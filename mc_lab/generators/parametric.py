"""Configurable parametric data-generating process."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.special import expit

from mc_lab.config import DGPConfig
from mc_lab.seeds import make_rng

from .base import DataGenerator, GeneratedDataset


def _covariates(config: DGPConfig, sample_size: int, rng: np.random.Generator) -> np.ndarray:
    p = config.num_covariates
    if config.covariate_distribution == "normal":
        return rng.normal(size=(sample_size, p))
    if config.covariate_distribution == "uniform":
        bound = math.sqrt(3.0)
        return rng.uniform(-bound, bound, size=(sample_size, p))
    if config.covariate_distribution == "bernoulli":
        return rng.binomial(1, 0.5, size=(sample_size, p)).astype(float)
    if config.covariate_distribution == "correlated_normal":
        rho = config.covariate_correlation
        covariance = np.full((p, p), rho, dtype=float)
        np.fill_diagonal(covariance, 1.0)
        # The UI limits rho to a safe range; add a clear failure for invalid custom manifests.
        eig_min = float(np.linalg.eigvalsh(covariance).min())
        if eig_min <= 1e-10:
            raise ValueError("covariate_correlation yields a non-positive-definite covariance matrix")
        return rng.multivariate_normal(np.zeros(p), covariance, size=sample_size)
    raise ValueError(f"Unknown covariate distribution: {config.covariate_distribution}")


def _beta_vector(config: DGPConfig) -> np.ndarray:
    # Decreasing magnitudes keep high-dimensional presets numerically well behaved.
    return config.beta_scale / np.sqrt(np.arange(1, config.num_covariates + 1, dtype=float))


def _systematic_outcome(x: np.ndarray, config: DGPConfig) -> np.ndarray:
    base = config.beta0 + x @ _beta_vector(config)
    x1 = x[:, 0]
    if config.outcome_model == "quadratic":
        base = base + config.quadratic_strength * (x1**2 - np.mean(x1**2))
    elif config.outcome_model == "nonlinear":
        strength = config.nonlinear_strength
        base = base + strength * np.sin(x1)
        if x.shape[1] > 1:
            base = base + 0.5 * strength * (x[:, 1] ** 2 - np.mean(x[:, 1] ** 2))
    return base


def _unit_effect(x: np.ndarray, config: DGPConfig) -> np.ndarray:
    tau = np.full(x.shape[0], config.tau0, dtype=float)
    if config.outcome_model == "heterogeneous" or config.tau1 != 0:
        tau = tau + config.tau1 * x[:, 0]
    if config.outcome_model == "interaction" or config.interaction_strength != 0:
        tau = tau + config.interaction_strength * x[:, 0]
    return tau


def _propensity(x: np.ndarray, config: DGPConfig) -> np.ndarray:
    if config.assignment_mode == "randomized":
        return np.full(x.shape[0], config.treatment_probability, dtype=float)
    score = np.full(x.shape[0], config.treatment_intercept, dtype=float)
    score += config.confounding_strength * x[:, 0]
    if x.shape[1] > 1:
        score += 0.35 * config.confounding_strength * x[:, 1]
    propensity = expit(score)
    if config.propensity_clip > 0:
        propensity = np.clip(propensity, config.propensity_clip, 1.0 - config.propensity_clip)
    return propensity


def _continuous_error(x: np.ndarray, config: DGPConfig, rng: np.random.Generator) -> np.ndarray:
    n = x.shape[0]
    if config.error_distribution == "gaussian":
        return rng.normal(scale=config.noise_sd, size=n)
    if config.error_distribution == "student_t":
        # Scale t draws to the requested standard deviation.
        scale = config.noise_sd * math.sqrt((config.student_t_df - 2.0) / config.student_t_df)
        return rng.standard_t(config.student_t_df, size=n) * scale
    if config.error_distribution == "skewed":
        shape = 0.7
        raw = rng.lognormal(mean=0.0, sigma=shape, size=n)
        expected = math.exp(shape**2 / 2)
        variance = (math.exp(shape**2) - 1.0) * math.exp(shape**2)
        return config.noise_sd * (raw - expected) / math.sqrt(variance)
    if config.error_distribution == "heteroskedastic":
        local_sd = config.noise_sd * np.sqrt(1.0 + config.heteroskedastic_strength * x[:, 0] ** 2)
        return rng.normal(scale=local_sd, size=n)
    raise ValueError(f"Unknown error distribution: {config.error_distribution}")


class ParametricDataGenerator(DataGenerator):
    """Flexible structural DGP with known sample-level ATE."""

    name = "parametric"

    def generate(self, config: DGPConfig, sample_size: int, seed: int) -> GeneratedDataset:
        config.validate()
        rng = make_rng(seed)
        x = _covariates(config, sample_size, rng)
        propensity = _propensity(x, config)
        treatment = rng.binomial(1, propensity, size=sample_size)
        base = _systematic_outcome(x, config)
        tau = _unit_effect(x, config)

        if config.outcome_type == "continuous":
            error = _continuous_error(x, config, rng)
            y0_mean = base
            y1_mean = base + tau
            outcome = base + treatment * tau + error
            true_ate = float(np.mean(y1_mean - y0_mean))
        else:
            p0 = expit(base)
            p1 = expit(base + tau)
            observed_probability = np.where(treatment == 1, p1, p0)
            outcome = rng.binomial(1, observed_probability, size=sample_size).astype(float)
            y0_mean = p0
            y1_mean = p1
            true_ate = float(np.mean(p1 - p0))

        columns = {f"X{j + 1}": x[:, j] for j in range(config.num_covariates)}
        data = pd.DataFrame(columns)
        data.insert(0, "A", treatment.astype(int))
        data.insert(0, "Y", outcome.astype(float))
        data["true_propensity"] = propensity
        data["true_y0_mean"] = y0_mean
        data["true_y1_mean"] = y1_mean
        data["true_unit_effect"] = y1_mean - y0_mean
        return GeneratedDataset(data=data, true_estimand=true_ate, description=self.describe(config))

    def true_estimand(self, data: pd.DataFrame, config: DGPConfig) -> float:
        if "true_unit_effect" not in data:
            raise ValueError("Generated data do not contain structural potential-outcome means")
        return float(data["true_unit_effect"].mean())

    def describe(self, config: DGPConfig) -> str:
        if config.assignment_mode == "randomized":
            treatment = f"A ~ Bernoulli({config.treatment_probability:.3g}) independent of X"
        else:
            score = "alpha0 + alpha1*X1"
            if config.num_covariates > 1:
                score += " + 0.35*alpha1*X2"
            treatment = f"A ~ Bernoulli(expit({score})), alpha1={config.confounding_strength:.3g}"
        if config.outcome_type == "continuous":
            error_label = {
                "gaussian": "Gaussian",
                "student_t": f"Student-t(df={config.student_t_df:g})",
                "skewed": "centered log-normal",
                "heteroskedastic": "Gaussian with Var(e|X) depending on X1",
            }[config.error_distribution]
            outcome = "Y = beta0 + beta'X + tau(X) A + epsilon"
            heterogeneity = config.tau1 + config.interaction_strength
            return f"{outcome}; tau(X)={config.tau0:g}+{heterogeneity:g}X1; epsilon: {error_label}. {treatment}."
        return (
            "Y|A,X ~ Bernoulli(expit(beta0 + beta'X + tau(X) A)); "
            f"tau(X)={config.tau0:g}+{config.tau1:g}X1 on the log-odds scale; {treatment}. "
            "The evaluated estimand is the sample-average risk difference E[Y(1)-Y(0)|X]."
        )


DEFAULT_GENERATOR = ParametricDataGenerator()
