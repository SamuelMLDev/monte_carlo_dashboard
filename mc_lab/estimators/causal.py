"""Propensity-score and doubly robust estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from .base import EstimateResult, Estimator
from .core import covariate_columns


def _critical_value(confidence_level: float) -> float:
    return float(norm.ppf(0.5 + confidence_level / 2.0))


def _finalize(estimate: float, influence: np.ndarray, confidence_level: float, diagnostics: dict[str, Any]) -> EstimateResult:
    influence = np.asarray(influence, dtype=float)
    if len(influence) < 2 or not np.all(np.isfinite(influence)):
        se = float("nan")
    else:
        se = float(np.std(influence, ddof=1) / np.sqrt(len(influence)))
    critical = _critical_value(confidence_level)
    lower = estimate - critical * se if np.isfinite(se) else float("nan")
    upper = estimate + critical * se if np.isfinite(se) else float("nan")
    return EstimateResult(float(estimate), se, float(lower), float(upper), diagnostics=diagnostics)


def _design_matrix(data: pd.DataFrame, covariates: list[str], indices: np.ndarray) -> np.ndarray:
    if covariates:
        x = data.iloc[indices][covariates].to_numpy(dtype=float)
        return np.column_stack([np.ones(len(indices)), x])
    return np.ones((len(indices), 1), dtype=float)


def _logistic_fit_predict(x_train: np.ndarray, y_train: np.ndarray, x_predict: np.ndarray) -> tuple[np.ndarray, bool]:
    """Small unpenalized IRLS logistic fit used only for nuisance prediction."""
    beta = np.zeros(x_train.shape[1], dtype=float)
    converged = False
    for _ in range(60):
        eta = x_train @ beta
        probability = expit(np.clip(eta, -30.0, 30.0))
        weights = np.clip(probability * (1.0 - probability), 1e-8, None)
        gradient = x_train.T @ (y_train - probability)
        information = x_train.T @ (weights[:, None] * x_train)
        information.flat[:: information.shape[0] + 1] += 1e-10
        try:
            step = np.linalg.solve(information, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(information) @ gradient
        beta = beta + step
        if float(np.max(np.abs(step))) < 1e-8:
            converged = True
            break
    predicted = expit(np.clip(x_predict @ beta, -30.0, 30.0))
    return np.asarray(predicted, dtype=float), converged


def propensity_scores(
    data: pd.DataFrame,
    settings: dict[str, Any] | None = None,
    train_index: np.ndarray | None = None,
    predict_index: np.ndarray | None = None,
) -> tuple[np.ndarray, bool]:
    """Fit an unpenalized logistic propensity model and return clipped probabilities."""
    settings = settings or {}
    covariates = covariate_columns(data)
    selected = settings.get("propensity_covariates")
    if selected is not None:
        covariates = [column for column in covariates if column in set(selected)]
    clip = float(settings.get("propensity_clip", 0.01))

    train_index = np.arange(len(data)) if train_index is None else np.asarray(train_index)
    predict_index = np.arange(len(data)) if predict_index is None else np.asarray(predict_index)
    x_train = _design_matrix(data, covariates, train_index)
    x_predict = _design_matrix(data, covariates, predict_index)
    y_train = data.iloc[train_index]["A"].to_numpy(dtype=float)
    probabilities, converged = _logistic_fit_predict(x_train, y_train, x_predict)
    probabilities = np.clip(probabilities, clip, 1.0 - clip)
    return probabilities, converged


def observed_ipw_weights(data: pd.DataFrame, propensity: np.ndarray, stabilized: bool = False) -> np.ndarray:
    """Return observation-level inverse probability weights."""
    a = data["A"].to_numpy(dtype=float)
    if stabilized:
        p = float(a.mean())
        return a * p / propensity + (1.0 - a) * (1.0 - p) / (1.0 - propensity)
    return a / propensity + (1.0 - a) / (1.0 - propensity)


def weight_diagnostics(weights: np.ndarray) -> dict[str, float]:
    weights = np.asarray(weights, dtype=float)
    total = float(weights.sum())
    ess = (total**2 / float(np.sum(weights**2))) if np.sum(weights**2) > 0 else 0.0
    return {
        "max_weight": float(np.max(weights)),
        "weight_mean": float(np.mean(weights)),
        "effective_sample_size": ess,
        "extreme_weight_fraction": float(np.mean(weights > 10.0)),
    }


class IPWEstimator(Estimator):
    name = "Inverse probability weighting"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        settings = settings or {}
        e, converged = propensity_scores(data, settings)
        a = data["A"].to_numpy(dtype=float)
        y = data["Y"].to_numpy(dtype=float)
        score = a * y / e - (1.0 - a) * y / (1.0 - e)
        estimate = float(np.mean(score))
        influence = score - estimate
        weights = observed_ipw_weights(data, e, stabilized=False)
        diagnostics: dict[str, Any] = weight_diagnostics(weights)
        diagnostics.update({"propensity_converged": converged, "min_propensity": float(e.min()), "max_propensity": float(e.max()), "se_note": "IF treats fitted propensity as fixed"})
        result = _finalize(estimate, influence, confidence_level, diagnostics)
        return EstimateResult(**{**result.__dict__, "converged": converged, "status": "ok" if converged else "not_converged"})


class StabilizedIPWEstimator(Estimator):
    name = "Stabilized IPW"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        settings = settings or {}
        e, converged = propensity_scores(data, settings)
        a = data["A"].to_numpy(dtype=float)
        y = data["Y"].to_numpy(dtype=float)
        w1 = a / e
        w0 = (1.0 - a) / (1.0 - e)
        if w1.sum() <= 0 or w0.sum() <= 0:
            raise ValueError("Both treatment groups require positive total weight")
        mu1 = float(np.sum(w1 * y) / np.sum(w1))
        mu0 = float(np.sum(w0 * y) / np.sum(w0))
        estimate = mu1 - mu0
        # Ratio-estimator influence function with fitted propensity treated as fixed.
        influence = w1 * (y - mu1) / np.mean(w1) - w0 * (y - mu0) / np.mean(w0)
        weights = observed_ipw_weights(data, e, stabilized=True)
        diagnostics: dict[str, Any] = weight_diagnostics(weights)
        diagnostics.update({"propensity_converged": converged, "min_propensity": float(e.min()), "max_propensity": float(e.max()), "se_note": "Hajek IF treats fitted propensity as fixed"})
        result = _finalize(estimate, influence, confidence_level, diagnostics)
        return EstimateResult(**{**result.__dict__, "converged": converged, "status": "ok" if converged else "not_converged"})


def _fit_outcome_model(
    train: pd.DataFrame,
    predict: pd.DataFrame,
    treatment: int,
    settings: dict[str, Any],
) -> tuple[np.ndarray, bool]:
    covariates = covariate_columns(train)
    selected = settings.get("outcome_covariates")
    if selected is not None:
        covariates = [column for column in covariates if column in set(selected)]
    group = train.loc[train["A"] == treatment]
    if len(group) < max(4, len(covariates) + 2):
        raise ValueError("Insufficient treatment-specific observations for outcome nuisance model")
    train_idx = np.arange(len(group))
    predict_idx = np.arange(len(predict))
    # Work on the group/predict frames directly so indices need not align.
    if covariates:
        x_train = np.column_stack([np.ones(len(group)), group[covariates].to_numpy(dtype=float)])
        x_predict = np.column_stack([np.ones(len(predict)), predict[covariates].to_numpy(dtype=float)])
    else:
        x_train = np.ones((len(group), 1), dtype=float)
        x_predict = np.ones((len(predict), 1), dtype=float)
    y = group["Y"].to_numpy(dtype=float)
    binary = bool(settings.get("binary_outcome", set(np.unique(train["Y"].dropna())).issubset({0.0, 1.0})))
    if binary:
        prediction, converged = _logistic_fit_predict(x_train, y, x_predict)
        return prediction, converged
    beta, *_ = np.linalg.lstsq(x_train, y, rcond=None)
    return np.asarray(x_predict @ beta, dtype=float), True


class AIPWEstimator(Estimator):
    """Two-fold cross-fitted augmented inverse-probability-weighted ATE."""

    name = "Doubly robust AIPW"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        settings = settings or {}
        n = len(data)
        if n < 20:
            raise ValueError("AIPW requires at least 20 observations")
        e = np.empty(n, dtype=float)
        m0 = np.empty(n, dtype=float)
        m1 = np.empty(n, dtype=float)
        converged = True
        indices = np.arange(n)
        # Deterministic interleaved folds avoid introducing an estimator-level RNG.
        folds = indices % 2
        for fold in (0, 1):
            test_idx = indices[folds == fold]
            train_idx = indices[folds != fold]
            fold_e, prop_converged = propensity_scores(data, settings, train_idx, test_idx)
            e[test_idx] = fold_e
            train = data.iloc[train_idx]
            predict = data.iloc[test_idx]
            fold_m0, m0_converged = _fit_outcome_model(train, predict, 0, settings)
            fold_m1, m1_converged = _fit_outcome_model(train, predict, 1, settings)
            m0[test_idx] = fold_m0
            m1[test_idx] = fold_m1
            converged = converged and prop_converged and m0_converged and m1_converged

        a = data["A"].to_numpy(dtype=float)
        y = data["Y"].to_numpy(dtype=float)
        pseudo = m1 - m0 + a * (y - m1) / e - (1.0 - a) * (y - m0) / (1.0 - e)
        estimate = float(np.mean(pseudo))
        influence = pseudo - estimate
        weights = observed_ipw_weights(data, e, stabilized=False)
        diagnostics: dict[str, Any] = weight_diagnostics(weights)
        diagnostics.update({"propensity_converged": converged, "min_propensity": float(e.min()), "max_propensity": float(e.max()), "cross_fitted": True})
        result = _finalize(estimate, influence, confidence_level, diagnostics)
        return EstimateResult(**{**result.__dict__, "converged": converged, "status": "ok" if converged else "not_converged"})
