"""Factorial experiment-grid construction."""

from __future__ import annotations

from dataclasses import replace
from itertools import product
from typing import Any

from .config import DGPConfig, ExperimentSpec, SimulationCondition
from .seeds import stable_seed


FACTOR_LABELS: dict[str, str] = {
    "sample_size": "Sample size",
    "noise_sd": "Noise SD",
    "confounding_strength": "Confounding strength",
    "tau0": "Effect size",
    "confidence_level": "Confidence level",
    "num_covariates": "Number of covariates",
    "error_distribution": "Error distribution",
    "outcome_model": "Outcome model",
    "stress_level": "Stress level",
}


def _stress_dgp(base: DGPConfig, level: int) -> DGPConfig:
    """Return cumulative, increasingly difficult methodological stress scenarios."""
    common = dict(outcome_type="continuous", covariate_distribution="normal", num_covariates=max(1, base.num_covariates))
    if level == 1:
        return replace(base, **common, outcome_model="linear", assignment_mode="randomized", confounding_strength=0.0, error_distribution="gaussian", noise_sd=1.0, propensity_clip=0.01)
    if level == 2:
        return replace(base, **common, outcome_model="linear", assignment_mode="randomized", confounding_strength=0.0, error_distribution="gaussian", noise_sd=2.0, propensity_clip=0.01)
    if level == 3:
        return replace(base, **common, outcome_model="linear", assignment_mode="logistic", confounding_strength=0.9, error_distribution="gaussian", noise_sd=2.0, propensity_clip=0.01)
    if level == 4:
        return replace(base, **common, outcome_model="nonlinear", nonlinear_strength=1.0, assignment_mode="logistic", confounding_strength=0.9, error_distribution="gaussian", noise_sd=2.0, propensity_clip=0.01)
    if level == 5:
        return replace(base, **common, outcome_model="nonlinear", nonlinear_strength=1.0, assignment_mode="logistic", confounding_strength=0.9, error_distribution="heteroskedastic", heteroskedastic_strength=1.5, noise_sd=2.0, propensity_clip=0.01)
    if level == 6:
        return replace(base, **common, outcome_model="nonlinear", nonlinear_strength=1.0, assignment_mode="logistic", confounding_strength=2.5, error_distribution="heteroskedastic", heteroskedastic_strength=1.5, noise_sd=2.0, propensity_clip=0.001)
    if level == 7:
        return replace(base, **common, outcome_model="nonlinear", nonlinear_strength=1.0, assignment_mode="logistic", confounding_strength=2.5, error_distribution="student_t", student_t_df=3.0, noise_sd=2.0, propensity_clip=0.001)
    raise ValueError("stress_level must be between 1 and 7")


def _apply_factor(sample_size: int, confidence_level: float, dgp: DGPConfig, name: str, value: Any) -> tuple[int, float, DGPConfig]:
    if name == "sample_size":
        return int(value), confidence_level, dgp
    if name == "confidence_level":
        return sample_size, float(value), dgp
    if name == "stress_level":
        return sample_size, confidence_level, _stress_dgp(dgp, int(value))
    dgp_field = {
        "noise_sd": "noise_sd",
        "confounding_strength": "confounding_strength",
        "tau0": "tau0",
        "effect_size": "tau0",
        "num_covariates": "num_covariates",
        "error_distribution": "error_distribution",
        "outcome_model": "outcome_model",
        "propensity_clip": "propensity_clip",
        "tau1": "tau1",
    }.get(name)
    if dgp_field is None:
        raise ValueError(f"Unsupported grid factor: {name}")
    return sample_size, confidence_level, replace(dgp, **{dgp_field: value})


def build_conditions(spec: ExperimentSpec) -> list[SimulationCondition]:
    """Expand a study specification into the Cartesian product of factors."""
    spec.validate()
    factors = {key: tuple(values) for key, values in spec.factors.items() if tuple(values)}
    if not factors:
        combinations = [({}, ())]
    else:
        keys = list(factors)
        combinations = [
            (dict(zip(keys, values, strict=True)), values)
            for values in product(*(factors[key] for key in keys))
        ]

    conditions: list[SimulationCondition] = []
    for index, (factor_values, _) in enumerate(combinations, start=1):
        sample_size = spec.sample_size
        confidence_level = spec.confidence_level
        dgp = replace(spec.dgp, tau0=0.0, tau1=0.0, interaction_strength=0.0) if spec.mode == "type_i" else spec.dgp
        for name, value in factor_values.items():
            sample_size, confidence_level, dgp = _apply_factor(sample_size, confidence_level, dgp, name, value)
        if spec.mode == "type_i":
            dgp = replace(dgp, tau0=0.0, tau1=0.0, interaction_strength=0.0)
        if spec.generator == "parametric":
            dgp.validate()
        condition_seed = stable_seed(spec.seed, "condition", dict(sorted(factor_values.items())))
        conditions.append(
            SimulationCondition(
                condition_id=f"C{index:03d}",
                condition_index=index,
                estimand=spec.estimand,
                generator=spec.generator,
                generator_config=spec.generator_config,
                sample_size=sample_size,
                replications=spec.replications,
                confidence_level=confidence_level,
                estimators=spec.estimators,
                experiment_seed=spec.seed,
                condition_seed=condition_seed,
                null_value=spec.null_value,
                dgp=dgp,
                factor_values=factor_values,
                estimator_settings=spec.estimator_settings,
            )
        )
    return conditions


def estimate_workload(spec: ExperimentSpec) -> dict[str, int]:
    """Return simple transparent workload counts; not a wall-clock promise."""
    conditions = build_conditions(spec)
    datasets = sum(condition.replications for condition in conditions)
    estimator_fits = sum(condition.replications * len(condition.estimators) for condition in conditions)
    return {"conditions": len(conditions), "datasets": datasets, "estimator_fits": estimator_fits}
