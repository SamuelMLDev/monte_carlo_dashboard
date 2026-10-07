"""Treatment, weighting, balance, and model diagnostics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .config import SimulationCondition
from .estimators.causal import observed_ipw_weights, propensity_scores, weight_diagnostics
from .estimators.core import covariate_columns
from .generators import get_generator
from .seeds import replication_seed


def generate_diagnostic_dataset(condition: SimulationCondition, replication: int = 1) -> pd.DataFrame:
    """Regenerate a specific replication exactly from its hierarchical seed."""
    seed = replication_seed(condition.condition_seed, replication)
    generator = get_generator(condition.generator)
    config = condition.dgp if condition.generator == "parametric" else condition.generator_config
    return generator.generate(config, condition.sample_size, seed).data


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    return float(np.sum(values * weights) / total) if total > 0 else float("nan")


def weighted_variance(values: np.ndarray, weights: np.ndarray) -> float:
    mean = weighted_mean(values, weights)
    total = float(np.sum(weights))
    return float(np.sum(weights * (values - mean) ** 2) / total) if total > 0 else float("nan")


def standardized_mean_differences(data: pd.DataFrame, weights: np.ndarray | None = None) -> pd.DataFrame:
    """Compute absolute and signed standardized mean differences by covariate."""
    rows = []
    a = data["A"].to_numpy(dtype=int)
    if weights is None:
        weights = np.ones(len(data), dtype=float)
    weights = np.asarray(weights, dtype=float)
    for column in covariate_columns(data):
        x = data[column].to_numpy(dtype=float)
        treated = a == 1
        control = a == 0
        mt = weighted_mean(x[treated], weights[treated])
        mc = weighted_mean(x[control], weights[control])
        vt = weighted_variance(x[treated], weights[treated])
        vc = weighted_variance(x[control], weights[control])
        pooled = np.sqrt(max(0.0, (vt + vc) / 2.0))
        smd = (mt - mc) / pooled if pooled > 0 else 0.0
        rows.append({"covariate": column, "smd": float(smd), "abs_smd": abs(float(smd))})
    return pd.DataFrame(rows)


def treatment_diagnostics(data: pd.DataFrame, stabilized: bool = False, settings: dict | None = None) -> dict[str, object]:
    """Fit a propensity model and return overlap, balance, and weight diagnostics."""
    propensity, converged = propensity_scores(data, settings or {})
    weights = observed_ipw_weights(data, propensity, stabilized=stabilized)
    before = standardized_mean_differences(data).rename(columns={"smd": "smd_before", "abs_smd": "abs_smd_before"})
    after = standardized_mean_differences(data, weights).rename(columns={"smd": "smd_after", "abs_smd": "abs_smd_after"})
    balance = before.merge(after, on="covariate", how="outer")
    treated = data["A"].to_numpy(dtype=int) == 1
    support_lower = max(float(propensity[treated].min(initial=1.0)), float(propensity[~treated].min(initial=1.0)))
    support_upper = min(float(propensity[treated].max(initial=0.0)), float(propensity[~treated].max(initial=0.0)))
    overlap_width = max(0.0, support_upper - support_lower)
    result: dict[str, object] = {
        "propensity": propensity,
        "weights": weights,
        "balance": balance,
        "propensity_converged": converged,
        "overlap_lower": support_lower,
        "overlap_upper": support_upper,
        "overlap_width": overlap_width,
    }
    result.update(weight_diagnostics(weights))
    return result


def ols_residual_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    """Return fitted values and residuals for a simple regression diagnostic."""
    columns = ["A", *covariate_columns(data)]
    design = sm.add_constant(data[columns].astype(float), has_constant="add")
    fit = sm.OLS(data["Y"].astype(float), design).fit()
    residuals = np.asarray(fit.resid, dtype=float)
    fitted = np.asarray(fit.fittedvalues, dtype=float)
    return pd.DataFrame({"fitted": fitted, "residual": residuals, "abs_residual": np.abs(residuals)})


def heteroskedasticity_indicator(data: pd.DataFrame) -> float:
    """Simple descriptive correlation between |OLS residual| and |X1|."""
    diag = ols_residual_diagnostics(data)
    x = np.abs(data[covariate_columns(data)[0]].to_numpy(dtype=float))
    if np.std(x) == 0 or np.std(diag["abs_residual"]) == 0:
        return 0.0
    return float(np.corrcoef(x, diag["abs_residual"])[0, 1])
