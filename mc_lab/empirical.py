"""Empirical-data estimator comparison helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .estimators import estimate_effect
from .seeds import make_rng


def prepare_empirical_data(
    frame: pd.DataFrame,
    outcome: str,
    treatment: str,
    covariates: list[str],
    treated_value: Any | None = None,
) -> pd.DataFrame:
    """Map arbitrary user columns into the estimator library's Y/A/X schema."""
    columns = [outcome, treatment, *covariates]
    data = frame[columns].dropna().copy()
    if treated_value is None:
        unique = list(pd.unique(data[treatment]))
        if set(unique).issubset({0, 1, 0.0, 1.0, False, True}):
            a = data[treatment].astype(int)
        elif len(unique) == 2:
            treated_value = unique[-1]
            a = (data[treatment] == treated_value).astype(int)
        else:
            raise ValueError("Treatment must be binary or a treated value must be selected")
    else:
        a = (data[treatment] == treated_value).astype(int)
    if a.nunique() != 2:
        raise ValueError("Mapped treatment does not contain both groups")
    result = pd.DataFrame({"Y": pd.to_numeric(data[outcome], errors="raise").astype(float), "A": a.to_numpy(dtype=int)})
    for index, column in enumerate(covariates, start=1):
        result[f"X{index}"] = pd.to_numeric(data[column], errors="raise").astype(float).to_numpy()
    return result.reset_index(drop=True)


def compare_estimators(data: pd.DataFrame, estimators: list[str], confidence_level: float = 0.95) -> pd.DataFrame:
    """Estimate effects on empirical data without inventing a population truth."""
    rows = []
    binary = set(np.unique(data["Y"])).issubset({0.0, 1.0})
    for estimator in estimators:
        try:
            result = estimate_effect(data, estimator, confidence_level, {"binary_outcome": binary})
            rows.append({
                "estimator": estimator,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "converged": result.converged,
                "status": result.status,
                "error": "",
            })
        except Exception as exc:
            rows.append({"estimator": estimator, "estimate": np.nan, "standard_error": np.nan, "ci_lower": np.nan, "ci_upper": np.nan, "converged": False, "status": "failed", "error": f"{type(exc).__name__}: {exc}"})
    return pd.DataFrame(rows)


def bootstrap_estimator(
    data: pd.DataFrame,
    estimator: str,
    replications: int = 300,
    seed: int = 42,
    confidence_level: float = 0.95,
) -> dict[str, float]:
    """Nonparametric bootstrap uncertainty for exploratory empirical analysis."""
    rng = make_rng(seed)
    estimates: list[float] = []
    n = len(data)
    for _ in range(replications):
        indices = rng.integers(0, n, size=n)
        sample = data.iloc[indices].reset_index(drop=True)
        try:
            value = estimate_effect(sample, estimator, confidence_level).estimate
            if np.isfinite(value):
                estimates.append(float(value))
        except Exception:
            continue
    if len(estimates) < max(20, replications // 10):
        raise RuntimeError("Too few successful bootstrap fits")
    alpha = 1.0 - confidence_level
    array = np.asarray(estimates)
    return {
        "bootstrap_mean": float(array.mean()),
        "bootstrap_se": float(array.std(ddof=1)),
        "bootstrap_ci_lower": float(np.quantile(array, alpha / 2)),
        "bootstrap_ci_upper": float(np.quantile(array, 1 - alpha / 2)),
        "successful_bootstraps": int(len(array)),
    }
