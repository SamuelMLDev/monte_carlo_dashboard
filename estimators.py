"""Backwards-compatible estimator facade plus access to the expanded library."""

from __future__ import annotations

import pandas as pd

from mc_lab.estimators.base import EstimateResult
from mc_lab.estimators.registry import ESTIMATOR_NAMES as RESEARCH_ESTIMATORS
from mc_lab.estimators.registry import estimate_effect as _research_estimate


def difference_in_means(data: pd.DataFrame) -> EstimateResult:
    return _research_estimate(data, "Difference in means")


def ols_regression(data: pd.DataFrame) -> EstimateResult:
    return _research_estimate(data, "OLS regression")


def robust_regression(data: pd.DataFrame) -> EstimateResult:
    return _research_estimate(data, "Robust regression (Huber)")


ESTIMATORS = {
    "Difference in means": difference_in_means,
    "OLS regression": ols_regression,
    "Robust regression": robust_regression,
}


def estimate_effect(data: pd.DataFrame, estimator: str) -> EstimateResult:
    """Dispatch legacy estimator names while also accepting new registry names."""
    if estimator in ESTIMATORS:
        return ESTIMATORS[estimator](data)
    return _research_estimate(data, estimator)
