"""Estimator registry and dispatch."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .base import EstimateResult, Estimator
from .causal import AIPWEstimator, IPWEstimator, StabilizedIPWEstimator
from .core import (
    DifferenceInMeansEstimator,
    OLSEstimator,
    OLSHC3Estimator,
    OutcomeRegressionEstimator,
    RobustRegressionEstimator,
)

_ESTIMATOR_CLASSES: tuple[type[Estimator], ...] = (
    DifferenceInMeansEstimator,
    OLSEstimator,
    OLSHC3Estimator,
    IPWEstimator,
    StabilizedIPWEstimator,
    OutcomeRegressionEstimator,
    AIPWEstimator,
    RobustRegressionEstimator,
)

ESTIMATOR_REGISTRY: dict[str, Estimator] = {cls.name: cls() for cls in _ESTIMATOR_CLASSES}
ESTIMATOR_NAMES: tuple[str, ...] = tuple(ESTIMATOR_REGISTRY)


def get_estimator(name: str) -> Estimator:
    try:
        return ESTIMATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown estimator: {name}") from exc


def estimate_effect(
    data: pd.DataFrame,
    estimator: str,
    confidence_level: float = 0.95,
    settings: dict[str, Any] | None = None,
) -> EstimateResult:
    method = get_estimator(estimator)
    settings = settings or {}
    if bool(settings.get("binary_outcome", False)) and not method.supports_binary:
        raise ValueError(f"{estimator} does not support binary outcomes")
    return method.estimate(data, confidence_level=confidence_level, settings=settings)
