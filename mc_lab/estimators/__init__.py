"""Estimator library."""

from .base import EstimateResult, Estimator
from .causal import observed_ipw_weights, propensity_scores, weight_diagnostics
from .registry import ESTIMATOR_NAMES, ESTIMATOR_REGISTRY, estimate_effect, get_estimator

__all__ = [
    "EstimateResult",
    "Estimator",
    "ESTIMATOR_NAMES",
    "ESTIMATOR_REGISTRY",
    "estimate_effect",
    "get_estimator",
    "propensity_scores",
    "observed_ipw_weights",
    "weight_diagnostics",
]
