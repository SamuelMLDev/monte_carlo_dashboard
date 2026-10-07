"""Estimator plug-in interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class EstimateResult:
    """Standard output contract for every estimator."""

    estimate: float
    standard_error: float
    ci_lower: float
    ci_upper: float
    converged: bool = True
    status: str = "ok"
    diagnostics: dict[str, Any] = field(default_factory=dict)


class Estimator(ABC):
    """Plug-in contract for treatment-effect estimators."""

    name: str
    supports_binary: bool = True

    @abstractmethod
    def estimate(
        self,
        data: pd.DataFrame,
        confidence_level: float = 0.95,
        settings: dict[str, Any] | None = None,
    ) -> EstimateResult:
        """Estimate the average treatment effect and uncertainty."""
