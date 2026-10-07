"""Extension interface for data generators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import pandas as pd

from typing import Any


@dataclass(frozen=True)
class GeneratedDataset:
    """Observed data plus known structural truth for one replication."""

    data: pd.DataFrame
    true_estimand: float
    description: str


class DataGenerator(ABC):
    """Plug-in contract for simulation data generators.

    Future generators (for example CTGAN, Gaussian copulas, or externally
    supplied synthetic data) can implement this interface without modifying
    the Monte Carlo engine.
    """

    name: str

    @abstractmethod
    def generate(self, config: Any, sample_size: int, seed: int) -> GeneratedDataset:
        """Generate one dataset for a deterministic seed."""

    @abstractmethod
    def true_estimand(self, data: pd.DataFrame, config: Any) -> float:
        """Return the known estimand for this generated covariate sample."""

    @abstractmethod
    def describe(self, config: Any) -> str:
        """Return a concise mathematical description suitable for reports."""
