"""Backwards-compatible metric facade for the research package."""

from __future__ import annotations

import pandas as pd

from mc_lab.metrics import calculate_metrics as _calculate_metrics


def calculate_metrics(results: pd.DataFrame, true_effect: float) -> dict[str, float | int]:
    """Calculate Monte Carlo metrics; preserves the original public function."""
    return _calculate_metrics(results, true_effect=true_effect)


def summarize_results(results: pd.DataFrame, true_effect: float, sample_size: int) -> pd.DataFrame:
    metrics = calculate_metrics(results, true_effect)
    metrics["sample_size"] = int(sample_size)
    return pd.DataFrame([metrics])
