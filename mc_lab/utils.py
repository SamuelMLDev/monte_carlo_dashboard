"""Serialization and formatting helpers."""

from __future__ import annotations

import json

import pandas as pd


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(index=False).encode("utf-8")


def config_to_json(config: dict[str, object]) -> str:
    return json.dumps(config, indent=2, sort_keys=True, default=str)


def format_mc(value: float, mcse: float, percent: bool = False) -> str:
    """Format an estimate with its Monte Carlo standard error."""
    if pd.isna(value):
        return "Unavailable"
    if percent:
        if pd.isna(mcse):
            return f"{100 * value:.1f}%"
        return f"{100 * value:.1f}% ± {100 * mcse:.1f}% MCSE"
    if pd.isna(mcse):
        return f"{value:.4f}"
    return f"{value:.4f} ± {mcse:.4f} MCSE"
