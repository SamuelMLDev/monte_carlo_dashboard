"""Synthetic data generation and Monte Carlo execution."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from estimators import estimate_effect


@dataclass(frozen=True)
class SimulationConfig:
    """Complete reproducibility configuration for one Monte Carlo run."""

    sample_size: int = 500
    replications: int = 500
    true_effect: float = 0.5
    noise_sd: float = 1.0
    estimator: str = "OLS regression"
    assignment_mode: str = "randomized"
    seed: int = 42
    beta0: float = 0.0
    beta1: float = 1.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _expit(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def generate_dataset(config: SimulationConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Generate one dataset from Y = beta0 + tau*A + beta1*X + epsilon."""
    x = rng.normal(0.0, 1.0, config.sample_size)

    if config.assignment_mode == "randomized":
        treatment_prob = np.full(config.sample_size, 0.5)
    elif config.assignment_mode == "confounded":
        treatment_prob = np.clip(_expit(0.9 * x), 0.05, 0.95)
    else:
        raise ValueError(f"Unknown assignment mode: {config.assignment_mode}")

    a = rng.binomial(1, treatment_prob)
    epsilon = rng.normal(0.0, config.noise_sd, config.sample_size)
    y = config.beta0 + config.true_effect * a + config.beta1 * x + epsilon

    return pd.DataFrame({"Y": y, "A": a, "X": x, "propensity": treatment_prob})


def run_monte_carlo(
    config: SimulationConfig,
    progress_callback: Callable[[float], None] | None = None,
) -> pd.DataFrame:
    """Run independent replications using a single seeded NumPy generator."""
    if config.sample_size < 4:
        raise ValueError("sample_size must be at least 4")
    if config.replications < 1:
        raise ValueError("replications must be positive")
    if config.noise_sd <= 0:
        raise ValueError("noise_sd must be positive")

    rng = np.random.default_rng(config.seed)
    rows: list[dict[str, float | int | bool]] = []

    for replication in range(1, config.replications + 1):
        # Extremely small samples can occasionally have only one treatment group.
        # Regenerate deterministically from the same RNG stream until both groups exist.
        for _ in range(100):
            data = generate_dataset(config, rng)
            if data["A"].nunique() == 2 and data.groupby("A").size().min() >= 2:
                break
        else:
            raise RuntimeError("Could not generate a dataset with both treatment groups.")

        result = estimate_effect(data, config.estimator)
        contains_true = result.ci_lower <= config.true_effect <= result.ci_upper
        rows.append(
            {
                "replication": replication,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "covers_true_effect": bool(contains_true),
            }
        )
        if progress_callback is not None:
            progress_callback(replication / config.replications)

    return pd.DataFrame(rows)
