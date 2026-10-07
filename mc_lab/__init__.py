"""Research-grade Monte Carlo simulation laboratory."""

from .config import DGPConfig, ExperimentSpec, SimulationCondition
from .engine import ExperimentResult, run_experiment

__all__ = [
    "DGPConfig",
    "ExperimentSpec",
    "SimulationCondition",
    "ExperimentResult",
    "run_experiment",
]

__version__ = "0.3.0"
