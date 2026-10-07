"""Deterministic Monte Carlo execution engine."""

from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import pandas as pd

from .config import ExperimentSpec, SimulationCondition
from .estimators import estimate_effect
from .generators import get_generator
from .grid import build_conditions
from .metrics import summarize_long_results
from .seeds import replication_seed


@dataclass
class ExperimentResult:
    """All outputs from a simulation experiment."""

    replication_results: pd.DataFrame
    summary: pd.DataFrame
    conditions: list[SimulationCondition]
    elapsed_seconds: float

    @property
    def failures(self) -> pd.DataFrame:
        if self.replication_results.empty:
            return self.replication_results.copy()
        return self.replication_results.loc[~self.replication_results["success"].fillna(False)].copy()


def _failure_row(condition: SimulationCondition, replication: int, rep_seed: int, estimator: str, message: str) -> dict[str, object]:
    row: dict[str, object] = {
        "condition_id": condition.condition_id,
        "condition_index": condition.condition_index,
        "estimand": condition.estimand,
        "generator": condition.generator,
        "replication": replication,
        "replication_seed": rep_seed,
        "sample_size": condition.sample_size,
        "estimator": estimator,
        "confidence_level": condition.confidence_level,
        "true_estimand": float("nan"),
        "estimate": float("nan"),
        "standard_error": float("nan"),
        "ci_lower": float("nan"),
        "ci_upper": float("nan"),
        "covers_true_effect": pd.NA,
        "null_rejected": pd.NA,
        "success": False,
        "converged": False,
        "status": "failed",
        "error_message": message,
        "standard_error_available": False,
        "extreme_weights": pd.NA,
    }
    row.update({f"factor_{key}": value for key, value in condition.factor_values.items()})
    return row


def _run_replication(condition: SimulationCondition, replication: int) -> list[dict[str, object]]:
    rep_seed = replication_seed(condition.condition_seed, replication)
    try:
        generator = get_generator(condition.generator)
        generator_config = condition.dgp if condition.generator == "parametric" else condition.generator_config
        generated = generator.generate(generator_config, condition.sample_size, rep_seed)
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        return [_failure_row(condition, replication, rep_seed, estimator, f"data_generation: {type(exc).__name__}: {exc}") for estimator in condition.estimators]

    rows: list[dict[str, object]] = []
    for estimator in condition.estimators:
        try:
            settings = dict(condition.estimator_settings.get(estimator, {}))
            if condition.generator == "parametric":
                binary_outcome = condition.dgp.outcome_type == "binary"
                default_propensity_clip = max(1e-6, condition.dgp.propensity_clip)
            else:
                observed_y = generated.data["Y"].dropna().to_numpy()
                binary_outcome = bool(observed_y.size and np.isin(observed_y, [0, 1]).all())
                default_propensity_clip = 0.01
            settings.setdefault("binary_outcome", binary_outcome)
            settings.setdefault("propensity_clip", default_propensity_clip)
            result = estimate_effect(generated.data, estimator, condition.confidence_level, settings)
            finite_estimate = bool(np.isfinite(result.estimate))
            finite_ci = bool(np.isfinite(result.ci_lower) and np.isfinite(result.ci_upper))
            success = finite_estimate
            covers = bool(result.ci_lower <= generated.true_estimand <= result.ci_upper) if finite_ci else pd.NA
            rejected = bool(result.ci_upper < condition.null_value or result.ci_lower > condition.null_value) if finite_ci else pd.NA
            row: dict[str, object] = {
                "condition_id": condition.condition_id,
                "condition_index": condition.condition_index,
                "estimand": condition.estimand,
                "generator": condition.generator,
                "replication": replication,
                "replication_seed": rep_seed,
                "sample_size": condition.sample_size,
                "estimator": estimator,
                "confidence_level": condition.confidence_level,
                "true_estimand": generated.true_estimand,
                "estimate": result.estimate,
                "standard_error": result.standard_error,
                "ci_lower": result.ci_lower,
                "ci_upper": result.ci_upper,
                "covers_true_effect": covers,
                "null_rejected": rejected,
                "success": success,
                "converged": bool(result.converged),
                "status": result.status,
                "error_message": "",
                "standard_error_available": bool(np.isfinite(result.standard_error)),
                "extreme_weights": bool(float(result.diagnostics.get("max_weight", 0.0)) > 10.0) if "max_weight" in result.diagnostics else pd.NA,
            }
            for key, value in result.diagnostics.items():
                if isinstance(value, (str, bool, int, float, np.integer, np.floating)):
                    row[f"diag_{key}"] = value
            row.update({f"factor_{key}": value for key, value in condition.factor_values.items()})
            rows.append(row)
        except Exception as exc:
            failed = _failure_row(condition, replication, rep_seed, estimator, f"estimation: {type(exc).__name__}: {exc}")
            failed["true_estimand"] = generated.true_estimand
            rows.append(failed)
    return rows


def _task(condition: SimulationCondition, replication: int) -> tuple[int, int, list[dict[str, object]]]:
    return condition.condition_index, replication, _run_replication(condition, replication)


def run_experiment(
    spec: ExperimentSpec,
    progress_callback: Callable[[int, int], None] | None = None,
    workers: int | None = None,
) -> ExperimentResult:
    """Run all grid conditions with deterministic per-replication seeds.

    Seeds depend only on experiment seed, condition values, and replication
    index, so serial and parallel execution produce identical sorted results.
    """
    spec.validate()
    conditions = build_conditions(spec)
    worker_count = int(workers if workers is not None else spec.workers)
    if worker_count < 1:
        raise ValueError("workers must be at least 1")
    tasks = [(condition, replication) for condition in conditions for replication in range(1, condition.replications + 1)]
    total = len(tasks)
    completed = 0
    collected: list[tuple[int, int, list[dict[str, object]]]] = []
    start = perf_counter()

    if worker_count == 1:
        for condition, replication in tasks:
            collected.append(_task(condition, replication))
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, total)
    else:
        # Spawn avoids fork-with-threads hazards in Streamlit and scientific BLAS runtimes.
        with ProcessPoolExecutor(max_workers=worker_count, mp_context=mp.get_context("spawn")) as executor:
            futures = {executor.submit(_task, condition, replication): (condition, replication) for condition, replication in tasks}
            for future in as_completed(futures):
                condition, replication = futures[future]
                try:
                    collected.append(future.result())
                except Exception as exc:  # defensive boundary for unexpected worker-level failures
                    rep_seed = replication_seed(condition.condition_seed, replication)
                    rows = [
                        _failure_row(condition, replication, rep_seed, estimator, f"worker: {type(exc).__name__}: {exc}")
                        for estimator in condition.estimators
                    ]
                    collected.append((condition.condition_index, replication, rows))
                completed += 1
                if progress_callback is not None:
                    progress_callback(completed, total)

    collected.sort(key=lambda item: (item[0], item[1]))
    rows = [row for _, _, replication_rows in collected for row in replication_rows]
    results = pd.DataFrame(rows)
    if not results.empty:
        estimator_order = {name: index for index, name in enumerate(spec.estimators)}
        results["_estimator_order"] = results["estimator"].map(estimator_order).fillna(len(estimator_order))
        results = results.sort_values(["condition_index", "replication", "_estimator_order"], kind="stable").drop(columns="_estimator_order").reset_index(drop=True)
    summary = summarize_long_results(results, spec.estimators)
    return ExperimentResult(results, summary, conditions, perf_counter() - start)
