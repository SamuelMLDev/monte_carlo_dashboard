"""Small deterministic serial-vs-parallel benchmark."""

from __future__ import annotations

from dataclasses import replace
from time import perf_counter

from mc_lab.config import DGPConfig, ExperimentSpec
from mc_lab.engine import run_experiment


def main() -> None:
    spec = ExperimentSpec(
        title="Parallel benchmark",
        sample_size=300,
        replications=120,
        estimators=("OLS regression", "OLS regression (HC3)", "Doubly robust AIPW"),
        seed=2026,
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.0),
    )
    timings = {}
    serial_result = None
    for workers in (1, 2):
        start = perf_counter()
        result = run_experiment(replace(spec, workers=workers), workers=workers)
        timings[workers] = perf_counter() - start
        if serial_result is None:
            serial_result = result
        else:
            if not serial_result.replication_results.equals(result.replication_results):
                raise RuntimeError("Parallel results differ from serial results")
    speedup = timings[1] / timings[2]
    print(f"serial_seconds={timings[1]:.3f}")
    print(f"parallel_2_seconds={timings[2]:.3f}")
    print(f"speedup_2_workers={speedup:.3f}x")
    print("results_identical=true")


if __name__ == "__main__":
    main()
