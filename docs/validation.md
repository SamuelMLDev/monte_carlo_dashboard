# Validation record

This file records validation performed on the repository build supplied with the project. It is not a guarantee for arbitrary future code changes or environments.

## Automated tests

Command:

```bash
pytest -q
```

Result in the final build environment:

```text
44 passed
```

The suite includes compatibility/regression tests plus research-framework tests for deterministic hierarchical seeds, serial/parallel identity, factorial grids, manifest round trips, failure retention, estimator capability checks, metric/MCSE formulas, empirical bootstrap reproducibility, synthetic-data diagnostics, statistical recovery, coverage, Type-I error, power, poor-overlap weight behavior, and IPW/AIPW behavior.

## Full built-in study presets

All six built-in research presets were executed at their configured replication counts in single-worker mode:

| Preset | Conditions | Estimator-fit rows | Failures |
| --- | ---: | ---: | ---: |
| Why regression adjustment matters under confounding | 1 | 2,000 | 0 |
| Coverage improves with correct standard errors | 4 | 4,000 | 0 |
| Poor overlap destabilizes IPW | 4 | 8,000 | 0 |
| Heavy-tailed errors stress OLS inference | 4 | 6,000 | 0 |
| Bias under nonlinear model misspecification | 3 | 4,500 | 0 |
| Doubly robust estimation when one nuisance model is misspecified | 1 | 1,500 | 0 |

Total: **26,000 estimator fits, 0 recorded failures** in these six configured examples.

## Additional end-to-end workflow checks

The following were executed directly in addition to the automated suite:

- Binary-outcome simulation with logistic confounding and seven supported estimators: 245 estimator-fit rows, 0 failures.
- Seven-level Method Stress Test with OLS-HC3: 7 conditions, 210 estimator-fit rows, 0 failures.
- Confidence-interval calibration grid at nominal 80%, 90%, 95%, and 99% levels.
- JSON and YAML manifest export/import equality.
- Replication and summary CSV serialization.
- Self-contained HTML research report rendering.
- Sample-size grid showing lower OLS-HC3 empirical SD at `n=1000` than at `n=100` under the clean randomized DGP.
- Unsupported Huber/binary combinations retained as explicit failure rows rather than silently producing results.

Representative uncertainty check from the final validation run:

```text
n=100:  empirical SD ≈ 0.2088, RMSE ≈ 0.2080
n=1000: empirical SD ≈ 0.0654, RMSE ≈ 0.0652
```

These are fixed-seed validation examples under the configured DGP, not universal performance claims.

## Statistical regression checks

The test suite verifies, with tolerance-based fixed-seed simulations:

- randomized OLS is approximately unbiased;
- randomized difference in means is approximately unbiased;
- HC3 coverage is near nominal under clean randomized conditions;
- unadjusted difference in means is biased under confounding;
- correctly specified regression adjustment recovers the effect under confounding;
- correctly specified IPW recovers the effect;
- AIPW remains well behaved when the propensity nuisance model is deliberately misspecified but the outcome nuisance model is correct;
- poor overlap produces larger weights and lower effective sample size;
- Type-I error is close to nominal in a clean null design;
- power rises strongly under a large alternative.

## Reproducibility/export checks

The following were exercised directly or by automated tests:

- JSON manifest round trip;
- YAML manifest round trip;
- deterministic condition/replication seeds;
- condition seeds independent of factor declaration order;
- exact serial versus two-worker replication-result equality;
- CSV serialization;
- HTML report rendering;
- empirical bootstrap reproducibility;
- configuration restoration from exported manifests.

## Parallel benchmark

Command:

```bash
python -m scripts.benchmark_parallel
```

Final build result:

```text
serial_seconds=1.108
parallel_2_seconds=2.422
speedup_2_workers=0.457x
results_identical=true
```

The workload is intentionally small and does not amortize process startup. The default therefore remains one worker. Exact equality is the important reproducibility result; see `docs/benchmark.md`.

## Package/build configuration

Editable package construction was validated locally with build isolation disabled because the execution environment cannot reach package indexes:

```bash
python -m pip install -e . --no-deps --no-build-isolation
```

The installed package reported version `0.3.0`.

## Local Windows validation

The published repository was subsequently validated in a Windows PowerShell environment with Python 3.14.2.

Commands and observed results:

```text
python -m pytest -q
44 passed

ruff check .
All checks passed!

streamlit run app.py
Uvicorn server started on :::8501
Local URL: http://localhost:8501
```

The application therefore has a confirmed local Streamlit launch in addition to the backend validation recorded above. The current source also replaces Streamlit's deprecated `use_container_width=True` argument with `width="stretch"`.

## Continuous integration

The repository includes a GitHub Actions workflow that runs Ruff and the full pytest suite on Python 3.10, 3.11, 3.12, 3.13, and 3.14 for pushes and pull requests targeting `main`.
