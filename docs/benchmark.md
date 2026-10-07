# Serial vs parallel benchmark

The project includes a small reproducible benchmark:

```bash
python -m scripts.benchmark_parallel
```

Configuration used by the script:

- sample size: 300
- replications: 120
- treatment: logistic confounding with strength 1.0
- estimators: OLS, OLS-HC3, AIPW
- deterministic experiment seed: 2026
- comparison: 1 worker vs 2 workers

Validation-container result:

```text
serial_seconds=1.108
parallel_2_seconds=2.422
speedup_2_workers=0.457x
results_identical=true
```

The two-worker run was slower because process-spawn and serialization overhead dominated this small workload. This result is intentional documentation of measured behavior rather than a claim that parallel execution is always faster.

Heavier studies can amortize startup costs better, especially when per-replication estimation is expensive. Researchers should benchmark their own target design. The default therefore remains `workers=1`.

The important reproducibility result is independent of speed: the benchmark asserts exact equality of serial and parallel replication-level results.
