# Extending the research laboratory

The package is intentionally organized around small plug-in contracts so new methodology can be added without rewriting the Streamlit app or Monte Carlo scheduler.

## 1. Add a data generator

Implement `mc_lab.generators.base.DataGenerator`:

```python
from mc_lab.generators.base import DataGenerator, GeneratedDataset

class MyGenerator(DataGenerator):
    name = "my_generator"

    def generate(self, config, sample_size, seed):
        ...
        return GeneratedDataset(
            data=frame,
            true_estimand=true_value,
            description="...",
        )

    def true_estimand(self, data, config):
        return ...

    def describe(self, config):
        return "mathematical description"
```

The generated `DataFrame` should expose `Y`, `A`, and `X1`, `X2`, ... columns for the current estimator library. Generator-specific truth columns may be retained for diagnostics.

The current `ParametricDataGenerator` is one implementation. Future implementations could wrap:

- CTGAN;
- PAR/CPAR;
- Gaussian copulas;
- externally generated synthetic datasets;
- domain-specific simulators.

Register the generator with `mc_lab.generators.register_generator(name, generator)`, then set `ExperimentSpec.generator` to that registry name and place generator-specific serializable settings in `ExperimentSpec.generator_config`. The Monte Carlo engine resolves generators through this registry, so adding CTGAN/copula/external generators does not require changing the scheduler or estimator loop.

The grid builder currently provides first-class factor mappings for the parametric DGP. A custom generator can still run fixed configurations immediately; generator-specific factorial controls can be added by extending the grid factor resolver without changing the execution engine.

## 2. Add an estimator

Implement `mc_lab.estimators.base.Estimator`:

```python
from mc_lab.estimators.base import EstimateResult, Estimator

class MyEstimator(Estimator):
    name = "My estimator"

    def estimate(self, data, confidence_level=0.95, settings=None):
        ...
        return EstimateResult(
            estimate=estimate,
            standard_error=se,
            ci_lower=lower,
            ci_upper=upper,
            converged=True,
            status="ok",
            diagnostics={"diagnostic_name": value},
        )
```

Then add the class to `_ESTIMATOR_CLASSES` in `mc_lab/estimators/registry.py`.

### Estimator requirements

- State the estimand explicitly.
- Do not silently discard convergence failures.
- Return `NaN` uncertainty when a valid SE/CI is genuinely unavailable rather than inventing one.
- Put compact scalar diagnostics in `diagnostics`; the engine flattens them into replication-level columns with a `diag_` prefix.
- Add a statistical recovery test under a known DGP and a failure-path test where relevant.

## 3. Add a metric

Core metrics live in `mc_lab.metrics.calculate_metrics`. A new metric should:

1. define its valid denominator (all attempts, successful estimates, finite CIs, etc.);
2. return `NaN` if the metric is not meaningful;
3. add an appropriate MCSE when the metric is reported as a Monte Carlo performance estimate;
4. receive a test with a hand-calculable example.

Summary rows are assembled in `summarize_long_results`.

## 4. Add a grid factor

Grid factors are resolved in `mc_lab.grid._apply_factor`. Add a human-readable entry to `FACTOR_LABELS`, map the factor to a `DGPConfig` or experiment field, and add a Cartesian-product test.

For non-factorial ordered stress sequences, follow the `stress_level` pattern: one factor value can resolve to a complete predefined DGP difficulty level.

## 5. Add a visualization

All figures live in `mc_lab.plots` and return a `matplotlib.figure.Figure`. Keep functions pure where possible: accept tidy DataFrames/arrays and return the figure without calling Streamlit.

A research figure should include:

- explicit axis labels;
- estimator labels where methods are compared;
- relevant truth/nominal reference lines;
- a concise title;
- a caption when an interpretive caveat matters.

The Streamlit layer should only choose inputs and render the figure.

## 6. Add a diagnostic

Diagnostics should be separated from estimation. `mc_lab.diagnostics` currently contains propensity, balance, weight, and OLS residual diagnostics. A new diagnostic should accept a concrete dataset or result table and return structured values that are independently testable.

## 7. Add a report component

`mc_lab.reports.build_html_report` is intentionally static and deterministic. Add tables/figures based on already computed results; do not add narrative generation or external services. The report should remain factual and neutral.

## 8. Add synthetic-data evaluation

`mc_lab.synthetic_eval` is optional and does not participate in the core Monte Carlo engine. Add tabular metrics to `evaluate_tabular_synthetic` or create a separate module for explicitly sequential diagnostics.

The `SequentialEvaluationConfig` scaffold is reserved for future sequence-aware comparisons such as unigram/bigram/trigram frequencies, Jensen-Shannon divergence, and sequence-length Wasserstein distance. Do not infer sequence semantics from arbitrary tabular columns.

## 9. Compatibility layer

The repository deliberately retains the original top-level modules (`simulation.py`, `estimators.py`, `metrics.py`, `plots.py`, `utils.py`). Avoid breaking those import paths unless a major-version migration is intended. New functionality belongs in `mc_lab/`.
