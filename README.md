# Monte Carlo Research Laboratory

[![Tests](https://github.com/SamuelMLDev/monte_carlo_dashboard/actions/workflows/tests.yml/badge.svg)](https://github.com/SamuelMLDev/monte_carlo_dashboard/actions/workflows/tests.yml)

A local-first Streamlit application and Python simulation framework for designing, running, diagnosing, and reproducing Monte Carlo studies of treatment-effect estimators.

The project is intended for quantitative methodology exploration rather than estimator marketing. It reports how methods behave **under the assumptions you configure**, preserves failed fits, quantifies Monte Carlo uncertainty, and exports machine-readable experiment manifests that can be rerun exactly.

> Primary local workflow: `streamlit run app.py`

## What the laboratory supports

The application has three deliberately separate workflows:

1. **Parametric Monte Carlo simulation** — generated data have a known structural truth, so bias, RMSE, coverage, power, and calibration can be evaluated directly.
2. **Empirical Data Exploration** — uploaded CSV data can be analyzed with the same estimator/diagnostic library, but population truth is not inferred. An externally known reference estimand can optionally be supplied for error/interval-containment checks.
3. **Synthetic-data diagnostics** — optional real-vs-synthetic tabular comparisons assess marginal similarity, dependence preservation, novelty, and train-on-synthetic/test-on-real utility. This is not treated as Monte Carlo truth evaluation.

The simulation engine is separated from the data generator interface so future generators such as CTGAN, PAR/CPAR, Gaussian copulas, or externally generated synthetic datasets can be plugged in without rewriting the Monte Carlo engine.

## Major features

### Scenario / DGP builder

Configure:

- continuous or binary outcomes;
- linear, quadratic, nonlinear, interaction, or heterogeneous-effect outcome mechanisms;
- 1–10 covariates from Normal, Uniform, Bernoulli, or correlated multivariate Normal distributions;
- randomized treatment or logistic propensity assignment;
- weak, moderate, strong, and poor-overlap confounding regimes;
- Gaussian, Student-t, skewed, or heteroskedastic continuous-outcome errors;
- known constant or covariate-dependent treatment effects;
- reusable DGP and full-study presets.

For continuous outcomes the structural template is

$$
Y = \beta_0 + \beta^\top X + f(X) + A\tau(X) + \varepsilon,
\qquad
\tau(X)=\tau_0 + \tau_1 X_1,
$$

with optional quadratic/nonlinear structure. Logistic treatment assignment uses

$$
P(A=1\mid X)=\mathrm{logit}^{-1}(\alpha_0 + \alpha_1X_1 + 0.35\alpha_1X_2),
$$

when a second covariate is available. For binary outcomes,

$$
P(Y=1\mid A,X)=\mathrm{logit}^{-1}\{\beta_0+\beta^\top X+A\tau(X)\}.
$$

The simulation truth is the **replication-specific sample average treatment effect (sample ATE)** implied by the structural potential-outcome means. This keeps the ground truth well defined under heterogeneous and binary-outcome designs.

### Estimator plug-in library

Every estimator implements one interface and returns:

- point estimate;
- standard error;
- confidence interval;
- convergence status;
- status message;
- optional diagnostics.

Included methods:

- Difference in means
- OLS regression adjustment
- OLS regression with HC3 robust covariance
- Inverse probability weighting (IPW)
- Stabilized / normalized IPW
- Outcome regression (parametric g-computation)
- Doubly robust AIPW with deterministic two-fold cross-fitting
- Huber robust regression retained for continuous-outcome stress studies

The core library intentionally stops here rather than adding superficially implemented matching or machine-learning estimators with poorly justified uncertainty estimates.

### Factorial experiment grids

A study can vary factors such as:

```text
sample_size:          [100, 250, 500, 1000]
noise_sd:             [0.5, 1.0, 2.0]
confounding_strength: [0.0, 0.9, 1.6]
tau0:                 [0.0, 0.25, 0.5, 0.75]
confidence_level:     [0.80, 0.90, 0.95, 0.99]
```

The engine constructs the Cartesian product, reports condition/dataset/fit counts, evaluates every selected estimator on the **same generated dataset within a replication**, and returns tidy replication-level and condition-estimator summary tables.

### Performance metrics and Monte Carlo uncertainty

For successful replications the summary includes, where meaningful:

- bias and absolute bias;
- relative bias;
- MSE and RMSE;
- empirical variance and empirical SD;
- mean estimator-reported SE;
- SE calibration ratio = mean estimated SE / empirical SD;
- confidence-interval coverage;
- average CI width;
- null rejection rate;
- Type-I error under a null DGP;
- power under non-null DGPs;
- convergence and failure rates;
- unavailable-SE rate;
- extreme-weight replication rate;
- paired relative RMSE versus the first selected estimator;
- RMSE rank within each simulation condition.

The project also reports **Monte Carlo standard errors (MCSEs)** for finite-replication uncertainty. Examples include

$$
MCSE(\widehat{\text{coverage}})=\sqrt{\hat p(1-\hat p)/R}
$$

and

$$
MCSE(\widehat{\text{bias}})=SD(\hat\tau_r-\tau_r)/\sqrt{R}.
$$

RMSE uses a delta-method MCSE from the Monte Carlo distribution of squared errors. Variance/SD MCSEs use empirical influence-function calculations rather than assuming normally distributed Monte Carlo estimates.

### Diagnostics

The Diagnostics page can regenerate any selected replication exactly and show:

- fitted propensity distributions by treatment group;
- overlap/common-support width;
- standardized mean differences before and after weighting;
- inverse-probability weight distribution;
- maximum weight;
- effective sample size;
- fraction of weights above the descriptive extreme-weight threshold of 10;
- OLS residual plots;
- a descriptive heteroskedasticity indicator;
- all failed estimator fits and non-OK statuses.

Failed replications are retained in the replication-level table and exports; they are never silently deleted.

### Calibration and stress-test modes

Built-in study modes include:

- standard simulation;
- Type-I error calibration (structural effect forced to zero);
- power studies over effect size and/or sample size;
- confidence-interval calibration over nominal confidence levels;
- seven-level cumulative **Method Stress Test** from clean randomized Gaussian data through higher noise, confounding, nonlinearity, heteroskedasticity, poor overlap, and heavy tails.

### Research visualizations

Matplotlib figures include:

- estimator sampling distributions;
- replication-level confidence intervals;
- bias vs sample size;
- RMSE vs sample size;
- coverage vs sample size with nominal reference;
- power/rejection curves;
- bias–variance tradeoff;
- factor heatmaps;
- forest-style estimator summaries;
- Monte Carlo convergence plots for running bias, RMSE, and coverage with MC uncertainty bands where appropriate;
- propensity overlap, balance, weights, and residual diagnostics.

### Reproducibility manifests

Every completed experiment can export JSON and YAML manifests containing:

- full study specification;
- DGP settings;
- estimator settings;
- factors and resolved conditions;
- sample sizes and replications;
- experiment seed;
- deterministic condition seeds;
- deterministic replication-seed rule;
- confidence level;
- application version;
- Python version;
- package versions;
- platform metadata;
- UTC timestamp;
- Git commit hash when available;
- measured run time.

A manifest can be loaded back into the Streamlit controls and rerun.

Random numbers use NumPy's modern `Generator` / `SeedSequence` API. Child seeds are derived from stable BLAKE2b hashes:

```text
experiment seed
    └── condition seed = hash(experiment seed, sorted factor values)
            └── replication seed = hash(condition seed, replication index)
```

Therefore results do not depend on process completion order. Serial and parallel execution are regression-tested for exact equality.

### Local parallel execution

`ProcessPoolExecutor` provides optional local process parallelism. The default remains one worker because small simulations can be slower in parallel due to process-spawn/serialization overhead. See `docs/benchmark.md` and run:

```bash
python -m scripts.benchmark_parallel
```

No cloud, distributed framework, database, authentication, API, or LLM service is used.

### Automatic HTML research report

A self-contained HTML report can be downloaded with:

- title and methodological description;
- mathematical DGP;
- estimators and simulation design;
- full main-results table;
- metric definitions;
- selected plots and diagnostics;
- complete reproducibility manifest;
- warnings and limitations.

The report is factual and deterministic; it does not use generative AI to interpret results.

### Session experiment history

Up to eight experiments are retained in Streamlit session state for within-session comparison of study design, estimator set, mean absolute bias, RMSE, and coverage. No database is required.

## Built-in study presets

Six research-oriented examples are provided:

1. **Why regression adjustment matters under confounding**
2. **Coverage improves with correct standard errors**
3. **Poor overlap destabilizes IPW**
4. **Heavy-tailed errors stress OLS inference**
5. **Bias under nonlinear model misspecification**
6. **Doubly robust estimation when one nuisance model is misspecified**

DGP-only presets are also available for clean randomization, weak/moderate/strong confounding, nonlinear outcomes, heterogeneous effects, poor overlap, heavy tails, heteroskedasticity, and combined misspecification stress.

## Installation

Python 3.10–3.14 is supported by the current project configuration. Python 3.14.2 has been validated locally.

### `venv` / pip

```bash
python -m venv .venv
source .venv/bin/activate        # Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For development tooling:

```bash
pip install -e '.[dev]'
```

### Conda

```bash
conda env create -f environment.yml
conda activate monte-carlo-research-lab
```

## Running the application

From the repository root:

```bash
streamlit run app.py
```

The app is intentionally a single local Streamlit process; there is no separate frontend/backend service.

## Tests and code-quality checks

Run the complete test suite:

```bash
pytest -q
```

Validated result for this build: **44 tests passed**. See [`docs/validation.md`](docs/validation.md) for the recorded statistical, preset, export, and environment checks.

Optional development checks after `pip install -e '.[dev]'`:

```bash
ruff check .
ruff format --check .
mypy mc_lab
```

Run all built-in research presets programmatically:

```bash
python -m scripts.validate_presets
```

Run the deterministic serial-vs-parallel benchmark:

```bash
python -m scripts.benchmark_parallel
```

The automated tests cover:

- legacy/original dashboard behavior;
- exact fixed-seed reproducibility;
- deterministic serial/parallel equality;
- factorial grid construction and order-independent condition seeds;
- configuration/manifest JSON and YAML round trips;
- metric and MCSE formulas;
- failure retention;
- randomized OLS and difference-in-means calibration;
- confounding bias of unadjusted means;
- regression-adjustment recovery under correct confounding adjustment;
- IPW recovery under correctly specified propensity models;
- AIPW behavior when one nuisance model is deliberately misspecified;
- poor-overlap weight instability;
- nominal coverage/type-I behavior;
- power increase under a large alternative;
- binary-outcome structural truth;
- empirical-data mapping and reproducible bootstrap;
- synthetic-data diagnostics.

## Statistical implementation notes

### Difference in means

$$
\hat\tau = \bar Y_1-\bar Y_0,
\qquad
SE(\hat\tau)=\sqrt{s_1^2/n_1+s_0^2/n_0}.
$$

Intervals use a normal critical value. The estimator is intentionally unadjusted and is expected to be biased under confounded assignment.

### OLS / OLS-HC3

The adjustment model is

$$
Y=\beta_0+\tau A+\beta^\top X+e.
$$

Conventional OLS and HC3 covariance variants are both available. Under deliberately nonlinear DGPs, this model can be misspecified; that is a feature of the simulation design, not silently repaired by the application.

### Outcome regression

Continuous outcomes use linear g-computation. Binary outcomes use logistic outcome regression and average the fitted risk difference

$$
\hat\tau = n^{-1}\sum_i \{\hat m_1(X_i)-\hat m_0(X_i)\}.
$$

### IPW

The propensity score is fit by an unpenalized logistic model. The unnormalized ATE estimate is

$$
\hat\tau_{IPW}=n^{-1}\sum_i \left(\frac{A_iY_i}{\hat e(X_i)}-\frac{(1-A_i)Y_i}{1-\hat e(X_i)}\right).
$$

The stabilized variant uses normalized/Hájek weighted means. Weight clipping is explicit and configurable.

### AIPW

The doubly robust estimator uses deterministic two-fold cross-fitting with the score

$$
\hat m_1(X)-\hat m_0(X)
+\frac{A\{Y-\hat m_1(X)\}}{\hat e(X)}
-\frac{(1-A)\{Y-\hat m_0(X)\}}{1-\hat e(X)}.
$$

See `docs/methodology.md` for the exact implementation assumptions and inference caveats.

## Architecture

```text
                                ┌──────────────────────────┐
                                │       Streamlit UI       │
                                │         app.py           │
                                └────────────┬─────────────┘
                                             │ ExperimentSpec
                                             ▼
┌──────────────────┐   registry   ┌──────────────────────────┐   registry   ┌──────────────────┐
│ DataGenerator    │◄────────────►│ Monte Carlo engine/grid  │◄────────────►│ Estimator        │
│ interface        │              │ mc_lab/engine.py         │              │ interface        │
└────────┬─────────┘              │ mc_lab/grid.py           │              └────────┬─────────┘
         │                        └────────────┬─────────────┘                       │
         ▼                                     │                                    ▼
┌──────────────────┐                           ▼                           ┌──────────────────┐
│ Parametric DGP   │                 ┌────────────────────┐               │ OLS/IPW/AIPW/...│
│ generators/      │                 │ replication table  │               │ estimators/      │
└──────────────────┘                 └──────────┬─────────┘               └──────────────────┘
                                               │
                  ┌────────────────────────────┼────────────────────────────┐
                  ▼                            ▼                            ▼
          ┌───────────────┐            ┌───────────────┐            ┌───────────────┐
          │ Metrics/MCSE  │            │ Diagnostics   │            │ Manifest/report│
          └───────────────┘            └───────────────┘            └───────────────┘
```

The engine depends only on the generator and estimator contracts. New implementations register with their respective registries rather than adding hard-coded branches to the engine.

## Project structure

```text
monte_carlo_dashboard/
├── app.py                         # Streamlit scientific UI
├── simulation.py                  # Backwards-compatible original simulation API
├── estimators.py                  # Backwards-compatible original estimator API
├── metrics.py                     # Backwards-compatible original metrics API
├── plots.py                       # Backwards-compatible original plotting API
├── utils.py                       # Backwards-compatible serialization helpers
├── mc_lab/
│   ├── config.py                  # DGP/study/condition dataclasses
│   ├── engine.py                  # deterministic serial/parallel Monte Carlo engine
│   ├── grid.py                    # factorial design expansion + stress levels
│   ├── seeds.py                   # hierarchical deterministic seed derivation
│   ├── metrics.py                 # research metrics + MCSEs
│   ├── diagnostics.py             # overlap, balance, weights, residuals
│   ├── plots.py                   # research-quality matplotlib figures
│   ├── manifests.py               # JSON/YAML reproducibility manifests
│   ├── reports.py                 # self-contained HTML reports
│   ├── presets.py                 # DGP + full-study presets
│   ├── empirical.py               # uploaded empirical-data workflow
│   ├── synthetic_eval.py          # optional real/synthetic diagnostics
│   ├── generators/
│   │   ├── base.py                # DataGenerator extension interface
│   │   ├── parametric.py          # built-in configurable structural DGP
│   │   └── registry.py            # generator registration
│   └── estimators/
│       ├── base.py                # Estimator / EstimateResult contracts
│       ├── core.py                # means, OLS, HC3, outcome regression, Huber
│       ├── causal.py              # propensity, IPW, stabilized IPW, AIPW
│       └── registry.py            # estimator registration
├── tests/                         # statistical + architectural regression tests
│   ├── test_dgp_validation.py
│   └── ...
├── scripts/
│   ├── validate_presets.py        # execute all built-in studies
│   └── benchmark_parallel.py      # deterministic serial/parallel benchmark
├── docs/
│   ├── methodology.md             # assumptions, estimators, metrics
│   ├── reproducibility.md         # seeds, manifests, versions
│   ├── extending.md               # add generators/estimators/metrics/plots
│   ├── benchmark.md               # measured local parallel benchmark
│   ├── validation.md              # checks performed for this build
│   └── screenshots/README.md      # screenshot placeholders/instructions
├── example_config.json            # importable bare ExperimentSpec example
├── pyproject.toml                 # package/test/lint/type-check configuration
├── requirements.txt
├── requirements-dev.txt
├── environment.yml
└── README.md
```

## Extending the laboratory

Detailed extension instructions are in `docs/extending.md`.

### New data generator

Implement `mc_lab.generators.base.DataGenerator`:

```python
class MyGenerator(DataGenerator):
    name = "my_generator"

    def generate(self, config, sample_size, seed): ...
    def true_estimand(self, data, config): ...
    def describe(self, config): ...
```

Then register it with `register_generator(...)`. No engine rewrite is required.

### New estimator

Implement `mc_lab.estimators.base.Estimator` and return an `EstimateResult`, then add/register the estimator in the registry. Diagnostics should be returned in the result's `diagnostics` mapping so they can flow into replication-level exports.

## Synthetic-data evaluation scaffold

When real and synthetic tabular datasets are supplied, the optional module supports:

- numeric KS statistic / complement and Wasserstein distance;
- categorical total-variation distance / complement;
- aggregate correlation preservation;
- pairwise correlation comparison;
- exact-row duplicate/novelty rate;
- real/synthetic sample-size comparison;
- random-forest train-on-synthetic/test-on-real utility.

`SequentialEvaluationConfig` intentionally reserves a separate future path for explicitly sequential data. Planned sequence-specific diagnostics include unigram/bigram/trigram comparisons, Jensen–Shannon divergence, and sequence-length Wasserstein distance. They are not forced into the tabular Monte Carlo workflow.

## Limitations and scientific cautions

- The implemented causal estimand is the **sample ATE**. ATT/ATC and other estimands are not yet implemented.
- Treatment is binary; multi-arm, continuous, clustered, longitudinal, survival, and missing-data designs are outside the current engine.
- IPW within-replication SEs use influence-function approximations that treat the fitted propensity model as fixed; this can understate uncertainty in some finite samples.
- AIPW uses simple parametric nuisance models and deterministic two-fold cross-fitting. It is not a general semiparametric ML framework.
- Under heterogeneous effects the truth varies by replication because it is the sample ATE. Some conventional estimator SE formulas are superpopulation-style approximations, so coverage in these designs should be interpreted accordingly.
- Binary-outcome structural `tau` parameters operate on the log-odds scale, while the evaluated ATE is a risk difference.
- Propensity clipping changes the numerical behavior of weighting estimators and must be reported as part of the estimator configuration.
- Poor overlap can make weighting estimators unstable even when no numerical exception occurs; weight diagnostics should be inspected.
- Robust Huber regression is included for continuous-outcome robustness studies, not as a general causal estimator for binary outcomes.
- Parallelism is local-process only and can be slower than serial execution for small jobs.
- The synthetic-data module is diagnostic scaffolding, not a privacy guarantee or full synthetic-data validation standard.
- The application does not choose a universally “best” estimator. Performance rankings are condition-specific.

## Further documentation

- [`docs/methodology.md`](docs/methodology.md) — mathematical assumptions and estimator details
- [`docs/reproducibility.md`](docs/reproducibility.md) — deterministic seed hierarchy and manifests
- [`docs/extending.md`](docs/extending.md) — adding generators, estimators, metrics, and figures
- [`docs/benchmark.md`](docs/benchmark.md) — serial/parallel benchmark methodology
- [`docs/validation.md`](docs/validation.md) — tests, preset runs, exports, benchmark, and launch limitation


## License

This project is licensed under the [MIT License](LICENSE).
