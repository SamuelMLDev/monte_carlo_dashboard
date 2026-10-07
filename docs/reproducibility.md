# Reproducibility model

## Design goal

A simulation result should not change because replications ran in a different order, because a condition appeared earlier or later in a factor grid, or because local parallel workers finished in a different sequence.

## Modern NumPy RNG

All generated datasets use `numpy.random.Generator` created through `numpy.random.default_rng` / `SeedSequence`. The code does not rely on mutable global NumPy random state.

## Hierarchical deterministic seeds

The seed hierarchy is:

```text
experiment seed
  -> condition seed
       -> replication seed
```

`mc_lab.seeds.stable_seed` builds 64-bit child seeds with BLAKE2b over a canonical JSON representation.

### Condition seed

A condition seed depends on:

- the experiment seed;
- the string label `"condition"`;
- the sorted resolved factor-value mapping.

It does **not** depend on the condition's display index. Consequently, changing factor declaration order does not change the seed assigned to the same resolved factor combination.

### Replication seed

A replication seed depends on:

- the condition seed;
- the label `"replication"`;
- the 1-based replication index.

Every replication can therefore be regenerated independently.

## Fair multi-estimator comparison

Within a replication, the DGP is generated once and every selected estimator is applied to that same dataset. Estimators do not receive separate dataset seeds. This ensures that differences between methods are not contaminated by simulation-noise differences across independently generated samples.

## Serial vs parallel execution

Parallel runs use `ProcessPoolExecutor` with the `spawn` multiprocessing context. Each submitted replication already has its deterministic seed. Results are sorted by condition, replication, and estimator order after workers finish.

The automated test suite checks exact DataFrame equality between serial and two-worker parallel runs.

## Manifest contents

Every completed experiment can export JSON or YAML containing:

- manifest schema and application version;
- study title/description;
- complete DGP configuration;
- estimator names and estimator nuisance-model settings;
- base sample size, replications, confidence level, and null value;
- experiment factors and their values;
- experiment seed;
- every resolved condition and its deterministic condition seed;
- seed-derivation rules;
- Python version and platform;
- package versions;
- UTC timestamp;
- Git commit hash when the repository is under Git and `git` is available;
- optional elapsed runtime.

The manifest intentionally stores enough information to reconstruct `ExperimentSpec` exactly.

## Importing a previous experiment

In the Streamlit sidebar, use **Load Experiment Configuration** and upload the exported JSON or YAML manifest. The app restores the study controls. Programmatically:

```python
from mc_lab.manifests import manifest_from_text, spec_from_manifest

manifest = manifest_from_text(open("experiment_manifest.json").read(), "experiment_manifest.json")
spec = spec_from_manifest(manifest)
```

A bare `ExperimentSpec` JSON/YAML mapping is also accepted for convenience.

## Reproducing one diagnostic replication

```python
from mc_lab.diagnostics import generate_diagnostic_dataset

condition = result.conditions[0]
data = generate_diagnostic_dataset(condition, replication=17)
```

This uses the exact condition and replication seed from the hierarchy.

## Why the manifest records versions

Statistical libraries can change solver behavior, covariance calculations, defaults, or numerical tolerances across versions. Seed reproducibility is necessary but not always sufficient for bitwise reproduction across environments. Recording package versions and platform information makes such differences auditable.

## Timestamp caveat

The manifest timestamp documents when a run was performed. It is metadata, not an input to simulation randomness; rerunning the same study produces the same simulation results even though the new manifest has a different creation timestamp.
