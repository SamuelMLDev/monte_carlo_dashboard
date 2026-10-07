import numpy as np
import pandas as pd

from mc_lab.empirical import bootstrap_estimator, compare_estimators, prepare_empirical_data


def _empirical_frame():
    rng = np.random.default_rng(123)
    n = 300
    x = rng.normal(size=n)
    treatment = rng.binomial(1, 0.5, size=n)
    outcome = 0.6 * treatment + x + rng.normal(size=n)
    return pd.DataFrame({"outcome": outcome, "treatment": treatment, "age_like": x})


def test_empirical_column_mapping_and_comparison():
    frame = _empirical_frame()
    mapped = prepare_empirical_data(frame, "outcome", "treatment", ["age_like"], treated_value=1)
    assert list(mapped.columns) == ["Y", "A", "X1"]
    comparison = compare_estimators(mapped, ["Difference in means", "OLS regression", "OLS regression (HC3)"])
    assert len(comparison) == 3
    assert comparison["estimate"].notna().all()
    assert comparison["converged"].all()


def test_empirical_bootstrap_is_reproducible():
    frame = _empirical_frame()
    mapped = prepare_empirical_data(frame, "outcome", "treatment", ["age_like"], treated_value=1)
    first = bootstrap_estimator(mapped, "OLS regression", replications=60, seed=77)
    second = bootstrap_estimator(mapped, "OLS regression", replications=60, seed=77)
    assert first == second
    assert first["successful_bootstraps"] == 60
    assert first["bootstrap_se"] > 0
