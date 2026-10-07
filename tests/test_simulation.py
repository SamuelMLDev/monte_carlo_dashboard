import numpy as np
import pandas as pd

from estimators import difference_in_means, ols_regression
from metrics import calculate_metrics
from simulation import SimulationConfig, run_monte_carlo


def test_reproducibility_with_fixed_seed():
    config = SimulationConfig(sample_size=100, replications=20, seed=123)
    first = run_monte_carlo(config)
    second = run_monte_carlo(config)
    pd.testing.assert_frame_equal(first, second)


def test_simulation_result_dimensions():
    config = SimulationConfig(sample_size=100, replications=17, seed=4)
    results = run_monte_carlo(config)
    assert results.shape == (17, 6)
    assert list(results.columns) == [
        "replication",
        "estimate",
        "standard_error",
        "ci_lower",
        "ci_upper",
        "covers_true_effect",
    ]


def test_ols_recovers_known_effect_in_large_correctly_specified_sample():
    rng = np.random.default_rng(2026)
    n = 20_000
    x = rng.normal(size=n)
    a = rng.binomial(1, 0.5, size=n)
    tau = 0.7
    y = tau * a + 1.0 * x + rng.normal(scale=1.0, size=n)
    data = pd.DataFrame({"Y": y, "A": a, "X": x})
    result = ols_regression(data)
    assert abs(result.estimate - tau) < 0.04


def test_difference_in_means_recovers_effect_when_no_covariate_signal():
    rng = np.random.default_rng(9)
    n = 20_000
    a = rng.binomial(1, 0.5, size=n)
    tau = 0.5
    y = tau * a + rng.normal(size=n)
    data = pd.DataFrame({"Y": y, "A": a, "X": np.zeros(n)})
    result = difference_in_means(data)
    assert abs(result.estimate - tau) < 0.04


def test_larger_sample_generally_reduces_ols_uncertainty():
    small = SimulationConfig(sample_size=100, replications=150, estimator="OLS regression", seed=77)
    large = SimulationConfig(sample_size=1000, replications=150, estimator="OLS regression", seed=77)
    small_metrics = calculate_metrics(run_monte_carlo(small), small.true_effect)
    large_metrics = calculate_metrics(run_monte_carlo(large), large.true_effect)
    assert large_metrics["std_estimate"] < small_metrics["std_estimate"]
    assert large_metrics["rmse"] < small_metrics["rmse"]
