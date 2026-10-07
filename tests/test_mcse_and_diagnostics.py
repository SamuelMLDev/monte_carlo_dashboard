import math

import numpy as np
import pandas as pd

from mc_lab.metrics import calculate_metrics
from mc_lab.synthetic_eval import evaluate_tabular_synthetic


def test_mcse_formulas_for_bias_and_coverage():
    estimates = np.array([0.8, 1.0, 1.2, 1.1])
    truth = np.ones(4)
    covered = [True, True, False, True]
    frame = pd.DataFrame(
        {
            "estimate": estimates,
            "true_estimand": truth,
            "standard_error": [0.1] * 4,
            "ci_lower": estimates - 0.196,
            "ci_upper": estimates + 0.196,
            "covers_true_effect": covered,
            "null_rejected": [True] * 4,
            "success": [True] * 4,
            "converged": [True] * 4,
        }
    )
    metrics = calculate_metrics(frame)
    errors = estimates - truth
    expected_bias_mcse = np.std(errors, ddof=1) / math.sqrt(4)
    expected_coverage_mcse = math.sqrt(0.75 * 0.25 / 4)
    assert math.isclose(metrics["bias_mcse"], expected_bias_mcse)
    assert math.isclose(metrics["coverage_mcse"], expected_coverage_mcse)
    assert metrics["coverage"] == 0.75


def test_rmse_mcse_is_finite_for_nonzero_error():
    frame = pd.DataFrame(
        {
            "estimate": [0.0, 1.0, 2.0, 1.5, 0.5],
            "true_estimand": [1.0] * 5,
            "standard_error": [0.2] * 5,
            "ci_lower": [-0.4, 0.6, 1.6, 1.1, 0.1],
            "ci_upper": [0.4, 1.4, 2.4, 1.9, 0.9],
            "covers_true_effect": [False, True, False, False, False],
            "null_rejected": [False, True, True, True, False],
            "success": [True] * 5,
            "converged": [True] * 5,
        }
    )
    metrics = calculate_metrics(frame)
    assert metrics["rmse"] > 0
    assert np.isfinite(metrics["rmse_mcse"])
    assert np.isfinite(metrics["empirical_sd_mcse"])


def test_synthetic_data_diagnostics_detect_identical_tables():
    real = pd.DataFrame({"x": [0.0, 1.0, 2.0, 3.0], "z": [1.0, 3.0, 2.0, 4.0], "cat": ["a", "a", "b", "b"]})
    evaluation = evaluate_tabular_synthetic(real, real.copy())
    assert (evaluation["numerical"]["ks_statistic"] == 0).all()
    assert evaluation["correlation"]["mean_absolute_correlation_error"] == 0
    assert not evaluation["pairwise_dependencies"].empty
    assert (evaluation["pairwise_dependencies"]["absolute_error"] == 0).all()
    assert evaluation["novelty"]["duplicate_rate"] == 1.0


def test_paired_relative_rmse_baseline_has_zero_mcse():
    from mc_lab.config import ExperimentSpec
    from mc_lab.engine import run_experiment

    result = run_experiment(
        ExperimentSpec(
            sample_size=120,
            replications=30,
            estimators=("OLS regression", "OLS regression (HC3)"),
            seed=19,
        )
    )
    baseline = result.summary.loc[result.summary["estimator"] == "OLS regression"].iloc[0]
    assert math.isclose(baseline["relative_rmse"], 1.0)
    assert math.isclose(baseline["relative_rmse_mcse"], 0.0, abs_tol=1e-12)
