import math

import pandas as pd

from metrics import calculate_metrics


def _results(estimates, covered):
    return pd.DataFrame({"estimate": estimates, "covers_true_effect": covered})


def test_bias_calculation():
    metrics = calculate_metrics(_results([1.0, 2.0, 3.0], [True] * 3), true_effect=1.5)
    assert metrics["bias"] == 0.5


def test_rmse_calculation():
    metrics = calculate_metrics(_results([0.0, 2.0], [True, True]), true_effect=1.0)
    assert math.isclose(metrics["rmse"], 1.0)


def test_variance_calculation():
    metrics = calculate_metrics(_results([1.0, 2.0, 3.0], [True] * 3), true_effect=2.0)
    assert math.isclose(metrics["variance"], 1.0)


def test_coverage_calculation():
    metrics = calculate_metrics(_results([1.0, 1.0, 1.0, 1.0], [True, False, True, False]), true_effect=1.0)
    assert math.isclose(metrics["coverage"], 0.5)
