import numpy as np
import pytest

from mc_lab.config import DGPConfig, ExperimentSpec
from mc_lab.diagnostics import treatment_diagnostics
from mc_lab.engine import run_experiment
from mc_lab.generators import DEFAULT_GENERATOR


@pytest.fixture(scope="module")
def randomized_summary():
    spec = ExperimentSpec(
        sample_size=500,
        replications=220,
        estimators=("Difference in means", "OLS regression", "OLS regression (HC3)"),
        seed=101,
        dgp=DGPConfig(),
    )
    return run_experiment(spec).summary.set_index("estimator")


@pytest.fixture(scope="module")
def confounded_summary():
    spec = ExperimentSpec(
        sample_size=500,
        replications=220,
        estimators=(
            "Difference in means",
            "OLS regression",
            "Inverse probability weighting",
            "Doubly robust AIPW",
        ),
        seed=102,
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.0),
    )
    return run_experiment(spec).summary.set_index("estimator")


def test_randomized_ols_is_approximately_unbiased(randomized_summary):
    assert abs(randomized_summary.loc["OLS regression", "bias"]) < 0.04


def test_randomized_difference_in_means_is_approximately_unbiased(randomized_summary):
    assert abs(randomized_summary.loc["Difference in means", "bias"]) < 0.05


def test_nominal_coverage_under_correct_randomized_conditions(randomized_summary):
    coverage = randomized_summary.loc["OLS regression (HC3)", "coverage"]
    assert 0.90 <= coverage <= 0.99


def test_standard_error_calibration_is_sensible(randomized_summary):
    ratio = randomized_summary.loc["OLS regression (HC3)", "se_calibration_ratio"]
    assert 0.8 <= ratio <= 1.2


def test_difference_in_means_is_biased_under_confounding(confounded_summary):
    assert abs(confounded_summary.loc["Difference in means", "bias"]) > 0.4


def test_regression_adjustment_recovers_effect_under_correct_confounding(confounded_summary):
    assert abs(confounded_summary.loc["OLS regression", "bias"]) < 0.05


def test_ipw_recovers_effect_with_correct_propensity_model(confounded_summary):
    assert abs(confounded_summary.loc["Inverse probability weighting", "bias"]) < 0.08


def test_aipw_is_doubly_robust_when_propensity_is_misspecified():
    spec = ExperimentSpec(
        sample_size=600,
        replications=180,
        estimators=("Doubly robust AIPW",),
        seed=103,
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.0),
        estimator_settings={"Doubly robust AIPW": {"propensity_covariates": []}},
    )
    summary = run_experiment(spec).summary.iloc[0]
    assert abs(summary["bias"]) < 0.06
    assert summary["failure_rate"] == 0.0


def test_poor_overlap_creates_unstable_weights():
    clean = DEFAULT_GENERATOR.generate(
        DGPConfig(assignment_mode="logistic", confounding_strength=0.3, propensity_clip=0.001),
        sample_size=2500,
        seed=12345,
    ).data
    poor = DEFAULT_GENERATOR.generate(
        DGPConfig(assignment_mode="logistic", confounding_strength=2.8, propensity_clip=0.001),
        sample_size=2500,
        seed=12345,
    ).data
    clean_diag = treatment_diagnostics(clean, settings={"propensity_clip": 0.001})
    poor_diag = treatment_diagnostics(poor, settings={"propensity_clip": 0.001})
    assert poor_diag["max_weight"] > clean_diag["max_weight"] * 5
    assert poor_diag["effective_sample_size"] < clean_diag["effective_sample_size"] * 0.5
    assert poor_diag["extreme_weight_fraction"] > clean_diag["extreme_weight_fraction"]


def test_binary_outcome_has_known_replication_specific_ate():
    generated = DEFAULT_GENERATOR.generate(
        DGPConfig(outcome_type="binary", outcome_model="heterogeneous", tau0=0.5, tau1=0.4),
        sample_size=400,
        seed=7,
    )
    assert 0 <= generated.true_estimand <= 1
    assert np.isclose(generated.true_estimand, generated.data["true_unit_effect"].mean())


def test_every_registered_estimator_runs_on_clean_continuous_data():
    from mc_lab.estimators import ESTIMATOR_NAMES, estimate_effect

    data = DEFAULT_GENERATOR.generate(DGPConfig(), sample_size=500, seed=888).data
    for estimator in ESTIMATOR_NAMES:
        result = estimate_effect(data, estimator, confidence_level=0.95, settings={"binary_outcome": False})
        assert np.isfinite(result.estimate), estimator
        assert np.isfinite(result.standard_error), estimator
        assert result.ci_lower <= result.ci_upper, estimator


def test_type_i_error_is_close_to_nominal_under_clean_conditions():
    spec = ExperimentSpec(
        mode="type_i",
        sample_size=500,
        replications=300,
        confidence_level=0.95,
        estimators=("OLS regression (HC3)",),
        seed=404,
        dgp=DGPConfig(tau0=2.0),  # backend must override this to zero in type-I mode
    )
    summary = run_experiment(spec).summary.iloc[0]
    assert 0.02 <= summary["type_i_error"] <= 0.08
    assert np.isfinite(summary["type_i_error_mcse"])


def test_power_increases_for_a_large_effect():
    spec = ExperimentSpec(
        mode="power",
        sample_size=250,
        replications=220,
        estimators=("OLS regression (HC3)",),
        seed=405,
        dgp=DGPConfig(tau0=0.0),
        factors={"tau0": (0.0, 0.7)},
    )
    summary = run_experiment(spec).summary.sort_values("factor_tau0")
    null_rejection = float(summary.iloc[0]["rejection_rate"])
    alternative_power = float(summary.iloc[1]["rejection_rate"])
    assert alternative_power > null_rejection + 0.6
    assert alternative_power > 0.8
