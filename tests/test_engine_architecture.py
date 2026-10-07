import json

import pandas as pd

from mc_lab.config import DGPConfig, ExperimentSpec
from mc_lab.engine import run_experiment
from mc_lab.grid import build_conditions, estimate_workload
from mc_lab.manifests import (
    create_manifest,
    manifest_from_text,
    manifest_to_json,
    manifest_to_yaml,
    spec_from_manifest,
)
from mc_lab.presets import stress_test_spec
from mc_lab.reports import build_html_report


def test_experiment_grid_is_cartesian_product():
    spec = ExperimentSpec(
        replications=10,
        factors={
            "sample_size": (100, 250),
            "noise_sd": (0.5, 1.0, 2.0),
            "confounding_strength": (0.0, 0.9),
        },
    )
    conditions = build_conditions(spec)
    assert len(conditions) == 12
    workload = estimate_workload(spec)
    assert workload == {"conditions": 12, "datasets": 120, "estimator_fits": 120}


def test_condition_seeds_do_not_depend_on_factor_declaration_order():
    a = ExperimentSpec(seed=9, factors={"sample_size": (100, 200), "noise_sd": (1.0, 2.0)})
    b = ExperimentSpec(seed=9, factors={"noise_sd": (1.0, 2.0), "sample_size": (100, 200)})
    seeds_a = {tuple(sorted(c.factor_values.items())): c.condition_seed for c in build_conditions(a)}
    seeds_b = {tuple(sorted(c.factor_values.items())): c.condition_seed for c in build_conditions(b)}
    assert seeds_a == seeds_b


def test_parallel_and_serial_results_are_identical():
    spec = ExperimentSpec(
        sample_size=100,
        replications=12,
        estimators=("OLS regression", "Doubly robust AIPW"),
        seed=99,
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=0.8),
    )
    serial = run_experiment(spec, workers=1)
    parallel = run_experiment(spec, workers=2)
    pd.testing.assert_frame_equal(serial.replication_results, parallel.replication_results)
    pd.testing.assert_frame_equal(serial.summary, parallel.summary)


def test_failure_rows_are_never_silently_discarded():
    spec = ExperimentSpec(sample_size=100, replications=7, estimators=("Unknown estimator",), seed=3)
    result = run_experiment(spec)
    assert len(result.replication_results) == 7
    assert len(result.failures) == 7
    assert result.summary.iloc[0]["failure_rate"] == 1.0
    assert result.replication_results["error_message"].str.contains("Unknown estimator").all()


def test_manifest_json_yaml_roundtrip_preserves_study_spec():
    spec = ExperimentSpec(
        title="Round trip",
        sample_size=250,
        replications=20,
        estimators=("OLS regression", "Stabilized IPW"),
        seed=777,
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.2),
        factors={"sample_size": (100, 250), "noise_sd": (0.5, 1.0)},
    )
    conditions = build_conditions(spec)
    manifest = create_manifest(spec, conditions)
    from_json = spec_from_manifest(manifest_from_text(manifest_to_json(manifest), "manifest.json"))
    from_yaml = spec_from_manifest(manifest_from_text(manifest_to_yaml(manifest), "manifest.yaml"))
    assert from_json == spec
    assert from_yaml == spec
    assert json.loads(manifest_to_json(manifest))["seed_model"]["execution_order_independent"] is True


def test_stress_test_constructs_seven_resolved_conditions():
    spec = stress_test_spec("OLS regression", replications=10, sample_size=100, seed=5)
    conditions = build_conditions(spec)
    assert len(conditions) == 7
    assert [c.factor_values["stress_level"] for c in conditions] == list(range(1, 8))
    assert conditions[0].dgp.assignment_mode == "randomized"
    assert conditions[-1].dgp.error_distribution == "student_t"
    assert conditions[-1].dgp.confounding_strength >= 2.0


def test_html_report_contains_design_and_results():
    spec = ExperimentSpec(sample_size=80, replications=8, estimators=("OLS regression",), title="Report smoke test")
    result = run_experiment(spec)
    manifest = create_manifest(spec, result.conditions)
    html = build_html_report(spec, result.summary, manifest_to_json(manifest))
    assert "Report smoke test" in html
    assert "Monte Carlo" in html
    assert "OLS regression" in html


def test_type_i_mode_forces_zero_structural_effect():
    spec = ExperimentSpec(mode="type_i", dgp=DGPConfig(tau0=1.5), replications=5)
    conditions = build_conditions(spec)
    assert conditions[0].dgp.tau0 == 0.0


def test_confidence_calibration_grid_resolves_nominal_levels():
    spec = ExperimentSpec(replications=5, factors={"confidence_level": (0.8, 0.9, 0.95, 0.99)})
    conditions = build_conditions(spec)
    assert [condition.confidence_level for condition in conditions] == [0.8, 0.9, 0.95, 0.99]


def test_type_i_mode_removes_heterogeneous_and_interaction_effects():
    spec = ExperimentSpec(
        mode="type_i",
        replications=5,
        dgp=DGPConfig(outcome_model="interaction", tau0=1.0, tau1=0.7, interaction_strength=0.8),
    )
    condition = build_conditions(spec)[0]
    assert condition.dgp.tau0 == 0.0
    assert condition.dgp.tau1 == 0.0
    assert condition.dgp.interaction_strength == 0.0


def test_external_generator_registry_runs_without_engine_changes():
    import numpy as np

    from mc_lab.generators import DataGenerator, GeneratedDataset, register_generator

    class TestGenerator(DataGenerator):
        name = "test_external"

        def generate(self, config, sample_size, seed):
            del seed
            treatment = np.arange(sample_size) % 2
            tau = float(config["tau"])
            frame = pd.DataFrame(
                {
                    "Y": tau * treatment.astype(float),
                    "A": treatment.astype(int),
                    "X1": np.zeros(sample_size),
                }
            )
            return GeneratedDataset(frame, tau, "Deterministic external-generator test DGP")

        def true_estimand(self, data, config):
            del data
            return float(config["tau"])

        def describe(self, config):
            return f"Y = {float(config['tau'])} A"

    register_generator("test_external", TestGenerator(), replace=True)
    spec = ExperimentSpec(
        generator="test_external",
        generator_config={"tau": 0.75},
        sample_size=100,
        replications=4,
        estimators=("Difference in means",),
    )
    result = run_experiment(spec, workers=1)
    assert result.failures.empty
    assert (result.replication_results["generator"] == "test_external").all()
    assert np.allclose(result.replication_results["estimate"], 0.75)
    assert np.allclose(result.replication_results["true_estimand"], 0.75)
