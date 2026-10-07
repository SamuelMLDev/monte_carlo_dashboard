from mc_lab.config import DGPConfig, ExperimentSpec
from mc_lab.engine import run_experiment


def test_unsupported_binary_estimator_is_recorded_as_failure():
    spec = ExperimentSpec(
        sample_size=120,
        replications=5,
        estimators=("Robust regression (Huber)",),
        seed=12,
        dgp=DGPConfig(outcome_type="binary"),
    )
    result = run_experiment(spec)
    assert len(result.replication_results) == 5
    assert len(result.failures) == 5
    assert result.replication_results["error_message"].str.contains("does not support binary outcomes").all()
