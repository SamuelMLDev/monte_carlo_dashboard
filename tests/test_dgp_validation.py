import pytest

from mc_lab.config import DGPConfig
from mc_lab.generators import DEFAULT_GENERATOR


def test_equicorrelation_validation_respects_covariate_dimension():
    DGPConfig(num_covariates=3, covariate_distribution="correlated_normal", covariate_correlation=-0.4).validate()
    with pytest.raises(ValueError, match="positive-definite"):
        DGPConfig(num_covariates=10, covariate_distribution="correlated_normal", covariate_correlation=-0.2).validate()


def test_dgp_description_includes_interaction_heterogeneity():
    config = DGPConfig(outcome_model="interaction", tau0=0.5, tau1=0.2, interaction_strength=0.3)
    text = DEFAULT_GENERATOR.describe(config)
    assert "tau(X)=0.5+0.5X1" in text
