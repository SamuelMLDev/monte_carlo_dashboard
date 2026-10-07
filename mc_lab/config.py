"""Configuration objects for simulation studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class DGPConfig:
    """Parametric data-generating process configuration.

    The estimand is the sample average treatment effect (ATE) implied by the
    structural model for each generated covariate matrix. For constant-effect
    continuous models this equals ``tau0`` exactly in every replication.
    """

    outcome_type: str = "continuous"  # continuous | binary
    outcome_model: str = "linear"  # linear | quadratic | nonlinear | heterogeneous | interaction
    covariate_distribution: str = "normal"  # normal | uniform | bernoulli | correlated_normal
    num_covariates: int = 1
    covariate_correlation: float = 0.3
    beta0: float = 0.0
    beta_scale: float = 1.0
    tau0: float = 0.5
    tau1: float = 0.0
    quadratic_strength: float = 0.0
    nonlinear_strength: float = 0.0
    interaction_strength: float = 0.0
    assignment_mode: str = "randomized"  # randomized | logistic
    treatment_probability: float = 0.5
    treatment_intercept: float = 0.0
    confounding_strength: float = 0.0
    propensity_clip: float = 0.01
    error_distribution: str = "gaussian"  # gaussian | student_t | skewed | heteroskedastic
    noise_sd: float = 1.0
    student_t_df: float = 5.0
    heteroskedastic_strength: float = 1.0

    def validate(self) -> None:
        if self.outcome_type not in {"continuous", "binary"}:
            raise ValueError("outcome_type must be 'continuous' or 'binary'")
        if self.outcome_model not in {"linear", "quadratic", "nonlinear", "heterogeneous", "interaction"}:
            raise ValueError(f"Unknown outcome_model: {self.outcome_model}")
        if self.covariate_distribution not in {"normal", "uniform", "bernoulli", "correlated_normal"}:
            raise ValueError(f"Unknown covariate_distribution: {self.covariate_distribution}")
        if not 1 <= self.num_covariates <= 10:
            raise ValueError("num_covariates must be between 1 and 10")
        lower_correlation = -1.0 / (self.num_covariates - 1) if self.num_covariates > 1 else -1.0
        if not lower_correlation < self.covariate_correlation < 0.95:
            raise ValueError(
                "covariate_correlation must yield a positive-definite equicorrelation matrix "
                f"(rho > {lower_correlation:.4g} for p={self.num_covariates}, and rho < 0.95)"
            )
        if self.assignment_mode not in {"randomized", "logistic"}:
            raise ValueError(f"Unknown assignment_mode: {self.assignment_mode}")
        if not 0 < self.treatment_probability < 1:
            raise ValueError("treatment_probability must be in (0, 1)")
        if not 0 <= self.propensity_clip < 0.5:
            raise ValueError("propensity_clip must be in [0, 0.5)")
        if self.error_distribution not in {"gaussian", "student_t", "skewed", "heteroskedastic"}:
            raise ValueError(f"Unknown error_distribution: {self.error_distribution}")
        if self.noise_sd <= 0:
            raise ValueError("noise_sd must be positive")
        if self.student_t_df <= 2:
            raise ValueError("student_t_df must exceed 2 so the variance exists")
        if self.heteroskedastic_strength < 0:
            raise ValueError("heteroskedastic_strength must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> DGPConfig:
        return cls(**payload)


@dataclass(frozen=True)
class ExperimentSpec:
    """A complete study design before expansion into grid conditions."""

    title: str = "Monte Carlo study"
    description: str = ""
    estimand: str = "sample_ate"
    generator: str = "parametric"
    generator_config: dict[str, Any] = field(default_factory=dict)
    sample_size: int = 500
    replications: int = 500
    estimators: tuple[str, ...] = ("OLS regression",)
    confidence_level: float = 0.95
    seed: int = 42
    workers: int = 1
    mode: str = "standard"  # standard | type_i | power | calibration | stress_test
    null_value: float = 0.0
    dgp: DGPConfig = field(default_factory=DGPConfig)
    factors: dict[str, tuple[Any, ...]] = field(default_factory=dict)
    estimator_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def validate(self) -> None:
        if self.estimand != "sample_ate":
            raise ValueError("The current engine supports the sample ATE estimand")
        if not self.generator:
            raise ValueError("generator must be a non-empty registry name")
        if self.sample_size < 20:
            raise ValueError("sample_size must be at least 20")
        if self.replications < 1:
            raise ValueError("replications must be positive")
        if not self.estimators:
            raise ValueError("Select at least one estimator")
        if not 0.5 < self.confidence_level < 1:
            raise ValueError("confidence_level must be between 0.5 and 1")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.workers < 1:
            raise ValueError("workers must be at least 1")
        if self.generator == "parametric":
            self.dgp.validate()

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimators"] = list(self.estimators)
        payload["factors"] = {key: list(values) for key, values in self.factors.items()}
        payload["estimator_settings"] = self.estimator_settings
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentSpec:
        data = dict(payload)
        data["estimators"] = tuple(data.get("estimators", ("OLS regression",)))
        data["factors"] = {key: tuple(values) for key, values in data.get("factors", {}).items()}
        data["dgp"] = DGPConfig.from_dict(data.get("dgp", {}))
        data["estimator_settings"] = data.get("estimator_settings", {})
        return cls(**data)


@dataclass(frozen=True)
class SimulationCondition:
    """One fully resolved condition in a factorial experiment."""

    condition_id: str
    condition_index: int
    estimand: str
    generator: str
    generator_config: dict[str, Any]
    sample_size: int
    replications: int
    confidence_level: float
    estimators: tuple[str, ...]
    experiment_seed: int
    condition_seed: int
    null_value: float
    dgp: DGPConfig
    factor_values: dict[str, Any] = field(default_factory=dict)
    estimator_settings: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["estimators"] = list(self.estimators)
        return payload


def update_spec(spec: ExperimentSpec, **changes: Any) -> ExperimentSpec:
    """Functional helper used by presets and the grid builder."""
    return replace(spec, **changes)
