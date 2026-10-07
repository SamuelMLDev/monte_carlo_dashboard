"""DGP and complete simulation-study presets."""

from __future__ import annotations

from dataclasses import replace

from .config import DGPConfig, ExperimentSpec

DGP_PRESETS: dict[str, tuple[str, DGPConfig]] = {
    "Clean randomized experiment": (
        "Linear continuous outcome, randomized treatment, Gaussian noise.",
        DGPConfig(),
    ),
    "Weak confounding": (
        "Treatment depends weakly on observed covariates through a logistic propensity score.",
        DGPConfig(assignment_mode="logistic", confounding_strength=0.4),
    ),
    "Moderate confounding": (
        "Treatment depends moderately on observed covariates through a logistic propensity score.",
        DGPConfig(assignment_mode="logistic", confounding_strength=0.9),
    ),
    "Strong confounding": (
        "Stronger treatment selection on observed covariates while retaining usable overlap.",
        DGPConfig(assignment_mode="logistic", confounding_strength=1.6, propensity_clip=0.005),
    ),
    "Nonlinear outcome": (
        "Outcome regression contains nonlinear sine and quadratic structure not included in basic linear estimators.",
        DGPConfig(outcome_model="nonlinear", nonlinear_strength=1.0, num_covariates=2),
    ),
    "Heterogeneous treatment effects": (
        "The unit-level effect varies with X1: tau(X)=tau0+tau1*X1.",
        DGPConfig(outcome_model="heterogeneous", tau1=0.6),
    ),
    "Poor overlap": (
        "Strong covariate-dependent treatment assignment produces propensity scores close to zero or one.",
        DGPConfig(assignment_mode="logistic", confounding_strength=2.8, propensity_clip=0.001),
    ),
    "Heavy-tailed noise": (
        "Continuous outcomes with Student-t errors scaled to the requested standard deviation.",
        DGPConfig(error_distribution="student_t", student_t_df=3.0),
    ),
    "Heteroskedastic outcomes": (
        "Outcome error variance rises with squared X1.",
        DGPConfig(error_distribution="heteroskedastic", heteroskedastic_strength=1.5),
    ),
    "Model misspecification stress test": (
        "Nonlinear outcomes, confounding, and heteroskedasticity stress linear-model assumptions.",
        DGPConfig(outcome_model="nonlinear", nonlinear_strength=1.2, num_covariates=2, assignment_mode="logistic", confounding_strength=1.2, error_distribution="heteroskedastic", heteroskedastic_strength=1.5),
    ),
}


STUDY_PRESETS: dict[str, ExperimentSpec] = {
    "Experiment 1 — Why regression adjustment matters under confounding": ExperimentSpec(
        title="Why regression adjustment matters under confounding",
        description="Compares unadjusted, regression-adjusted, weighting, and doubly robust estimators when treatment depends on X.",
        sample_size=500,
        replications=500,
        estimators=("Difference in means", "OLS regression", "Inverse probability weighting", "Doubly robust AIPW"),
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.0),
    ),
    "Experiment 2 — Coverage improves with correct standard errors": ExperimentSpec(
        title="Coverage under heteroskedasticity",
        description="Contrasts conventional and HC3 OLS standard errors when residual variance depends on X.",
        sample_size=250,
        replications=500,
        estimators=("OLS regression", "OLS regression (HC3)"),
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.2, error_distribution="heteroskedastic", heteroskedastic_strength=4.0),
        factors={"sample_size": (100, 250, 500, 1000)},
    ),
    "Experiment 3 — Poor overlap destabilizes IPW": ExperimentSpec(
        title="Poor overlap and IPW instability",
        description="Examines weight instability as treatment assignment becomes more deterministic.",
        sample_size=500,
        replications=500,
        estimators=("OLS regression", "Inverse probability weighting", "Stabilized IPW", "Doubly robust AIPW"),
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=0.5, propensity_clip=0.001),
        factors={"confounding_strength": (0.5, 1.0, 1.8, 2.8)},
    ),
    "Experiment 4 — Heavy-tailed errors stress OLS inference": ExperimentSpec(
        title="Heavy-tailed error stress test",
        description="Studies conventional, HC3, and Huber regression under Student-t errors.",
        sample_size=250,
        replications=500,
        estimators=("OLS regression", "OLS regression (HC3)", "Robust regression (Huber)"),
        dgp=DGPConfig(error_distribution="student_t", student_t_df=3.0),
        factors={"sample_size": (100, 250, 500, 1000)},
    ),
    "Experiment 5 — Bias under nonlinear model misspecification": ExperimentSpec(
        title="Nonlinear outcome misspecification",
        description="Assesses linear regression and propensity-based estimators when the outcome surface is nonlinear.",
        sample_size=500,
        replications=500,
        estimators=("OLS regression", "Inverse probability weighting", "Doubly robust AIPW"),
        dgp=DGPConfig(outcome_model="nonlinear", nonlinear_strength=2.0, num_covariates=2, assignment_mode="logistic", confounding_strength=1.5),
        factors={"sample_size": (250, 500, 1000)},
    ),
    "Experiment 6 — Doubly robust estimation with one nuisance model misspecified": ExperimentSpec(
        title="Doubly robust estimation under propensity misspecification",
        description="The outcome model is correctly linear while AIPW is deliberately given an intercept-only propensity model; outcome correctness should protect consistency.",
        sample_size=750,
        replications=500,
        estimators=("Inverse probability weighting", "Outcome regression", "Doubly robust AIPW"),
        dgp=DGPConfig(assignment_mode="logistic", confounding_strength=1.0, num_covariates=1),
        estimator_settings={
            "Inverse probability weighting": {"propensity_covariates": []},
            "Doubly robust AIPW": {"propensity_covariates": []},
        },
    ),
}


def dgp_preset(name: str) -> DGPConfig:
    return replace(DGP_PRESETS[name][1])


def study_preset(name: str) -> ExperimentSpec:
    return replace(STUDY_PRESETS[name])


def stress_test_spec(estimator: str, replications: int = 300, sample_size: int = 500, seed: int = 42) -> ExperimentSpec:
    return ExperimentSpec(
        title=f"Method stress test — {estimator}",
        description="Cumulative robustness profile from clean randomized data to poor overlap, nonlinearity, heteroskedasticity, and heavy tails.",
        sample_size=sample_size,
        replications=replications,
        estimators=(estimator,),
        seed=seed,
        mode="stress_test",
        dgp=DGPConfig(),
        factors={"stress_level": tuple(range(1, 8))},
    )
