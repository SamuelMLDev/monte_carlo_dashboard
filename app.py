"""Streamlit interface for the Monte Carlo Research Laboratory."""

from __future__ import annotations

from dataclasses import replace
from io import BytesIO
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from mc_lab.config import DGPConfig, ExperimentSpec
from mc_lab.diagnostics import (
    generate_diagnostic_dataset,
    heteroskedasticity_indicator,
    ols_residual_diagnostics,
    treatment_diagnostics,
)
from mc_lab.empirical import bootstrap_estimator, compare_estimators, prepare_empirical_data
from mc_lab.engine import ExperimentResult, run_experiment
from mc_lab.estimators import ESTIMATOR_NAMES
from mc_lab.generators import DEFAULT_GENERATOR
from mc_lab.grid import estimate_workload
from mc_lab.manifests import (
    create_manifest,
    manifest_from_text,
    manifest_to_json,
    manifest_to_yaml,
    package_versions,
    spec_from_manifest,
)
from mc_lab.plots import (
    balance_plot,
    bias_variance_plot,
    confidence_interval_plot,
    estimate_distribution_plot,
    forest_summary_plot,
    heatmap_plot,
    metric_vs_factor,
    metric_vs_sample_size,
    monte_carlo_convergence_plot,
    power_curve_plot,
    propensity_plot,
    residual_plot,
    weight_distribution_plot,
)
from mc_lab.presets import DGP_PRESETS, STUDY_PRESETS, dgp_preset, study_preset
from mc_lab.reports import build_html_report
from mc_lab.synthetic_eval import evaluate_tabular_synthetic, train_synthetic_test_real
from mc_lab.utils import dataframe_to_csv_bytes, format_mc

st.set_page_config(page_title="Monte Carlo Research Laboratory", layout="wide")


ESTIMATOR_DESCRIPTIONS = {
    "Difference in means": "Unadjusted treated-minus-control mean difference with unequal-variance standard error.",
    "OLS regression": "Linear regression adjustment using treatment and all configured covariates; conventional covariance.",
    "OLS regression (HC3)": "The same linear adjustment with HC3 heteroskedasticity-robust covariance.",
    "Inverse probability weighting": "Horvitz–Thompson ATE using a fitted logistic propensity model.",
    "Stabilized IPW": "Hájek-style normalized weighted mean contrast using stabilized observed-treatment weights.",
    "Outcome regression": "Parametric g-computation; linear for continuous outcomes and logistic for binary outcomes.",
    "Doubly robust AIPW": "Two-fold cross-fitted augmented IPW with logistic propensity and treatment-specific outcome models.",
    "Robust regression (Huber)": "Huber M-estimation retained from the original dashboard for continuous-outcome robustness checks.",
}

MODE_LABELS = {
    "Standard simulation": "standard",
    "Type-I error calibration": "type_i",
    "Power study": "power",
    "Confidence-interval calibration": "calibration",
    "Method Stress Test": "stress_test",
}


def _numeric_list(text: str, cast: type = float) -> tuple[Any, ...]:
    values = []
    for token in text.split(","):
        token = token.strip()
        if token:
            values.append(cast(token))
    if not values:
        raise ValueError("Enter at least one comma-separated value")
    return tuple(values)


def _figure_png(figure: plt.Figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    return buffer.getvalue()


def _apply_spec_to_state(spec: ExperimentSpec) -> None:
    dgp = spec.dgp
    mapping = {
        "study_title": spec.title,
        "estimand_label": "Sample average treatment effect (ATE)",
        "study_description": spec.description,
        "sample_size": spec.sample_size,
        "replications": spec.replications,
        "confidence_level": spec.confidence_level,
        "experiment_seed": spec.seed,
        "workers": spec.workers,
        "estimators": list(spec.estimators),
        "outcome_type": dgp.outcome_type,
        "outcome_model": dgp.outcome_model,
        "covariate_distribution": dgp.covariate_distribution,
        "num_covariates": dgp.num_covariates,
        "covariate_correlation": dgp.covariate_correlation,
        "beta0": dgp.beta0,
        "beta_scale": dgp.beta_scale,
        "tau0": dgp.tau0,
        "tau1": dgp.tau1,
        "quadratic_strength": dgp.quadratic_strength,
        "nonlinear_strength": dgp.nonlinear_strength,
        "interaction_strength": dgp.interaction_strength,
        "assignment_mode": dgp.assignment_mode,
        "treatment_probability": dgp.treatment_probability,
        "treatment_intercept": dgp.treatment_intercept,
        "confounding_strength": dgp.confounding_strength,
        "confounding_level": "Custom",
        "propensity_clip": dgp.propensity_clip,
        "error_distribution": dgp.error_distribution,
        "noise_sd": dgp.noise_sd,
        "student_t_df": dgp.student_t_df,
        "heteroskedastic_strength": dgp.heteroskedastic_strength,
    }
    for key, value in mapping.items():
        st.session_state[key] = value
    def _selection_label(values: Any, num_covariates: int) -> str:
        if values is None:
            return "All covariates"
        values = list(values)
        if not values:
            return "Intercept only (deliberate misspecification)"
        if values == ["X1"]:
            return "X1 only"
        if values == [f"X{i}" for i in range(1, num_covariates + 1)]:
            return "All covariates"
        return "All covariates"

    propensity_settings = spec.estimator_settings.get("Doubly robust AIPW", spec.estimator_settings.get("Inverse probability weighting", {}))
    aipw_settings = spec.estimator_settings.get("Doubly robust AIPW", {})
    st.session_state["propensity_covariates"] = _selection_label(propensity_settings.get("propensity_covariates"), dgp.num_covariates)
    st.session_state["outcome_covariates"] = _selection_label(aipw_settings.get("outcome_covariates"), dgp.num_covariates)
    st.session_state["estimation_clip"] = float(propensity_settings.get("propensity_clip", 0.01))
    reverse_mode = {value: key for key, value in MODE_LABELS.items()}
    st.session_state["study_mode"] = reverse_mode.get(spec.mode, "Standard simulation")
    factors = spec.factors
    st.session_state["vary_sample_size"] = "sample_size" in factors
    st.session_state["sample_size_grid"] = ", ".join(map(str, factors.get("sample_size", (100, 250, 500, 1000))))
    st.session_state["vary_noise"] = "noise_sd" in factors
    st.session_state["noise_grid"] = ", ".join(map(str, factors.get("noise_sd", (0.5, 1.0, 2.0))))
    st.session_state["vary_confounding"] = "confounding_strength" in factors
    st.session_state["confounding_grid"] = ", ".join(map(str, factors.get("confounding_strength", (0.0, 0.9, 1.6))))
    st.session_state["vary_effect"] = "tau0" in factors
    st.session_state["effect_grid"] = ", ".join(map(str, factors.get("tau0", (0.0, 0.25, 0.5, 0.75))))
    st.session_state["vary_confidence"] = "confidence_level" in factors
    st.session_state["confidence_grid"] = ", ".join(map(str, factors.get("confidence_level", (0.8, 0.9, 0.95, 0.99))))


def _dgp_from_widgets() -> DGPConfig:
    tau0 = float(st.session_state.get("tau0", 0.5))
    confounding_levels = {
        "Weak": 0.4,
        "Moderate": 0.9,
        "Strong": 1.6,
        "Poor overlap": 2.8,
    }
    confounding_level = st.session_state.get("confounding_level", "Custom")
    confounding_strength = confounding_levels.get(confounding_level, float(st.session_state.get("confounding_strength", 0.0)))
    propensity_clip = float(st.session_state.get("propensity_clip", 0.01))
    if confounding_level == "Poor overlap":
        propensity_clip = min(propensity_clip, 0.001)
    if MODE_LABELS.get(st.session_state.get("study_mode", "Standard simulation")) == "type_i":
        tau0 = 0.0
    return DGPConfig(
        outcome_type=st.session_state.get("outcome_type", "continuous"),
        outcome_model=st.session_state.get("outcome_model", "linear"),
        covariate_distribution=st.session_state.get("covariate_distribution", "normal"),
        num_covariates=int(st.session_state.get("num_covariates", 1)),
        covariate_correlation=float(st.session_state.get("covariate_correlation", 0.3)),
        beta0=float(st.session_state.get("beta0", 0.0)),
        beta_scale=float(st.session_state.get("beta_scale", 1.0)),
        tau0=tau0,
        tau1=float(st.session_state.get("tau1", 0.0)),
        quadratic_strength=float(st.session_state.get("quadratic_strength", 0.0)),
        nonlinear_strength=float(st.session_state.get("nonlinear_strength", 0.0)),
        interaction_strength=float(st.session_state.get("interaction_strength", 0.0)),
        assignment_mode=st.session_state.get("assignment_mode", "randomized"),
        treatment_probability=float(st.session_state.get("treatment_probability", 0.5)),
        treatment_intercept=float(st.session_state.get("treatment_intercept", 0.0)),
        confounding_strength=float(confounding_strength),
        propensity_clip=float(propensity_clip),
        error_distribution=st.session_state.get("error_distribution", "gaussian"),
        noise_sd=float(st.session_state.get("noise_sd", 1.0)),
        student_t_df=float(st.session_state.get("student_t_df", 5.0)),
        heteroskedastic_strength=float(st.session_state.get("heteroskedastic_strength", 1.0)),
    )


def _current_spec() -> ExperimentSpec:
    mode = MODE_LABELS.get(st.session_state.get("study_mode", "Standard simulation"), "standard")
    dgp = _dgp_from_widgets()
    estimators = tuple(st.session_state.get("estimators", ["OLS regression"]))
    factors: dict[str, tuple[Any, ...]] = {}
    if mode == "stress_test":
        factors["stress_level"] = tuple(range(1, 8))
    else:
        if st.session_state.get("vary_sample_size", False):
            factors["sample_size"] = _numeric_list(st.session_state.get("sample_size_grid", "100, 250, 500, 1000"), int)
        if st.session_state.get("vary_noise", False):
            factors["noise_sd"] = _numeric_list(st.session_state.get("noise_grid", "0.5, 1.0, 2.0"), float)
        if st.session_state.get("vary_confounding", False):
            factors["confounding_strength"] = _numeric_list(st.session_state.get("confounding_grid", "0.0, 0.9, 1.6"), float)
        if st.session_state.get("vary_effect", False):
            factors["tau0"] = _numeric_list(st.session_state.get("effect_grid", "0.0, 0.25, 0.5, 0.75"), float)
        if st.session_state.get("vary_confidence", False):
            factors["confidence_level"] = _numeric_list(st.session_state.get("confidence_grid", "0.8, 0.9, 0.95, 0.99"), float)

    if mode == "power" and not any(key in factors for key in ("tau0", "sample_size")):
        factors["tau0"] = (0.0, 0.2, 0.4, 0.6, 0.8)
    if mode == "calibration" and "confidence_level" not in factors:
        factors["confidence_level"] = (0.8, 0.9, 0.95, 0.99)

    propensity_choice = st.session_state.get("propensity_covariates", "All covariates")
    outcome_choice = st.session_state.get("outcome_covariates", "All covariates")
    all_covariates = [f"X{i}" for i in range(1, dgp.num_covariates + 1)]
    selection_map = {
        "All covariates": all_covariates,
        "X1 only": ["X1"],
        "Intercept only (deliberate misspecification)": [],
    }
    p_selection = selection_map.get(propensity_choice, all_covariates)
    o_selection = selection_map.get(outcome_choice, all_covariates)
    estimator_settings = {
        "Inverse probability weighting": {"propensity_covariates": p_selection, "propensity_clip": float(st.session_state.get("estimation_clip", 0.01))},
        "Stabilized IPW": {"propensity_covariates": p_selection, "propensity_clip": float(st.session_state.get("estimation_clip", 0.01))},
        "Doubly robust AIPW": {
            "propensity_covariates": p_selection,
            "outcome_covariates": o_selection,
            "propensity_clip": float(st.session_state.get("estimation_clip", 0.01)),
        },
    }
    return ExperimentSpec(
        title=st.session_state.get("study_title", "Monte Carlo study"),
        description=st.session_state.get("study_description", ""),
        estimand="sample_ate",
        generator="parametric",
        sample_size=int(st.session_state.get("sample_size", 500)),
        replications=int(st.session_state.get("replications", 500)),
        estimators=estimators,
        confidence_level=float(st.session_state.get("confidence_level", 0.95)),
        seed=int(st.session_state.get("experiment_seed", 42)),
        workers=int(st.session_state.get("workers", 1)),
        mode=mode,
        null_value=0.0,
        dgp=dgp,
        factors=factors,
        estimator_settings=estimator_settings,
    )


def _show_dgp_math(dgp: DGPConfig) -> None:
    if dgp.outcome_type == "continuous":
        extras = []
        if dgp.outcome_model == "quadratic":
            extras.append(r"qX_1^2")
        if dgp.outcome_model == "nonlinear":
            extras.append(r"f(X)")
        tau = r"\tau_0 + \tau_1X_1"
        if dgp.outcome_model == "interaction" or dgp.interaction_strength:
            tau += r" + \gamma X_1"
        rhs = r"\beta_0 + \sum_j \beta_j X_j"
        if extras:
            rhs += " + " + " + ".join(extras)
        st.latex(rf"Y = {rhs} + A({tau}) + \varepsilon")
    else:
        st.latex(r"P(Y=1\mid A,X)=\operatorname{logit}^{-1}\{\beta_0+\beta'X+A\tau(X)\}")
        st.caption("For binary outcomes, estimators target the risk-difference ATE; τ is the structural log-odds shift.")
    if dgp.assignment_mode == "randomized":
        st.latex(rf"P(A=1\mid X)={dgp.treatment_probability:.3g}")
    else:
        score = rf"\alpha_0+{dgp.confounding_strength:.3g}X_1"
        if dgp.num_covariates > 1:
            score += rf"+0.35\,{dgp.confounding_strength:.3g}X_2"
        st.latex(rf"P(A=1\mid X)=\operatorname{{logit}}^{{-1}}({score})")
    st.caption(DEFAULT_GENERATOR.describe(dgp))


def _result_from_state() -> tuple[ExperimentResult | None, ExperimentSpec | None, dict[str, Any] | None]:
    return (
        st.session_state.get("experiment_result"),
        st.session_state.get("experiment_spec"),
        st.session_state.get("experiment_manifest"),
    )


if "history" not in st.session_state:
    st.session_state["history"] = []

st.title("Monte Carlo Research Laboratory")
st.write(
    "Design reproducible simulation experiments under known structural assumptions, compare estimators on the same generated datasets, "
    "quantify Monte Carlo uncertainty, diagnose overlap and weighting behavior, and export complete study manifests and reports."
)
st.caption("Scientific scope: reported performance is conditional on the configured data-generating process; the application does not claim a universally best method.")

with st.sidebar:
    st.header("Study presets")
    study_preset_name = st.selectbox("Load simulation study", ["Custom"] + list(STUDY_PRESETS), key="study_preset_selector")
    if st.button("Load study preset", use_container_width=True, disabled=study_preset_name == "Custom"):
        _apply_spec_to_state(study_preset(study_preset_name))
        st.rerun()

    dgp_preset_name = st.selectbox("Load DGP preset", ["Custom"] + list(DGP_PRESETS), key="dgp_preset_selector")
    if dgp_preset_name != "Custom":
        st.caption(DGP_PRESETS[dgp_preset_name][0])
    if st.button("Load DGP preset", use_container_width=True, disabled=dgp_preset_name == "Custom"):
        current = _current_spec() if "study_title" in st.session_state else ExperimentSpec()
        _apply_spec_to_state(replace(current, dgp=dgp_preset(dgp_preset_name)))
        st.rerun()

    st.divider()
    st.subheader("Load Experiment Configuration")
    manifest_upload = st.file_uploader("JSON or YAML manifest", type=["json", "yaml", "yml"], key="manifest_upload")
    if st.button("Load configuration", use_container_width=True, disabled=manifest_upload is None):
        try:
            payload = manifest_from_text(manifest_upload.getvalue().decode("utf-8"), manifest_upload.name)
            _apply_spec_to_state(spec_from_manifest(payload))
            st.success("Configuration loaded into the study controls.")
            st.rerun()
        except Exception as exc:
            st.error(f"Could not load configuration: {exc}")


tabs = st.tabs([
    "Study Design",
    "Estimators",
    "Run",
    "Results",
    "Diagnostics",
    "Visualizations",
    "Reproducibility",
    "Export",
    "Empirical Data Exploration",
])

with tabs[0]:
    st.subheader("Study Design")
    left, right = st.columns([1.15, 0.85])
    with left:
        st.selectbox("Estimand", ["Sample average treatment effect (ATE)"], key="estimand_label", help="The current research engine targets the sample ATE. The architecture is ready for future estimands such as ATT/ATC.")
        st.selectbox("Study mode", list(MODE_LABELS), key="study_mode", index=0)
        st.text_input("Study title", value="Monte Carlo study", key="study_title")
        st.text_area("Methodological description", value="", key="study_description", height=80)
        c1, c2, c3 = st.columns(3)
        c1.number_input("Base sample size", min_value=20, max_value=100_000, value=500, step=10, key="sample_size")
        c2.number_input("Replications / condition", min_value=20, max_value=100_000, value=500, step=50, key="replications")
        c3.number_input("Experiment seed", min_value=0, max_value=2**31 - 1, value=42, step=1, key="experiment_seed")
        c4, c5 = st.columns(2)
        c4.slider("Confidence level", min_value=0.80, max_value=0.99, value=0.95, step=0.01, key="confidence_level")
        c5.number_input("Local workers", min_value=1, max_value=16, value=1, step=1, key="workers", help="Parallel execution uses deterministic per-replication seeds. Single-worker mode remains available.")

        with st.expander("Covariates", expanded=True):
            a, b, c = st.columns(3)
            a.number_input("Number of covariates", min_value=1, max_value=10, value=1, step=1, key="num_covariates")
            b.selectbox("Covariate distribution", ["normal", "uniform", "bernoulli", "correlated_normal"], key="covariate_distribution")
            c.slider("Correlation (MVN)", 0.0, 0.8, 0.3, 0.05, key="covariate_correlation", disabled=st.session_state.get("covariate_distribution", "normal") != "correlated_normal")
            d, e = st.columns(2)
            d.number_input("Outcome intercept β₀", value=0.0, step=0.1, key="beta0")
            e.number_input("Covariate effect scale", value=1.0, step=0.1, key="beta_scale")

        with st.expander("Outcome mechanism", expanded=True):
            a, b = st.columns(2)
            a.selectbox("Outcome type", ["continuous", "binary"], key="outcome_type")
            b.selectbox("Outcome model", ["linear", "quadratic", "nonlinear", "heterogeneous", "interaction"], key="outcome_model")
            c1, c2 = st.columns(2)
            c1.number_input("Structural treatment effect τ₀", value=0.5, step=0.1, key="tau0", disabled=st.session_state.get("study_mode") == "Type-I error calibration")
            c2.number_input("Treatment heterogeneity τ₁", value=0.0, step=0.1, key="tau1")
            c3, c4, c5 = st.columns(3)
            c3.number_input("Quadratic strength", value=0.0, step=0.1, key="quadratic_strength")
            c4.number_input("Nonlinear strength", value=0.0, step=0.1, key="nonlinear_strength")
            c5.number_input("Interaction strength", value=0.0, step=0.1, key="interaction_strength")

        with st.expander("Treatment assignment", expanded=True):
            a, b = st.columns(2)
            a.selectbox("Assignment mechanism", ["randomized", "logistic"], key="assignment_mode")
            b.number_input("Randomized P(A=1)", min_value=0.05, max_value=0.95, value=0.5, step=0.05, key="treatment_probability", disabled=st.session_state.get("assignment_mode", "randomized") != "randomized")
            st.selectbox("Confounding level", ["Custom", "Weak", "Moderate", "Strong", "Poor overlap"], key="confounding_level", disabled=st.session_state.get("assignment_mode", "randomized") != "logistic", help="Preset α₁ values are 0.4, 0.9, 1.6, and 2.8 respectively. Poor-overlap mode also permits propensities close to 0 and 1.")
            c1, c2, c3 = st.columns(3)
            c1.number_input("Treatment intercept α₀", value=0.0, step=0.1, key="treatment_intercept", disabled=st.session_state.get("assignment_mode", "randomized") != "logistic")
            c2.number_input("Custom confounding strength α₁", min_value=0.0, max_value=5.0, value=0.0, step=0.1, key="confounding_strength", disabled=st.session_state.get("assignment_mode", "randomized") != "logistic" or st.session_state.get("confounding_level", "Custom") != "Custom")
            c3.number_input("True propensity floor", min_value=0.0, max_value=0.1, value=0.01, step=0.001, format="%.3f", key="propensity_clip", disabled=st.session_state.get("assignment_mode", "randomized") != "logistic")

        with st.expander("Error distribution", expanded=True):
            a, b = st.columns(2)
            a.selectbox("Error distribution", ["gaussian", "student_t", "skewed", "heteroskedastic"], key="error_distribution")
            b.number_input("Noise SD", min_value=0.01, value=1.0, step=0.1, key="noise_sd")
            c1, c2 = st.columns(2)
            c1.number_input("Student-t degrees of freedom", min_value=2.1, value=5.0, step=0.5, key="student_t_df", disabled=st.session_state.get("error_distribution", "gaussian") != "student_t")
            c2.number_input("Heteroskedastic strength", min_value=0.0, value=1.0, step=0.1, key="heteroskedastic_strength", disabled=st.session_state.get("error_distribution", "gaussian") != "heteroskedastic")

        st.markdown("#### Simulation-design explorer")
        st.caption("Choose factors to vary. The engine constructs their Cartesian product; estimators are evaluated on the same generated dataset within each condition.")
        f1, f2 = st.columns(2)
        with f1:
            st.checkbox("Vary sample size", value=False, key="vary_sample_size")
            st.text_input("Sample sizes", value="100, 250, 500, 1000", key="sample_size_grid", disabled=not st.session_state.get("vary_sample_size", False))
            st.checkbox("Vary noise SD", value=False, key="vary_noise")
            st.text_input("Noise levels", value="0.5, 1.0, 2.0", key="noise_grid", disabled=not st.session_state.get("vary_noise", False))
            st.checkbox("Vary confounding strength", value=False, key="vary_confounding")
            st.text_input("Confounding strengths", value="0.0, 0.9, 1.6", key="confounding_grid", disabled=not st.session_state.get("vary_confounding", False))
        with f2:
            st.checkbox("Vary effect size", value=False, key="vary_effect")
            st.text_input("Effect sizes", value="0.0, 0.25, 0.5, 0.75", key="effect_grid", disabled=not st.session_state.get("vary_effect", False))
            st.checkbox("Vary confidence level", value=False, key="vary_confidence")
            st.text_input("Confidence levels", value="0.80, 0.90, 0.95, 0.99", key="confidence_grid", disabled=not st.session_state.get("vary_confidence", False))
            if st.session_state.get("study_mode") == "Method Stress Test":
                st.info("Stress Test mode fixes a seven-level cumulative robustness grid; manual factor grids are ignored.")

    with right:
        st.markdown("#### Mathematical DGP")
        try:
            _show_dgp_math(_dgp_from_widgets())
        except Exception as exc:
            st.warning(f"Current controls do not form a valid DGP yet: {exc}")
        st.markdown("#### Known estimand")
        st.write(
            "Each replication records the exact sample-average treatment effect implied by the structural potential-outcome means. "
            "This permits heterogeneous and binary-outcome simulations without pretending the truth is constant when it is not."
        )
        st.markdown("#### Monte Carlo error")
        st.write(
            "Finite simulation studies have sampling error of their own. The results report MCSEs for bias, RMSE, empirical SD, mean SE, coverage, CI width, and rejection rates."
        )

with tabs[1]:
    st.subheader("Estimators")
    st.multiselect("Evaluate estimators on every generated dataset", list(ESTIMATOR_NAMES), default=["OLS regression"], key="estimators")
    selected = st.session_state.get("estimators", ["OLS regression"])
    st.dataframe(pd.DataFrame({"Estimator": selected, "Method": [ESTIMATOR_DESCRIPTIONS.get(name, "") for name in selected]}), hide_index=True, use_container_width=True)
    if st.session_state.get("outcome_type") == "binary" and "Robust regression (Huber)" in selected:
        st.warning("Huber RLM is a continuous-outcome method in this laboratory. If selected for a binary-outcome study, each fit is retained as an explicit unsupported-method failure rather than producing a misleading estimate.")
    with st.expander("Advanced nuisance-model controls"):
        st.caption("These controls make model misspecification explicit rather than hidden. They primarily affect propensity and AIPW nuisance models.")
        a, b, c = st.columns(3)
        options = ["All covariates", "X1 only", "Intercept only (deliberate misspecification)"]
        a.selectbox("Propensity model covariates", options, key="propensity_covariates")
        b.selectbox("AIPW outcome nuisance covariates", options, key="outcome_covariates")
        c.number_input("Estimation propensity clip", min_value=0.001, max_value=0.1, value=0.01, step=0.001, format="%.3f", key="estimation_clip")
        st.caption("IPW standard errors use influence-function approximations treating the fitted propensity model as fixed. AIPW uses deterministic two-fold cross-fitting and an empirical influence-function SE.")

with tabs[2]:
    st.subheader("Run")
    try:
        spec = _current_spec()
        spec.validate()
        workload = estimate_workload(spec)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Simulation conditions", f"{workload['conditions']:,}")
        c2.metric("Replications / condition", f"{spec.replications:,}")
        c3.metric("Generated datasets", f"{workload['datasets']:,}")
        c4.metric("Estimator fits", f"{workload['estimator_fits']:,}")
        st.write(f"**Estimators:** {', '.join(spec.estimators)}")
        st.write(f"**Factors varied:** {', '.join(spec.factors) if spec.factors else 'None — single condition'}")
        st.write(f"**Execution:** {spec.workers} local worker{'s' if spec.workers != 1 else ''}; deterministic hierarchical seeds.")
        if "confounding_strength" in spec.factors and spec.dgp.assignment_mode == "randomized":
            st.warning("Confounding strength is varied, but the treatment mechanism is randomized; that factor will not change treatment assignment unless logistic assignment is selected.")
        _show_dgp_math(spec.dgp)
    except Exception as exc:
        spec = None
        st.error(f"Study design is not valid: {exc}")

    if st.button("Run Monte Carlo Experiment", type="primary", use_container_width=True, disabled=spec is None):
        progress = st.progress(0.0, text="Running simulation study…")

        def update_progress(done: int, total: int) -> None:
            progress.progress(done / total, text=f"Completed {done:,} / {total:,} generated datasets")

        try:
            result = run_experiment(spec, progress_callback=update_progress)
            manifest = create_manifest(spec, result.conditions, result.elapsed_seconds)
            st.session_state["experiment_result"] = result
            st.session_state["experiment_spec"] = spec
            st.session_state["experiment_manifest"] = manifest
            st.session_state["history"].append({"spec": spec, "result": result, "manifest": manifest})
            st.session_state["history"] = st.session_state["history"][-8:]
            progress.empty()
            st.success(f"Completed {len(result.conditions):,} conditions in {result.elapsed_seconds:.2f} seconds. Failed fits are retained and reported.")
        except Exception as exc:
            progress.empty()
            st.exception(exc)

result, active_spec, active_manifest = _result_from_state()

with tabs[3]:
    st.subheader("Results")
    if result is None or active_spec is None:
        st.info("Run an experiment to populate research results.")
    else:
        workload = estimate_workload(active_spec)
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Study", f"{len(result.conditions)} conditions")
        c2.metric("Replications", f"{active_spec.replications:,} / condition")
        c3.metric("Datasets", f"{workload['datasets']:,}")
        c4.metric("Estimators", str(len(active_spec.estimators)))
        c5.metric("Failures", f"{100 * len(result.failures) / max(1, len(result.replication_results)):.2f}%")
        st.caption(f"Factors varied: {', '.join(active_spec.factors) if active_spec.factors else 'none'}. Confidence level: {100 * active_spec.confidence_level:.1f}%. True estimand is recorded per replication.")
        st.caption(f"Relative RMSE is paired to the first selected estimator ({active_spec.estimators[0]}) as the baseline; its MCSE uses replication-level covariance.")

        summary = result.summary.copy()
        display_columns = [column for column in [
            "condition_id", "estimator", "sample_size", "true_effect",
            "bias", "bias_mcse", "absolute_bias", "absolute_bias_mcse", "relative_bias", "relative_bias_mcse",
            "mse", "mse_mcse", "rmse", "rmse_mcse", "variance", "variance_mcse",
            "empirical_sd", "empirical_sd_mcse", "mean_se", "mean_se_mcse",
            "se_calibration_ratio", "se_calibration_ratio_mcse", "coverage", "coverage_mcse",
            "coverage_deviation_abs", "average_ci_width", "average_ci_width_mcse",
            "rejection_rate", "rejection_rate_mcse", "type_i_error", "type_i_error_mcse", "power", "power_mcse",
            "failure_rate", "failure_rate_mcse", "convergence_rate", "convergence_rate_mcse", "se_unavailable_rate",
            "extreme_weight_replication_rate", "relative_rmse", "relative_rmse_mcse", "rmse_rank",
            "replications", "successful_replications", "reported_se_replications", "valid_ci_replications", "rejection_replications",
        ] if column in summary]
        st.dataframe(summary[display_columns], hide_index=True, use_container_width=True, height=440)
        with st.expander("Replication-level results", expanded=False):
            st.dataframe(result.replication_results, hide_index=True, use_container_width=True, height=360)

        first = summary.iloc[0]
        st.markdown("#### First estimator-condition MC summary")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Bias", format_mc(float(first["bias"]), float(first["bias_mcse"])))
        m2.metric("RMSE", format_mc(float(first["rmse"]), float(first["rmse_mcse"])))
        m3.metric("Coverage", format_mc(float(first["coverage"]), float(first["coverage_mcse"]), percent=True))
        m4.metric("Empirical SD / Mean SE", f"{first['empirical_sd']:.4f} / {first['mean_se']:.4f}")
        st.caption("MCSE quantifies uncertainty in the simulation summary itself. It shrinks as the number of successful Monte Carlo replications increases.")

        st.markdown("#### Experiment history in this session")
        history_rows = []
        for index, item in enumerate(st.session_state.get("history", []), start=1):
            hist_summary = item["result"].summary
            history_rows.append({
                "Experiment": chr(64 + index),
                "Title": item["spec"].title,
                "DGP": item["spec"].dgp.outcome_model + "/" + item["spec"].dgp.assignment_mode,
                "Conditions": len(item["result"].conditions),
                "Estimators": ", ".join(item["spec"].estimators),
                "Mean |bias|": float(hist_summary["absolute_bias"].mean()),
                "Mean RMSE": float(hist_summary["rmse"].mean()),
                "Mean coverage": float(hist_summary["coverage"].mean()),
            })
        st.dataframe(pd.DataFrame(history_rows), hide_index=True, use_container_width=True)

with tabs[4]:
    st.subheader("Diagnostics")
    if result is None or active_spec is None:
        st.info("Run an experiment to inspect overlap, balance, weights, model diagnostics, and failed replications.")
    else:
        condition_labels = {condition.condition_id: condition for condition in result.conditions}
        selected_condition_id = st.selectbox("Diagnostic condition", list(condition_labels), key="diagnostic_condition")
        diagnostic_replication = st.number_input("Replication to regenerate", min_value=1, max_value=active_spec.replications, value=1, step=1)
        condition = condition_labels[selected_condition_id]
        data = generate_diagnostic_dataset(condition, int(diagnostic_replication))
        try:
            weighting_options = [name for name in active_spec.estimators if name in {"Inverse probability weighting", "Stabilized IPW", "Doubly robust AIPW"}]
            diagnostic_method = st.selectbox("Weighting diagnostic model", weighting_options or ["Default logistic propensity"], key="diagnostic_weighting_method")
            diagnostic_settings = dict(active_spec.estimator_settings.get(diagnostic_method, {})) if diagnostic_method in active_spec.estimator_settings else {}
            diagnostic_settings.setdefault("propensity_clip", float(st.session_state.get("estimation_clip", 0.01)))
            diag = treatment_diagnostics(data, stabilized=diagnostic_method == "Stabilized IPW", settings=diagnostic_settings)
            a, b = st.columns(2)
            with a:
                st.pyplot(propensity_plot(data, np.asarray(diag["propensity"])), clear_figure=True)
            with b:
                st.pyplot(balance_plot(diag["balance"]), clear_figure=True)
            c, d = st.columns(2)
            with c:
                st.pyplot(weight_distribution_plot(np.asarray(diag["weights"])), clear_figure=True)
            with d:
                st.pyplot(residual_plot(ols_residual_diagnostics(data)), clear_figure=True)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Maximum weight", f"{diag['max_weight']:.2f}")
            k2.metric("Effective sample size", f"{diag['effective_sample_size']:.1f}")
            k3.metric("Extreme weights >10", f"{100 * diag['extreme_weight_fraction']:.1f}%")
            k4.metric("Common-support width", f"{diag['overlap_width']:.3f}")
            st.caption(f"Descriptive heteroskedasticity indicator corr(|residual|, |X1|): {heteroskedasticity_indicator(data):.3f}")
            st.dataframe(diag["balance"], hide_index=True, use_container_width=True)
        except Exception as exc:
            st.warning(f"Treatment diagnostics were not available for this replication: {exc}")

        st.markdown("#### Estimation failures and warnings")
        if result.failures.empty:
            st.success("No failed estimator fits were recorded in this experiment.")
        else:
            st.warning(f"{len(result.failures):,} estimator fits failed. They are retained below rather than silently dropped.")
            st.dataframe(result.failures, hide_index=True, use_container_width=True, height=300)
        warning_rows = result.replication_results.loc[result.replication_results["status"].ne("ok")]
        if not warning_rows.empty:
            st.dataframe(warning_rows[["condition_id", "replication", "estimator", "status", "error_message"]], hide_index=True, use_container_width=True)

with tabs[5]:
    st.subheader("Visualizations")
    if result is None or active_spec is None:
        st.info("Run an experiment to create simulation figures.")
    else:
        condition_ids = list(dict.fromkeys(result.replication_results["condition_id"]))
        condition_id = st.selectbox("Condition for distribution and convergence plots", condition_ids, key="viz_condition")
        condition_results = result.replication_results.loc[result.replication_results["condition_id"] == condition_id]
        est = st.selectbox("Estimator for convergence diagnostic", list(active_spec.estimators), key="viz_estimator")
        a, b = st.columns(2)
        with a:
            st.pyplot(estimate_distribution_plot(condition_results), clear_figure=True)
        with b:
            st.pyplot(forest_summary_plot(condition_results), clear_figure=True)
        st.pyplot(confidence_interval_plot(condition_results.loc[condition_results["estimator"] == est]), clear_figure=True)
        st.pyplot(monte_carlo_convergence_plot(condition_results, est), clear_figure=True)
        st.caption("The convergence plot asks whether the finite-replication estimates of bias, RMSE, and coverage have stabilized; approximate MC uncertainty bands are shown for running bias and coverage.")

        factor_columns = [column for column in result.summary.columns if column.startswith("factor_")]
        if "factor_sample_size" in factor_columns or result.summary["sample_size"].nunique() > 1:
            x_summary = result.summary.copy()
            st.pyplot(metric_vs_sample_size(x_summary, "bias", reference=0.0, ylabel="Bias"), clear_figure=True)
            st.pyplot(metric_vs_sample_size(x_summary, "rmse", ylabel="RMSE"), clear_figure=True)
            st.pyplot(metric_vs_sample_size(x_summary, "coverage", reference=float(x_summary["confidence_level"].median()), ylabel="Coverage"), clear_figure=True)
        if active_spec.mode == "power":
            if "factor_tau0" in factor_columns:
                st.pyplot(power_curve_plot(result.summary, "factor_tau0"), clear_figure=True)
            elif "factor_sample_size" in factor_columns or result.summary["sample_size"].nunique() > 1:
                st.pyplot(power_curve_plot(result.summary, "sample_size"), clear_figure=True)
        if "factor_stress_level" in factor_columns:
            st.markdown("#### Method robustness profile")
            st.pyplot(metric_vs_factor(result.summary, "factor_stress_level", "rmse", title="Stress-test robustness profile: RMSE"), clear_figure=True)
            st.pyplot(metric_vs_factor(result.summary, "factor_stress_level", "coverage", reference=active_spec.confidence_level, title="Stress-test robustness profile: coverage"), clear_figure=True)
        st.pyplot(bias_variance_plot(result.summary), clear_figure=True)
        if len(factor_columns) >= 2:
            heatmap_estimator = st.selectbox("Estimator for factor heatmap", list(active_spec.estimators), key="heatmap_estimator")
            heatmap_metric = st.selectbox("Heatmap metric", ["rmse", "coverage", "absolute_bias", "failure_rate"], key="heatmap_metric")
            st.pyplot(heatmap_plot(result.summary, heatmap_metric, factor_columns[0], factor_columns[1], heatmap_estimator), clear_figure=True)

with tabs[6]:
    st.subheader("Reproducibility")
    if active_manifest is None or active_spec is None:
        st.info("Run an experiment to generate a complete manifest.")
    else:
        st.markdown("#### Seed hierarchy")
        st.write(
            "The experiment seed deterministically derives a seed for each resolved condition, and each condition seed deterministically derives a seed for each replication. "
            "Because those seeds depend on condition values and replication index rather than task completion order, serial and parallel runs are reproducible."
        )
        st.json(active_manifest["seed_model"])
        st.markdown("#### Software environment")
        st.dataframe(pd.DataFrame([{"Package": key, "Version": value} for key, value in package_versions().items()]), hide_index=True, use_container_width=True)
        with st.expander("Complete manifest", expanded=False):
            st.json(active_manifest)

with tabs[7]:
    st.subheader("Export")
    if result is None or active_spec is None or active_manifest is None:
        st.info("Run an experiment before exporting outputs.")
    else:
        manifest_json = manifest_to_json(active_manifest)
        manifest_yaml = manifest_to_yaml(active_manifest)
        c1, c2, c3, c4 = st.columns(4)
        c1.download_button("Replication results CSV", dataframe_to_csv_bytes(result.replication_results), "replication_results.csv", "text/csv", use_container_width=True)
        c2.download_button("Summary metrics CSV", dataframe_to_csv_bytes(result.summary), "summary_metrics.csv", "text/csv", use_container_width=True)
        c3.download_button("Manifest JSON", manifest_json, "experiment_manifest.json", "application/json", use_container_width=True)
        c4.download_button("Manifest YAML", manifest_yaml, "experiment_manifest.yaml", "application/x-yaml", use_container_width=True)

        first_condition = result.replication_results.loc[result.replication_results["condition_id"] == result.replication_results["condition_id"].iloc[0]]
        if st.button("Prepare HTML research report", use_container_width=True):
            report_figures = [
                (estimate_distribution_plot(first_condition), "Sampling distributions for the first simulation condition."),
                (forest_summary_plot(first_condition), "Estimator means and empirical standard deviations for the first condition."),
                (monte_carlo_convergence_plot(first_condition, active_spec.estimators[0]), "Monte Carlo convergence diagnostic for the baseline estimator."),
            ]
            try:
                diagnostic_data = generate_diagnostic_dataset(result.conditions[0], 1)
                report_diag = treatment_diagnostics(diagnostic_data)
                report_figures.extend([
                    (propensity_plot(diagnostic_data, np.asarray(report_diag["propensity"])), "Propensity-score overlap diagnostic for replication 1 of the first condition."),
                    (balance_plot(report_diag["balance"]), "Covariate balance before and after inverse-probability weighting."),
                ])
            except Exception:
                pass
            warnings = []
            if not result.failures.empty:
                warnings.append(f"{len(result.failures)} estimator fits failed and are retained in the replication-level export.")
            if active_spec.dgp.assignment_mode == "logistic" and active_spec.dgp.confounding_strength >= 2:
                warnings.append("Strong treatment selection can produce poor overlap and unstable inverse-probability weights.")
            st.session_state["prepared_report_html"] = build_html_report(active_spec, result.summary, manifest_json, report_figures, warnings)
            for figure, _ in report_figures:
                plt.close(figure)
        if "prepared_report_html" in st.session_state:
            st.download_button("Download HTML research report", st.session_state["prepared_report_html"], "monte_carlo_research_report.html", "text/html", type="primary", use_container_width=True)

        p1, p2 = st.columns(2)
        p1.download_button("Distribution plot PNG", _figure_png(estimate_distribution_plot(first_condition)), "estimate_distribution.png", "image/png", use_container_width=True)
        p2.download_button("Convergence plot PNG", _figure_png(monte_carlo_convergence_plot(first_condition, active_spec.estimators[0])), "monte_carlo_convergence.png", "image/png", use_container_width=True)

with tabs[8]:
    st.subheader("Empirical Data Exploration")
    st.warning("This mode compares estimators on observed data. Population truth is generally unknown, so bias, RMSE, and coverage against a true effect are not reported unless truth comes from an external design—not inferred by this application.")
    empirical_upload = st.file_uploader("Upload empirical CSV", type=["csv"], key="empirical_csv")
    if empirical_upload is not None:
        empirical_frame = pd.read_csv(empirical_upload)
        st.dataframe(empirical_frame.head(20), use_container_width=True)
        columns = list(empirical_frame.columns)
        if len(columns) >= 2:
            c1, c2 = st.columns(2)
            outcome_column = c1.selectbox("Outcome column", columns, key="empirical_outcome")
            treatment_column = c2.selectbox("Treatment column", columns, index=min(1, len(columns) - 1), key="empirical_treatment")
            candidate_covariates = [column for column in columns if column not in {outcome_column, treatment_column}]
            covariates = st.multiselect("Covariates", candidate_covariates, default=candidate_covariates[: min(3, len(candidate_covariates))], key="empirical_covariates")
            treatment_values = list(pd.unique(empirical_frame[treatment_column].dropna()))
            treated_value = st.selectbox("Value treated as A=1", treatment_values, index=min(1, len(treatment_values) - 1), key="empirical_treated_value") if treatment_values else None
            empirical_estimators = st.multiselect("Estimators", list(ESTIMATOR_NAMES), default=["Difference in means", "OLS regression", "OLS regression (HC3)"], key="empirical_estimators")
            with st.expander("Optional externally known reference estimand"):
                use_reference = st.checkbox(
                    "I have a design-based or externally established reference effect",
                    value=False,
                    key="empirical_use_reference",
                    help="The application will never infer population truth from the uploaded dataset. If supplied, this value is only used for estimation-error and interval-containment diagnostics, not to claim Monte Carlo bias from one observed dataset.",
                )
                reference_estimand = st.number_input("Reference estimand", value=0.0, step=0.1, key="empirical_reference_estimand", disabled=not use_reference)
            if st.button("Compare methods on empirical data", type="primary"):
                try:
                    mapped = prepare_empirical_data(empirical_frame, outcome_column, treatment_column, covariates, treated_value)
                    comparison = compare_estimators(mapped, empirical_estimators, float(st.session_state.get("confidence_level", 0.95)))
                    if use_reference:
                        reference = float(reference_estimand)
                        comparison["reference_estimand"] = reference
                        comparison["estimation_error_to_reference"] = comparison["estimate"] - reference
                        comparison["ci_contains_reference"] = (comparison["ci_lower"] <= reference) & (reference <= comparison["ci_upper"])
                    st.session_state["empirical_mapped"] = mapped
                    st.session_state["empirical_comparison"] = comparison
                except Exception as exc:
                    st.error(f"Empirical comparison failed: {exc}")
            if "empirical_comparison" in st.session_state:
                st.dataframe(st.session_state["empirical_comparison"], hide_index=True, use_container_width=True)
                mapped = st.session_state["empirical_mapped"]
                try:
                    ediag = treatment_diagnostics(mapped)
                    a, b = st.columns(2)
                    a.pyplot(propensity_plot(mapped, np.asarray(ediag["propensity"])), clear_figure=True)
                    b.pyplot(balance_plot(ediag["balance"]), clear_figure=True)
                except Exception as exc:
                    st.info(f"Propensity diagnostics unavailable: {exc}")
                with st.expander("Bootstrap uncertainty"):
                    bootstrap_name = st.selectbox("Estimator to bootstrap", empirical_estimators or ["OLS regression"], key="bootstrap_estimator_name")
                    bootstrap_reps = st.number_input("Bootstrap replications", 50, 2000, 300, 50, key="bootstrap_reps")
                    if st.button("Run bootstrap"):
                        try:
                            st.json(bootstrap_estimator(mapped, bootstrap_name, int(bootstrap_reps), seed=int(st.session_state.get("experiment_seed", 42))))
                        except Exception as exc:
                            st.error(f"Bootstrap failed: {exc}")

    st.divider()
    st.markdown("### Optional synthetic-data diagnostics")
    st.caption("This component is separate from Monte Carlo truth evaluation. It compares supplied real and synthetic tabular datasets without assuming how the synthetic data were generated.")
    real_upload = st.file_uploader("Real reference CSV", type=["csv"], key="synthetic_real_csv")
    synthetic_upload = st.file_uploader("Synthetic CSV", type=["csv"], key="synthetic_csv")
    if real_upload is not None and synthetic_upload is not None:
        real = pd.read_csv(real_upload)
        synthetic = pd.read_csv(synthetic_upload)
        evaluation = evaluate_tabular_synthetic(real, synthetic)
        st.write("**Sample sizes**", evaluation["sample_sizes"])
        st.write("**Correlation preservation**", evaluation["correlation"])
        st.write("**Novelty / duplicate rate**", evaluation["novelty"])
        if not evaluation["numerical"].empty:
            st.markdown("#### Numerical marginals")
            st.dataframe(evaluation["numerical"], hide_index=True, use_container_width=True)
        if not evaluation["categorical"].empty:
            st.markdown("#### Categorical marginals")
            st.dataframe(evaluation["categorical"], hide_index=True, use_container_width=True)
        if not evaluation["pairwise_dependencies"].empty:
            st.markdown("#### Pairwise dependency preservation")
            st.dataframe(evaluation["pairwise_dependencies"], hide_index=True, use_container_width=True)
        common_targets = [column for column in real.columns if column in synthetic.columns]
        if common_targets:
            target = st.selectbox("Optional TSTR target", ["None"] + common_targets, key="tstr_target")
            if target != "None" and st.button("Run train-on-synthetic/test-on-real utility check"):
                try:
                    st.json(train_synthetic_test_real(real, synthetic, target))
                except Exception as exc:
                    st.error(f"TSTR check failed: {exc}")
