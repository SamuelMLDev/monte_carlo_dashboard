"""Regression and unadjusted estimators."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.special import expit
from scipy.stats import norm

from .base import EstimateResult, Estimator


def covariate_columns(data: pd.DataFrame) -> list[str]:
    """Return observed covariate columns in a stable order."""
    cols = [column for column in data.columns if column == "X" or (column.startswith("X") and column[1:].isdigit())]
    return sorted(cols, key=lambda name: (0, 0) if name == "X" else (1, int(name[1:])))


def _critical_value(confidence_level: float) -> float:
    return float(norm.ppf(0.5 + confidence_level / 2.0))


def _result(estimate: float, se: float, confidence_level: float, **kwargs: Any) -> EstimateResult:
    critical = _critical_value(confidence_level)
    if not np.isfinite(se) or se < 0:
        lower = upper = float("nan")
    else:
        lower = estimate - critical * se
        upper = estimate + critical * se
    return EstimateResult(float(estimate), float(se), float(lower), float(upper), **kwargs)


class DifferenceInMeansEstimator(Estimator):
    name = "Difference in means"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        treated = data.loc[data["A"] == 1, "Y"].to_numpy(dtype=float)
        control = data.loc[data["A"] == 0, "Y"].to_numpy(dtype=float)
        if len(treated) < 2 or len(control) < 2:
            raise ValueError("Both treatment groups need at least two observations")
        estimate = treated.mean() - control.mean()
        se = np.sqrt(treated.var(ddof=1) / len(treated) + control.var(ddof=1) / len(control))
        return _result(estimate, se, confidence_level, diagnostics={"n_treated": len(treated), "n_control": len(control)})


class OLSEstimator(Estimator):
    name = "OLS regression"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        columns = ["A", *covariate_columns(data)]
        design = sm.add_constant(data[columns].astype(float), has_constant="add")
        fit = sm.OLS(data["Y"].astype(float), design).fit()
        estimate = float(fit.params["A"])
        se = float(fit.bse["A"])
        return _result(estimate, se, confidence_level, diagnostics={"r_squared": float(fit.rsquared)})


class OLSHC3Estimator(Estimator):
    name = "OLS regression (HC3)"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        columns = ["A", *covariate_columns(data)]
        design = sm.add_constant(data[columns].astype(float), has_constant="add")
        fit = sm.OLS(data["Y"].astype(float), design).fit(cov_type="HC3")
        estimate = float(fit.params["A"])
        se = float(fit.bse["A"])
        return _result(estimate, se, confidence_level, diagnostics={"covariance": "HC3"})


class OutcomeRegressionEstimator(Estimator):
    """Parametric g-computation with delta-method uncertainty."""

    name = "Outcome regression"

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        settings = settings or {}
        columns = ["A", *covariate_columns(data)]
        design = sm.add_constant(data[columns].astype(float), has_constant="add")
        outcome = data["Y"].astype(float)
        binary = bool(settings.get("binary_outcome", set(np.unique(outcome.dropna())).issubset({0.0, 1.0})))

        x1 = design.copy()
        x0 = design.copy()
        x1["A"] = 1.0
        x0["A"] = 0.0

        if binary:
            fit = sm.GLM(outcome, design, family=sm.families.Binomial()).fit()
            beta = fit.params.to_numpy(dtype=float)
            p1 = expit(x1.to_numpy(dtype=float) @ beta)
            p0 = expit(x0.to_numpy(dtype=float) @ beta)
            estimate = float(np.mean(p1 - p0))
            gradient = np.mean(
                p1[:, None] * (1.0 - p1[:, None]) * x1.to_numpy(dtype=float)
                - p0[:, None] * (1.0 - p0[:, None]) * x0.to_numpy(dtype=float),
                axis=0,
            )
            covariance = np.asarray(fit.cov_params(), dtype=float)
            se = float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))
            converged = bool(getattr(fit, "converged", True))
        else:
            fit = sm.OLS(outcome, design).fit()
            contrast = np.mean(x1.to_numpy(dtype=float) - x0.to_numpy(dtype=float), axis=0)
            beta = fit.params.to_numpy(dtype=float)
            estimate = float(contrast @ beta)
            covariance = np.asarray(fit.cov_params(), dtype=float)
            se = float(np.sqrt(max(0.0, contrast @ covariance @ contrast)))
            converged = True

        return _result(
            estimate,
            se,
            confidence_level,
            converged=converged,
            status="ok" if converged else "not_converged",
            diagnostics={"model": "logistic" if binary else "linear"},
        )


class RobustRegressionEstimator(Estimator):
    """Legacy Huber M-estimator preserved for continuous-outcome stress studies."""

    name = "Robust regression (Huber)"
    supports_binary = False

    def estimate(self, data: pd.DataFrame, confidence_level: float = 0.95, settings: dict[str, Any] | None = None) -> EstimateResult:
        columns = ["A", *covariate_columns(data)]
        design = sm.add_constant(data[columns].astype(float), has_constant="add")
        fit = sm.RLM(data["Y"].astype(float), design, M=sm.robust.norms.HuberT()).fit()
        estimate = float(fit.params["A"])
        se = float(fit.bse["A"])
        return _result(estimate, se, confidence_level, diagnostics={"scale": float(fit.scale)})
