"""Monte Carlo performance metrics and Monte Carlo standard errors."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _mean_mcse(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return float("nan")
    return float(np.std(values, ddof=1) / math.sqrt(len(values)))


def _proportion_mcse(p: float, n: int) -> float:
    if n <= 0 or not np.isfinite(p):
        return float("nan")
    return float(math.sqrt(max(0.0, p * (1.0 - p) / n)))


def calculate_metrics(results: pd.DataFrame, true_effect: float | None = None, nominal_level: float = 0.95) -> dict[str, float | int]:
    """Summarize one estimator-condition group with explicit MC uncertainty.

    If ``true_estimand`` exists in ``results`` it is used replication by
    replication. Otherwise ``true_effect`` supplies a constant truth for
    backwards compatibility.
    """
    if results.empty:
        raise ValueError("Cannot calculate metrics for an empty result table")
    total = len(results)
    if "success" in results:
        successful = results.loc[results["success"].fillna(False)].copy()
    else:
        successful = results.copy()
    successful = successful.loc[np.isfinite(pd.to_numeric(successful["estimate"], errors="coerce"))].copy()
    n_success = len(successful)
    if n_success == 0:
        base: dict[str, float | int] = {key: float("nan") for key in [
            "bias", "bias_mcse", "absolute_bias", "absolute_bias_mcse", "relative_bias", "relative_bias_mcse",
            "mse", "mse_mcse", "rmse", "rmse_mcse", "variance", "variance_mcse", "empirical_sd",
            "empirical_sd_mcse", "std_estimate", "mean_estimate", "mean_estimate_mcse", "mean_se", "mean_se_mcse",
            "se_calibration_ratio", "se_calibration_ratio_mcse", "coverage", "coverage_mcse", "coverage_deviation",
            "coverage_deviation_mcse", "average_ci_width", "average_ci_width_mcse", "rejection_rate",
            "rejection_rate_mcse", "type_i_error", "type_i_error_mcse", "power", "power_mcse", "true_effect", "true_effect_mcse"
        ]}
        base.update({"replications": total, "successful_replications": 0, "reported_se_replications": 0, "valid_ci_replications": 0, "rejection_replications": 0, "failure_rate": 1.0, "failure_rate_mcse": 0.0, "convergence_rate": 0.0, "convergence_rate_mcse": 0.0, "se_available_rate": 0.0, "se_unavailable_rate": float("nan"), "se_unavailable_rate_mcse": float("nan"), "extreme_weight_replication_rate": float("nan"), "extreme_weight_replication_rate_mcse": float("nan")})
        return base

    estimates = successful["estimate"].to_numpy(dtype=float)
    if "true_estimand" in successful:
        truth = successful["true_estimand"].to_numpy(dtype=float)
    elif true_effect is not None:
        truth = np.full(n_success, float(true_effect))
    else:
        raise ValueError("true_effect is required when results lack true_estimand")
    errors = estimates - truth
    error_sq = errors**2
    mean_truth = float(np.mean(truth))
    bias = float(np.mean(errors))
    empirical_sd = float(np.std(estimates, ddof=1)) if n_success > 1 else 0.0
    variance = float(empirical_sd**2)
    mse = float(np.mean(error_sq))
    rmse = float(math.sqrt(mse))
    bias_mcse = _mean_mcse(errors)
    true_effect_mcse = _mean_mcse(truth)
    if abs(mean_truth) > 1e-12:
        relative_bias = float(bias / mean_truth)
        if n_success > 1:
            covariance = np.cov(np.column_stack([errors, truth]).T, ddof=1) / n_success
            gradient = np.array([1.0 / mean_truth, -bias / (mean_truth**2)], dtype=float)
            relative_bias_mcse = float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))
        else:
            relative_bias_mcse = float("nan")
    else:
        relative_bias = relative_bias_mcse = float("nan")
    mse_mcse = _mean_mcse(error_sq)
    rmse_mcse = float(mse_mcse / (2.0 * rmse)) if rmse > 0 and np.isfinite(mse_mcse) else float("nan")
    # Monte Carlo uncertainty for the variance/SD is estimated from the
    # empirical influence function rather than the normal-theory chi-square
    # approximation. This remains meaningful under heavy-tailed simulation
    # designs (provided the relevant fourth moment is finite in the simulated
    # distribution). The reported variance itself remains the usual ddof=1
    # empirical variance across successful replications.
    if n_success > 1 and empirical_sd > 0:
        centered_sq = (estimates - float(np.mean(estimates))) ** 2
        pop_variance = float(np.mean(centered_sq))
        variance_scale = n_success / (n_success - 1)
        variance_if = variance_scale * (centered_sq - pop_variance)
        variance_mcse = _mean_mcse(variance_if)
        empirical_sd_if = variance_if / (2.0 * empirical_sd)
        empirical_sd_mcse = _mean_mcse(empirical_sd_if)
    elif n_success > 1:
        variance_mcse = 0.0
        empirical_sd_mcse = 0.0
    else:
        variance_mcse = empirical_sd_mcse = float("nan")

    se_raw = pd.to_numeric(successful.get("standard_error", pd.Series(index=successful.index, dtype=float)), errors="coerce").to_numpy(dtype=float)
    se_mask = np.isfinite(se_raw) & (se_raw >= 0)
    se_values = se_raw[se_mask]
    mean_se = float(np.mean(se_values)) if len(se_values) else float("nan")
    mean_se_mcse = _mean_mcse(se_values) if len(se_values) else float("nan")
    if empirical_sd > 0 and np.isfinite(mean_se):
        se_ratio = mean_se / empirical_sd
        if len(se_values) > 1:
            # Preserve the pairing between estimator-reported SEs and the
            # replication estimates when quantifying MC uncertainty in the
            # calibration ratio.
            paired_estimates = estimates[se_mask]
            paired_centered_sq = (paired_estimates - float(np.mean(estimates))) ** 2
            pop_variance = float(np.mean((estimates - float(np.mean(estimates))) ** 2))
            variance_scale = n_success / (n_success - 1) if n_success > 1 else 1.0
            sd_if = variance_scale * (paired_centered_sq - pop_variance) / (2.0 * empirical_sd)
            ratio_if = (se_values - mean_se) / empirical_sd - (mean_se / empirical_sd**2) * sd_if
            se_ratio_mcse = _mean_mcse(ratio_if)
        else:
            se_ratio_mcse = float("nan")
    else:
        se_ratio = se_ratio_mcse = float("nan")

    if "covers_true_effect" in successful:
        coverage_series = successful["covers_true_effect"].dropna().astype(float)
        coverage = float(coverage_series.mean()) if len(coverage_series) else float("nan")
        coverage_mcse = _proportion_mcse(coverage, len(coverage_series))
        valid_ci_replications = int(len(coverage_series))
    else:
        coverage = coverage_mcse = float("nan")
        valid_ci_replications = 0

    if {"ci_lower", "ci_upper"}.issubset(successful.columns):
        widths = (successful["ci_upper"] - successful["ci_lower"]).to_numpy(dtype=float)
        widths = widths[np.isfinite(widths)]
        average_width = float(np.mean(widths)) if len(widths) else float("nan")
        average_width_mcse = _mean_mcse(widths) if len(widths) else float("nan")
    else:
        average_width = average_width_mcse = float("nan")

    if "null_rejected" in successful:
        rejection = successful["null_rejected"].dropna().astype(float)
        rejection_rate = float(rejection.mean()) if len(rejection) else float("nan")
        rejection_mcse = _proportion_mcse(rejection_rate, len(rejection))
        rejection_replications = int(len(rejection))
    else:
        rejection_rate = rejection_mcse = float("nan")
        rejection_replications = 0

    all_null = bool(np.all(np.isclose(truth, 0.0, atol=1e-12)))
    type_i_error = rejection_rate if all_null else float("nan")
    power = rejection_rate if not all_null else float("nan")
    failure_rate = float(1.0 - n_success / total)
    if "converged" in results:
        convergence_rate = float(results["converged"].fillna(False).astype(bool).mean())
    else:
        convergence_rate = float(n_success / total)
    if "standard_error_available" in successful:
        se_available_rate = float(successful["standard_error_available"].fillna(False).astype(bool).mean())
    else:
        se_available_rate = float(len(se_values) / n_success)
    se_unavailable_rate = 1.0 - se_available_rate
    if "extreme_weights" in successful:
        extreme = successful["extreme_weights"].dropna().astype(bool)
        extreme_weight_rate = float(extreme.mean()) if len(extreme) else float("nan")
        extreme_weight_rate_mcse = _proportion_mcse(extreme_weight_rate, len(extreme)) if len(extreme) else float("nan")
    else:
        extreme_weight_rate = extreme_weight_rate_mcse = float("nan")

    return {
        "bias": bias,
        "bias_mcse": bias_mcse,
        "absolute_bias": abs(bias),
        "absolute_bias_mcse": bias_mcse,
        "relative_bias": relative_bias,
        "relative_bias_mcse": relative_bias_mcse,
        "mse": mse,
        "mse_mcse": mse_mcse,
        "rmse": rmse,
        "rmse_mcse": rmse_mcse,
        "variance": variance,
        "variance_mcse": variance_mcse,
        "empirical_sd": empirical_sd,
        "empirical_sd_mcse": empirical_sd_mcse,
        "std_estimate": empirical_sd,
        "mean_estimate": float(np.mean(estimates)),
        "mean_estimate_mcse": _mean_mcse(estimates),
        "mean_se": mean_se,
        "mean_se_mcse": mean_se_mcse,
        "se_calibration_ratio": se_ratio,
        "se_calibration_ratio_mcse": se_ratio_mcse,
        "coverage": coverage,
        "coverage_mcse": coverage_mcse,
        "coverage_deviation": float(coverage - nominal_level) if np.isfinite(coverage) else float("nan"),
        "coverage_deviation_mcse": coverage_mcse,
        "average_ci_width": average_width,
        "average_ci_width_mcse": average_width_mcse,
        "rejection_rate": rejection_rate,
        "rejection_rate_mcse": rejection_mcse,
        "type_i_error": type_i_error,
        "type_i_error_mcse": rejection_mcse if all_null else float("nan"),
        "power": power,
        "power_mcse": rejection_mcse if not all_null else float("nan"),
        "true_effect": mean_truth,
        "true_effect_mcse": true_effect_mcse,
        "replications": total,
        "successful_replications": n_success,
        "reported_se_replications": int(len(se_values)),
        "valid_ci_replications": valid_ci_replications,
        "rejection_replications": rejection_replications,
        "failure_rate": failure_rate,
        "failure_rate_mcse": _proportion_mcse(failure_rate, total),
        "convergence_rate": convergence_rate,
        "convergence_rate_mcse": _proportion_mcse(convergence_rate, total),
        "se_available_rate": se_available_rate,
        "se_unavailable_rate": se_unavailable_rate,
        "se_unavailable_rate_mcse": _proportion_mcse(se_unavailable_rate, n_success),
        "extreme_weight_replication_rate": extreme_weight_rate,
        "extreme_weight_replication_rate_mcse": extreme_weight_rate_mcse,
    }


def summarize_long_results(results: pd.DataFrame, estimator_order: tuple[str, ...] | None = None) -> pd.DataFrame:
    """Return one tidy summary row per condition and estimator."""
    if results.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    group_columns = ["condition_id", "condition_index", "estimator"]
    for keys, group in results.groupby(group_columns, sort=False, dropna=False):
        condition_id, condition_index, estimator = keys
        nominal = float(group["confidence_level"].iloc[0])
        metrics = calculate_metrics(group, nominal_level=nominal)
        row: dict[str, Any] = {
            "condition_id": condition_id,
            "condition_index": int(condition_index),
            "estimator": estimator,
            "estimand": group["estimand"].iloc[0] if "estimand" in group else "sample_ate",
            "generator": group["generator"].iloc[0] if "generator" in group else "parametric",
            "sample_size": int(group["sample_size"].iloc[0]),
            "confidence_level": nominal,
        }
        factor_columns = [column for column in group.columns if column.startswith("factor_")]
        for column in factor_columns:
            row[column] = group[column].iloc[0]
        row.update(metrics)
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary["coverage_deviation_abs"] = (summary["coverage"] - summary["confidence_level"]).abs()
    summary["rmse_rank"] = summary.groupby("condition_id")["rmse"].rank(method="min", na_option="bottom")

    order = estimator_order or tuple(summary["estimator"].drop_duplicates())
    baseline_name = order[0] if order else None
    baseline = (
        summary.loc[summary["estimator"] == baseline_name, ["condition_id", "rmse"]]
        .rename(columns={"rmse": "baseline_rmse"})
        if baseline_name is not None
        else pd.DataFrame(columns=["condition_id", "baseline_rmse"])
    )
    summary = summary.merge(baseline, on="condition_id", how="left")
    summary["relative_rmse"] = summary["rmse"] / summary["baseline_rmse"]
    summary["relative_rmse_mcse"] = np.nan

    if baseline_name is not None:
        for condition_id, condition_rows in results.groupby("condition_id", sort=False):
            baseline_rows = condition_rows.loc[condition_rows["estimator"] == baseline_name].copy()
            baseline_rows = baseline_rows.loc[baseline_rows["success"].fillna(False)] if "success" in baseline_rows else baseline_rows
            baseline_rows = baseline_rows[["replication", "estimate", "true_estimand"]].rename(columns={"estimate": "estimate_base", "true_estimand": "truth_base"})
            for estimator in condition_rows["estimator"].drop_duplicates():
                estimator_rows = condition_rows.loc[condition_rows["estimator"] == estimator].copy()
                estimator_rows = estimator_rows.loc[estimator_rows["success"].fillna(False)] if "success" in estimator_rows else estimator_rows
                estimator_rows = estimator_rows[["replication", "estimate", "true_estimand"]].rename(columns={"estimate": "estimate_est", "true_estimand": "truth_est"})
                paired = estimator_rows.merge(baseline_rows, on="replication", how="inner")
                if len(paired) < 2:
                    continue
                qa = (paired["estimate_est"].to_numpy(dtype=float) - paired["truth_est"].to_numpy(dtype=float)) ** 2
                qb = (paired["estimate_base"].to_numpy(dtype=float) - paired["truth_base"].to_numpy(dtype=float)) ** 2
                mse_a = float(np.mean(qa))
                mse_b = float(np.mean(qb))
                if mse_a <= 0 or mse_b <= 0:
                    continue
                ratio = float(np.sqrt(mse_a / mse_b))
                covariance = np.cov(np.column_stack([qa, qb]).T, ddof=1) / len(paired)
                gradient = np.array([ratio / (2.0 * mse_a), -ratio / (2.0 * mse_b)], dtype=float)
                ratio_mcse = float(np.sqrt(max(0.0, gradient @ covariance @ gradient)))
                mask = (summary["condition_id"] == condition_id) & (summary["estimator"] == estimator)
                summary.loc[mask, "relative_rmse"] = ratio
                summary.loc[mask, "relative_rmse_mcse"] = ratio_mcse

    summary["coverage_deviation_abs_mcse"] = summary["coverage_mcse"]
    return summary.sort_values(["condition_index", "rmse_rank", "estimator"], kind="stable").reset_index(drop=True)
