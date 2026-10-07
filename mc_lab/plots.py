"""Research-quality matplotlib figures for simulation studies."""

from __future__ import annotations

import math
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure


def _finish(fig: Figure, caption: str | None = None) -> Figure:
    if caption:
        fig.text(0.01, 0.01, caption, ha="left", va="bottom", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
    else:
        fig.tight_layout()
    return fig


def estimate_distribution_plot(results: pd.DataFrame, true_effect: float | None = None) -> Figure:
    """Histogram of successful estimates, optionally overlaying estimators."""
    successful = results.loc[results.get("success", True).astype(bool)] if "success" in results else results
    fig, ax = plt.subplots(figsize=(8, 4.8))
    estimators = list(successful["estimator"].dropna().unique()) if "estimator" in successful else ["Estimator"]
    if len(estimators) == 1:
        values = successful["estimate"].dropna()
        ax.hist(values, bins="auto", alpha=0.78, edgecolor="white", label=estimators[0])
        ax.axvline(float(values.mean()), linestyle=":", linewidth=2, label=f"Mean = {values.mean():.3f}")
    else:
        common = successful["estimate"].dropna().to_numpy(dtype=float)
        bins = np.histogram_bin_edges(common, bins="auto") if len(common) else 10
        for estimator, group in successful.groupby("estimator", sort=False):
            ax.hist(group["estimate"].dropna(), bins=bins, histtype="step", linewidth=1.8, label=str(estimator))
    if true_effect is None and "true_estimand" in successful and len(successful):
        true_effect = float(successful["true_estimand"].mean())
    if true_effect is not None and np.isfinite(true_effect):
        ax.axvline(true_effect, linestyle="--", linewidth=2, label=f"True estimand ≈ {true_effect:.3f}")
    ax.set_title("Estimator sampling distribution")
    ax.set_xlabel("Estimated treatment effect")
    ax.set_ylabel("Replication count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _finish(fig, "Sampling distributions are conditional on the configured data-generating process.")


def confidence_interval_plot(results: pd.DataFrame, true_effect: float | None = None, max_intervals: int = 40) -> Figure:
    """Forest-style subset of replication-level confidence intervals."""
    data = results.copy()
    if "success" in data:
        data = data.loc[data["success"].fillna(False)]
    if "estimator" in data and data["estimator"].nunique() > 1:
        data = data.loc[data["estimator"] == data["estimator"].iloc[0]]
    if {"ci_lower", "ci_upper"}.issubset(data.columns):
        data = data.loc[np.isfinite(data["ci_lower"]) & np.isfinite(data["ci_upper"])].copy()
    subset = data.head(max_intervals).copy()
    if true_effect is None and "true_estimand" in subset and len(subset):
        true_effect = float(subset["true_estimand"].mean())
    y = np.arange(len(subset))
    covered = subset.get("covers_true_effect", pd.Series([True] * len(subset))).fillna(False).to_numpy(dtype=bool)
    fig, ax = plt.subplots(figsize=(8, max(4.5, len(subset) * 0.14)))
    for mask, label in [(covered, "Covers truth"), (~covered, "Misses truth")]:
        part = subset.loc[mask]
        ypos = y[mask]
        if len(part):
            ax.errorbar(part["estimate"], ypos, xerr=[part["estimate"] - part["ci_lower"], part["ci_upper"] - part["estimate"]], fmt="o", markersize=4, capsize=2, linestyle="none", label=label)
    if true_effect is not None and np.isfinite(true_effect):
        ax.axvline(true_effect, linestyle="--", linewidth=2, label="True estimand")
    ax.set_title(f"Confidence-interval diagnostic: first {len(subset)} successful replications")
    ax.set_xlabel("Treatment effect")
    ax.set_ylabel("Replication")
    ax.invert_yaxis()
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    return _finish(fig)


def metric_vs_sample_size(summary: pd.DataFrame, metric: str, reference: float | None = None, ylabel: str | None = None) -> Figure:
    """Plot a metric against sample size with one line per estimator."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = summary.sort_values("sample_size")
    for estimator, group in ordered.groupby("estimator", sort=False):
        ax.plot(group["sample_size"], group[metric], marker="o", label=str(estimator))
    if reference is not None:
        ax.axhline(reference, linestyle="--", linewidth=1.5, label=f"Reference = {reference:g}")
    ax.set_title(f"{(ylabel or metric).replace('_', ' ').title()} vs sample size")
    ax.set_xlabel("Sample size")
    ax.set_ylabel(ylabel or metric.replace("_", " ").title())
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig)


def sample_size_performance_plot(summary: pd.DataFrame) -> Figure:
    """Legacy-compatible three-panel bias/RMSE/coverage plot."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for estimator, group in summary.sort_values("sample_size").groupby("estimator", sort=False):
        axes[0].plot(group["sample_size"], group["bias"], marker="o", label=str(estimator))
        axes[1].plot(group["sample_size"], group["rmse"], marker="o", label=str(estimator))
        axes[2].plot(group["sample_size"], 100 * group["coverage"], marker="o", label=str(estimator))
    axes[0].axhline(0.0, linestyle="--", linewidth=1)
    axes[2].axhline(95.0, linestyle="--", linewidth=1)
    for ax, title, ylabel in zip(axes, ["Bias", "RMSE", "Coverage"], ["Bias", "RMSE", "Coverage (%)"], strict=True):
        ax.set_title(title)
        ax.set_xlabel("Sample size")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    axes[-1].legend(frameon=False, fontsize=8)
    fig.suptitle("Performance across sample sizes", y=1.02)
    return _finish(fig)


def power_curve_plot(summary: pd.DataFrame, x: str = "factor_tau0") -> Figure:
    """Plot rejection probability (power/type-I error) against a varied factor."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    metric = "rejection_rate"
    for estimator, group in summary.groupby("estimator", sort=False):
        group = group.sort_values(x)
        ax.plot(group[x], group[metric], marker="o", label=str(estimator))
    ax.axhline(0.05, linestyle="--", linewidth=1, label="5% reference")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("Rejection probability / power curve")
    ax.set_xlabel(x.replace("factor_", "").replace("_", " ").title())
    ax.set_ylabel("Rejection probability")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig, "For a zero true effect this rejection probability is the empirical Type-I error rate.")


def bias_variance_plot(summary: pd.DataFrame) -> Figure:
    """Scatter absolute bias against empirical variance."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for estimator, group in summary.groupby("estimator", sort=False):
        ax.scatter(group["absolute_bias"], group["variance"], label=str(estimator), alpha=0.8)
    ax.set_title("Bias–variance tradeoff")
    ax.set_xlabel("Absolute bias")
    ax.set_ylabel("Empirical variance")
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig)


def heatmap_plot(summary: pd.DataFrame, value: str, row_factor: str, column_factor: str, estimator: str) -> Figure:
    """Heatmap a metric over two simulation factors for one estimator."""
    data = summary.loc[summary["estimator"] == estimator].copy()
    pivot = data.pivot_table(index=row_factor, columns=column_factor, values=value, aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.5, 5))
    image = ax.imshow(pivot.to_numpy(dtype=float), aspect="auto", origin="lower")
    ax.set_xticks(np.arange(len(pivot.columns)), labels=[str(v) for v in pivot.columns])
    ax.set_yticks(np.arange(len(pivot.index)), labels=[str(v) for v in pivot.index])
    ax.set_xlabel(column_factor.replace("factor_", "").replace("_", " ").title())
    ax.set_ylabel(row_factor.replace("factor_", "").replace("_", " ").title())
    ax.set_title(f"{value.replace('_', ' ').title()} — {estimator}")
    fig.colorbar(image, ax=ax, label=value.replace("_", " ").title())
    return _finish(fig)


def forest_summary_plot(results: pd.DataFrame) -> Figure:
    """Forest-style estimator mean ± empirical SD with average truth."""
    successful = results.loc[results["success"].fillna(False)].copy()
    grouped = successful.groupby("estimator", sort=False)
    labels, means, sds = [], [], []
    for name, group in grouped:
        labels.append(str(name))
        means.append(float(group["estimate"].mean()))
        sds.append(float(group["estimate"].std(ddof=1)))
    truth = float(successful["true_estimand"].mean()) if len(successful) else float("nan")
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(8, max(4, 0.5 * len(labels) + 2)))
    ax.errorbar(means, y, xerr=sds, fmt="o", capsize=3, linestyle="none", label="Mean ± empirical SD")
    if np.isfinite(truth):
        ax.axvline(truth, linestyle="--", linewidth=1.5, label="Mean true estimand")
    ax.set_yticks(y, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("Treatment effect")
    ax.set_title("Simulation summary by estimator")
    ax.grid(axis="x", alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig)


def monte_carlo_convergence_plot(results: pd.DataFrame, estimator: str | None = None) -> Figure:
    """Running bias, RMSE, and coverage with approximate Monte Carlo bands."""
    data = results.loc[results["success"].fillna(False)].copy()
    if estimator is None and "estimator" in data and len(data):
        estimator = str(data["estimator"].iloc[0])
    if estimator is not None:
        data = data.loc[data["estimator"] == estimator]
    data = data.sort_values("replication")
    errors = data["estimate"].to_numpy(dtype=float) - data["true_estimand"].to_numpy(dtype=float)
    coverage_values = pd.to_numeric(data["covers_true_effect"], errors="coerce").to_numpy(dtype=float)
    n = np.arange(1, len(data) + 1)
    running_bias = np.cumsum(errors) / n
    running_rmse = np.sqrt(np.cumsum(errors**2) / n)
    coverage_valid = np.isfinite(coverage_values)
    coverage_count = np.cumsum(coverage_valid.astype(int))
    coverage_sum = np.cumsum(np.where(coverage_valid, coverage_values, 0.0))
    running_coverage = np.divide(coverage_sum, coverage_count, out=np.full(len(data), np.nan), where=coverage_count > 0)
    bias_sd = pd.Series(errors).expanding(2).std().to_numpy(dtype=float)
    bias_mcse = bias_sd / np.sqrt(n)
    coverage_mcse = np.sqrt(np.divide(np.maximum(0, running_coverage * (1 - running_coverage)), coverage_count, out=np.full(len(data), np.nan), where=coverage_count > 0))

    fig, axes = plt.subplots(3, 1, figsize=(9, 9), sharex=True)
    axes[0].plot(n, running_bias)
    axes[0].fill_between(n, running_bias - 1.96 * bias_mcse, running_bias + 1.96 * bias_mcse, alpha=0.2)
    axes[0].axhline(0, linestyle="--", linewidth=1)
    axes[0].set_ylabel("Running bias")
    axes[1].plot(n, running_rmse)
    axes[1].set_ylabel("Running RMSE")
    axes[2].plot(n, running_coverage)
    axes[2].fill_between(n, running_coverage - 1.96 * coverage_mcse, running_coverage + 1.96 * coverage_mcse, alpha=0.2)
    axes[2].axhline(float(data["confidence_level"].iloc[0]) if len(data) else 0.95, linestyle="--", linewidth=1)
    axes[2].set_ylabel("Running coverage")
    axes[2].set_xlabel("Completed replications")
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(f"Monte Carlo convergence — {estimator or 'estimator'}", y=1.01)
    return _finish(fig, "Bands show approximate ±1.96 Monte Carlo standard errors for running bias and coverage.")


def propensity_plot(data: pd.DataFrame, propensity: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for value, label in [(0, "Control"), (1, "Treated")]:
        mask = data["A"].to_numpy(dtype=int) == value
        ax.hist(propensity[mask], bins=20, histtype="step", linewidth=1.8, density=True, label=label)
    ax.set_title("Estimated propensity-score overlap")
    ax.set_xlabel("Estimated propensity score")
    ax.set_ylabel("Density")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _finish(fig)


def balance_plot(balance: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, max(4, 0.38 * len(balance) + 2)))
    y = np.arange(len(balance))
    ax.scatter(balance["smd_before"], y, label="Before weighting")
    ax.scatter(balance["smd_after"], y, label="After weighting")
    ax.axvline(0.1, linestyle="--", linewidth=1)
    ax.axvline(-0.1, linestyle="--", linewidth=1)
    ax.axvline(0, linewidth=1)
    ax.set_yticks(y, labels=balance["covariate"])
    ax.set_xlabel("Standardized mean difference")
    ax.set_title("Covariate balance")
    ax.legend(frameon=False)
    ax.grid(axis="x", alpha=0.2)
    return _finish(fig, "Dashed lines at ±0.10 are descriptive balance reference thresholds, not formal tests.")


def weight_distribution_plot(weights: np.ndarray) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(weights, bins="auto", edgecolor="white")
    ax.axvline(10.0, linestyle="--", linewidth=1.5, label="Extreme-weight reference = 10")
    ax.set_title("Inverse-probability weight distribution")
    ax.set_xlabel("Weight")
    ax.set_ylabel("Count")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    return _finish(fig)


def residual_plot(residual_data: pd.DataFrame) -> Figure:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(residual_data["fitted"], residual_data["residual"], alpha=0.5, s=18)
    ax.axhline(0, linestyle="--", linewidth=1)
    ax.set_title("OLS residual diagnostic")
    ax.set_xlabel("Fitted value")
    ax.set_ylabel("Residual")
    ax.grid(alpha=0.2)
    return _finish(fig)


def figures_to_close(figures: Iterable[Figure]) -> None:
    for figure in figures:
        plt.close(figure)


def metric_vs_factor(summary: pd.DataFrame, factor: str, metric: str, reference: float | None = None, title: str | None = None) -> Figure:
    """Plot any summary metric against a factor with one line per estimator."""
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = summary.sort_values(factor)
    for estimator, group in ordered.groupby("estimator", sort=False):
        ax.plot(group[factor], group[metric], marker="o", label=str(estimator))
    if reference is not None:
        ax.axhline(reference, linestyle="--", linewidth=1.5, label=f"Reference = {reference:g}")
    ax.set_title(title or f"{metric.replace('_', ' ').title()} vs {factor.replace('factor_', '').replace('_', ' ').title()}")
    ax.set_xlabel(factor.replace("factor_", "").replace("_", " ").title())
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.grid(alpha=0.2)
    ax.legend(frameon=False)
    return _finish(fig)
