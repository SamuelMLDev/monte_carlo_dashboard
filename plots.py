"""Backwards-compatible plotting facade."""

from mc_lab.plots import confidence_interval_plot, estimate_distribution_plot, sample_size_performance_plot

__all__ = ["estimate_distribution_plot", "confidence_interval_plot", "sample_size_performance_plot"]
