"""Static HTML research-report generation."""

from __future__ import annotations

import base64
import html
import json
from collections.abc import Iterable
from io import BytesIO

import pandas as pd
from jinja2 import Template
from matplotlib.figure import Figure

from .config import ExperimentSpec
from .generators import get_generator

_TEMPLATE = Template(
    """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>{{ title }}</title>
<style>
body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 2.5rem auto; max-width: 1100px; padding: 0 1.2rem; color: #1f2937; line-height: 1.5; }
h1, h2, h3 { color: #111827; }
table { border-collapse: collapse; width: 100%; font-size: 0.9rem; margin: 1rem 0 2rem; }
th, td { border: 1px solid #d1d5db; padding: 0.45rem 0.55rem; text-align: right; }
th:first-child, td:first-child { text-align: left; }
th { background: #f3f4f6; }
pre { background: #f8fafc; border: 1px solid #e5e7eb; padding: 1rem; overflow-x: auto; }
figure { margin: 2rem 0; }
figure img { max-width: 100%; height: auto; border: 1px solid #e5e7eb; }
figcaption { color: #4b5563; font-size: 0.9rem; }
.note { background: #f8fafc; border-left: 4px solid #64748b; padding: 0.8rem 1rem; }
.warn { background: #fff7ed; border-left: 4px solid #c2410c; padding: 0.8rem 1rem; }
</style>
</head>
<body>
<h1>{{ title }}</h1>
<p>{{ description }}</p>
<div class="note"><strong>Monte Carlo scope.</strong> Results describe estimator behavior only under the configured simulation assumptions. They are not universal rankings.</div>
<h2>Study design</h2>
<p><strong>Mathematical DGP:</strong> {{ dgp }}</p>
<p><strong>Estimators:</strong> {{ estimators }}</p>
<p><strong>Replications per condition:</strong> {{ replications }}. <strong>Confidence level:</strong> {{ confidence_level }}.</p>
<p><strong>Conditions:</strong> {{ condition_count }}. <strong>Generated datasets:</strong> {{ dataset_count }}. <strong>Factors varied:</strong> {{ factors }}.</p>
<h3>Configuration</h3>
<pre>{{ config }}</pre>
<h2>Main results</h2>
{{ summary_table | safe }}
<h2>Metric definitions</h2>
<ul>
<li><strong>Bias:</strong> mean estimation error relative to the known replication-specific estimand.</li>
<li><strong>Absolute/relative bias:</strong> magnitude of bias and bias relative to the mean known estimand when the latter is nonzero.</li>
<li><strong>MSE/RMSE:</strong> mean squared estimation error and its square root.</li>
<li><strong>Empirical SD:</strong> standard deviation of successful estimates across replications.</li>
<li><strong>Mean SE:</strong> mean estimator-reported standard error; its ratio to empirical SD is the SE-calibration ratio.</li>
<li><strong>Coverage:</strong> fraction of valid confidence intervals containing the known replication-specific estimand.</li>
<li><strong>Rejection rate:</strong> fraction of valid intervals excluding the configured null value; interpreted as Type-I error only under the null and as power under an alternative.</li>
<li><strong>Failure/convergence rates:</strong> estimator execution and convergence diagnostics; failed fits remain in the replication-level output.</li>
<li><strong>MCSE:</strong> Monte Carlo standard error quantifying uncertainty caused by a finite number of replications.</li>
</ul>
{% for figure in figures %}
<figure><img src="data:image/png;base64,{{ figure.image }}" alt="{{ figure.caption }}"><figcaption>{{ figure.caption }}</figcaption></figure>
{% endfor %}
<h2>Reproducibility</h2>
<pre>{{ manifest }}</pre>
<h2>Warnings and limitations</h2>
{% if warnings %}<div class="warn"><ul>{% for warning in warnings %}<li>{{ warning }}</li>{% endfor %}</ul></div>{% else %}<p>No estimator failures were recorded. Standard model-specific limitations still apply.</p>{% endif %}
<p>Propensity-based standard errors use influence-function approximations; IPW variants treat the fitted propensity score as fixed for their reported within-replication standard error. AIPW uses deterministic two-fold cross-fitting. For heterogeneous effects, the simulation truth is the replication-specific sample ATE; some estimator standard errors are conventional superpopulation-style approximations, so coverage under heterogeneity should be interpreted with that distinction in mind.</p>
</body>
</html>"""
)


def _figure_base64(figure: Figure) -> str:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=140, bbox_inches="tight")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def build_html_report(
    spec: ExperimentSpec,
    summary: pd.DataFrame,
    manifest_json: str,
    figures: Iterable[tuple[Figure, str]] = (),
    warnings: list[str] | None = None,
) -> str:
    """Render a self-contained factual HTML report."""
    display_columns = [
        column for column in [
            "condition_id", "estimator", "sample_size", "true_effect", "bias", "bias_mcse",
            "absolute_bias", "relative_bias", "mse", "rmse", "rmse_mcse", "variance",
            "empirical_sd", "mean_se", "se_calibration_ratio", "coverage", "coverage_mcse",
            "average_ci_width", "rejection_rate", "type_i_error", "power", "failure_rate",
            "convergence_rate", "relative_rmse", "rmse_rank",
        ] if column in summary.columns
    ]
    display = summary[display_columns].copy()
    figure_payload = [{"image": _figure_base64(fig), "caption": caption} for fig, caption in figures]
    generator = get_generator(spec.generator)
    generator_config = spec.dgp if spec.generator == "parametric" else spec.generator_config
    dgp_text = generator.describe(generator_config)
    return _TEMPLATE.render(
        title=html.escape(spec.title),
        description=html.escape(spec.description or "Reproducible Monte Carlo simulation study."),
        dgp=html.escape(dgp_text),
        estimators=html.escape(", ".join(spec.estimators)),
        replications=spec.replications,
        confidence_level=f"{100 * spec.confidence_level:.1f}%",
        condition_count=int(summary["condition_id"].nunique()) if "condition_id" in summary else 1,
        dataset_count=int((summary[["condition_id", "replications"]].drop_duplicates()["replications"].sum()) if {"condition_id", "replications"}.issubset(summary.columns) else spec.replications),
        factors=html.escape(", ".join(spec.factors) if spec.factors else "none"),
        config=html.escape(json.dumps(spec.to_dict(), indent=2, sort_keys=True, default=str)),
        summary_table=display.to_html(index=False, float_format=lambda x: f"{x:.4g}", border=0),
        figures=figure_payload,
        manifest=html.escape(manifest_json),
        warnings=warnings or [],
    )
