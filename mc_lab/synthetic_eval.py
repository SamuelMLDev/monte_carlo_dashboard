"""Optional diagnostics for comparing real and synthetic tabular datasets."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, wasserstein_distance
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score


@dataclass(frozen=True)
class SequentialEvaluationConfig:
    """Reserved configuration for future explicitly sequential diagnostics."""

    sequence_column: str
    entity_column: str | None = None
    order_column: str | None = None


def numerical_similarity(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = [column for column in real.columns if column in synthetic.columns and pd.api.types.is_numeric_dtype(real[column]) and pd.api.types.is_numeric_dtype(synthetic[column])]
    for column in common:
        r = real[column].dropna().to_numpy(dtype=float)
        s = synthetic[column].dropna().to_numpy(dtype=float)
        if len(r) and len(s):
            ks = float(ks_2samp(r, s).statistic)
            rows.append({"column": column, "ks_statistic": ks, "ks_similarity": 1.0 - ks, "wasserstein_distance": float(wasserstein_distance(r, s))})
    return pd.DataFrame(rows)


def categorical_similarity(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    rows = []
    common = [column for column in real.columns if column in synthetic.columns and not pd.api.types.is_numeric_dtype(real[column])]
    for column in common:
        categories = sorted(set(real[column].dropna().astype(str)) | set(synthetic[column].dropna().astype(str)))
        r = real[column].astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
        s = synthetic[column].astype(str).value_counts(normalize=True).reindex(categories, fill_value=0.0)
        tv = float(0.5 * np.abs(r.to_numpy() - s.to_numpy()).sum())
        rows.append({"column": column, "tv_distance": tv, "tv_similarity": 1.0 - tv})
    return pd.DataFrame(rows)


def correlation_preservation(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    numeric = [column for column in real.columns if column in synthetic.columns and pd.api.types.is_numeric_dtype(real[column]) and pd.api.types.is_numeric_dtype(synthetic[column])]
    if len(numeric) < 2:
        return {"mean_absolute_correlation_error": float("nan"), "correlation_similarity": float("nan")}
    rc = real[numeric].corr().to_numpy(dtype=float)
    sc = synthetic[numeric].corr().to_numpy(dtype=float)
    mask = np.triu(np.ones_like(rc, dtype=bool), k=1)
    error = float(np.nanmean(np.abs(rc[mask] - sc[mask])))
    return {"mean_absolute_correlation_error": error, "correlation_similarity": max(0.0, 1.0 - error / 2.0)}



def pairwise_dependency_comparison(real: pd.DataFrame, synthetic: pd.DataFrame) -> pd.DataFrame:
    """Compare pairwise Pearson dependencies for shared numeric columns.

    The table is deliberately transparent rather than collapsing all structure
    to one score: researchers can inspect which relationships are preserved or
    distorted. ``correlation_preservation`` remains the compact aggregate.
    """
    numeric = [
        column
        for column in real.columns
        if column in synthetic.columns
        and pd.api.types.is_numeric_dtype(real[column])
        and pd.api.types.is_numeric_dtype(synthetic[column])
    ]
    rows: list[dict[str, float | str]] = []
    for i, first in enumerate(numeric):
        for second in numeric[i + 1 :]:
            real_pair = real[[first, second]].dropna()
            synthetic_pair = synthetic[[first, second]].dropna()
            if len(real_pair) < 2 or len(synthetic_pair) < 2:
                continue
            real_corr = float(real_pair[first].corr(real_pair[second]))
            synthetic_corr = float(synthetic_pair[first].corr(synthetic_pair[second]))
            rows.append(
                {
                    "variable_1": first,
                    "variable_2": second,
                    "real_correlation": real_corr,
                    "synthetic_correlation": synthetic_corr,
                    "absolute_error": abs(real_corr - synthetic_corr),
                }
            )
    return pd.DataFrame(rows)

def novelty_duplicate_rate(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, float]:
    common = [column for column in real.columns if column in synthetic.columns]
    if not common or synthetic.empty:
        return {"duplicate_rate": float("nan"), "novelty_rate": float("nan")}
    real_hashes = set(pd.util.hash_pandas_object(real[common].astype(str), index=False).to_numpy())
    synthetic_hashes = pd.util.hash_pandas_object(synthetic[common].astype(str), index=False).to_numpy()
    duplicate = float(np.mean([value in real_hashes for value in synthetic_hashes]))
    return {"duplicate_rate": duplicate, "novelty_rate": 1.0 - duplicate}


def train_synthetic_test_real(
    real: pd.DataFrame,
    synthetic: pd.DataFrame,
    target: str,
    task: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    """Train-on-synthetic/test-on-real utility using a compact random forest."""
    common = [column for column in real.columns if column in synthetic.columns and column != target]
    numeric = [column for column in common if pd.api.types.is_numeric_dtype(real[column]) and pd.api.types.is_numeric_dtype(synthetic[column])]
    if not numeric:
        raise ValueError("TSTR currently requires at least one shared numeric predictor")
    train = synthetic[[*numeric, target]].dropna()
    test = real[[*numeric, target]].dropna()
    if task == "auto":
        unique = pd.unique(real[target].dropna())
        task = "classification" if len(unique) <= 10 and not pd.api.types.is_float_dtype(real[target]) else "regression"
    if task == "classification":
        model = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
        model.fit(train[numeric], train[target])
        prediction = model.predict(test[numeric])
        result: dict[str, Any] = {"task": task, "accuracy": float(accuracy_score(test[target], prediction))}
        if len(pd.unique(test[target])) == 2 and hasattr(model, "predict_proba"):
            classes = list(model.classes_)
            positive = classes[-1]
            probs = model.predict_proba(test[numeric])[:, -1]
            truth = (test[target] == positive).astype(int)
            result["roc_auc"] = float(roc_auc_score(truth, probs))
        return result
    model = RandomForestRegressor(n_estimators=200, random_state=seed, n_jobs=1)
    model.fit(train[numeric], train[target])
    prediction = model.predict(test[numeric])
    return {"task": "regression", "rmse": float(mean_squared_error(test[target], prediction) ** 0.5)}


def evaluate_tabular_synthetic(real: pd.DataFrame, synthetic: pd.DataFrame) -> dict[str, Any]:
    """Run lightweight marginal, dependency, novelty, and size diagnostics."""
    return {
        "numerical": numerical_similarity(real, synthetic),
        "categorical": categorical_similarity(real, synthetic),
        "correlation": correlation_preservation(real, synthetic),
        "pairwise_dependencies": pairwise_dependency_comparison(real, synthetic),
        "novelty": novelty_duplicate_rate(real, synthetic),
        "sample_sizes": {"real": len(real), "synthetic": len(synthetic), "ratio": len(synthetic) / len(real) if len(real) else float("nan")},
    }


def sequential_diagnostics_scaffold(config: SequentialEvaluationConfig) -> dict[str, str]:
    """Describe reserved sequence metrics without forcing them into tabular workflows."""
    return {
        "status": "scaffold",
        "planned_metrics": "unigram, bigram, trigram, Jensen-Shannon divergence, sequence-length Wasserstein distance",
        "sequence_column": config.sequence_column,
    }
