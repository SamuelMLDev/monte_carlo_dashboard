"""Run all built-in study presets and basic export/report smoke checks."""

from __future__ import annotations

from dataclasses import replace

from mc_lab.engine import run_experiment
from mc_lab.manifests import create_manifest, manifest_to_json, manifest_to_yaml, spec_from_manifest
from mc_lab.presets import STUDY_PRESETS
from mc_lab.reports import build_html_report


def main() -> None:
    for name, original in STUDY_PRESETS.items():
        spec = replace(original, workers=1)
        result = run_experiment(spec, workers=1)
        manifest = create_manifest(spec, result.conditions, result.elapsed_seconds)
        restored = spec_from_manifest(manifest)
        assert restored == spec
        assert len(result.replication_results) == sum(c.replications * len(c.estimators) for c in result.conditions)
        assert not result.summary.empty
        assert manifest_to_json(manifest)
        assert manifest_to_yaml(manifest)
        assert "<html" in build_html_report(spec, result.summary, manifest_to_json(manifest)).lower()
        print(
            f"PASS | {name} | conditions={len(result.conditions)} | rows={len(result.replication_results)} "
            f"| failures={len(result.failures)} | seconds={result.elapsed_seconds:.2f}"
        )


if __name__ == "__main__":
    main()
