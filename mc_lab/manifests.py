"""Machine-readable experiment manifests and round-trip serialization."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import datetime, timezone
from importlib import metadata
from typing import Any

import yaml

from . import __version__
from .config import ExperimentSpec, SimulationCondition

PACKAGE_NAMES = ["numpy", "pandas", "scipy", "statsmodels", "scikit-learn", "matplotlib", "streamlit", "PyYAML", "Jinja2"]


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def git_commit_hash() -> str | None:
    try:
        completed = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, timeout=2)
        return completed.stdout.strip() or None
    except Exception:
        return None


def create_manifest(spec: ExperimentSpec, conditions: list[SimulationCondition], elapsed_seconds: float | None = None) -> dict[str, Any]:
    """Build a complete reproducibility manifest for an experiment."""
    manifest: dict[str, Any] = {
        "manifest_schema": "mc-lab/v1",
        "software": {
            "name": "monte-carlo-research-lab",
            "version": __version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": package_versions(),
            "git_commit": git_commit_hash(),
        },
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": spec.to_dict(),
        "seed_model": {
            "experiment_seed": spec.seed,
            "condition_seed_rule": "BLAKE2b(experiment_seed, sorted factor values)",
            "replication_seed_rule": "BLAKE2b(condition_seed, replication index)",
            "execution_order_independent": True,
        },
        "conditions": [condition.to_dict() for condition in conditions],
    }
    if elapsed_seconds is not None:
        manifest["runtime"] = {"elapsed_seconds": float(elapsed_seconds)}
    return manifest


def manifest_to_json(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, default=str)


def manifest_to_yaml(manifest: dict[str, Any]) -> str:
    return yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True)


def manifest_from_text(text: str, filename: str | None = None) -> dict[str, Any]:
    """Parse JSON or YAML manifest text."""
    suffix = (filename or "").lower()
    if suffix.endswith(".json"):
        payload = json.loads(text)
    elif suffix.endswith((".yaml", ".yml")):
        payload = yaml.safe_load(text)
    else:
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Manifest must decode to a mapping")
    return payload


def spec_from_manifest(manifest: dict[str, Any]) -> ExperimentSpec:
    if "study" not in manifest:
        # Accept a bare exported ExperimentSpec as a convenience.
        return ExperimentSpec.from_dict(manifest)
    return ExperimentSpec.from_dict(manifest["study"])
