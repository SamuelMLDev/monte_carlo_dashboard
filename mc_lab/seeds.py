"""Deterministic hierarchical seed utilities."""

from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np


def canonical_json(value: Any) -> str:
    """Return a stable JSON encoding for seed derivation and manifests."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def stable_seed(root_seed: int, *parts: Any) -> int:
    """Derive a deterministic 64-bit child seed from a root seed and labels."""
    payload = f"{int(root_seed)}|" + "|".join(canonical_json(part) for part in parts)
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False)


def replication_seed(condition_seed: int, replication_index: int) -> int:
    """Derive a replication seed independent of execution order."""
    return stable_seed(condition_seed, "replication", int(replication_index))


def make_rng(seed: int) -> np.random.Generator:
    """Create a modern NumPy Generator from a deterministic integer seed."""
    return np.random.default_rng(np.random.SeedSequence(int(seed)))
