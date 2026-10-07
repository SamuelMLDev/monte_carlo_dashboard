"""Data-generator registry for pluggable simulation backends."""

from __future__ import annotations

from .base import DataGenerator
from .parametric import DEFAULT_GENERATOR

GENERATOR_REGISTRY: dict[str, DataGenerator] = {"parametric": DEFAULT_GENERATOR}


def register_generator(name: str, generator: DataGenerator, *, replace: bool = False) -> None:
    """Register a generator plug-in without modifying the simulation engine."""
    if name in GENERATOR_REGISTRY and not replace:
        raise ValueError(f"Generator already registered: {name}")
    GENERATOR_REGISTRY[name] = generator


def get_generator(name: str) -> DataGenerator:
    try:
        return GENERATOR_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown data generator: {name}") from exc
