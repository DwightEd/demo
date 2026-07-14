"""Causal pullback package with dependency-light lazy public imports."""

from __future__ import annotations

from importlib import import_module


_EXPORTS = {
    "CausalPullbackArtifact": ("schema", "CausalPullbackArtifact"),
    "CausalPullbackAuditConfig": ("audit", "CausalPullbackAuditConfig"),
    "CausalPullbackConfig": ("schema", "CausalPullbackConfig"),
    "PullbackFeatureCollection": ("features", "PullbackFeatureCollection"),
    "PullbackSource": ("data", "PullbackSource"),
    "build_pullback_features": ("features", "build_pullback_features"),
    "load_ordered_processbench_questions": (
        "data",
        "load_ordered_processbench_questions",
    ),
    "load_pullback_source": ("data", "load_pullback_source"),
    "run_causal_pullback_audit": ("audit", "run_causal_pullback_audit"),
    "run_causal_pullback_extraction": (
        "extraction",
        "run_causal_pullback_extraction",
    ),
    "source_preflight": ("data", "source_preflight"),
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module_name, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
