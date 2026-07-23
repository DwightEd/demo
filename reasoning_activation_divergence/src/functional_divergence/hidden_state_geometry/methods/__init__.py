from __future__ import annotations

from importlib import import_module


def load_builtin_methods() -> None:
    """Import built-ins once; a new method only needs its own module import here."""
    import_module("functional_divergence.hidden_state_geometry.methods.raw_functional_probe")


__all__ = ["load_builtin_methods"]
