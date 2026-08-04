from __future__ import annotations

from importlib import import_module


def load_builtin_methods() -> None:
    """Import built-ins once; a new method only needs its own module import here."""
    import_module("functional_divergence.hidden_state_geometry.methods.raw_functional_probe")
    import_module("functional_divergence.hidden_state_geometry.methods.full_tensor_ridge")
    import_module("functional_divergence.hidden_state_geometry.methods.innovation_hazard")
    import_module(
        "functional_divergence.hidden_state_geometry.methods.component_resolved_hazard"
    )


__all__ = ["load_builtin_methods"]
