"""Causal belief-update decomposition in pretrained transformers."""

from .world import AliasWorldConfig, PredictiveAliasWorld, generate_alias_worlds

__all__ = [
    "AliasWorldConfig",
    "PredictiveAliasWorld",
    "generate_alias_worlds",
]
