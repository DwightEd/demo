"""Discriminative analysis of real Llama residual-stream trajectories."""

from .contracts import ChainSample, HiddenGeometryDataset, TraceSource
from .data import load_hidden_geometry_dataset, load_step_end_states
from .experiment import inspect_hidden_geometry_sources, run_hidden_geometry_experiment

__all__ = [
    "ChainSample",
    "HiddenGeometryDataset",
    "TraceSource",
    "inspect_hidden_geometry_sources",
    "load_hidden_geometry_dataset",
    "load_step_end_states",
    "run_hidden_geometry_experiment",
]
