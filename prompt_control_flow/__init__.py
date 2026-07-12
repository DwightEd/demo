"""Prompt-controlled residual-flow diagnostics."""

from .config import ExtractionConfig
from .data import ChainRecord, load_chain_records
from .extraction import MechanismExtraction, extract_chain_mechanisms
from .extractors import ICRResidualMismatchExtractor, PromptResidualFlowExtractor, UncertaintyExtractor
from .representation_geometry import GeometryAuditConfig, append_geometry_audit
from .spectral_chain_dynamics import SpectralChainConfig, append_spectral_chain_dynamics
from .teacher_forcing import ForwardCache, build_prompt_response, run_teacher_forcing

__all__ = [
    "ExtractionConfig",
    "ChainRecord",
    "load_chain_records",
    "ForwardCache",
    "MechanismExtraction",
    "build_prompt_response",
    "run_teacher_forcing",
    "extract_chain_mechanisms",
    "GeometryAuditConfig",
    "append_geometry_audit",
    "SpectralChainConfig",
    "append_spectral_chain_dynamics",
    "PromptResidualFlowExtractor",
    "UncertaintyExtractor",
    "ICRResidualMismatchExtractor",
]
