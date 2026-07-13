"""Prompt-controlled residual-flow diagnostics."""

from .config import ExtractionConfig
from .data import ChainRecord, load_chain_records
from .extraction import MechanismExtraction, extract_chain_mechanisms
from .extractors import ICRResidualMismatchExtractor, PromptResidualFlowExtractor, UncertaintyExtractor
from .flow_signature_audit import FlowAuditConfig, run_flow_signature_audit
from .flow_signature_data import FlowTrajectoryDataset, load_flow_trajectory_dataset
from .flow_signatures import FlowSignatureConfig, FlowSignatureEncoding, encode_reasoning_flows
from .first_error_geometry import (
    FirstErrorGeometryConfig,
    GeometryAuditResult,
    load_step_geometry_dataset,
    load_token_axis,
    run_first_error_geometry_audit,
)
from .layer_time_geometry import LayerTimeGeometryConfig, append_layer_time_geometry
from .layer_time_evaluate import LayerTimeValidationConfig, evaluate_layer_time_geometry
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
    "LayerTimeGeometryConfig",
    "append_layer_time_geometry",
    "LayerTimeValidationConfig",
    "evaluate_layer_time_geometry",
    "SpectralChainConfig",
    "append_spectral_chain_dynamics",
    "PromptResidualFlowExtractor",
    "UncertaintyExtractor",
    "ICRResidualMismatchExtractor",
    "FlowSignatureConfig",
    "FlowSignatureEncoding",
    "FlowTrajectoryDataset",
    "FlowAuditConfig",
    "encode_reasoning_flows",
    "load_flow_trajectory_dataset",
    "run_flow_signature_audit",
    "FirstErrorGeometryConfig",
    "GeometryAuditResult",
    "load_step_geometry_dataset",
    "load_token_axis",
    "run_first_error_geometry_audit",
]
