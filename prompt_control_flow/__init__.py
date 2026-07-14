"""Prompt-controlled residual-flow diagnostics.

Public symbols are loaded lazily so lightweight schema/statistics tools do not
import PyTorch, attention kernels, or plotting dependencies at package import
time.  Existing ``from prompt_control_flow import ...`` calls remain valid.
"""

from __future__ import annotations

from importlib import import_module


_LAZY_EXPORTS = {
    "ExtractionConfig": (".config", "ExtractionConfig"),
    "ChainRecord": (".data", "ChainRecord"),
    "load_chain_records": (".data", "load_chain_records"),
    "ForwardCache": (".teacher_forcing", "ForwardCache"),
    "MechanismExtraction": (".extraction", "MechanismExtraction"),
    "build_prompt_response": (".teacher_forcing", "build_prompt_response"),
    "run_teacher_forcing": (".teacher_forcing", "run_teacher_forcing"),
    "extract_chain_mechanisms": (".extraction", "extract_chain_mechanisms"),
    "GeometryAuditConfig": (".representation_geometry", "GeometryAuditConfig"),
    "append_geometry_audit": (".representation_geometry", "append_geometry_audit"),
    "LayerTimeGeometryConfig": (".layer_time_geometry", "LayerTimeGeometryConfig"),
    "append_layer_time_geometry": (
        ".layer_time_geometry",
        "append_layer_time_geometry",
    ),
    "LayerTimeValidationConfig": (".layer_time_evaluate", "LayerTimeValidationConfig"),
    "evaluate_layer_time_geometry": (
        ".layer_time_evaluate",
        "evaluate_layer_time_geometry",
    ),
    "SpectralChainConfig": (".spectral_chain_dynamics", "SpectralChainConfig"),
    "append_spectral_chain_dynamics": (
        ".spectral_chain_dynamics",
        "append_spectral_chain_dynamics",
    ),
    "PromptResidualFlowExtractor": (".extractors", "PromptResidualFlowExtractor"),
    "UncertaintyExtractor": (".extractors", "UncertaintyExtractor"),
    "ICRResidualMismatchExtractor": (".extractors", "ICRResidualMismatchExtractor"),
    "FlowSignatureConfig": (".flow_signatures", "FlowSignatureConfig"),
    "FlowSignatureEncoding": (".flow_signatures", "FlowSignatureEncoding"),
    "FlowTrajectoryDataset": (".flow_signature_data", "FlowTrajectoryDataset"),
    "FlowAuditConfig": (".flow_signature_audit", "FlowAuditConfig"),
    "encode_reasoning_flows": (".flow_signatures", "encode_reasoning_flows"),
    "load_flow_trajectory_dataset": (
        ".flow_signature_data",
        "load_flow_trajectory_dataset",
    ),
    "run_flow_signature_audit": (".flow_signature_audit", "run_flow_signature_audit"),
    "FirstErrorGeometryConfig": (".first_error_geometry", "FirstErrorGeometryConfig"),
    "GeometryAuditResult": (".first_error_geometry", "GeometryAuditResult"),
    "load_step_geometry_dataset": (
        ".first_error_geometry",
        "load_step_geometry_dataset",
    ),
    "load_token_axis": (".first_error_geometry", "load_token_axis"),
    "run_first_error_geometry_audit": (
        ".first_error_geometry",
        "run_first_error_geometry_audit",
    ),
}

__all__ = list(_LAZY_EXPORTS)


def __getattr__(name: str):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(name)
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
