from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np


PROMPT_TEXT_KEYS = ("prompts", "questions", "problem_text", "problems", "problem")
RESPONSE_KEYS = ("responses", "response")
STEP_VECTOR_PREFIXES = ("sv_vec_",)
PROMPT_HIDDEN_KEYS = ("prompt_hidden", "prompt_states", "prompt_hidden_states")


def inspect_npz_schema(path: str | Path) -> Dict[str, Any]:
    """Inspect whether an existing npz can support prompt-control extraction.

    The central distinction is text vs hidden-state availability:
    saved prompt/problem text lets us reconstruct the prompt string, but prompt
    SVD needs prompt-token hidden states at extraction time.  Step vectors alone
    are not enough because they summarize response-step states after the prompt
    tokens have already been discarded.
    """

    path = Path(path)
    z = np.load(path, allow_pickle=True)
    files = tuple(str(k) for k in z.files)

    has_prompt_text = any(k in files for k in PROMPT_TEXT_KEYS)
    has_explicit_prompt = "prompts" in files
    has_steps_text = "steps_text" in files or "steps" in files
    has_response = any(k in files for k in RESPONSE_KEYS)
    has_stepvec = "stepvec" in files or any(k.startswith(STEP_VECTOR_PREFIXES) for k in files)
    has_qvec = "qvec" in files
    has_prompt_hidden = any(k in files for k in PROMPT_HIDDEN_KEYS)
    has_hidden_shards = ("hidden_files" in files or "hidden_ids" in files) and (
        "hidden_dir" in files or "hidden_layers" in files or "layers_used" in files
    )
    has_prompt_flow_metrics = "step_score_names" in files and "step_scores" in files and _contains_name(z, "step_score_names", "prompt_frac")
    has_layer_tensor = "step_layer_state_vectors" in files
    has_layer_memmap = "step_layer_state_memmap_path" in files
    has_mean_step_vectors = "sv_vec_mean" in files
    layer_values = None
    for key in ("step_layer_state_vector_layers", "layers_used", "sv_layers", "layers"):
        if key in files:
            layer_values = np.asarray(z[key], dtype=np.int64).reshape(-1)
            break
    if layer_values is not None and layer_values.size >= 2 and layer_values[0] == 0:
        # Exact sv writers include the embedding output; the LTG adapter drops it.
        layer_values = layer_values[1:]
    contiguous_layers = bool(
        layer_values is not None
        and layer_values.size >= 2
        and np.all(np.diff(layer_values) == 1)
    )
    pooling_kind = _scalar_value(z, "state_pooling_kind", "")
    if not pooling_kind and has_mean_step_vectors:
        pooling_kind = "arithmetic_mean_over_step_tokens"
    representation_kind = _scalar_value(z, "state_representation_kind", "")
    if not representation_kind:
        projected = bool(np.asarray(z["reasoning_subspace_used"]).item()) if "reasoning_subspace_used" in files else False
        representation_kind = "reasoning_subspace_projection" if projected else "hidden_state"
    has_layer_time_states = bool(has_layer_tensor or has_layer_memmap or has_mean_step_vectors)
    layer_time_mainline_ready = bool(
        has_layer_time_states
        and contiguous_layers
        and pooling_kind == "arithmetic_mean_over_step_tokens"
        and representation_kind == "hidden_state"
    )
    exact_required = {
        "trace_schema_version",
        "trace_token_add_special_tokens",
        "token_offset_convention",
        "step_token_range_convention",
        "span_range_convention",
        "prompts",
        "responses",
        "input_ids",
        "attention_mask",
        "token_offsets",
        "step_token_ranges",
        "response_token_ranges",
    }
    exact_trace_declared = "trace_schema_version" in files or "input_ids" in files
    exact_trace_complete = bool(exact_trace_declared and exact_required.issubset(files))

    can_reconstruct_prompt_text = has_prompt_text
    can_compute_prompt_svd_without_reextract = bool(has_prompt_hidden or has_hidden_shards)
    needs_teacher_forcing_reextract = not (has_prompt_flow_metrics or can_compute_prompt_svd_without_reextract)

    return {
        "path": str(path),
        "n_files": len(files),
        "keys": files,
        "has_prompt_text": bool(has_prompt_text),
        "has_explicit_prompt": bool(has_explicit_prompt),
        "has_steps_text": bool(has_steps_text),
        "has_response": bool(has_response),
        "has_step_vectors": bool(has_stepvec),
        "has_qvec_anchor": bool(has_qvec),
        "has_prompt_hidden": bool(has_prompt_hidden),
        "has_hidden_shards": bool(has_hidden_shards),
        "has_prompt_flow_metrics": bool(has_prompt_flow_metrics),
        "can_reconstruct_prompt_text": bool(can_reconstruct_prompt_text),
        "can_compute_prompt_svd_without_reextract": bool(can_compute_prompt_svd_without_reextract),
        "needs_teacher_forcing_reextract": bool(needs_teacher_forcing_reextract),
        "has_layer_time_states": has_layer_time_states,
        "has_layer_state_memmap": bool(has_layer_memmap),
        "layer_time_contiguous_layers": contiguous_layers,
        "layer_time_pooling_kind": pooling_kind,
        "layer_time_representation_kind": representation_kind,
        "layer_time_mainline_ready": layer_time_mainline_ready,
        "exact_trace_declared": exact_trace_declared,
        "exact_trace_complete": exact_trace_complete,
    }


def _contains_name(z: np.lib.npyio.NpzFile, key: str, target: str) -> bool:
    if key not in z.files:
        return False
    try:
        return target in [str(x) for x in z[key].tolist()]
    except Exception:
        return False


def _scalar_value(z: np.lib.npyio.NpzFile, key: str, default: str) -> str:
    if key not in z.files:
        return default
    try:
        return str(np.asarray(z[key]).item())
    except Exception:
        return default
