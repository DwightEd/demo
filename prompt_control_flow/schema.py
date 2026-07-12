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
    }


def _contains_name(z: np.lib.npyio.NpzFile, key: str, target: str) -> bool:
    if key not in z.files:
        return False
    try:
        return target in [str(x) for x in z[key].tolist()]
    except Exception:
        return False
