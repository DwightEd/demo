from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class Trace:
    idx: int
    chain_id: str
    problem_id: int
    dataset: str
    correct: bool
    gold_error_step: int
    step_token_ranges: np.ndarray
    steps_text: List[str]
    response_text: str
    prompt_text: str
    features: Dict[str, np.ndarray]
    stepvec: Optional[np.ndarray]
    qvec: Optional[np.ndarray]
    sv_layers: List[int]
    hidden_path: Optional[str]
    layer: int
    prompt_offsets: Optional[np.ndarray] = None
    question_char_span: Optional[Tuple[int, int]] = None
    prompt_hidden: Optional[np.ndarray] = None
    prompt_hidden_layers: List[int] = field(default_factory=list)
    kept_steps: Optional[np.ndarray] = None
    step_clouds: Optional[List[np.ndarray]] = None
    schema_version: str = ""
    time_axis_kind: str = "step"

    @property
    def n_steps(self) -> int:
        return int(len(self.step_token_ranges))


@dataclass
class StepWindow:
    trace_idx: int
    step_id: int
    token_start: int
    token_end: int


def unit(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    return v / max(float(np.linalg.norm(v)), EPS)


def layer_index(layers: List[int], layer: int, *, nearest: bool = False) -> Optional[int]:
    vals = [int(x) for x in layers]
    if layer in vals:
        return vals.index(layer)
    if nearest and vals:
        return int(np.argmin([abs(x - layer) for x in vals]))
    return None


def _obj_get(arr, i, default=None):
    if arr is None or i >= len(arr):
        return default
    val = arr[i]
    return default if val is None else val


def _str_array_get(arr, i, default: str = "") -> str:
    val = _obj_get(arr, i, default)
    if val is None:
        return default
    return str(val)


def _scalar_string(value, default: str = "") -> str:
    if value is None:
        return default
    arr = np.asarray(value)
    if arr.size == 0:
        return default
    return str(arr.reshape(-1)[0])


def _select_layer_tensor(value, layers: List[int], layer: int) -> Optional[np.ndarray]:
    """Select ``[tokens, hidden]`` from ``[tokens, layers, hidden]`` data."""
    if value is None:
        return None
    arr = np.asarray(value, float)
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3 or arr.shape[1] == 0:
        return None
    li = layer_index(layers, layer, nearest=True) if layers else 0
    return arr[:, int(li), :] if li is not None else None


def _split_step_clouds(value, sizes, layers: List[int], layer: int) -> Optional[List[np.ndarray]]:
    """Recover per-step clouds from the compact concatenated NPZ payload."""
    if value is None or sizes is None:
        return None
    cloud = np.asarray(value, float)
    counts = np.asarray(sizes, int).reshape(-1)
    if cloud.ndim == 3:
        li = layer_index(layers, layer, nearest=True) if layers else 0
        cloud = cloud[:, int(li), :] if li is not None else cloud[:, 0, :]
    if cloud.ndim != 2 or np.any(counts < 0) or int(counts.sum()) != len(cloud):
        return None
    cuts = np.cumsum(counts)[:-1]
    return [np.asarray(x, float) for x in np.split(cloud, cuts)]


def _step_ranges(rng: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(rng, int)
    if rr.ndim != 2 or rr.shape[1] != 2:
        return np.array([], float), np.array([], float)
    # Existing project features use inclusive-looking ranges in several audits.
    n_tok = np.maximum(1, rr[:, 1] - rr[:, 0] + 1).astype(float)
    pos = np.arange(len(rr), dtype=float) / max(1, len(rr) - 1)
    return n_tok, pos


def _per_step_mean(tok: Optional[np.ndarray], rng: np.ndarray) -> np.ndarray:
    out = np.full(len(rng), np.nan)
    if tok is None:
        return out
    arr = np.asarray(tok, float)
    if len(arr) == 0:
        return out
    a0 = int(rng[0, 0])
    for t, (lo0, hi0) in enumerate(np.asarray(rng, int)):
        lo = max(0, int(lo0) - a0)
        hi = min(len(arr), int(hi0) - a0 + 1)
        if hi > lo:
            out[t] = float(np.nanmean(arr[lo:hi]))
    return out


def _delta(x: np.ndarray) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) > 1:
        out[1:] = v[1:] - v[:-1]
    return out


def _hidden_path(hidden_dir: Optional[str], hidden_files, ids, i: int) -> Optional[str]:
    if not hidden_dir:
        return None
    fname = _obj_get(hidden_files, i, None)
    if fname is None and ids is not None:
        raw = _obj_get(ids, i, i)
        fname = "".join(c if c.isalnum() or c in "._-" else "_" for c in str(raw)) + ".npy"
    if fname is None:
        return None
    path = os.path.join(hidden_dir, str(fname))
    return path if os.path.exists(path) else path


def load_traces(
    npz_path: str,
    *,
    dataset: str = "",
    layer: int = 14,
    max_chains: int = 0,
    hidden_dir: Optional[str] = None,
) -> Tuple[List[Trace], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    files = set(z.files)
    if "gold_error_step" in files:
        ges = z["gold_error_step"].astype(int)
        correct_flags = ges < 0
    elif "is_correct" in files:
        correct_flags = z["is_correct"].astype(bool)
        # -2 means incorrect but first-error location is not annotated.  Such
        # chains remain usable for chain-level analysis but are masked from the
        # first-error hazard risk set by ``make_labels``.
        ges = np.where(correct_flags, -1, -2).astype(int)
    else:
        raise KeyError("NPZ needs gold_error_step or is_correct")
    groups = z["problem_ids"].astype(int) if "problem_ids" in files else np.arange(len(ges))
    ids = z["ids"] if "ids" in files else np.arange(len(ges)).astype(object)
    source = z["source"] if "source" in files else None
    ranges = z["step_token_ranges"]
    kept_steps_arr = z["kept_steps"] if "kept_steps" in files else None
    steps_text = z["steps_text"] if "steps_text" in files else None
    responses = z["responses"] if "responses" in files else None
    prompts = None
    for key in ("prompts", "questions", "problem_text", "problems"):
        if key in files:
            prompts = z[key]
            break

    stepcloud = z["stepcloud"] if "stepcloud" in files else None
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in files else []
    cloud_layers = [int(x) for x in z["layers_used"]] if "layers_used" in files else []
    ci = layer_index(cloud_layers, layer, nearest=False)

    tok_ud = z["tok_U_D"] if "tok_U_D" in files else None
    tok_uc = z["tok_U_C"] if "tok_U_C" in files else None
    stepvec = z["stepvec"] if "stepvec" in files else None
    qvec = z["qvec"] if "qvec" in files else None
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in files else []
    svi = layer_index(sv_layers, layer, nearest=True) if stepvec is not None and sv_layers else None

    hidden_files = z["hidden_files"] if "hidden_files" in files else None
    if hidden_dir is None and "hidden_dir" in files:
        val = np.asarray(z["hidden_dir"]).item()
        hidden_dir = str(val) if val else None

    prompt_offsets_arr = z["token_offsets"] if "token_offsets" in files else None
    prompt_counts = z["prompt_token_counts"] if "prompt_token_counts" in files else None
    question_spans = z["question_char_spans"] if "question_char_spans" in files else None
    prompt_hidden_arr = z["prompt_hidden"] if "prompt_hidden" in files else None
    prompt_hidden_layers = (
        [int(x) for x in np.asarray(z["prompt_hidden_layers"]).reshape(-1)]
        if "prompt_hidden_layers" in files else []
    )
    raw_clouds = z["sv_clouds"] if "sv_clouds" in files else None
    raw_cloud_sizes = z["cloud_sizes"] if "cloud_sizes" in files else None
    raw_cloud_layers = (
        [int(x) for x in np.asarray(z["cloud_layers"]).reshape(-1)]
        if "cloud_layers" in files else []
    )
    schema_version = _scalar_string(z["trace_schema_version"], "") if "trace_schema_version" in files else ""
    time_axis_kind = _scalar_string(z["time_axis_kind"], "step") if "time_axis_kind" in files else "step"

    n = len(ges) if not max_chains else min(int(max_chains), len(ges))
    traces: List[Trace] = []
    missing = {"stepvec": 0, "qvec": 0, "prompt_text": 0, "stepcloud": 0}
    for i in range(n):
        rng = np.asarray(ranges[i], int)
        if rng.ndim != 2 or len(rng) == 0:
            continue
        kept = (
            np.asarray(_obj_get(kept_steps_arr, i, np.arange(len(rng))), int).reshape(-1)
            if kept_steps_arr is not None else np.arange(len(rng), dtype=int)
        )
        if len(kept) != len(rng):
            continue
        original_error = int(ges[i])
        mapped_error = original_error
        if original_error >= 0 and kept_steps_arr is not None:
            hit = np.where(kept == original_error)[0]
            mapped_error = int(hit[0]) if hit.size else -2
        T = len(rng)
        n_tok, pos = _step_ranges(rng)
        feats: Dict[str, np.ndarray] = {
            "n_tok": n_tok,
            "logN": np.log1p(n_tok),
            "pos": pos,
        }

        if stepcloud is not None and ci is not None and stepcloud[i] is not None:
            sc = np.asarray(stepcloud[i], float)
            for name in ("resultant", "coherence", "cloud_D", "cloud_V", "cloud_C"):
                if name in cloud_names and sc.ndim == 3 and sc.shape[0] >= T:
                    feats[name] = sc[:T, ci, cloud_names.index(name)]
        else:
            missing["stepcloud"] += 1

        ud = np.asarray(tok_ud[i], float) if tok_ud is not None else None
        uc = np.asarray(tok_uc[i], float) if tok_uc is not None else None
        feats["U_D_mean"] = _per_step_mean(ud, rng)
        feats["U_C_mean"] = _per_step_mean(uc, rng)

        sv_sel = None
        q_sel = None
        if stepvec is not None and svi is not None and stepvec[i] is not None:
            sv = np.asarray(stepvec[i], float)
            if sv.ndim == 3 and sv.shape[0] >= T:
                sv_sel = sv[:T, svi, :]
        if sv_sel is None:
            missing["stepvec"] += 1
        if qvec is not None:
            q_all = np.asarray(qvec)
            if q_all.dtype != object and q_all.ndim == 1:
                qraw = np.asarray(q_all, float)
            else:
                qraw = np.asarray(_obj_get(qvec, i, None), float)
            if qraw.ndim == 2:
                qi = min(svi if svi is not None else 0, qraw.shape[0] - 1)
                q_sel = qraw[qi]
            elif qraw.ndim == 1:
                q_sel = qraw
        if q_sel is None:
            missing["qvec"] += 1

        if sv_sel is not None and q_sel is not None:
            q_unit = unit(q_sel)
            dirs = np.asarray([unit(v) for v in sv_sel], float)
            feats["q_align"] = dirs @ q_unit
            jump = np.full(T, np.nan)
            if T > 1:
                jump[1:] = 1.0 - np.sum(dirs[1:] * dirs[:-1], axis=1)
            feats["step_direction_jump"] = jump

        if "resultant" in feats:
            feats["spread"] = 1.0 - np.asarray(feats["resultant"], float)
        elif "coherence" in feats:
            feats["spread"] = 1.0 - np.asarray(feats["coherence"], float)
        else:
            feats["spread"] = np.full(T, np.nan)
        if "q_align" in feats:
            feats["anchor_loss"] = 1.0 - np.asarray(feats["q_align"], float)
        else:
            feats["anchor_loss"] = np.full(T, np.nan)
        feats["d_spread"] = _delta(feats["spread"])
        feats["d_anchor_loss"] = _delta(feats["anchor_loss"])

        st = [str(x) for x in list(_obj_get(steps_text, i, []))] if steps_text is not None else []
        prompt_text = _str_array_get(prompts, i, "") if prompts is not None else ""
        if not prompt_text:
            missing["prompt_text"] += 1

        p_offsets = None
        if prompt_offsets_arr is not None:
            raw_offsets = np.asarray(_obj_get(prompt_offsets_arr, i, None), int)
            if raw_offsets.ndim == 2 and raw_offsets.shape[1] == 2:
                pcount = int(prompt_counts[i]) if prompt_counts is not None else len(raw_offsets)
                p_offsets = raw_offsets[:pcount]
                feats["prompt_offsets"] = p_offsets
        p_hidden = _select_layer_tensor(
            _obj_get(prompt_hidden_arr, i, None) if prompt_hidden_arr is not None else None,
            prompt_hidden_layers,
            layer,
        )
        if p_hidden is not None:
            if p_offsets is not None and len(p_hidden) != len(p_offsets):
                p_hidden = None
            else:
                feats["prompt_hidden"] = p_hidden
        q_span = None
        if question_spans is not None:
            raw_span = np.asarray(question_spans[i], int).reshape(-1)
            if len(raw_span) == 2 and raw_span[0] >= 0:
                q_span = (int(raw_span[0]), int(raw_span[1]))
        step_clouds = _split_step_clouds(
            _obj_get(raw_clouds, i, None) if raw_clouds is not None else None,
            _obj_get(raw_cloud_sizes, i, None) if raw_cloud_sizes is not None else None,
            raw_cloud_layers,
            layer,
        )
        if step_clouds is not None and len(step_clouds) != T:
            step_clouds = None
        traces.append(
            Trace(
                idx=i,
                chain_id=str(_obj_get(ids, i, i)),
                problem_id=int(groups[i]),
                dataset=dataset or str(_obj_get(source, i, "")),
                correct=bool(correct_flags[i]),
                gold_error_step=int(mapped_error),
                step_token_ranges=rng,
                steps_text=st,
                response_text=_str_array_get(responses, i, ""),
                prompt_text=prompt_text,
                features=feats,
                stepvec=sv_sel,
                qvec=q_sel,
                sv_layers=sv_layers,
                hidden_path=_hidden_path(hidden_dir, hidden_files, ids, i),
                layer=int(layer),
                prompt_offsets=p_offsets,
                question_char_span=q_span,
                prompt_hidden=p_hidden,
                prompt_hidden_layers=prompt_hidden_layers,
                kept_steps=kept,
                step_clouds=step_clouds,
                schema_version=schema_version,
                time_axis_kind=time_axis_kind,
            )
        )

    meta = {
        "npz": npz_path,
        "dataset": dataset,
        "layer": int(layer),
        "n_loaded": len(traces),
        "sv_layers": sv_layers,
        "cloud_layers": cloud_layers,
        "has_stepvec": bool(stepvec is not None),
        "has_qvec": bool(qvec is not None),
        "has_prompt_text": bool(prompts is not None),
        "has_exact_trace": bool(schema_version),
        "has_prompt_hidden": bool(prompt_hidden_arr is not None),
        "has_step_clouds": bool(raw_clouds is not None),
        "trace_schema_version": schema_version,
        "time_axis_kind": time_axis_kind,
        "missing": missing,
    }
    return traces, meta


def iter_step_windows(trace: Trace) -> Iterator[StepWindow]:
    for t, (lo, hi) in enumerate(np.asarray(trace.step_token_ranges, int)):
        yield StepWindow(trace.idx, int(t), int(lo), int(hi))


def make_labels(trace: Trace, *, mask_post_error: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    y = np.zeros(trace.n_steps, dtype=int)
    mask = np.ones(trace.n_steps, dtype=bool)
    if not trace.correct and 0 <= trace.gold_error_step < trace.n_steps:
        y[trace.gold_error_step] = 1
        if mask_post_error:
            mask[np.arange(trace.n_steps) > trace.gold_error_step] = False
    elif not trace.correct:
        # A chain-level error label without a localized first-error annotation
        # is not a right-censored clean chain.  Mask it rather than fabricating
        # negative hazard targets.
        mask[:] = False
    return y, mask
