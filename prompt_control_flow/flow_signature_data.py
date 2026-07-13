from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np


VECTOR_KEYS = (
    "sv_vec_step_exp",
    "stepvec",
    "sv_vec_mean",
    "step_layer_state_vectors",
    "step_state_vectors",
)


@dataclass
class FlowTrajectoryDataset:
    source_path: str
    vector_key: str
    trajectories: list[np.ndarray]
    original_indices: np.ndarray
    problem_ids: np.ndarray
    sample_idx: np.ndarray
    y_error: np.ndarray
    is_correct: np.ndarray
    n_steps: np.ndarray
    response_chars: np.ndarray
    layer_ids: np.ndarray
    hidden_dim: int
    label_policy: str
    skipped: dict[str, int]
    metadata: dict[str, Any]

    @property
    def n_samples(self) -> int:
        return len(self.trajectories)


def _scalar_text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    array = np.asarray(value, dtype=object)
    if array.size == 0:
        return default
    return str(array.reshape(-1)[0])


def resolve_vector_key(files: Sequence[str], requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in files:
            raise FileNotFoundError(
                f"requested vector key {requested!r} is absent; available vector keys: "
                f"{[key for key in VECTOR_KEYS if key in files]}"
            )
        return requested
    for key in VECTOR_KEYS:
        if key in files:
            return key
    raise FileNotFoundError(
        "no raw step-vector trajectory found. Expected one of "
        f"{VECTOR_KEYS}. Multisample extraction needs --store_vectors."
    )


def _resolve_layer_ids(z: np.lib.npyio.NpzFile, vector_key: str, depth: int) -> np.ndarray:
    if vector_key == "stepvec":
        candidates = ("sv_layers", "layers_used", "layers")
    elif vector_key.startswith("sv_vec_"):
        candidates = ("layers_used", "sv_layers", "layers")
    else:
        candidates = ("step_layer_state_vector_layers", "layers_used", "sv_layers", "layers")
    present: list[tuple[str, int]] = []
    for key in candidates:
        if key not in z.files:
            continue
        values = np.asarray(z[key]).reshape(-1)
        present.append((key, int(values.size)))
        if values.size == depth:
            return values.astype(np.int64)
    if present:
        raise ValueError(
            f"{vector_key} stores {depth} layers, but layer metadata sizes are {present}; "
            "refusing to guess the layer mapping"
        )
    return np.arange(depth, dtype=np.int64)


def parse_layer_selection(spec: str, layer_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(layer_ids, dtype=np.int64).reshape(-1)
    if values.size == 0:
        raise ValueError("empty layer axis")
    text = str(spec).strip().lower()
    if text in {"", "all"}:
        pos = np.arange(values.size, dtype=np.int64)
    elif text == "mid":
        lo = int(np.floor(values.size / 3))
        hi = int(np.ceil(2 * values.size / 3))
        pos = np.arange(lo, max(lo + 1, hi), dtype=np.int64)
    else:
        requested = [int(x.strip()) for x in text.split(",") if x.strip()]
        missing = [x for x in requested if x not in set(values.tolist())]
        if missing:
            raise ValueError(f"requested layers {missing} are absent; available={values.tolist()}")
        pos = np.asarray([int(np.where(values == x)[0][0]) for x in requested], dtype=np.int64)
    return pos, values[pos]


def _labels(
    z: np.lib.npyio.NpzFile,
    policy: str,
    n: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    policy = str(policy)
    if policy not in {"answer", "strict", "answer_format_ok", "processbench"}:
        raise ValueError(f"unknown label policy {policy!r}")
    gold = np.asarray(z["gold_error_step"], dtype=np.int64) if "gold_error_step" in z.files else None
    if policy == "processbench" or (
        gold is not None and "is_correct" not in z.files and "is_correct_strict" not in z.files
    ):
        y_error = (gold >= 0).astype(np.int64)
        mask = np.ones(n, dtype=bool)
    elif policy == "strict":
        if "is_correct_strict" not in z.files:
            raise ValueError("strict policy requires is_correct_strict")
        y_error = (np.asarray(z["is_correct_strict"], dtype=np.int64) == 0).astype(np.int64)
        mask = np.ones(n, dtype=bool)
    else:
        if "is_correct" not in z.files:
            if gold is None:
                raise ValueError("answer policy requires is_correct or gold_error_step")
            y_error = (gold >= 0).astype(np.int64)
        else:
            y_error = (np.asarray(z["is_correct"], dtype=np.int64) == 0).astype(np.int64)
        if policy == "answer_format_ok":
            if "format_ok" not in z.files:
                raise ValueError("answer_format_ok policy requires format_ok")
            mask = np.asarray(z["format_ok"], dtype=bool)
        else:
            mask = np.ones(n, dtype=bool)
    if y_error.shape[0] != n:
        raise ValueError("label length does not match vector records")
    return y_error, mask, 1 - y_error


def _trajectory_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    if array.ndim == 2:
        array = array[:, None, :]
    if array.ndim != 3:
        raise ValueError(f"expected [step, layer, hidden], got shape={array.shape}")
    return np.asarray(array, dtype=np.float32)


def inspect_flow_source(path: str | Path, vector_key: str = "auto") -> dict[str, Any]:
    path = Path(path)
    z = np.load(path, allow_pickle=True)
    key = resolve_vector_key(z.files, vector_key)
    raw = z[key]
    example = None
    for value in raw:
        try:
            candidate = _trajectory_array(value)
        except (TypeError, ValueError):
            continue
        if candidate.shape[0] >= 2:
            example = candidate
            break
    if example is None:
        raise ValueError(f"{path}: {key} contains no valid trajectory")
    layers = _resolve_layer_ids(z, key, example.shape[1])
    return {
        "path": str(path),
        "vector_key": key,
        "n_records": int(len(raw)),
        "example_shape": list(example.shape),
        "layer_ids": layers.tolist(),
        "has_same_problem_ids": "problem_ids" in z.files,
        "has_sample_idx": "sample_idx" in z.files,
        "has_answer_labels": "is_correct" in z.files,
        "has_strict_labels": "is_correct_strict" in z.files,
        "has_process_labels": "gold_error_step" in z.files,
        "model_name": _scalar_text(z["model_name"] if "model_name" in z.files else None),
    }


def load_flow_trajectory_dataset(
    path: str | Path,
    *,
    vector_key: str = "auto",
    layers: str = "all",
    label_policy: str = "answer_format_ok",
    max_samples: int = 0,
) -> FlowTrajectoryDataset:
    """Load canonical full or same-problem multisample hidden-state paths."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    z = np.load(path, allow_pickle=True)
    key = resolve_vector_key(z.files, vector_key)
    raw = z[key]
    n = int(len(raw))
    if "problem_ids" not in z.files:
        problem_ids = np.arange(n, dtype=np.int64)
    else:
        problem_ids = np.asarray(z["problem_ids"])
    sample_idx = (
        np.asarray(z["sample_idx"], dtype=np.int64)
        if "sample_idx" in z.files
        else np.arange(n, dtype=np.int64)
    )
    y_error, label_mask, is_correct = _labels(z, label_policy, n)
    responses = np.asarray(z["responses"], dtype=object) if "responses" in z.files else None

    example = None
    for value in raw:
        try:
            candidate = _trajectory_array(value)
        except (TypeError, ValueError):
            continue
        if candidate.shape[0] >= 2:
            example = candidate
            break
    if example is None:
        raise ValueError(f"{path}: no trajectory has at least two states")
    all_layer_ids = _resolve_layer_ids(z, key, example.shape[1])
    layer_positions, selected_layer_ids = parse_layer_selection(layers, all_layer_ids)

    trajectories: list[np.ndarray] = []
    original_indices: list[int] = []
    kept_problem_ids: list[Any] = []
    kept_sample_idx: list[int] = []
    kept_y: list[int] = []
    kept_correct: list[int] = []
    kept_steps: list[int] = []
    kept_chars: list[int] = []
    skipped = {"label_policy": 0, "invalid_shape": 0, "too_short": 0, "nonfinite": 0}

    for i, value in enumerate(raw):
        if max_samples and len(trajectories) >= int(max_samples):
            break
        if not bool(label_mask[i]):
            skipped["label_policy"] += 1
            continue
        try:
            trajectory = _trajectory_array(value)
        except (TypeError, ValueError):
            skipped["invalid_shape"] += 1
            continue
        if trajectory.shape[0] < 2:
            skipped["too_short"] += 1
            continue
        if trajectory.shape[1] != len(all_layer_ids):
            skipped["invalid_shape"] += 1
            continue
        trajectory = trajectory[:, layer_positions, :]
        if not np.isfinite(trajectory).all():
            skipped["nonfinite"] += 1
            continue
        trajectories.append(np.ascontiguousarray(trajectory, dtype=np.float32))
        original_indices.append(i)
        kept_problem_ids.append(problem_ids[i])
        kept_sample_idx.append(int(sample_idx[i]))
        kept_y.append(int(y_error[i]))
        kept_correct.append(int(is_correct[i]))
        kept_steps.append(int(trajectory.shape[0]))
        kept_chars.append(len(str(responses[i])) if responses is not None else 0)

    if not trajectories:
        raise ValueError(f"{path}: no valid trajectories remain after policy and schema filters")
    hidden_dim = int(trajectories[0].shape[2])
    if any(x.shape[1:] != (len(selected_layer_ids), hidden_dim) for x in trajectories):
        raise ValueError("selected trajectories do not share layer/hidden dimensions")

    metadata = {
        "model_name": _scalar_text(z["model_name"] if "model_name" in z.files else None),
        "prompt_style": _scalar_text(z["prompt_style"] if "prompt_style" in z.files else None),
        "step_split": _scalar_text(z["step_split"] if "step_split" in z.files else None),
        "available_layer_ids": all_layer_ids.tolist(),
        "selected_layer_positions": layer_positions.tolist(),
        "selected_layer_ids": selected_layer_ids.tolist(),
    }
    return FlowTrajectoryDataset(
        source_path=str(path.resolve()),
        vector_key=key,
        trajectories=trajectories,
        original_indices=np.asarray(original_indices, dtype=np.int64),
        problem_ids=np.asarray(kept_problem_ids),
        sample_idx=np.asarray(kept_sample_idx, dtype=np.int64),
        y_error=np.asarray(kept_y, dtype=np.int64),
        is_correct=np.asarray(kept_correct, dtype=np.int64),
        n_steps=np.asarray(kept_steps, dtype=np.int64),
        response_chars=np.asarray(kept_chars, dtype=np.int64),
        layer_ids=selected_layer_ids,
        hidden_dim=hidden_dim,
        label_policy=label_policy,
        skipped=skipped,
        metadata=metadata,
    )
