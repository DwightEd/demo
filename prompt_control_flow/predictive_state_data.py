from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from .directional_consensus import (
    DirectionalCloudDataset,
    load_directional_cloud_dataset,
)


@dataclass
class PredictiveStateDataset:
    cloud: DirectionalCloudDataset
    token_ids: list[np.ndarray]
    token_positions: list[np.ndarray]
    token_range_key: str

    @property
    def n_samples(self) -> int:
        return self.cloud.n_samples


@dataclass(frozen=True)
class ProjectionConfig:
    projection_dim: int = 96
    batch_size: int = 64
    max_batch_tokens: int = 8192
    seed: int = 13
    compute_device: str = "cuda"

    def validate(self) -> None:
        if self.projection_dim < 2:
            raise ValueError("projection_dim must be at least 2")
        if self.batch_size <= 0 or self.max_batch_tokens <= 0:
            raise ValueError("projection batch limits must be positive")


@dataclass(frozen=True)
class WindowConfig:
    window_tokens: int = 16
    window_stride: int = 16
    max_skipped_tokens: int = 4
    window_batch_size: int = 4096
    compute_device: str = "cuda"

    def validate(self) -> None:
        if self.window_tokens < 2:
            raise ValueError("window_tokens must be at least 2")
        if self.window_stride <= 0:
            raise ValueError("window_stride must be positive")
        if self.max_skipped_tokens < 0:
            raise ValueError("max_skipped_tokens must be non-negative")
        if self.window_batch_size <= 0:
            raise ValueError("window_batch_size must be positive")


@dataclass
class TransitionBundle:
    x: np.ndarray
    y: np.ndarray
    weights: np.ndarray
    sample_indices: np.ndarray
    problem_ids: np.ndarray
    transition_positions: np.ndarray
    horizon: int

    @property
    def n_rows(self) -> int:
        return int(self.x.shape[0])


def _flat_int_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    try:
        array = np.asarray(array, dtype=np.int64).reshape(-1)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not a flat integer array") from exc
    return array


def _range_array(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.dtype == object:
        array = np.asarray(value.tolist())
    try:
        array = np.asarray(array, dtype=np.int64).reshape(-1, 2)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} is not an integer [time, 2] array") from exc
    return array


def _resolve_time_range_key(files: Sequence[str]) -> str:
    for key in ("time_axis_token_ranges", "step_token_ranges"):
        if key in files:
            return key
    raise FileNotFoundError(
        "predictive-state analysis requires time_axis_token_ranges or "
        "step_token_ranges so every cloud token can be tied to its exact token ID"
    )


def load_predictive_state_dataset(
    path: str | Path,
    *,
    vector_key: str = "auto",
    cloud_layers: str = "all",
    label_policy: str = "answer_format_ok",
    max_samples: int = 0,
) -> PredictiveStateDataset:
    """Load token clouds with exact cloud-to-token alignment.

    The stored cloud is a concatenation of inclusive semantic-step token
    ranges. We reconstruct the corresponding token IDs from the exact model
    input and reject the artifact if any stored cloud size disagrees with its
    declared absolute token range. No response re-tokenization fallback is
    allowed for the confirmatory analysis.
    """

    path = Path(path)
    cloud = load_directional_cloud_dataset(
        path,
        vector_key=vector_key,
        cloud_layers=cloud_layers,
        label_policy=label_policy,
        max_samples=max_samples,
    )
    z = np.load(path, allow_pickle=True)
    if "input_ids" not in z.files:
        raise FileNotFoundError(
            f"{path}: missing input_ids; exact lexical-nuisance control cannot be run"
        )
    range_key = _resolve_time_range_key(z.files)
    raw_input_ids = z["input_ids"]
    raw_ranges = z[range_key]

    token_ids: list[np.ndarray] = []
    token_positions: list[np.ndarray] = []
    for local_index, original in enumerate(cloud.base.original_indices.tolist()):
        ids = _flat_int_array(raw_input_ids[int(original)], "input_ids")
        ranges = _range_array(raw_ranges[int(original)], range_key)
        sizes = np.asarray(cloud.step_sizes[local_index], dtype=np.int64).reshape(-1)
        if ranges.shape[0] != sizes.size:
            raise ValueError(
                f"record {original}: {range_key} has {ranges.shape[0]} rows but "
                f"cloud_sizes has {sizes.size}"
            )
        pieces: list[np.ndarray] = []
        positions: list[np.ndarray] = []
        for step, ((start, stop), expected) in enumerate(zip(ranges.tolist(), sizes.tolist())):
            start, stop, expected = int(start), int(stop), int(expected)
            if start < 0 or stop < start or stop >= ids.size:
                raise ValueError(
                    f"record {original} step {step}: invalid inclusive token range "
                    f"[{start}, {stop}] for input length {ids.size}"
                )
            if stop - start + 1 != expected:
                raise ValueError(
                    f"record {original} step {step}: range length {stop - start + 1} "
                    f"does not equal cloud size {expected}"
                )
            pieces.append(ids[start : stop + 1])
            positions.append(np.arange(start, stop + 1, dtype=np.int64))
        aligned_ids = np.concatenate(pieces)
        aligned_positions = np.concatenate(positions)
        cloud_tokens = int(cloud.clouds[local_index].shape[0])
        if aligned_ids.size != cloud_tokens:
            raise ValueError(
                f"record {original}: reconstructed {aligned_ids.size} token IDs but "
                f"cloud stores {cloud_tokens} states"
            )
        token_ids.append(np.ascontiguousarray(aligned_ids, dtype=np.int64))
        token_positions.append(np.ascontiguousarray(aligned_positions, dtype=np.int64))
    return PredictiveStateDataset(
        cloud=cloud,
        token_ids=token_ids,
        token_positions=token_positions,
        token_range_key=range_key,
    )


def inspect_predictive_state_source(
    path: str | Path,
    *,
    vector_key: str = "auto",
    cloud_layers: str = "all",
    label_policy: str = "answer_format_ok",
    max_samples: int = 0,
) -> dict[str, Any]:
    dataset = load_predictive_state_dataset(
        path,
        vector_key=vector_key,
        cloud_layers=cloud_layers,
        label_policy=label_policy,
        max_samples=max_samples,
    )
    base = dataset.cloud.base
    contrastive = sum(
        np.any(base.y_error[base.problem_ids == problem] == 0)
        and np.any(base.y_error[base.problem_ids == problem] == 1)
        for problem in np.unique(base.problem_ids)
    )
    return {
        "path": str(Path(path)),
        "samples": dataset.n_samples,
        "errors": int(np.sum(base.y_error == 1)),
        "correct": int(np.sum(base.y_error == 0)),
        "problems": int(np.unique(base.problem_ids).size),
        "contrastive_problems": int(contrastive),
        "cloud_layers": dataset.cloud.cloud_layer_ids.tolist(),
        "cloud_hidden_dim": int(dataset.cloud.cloud_hidden_dim),
        "token_range_key": dataset.token_range_key,
        "exact_token_alignment": True,
        "min_tokens": int(min(len(x) for x in dataset.token_ids)),
        "median_tokens": float(np.median([len(x) for x in dataset.token_ids])),
        "max_tokens": int(max(len(x) for x in dataset.token_ids)),
        "ready": bool(contrastive > 0),
    }


def _batch_positions(lengths: np.ndarray, batch_size: int, max_tokens: int) -> list[np.ndarray]:
    order = np.argsort(np.asarray(lengths, dtype=np.int64), kind="stable")
    batches: list[np.ndarray] = []
    current: list[int] = []
    total = 0
    for index in order.tolist():
        length = int(lengths[index])
        if current and (len(current) >= batch_size or total + length > max_tokens):
            batches.append(np.asarray(current, dtype=np.int64))
            current, total = [], 0
        current.append(index)
        total += length
    if current:
        batches.append(np.asarray(current, dtype=np.int64))
    return batches


@torch.inference_mode()
def project_token_clouds(
    dataset: PredictiveStateDataset,
    cfg: ProjectionConfig,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Apply a fixed label-free Gaussian sketch to every selected layer."""

    cfg.validate()
    device = torch.device(cfg.compute_device)
    layers = int(dataset.cloud.cloud_layer_ids.size)
    hidden = int(dataset.cloud.cloud_hidden_dim)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(cfg.seed))
    projection = torch.randn(
        (layers, hidden, cfg.projection_dim),
        generator=generator,
        device=device,
        dtype=torch.float32,
    ) / math.sqrt(float(cfg.projection_dim))
    lengths = np.asarray([cloud.shape[0] for cloud in dataset.cloud.clouds], dtype=np.int64)
    outputs: list[np.ndarray | None] = [None] * dataset.n_samples
    for batch in _batch_positions(lengths, cfg.batch_size, cfg.max_batch_tokens):
        tensors = [
            torch.as_tensor(
                dataset.cloud.clouds[int(index)],
                device=device,
                dtype=torch.float32,
            )
            for index in batch.tolist()
        ]
        joined = torch.cat(tensors, dim=0)
        sketched = torch.einsum("nld,ldp->nlp", joined, projection)
        sketched = sketched.reshape(sketched.shape[0], layers * cfg.projection_dim)
        offset = 0
        for index, tensor in zip(batch.tolist(), tensors):
            stop = offset + int(tensor.shape[0])
            outputs[int(index)] = np.ascontiguousarray(
                sketched[offset:stop].detach().cpu().numpy(), dtype=np.float32
            )
            offset = stop
    if any(value is None for value in outputs):
        raise RuntimeError("projection did not produce every response")
    return [value for value in outputs if value is not None], {
        "projection_kind": "fixed_gaussian_jl",
        "projection_dim_per_layer": int(cfg.projection_dim),
        "projected_dim": int(layers * cfg.projection_dim),
        "seed": int(cfg.seed),
    }


@torch.inference_mode()
def build_window_observations(
    sequences: Sequence[np.ndarray],
    token_positions: Sequence[np.ndarray],
    cfg: WindowConfig,
) -> tuple[list[np.ndarray], list[np.ndarray], np.ndarray]:
    """Create batched fixed-token observations without semantic aggregation.

    Each observation concatenates the window mean, endpoint, and displacement.
    A window is excluded when omitted model-input tokens make its absolute span
    exceed ``window_tokens + max_skipped_tokens``.
    """

    cfg.validate()
    if len(sequences) != len(token_positions):
        raise ValueError("sequence/token-position lengths differ")
    if not sequences:
        return [], [], np.empty(0, dtype=np.int64)
    feature_dim = int(np.asarray(sequences[0]).shape[1])
    index_rows: list[np.ndarray] = []
    sample_rows: list[int] = []
    span_rows: list[tuple[int, int]] = []
    offsets = np.cumsum([0] + [len(sequence) for sequence in sequences[:-1]])
    for sample, (sequence, positions, offset) in enumerate(
        zip(sequences, token_positions, offsets.tolist())
    ):
        sequence = np.asarray(sequence)
        positions = np.asarray(positions, dtype=np.int64)
        if sequence.ndim != 2 or sequence.shape[1] != feature_dim:
            raise ValueError("projected sequences do not share feature dimensions")
        if sequence.shape[0] != positions.size:
            raise ValueError("projected sequence/token-position mismatch")
        for start in range(0, sequence.shape[0] - cfg.window_tokens + 1, cfg.window_stride):
            stop = start + cfg.window_tokens
            absolute_span = int(positions[stop - 1] - positions[start] + 1)
            if absolute_span > cfg.window_tokens + cfg.max_skipped_tokens:
                continue
            index_rows.append(np.arange(start + offset, stop + offset, dtype=np.int64))
            sample_rows.append(sample)
            span_rows.append((int(positions[start]), int(positions[stop - 1])))
    outputs: list[list[np.ndarray]] = [[] for _ in sequences]
    spans: list[list[np.ndarray]] = [[] for _ in sequences]
    if not index_rows:
        return (
            [np.empty((0, 3 * feature_dim), dtype=np.float32) for _ in sequences],
            [np.empty((0, 2), dtype=np.int64) for _ in sequences],
            np.zeros(len(sequences), dtype=np.int64),
        )
    device = torch.device(cfg.compute_device)
    joined = torch.as_tensor(
        np.concatenate([np.asarray(x, dtype=np.float32) for x in sequences], axis=0),
        device=device,
        dtype=torch.float32,
    )
    all_indices = np.stack(index_rows)
    for start in range(0, all_indices.shape[0], cfg.window_batch_size):
        stop = min(start + cfg.window_batch_size, all_indices.shape[0])
        index = torch.as_tensor(all_indices[start:stop], device=device, dtype=torch.long)
        windows = joined[index]
        observation = torch.cat(
            [windows.mean(dim=1), windows[:, -1], windows[:, -1] - windows[:, 0]],
            dim=-1,
        ).detach().cpu().numpy()
        for row, sample in enumerate(sample_rows[start:stop]):
            outputs[sample].append(np.ascontiguousarray(observation[row], dtype=np.float32))
            spans[sample].append(np.asarray(span_rows[start + row], dtype=np.int64))
    packed = [
        np.stack(rows).astype(np.float32, copy=False)
        if rows
        else np.empty((0, 3 * feature_dim), dtype=np.float32)
        for rows in outputs
    ]
    packed_spans = [
        np.stack(rows).astype(np.int64, copy=False)
        if rows
        else np.empty((0, 2), dtype=np.int64)
        for rows in spans
    ]
    return packed, packed_spans, np.asarray([len(rows) for rows in packed], dtype=np.int64)


def build_transition_bundle(
    window_observations: Sequence[np.ndarray],
    window_token_ranges: Sequence[np.ndarray],
    sample_indices: Sequence[int],
    problem_ids: np.ndarray,
    *,
    horizon: int,
    context_windows: int = 1,
    max_transition_gap: int = 4,
) -> TransitionBundle:
    if horizon <= 0 or context_windows <= 0:
        raise ValueError("horizon and context_windows must be positive")
    x_rows: list[np.ndarray] = []
    y_rows: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    sample_rows: list[np.ndarray] = []
    problem_rows: list[np.ndarray] = []
    positions: list[np.ndarray] = []
    for sample in sample_indices:
        sample = int(sample)
        observations = np.asarray(window_observations[sample], dtype=np.float32)
        ranges = np.asarray(window_token_ranges[sample], dtype=np.int64).reshape(-1, 2)
        if observations.shape[0] != ranges.shape[0]:
            raise ValueError("window observation/range count mismatch")
        starts = np.arange(context_windows - 1, observations.shape[0] - horizon, dtype=np.int64)
        if starts.size:
            valid = []
            for current in starts.tolist():
                first = current - context_windows + 1
                last = current + horizon
                gaps = ranges[first + 1 : last + 1, 0] - ranges[first:last, 1] - 1
                valid.append(bool(np.all(gaps <= int(max_transition_gap))))
            starts = starts[np.asarray(valid, dtype=bool)]
        if not starts.size:
            continue
        contexts = np.stack(
            [
                observations[start - context_windows + 1 : start + 1].reshape(-1)
                for start in starts.tolist()
            ]
        )
        targets = observations[starts + horizon]
        x_rows.append(contexts)
        y_rows.append(targets)
        weights.append(np.full(starts.size, 1.0 / starts.size, dtype=np.float32))
        sample_rows.append(np.full(starts.size, sample, dtype=np.int64))
        problem_rows.append(
            np.full(starts.size, problem_ids[sample], dtype=problem_ids.dtype)
        )
        positions.append(starts)
    if not x_rows:
        target_dim = 0
        for value in window_observations:
            if np.asarray(value).ndim == 2 and np.asarray(value).shape[1] > 0:
                target_dim = int(np.asarray(value).shape[1])
                break
        return TransitionBundle(
            x=np.empty((0, context_windows * target_dim), dtype=np.float32),
            y=np.empty((0, target_dim), dtype=np.float32),
            weights=np.empty(0, dtype=np.float32),
            sample_indices=np.empty(0, dtype=np.int64),
            problem_ids=np.empty(0, dtype=problem_ids.dtype),
            transition_positions=np.empty(0, dtype=np.int64),
            horizon=int(horizon),
        )
    return TransitionBundle(
        x=np.concatenate(x_rows, axis=0),
        y=np.concatenate(y_rows, axis=0),
        weights=np.concatenate(weights),
        sample_indices=np.concatenate(sample_rows),
        problem_ids=np.concatenate(problem_rows),
        transition_positions=np.concatenate(positions),
        horizon=int(horizon),
    )
