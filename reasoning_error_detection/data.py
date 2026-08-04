from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


OUTPUT_PATTERNS = (
    "entropy",
    "nll",
    "logprob",
    "log_prob",
    "margin",
    "topk",
    "top_",
    "mass",
    "rank",
)
ROUTING_PATTERNS = (
    "attention",
    "routing",
    "icr",
    "prompt_frac",
    "prefix_frac",
    "off_prompt",
)


@dataclass(frozen=True)
class FeatureConfig:
    layers: tuple[int, ...] = ()
    projection_dim: int = 16
    projection_batch_size: int = 256
    seed: int = 17
    device: str = "auto"

    def validate(self) -> None:
        if self.projection_dim < 1 or self.projection_batch_size < 1:
            raise ValueError("projection dimensions and batch size must be positive")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if len(set(self.layers)) != len(self.layers):
            raise ValueError("selected layers must be unique")


@dataclass
class StepFeatureDataset:
    source_path: str
    chain_idx: np.ndarray
    problem_groups: np.ndarray
    step_idx: np.ndarray
    gold_error_step: np.ndarray
    onset_label: np.ndarray
    onset_eligible: np.ndarray
    chain_error: np.ndarray
    control_features: np.ndarray
    output_features: np.ndarray
    routing_features: np.ndarray
    residual_features: np.ndarray
    control_names: tuple[str, ...]
    output_names: tuple[str, ...]
    routing_names: tuple[str, ...]
    residual_names: tuple[str, ...]
    selected_layers: tuple[int, ...]

    @property
    def n_rows(self) -> int:
        return int(len(self.chain_idx))

    def validate(self) -> None:
        arrays = (
            self.problem_groups,
            self.step_idx,
            self.gold_error_step,
            self.onset_label,
            self.onset_eligible,
            self.chain_error,
        )
        if any(len(values) != self.n_rows for values in arrays):
            raise ValueError("step metadata arrays have inconsistent row counts")
        feature_sets = (
            (self.control_features, self.control_names),
            (self.output_features, self.output_names),
            (self.routing_features, self.routing_names),
            (self.residual_features, self.residual_names),
        )
        for values, names in feature_sets:
            if values.shape != (self.n_rows, len(names)):
                raise ValueError("feature matrix does not match its row/name contract")
        if not np.isin(self.onset_label, (-1, 0, 1)).all():
            raise ValueError("onset labels must use -1=excluded, 0=valid, 1=first error")
        if not np.array_equal(self.onset_eligible, self.onset_label >= 0):
            raise ValueError("onset eligibility disagrees with onset labels")
        if not np.isfinite(self.control_features).all():
            raise ValueError("control features contain non-finite values")
        if not np.isfinite(self.residual_features).all():
            raise ValueError("residual features contain non-finite values")


@dataclass(frozen=True)
class _StateView:
    values: np.ndarray
    chain_idx: np.ndarray
    step_idx: np.ndarray
    layers: tuple[int, ...]

    def row_lookup(self) -> dict[tuple[int, int], int]:
        keys = [
            (int(chain), int(step))
            for chain, step in zip(self.chain_idx, self.step_idx)
        ]
        if len(set(keys)) != len(keys):
            raise ValueError("state view contains duplicate chain/step rows")
        return {key: index for index, key in enumerate(keys)}


def _scalar_path(value: np.ndarray, manifest: Path) -> Path:
    path = Path(str(np.asarray(value, dtype=object).item()))
    return path if path.is_absolute() else manifest.parent / path


def _load_state_view(
    z: np.lib.npyio.NpzFile,
    manifest: Path,
    view: str,
) -> _StateView:
    vector_key = f"{view}_vectors"
    memmap_key = f"{view}_memmap_path"
    if vector_key in z.files:
        values = np.asarray(z[vector_key])
    elif memmap_key in z.files:
        state_path = _scalar_path(z[memmap_key], manifest)
        if not state_path.exists():
            raise FileNotFoundError(f"{view} state store does not exist: {state_path}")
        values = np.load(state_path, mmap_mode="r")
    else:
        raise ValueError(
            f"ProcessBench features require {view}_vectors or {view}_memmap_path"
        )
    count_key = f"{view}_memmap_count"
    count = int(np.asarray(z[count_key]).item()) if count_key in z.files else len(values)
    if values.ndim != 3 or not 0 < count <= len(values):
        raise ValueError(f"{view} must have shape [step, layer, hidden]")
    chain_key = f"{view}_vector_chain_idx"
    step_key = f"{view}_vector_step_idx"
    if chain_key not in z.files or step_key not in z.files:
        raise ValueError(f"{view} is missing chain/step alignment arrays")
    chain_idx = np.asarray(z[chain_key], dtype=np.int64)[:count]
    step_idx = np.asarray(z[step_key], dtype=np.int64)[:count]
    if len(chain_idx) != count or len(step_idx) != count:
        raise ValueError(f"{view} alignment arrays do not match the state count")
    layer_key = f"{view}_vector_layers"
    if layer_key not in z.files:
        layer_key = "step_layer_state_vector_layers"
    if layer_key not in z.files:
        raise ValueError(f"{view} is missing its layer IDs")
    layers = tuple(int(value) for value in np.asarray(z[layer_key]).reshape(-1))
    if values.shape[1] != len(layers):
        raise ValueError(f"{view} layer IDs do not match the state tensor")
    return _StateView(values=values[:count], chain_idx=chain_idx, step_idx=step_idx, layers=layers)


def _step_ranges(value: np.ndarray, n_steps: int) -> np.ndarray:
    ranges = np.asarray(value)
    if ranges.dtype == object:
        ranges = np.asarray(ranges.tolist())
    ranges = np.asarray(ranges, dtype=np.int64)
    if ranges.ndim != 2 or ranges.shape[1] != 2 or len(ranges) < n_steps:
        raise ValueError("step_token_ranges must provide [start, stop] for every step")
    ranges = ranges[:n_steps]
    if np.any(ranges[:, 0] < 0) or np.any(ranges[:, 1] < ranges[:, 0]):
        raise ValueError("step_token_ranges contains an invalid inclusive range")
    return ranges


def _feature_indices(names: Sequence[str], patterns: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, name in enumerate(names)
            if any(pattern in str(name).lower() for pattern in patterns)
        ],
        dtype=np.int64,
    )


class ProcessBenchFeatureLoader:
    def __init__(self, config: FeatureConfig) -> None:
        config.validate()
        self.config = config

    def load(self, path: str | Path) -> StepFeatureDataset:
        manifest = Path(path)
        if not manifest.exists():
            raise FileNotFoundError(manifest)
        with np.load(manifest, allow_pickle=True) as z:
            required = ("gold_error_step", "n_steps", "step_token_ranges")
            missing = [name for name in required if name not in z.files]
            if missing:
                raise ValueError(f"ProcessBench feature manifest is missing {missing}")
            pre = _load_state_view(z, manifest, "step_pre_state")
            end = _load_state_view(z, manifest, "step_end_state")
            if pre.layers != end.layers:
                raise ValueError("pre/end state views use different layer axes")

            n_steps = np.asarray(z["n_steps"], dtype=np.int64)
            gold = np.asarray(z["gold_error_step"], dtype=np.int64)
            chain_ids = (
                np.asarray(z["chain_idx"], dtype=np.int64)
                if "chain_idx" in z.files
                else np.arange(len(n_steps), dtype=np.int64)
            )
            if not (len(n_steps) == len(gold) == len(chain_ids)):
                raise ValueError("chain labels and n_steps are misaligned")
            group_values = None
            for key in ("problem_group_ids", "problem_group_id", "problem_ids", "problem_id"):
                if key in z.files:
                    group_values = np.asarray(z[key], dtype=object)
                    break
            if group_values is None or len(group_values) != len(chain_ids):
                raise ValueError("ProcessBench features require one problem group per chain")
            _, encoded_groups = np.unique(group_values.astype(str), return_inverse=True)

            records: list[int] = []
            row_chain: list[int] = []
            row_group: list[int] = []
            row_step: list[int] = []
            row_gold: list[int] = []
            labels: list[int] = []
            controls: list[np.ndarray] = []
            for record, (chain, group, count, first_error) in enumerate(
                zip(chain_ids, encoded_groups, n_steps, gold)
            ):
                count = int(count)
                first_error = int(first_error)
                if count < 1 or first_error < -1 or first_error >= count:
                    raise ValueError(
                        f"chain {int(chain)} has invalid n_steps/gold_error_step"
                    )
                ranges = _step_ranges(z["step_token_ranges"][record], count)
                lengths = ranges[:, 1] - ranges[:, 0] + 1
                cumulative = np.cumsum(lengths)
                previous = np.concatenate([lengths[:1], lengths[:-1]])
                for step in range(count):
                    records.append(record)
                    row_chain.append(int(chain))
                    row_group.append(int(group))
                    row_step.append(step)
                    row_gold.append(first_error)
                    labels.append(
                        0
                        if first_error < 0 or step < first_error
                        else (1 if step == first_error else -1)
                    )
                    controls.append(
                        np.log1p(
                            np.asarray(
                                [step, lengths[step], previous[step], cumulative[step]],
                                dtype=np.float64,
                            )
                        )
                    )

            record_index = np.asarray(records, dtype=np.int64)
            step_index = np.asarray(row_step, dtype=np.int64)
            output, output_names, routing, routing_names = self._scalar_features(
                z, record_index, step_index
            )

        selected_layers = self._selected_layers(pre.layers)
        residual, residual_names = self._residual_features(
            pre,
            end,
            np.asarray(row_chain, dtype=np.int64),
            step_index,
            selected_layers,
        )
        onset_label = np.asarray(labels, dtype=np.int8)
        row_gold_array = np.asarray(row_gold, dtype=np.int64)
        dataset = StepFeatureDataset(
            source_path=str(manifest.resolve()),
            chain_idx=np.asarray(row_chain, dtype=np.int64),
            problem_groups=np.asarray(row_group, dtype=np.int64),
            step_idx=step_index,
            gold_error_step=row_gold_array,
            onset_label=onset_label,
            onset_eligible=onset_label >= 0,
            chain_error=(row_gold_array >= 0).astype(np.int8),
            control_features=np.asarray(controls, dtype=np.float32),
            output_features=output,
            routing_features=routing,
            residual_features=residual,
            control_names=(
                "control.log1p_step_index",
                "control.log1p_step_tokens",
                "control.log1p_previous_step_tokens",
                "control.log1p_cumulative_tokens",
            ),
            output_names=output_names,
            routing_names=routing_names,
            residual_names=residual_names,
            selected_layers=selected_layers,
        )
        dataset.validate()
        return dataset

    def _selected_layers(self, available: tuple[int, ...]) -> tuple[int, ...]:
        selected = self.config.layers or available
        missing = [layer for layer in selected if layer not in available]
        if missing:
            raise ValueError(f"selected layers are absent from ProcessBench states: {missing}")
        return tuple(int(layer) for layer in selected)

    def _scalar_features(
        self,
        z: np.lib.npyio.NpzFile,
        record_index: np.ndarray,
        step_index: np.ndarray,
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, tuple[str, ...]]:
        if "step_scores" not in z.files or "step_score_names" not in z.files:
            empty = np.empty((len(record_index), 0), dtype=np.float32)
            return empty, (), empty.copy(), ()
        scores = np.asarray(z["step_scores"], dtype=np.float32)
        names = tuple(str(name) for name in np.asarray(z["step_score_names"]).tolist())
        if scores.ndim != 3 or scores.shape[2] != len(names):
            raise ValueError("step_scores does not match step_score_names")
        output_index = _feature_indices(names, OUTPUT_PATTERNS)
        routing_index = _feature_indices(names, ROUTING_PATTERNS)
        rows = scores[record_index, step_index]
        output = rows[:, output_index] if len(output_index) else np.empty((len(rows), 0))
        routing = rows[:, routing_index] if len(routing_index) else np.empty((len(rows), 0))
        return (
            np.asarray(output, dtype=np.float32),
            tuple(f"output.{names[index]}" for index in output_index),
            np.asarray(routing, dtype=np.float32),
            tuple(f"routing.{names[index]}" for index in routing_index),
        )

    def _residual_features(
        self,
        pre: _StateView,
        end: _StateView,
        chain_idx: np.ndarray,
        step_idx: np.ndarray,
        selected_layers: tuple[int, ...],
    ) -> tuple[np.ndarray, tuple[str, ...]]:
        pre_lookup = pre.row_lookup()
        end_lookup = end.row_lookup()
        keys = [(int(chain), int(step)) for chain, step in zip(chain_idx, step_idx)]
        missing = [key for key in keys if key not in pre_lookup or key not in end_lookup]
        if missing:
            raise ValueError(f"pre/end state views miss chain-step rows: {missing[:8]}")
        pre_rows = np.asarray([pre_lookup[key] for key in keys], dtype=np.int64)
        end_rows = np.asarray([end_lookup[key] for key in keys], dtype=np.int64)
        layer_positions = [pre.layers.index(layer) for layer in selected_layers]
        width = self.config.projection_dim + 4
        output = np.empty((len(keys), len(selected_layers) * width), dtype=np.float32)
        names: list[str] = []
        matrices: list[np.ndarray] = []
        hidden = int(pre.values.shape[2])
        for layer in selected_layers:
            rng = np.random.default_rng(self.config.seed + 1009 * int(layer))
            matrices.append(
                (rng.normal(size=(hidden, self.config.projection_dim)) / np.sqrt(self.config.projection_dim)).astype(
                    np.float32
                )
            )
            names.extend(
                [
                    *(f"residual.layer{layer}.direction_rp{j}" for j in range(self.config.projection_dim)),
                    f"residual.layer{layer}.log_delta_norm",
                    f"residual.layer{layer}.log_pre_norm",
                    f"residual.layer{layer}.log_end_norm",
                    f"residual.layer{layer}.pre_end_cosine",
                ]
            )
        device = self._projection_device()
        for start in range(0, len(keys), self.config.projection_batch_size):
            stop = min(start + self.config.projection_batch_size, len(keys))
            before = np.asarray(
                pre.values[pre_rows[start:stop]][:, layer_positions, :], dtype=np.float32
            )
            after = np.asarray(
                end.values[end_rows[start:stop]][:, layer_positions, :], dtype=np.float32
            )
            output[start:stop] = self._project_batch(before, after, matrices, device)
        return output, tuple(names)

    def _projection_device(self) -> str:
        if self.config.device == "cpu":
            return "cpu"
        import torch

        if self.config.device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if self.config.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA projection requested but CUDA is unavailable")
        return self.config.device

    @staticmethod
    def _project_batch(
        before: np.ndarray,
        after: np.ndarray,
        matrices: Sequence[np.ndarray],
        device: str,
    ) -> np.ndarray:
        if device == "cpu":
            blocks: list[np.ndarray] = []
            for position, matrix in enumerate(matrices):
                pre = before[:, position]
                end = after[:, position]
                delta = end - pre
                delta_norm = np.linalg.norm(delta, axis=1)
                pre_norm = np.linalg.norm(pre, axis=1)
                end_norm = np.linalg.norm(end, axis=1)
                unit = delta / np.maximum(delta_norm[:, None], 1e-8)
                cosine = np.sum(pre * end, axis=1) / np.maximum(pre_norm * end_norm, 1e-8)
                blocks.append(
                    np.column_stack(
                        [
                            unit @ matrix,
                            np.log1p(delta_norm),
                            np.log1p(pre_norm),
                            np.log1p(end_norm),
                            cosine,
                        ]
                    )
                )
            return np.concatenate(blocks, axis=1).astype(np.float32)

        import torch

        pre_tensor = torch.as_tensor(before, device=device, dtype=torch.float32)
        end_tensor = torch.as_tensor(after, device=device, dtype=torch.float32)
        blocks = []
        for position, matrix in enumerate(matrices):
            pre = pre_tensor[:, position]
            end = end_tensor[:, position]
            delta = end - pre
            delta_norm = torch.linalg.vector_norm(delta, dim=1)
            pre_norm = torch.linalg.vector_norm(pre, dim=1)
            end_norm = torch.linalg.vector_norm(end, dim=1)
            unit = delta / delta_norm.clamp_min(1e-8).unsqueeze(1)
            projection = unit @ torch.as_tensor(matrix, device=device)
            cosine = torch.sum(pre * end, dim=1) / (pre_norm * end_norm).clamp_min(1e-8)
            blocks.append(
                torch.column_stack(
                    (
                        projection,
                        torch.log1p(delta_norm),
                        torch.log1p(pre_norm),
                        torch.log1p(end_norm),
                        cosine,
                    )
                )
            )
        return torch.cat(blocks, dim=1).cpu().numpy().astype(np.float32)
