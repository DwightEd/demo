from __future__ import annotations

import copy
import hashlib
import json
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

import numpy as np

from ..evaluation import (
    BinaryReport,
    GroupedBootstrapReport,
    LocalizationReport,
    PredictionRow,
)
from ..splitting import FixedHoldoutSplitter, SplitAssignment
from .contracts import CausalHypergraph, ConstraintGeometry
from .data import CausalTrace
from .hazard import FirstErrorSurvival
from .model import ConstraintTransportDetector, require_torch


@dataclass(frozen=True)
class TrainingConfig:
    hidden_dim: int = 128
    num_layers: int = 2
    epochs: int = 100
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 12
    gradient_clip: float = 1.0
    batch_size: int = 8
    seed: int = 17
    device: str = "cuda"
    bootstrap_replicates: int = 2000
    bootstrap_confidence: float = 0.95

    def validate(self) -> None:
        positive = (
            self.hidden_dim,
            self.num_layers,
            self.epochs,
            self.learning_rate,
            self.patience,
            self.gradient_clip,
            self.batch_size,
        )
        if (
            any(not math.isfinite(float(value)) for value in positive)
            or any(value <= 0 for value in positive)
            or not math.isfinite(float(self.weight_decay))
            or self.weight_decay < 0
            or self.bootstrap_replicates <= 0
            or not 0.0 < self.bootstrap_confidence < 1.0
        ):
            raise ValueError("training dimensions, rates, and limits are invalid")


@dataclass(frozen=True)
class FeatureNormalizer:
    mean: np.ndarray
    scale: np.ndarray
    content_dim: int

    @classmethod
    def fit(cls, traces: Sequence[CausalTrace]) -> "FeatureNormalizer":
        if not traces:
            raise ValueError("cannot fit feature normalization without training traces")
        features = np.concatenate(
            [trace.graph.node_features.astype(np.float64) for trace in traces], axis=0
        )
        geometry_dim = len(ConstraintGeometry.feature_names)
        content_dim = features.shape[1] - geometry_dim - 1
        if content_dim <= 0:
            raise ValueError("node features lack content or geometry columns")
        receiver = features[:, -1] > 0.5
        if not receiver.any():
            raise ValueError("training traces contain no step receiver nodes")
        mean = np.zeros(features.shape[1], dtype=np.float64)
        scale = np.ones(features.shape[1], dtype=np.float64)
        mean[:content_dim] = features[:, :content_dim].mean(axis=0)
        scale[:content_dim] = features[:, :content_dim].std(axis=0)
        geometry = features[receiver, content_dim:-1]
        mean[content_dim:-1] = geometry.mean(axis=0)
        scale[content_dim:-1] = geometry.std(axis=0)
        scale[scale < 1e-8] = 1.0
        return cls(mean=mean, scale=scale, content_dim=content_dim)

    def transform(self, trace: CausalTrace) -> CausalTrace:
        graph = trace.graph
        if graph.node_features.shape[1] != len(self.mean):
            raise ValueError("normalizer and node feature dimensions do not match")
        features = graph.node_features.astype(np.float64, copy=False)
        normalized = np.zeros_like(features)
        normalized[:, : self.content_dim] = (
            features[:, : self.content_dim] - self.mean[: self.content_dim]
        ) / self.scale[: self.content_dim]
        receiver = features[:, -1] > 0.5
        normalized[receiver, self.content_dim : -1] = (
            features[receiver, self.content_dim : -1] - self.mean[self.content_dim : -1]
        ) / self.scale[self.content_dim : -1]
        normalized[:, -1] = features[:, -1]
        return replace(
            trace,
            graph=CausalHypergraph(
                node_features=normalized.astype(np.float32),
                incidence=graph.incidence,
                receivers=graph.receivers,
                edge_features=graph.edge_features,
                edge_kind=graph.edge_kind,
                response_nodes=graph.response_nodes,
            ),
        )


@dataclass(frozen=True)
class TraceBatch:
    """A disjoint union of variable-size causal graphs for one GPU forward."""

    graph: CausalHypergraph
    traces: tuple[CausalTrace, ...]
    step_ranges: tuple[tuple[int, int], ...]

    @classmethod
    def from_traces(cls, traces: Sequence[CausalTrace]) -> "TraceBatch":
        if not traces:
            raise ValueError("cannot batch an empty trace sequence")
        node_offset = edge_offset = step_offset = 0
        nodes: list[np.ndarray] = []
        incidence: list[np.ndarray] = []
        receivers: list[np.ndarray] = []
        edge_features: list[np.ndarray] = []
        edge_kind: list[np.ndarray] = []
        response_nodes: list[np.ndarray] = []
        step_ranges: list[tuple[int, int]] = []
        for trace in traces:
            graph = trace.graph
            nodes.append(graph.node_features)
            shifted = graph.incidence.copy()
            shifted[0] += node_offset
            shifted[1] += edge_offset
            incidence.append(shifted)
            receivers.append(graph.receivers + node_offset)
            edge_features.append(graph.edge_features)
            edge_kind.append(graph.edge_kind)
            response_nodes.append(graph.response_nodes + node_offset)
            stop = step_offset + trace.labels.num_steps
            step_ranges.append((step_offset, stop))
            node_offset += graph.num_nodes
            edge_offset += graph.num_edges
            step_offset = stop
        return cls(
            graph=CausalHypergraph(
                node_features=np.concatenate(nodes, axis=0),
                incidence=np.concatenate(incidence, axis=1),
                receivers=np.concatenate(receivers),
                edge_features=np.concatenate(edge_features, axis=0),
                edge_kind=np.concatenate(edge_kind),
                response_nodes=np.concatenate(response_nodes),
            ),
            traces=tuple(traces),
            step_ranges=tuple(step_ranges),
        )


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_auroc: float | None


@dataclass(frozen=True)
class EvaluationResult:
    mean_loss: float
    response: BinaryReport
    localization: LocalizationReport
    uncertainty: GroupedBootstrapReport | None
    predictions: tuple[PredictionRow, ...]


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    history: tuple[EpochRecord, ...]
    validation: EvaluationResult
    test: EvaluationResult
    split: SplitAssignment
    normalizer: FeatureNormalizer
    model_state: dict


class CausalTransportTrainer:
    """Train one equal-weight survival loss per response, never per token."""

    def __init__(self, config: TrainingConfig) -> None:
        config.validate()
        self.config = config

    def fit(
        self,
        traces: Sequence[CausalTrace],
        splitter: FixedHoldoutSplitter,
    ) -> TrainingResult:
        require_torch()
        import torch

        if not traces:
            raise ValueError("training requires causal traces")
        self._validate_dimensions(traces)
        split = splitter.split([trace.split_record() for trace in traces])
        train = [traces[index] for index in split.train.indices]
        validation = [traces[index] for index in split.validation.indices]
        test = [traces[index] for index in split.test.indices]
        normalizer = FeatureNormalizer.fit(train)
        train, validation, test = (
            [normalizer.transform(trace) for trace in partition]
            for partition in (train, validation, test)
        )

        torch.manual_seed(self.config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.config.seed)
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA training requested but CUDA is unavailable")
        model = ConstraintTransportDetector(
            node_dim=train[0].graph.node_features.shape[1],
            edge_dim=train[0].graph.edge_features.shape[1],
            hidden_dim=self.config.hidden_dim,
            num_layers=self.config.num_layers,
        ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        history: list[EpochRecord] = []
        best_state = copy.deepcopy(model.state_dict())
        best_epoch, best_score, stale = 0, float("-inf"), 0
        rng = np.random.default_rng(self.config.seed)
        for epoch in range(1, self.config.epochs + 1):
            model.train()
            losses: list[float] = []
            order = rng.permutation(len(train))
            for start in range(0, len(order), self.config.batch_size):
                batch = TraceBatch.from_traces(
                    [
                        train[int(index)]
                        for index in order[start : start + self.config.batch_size]
                    ]
                )
                optimizer.zero_grad(set_to_none=True)
                logits = model(batch.graph)
                loss = torch.stack(
                    [
                        self._normalized_loss(logits[left:right], trace)
                        for trace, (left, right) in zip(batch.traces, batch.step_ranges)
                    ]
                ).mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), self.config.gradient_clip
                )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
            validation_result = self.evaluate(
                model, validation, batch_size=self.config.batch_size
            )
            selection_score = (
                validation_result.response.auroc
                if validation_result.response.auroc is not None
                else -validation_result.mean_loss
            )
            history.append(
                EpochRecord(
                    epoch=epoch,
                    train_loss=float(np.mean(losses)),
                    validation_loss=validation_result.mean_loss,
                    validation_auroc=validation_result.response.auroc,
                )
            )
            if selection_score > best_score:
                best_score, best_epoch, stale = selection_score, epoch, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                stale += 1
                if stale >= self.config.patience:
                    break

        model.load_state_dict(best_state)
        return TrainingResult(
            best_epoch=best_epoch,
            history=tuple(history),
            validation=self.evaluate(
                model,
                validation,
                bootstrap_replicates=self.config.bootstrap_replicates,
                bootstrap_confidence=self.config.bootstrap_confidence,
                bootstrap_seed=self.config.seed,
                batch_size=self.config.batch_size,
            ),
            test=self.evaluate(
                model,
                test,
                bootstrap_replicates=self.config.bootstrap_replicates,
                bootstrap_confidence=self.config.bootstrap_confidence,
                bootstrap_seed=self.config.seed + 1,
                batch_size=self.config.batch_size,
            ),
            split=split,
            normalizer=normalizer,
            model_state={
                key: value.detach().cpu() for key, value in best_state.items()
            },
        )

    @staticmethod
    def evaluate(
        model,
        traces: Sequence[CausalTrace],
        *,
        bootstrap_replicates: int = 0,
        bootstrap_confidence: float = 0.95,
        bootstrap_seed: int = 17,
        batch_size: int = 8,
    ) -> EvaluationResult:
        import torch

        model.eval()
        losses: list[float] = []
        first_errors: list[int] = []
        step_scores: list[np.ndarray] = []
        predictions: list[PredictionRow] = []
        if not traces or batch_size <= 0:
            raise ValueError("evaluation traces and batch_size must be non-empty")
        with torch.no_grad():
            for start in range(0, len(traces), batch_size):
                batch = TraceBatch.from_traces(traces[start : start + batch_size])
                batch_logits = model(batch.graph)
                for trace, (left, right) in zip(batch.traces, batch.step_ranges):
                    logits = batch_logits[left:right]
                    losses.append(
                        float(CausalTransportTrainer._normalized_loss(logits, trace))
                    )
                    probability = FirstErrorSurvival.response_error_probability(logits)
                    scores = torch.sigmoid(logits).cpu().numpy()
                    first_errors.append(trace.labels.first_error)
                    step_scores.append(scores)
                    predictions.append(
                        PredictionRow(
                            trace_id=trace.trace_id,
                            problem_id=trace.problem_id,
                            label=trace.response_label,
                            probability=probability,
                            first_error=trace.labels.first_error,
                            predicted_step=int(np.argmax(scores)),
                            step_probabilities=tuple(float(value) for value in scores),
                        )
                    )
        labels = np.asarray([row.label for row in predictions], dtype=np.int64)
        probabilities = np.asarray(
            [row.probability for row in predictions], dtype=np.float64
        )
        uncertainty = (
            GroupedBootstrapReport.from_predictions(
                predictions,
                replicates=bootstrap_replicates,
                confidence=bootstrap_confidence,
                seed=bootstrap_seed,
            )
            if bootstrap_replicates
            else None
        )
        return EvaluationResult(
            mean_loss=float(np.mean(losses)),
            response=BinaryReport.from_scores(labels=labels, scores=probabilities),
            localization=LocalizationReport.from_traces(first_errors, step_scores),
            uncertainty=uncertainty,
            predictions=tuple(predictions),
        )

    @staticmethod
    def _normalized_loss(logits, trace: CausalTrace):
        loss = FirstErrorSurvival.loss(logits, trace.labels)
        observed = (
            trace.labels.num_steps
            if trace.labels.first_error < 0
            else trace.labels.first_error + 1
        )
        return loss / observed

    @staticmethod
    def _validate_dimensions(traces: Sequence[CausalTrace]) -> None:
        node_dims = {trace.graph.node_features.shape[1] for trace in traces}
        edge_dims = {trace.graph.edge_features.shape[1] for trace in traces}
        if len(node_dims) != 1 or len(edge_dims) != 1:
            raise ValueError("all traces must share node and edge feature dimensions")


def _checkpoint_file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def trace_cohort_sha256(root: str | Path) -> str:
    files = sorted(Path(root).glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no causal traces in {root}")
    member_digests = sorted(
        bytes.fromhex(_checkpoint_file_sha256(path)) for path in files
    )
    digest = hashlib.sha256()
    for member in member_digests:
        digest.update(len(member).to_bytes(8, "big"))
        digest.update(member)
    return digest.hexdigest()


def trace_sidecar_sha256(root: str | Path) -> str | None:
    files = sorted(Path(root).glob("*.json"))
    if not files:
        return None
    digest = hashlib.sha256()
    for path in files:
        relative = path.name.encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(
            bytes.fromhex(_checkpoint_file_sha256(path))
        )
    return digest.hexdigest()


def load_training_checkpoint(
    output_dir: str | Path,
    *,
    device: str = "cpu",
):
    """Load a schema-checked CCT model and normalizer without pickle."""

    require_torch()
    from safetensors.torch import load_file

    root = Path(output_dir)
    checkpoint_path = root / "checkpoint.json"
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    if checkpoint.get("schema_version") != 1:
        raise ValueError("unsupported CCT checkpoint schema")
    if checkpoint.get("artifact_type") != "cct_hg_training_checkpoint":
        raise ValueError("unexpected CCT checkpoint artifact type")

    model_config = checkpoint.get("model")
    expected_model_keys = {
        "node_dim",
        "edge_dim",
        "hidden_dim",
        "num_layers",
    }
    if not isinstance(model_config, dict) or set(model_config) != expected_model_keys:
        raise ValueError("checkpoint model configuration is incomplete")
    model_values = {
        key: int(model_config[key])
        for key in expected_model_keys
    }
    if any(value <= 0 for value in model_values.values()):
        raise ValueError("checkpoint model dimensions must be positive")

    split_path = root / "split.json"
    if _checkpoint_file_sha256(split_path) != checkpoint.get("split_sha256"):
        raise ValueError("checkpoint split hash mismatch")
    weights_path = root / str(checkpoint.get("weights_file", ""))
    if (
        weights_path.name != checkpoint.get("weights_file")
        or _checkpoint_file_sha256(weights_path)
        != checkpoint.get("weights_sha256")
    ):
        raise ValueError("checkpoint weight hash mismatch")
    normalizer_path = root / str(checkpoint.get("normalizer_file", ""))
    if (
        normalizer_path.name != checkpoint.get("normalizer_file")
        or _checkpoint_file_sha256(normalizer_path)
        != checkpoint.get("normalizer_sha256")
    ):
        raise ValueError("checkpoint normalizer hash mismatch")

    with np.load(normalizer_path, allow_pickle=False) as archive:
        mean = np.asarray(archive["mean"], dtype=np.float64)
        scale = np.asarray(archive["scale"], dtype=np.float64)
        content_dim = int(archive["content_dim"])
    if (
        mean.ndim != 1
        or scale.shape != mean.shape
        or len(mean) != model_values["node_dim"]
        or not np.isfinite(mean).all()
        or not np.isfinite(scale).all()
        or np.any(scale <= 0.0)
        or not 0 < content_dim < len(mean)
    ):
        raise ValueError("checkpoint normalizer is invalid")
    normalizer_manifest = checkpoint.get("normalizer")
    if normalizer_manifest != {
        "feature_dim": len(mean),
        "content_dim": content_dim,
    }:
        raise ValueError("checkpoint normalizer manifest mismatch")

    model = ConstraintTransportDetector(
        node_dim=model_values["node_dim"],
        edge_dim=model_values["edge_dim"],
        hidden_dim=model_values["hidden_dim"],
        num_layers=model_values["num_layers"],
    ).to(device)
    state = load_file(str(weights_path), device=device)
    model.load_state_dict(state, strict=True)
    return model, FeatureNormalizer(
        mean=mean,
        scale=scale,
        content_dim=content_dim,
    )
