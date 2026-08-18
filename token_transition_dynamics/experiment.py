from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score, roc_auc_score

from .data import ChainRecord, RawDomain, RawHiddenRepository
from .dynamics import MixtureLinearDynamics, transition_coordinates


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    domains: tuple[str, ...]
    output_dir: Path
    manifest_name: str = "trace.raw_residual_stream.npz"
    pca_dim: int = 16
    clusters: tuple[int, ...] = (1, 2)
    tokens_per_chain: int = 8
    max_test_tokens_per_chain: int = 64
    max_pca_rows: int = 4096
    bootstrap_samples: int = 2000
    seed: int = 17
    max_records_per_domain: int | None = None


@dataclass(frozen=True)
class _RecordRef:
    source: RawDomain
    record: ChainRecord
    split: str


@dataclass(frozen=True)
class _ScoredTokens:
    scores: np.ndarray
    layer_scores: np.ndarray
    labels: np.ndarray
    chain_ids: np.ndarray
    problem_groups: np.ndarray
    error_chains: np.ndarray


def _split_name(domain: str, problem_group: str, seed: int) -> str:
    value = f"{seed}::{domain}::{problem_group}".encode("utf-8")
    bucket = int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "calibration"
    return "test"


def _even_indices(size: int, limit: int) -> np.ndarray:
    if size <= 0 or limit <= 0:
        return np.empty(0, dtype=np.int64)
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, limit, dtype=np.int64))


def _safe_metric(labels: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def _bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    groups: np.ndarray,
    metric: str,
    samples: int,
    seed: int,
) -> list[float] | None:
    unique = np.unique(groups)
    if samples <= 0 or unique.size < 2:
        return None
    rows = {group: np.flatnonzero(groups == group) for group in unique}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        index = np.concatenate([rows[group] for group in chosen])
        value = _safe_metric(labels[index], scores[index], metric)
        if value is not None:
            values.append(value)
    if not values:
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


class TokenTransitionExperiment:
    """Fit correct-only token transition fields and score first-error token intervals."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def run(self) -> dict[str, Any]:
        self._validate_config()
        repository = RawHiddenRepository(
            self.config.data_root,
            manifest_name=self.config.manifest_name,
            max_records_per_domain=self.config.max_records_per_domain,
        )
        domains = [repository.read_domain(name) for name in self.config.domains]
        self._validate_layers(domains)
        records = self._split_records(domains)
        self._validate_splits(records)

        train_correct = [
            value for value in records if value.split == "train" and value.record.error_step < 0
        ]
        calibration_correct = [
            value
            for value in records
            if value.split == "calibration" and value.record.error_step < 0
        ]
        test = [value for value in records if value.split == "test"]
        projector = self._fit_projector(train_correct)

        models: dict[str, Any] = {}
        fitted: dict[tuple[str, int], list[MixtureLinearDynamics]] = {}
        train_rows = 0
        for geometry in ("euclidean", "spherical"):
            batches = self._transition_batches(train_correct, projector, geometry)
            train_rows = sum(batch[0].shape[0] for batch in batches)
            baseline_models = [
                MixtureLinearDynamics(1, seed=self.config.seed + layer_index).fit(
                    np.empty((state.shape[0], 0)), target, position
                )
                for layer_index, (state, target, position) in enumerate(batches)
            ]
            calibration_baseline = self._score(
                calibration_correct,
                projector,
                geometry,
                baseline_models,
                test_mode=False,
                position_only=True,
            )
            for clusters in self.config.clusters:
                layer_models = []
                for layer_index, (state, target, position) in enumerate(batches):
                    model = MixtureLinearDynamics(
                        clusters,
                        seed=self.config.seed + layer_index,
                    ).fit(state, target, position)
                    layer_models.append(model)
                fitted[(geometry, clusters)] = layer_models

                calibration = self._score(
                    calibration_correct, projector, geometry, layer_models, test_mode=False
                )
                evaluated = self._score(test, projector, geometry, layer_models, test_mode=True)
                name = f"{geometry}_k{clusters}"
                models[name] = self._model_report(
                    calibration, evaluated, calibration_baseline, layer_models
                )

        split_counts = {
            f"{split}_{kind}": sum(
                value.split == split
                and ((value.record.error_step >= 0) if kind == "error" else (value.record.error_step < 0))
                for value in records
            )
            for split in ("train", "calibration", "test")
            for kind in ("correct", "error")
        }
        report = {
            "protocol": {
                "unit": "individual response token across every adjacent stored layer pair",
                "uses_step_pooling": False,
                "uses_step_scores": False,
                "uses_error_labels_for_model_fit": False,
                "fit_cohort": "train-split fully-correct chains only",
                "target_label_semantics": "interval_censored_first_error_step_tokens",
                "post_error_tokens": "excluded",
                "split_unit": "domain plus problem_group",
                "geometry_arms": ["euclidean", "spherical_log_map"],
            },
            "data": {
                "domains": {
                    domain.name: {
                        "records": len(domain.records),
                        "layers": domain.layers.tolist(),
                        "manifest": str(domain.manifest_path),
                        "source_format": domain.source_format,
                    }
                    for domain in domains
                },
                "split_counts": split_counts,
                "fit_counts": {
                    "correct_chains": len(train_correct),
                    "error_chains": 0,
                    "transition_rows_per_geometry": train_rows,
                    "pca_rows": int(projector.n_samples_),
                    "pca_dim": int(projector.n_components_),
                },
            },
            "models": models,
            "comparisons": self._comparisons(models),
        }
        report["verdict"] = self._verdict(report["comparisons"])
        self._save(report)
        return report

    def inspect(self) -> dict[str, Any]:
        """Read manifests and one shard per domain without fitting a model."""
        self._validate_config()
        repository = RawHiddenRepository(
            self.config.data_root,
            manifest_name=self.config.manifest_name,
            max_records_per_domain=self.config.max_records_per_domain,
        )
        domains = [repository.read_domain(name) for name in self.config.domains]
        self._validate_layers(domains)
        return {
            domain.name: {
                "manifest": str(domain.manifest_path),
                "records": len(domain.records),
                "layers": domain.layers.tolist(),
                "first_shard": str(domain.records[0].state_path),
                "first_shard_shape": list(domain.load_states(domain.records[0]).shape),
            }
            for domain in domains
        }

    def _validate_config(self) -> None:
        if not self.config.domains:
            raise ValueError("at least one domain is required")
        if self.config.pca_dim < 1:
            raise ValueError("pca_dim must be positive")
        if not self.config.clusters or any(value < 1 for value in self.config.clusters):
            raise ValueError("clusters must contain positive integers")
        if self.config.tokens_per_chain < 1 or self.config.max_test_tokens_per_chain < 1:
            raise ValueError("token sampling limits must be positive")

    @staticmethod
    def _validate_layers(domains: list[RawDomain]) -> None:
        reference = domains[0].layers
        for domain in domains[1:]:
            if not np.array_equal(reference, domain.layers):
                raise ValueError(
                    "all domains must store identical consecutive layers; "
                    f"{domains[0].name}={reference.tolist()}, "
                    f"{domain.name}={domain.layers.tolist()}"
                )

    def _split_records(self, domains: list[RawDomain]) -> list[_RecordRef]:
        return [
            _RecordRef(
                domain,
                record,
                _split_name(domain.name, record.problem_group, self.config.seed),
            )
            for domain in domains
            for record in domain.records
        ]

    @staticmethod
    def _validate_splits(records: list[_RecordRef]) -> None:
        train_correct = sum(
            value.split == "train" and value.record.error_step < 0 for value in records
        )
        calibration_correct = sum(
            value.split == "calibration" and value.record.error_step < 0 for value in records
        )
        test_correct = sum(
            value.split == "test" and value.record.error_step < 0 for value in records
        )
        test_error = sum(
            value.split == "test" and value.record.error_step >= 0 for value in records
        )
        if min(train_correct, calibration_correct, test_correct, test_error) == 0:
            raise ValueError(
                "problem-group split needs train/calibration correct chains and test correct/error "
                f"chains; got train_correct={train_correct}, "
                f"calibration_correct={calibration_correct}, test_correct={test_correct}, "
                f"test_error={test_error}"
            )

    def _fit_projector(self, records: list[_RecordRef]) -> PCA:
        rows_per_chain = max(1, self.config.max_pca_rows // len(records))
        chunks = []
        for value in records:
            states = value.source.load_states(value.record)
            flattened = np.asarray(states).reshape(-1, states.shape[-1])
            chunks.append(flattened[_even_indices(flattened.shape[0], rows_per_chain)])
        matrix = np.concatenate(chunks, axis=0)
        if matrix.shape[0] > self.config.max_pca_rows:
            matrix = matrix[_even_indices(matrix.shape[0], self.config.max_pca_rows)]
        maximum = min(matrix.shape)
        if self.config.pca_dim > maximum:
            raise ValueError(
                f"pca_dim={self.config.pca_dim} exceeds available rank bound {maximum}"
            )
        return PCA(
            n_components=self.config.pca_dim,
            svd_solver="randomized",
            random_state=self.config.seed,
        ).fit(matrix)

    def _sample_tokens(self, value: _RecordRef, state_count: int, test_mode: bool) -> np.ndarray:
        if not test_mode:
            return _even_indices(state_count, self.config.tokens_per_chain)
        record = value.record
        if record.error_step < 0:
            return _even_indices(state_count, self.config.max_test_tokens_per_chain)

        event = record.step_ranges[record.error_step] - record.response_start
        event_start, event_end = int(event[0]), int(event[1])
        if event_start < 0 or event_end >= state_count:
            raise ValueError(
                f"{record.state_path}: first-error interval {event.tolist()} lies outside "
                f"response-state tokens [0,{state_count - 1}]"
            )
        prefix_size = event_end + 1
        cap = self.config.max_test_tokens_per_chain
        if prefix_size <= cap:
            return np.arange(prefix_size, dtype=np.int64)
        event_indices = np.arange(event_start, event_end + 1, dtype=np.int64)
        prior_indices = np.arange(event_start, dtype=np.int64)
        event_budget = min(event_indices.size, max(1, cap // 2))
        prior_budget = min(prior_indices.size, cap - event_budget)
        remaining = cap - event_budget - prior_budget
        event_budget = min(event_indices.size, event_budget + remaining)
        return np.sort(
            np.concatenate(
                [
                    event_indices[_even_indices(event_indices.size, event_budget)],
                    prior_indices[_even_indices(prior_indices.size, prior_budget)],
                ]
            )
        )

    @staticmethod
    def _project(projector: PCA, states: np.ndarray, indices: np.ndarray) -> np.ndarray:
        selected = np.asarray(states[indices], dtype=np.float64)
        shape = selected.shape
        return projector.transform(selected.reshape(-1, shape[-1])).reshape(
            shape[0], shape[1], -1
        )

    def _transition_batches(
        self, records: list[_RecordRef], projector: PCA, geometry: str
    ) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        layer_count = records[0].source.layers.size - 1
        states_by_layer: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
        targets_by_layer: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
        positions_by_layer: list[list[np.ndarray]] = [[] for _ in range(layer_count)]
        for value in records:
            raw = value.source.load_states(value.record)
            index = self._sample_tokens(value, raw.shape[0], test_mode=False)
            projected = self._project(projector, raw, index)
            state, target = transition_coordinates(projected, geometry)
            position = index.astype(np.float64) / max(raw.shape[0] - 1, 1)
            for layer in range(layer_count):
                states_by_layer[layer].append(state[:, layer])
                targets_by_layer[layer].append(target[:, layer])
                positions_by_layer[layer].append(position)
        return [
            (
                np.concatenate(states_by_layer[layer]),
                np.concatenate(targets_by_layer[layer]),
                np.concatenate(positions_by_layer[layer]),
            )
            for layer in range(layer_count)
        ]

    def _score(
        self,
        records: list[_RecordRef],
        projector: PCA,
        geometry: str,
        models: list[MixtureLinearDynamics],
        *,
        test_mode: bool,
        position_only: bool = False,
    ) -> _ScoredTokens:
        scores: list[np.ndarray] = []
        scores_by_layer: list[np.ndarray] = []
        labels: list[np.ndarray] = []
        chain_ids: list[np.ndarray] = []
        problem_groups: list[np.ndarray] = []
        error_chains: list[np.ndarray] = []
        for value in records:
            raw = value.source.load_states(value.record)
            index = self._sample_tokens(value, raw.shape[0], test_mode=test_mode)
            projected = self._project(projector, raw, index)
            state, target = transition_coordinates(projected, geometry)
            position = index.astype(np.float64) / max(raw.shape[0] - 1, 1)
            layer_scores = np.column_stack(
                [
                    model.nll(
                        (
                            np.empty((index.size, 0))
                            if position_only
                            else state[:, layer]
                        ),
                        target[:, layer],
                        position,
                    )
                    for layer, model in enumerate(models)
                ]
            )
            token_labels = np.zeros(index.size, dtype=np.int64)
            if value.record.error_step >= 0:
                event = (
                    value.record.step_ranges[value.record.error_step]
                    - value.record.response_start
                )
                token_labels = ((index >= event[0]) & (index <= event[1])).astype(np.int64)
            identity = f"{value.source.name}::{value.record.row}"
            group = f"{value.source.name}::{value.record.problem_group}"
            scores.append(np.mean(layer_scores, axis=1))
            scores_by_layer.append(layer_scores)
            labels.append(token_labels)
            chain_ids.append(np.full(index.size, identity, dtype=object))
            problem_groups.append(np.full(index.size, group, dtype=object))
            error_chains.append(
                np.full(index.size, value.record.error_step >= 0, dtype=bool)
            )
        return _ScoredTokens(
            scores=np.concatenate(scores),
            layer_scores=np.concatenate(scores_by_layer),
            labels=np.concatenate(labels),
            chain_ids=np.concatenate(chain_ids),
            problem_groups=np.concatenate(problem_groups),
            error_chains=np.concatenate(error_chains),
        )

    def _model_report(
        self,
        calibration: _ScoredTokens,
        evaluated: _ScoredTokens,
        calibration_baseline: _ScoredTokens,
        layer_models: list[MixtureLinearDynamics],
    ) -> dict[str, Any]:
        auroc = _safe_metric(evaluated.labels, evaluated.scores, "auroc")
        auprc = _safe_metric(evaluated.labels, evaluated.scores, "auprc")
        within = []
        for chain in np.unique(evaluated.chain_ids[evaluated.error_chains]):
            mask = evaluated.chain_ids == chain
            value = _safe_metric(evaluated.labels[mask], evaluated.scores[mask], "auroc")
            if value is not None:
                within.append(value)
        negative = evaluated.labels == 0
        calibration_nll = float(np.mean(calibration.scores))
        baseline_nll = float(np.mean(calibration_baseline.scores))
        layer_metrics = [
            {
                "transition_index": layer,
                "auroc": _safe_metric(
                    evaluated.labels, evaluated.layer_scores[:, layer], "auroc"
                ),
                "auprc": _safe_metric(
                    evaluated.labels, evaluated.layer_scores[:, layer], "auprc"
                ),
            }
            for layer in range(evaluated.layer_scores.shape[1])
        ]
        return {
            "calibration_correct_nll": calibration_nll,
            "calibration_position_only_nll": baseline_nll,
            "calibration_nll_gain_over_position_only": baseline_nll - calibration_nll,
            "test_negative_nll": float(np.mean(evaluated.scores[negative])),
            "test_token": {
                "auroc": auroc,
                "auroc_ci": _bootstrap_interval(
                    evaluated.labels,
                    evaluated.scores,
                    evaluated.problem_groups,
                    "auroc",
                    self.config.bootstrap_samples,
                    self.config.seed,
                ),
                "auprc": auprc,
                "auprc_ci": _bootstrap_interval(
                    evaluated.labels,
                    evaluated.scores,
                    evaluated.problem_groups,
                    "auprc",
                    self.config.bootstrap_samples,
                    self.config.seed + 1,
                ),
                "within_error_chain_macro_auroc": (
                    None if not within else float(np.mean(within))
                ),
                "rows": int(evaluated.labels.size),
                "positive_rows": int(np.sum(evaluated.labels)),
                "problem_groups": int(np.unique(evaluated.problem_groups).size),
            },
            "test_token_by_layer_transition": layer_metrics,
            "cluster_weights_by_layer_transition": [
                model.weights_.tolist() for model in layer_models
            ],
        }

    def _comparisons(self, models: dict[str, Any]) -> dict[str, Any]:
        geometry = {}
        for clusters in self.config.clusters:
            euclidean = models[f"euclidean_k{clusters}"]
            spherical = models[f"spherical_k{clusters}"]
            geometry[f"k{clusters}"] = {
                "calibration_gain_delta_spherical": spherical[
                    "calibration_nll_gain_over_position_only"
                ]
                - euclidean["calibration_nll_gain_over_position_only"],
                "test_auroc_delta_spherical": spherical["test_token"]["auroc"]
                - euclidean["test_token"]["auroc"],
            }
        clusters = {}
        if 1 in self.config.clusters and 2 in self.config.clusters:
            for geometry_name in ("euclidean", "spherical"):
                one = models[f"{geometry_name}_k1"]
                two = models[f"{geometry_name}_k2"]
                clusters[geometry_name] = {
                    "calibration_nll_reduction_k2": one["calibration_correct_nll"]
                    - two["calibration_correct_nll"],
                    "test_auroc_delta_k2": two["test_token"]["auroc"]
                    - one["test_token"]["auroc"],
                }
        return {"geometry": geometry, "clusters": clusters}

    @staticmethod
    def _verdict(comparisons: dict[str, Any]) -> dict[str, Any]:
        manifold = {
            name: values["calibration_gain_delta_spherical"] > 0
            and values["test_auroc_delta_spherical"] > 0
            for name, values in comparisons["geometry"].items()
        }
        clusters = {
            name: values["calibration_nll_reduction_k2"] > 0
            and values["test_auroc_delta_k2"] > 0
            for name, values in comparisons["clusters"].items()
        }
        return {
            "spherical_geometry_point_gate_passed": any(manifold.values()),
            "latent_clusters_point_gate_passed": any(clusters.values()),
            "claim_supported": False,
            "rule": (
                "the point gate requires improvement on correct-only calibration gain and "
                "held-out first-error token AUROC; paired delta uncertainty and replication "
                "are still required, and any result is predictive rather than LLM-causal"
            ),
        }

    def _save(self, report: dict[str, Any]) -> None:
        output = Path(self.config.output_dir).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        json_temporary = output / "results.json.tmp"
        json_temporary.write_text(
            json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        json_temporary.replace(output / "results.json")

        lines = [
            "token_transition_dynamics",
            "unit: response token x adjacent hidden-layer transition",
            "fit: train-split fully-correct chains only; no step pooling or step_scores",
            "",
        ]
        def displayed(value: float | None) -> str:
            return "NA" if value is None else f"{value:.4f}"

        for name, values in report["models"].items():
            token = values["test_token"]
            lines.append(
                f"{name}: calibration_nll={values['calibration_correct_nll']:.6f} | "
                f"gain_over_position_only={values['calibration_nll_gain_over_position_only']:+.6f} | "
                f"token_AUROC={displayed(token['auroc'])} | "
                f"token_AUPRC={displayed(token['auprc'])} | "
                f"within_error_chain_AUROC={displayed(token['within_error_chain_macro_auroc'])}"
            )
        lines.extend(
            [
                "",
                f"spherical_geometry_point_gate_passed: "
                f"{report['verdict']['spherical_geometry_point_gate_passed']}",
                f"latent_clusters_point_gate_passed: "
                f"{report['verdict']['latent_clusters_point_gate_passed']}",
                f"claim_supported: {report['verdict']['claim_supported']}",
                f"decision_rule: {report['verdict']['rule']}",
            ]
        )
        text_temporary = output / "summary.txt.tmp"
        text_temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text_temporary.replace(output / "summary.txt")
