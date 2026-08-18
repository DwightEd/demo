from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from .baseline import CorrectOnlyBaseline
from .data import ChainRecord, RawDomain, RawHiddenRepository
from .window_geometry import FEATURE_NAMES, window_features


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: Path
    domains: tuple[str, ...]
    output_dir: Path
    manifest_name: str = "trace.raw_residual_stream.npz"
    window_size: int = 24
    neighbors: int = 20
    tle_centers: int = 6
    train_windows_per_chain: int = 4
    calibration_windows_per_chain: int = 4
    max_test_windows_per_chain: int = 12
    position_bins: int = 4
    min_baseline_samples: int = 8
    bootstrap_samples: int = 1000
    seed: int = 17
    max_records_per_domain: int | None = None


@dataclass(frozen=True)
class _RecordRef:
    source: RawDomain
    record: ChainRecord
    split: str


@dataclass(frozen=True)
class _WindowBatch:
    features: np.ndarray
    labels: np.ndarray
    domains: np.ndarray
    position_bins: np.ndarray
    chain_ids: np.ndarray
    problem_groups: np.ndarray
    error_chains: np.ndarray
    records_considered: int
    records_with_windows: int
    early_error_records_skipped: int


def _split_name(domain: str, problem_group: str, seed: int) -> str:
    value = f"{seed}::{domain}::{problem_group}".encode()
    bucket = int.from_bytes(hashlib.sha256(value).digest()[:8], "big") % 100
    if bucket < 60:
        return "train"
    if bucket < 80:
        return "calibration"
    return "test"


def _even_indices(size: int, limit: int) -> np.ndarray:
    if size <= limit:
        return np.arange(size, dtype=np.int64)
    return np.unique(np.linspace(0, size - 1, limit, dtype=np.int64))


def _safe_metric(labels: np.ndarray, scores: np.ndarray, metric: str) -> float | None:
    if labels.size == 0 or np.unique(labels).size < 2:
        return None
    if metric == "auroc":
        return float(roc_auc_score(labels, scores))
    return float(average_precision_score(labels, scores))


def _domain_macro_metric(
    labels: np.ndarray,
    scores: np.ndarray,
    domains: np.ndarray,
    metric: str,
) -> float | None:
    values = [
        _safe_metric(labels[domains == domain], scores[domains == domain], metric)
        for domain in np.unique(domains)
    ]
    valid = [value for value in values if value is not None]
    return None if not valid else float(np.mean(valid))


def _bootstrap_interval(
    labels: np.ndarray,
    scores: np.ndarray,
    domains: np.ndarray,
    groups: np.ndarray,
    metric: str,
    samples: int,
    seed: int,
) -> list[float] | None:
    if samples <= 0 or np.unique(groups).size < 2:
        return None
    domain_groups = {
        domain: np.unique(groups[domains == domain]) for domain in np.unique(domains)
    }
    rows = {group: np.flatnonzero(groups == group) for group in np.unique(groups)}
    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(samples):
        chosen = np.concatenate(
            [
                rng.choice(group_values, size=group_values.size, replace=True)
                for group_values in domain_groups.values()
            ]
        )
        index = np.concatenate([rows[group] for group in chosen])
        value = _domain_macro_metric(
            labels[index], scores[index], domains[index], metric
        )
        if value is not None:
            values.append(value)
    if not values:
        return None
    low, high = np.quantile(values, [0.025, 0.975])
    return [float(low), float(high)]


class TokenTransitionExperiment:
    """Score raw-space token-window geometry against a correct-only baseline."""

    _ARMS: ClassVar[dict[str, tuple[int, ...]]] = {
        "geometry": (0, 1),
        "dynamics": (2, 3),
        "combined": (0, 1, 2, 3),
    }

    def __init__(
        self,
        config: ExperimentConfig,
        progress: Callable[[str], None] | None = None,
    ) -> None:
        self.config = config
        self.progress = progress

    def run(self) -> dict[str, Any]:
        self._validate_config()
        self._progress("stage=load status=started")
        domains = self._read_domains()
        records = self._split_records(domains)
        self._validate_splits(records)
        self._progress(
            f"stage=load status=complete domains={len(domains)} records={len(records)}"
        )

        train_correct = [
            value for value in records if value.split == "train" and value.record.error_step < 0
        ]
        calibration_correct = [
            value
            for value in records
            if value.split == "calibration" and value.record.error_step < 0
        ]
        test_records = [value for value in records if value.split == "test"]

        train = self._extract_windows(
            train_correct,
            self.config.train_windows_per_chain,
            test_mode=False,
            stage="train",
        )
        calibration = self._extract_windows(
            calibration_correct,
            self.config.calibration_windows_per_chain,
            test_mode=False,
            stage="calibration",
        )
        evaluated = self._extract_windows(
            test_records,
            self.config.max_test_windows_per_chain,
            test_mode=True,
            stage="test",
        )
        self._progress("stage=baseline status=started")
        baseline = CorrectOnlyBaseline(
            position_bins=self.config.position_bins,
            min_bin_samples=self.config.min_baseline_samples,
        ).fit(train.features, train.domains, train.position_bins)
        calibration_z = baseline.standardize(
            calibration.features, calibration.domains, calibration.position_bins
        )
        evaluated_z = baseline.standardize(
            evaluated.features, evaluated.domains, evaluated.position_bins
        )
        self._progress("stage=baseline status=complete")

        self._progress("stage=evaluate status=started")
        models = {
            name: self._arm_report(
                calibration_z,
                evaluated_z,
                evaluated,
                feature_indices,
                domains[0].layers,
            )
            for name, feature_indices in self._ARMS.items()
        }
        self._progress("stage=evaluate status=complete")
        split_counts = {
            f"{split}_{kind}": sum(
                value.split == split
                and (
                    value.record.error_step >= 0
                    if kind == "error"
                    else value.record.error_step < 0
                )
                for value in records
            )
            for split in ("train", "calibration", "test")
            for kind in ("correct", "error")
        }
        report = {
            "protocol": {
                "analysis_axis": "token_time_within_each_stored_layer",
                "depth_semantics": "stored_layers_are_independent_sparse_observations",
                "window_unit": "fixed_length_trailing_response_token_window",
                "uses_raw_hidden_coordinates": True,
                "uses_pca": False,
                "uses_kmeans": False,
                "uses_step_pooling": False,
                "uses_step_scores": False,
                "uses_chain_correctness_to_select_fit_cohort": True,
                "uses_first_error_locations_for_model_fit": False,
                "fit_cohort": "train-split fully-correct chains only",
                "target_label_semantics": "window_endpoint_inside_first_error_step",
                "post_error_tokens": "excluded",
                "test_sampling": (
                    "error chains retain sampled pre-error endpoints and sampled first-error "
                    "endpoints; reported AUPRC therefore describes the sampled evaluation set"
                ),
                "split_unit": "domain plus problem_group",
                "position_control": "domain-layer-relative-position-bin robust norm",
                "geometry_features_are_order_invariant": list(FEATURE_NAMES[:2]),
                "dynamics_features_are_order_sensitive": list(FEATURE_NAMES[2:]),
            },
            "parameters": {
                "window_size": self.config.window_size,
                "neighbors": self.config.neighbors,
                "tle_centers": self.config.tle_centers,
                "position_bins": self.config.position_bins,
                "feature_names": list(FEATURE_NAMES),
            },
            "data": {
                "domains": {
                    domain.name: {
                        "records": len(domain.records),
                        "layers": domain.layers.tolist(),
                        "manifest": str(domain.manifest_path),
                        "source_format": domain.source_format,
                        "first_shard_shape": list(
                            domain.load_states(domain.records[0]).shape
                        ),
                    }
                    for domain in domains
                },
                "split_counts": split_counts,
                "fit_counts": {
                    "correct_chains": len(train_correct),
                    "error_chains": 0,
                    "windows": int(train.features.shape[0]),
                    "position_fallback_bins": baseline.fallback_bin_count,
                },
                "window_counts": {
                    "calibration": int(calibration.features.shape[0]),
                    "test": int(evaluated.features.shape[0]),
                    "test_positive": int(np.sum(evaluated.labels)),
                    "test_records_considered": evaluated.records_considered,
                    "test_records_with_windows": evaluated.records_with_windows,
                    "early_error_records_skipped": evaluated.early_error_records_skipped,
                },
            },
            "models": models,
            "comparisons": {
                "combined_minus_geometry_auroc": self._difference(
                    models["combined"]["test_token"]["auroc"],
                    models["geometry"]["test_token"]["auroc"],
                ),
                "combined_minus_dynamics_auroc": self._difference(
                    models["combined"]["test_token"]["auroc"],
                    models["dynamics"]["test_token"]["auroc"],
                ),
            },
            "verdict": {
                "claim_supported": False,
                "scope": "exploratory predictive association, not an LLM-causal mechanism",
                "decision_rule": (
                    "replication across seeds/models and matched negative controls are required; "
                    "window length is fixed and relative position is conditioned, but first-error "
                    "annotations remain interval censored"
                ),
            },
        }
        self._progress("stage=save status=started")
        self._save(report)
        self._progress("stage=complete status=complete")
        return report

    def inspect(self) -> dict[str, Any]:
        """Read manifests and one shard per domain without fitting a baseline."""
        self._validate_config()
        domains = self._read_domains()
        return {
            domain.name: {
                "manifest": str(domain.manifest_path),
                "records": len(domain.records),
                "layers": domain.layers.tolist(),
                "analysis_axis": "token_time_within_each_stored_layer",
                "depth_semantics": (
                    "adjacent_layers"
                    if domain.layers.size > 1 and np.all(np.diff(domain.layers) == 1)
                    else "sparse_observation_layers"
                ),
                "first_shard": str(domain.records[0].state_path),
                "first_shard_shape": list(domain.load_states(domain.records[0]).shape),
            }
            for domain in domains
        }

    def _read_domains(self) -> list[RawDomain]:
        repository = RawHiddenRepository(
            self.config.data_root,
            manifest_name=self.config.manifest_name,
            max_records_per_domain=self.config.max_records_per_domain,
        )
        domains = [repository.read_domain(name) for name in self.config.domains]
        reference = domains[0].layers
        for domain in domains[1:]:
            if not np.array_equal(reference, domain.layers):
                raise ValueError(
                    "all domains must store identical observation layers; "
                    f"{domains[0].name}={reference.tolist()}, "
                    f"{domain.name}={domain.layers.tolist()}"
                )
        return domains

    def _validate_config(self) -> None:
        if not self.config.domains:
            raise ValueError("at least one domain is required")
        if self.config.window_size <= self.config.neighbors:
            raise ValueError("window_size must be larger than neighbors")
        if self.config.neighbors < 2 or self.config.tle_centers < 1:
            raise ValueError("neighbors must be at least 2 and tle_centers must be positive")
        limits = (
            self.config.train_windows_per_chain,
            self.config.calibration_windows_per_chain,
            self.config.max_test_windows_per_chain,
            self.config.position_bins,
            self.config.min_baseline_samples,
        )
        if any(value < 1 for value in limits):
            raise ValueError("window limits, position bins, and baseline samples must be positive")

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
        counts = {
            "train_correct": sum(
                value.split == "train" and value.record.error_step < 0 for value in records
            ),
            "calibration_correct": sum(
                value.split == "calibration" and value.record.error_step < 0
                for value in records
            ),
            "test_correct": sum(
                value.split == "test" and value.record.error_step < 0 for value in records
            ),
            "test_error": sum(
                value.split == "test" and value.record.error_step >= 0 for value in records
            ),
        }
        if min(counts.values()) == 0:
            raise ValueError(
                "problem-group split needs train/calibration correct chains and test "
                f"correct/error chains; got {counts}"
            )

    def _selected_endpoints(
        self,
        value: _RecordRef,
        state_count: int,
        limit: int,
        test_mode: bool,
    ) -> tuple[np.ndarray, np.ndarray, bool]:
        valid = np.arange(self.config.window_size - 1, state_count, dtype=np.int64)
        if valid.size == 0:
            return valid, np.empty(0, dtype=np.int64), value.record.error_step >= 0
        if not test_mode or value.record.error_step < 0:
            index = valid[_even_indices(valid.size, limit)]
            return index, np.zeros(index.size, dtype=np.int64), False

        event = value.record.step_ranges[value.record.error_step] - value.record.response_start
        event_start, event_end = int(event[0]), int(event[1])
        if event_start < 0 or event_end >= state_count:
            raise ValueError(
                f"{value.record.state_path}: first-error interval {event.tolist()} lies "
                f"outside response-state tokens [0,{state_count - 1}]"
            )
        positive = valid[(valid >= event_start) & (valid <= event_end)]
        if positive.size == 0:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), True
        negative = valid[valid < event_start]
        if positive.size + negative.size <= limit:
            index = np.concatenate([negative, positive])
        else:
            positive_budget = min(positive.size, max(1, limit // 2))
            negative_budget = min(negative.size, limit - positive_budget)
            positive_budget = min(positive.size, limit - negative_budget)
            index = np.sort(
                np.concatenate(
                    [
                        negative[_even_indices(negative.size, negative_budget)],
                        positive[_even_indices(positive.size, positive_budget)],
                    ]
                )
            )
        labels = ((index >= event_start) & (index <= event_end)).astype(np.int64)
        return index, labels, False

    def _extract_windows(
        self,
        records: list[_RecordRef],
        limit: int,
        *,
        test_mode: bool,
        stage: str,
    ) -> _WindowBatch:
        feature_rows: list[np.ndarray] = []
        label_rows: list[np.ndarray] = []
        domain_rows: list[np.ndarray] = []
        bin_rows: list[np.ndarray] = []
        chain_rows: list[np.ndarray] = []
        group_rows: list[np.ndarray] = []
        error_rows: list[np.ndarray] = []
        records_with_windows = 0
        early_error_records_skipped = 0
        window_count = 0
        started = time.monotonic()
        update_every = max(1, len(records) // 20)
        self._progress(
            f"stage={stage} records=0/{len(records)} percent=0.0 windows=0 "
            "elapsed=0.0s eta=pending"
        )
        for record_index, value in enumerate(records, start=1):
            states = value.source.load_states(value.record)
            endpoints, labels, skipped = self._selected_endpoints(
                value, states.shape[0], limit, test_mode
            )
            early_error_records_skipped += int(skipped)
            if endpoints.size == 0:
                if record_index % update_every == 0 or record_index == len(records):
                    self._window_progress(
                        stage, record_index, len(records), window_count, started
                    )
                continue
            records_with_windows += 1
            feature_rows.append(
                np.stack(
                    [
                        window_features(
                            states[
                                endpoint - self.config.window_size + 1 : endpoint + 1
                            ],
                            neighbors=self.config.neighbors,
                            tle_centers=self.config.tle_centers,
                        )
                        for endpoint in endpoints
                    ]
                )
            )
            relative_position = endpoints / max(states.shape[0] - 1, 1)
            position_bin = np.minimum(
                (relative_position * self.config.position_bins).astype(np.int64),
                self.config.position_bins - 1,
            )
            row_count = endpoints.size
            window_count += row_count
            identity = f"{value.source.name}::{value.record.row}"
            group = f"{value.source.name}::{value.record.problem_group}"
            label_rows.append(labels)
            domain_rows.append(np.full(row_count, value.source.name, dtype=object))
            bin_rows.append(position_bin)
            chain_rows.append(np.full(row_count, identity, dtype=object))
            group_rows.append(np.full(row_count, group, dtype=object))
            error_rows.append(
                np.full(row_count, value.record.error_step >= 0, dtype=bool)
            )
            if record_index % update_every == 0 or record_index == len(records):
                self._window_progress(
                    stage, record_index, len(records), window_count, started
                )
        if not feature_rows:
            raise ValueError(
                f"no token windows remain; window_size={self.config.window_size} is too large"
            )
        return _WindowBatch(
            features=np.concatenate(feature_rows),
            labels=np.concatenate(label_rows),
            domains=np.concatenate(domain_rows),
            position_bins=np.concatenate(bin_rows),
            chain_ids=np.concatenate(chain_rows),
            problem_groups=np.concatenate(group_rows),
            error_chains=np.concatenate(error_rows),
            records_considered=len(records),
            records_with_windows=records_with_windows,
            early_error_records_skipped=early_error_records_skipped,
        )

    def _progress(self, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[progress] {message}")

    def _window_progress(
        self,
        stage: str,
        completed: int,
        total: int,
        windows: int,
        started: float,
    ) -> None:
        elapsed = time.monotonic() - started
        eta = elapsed / completed * (total - completed)
        fraction = completed / total
        filled = round(20 * fraction)
        bar = "#" * filled + "-" * (20 - filled)
        self._progress(
            f"stage={stage} [{bar}] records={completed}/{total} "
            f"percent={100.0 * fraction:.1f} windows={windows} "
            f"elapsed={elapsed:.1f}s eta={eta:.1f}s"
        )

    def _arm_report(
        self,
        calibration_z: np.ndarray,
        evaluated_z: np.ndarray,
        evaluated: _WindowBatch,
        feature_indices: tuple[int, ...],
        stored_layers: np.ndarray,
    ) -> dict[str, Any]:
        calibration_layer_scores = np.mean(
            calibration_z[:, :, feature_indices] ** 2, axis=2
        )
        test_layer_scores = np.mean(evaluated_z[:, :, feature_indices] ** 2, axis=2)
        test_scores = np.mean(test_layer_scores, axis=1)
        within = []
        for chain in np.unique(evaluated.chain_ids[evaluated.error_chains]):
            selected = evaluated.chain_ids == chain
            value = _safe_metric(
                evaluated.labels[selected], test_scores[selected], "auroc"
            )
            if value is not None:
                within.append(value)
        return {
            "features": [FEATURE_NAMES[index] for index in feature_indices],
            "fit": "squared robust z distance to correct-only norms; no neural network",
            "calibration_correct_score": {
                "mean": float(np.mean(calibration_layer_scores)),
                "median": float(np.median(calibration_layer_scores)),
                "windows": int(calibration_layer_scores.shape[0]),
            },
            "test_token": self._metric_report(test_scores, evaluated, within),
            "test_token_by_stored_layer": [
                {
                    "stored_layer": int(layer),
                    "auroc": _domain_macro_metric(
                        evaluated.labels,
                        test_layer_scores[:, index],
                        evaluated.domains,
                        "auroc",
                    ),
                    "auprc": _domain_macro_metric(
                        evaluated.labels,
                        test_layer_scores[:, index],
                        evaluated.domains,
                        "auprc",
                    ),
                }
                for index, layer in enumerate(stored_layers)
            ],
        }

    def _metric_report(
        self,
        scores: np.ndarray,
        evaluated: _WindowBatch,
        within: list[float],
    ) -> dict[str, Any]:
        return {
            "auroc": _domain_macro_metric(
                evaluated.labels, scores, evaluated.domains, "auroc"
            ),
            "auroc_ci": _bootstrap_interval(
                evaluated.labels,
                scores,
                evaluated.domains,
                evaluated.problem_groups,
                "auroc",
                self.config.bootstrap_samples,
                self.config.seed,
            ),
            "auprc": _domain_macro_metric(
                evaluated.labels, scores, evaluated.domains, "auprc"
            ),
            "auprc_ci": _bootstrap_interval(
                evaluated.labels,
                scores,
                evaluated.domains,
                evaluated.problem_groups,
                "auprc",
                self.config.bootstrap_samples,
                self.config.seed + 1,
            ),
            "pooled_auroc": _safe_metric(evaluated.labels, scores, "auroc"),
            "pooled_auprc": _safe_metric(evaluated.labels, scores, "auprc"),
            "within_error_chain_macro_auroc": (
                None if not within else float(np.mean(within))
            ),
            "windows": int(scores.size),
            "positive_windows": int(np.sum(evaluated.labels)),
            "problem_groups": int(np.unique(evaluated.problem_groups).size),
        }

    @staticmethod
    def _difference(left: float | None, right: float | None) -> float | None:
        return None if left is None or right is None else float(left - right)

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
            "unit: fixed-length token window within each independently observed stored layer",
            "coordinates: raw hidden states; no PCA, KMeans, step pooling, or neural network",
            "fit: train-split fully-correct chains only",
            "",
        ]
        for name, values in report["models"].items():
            token = values["test_token"]
            auroc = "NA" if token["auroc"] is None else f"{token['auroc']:.4f}"
            auprc = "NA" if token["auprc"] is None else f"{token['auprc']:.4f}"
            within = (
                "NA"
                if token["within_error_chain_macro_auroc"] is None
                else f"{token['within_error_chain_macro_auroc']:.4f}"
            )
            lines.append(
                f"{name}: AUROC={auroc} | AUPRC={auprc} | "
                f"within_error_chain_AUROC={within}"
            )
        lines.extend(
            [
                "",
                (
                    "early_error_records_skipped: "
                    f"{report['data']['window_counts']['early_error_records_skipped']}"
                ),
                f"claim_supported: {report['verdict']['claim_supported']}",
                f"scope: {report['verdict']['scope']}",
            ]
        )
        text_temporary = output / "summary.txt.tmp"
        text_temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
        text_temporary.replace(output / "summary.txt")
