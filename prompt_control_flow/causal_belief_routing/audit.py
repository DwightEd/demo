from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .charts import (
    RandomProjection,
    build_group_fold_ids,
    cross_fit_ridge_accelerated,
    fit_ridge_accelerated,
    fit_layer_chart_bundle,
    project_features,
)
from .metrics import (
    conditional_usable_bits,
    evaluate_fourier_predictions,
    paired_alias_js,
    paired_target_identification_accuracy,
    softmax,
)
from .schema import CausalBeliefTrace


@dataclass(frozen=True)
class RepresentationAuditConfig:
    folds: int = 5
    projection_dim: int = 32
    ridge_alpha: float = 10.0
    bootstrap: int = 2000
    seed: int = 17
    max_current_alias_js: float = 0.05
    min_future_accuracy_gain: float = 0.10
    compute_device: str = "cpu"

    def validate(self) -> None:
        if int(self.folds) < 2:
            raise ValueError("folds must be at least two")
        if int(self.projection_dim) < 1:
            raise ValueError("projection_dim must be positive")
        if float(self.ridge_alpha) < 0.0:
            raise ValueError("ridge_alpha must be non-negative")
        if int(self.bootstrap) < 0:
            raise ValueError("bootstrap must be non-negative")
        if int(self.seed) < 0:
            raise ValueError("seed must be non-negative")
        if not 0.0 <= float(self.max_current_alias_js) <= np.log(2.0):
            raise ValueError("max_current_alias_js is outside the JS range")
        if not str(self.compute_device):
            raise ValueError("compute_device must be non-empty")


def _future_lookup(trace: CausalBeliefTrace) -> dict[tuple[int, int], int]:
    result: dict[tuple[int, int], int] = {}
    for index in np.flatnonzero(trace.query_roles == "future"):
        key = (int(trace.pair_ids[index]), int(trace.branches[index]))
        if key in result:
            raise ValueError(f"duplicate future observation for pair/branch {key}")
        result[key] = int(index)
    return result


def _current_future_task(trace: CausalBeliefTrace) -> dict[str, np.ndarray]:
    current_indices = np.flatnonzero(trace.current_mask)
    if len(current_indices) < 2:
        raise ValueError("trace does not contain enough current-query rows")
    lookup = _future_lookup(trace)
    future_indices = np.asarray(
        [
            lookup[(int(trace.pair_ids[index]), int(trace.branches[index]))]
            for index in current_indices
        ],
        dtype=np.int64,
    )
    labels = np.argmax(
        trace.exact_query_distributions[future_indices], axis=1
    ).astype(np.int64)
    return {
        "current_indices": current_indices,
        "future_indices": future_indices,
        "pair_ids": trace.pair_ids[current_indices],
        "branches": trace.branches[current_indices],
        "targets": trace.fourier_targets[current_indices].astype(np.float32),
        "future_queries": trace.query_vectors[future_indices],
        "future_labels": labels,
    }


def _output_features(
    trace: CausalBeliefTrace,
    current_indices: np.ndarray,
    *,
    include_controls: bool,
) -> np.ndarray:
    residue = trace.residue_logits[current_indices].astype(np.float32)
    residue -= residue.mean(axis=1, keepdims=True)
    sketch = trace.logit_sketch[current_indices].astype(np.float32)
    lengths = np.log1p(
        np.asarray([len(trace.input_ids[index]) for index in current_indices])
    ).astype(np.float32)[:, None]
    families = trace.template_families[current_indices].astype(np.int64)
    width = int(families.max(initial=0)) + 1
    one_hot = np.zeros((len(families), width), dtype=np.float32)
    one_hot[np.arange(len(families)), families] = 1.0
    output = [residue, sketch]
    if include_controls:
        output.extend([lengths, one_hot])
    return np.concatenate(output, axis=1)


def _project_layers(
    states: np.ndarray,
    layers: np.ndarray,
    *,
    projection_dim: int,
    seed: int,
    compute_device: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(states, dtype=np.float32)
    output_dim = min(int(projection_dim), int(values.shape[2]))
    projected = []
    seeds = []
    for position, layer in enumerate(np.asarray(layers, dtype=np.int64)):
        projection_seed = int(seed) + 1009 * int(layer)
        projection = RandomProjection(values.shape[2], output_dim, projection_seed)
        projected.append(
            project_features(
                values[:, position],
                projection,
                compute_device=compute_device,
            )
        )
        seeds.append(projection_seed)
    return np.stack(projected, axis=1), np.asarray(seeds, dtype=np.int64)


def _cross_fit_permuted_targets(
    features: np.ndarray,
    targets: np.ndarray,
    fold_ids: np.ndarray,
    *,
    alpha: float,
    seed: int,
    compute_device: str,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    y = np.asarray(targets, dtype=np.float32)
    folds = np.asarray(fold_ids, dtype=np.int16)
    rng = np.random.default_rng(int(seed))
    result = np.empty_like(y)
    for fold in sorted(np.unique(folds).tolist()):
        test = folds == int(fold)
        train_indices = np.flatnonzero(~test)
        permuted = train_indices[rng.permutation(len(train_indices))]
        model = fit_ridge_accelerated(
            x[train_indices],
            y[permuted],
            alpha=alpha,
            compute_device=compute_device,
        )
        result[test] = model.predict(x[test])
    return result


def _serializable_metrics(metrics: dict[str, float | np.ndarray]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metrics.items()
        if np.asarray(value).ndim == 0
    }


def _evaluate_prediction_set(
    predictions: dict[str, np.ndarray],
    *,
    targets: np.ndarray,
    frequencies: np.ndarray,
    future_queries: np.ndarray,
    future_labels: np.ndarray,
    pair_ids: np.ndarray,
    branches: np.ndarray,
    modulus: int,
) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, np.ndarray]]]:
    summaries: dict[str, dict[str, float]] = {}
    row_metrics: dict[str, dict[str, np.ndarray]] = {}
    for name, values in predictions.items():
        evaluated = evaluate_fourier_predictions(
            values,
            targets,
            frequencies,
            future_queries,
            future_labels,
            modulus=modulus,
        )
        summaries[name] = _serializable_metrics(evaluated)
        summaries[name]["paired_target_accuracy"] = paired_target_identification_accuracy(
            values, targets, pair_ids, branches
        )
        row_metrics[name] = {
            key: np.asarray(value)
            for key, value in evaluated.items()
            if np.asarray(value).ndim > 0
        }
    return summaries, row_metrics


def _write_score_table(
    path: Path,
    pair_ids: np.ndarray,
    branches: np.ndarray,
    labels: np.ndarray,
    row_metrics: dict[str, dict[str, np.ndarray]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    model_names = sorted(row_metrics)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        header = ["pair_id", "branch", "future_label"]
        for name in model_names:
            header.extend([f"{name}.future_nll_nats", f"{name}.fourier_mse"])
        writer.writerow(header)
        for row in range(len(pair_ids)):
            values: list[Any] = [int(pair_ids[row]), int(branches[row]), int(labels[row])]
            for name in model_names:
                values.extend(
                    [
                        float(row_metrics[name]["row_future_nll_nats"][row]),
                        float(row_metrics[name]["row_fourier_mse"][row]),
                    ]
                )
            writer.writerow(values)


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Causal Belief Routing Representation Audit",
        "",
        "The primary test asks whether current residual states preserve a future-relevant "
        "belief distinction after the current output distribution has been matched by construction.",
        "",
        f"- Alias pairs: `{report['data']['pairs']}`",
        f"- Current observations: `{report['data']['current_rows']}`",
        f"- Layers: `{report['data']['layers']}`",
        f"- Modulus: `GF({report['data']['modulus']})`",
        "",
        "## Predictive-Alias Checks",
        "",
        f"- Exact current alias: `{report['alias_checks']['exact_current_alias']}`",
        f"- Median model current JS: `{report['alias_checks']['model_current_js_median']:.6f}` nats",
        f"- Model future-query accuracy: `{report['alias_checks']['model_future_accuracy']:.4f}`",
        "",
        "## Held-Out Belief Decoding",
        "",
        "| representation | Fourier R2 | future NLL | future accuracy | pair target acc. |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, metrics in report["prediction_metrics"].items():
        lines.append(
            f"| {name} | {metrics['fourier_r2']:.4f} | "
            f"{metrics['future_nll_nats']:.4f} | {metrics['future_accuracy']:.4f} | "
            f"{metrics['paired_target_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Information",
            "",
            "| comparison | usable bits | 95% CI | pairs |",
            "|---|---:|---|---:|",
        ]
    )
    for name, metric in report["conditional_information"].items():
        lines.append(
            f"| {name} | {metric['point']:.4f} | "
            f"[{metric['ci_low']:.4f}, {metric['ci_high']:.4f}] | {metric['groups']} |"
        )
    lines.extend(
        [
            "",
            "## Layer Trajectory",
            "",
            "| layer | Fourier R2 | future NLL | future accuracy | pair target acc. |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in report["layer_metrics"]:
        lines.append(
            f"| {row['layer']} | {row['fourier_r2']:.4f} | "
            f"{row['future_nll_nats']:.4f} | {row['future_accuracy']:.4f} | "
            f"{row['paired_target_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Representation supported: `{report['decision_gate']['representation_supported']}`",
            f"- Ready for routing analysis: `{report['decision_gate']['ready_for_routing_analysis']}`",
        ]
    )
    for name, value in report["decision_gate"]["conditions"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "A positive result is not a claim that hidden states are Bayesian in general. "
            "It is evidence that this controlled task induces a cross-layer representation "
            "of an analytically known posterior that is not recoverable from the matched "
            "current output alone.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_representation_audit(
    trace_path: str | Path,
    output_dir: str | Path,
    cfg: RepresentationAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    trace = CausalBeliefTrace.load(trace_path)
    task = _current_future_task(trace)
    current = task["current_indices"]
    pair_ids = task["pair_ids"]
    targets = task["targets"]
    modulus = int(trace.metadata.get("modulus", trace.residue_logits.shape[1]))
    unique_pairs = np.unique(pair_ids)
    if len(unique_pairs) < int(cfg.folds):
        raise ValueError("fewer alias pairs than requested folds")

    fold_ids = build_group_fold_ids(pair_ids, num_folds=cfg.folds, seed=cfg.seed)
    output_only_features = _output_features(
        trace, current, include_controls=False
    )
    output_features = _output_features(trace, current, include_controls=True)
    output_only_predictions, _ = cross_fit_ridge_accelerated(
        output_only_features,
        targets,
        fold_ids,
        alpha=cfg.ridge_alpha,
        compute_device=cfg.compute_device,
    )
    output_predictions, _ = cross_fit_ridge_accelerated(
        output_features,
        targets,
        fold_ids,
        alpha=cfg.ridge_alpha,
        compute_device=cfg.compute_device,
    )

    layer_projection_dim = min(cfg.projection_dim, trace.states.shape[2])
    chart_bundle, layer_predictions, chart_fold_ids = fit_layer_chart_bundle(
        trace.states[current],
        targets,
        pair_ids,
        trace.layers,
        num_folds=cfg.folds,
        projection_dim=layer_projection_dim,
        projection_seed=cfg.seed + 101,
        alpha=cfg.ridge_alpha,
        split_seed=cfg.seed,
        compute_device=cfg.compute_device,
    )
    if not np.array_equal(fold_ids, chart_fold_ids):
        raise RuntimeError("layer and output cross-fitting folds diverged")

    projected_layers, _ = _project_layers(
        trace.states[current],
        trace.layers,
        projection_dim=layer_projection_dim,
        seed=cfg.seed + 101,
        compute_device=cfg.compute_device,
    )
    hidden_features = projected_layers.reshape(len(current), -1)
    hidden_predictions, _ = cross_fit_ridge_accelerated(
        hidden_features,
        targets,
        fold_ids,
        alpha=cfg.ridge_alpha,
        compute_device=cfg.compute_device,
    )
    joint_predictions, _ = cross_fit_ridge_accelerated(
        np.concatenate([output_features, hidden_features], axis=1),
        targets,
        fold_ids,
        alpha=cfg.ridge_alpha,
        compute_device=cfg.compute_device,
    )
    shuffled_predictions = _cross_fit_permuted_targets(
        hidden_features,
        targets,
        fold_ids,
        alpha=cfg.ridge_alpha,
        seed=cfg.seed + 202,
        compute_device=cfg.compute_device,
    )

    predictions = {
        "output_only": output_only_predictions,
        "output_plus_controls": output_predictions,
        "hidden_all_layers": hidden_predictions,
        "output_controls_plus_hidden": joint_predictions,
        "hidden_shuffled_null": shuffled_predictions,
    }
    summaries, rows = _evaluate_prediction_set(
        predictions,
        targets=targets,
        frequencies=trace.frequencies,
        future_queries=task["future_queries"],
        future_labels=task["future_labels"],
        pair_ids=pair_ids,
        branches=task["branches"],
        modulus=modulus,
    )
    layer_metrics = []
    for position, layer in enumerate(trace.layers):
        evaluated = evaluate_fourier_predictions(
            layer_predictions[:, position],
            targets,
            trace.frequencies,
            task["future_queries"],
            task["future_labels"],
            modulus=modulus,
        )
        layer_metrics.append(
            {
                "layer": int(layer),
                **_serializable_metrics(evaluated),
                "paired_target_accuracy": paired_target_identification_accuracy(
                    layer_predictions[:, position],
                    targets,
                    pair_ids,
                    task["branches"],
                ),
            }
        )

    conditional = {
        "hidden_over_output_plus_controls": conditional_usable_bits(
            rows["output_plus_controls"]["row_future_nll_nats"],
            rows["hidden_all_layers"]["row_future_nll_nats"],
            pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 301,
        ),
        "joint_over_output_plus_controls": conditional_usable_bits(
            rows["output_plus_controls"]["row_future_nll_nats"],
            rows["output_controls_plus_hidden"]["row_future_nll_nats"],
            pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 302,
        ),
        "hidden_over_shuffled": conditional_usable_bits(
            rows["hidden_shuffled_null"]["row_future_nll_nats"],
            rows["hidden_all_layers"]["row_future_nll_nats"],
            pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 303,
        ),
    }
    serializable_conditional = {
        name: {key: value for key, value in metric.items() if key != "row_bits"}
        for name, metric in conditional.items()
    }

    current_probabilities = softmax(trace.residue_logits[current])
    current_js = paired_alias_js(current_probabilities, pair_ids, task["branches"])
    exact_current = trace.exact_query_distributions[current]
    exact_js = paired_alias_js(exact_current, pair_ids, task["branches"])
    future_probabilities = softmax(trace.residue_logits[task["future_indices"]])
    model_future_accuracy = float(
        np.mean(np.argmax(future_probabilities, axis=1) == task["future_labels"])
    )
    chance = 1.0 / float(modulus)
    conditions = {
        "exact_current_alias_verified": bool(np.max(exact_js, initial=0.0) < 1e-10),
        "model_current_alias_preserved": bool(
            np.median(current_js) <= cfg.max_current_alias_js
        ),
        "model_future_behavior_above_chance": bool(
            model_future_accuracy >= chance + cfg.min_future_accuracy_gain
        ),
        "hidden_future_bits_ci_above_zero": bool(
            conditional["hidden_over_output_plus_controls"]["ci_low"] > 0.0
        ),
        "joint_future_bits_ci_above_zero": bool(
            conditional["joint_over_output_plus_controls"]["ci_low"] > 0.0
        ),
        "hidden_beats_shuffled_null": bool(
            conditional["hidden_over_shuffled"]["ci_low"] > 0.0
        ),
    }
    representation_supported = all(
        conditions[name]
        for name in (
            "exact_current_alias_verified",
            "model_current_alias_preserved",
            "joint_future_bits_ci_above_zero",
            "hidden_beats_shuffled_null",
        )
    )
    ready_for_routing = bool(
        representation_supported and conditions["model_future_behavior_above_chance"]
    )

    report: dict[str, Any] = {
        "method": "causal_belief_routing_predictive_alias_v1",
        "data": {
            "trace": str(trace_path),
            "pairs": int(len(unique_pairs)),
            "current_rows": int(len(current)),
            "layers": [int(layer) for layer in trace.layers],
            "modulus": modulus,
        },
        "config": {
            "folds": cfg.folds,
            "projection_dim": layer_projection_dim,
            "ridge_alpha": cfg.ridge_alpha,
            "bootstrap": cfg.bootstrap,
            "seed": cfg.seed,
            "max_current_alias_js": cfg.max_current_alias_js,
            "min_future_accuracy_gain": cfg.min_future_accuracy_gain,
            "compute_device": cfg.compute_device,
        },
        "alias_checks": {
            "exact_current_alias": bool(np.max(exact_js, initial=0.0) < 1e-10),
            "exact_current_js_max": float(np.max(exact_js, initial=0.0)),
            "model_current_js_mean": float(np.mean(current_js)),
            "model_current_js_median": float(np.median(current_js)),
            "model_current_js_p95": float(np.quantile(current_js, 0.95)),
            "model_future_accuracy": model_future_accuracy,
            "chance_accuracy": chance,
        },
        "prediction_metrics": summaries,
        "conditional_information": serializable_conditional,
        "layer_metrics": layer_metrics,
        "decision_gate": {
            "conditions": conditions,
            "representation_supported": representation_supported,
            "ready_for_routing_analysis": ready_for_routing,
        },
    }

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    chart_bundle.metadata.update(
        {
            "trace": str(trace_path),
            "target": "split_finite_field_fourier_belief",
            "decision_gate": report["decision_gate"],
        }
    )
    chart_bundle.save(output / "layer_charts.npz")
    np.savez(
        output / "oof_predictions.npz",
        pair_ids=pair_ids,
        branches=task["branches"],
        future_labels=task["future_labels"],
        fold_ids=fold_ids,
        targets=targets,
        output_only=output_only_predictions,
        output_plus_controls=output_predictions,
        hidden_all_layers=hidden_predictions,
        output_controls_plus_hidden=joint_predictions,
        hidden_shuffled_null=shuffled_predictions,
        layer_predictions=layer_predictions,
    )
    _write_score_table(
        output / "row_scores.csv",
        pair_ids,
        task["branches"],
        task["future_labels"],
        rows,
    )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    (output / "summary.md").write_text(_render_report(report), encoding="utf-8")
    return report
