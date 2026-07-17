from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .artifact import BeliefTraceArtifact
from .belief import (
    categorical_entropy,
    fisher_rao_distance,
    mask_to_belief,
    transition_diagnostics,
)
from .decoder import (
    DecoderConfig,
    build_control_features,
    cross_fit_belief_decoder,
    problem_balanced_row_weights,
)


@dataclass(frozen=True)
class BeliefAuditConfig:
    decoder: DecoderConfig = DecoderConfig()
    primary_layer: int = 16
    bootstrap: int = 2000
    seed: int = 17
    max_null_information_gain_gap: float = 1e-8

    def validate(self) -> None:
        self.decoder.validate()
        if self.bootstrap < 100:
            raise ValueError("bootstrap must be at least 100")
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.max_null_information_gain_gap < 0.0:
            raise ValueError("max_null_information_gain_gap must be non-negative")


def _normalize(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all() or np.any(values < 0.0):
        raise ValueError("belief predictions must be a finite non-negative matrix")
    mass = values.sum(axis=1, keepdims=True)
    if np.any(mass <= 0.0):
        raise ValueError("belief predictions must have positive row mass")
    return values / mass


def belief_prediction_metrics(
    target: np.ndarray,
    prediction: np.ndarray,
    feasible_mask: np.ndarray,
    groups: np.ndarray | None = None,
) -> dict[str, float]:
    truth = _normalize(target)
    estimate = _normalize(prediction)
    support = np.asarray(feasible_mask, dtype=bool)
    if truth.shape != estimate.shape or support.shape != truth.shape:
        raise ValueError("belief metric inputs have incompatible shapes")
    eps = 1e-9
    per_row_kl = np.sum(
        np.where(
            truth > 0.0,
            truth * (np.log(np.clip(truth, eps, None)) - np.log(np.clip(estimate, eps, None))),
            0.0,
        ),
        axis=1,
    )
    row_weight = (
        np.ones(len(truth), dtype=np.float64)
        if groups is None
        else problem_balanced_row_weights(np.asarray(groups))
    )
    row_weight = row_weight / np.sum(row_weight)
    support_auc = []
    support_auc_weight = []
    for index, (row_support, row_score) in enumerate(zip(support, estimate)):
        if np.any(row_support) and np.any(~row_support):
            support_auc.append(roc_auc_score(row_support.astype(np.int64), row_score))
            support_auc_weight.append(row_weight[index])
    exact_entropy = categorical_entropy(truth)
    predicted_entropy = categorical_entropy(estimate)
    if np.std(exact_entropy) <= 1e-12 or np.std(predicted_entropy) <= 1e-12:
        correlation = float("nan")
    else:
        correlation = spearmanr(exact_entropy, predicted_entropy).statistic
    feasible_mass = np.sum(estimate * support, axis=1)
    return {
        "soft_ce_nats": float(
            np.sum(
                row_weight
                * -np.sum(truth * np.log(np.clip(estimate, eps, None)), axis=1)
            )
        ),
        "kl_nats": float(np.sum(row_weight * per_row_kl)),
        "tvd": float(
            np.sum(row_weight * 0.5 * np.sum(np.abs(truth - estimate), axis=1))
        ),
        "fisher_rao": float(
            np.sum(row_weight * fisher_rao_distance(truth, estimate))
        ),
        "support_auc": float(
            np.average(support_auc, weights=support_auc_weight)
        )
        if support_auc
        else float("nan"),
        "feasible_mass": float(np.sum(row_weight * feasible_mass)),
        "entropy_spearman": float(correlation) if np.isfinite(correlation) else float("nan"),
    }


def build_matched_wrong_masks(
    artifact: BeliefTraceArtifact,
    *,
    seed: int,
) -> np.ndarray:
    """Build a minimally perturbed, exactly rank-matched support operator.

    For each legal transition, remove one hypothesis from the true posterior
    support and insert one hypothesis rejected by the true condition but still
    present in the prior.  The null therefore has exactly the same support
    cardinality and differs by the smallest possible non-zero support edit.
    """

    artifact.validate()
    rng = np.random.default_rng(int(seed))
    wrong = np.asarray(artifact.condition_mask, dtype=bool).copy()
    lookup = {
        (int(problem), int(prefix)): row
        for row, (problem, prefix) in enumerate(
            zip(artifact.problem_ids, artifact.prefix_index)
        )
    }
    for index in np.flatnonzero(artifact.prefix_index > 0):
        prior_index = lookup[
            (int(artifact.problem_ids[index]), int(artifact.prefix_index[index]) - 1)
        ]
        prior_support = artifact.feasible_mask[prior_index]
        true_support = artifact.feasible_mask[index]
        retained = np.flatnonzero(true_support)
        rejected = np.flatnonzero(prior_support & ~true_support)
        if len(retained) == 0 or len(rejected) == 0:
            raise ValueError("every transition must be a strict non-empty support reduction")
        remove = int(retained[int(rng.integers(0, len(retained)))])
        insert = int(rejected[int(rng.integers(0, len(rejected)))])
        null_support = true_support.copy()
        null_support[remove] = False
        null_support[insert] = True
        if int(null_support.sum()) != int(true_support.sum()):
            raise RuntimeError("rank-matched null construction changed support size")
        wrong[index] = null_support
    return wrong


def directional_transport_rows(
    artifact: BeliefTraceArtifact,
    predictions: np.ndarray,
    wrong_masks: np.ndarray,
) -> dict[str, np.ndarray]:
    artifact.validate()
    belief = _normalize(predictions)
    wrong = np.asarray(wrong_masks, dtype=bool)
    if belief.shape != artifact.feasible_mask.shape or wrong.shape != belief.shape:
        raise ValueError("directional transport inputs do not match the trace")
    lookup = {
        (int(problem), int(prefix)): index
        for index, (problem, prefix) in enumerate(
            zip(artifact.problem_ids, artifact.prefix_index)
        )
    }
    rows: dict[str, list[float | int]] = {
        "problem_id": [],
        "prefix_index": [],
        "true_transport_residual": [],
        "wrong_transport_residual": [],
        "operator_advantage": [],
        "true_support_gain": [],
        "wrong_support_gain": [],
        "true_unsupported_contraction": [],
        "wrong_unsupported_contraction": [],
        "unsupported_advantage": [],
        "true_information_gain": [],
        "wrong_information_gain": [],
        "information_gain_gap": [],
    }
    for after_index in np.flatnonzero(artifact.prefix_index > 0):
        problem = int(artifact.problem_ids[after_index])
        prefix = int(artifact.prefix_index[after_index])
        before_index = lookup[(problem, prefix - 1)]
        true = transition_diagnostics(
            belief[before_index],
            belief[after_index],
            artifact.condition_mask[after_index],
            epsilon_prior=0.0,
        )
        null = transition_diagnostics(
            belief[before_index],
            belief[after_index],
            wrong[after_index],
        )
        prior_support = artifact.feasible_mask[before_index]
        prior_count = int(prior_support.sum())
        true_count = int((prior_support & artifact.condition_mask[after_index]).sum())
        wrong_count = int((prior_support & wrong[after_index]).sum())
        true_gain = float(np.log(prior_count / true_count))
        wrong_gain = float(np.log(prior_count / wrong_count))
        rows["problem_id"].append(problem)
        rows["prefix_index"].append(prefix)
        rows["true_transport_residual"].append(true["transport_residual"])
        rows["wrong_transport_residual"].append(null["transport_residual"])
        rows["operator_advantage"].append(
            null["transport_residual"] - true["transport_residual"]
        )
        rows["true_support_gain"].append(true["support_gain"])
        rows["wrong_support_gain"].append(null["support_gain"])
        rows["true_unsupported_contraction"].append(
            true["unsupported_contraction"]
        )
        rows["wrong_unsupported_contraction"].append(
            null["unsupported_contraction"]
        )
        rows["unsupported_advantage"].append(
            null["unsupported_contraction"] - true["unsupported_contraction"]
        )
        rows["true_information_gain"].append(true_gain)
        rows["wrong_information_gain"].append(wrong_gain)
        rows["information_gain_gap"].append(abs(wrong_gain - true_gain))
    return {
        name: np.asarray(values, dtype=np.int64 if name in {"problem_id", "prefix_index"} else np.float64)
        for name, values in rows.items()
    }


def _soft_ce_rows(target: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    truth = _normalize(target)
    estimate = _normalize(prediction)
    return -np.sum(truth * np.log(np.clip(estimate, 1e-9, None)), axis=1)


def cluster_bootstrap_mean(
    values: np.ndarray,
    groups: np.ndarray,
    *,
    repetitions: int,
    seed: int,
) -> dict[str, float]:
    score = np.asarray(values, dtype=np.float64)
    group_values = np.asarray(groups)
    if score.shape != (len(group_values),):
        raise ValueError("bootstrap score must have one value per row")
    unique = np.unique(group_values)
    group_means = np.asarray(
        [np.mean(score[group_values == group]) for group in unique], dtype=np.float64
    )
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(repetitions), dtype=np.float64)
    for index in range(int(repetitions)):
        sampled = rng.integers(0, len(group_means), size=len(group_means))
        draws[index] = float(np.mean(group_means[sampled]))
    return {
        "mean": float(np.mean(group_means)),
        "ci_low": float(np.quantile(draws, 0.025)),
        "ci_high": float(np.quantile(draws, 0.975)),
        "groups": int(len(unique)),
    }


def _paired_operator_auc(true_residual: np.ndarray, wrong_residual: np.ndarray) -> float:
    true = np.asarray(true_residual, dtype=np.float64)
    wrong = np.asarray(wrong_residual, dtype=np.float64)
    if true.shape != wrong.shape:
        raise ValueError("paired residual arrays differ in shape")
    return float(np.mean((true < wrong).astype(float) + 0.5 * (true == wrong)))


def _write_layer_metrics(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_transition_rows(path: Path, rows: Mapping[str, np.ndarray]) -> None:
    names = list(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(names)
        for index in range(len(rows[names[0]])):
            writer.writerow([rows[name][index] for name in names])


def _summary_markdown(report: Mapping[str, Any]) -> str:
    lines = [
        "# Constraint-Supported Belief Transport Audit",
        "",
        f"- Rows: `{report['rows']}`",
        f"- Problems: `{report['problems']}`",
        f"- Primary layer: `{report['primary_layer']}`",
        f"- Hypotheses: `{report['hypotheses']}`",
        "",
        "## Belief Readout",
        "",
        "| source | support AUROC | KL (nats) | Fisher-Rao | entropy rho |",
        "|---|---:|---:|---:|---:|",
    ]
    for name in ("nuisance", "output", "primary_hidden", "primary_joint"):
        metrics = report["belief_metrics"][name]
        lines.append(
            f"| {name} | {metrics['support_auc']:.4f} | {metrics['kl_nats']:.4f} | "
            f"{metrics['fisher_rao']:.4f} | {metrics['entropy_spearman']:.4f} |"
        )
    increment = report["joint_over_output_usable_bits"]
    lines.extend(
        [
            "",
            "## Conditional Information",
            "",
            f"Adding the primary hidden state to output controls contributes "
            f"`{increment['mean']:.4f}` usable bits per prefix "
            f"over nuisance + compact logits, 95% CI "
            f"`[{increment['ci_low']:.4f}, {increment['ci_high']:.4f}]`.",
            "",
            "## Directional Transport",
            "",
            f"- Matched operator AUROC: `{report['directionality']['operator_auc']:.4f}`",
            f"- Wrong-minus-true Fisher-Rao residual: "
            f"`{report['directionality']['operator_advantage']['mean']:.4f}` "
            f"(95% CI `[{report['directionality']['operator_advantage']['ci_low']:.4f}, "
            f"{report['directionality']['operator_advantage']['ci_high']:.4f}]`)",
            f"- Null information-gain gap p95: "
            f"`{report['directionality']['null_information_gain_gap_p95']:.4f}` nats",
            "",
            "## Decision Gate",
            "",
            f"- Belief state decodable: `{report['decision_gate']['belief_state_decodable']}`",
            f"- Constraint direction supported: "
            f"`{report['decision_gate']['constraint_direction_supported']}`",
            f"- Ready for ProcessBench transfer: "
            f"`{report['decision_gate']['ready_for_processbench_transfer']}`",
            "",
            "This stage does not claim ProcessBench error detection. It tests whether the "
            "proposed state and transport objects exist in a setting with exact semantics.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_belief_transport_audit(
    trace_path: str | Path,
    output_dir: str | Path,
    cfg: BeliefAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    artifact = BeliefTraceArtifact.load(trace_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if int(cfg.primary_layer) not in set(artifact.layers.tolist()):
        raise ValueError(
            f"primary_layer={cfg.primary_layer} is absent; available={artifact.layers.tolist()}"
        )
    target = np.asarray(
        [mask_to_belief(mask) for mask in artifact.feasible_mask], dtype=np.float32
    )
    groups = np.asarray(artifact.problem_ids)
    nuisance_features = build_control_features(artifact, include_output=False)
    output_features = build_control_features(artifact, include_output=True)
    print("cross-fitting nuisance-only belief decoder")
    nuisance_result = cross_fit_belief_decoder(
        nuisance_features, target, groups, cfg.decoder
    )
    print("cross-fitting nuisance + compact-logits belief decoder")
    output_result = cross_fit_belief_decoder(
        output_features, target, groups, cfg.decoder
    )

    layer_predictions = np.empty(
        (artifact.n_rows, len(artifact.layers), target.shape[1]), dtype=np.float32
    )
    joint_predictions = np.empty_like(layer_predictions)
    layer_metric_rows: list[dict[str, Any]] = []
    fold_diagnostics: dict[str, Any] = {
        "nuisance": nuisance_result.fold_diagnostics,
        "output": output_result.fold_diagnostics,
    }
    for position, layer in enumerate(artifact.layers):
        print(f"cross-fitting hidden belief decoder at layer {int(layer)}")
        result = cross_fit_belief_decoder(
            np.asarray(artifact.states[:, position], dtype=np.float32),
            target,
            groups,
            cfg.decoder,
        )
        layer_predictions[:, position] = result.predictions
        hidden_metrics = belief_prediction_metrics(
            target, result.predictions, artifact.feasible_mask, groups
        )
        joint_result = cross_fit_belief_decoder(
            np.concatenate(
                [
                    output_features,
                    np.asarray(artifact.states[:, position], dtype=np.float32),
                ],
                axis=1,
            ),
            target,
            groups,
            cfg.decoder,
        )
        joint_predictions[:, position] = joint_result.predictions
        joint_metrics = belief_prediction_metrics(
            target, joint_result.predictions, artifact.feasible_mask, groups
        )
        conditional_bits_rows = (
            _soft_ce_rows(target, output_result.predictions)
            - _soft_ce_rows(target, joint_result.predictions)
        ) / np.log(2.0)
        conditional_bits = float(
            cluster_bootstrap_mean(
                conditional_bits_rows,
                groups,
                repetitions=100,
                seed=cfg.seed + 10 + position,
            )["mean"]
        )
        layer_metric_rows.append(
            {
                "layer": int(layer),
                **{f"hidden_{name}": value for name, value in hidden_metrics.items()},
                **{f"joint_{name}": value for name, value in joint_metrics.items()},
                "joint_vs_output_usable_bits": conditional_bits,
            }
        )
        fold_diagnostics[f"layer_{int(layer)}_hidden"] = result.fold_diagnostics
        fold_diagnostics[f"layer_{int(layer)}_joint"] = joint_result.fold_diagnostics

    primary_position = int(np.flatnonzero(artifact.layers == cfg.primary_layer)[0])
    primary_prediction = layer_predictions[:, primary_position]
    primary_joint_prediction = joint_predictions[:, primary_position]
    nuisance_metrics = belief_prediction_metrics(
        target, nuisance_result.predictions, artifact.feasible_mask, groups
    )
    output_metrics = belief_prediction_metrics(
        target, output_result.predictions, artifact.feasible_mask, groups
    )
    primary_metrics = belief_prediction_metrics(
        target, primary_prediction, artifact.feasible_mask, groups
    )
    primary_joint_metrics = belief_prediction_metrics(
        target, primary_joint_prediction, artifact.feasible_mask, groups
    )
    usable_nats = _soft_ce_rows(target, output_result.predictions) - _soft_ce_rows(
        target, primary_joint_prediction
    )
    usable_bits = cluster_bootstrap_mean(
        usable_nats / np.log(2.0),
        groups,
        repetitions=cfg.bootstrap,
        seed=cfg.seed + 100,
    )
    wrong_masks = build_matched_wrong_masks(artifact, seed=cfg.seed + 200)
    transitions = directional_transport_rows(
        artifact, primary_prediction, wrong_masks
    )
    direction_bootstrap = cluster_bootstrap_mean(
        transitions["operator_advantage"],
        transitions["problem_id"],
        repetitions=cfg.bootstrap,
        seed=cfg.seed + 300,
    )
    unsupported_bootstrap = cluster_bootstrap_mean(
        transitions["unsupported_advantage"],
        transitions["problem_id"],
        repetitions=cfg.bootstrap,
        seed=cfg.seed + 400,
    )
    operator_auc = _paired_operator_auc(
        transitions["true_transport_residual"],
        transitions["wrong_transport_residual"],
    )
    null_gain_gap_p95 = float(np.quantile(transitions["information_gain_gap"], 0.95))
    belief_state_decodable = bool(
        primary_metrics["support_auc"] >= 0.70
        and primary_metrics["entropy_spearman"] >= 0.50
        and usable_bits["ci_low"] > 0.0
    )
    direction_supported = bool(
        operator_auc >= 0.60
        and direction_bootstrap["ci_low"] > 0.0
        and null_gain_gap_p95 <= cfg.max_null_information_gain_gap
    )
    report: dict[str, Any] = {
        "schema": "constraint_supported_belief_transport_audit_v1",
        "trace": str(Path(trace_path)),
        "rows": int(artifact.n_rows),
        "problems": int(len(np.unique(groups))),
        "hypotheses": int(target.shape[1]),
        "layers": artifact.layers.tolist(),
        "primary_layer": int(cfg.primary_layer),
        "config": {
            "audit": {
                "primary_layer": cfg.primary_layer,
                "bootstrap": cfg.bootstrap,
                "seed": cfg.seed,
                "max_null_information_gain_gap": cfg.max_null_information_gain_gap,
            },
            "decoder": asdict(cfg.decoder),
        },
        "belief_metrics": {
            "nuisance": nuisance_metrics,
            "output": output_metrics,
            "primary_hidden": primary_metrics,
            "primary_joint": primary_joint_metrics,
            "by_layer": layer_metric_rows,
        },
        "joint_over_output_usable_bits": usable_bits,
        "directionality": {
            "operator_auc": operator_auc,
            "operator_advantage": direction_bootstrap,
            "unsupported_contraction_advantage": unsupported_bootstrap,
            "null_information_gain_gap_mean": float(
                np.mean(transitions["information_gain_gap"])
            ),
            "null_information_gain_gap_p95": null_gain_gap_p95,
        },
        "decision_gate": {
            "belief_state_decodable": belief_state_decodable,
            "constraint_direction_supported": direction_supported,
            "ready_for_processbench_transfer": bool(
                belief_state_decodable and direction_supported
            ),
        },
    }
    _write_layer_metrics(output / "layer_metrics.csv", layer_metric_rows)
    _write_transition_rows(output / "transition_rows.csv", transitions)
    (output / "fold_diagnostics.json").write_text(
        json.dumps(fold_diagnostics, indent=2), encoding="utf-8"
    )
    (output / "summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    (output / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    with (output / "oof_predictions.npz").open("wb") as handle:
        np.savez(
            handle,
            problem_ids=artifact.problem_ids,
            prefix_index=artifact.prefix_index,
            layers=artifact.layers,
            target_belief=target,
            nuisance_predictions=nuisance_result.predictions,
            output_predictions=output_result.predictions,
            layer_predictions=layer_predictions,
            joint_predictions=joint_predictions,
            fold_ids=nuisance_result.fold_ids,
            wrong_condition_masks=wrong_masks,
        )
    return report
