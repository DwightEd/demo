from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import json
from pathlib import Path
from typing import Any, Hashable, Mapping

import numpy as np

from ..evaluate import auprc, auroc, finite_json
from ..flow_signature_audit import FlowAuditConfig, crossfit_residualize_score
from ..ocgpi.metrics import ranked_feature_importance, summarize_binary_increment
from ..ocgpi.models import CrossFitConfig, crossfit_binary_increment
from .features import PullbackFeatureCollection, build_pullback_features
from .schema import CausalPullbackArtifact


@dataclass(frozen=True)
class CausalPullbackAuditConfig:
    phase_grid: int = 4
    bootstrap: int = 2000
    min_coverage: float = 0.80
    min_contrastive_problems: int = 100
    max_finite_difference_error: float = 0.50
    max_acausal_fisher_leakage: float = 1e-5
    replay_cosine_threshold: float = 0.98
    random_seed: int = 17
    crossfit: CrossFitConfig = CrossFitConfig()

    def validate(self) -> None:
        if self.phase_grid < 2:
            raise ValueError("phase_grid must be at least two")
        if self.bootstrap < 10:
            raise ValueError("bootstrap must be at least ten")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must lie in (0, 1]")
        if self.min_contrastive_problems < 2:
            raise ValueError("min_contrastive_problems must be at least two")
        if self.max_finite_difference_error <= 0.0:
            raise ValueError("max_finite_difference_error must be positive")
        if self.max_acausal_fisher_leakage < 0.0:
            raise ValueError("max_acausal_fisher_leakage cannot be negative")
        self.crossfit.validate()


def _pair_auc(error: np.ndarray, correct: np.ndarray) -> float:
    difference = error[:, None] - correct[None, :]
    return float(np.mean((difference > 0.0) + 0.5 * (difference == 0.0)))


def _problem_auc_values(
    score: np.ndarray,
    labels: np.ndarray,
    problem_ids: np.ndarray,
) -> dict[Hashable, float]:
    output: dict[Hashable, float] = {}
    for problem in np.unique(problem_ids):
        index = np.where(problem_ids == problem)[0]
        local = np.asarray(score[index], dtype=np.float64)
        finite = np.isfinite(local)
        error = local[finite & (labels[index] == 1)]
        correct = local[finite & (labels[index] == 0)]
        if error.size and correct.size:
            key = problem.item() if isinstance(problem, np.generic) else problem
            output[key] = _pair_auc(error, correct)
    return output


def _bootstrap_problem_mean(
    values: Mapping[Hashable, float],
    *,
    draws: int,
    seed: int,
) -> tuple[float, list[float]]:
    array = np.asarray(list(values.values()), dtype=np.float64)
    point = float(np.mean(array)) if array.size else float("nan")
    if array.size < 2:
        return point, [float("nan"), float("nan")]
    rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
    samples = np.asarray(
        [np.mean(rng.choice(array, size=array.size, replace=True)) for _ in range(draws)]
    )
    return point, [float(value) for value in np.percentile(samples, [2.5, 97.5])]


def _bootstrap_auc_delta(
    first: Mapping[Hashable, float],
    second: Mapping[Hashable, float],
    *,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    keys = sorted(set(first) & set(second), key=str)
    difference = np.asarray([first[key] - second[key] for key in keys], dtype=np.float64)
    point = float(np.mean(difference)) if difference.size else float("nan")
    if difference.size < 2:
        ci = [float("nan"), float("nan")]
    else:
        rng = np.random.default_rng(int(seed) & 0xFFFFFFFF)
        samples = np.asarray(
            [
                np.mean(rng.choice(difference, size=difference.size, replace=True))
                for _ in range(draws)
            ]
        )
        ci = [float(value) for value in np.percentile(samples, [2.5, 97.5])]
    return {"delta": point, "ci95": ci, "problems": len(keys)}


def _score_summary(
    name: str,
    score: np.ndarray,
    labels: np.ndarray,
    problem_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
) -> tuple[dict[str, Any], dict[Hashable, float]]:
    values = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(values)
    problem_auc = _problem_auc_values(values, labels, problem_ids)
    point, ci = _bootstrap_problem_mean(
        problem_auc,
        draws=draws,
        seed=seed + sum(ord(char) for char in name),
    )
    return (
        {
            "name": name,
            "n": int(finite.sum()),
            "coverage": float(finite.mean()),
            "errors": int(np.sum(finite & (labels == 1))),
            "pooled_auroc": auroc(labels, values),
            "pooled_auprc": auprc(labels, values),
            "within_problem_auroc": point,
            "within_problem_ci95": ci,
            "contrastive_problems": len(problem_auc),
        },
        problem_auc,
    )


def _identity_matches(source: str, observer: str) -> bool:
    source = str(source).strip().replace("\\", "/").rstrip("/").lower()
    observer = str(observer).strip().replace("\\", "/").rstrip("/").lower()
    if not source:
        return False
    return source == observer or source.rsplit("/", 1)[-1] == observer.rsplit("/", 1)[-1]


def _run_increment(
    features: PullbackFeatureCollection,
    index: np.ndarray,
    geometry: np.ndarray,
    geometry_names: tuple[str, ...],
    geometry_groups: tuple[str, ...],
    cfg: CausalPullbackAuditConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    # Length and replay diagnostics are baseline controls. They never enter
    # only the geometry branch, so an apparent increment cannot come from
    # withholding an obvious nuisance from the output-only model.
    x_output = np.concatenate(
        [features.nuisance[index], features.x_output[index]], axis=1
    )
    result = crossfit_binary_increment(
        x_output,
        geometry[index],
        features.y_error[index],
        features.problem_ids[index],
        features.nuisance[index],
        cfg.crossfit,
    )
    summary = summarize_binary_increment(
        result,
        n_boot=cfg.bootstrap,
        seed=cfg.random_seed,
    )
    summary["feature_importance"] = ranked_feature_importance(
        geometry_names,
        geometry_groups,
        result.feature_importance,
        limit=30,
    )
    predictions = {
        "chain_idx": features.chain_idx[index],
        "problem_ids": features.problem_ids[index],
        "y_error": features.y_error[index],
        "output_only": result.base_probability,
        "controls_plus_mechanism": result.geometry_probability,
        "output_plus_mechanism": result.full_probability,
        "length_matched_null": result.null_probability,
    }
    return summary, predictions


def _write_report(
    report: dict[str, Any],
    score_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(finite_json(report), handle, indent=2, ensure_ascii=False)
    if score_rows:
        with (output_dir / "direct_scores.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(score_rows[0]))
            writer.writeheader()
            writer.writerows(finite_json(score_rows))

    validation = report["validation"]
    lines = [
        "# Causal Pullback Flow Field Audit",
        "",
        "## Research Question",
        "",
        "Do geometry-derived directions carry abnormal downstream output sensitivity, and ",
        "does the resulting causal operator add information beyond output uncertainty and length?",
        "",
        "## Preflight",
        "",
        f"- Responses: `{report['preflight']['responses']}`",
        f"- Errors: `{report['preflight']['errors']}`",
        f"- Contrastive problems: `{report['preflight']['contrastive_problems']}`",
        f"- Valid coverage: `{report['preflight']['valid_coverage']:.4f}`",
        f"- Evidence tier: `{report['preflight']['evidence_tier']}`",
        "",
        "## Direct Same-Problem Diagnosis",
        "",
        "| score | within AUROC | CI95 | problems |",
        "|---|---:|---|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['name']} | {row['within_problem_auroc']:.4f} | "
            f"{row['within_problem_ci95']} | {row['contrastive_problems']} |"
        )
    lines.extend(
        [
            "",
            "## Conditional Increment",
            "",
            "| ablation | output AUROC | + mechanism AUROC | usable bits | delta AUROC |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, summary in report["conditional_increment"].items():
        increment = summary["increment"]
        lines.append(
            f"| {name} | {summary['output_only']['auroc']:.4f} | "
            f"{summary['output_plus_geometry']['auroc']:.4f} | "
            f"{increment['conditional_usable_information']['point_bits']:.5f} | "
            f"{increment['delta_auroc']['point']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gates",
            "",
            f"- Numerical validity: `{validation['numerical_validity']['pass']}`",
            f"- Mechanism supported: `{validation['mechanism_supported']['pass']}`",
            f"- Detector increment supported: `{validation['detector_increment_supported']['pass']}`",
            f"- Confirmatory ready: `{validation['confirmatory_ready']}`",
            "",
            "A failure is a falsification of the current witness/operator construction, not evidence ",
            "that hidden representations contain no useful information.",
        ]
    )
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_causal_pullback_audit(
    artifact_path: str | Path,
    output_dir: str | Path,
    cfg: CausalPullbackAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    output_dir = Path(output_dir)
    artifact = CausalPullbackArtifact.load(artifact_path)
    features = build_pullback_features(
        artifact,
        phase_grid=cfg.phase_grid,
        replay_cosine_threshold=cfg.replay_cosine_threshold,
    )
    finite_difference = features.direct_scores["finite_difference_relative_error"]
    valid = features.valid & (
        finite_difference <= float(cfg.max_finite_difference_error)
    )
    index = np.where(valid)[0]
    if index.size < 20:
        raise ValueError(
            f"only {index.size} responses pass replay and finite-difference controls"
        )
    labels = features.y_error[index]
    problems = features.problem_ids[index]
    contrastive = len(_problem_auc_values(np.zeros(len(index)), labels, problems))

    controls = features.nuisance[index, :2]
    residual_cfg = FlowAuditConfig(
        folds=cfg.crossfit.outer_folds,
        bootstrap=0,
        permutations=0,
        seed=cfg.random_seed,
        compute_device="cpu",
    )
    direct_values: dict[str, np.ndarray] = {}
    for name, score in features.direct_scores.items():
        local = np.asarray(score[index], dtype=np.float64)
        direct_values[name] = local
        if name != "finite_difference_relative_error":
            direct_values[f"{name}.length_residual"] = crossfit_residualize_score(
                local, controls, problems, residual_cfg
            )

    score_rows: list[dict[str, Any]] = []
    auc_by_name: dict[str, dict[Hashable, float]] = {}
    for name, score in direct_values.items():
        row, problem_auc = _score_summary(
            name,
            score,
            labels,
            problems,
            draws=cfg.bootstrap,
            seed=cfg.random_seed,
        )
        score_rows.append(row)
        auc_by_name[name] = problem_auc
    score_rows.sort(
        key=lambda row: abs(float(row["within_problem_auroc"]) - 0.5),
        reverse=True,
    )

    field_groups = tuple("field_geometry" for _ in features.field_names)
    increment: dict[str, Any] = {}
    prediction_payload: dict[str, np.ndarray] = {}
    for name, geometry, names, groups in (
        (
            "field_only",
            features.x_field,
            features.field_names,
            field_groups,
        ),
        (
            "causal_pullback_only",
            features.x_pullback,
            features.pullback_names,
            features.pullback_groups,
        ),
        (
            "field_plus_causal_pullback",
            np.concatenate([features.x_field, features.x_pullback], axis=1),
            features.field_names + features.pullback_names,
            field_groups + features.pullback_groups,
        ),
    ):
        summary, predictions = _run_increment(
            features,
            index,
            geometry,
            names,
            groups,
            cfg,
        )
        increment[name] = summary
        for key, value in predictions.items():
            prediction_payload[f"{name}.{key}"] = value

    primary = "field_consequential_mean.length_residual"
    shuffle = "shuffle_consequential_mean.length_residual"
    random = "random_consequential_mean.length_residual"
    primary_row = next(row for row in score_rows if row["name"] == primary)
    primary_vs_shuffle = _bootstrap_auc_delta(
        auc_by_name[primary],
        auc_by_name[shuffle],
        draws=cfg.bootstrap,
        seed=cfg.random_seed + 101,
    )
    primary_vs_random = _bootstrap_auc_delta(
        auc_by_name[primary],
        auc_by_name[random],
        draws=cfg.bootstrap,
        seed=cfg.random_seed + 103,
    )

    median_replay = np.asarray(
        [np.nanmedian(item.replay_cosine) for item in artifact.items], dtype=np.float64
    )
    acausal = np.asarray(
        [
            float(item.metadata.get("maximum_acausal_fisher_leakage", np.nan))
            for item in artifact.items
        ],
        dtype=np.float64,
    )
    numerical_conditions = {
        "valid_coverage_at_least_threshold": float(valid.mean()) >= cfg.min_coverage,
        "median_replay_cosine_at_least_threshold": float(np.nanmedian(median_replay))
        >= cfg.replay_cosine_threshold,
        "median_finite_difference_error_at_most_threshold": float(
            np.nanmedian(finite_difference)
        )
        <= cfg.max_finite_difference_error,
        "maximum_acausal_fisher_leakage_at_most_threshold": float(
            np.nanmax(acausal)
        )
        <= cfg.max_acausal_fisher_leakage,
        "contrastive_problem_count_at_least_minimum": contrastive
        >= cfg.min_contrastive_problems,
    }
    mechanism_conditions = {
        "primary_length_residual_auc_ci_above_chance": primary_row[
            "within_problem_ci95"
        ][0]
        > 0.5,
        "field_beats_time_shuffle_ci_above_zero": primary_vs_shuffle["ci95"][0]
        > 0.0,
        "field_beats_random_tangent_ci_above_zero": primary_vs_random["ci95"][0]
        > 0.0,
    }
    combined = increment["field_plus_causal_pullback"]
    detector_conditions = {
        "conditional_usable_information_ci_above_zero": combined["increment"][
            "conditional_usable_information"
        ]["ci_low_bits"]
        > 0.0,
        "conditional_auroc_gain_ci_above_zero": combined["increment"][
            "delta_auroc"
        ]["ci_low"]
        > 0.0,
        "beats_length_matched_null_ci_above_zero": combined["increment"][
            "delta_auroc_vs_null"
        ]["ci_low"]
        > 0.0,
    }
    evidence_tier = str(artifact.metadata.get("evidence_tier", "unknown"))
    source_model = str(artifact.metadata.get("source_model", ""))
    observer_model = str(artifact.metadata.get("observer_model", ""))
    model_match = _identity_matches(source_model, observer_model)
    numerical_pass = all(numerical_conditions.values())
    mechanism_pass = numerical_pass and all(mechanism_conditions.values())
    detector_pass = numerical_pass and all(detector_conditions.values())
    confirmatory = bool(
        mechanism_pass
        and detector_pass
        and evidence_tier == "exact_trace_candidate"
        and model_match
    )

    report: dict[str, Any] = {
        "method": "Causal Pullback Flow Field",
        "schema_version": "causal_pullback_audit_v1",
        "research_hypothesis": (
            "Errors need not leave the feasible hidden-state field. They may occupy plausible "
            "geometry whose field-normal directions are abnormally amplified by the downstream "
            "categorical output map."
        ),
        "operator": (
            "central-KL estimate of w_t^T J_{s<-t}^T F(p_s) J_{s<-t} w_t, "
            "for strictly future output steps"
        ),
        "config": {**asdict(cfg), "crossfit": asdict(cfg.crossfit)},
        "preflight": {
            "responses": len(artifact.items),
            "errors": int(features.y_error.sum()),
            "valid_responses": int(valid.sum()),
            "valid_coverage": float(valid.mean()),
            "problems": int(np.unique(features.problem_ids).size),
            "contrastive_problems": contrastive,
            "median_replay_cosine": float(np.nanmedian(median_replay)),
            "median_finite_difference_error": float(
                np.nanmedian(finite_difference)
            ),
            "maximum_acausal_fisher_leakage": float(np.nanmax(acausal)),
            "evidence_tier": evidence_tier,
            "source_model": source_model,
            "observer_model": observer_model,
            "observer_model_identity_matches": model_match,
            "skipped_extraction_rows": len(artifact.skipped),
        },
        "direct_same_problem_diagnosis": {
            "primary_score": primary,
            "scores": score_rows,
            "primary_minus_time_shuffle_auc": primary_vs_shuffle,
            "primary_minus_random_tangent_auc": primary_vs_random,
        },
        "conditional_increment": increment,
        "validation": {
            "numerical_validity": {
                "pass": numerical_pass,
                "conditions": numerical_conditions,
            },
            "mechanism_supported": {
                "pass": mechanism_pass,
                "conditions": mechanism_conditions,
            },
            "detector_increment_supported": {
                "pass": detector_pass,
                "conditions": detector_conditions,
            },
            "confirmatory_ready": confirmatory,
            "status": (
                "confirmatory_support"
                if confirmatory
                else "exploratory_support"
                if mechanism_pass or detector_pass
                else "current_causal_pullback_hypothesis_not_supported"
            ),
        },
        "claim_boundary": (
            "The same-problem correct ensemble is an offline reference teacher. A passing result "
            "does not yet establish a single-trajectory real-time detector."
        ),
        "artifact_metadata": artifact.metadata,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_dir / "oof_predictions.npz", **prediction_payload)
    _write_report(report, score_rows, output_dir)
    return report
