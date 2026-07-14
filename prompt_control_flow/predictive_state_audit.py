from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .directional_consensus import (
    DirectionalConsensusConfig,
    _bh_qvalues,
    _bootstrap_auc_delta,
    _component_auc,
    _component_bootstrap,
    _length_matched_components,
    _problem_pair_components,
    _safe_spearman,
    _same_problem_permutation_p,
    compute_directional_consensus,
)
from .evaluate import finite_json
from .flow_signature_audit import FlowAuditConfig, crossfit_residualize_score, evaluate_score
from .predictive_state_data import (
    PredictiveStateDataset,
    ProjectionConfig,
    WindowConfig,
    build_transition_bundle,
    build_window_observations,
    project_token_clouds,
)
from .predictive_state_model import (
    PredictiveModelConfig,
    aggregate_transition_scores,
    average_horizon_scores,
    fit_reduced_rank_gaussian,
    fit_token_bigram,
    fit_token_nuisance_transform,
    permute_transition_targets,
    transform_projected_sequences,
)


@dataclass(frozen=True)
class PredictiveStateAuditConfig:
    folds: int = 5
    horizons: tuple[int, ...] = (1, 2)
    context_windows: int = 1
    min_token_count: int = 4
    bootstrap: int = 1000
    permutations: int = 500
    length_match_ratio: float = 1.25
    seed: int = 13
    verbose: bool = True

    def validate(self) -> None:
        if self.folds < 2:
            raise ValueError("folds must be at least 2")
        if not self.horizons or any(int(value) <= 0 for value in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")
        if self.context_windows <= 0:
            raise ValueError("context_windows must be positive")
        if self.min_token_count <= 0:
            raise ValueError("min_token_count must be positive")
        if self.bootstrap < 0 or self.permutations < 0:
            raise ValueError("bootstrap and permutations must be non-negative")
        if self.length_match_ratio < 1.0:
            raise ValueError("length_match_ratio must be at least 1")


def _group_folds(
    problem_ids: np.ndarray,
    folds: int,
    seed: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(problem_ids)
    if unique.size < folds:
        raise ValueError(f"{folds} folds requested but only {unique.size} problems are available")
    rng = np.random.default_rng(int(seed))
    shuffled = unique.copy()
    rng.shuffle(shuffled)
    parts = np.array_split(shuffled, folds)
    result = []
    all_indices = np.arange(problem_ids.size, dtype=np.int64)
    for part in parts:
        test = all_indices[np.isin(problem_ids, part)]
        train = all_indices[~np.isin(problem_ids, part)]
        result.append((train, test))
    return result


def _empty_scores(n_samples: int) -> dict[str, np.ndarray]:
    names = [
        "predictive.raw.mahalanobis_mean",
        "predictive.raw.euclidean_mean",
        "predictive.token_residual.mahalanobis_mean",
        "predictive.token_residual.euclidean_mean",
        "predictive.token_residual.transverse_mean",
        "predictive.token_residual.parallel_mean",
        "predictive.token_residual.offchart_mean",
        "static.token_residual.mahalanobis_mean",
        "null.shuffle.mahalanobis_mean",
        "null.same_problem_mismatch.mahalanobis_mean",
        "control.token_bigram_nll",
    ]
    return {name: np.full(n_samples, np.nan, dtype=np.float64) for name in names}


def _assign_test_scores(
    destination: dict[str, np.ndarray],
    test_indices: np.ndarray,
    channel: str,
    ordered_by_horizon: Mapping[str, Sequence[np.ndarray]],
    null_by_horizon: Mapping[str, Sequence[np.ndarray]],
) -> None:
    ordered = {
        name: average_horizon_scores(values)
        for name, values in ordered_by_horizon.items()
    }
    if channel == "raw":
        destination["predictive.raw.mahalanobis_mean"][test_indices] = ordered[
            "mahalanobis"
        ][test_indices]
        destination["predictive.raw.euclidean_mean"][test_indices] = ordered[
            "euclidean"
        ][test_indices]
        return
    mapping = {
        "predictive.token_residual.mahalanobis_mean": "mahalanobis",
        "predictive.token_residual.euclidean_mean": "euclidean",
        "predictive.token_residual.transverse_mean": "transverse",
        "predictive.token_residual.parallel_mean": "parallel",
        "predictive.token_residual.offchart_mean": "offchart",
        "static.token_residual.mahalanobis_mean": "static_mahalanobis",
    }
    for destination_name, source_name in mapping.items():
        destination[destination_name][test_indices] = ordered[source_name][test_indices]
    for destination_name, source_name in (
        ("null.shuffle.mahalanobis_mean", "shuffle"),
        ("null.same_problem_mismatch.mahalanobis_mean", "mismatch"),
    ):
        values = average_horizon_scores(null_by_horizon[source_name])
        destination[destination_name][test_indices] = values[test_indices]


def _correct_problem_difference(
    first: np.ndarray,
    second: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    *,
    draws: int,
    seed: int,
    compute_device: str,
) -> dict[str, Any]:
    """Bootstrap mean ``first - second`` over held-out correct responses."""

    effects: list[float] = []
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    for problem in np.unique(problem_ids):
        keep = (
            (problem_ids == problem)
            & (y_error == 0)
            & np.isfinite(first)
            & np.isfinite(second)
        )
        if np.any(keep):
            effects.append(float(np.mean(first[keep] - second[keep])))
    point = float(np.mean(effects)) if effects else float("nan")
    if draws <= 0 or len(effects) < 2:
        return {"point": point, "ci95": [float("nan"), float("nan")], "problems": len(effects)}
    device = torch.device(compute_device)
    values = torch.as_tensor(effects, device=device, dtype=torch.float64)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    selected = torch.randint(
        0,
        values.numel(),
        (draws, values.numel()),
        device=device,
        generator=generator,
    )
    estimates = values[selected].mean(dim=1)
    interval = torch.quantile(
        estimates,
        torch.as_tensor([0.025, 0.975], device=device, dtype=torch.float64),
    )
    return {
        "point": point,
        "ci95": [float(value) for value in interval.detach().cpu().tolist()],
        "problems": len(effects),
    }


def _fit_fold_channel(
    dataset: PredictiveStateDataset,
    projected: Sequence[np.ndarray],
    train_correct: np.ndarray,
    test_indices: np.ndarray,
    *,
    token_residual: bool,
    window_cfg: WindowConfig,
    model_cfg: PredictiveModelConfig,
    audit_cfg: PredictiveStateAuditConfig,
    fold: int,
) -> tuple[
    dict[str, list[np.ndarray]],
    dict[str, list[np.ndarray]],
    np.ndarray,
    list[dict[str, Any]],
]:
    transform = fit_token_nuisance_transform(
        projected,
        dataset.token_ids,
        train_correct,
        token_residual=token_residual,
        min_token_count=audit_cfg.min_token_count,
        compute_device=window_cfg.compute_device,
    )
    transformed = transform_projected_sequences(
        projected,
        dataset.token_ids,
        transform,
        compute_device=window_cfg.compute_device,
    )
    windows, window_ranges, window_counts = build_window_observations(
        transformed,
        dataset.token_positions,
        window_cfg,
    )
    ordered_scores: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "mahalanobis",
            "euclidean",
            "parallel",
            "transverse",
            "static_mahalanobis",
            "offchart",
        )
    }
    null_scores: dict[str, list[np.ndarray]] = {"shuffle": [], "mismatch": []}
    diagnostics: list[dict[str, Any]] = []
    for horizon in audit_cfg.horizons:
        train_bundle = build_transition_bundle(
            windows,
            window_ranges,
            train_correct,
            dataset.cloud.base.problem_ids,
            horizon=int(horizon),
            context_windows=audit_cfg.context_windows,
            max_transition_gap=window_cfg.max_skipped_tokens,
        )
        test_bundle = build_transition_bundle(
            windows,
            window_ranges,
            test_indices,
            dataset.cloud.base.problem_ids,
            horizon=int(horizon),
            context_windows=audit_cfg.context_windows,
            max_transition_gap=window_cfg.max_skipped_tokens,
        )
        if train_bundle.n_rows == 0 or test_bundle.n_rows == 0:
            raise ValueError(
                f"fold {fold}, horizon {horizon}: no train/test transitions; reduce window size"
            )
        ordered_model = fit_reduced_rank_gaussian(
            train_bundle,
            model_cfg,
            compute_device=window_cfg.compute_device,
        )
        ordered_transition = ordered_model.score(
            test_bundle.x,
            test_bundle.y,
            device=window_cfg.compute_device,
        )
        for name, values in ordered_transition.items():
            ordered_scores[name].append(
                aggregate_transition_scores(
                    values,
                    test_bundle,
                    n_samples=dataset.n_samples,
                    compute_device=window_cfg.compute_device,
                )
            )
        if token_residual:
            for mode, output_name in (
                ("within_response", "shuffle"),
                ("same_problem_mismatch", "mismatch"),
            ):
                permuted = permute_transition_targets(
                    train_bundle,
                    mode=mode,
                    seed=audit_cfg.seed + fold * 101 + int(horizon) * 17,
                )
                null_model = fit_reduced_rank_gaussian(
                    train_bundle,
                    model_cfg,
                    compute_device=window_cfg.compute_device,
                    fit_targets=permuted,
                    fixed_chart=ordered_model.chart,
                    fixed_x_mean=ordered_model.x_mean,
                    fixed_y_mean=ordered_model.y_mean,
                )
                transition_score = null_model.score(
                    test_bundle.x,
                    test_bundle.y,
                    device=window_cfg.compute_device,
                )["mahalanobis"]
                null_scores[output_name].append(
                    aggregate_transition_scores(
                        transition_score,
                        test_bundle,
                        n_samples=dataset.n_samples,
                        compute_device=window_cfg.compute_device,
                    )
                )
        diagnostics.append(
            {
                "fold": int(fold),
                "channel": "token_residual" if token_residual else "raw",
                "horizon": int(horizon),
                "train_correct_responses": int(train_correct.size),
                "train_transitions": int(train_bundle.n_rows),
                "test_transitions": int(test_bundle.n_rows),
                "latent_dim": int(ordered_model.latent_dim),
                "tangent_rank": int(ordered_model.tangent_rank),
                "observed_token_types": int(transform.observed_token_count),
            }
        )
    return ordered_scores, null_scores, window_counts, diagnostics


def run_predictive_state_audit(
    dataset: PredictiveStateDataset,
    projection_cfg: ProjectionConfig,
    window_cfg: WindowConfig,
    model_cfg: PredictiveModelConfig,
    audit_cfg: PredictiveStateAuditConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Run leakage-safe correct-only predictive-state validation."""

    audit_cfg.validate()
    projection_cfg.validate()
    window_cfg.validate()
    model_cfg.validate()
    base = dataset.cloud.base
    projected, projection_meta = project_token_clouds(dataset, projection_cfg)
    n_samples = dataset.n_samples
    score_map = _empty_scores(n_samples)
    oof_window_counts = np.full(n_samples, -1, dtype=np.int64)
    fold_rows: list[dict[str, Any]] = []

    for fold, (train_indices, test_indices) in enumerate(
        _group_folds(base.problem_ids, audit_cfg.folds, audit_cfg.seed)
    ):
        train_correct = train_indices[base.y_error[train_indices] == 0]
        if train_correct.size < max(8, model_cfg.latent_dim + 1):
            raise ValueError(
                f"fold {fold}: only {train_correct.size} correct training responses"
            )
        if audit_cfg.verbose:
            print(
                f"fold {fold + 1}/{audit_cfg.folds}: train_correct={train_correct.size} "
                f"test={test_indices.size}"
            )
        bigram = fit_token_bigram(
            dataset.token_ids,
            dataset.token_positions,
            train_correct,
        )
        for sample in test_indices.tolist():
            score_map["control.token_bigram_nll"][sample] = bigram.score(
                dataset.token_ids[sample], dataset.token_positions[sample]
            )
        for channel, token_residual in (("raw", False), ("token_residual", True)):
            ordered, null, window_counts, diagnostics = _fit_fold_channel(
                dataset,
                projected,
                train_correct,
                test_indices,
                token_residual=token_residual,
                window_cfg=window_cfg,
                model_cfg=model_cfg,
                audit_cfg=audit_cfg,
                fold=fold,
            )
            _assign_test_scores(score_map, test_indices, channel, ordered, null)
            if token_residual:
                oof_window_counts[test_indices] = window_counts[test_indices]
            fold_rows.extend(diagnostics)
        if audit_cfg.verbose:
            finite = np.isfinite(
                score_map["predictive.token_residual.mahalanobis_mean"][test_indices]
            )
            print(
                f"  fold {fold + 1}: scored={int(np.sum(finite))}/{test_indices.size} "
                f"median_windows={float(np.median(oof_window_counts[test_indices])):.1f}"
            )

    if np.any(oof_window_counts < 0):
        raise RuntimeError("some OOF samples never received a window count")
    consensus = compute_directional_consensus(
        dataset.cloud,
        DirectionalConsensusConfig(
            fixed_window_tokens=window_cfg.window_tokens,
            batch_size=projection_cfg.batch_size,
            max_batch_tokens=projection_cfg.max_batch_tokens,
            compute_device=projection_cfg.compute_device,
        ),
    )
    score_map["baseline.fixed_window_consensus"] = consensus.chain_scores[
        "consensus.fixed_window_dispersion.mean"
    ]
    controls = np.column_stack(
        [
            np.log1p(base.n_steps.astype(np.float64)),
            np.log1p(base.response_chars.astype(np.float64)),
            np.log1p(dataset.cloud.response_tokens.astype(np.float64)),
            np.log1p(oof_window_counts.astype(np.float64)),
        ]
    )
    control_names = (
        "control.log1p_n_steps",
        "control.log1p_response_chars",
        "control.log1p_response_tokens",
        "control.log1p_valid_windows",
    )
    for index, name in enumerate(control_names):
        score_map[name] = controls[:, index]

    statistical_cfg = FlowAuditConfig(
        folds=audit_cfg.folds,
        bootstrap=0,
        permutations=0,
        seed=audit_cfg.seed,
        compute_device=projection_cfg.compute_device,
    )
    residualize = [
        "predictive.raw.mahalanobis_mean",
        "predictive.token_residual.mahalanobis_mean",
        "static.token_residual.mahalanobis_mean",
        "null.shuffle.mahalanobis_mean",
        "null.same_problem_mismatch.mahalanobis_mean",
        "control.token_bigram_nll",
        "baseline.fixed_window_consensus",
    ]
    for name in residualize:
        score_map[f"{name}.length_residual"] = crossfit_residualize_score(
            score_map[name], controls, base.problem_ids, statistical_cfg
        )

    confirmatory = [
        "predictive.token_residual.mahalanobis_mean",
        "predictive.token_residual.mahalanobis_mean.length_residual",
        "null.shuffle.mahalanobis_mean",
        "static.token_residual.mahalanobis_mean",
        "baseline.fixed_window_consensus",
    ]
    rows: list[dict[str, Any]] = []
    for name, score in score_map.items():
        row = evaluate_score(name, score, base, statistical_cfg)
        components = _problem_pair_components(score, base.y_error, base.problem_ids)
        row["same_problem_ci95"] = _component_bootstrap(
            components,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + sum(ord(char) for char in name),
            compute_device=projection_cfg.compute_device,
        )
        row["same_problem_permutation_p"] = (
            _same_problem_permutation_p(
                score,
                base.y_error,
                base.problem_ids,
                row["same_problem_auroc"],
                permutations=audit_cfg.permutations,
                seed=audit_cfg.seed + 101 + sum(ord(char) for char in name),
                compute_device=projection_cfg.compute_device,
            )
            if name in confirmatory
            else float("nan")
        )
        matched = _length_matched_components(
            score,
            base.y_error,
            base.problem_ids,
            dataset.cloud.response_tokens,
            audit_cfg.length_match_ratio,
        )
        row["token_length_matched_auroc"], row["token_length_matched_pairs"] = _component_auc(
            matched
        )
        row["token_length_matched_ci95"] = _component_bootstrap(
            matched,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 1701 + sum(ord(char) for char in name),
            compute_device=projection_cfg.compute_device,
        )
        row["spearman_response_tokens"] = _safe_spearman(
            score, dataset.cloud.response_tokens
        )
        row["confirmatory"] = bool(name in confirmatory)
        rows.append(row)
    row_by_name = {row["name"]: row for row in rows}
    q_values = _bh_qvalues(
        [row_by_name[name]["same_problem_permutation_p"] for name in confirmatory]
    )
    for row in rows:
        row["same_problem_bh_q"] = float("nan")
    for name, q_value in zip(confirmatory, q_values):
        row_by_name[name]["same_problem_bh_q"] = float(q_value)

    primary_name = "predictive.token_residual.mahalanobis_mean"
    primary_residual_name = f"{primary_name}.length_residual"
    deltas = {
        "token_residual_minus_raw": _bootstrap_auc_delta(
            score_map[primary_name],
            score_map["predictive.raw.mahalanobis_mean"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 301,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_shuffle": _bootstrap_auc_delta(
            score_map[primary_name],
            score_map["null.shuffle.mahalanobis_mean"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 401,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_shuffle_length_residual": _bootstrap_auc_delta(
            score_map[primary_residual_name],
            score_map["null.shuffle.mahalanobis_mean.length_residual"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 451,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_same_problem_mismatch": _bootstrap_auc_delta(
            score_map[primary_name],
            score_map["null.same_problem_mismatch.mahalanobis_mean"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 501,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_same_problem_mismatch_length_residual": _bootstrap_auc_delta(
            score_map[primary_residual_name],
            score_map["null.same_problem_mismatch.mahalanobis_mean.length_residual"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 551,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_static": _bootstrap_auc_delta(
            score_map[primary_name],
            score_map["static.token_residual.mahalanobis_mean"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 601,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_static_length_residual": _bootstrap_auc_delta(
            score_map[primary_residual_name],
            score_map["static.token_residual.mahalanobis_mean.length_residual"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 651,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_fixed_consensus": _bootstrap_auc_delta(
            score_map[primary_name],
            score_map["baseline.fixed_window_consensus"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 701,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_fixed_consensus_length_residual": _bootstrap_auc_delta(
            score_map[primary_residual_name],
            score_map["baseline.fixed_window_consensus.length_residual"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 801,
            compute_device=projection_cfg.compute_device,
        ),
        "ordered_minus_token_bigram_length_residual": _bootstrap_auc_delta(
            score_map[primary_residual_name],
            score_map["control.token_bigram_nll.length_residual"],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 851,
            compute_device=projection_cfg.compute_device,
        ),
    }
    correct_predictive_advantage = {
        "shuffle_minus_ordered": _correct_problem_difference(
            score_map["null.shuffle.mahalanobis_mean"],
            score_map[primary_name],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 901,
            compute_device=projection_cfg.compute_device,
        ),
        "same_problem_mismatch_minus_ordered": _correct_problem_difference(
            score_map["null.same_problem_mismatch.mahalanobis_mean"],
            score_map[primary_name],
            base.y_error,
            base.problem_ids,
            draws=audit_cfg.bootstrap,
            seed=audit_cfg.seed + 1001,
            compute_device=projection_cfg.compute_device,
        ),
    }
    primary = row_by_name[primary_residual_name]
    chronology_delta = deltas["ordered_minus_shuffle_length_residual"]
    mismatch_delta = deltas[
        "ordered_minus_same_problem_mismatch_length_residual"
    ]
    static_delta = deltas["ordered_minus_static_length_residual"]
    consensus_delta = deltas["ordered_minus_fixed_consensus_length_residual"]
    lexical_delta = deltas["ordered_minus_token_bigram_length_residual"]
    correct_advantage = correct_predictive_advantage["shuffle_minus_ordered"]
    mismatch_advantage = correct_predictive_advantage[
        "same_problem_mismatch_minus_ordered"
    ]
    gate_checks = {
        "length_residual_ci_above_chance": bool(
            np.isfinite(primary["same_problem_ci95"][0])
            and primary["same_problem_ci95"][0] > 0.5
        ),
        "primary_bh_q_below_005": bool(
            np.isfinite(primary["same_problem_bh_q"])
            and primary["same_problem_bh_q"] < 0.05
        ),
        "chronology_auc_gain_at_least_003": bool(
            chronology_delta["point"] >= 0.03
            and np.isfinite(chronology_delta["ci95"][0])
            and chronology_delta["ci95"][0] > 0.0
        ),
        "same_problem_mismatch_gain_at_least_003": bool(
            mismatch_delta["point"] >= 0.03
            and np.isfinite(mismatch_delta["ci95"][0])
            and mismatch_delta["ci95"][0] > 0.0
        ),
        "dynamic_beats_static_by_003": bool(
            static_delta["point"] >= 0.03
            and np.isfinite(static_delta["ci95"][0])
            and static_delta["ci95"][0] > 0.0
        ),
        "beats_fixed_consensus_by_003": bool(
            consensus_delta["point"] >= 0.03
            and np.isfinite(consensus_delta["ci95"][0])
            and consensus_delta["ci95"][0] > 0.0
        ),
        "beats_lexical_bigram": bool(
            lexical_delta["point"] > 0.0
            and np.isfinite(lexical_delta["ci95"][0])
            and lexical_delta["ci95"][0] > 0.0
        ),
        "ordered_predicts_correct_future_better": bool(
            np.isfinite(correct_advantage["ci95"][0])
            and correct_advantage["ci95"][0] > 0.0
        ),
        "ordered_beats_mismatch_on_correct_future": bool(
            np.isfinite(mismatch_advantage["ci95"][0])
            and mismatch_advantage["ci95"][0] > 0.0
        ),
        "coverage_at_least_080": bool(primary["coverage"] >= 0.8),
    }
    report = {
        "meta": {
            "source": base.source_path,
            "samples": n_samples,
            "errors": int(np.sum(base.y_error == 1)),
            "correct": int(np.sum(base.y_error == 0)),
            "problems": int(np.unique(base.problem_ids).size),
            "contrastive_problems": int(
                sum(
                    np.any(base.y_error[base.problem_ids == problem] == 0)
                    and np.any(base.y_error[base.problem_ids == problem] == 1)
                    for problem in np.unique(base.problem_ids)
                )
            ),
            "cloud_layers": dataset.cloud.cloud_layer_ids.tolist(),
            "token_range_key": dataset.token_range_key,
            "projection": projection_meta,
            "window_tokens": int(window_cfg.window_tokens),
            "window_stride": int(window_cfg.window_stride),
            "max_skipped_tokens": int(window_cfg.max_skipped_tokens),
            "horizons": list(audit_cfg.horizons),
            "context_windows": int(audit_cfg.context_windows),
            "latent_dim": int(model_cfg.latent_dim),
            "fit_policy": "problem-group OOF; correct training responses only",
            "compute_device": projection_cfg.compute_device,
        },
        "hypothesis": (
            "Correct reasoning admits a compact predictive state whose future-window "
            "innovation is smaller than that of incorrect reasoning after lexical and "
            "length nuisance control."
        ),
        "method": {
            "preconditioner": "fixed label-free Gaussian projection",
            "lexical_quotient": "training-fold correct-only token-ID conditional mean subtraction",
            "chart": "top predictable target directions of correct-only ridge dynamics",
            "risk": "shrinkage-Gaussian Mahalanobis innovation averaged equally over horizons",
            "response_weighting": "each response has unit total transition weight during fitting",
        },
        "scores": rows,
        "auc_deltas": deltas,
        "correct_predictive_advantage": correct_predictive_advantage,
        "fold_diagnostics": fold_rows,
        "decision_gate": {
            "primary_score": primary_residual_name,
            "checks": gate_checks,
            "passes": bool(all(gate_checks.values())),
            "replication_requirement": (
                "The gate must pass without changing hyperparameters on both gsm8k_v2_custom "
                "and gsm8k_v2_5shot before training a nonlinear predictive encoder."
            ),
        },
        "claim_scope": {
            "supported_if_replicated": (
                "response-level predictive-state innovation adds same-problem error "
                "discrimination beyond static geometry, shuffled chronology, lexical NLL, "
                "and fixed-window directional consensus"
            ),
            "not_supported": (
                "first-error localization, a globally low-dimensional reasoning manifold, "
                "causal error generation, model self-awareness, or output-logit sensitivity"
            ),
        },
    }
    packed = {
        "original_indices": base.original_indices,
        "problem_ids": base.problem_ids,
        "sample_idx": base.sample_idx,
        "y_error": base.y_error,
        "is_correct": base.is_correct,
        "n_steps": base.n_steps,
        "response_chars": base.response_chars,
        "response_tokens": dataset.cloud.response_tokens,
        "valid_windows": oof_window_counts,
        "cloud_layer_ids": dataset.cloud.cloud_layer_ids,
        "score_names": np.asarray(list(score_map), dtype=object),
        "scores": np.column_stack([score_map[name] for name in score_map]).astype(np.float32),
        "metadata_json": np.asarray(
            json.dumps(finite_json(report["meta"]), ensure_ascii=False), dtype=object
        ),
    }
    return report, packed


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key not in columns and not isinstance(value, (list, dict)):
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    meta = report["meta"]
    gate = report["decision_gate"]
    lines = [
        "# Predictive State Geometry Pilot",
        "",
        f"- Samples: `{meta['samples']}` (`{meta['errors']}` errors, `{meta['correct']}` correct)",
        f"- Problems: `{meta['problems']}`; contrastive: `{meta['contrastive_problems']}`",
        f"- Layers: `{meta['cloud_layers']}`; exact token alignment: `{meta['token_range_key']}`",
        f"- Window/horizons: `{meta['window_tokens']}` tokens / `{meta['horizons']}`",
        f"- Predictive latent dimension: `{meta['latent_dim']}`",
        "",
        "## Fixed Hypothesis",
        "",
        report["hypothesis"],
        "",
        "The Gaussian random projection is only a label-free computational sketch. The "
        "scientific latent chart is learned from future directions predictable on correct "
        "training trajectories. No absolute position, response length, or error label is a "
        "model input.",
        "",
        "## Scores",
        "",
        "| score | confirmatory | coverage | within AUROC | CI95 | BH q | "
        "token-match | token rho | cross AUROC |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for row in sorted(
        report["scores"],
        key=lambda value: np.nan_to_num(value["same_problem_auroc"], nan=-1.0),
        reverse=True,
    ):
        ci = row["same_problem_ci95"]
        lines.append(
            f"| {row['name']} | {int(row['confirmatory'])} | {_fmt(row['coverage'], 3)} | "
            f"{_fmt(row['same_problem_auroc'])} | [{_fmt(ci[0])}, {_fmt(ci[1])}] | "
            f"{_fmt(row['same_problem_bh_q'])} | {_fmt(row['token_length_matched_auroc'])} | "
            f"{_fmt(row['spearman_response_tokens'])} | {_fmt(row['cross_problem_auroc'])} |"
        )
    lines += [
        "",
        "## Falsification Deltas",
        "",
        "| comparison | delta AUROC | CI95 | problems |",
        "|---|---:|---|---:|",
    ]
    for name, row in report["auc_deltas"].items():
        lines.append(
            f"| {name} | {_fmt(row['point'])} | [{_fmt(row['ci95'][0])}, "
            f"{_fmt(row['ci95'][1])}] | {row['problems']} |"
        )
    lines += [
        "",
        "## Correct-Trajectory Predictive Advantage",
        "",
        "Positive values mean the null has higher innovation than the ordered model on "
        "held-out correct responses.",
        "",
    ]
    for name, row in report["correct_predictive_advantage"].items():
        lines.append(
            f"- `{name}`: `{_fmt(row['point'])}`, CI "
            f"`[{_fmt(row['ci95'][0])}, {_fmt(row['ci95'][1])}]`."
        )
    lines += [
        "",
        "## Decision Gate",
        "",
    ]
    for name, passed in gate["checks"].items():
        lines.append(f"- `{name}`: `{bool(passed)}`")
    lines += [
        f"- **PASS: `{gate['passes']}`**",
        f"- Replication: {gate['replication_requirement']}",
        "",
        "## Claim Boundary",
        "",
        f"- Supported only after replication: {report['claim_scope']['supported_if_replicated']}.",
        f"- Not supported: {report['claim_scope']['not_supported']}.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _render_plots(
    output_dir: Path,
    packed: Mapping[str, np.ndarray],
) -> list[Path]:
    import matplotlib.pyplot as plt

    names = [str(value) for value in packed["score_names"].tolist()]
    matrix = np.asarray(packed["scores"], dtype=np.float64)
    labels = np.asarray(packed["y_error"], dtype=np.int64)
    selected = [
        "predictive.token_residual.mahalanobis_mean",
        "null.shuffle.mahalanobis_mean",
        "static.token_residual.mahalanobis_mean",
        "baseline.fixed_window_consensus",
    ]
    figure, axes = plt.subplots(1, len(selected), figsize=(16, 4))
    for axis, name in zip(axes, selected):
        values = matrix[:, names.index(name)]
        axis.hist(values[labels == 0], bins=35, density=True, alpha=0.55, label="correct")
        axis.hist(values[labels == 1], bins=35, density=True, alpha=0.55, label="error")
        axis.set_title(name.replace(".mahalanobis_mean", ""), fontsize=8)
        axis.set_xlabel("risk")
    axes[0].set_ylabel("density")
    axes[-1].legend()
    figure.tight_layout()
    path = output_dir / "score_distributions.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    return [path]


def write_predictive_state_outputs(
    report: Mapping[str, Any],
    packed: Mapping[str, np.ndarray],
    *,
    output: str | Path,
    output_dir: str | Path,
    render_plots: bool = True,
) -> dict[str, str]:
    output = Path(output)
    output_dir = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    summary_json = output_dir / "summary.json"
    summary_json.write_text(
        json.dumps(finite_json(report), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_md = output_dir / "summary.md"
    _write_markdown(summary_md, report)
    score_table = output_dir / "score_table.csv"
    _write_csv(score_table, report["scores"])
    fold_table = output_dir / "fold_diagnostics.csv"
    _write_csv(fold_table, report["fold_diagnostics"])
    paths = {
        "scores_npz": str(output),
        "summary_json": str(summary_json),
        "summary_md": str(summary_md),
        "score_table": str(score_table),
        "fold_diagnostics": str(fold_table),
    }
    if render_plots:
        for path in _render_plots(output_dir, packed):
            paths[path.stem] = str(path)
    return paths
