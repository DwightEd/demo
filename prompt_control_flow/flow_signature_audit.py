from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .evaluate import auprc, auroc, finite_json
from .flow_signature_data import FlowTrajectoryDataset
from .flow_signatures import FlowSignatureEncoding


EPS = 1e-10


@dataclass(frozen=True)
class FlowAuditConfig:
    folds: int = 5
    bootstrap: int = 1000
    permutations: int = 500
    min_correct_support: int = 2
    seed: int = 0
    compute_device: str = "cpu"


def _group_folds(problem_ids: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    unique = np.unique(problem_ids)
    if unique.size < 2:
        raise ValueError("at least two distinct problems are required for cross-fitting")
    rng = np.random.default_rng(int(seed))
    unique = np.array(unique, copy=True)
    rng.shuffle(unique)
    k = min(max(2, int(folds)), int(unique.size))
    fold_map = {value: i % k for i, value in enumerate(unique.tolist())}
    assignment = np.asarray([fold_map[value] for value in problem_ids.tolist()], dtype=np.int64)
    return [
        (np.where(assignment != fold)[0], np.where(assignment == fold)[0])
        for fold in range(k)
    ]


def _robust_center_scale(x: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    center = torch.median(x, dim=0).values
    mad = 1.4826 * torch.median(torch.abs(x - center), dim=0).values
    std = torch.std(x, dim=0, unbiased=False)
    raw_scale = torch.where(mad > eps, mad, std)
    valid = torch.isfinite(raw_scale) & (raw_scale > eps)
    positive = raw_scale[valid]
    floor = torch.median(positive) * 0.05 if positive.numel() else torch.tensor(eps, device=x.device)
    floor_value = max(float(eps), float(floor.detach().cpu()))
    scale = raw_scale.clamp_min(floor_value)
    scale = torch.where(valid, scale, torch.full_like(scale, float("inf")))
    return center, scale


def _mean_energy(features: torch.Tensor, center: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    z = (features - center) / scale
    return torch.mean(z.square(), dim=-1)


def score_conditional_support(
    features: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    cfg: FlowAuditConfig,
) -> dict[str, np.ndarray]:
    """Cross-fit feature scale, then score global and same-problem support energy.

    The global score is deployable but not prompt conditioned. The support
    score is a diagnostic upper bound: correct samples from the target problem
    define its flow center, with leave-one-out scoring for correct candidates.
    """

    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 3:
        raise ValueError("features must have shape [sample, phase, feature]")
    if array.shape[0] != len(y_error):
        raise ValueError("feature and label lengths differ")
    device = torch.device(cfg.compute_device)
    tensor = torch.as_tensor(array, device=device)
    n, phase_points, _ = array.shape
    global_profile = np.full((n, phase_points), np.nan, dtype=np.float32)
    support_profile = np.full((n, phase_points), np.nan, dtype=np.float32)
    support_size = np.zeros(n, dtype=np.int64)

    for train, test in _group_folds(problem_ids, cfg.folds, cfg.seed):
        correct_train = train[y_error[train] == 0]
        if correct_train.size < 2:
            continue
        train_tensor = tensor[torch.as_tensor(correct_train, device=device)]
        center, scale = _robust_center_scale(train_tensor)
        test_index = torch.as_tensor(test, device=device)
        global_profile[test] = (
            _mean_energy(tensor[test_index], center, scale).detach().cpu().numpy()
        )

        for problem in np.unique(problem_ids[test]):
            group = test[problem_ids[test] == problem]
            correct = group[y_error[group] == 0]
            if correct.size < int(cfg.min_correct_support):
                continue
            for candidate in group:
                refs = correct[correct != candidate] if y_error[candidate] == 0 else correct
                if refs.size == 0:
                    continue
                prototype = torch.mean(
                    tensor[torch.as_tensor(refs, device=device)],
                    dim=0,
                )
                energy = _mean_energy(tensor[candidate], prototype, scale)
                support_profile[candidate] = energy.detach().cpu().numpy()
                support_size[candidate] = int(refs.size)

    return {
        "global_profile": global_profile,
        "support_profile": support_profile,
        "support_size": support_size,
    }


def _profile_integral(profile: np.ndarray, phase_grid: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=np.float64)
    phase = np.asarray(phase_grid, dtype=np.float64)
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(values, phase, axis=1)
    else:
        integral = np.trapz(values, phase, axis=1)
    width = float(phase[-1] - phase[0])
    return integral / max(width, EPS)


def _profile_scores(prefix: np.ndarray, phase_grid: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "endpoint": np.asarray(prefix[:, -1], dtype=np.float64),
        "prefix_integral": _profile_integral(prefix, phase_grid),
    }


def _crossfit_residualize(
    score: np.ndarray,
    controls: np.ndarray,
    problem_ids: np.ndarray,
    cfg: FlowAuditConfig,
) -> np.ndarray:
    out = np.full(len(score), np.nan, dtype=np.float64)
    score = np.asarray(score, dtype=np.float64)
    controls = np.asarray(controls, dtype=np.float64)
    for train, test in _group_folds(problem_ids, cfg.folds, cfg.seed + 991):
        finite_train = np.isfinite(score[train]) & np.isfinite(controls[train]).all(axis=1)
        finite_test = np.isfinite(score[test]) & np.isfinite(controls[test]).all(axis=1)
        tr = train[finite_train]
        te = test[finite_test]
        if tr.size < controls.shape[1] + 2 or te.size == 0:
            continue
        mean = controls[tr].mean(axis=0)
        std = controls[tr].std(axis=0)
        std[std < 1e-8] = 1.0
        x_train = np.column_stack([np.ones(tr.size), (controls[tr] - mean) / std])
        x_test = np.column_stack([np.ones(te.size), (controls[te] - mean) / std])
        x_train_tensor = torch.as_tensor(x_train, dtype=torch.float64)
        target_tensor = torch.as_tensor(score[tr], dtype=torch.float64)
        gram = x_train_tensor.T @ x_train_tensor
        ridge = 1e-8 * torch.eye(gram.shape[0], dtype=gram.dtype)
        beta = torch.linalg.solve(gram + ridge, x_train_tensor.T @ target_tensor).numpy()
        out[te] = score[te] - np.sum(x_test * beta[None, :], axis=1)
    return out


def _problem_pair_components(
    score: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
) -> dict[Any, tuple[float, int]]:
    components: dict[Any, tuple[float, int]] = {}
    for problem in np.unique(problem_ids):
        idx = np.where(problem_ids == problem)[0]
        err = score[idx[(y_error[idx] == 1) & np.isfinite(score[idx])]]
        cor = score[idx[(y_error[idx] == 0) & np.isfinite(score[idx])]]
        pairs = int(err.size * cor.size)
        if pairs == 0:
            continue
        concordance = 0.0
        for a in err:
            concordance += float(np.sum(a > cor) + 0.5 * np.sum(a == cor))
        components[problem] = (concordance, pairs)
    return components


def _micro_pair_auc(components: Mapping[Any, tuple[float, int]]) -> tuple[float, int]:
    pairs = int(sum(value[1] for value in components.values()))
    concordance = float(sum(value[0] for value in components.values()))
    return (concordance / pairs if pairs else float("nan")), pairs


def _bootstrap_pair_auc(
    components: Mapping[Any, tuple[float, int]],
    *,
    draws: int,
    seed: int,
) -> tuple[float, float]:
    if draws <= 0 or len(components) < 2:
        return float("nan"), float("nan")
    keys = list(components)
    rng = np.random.default_rng(int(seed))
    values = []
    for _ in range(int(draws)):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        concordance = sum(components[keys[i]][0] for i in chosen)
        pairs = sum(components[keys[i]][1] for i in chosen)
        if pairs:
            values.append(concordance / pairs)
    if not values:
        return float("nan"), float("nan")
    return tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))


def _within_problem_permutation_p(
    score: np.ndarray,
    y_error: np.ndarray,
    problem_ids: np.ndarray,
    observed: float,
    *,
    permutations: int,
    seed: int,
) -> float:
    if permutations <= 0 or not np.isfinite(observed):
        return float("nan")
    rng = np.random.default_rng(int(seed))
    values = []
    permuted = np.array(y_error, copy=True)
    for _ in range(int(permutations)):
        for problem in np.unique(problem_ids):
            idx = np.where(problem_ids == problem)[0]
            permuted[idx] = rng.permutation(y_error[idx])
        value, _ = _micro_pair_auc(_problem_pair_components(score, permuted, problem_ids))
        if np.isfinite(value):
            values.append(value)
    if not values:
        return float("nan")
    return float((1 + np.sum(np.asarray(values) >= observed)) / (len(values) + 1))


def evaluate_score(
    name: str,
    score: np.ndarray,
    dataset: FlowTrajectoryDataset,
    cfg: FlowAuditConfig,
) -> dict[str, Any]:
    score = np.asarray(score, dtype=np.float64)
    finite = np.isfinite(score)
    y = dataset.y_error
    components = _problem_pair_components(score, y, dataset.problem_ids)
    within, pairs = _micro_pair_auc(components)
    ci_low, ci_high = _bootstrap_pair_auc(
        components,
        draws=cfg.bootstrap,
        seed=cfg.seed + sum(ord(ch) for ch in name),
    )
    return {
        "name": name,
        "n": int(finite.sum()),
        "coverage": float(finite.mean()),
        "n_error": int(np.sum(finite & (y == 1))),
        "cross_problem_auroc": auroc(y, score),
        "cross_problem_auprc": auprc(y, score),
        "same_problem_auroc": within,
        "same_problem_pairs": pairs,
        "same_problem_problems": int(len(components)),
        "same_problem_ci95": [ci_low, ci_high],
        "same_problem_permutation_p": _within_problem_permutation_p(
            score,
            y,
            dataset.problem_ids,
            within,
            permutations=cfg.permutations,
            seed=cfg.seed + 17 + sum(ord(ch) for ch in name),
        ),
        "error_mean": float(np.nanmean(score[y == 1])) if np.any(finite & (y == 1)) else float("nan"),
        "correct_mean": float(np.nanmean(score[y == 0])) if np.any(finite & (y == 0)) else float("nan"),
    }


def _bootstrap_auc_delta(
    first: np.ndarray,
    second: np.ndarray,
    dataset: FlowTrajectoryDataset,
    cfg: FlowAuditConfig,
) -> dict[str, Any]:
    first_components = _problem_pair_components(first, dataset.y_error, dataset.problem_ids)
    second_components = _problem_pair_components(second, dataset.y_error, dataset.problem_ids)
    keys = sorted(set(first_components) & set(second_components), key=str)
    first_auc, _ = _micro_pair_auc({key: first_components[key] for key in keys})
    second_auc, _ = _micro_pair_auc({key: second_components[key] for key in keys})
    point = first_auc - second_auc
    if cfg.bootstrap <= 0 or len(keys) < 2:
        return {"point": point, "ci95": [float("nan"), float("nan")], "problems": len(keys)}
    rng = np.random.default_rng(cfg.seed + 701)
    values = []
    for _ in range(cfg.bootstrap):
        chosen = rng.choice(len(keys), size=len(keys), replace=True)
        first_conc = sum(first_components[keys[i]][0] for i in chosen)
        first_pairs = sum(first_components[keys[i]][1] for i in chosen)
        second_conc = sum(second_components[keys[i]][0] for i in chosen)
        second_pairs = sum(second_components[keys[i]][1] for i in chosen)
        if first_pairs and second_pairs:
            values.append(first_conc / first_pairs - second_conc / second_pairs)
    ci = np.percentile(values, [2.5, 97.5]) if values else [np.nan, np.nan]
    return {"point": point, "ci95": [float(ci[0]), float(ci[1])], "problems": len(keys)}


def _effective_rank(
    x: np.ndarray,
    max_rank: int = 128,
    *,
    compute_device: str = "cpu",
) -> float:
    values = np.asarray(x, dtype=np.float64)
    values = values[np.isfinite(values).all(axis=1)]
    if values.shape[0] < 2:
        return float("nan")
    values = values - values.mean(axis=0, keepdims=True)
    tensor = torch.as_tensor(values, dtype=torch.float32, device=torch.device(compute_device))
    if values.shape[1] > max_rank and values.shape[0] > max_rank:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        sketch = torch.randn(
            values.shape[1],
            max_rank,
            generator=generator,
            dtype=torch.float32,
        ).to(tensor.device) / math.sqrt(max_rank)
        tensor = tensor @ sketch
    singular = torch.linalg.svdvals(tensor)
    eigen = singular.square()
    denom = torch.sum(eigen.square())
    if float(denom.detach().cpu()) <= EPS:
        return 0.0
    return float((torch.sum(eigen).square() / denom).detach().cpu())


def _conditional_residual_geometry(
    endpoint_features: np.ndarray,
    dataset: FlowTrajectoryDataset,
    *,
    compute_device: str,
) -> dict[str, Any]:
    correct_residuals = []
    error_residuals = []
    for problem in np.unique(dataset.problem_ids):
        idx = np.where(dataset.problem_ids == problem)[0]
        correct = idx[dataset.y_error[idx] == 0]
        if correct.size < 2:
            continue
        for candidate in idx:
            refs = correct[correct != candidate] if dataset.y_error[candidate] == 0 else correct
            if refs.size == 0:
                continue
            residual = endpoint_features[candidate] - endpoint_features[refs].mean(axis=0)
            if dataset.y_error[candidate] == 0:
                correct_residuals.append(residual)
            else:
                error_residuals.append(residual)
    correct_array = np.asarray(correct_residuals, dtype=np.float64)
    error_array = np.asarray(error_residuals, dtype=np.float64)
    correct_radius = (
        torch.linalg.vector_norm(
            torch.as_tensor(correct_array, dtype=torch.float32, device=torch.device(compute_device)),
            dim=1,
        ).detach().cpu().numpy()
        if len(correct_array)
        else np.empty(0)
    )
    error_radius = (
        torch.linalg.vector_norm(
            torch.as_tensor(error_array, dtype=torch.float32, device=torch.device(compute_device)),
            dim=1,
        ).detach().cpu().numpy()
        if len(error_array)
        else np.empty(0)
    )
    return {
        "n_correct": int(len(correct_array)),
        "n_error": int(len(error_array)),
        "effective_rank_correct": _effective_rank(correct_array, compute_device=compute_device),
        "effective_rank_error": _effective_rank(error_array, compute_device=compute_device),
        "median_radius_correct": float(np.median(correct_radius)) if len(correct_radius) else float("nan"),
        "median_radius_error": float(np.median(error_radius)) if len(error_radius) else float("nan"),
    }


def run_flow_signature_audit(
    dataset: FlowTrajectoryDataset,
    chronological: FlowSignatureEncoding,
    shuffled: FlowSignatureEncoding,
    cfg: FlowAuditConfig,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if chronological.order1_prefix.shape[0] != dataset.n_samples:
        raise ValueError("encoding does not match dataset")
    if shuffled is None:
        raise ValueError("the chronology control requires a shuffled encoding")

    feature_sets = {
        "o1": chronological.order1_prefix,
        "o2": chronological.order2_prefix,
        "o2_shuffled": shuffled.order2_prefix,
    }
    profiles: dict[str, np.ndarray] = {}
    scalar_scores: dict[str, np.ndarray] = {}
    for feature_name, features in feature_sets.items():
        result = score_conditional_support(features, dataset.y_error, dataset.problem_ids, cfg)
        for reference in ("global", "support"):
            profile_name = f"{reference}_{feature_name}_profile"
            profile = result[f"{reference}_profile"]
            profiles[profile_name] = profile
            summaries = _profile_scores(profile, chronological.phase_grid)
            scalar_scores[f"{reference}_{feature_name}_endpoint"] = summaries["endpoint"]
            if feature_name == "o2":
                scalar_scores[f"{reference}_{feature_name}_prefix_integral"] = summaries[
                    "prefix_integral"
                ]
        if feature_name == "o2":
            scalar_scores["support_size"] = result["support_size"].astype(np.float64)

    controls = np.column_stack(
        [
            np.log1p(dataset.n_steps.astype(np.float64)),
            np.log1p(dataset.response_chars.astype(np.float64)),
        ]
    )
    scalar_scores["control_n_steps"] = dataset.n_steps.astype(np.float64)
    scalar_scores["control_response_chars"] = dataset.response_chars.astype(np.float64)
    scalar_scores["control_total_variation"] = np.mean(
        chronological.total_variation,
        axis=1,
    ).astype(np.float64)
    for reference in ("global", "support"):
        key = f"{reference}_o2_endpoint"
        scalar_scores[f"{key}_length_residual"] = _crossfit_residualize(
            scalar_scores[key],
            controls,
            dataset.problem_ids,
            cfg,
        )

    score_table = {
        name: evaluate_score(name, values, dataset, cfg)
        for name, values in scalar_scores.items()
        if name != "support_size"
    }
    support_available = score_table["support_o2_endpoint"]["same_problem_pairs"] > 0
    prefix = "support" if support_available else "global"
    primary_name = f"{prefix}_o2_endpoint"
    order2_ablation_name = f"{prefix}_o2_endpoint"
    order1_name = f"{prefix}_o1_endpoint"
    shuffle_name = f"{prefix}_o2_shuffled_endpoint"
    length_name = f"{primary_name}_length_residual"
    order_gain = _bootstrap_auc_delta(
        scalar_scores[order2_ablation_name],
        scalar_scores[order1_name],
        dataset,
        cfg,
    )
    chronology_gain = _bootstrap_auc_delta(
        scalar_scores[primary_name],
        scalar_scores[shuffle_name],
        dataset,
        cfg,
    )
    primary = score_table[primary_name]
    length_residual = score_table[length_name]
    control_best = max(
        score_table[name]["same_problem_auroc" if support_available else "cross_problem_auroc"]
        for name in ("control_n_steps", "control_response_chars", "control_total_variation")
    )
    primary_auc_key = "same_problem_auroc" if support_available else "cross_problem_auroc"
    ci_low = primary["same_problem_ci95"][0]
    report: dict[str, Any] = {
        "method": "conditional_ordered_reasoning_flow_logsignature",
        "meta": {
            "source": dataset.source_path,
            "vector_key": dataset.vector_key,
            "label_policy": dataset.label_policy,
            "n_samples": dataset.n_samples,
            "n_error": int(dataset.y_error.sum()),
            "n_problems": int(np.unique(dataset.problem_ids).size),
            "layers": dataset.layer_ids.tolist(),
            "hidden_dim": dataset.hidden_dim,
            "phase_points": int(len(chronological.phase_grid)),
            "feature_metadata": chronological.feature_metadata,
            "dataset_metadata": dataset.metadata,
            "skipped": dataset.skipped,
            "reference_semantics": {
                "global": "correct trajectories from training problems only",
                "support": "same-problem correct support; leave-one-out for correct candidates; diagnostic, not zero-shot deployable",
            },
        },
        "headline": {
            "primary_score": primary_name,
            "primary_auc_kind": primary_auc_key,
            "primary_auc": primary[primary_auc_key],
            "primary_auprc": primary["cross_problem_auprc"],
            "support_diagnostic_available": bool(support_available),
            "order_ablation_scores": [order2_ablation_name, order1_name],
            "order2_minus_order1": order_gain,
            "chronological_minus_shuffled": chronology_gain,
            "length_residual_auc": length_residual[primary_auc_key],
            "best_length_control_auc": control_best,
        },
        "hypothesis_gates": {
            "conditional_auc_above_chance": bool(
                (np.isfinite(ci_low) and ci_low > 0.5)
                if support_available
                else primary[primary_auc_key] > 0.5
            ),
            "order2_adds_over_order1": bool(
                np.isfinite(order_gain["ci95"][0]) and order_gain["ci95"][0] > 0
            ),
            "chronology_adds_over_shuffled": bool(
                np.isfinite(chronology_gain["ci95"][0]) and chronology_gain["ci95"][0] > 0
            ),
            "beats_length_controls": bool(primary[primary_auc_key] > control_best),
            "survives_length_residualization": bool(length_residual[primary_auc_key] > 0.5),
        },
        "scores": score_table,
        "conditional_geometry": _conditional_residual_geometry(
            chronological.order2_prefix[:, -1],
            dataset,
            compute_device=cfg.compute_device,
        ),
    }
    packed = {
        "original_indices": dataset.original_indices,
        "problem_ids": dataset.problem_ids,
        "sample_idx": dataset.sample_idx,
        "y_error": dataset.y_error,
        "is_correct": dataset.is_correct,
        "n_steps": dataset.n_steps,
        "response_chars": dataset.response_chars,
        "layers": dataset.layer_ids,
        "phase_grid": chronological.phase_grid,
        "score_names": np.asarray(list(scalar_scores), dtype=object),
        "scores": np.column_stack(list(scalar_scores.values())).astype(np.float32),
        "profile_names": np.asarray(list(profiles), dtype=object),
        "profiles": np.stack(list(profiles.values()), axis=-1).astype(np.float32),
        "projection": chronological.projection,
        "vector_key": np.asarray(dataset.vector_key, dtype=object),
        "method": np.asarray(report["method"], dtype=object),
        "metadata_json": np.asarray(json.dumps(finite_json(report["meta"]), ensure_ascii=False), dtype=object),
    }
    return report, packed


def write_flow_signature_outputs(
    report: Mapping[str, Any],
    packed: Mapping[str, np.ndarray],
    *,
    output: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path, Path]:
    output = Path(output)
    output_dir = Path(output_dir)
    output.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **packed)
    json_path = output_dir / "reasoning_flow_signature_audit.json"
    markdown_path = output_dir / "reasoning_flow_signature_audit.md"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(finite_json(report), handle, ensure_ascii=False, indent=2)
    _write_markdown(markdown_path, report)
    return output, json_path, markdown_path


def _fmt(value: Any, digits: int = 4) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "NA"
    return f"{number:.{digits}f}" if np.isfinite(number) else "NA"


def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
    meta = report["meta"]
    headline = report["headline"]
    lines = [
        "# Conditional Ordered Reasoning-Flow Signature Audit",
        "",
        f"- Source: `{meta['source']}`",
        f"- Samples: `{meta['n_samples']}`; errors: `{meta['n_error']}`; problems: `{meta['n_problems']}`",
        f"- Vector key: `{meta['vector_key']}`; layers: `{meta['layers']}`",
        f"- Primary: `{headline['primary_score']}` = `{_fmt(headline['primary_auc'])}` ({headline['primary_auc_kind']})",
        "",
        "## Claim Gates",
        "",
        "| gate | pass |",
        "|---|---:|",
    ]
    for name, passed in report["hypothesis_gates"].items():
        lines.append(f"| `{name}` | `{bool(passed)}` |")
    lines += [
        "",
        "## Falsification Contrasts",
        "",
        f"- Order-2 minus order-1: `{_fmt(headline['order2_minus_order1']['point'])}`; "
        f"95% CI `{[_fmt(x) for x in headline['order2_minus_order1']['ci95']]}`.",
        f"- Chronological minus shuffled: `{_fmt(headline['chronological_minus_shuffled']['point'])}`; "
        f"95% CI `{[_fmt(x) for x in headline['chronological_minus_shuffled']['ci95']]}`.",
        f"- Length-residualized primary: `{_fmt(headline['length_residual_auc'])}`.",
        f"- Best length/scale control: `{_fmt(headline['best_length_control_auc'])}`.",
        "",
        "## Scores",
        "",
        "| score | coverage | cross AUROC | same-problem AUROC | pairs | CI95 | permutation p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    rows = sorted(
        report["scores"].items(),
        key=lambda item: np.nan_to_num(item[1]["same_problem_auroc"], nan=item[1]["cross_problem_auroc"]),
        reverse=True,
    )
    for name, row in rows:
        ci = row["same_problem_ci95"]
        lines.append(
            f"| `{name}` | {_fmt(row['coverage'], 3)} | {_fmt(row['cross_problem_auroc'])} | "
            f"{_fmt(row['same_problem_auroc'])} | {row['same_problem_pairs']} | "
            f"[{_fmt(ci[0])}, {_fmt(ci[1])}] | {_fmt(row['same_problem_permutation_p'])} |"
        )
    geometry = report["conditional_geometry"]
    lines += [
        "",
        "## Conditional Geometry",
        "",
        f"- Correct residual effective rank: `{_fmt(geometry['effective_rank_correct'])}`.",
        f"- Error residual effective rank: `{_fmt(geometry['effective_rank_error'])}`.",
        f"- Correct median radius: `{_fmt(geometry['median_radius_correct'])}`.",
        f"- Error median radius: `{_fmt(geometry['median_radius_error'])}`.",
        "",
        "The support score is an oracle diagnostic of the conditional-flow hypothesis; the global score is the zero-support baseline.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_order_sensitive_synthetic_dataset(
    *,
    n_problems: int = 24,
    correct_per_problem: int = 3,
    error_per_problem: int = 2,
    hidden_dim: int = 8,
    layers: int = 2,
    seed: int = 0,
) -> FlowTrajectoryDataset:
    """Create equal-length paths where only increment order carries the label."""

    rng = np.random.default_rng(int(seed))
    trajectories: list[np.ndarray] = []
    problem_ids = []
    sample_idx = []
    labels = []
    for problem in range(n_problems):
        first = rng.normal(size=hidden_dim)
        first /= math.sqrt(float(np.sum(first * first)))
        second = rng.normal(size=hidden_dim)
        second -= first * np.dot(first, second)
        second /= math.sqrt(float(np.sum(second * second)))
        offset = rng.normal(scale=3.0, size=hidden_dim)
        layer_transforms = []
        for _ in range(layers):
            permutation = rng.permutation(hidden_dim)
            signs = rng.choice(np.asarray([-1.0, 1.0]), size=hidden_dim)
            layer_transforms.append((permutation, signs))
        for sample in range(correct_per_problem + error_per_problem):
            is_error = sample >= correct_per_problem
            ordered = [second, second, first, first] if is_error else [first, first, second, second]
            layer_paths = []
            for permutation, signs in layer_transforms:
                increments = np.asarray(ordered)[:, permutation] * signs
                increments += rng.normal(scale=0.01, size=increments.shape)
                states = np.vstack([np.zeros(hidden_dim), np.cumsum(increments, axis=0)])
                states += offset[permutation] * signs
                layer_paths.append(states)
            trajectories.append(np.stack(layer_paths, axis=1).astype(np.float32))
            problem_ids.append(problem)
            sample_idx.append(sample)
            labels.append(int(is_error))
    n = len(trajectories)
    labels_array = np.asarray(labels, dtype=np.int64)
    return FlowTrajectoryDataset(
        source_path="synthetic://ordered-flow",
        vector_key="synthetic_stepvec",
        trajectories=trajectories,
        original_indices=np.arange(n, dtype=np.int64),
        problem_ids=np.asarray(problem_ids, dtype=np.int64),
        sample_idx=np.asarray(sample_idx, dtype=np.int64),
        y_error=labels_array,
        is_correct=1 - labels_array,
        n_steps=np.full(n, 5, dtype=np.int64),
        response_chars=np.full(n, 100, dtype=np.int64),
        layer_ids=np.arange(layers, dtype=np.int64),
        hidden_dim=hidden_dim,
        label_policy="synthetic",
        skipped={},
        metadata={"construction": "same endpoint and increment multiset; label changes order only"},
    )
