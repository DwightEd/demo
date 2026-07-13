from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np

from .layer_time_geometry import LAYER_TIME_FIELD_NAMES


EPS = 1e-12


@dataclass(frozen=True)
class LayerTimeValidationConfig:
    """Claim-driven validation settings for the layer-time geometry field."""

    event_offsets: tuple[int, ...] = (-2, -1, 0, 1, 2)
    bootstrap: int = 1000
    random_seed: int = 13
    layer_reduction: str = "median"


def evaluate_layer_time_geometry(
    metrics: Mapping[str, Any],
    cfg: LayerTimeValidationConfig = LayerTimeValidationConfig(),
) -> dict[str, Any]:
    """Evaluate field structure without turning observables into a detector stack.

    The primary outputs are first-error-aligned event curves, their nuisance-
    residualized counterparts, and same-problem paired response contrasts.  A
    full per-layer event map is retained so a scalar reduction cannot hide the
    claimed layer-time band.
    """

    field = np.asarray(metrics["layer_time_geometry_field"], dtype=np.float64)
    names = [str(x) for x in np.asarray(metrics["layer_time_geometry_field_names"]).tolist()]
    if field.ndim != 4 or field.shape[-1] != len(names):
        raise ValueError("layer_time_geometry_field must be [chain,step,layer,observable]")
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    problem_id = np.asarray(metrics.get("problem_id", metrics["chain_idx"]), dtype=np.int64)
    correct = np.asarray(metrics.get("is_correct", (gold < 0).astype(np.int64)), dtype=np.int64)
    layers = np.asarray(metrics.get("layer_time_geometry_layers", np.arange(field.shape[2])), dtype=np.int64)
    if n_steps.shape[0] != field.shape[0] or gold.shape[0] != field.shape[0]:
        raise ValueError("chain metadata does not match layer-time field")
    if cfg.layer_reduction not in {"median", "mean"}:
        raise ValueError("layer_reduction must be 'median' or 'mean'")

    step_len, rel_pos = _step_controls(metrics, n_steps, field.shape[1])
    rng = np.random.default_rng(int(cfg.random_seed))
    observables: dict[str, Any] = {}
    for name in names:
        values = field[..., names.index(name)]
        reduced = reduce_layers(values, cfg.layer_reduction)
        adjusted = residualize_step_signal(
            reduced,
            n_steps=n_steps,
            problem_id=problem_id,
            controls=(rel_pos, np.log1p(step_len)),
        )
        observables[name] = {
            "event": event_study(
                reduced,
                gold=gold,
                problem_id=problem_id,
                n_steps=n_steps,
                offsets=cfg.event_offsets,
                bootstrap=cfg.bootstrap,
                rng=rng,
            ),
            "event_adjusted": event_study(
                adjusted,
                gold=gold,
                problem_id=problem_id,
                n_steps=n_steps,
                offsets=cfg.event_offsets,
                bootstrap=cfg.bootstrap,
                rng=rng,
            ),
            "same_problem": same_problem_paired(
                reduced,
                correct=correct,
                problem_id=problem_id,
                n_steps=n_steps,
                bootstrap=cfg.bootstrap,
                rng=rng,
            ),
            "layer_event": layer_event_map(
                values,
                layers=layers,
                gold=gold,
                problem_id=problem_id,
                n_steps=n_steps,
                offsets=cfg.event_offsets,
                bootstrap=cfg.bootstrap,
                rng=rng,
            ),
        }

    conditional = conditional_wilson_event(
        field,
        names=names,
        n_steps=n_steps,
        gold=gold,
        problem_id=problem_id,
        rel_pos=rel_pos,
        step_len=step_len,
        offsets=cfg.event_offsets,
        bootstrap=cfg.bootstrap,
        rng=rng,
        layer_reduction=cfg.layer_reduction,
    )
    reliability = curvature_reliability_diagnostics(field, names, n_steps)
    return {
        "n_chains": int(field.shape[0]),
        "n_problems": int(np.unique(problem_id).size),
        "layers": layers.tolist(),
        "layer_reduction": cfg.layer_reduction,
        "event_offsets": list(cfg.event_offsets),
        "observables": observables,
        "conditional_wilson_event": conditional,
        "curvature_reliability": reliability,
        "claim_gate": build_claim_gate(observables, conditional, reliability),
    }


def reduce_layers(values: np.ndarray, reduction: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        if reduction == "median":
            return np.nanmedian(values, axis=2)
        if reduction == "mean":
            return np.nanmean(values, axis=2)
    raise ValueError(f"unsupported layer reduction: {reduction}")


def residualize_step_signal(
    signal: np.ndarray,
    *,
    n_steps: np.ndarray,
    problem_id: np.ndarray,
    controls: Sequence[np.ndarray],
) -> np.ndarray:
    """Remove problem fixed effects plus position/length effects by OLS."""

    signal = np.asarray(signal, dtype=np.float64)
    out = np.full(signal.shape, np.nan, dtype=np.float64)
    rows: list[tuple[int, int]] = []
    y: list[float] = []
    x: list[list[float]] = []
    groups: list[int] = []
    for i, count in enumerate(np.asarray(n_steps, dtype=np.int64)):
        for step in range(int(count)):
            control_row = [float(np.asarray(c)[i, step]) for c in controls]
            if np.isfinite(signal[i, step]) and np.isfinite(control_row).all():
                rows.append((i, step))
                y.append(float(signal[i, step]))
                x.append(control_row)
                groups.append(int(problem_id[i]))
    if len(rows) <= len(controls) + 1:
        return out
    y_arr = np.asarray(y, dtype=np.float64)
    x_arr = np.asarray(x, dtype=np.float64)
    groups_arr = np.asarray(groups, dtype=np.int64)
    y_within = y_arr.copy()
    x_within = x_arr.copy()
    for group in np.unique(groups_arr):
        mask = groups_arr == group
        y_within[mask] -= np.mean(y_within[mask])
        x_within[mask] -= np.mean(x_within[mask], axis=0, keepdims=True)
    beta, *_ = np.linalg.lstsq(x_within, y_within, rcond=None)
    residual = y_within - x_within @ beta
    for (i, step), value in zip(rows, residual):
        out[i, step] = float(value)
    return out


def event_study(
    signal: np.ndarray,
    *,
    gold: np.ndarray,
    problem_id: np.ndarray,
    n_steps: np.ndarray,
    offsets: Sequence[int],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    curves = []
    by_offset: dict[int, dict[int, list[float]]] = {}
    for offset in offsets:
        grouped: dict[int, list[float]] = {}
        for i, g in enumerate(np.asarray(gold, dtype=np.int64)):
            step = int(g) + int(offset)
            if g < 0 or step < 0 or step >= int(n_steps[i]):
                continue
            value = float(signal[i, step])
            if np.isfinite(value):
                grouped.setdefault(int(problem_id[i]), []).append(value)
        by_offset[int(offset)] = grouped
        problem_values = np.asarray([np.mean(v) for v in grouped.values()], dtype=np.float64)
        low, high = bootstrap_mean_ci(problem_values, bootstrap, rng)
        curves.append(
            {
                "offset": int(offset),
                "mean": _finite_or_nan(np.mean(problem_values) if problem_values.size else np.nan),
                "ci_low": low,
                "ci_high": high,
                "n_problems": int(problem_values.size),
                "n_chains": int(sum(len(v) for v in grouped.values())),
            }
        )
    contrast = local_peak_contrast(by_offset, bootstrap, rng)
    finite_rows = [
        row for row in curves if row["mean"] is not None and np.isfinite(float(row["mean"]))
    ]
    peak_offset = max(finite_rows, key=lambda row: row["mean"])["offset"] if finite_rows else None
    return {"curve": curves, "peak_offset": peak_offset, "local_peak_contrast": contrast}


def layer_event_map(
    values: np.ndarray,
    *,
    layers: np.ndarray,
    gold: np.ndarray,
    problem_id: np.ndarray,
    n_steps: np.ndarray,
    offsets: Sequence[int],
    bootstrap: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    out = []
    for layer_pos, layer in enumerate(layers):
        result = event_study(
            values[:, :, layer_pos],
            gold=gold,
            problem_id=problem_id,
            n_steps=n_steps,
            offsets=offsets,
            bootstrap=bootstrap,
            rng=rng,
        )
        out.append({"layer": int(layer), **result})
    return out


def same_problem_paired(
    signal: np.ndarray,
    *,
    correct: np.ndarray,
    problem_id: np.ndarray,
    n_steps: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    chain_score = np.full(signal.shape[0], np.nan, dtype=np.float64)
    for i, count in enumerate(n_steps):
        values = signal[i, : int(count)]
        values = values[np.isfinite(values)]
        if values.size:
            chain_score[i] = float(np.mean(values))
    deltas = []
    pair_aucs = []
    eligible = []
    for problem in np.unique(problem_id):
        idx = np.where(problem_id == problem)[0]
        wrong = chain_score[idx[(correct[idx] == 0) & np.isfinite(chain_score[idx])]]
        right = chain_score[idx[(correct[idx] == 1) & np.isfinite(chain_score[idx])]]
        if wrong.size == 0 or right.size == 0:
            continue
        eligible.append(int(problem))
        deltas.append(float(np.median(wrong) - np.median(right)))
        comparisons = wrong[:, None] - right[None, :]
        pair_aucs.append(float(np.mean((comparisons > 0) + 0.5 * (comparisons == 0))))
    delta_arr = np.asarray(deltas, dtype=np.float64)
    auc_arr = np.asarray(pair_aucs, dtype=np.float64)
    delta_low, delta_high = bootstrap_mean_ci(delta_arr, bootstrap, rng)
    auc_low, auc_high = bootstrap_mean_ci(auc_arr, bootstrap, rng)
    return {
        "eligible_problems": int(len(eligible)),
        "mean_problem_delta": _finite_or_nan(np.mean(delta_arr) if delta_arr.size else np.nan),
        "delta_ci_low": delta_low,
        "delta_ci_high": delta_high,
        "pair_micro_auroc": _finite_or_nan(np.mean(auc_arr) if auc_arr.size else np.nan),
        "auroc_ci_low": auc_low,
        "auroc_ci_high": auc_high,
    }


def conditional_wilson_event(
    field: np.ndarray,
    *,
    names: Sequence[str],
    n_steps: np.ndarray,
    gold: np.ndarray,
    problem_id: np.ndarray,
    rel_pos: np.ndarray,
    step_len: np.ndarray,
    offsets: Sequence[int],
    bootstrap: int,
    rng: np.random.Generator,
    layer_reduction: str,
) -> dict[str, Any] | None:
    target_name = (
        "plaquette_reliable_wilson"
        if "plaquette_reliable_wilson" in names
        else "plaquette_wilson_curvature"
    )
    if target_name not in names:
        return None
    target = reduce_layers(field[..., names.index(target_name)], layer_reduction)
    controls = [rel_pos, np.log1p(step_len)]
    for name in (
        "lid",
        "fiber_rank",
        "depth_neighbor_rewire",
        "time_neighbor_rewire",
        "plaquette_transport_residual",
        "rank_singularity",
    ):
        if name in names:
            controls.append(reduce_layers(field[..., names.index(name)], layer_reduction))
    residual = residualize_step_signal(
        target,
        n_steps=n_steps,
        problem_id=problem_id,
        controls=controls,
    )
    result = event_study(
        residual,
        gold=gold,
        problem_id=problem_id,
        n_steps=n_steps,
        offsets=offsets,
        bootstrap=bootstrap,
        rng=rng,
    )
    result["target"] = target_name
    result["conditioned_on"] = [
        "rel_pos",
        "log1p_step_len",
        "lid",
        "fiber_rank",
        "depth_neighbor_rewire",
        "time_neighbor_rewire",
        "plaquette_transport_residual",
        "rank_singularity",
    ]
    return result


def curvature_reliability_diagnostics(
    field: np.ndarray,
    names: Sequence[str],
    n_steps: np.ndarray,
) -> dict[str, Any]:
    if "plaquette_wilson_curvature" not in names or "plaquette_transport_residual" not in names:
        return {}
    curvature = field[..., names.index("plaquette_wilson_curvature")]
    residual = field[..., names.index("plaquette_transport_residual")]
    mask = np.zeros(curvature.shape, dtype=bool)
    for i, count in enumerate(n_steps):
        mask[i, : int(count)] = True
    valid = mask & np.isfinite(curvature) & np.isfinite(residual)
    corr = spearman(curvature[valid], residual[valid]) if np.any(valid) else np.nan
    reliable_name = "plaquette_reliable_wilson"
    reliable_fraction = np.nan
    if reliable_name in names:
        reliable = field[..., names.index(reliable_name)]
        denom = int(np.sum(mask & np.isfinite(curvature)))
        reliable_fraction = float(np.sum(mask & np.isfinite(reliable)) / denom) if denom else np.nan
    return {
        "n_plaquettes": int(np.sum(valid)),
        "wilson_transport_spearman": _finite_or_nan(corr),
        "reliable_fraction": _finite_or_nan(reliable_fraction),
    }


def build_claim_gate(
    observables: Mapping[str, Any],
    conditional: Mapping[str, Any] | None,
    reliability: Mapping[str, Any],
) -> dict[str, Any]:
    primary = observables.get("plaquette_reliable_wilson", {})
    event = primary.get("event_adjusted", {})
    contrast = event.get("local_peak_contrast", {})
    paired = primary.get("same_problem", {})
    conditional_contrast = (conditional or {}).get("local_peak_contrast", {})
    return {
        "event_peak_at_zero": event.get("peak_offset") == 0,
        "adjusted_local_peak_ci_above_zero": _ci_above_zero(contrast),
        "conditional_local_peak_ci_above_zero": _ci_above_zero(conditional_contrast),
        "same_problem_delta_ci_above_zero": _ci_above_zero(
            {
                "ci_low": paired.get("delta_ci_low"),
                "ci_high": paired.get("delta_ci_high"),
            }
        ),
        "transport_correlation_below_0_5": (
            abs(float(reliability.get("wilson_transport_spearman"))) < 0.5
            if reliability.get("wilson_transport_spearman") is not None
            else False
        ),
        "note": "These gates are descriptive prerequisites, not a causal verdict.",
    }


def local_peak_contrast(
    by_offset: Mapping[int, Mapping[int, Sequence[float]]],
    bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    common = set(by_offset.get(0, {}))
    common &= set(by_offset.get(-1, {}))
    common &= set(by_offset.get(1, {}))
    values = []
    for problem in sorted(common):
        center = float(np.mean(by_offset[0][problem]))
        side = 0.5 * (
            float(np.mean(by_offset[-1][problem])) + float(np.mean(by_offset[1][problem]))
        )
        values.append(center - side)
    arr = np.asarray(values, dtype=np.float64)
    low, high = bootstrap_mean_ci(arr, bootstrap, rng)
    return {
        "mean": _finite_or_nan(np.mean(arr) if arr.size else np.nan),
        "ci_low": low,
        "ci_high": high,
        "n_problems": int(arr.size),
    }


def bootstrap_mean_ci(
    values: np.ndarray,
    bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float | None, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None, None
    if bootstrap <= 0 or values.size == 1:
        mean = float(np.mean(values))
        return mean, mean
    draws = rng.integers(0, values.size, size=(int(bootstrap), values.size))
    means = np.mean(values[draws], axis=1)
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(a) & np.isfinite(b)
    if np.sum(valid) < 3:
        return float("nan")
    ar = rankdata(a[valid])
    br = rankdata(b[valid])
    if np.std(ar) <= EPS or np.std(br) <= EPS:
        return float("nan")
    return float(np.corrcoef(ar, br)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < values.size:
        stop = start + 1
        while stop < values.size and sorted_values[stop] == sorted_values[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    return ranks


def _step_controls(
    metrics: Mapping[str, Any],
    n_steps: np.ndarray,
    max_steps: int,
) -> tuple[np.ndarray, np.ndarray]:
    step_len = _named_step_score(metrics, "step_len", n_steps, max_steps)
    rel_pos = _named_step_score(metrics, "rel_pos", n_steps, max_steps)
    ranges = np.asarray(metrics.get("step_token_ranges", []), dtype=object)
    for i, count in enumerate(n_steps):
        for step in range(int(count)):
            if not np.isfinite(rel_pos[i, step]):
                rel_pos[i, step] = step / max(int(count) - 1, 1)
            if not np.isfinite(step_len[i, step]):
                length = 1.0
                try:
                    a, b = ranges[i][step]
                    length = max(float(b) - float(a) + 1.0, 1.0)
                except (IndexError, TypeError, ValueError):
                    pass
                step_len[i, step] = length
    return step_len, rel_pos


def _named_step_score(
    metrics: Mapping[str, Any],
    name: str,
    n_steps: np.ndarray,
    max_steps: int,
) -> np.ndarray:
    out = np.full((n_steps.size, max_steps), np.nan, dtype=np.float64)
    names = [str(x) for x in np.asarray(metrics.get("step_score_names", [])).tolist()]
    if name not in names:
        return out
    scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    out[:, : scores.shape[1]] = scores[:, :, names.index(name)]
    return out


def _ci_above_zero(values: Mapping[str, Any]) -> bool:
    low = values.get("ci_low")
    return low is not None and np.isfinite(float(low)) and float(low) > 0.0


def _finite_or_nan(value: float) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def render_layer_time_validation(summary: Mapping[str, Any]) -> str:
    lines = ["# Layer-Time Geometry Validation", ""]
    lines.append(f"- Chains: `{summary.get('n_chains')}`")
    lines.append(f"- Problems: `{summary.get('n_problems')}`")
    lines.append(f"- Layers: `{summary.get('layers')}`")
    lines.extend(["", "## Claim Gate", "", "| gate | pass |", "|---|---:|"])
    for name, value in summary.get("claim_gate", {}).items():
        if name == "note":
            continue
        lines.append(f"| {name} | {value} |")
    lines.extend(["", "## Primary Observables", ""])
    lines.append("| observable | adjusted peak offset | adjusted local contrast | paired delta | paired AUROC |")
    lines.append("|---|---:|---:|---:|---:|")
    for name in (
        "lid",
        "fiber_rank",
        "depth_neighbor_rewire",
        "time_neighbor_rewire",
        "plaquette_wilson_curvature",
        "plaquette_transport_residual",
        "plaquette_reliable_wilson",
        "rank_singularity",
    ):
        row = summary.get("observables", {}).get(name)
        if not row:
            continue
        event = row.get("event_adjusted", {})
        contrast = event.get("local_peak_contrast", {})
        paired = row.get("same_problem", {})
        lines.append(
            f"| {name} | {event.get('peak_offset')} | {_fmt(contrast.get('mean'))} | "
            f"{_fmt(paired.get('mean_problem_delta'))} | {_fmt(paired.get('pair_micro_auroc'))} |"
        )
    reliability = summary.get("curvature_reliability", {})
    lines.extend(["", "## Reliability", ""])
    lines.append(f"- Wilson/transport Spearman: `{_fmt(reliability.get('wilson_transport_spearman'))}`")
    lines.append(f"- Reliable plaquette fraction: `{_fmt(reliability.get('reliable_fraction'))}`")
    lines.append("")
    lines.append("The claim gate is descriptive. It does not establish causal influence on generated tokens.")
    return "\n".join(lines)


def _fmt(value: Any) -> str:
    if value is None:
        return "NA"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.4f}" if np.isfinite(number) else "NA"
