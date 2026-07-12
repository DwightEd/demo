#!/usr/bin/env python3
"""End-to-end validation for the deployable ECGH primitives.

The synthetic path is CPU-only and deliberately exercises the full causal
contract: semantic prompt-span anchors, compact lookback, anchor-residual Gram
geometry, boundary-free change detection, first-error right censoring, and a
pure micro-replay plan.  The real-data path audits a canonical NPZ without
claiming model-quality results.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import numpy as np

from anchorflow.anchor_repr import build_anchor_bank
from anchorflow.anchors import Anchor, parse_anchors
from anchorflow.data import Trace, load_traces
from anchorflow.hazard import grouped_oof_hazard, make_first_error_hazard_targets
from anchorflow.intervention import build_micro_replay
from anchorflow.lookback import compact_hidden_lookback
from anchorflow.phase import (
    calibrate_chain_fpr_threshold,
    causal_boundary_events,
    causal_change_scores,
)
from anchorflow.volume import conditional_gram_geometry


def _finite_json(value):
    if isinstance(value, dict):
        return {str(k): _finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return _finite_json(value.tolist())
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def synthetic_validation(seed: int = 7) -> Dict[str, object]:
    rng = np.random.default_rng(seed)
    prompt = "Tom starts with 7 books and gives away 2. How many books remain?"
    offsets = np.asarray([(i, i + 1) for i in range(len(prompt))], int)
    hidden = np.zeros((len(prompt), 6), float)
    hidden[:, 0] = 0.1

    anchors = [
        Anchor(0, "number", "7", (prompt.index("7"), prompt.index("7") + 1)),
        Anchor(1, "number", "2", (prompt.index("2"), prompt.index("2") + 1)),
        Anchor(2, "goal", "How many books remain?", (prompt.index("How"), len(prompt))),
    ]
    hidden[anchors[0].char_span[0], 1] = 3.0
    hidden[anchors[1].char_span[0], 2] = 3.0
    hidden[anchors[2].char_span[0] : anchors[2].char_span[1], 3] = 2.0
    trace = Trace(
        idx=0,
        chain_id="synthetic",
        problem_id=0,
        dataset="synthetic",
        correct=False,
        gold_error_step=8,
        step_token_ranges=np.asarray([[i, i] for i in range(14)]),
        steps_text=[str(i) for i in range(14)],
        response_text="",
        prompt_text=prompt,
        features={"prompt_offsets": offsets, "prompt_hidden": hidden},
        stepvec=None,
        qvec=None,
        sv_layers=[],
        hidden_path=None,
        layer=0,
        prompt_offsets=offsets,
        prompt_hidden=hidden,
        question_char_span=(0, len(prompt)),
    )
    bank = build_anchor_bank(trace, anchors, prompt_offsets=offsets, prompt_hidden=hidden)
    if not bank.semantic:
        raise AssertionError("synthetic prompt spans must produce a fully semantic bank")

    T, d = 14, hidden.shape[1]
    response = np.empty((T, d), float)
    clouds = []
    grounded = np.mean(bank.vectors, axis=0)
    grounded /= np.linalg.norm(grounded)
    for t in range(T):
        if t < 8:
            center = grounded + 0.01 * rng.normal(size=d)
            cloud = center + 0.015 * rng.normal(size=(6, d))
        else:
            center = np.asarray([0, 0, 0, 0, 1, 1], float)
            cloud = center + 0.55 * rng.normal(size=(6, d))
        response[t] = center
        clouds.append(cloud)

    lookback = compact_hidden_lookback(response, bank.vectors, window=2)
    gram = conditional_gram_geometry(
        clouds,
        anchor_vectors=bank.vectors,
        condition=np.ones(T, dtype=bool),
    )
    phase_x = np.column_stack([
        lookback["detach"],
        gram["residual_energy_ratio"],
        gram["residual_spectral_js_change"],
    ])
    phase = causal_change_scores(phase_x, min_history=4, recent_window=1)
    finite = np.where(np.isfinite(phase["change_score"]))[0]
    peak = int(finite[np.argmax(phase["change_score"][finite])])
    if not 8 <= peak <= 10:
        raise AssertionError(f"boundary-free peak {peak} missed the injected rupture at 8")
    pre = phase["change_score"][finite[finite < 8]]
    threshold = float(np.max(pre) + 1e-6) if len(pre) else 1.0
    events = causal_boundary_events(phase["change_score"], threshold, refractory=2)
    first_event = int(np.where(events)[0][0])
    if first_event > 10:
        raise AssertionError("causal event arrived too late")

    targets = make_first_error_hazard_targets([T, T], [8, None])
    if int(targets.at_risk[0].sum()) != 9 or int(targets.at_risk[1].sum()) != T:
        raise AssertionError("first-error/right-censor risk sets are wrong")
    if targets.at_risk[0, 9:].any():
        raise AssertionError("post-error positions leaked into the hazard risk set")

    replay = build_micro_replay(
        list(range(T)),
        first_event,
        rollback=1,
        repair_instruction=(9001, 9002),
    )
    if replay.cut_index >= first_event or replay.model_input[-2:] != (9001, 9002):
        raise AssertionError("micro-replay did not preserve a safe rollback prefix")

    return {
        "status": "passed",
        "seed": int(seed),
        "semantic_anchor_count": int(len(bank.anchors)),
        "fallback_anchor_count": int(np.sum(bank.fallback_mask)),
        "injected_first_error": 8,
        "change_peak": peak,
        "first_event": first_event,
        "pre_detach_mean": float(np.mean(lookback["detach"][:8])),
        "post_detach_mean": float(np.mean(lookback["detach"][8:])),
        "pre_residual_energy": float(np.mean(gram["residual_energy_ratio"][:8])),
        "post_residual_energy": float(np.mean(gram["residual_energy_ratio"][8:])),
        "hazard_at_risk_counts": targets.at_risk.sum(axis=1).astype(int).tolist(),
        "replay_cut_index": int(replay.cut_index),
    }


def audit_npz(path: str, layer: int, max_chains: int = 0) -> Dict[str, object]:
    traces, meta = load_traces(path, layer=layer, max_chains=max_chains)
    semantic = fallback = with_clouds = geometry_ready = 0
    errors = []
    labeled_traces = []
    baseline_sequences = []
    ecgh_sequences = []

    def resultant(cloud):
        H = np.asarray(cloud, float)
        norms = np.linalg.norm(H, axis=1, keepdims=True)
        good = np.isfinite(H).all(axis=1) & (norms[:, 0] > 1e-9)
        if not good.any():
            return np.nan
        U = H[good] / norms[good]
        return float(np.linalg.norm(np.mean(U, axis=0)))

    for trace in traces:
        try:
            anchors = parse_anchors(
                trace.prompt_text,
                char_span=trace.question_char_span,
            )
            bank = build_anchor_bank(trace, anchors)
            semantic += int(bank.semantic)
            fallback += int(not bank.semantic)
            if trace.step_clouds:
                with_clouds += 1
                if bank.semantic:
                    clouds = trace.step_clouds
                    response_hidden = np.asarray([
                        np.nanmean(np.asarray(cloud, float), axis=0) for cloud in clouds
                    ])
                    lookback = compact_hidden_lookback(response_hidden, bank.vectors, window=2)
                    gram = conditional_gram_geometry(
                        trace.step_clouds,
                        anchor_vectors=bank.vectors,
                    )
                    geometry_ready += 1
                    spread = 1.0 - np.asarray([resultant(x) for x in clouds], float)
                    uncertainty = np.asarray(
                        trace.features.get("U_D_mean", np.full(trace.n_steps, np.nan)),
                        float,
                    )
                    logn = np.log1p(np.asarray([len(x) for x in clouds], float))
                    pos = np.arange(trace.n_steps, dtype=float) / max(1, trace.n_steps - 1)
                    baseline = np.column_stack([logn, pos, spread, uncertainty])
                    primitive = np.column_stack([
                        lookback["detach"],
                        lookback["transport_shift"],
                        lookback["anchor_entropy"],
                        gram["residual_energy_ratio"],
                        gram["residual_eff_rank"],
                        gram["residual_spectral_js_change"],
                    ])
                    phase = causal_change_scores(primitive, min_history=4)
                    full = np.column_stack([
                        baseline,
                        primitive,
                        phase["change_score"],
                        phase["direction_jump"],
                    ])
                    if trace.correct or trace.gold_error_step >= 0:
                        labeled_traces.append(trace)
                        baseline_sequences.append(baseline)
                        ecgh_sequences.append(full)
        except Exception as exc:  # audit all records and report, never conceal
            errors.append({"chain_id": trace.chain_id, "error": str(exc)})

    hazard = None
    if labeled_traces and len({t.problem_id for t in labeled_traces}) >= 2:
        lengths = [t.n_steps for t in labeled_traces]
        first_error = [None if t.correct else t.gold_error_step for t in labeled_traces]
        groups = [t.problem_id for t in labeled_traces]

        def metrics(result):
            scores = []
            labels = []
            top1 = []
            correct_max = []
            for i, trace in enumerate(labeled_traces):
                h = np.asarray(result["hazard"][i], float)
                risk = np.asarray(result["at_risk"][i, : len(h)], bool)
                target = np.asarray(result["target"][i, : len(h)], int)
                keep = risk & np.isfinite(h)
                scores.extend(h[keep].tolist())
                labels.extend(target[keep].tolist())
                if trace.correct and keep.any():
                    correct_max.append(float(np.max(h[keep])))
                elif trace.gold_error_step >= 0 and keep.any():
                    allowed = np.where(keep)[0]
                    top1.append(int(allowed[np.argmax(h[allowed])] == trace.gold_error_step))
            from sklearn.metrics import roc_auc_score
            auc = (
                float(roc_auc_score(labels, scores))
                if len(set(labels)) == 2 else float("nan")
            )
            threshold = calibrate_chain_fpr_threshold(
                [[x] for x in correct_max], target_fpr=0.05
            ) if correct_max else float("nan")
            detections = []
            delays = []
            if np.isfinite(threshold):
                for i, trace in enumerate(labeled_traces):
                    if trace.correct:
                        continue
                    h = np.asarray(result["hazard"][i], float)
                    hit = np.where(np.isfinite(h) & (h >= threshold))[0]
                    detections.append(int(hit.size > 0 and hit[0] <= trace.gold_error_step))
                    if hit.size:
                        delays.append(int(hit[0] - trace.gold_error_step))
            return {
                "at_risk_auroc": auc,
                "first_error_top1": float(np.mean(top1)) if top1 else float("nan"),
                "threshold_at_5pct_correct_chain_fpr": threshold,
                "recall_by_first_error_at_threshold": (
                    float(np.mean(detections)) if detections else float("nan")
                ),
                "median_signed_delay": float(np.median(delays)) if delays else float("nan"),
                "n_labeled_chains": len(labeled_traces),
            }

        try:
            base_oof = grouped_oof_hazard(
                baseline_sequences, lengths, first_error, groups, folds=min(5, len(set(groups)))
            )
            full_oof = grouped_oof_hazard(
                ecgh_sequences, lengths, first_error, groups, folds=min(5, len(set(groups)))
            )
            base_metrics = metrics(base_oof)
            full_metrics = metrics(full_oof)
            hazard = {
                "status": "descriptive_grouped_oof",
                "baseline_features": ["logN", "position", "spread", "uncertainty"],
                "ecgh_additions": [
                    "anchor_detach", "transport_shift", "anchor_entropy",
                    "residual_energy", "residual_eff_rank", "residual_spectral_js",
                    "causal_change", "direction_jump",
                ],
                "baseline": base_metrics,
                "ecgh": full_metrics,
                "increment_at_risk_auroc": (
                    full_metrics["at_risk_auroc"] - base_metrics["at_risk_auroc"]
                ),
                "limitation": "threshold and OOF summary are audit diagnostics; use nested calibration and problem bootstrap for paper claims",
            }
        except ValueError as exc:
            hazard = {"status": "not_estimable", "reason": str(exc)}

    return {
        "status": "passed" if not errors else "failed",
        "input": os.path.abspath(path),
        "layer": int(layer),
        "n_loaded": len(traces),
        "semantic_anchor_ready": semantic,
        "fallback_only": fallback,
        "with_step_clouds": with_clouds,
        "conditional_geometry_ready": geometry_ready,
        "hazard_audit": hazard,
        "meta": meta,
        "errors": errors[:20],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--input", default=None)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--max_chains", type=int, default=0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", default="outputs/anchorflow_validation/report.json")
    parser.add_argument("--no_write", action="store_true")
    args = parser.parse_args()
    if not args.selftest and not args.input:
        parser.error("choose --selftest or --input NPZ")

    result = synthetic_validation(args.seed) if args.selftest else audit_npz(
        args.input, args.layer, args.max_chains
    )
    result = _finite_json(result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not args.no_write:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
    if result.get("status") != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
