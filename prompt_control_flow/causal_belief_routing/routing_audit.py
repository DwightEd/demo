from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .charts import build_group_fold_ids
from .metrics import cluster_bootstrap_mean
from .routing_schema import EvidenceRoutingTrace


@dataclass(frozen=True)
class RoutingAuditConfig:
    folds: int = 5
    top_heads: int = 16
    bootstrap: int = 2000
    seed: int = 29

    def validate(self) -> None:
        if int(self.folds) < 2:
            raise ValueError("folds must be at least two")
        if int(self.top_heads) < 1:
            raise ValueError("top_heads must be positive")
        if int(self.bootstrap) < 0 or int(self.seed) < 0:
            raise ValueError("bootstrap and seed must be non-negative")


def cross_fit_routed_update_scores(
    trace: EvidenceRoutingTrace,
    cfg: RoutingAuditConfig,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    cfg.validate()
    fold_ids = build_group_fold_ids(trace.pair_ids, num_folds=cfg.folds, seed=cfg.seed)
    evidence_margin = trace.evidence_margin
    control_margin = trace.control_margin
    evidence_score = trace.evidence_mass * evidence_margin
    control_score = trace.control_mass * control_margin
    row_primary = np.empty(len(trace.pair_ids), dtype=np.float64)
    row_control = np.empty_like(row_primary)
    row_margin = np.empty_like(row_primary)
    selections: list[dict[str, Any]] = []
    flat_width = evidence_score.shape[1] * evidence_score.shape[2]
    top_k = min(int(cfg.top_heads), flat_width)
    for fold in sorted(np.unique(fold_ids).tolist()):
        train = fold_ids != int(fold)
        test = ~train
        train_score = evidence_score[train].mean(axis=0).reshape(-1)
        selected = np.argsort(train_score, kind="stable")[-top_k:][::-1]
        layer_positions, heads = np.unravel_index(
            selected, evidence_score.shape[1:]
        )
        test_evidence = evidence_score[test].reshape(int(test.sum()), -1)[:, selected]
        test_control = control_score[test].reshape(int(test.sum()), -1)[:, selected]
        test_margin = evidence_margin[test].reshape(int(test.sum()), -1)[:, selected]
        row_primary[test] = test_evidence.mean(axis=1)
        row_control[test] = test_control.mean(axis=1)
        row_margin[test] = test_margin.mean(axis=1)
        selections.append(
            {
                "fold": int(fold),
                "train_rows": int(train.sum()),
                "test_rows": int(test.sum()),
                "selected": [
                    {
                        "layer": int(trace.layers[layer_position]),
                        "head": int(head),
                        "train_score": float(train_score[index]),
                    }
                    for index, layer_position, head in zip(
                        selected, layer_positions, heads, strict=True
                    )
                ],
            }
        )
    return (
        {
            "fold_ids": fold_ids,
            "routed_update_score": row_primary,
            "control_routed_update_score": row_control,
            "unweighted_update_margin": row_margin,
            "evidence_over_control": row_primary - row_control,
        },
        selections,
    )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Evidence Routing Audit",
        "",
        "Heads are selected only on training alias pairs. The primary score is "
        "evidence attention mass multiplied by the held-out geometric alignment "
        "margin between the true and opposite belief updates.",
        "",
        f"- Rows: `{report['data']['rows']}`",
        f"- Pairs: `{report['data']['pairs']}`",
        f"- Layers: `{report['data']['layers']}`",
        "",
        "| test | mean | 95% CI | pairs |",
        "|---|---:|---|---:|",
    ]
    for name, metric in report["tests"].items():
        lines.append(
            f"| {name} | {metric['point']:.5f} | "
            f"[{metric['ci_low']:.5f}, {metric['ci_high']:.5f}] | {metric['groups']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Routing supported: `{report['decision_gate']['routing_supported']}`",
        ]
    )
    for name, value in report["decision_gate"]["conditions"].items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "Passing this gate establishes source-specific directional mediation, "
            "not causality. Donor-to-recipient head patching is required for the "
            "causal claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_routing_audit(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: RoutingAuditConfig,
) -> dict[str, Any]:
    trace = EvidenceRoutingTrace.load(input_path)
    scores, selections = cross_fit_routed_update_scores(trace, cfg)
    tests = {
        "routed_update_above_zero": cluster_bootstrap_mean(
            scores["routed_update_score"],
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 1,
        ),
        "evidence_over_length_matched_control": cluster_bootstrap_mean(
            scores["evidence_over_control"],
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 2,
        ),
        "true_over_opposite_update": cluster_bootstrap_mean(
            scores["unweighted_update_margin"],
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 3,
        ),
    }
    representation_ready = bool(
        trace.metadata.get("representation_gate", {}).get(
            "ready_for_routing_analysis", False
        )
    )
    conditions = {
        "representation_gate_passed": representation_ready,
        "crossfit_routed_update_ci_above_zero": bool(
            tests["routed_update_above_zero"]["ci_low"] > 0.0
        ),
        "evidence_beats_length_matched_control": bool(
            tests["evidence_over_length_matched_control"]["ci_low"] > 0.0
        ),
        "true_update_beats_opposite_update": bool(
            tests["true_over_opposite_update"]["ci_low"] > 0.0
        ),
    }
    report: dict[str, Any] = {
        "method": "causal_belief_routing_evidence_ov_v1",
        "data": {
            "input": str(input_path),
            "rows": int(len(trace.pair_ids)),
            "pairs": int(len(np.unique(trace.pair_ids))),
            "layers": [int(layer) for layer in trace.layers],
            "heads": int(trace.evidence_mass.shape[2]),
        },
        "config": {
            "folds": cfg.folds,
            "top_heads": cfg.top_heads,
            "bootstrap": cfg.bootstrap,
            "seed": cfg.seed,
        },
        "tests": tests,
        "head_selections": selections,
        "decision_gate": {
            "conditions": conditions,
            "routing_supported": bool(all(conditions.values())),
            "ready_for_causal_patching": bool(all(conditions.values())),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "row_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair_id",
                "branch",
                "fold",
                "routed_update_score",
                "control_routed_update_score",
                "unweighted_update_margin",
                "evidence_over_control",
            ]
        )
        for row in range(len(trace.pair_ids)):
            writer.writerow(
                [
                    int(trace.pair_ids[row]),
                    int(trace.branches[row]),
                    int(scores["fold_ids"][row]),
                    float(scores["routed_update_score"][row]),
                    float(scores["control_routed_update_score"][row]),
                    float(scores["unweighted_update_margin"][row]),
                    float(scores["evidence_over_control"][row]),
                ]
            )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    (output / "summary.md").write_text(_render_report(report), encoding="utf-8")
    return report
