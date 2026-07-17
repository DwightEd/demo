from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import cluster_bootstrap_mean
from .patch_schema import SourcePatchTrace


@dataclass(frozen=True)
class SourcePatchAuditConfig:
    bootstrap: int = 2000
    seed: int = 43
    min_coverage: float = 0.80

    def validate(self) -> None:
        if int(self.bootstrap) < 0 or int(self.seed) < 0:
            raise ValueError("bootstrap and seed must be non-negative")
        if not 0.0 < float(self.min_coverage) <= 1.0:
            raise ValueError("min_coverage must lie in (0, 1]")


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Source-Specific Causal Patch Audit",
        "",
        "For each held-out alias pair and direction, donor evidence components are "
        "inserted into the recipient's cross-fitted attention-head paths. The null "
        "uses the same heads and a length-matched control source window.",
        "",
        f"- Pair directions: `{report['data']['directions']}`",
        f"- Unique pairs: `{report['data']['pairs']}`",
        f"- Extraction coverage: `{report['data']['coverage']:.4f}`",
        f"- Max replay JS: `{report['data']['max_replay_js']:.6f}`",
        "",
        "| causal test | effect | 95% CI | pairs |",
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
            f"- Causal routing supported: `{report['decision_gate']['causal_routing_supported']}`",
        ]
    )
    for name, value in report["decision_gate"]["conditions"].items():
        lines.append(f"- `{name}`: `{value}`")
    return "\n".join(lines) + "\n"


def run_source_patch_audit(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: SourcePatchAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    trace = SourcePatchTrace.load(input_path)
    evidence_vs_control_logodds = (
        trace.evidence_logodds_shift - trace.control_logodds_shift
    )
    evidence_vs_control_probability = (
        trace.evidence_donor_probability_shift
        - trace.control_donor_probability_shift
    )
    evidence_vs_random_logodds = (
        trace.evidence_logodds_shift - trace.random_head_logodds_shift
    )
    evidence_vs_random_probability = (
        trace.evidence_donor_probability_shift
        - trace.random_head_donor_probability_shift
    )
    tests = {
        "evidence_patch_moves_toward_donor_logodds": cluster_bootstrap_mean(
            trace.evidence_logodds_shift,
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 1,
        ),
        "evidence_beats_control_logodds": cluster_bootstrap_mean(
            evidence_vs_control_logodds,
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 2,
        ),
        "evidence_beats_control_donor_probability": cluster_bootstrap_mean(
            evidence_vs_control_probability,
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 3,
        ),
        "selected_heads_beat_random_heads_logodds": cluster_bootstrap_mean(
            evidence_vs_random_logodds,
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 4,
        ),
        "selected_heads_beat_random_heads_probability": cluster_bootstrap_mean(
            evidence_vs_random_probability,
            trace.pair_ids,
            draws=cfg.bootstrap,
            seed=cfg.seed + 5,
        ),
    }
    coverage = float(trace.metadata.get("coverage", 0.0))
    routing_ready = bool(
        trace.metadata.get("routing_gate", {}).get(
            "ready_for_causal_patching", False
        )
    )
    conditions = {
        "routing_gate_passed": routing_ready,
        "coverage_at_least_threshold": bool(coverage >= cfg.min_coverage),
        "evidence_patch_logodds_ci_above_zero": bool(
            tests["evidence_patch_moves_toward_donor_logodds"]["ci_low"] > 0.0
        ),
        "evidence_beats_control_logodds": bool(
            tests["evidence_beats_control_logodds"]["ci_low"] > 0.0
        ),
        "evidence_beats_control_probability": bool(
            tests["evidence_beats_control_donor_probability"]["ci_low"] > 0.0
        ),
        "selected_heads_beat_random_heads": bool(
            tests["selected_heads_beat_random_heads_logodds"]["ci_low"] > 0.0
            and tests["selected_heads_beat_random_heads_probability"]["ci_low"] > 0.0
        ),
    }
    report: dict[str, Any] = {
        "method": "causal_belief_routing_source_patch_v1",
        "data": {
            "input": str(input_path),
            "directions": int(len(trace.pair_ids)),
            "pairs": int(len(np.unique(trace.pair_ids))),
            "coverage": coverage,
            "max_replay_js": float(np.max(trace.replay_js, initial=0.0)),
            "skip_reasons": trace.metadata.get("skip_reasons", {}),
        },
        "config": {
            "bootstrap": cfg.bootstrap,
            "seed": cfg.seed,
            "min_coverage": cfg.min_coverage,
        },
        "tests": tests,
        "decision_gate": {
            "conditions": conditions,
            "causal_routing_supported": bool(all(conditions.values())),
        },
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "patch_scores.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "pair_id",
                "recipient_branch",
                "donor_branch",
                "fold",
                "selected_heads",
                "replay_js",
                "evidence_logodds_shift",
                "control_logodds_shift",
                "random_head_logodds_shift",
                "evidence_donor_probability_shift",
                "control_donor_probability_shift",
                "random_head_donor_probability_shift",
            ]
        )
        for row in range(len(trace.pair_ids)):
            writer.writerow(
                [
                    int(trace.pair_ids[row]),
                    int(trace.recipient_branches[row]),
                    int(trace.donor_branches[row]),
                    int(trace.fold_ids[row]),
                    int(trace.selected_head_counts[row]),
                    float(trace.replay_js[row]),
                    float(trace.evidence_logodds_shift[row]),
                    float(trace.control_logodds_shift[row]),
                    float(trace.random_head_logodds_shift[row]),
                    float(trace.evidence_donor_probability_shift[row]),
                    float(trace.control_donor_probability_shift[row]),
                    float(trace.random_head_donor_probability_shift[row]),
                ]
            )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    (output / "summary.md").write_text(_render_report(report), encoding="utf-8")
    return report
