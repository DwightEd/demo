from __future__ import annotations

from dataclasses import dataclass
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .metrics import cluster_bootstrap_mean
from .update_schema import BeliefUpdateTrace


@dataclass(frozen=True)
class BeliefUpdateAuditConfig:
    primary_layer: int
    bootstrap: int = 2000
    seed: int = 233
    max_reconstruction_p95: float = 1e-4
    max_state_replay_p95: float = 1e-3

    def validate(self) -> None:
        if int(self.primary_layer) < 1:
            raise ValueError("primary_layer must identify a raw decoder block depth")
        if int(self.bootstrap) < 0 or int(self.seed) < 0:
            raise ValueError("bootstrap and seed must be non-negative")
        if not 0.0 <= float(self.max_reconstruction_p95) < 1.0:
            raise ValueError("max_reconstruction_p95 must lie in [0, 1)")
        if not 0.0 <= float(self.max_state_replay_p95) < 1.0:
            raise ValueError("max_state_replay_p95 must lie in [0, 1)")


def _bootstrap(
    values: np.ndarray,
    pair_ids: np.ndarray,
    cfg: BeliefUpdateAuditConfig,
    offset: int,
) -> dict[str, float | int]:
    return cluster_bootstrap_mean(
        values,
        pair_ids,
        draws=cfg.bootstrap,
        seed=cfg.seed + int(offset),
    )


def _render_report(report: dict[str, Any]) -> str:
    lines = [
        "# Causal Belief Update Decomposition Audit",
        "",
        "The primary layer is preregistered. Target progress measures signed "
        "coverage of the analytic update; target error measures remaining "
        "direction-and-magnitude error. MLP results are observational update "
        "signatures until factorial patching is run.",
        "",
        f"- Rows: `{report['data']['rows']}`",
        f"- Pairs: `{report['data']['pairs']}`",
        f"- Primary layer: `{report['data']['primary_layer']}`",
        f"- Reconstruction p95: `{report['data']['reconstruction_p95']:.8f}`",
        f"- State replay p95: `{report['data']['state_replay_p95']:.8f}`",
        "",
        "| test | mean | 95% CI | pairs |",
        "|---|---:|---|---:|",
    ]
    for name, metric in report["tests"].items():
        lines.append(
            f"| {name} | {metric['point']:.5f} | "
            f"[{metric['ci_low']:.5f}, {metric['ci_high']:.5f}] | "
            f"{metric['groups']} |"
        )
    lines.extend(
        [
            "",
            "## Decision Gate",
            "",
            f"- Decomposition valid: `{report['decision_gate']['decomposition_valid']}`",
            "- Attention update signature: "
            f"`{report['decision_gate']['attention_update_signature']}`",
            f"- MLP update signature: `{report['decision_gate']['mlp_update_signature']}`",
            "- Ready for factorial patching: "
            f"`{report['decision_gate']['ready_for_factorial_patching']}`",
            "",
            "This gate does not establish that the MLP causally corrects belief. "
            "That claim requires attention-only, MLP-only, and joint patches.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_belief_update_audit(
    input_path: str | Path,
    output_dir: str | Path,
    cfg: BeliefUpdateAuditConfig,
) -> dict[str, Any]:
    cfg.validate()
    trace = BeliefUpdateTrace.load(input_path)
    matches = np.flatnonzero(trace.layers == int(cfg.primary_layer))
    if len(matches) != 1:
        raise ValueError(
            f"primary layer {cfg.primary_layer} is absent or duplicated in the artifact"
        )
    position = int(matches[0])
    primary = {
        "attention_margin": trace.attention_margin[:, position],
        "block_margin": trace.block_margin[:, position],
        "attention_progress": trace.attention_target_progress[:, position],
        "block_progress": trace.block_target_progress[:, position],
        "mlp_margin_gain": trace.mlp_margin_gain[:, position],
        "mlp_error_reduction": trace.mlp_target_error_reduction[:, position],
    }
    tests = {
        "attention_target_progress_above_zero": _bootstrap(
            primary["attention_progress"], trace.pair_ids, cfg, 1
        ),
        "block_target_progress_above_zero": _bootstrap(
            primary["block_progress"], trace.pair_ids, cfg, 2
        ),
        "attention_direction_margin_above_zero": _bootstrap(
            primary["attention_margin"], trace.pair_ids, cfg, 3
        ),
        "block_direction_margin_above_zero": _bootstrap(
            primary["block_margin"], trace.pair_ids, cfg, 4
        ),
        "mlp_incremental_direction_gain": _bootstrap(
            primary["mlp_margin_gain"], trace.pair_ids, cfg, 5
        ),
        "mlp_target_error_reduction": _bootstrap(
            primary["mlp_error_reduction"], trace.pair_ids, cfg, 6
        ),
    }
    reconstruction_p95 = float(
        np.quantile(trace.reconstruction_relative_error[:, position], 0.95)
    )
    state_replay_p95 = float(
        np.quantile(trace.state_replay_relative_error[:, position], 0.95)
    )
    representation_ready = bool(
        trace.metadata.get("representation_gate", {}).get(
            "ready_for_routing_analysis", False
        )
    )
    conditions = {
        "representation_gate_passed": representation_ready,
        "component_reconstruction_within_threshold": bool(
            reconstruction_p95 <= cfg.max_reconstruction_p95
        ),
        "state_replay_within_threshold": bool(
            state_replay_p95 <= cfg.max_state_replay_p95
        ),
        "attention_progress_ci_above_zero": bool(
            tests["attention_target_progress_above_zero"]["ci_low"] > 0.0
        ),
        "attention_direction_ci_above_zero": bool(
            tests["attention_direction_margin_above_zero"]["ci_low"] > 0.0
        ),
        "block_progress_ci_above_zero": bool(
            tests["block_target_progress_above_zero"]["ci_low"] > 0.0
        ),
        "block_direction_ci_above_zero": bool(
            tests["block_direction_margin_above_zero"]["ci_low"] > 0.0
        ),
        "mlp_direction_gain_ci_above_zero": bool(
            tests["mlp_incremental_direction_gain"]["ci_low"] > 0.0
        ),
        "mlp_error_reduction_ci_above_zero": bool(
            tests["mlp_target_error_reduction"]["ci_low"] > 0.0
        ),
    }
    decomposition_valid = bool(
        conditions["representation_gate_passed"]
        and conditions["component_reconstruction_within_threshold"]
        and conditions["state_replay_within_threshold"]
    )
    attention_signature = bool(
        decomposition_valid
        and conditions["attention_progress_ci_above_zero"]
        and conditions["attention_direction_ci_above_zero"]
    )
    mlp_signature = bool(
        decomposition_valid
        and conditions["block_progress_ci_above_zero"]
        and conditions["block_direction_ci_above_zero"]
        and conditions["mlp_direction_gain_ci_above_zero"]
        and conditions["mlp_error_reduction_ci_above_zero"]
    )
    layer_summary = [
        {
            "layer": int(layer),
            "attention_progress_mean": float(
                np.mean(trace.attention_target_progress[:, index])
            ),
            "block_progress_mean": float(np.mean(trace.block_target_progress[:, index])),
            "mlp_margin_gain_mean": float(np.mean(trace.mlp_margin_gain[:, index])),
            "mlp_error_reduction_mean": float(
                np.mean(trace.mlp_target_error_reduction[:, index])
            ),
            "reconstruction_p95": float(
                np.quantile(trace.reconstruction_relative_error[:, index], 0.95)
            ),
            "state_replay_p95": float(
                np.quantile(trace.state_replay_relative_error[:, index], 0.95)
            ),
        }
        for index, layer in enumerate(trace.layers)
    ]
    report: dict[str, Any] = {
        "method": "causal_belief_update_decomposition_audit_v1",
        "data": {
            "input": str(input_path),
            "rows": int(len(trace.pair_ids)),
            "pairs": int(len(np.unique(trace.pair_ids))),
            "layers": [int(layer) for layer in trace.layers],
            "primary_layer": int(cfg.primary_layer),
            "reconstruction_p95": reconstruction_p95,
            "state_replay_p95": state_replay_p95,
        },
        "config": {
            "primary_layer": int(cfg.primary_layer),
            "bootstrap": int(cfg.bootstrap),
            "seed": int(cfg.seed),
            "max_reconstruction_p95": float(cfg.max_reconstruction_p95),
            "max_state_replay_p95": float(cfg.max_state_replay_p95),
        },
        "tests": tests,
        "layer_summary": layer_summary,
        "decision_gate": {
            "conditions": conditions,
            "decomposition_valid": decomposition_valid,
            "attention_update_signature": attention_signature,
            "mlp_update_signature": mlp_signature,
            "ready_for_factorial_patching": bool(
                attention_signature and mlp_signature
            ),
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
                *primary.keys(),
                "reconstruction_error",
                "state_replay_error",
            ]
        )
        for row in range(len(trace.pair_ids)):
            writer.writerow(
                [
                    int(trace.pair_ids[row]),
                    int(trace.branches[row]),
                    *[float(values[row]) for values in primary.values()],
                    float(trace.reconstruction_relative_error[row, position]),
                    float(trace.state_replay_relative_error[row, position]),
                ]
            )
    with (output / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
    (output / "summary.md").write_text(_render_report(report), encoding="utf-8")
    return report
