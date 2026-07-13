from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def render_markdown(
    summary: Mapping[str, Any],
    *,
    title: str = "Prompt-Controlled Residual Flow Audit",
) -> str:
    lines = [f"# {title}", ""]
    lines.append(f"- Chains: `{summary.get('n_chains')}`")
    lines.append("")
    fe = summary.get("first_error", {})
    lines.extend(["## First-Error Step Diagnosis", ""])
    lines.append(f"- Rows: `{fe.get('rows')}`")
    lines.append(f"- Positives: `{fe.get('pos')}`")
    lines.append("")
    lines.extend(["| score | n | positives | coverage | AUROC |", "|---|---:|---:|---:|---:|"])
    fe_stats = fe.get("metric_stats", {})
    for k, v in sorted(fe.get("single", {}).items(), key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -1), kv[0])):
        stats = fe_stats.get(k, {})
        cov = stats.get("coverage", float("nan"))
        auc = f"{v:.4f}" if np.isfinite(v) else "NA"
        coverage = f"{cov:.3f}" if np.isfinite(cov) else "NA"
        lines.append(f"| {k} | {stats.get('n', 0)} | {stats.get('pos', 0)} | {coverage} | {auc} |")
    response = summary.get("response", {})
    lines.extend(["", "## Response Diagnosis", ""])
    lines.append(f"- Responses: `{response.get('n')}`")
    lines.append(f"- Error responses: `{response.get('pos')}`")
    lines.extend(["", "| score | n | errors | coverage | AUROC | AUPRC |", "|---|---:|---:|---:|---:|---:|"])
    auprc = summary.get("response", {}).get("auprc", {})
    response_stats = response.get("metric_stats", {})
    for k, v in sorted(summary.get("response", {}).get("single", {}).items(), key=lambda kv: (-(kv[1] if np.isfinite(kv[1]) else -1), kv[0])):
        p = auprc.get(k, float("nan"))
        stats = response_stats.get(k, {})
        cov = stats.get("coverage", float("nan"))
        av = f"{v:.4f}" if np.isfinite(v) else "NA"
        pv = f"{p:.4f}" if np.isfinite(p) else "NA"
        coverage = f"{cov:.3f}" if np.isfinite(cov) else "NA"
        lines.append(
            f"| {k} | {stats.get('n', 0)} | {stats.get('pos', 0)} | "
            f"{coverage} | {av} | {pv} |"
        )
    ab = summary.get("response", {}).get("ablation_best", {})
    if ab:
        lines.extend(["", "## Response Ablation Best", ""])
        lines.append("Only metrics with at least 80% finite response coverage are eligible.")
        lines.extend(["", "| group | best score | n | coverage | AUROC | AUPRC |", "|---|---|---:|---:|---:|---:|"])
        for group, d in sorted(ab.items()):
            auc = d.get("auroc", float("nan"))
            pr = d.get("auprc", float("nan"))
            cov = d.get("coverage", float("nan"))
            lines.append(
                f"| {group} | {d.get('best_metric') or 'NA'} | {d.get('n', 0)} | "
                f"{cov:.3f} | {auc:.4f} | {pr:.4f} |"
                if np.isfinite(auc) and np.isfinite(pr) and np.isfinite(cov)
                else f"| {group} | {d.get('best_metric') or 'NA'} | 0 | NA | NA | NA |"
            )
    lines.extend([
        "",
        "## First-Error Ranks",
        "",
        "Ranks use expected uniform random tie-breaking.",
        "",
        "| score | n | eligible fraction | mean candidates | top1 | mean rank |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for k, d in sorted(summary.get("rank", {}).items()):
        top1 = d.get("top1", float("nan"))
        mr = d.get("mean_rank", float("nan"))
        eligible = d.get("eligible_fraction", float("nan"))
        candidates = d.get("mean_candidates", float("nan"))
        lines.append(
            f"| {k} | {d.get('n', 0)} | {eligible:.3f} | {candidates:.2f} | "
            f"{top1:.4f} | {mr:.2f} |"
            if all(np.isfinite(v) for v in (top1, mr, eligible, candidates))
            else f"| {k} | {d.get('n', 0)} | NA | NA | NA | NA |"
        )
    lines.append("")
    lines.append("## Interpretation Guardrails")
    lines.append("")
    lines.append("- `step_len` and `rel_pos` are controls, not definitions of healthy reasoning.")
    lines.append("- `random_frac` is a matched-rank negative control for prompt SVD.")
    lines.append("- `icr_*` scores require explicit attention extraction and are reported separately from hidden-only prompt-flow scores.")
    lines.append("- `geom_*` scores are cross-fitted point-cloud geometry diagnostics; they should beat length/position controls before supporting a mechanism claim.")
    lines.append("- `sd_*` scores summarize whole-chain spectral-manifold dynamics; use `full_*.npz` for ProcessBench first-error/cross-problem response claims, not same-problem paired AUROC.")
    lines.append("- `ltg_*` scores are compatibility reductions of the full layer-time field; the claim-driven event and paired tests live in `layer_time_validation.*`.")
    lines.append("- Geometry scores indicate separability, local dimensional expansion, and neighborhood rearrangement; causal claims still require interventions or patching.")
    lines.append("- A useful prompt-control signal should beat both controls and random subspaces.")
    return "\n".join(lines)


def write_step_csv(metrics: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    step_scores = np.asarray(metrics["step_scores"], dtype=np.float64)
    names = [str(x) for x in metrics["step_score_names"].tolist()]
    n_steps = np.asarray(metrics["n_steps"], dtype=np.int64)
    gold = np.asarray(metrics["gold_error_step"], dtype=np.int64)
    chain_idx = np.asarray(metrics["chain_idx"], dtype=np.int64)
    problem_id = np.asarray(metrics["problem_id"], dtype=np.int64)
    with path.open("w", encoding="utf-8", newline="") as f:
        cols = ["chain_idx", "problem_id", "step_idx", "gold_error_step", "is_gold_error_step"] + names
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for i in range(step_scores.shape[0]):
            for j in range(int(n_steps[i])):
                row = {
                    "chain_idx": int(chain_idx[i]),
                    "problem_id": int(problem_id[i]),
                    "step_idx": int(j),
                    "gold_error_step": int(gold[i]),
                    "is_gold_error_step": int(j == int(gold[i])),
                }
                for k, name in enumerate(names):
                    row[name] = float(step_scores[i, j, k])
                w.writerow(row)
