"""
Q1 Pilot: Evaluate geometric hypothesis on ProcessBench.

Core hypothesis: correct reasoning trajectories evolve in a "constrained but
non-degenerate" subspace.

Testable predictions:
- Correct steps have STABLE displacement (low variance)
- Correct steps have HIGH cosine similarity (smooth evolution)
- Correct steps have BOUNDED effective rank (constrained but non-degenerate)
- Error steps show SPIKES in curvature, displacement, or rank collapse/explosion

Metrics: AUROC, Cohen's d, Mann-Whitney U for each geometric feature.
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict

try:
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    print("[WARN] scipy/sklearn not found. Install: pip install scipy scikit-learn")


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = np.var(g1, ddof=1), np.var(g2, ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(g1) - np.mean(g2)) / pooled if pooled > 1e-10 else 0.0


def evaluate_metric(correct_vals, error_vals, name, higher_means_correct=True):
    """Evaluate one metric. Returns dict with stats."""
    c, e = np.array(correct_vals), np.array(error_vals)
    r = {
        "metric": name,
        "n_correct": len(c), "n_error": len(e),
        "mean_correct": float(np.mean(c)), "std_correct": float(np.std(c)),
        "mean_error": float(np.mean(e)), "std_error": float(np.std(e)),
        "cohens_d": cohens_d(c, e),
    }

    if not HAS_STATS or len(c) < 2 or len(e) < 2:
        return r

    # AUROC: we want to detect errors
    labels = np.concatenate([np.zeros(len(c)), np.ones(len(e))])
    if higher_means_correct:
        scores = np.concatenate([-c, -e])  # negate: lower metric -> more likely error
    else:
        scores = np.concatenate([c, e])  # higher metric -> more likely error

    try:
        r["auroc"] = roc_auc_score(labels, scores)
    except ValueError:
        r["auroc"] = None

    try:
        stat, p = stats.mannwhitneyu(c, e, alternative='two-sided')
        r["mw_p"] = float(p)
    except Exception:
        r["mw_p"] = None

    return r


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k")
    parser.add_argument("--output", type=str, default="results/q1_evaluation.json")
    args = parser.parse_args()

    # Metrics to evaluate and whether higher value means "more correct"
    METRICS = {
        "displacement":       {"higher_correct": False, "desc": "step-to-step L2 distance"},
        "displacement_normed":{"higher_correct": False, "desc": "displacement / norm"},
        "cosine_sim":         {"higher_correct": True,  "desc": "consecutive step cosine"},
        "norm":               {"higher_correct": None,  "desc": "hidden state L2 norm"},
        "curvature":          {"higher_correct": False, "desc": "angle between displacements"},
        "effective_rank":     {"higher_correct": None,  "desc": "cross-layer effective rank"},
    }

    all_results = {}

    for split in args.splits.split(","):
        split = split.strip()
        path = os.path.join(args.results_dir, f"{split}_geometry.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating: {split}")
        print(f"{'='*60}")

        correct = defaultdict(list)
        error = defaultdict(list)
        n_ex = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                n_ex += 1
                for sm in rec["step_metrics"]:
                    for m in METRICS:
                        val = sm.get(m)
                        if val is None:
                            continue
                        if sm["is_first_error"] == 1:
                            error[m].append(val)
                        elif sm["is_error"] == 0:
                            correct[m].append(val)

        n_correct_steps = len(correct.get("norm", []))
        n_error_steps = len(error.get("norm", []))
        print(f"  Examples: {n_ex}")
        print(f"  Correct steps: {n_correct_steps}, First-error steps: {n_error_steps}")

        split_results = []
        for m, info in METRICS.items():
            if not correct[m] or not error[m]:
                continue

            hc = info["higher_correct"]
            if hc is None:
                # Try both directions, pick better AUROC
                r1 = evaluate_metric(correct[m], error[m], m, higher_means_correct=True)
                r2 = evaluate_metric(correct[m], error[m], m, higher_means_correct=False)
                a1 = r1.get("auroc", 0) or 0
                a2 = r2.get("auroc", 0) or 0
                r = r1 if a1 >= a2 else r2
                r["direction"] = "higher=correct" if a1 >= a2 else "higher=error"
            else:
                r = evaluate_metric(correct[m], error[m], m, higher_means_correct=hc)
                r["direction"] = "higher=correct" if hc else "higher=error"

            r["description"] = info["desc"]
            split_results.append(r)

            auroc = f"{r['auroc']:.4f}" if r.get('auroc') else "N/A"
            p_str = f"{r['mw_p']:.2e}" if r.get('mw_p') else "N/A"
            d_str = f"{r['cohens_d']:.3f}"
            print(f"\n  {m} ({info['desc']}):")
            print(f"    Correct: {r['mean_correct']:.4f} +/- {r['std_correct']:.4f}")
            print(f"    Error:   {r['mean_error']:.4f} +/- {r['std_error']:.4f}")
            print(f"    Cohen's d={d_str}  AUROC={auroc}  MW-p={p_str}  [{r.get('direction','')}]")

        all_results[split] = split_results

    # ── Summary ──
    print(f"\n{'='*60}")
    print("Q1 HYPOTHESIS VERIFICATION SUMMARY")
    print(f"{'='*60}")
    print()
    print("Hypothesis: correct steps evolve in a constrained-but-non-degenerate subspace.")
    print("Evidence for each metric:\n")

    for split, results in all_results.items():
        print(f"  [{split}]")
        for r in sorted(results, key=lambda x: -(x.get("auroc", 0) or 0)):
            auroc = r.get("auroc", 0) or 0
            d = abs(r.get("cohens_d", 0))
            tag = ""
            if auroc >= 0.65 and d >= 0.3:
                tag = " *** SIGNAL"
            elif auroc >= 0.55:
                tag = " *  weak"
            print(f"    {r['metric']:25s}  AUROC={auroc:.3f}  d={d:.3f}{tag}")
        print()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full results -> {args.output}")


if __name__ == "__main__":
    main()
