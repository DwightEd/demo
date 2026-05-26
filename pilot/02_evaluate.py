"""
Q1 Pilot: Evaluate AFC metrics on ProcessBench.

Reads results/{split}_afc.jsonl, computes:
- AUROC for each metric (correct step vs first-error step)
- Cohen's d effect size
- Mann-Whitney U test
- Per-metric distribution plots

Decision criteria (from research_motivation_v1.md):
- At least one metric AUROC >= 0.70 and Cohen's d >= 0.5  => AFC direction confirmed
- All metrics AUROC < 0.65  => AFC hypothesis fails, need pivot
"""

import argparse
import json
import os
import numpy as np
from collections import defaultdict

try:
    from scipy import stats
    from sklearn.metrics import roc_auc_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("[WARN] scipy/sklearn not found, install them for full evaluation")


def cohens_d(group1, group2):
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std < 1e-10:
        return 0.0
    return (np.mean(group1) - np.mean(group2)) / pooled_std


def evaluate_metric(correct_vals, error_vals, metric_name):
    """Evaluate a single metric: AUROC, Cohen's d, Mann-Whitney U."""
    correct_vals = np.array(correct_vals)
    error_vals = np.array(error_vals)

    result = {
        "metric": metric_name,
        "n_correct": len(correct_vals),
        "n_error": len(error_vals),
        "mean_correct": float(np.mean(correct_vals)),
        "mean_error": float(np.mean(error_vals)),
        "std_correct": float(np.std(correct_vals)),
        "std_error": float(np.std(error_vals)),
    }

    # Cohen's d
    d = cohens_d(correct_vals, error_vals)
    result["cohens_d"] = d

    if not HAS_SKLEARN:
        return result

    # AUROC: higher metric value = more likely correct
    # For detection, we want to detect errors, so we flip:
    # label=1 for error, label=0 for correct
    # score = -metric_value (lower AFC = more likely error)
    labels = np.concatenate([np.zeros(len(correct_vals)), np.ones(len(error_vals))])
    scores = np.concatenate([-correct_vals, -error_vals])  # negate so higher score = more error

    try:
        auroc = roc_auc_score(labels, scores)
        result["auroc"] = auroc
    except ValueError:
        result["auroc"] = None

    # Mann-Whitney U
    try:
        stat, p_val = stats.mannwhitneyu(correct_vals, error_vals, alternative='two-sided')
        result["mannwhitney_U"] = float(stat)
        result["mannwhitney_p"] = float(p_val)
    except Exception:
        result["mannwhitney_U"] = None
        result["mannwhitney_p"] = None

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k",
                        help="Comma-separated splits")
    parser.add_argument("--output", type=str, default="results/q1_evaluation.json")
    args = parser.parse_args()

    metrics_to_eval = ["afc_cos", "afc_vocab_jsd", "afc_proj"]
    all_results = {}

    for split in args.splits.split(","):
        split = split.strip()
        path = os.path.join(args.results_dir, f"{split}_afc.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found, skipping")
            continue

        print(f"\n{'='*60}")
        print(f"Evaluating {split}")
        print(f"{'='*60}")

        # Collect per-step values grouped by label
        metric_correct = defaultdict(list)  # metric_name -> list of values (correct steps)
        metric_error = defaultdict(list)    # metric_name -> list of values (first-error steps)

        n_examples = 0
        n_correct_steps = 0
        n_first_error_steps = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                n_examples += 1
                for sm in record["step_metrics"]:
                    for m in metrics_to_eval:
                        val = sm.get(m)
                        if val is None:
                            continue
                        if sm["is_first_error"] == 1:
                            metric_error[m].append(val)
                        elif sm["is_error"] == 0:
                            metric_correct[m].append(val)
                        # Skip non-first error steps (after first error)

                    # Count
                    if sm["is_first_error"] == 1:
                        n_first_error_steps += 1
                    elif sm["is_error"] == 0:
                        n_correct_steps += 1

        print(f"  Examples: {n_examples}")
        print(f"  Correct steps: {n_correct_steps}")
        print(f"  First-error steps: {n_first_error_steps}")

        split_results = []
        for m in metrics_to_eval:
            if not metric_correct[m] or not metric_error[m]:
                print(f"  {m}: insufficient data")
                continue

            r = evaluate_metric(metric_correct[m], metric_error[m], m)
            split_results.append(r)

            # Print
            auroc_str = f"{r['auroc']:.4f}" if r.get('auroc') is not None else "N/A"
            p_str = f"{r['mannwhitney_p']:.2e}" if r.get('mannwhitney_p') is not None else "N/A"
            print(f"\n  {m}:")
            print(f"    Correct: mean={r['mean_correct']:.4f} +/- {r['std_correct']:.4f}")
            print(f"    Error:   mean={r['mean_error']:.4f} +/- {r['std_error']:.4f}")
            print(f"    Cohen's d = {r['cohens_d']:.4f}")
            print(f"    AUROC     = {auroc_str}")
            print(f"    M-W p     = {p_str}")

        all_results[split] = split_results

    # Decision summary
    print(f"\n{'='*60}")
    print("Q1 DECISION SUMMARY")
    print(f"{'='*60}")

    best_auroc = 0
    best_metric = None
    best_d = 0
    for split, results in all_results.items():
        for r in results:
            auroc = r.get("auroc", 0) or 0
            d = abs(r.get("cohens_d", 0))
            if auroc > best_auroc:
                best_auroc = auroc
                best_metric = f"{split}/{r['metric']}"
                best_d = d

    print(f"Best AUROC: {best_auroc:.4f} ({best_metric}), Cohen's d = {best_d:.4f}")
    if best_auroc >= 0.70 and best_d >= 0.5:
        print(">>> PASS: AFC hypothesis direction confirmed. Proceed to P2.")
    elif best_auroc >= 0.65:
        print(">>> MARGINAL: AFC shows signal but weak. Consider augmentation or layer tuning.")
    else:
        print(">>> FAIL: AFC signal too weak. Consider pivoting or checking layer selection.")

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\nFull results saved to {args.output}")


if __name__ == "__main__":
    main()
