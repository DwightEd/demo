"""
Q1 Pilot v2: Evaluate manifold-level geometric hypothesis on ProcessBench.

v2 evaluates trajectory-aware metrics alongside basic metrics.
Groups metrics into tiers for clearer interpretation.
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

    labels = np.concatenate([np.zeros(len(c)), np.ones(len(e))])
    if higher_means_correct:
        scores = np.concatenate([-c, -e])
    else:
        scores = np.concatenate([c, e])

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


# Metrics organized by category
METRICS = {
    # ── Trajectory-level (the real hypothesis test) ──
    "subspace_deviation":       {"higher_correct": False, "cat": "trajectory",
                                 "desc": "recon error projecting onto prior PCA subspace"},
    "subspace_angle_change":    {"higher_correct": False, "cat": "trajectory",
                                 "desc": "principal angle rotation when step added"},
    "traj_effective_rank":      {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "effective rank of full trajectory matrix"},
    "traj_rank_normed":         {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "trajectory effective rank / n_steps"},
    "traj_spectral_gap":        {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "sigma1/sigma2 of trajectory matrix"},
    "window_rank":              {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "effective rank of last-5-steps window"},
    "window_rank_normed":       {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "window rank / window_size"},
    "window_spectral_gap":      {"higher_correct": None,  "cat": "trajectory",
                                 "desc": "sigma1/sigma2 of window matrix"},

    # ── Cross-layer coherence ──
    "crosslayer_disp_align":    {"higher_correct": True,  "cat": "crosslayer",
                                 "desc": "mean pairwise cosine of cross-layer displacements"},
    "crosslayer_rank":          {"higher_correct": None,  "cat": "crosslayer",
                                 "desc": "cross-layer effective rank (single step)"},
    "crosslayer_rank_normed":   {"higher_correct": None,  "cat": "crosslayer",
                                 "desc": "cross-layer rank / n_layers"},

    # ── Running z-scores (anomaly detection) ──
    "displacement_zscore":      {"higher_correct": False, "cat": "zscore",
                                 "desc": "displacement z-score vs trajectory history"},
    "cosine_sim_zscore":        {"higher_correct": True,  "cat": "zscore",
                                 "desc": "cosine_sim z-score vs trajectory history"},
    "norm_zscore":              {"higher_correct": None,  "cat": "zscore",
                                 "desc": "norm z-score vs trajectory history"},
    "curvature_zscore":         {"higher_correct": False, "cat": "zscore",
                                 "desc": "curvature z-score vs trajectory history"},
    "subspace_deviation_zscore":{"higher_correct": False, "cat": "zscore",
                                 "desc": "subspace_deviation z-score"},
    "crosslayer_disp_align_zscore": {"higher_correct": True, "cat": "zscore",
                                 "desc": "cross-layer alignment z-score"},
    "displacement_normed_zscore":{"higher_correct": False, "cat": "zscore",
                                 "desc": "normalized displacement z-score"},
    "traj_effective_rank_zscore":{"higher_correct": None, "cat": "zscore",
                                 "desc": "trajectory rank z-score"},

    # ── Basic (v1 reference) ──
    "displacement":             {"higher_correct": False, "cat": "basic",
                                 "desc": "step-to-step L2 distance"},
    "displacement_normed":      {"higher_correct": False, "cat": "basic",
                                 "desc": "displacement / norm"},
    "cosine_sim":               {"higher_correct": True,  "cat": "basic",
                                 "desc": "consecutive step cosine"},
    "norm":                     {"higher_correct": None,  "cat": "basic",
                                 "desc": "hidden state L2 norm"},
    "curvature":                {"higher_correct": False, "cat": "basic",
                                 "desc": "angle between displacements"},
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--splits", type=str, default="gsm8k")
    parser.add_argument("--output", type=str, default="results/q1_evaluation.json")
    args = parser.parse_args()

    all_results = {}

    for split in args.splits.split(","):
        split = split.strip()
        path = os.path.join(args.results_dir, f"{split}_geometry.jsonl")
        if not os.path.exists(path):
            print(f"[WARN] {path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Evaluating: {split}")
        print(f"{'='*70}")

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
            r["category"] = info["cat"]
            split_results.append(r)

        all_results[split] = split_results

        # Print by category
        categories = ["trajectory", "crosslayer", "zscore", "basic"]
        cat_names = {
            "trajectory": "TRAJECTORY-LEVEL (manifold hypothesis)",
            "crosslayer": "CROSS-LAYER COHERENCE",
            "zscore":     "RUNNING Z-SCORES (anomaly)",
            "basic":      "BASIC (v1 reference)",
        }

        for cat in categories:
            cat_results = [r for r in split_results if r.get("category") == cat]
            if not cat_results:
                continue
            cat_results.sort(key=lambda x: -(x.get("auroc", 0) or 0))

            print(f"\n  --- {cat_names[cat]} ---")
            for r in cat_results:
                auroc = f"{r['auroc']:.4f}" if r.get('auroc') else "N/A"
                p_str = f"{r['mw_p']:.2e}" if r.get('mw_p') else "N/A"
                d_str = f"{r['cohens_d']:.3f}"
                print(f"\n  {r['metric']} ({r['description']}):")
                print(f"    Correct: {r['mean_correct']:.4f} +/- {r['std_correct']:.4f}")
                print(f"    Error:   {r['mean_error']:.4f} +/- {r['std_error']:.4f}")
                print(f"    Cohen's d={d_str}  AUROC={auroc}  MW-p={p_str}  [{r.get('direction','')}]")

    # ── Summary ──
    print(f"\n{'='*70}")
    print("Q1 HYPOTHESIS VERIFICATION SUMMARY (v2)")
    print(f"{'='*70}")
    print()
    print("Hypothesis: correct steps evolve in a constrained-but-non-degenerate subspace.")
    print("Key: *** STRONG (AUROC>=0.65, |d|>=0.3), ** MODERATE (AUROC>=0.60),")
    print("     *  weak (AUROC>=0.55), .  noise (<0.55)\n")

    for split, results in all_results.items():
        for cat in ["trajectory", "crosslayer", "zscore", "basic"]:
            cat_results = [r for r in results if r.get("category") == cat]
            if not cat_results:
                continue
            cat_results.sort(key=lambda x: -(x.get("auroc", 0) or 0))

            print(f"  [{split} / {cat}]")
            for r in cat_results:
                auroc = r.get("auroc", 0) or 0
                d = abs(r.get("cohens_d", 0))
                if auroc >= 0.65 and d >= 0.3:
                    tag = " *** STRONG"
                elif auroc >= 0.60:
                    tag = " **  moderate"
                elif auroc >= 0.55:
                    tag = " *   weak"
                else:
                    tag = " .   noise"
                print(f"    {r['metric']:35s}  AUROC={auroc:.3f}  d={d:.3f}{tag}")
            print()

    # Save
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"Full results -> {args.output}")


if __name__ == "__main__":
    main()
