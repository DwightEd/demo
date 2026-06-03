"""Step 12: within-problem LEARNED probe — make correct vs incorrect more separable.

The unsupervised scalars (participation / Mahalanobis) reach ~0.58-0.66
difficulty-controlled. A supervised linear probe LEARNS the direction in
activation space that best separates failing from succeeding solutions, which
typically lifts AUROC substantially IF the signal is real.

Rigor (so the lift is honest, not overfit / not difficulty):
  - Feature per chain: mean over (steps, band-layers) of the stored step vector
    (the d-dim activation), from 10 --store_vectors.
  - GROUP k-fold over PROBLEMS: the probe direction is learned on TRAIN problems
    and scored on HELD-OUT problems -> difficulty-controlled (never sees the test
    problem) AND not overfit (test problems are unseen).
  - Report within-problem (same-problem pooled-pair) AUROC on the held-out test
    predictions, vs the unsupervised band-mean participation baseline.

Probe = StandardScaler + L2-logistic (strong reg, since d >> n_incorrect).
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline

from utils.step_vector import participation_ratio, activation_entropy


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    if band == "early":
        return np.arange(0, max(1, int(L_sub * 0.3)))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def within_pair_auroc(idx_groups, feats, y_inc):
    conc = 0.0; npair = 0
    for idx in idx_groups:
        inc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        cor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if not inc or not cor:
            continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc) * len(cor)
    return (conc / npair if npair else float("nan")), npair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="multisample npz from 10 --store_vectors")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.05, help="inverse L2 strength")
    ap.add_argument("--n_seeds", type=int, default=5,
                    help="average the probe over this many group-kfold shufflings")
    ap.add_argument("--metric", default="ae", choices=["pr", "ae"],
                    help="unsupervised baseline metric for comparison")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/within_probe.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("sv_vectors_stored", np.array(False))):
        raise SystemExit("npz has no stored step vectors (need 10 --store_vectors).")
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int) if "problem_ids" in data \
        else np.arange(len(VEC))
    is_correct = data["is_correct"].astype(int)
    N = len(VEC)
    y = (is_correct == 0).astype(int)                 # 1 = incorrect (failing)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # per-chain feature: mean over (steps, band-layers) of the step vector
    X = np.full((N, d), np.nan, dtype=np.float64)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]      # (T, nc, d)
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(V.reshape(-1, d), axis=0)
    ok = np.isfinite(X).all(axis=1)
    X, y, problem_ids = X[ok], y[ok], problem_ids[ok]
    N = X.shape[0]

    # contrastive problems (for the within-problem AUROC)
    prob_to_idx = {}
    for i, p in enumerate(problem_ids):
        prob_to_idx.setdefault(int(p), []).append(i)

    print(f"Loaded {N} chains over {len(prob_to_idx)} problems "
          f"(d={d}, band={args.layer_band}, mode={args.mode}); "
          f"{int(y.sum())} incorrect / {int((1 - y).sum())} correct")

    n_steps = (data["n_steps"].astype(float)[ok] if "n_steps" in data
               else np.full(N, np.nan))

    idx_groups = [np.array(v) for v in prob_to_idx.values()
                  if any(y[v] == 1) and any(y[v] == 0)]
    n_contrastive = len(idx_groups)

    def bd(a):
        return (max(a, 1 - a), "+" if a >= 0.5 else "-") if np.isfinite(a) else (a, "?")

    # ---- baselines: unsupervised participation + LENGTH (the thing to beat) ----
    mfn = participation_ratio if args.metric == "pr" else activation_entropy
    base = np.array([mfn(X[i]) for i in range(N)])
    base_wa, npair = within_pair_auroc(idx_groups, base, y)
    len_wa, _ = within_pair_auroc(idx_groups, n_steps, y)
    b_bd, b_d = bd(base_wa); l_bd, l_d = bd(len_wa)

    # ---- LEARNED probe: group-kfold over problems, averaged over seeds ----
    def group_folds(groups, k, seed):
        uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
        fold_of = {int(g): i % k for i, g in enumerate(uniq)}
        f = np.array([fold_of[int(g)] for g in groups])
        return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]

    seeds = [args.seed + s for s in range(args.n_seeds)]
    probe_was = []
    oof_last = None
    for sd in seeds:
        oof = np.full(N, np.nan)
        for tr, te in group_folds(problem_ids, args.kfold, sd):
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced"))
            clf.fit(X[tr], y[tr])
            oof[te] = clf.predict_proba(X[te])[:, 1]
        probe_was.append(within_pair_auroc(idx_groups, oof, y)[0])
        oof_last = oof
    probe_was = np.array(probe_was)
    p_mean, p_std = float(probe_was.mean()), float(probe_was.std())
    p_d = "+" if p_mean >= 0.5 else "-"; p_bd = max(p_mean, 1 - p_mean)

    print(f"\n=== Within-problem AUROC (group-{args.kfold}fold x {len(seeds)} seeds, "
          f"held-out problems; {n_contrastive} contrastive problems, {npair} pairs) ===")
    print(f"  unsupervised participation ({args.metric}) = {b_bd:.4f}  (dir {b_d})")
    print(f"  LENGTH baseline (n_steps)                  = {l_bd:.4f}  (dir {l_d})  <- must beat this")
    print(f"  LEARNED probe (OOF)                        = {p_bd:.4f} +/- {p_std:.4f}  (dir {p_d})")
    print(f"  -> probe vs length: delta {p_bd - l_bd:+.4f}   probe vs unsup: delta {p_bd - b_bd:+.4f}")
    if p_bd > l_bd + 0.03:
        print("  -> probe BEATS the length baseline (signal is not just length).")
    else:
        print("  -> probe ~ length: the lift may be length-driven; check.")

    np.savez(args.output, oof=oof_last, base=base, y=y, problem_ids=problem_ids,
             n_steps=n_steps, probe_within_auroc=np.array(p_bd),
             probe_within_std=np.array(p_std), base_within_auroc=np.array(b_bd),
             length_within_auroc=np.array(l_bd), n_contrastive=np.array(n_contrastive))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
