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

    # ---- unsupervised baseline: participation of the chain-mean vector ----
    mfn = participation_ratio if args.metric == "pr" else activation_entropy
    base = np.array([mfn(X[i]) for i in range(N)])

    # ---- LEARNED probe: group-kfold over problems ----
    gkf = GroupKFold(n_splits=args.kfold)
    oof = np.full(N, np.nan)                            # out-of-fold probe scores
    for tr, te in gkf.split(X, y, groups=problem_ids):
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=args.C, max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    idx_groups = [np.array(v) for v in prob_to_idx.values()
                  if any(y[v] == 1) and any(y[v] == 0)]
    n_contrastive = len(idx_groups)

    base_wa, npair = within_pair_auroc(idx_groups, base, y)
    probe_wa, _ = within_pair_auroc(idx_groups, oof, y)

    def bd(a):
        return (max(a, 1 - a), "+" if a >= 0.5 else "-") if np.isfinite(a) else (a, "?")
    b_bd, b_d = bd(base_wa); p_bd, p_d = bd(probe_wa)

    print(f"\n=== Within-problem AUROC (group-{args.kfold}fold, held-out problems; "
          f"{n_contrastive} contrastive problems, {npair} pairs) ===")
    print(f"  unsupervised participation ({args.metric}) = {b_bd:.4f}  (dir {b_d})")
    print(f"  LEARNED probe (OOF)                        = {p_bd:.4f}  (dir {p_d})")
    print(f"  -> probe {'LIFTS' if p_bd > b_bd + 0.02 else 'does not lift'} "
          f"separability (delta {p_bd - b_bd:+.4f}).")
    print("  (probe scored on UNSEEN problems -> difficulty-controlled + not overfit.)")

    np.savez(args.output, oof=oof, base=base, y=y, problem_ids=problem_ids,
             probe_within_auroc=np.array(p_bd), base_within_auroc=np.array(b_bd),
             n_contrastive=np.array(n_contrastive))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
