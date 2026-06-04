"""Step 18: STRENGTHEN the within-problem failure signal (push past the 0.71 ceiling).

The phenomenology is settled: failure is a late-emerging, low-rank, difficulty-
orthogonal direction, but every pooled-vector probe caps at ~0.71 within-problem.
That is too weak to act on (let alone causally steer). Before any downstream use we
need a STRONGER representation. We test two principled levers, both on the already-
stored step vectors (no GPU), all evaluated honestly (GroupKFold over problems,
within-problem PAIRED AUROC; cross-problem pooled shown only for reference):

  Lever 1 - PER-PROBLEM RELATIVE representation (remove the difficulty baseline).
    The pooled probe wastes capacity on cross-problem difficulty variance (the same
    variance that inflates cross-problem AUROC to ~0.90). Subtract each solution's
    SIBLING mean (leave-self-out, unsupervised, no labels) -> the residual isolates
    "how this solution deviates within its problem". For a fixed linear direction the
    within-pair AUROC is INVARIANT to this additive per-problem shift, so any lift is
    purely from forcing the TRAINED probe onto the within-problem failure axis instead
    of the difficulty axis. Confirmation: centering should collapse cross-problem AUROC
    toward chance (difficulty drained) while within-problem rises.

  Lever 2 - EMERGENCE TRAJECTORY (use the dynamics, not one pooled vector).
    Our own result says failure EMERGES along the chain; pooling last-3 steps into one
    vector throws the trajectory away. Add delta = late_window - early_window (the
    emergence increment) as features.

Ladder (same band): baseline late-window -> +centering -> delta -> centered late (+)
centered delta -> centered + PCA-k. We report within/cross for each so the lift (and
the cross-problem collapse that proves it is difficulty-removal) is visible.
"""

from __future__ import annotations

import argparse
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


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


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def random_folds(n, k, seed):
    idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    f = np.empty(n, int); f[idx] = np.arange(n) % k
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def per_problem_center(F, problem_ids):
    """Leave-self-out per-problem centering (unsupervised, no labels).
    F'[i] = F[i] - mean(F[j] : same problem, j != i). Singletons -> global-mean
    centered (they are never contrastive, only affect probe training)."""
    F = np.asarray(F, dtype=np.float64)
    out = np.empty_like(F)
    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    gmean = F.mean(axis=0)
    for p, v in prob.items():
        v = np.array(v)
        if len(v) > 1:
            s = F[v].sum(axis=0)
            out[v] = F[v] - (s - F[v]) / (len(v) - 1)
        else:
            out[v] = F[v] - gmean
    return out


def make_clf(C, pca_k=None):
    steps = [StandardScaler()]
    if pca_k:
        steps.append(PCA(n_components=pca_k, random_state=0))
    steps.append(LogisticRegression(C=C, max_iter=2000, class_weight="balanced"))
    return make_pipeline(*steps)


def eval_features(F, y, problem_ids, idx_groups, kfold, n_seeds, seed, C, pca_k=None):
    """within-problem paired AUROC (GroupKFold OOF) + cross-problem pooled AUROC
    (random-fold OOF), averaged over seeds."""
    N = len(y)
    ok = np.isfinite(F).all(axis=1)
    win, cross = [], []
    for s in range(n_seeds):
        og = np.full(N, np.nan); orr = np.full(N, np.nan)
        for tr, te in group_folds(problem_ids, kfold, seed + s):
            tr = tr[ok[tr]]; te2 = te[ok[te]]
            if len(np.unique(y[tr])) < 2 or len(te2) == 0:
                continue
            clf = make_clf(C, pca_k).fit(F[tr], y[tr])
            og[te2] = clf.predict_proba(F[te2])[:, 1]
        for tr, te in random_folds(N, kfold, seed + s):
            tr = tr[ok[tr]]; te2 = te[ok[te]]
            if len(np.unique(y[tr])) < 2 or len(te2) == 0:
                continue
            clf = make_clf(C, pca_k).fit(F[tr], y[tr])
            orr[te2] = clf.predict_proba(F[te2])[:, 1]
        a = within_pair_auroc(idx_groups, og, y)[0]
        win.append(max(a, 1 - a))
        m = np.isfinite(orr)
        if len(np.unique(y[m])) == 2:
            cr = roc_auc_score(y[m], orr[m]); cross.append(max(cr, 1 - cr))
    return (float(np.mean(win)), float(np.std(win)),
            float(np.mean(cross)) if cross else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="all")
    ap.add_argument("--early_hi", type=float, default=0.4, help="early window = frac [0, early_hi)")
    ap.add_argument("--late_lo", type=float, default=0.6, help="late window = frac [late_lo, 1]")
    ap.add_argument("--pca_k", type=int, default=25)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/strengthen.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)         # 1 = incorrect
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    # per-solution windowed band-mean vectors: late, early, all-mean -> delta
    late = np.full((N, d), np.nan); early = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            ps = np.nanmean(V, axis=1)                             # (T, d)
        T = ps.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        lo = fr >= args.late_lo
        eo = fr < args.early_hi
        if not lo.any():
            lo = fr >= fr.max()                                   # at least last step
        if not eo.any():
            eo = fr <= fr.min()                                   # at least first step
        with np.errstate(invalid="ignore"):
            late[i] = np.nanmean(ps[lo], axis=0)
            early[i] = np.nanmean(ps[eo], axis=0)
    delta = late - early

    ok = np.isfinite(late).all(axis=1) & np.isfinite(delta).all(axis=1)
    late, early, delta = late[ok], early[ok], delta[ok]
    y, problem_ids = y[ok], problem_ids[ok]
    N = len(y)

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    late_c = per_problem_center(late, problem_ids)
    delta_c = per_problem_center(delta, problem_ids)

    print(f"Loaded {N} chains over {len(prob)} problems ({len(idx_groups)} contrastive); "
          f"d={d}, band={args.layer_band}, early<{args.early_hi} late>={args.late_lo}; "
          f"{int(y.sum())} incorrect / {int((1 - y).sum())} correct")

    kw = dict(y=y, problem_ids=problem_ids, idx_groups=idx_groups,
              kfold=args.kfold, n_seeds=args.n_seeds, seed=args.seed, C=args.C)

    ladder = [
        ("A baseline: late-window vec", lambda: eval_features(late, **kw)),
        ("B + per-problem centering", lambda: eval_features(late_c, **kw)),
        ("C emergence delta (late-early)", lambda: eval_features(delta, **kw)),
        ("D centered late (+) centered delta", lambda: eval_features(np.hstack([late_c, delta_c]), **kw)),
        (f"E centered late + PCA{args.pca_k}", lambda: eval_features(late_c, pca_k=args.pca_k, **kw)),
    ]

    print(f"\n=== Signal-strengthening ladder (within = HONEST headline; cross = reference) ===")
    print(f"{'variant':38s}  {'within':>14s}   {'cross':>7s}")
    results = {}
    for name, fn in ladder:
        w, ws, c = fn()
        results[name] = (w, ws, c)
        print(f"{name:38s}  {w:.4f} +/- {ws:.4f}   {c:7.4f}")

    base_w = results["A baseline: late-window vec"][0]
    best = max(results.items(), key=lambda kv: kv[1][0])
    print(f"\n  baseline within = {base_w:.4f}")
    print(f"  best  = {best[0]}  within = {best[1][0]:.4f}  (delta {best[1][0]-base_w:+.4f})")
    print("  Read: if centering LIFTS within AND DROPS cross toward ~0.5, the gain is "
          "genuine difficulty-removal (probe forced onto the within-problem failure axis).")

    np.savez(args.output,
             variants=np.array(list(results.keys()), dtype=object),
             within=np.array([results[k][0] for k in results]),
             within_std=np.array([results[k][1] for k in results]),
             cross=np.array([results[k][2] for k in results]),
             band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
