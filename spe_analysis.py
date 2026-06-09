"""SPE / subspace-leakage test of the core anchor -- the SUBSPACE sense of
"error occupies more dimensions", not the single-vector participation sense.

Build a k-dim "healthy" subspace U_k from CORRECT chains' per-step vectors
(cross-fit by problem so the subspace never sees the test chain), then for every
step vector z measure the Jackson-Mudholkar Q-statistic / squared prediction
error -- the fraction of energy OUTSIDE the healthy subspace:

    SPE(z) = || (z-mu) - U_k U_k^T (z-mu) ||^2 / || z-mu ||^2

Anchor prediction: error steps leak more (higher SPE) = occupy directions that
correct reasoning does not. Well-conditioned (thousands of correct step vectors),
so it avoids the n<<d degeneracy of per-chain / per-step rank.

Reports, per layer and per k:
  within / cross AUROC of chain-mean SPE (error vs correct), format_ok control,
  the mean SPE direction, and (ProcessBench) per-step SPE localization at the
  gold error step.

Needs an npz produced with extract_features.py --store_step_vectors.
"""

from __future__ import annotations

import argparse
import numpy as np


def healthy_subspace(X, k):
    """PCA subspace of rows of X (M,d): return (mu, Vt_k) with Vt_k (k,d)."""
    mu = X.mean(0)
    Xc = X - mu
    # economy SVD; right singular vectors are the principal axes
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return mu, Vt[:k]


def spe(z, mu, Vt_k, eps=1e-12):
    zc = z - mu
    tot = float(zc @ zc)
    if tot <= eps:
        return np.nan
    proj = zc @ Vt_k.T                      # (k,)
    return float(tot - proj @ proj) / tot   # residual energy fraction


def auroc(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    if (yy == 1).sum() == 0 or (yy == 0).sum() == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    r = np.empty(len(s)); sr = s[order]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[order[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    npos = (yy == 1).sum(); nneg = (yy == 0).sum()
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def within_auroc(score, y, pid):
    conc = tie = npair = 0.0
    for p in np.unique(pid):
        m = (pid == p) & np.isfinite(score)
        se, sc = score[m & (y == 1)], score[m & (y == 0)]
        if se.size and sc.size:
            diff = se[:, None] - sc[None, :]
            conc += (diff > 0).sum(); tie += (diff == 0).sum(); npair += diff.size
    return (conc + 0.5 * tie) / npair if npair else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=16, help="layer NUMBER (must be in layers_used)")
    ap.add_argument("--ks", default="10,20,50,100")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--format_ok_only", action="store_true")
    ap.add_argument("--healthy_label", default="answer", choices=["answer", "strict"],
                    help="which chains define 'correct' for the healthy subspace.")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("npz has no stored step vectors. Re-extract with "
                         "--store_step_vectors.")
    layers = [int(x) for x in z["layers_used"]]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in {layers}")
    li = layers.index(args.layer)
    SV = z["stepvec"]
    pid = z["problem_ids"].astype(int)
    y_ans = (z["is_correct"].astype(int) == 0).astype(int)          # error=1 (answer-only)
    y_str = (z["is_correct_strict"].astype(int) == 0).astype(int)
    correct = (z["is_correct"].astype(int) == 1) if args.healthy_label == "answer" \
        else (z["is_correct_strict"].astype(int) == 1)
    ges = z["gold_error_step"].astype(int) if "gold_error_step" in z.files \
        else np.full(len(SV), -1)
    keep = np.ones(len(SV), bool)
    if args.format_ok_only and "format_ok" in z.files:
        keep = z["format_ok"].astype(int) == 1
        print(f"[format_ok=1 subset: {int(keep.sum())}/{len(keep)}]")

    N = len(SV)
    # per-chain list of (T,d) step-vector matrices at the chosen layer
    mats = []
    for i in range(N):
        sv = np.asarray(SV[i], np.float32)         # (T,L,d)
        m = sv[:, li, :] if sv.ndim == 3 else np.empty((0, 0))
        mats.append(m)

    uniq = np.unique(pid)
    fold = {p: idx % args.folds for idx, p in enumerate(uniq)}
    foldof = np.array([fold[p] for p in pid])

    print(f"file: {args.npz} | layer {args.layer} | chains {N} | "
          f"error(answer) {int(y_ans.sum())} | folds {args.folds}")
    print(f"{'k':>5s} {'within_ans':>10s} {'cross_ans':>9s} {'within_str':>10s} "
          f"{'meanSPE_err':>11s} {'meanSPE_cor':>11s} {'loc_auroc':>9s}")
    print("-" * 72)

    for k in [int(x) for x in args.ks.split(",")]:
        chain_spe = np.full(N, np.nan)
        # per-step SPE kept for localization
        step_spe = [None] * N
        for f in range(args.folds):
            tr = keep & correct & (foldof != f)
            X = [mats[i] for i in np.where(tr)[0] if mats[i].size]
            if not X:
                continue
            X = np.concatenate(X, 0)
            X = X[np.isfinite(X).all(1)]
            if X.shape[0] <= k:
                continue
            mu, Vt = healthy_subspace(X, k)
            for i in np.where(keep & (foldof == f))[0]:
                m = mats[i]
                if m.size == 0:
                    continue
                s = np.array([spe(m[t], mu, Vt) for t in range(m.shape[0])])
                step_spe[i] = s
                chain_spe[i] = np.nanmean(s)

        wa = within_auroc(chain_spe[keep], y_ans[keep], pid[keep])
        ca = auroc(chain_spe[keep], y_ans[keep])
        ws = within_auroc(chain_spe[keep], y_str[keep], pid[keep])
        me = np.nanmean(chain_spe[keep & (y_ans == 1)])
        mc = np.nanmean(chain_spe[keep & (y_ans == 0)])
        # per-step localization at gold error step (error chains)
        conc = tie = npair = 0.0
        for i in range(N):
            if not keep[i] or ges[i] < 0 or step_spe[i] is None:
                continue
            s = step_spe[i]
            if ges[i] >= len(s) or not np.isfinite(s[ges[i]]):
                continue
            others = s[np.arange(len(s)) != ges[i]]
            others = others[np.isfinite(others)]
            if others.size:
                conc += (s[ges[i]] > others).sum(); tie += (s[ges[i]] == others).sum()
                npair += others.size
        loc = (conc + 0.5 * tie) / npair if npair else float("nan")
        print(f"{k:5d} {wa:10.3f} {ca:9.3f} {ws:10.3f} {me:11.4f} {mc:11.4f} {loc:9.3f}")

    print("\nwithin_ans>0.5 => error chains leak MORE outside the healthy subspace "
          "(anchor holds). meanSPE_err vs _cor shows direction; loc_auroc>0.5 => "
          "SPE higher at the gold error step.")


if __name__ == "__main__":
    main()
