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

try:
    from tqdm import tqdm
except ImportError:                      # graceful fallback if tqdm is absent
    def tqdm(it, **kw):
        return it


def _r(x, nd=3):
    """Round for JSON; NaN/inf -> None."""
    return round(float(x), nd) if x is not None and np.isfinite(x) else None


def exact_top_k(Xc, k):
    """EXACT top-k principal axes via LAPACK partial eigendecomposition of the
    d x d covariance (scipy eigh subset_by_index) -- only the top k eigenpairs,
    not all d. Returns (k, d). Use when you want the exact subspace; needs scipy.
    """
    from scipy.linalg import eigh
    d = Xc.shape[1]
    C = Xc.T @ Xc                                       # (d, d) gemm, BLAS-fast
    _, V = eigh(C, subset_by_index=[d - k, d - 1])      # ascending, top-k only
    return np.ascontiguousarray(V[:, ::-1].T)           # (k, d), descending


def randomized_top_k(Xc, k, oversample=10, n_power=2, seed=0):
    """Top-k right singular vectors of Xc (M,d) via randomized SVD (Halko et al.).

    Returns (k, d). Avoids the full d-dim eigendecomposition of X^T X, which is
    very slow for d=4096; we only ever need the top ~100 components.
    """
    M, d = Xc.shape
    p = min(k + oversample, d, M)
    rng = np.random.default_rng(seed)
    Y = Xc @ rng.standard_normal((d, p))         # (M, p) range sketch
    for _ in range(n_power):                     # power iterations for accuracy
        Y = Xc @ (Xc.T @ Y)
    Q, _ = np.linalg.qr(Y)                        # (M, p)
    B = Q.T @ Xc                                  # (p, d)
    _, _, Vt = np.linalg.svd(B, full_matrices=False)
    return np.ascontiguousarray(Vt[:k])


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
    ap.add_argument("--json", action="store_true", help="also print results as JSON.")
    ap.add_argument("--exact", action="store_true",
                    help="exact top-k eigendecomposition (scipy) instead of the "
                         "default randomized SVD; for cross-checking the subspace.")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("npz has no stored step vectors. Re-extract with "
                         "--store_step_vectors.")
    layers = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files
                               else z["layers_used"])]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in stored step-vector layers {layers}")
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

    ks = [int(x) for x in args.ks.split(",")]
    max_k = max(ks)

    # Build each fold's healthy subspace ONCE (top max_k components, via the
    # covariance eigendecomposition -- far cheaper than a full SVD per k).
    fold_sub = {}
    for f in tqdm(range(args.folds), desc="build subspaces"):
        tr = keep & correct & (foldof != f)
        X = [mats[i] for i in np.where(tr)[0] if mats[i].size]
        if not X:
            fold_sub[f] = None
            continue
        X = np.concatenate(X, 0).astype(np.float32)
        X = X[np.isfinite(X).all(1)]
        if X.shape[0] <= max_k:
            fold_sub[f] = None
            continue
        mu = X.mean(0)
        Vt = (exact_top_k(X - mu, max_k) if args.exact
              else randomized_top_k(X - mu, max_k))   # (max_k, d) top components
        fold_sub[f] = (mu, Vt)

    # Per chain/step: total energy + cumulative explained energy of top-1..max_k,
    # so SPE at any k is one slice (no recomputation).
    chain_steps = [None] * N
    for i in tqdm(range(N), desc="project chains"):
        if not keep[i] or mats[i].size == 0:
            continue
        sub = fold_sub.get(foldof[i])
        if sub is None:
            continue
        mu, Vt = sub
        m = mats[i].astype(np.float32)
        steps = []
        for t in range(m.shape[0]):
            zc = m[t] - mu
            tot = float(zc @ zc)
            if tot <= 1e-12:
                steps.append(None)
                continue
            proj = Vt @ zc
            steps.append((tot, np.cumsum(proj * proj)))
        chain_steps[i] = steps

    results = []
    for k in ks:
        chain_spe = np.full(N, np.nan)
        step_spe = [None] * N
        for i in range(N):
            if chain_steps[i] is None:
                continue
            ss = np.full(len(chain_steps[i]), np.nan)
            for t, sp in enumerate(chain_steps[i]):
                if sp is None:
                    continue
                tot, csum = sp
                expl = csum[min(k, len(csum)) - 1]
                ss[t] = (tot - expl) / tot
            step_spe[i] = ss
            if np.isfinite(ss).any():
                chain_spe[i] = np.nanmean(ss)

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
        results.append({"k": k, "within_ans": _r(wa), "cross_ans": _r(ca),
                        "within_strict": _r(ws), "meanSPE_err": _r(me, 4),
                        "meanSPE_cor": _r(mc, 4), "loc_auroc": _r(loc)})

    print("\nwithin_ans>0.5 => error chains leak MORE outside the healthy subspace "
          "(anchor holds). meanSPE_err vs _cor shows direction; loc_auroc>0.5 => "
          "SPE higher at the gold error step.")

    if args.json:
        import json
        meta = {"file": args.npz, "layer": args.layer, "chains": int(N),
                "n_error_answer": int(y_ans[keep].sum()),
                "format_ok_only": bool(args.format_ok_only),
                "healthy_label": args.healthy_label}
        print("\n=== JSON ===")
        print(json.dumps({"meta": meta, "results": results}, indent=2))


if __name__ == "__main__":
    main()
