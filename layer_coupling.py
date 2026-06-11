"""Cross-layer coupling -- the "spectral field" falsification test (step-level).

Decision is at the STEP level, ProcessBench process-level labels:
  positive = the gold first-error step (one per error chain)
  negative = every step of a process-correct chain (label==-1) + the steps BEFORE
             the first error in an error chain.  Post-error steps are EXCLUDED.

For a base metric M (e.g. cloud_D), each step j has a layer profile m_j in R^L
(M across the L layers).  Healthy subspace V* = top-k principal directions of the
profiles of CORRECT-chain steps, cross-fit by chain (GroupKFold).  Coordination
  c_j = ||V*^T m_j||^2 / ||m_j||^2.

Ladder of step-level AUROCs (the spectral-field narrative requires (iv-a) to beat
BOTH (ii') and (iii) -- significantly, not on the point estimate):
  (i)   single best layer        : best single m_j[l]
  (ii)  pooling                  : mean_l / max_l m_j
  (ii') per-layer z then aggregate: each layer z-scored vs correct (cross-fit),
                                    aggregated max/mean -- strong "layers independent" baseline
  (iii) linear bag               : logistic on the L components of m_j
  (iv-a)(iii) + c                : the JUDGMENT (c is the only quadratic term beyond the linear bag)
  (iv-b)(iii) + c + std          : + nonlinear std (separated so the gain attributable to c is clean)
  ctrl  (iii) + random quad       : (iii)+||R m_j||^2/||m_j||^2, R random k x L -- rules out
                                    "any quadratic feature raises AUROC"

Significance: chain-paired bootstrap CI for AUROC differences.
Direct hypothesis test: mean c at first-error vs correct steps + standalone AUROC of (-c).

Needs an 8-layer ProcessBench npz: extract_features --layers 2,6,10,14,18,22,26,30 --cloud_eff_rank
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def auroc(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def best_dir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def top_k_subspace(P, k):
    """Top-k right singular vectors (R^L) of profile matrix P (n, L), uncentered."""
    _, _, Vt = np.linalg.svd(P, full_matrices=False)
    return Vt[:min(k, Vt.shape[0])]                       # (k, L)


def coord(M, Vt, eps=1e-12):
    """c = ||Vt M^T||^2 / ||M||^2 per row of M (n, L)."""
    num = (M @ Vt.T) ** 2                                 # (n, k)
    den = (M ** 2).sum(1)                                 # (n,)
    return num.sum(1) / np.maximum(den, eps)


def smooth_basis(L, k):
    """Top-k low-frequency DCT-II modes on the layer axis (k, L). This is the
    GENERIC residual-stream smoothness null: adjacent layers correlate, so every
    chain's layer profile is smooth. c built on THIS basis measures 'how smooth',
    not 'reasoning-specific coordination'. The healthy-V* c must beat it."""
    n = np.arange(L)
    B = np.stack([np.cos(np.pi * (n + 0.5) * m / L) for m in range(k)])
    return B / np.linalg.norm(B, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--metric", default="cloud_D")
    ap.add_argument("--k", type=int, default=3, help="V* subspace dimension")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--n_boot", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    geom_names = [str(x) for x in z["geom_feature_names"]]
    cloud_names = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    if args.metric == "grad":                              # GRADIENT spectral field
        if not bool(z.get("grad_stored", np.array(False))):
            raise SystemExit("no gradprof; re-extract with --grad_profile")
        SRC, fi = z["gradprof"], None
        layers = [int(x) for x in z["grad_layers"]]
    elif args.metric in geom_names:
        SRC, fi = z["stepgeom"], geom_names.index(args.metric)
        layers = [int(x) for x in z["layers_used"]]
    elif args.metric in cloud_names and bool(z.get("cloud_stored", np.array(False))):
        SRC, fi = z["stepcloud"], cloud_names.index(args.metric)
        layers = [int(x) for x in z["layers_used"]]
    else:
        raise SystemExit(f"metric {args.metric} not in grad / geom {geom_names} / cloud {cloud_names}")
    L = len(layers)
    ges = z["gold_error_step"].astype(int)

    # ---- build step-level dataset (m_j, y, chain group, healthy flag) ----
    rows_m, rows_y, rows_g, rows_h = [], [], [], []
    for i in range(len(SRC)):
        if SRC[i] is None:
            continue
        g = np.asarray(SRC[i], float)                     # (T, L) for this metric
        if fi is not None and g.ndim == 3:
            g = g[:, :, fi]
        T = g.shape[0]
        k = int(ges[i])
        correct_chain = (k < 0)
        for j in range(T):
            if not np.isfinite(g[j]).all():
                continue
            if correct_chain:
                y, keepit = 0, True
            elif j < k:
                y, keepit = 0, True
            elif j == k:
                y, keepit = 1, True
            else:
                keepit = False                            # post-error: exclude
            if keepit:
                rows_m.append(g[j]); rows_y.append(y); rows_g.append(i)
                rows_h.append(correct_chain)
    M = np.asarray(rows_m, float); Y = np.asarray(rows_y, int)
    G = np.asarray(rows_g, int); H = np.asarray(rows_h, bool)
    print(f"file: {args.npz} | metric {args.metric} | layers {layers}")
    print(f"steps: {len(Y)} | positive(first-error): {int(Y.sum())} | "
          f"healthy(correct-chain) steps: {int(H.sum())} | chains: {len(np.unique(G))}")

    rng = np.random.default_rng(args.seed)
    n = len(Y)
    oof = {s: np.full(n, np.nan) for s in
           ["iii", "iv_a", "iv_b", "ctrl", "iv_smooth", "iip_max", "iip_mean", "c"]}
    Vs = smooth_basis(L, args.k)                          # generic smoothness null (fixed)
    gkf = GroupKFold(args.folds)
    for tr, te in gkf.split(M, Y, G):
        Htr = H[tr]
        Pc = M[tr][Htr]                                   # correct-chain step profiles
        if Pc.shape[0] <= args.k:
            continue
        mu, sd = Pc.mean(0), Pc.std(0) + 1e-9
        # CENTER on the correct-step mean: c measures how much the DEVIATION
        # (m-mu) lies in the healthy deviation-subspace (else c is mean-dominated).
        Vt = top_k_subspace(Pc - mu, args.k)
        R = top_k_subspace(rng.standard_normal((max(args.k * 4, args.k + 5), L)), args.k)
        c_tr, c_te = coord(M[tr] - mu, Vt), coord(M[te] - mu, Vt)
        cr_tr, cr_te = coord(M[tr] - mu, R), coord(M[te] - mu, R)
        cs_tr, cs_te = coord(M[tr] - mu, Vs), coord(M[te] - mu, Vs)   # smoothness null
        std_tr = M[tr].std(1); std_te = M[te].std(1)
        oof["c"][te] = c_te
        # per-layer z, aggregate (cross-fit calibration)
        zte = (M[te] - mu) / sd
        oof["iip_max"][te] = np.abs(zte).max(1)
        oof["iip_mean"][te] = np.abs(zte).mean(1)

        def fit(Xtr, Xte):
            p = make_pipeline(StandardScaler(),
                              LogisticRegression(C=args.C, max_iter=2000))
            if len(np.unique(Y[tr])) < 2:
                return np.full(len(te), np.nan)
            p.fit(Xtr, Y[tr]); return p.predict_proba(Xte)[:, 1]
        oof["iii"][te] = fit(M[tr], M[te])
        oof["iv_a"][te] = fit(np.c_[M[tr], c_tr], np.c_[M[te], c_te])
        oof["iv_b"][te] = fit(np.c_[M[tr], c_tr, std_tr], np.c_[M[te], c_te, std_te])
        oof["ctrl"][te] = fit(np.c_[M[tr], cr_tr], np.c_[M[te], cr_te])
        oof["iv_smooth"][te] = fit(np.c_[M[tr], cs_tr], np.c_[M[te], cs_te])

    # ---- point AUROCs ----
    a_i = best_dir(max(auroc(M[:, l], Y) for l in range(L)))
    a_ii = max(best_dir(auroc(M.mean(1), Y)), best_dir(auroc(M.max(1), Y)))
    res = {
        "(i) single best layer": a_i,
        "(ii) pool mean/max": a_ii,
        "(ii') per-layer z agg": max(best_dir(auroc(oof["iip_max"], Y)),
                                     best_dir(auroc(oof["iip_mean"], Y))),
        "(iii) linear bag": auroc(oof["iii"], Y),
        "(iv-a) iii + c": auroc(oof["iv_a"], Y),
        "(iv-b) iii + c + std": auroc(oof["iv_b"], Y),
        "ctrl iii + rand-quad": auroc(oof["ctrl"], Y),
        "ctrl iii + smooth-c": auroc(oof["iv_smooth"], Y),
    }
    print(f"\n{'scheme':26s} {'AUROC':>7s}")
    print("-" * 36)
    for k, v in res.items():
        print(f"{k:26s} {v:7.3f}")

    # ---- chain-paired bootstrap for the decisive gaps ----
    chains = np.unique(G)

    def boot_gap(sa, sb):
        diffs = []
        for _ in range(args.n_boot):
            cb = rng.choice(chains, size=len(chains), replace=True)
            mask = np.concatenate([np.where(G == c)[0] for c in cb])
            diffs.append(auroc(oof[sa][mask], Y[mask]) - auroc(oof[sb][mask], Y[mask]))
        d = np.array(diffs)
        return float(np.nanmean(d)), float(np.nanpercentile(d, 2.5)), float(np.nanpercentile(d, 97.5))

    print(f"\n=== decisive gaps (chain-paired bootstrap, n={args.n_boot}) ===")
    for sa, sb, name in [("iv_a", "iii", "(iv-a) - (iii)"),
                         ("iv_a", "ctrl", "(iv-a) - ctrl(rand-quad)"),
                         ("iv_a", "iv_smooth", "(iv-a) - ctrl(smooth-c)")]:
        m, lo, hi = boot_gap(sa, sb)
        sig = "SIGNIFICANT" if lo > 0 else ("neg" if hi < 0 else "ns")
        print(f"  {name:26s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}]  {sig}")

    # ---- direct hypothesis: c_j lower at first-error step? ----
    c = oof["c"]
    ce, cc = c[(Y == 1) & np.isfinite(c)], c[(Y == 0) & np.isfinite(c)]
    print(f"\n=== coordination c direct test ===")
    print(f"  mean c  error-step {np.mean(ce):.4f}  vs correct-step {np.mean(cc):.4f}  "
          f"(diff {np.mean(ce)-np.mean(cc):+.4f})")
    print(f"  standalone AUROC of (-c): {auroc(-c, Y):.3f}  "
          f"(>0.5 => c LOWER at error step, anchor holds)")
    print("\nverdict: (iv-a) must significantly beat (iii) AND (ii'), with c lower at "
          "error, for the spectral-field claim. else -> bag of per-layer features.")


if __name__ == "__main__":
    main()
