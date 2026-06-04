"""Step 21: does the WITHIN-STEP token cloud carry failure signal that step-vector
pooling destroys?

Every prior feature is built on ONE pooled vector per step (step_exp), which collapses
the step's token cloud H_j in R^{n_j x d} to its (weighted) mean. But the anchor --
"error reasoning diffuses over more dimensions; correct concentrates in a low-dim subset"
-- is a statement about the DISTRIBUTION of the tokens within the step, exactly what the
mean throws away. With raw token clouds stored (10 --store_clouds), we test, per step,
cloud-geometry features measured RELATIVE TO THE HEALTHY TOKEN MANIFOLD:

  healthy token subspace U_k : PCA of all CORRECT-step tokens (train problems only),
                               standardized -- estimated from ~millions of tokens, so no
                               n<<d problem (unlike per-step effective rank M_D).

  per-step features (late-window aggregated, within-problem paired AUROC, cross-fit):
    centroid_spe : SPE of the cloud MEAN (~= the pooled step_exp SPE of 19) -> BASELINE.
    tok_spe      : mean over the cloud's tokens of out-of-manifold energy fraction
                   ||z_perp||^2/||z||^2  -- uses every token, not just the mean.
    cloud_disp   : mean token distance to the cloud centroid (healthy-standardized)
                   -- how SPREAD the token cloud is (diffuse vs concentrated).
    cloud_pr     : participation ratio of the cloud covariance spectrum (effective #
                   dims the cloud spans), via the n x n Gram (robust to n<<d). Length-
                   sensitive, so optionally cap tokens per step (--tok_cap).

Decision: if any cloud feature (tok_spe / cloud_disp / cloud_pr) beats centroid_spe (and
the pooled ceilings ~0.68-0.73), the within-step structure carried signal the pooling
lost -> richer representation is the way to strengthen. If not, pooling was sufficient.
"""

from __future__ import annotations

import argparse
import numpy as np


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


def healthy_subspace(tokens, kmax):
    """Top-kmax principal directions of standardized healthy tokens (cov eigh)."""
    H = tokens.astype(np.float32)
    mu = H.mean(0); sd = H.std(0) + 1e-6
    Hc = (H - mu) / sd
    cm = Hc.mean(0); Hc -= cm
    C = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
    _, evecs = np.linalg.eigh(C)
    B = np.ascontiguousarray(evecs[:, ::-1][:, :kmax].T)
    return mu, sd, cm, B


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="multisample npz from 10 --store_clouds")
    ap.add_argument("--cloud_layer", type=int, default=None,
                    help="which stored cloud layer to use (default: first stored)")
    ap.add_argument("--k", type=int, default=50, help="healthy subspace dim")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--tok_cap", type=int, default=0,
                    help="cap tokens per step (0 = use all) to length-match cloud_disp/pr")
    ap.add_argument("--heal_cap", type=int, default=40000,
                    help="max correct-train tokens used to fit the healthy subspace")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/token_cloud.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    if not bool(data.get("clouds_stored", np.array(False))):
        raise SystemExit("npz has no stored token clouds (need 10 --store_clouds).")
    CL = data["sv_clouds"]; SZ = data["cloud_sizes"]
    cloud_layers = data["cloud_layers"].astype(int)
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)            # 1 = incorrect
    N = len(CL)
    li = 0 if args.cloud_layer is None else int(np.where(cloud_layers == args.cloud_layer)[0][0])
    print(f"Loaded {N} solutions; cloud layers={cloud_layers.tolist()}; using layer "
          f"{cloud_layers[li]} (col {li}); k={args.k}, tok_cap={args.tok_cap or 'all'}")

    # reconstruct per-step token clouds (split concatenated tokens by per-step sizes)
    steps_per = [None] * N                                          # list of (n_j,d) arrays
    rng0 = np.random.default_rng(args.seed)
    for i in range(N):
        if CL[i] is None:
            continue
        toks = np.asarray(CL[i])[:, li, :].astype(np.float32)       # (n_tot, d)
        sizes = np.asarray(SZ[i]).astype(int)
        off = np.concatenate([[0], np.cumsum(sizes)])
        cl = []
        for t in range(len(sizes)):
            c = toks[off[t]:off[t + 1]]
            if args.tok_cap and c.shape[0] > args.tok_cap:
                sel = rng0.choice(c.shape[0], args.tok_cap, replace=False)
                c = c[sel]
            cl.append(c)
        steps_per[i] = cl

    valid = np.array([steps_per[i] is not None and len(steps_per[i]) > 0 for i in range(N)])
    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    def late_mask(T):
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        m = fr >= args.late_lo
        return m if m.any() else (fr >= fr.max())

    feat_names = ["centroid_spe", "tok_spe", "cloud_disp", "cloud_pr"]
    oof = {f: np.zeros(N) for f in feat_names}; cnt = np.zeros(N)

    for s in range(args.n_seeds):
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            # healthy token subspace from CORRECT-train tokens (subsampled)
            pool = []
            for i in tr:
                if valid[i] and y[i] == 0:
                    pool.extend(steps_per[i])
            if not pool:
                continue
            Htok = np.vstack(pool)
            if Htok.shape[0] > args.heal_cap:
                sel = np.random.default_rng(1000 + s).choice(Htok.shape[0], args.heal_cap, replace=False)
                Htok = Htok[sel]
            mu, sd, cm, B = healthy_subspace(Htok, min(args.k, Htok.shape[1]))
            k = B.shape[0]

            for i in te:
                if not valid[i]:
                    continue
                cl = steps_per[i]; T = len(cl)
                lm = late_mask(T)
                pf = {f: [] for f in feat_names}
                for t in range(T):
                    Z = (cl[t] - mu) / sd - cm                      # (n_j, d) standardized
                    n_j = Z.shape[0]
                    tot = (Z ** 2).sum(1) + 1e-12
                    ink = ((Z @ B.T) ** 2).sum(1)
                    pf["tok_spe"].append(float(np.mean((tot - ink) / tot)))
                    c = Z.mean(0)                                   # cloud centroid
                    tc = (c ** 2).sum() + 1e-12; ic = ((c @ B.T) ** 2).sum()
                    pf["centroid_spe"].append(float((tc - ic) / tc))
                    pf["cloud_disp"].append(float(np.mean(np.sqrt(((Z - c) ** 2).sum(1)))))
                    if n_j >= 2:
                        g = Z @ Z.T                                 # n x n Gram
                        ev = np.linalg.eigvalsh(g); ev = ev[ev > 1e-9]
                        pr = (ev.sum() ** 2 / (ev ** 2).sum()) if ev.size else np.nan
                    else:
                        pr = np.nan
                    pf["cloud_pr"].append(pr)
                for f in feat_names:
                    v = np.array(pf[f])[lm]
                    oof[f][i] += np.nanmean(v) if np.isfinite(v).any() else np.nan
                cnt[i] += 1

    feats = {}
    for f in feat_names:
        v = np.full(N, np.nan); m = cnt > 0
        v[m] = oof[f][m] / cnt[m]; feats[f] = v

    print(f"\n=== token-cloud features: within-problem PAIRED AUROC "
          f"({len(idx_groups)} contrastive) ===")
    res = {}
    for f in feat_names:
        a = within_pair_auroc(idx_groups, feats[f], y)[0]
        res[f] = max(a, 1 - a) if np.isfinite(a) else float("nan")
        tag = "  <- pooled baseline" if f == "centroid_spe" else ""
        print(f"  {f:13s} {res[f]:.4f}{tag}")
    base = res["centroid_spe"]
    best = max((f for f in feat_names if f != "centroid_spe"), key=lambda f: res[f])
    print(f"\n  pooled centroid baseline = {base:.4f}")
    print(f"  best cloud feature       = {best} = {res[best]:.4f}  (delta {res[best]-base:+.4f})")
    if res[best] > base + 0.01:
        print("  -> the WITHIN-STEP token structure carries signal the pooled mean lost. "
              "Richer (cloud) representation strengthens the signal.")
    else:
        print("  -> cloud features do NOT beat the pooled centroid -> step_exp pooling "
              "kept the relevant signal; the structure loss is not the bottleneck.")

    np.savez(args.output, feat_names=np.array(feat_names, dtype=object),
             within=np.array([res[f] for f in feat_names]),
             cloud_layer=np.array(int(cloud_layers[li])), k=np.array(args.k),
             tok_cap=np.array(args.tok_cap))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
