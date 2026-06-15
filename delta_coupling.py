"""CROSS-LAYER DELTA verdict: does the per-block residual-stream WRITE carry
cross-layer structure beyond a flat per-layer linear aggregate?

Inputs: an extract_features npz produced with `--layers all --cloud_eff_rank
--cloud_delta` (so stepdelta = norm_delta, resultant_delta per step x layer, and
stepcloud carries cumulative `resultant`, stepgeom carries cumulative `norm`).

Design (per the confirmed plan):
  (1) ORTHOGONALITY CHECK: corr(resultant_delta@L, cumulative norm@L) over steps.
      If ~0 -> Delta direction-concentration is an independent new axis; if high ->
      it is the same ~0.8 norm signal in disguise (re-think what it measures).
  (2) V* / mu / sigma fit on FULLY-CORRECT chains ONLY (k<0), GroupKFold by chain.
      Error-chain correct-prefix steps are NEVER in V* (the optimism-leak fix); they
      stay as ordinary logistic negatives and we also report c on them as a byproduct
      (does c dip before the error step? = anchoring pre-weakening).
  (3) THREE verdicts, each "linear bag (per-layer profile) vs +c (coordination)":
        A norm_delta profile     bag vs bag+c
        B resultant_delta profile bag vs bag+c
        C joint [norm_delta, resultant_delta] bag vs bag+c   <- the complete verdict
      c = ||V*^T p~||^2 / ||p~||^2 in [0,1] = fraction of the (standardized) cross-layer
      profile lying in the healthy coordination subspace (1 - SPE). High c = the step's
      cross-layer write pattern looks coordinated like a healthy step.
  Two evaluations of every OOF score: raw pooled AUROC, and within-chain-z AUROC
  (chain-centered -> within-chain component only). chain-paired bootstrap on the
  bag+c minus bag delta.

Verdict: +c beats bag (CI above 0) for some group -> cross-layer COORDINATION adds
signal beyond marginal per-layer aggregation. All groups ns -> the cross-layer
organization of the WRITE is a non-finding (note: this only buries direction/energy
of the *delta write*, not norm's cross-layer or module-coupling).

Usage:
  python delta_coupling.py <delta.npz> --layer 14 --rank 3 --folds 5
"""
from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    _HAVE_SK = True
except Exception:
    _HAVE_SK = False


def auroc(score, y):
    m = np.isfinite(score)
    s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort")
    r = np.empty(len(s))
    sr = s[o]
    i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1
        i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def wcz_scores(score, chain):
    """within-chain z-score a flat score vector keyed by chain id."""
    out = np.full_like(score, np.nan, dtype=float)
    for g in np.unique(chain):
        idx = np.where(chain == g)[0]
        v = score[idx]
        fin = np.isfinite(v)
        if fin.sum() < 2:
            continue
        sd = v[fin].std()
        if sd < 1e-12:
            continue
        out[idx[fin]] = (v[fin] - v[fin].mean()) / sd
    return out


def fit_logreg(X, y):
    lr = LogisticRegression(max_iter=2000, C=1.0)
    lr.fit(X, y)
    return lr


def coordination_c(P, mu, sd, Vstar):
    """standardized profile -> c = ||V*^T p~||^2/||p~||^2 in [0,1] (1 - SPE)."""
    Pt = (P - mu) / sd
    Pt = np.nan_to_num(Pt, nan=0.0)
    proj = Pt @ Vstar                         # (n, r)
    num = (proj ** 2).sum(1)
    den = (Pt ** 2).sum(1)
    return np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14, help="layer for the orthogonality check")
    ap.add_argument("--rank", type=int, default=3, help="V* subspace rank")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_finite", type=float, default=0.8,
                    help="keep a layer if this frac of steps have finite delta")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if not _HAVE_SK:
        raise SystemExit("needs scikit-learn (LogisticRegression, GroupKFold).")

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z["cloud_delta_stored"]) if "cloud_delta_stored" in z.files else True:
        print("WARN: cloud_delta_stored is False / missing -- did you extract with --cloud_delta?")
    dnames = [str(x) for x in z["cloud_delta_names"]]
    gnames = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]
    SD = z["stepdelta"]; SG = z["stepgeom"]        # stepcloud NOT needed (minimal Δ config)
    SR = z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    nidx = dnames.index("norm_delta"); ridx = dnames.index("resultant_delta")
    cum_norm_i = gnames.index("norm")              # orthogonality uses cumulative norm (always stored)
    li_chk = layers.index(args.layer)

    # ---- pick layers with enough finite delta (drops layer 0 etc.) ----
    fin_frac = np.zeros(len(layers))
    tot = 0
    for i in range(len(SD)):
        sd = np.asarray(SD[i], float)
        fin_frac += np.isfinite(sd[:, :, nidx]).sum(0)
        tot += sd.shape[0]
    fin_frac /= max(tot, 1)
    keep_li = [j for j in range(len(layers)) if fin_frac[j] >= args.min_finite]
    print(f"file: {args.npz}")
    print(f"layers_used={layers}  -> delta-valid layers (>= {args.min_finite} finite): "
          f"{[layers[j] for j in keep_li]} ({len(keep_li)} of {len(layers)})")

    # ---- build per-step table ----
    rows_norm, rows_res = [], []          # profiles over keep_li
    ocn, orr = [], []                     # orthogonality: cumulative norm, resultant_delta @ layer
    ylab, chain, is_corr_chain, is_prefix = [], [], [], []
    for i in range(len(SD)):
        sd = np.asarray(SD[i], float); sg = np.asarray(SG[i], float)
        k = int(ges[i]); correct = (k < 0); T = sd.shape[0]
        if not correct and (k < 0 or k >= T):
            continue
        keep_steps = range(T) if correct else range(0, k + 1)   # post-error excluded
        for j in keep_steps:
            rows_norm.append(sd[j, keep_li, nidx])
            rows_res.append(sd[j, keep_li, ridx])
            ocn.append(sg[j, li_chk, cum_norm_i]); orr.append(sd[j, li_chk, ridx])
            lab = 1 if (not correct and j == k) else 0
            ylab.append(lab); chain.append(i)
            is_corr_chain.append(correct)
            is_prefix.append((not correct) and j < k)
    Xn = np.asarray(rows_norm, float); Xr = np.asarray(rows_res, float)
    y = np.asarray(ylab, int); chain = np.asarray(chain, int)
    is_corr_chain = np.asarray(is_corr_chain, bool); is_prefix = np.asarray(is_prefix, bool)
    print(f"steps: {len(y)} (pos {int(y.sum())}, neg {int((y==0).sum())}); "
          f"chains {len(np.unique(chain))} "
          f"(correct {int(np.unique(chain[is_corr_chain]).size)}, "
          f"error {int(np.unique(chain[~is_corr_chain]).size)})")

    # ---- (1) orthogonality check ----
    ocn = np.asarray(ocn, float); orr = np.asarray(orr, float)
    m = np.isfinite(ocn) & np.isfinite(orr)
    r_orth = np.corrcoef(ocn[m], orr[m])[0, 1] if m.sum() > 2 else float("nan")
    print(f"\n(1) orthogonality @L{args.layer}: corr(resultant_delta, cumulative norm) = "
          f"{r_orth:+.3f}  (~0 => independent new axis; |r|>~0.5 => same signal in disguise)")

    # ---- (3) three verdicts via GroupKFold (V*/mu/sigma on fully-correct train chains) ----
    groups = {"A norm_delta": [Xn], "B resultant_delta": [Xr],
              "C joint": [Xn, Xr]}
    gkf = GroupKFold(n_splits=args.folds)
    rng = np.random.default_rng(args.seed)

    print(f"\n{'group':18s} {'eval':8s} {'bag':>7s} {'bag+c':>7s} {'delta':>7s} {'95% CI':>18s}")
    cprefix_summary = {}
    for gname, blocks in groups.items():
        Xfull = np.hstack(blocks)                       # (n, L*nblocks)
        oof_bag = np.full(len(y), np.nan); oof_bagc = np.full(len(y), np.nan)
        c_all = np.full(len(y), np.nan)
        for tr, te in gkf.split(Xfull, y, groups=chain):
            # V*/mu/sigma from fully-correct TRAIN chains only
            hmask = tr[is_corr_chain[tr]]
            Ph = Xfull[hmask]
            mu = np.nanmean(Ph, 0); sd_ = np.nanstd(Ph, 0); sd_[sd_ < 1e-9] = 1.0
            Pht = np.nan_to_num((Ph - mu) / sd_)
            # V* = top-r right singular vectors of healthy standardized profiles
            r = min(args.rank, Pht.shape[1], max(1, Pht.shape[0] - 1))
            _, _, Vt = np.linalg.svd(Pht - Pht.mean(0), full_matrices=False)
            Vstar = Vt[:r].T                              # (D, r)
            c_all_fold = coordination_c(Xfull, mu, sd_, Vstar)
            c_all[te] = c_all_fold[te]
            # logistic bag (standardized profile) and bag+c, trained on ALL train steps
            Xs = np.nan_to_num((Xfull - mu) / sd_)
            lr_b = fit_logreg(Xs[tr], y[tr])
            oof_bag[te] = lr_b.decision_function(Xs[te])
            Xc = np.column_stack([Xs, c_all_fold])
            lr_c = fit_logreg(Xc[tr], y[tr])
            oof_bagc[te] = lr_c.decision_function(Xc[te])
        # byproduct: c on error-chain prefix vs correct-chain steps (anchoring pre-weakening)
        cprefix_summary[gname] = (np.nanmean(c_all[is_prefix]),
                                  np.nanmean(c_all[is_corr_chain]))
        for evname, transform in (("raw", lambda s: s),
                                  ("wcz", lambda s: wcz_scores(s, chain))):
            sb = transform(oof_bag); sc_ = transform(oof_bagc)
            ab, ac = auroc(sb, y), auroc(sc_, y)
            deltas = np.empty(args.boot)
            uch = np.unique(chain)
            for b in range(args.boot):
                samp = rng.choice(uch, len(uch), replace=True)
                idx = np.concatenate([np.where(chain == g)[0] for g in samp])
                deltas[b] = auroc(sc_[idx], y[idx]) - auroc(sb[idx], y[idx])
            lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
            print(f"{gname:18s} {evname:8s} {ab:7.3f} {ac:7.3f} {ac-ab:7.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]")

    print("\n(2) anchoring byproduct: mean c on error-chain PREFIX steps vs correct-chain steps")
    print("    (c lower on prefix => coordination weakens BEFORE the first error step):")
    for gname, (cp, cc) in cprefix_summary.items():
        print(f"    {gname:18s} prefix {cp:.3f}  vs  correct {cc:.3f}  (diff {cp-cc:+.3f})")
    print("\nverdict: +c CI above 0 for any group => cross-layer COORDINATION of the block")
    print("write adds signal beyond per-layer linear aggregation. All ns => the cross-layer")
    print("organization of the delta write is a non-finding (does not bury norm's cross-layer")
    print("or module-coupling, which are separate propositions).")


if __name__ == "__main__":
    main()
