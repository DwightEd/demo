"""Decompose the step-vector norm into MASSIVE-activation dims vs the BULK.

Massive activations (Sun et al. 2024) live in a few FIXED dimensions that are
enormous for (almost) every token. We identify them data-driven: per dim, the
median |value| over correct-chain step vectors; the top-k dims are the massive
set (no hard-coded indices, works for any model). Then per step:
    norm_massive = || z[massive dims] ||      norm_bulk = || z[other dims] ||
and we test, at the step level (ProcessBench first-error vs good steps), the
AUROC of each -- raw and within length buckets (length held ~constant) -- to see
whether the norm signal lives in the massive dims or the bulk, and whether it
survives controlling for length.

Needs an npz with stored step vectors at the layer: extract_features
  --store_step_vectors --sv_layers <layer>.
"""

from __future__ import annotations

import argparse
import numpy as np


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
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


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def bucket_auroc(score, y, nt, nb=5):
    edges = np.quantile(nt, np.linspace(0, 1, nb + 1)); edges[-1] += 1
    b = np.clip(np.digitize(nt, edges[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb
        ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        a = bdir(auroc(score[m], y[m]))
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--k_massive", type=int, default=5, help="# massive dims")
    ap.add_argument("--per_token", action="store_true",
                    help="also compare pooled norm vs MEAN per-token norm (needs "
                         "tokgeom): low pooled + normal per-token = cancellation/"
                         "diffuseness; low per-token = individually weak tokens.")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("no step vectors; re-extract with --store_step_vectors --sv_layers ...")
    svl = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files else z["layers_used"])]
    if args.layer not in svl:
        raise SystemExit(f"layer {args.layer} not in stored sv_layers {svl}")
    li = svl.index(args.layer)
    SV = z["stepvec"]; SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    # optional per-token norm from tokgeom (to separate cancellation vs weak tokens)
    TG = lit = ni = None
    if args.per_token:
        if "tokgeom" not in z.files or z["tokgeom"][0] is None:
            raise SystemExit("--per_token needs tokgeom (extract without --no_token_geom)")
        TG = z["tokgeom"]
        tlayers = [int(x) for x in z["layers_used"]]
        lit = tlayers.index(args.layer)
        ni = [str(x) for x in z["geom_feature_names"]].index("norm")

    Z, NT, Y, H, MTN = [], [], [], [], []
    for i in range(len(SV)):
        sv = np.asarray(SV[i], np.float32)
        if sv.ndim != 3 or sv.shape[1] <= li:
            continue
        rng = np.asarray(SR[i], int); k = int(ges[i]); corr = (k < 0)
        a0 = int(rng[0, 0]); T = sv.shape[0]
        tg = np.asarray(TG[i], float) if TG is not None else None
        for j in range(T):
            zj = sv[j, li, :]
            if not np.isfinite(zj).all():
                continue
            if corr or j < k:
                y, keep = 0, True
            elif j == k:
                y, keep = 1, True
            else:
                keep = False
            if keep:
                Z.append(zj); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); Y.append(y); H.append(corr)
                if tg is not None:
                    lo, hi = int(rng[j, 0]) - a0, int(rng[j, 1]) - a0 + 1
                    lo, hi = max(0, lo), min(tg.shape[0], hi)
                    MTN.append(np.nanmean(tg[lo:hi, lit, ni]) if hi > lo else np.nan)
    Z = np.asarray(Z, np.float64); NT = np.asarray(NT, float); Y = np.asarray(Y, int); H = np.asarray(H, bool)
    MTN = np.asarray(MTN, float) if MTN else None

    # identify massive dims from correct-chain step vectors (median |value| per dim)
    med = np.median(np.abs(Z[H]), axis=0)
    massive = np.argsort(med)[::-1][:args.k_massive]
    other = np.setdiff1d(np.arange(Z.shape[1]), massive)
    print(f"layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())} | d={Z.shape[1]}")
    print(f"massive dims (top-{args.k_massive} by median |val| on correct steps): {sorted(massive.tolist())}")
    print(f"  their median|val|: {np.round(med[massive],1).tolist()}  vs overall median {med.mean():.2f}")

    n_tot = np.linalg.norm(Z, axis=1)
    n_mas = np.linalg.norm(Z[:, massive], axis=1)
    n_bulk = np.linalg.norm(Z[:, other], axis=1)
    frac = (n_mas ** 2) / np.maximum(n_tot ** 2, 1e-12)     # fraction of energy in massive dims
    print(f"  massive energy fraction: error-step {frac[Y==1].mean():.3f} vs "
          f"good {frac[Y==0].mean():.3f}")

    print(f"\n{'component':14s} {'err mean':>9s} {'good mean':>9s} {'raw AUROC':>10s} {'bucket AUROC':>13s}")
    comps = [("norm_total", n_tot), ("norm_massive", n_mas), ("norm_bulk", n_bulk),
             ("massive_frac", frac)]
    if MTN is not None:
        coherence = n_tot / np.maximum(MTN, 1e-9)        # pooled / mean-token = alignment
        comps += [("mean_tok_norm", MTN), ("coherence(pool/tok)", coherence)]
    for name, v in comps:
        print(f"{name:20s} {v[Y==1].mean():9.2f} {v[Y==0].mean():9.2f} "
              f"{bdir(auroc(v, Y)):10.3f} {bucket_auroc(v, Y, NT):13.3f}")
    print("\nbucket AUROC = within-length-bucket (length ~fixed). >0.5 => real beyond length.")
    if MTN is not None:
        print("per-token read: if error has LOW pooled but NORMAL mean_tok_norm (so LOW "
              "coherence) -> tokens cancel = diffuseness (anchor revived). If mean_tok_norm "
              "also LOW -> individual tokens are weaker.")


if __name__ == "__main__":
    main()
