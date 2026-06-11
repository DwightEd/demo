"""Baseline-normalization sweep for the per-step geometry.

For each reasoning step vector z_j (exp-pooled, per layer), recompute the
activation-degree geometry on z_j MINUS a baseline reference, for four baselines:

  none      : z_j                 (raw, current)
  first(x0) : z_j - z_0           (deviation from the 1st generated step)
  prev(x-1) : z_j - z_{j-1}       (step-to-step drift)
  question  : z_j - q             (deviation from the exp-pooled question tokens)

then aggregate per chain (mean over steps) and report within-problem AUROC
(error vs correct, answer-only) for each (baseline, feature). Tests whether
removing a common-mode baseline makes the geometric signal cleaner/stronger.

Needs an npz from extract_features.py --store_step_vectors (stores stepvec + qvec).
"""

from __future__ import annotations

import argparse
import numpy as np

from features.geometry import vector_features, GEOM_FEATURE_NAMES


def within_auroc(score, y, pid):
    conc = tie = npair = 0.0
    for p in np.unique(pid):
        m = (pid == p) & np.isfinite(score)
        se, sc = score[m & (y == 1)], score[m & (y == 0)]
        if se.size and sc.size:
            d = se[:, None] - sc[None, :]
            conc += (d > 0).sum(); tie += (d == 0).sum(); npair += d.size
    return (conc + 0.5 * tie) / npair if npair else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=16, help="layer NUMBER in layers_used")
    ap.add_argument("--massive_m", type=int, default=4)
    ap.add_argument("--format_ok_only", action="store_true")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("npz has no step vectors. Re-extract with --store_step_vectors.")
    layers = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files
                               else z["layers_used"])]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in stored step-vector layers {layers}")
    li = layers.index(args.layer)

    SV = z["stepvec"]; QV = z["qvec"]
    pid = z["problem_ids"].astype(int)
    y = (z["is_correct"].astype(int) == 0).astype(int)            # error = 1
    keep = np.ones(len(SV), bool)
    if args.format_ok_only and "format_ok" in z.files:
        keep = z["format_ok"].astype(int) == 1
        print(f"[format_ok=1 subset: {int(keep.sum())}/{len(keep)}]")

    feats = list(GEOM_FEATURE_NAMES)
    baselines = ["none", "first", "prev", "question"]
    # chain-level feature per (baseline, feature)
    N = len(SV)
    acc = {bl: {f: np.full(N, np.nan) for f in feats} for bl in baselines}

    for i in range(N):
        if not keep[i]:
            continue
        sv = np.asarray(SV[i], np.float32)                        # (T, L, d)
        if sv.ndim != 3 or sv.shape[1] <= li:
            continue
        Z = sv[:, li, :]                                          # (T, d) step vectors
        T = Z.shape[0]
        q = np.asarray(QV[i], np.float32)[li] if QV[i] is not None else None
        for bl in baselines:
            rows = []
            for j in range(T):
                zj = Z[j]
                if not np.isfinite(zj).all():
                    continue
                if bl == "none":
                    zp = zj
                elif bl == "first":
                    zp = zj - Z[0]
                elif bl == "prev":
                    if j == 0:
                        continue
                    zp = zj - Z[j - 1]
                else:                                            # question
                    if q is None or not np.isfinite(q).all():
                        continue
                    zp = zj - q
                rows.append(vector_features(zp, massive_m=args.massive_m))
            if rows:
                for f in feats:
                    acc[bl][f][i] = np.nanmean([r[f] for r in rows])

    pid_k, y_k = pid[keep], y[keep]
    print(f"file: {args.npz} | layer {args.layer} | chains {int(keep.sum())} | "
          f"error(answer) {int(y_k.sum())}")
    print(f"\n{'feature':12s} " + " ".join(f"{bl:>9s}" for bl in baselines))
    print("-" * (12 + 10 * len(baselines)))
    for f in feats:
        cells = []
        for bl in baselines:
            wa = within_auroc(acc[bl][f][keep], y_k, pid_k)
            cells.append(f"{wa:9.3f}")
        print(f"{f:12s} " + " ".join(cells))
    print("\nwithin-AUROC (error vs correct, answer-only). >0.5 error higher, "
          "<0.5 correct higher; |dev from 0.5| = strength.")


if __name__ == "__main__":
    main()
