"""Trajectory dynamics: how far step t moves relative to the PRECEDING context.

For each reasoning step vector z_t (exp-pooled, per layer), using only the chain's
own history z_0..z_{t-1}, compute:

  disp_cent     ||z_t - mean(z_<t)||              displacement from running centroid
  disp_cent_rel disp_cent / ||mean(z_<t)||        relative displacement
  disp_prev     ||z_t - z_{t-1}||                 jump from the previous step
  cos_cent      cos(z_t, mean(z_<t))              alignment with accumulated context
                                                  (low = diverging from the context)
  curv          1 - cos(z_t-z_{t-1}, z_{t-1}-z_{t-2})   direction change (curvature)

Then two views:
  (1) chain-level within-problem AUROC (mean + late-window of each series);
  (2) step-level localization at the ProcessBench gold error step (does the
      displacement spike exactly where the reasoning goes wrong?).

Post-hoc from stepvec -- needs extract_features.py --store_step_vectors. No re-extraction.
"""

from __future__ import annotations

import argparse
import numpy as np

FEATS = ["disp_cent", "disp_cent_rel", "disp_prev", "cos_cent", "curv"]


def chain_series(Z):
    """Z (T,d) step vectors at one layer -> dict feat -> (T,) per-step series."""
    T = Z.shape[0]
    out = {k: np.full(T, np.nan) for k in FEATS}
    for t in range(T):
        if t >= 1:
            c = Z[:t].mean(0)
            nc = float(np.linalg.norm(c))
            zt = float(np.linalg.norm(Z[t]))
            out["disp_cent"][t] = float(np.linalg.norm(Z[t] - c))
            out["disp_prev"][t] = float(np.linalg.norm(Z[t] - Z[t - 1]))
            if nc > 0:
                out["disp_cent_rel"][t] = out["disp_cent"][t] / nc
                if zt > 0:
                    out["cos_cent"][t] = float(Z[t] @ c / (zt * nc))
        if t >= 2:
            d1, d2 = Z[t] - Z[t - 1], Z[t - 1] - Z[t - 2]
            n1, n2 = float(np.linalg.norm(d1)), float(np.linalg.norm(d2))
            if n1 > 0 and n2 > 0:
                out["curv"][t] = 1.0 - float(d1 @ d2 / (n1 * n2))
    return out


def within_auroc(score, y, pid):
    conc = tie = npair = 0.0
    for p in np.unique(pid):
        m = (pid == p) & np.isfinite(score)
        se, sc = score[m & (y == 1)], score[m & (y == 0)]
        if se.size and sc.size:
            d = se[:, None] - sc[None, :]
            conc += (d > 0).sum(); tie += (d == 0).sum(); npair += d.size
    return (conc + 0.5 * tie) / npair if npair else float("nan")


def late_mean(x, frac=0.25):
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan")
    k = max(1, int(round(x.size * frac)))
    return float(x[-k:].mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=16)
    ap.add_argument("--format_ok_only", action="store_true")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("needs --store_step_vectors npz")
    layers = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files
                               else z["layers_used"])]
    if args.layer not in layers:
        raise SystemExit(f"layer {args.layer} not in stored step-vector layers {layers}")
    li = layers.index(args.layer)
    SV = z["stepvec"]
    pid = z["problem_ids"].astype(int)
    y = (z["is_correct"].astype(int) == 0).astype(int)
    ges = z["gold_error_step"].astype(int) if "gold_error_step" in z.files \
        else np.full(len(SV), -1)
    keep = np.ones(len(SV), bool)
    if args.format_ok_only and "format_ok" in z.files:
        keep = z["format_ok"].astype(int) == 1
        print(f"[format_ok=1 subset: {int(keep.sum())}/{len(keep)}]")

    N = len(SV)
    series = [None] * N
    for i in range(N):
        sv = np.asarray(SV[i], np.float32)
        if sv.ndim == 3 and sv.shape[1] > li:
            series[i] = chain_series(sv[:, li, :])
    # length control: tokens per step (so curv/disp can be compared to length)
    feat_list = list(FEATS)
    if "step_token_ranges" in z.files:
        SR = z["step_token_ranges"]
        for i in range(N):
            if series[i] is not None:
                r = np.asarray(SR[i], int)
                if r.shape[0] == len(series[i][FEATS[0]]):
                    series[i]["ntok"] = (r[:, 1] - r[:, 0] + 1).astype(float)
        feat_list.append("ntok")

    # (1) chain-level within-problem AUROC
    print(f"file: {args.npz} | layer {args.layer} | chains {int(keep.sum())} | "
          f"error(answer) {int(y[keep].sum())}")
    print(f"\n=== chain-level within-AUROC (mean / late) ===")
    print(f"{'feature':14s} {'mean':>8s} {'late':>8s}")
    for f in feat_list:
        cm = np.array([np.nanmean(series[i][f]) if (series[i] and f in series[i])
                       else np.nan for i in range(N)])
        cl = np.array([late_mean(series[i][f]) if (series[i] and f in series[i])
                       else np.nan for i in range(N)])
        print(f"{f:14s} {within_auroc(cm[keep], y[keep], pid[keep]):8.3f} "
              f"{within_auroc(cl[keep], y[keep], pid[keep]):8.3f}")

    # (2) step-level localization at the gold error step (ProcessBench)
    usable = [i for i in range(N) if keep[i] and series[i] is not None
              and 0 <= ges[i] < len(series[i][FEATS[0]])]
    if usable:
        print(f"\n=== error-step localization (within error chains, n={len(usable)}) ===")
        print(f"{'feature':14s} {'loc_auroc':>9s} {'z_at_err':>8s}")
        for f in feat_list:
            conc = tie = npair = 0.0
            zes = []
            for i in usable:
                s = series[i].get(f)
                if s is None:
                    continue
                k = int(ges[i])
                if not np.isfinite(s[k]):
                    continue
                others = s[np.arange(len(s)) != k]
                others = others[np.isfinite(others)]
                if others.size:
                    conc += (s[k] > others).sum(); tie += (s[k] == others).sum()
                    npair += others.size
                    sd = np.nanstd(s)
                    if sd > 0:
                        zes.append((s[k] - np.nanmean(s)) / sd)
            loc = (conc + 0.5 * tie) / npair if npair else float("nan")
            print(f"{f:14s} {loc:9.3f} {np.mean(zes) if zes else float('nan'):8.2f}")
        print("loc_auroc>0.5 => feature HIGHER at the gold error step.")


if __name__ == "__main__":
    main()
