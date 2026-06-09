"""Lay out the WHOLE reasoning trace -- no chain-level pooling.

Produces, for a chosen feature set, both text tables and PNG figures:

  (1) traj_*.png   : mean feature vs normalized step position, correct vs error
                     (with SEM bands) -- where along the chain do they diverge?
  (2) aligned_*.png: ProcessBench only -- per-step z aligned at the gold error
                     step (offset 0), window +/-W.
  (3) examples_*.png: a few individual chains, per-token U_D laid out along the
                     full token axis with step boundaries and the gold error step
                     marked, plus the linear trend (the paper's Fig.5 view).

Per-step features come from localize_signal.per_step_series (U_D, U_C per-step
means + geometry stepgeom columns). Text tables are printed so results can be
read without opening the PNGs.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from localize_signal import per_step_series, zscore

DEFAULT_FEATS = ["U_D", "U_C", "norm_L16", "norm_L8",
                 "ed_half_L8", "anom_k10_L8", "e90_L8", "ae_robust_L16"]


def trajectories(series, names, is_err, feats, nb=10):
    idx = {n: i for i, n in enumerate(names)}
    res = {}
    for f in feats:
        if f not in idx:
            continue
        c = idx[f]
        acc = {0: np.zeros(nb), 1: np.zeros(nb)}
        sq = {0: np.zeros(nb), 1: np.zeros(nb)}
        cnt = {0: np.zeros(nb), 1: np.zeros(nb)}
        for i in range(len(series)):
            col = series[i][:, c]
            T = len(col)
            for j in range(T):
                v = col[j]
                if not np.isfinite(v):
                    continue
                b = min(nb - 1, int(nb * j / max(1, T)))
                g = int(is_err[i])
                acc[g][b] += v
                sq[g][b] += v * v
                cnt[g][b] += 1
        out = {}
        for g in (0, 1):
            m = np.where(cnt[g] > 0, acc[g] / np.maximum(cnt[g], 1), np.nan)
            var = np.where(cnt[g] > 0, sq[g] / np.maximum(cnt[g], 1) - m * m, np.nan)
            sem = np.sqrt(np.maximum(var, 0) / np.maximum(cnt[g], 1))
            out[g] = (m, sem)
        res[f] = out
    return res


def aligned_curves(series, names, ges, feats, win=4):
    idx = {n: i for i, n in enumerate(names)}
    res = {}
    for f in feats:
        if f not in idx:
            continue
        c = idx[f]
        s = np.zeros(2 * win + 1)
        n = np.zeros(2 * win + 1)
        for i in range(len(series)):
            k = int(ges[i])
            if k < 0 or k >= series[i].shape[0]:
                continue
            zc = zscore(series[i][:, c])
            for d in range(-win, win + 1):
                p = k + d
                if 0 <= p < len(zc) and np.isfinite(zc[p]):
                    s[d + win] += zc[p]
                    n[d + win] += 1
        res[f] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--feats", default=",".join(DEFAULT_FEATS))
    ap.add_argument("--nb", type=int, default=10)
    ap.add_argument("--win", type=int, default=4)
    ap.add_argument("--outdir", default="output/trajectories")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    tag = os.path.splitext(os.path.basename(args.npz))[0]

    z = np.load(args.npz, allow_pickle=True)
    names, series = per_step_series(z)
    is_err = (z["is_correct"].astype(int) == 0)
    ges = z["gold_error_step"].astype(int) if "gold_error_step" in z.files \
        else np.full(len(series), -1)
    feats = [f for f in args.feats.split(",") if f in names]
    n_err = int(is_err.sum())
    print(f"file: {args.npz} | chains: {len(series)} | "
          f"correct: {len(series)-n_err} | error: {n_err}")

    # ---- (1) trajectories ----
    res = trajectories(series, names, is_err, feats, args.nb)
    print(f"\n=== correct vs error trajectory ({args.nb} normalized-position bins) ===")
    for f in feats:
        mc, _ = res[f][0]
        me, _ = res[f][1]
        print(f"\n{f}")
        print("  bin    : " + " ".join(f"{b:6d}" for b in range(args.nb)))
        print("  correct: " + " ".join(f"{v:6.2f}" for v in mc))
        print("  error  : " + " ".join(f"{v:6.2f}" for v in me))
        print("  err-cor: " + " ".join(f"{e-c:+6.2f}" for c, e in zip(mc, me)))

    ncol = 2
    nrow = (len(feats) + ncol - 1) // ncol
    fig, ax = plt.subplots(nrow, ncol, figsize=(11, 3 * nrow), squeeze=False)
    x = (np.arange(args.nb) + 0.5) / args.nb
    for ix, f in enumerate(feats):
        a = ax[ix // ncol][ix % ncol]
        for g, lab, col in [(0, "correct", "tab:blue"), (1, "error", "tab:red")]:
            m, sem = res[f][g]
            a.plot(x, m, "-o", ms=3, color=col, label=lab)
            a.fill_between(x, m - sem, m + sem, color=col, alpha=0.15)
        a.set_title(f)
        a.set_xlabel("normalized step position")
        a.legend(fontsize=7)
    for ix in range(len(feats), nrow * ncol):
        ax[ix // ncol][ix % ncol].axis("off")
    fig.tight_layout()
    p1 = os.path.join(args.outdir, f"traj_{tag}.png")
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    print(f"\nsaved {p1}")

    # ---- (2) error-aligned curves (ProcessBench) ----
    usable = int(((ges >= 0) & (ges < np.array([s.shape[0] for s in series]))).sum())
    if usable > 0:
        ac = aligned_curves(series, names, ges, feats, args.win)
        offs = list(range(-args.win, args.win + 1))
        print(f"\n=== error-aligned mean z (offset 0 = gold error step; "
              f"usable={usable}) ===")
        print("  offset : " + " ".join(f"{o:+5d}" for o in offs))
        for f in feats:
            print(f"  {f:14s}: " + " ".join(f"{v:+5.2f}" for v in ac[f]))
        fig, a = plt.subplots(figsize=(7, 4))
        for f in feats:
            a.plot(offs, ac[f], "-o", ms=3, label=f)
        a.axvline(0, color="k", lw=0.8, ls="--")
        a.set_xlabel("step offset from gold error step")
        a.set_ylabel("within-chain z")
        a.legend(fontsize=7, ncol=2)
        a.set_title(f"error-aligned signal ({tag})")
        fig.tight_layout()
        p2 = os.path.join(args.outdir, f"aligned_{tag}.png")
        fig.savefig(p2, dpi=120)
        plt.close(fig)
        print(f"saved {p2}")

        # ---- (3) example individual traces ----
        ex = [i for i in range(len(series))
              if 0 <= ges[i] < series[i].shape[0]][:3]
        if ex and "tok_U_D" in z.files:
            fig, ax = plt.subplots(len(ex), 1, figsize=(9, 2.4 * len(ex)),
                                   squeeze=False)
            SR = z["step_token_ranges"]
            UD = z["tok_U_D"]
            for r, i in enumerate(ex):
                a = ax[r][0]
                ud = np.asarray(UD[i], float)
                a.plot(np.arange(len(ud)), ud, color="tab:gray", lw=0.8)
                ranges = np.asarray(SR[i], int)
                a0 = int(ranges[0, 0])
                for j in range(ranges.shape[0]):           # step boundaries
                    a.axvline(int(ranges[j, 0]) - a0, color="0.85", lw=0.5)
                k = int(ges[i])                            # gold error step span
                lo = int(ranges[k, 0]) - a0
                hi = int(ranges[k, 1]) - a0
                a.axvspan(lo, hi, color="tab:red", alpha=0.25, label="gold error step")
                if len(ud) >= 2:                           # linear trend (paper Fig.5)
                    xx = np.arange(len(ud))
                    b = np.polyfit(xx, ud, 1)
                    a.plot(xx, b[0] * xx + b[1], "b--", lw=1, label="linear fit")
                a.set_title(f"chain {i} (U_D per token; error step {k})")
                a.legend(fontsize=7)
            fig.tight_layout()
            p3 = os.path.join(args.outdir, f"examples_{tag}.png")
            fig.savefig(p3, dpi=120)
            plt.close(fig)
            print(f"saved {p3}")


if __name__ == "__main__":
    main()
