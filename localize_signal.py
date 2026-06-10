"""Per-step / per-token signal localization -- NO chain-level pooling.

Two questions, neither of which collapses a chain to one number:

(1) ERROR LOCALIZATION (ProcessBench, uses gold_error_step):
    Within each error chain, z-normalize a per-step feature across its own steps
    and ask whether the value at the gold error step stands out:
      loc_auroc  : within-chain P(feat[error step] > feat[other step]) pooled
      p_argmax   : fraction of error chains where the error step is the argmax
                   (baseline ~ mean 1/T)
      z_at_err   : mean within-chain z-score at the error step
    Also an error-aligned average curve (positions -2..+2 around the error step).

(2) CORRECT vs ERROR TRAJECTORY (any file): mean per-step feature vs normalized
    step position, for correct and error chains separately -- where do they
    diverge (paper Fig.1 gap), without averaging the whole chain away.

Per-step features: U_D, U_C (mean of tok_* over each step's token slice) and the
geometry stepgeom columns per layer.

Caveat: gold_error_step indexes the ORIGINAL ProcessBench steps; this script
assumes the kept steps align (true when no step was dropped during alignment).
Chains where gold_error_step is out of the kept range are skipped and counted.
"""

from __future__ import annotations

import argparse
import numpy as np


def per_step_series(z):
    """Return (names, list-of (T, n_feat) per chain). Columns: U_D, U_C, then
    each geometry (layer, feature)."""
    N = len(z["problem_ids"])
    geom_names = [str(x) for x in z["geom_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]
    SG = z["stepgeom"]
    SR = z["step_token_ranges"]
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    has_cloud = "stepcloud" in z.files and bool(z.get("cloud_stored", np.array(False)))
    cnames = [str(x) for x in z["cloud_feature_names"]] if has_cloud else []
    SC = z["stepcloud"] if has_cloud else None

    names = (["U_D", "U_C"]
             + [f"{fn}_L{ly}" for ly in layers for fn in geom_names]
             + [f"{fn}_L{ly}" for ly in layers for fn in cnames])
    out = []
    for i in range(N):
        sg = np.asarray(SG[i], float)              # (T, L, F)
        T = sg.shape[0]
        ranges = np.asarray(SR[i], int)            # (T, 2) absolute token idx
        a0 = int(ranges[0, 0])
        cols = []
        # per-step mean of token-level U_D / U_C over each step's slice
        for tok in (UD, UC):
            if tok is None:
                cols.append(np.full(T, np.nan))
                continue
            t = np.asarray(tok[i], float)
            s = np.full(T, np.nan)
            for j in range(T):
                lo = int(ranges[j, 0]) - a0
                hi = int(ranges[j, 1]) - a0 + 1
                lo, hi = max(0, lo), min(len(t), hi)
                if hi > lo:
                    s[j] = np.nanmean(t[lo:hi])
            cols.append(s)
        # geometry: flatten (L, F) -> columns in the same order as `names`
        for li in range(sg.shape[1]):
            for fi in range(sg.shape[2]):
                cols.append(sg[:, li, fi])
        # point-cloud D/V/C, same (L, F) flatten order
        if SC is not None:
            sc = np.asarray(SC[i], float)          # (T, L, 3)
            for li in range(sc.shape[1]):
                for fi in range(sc.shape[2]):
                    cols.append(sc[:, li, fi])
        out.append(np.column_stack(cols))          # (T, n_feat)
    return names, out


def zscore(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() < 2:
        return np.full_like(x, np.nan)
    mu, sd = x[m].mean(), x[m].std()
    if sd == 0:
        return np.full_like(x, np.nan)
    z = np.full_like(x, np.nan)
    z[m] = (x[m] - mu) / sd
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--win", type=int, default=2, help="aligned-curve half width")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    names, series = per_step_series(z)
    nfeat = len(names)
    ges = z["gold_error_step"].astype(int) if "gold_error_step" in z.files \
        else np.full(len(series), -1)

    # ---- (1) error localization ----
    err_idx = [i for i in range(len(series))
               if ges[i] >= 0 and ges[i] < series[i].shape[0]]
    n_err_total = int((ges >= 0).sum())
    n_oob = int(((ges >= 0) & (ges >= np.array([s.shape[0] for s in series]))).sum())

    print(f"file: {args.npz}")
    print(f"chains: {len(series)} | error chains: {n_err_total} | "
          f"usable (error step in kept range): {len(err_idx)} | "
          f"skipped out-of-range: {n_oob}")

    if err_idx:
        conc = np.zeros(nfeat); tie = np.zeros(nfeat); npair = np.zeros(nfeat)
        zatk = [[] for _ in range(nfeat)]
        argmax_hit = np.zeros(nfeat); argmax_n = np.zeros(nfeat)
        curve_sum = np.zeros((nfeat, 2 * args.win + 1))
        curve_cnt = np.zeros((nfeat, 2 * args.win + 1))
        for i in err_idx:
            k = int(ges[i])
            S = series[i]                           # (T, nfeat)
            T = S.shape[0]
            for c in range(nfeat):
                col = S[:, c]
                if not np.isfinite(col[k]):
                    continue
                others = col[np.arange(T) != k]
                others = others[np.isfinite(others)]
                if others.size:
                    conc[c] += np.sum(col[k] > others)
                    tie[c] += np.sum(col[k] == others)
                    npair[c] += others.size
                    argmax_hit[c] += int(col[k] >= np.nanmax(col))
                    argmax_n[c] += 1
                zc = zscore(col)
                if np.isfinite(zc[k]):
                    zatk[c].append(zc[k])
                for d in range(-args.win, args.win + 1):
                    p = k + d
                    if 0 <= p < T and np.isfinite(zc[p]):
                        curve_sum[c, d + args.win] += zc[p]
                        curve_cnt[c, d + args.win] += 1

        rows = []
        for c in range(nfeat):
            loc = (conc[c] + 0.5 * tie[c]) / npair[c] if npair[c] else np.nan
            rows.append({
                "name": names[c],
                "loc_auroc": loc,
                "z_at_err": float(np.mean(zatk[c])) if zatk[c] else np.nan,
                "p_argmax": argmax_hit[c] / argmax_n[c] if argmax_n[c] else np.nan,
            })
        rows.sort(key=lambda r: -abs((r["loc_auroc"] if np.isfinite(r["loc_auroc"])
                                      else 0.5) - 0.5))
        print(f"\n=== error-step localization (within error chains) ===")
        print(f"{'feature':16s} {'loc_auroc':>9s} {'z_at_err':>8s} {'p_argmax':>8s}")
        print("-" * 46)
        for r in rows[:args.top]:
            print(f"{r['name']:16s} {r['loc_auroc']:9.3f} {r['z_at_err']:8.2f} "
                  f"{r['p_argmax']:8.3f}")
        print("\n(loc_auroc>0.5 => feature is HIGHER at the error step than at "
              "other steps. p_argmax baseline ~ mean(1/T).)")

        # aligned curve for the best feature
        best = max(range(nfeat), key=lambda c:
                   abs(((conc[c] + 0.5 * tie[c]) / npair[c] if npair[c] else 0.5) - 0.5))
        curve = np.where(curve_cnt[best] > 0, curve_sum[best] / np.maximum(curve_cnt[best], 1), np.nan)
        offs = list(range(-args.win, args.win + 1))
        print(f"\nerror-aligned mean z-curve, feature '{names[best]}' "
              f"(offset 0 = error step):")
        print("  offset: " + " ".join(f"{o:+d}" for o in offs))
        print("  meanz : " + " ".join(f"{v:+.2f}" for v in curve))

    # ---- (2) correct vs error trajectory (normalized position) ----
    is_err = (z["is_correct"].astype(int) == 0)
    nb = 10
    for label, sel in [("U_D", 0), ("U_C", 1)]:
        acc = {0: np.zeros(nb), 1: np.zeros(nb)}
        cnt = {0: np.zeros(nb), 1: np.zeros(nb)}
        for i in range(len(series)):
            col = series[i][:, sel]
            T = len(col)
            for j in range(T):
                if not np.isfinite(col[j]):
                    continue
                b = min(nb - 1, int(nb * j / max(1, T)))
                g = int(is_err[i])
                acc[g][b] += col[j]; cnt[g][b] += 1
        cm = np.where(cnt[0] > 0, acc[0] / np.maximum(cnt[0], 1), np.nan)
        em = np.where(cnt[1] > 0, acc[1] / np.maximum(cnt[1], 1), np.nan)
        print(f"\n=== {label} trajectory by normalized step position (deciles) ===")
        print("  bin:     " + " ".join(f"{b:5d}" for b in range(nb)))
        print("  correct: " + " ".join(f"{v:5.2f}" for v in cm))
        print("  error:   " + " ".join(f"{v:5.2f}" for v in em))


if __name__ == "__main__":
    main()
