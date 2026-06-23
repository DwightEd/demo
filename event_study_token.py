"""TOKEN-level event study: align the windowed directional concentration R(t) to the first-error
ONSET and read its temporal profile -- is the collapse a PRECURSOR (R falls before the error token)
or SYNCHRONOUS, and is it sharper at token resolution than the step-level pooled scalar (which
dilutes the moment of error by averaging it with the step's earlier correct tokens)?

R(t) = ||mean of UNIT token vectors in the trailing window [t-w+1, t]|| -- the moving-window von
Mises-Fisher concentration (segmentation-free; this is also the plug-in signal). For each error
chain we align tokens by offset r = t - onset, onset = first token of the gold error step. We
average R over error chains (EVENT) and, as a within-chain placebo, over the SAME chains aligned at
an earlier CORRECT step boundary (PLACEBO). A dip near r=0 in EVENT but not PLACEBO = the collapse
is specific to the error onset; a dip at r<0 = a genuine precursor that step-level pooling hides.

Needs _cloud.npz: respcloud (clouds_stored) + cloud_store_layers + step_token_ranges + gold_error_step.
"""

from __future__ import annotations
import argparse
import numpy as np


def winres(U):
    """windowed resultant: ||mean of unit vectors|| over the rows of U (raw token vectors)."""
    nrm = np.linalg.norm(U, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return np.nan
    u = U[ok] / nrm[ok, None]
    return float(np.linalg.norm(u.mean(0)))


def profile_around(R, onset, W):
    """collect (offset, R) for offset r=t-onset in [-W, W] where R[t] is finite."""
    out = []
    for t in range(len(R)):
        r = t - onset
        if -W <= r <= W and np.isfinite(R[t]):
            out.append((r, R[t]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--w", type=int, default=6, help="trailing window width (tokens)")
    ap.add_argument("--W", type=int, default=24, help="event-study half-window (tokens)")
    ap.add_argument("--bin", type=int, default=4, help="offset bin size (tokens)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    nb = 2 * args.W // args.bin + 1
    centers = [(-args.W + args.bin // 2) + b * args.bin for b in range(nb)]
    ev_sum = np.zeros(nb); ev_cnt = np.zeros(nb)
    pb_sum = np.zeros(nb); pb_cnt = np.zeros(nb)
    step_err, step_prev = [], []                              # step-level R at error step vs previous step
    n_chains = 0
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        k = int(ges[i])
        if k < 2:                                             # need a prior correct boundary for placebo
            continue
        a0 = int(rng[0, 0]); T = rng.shape[0]
        onset = int(rng[k, 0]) - a0
        placebo = int(rng[max(1, k // 2), 0]) - a0            # an earlier correct step boundary
        if onset >= rcl.shape[0]:
            continue
        n_chains += 1
        # windowed R(t) over the whole response
        R = np.full(rcl.shape[0], np.nan)
        for t in range(args.w - 1, rcl.shape[0]):
            R[t] = winres(rcl[t - args.w + 1:t + 1])
        for r, val in profile_around(R, onset, args.W):
            b = int((r + args.W) // args.bin)
            if 0 <= b < nb:
                ev_sum[b] += val; ev_cnt[b] += 1
        for r, val in profile_around(R, placebo, args.W):
            b = int((r + args.W) // args.bin)
            if 0 <= b < nb:
                pb_sum[b] += val; pb_cnt[b] += 1
        # step-level pooled R: error step vs the step before it
        def steppool(j):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            return winres(rcl[lo:hi]) if hi - lo >= 2 else np.nan
        re, rp = steppool(k), steppool(k - 1)
        if np.isfinite(re):
            step_err.append(re)
        if np.isfinite(rp):
            step_prev.append(rp)

    ev = ev_sum / np.maximum(ev_cnt, 1); pb = pb_sum / np.maximum(pb_cnt, 1)
    print(f"file: {args.npz} | layer {args.layer} | w={args.w} W={args.W} | error chains {n_chains}")
    print(f"\n{'offset(tok)':>11s} {'R_event':>8s} {'R_placebo':>10s} {'n_ev':>6s}")
    for b in range(nb):
        print(f"  {centers[b]:>+9d} {ev[b]:8.3f} {pb[b]:10.3f} {int(ev_cnt[b]):6d}")
    pre = ev[:nb // 2][np.isfinite(ev[:nb // 2])]; post = ev[nb // 2:][np.isfinite(ev[nb // 2:])]
    dip_at = centers[int(np.nanargmin(ev))]
    print(f"\nR_event min at offset {dip_at:+d} tokens  (0 = synchronous, <0 = precursor)")
    print(f"step-level pooled R:  error step {np.nanmean(step_err):.3f}  vs prev step {np.nanmean(step_prev):.3f}  "
          f"(diff {np.nanmean(step_err)-np.nanmean(step_prev):+.3f})")
    print("\nread: if R_event dips near r=0 while R_placebo (same chains, a correct earlier boundary) stays flat, "
          "the directional collapse is SPECIFIC to the error onset. If the dip bottoms at r<0, there is a "
          "token-level PRECURSOR that step pooling hides. Compare the token-level dip depth to the step-level "
          "diff (error vs prev step): a deeper/sharper token dip means step pooling dilutes the moment of error "
          "-- motivating the moving-window (segmentation-free) signal for the plug-in and a change-point/Rayleigh "
          "treatment of R(t). This is descriptive (mechanism/localization), not a new detector by itself.")


if __name__ == "__main__":
    main()
