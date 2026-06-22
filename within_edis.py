"""WITHIN-STEP EDIS: EDIS's cumulative instability aggregation applied INSIDE one semantic step, on
the token-level fan-out sequence -- not a global cumulative signal.

The within-step fan-out per token is dev_t = 1 - cos(u_t, running-mean-before-t) (how far token t
breaks from the step's direction so far). within_fan showed its MEAN is just resultant (|r|res .93)
and its coarse 2-bin shape (conv) is independent but empty (.55). EDIS's contribution is a FINER
aggregation: cumulative bursts + peak-valley rebounds, the event structure between mean and halves.
We apply exactly that, inside the step:
  wedis     = 0.5(burst + peakvalley)(1+var)  on dev   (burst/pv thresholds = 1 within-step sigma)
  wedis_z   = same on the WITHIN-STEP z-scored dev      (level removed -> PURE order dynamic, ind. of res)
  var_dev   = within-step variance of dev               (the (1+var) amplifier alone)

Decisive: does wedis_z (level-free within-step instability) carry error signal with LOW |r|res and in
the LOW-resultant stratum? That is the cumulative within-step dynamic resultant's pool discards, in
EDIS's own event form. If it is empty too, the within-step order genuinely carries nothing beyond the
pooled level and the diagnostic content is the static concentration, full stop.

Needs _cloud.npz: respcloud (clouds_stored) + cloud_store_layers + step_token_ranges + gold_error_step.
"""

from __future__ import annotations
import argparse
import numpy as np


def auroc(s, y):
    m = np.isfinite(s); s, y = s[m], y[m]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if not p or not n:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def wbuck(s, y, key, nb=5):
    m = np.isfinite(s) & np.isfinite(key); s, y, key = s[m], y[m], key[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(key, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(key, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a[m], b[m])[0, 1]))


def edis_seq(s, w, tau_b, tau_r):
    """EDIS aggregation on a sequence s: cumulative burst (growth over window w) + peak-valley
    (rise above running min), threshold tau_b/tau_r. Returns 0.5*(burst+pv)."""
    s = np.asarray(s, float); T = len(s)
    if T < 3:
        return 0.0
    burst = int(sum(1 for t in range(T - w) if s[t + w] - s[t] > tau_b)) if T > w else 0
    pv = 0; rmin = s[0]
    for t in range(1, T):
        if s[t] - rmin > tau_r:
            pv += 1
        rmin = min(rmin, s[t])
    return 0.5 * (burst + pv)


def within_dev(H):
    """ordered token states of ONE step -> (resultant, dev sequence)."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    if n < 6:
        return None
    res = float(np.linalg.norm(u.mean(0)))
    csum = np.cumsum(u, 0); dev = np.empty(n - 1)
    for t in range(1, n):
        m = csum[t - 1]; mn = np.linalg.norm(m)
        dev[t - 1] = 1.0 - float(u[t] @ (m / mn)) if mn > 1e-9 else 0.0
    return res, dev


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    keys = ["res", "wedis", "wedis_z", "var_dev"]
    cols = {k: [] for k in keys}; Y, NT = [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i]); correct = (k < 0)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            wd = within_dev(rcl[lo:hi]) if hi - lo >= 6 else None
            if wd is None:
                continue
            res, dev = wd; sd = dev.std(); w = max(1, len(dev) // 4)
            wedis = edis_seq(dev, w, sd, sd) * (1.0 + float(dev.var()))      # within-step EDIS on dev
            zz = (dev - dev.mean()) / (sd + 1e-9)
            wedis_z = edis_seq(zz, w, 1.0, 1.0)                              # level-free pure dynamic
            cols["res"].append(res); cols["wedis"].append(wedis)
            cols["wedis_z"].append(wedis_z); cols["var_dev"].append(float(dev.var()))
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    for kk in keys:
        cols[kk] = np.asarray(cols[kk], float)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); RES = cols["res"]

    print(f"file: {args.npz} | layer {args.layer} | steps(n>=6) {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'within-step signal':22s} {'AUROC':>7s} {'b(len)':>7s} {'|r|res':>7s} {'lowRes AUROC':>13s}")
    q = np.nanquantile(RES, 1 / 3); low = RES <= q
    orient = {"res": -1, "wedis": 1, "wedis_z": 1, "var_dev": 1}
    for kk in keys:
        v = orient[kk] * cols[kk]
        rr = abscorr(cols[kk], RES) if kk != "res" else 1.0
        print(f"  {kk:22s} {bdir(auroc(v, Y)):7.3f} {wbuck(v, Y, NT):7.3f} {rr:7.3f} "
              f"{bdir(auroc(v[low], Y[low])):13.3f}")

    def zc(x):
        x = np.asarray(x, float); m = np.isfinite(x); s = np.full(len(x), 0.0)
        s[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-9); return s
    for kk in ["wedis", "wedis_z"]:
        comb = zc(-RES) + zc(orient[kk] * cols[kk])
        print(f"combined z(-res)+z({kk:8s})  AUROC {bdir(auroc(comb, Y)):.3f}  b(len) {wbuck(comb, Y, NT):.3f}")
    print("\nread: wedis_z is the level-free, within-step, EDIS-form cumulative dynamic. If it shows signal "
          "with low |r|res and in the low-resultant stratum, the cumulative within-step fan-out instability "
          "is a real axis the pooled resultant discards -> build the matched level x within-step-EDIS detector. "
          "If wedis_z ~0.5 (like conv), the within-step ORDER carries no error info beyond the static "
          "concentration, and the geometric content is fully in resultant.")


if __name__ == "__main__":
    main()
