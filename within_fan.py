"""WITHIN-STEP fan-out DYNAMICS: how the token directions disperse and (fail to) re-converge as a
single reasoning step unfolds. Stays entirely inside one semantic step -- NO cross-step pooling
(which is confounded by the global convergence trend of reasoning: later steps always look more
contracted, so cross-step departure measures convergence, not error).

The validated resultant pools a step's tokens ORDER-FREE into one scalar, discarding the sequence.
The dynamic lives in the order: a correct step should fan out then RE-CONVERGE (explore, then commit);
an error step fans out and stays diffuse (never commits) or keeps turning. Per step, in generation
order, with running direction m_{t-1} = normalized cumulative mean of tokens before t:
  dev_t   = 1 - cos(u_t, m_{t-1})            per-token fan-out from the step's direction-so-far
  fan_mean= mean dev                          order-aware dispersion (resultant's dynamic cousin)
  fan_late= mean dev over the last third      still fanning at the end = no commit
  conv    = mean(dev first half) - mean(dev second half)   >0 fan-then-converge (correct); ~0/neg error
  turn    = mean 1-cos(u_t, u_{t-1})          token-to-token turning (trajectory wiggle)

All are within the semantic step, causal, length-clean (cosines). Decisive: do conv / fan_late carry
error signal INDEPENDENT of resultant (low |r|res) and in the LOW-resultant stratum where the level
is blind? That would be the within-step DYNAMIC axis resultant's pooling throws away.

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


def within_step(H):
    """H: (n_tok, d) ordered token states of ONE step -> dict of within-step dynamic features."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    if n < 4:
        return None
    res = float(np.linalg.norm(u.mean(0)))                       # plain resultant (reference)
    csum = np.cumsum(u, 0)
    dev = np.empty(n - 1)                                        # dev_t for t=1..n-1 (0-indexed)
    for t in range(1, n):
        m = csum[t - 1]; mn = np.linalg.norm(m)
        dev[t - 1] = 1.0 - float(u[t] @ (m / mn)) if mn > 1e-9 else 0.0
    half = (n - 1) // 2
    fan_mean = float(dev.mean())
    fan_late = float(dev[-max(1, (n - 1) // 3):].mean())
    conv = float(dev[:half].mean() - dev[half:].mean()) if half >= 1 else 0.0
    turn = float(np.mean([1.0 - u[t] @ u[t - 1] for t in range(1, n)]))
    return dict(res=res, fan_mean=fan_mean, fan_late=fan_late, conv=conv, turn=turn)


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

    keys = ["res", "fan_mean", "fan_late", "conv", "turn"]
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
            f = within_step(rcl[lo:hi]) if hi - lo >= 4 else None
            if f is None:
                continue
            for kk in keys:
                cols[kk].append(f[kk])
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    for kk in keys:
        cols[kk] = np.asarray(cols[kk], float)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); RES = cols["res"]

    print(f"file: {args.npz} | layer {args.layer} | steps(n>=4) {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'within-step signal':24s} {'AUROC':>7s} {'b(len)':>7s} {'|r|res':>7s} {'lowRes AUROC':>13s}")
    q = np.nanquantile(RES, 1 / 3); low = RES <= q
    # orientation: error = high fan/turn, LOW conv -> flip conv
    orient = {"res": -1, "fan_mean": 1, "fan_late": 1, "conv": -1, "turn": 1}
    for kk in keys:
        v = orient[kk] * cols[kk]
        lr = bdir(auroc(v[low], Y[low]))
        rr = abscorr(cols[kk], RES) if kk != "res" else 1.0
        print(f"  {kk:24s} {bdir(auroc(v, Y)):7.3f} {wbuck(v, Y, NT):7.3f} {rr:7.3f} {lr:13.3f}")

    def zc(x):
        x = np.asarray(x, float); m = np.isfinite(x); s = np.full(len(x), 0.0)
        s[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-9); return s
    best = max(["fan_late", "conv"], key=lambda kk: bdir(auroc(orient[kk] * cols[kk], Y)))
    comb = zc(-RES) + zc(orient[best] * cols[best])
    print(f"\ncombined z(-res)+z({best})  AUROC {bdir(auroc(comb, Y)):.3f}  b(len) {wbuck(comb, Y, NT):.3f}")
    print("\nread: stays WITHIN the semantic step, order-aware, no cross-step pooling. Decisive = conv / "
          "fan_late: do they (a) carry error signal with LOW |r|res (an axis resultant's order-free pool "
          "discards) and (b) work in the LOW-resultant stratum? If yes, the within-step convergence DYNAMIC "
          "is the new axis -- correct steps fan then re-converge, error steps don't. If they track resultant "
          "(high |r|res) and die in the low stratum, the order carries nothing beyond the pooled level.")


if __name__ == "__main__":
    main()
