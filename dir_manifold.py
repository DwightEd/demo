"""Manifold-departure: does an error step move in a direction the chain's prior steps never used?

This is the project's core hypothesis (SMCD = Step-level Manifold Constraint Diagnostic) in its
PROPER geometric form, which we never actually tested. Past 'dynamic' attempts collapsed the
geometry to a SCALAR first (resultant residual ~0.68, step-velocity, D_j/D_0) and then took a
derivative -- throwing away the subspace. Here we keep the geometry: the prefix step-directions
span the manifold the chain has walked so far, and we measure how far OUTSIDE it the current step
goes.

Per step j (with >=2 prior steps), from respcloud:
  d_j           = normalized mean of the step's UNIT token vectors (its representative direction)
  V_{<j}        = top-r PCA basis of {d_0..d_{j-1}}  (the low-dim manifold walked so far)
  depart_lin    = ||d_j - V V^T d_j||                (orthogonal departure from the linear manifold)
  depart_nn     = 1 - max_{i<j} cos(d_j, d_i)        (departure from the NEAREST prior direction; kNN/nonlinear)

Everything is length-clean (unit directions) and relative to the chain's OWN history. depart_lin
mechanically shrinks as j grows (the subspace fills), so we ALSO report the AUROC of its residual
after regressing out step-index, and a step-index-bucketed AUROC -- the honest test that it carries
error info BEYOND position. Decisive: depart AUROC (esp. its j-residual) > resultant, and low |r|res
(independent axis). If not, geometry is mined out at resultant and we pivot to the confident stratum.

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


def _wbuck(s, y, key, nb=5):
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


def resid_on(s, key):
    """linear residual of s after regressing out [1, key, key^2] -- removes the positional trend."""
    s = np.asarray(s, float); key = np.asarray(key, float)
    m = np.isfinite(s) & np.isfinite(key); out = np.full(len(s), np.nan)
    if m.sum() < 10:
        return out
    X = np.c_[np.ones(m.sum()), key[m], key[m] ** 2]
    beta, *_ = np.linalg.lstsq(X, s[m], rcond=None)
    out[m] = s[m] - X @ beta
    return out


def stepdir(H):
    """H: (n_tok,d) -> normalized mean of unit token vectors (representative direction), or None."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return None, np.nan
    u = H[ok] / nrm[ok, None]
    mean = u.mean(0); res = float(np.linalg.norm(mean))           # resultant (uniform) for reference
    d = mean / (res + 1e-12)
    return d.astype(np.float64), res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=4, help="max manifold dim (top-r PCA of prefix dirs)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    DLIN, DNN, RES, SJ, Y, NT = [], [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i]); correct = (k < 0)
        dirs = []                                                  # representative direction per step
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            d, res = (stepdir(rcl[lo:hi]) if hi - lo >= 2 else (None, np.nan))
            dirs.append((d, res))
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            d, res = dirs[j]
            prior = [dirs[p][0] for p in range(j) if dirs[p][0] is not None]
            if d is None or len(prior) < 2:
                continue
            P = np.stack(prior)                                    # (j', d)
            # linear manifold = top-r right singular vectors of prefix directions
            r = min(args.rank, P.shape[0])
            _, _, Vt = np.linalg.svd(P, full_matrices=False)
            V = Vt[:r]                                             # (r, d) orthonormal
            proj = V.T @ (V @ d)
            dlin = float(np.linalg.norm(d - proj))                # orthogonal departure in [0,1]
            dnn = float(1.0 - np.max(P @ d))                       # 1 - nearest prior cosine
            DLIN.append(dlin); DNN.append(dnn); RES.append(res); SJ.append(j)
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    DLIN = np.asarray(DLIN); DNN = np.asarray(DNN); RES = np.asarray(RES)
    SJ = np.asarray(SJ, float); Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    print(f"file: {args.npz} | layer {args.layer} | rank {args.rank} | labeled steps(j>=2) {len(Y)} "
          f"| first-error {int(Y.sum())}")
    print(f"\n{'signal':30s} {'AUROC':>7s} {'b(len)':>7s} {'b(stepj)':>8s} {'res|j AUROC':>11s} "
          f"{'|r|res':>7s} {'|r|stepj':>8s}")
    for nm, v in [("resultant (level, ref)", -RES), ("depart_lin (manifold resid)", DLIN),
                  ("depart_nn (1-max prior cos)", DNN)]:
        rj = resid_on(v, SJ)
        print(f"  {nm:30s} {bdir(auroc(v, Y)):7.3f} {_wbuck(v, Y, NT):7.3f} {_wbuck(v, Y, SJ):8.3f} "
              f"{bdir(auroc(rj, Y)):11.3f} {abscorr(v, RES):7.3f} {abscorr(v, SJ):8.3f}")

    # does departure add to resultant where it is blind? combined standardized sum + low-res stratum
    def zc(x):
        x = np.asarray(x, float); m = np.isfinite(x)
        s = np.full(len(x), 0.0); s[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-9); return s
    comb = zc(-RES) + zc(DLIN)
    print(f"\ncombined z(-resultant)+z(depart_lin)  AUROC {bdir(auroc(comb, Y)):.3f}  b(len) {_wbuck(comb, Y, NT):.3f}")
    q = np.nanquantile(RES, 1 / 3); low = RES <= q
    print(f"LOW-resultant stratum (n={int(low.sum())}, err={int(Y[low].sum())}): "
          f"depart_lin AUROC {bdir(auroc(DLIN[low], Y[low])):.3f}  depart_nn AUROC {bdir(auroc(DNN[low], Y[low])):.3f}")
    print("\nread: DECISIVE = depart's 'res|j AUROC' (position-controlled) and its LOW-resultant-stratum AUROC. "
          "If depart_lin/nn beats resultant AND has low |r|res (independent axis) AND works in the low-res "
          "stratum where resultant is blind -> the manifold-departure hypothesis finally holds and we have a "
          "NEW geometric axis that breaks the 0.77 ceiling. If its AUROC ~ resultant with high |r|res, or its "
          "j-residual collapses to ~0.5, departure is just resultant/position in disguise -- then geometry is "
          "genuinely mined out and we pivot to the confident-stratum scoped exceed. Try --rank 2/3/6.")


if __name__ == "__main__":
    main()
