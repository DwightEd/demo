"""Rigorous manifold-departure: does an error step's TOKENS disperse into directions the chain's
PRIOR tokens never used -- measured at token level, after removing the shared common component.

Fixes two crudeness bugs in dir_manifold.py:
  (1) it never removed the dominant shared anisotropy direction, so every step's mean direction was
      near-parallel and 'departure' was pure noise (AUROC ~0.52);
  (2) it collapsed each step to ONE mean direction, discarding the within-step token spread that IS
      the validated signal (resultant = within-step token concentration, ~0.77).

Here we keep token-level richness AND remove the common component:
  c          = top principal direction of ALL chain token unit-vectors (the shared anisotropy axis)
  residual   r_t = u_t - (u_t . c) c   for every token (remove the common component)
  V_{<j}     = top-r PCA basis of the PRIOR steps' token residuals (manifold walked so far)
  off_frac_j = mean over step-j tokens of ||r - V V^T r||^2 / ||r||^2
             = fraction of the step's representational variance OUTSIDE the prior manifold
  on_disp_j  = within-step dispersion INSIDE the manifold (control: ordinary diffusion)

Hypothesis (SMCD): an error step has high off_frac (it explores off-manifold directions), beyond
plain diffusion. Decisive: off_frac beats resultant, is independent of it (low |r|res), survives
length + step-index control. If off_frac ~0.5 too, the cross-step manifold structure is genuinely
empty and geometry is mined out at within-step resultant.

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


def units(H):
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    return H[ok] / nrm[ok, None] if ok.sum() else np.zeros((0, H.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--rank", type=int, default=6, help="manifold dim (top-r PCA of prior token residuals)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    OFF, RES, SJ, Y, NT = [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i]); correct = (k < 0)
        # per-step token unit vectors
        toks = []
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            toks.append(units(rcl[lo:hi]) if hi - lo >= 2 else np.zeros((0, rcl.shape[1])))
        allu = np.concatenate([t for t in toks if len(t)], 0) if any(len(t) for t in toks) else None
        if allu is None or len(allu) < 4:
            continue
        # common component c = top principal axis of all chain token directions
        _, _, Vt = np.linalg.svd(allu - allu.mean(0), full_matrices=False)
        c = Vt[0]
        def resid(U):
            return U - np.outer(U @ c, c) if len(U) else U
        rtoks = [resid(t) for t in toks]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            prior = [rtoks[p] for p in range(j) if len(rtoks[p])]
            Rj = rtoks[j]
            if len(Rj) < 2 or sum(len(p) for p in prior) < 4:
                continue
            P = np.concatenate(prior, 0)
            r = min(args.rank, P.shape[0] - 1, P.shape[1])
            _, _, Pvt = np.linalg.svd(P - P.mean(0), full_matrices=False)
            V = Pvt[:r]                                            # (r, d) manifold basis
            nrm2 = (Rj ** 2).sum(1)                                # per-token total energy
            ortho = Rj - (Rj @ V.T) @ V                            # off-manifold component
            o2 = (ortho ** 2).sum(1)
            off_frac = float(np.mean(o2 / (nrm2 + 1e-12)))         # mean off-manifold fraction
            res = float(np.linalg.norm(units(np.asarray(toks[j])).mean(0)))  # plain resultant (ref)
            OFF.append(off_frac); RES.append(res); SJ.append(j)
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    OFF = np.asarray(OFF); RES = np.asarray(RES); SJ = np.asarray(SJ, float)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    print(f"file: {args.npz} | layer {args.layer} | rank {args.rank} | steps(j>=2) {len(Y)} "
          f"| first-error {int(Y.sum())}")
    print(f"\n{'signal':30s} {'AUROC':>7s} {'b(len)':>7s} {'b(stepj)':>8s} {'|r|res':>7s} {'|r|stepj':>8s}")
    for nm, v in [("resultant (level, ref)", -RES), ("off_frac (off-manifold)", OFF)]:
        print(f"  {nm:30s} {bdir(auroc(v, Y)):7.3f} {wbuck(v, Y, NT):7.3f} {wbuck(v, Y, SJ):8.3f} "
              f"{abscorr(v, RES):7.3f} {abscorr(v, SJ):8.3f}")

    def zc(x):
        x = np.asarray(x, float); m = np.isfinite(x); s = np.full(len(x), 0.0)
        s[m] = (x[m] - x[m].mean()) / (x[m].std() + 1e-9); return s
    comb = zc(-RES) + zc(OFF)
    q = np.nanquantile(RES, 1 / 3); low = RES <= q
    print(f"\ncombined z(-res)+z(off_frac)  AUROC {bdir(auroc(comb, Y)):.3f}  b(len) {wbuck(comb, Y, NT):.3f}")
    print(f"LOW-resultant stratum (n={int(low.sum())}, err={int(Y[low].sum())}): "
          f"off_frac AUROC {bdir(auroc(OFF[low], Y[low])):.3f}")
    print("\nread: this is the FAIR test (token-level, common-component removed). Decisive = off_frac AUROC, "
          "its independence |r|res, and the LOW-resultant-stratum AUROC. If off_frac > resultant with low "
          "|r|res and works in the low-res stratum -> the manifold hypothesis holds and my earlier impl was "
          "just crude. If off_frac ~0.5 here too, the cross-step manifold carries no error info beyond "
          "within-step resultant -> geometry is genuinely mined out, pivot to the confident stratum. "
          "Try --rank 3/10.")


if __name__ == "__main__":
    main()
