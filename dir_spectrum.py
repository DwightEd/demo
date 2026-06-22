"""Mechanism test: is 'directional dispersion' at an error STRUCTURED or ISOTROPIC?

Premise (the 0.77 ceiling): resultant measures only alignment to the MEAN direction, so it cannot
tell a legitimately-dispersed CORRECT step (casework / enumeration -> directions split into a few
TIGHT modes) from a genuinely-lost ERROR step (directions fan out ISOTROPICALLY). Both have low
resultant -> they overlap -> the ceiling. So correct reasoning *can* disperse, but with structure;
error disperses without it. The discriminator is the SHAPE of the unit-direction distribution, not
its mean length -- and it is length-clean (shape of the unit vectors, not token count).

Per step, from the raw JL token vectors (respcloud), unit-normalized:
  level      resultant       = ||exp-pooled unit vectors||                 (alignment to mean)
  structure  spec_entropy    = normalized entropy of the unit-vector Gram eigenvalues in [0,1]
                               0 = one direction; small = a few modes (casework); 1 = isotropic fan
             PR_norm         = participation ratio of the spectrum / n     (effective # modes)
             lam1_share      = top eigenvalue / n                          (dominant-mode share)

Decisive test: stratify steps by resultant. In the LOW-resultant (dispersed) stratum, does
spec_entropy separate correct from error where resultant cannot? i.e. is mean spec_entropy(error) >
mean spec_entropy(correct), and its AUROC > ~0.6? If yes, the mechanism holds: error dispersion is
isotropic, correct dispersion is structured -- the unmined axis that can break the resultant ceiling.

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


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5:
        return float("nan")
    a, b = a[m], b[m]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a, b)[0, 1]))


def step_spectrum(H):
    """H: (n_tok, d) raw token vectors -> (resultant, spec_entropy, PR_norm, lam1_share).
    spec_entropy/PR/lam1 are from the unit-direction Gram eigenvalues (length-clean shape)."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return None
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
    res = float(np.linalg.norm((w[:, None] * u).sum(0)))            # level: alignment to mean
    G = u @ u.T                                                     # n x n Gram of unit vectors
    e = np.linalg.eigvalsh(G); e = np.clip(e, 0, None); s = e.sum()
    if s <= 0:
        return None
    p = e / s; p = p[p > 1e-12]
    H_spec = float(-(p * np.log(p)).sum()) / np.log(n)              # in [0,1]: 0 aligned, 1 isotropic
    PR_norm = float((s * s) / (np.square(e).sum() + 1e-12)) / n     # effective # modes / n
    lam1 = float(e.max() / s)                                       # dominant-mode share
    return res, H_spec, PR_norm, lam1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="the _cloud.npz with populated respcloud")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    RES, HSP, PRN, LAM, Y, NT = [], [], [], [], [], []
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
            if hi - lo < 2:
                continue
            st = step_spectrum(rcl[lo:hi])
            if st is None:
                continue
            RES.append(st[0]); HSP.append(st[1]); PRN.append(st[2]); LAM.append(st[3])
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    RES = np.asarray(RES); HSP = np.asarray(HSP); PRN = np.asarray(PRN); LAM = np.asarray(LAM)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'signal':26s} {'AUROC':>7s} {'bucket':>7s} {'|r|len':>7s} {'|r|res':>7s}")
    for nm, v, flip in [("resultant (level)", RES, True), ("spec_entropy (isotropy)", HSP, False),
                        ("PR_norm (# modes)", PRN, False), ("lam1_share (dominant)", LAM, True)]:
        sv = -v if flip else v
        print(f"  {nm:26s} {bdir(auroc(sv, Y)):7.3f} {bucket(sv, Y, NT):7.3f} "
              f"{abscorr(v, NT):7.3f} {abscorr(v, RES):7.3f}")

    # DECISIVE: within resultant strata, does spectral SHAPE separate correct/error?
    q = np.quantile(RES, [1 / 3, 2 / 3]); strat = np.digitize(RES, q)
    names = ["LOW res (diffuse)", "MID res", "HIGH res (concentrated)"]
    print(f"\n{'resultant stratum':24s} {'n':>6s} {'err':>5s} {'specH AUROC':>12s} "
          f"{'meanH cor':>10s} {'meanH err':>10s}")
    for sname in range(3):
        m = strat == sname; ne = int(Y[m].sum())
        a = bdir(auroc(HSP[m], Y[m]))
        hc = float(np.nanmean(HSP[m][Y[m] == 0])) if (Y[m] == 0).any() else np.nan
        he = float(np.nanmean(HSP[m][Y[m] == 1])) if (Y[m] == 1).any() else np.nan
        print(f"  {names[sname]:24s} {int(m.sum()):>6d} {ne:>5d} {a:>12.3f} {hc:>10.3f} {he:>10.3f}")

    combo = (1.0 - RES) * HSP                                       # diffuse AND isotropic = error
    print(f"\ncombined (1-resultant)*spec_entropy   AUROC {bdir(auroc(combo, Y)):.3f}  "
          f"bucket {bucket(combo, Y, NT):.3f}")
    print("\nread: DECISIVE cell = specH AUROC in 'LOW res (diffuse)' with meanH(err) > meanH(cor). If specH "
          "separates correct from error THERE (where resultant is blind because both are diffuse), the "
          "mechanism holds: correct dispersion is STRUCTURED (few tight modes, low spec_entropy), error "
          "dispersion is ISOTROPIC (high spec_entropy). That is the unmined, length-clean axis (check |r|len "
          "is small) orthogonal to resultant (check |r|res). If specH AUROC ~0.5 in the low stratum, the two "
          "dispersions are NOT distinguishable by spectral shape and this axis is dead. The combined score "
          "previews whether level x structure beats level alone.")


if __name__ == "__main__":
    main()
