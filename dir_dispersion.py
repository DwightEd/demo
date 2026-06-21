"""Decisive control: DIRECT within-step token directional dispersion, bypassing exp-pooling.

Our `resultant` = ||sum_t w_t u_t|| uses unit tokens u_t BUT exp-pooling weights w_t. To make the
"directional collapse" mechanism claim hard, we must test pooling-free, weight-free, magnitude-
free direction concentration computed straight from the token cloud:

  R    = || (1/n) sum_i u_i ||                      mean resultant length (circular statistics)
  MPC  = (1/(n(n-1))) sum_{i!=j} <u_i, u_j>         mean pairwise cosine (excl. self) = directional concentration
  (note: MPC_incl_self = R^2, so AUROC(R) == AUROC(R^2); MPC excl-self has a mild length term.)
  1-R  = directional dispersion

Compared against exp_resultant = ||sum_t w_t u_t|| (recomputed from the SAME tokens) and the stored
`resultant`. Verdict:
  * if R / MPC drop synchronously at the error step AND AUROC ~0.7 like exp_resultant
    -> the signal IS pure direction concentration; the mechanism claim is confirmed.
  * if R / MPC are weak while exp_resultant is strong -> the signal needs the exp weighting or
    leaks token-magnitude (rho_i) -> the "direction collapse" claim must be revised.

Needs _cloud.npz: respcloud + cloud_store_layers + step_token_ranges + gold_error_step
(+ stepcloud/resultant for the stored baseline).
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
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def exp_w(n):
    if n <= 1:
        return np.ones(max(n, 1))
    e = np.exp(np.arange(n) / (n - 1)); return e / e.sum()


def causal_resid(y, sd_floor, clip=5.0, eps=1e-6):
    T = len(y); s = np.full(T, np.nan)
    for t in range(2, T):
        h = y[:t]; h = h[np.isfinite(h)]
        if len(h) >= 2 and np.isfinite(y[t]):
            mu = h.mean(); sd = max(h.std(), sd_floor) + eps
            s[t] = np.clip((y[t] - mu) / sd, -clip, clip)
    return s


def step_dispersion(tok):
    """from raw token cloud (n,d) -> R (mean resultant length), MPC (mean pairwise cos excl self),
    exp_res (exp-weighted unit pool norm). All on UNIT tokens (magnitude-free)."""
    nrm = np.linalg.norm(tok, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return None
    u = tok[ok] / nrm[ok, None]; n = u.shape[0]
    mean_u = u.mean(0); R = float(np.linalg.norm(mean_u))
    G = u @ u.T; mpc = float((G.sum() - n) / (n * (n - 1)))          # exclude self-pairs
    w = exp_w(n); exp_res = float(np.linalg.norm((w[:, None] * u).sum(0)))
    return R, mpc, exp_res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else None
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    SC = z["stepcloud"] if "stepcloud" in z.files else None
    use_res = SC is not None and lyu is not None and args.layer in lyu and "resultant" in cnames
    lic = lyu.index(args.layer) if use_res else None
    fi = cnames.index("resultant") if "resultant" in cnames else None

    chains = []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i])
        R = np.full(T, np.nan); MPC = np.full(T, np.nan); XR = np.full(T, np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo >= 2:
                st = step_dispersion(rcl[lo:hi])
                if st is not None:
                    R[j], MPC[j], XR[j] = st
        res = (np.asarray(SC[i], float)[:, lic, fi] if use_res else np.full(T, np.nan))
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)
        chains.append(dict(R=R, MPC=MPC, XR=XR, res=res, nt=nt, k=k, T=T, correct=k < 0))

    def sdf(key):
        v = [c[key][np.isfinite(c[key])].std() for c in chains if np.isfinite(c[key]).sum() >= 2]
        return 0.5 * float(np.median(v)) if v else 1.0
    sf = {k: sdf(k) for k in ["R", "MPC", "XR", "res"]}

    # per labeled step (first-error): collect raw + within-chain residual
    cols = {k: [] for k in ["R", "MPC", "XR", "res", "Rz", "MPCz", "XRz", "resz"]}
    Y, NT, EOFF, ER = [], [], [], []
    for c in chains:
        rz = {k: causal_resid(c[k], sf[k]) for k in ["R", "MPC", "XR", "res"]}
        for j in range(c["T"]):
            if not c["correct"]:
                EOFF.append(j - c["k"]); ER.append(c["R"][j])
            if c["correct"] or j < c["k"]:
                lab = 0
            elif j == c["k"]:
                lab = 1
            else:
                continue
            if not np.isfinite(c["R"][j]):
                continue
            for k in ["R", "MPC", "XR", "res"]:
                cols[k].append(c[k][j]); cols[k + "z"].append(rz[k][j])
            Y.append(lab); NT.append(c["nt"][j])
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    for k in cols:
        cols[k] = np.asarray(cols[k], float)
    EOFF = np.asarray(EOFF, int); ER = np.asarray(ER, float)
    R, MPC, XR, res = cols["R"], cols["MPC"], cols["XR"], cols["res"]

    print(f"file: {args.npz} | layer {args.layer} | labeled steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"corr: R~exp_res {np.corrcoef(R,XR)[0,1]:+.2f} | R~stored_res {np.corrcoef(R,res)[0,1]:+.2f} "
          f"| R~MPC {np.corrcoef(R,MPC)[0,1]:+.2f}")
    print(f"\n{'signal':28s} {'AUROC':>7s} {'bucket':>7s} {'within-z AUROC':>15s}")
    for nm, raw, zz in [("R = mean resultant len (UNIF)", R, cols["Rz"]),
                        ("MPC = mean pairwise cos", MPC, cols["MPCz"]),
                        ("exp_resultant (recomputed)", XR, cols["XRz"]),
                        ("stored resultant (baseline)", res, cols["resz"])]:
        # error = LOW concentration -> flip sign so positive = error-like for the within-z
        print(f"  {nm:28s} {bdir(auroc(raw,Y)):7.3f} {bucket(raw,Y,NT):7.3f} {bdir(auroc(-zz,Y)):15.3f}")

    print(f"\nevent study: mean R (uniform dir concentration) by offset from first error")
    print(f"  {'Δ=j-k':>6s} {'n':>5s} {'mean R':>9s} {'SE':>7s}")
    for dd in range(-4, 4):
        m = (EOFF == dd) & np.isfinite(ER)
        if m.sum() >= 5:
            star = " <-- error" if dd == 0 else ""
            print(f"  {dd:>6d} {int(m.sum()):>5d} {ER[m].mean():>9.4f} {ER[m].std()/np.sqrt(m.sum()):>7.4f}{star}")

    print("\nread: if R / MPC (pooling-free, weight-free, magnitude-free) reach ~0.7 AUROC and dip at "
          "the error step like exp_resultant, the directional-collapse mechanism is CONFIRMED -- the "
          "signal is pure within-step direction concentration, not pooling or token-magnitude. If R/MPC "
          "are weak while exp_resultant/stored stay strong, the signal leaks magnitude/weighting and the "
          "mechanism claim must be revised. corr(R, exp_res) near 1 = they are the same thing.")


if __name__ == "__main__":
    main()
