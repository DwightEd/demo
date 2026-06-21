"""Multivariate Kalman INNOVATION detector for reasoning-error detection.

WHAT THIS IS
------------
Tests one falsifiable claim: "reasoning failure = loss of observability", operationalized
as -- the multivariate INNOVATION nu_t (how far a step's observation deviates from what a
healthy-reasoning state-space model predicts) rises before/at the error, and detects errors
better than the raw observation.

WHY MULTIVARIATE (vs the previous 1-D version)
----------------------------------------------
A scalar latent state + single observation channel likely crushes the signal. Here:
    x_t in R^k   (low-dim latent reasoning "health", k small)
    y_t in R^m   (observation = MANY per-step signals: layers x metrics, + uncertainty)
    x_t = A x_{t-1} + w_t,  w~N(0,Q)
    y_t = C x_t + d + v_t,  v~N(0,R)
The data-dependent quantity is the innovation nu_t = y_t - C xhat_{t|t-1} - d, normalized by
its predicted covariance S_t via the Mahalanobis norm z2_t = nu_t^T S_t^{-1} nu_t. In a
linear KF the covariance P_t is data-independent (Riccati), so we do NOT use P_t -- only nu_t.

Params (A,C,Q,R,d) learned by EM on CORRECT chains only (healthy dynamics), cross-fit by
chain (GroupKFold) so innovations on error chains carry no leakage.

DIAGNOSTICS (so a null result is interpretable)
-----------------------------------------------
  * raw-observation AUROC (no temporal model) -- the baseline to beat
  * supervised within-step probe AUROC on the SAME features -- the information ceiling;
    if the probe sees signal but the innovation does not, the *filter model* is too weak,
    not the signal absent. If neither sees signal, the claim itself is likely false here.
  * event study: z2_t by offset from first error -- does deviation rise BEFORE the error?

INPUT npz: stepcloud/stepgeom (per-step, per-layer features) + step_token_ranges +
gold_error_step (=-1 for correct, else first-error step index). Optionally extra per-step
channels (e.g. uncertainty) concatenated in.
"""

from __future__ import annotations
import argparse
import numpy as np


# ----------------------------- metrics -----------------------------
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
    """length-matched AUROC: average within n_token buckets, weighted by pair count."""
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        a = bdir(auroc(s[m], y[m]))
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


# ----------------------------- EM for linear-Gaussian SSM -----------------------------
def em_ssm(chains, k, n_iter=30, reg=1e-4, seed=0):
    """EM on a list of (T_i x m) observation arrays. Returns A,C,Q,R,d,mu0,P0.

    Standard linear-Gaussian state-space EM (Ghahramani & Hinton style), m observations,
    k latent dims. Kept compact; regularized for stability on short chains.
    """
    rng = np.random.default_rng(seed)
    m = chains[0].shape[1]
    d = np.concatenate(chains, 0).mean(0)                      # observation offset
    Yc = [Y - d for Y in chains]
    # init
    A = 0.9 * np.eye(k)
    C = rng.standard_normal((m, k)) * 0.1
    # seed C with top-k PCA directions of the data (better init)
    allY = np.concatenate(Yc, 0)
    U, S, Vt = np.linalg.svd(allY - allY.mean(0), full_matrices=False)
    C = Vt[:k].T * (S[:k] / np.sqrt(len(allY)))                # (m,k)
    Q = np.eye(k); R = np.diag(allY.var(0) + reg); mu0 = np.zeros(k); P0 = np.eye(k)

    def kalman_smooth(Y):
        T = Y.shape[0]
        xp = np.zeros((T, k)); Pp = np.zeros((T, k, k))
        xf = np.zeros((T, k)); Pf = np.zeros((T, k, k))
        # filter
        xpre, Ppre = mu0.copy(), P0.copy()
        for t in range(T):
            xp[t], Pp[t] = xpre, Ppre
            S_ = C @ Ppre @ C.T + R
            Sinv = np.linalg.inv(S_ + reg * np.eye(m))
            K = Ppre @ C.T @ Sinv
            nu = Y[t] - C @ xpre
            xf[t] = xpre + K @ nu
            Pf[t] = (np.eye(k) - K @ C) @ Ppre
            xpre = A @ xf[t]; Ppre = A @ Pf[t] @ A.T + Q
        # RTS smoother
        xs = xf.copy(); Ps = Pf.copy(); Plag = np.zeros((T, k, k))
        for t in range(T - 2, -1, -1):
            J = Pf[t] @ A.T @ np.linalg.inv(Pp[t + 1] + reg * np.eye(k))
            xs[t] = xf[t] + J @ (xs[t + 1] - xp[t + 1])
            Ps[t] = Pf[t] + J @ (Ps[t + 1] - Pp[t + 1]) @ J.T
            Plag[t + 1] = J @ Ps[t + 1]
        return xs, Ps, Plag

    for _ in range(n_iter):
        S11 = np.zeros((k, k)); S10 = np.zeros((k, k)); S00 = np.zeros((k, k))
        Syx = np.zeros((m, k)); Sxx = np.zeros((k, k))
        Syy = np.zeros((m, m)); Ntot = 0; Ntrans = 0
        mu0a = np.zeros(k); P0a = np.zeros((k, k)); nchain = 0
        for Y in Yc:
            T = Y.shape[0]
            xs, Ps, Plag = kalman_smooth(Y)
            mu0a += xs[0]; P0a += Ps[0] + np.outer(xs[0], xs[0]); nchain += 1
            for t in range(T):
                Sxx += Ps[t] + np.outer(xs[t], xs[t])
                Syx += np.outer(Y[t], xs[t]); Syy += np.outer(Y[t], Y[t]); Ntot += 1
            for t in range(1, T):
                S11 += Ps[t] + np.outer(xs[t], xs[t])
                S00 += Ps[t - 1] + np.outer(xs[t - 1], xs[t - 1])
                S10 += Plag[t] + np.outer(xs[t], xs[t - 1]); Ntrans += 1
        A = S10 @ np.linalg.inv(S00 + reg * np.eye(k))
        Q = (S11 - A @ S10.T) / max(Ntrans, 1) + reg * np.eye(k)
        C = Syx @ np.linalg.inv(Sxx + reg * np.eye(k))
        R = (Syy - C @ Syx.T) / max(Ntot, 1) + reg * np.eye(m)
        Q = 0.5 * (Q + Q.T); R = 0.5 * (R + R.T)
        mu0 = mu0a / max(nchain, 1); P0 = P0a / max(nchain, 1) - np.outer(mu0, mu0) + reg * np.eye(k)
    return A, C, Q, R, d, mu0, P0


def innovations(Y, A, C, Q, R, d, mu0, P0, reg=1e-4):
    """per-step Mahalanobis innov z2_t = nu^T S^-1 nu AND raw innovation vector nu_t.

    Returns (z2 [T], NU [T,m]); both nan at t=0. NU keeps the signed/directional info that
    the scalar z2 squares away -- so a supervised probe on NU can tell whether the filter
    PRESERVES the signal (probe-on-nu ~ probe-on-raw) or the low z2 is just the squaring.
    """
    T, m = Y.shape; k = A.shape[0]
    z2 = np.full(T, np.nan); NU = np.full((T, m), np.nan)
    xpre, Ppre = mu0.copy(), P0.copy()
    for t in range(T):
        nu = (Y[t] - d) - C @ xpre
        S_ = C @ Ppre @ C.T + R
        Sinv = np.linalg.inv(S_ + reg * np.eye(m))
        if t >= 1:
            z2[t] = float(nu @ Sinv @ nu); NU[t] = nu
        K = Ppre @ C.T @ Sinv
        xf = xpre + K @ nu; Pf = (np.eye(k) - K @ C) @ Ppre
        xpre = A @ xf; Ppre = A @ Pf @ A.T + Q
    return z2, NU


# ----------------------------- data loading -----------------------------
def load_chains(npz_path, layers_sel, metrics_sel):
    z = np.load(npz_path, allow_pickle=True)
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gnames = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]
    SC = z["stepcloud"] if "stepcloud" in z.files else None
    SG = z["stepgeom"] if "stepgeom" in z.files else None
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    def feat(src, names, name, li):
        fi = names.index(name)
        out = []
        for i in range(len(src)):
            g = np.asarray(src[i], float)
            out.append(g[:, li, fi] if g.ndim == 3 else g[:, li])
        return out

    # build list of (m) channels, each a list-of-chains of 1-D arrays
    channels = []
    for lyr in layers_sel:
        li = layers.index(lyr)
        for mt in metrics_sel:
            if mt in cnames and SC is not None:
                channels.append(feat(SC, cnames, mt, li))
            elif mt in gnames and SG is not None:
                channels.append(feat(SG, gnames, mt, li))
            else:
                raise ValueError(f"metric {mt} not found")
    N = len(channels[0]); chains = []
    for i in range(N):
        cols = [ch[i] for ch in channels]
        T = min(len(c) for c in cols)
        Y = np.stack([c[:T] for c in cols], 1)               # (T, m)
        rng = np.asarray(SR[i], int); k = int(ges[i])
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(min(T, rng.shape[0]))], float)
        if np.isfinite(Y).all() and T >= 3 and len(nt) == T:
            chains.append({"Y": Y, "k": k, "nt": nt, "correct": k < 0})
    return chains


# ----------------------------- main -----------------------------
def run(chains, k_latent, folds, seed=0, resultant_idx=None):
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    idx = np.arange(len(chains))
    Zall = {i: None for i in idx}
    gkf = GroupKFold(folds)
    for tr, te in gkf.split(idx, idx, idx):
        Ytr = [chains[t]["Y"] for t in tr if chains[t]["correct"]]
        if len(Ytr) < 20:
            continue
        params = em_ssm(Ytr, k_latent, seed=seed)
        for t in te:
            Zall[t] = innovations(chains[t]["Y"], *params)

    # assemble labeled steps (j<k or correct -> 0; j==k -> 1; j>k skipped); GRP=chain id
    RAWnorm, Z2, CUM, Y, NT, FEAT, NUF, GRP = [], [], [], [], [], [], [], []
    EOFF, EZ2 = [], []
    for t in idx:
        if Zall[t] is None:
            continue
        z2, NU = Zall[t]
        c = chains[t]; kk = c["k"]; correct = c["correct"]; T = c["Y"].shape[0]
        cum = 0.0
        rawmag = np.linalg.norm(c["Y"], axis=1)              # raw obs magnitude (broken baseline)
        for j in range(T):
            if j >= 1 and np.isfinite(z2[j]):
                cum += z2[j]
            if j == 0:
                continue
            if not correct:
                EOFF.append(j - kk); EZ2.append(z2[j])
            if correct or j < kk:
                lab = 0
            elif j == kk:
                lab = 1
            else:
                continue
            RAWnorm.append(rawmag[j]); Z2.append(z2[j]); CUM.append(cum)
            Y.append(lab); NT.append(c["nt"][j]); FEAT.append(c["Y"][j])
            NUF.append(NU[j]); GRP.append(t)
    RAWnorm = np.asarray(RAWnorm); Z2 = np.asarray(Z2); CUM = np.asarray(CUM)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); FEAT = np.asarray(FEAT)
    NUF = np.asarray(NUF); GRP = np.asarray(GRP)
    EOFF = np.asarray(EOFF, int); EZ2 = np.asarray(EZ2, float)

    # signed directional innovation: mean SIGNED nu over resultant channels (unsquared) --
    # the multivariate analog of the 1-D signed innovation that DID spike at the error.
    if not resultant_idx:
        resultant_idx = list(range(NUF.shape[1]))
    SIGNED = NUF[:, resultant_idx].mean(axis=1)

    # supervised probes (grouped CV, leak-free): on RAW features (static ceiling) and on the
    # innovation vector NU. probe-on-nu ~ probe-on-raw => filter preserves info (z2 squaring is
    # the culprit). probe-on-nu << probe-on-raw => innovation removed the between-chain signal.
    def probe(X):
        out = np.full(len(Y), np.nan)
        if len(np.unique(GRP)) < folds:
            return out
        for tr, te in GroupKFold(folds).split(X, Y, GRP):
            if Y[tr].sum() < 3 or (Y[tr] == 0).sum() < 3:
                continue
            sc = StandardScaler().fit(X[tr])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced")
            lr.fit(sc.transform(X[tr]), Y[tr])
            out[te] = lr.decision_function(sc.transform(X[te]))
        return out
    probe_raw = probe(FEAT)
    probe_nu = probe(NUF)

    print(f"\n=== AUROC (pooled / length-bucket) — k_latent={k_latent} ===")
    print(f"  {'feature':30s} {'pooled':>8s} {'bucket':>8s}")
    rows = [("raw obs ||y|| (broken baseline)", RAWnorm),
            ("z2 (Mahalanobis innov)", Z2),
            ("signed innov (resultant dir)", SIGNED),
            ("cumulative z2", CUM),
            ("probe on RAW feats (ceiling)", probe_raw),
            ("probe on innovation nu", probe_nu)]
    for nm, v in rows:
        print(f"  {nm:30s} {bdir(auroc(v, Y)):8.3f} {bucket(v, Y, NT):8.3f}")

    print(f"\n=== event study: mean z2 by offset from first error (Δ=0) ===")
    print(f"  {'Δ=j-k':>6s} {'n':>5s} {'mean z2':>10s} {'SE':>8s}")
    for dd in range(-4, 4):
        m = EOFF == dd
        if m.sum() >= 5:
            se = EZ2[m].std() / np.sqrt(m.sum())
            star = " <-- error" if dd == 0 else ""
            print(f"  {dd:>6d} {int(m.sum()):>5d} {EZ2[m].mean():>10.3f} {se:>8.3f}{star}")
    pre = EZ2[(EOFF <= -3) & np.isfinite(EZ2)]; at0 = EZ2[(EOFF == 0) & np.isfinite(EZ2)]
    if len(pre) >= 5 and len(at0) >= 5:
        jump = at0.mean() - pre.mean()
        se = np.sqrt(at0.std()**2 / len(at0) + pre.std()**2 / len(pre))
        sig = "SIGNIFICANT" if abs(jump) - 2 * se > 0 else "ns"
        print(f"  jump z2(0) − z2(≤-3) = {jump:+.3f} [{jump-2*se:+.3f},{jump+2*se:+.3f}] {sig}")

    # interpretation: separate "filter destroys info" from "innovation removes difficulty (by design)"
    a_pr = bdir(auroc(probe_raw, Y)); a_pn = bdir(auroc(probe_nu, Y))
    a_sg = bdir(auroc(SIGNED, Y)); a_z2 = bdir(auroc(Z2, Y))
    print("\n=== DIAGNOSIS ===")
    print(f"  static ceiling (probe on raw)  = {a_pr:.3f}")
    print(f"  probe on innovation nu         = {a_pn:.3f}")
    print(f"  signed directional innov       = {a_sg:.3f}   (Mahalanobis z2 = {a_z2:.3f})")
    if np.isfinite(a_sg) and np.isfinite(a_z2) and a_sg > a_z2 + 0.05:
        print(f"  -> SIGNED ({a_sg:.3f}) >> z2 ({a_z2:.3f}): the error is a DIRECTIONAL drop; the\n"
              f"     Mahalanobis squaring/whitening discards the sign. Use the signed innovation.")
    if np.isfinite(a_pn) and np.isfinite(a_pr):
        if a_pn >= a_pr - 0.03:
            print(f"  -> filter PRESERVES the information (probe-on-nu {a_pn:.3f} ~ probe-on-raw {a_pr:.3f}):\n"
                  f"     the low z2 is the scalar readout, NOT a weak filter. Nonlinear won't help the readout.")
        else:
            print(f"  -> innovation drops the ceiling by {a_pr - a_pn:.3f}: the filter removed the\n"
                  f"     BETWEEN-chain (difficulty/absolute-level) component. The remainder is the\n"
                  f"     within-chain signal. A nonlinear filter is STILL within-chain -> won't recover\n"
                  f"     the between-chain signal. The signal is static/absolute, not dynamical.")


def make_synthetic(n=200, seed=0):
    """synthetic SSM data to verify the pipeline runs and innovation beats raw when an
    error injects a transient dynamics break. Mirrors the expected npz structure loosely."""
    rng = np.random.default_rng(seed)
    chains = []
    A = np.array([[0.9, 0.05], [0.0, 0.85]]); C = rng.standard_normal((4, 2))
    for i in range(n):
        correct = rng.random() < 0.6
        T = rng.integers(6, 18)
        x = rng.standard_normal(2) * 0.3
        Y = np.zeros((T, 4)); kk = -1
        err_t = -1 if correct else int(rng.integers(2, T - 1))
        for t in range(T):
            x = A @ x + rng.standard_normal(2) * 0.2
            if (not correct) and t == err_t:
                x = x + rng.standard_normal(2) * 1.5        # dynamics break at error
            Y[t] = C @ x + rng.standard_normal(4) * 0.3
        kk = err_t
        rngrng = np.array([[t * 10, t * 10 + 9] for t in range(T)])
        chains.append({"Y": Y, "k": kk, "nt": np.full(T, 10.0), "correct": correct})
    return chains


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", nargs="?", default=None)
    ap.add_argument("--layers", default="14", help="comma list, e.g. 14,20,26")
    ap.add_argument("--metrics", default="resultant", help="comma list of feature names")
    ap.add_argument("--k_latent", type=int, default=2)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--selftest", action="store_true", help="run on synthetic data")
    args = ap.parse_args()

    if args.selftest or args.npz is None:
        print("[selftest] synthetic SSM with error-induced dynamics breaks")
        chains = make_synthetic()
        print(f"  chains={len(chains)} correct={sum(c['correct'] for c in chains)}")
        run(chains, args.k_latent, args.folds)
        return

    layers_sel = [int(x) for x in args.layers.split(",")]
    metrics_sel = [x.strip() for x in args.metrics.split(",")]
    chains = load_chains(args.npz, layers_sel, metrics_sel)
    # channel order is (for lyr in layers_sel: for mt in metrics_sel) -> resultant channel indices
    ri = [li * len(metrics_sel) + mi for li in range(len(layers_sel))
          for mi, mt in enumerate(metrics_sel) if mt == "resultant"]
    print(f"file: {args.npz}")
    print(f"layers={layers_sel} metrics={metrics_sel} -> obs dim m={len(layers_sel)*len(metrics_sel)}")
    print(f"chains={len(chains)} correct={sum(c['correct'] for c in chains)} "
          f"error={sum(not c['correct'] for c in chains)}")
    run(chains, args.k_latent, args.folds, resultant_idx=ri)


if __name__ == "__main__":
    main()