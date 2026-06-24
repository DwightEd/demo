"""Proposal's three reasoning-spectral-field metrics per (step) at one layer, evaluated honestly.
  D  = effective rank: exp of spectral entropy of the CENTERED covariance eigenvalues (proposal eq 1).
  d  = 2-NN local intrinsic dimension (Facco-Laio): MLE over nearest/2nd-nearest distance ratios (eq 3).
  kc = step curvature: ||h_{j+1} - 2 h_j + h_{j-1}|| on step centroids (eq 4).
Reports POOLED AUROC (single + 3-combined) and the difficulty+length-controlled within-chain wc_loc(perp nt).
Runs on respcloud (JL-256 preserves covariance/distances/norms -> all three valid)."""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


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
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def oof(cols, y, g):
    X = np.column_stack(cols); s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(5).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def eff_rank(H):
    Hc = H - H.mean(0)
    s = np.linalg.svd(Hc, compute_uv=False)
    p = s ** 2; tot = p.sum()
    if tot <= 0 or len(p) < 2:
        return np.nan
    p = p / tot; p = p[p > 1e-12]
    return float(np.exp(-(p * np.log(p)).sum()))


def twonn(H):
    n = len(H)
    if n < 4:
        return np.nan
    G = H @ H.T; sq = np.diag(G); D2 = sq[:, None] + sq[None, :] - 2 * G
    np.fill_diagonal(D2, np.inf); D2 = np.maximum(D2, 0)
    Ds = np.sqrt(np.sort(D2, axis=1)[:, :2])   # nearest, 2nd nearest
    r = Ds[:, 1] / np.maximum(Ds[:, 0], 1e-12)
    r = r[np.isfinite(r) & (r > 1.0)]
    if len(r) < 3:
        return np.nan
    return float(len(r) / np.log(r).sum())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    P_D, P_d, P_k, P_nt, Y, G = [], [], [], [], [], []
    chains = []   # per error chain for within-chain wc_loc
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0])
        k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        D = np.full(T, np.nan); d = np.full(T, np.nan); cent = np.full((T, H.shape[1]), np.nan); nt = np.full(T, np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1); Hj = H[lo:hi]
            nt[j] = hi - lo
            if hi - lo >= 4:
                D[j] = eff_rank(Hj); d[j] = twonn(Hj); cent[j] = Hj.mean(0)
        kc = np.full(T, np.nan)
        for j in range(1, T - 1):
            if np.isfinite(cent[j - 1, 0]) and np.isfinite(cent[j, 0]) and np.isfinite(cent[j + 1, 0]):
                kc[j] = np.linalg.norm(cent[j + 1] - 2 * cent[j] + cent[j - 1])
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            P_D.append(D[j]); P_d.append(d[j]); P_k.append(kc[j]); P_nt.append(nt[j]); Y.append(lab); G.append(i)
        if not correct and T >= 4:
            chains.append({"D": D, "d": d, "kc": kc, "nt": nt, "k": k, "T": T})
    P_D = np.asarray(P_D); P_d = np.asarray(P_d); P_k = np.asarray(P_k); P_nt = np.asarray(P_nt, float); Y = np.asarray(Y, int); G = np.asarray(G, int)

    def wc(name_metric):
        locs, w = [], []
        # sign from pooled direction
        m = {"D": P_D, "d": P_d, "kc": P_k}[name_metric]
        sign = 1.0 if auroc(m, Y) >= 0.5 else -1.0
        for ch in chains:
            v = ch[name_metric]; nt = ch["nt"]; k = ch["k"]; fin = np.isfinite(v) & np.isfinite(nt)
            if fin.sum() < 3 or not fin[k]:
                continue
            b = np.polyfit(nt[fin], v[fin], 1); resid = sign * (v - (b[0] * nt + b[1]))
            others = np.array([j for j in range(len(v)) if j != k and fin[j]])
            if len(others) < 2:
                continue
            locs.append(np.mean(resid[others] < resid[k])); w.append(len(others))
        return np.average(locs, weights=np.asarray(w, float)) if locs else float("nan")

    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())} | err-chains {len(chains)}")
    print(f"  {'metric':18s} {'pooled':>8s} {'pool-bkt':>9s} {'wc_loc(perp nt)':>16s}")
    for nm, arr in [("D eff-rank", P_D), ("d 2-NN intrinsic", P_d), ("kc step-curvature", P_k)]:
        key = {"D eff-rank": "D", "d 2-NN intrinsic": "d", "kc step-curvature": "kc"}[nm]
        print(f"  {nm:18s} {bdir(auroc(arr, Y)):8.3f} {bucket(arr, Y, P_nt):9.3f} {wc(key):16.3f}")
    fin = np.isfinite(P_D) & np.isfinite(P_d) & np.isfinite(P_k)
    s3 = np.full(len(Y), np.nan); s3[fin] = oof([P_D[fin], P_d[fin], P_k[fin]], Y[fin], G[fin])
    print(f"  {'D+d+kc (pooled)':18s} {bdir(auroc(s3, Y)):8.3f} {bucket(s3, Y, P_nt):9.3f}")


if __name__ == "__main__":
    main()
