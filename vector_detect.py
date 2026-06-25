"""Discriminative power of the UNIFIED spectral-functional vector [HS, eff-rank D, lam1-frac, log-energy, 2-NN d (+entropy)].
Honest read: pooled OOF AUROC + length bucket, AND within-chain (same problem) + length-controlled localization of the
combined OOF score (the difficulty-free number). Per-component standalone for context. respcloud (JL-256)."""
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


def twonn(H):
    n = len(H)
    if n < 4:
        return np.nan
    G = H @ H.T; sq = np.diag(G); D2 = np.maximum(sq[:, None] + sq[None, :] - 2 * G, 0); np.fill_diagonal(D2, np.inf)
    Ds = np.sqrt(np.sort(D2, axis=1)[:, :2]); r = Ds[:, 1] / np.maximum(Ds[:, 0], 1e-12); r = r[np.isfinite(r) & (r > 1.0)]
    return float(len(r) / np.log(r).sum()) if len(r) >= 3 else np.nan


def feats(H):
    Hc = H - H.mean(0); s = np.linalg.svd(Hc, compute_uv=False); lam = s ** 2; lam = lam[lam > 1e-9]
    if len(lam) < 2:
        return [np.nan] * 5
    p = lam / lam.sum()
    hs = float(np.log(lam).mean())                       # log-volume (seq-normalized log-det)
    D = float(np.exp(-(p * np.log(p)).sum()))            # matrix entropy / eff-rank
    lam1 = float(p[0])                                   # top concentration
    en = float(np.log(lam.sum()))                        # log total energy
    return [hs, D, lam1, en, twonn(H)]


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    names = ["HS", "effrank_D", "lam1", "logE", "twoNN_d"] + (["entropy"] if UD is not None else [])
    X, NT, Y, G = [], [], [], []
    chains = {}   # chain -> list of (score_idx, j, k, nt) for within-chain
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        ud = np.asarray(UD[i], float) if UD is not None else None
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1); Hj = H[lo:hi]
            if hi - lo < 4:
                continue
            f = feats(Hj)
            if UD is not None:
                f = f + [float(np.nanmean(ud[lo:hi]))]
            X.append(f); NT.append(hi - lo); Y.append(lab); G.append(i)
            chains.setdefault(i, []).append((len(X) - 1, j, k))
    X = np.asarray(X); NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    fin = np.isfinite(X).all(1)
    X, NT, Y, G = X[fin], NT[fin], Y[fin], G[fin]
    idmap = {}; nxt = 0
    # OOF on the combined vector
    s = np.full(len(Y), np.nan)
    for tr, te in GroupKFold(5).split(X, Y, G):
        if len(np.unique(Y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], Y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())} | vector = [{', '.join(names)}]")
    for c, nm in enumerate(names):
        print(f"  {nm:12s} pooled {bdir(auroc(X[:, c], Y)):.3f}  bkt {bucket(X[:, c], Y, NT):.3f}")
    print(f"  {'VECTOR(OOF)':12s} pooled {bdir(auroc(s, Y)):.3f}  bkt {bucket(s, Y, NT):.3f}")
    # within-chain (same problem) + length-controlled wc_loc of the combined score
    # rebuild per-chain finite-index mapping
    keep = np.where(fin)[0]; old2new = {old: new for new, old in enumerate(keep)}
    locs, w = [], []
    for ci, lst in chains.items():
        rows = [(old2new[r], j, k) for (r, j, k) in lst if r in old2new]
        if not rows:
            continue
        k = rows[0][2]
        if k < 1:
            continue
        ek = [(ri, j) for (ri, j, _) in rows if j == k]; pr = [(ri, j) for (ri, j, _) in rows if j < k]
        if not ek or len(pr) < 2:
            continue
        eri = ek[0][0]; pri = [ri for (ri, _) in pr]
        sc = s[[eri] + pri]; nt = NT[[eri] + pri]
        if not np.isfinite(sc).all():
            continue
        b = np.polyfit(nt, sc, 1); res = sc - (b[0] * nt + b[1])
        locs.append(np.mean(res[1:] < res[0])); w.append(len(pri))
    if locs:
        print(f"  {'VECTOR within-chain (same problem)+length-controlled wc_loc':s} {np.average(locs, weights=np.asarray(w,float)):.3f}")


if __name__ == "__main__":
    main()
