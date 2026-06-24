"""Second-moment (Bingham/Watson) descriptors of the unit-token-vector cloud, beyond the first-moment kappa.
Per step: scatter eigenspectrum (lam1, eff-rank, gap, residual) + alignment-to-mean distribution (var, anti-frac).
Tests whether they add over [kappa + length] (strict, OOF + bootstrap CI) + standalone AUROC + bucket.
Key target: axial/bipolar steps where mean cancels (kappa low) but a dominant AXIS exists (lam1 high) -- kappa is blind to these."""
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


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    KA, LAM1, EFFR, GAP, RESID, AVAR, AFRAC, NT, Y, G = ([] for _ in range(10))
    EV5 = []   # top-5 scatter eigenvalues = the spectrum AS A VECTOR (richer than a scalar)
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0])
        k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9; U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(H), int(rng[j, 1]) - a0 + 1)
            Uj = U[lo:hi][ok[lo:hi]]
            n = len(Uj)
            if n < 3:
                continue
            w = np.exp(np.arange(n) / (n - 1)); w /= w.sum()
            mu = w @ Uj; ka = np.linalg.norm(mu); muhat = mu / ka if ka > 1e-9 else mu
            ev = np.linalg.eigvalsh(Uj @ Uj.T)[::-1] / n; ev = np.clip(ev, 1e-12, None); ev = ev / ev.sum()
            a = Uj @ muhat
            KA.append(ka); LAM1.append(ev[0]); EFFR.append(float(np.exp(-(ev * np.log(ev)).sum())))
            GAP.append(ev[0] - (ev[1] if n > 1 else 0.0)); RESID.append(1 - ev[0])
            AVAR.append(float(a.var())); AFRAC.append(float((a < 0).mean()))
            EV5.append(np.pad(ev[:5], (0, max(0, 5 - len(ev))))[:5])
            NT.append(n); Y.append(lab); G.append(i)
    KA = np.asarray(KA); NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    feats = {"lam1": LAM1, "eff_rank": EFFR, "gap": GAP, "residual": RESID, "align_var": AVAR, "anti_frac": AFRAC}
    feats = {k: np.asarray(v, float) for k, v in feats.items()}
    print(f"{args.npz} | L{args.layer} | steps {len(Y)} err {int(Y.sum())}")
    print(f"  kappa (1st moment)   AUROC {bdir(auroc(-KA, Y)):.3f}  bkt {bucket(-KA, Y, NT):.3f}")
    for nm, v in feats.items():
        print(f"  {nm:20s} AUROC {bdir(auroc(v, Y)):.3f}  bkt {bucket(v, Y, NT):.3f}")
    EV5 = np.asarray(EV5)
    print(f"  spectrum top-5 (vector)  AUROC {bdir(auroc(oof([EV5[:, c] for c in range(5)], Y, G), Y)):.3f}  bkt {bucket(oof([EV5[:, c] for c in range(5)], Y, G), Y, NT):.3f}")

    ch = np.unique(G); rng = np.random.default_rng(0); ci = {c: np.where(G == c)[0] for c in ch}

    def strict(name, extra):
        base = oof([-KA, NT], Y, G); full = oof([-KA, NT] + extra, Y, G)
        ds = []
        for _ in range(200):
            idx = np.concatenate([ci[c] for c in rng.choice(ch, len(ch), replace=True)])
            ds.append(auroc(full[idx], Y[idx]) - auroc(base[idx], Y[idx]))
        cl, cu = np.percentile(ds, [2.5, 97.5])
        print(f"  STRICT {name:22s} over [kappa+len]  {auroc(base,Y):.3f} -> {auroc(full,Y):.3f}  (+{auroc(full,Y)-auroc(base,Y):.3f}, CI [{cl:+.3f}, {cu:+.3f}])")

    # minimal principled combinations: keep the KEY part, drop redundant eigenvalue summaries
    strict("eff_rank only", [feats["eff_rank"]])
    strict("lam1 + eff_rank", [feats["lam1"], feats["eff_rank"]])
    strict("spectrum top-5", [EV5[:, c] for c in range(5)])
    strict("all 2nd-moment feats", list(feats.values()))

    # combination: does a multiplicative kappa x eff_rank term add over the two linear terms?
    def strict_b(name, base, extra):
        sb = oof(base, Y, G); sf = oof(base + extra, Y, G); ds = []
        for _ in range(200):
            idx = np.concatenate([ci[c] for c in rng.choice(ch, len(ch), replace=True)])
            ds.append(auroc(sf[idx], Y[idx]) - auroc(sb[idx], Y[idx]))
        cl, cu = np.percentile(ds, [2.5, 97.5])
        print(f"  COMBO {name:30s} {auroc(sb,Y):.3f} -> {auroc(sf,Y):.3f}  (+{auroc(sf,Y)-auroc(sb,Y):.3f}, CI [{cl:+.3f}, {cu:+.3f}])")

    er = feats["eff_rank"]
    print(f"  kappa*eff_rank (product) AUROC {bdir(auroc(KA * er, Y)):.3f}")
    strict_b("kappa*eff_rank over [k+eff+len]", [-KA, er, NT], [KA * er])
    strict_b("kappa/eff_rank over [k+eff+len]", [-KA, er, NT], [KA / np.maximum(er, 1e-9)])

    # SIGN (mechanism): >0.5 = HIGH value is error. Compare across configs -- does eff_rank FLIP easy vs hard?
    print(f"  SIGN raw AUROC (>0.5: high=error)  eff_rank {auroc(er, Y):.3f}  lam1 {auroc(feats['lam1'], Y):.3f}  kappa {auroc(KA, Y):.3f}")
    # DECISIVE: rule out residual NONLINEAR length -- put log n, 1/n, sqrt n, n^2 into the baseline
    NL = [NT, np.log(NT + 1.0), 1.0 / np.maximum(NT, 1.0), np.sqrt(NT), NT ** 2]
    strict_b("spectrum5 over [kappa+NONLIN-len]", [-KA] + NL, [EV5[:, c] for c in range(5)])
    strict_b("eff_rank  over [kappa+NONLIN-len]", [-KA] + NL, [er])


if __name__ == "__main__":
    main()
