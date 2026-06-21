"""Deepen the 'directional consistency' axis beyond the resultant SCALAR.

resultant = ||sum_t w_t uhat_t|| is only the within-step CONCENTRATION of the mean direction.
It throws away the step DIRECTION VECTOR itself -- and therefore WHAT that direction aligns
with. Using the stored full-dim pooled step vectors (stepvec) and the question vector (qvec),
we test the unexplored, magnitude-immune cells:

  step direction   d_j = normalize(stepvec[j])           question direction d_q = normalize(qvec)
  -- external anchoring (the anchor, in representation space) --
  anchor_q = cos(d_j, d_q)                  is the step still pointing at the problem?
  -- trajectory directional coherence --
  coh_prev = cos(d_j, d_{j-1})              turn from the previous step
  coh_run  = cos(d_j, normalize(sum_{t<j} d_t))   consistency with the reasoning so far

Internal concentration (resultant) tells whether the tokens within a step agree; these tell
whether the step points at the QUESTION (anchor_q) and stays consistent with the trajectory
(coh_*). A step can be internally concentrated yet collectively drift off the question --
resultant cannot see that.

VERDICT per signal: single AUROC (pooled + length-bucket); residualized-on-resultant AUROC
(cross-fit GBR on correct steps) -- survives removing the known signal; GroupKFold logistic
INCREMENT over [resultant + position] with chain-paired bootstrap CI ('*' = CI clears 0 = a NEW
axis). Position is a control because later steps naturally drift from the question.

Needs npz from extract_features.py --store_step_vectors (stepvec + qvec) + stepcloud(resultant).
"""

from __future__ import annotations
import argparse
import numpy as np

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    raise SystemExit("needs scikit-learn")


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
        m = b == bb
        a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def residualize_on(sig, base, correct, grp, folds):
    out = np.full(len(sig), np.nan); X = base.reshape(-1, 1)
    for tr, te in GroupKFold(folds).split(X, np.zeros(len(X)), grp):
        ctr = tr[correct[tr]]
        if len(ctr) < 50:
            continue
        reg = GradientBoostingRegressor(n_estimators=120, max_depth=3, random_state=0)
        reg.fit(X[ctr], sig[ctr]); out[te] = sig[te] - reg.predict(X[te])
    return out


def oof_logit(X, y, grp, folds):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def boot_delta(sa, sb, y, grp, n=1000, seed=0):
    rng = np.random.default_rng(seed); chains = np.unique(grp)
    idx_by = {c: np.where(grp == c)[0] for c in chains}
    d0 = bdir(auroc(sb, y)) - bdir(auroc(sa, y)); ds = []
    for _ in range(n):
        pick = rng.choice(chains, len(chains), replace=True)
        ii = np.concatenate([idx_by[c] for c in pick])
        a = auroc(sa[ii], y[ii]); b = auroc(sb[ii], y[ii])
        if np.isfinite(a) and np.isfinite(b):
            ds.append(max(b, 1 - b) - max(a, 1 - a))
    lo, hi = np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan)
    return d0, lo, hi


def unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("no step vectors; re-extract with --store_step_vectors --sv_layers ...")
    svl = [int(x) for x in (z["sv_layers"] if "sv_layers" in z.files else z["layers_used"])]
    if args.layer not in svl:
        raise SystemExit(f"layer {args.layer} not in stored sv_layers {svl} -- pick one of these")
    lisv = svl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; lic = lyu.index(args.layer)
    SV, QV, SC = z["stepvec"], z["qvec"], z["stepcloud"]
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    feats = {k: [] for k in ["anchor_q", "coh_prev", "coh_run", "res"]}
    Y, NT, POS, G = [], [], [], []
    for i in range(len(SV)):
        sv = np.asarray(SV[i], np.float32)
        if sv.ndim != 3 or sv.shape[1] <= lisv:
            continue
        Z = sv[:, lisv, :]                                          # (T, d) pooled step vectors
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int)
        k = int(ges[i]); correct = (k < 0); T = Z.shape[0]
        q = unit(np.asarray(QV[i], np.float32)[lisv]) if QV[i] is not None else None
        res_seq = sc[:, lic, cnames.index("resultant")] if "resultant" in cnames else np.full(T, np.nan)
        dlist = [unit(Z[j]) for j in range(T)]

        run = np.zeros(Z.shape[1]); have_run = False
        for j in range(T):
            dj = dlist[j]
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            aq = cp = cr = np.nan
            if dj is not None:
                if q is not None:
                    aq = float(dj @ q)
                if j >= 1 and dlist[j - 1] is not None:
                    cp = float(dj @ dlist[j - 1])
                if have_run and np.linalg.norm(run) > 1e-9:
                    cr = float(dj @ (run / np.linalg.norm(run)))
                run = run + dj; have_run = True
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            feats["anchor_q"].append(aq); feats["coh_prev"].append(cp); feats["coh_run"].append(cr)
            feats["res"].append(res_seq[j])
            Y.append(y); NT.append(ntok); POS.append(j / max(1, T - 1)); G.append(i)

    for kk in feats:
        c = np.asarray(feats[kk], float)
        c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        feats[kk] = c
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); POS = np.asarray(POS, float); G = np.asarray(G, int)
    correct_step = Y == 0; res = feats["res"]

    print(f"file: {args.npz} | layer {args.layer} (sv_layers {svl}) | labeled steps {len(Y)} | "
          f"first-error {int(Y.sum())} | error-chains {len(np.unique(G[Y==1]))}")
    print(f"baseline resultant AUROC {bdir(auroc(res,Y)):.3f}  bucket {bucket(res,Y,NT):.3f}  "
          f"| position AUROC {bdir(auroc(POS,Y)):.3f}")

    base = np.c_[res, POS]; sa = oof_logit(base, Y, G, args.folds)
    print(f"\n{'signal':10s} {'AUROC':>7s} {'bucket':>7s} {'resid⊥res':>10s} {'Δ[+res,pos]':>12s} {'95% CI':>18s}")
    for nm in ["anchor_q", "coh_prev", "coh_run"]:
        sig = feats[nm]
        a = bdir(auroc(sig, Y)); bk = bucket(sig, Y, NT)
        rsd = residualize_on(sig, res, correct_step, G, args.folds); ar = bdir(auroc(rsd, Y))
        sb = oof_logit(np.c_[res, POS, sig], Y, G, args.folds)
        d0, lo, hi = boot_delta(sa, sb, Y, G, n=args.boot)
        flag = " *" if lo > 0 else ""
        print(f"{nm:10s} {a:7.3f} {bk:7.3f} {ar:10.3f} {d0:+12.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s}{flag}")

    print("\nread: resid⊥res = AUROC after regressing the signal on resultant (cross-fit on "
          "correct) -> survives if > ~0.55. Δ[+res,pos] = logistic increment over [resultant + "
          "position] with chain-paired bootstrap CI; '*' = CI clears 0 = a NEW directional-"
          "consistency axis (internal concentration + external/trajectory anchoring). Else it "
          "collapses into resultant+position (honest negative). anchor_q controls position "
          "because later steps naturally drift from the question.")


if __name__ == "__main__":
    main()
