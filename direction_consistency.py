"""Deepen the 'directional consistency' axis.

resultant is only the SCALAR concentration of the within-step mean direction. It throws away
(i) the step DIRECTION VECTOR itself and (ii) the higher-order shape of the direction
distribution. This script tests the UNEXPLORED cells, all magnitude-immune, in the stored JL
space (JL approximately preserves cosines):

  step direction        d_j = normalize( sum_t w_t * uhat_t )         (w = exp-pool, uhat=unit token)
  -- (2) BETWEEN-step directional coherence (trajectory) --
  coh_prev = cos(d_j, d_{j-1})                       turn from the previous step
  coh_run  = cos(d_j, normalize(sum_{t<j} d_t))      consistency with the reasoning so far
  -- (3) WITHIN-step directional higher-order shape (multimodality) --
  dir_lam2 = lambda_2 / sum(lambda)  of  sum_t w_t uhat_t uhat_t^T    second-direction strength = split
  dir_D    = exp(-sum p_i log p_i), p_i=lambda_i/sum                  directional effective rank
  -- baseline recomputed in the SAME JL space --
  res_jl   = || sum_t w_t uhat_t ||

VERDICT per signal: (a) single AUROC pooled + length-bucket; (b) residualized-on-resultant
AUROC (cross-fit on correct steps) -- does it survive removing the known signal; (c) GroupKFold
logistic INCREMENT of [resultant] -> [resultant + signal] with chain-paired bootstrap CI.
Position is included as a control feature in (c). If a signal survives (b) AND the increment CI
clears 0, it is a NEW directional-consistency axis (write it up). If not, it collapses into
resultant -- honest negative.

Needs npz with respcloud (--store_clouds) + stepcloud(resultant) + step_token_ranges +
gold_error_step + cloud_store_layers.
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


def exp_w(n):
    if n <= 1:
        return np.ones(max(n, 1))
    e = np.exp(np.arange(n) / (n - 1)); return e / e.sum()


def step_dir_stats(tok):
    """tok: (n,d) raw token vectors -> (d_unit, res_jl, dir_lam2, dir_D)."""
    n = tok.shape[0]
    nrm = np.linalg.norm(tok, axis=1)
    ok = nrm > 1e-9
    if ok.sum() < 2:
        return None
    u = tok[ok] / nrm[ok, None]; w = exp_w(u.shape[0])
    pooled = (w[:, None] * u).sum(0); res = float(np.linalg.norm(pooled))
    d = pooled / (res + 1e-12)
    # nonzero eigenvalues of sum_t w_t u_t u_t^T via the (n x n) weighted Gram (cheap, same spectrum)
    B = np.sqrt(w)[:, None] * u                       # (n,d)
    G = B @ B.T                                        # (n,n), trace = sum w = 1
    lam = np.linalg.eigvalsh(G); lam = np.clip(lam[::-1], 0, None); s = lam.sum()
    if s <= 1e-12:
        return None
    p = lam / s; p = p[p > 1e-12]
    dir_D = float(np.exp(-(p * np.log(p)).sum()))
    lam2 = float(lam[1] / s) if len(lam) > 1 else 0.0
    return d, res, lam2, dir_D


def residualize_on(sig, base, correct, grp, folds):
    """cross-fit GBR(sig ~ base) on CORRECT steps; return residual sig - pred (all steps)."""
    out = np.full(len(sig), np.nan)
    X = base.reshape(-1, 1)
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
    """chain-paired bootstrap CI of AUROC(sb) - AUROC(sa)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_tok", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files:
        raise SystemExit("no respcloud; re-extract with --store_clouds")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    RC, SC, SR = z["respcloud"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)

    feats = {k: [] for k in ["coh_prev", "coh_run", "dir_lam2", "dir_D", "res_jl", "res_stored"]}
    Y, NT, POS, G = [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        res_seq = sc[:, li, cnames.index("resultant")] if "resultant" in cnames else np.full(T, np.nan)

        # pass 1: step directions + within-step shape for ALL steps (history needed for coh_run)
        dlist = [None] * T; lam2 = np.full(T, np.nan); dirD = np.full(T, np.nan); rjl = np.full(T, np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < args.min_tok:
                continue
            st = step_dir_stats(rcl[lo:hi])
            if st is None:
                continue
            dlist[j], rjl[j], lam2[j], dirD[j] = st

        # pass 2: labeled steps -> between-step coherence + collect.
        # coh_run uses the running mean of PRIOR step directions, so update run AFTER scoring j.
        # post-error steps (j>k) come after all labeled steps -> never history -> safe to skip.
        run = np.zeros(rcl.shape[1]); have_run = False
        for j in range(T):
            dj = dlist[j]
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            cohp = cohr = np.nan
            if dj is not None:
                if j >= 1 and dlist[j - 1] is not None:
                    cohp = float(dj @ dlist[j - 1])
                if have_run and np.linalg.norm(run) > 1e-9:
                    rd = run / np.linalg.norm(run); cohr = float(dj @ rd)
                run = run + dj; have_run = True              # add this step to history
            ntok = int(rng[j, 1] - rng[j, 0] + 1)
            feats["coh_prev"].append(cohp); feats["coh_run"].append(cohr)
            feats["dir_lam2"].append(lam2[j]); feats["dir_D"].append(dirD[j])
            feats["res_jl"].append(rjl[j]); feats["res_stored"].append(res_seq[j])
            Y.append(y); NT.append(ntok); POS.append(j / max(1, T - 1)); G.append(i)

    for kk in feats:
        c = np.asarray(feats[kk], float)
        c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        feats[kk] = c
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); POS = np.asarray(POS, float); G = np.asarray(G, int)
    correct_step = Y == 0
    res = feats["res_stored"]

    print(f"file: {args.npz} | layer {args.layer} | labeled steps {len(Y)} | first-error {int(Y.sum())} "
          f"| error-chains {len(np.unique(G[Y==1]))}")
    print(f"baseline  resultant(stored) AUROC {bdir(auroc(res,Y)):.3f}  bucket {bucket(res,Y,NT):.3f} | "
          f"res_jl {bdir(auroc(feats['res_jl'],Y)):.3f}  (corr res_jl,stored "
          f"{np.corrcoef(feats['res_jl'],res)[0,1]:+.2f})")

    base2 = np.c_[res, POS]                                  # resultant + position baseline
    sa = oof_logit(base2, Y, G, args.folds)
    print(f"\n{'signal':10s} {'AUROC':>7s} {'bucket':>7s} {'resid⊥res':>10s} {'Δ[+res,pos]':>12s} {'95% CI':>18s}")
    for nm in ["coh_prev", "coh_run", "dir_lam2", "dir_D"]:
        sig = feats[nm]
        a = bdir(auroc(sig, Y)); bk = bucket(sig, Y, NT)
        rsd = residualize_on(sig, res, correct_step, G, args.folds)
        ar = bdir(auroc(rsd, Y))
        sb = oof_logit(np.c_[res, POS, sig], Y, G, args.folds)
        d0, lo, hi = boot_delta(sa, sb, Y, G, n=args.boot)
        flag = " *" if lo > 0 else ""
        print(f"{nm:10s} {a:7.3f} {bk:7.3f} {ar:10.3f} {d0:+12.3f} {f'[{lo:+.3f},{hi:+.3f}]':>18s}{flag}")

    print("\nread: AUROC/bucket = single-feature (length-controlled). resid⊥res = AUROC after "
          "regressing the signal on resultant (cross-fit on correct) -> survives if > ~0.55. "
          "Δ[+res,pos] = logistic increment over [resultant+position] with chain-paired bootstrap "
          "CI; '*' = CI clears 0 = a NEW directional-consistency axis. Otherwise it collapses into "
          "resultant (honest negative).")


if __name__ == "__main__":
    main()
