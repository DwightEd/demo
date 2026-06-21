"""Deepen the 'directional consistency' axis beyond the resultant SCALAR.

resultant = ||sum_t w_t uhat_t|| is only the within-step CONCENTRATION of the mean direction.
It throws away (i) the step DIRECTION VECTOR and (ii) the higher-order shape of the direction
distribution. Using the stored JL token clouds (respcloud, _cloud.npz), we test the
unexplored, magnitude-immune cells (JL preserves cosines):

  step direction  d_j = normalize(sum_t w_t uhat_t)        (w = exp-pool, uhat = unit token)
  -- (2) BETWEEN-step trajectory directional coherence --
  coh_prev = cos(d_j, d_{j-1})                  turn from the previous step
  coh_run  = cos(d_j, normalize(sum_{t<j} d_t)) consistency with the reasoning so far
  -- (3) WITHIN-step directional higher-order shape (multimodality) --
  dir_lam2 = lambda_2 / sum(lambda)  of  sum_t w_t uhat uhat^T   second-direction strength = split
  dir_D    = exp(-sum p_i log p_i), p_i = lambda_i/sum            directional effective rank
  res_jl   = ||sum_t w_t uhat_t||  (baseline recomputed in JL space)

(anchor_q = cos(d_j, question direction) is the most promising cell but needs qvec ->
 re-extract with --store_step_vectors. Not computable from respcloud alone.)

VERDICT per signal: single AUROC (pooled + length-bucket); residualized-on-resultant AUROC
(cross-fit GBR on correct steps); GroupKFold logistic INCREMENT over [resultant + position]
with chain-paired bootstrap CI ('*' = CI clears 0 = a NEW axis). Else it collapses into
resultant (honest negative).

Needs _cloud.npz: respcloud (clouds_stored) + cloud_store_layers + step_token_ranges +
gold_error_step (+ stepcloud/cloud_feature_names/layers_used for the stored-resultant baseline).
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


def exp_w(n):
    if n <= 1:
        return np.ones(max(n, 1))
    e = np.exp(np.arange(n) / (n - 1)); return e / e.sum()


def step_dir_stats(tok):
    """tok: (n,d) JL token vectors -> (d_unit, res_jl, dir_lam2, dir_D) or None."""
    nrm = np.linalg.norm(tok, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return None
    u = tok[ok] / nrm[ok, None]; w = exp_w(u.shape[0])
    pooled = (w[:, None] * u).sum(0); res = float(np.linalg.norm(pooled))
    if res < 1e-12:
        return None
    d = pooled / res
    B = np.sqrt(w)[:, None] * u                          # (n,d); eig of B B^T = nonzero eig of sum w uu^T
    lam = np.linalg.eigvalsh(B @ B.T); lam = np.clip(lam[::-1], 0, None); s = lam.sum()
    if s <= 1e-12:
        return None
    p = lam / s; pp = p[p > 1e-12]
    dir_D = float(np.exp(-(pp * np.log(pp)).sum()))
    lam2 = float(lam[1] / s) if len(lam) > 1 else 0.0
    return d, res, lam2, dir_D


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz", help="the _cloud.npz with populated respcloud")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min_tok", type=int, default=4)
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no stored respcloud; use the _cloud.npz with --store_clouds")
    csl = [int(x) for x in z["cloud_store_layers"]]
    if args.layer not in csl:
        raise SystemExit(f"layer {args.layer} not in cloud_store_layers {csl}")
    cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None     # per-token entropy -> per-step mean (control)
    # stored full-dim resultant baseline if available, else fall back to JL res
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else None
    SC = z["stepcloud"] if "stepcloud" in z.files else None
    use_stored_res = (SC is not None and lyu is not None and args.layer in lyu and "resultant" in cnames)
    lic = lyu.index(args.layer) if use_stored_res else None

    feats = {k: [] for k in ["coh_prev", "coh_run", "dir_lam2", "dir_D", "res"]}
    rjl_all = []; UDcol = []
    Y, NT, POS, G = [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        k = int(ges[i]); correct = (k < 0); a0 = int(rng[0, 0]); T = rng.shape[0]
        sc = np.asarray(SC[i], float) if use_stored_res else None
        ud = np.asarray(UD[i], float) if UD is not None else None

        dlist = [None] * T; lam2 = np.full(T, np.nan); dirD = np.full(T, np.nan); rjl = np.full(T, np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < args.min_tok:
                continue
            st = step_dir_stats(rcl[lo:hi])
            if st is not None:
                dlist[j], rjl[j], lam2[j], dirD[j] = st

        run = np.zeros(rcl.shape[1]); have_run = False
        for j in range(T):
            dj = dlist[j]
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            cp = cr = np.nan
            if dj is not None:
                if j >= 1 and dlist[j - 1] is not None:
                    cp = float(dj @ dlist[j - 1])
                if have_run and np.linalg.norm(run) > 1e-9:
                    cr = float(dj @ (run / np.linalg.norm(run)))
                run = run + dj; have_run = True
            res_v = (sc[j, lic, cnames.index("resultant")] if use_stored_res else rjl[j])
            lo = max(0, int(rng[j, 0]) - a0); hi = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            uv = np.nanmean(ud[lo:hi]) if (ud is not None and hi > lo) else np.nan
            feats["coh_prev"].append(cp); feats["coh_run"].append(cr)
            feats["dir_lam2"].append(lam2[j]); feats["dir_D"].append(dirD[j]); feats["res"].append(res_v)
            rjl_all.append(rjl[j]); UDcol.append(uv)
            Y.append(y); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); POS.append(j / max(1, T - 1)); G.append(i)

    for kk in feats:
        c = np.asarray(feats[kk], float)
        c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        feats[kk] = c
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); POS = np.asarray(POS, float); G = np.asarray(G, int)
    rjl_all = np.asarray(rjl_all, float); correct_step = Y == 0; res = feats["res"]
    UDcol = np.asarray(UDcol, float)
    if np.isfinite(UDcol).any():
        UDcol[~np.isfinite(UDcol)] = np.nanmean(UDcol[np.isfinite(UDcol)])
    else:
        UDcol[:] = 0.0
    has_ud = UD is not None and np.std(UDcol) > 1e-9

    print(f"file: {args.npz} | layer {args.layer} (store {csl}) | labeled steps {len(Y)} | "
          f"first-error {int(Y.sum())} | error-chains {len(np.unique(G[Y==1]))}")
    rtag = "stored" if use_stored_res else "JL"
    cc = np.corrcoef(res, rjl_all)[0, 1] if len(res) > 2 else np.nan
    print(f"baseline resultant({rtag}) AUROC {bdir(auroc(res,Y)):.3f}  bucket {bucket(res,Y,NT):.3f}  "
          f"| position AUROC {bdir(auroc(POS,Y)):.3f}  | corr(res,res_jl) {cc:+.2f}")

    # baselines: [res,pos] and the STRICT [res,pos,log n_tok] -- dir_D/dir_lam2 are rank-bounded
    # by n_tok, so an increment that survives the length control is real directional structure;
    # one that vanishes was just length resultant didn't fully absorb.
    # decisive baselines: length-controlled, and length+uncertainty-controlled (kill the obvious
    # remaining confounds for dir_lam2 -- rank=length, and hesitation=high-entropy spread).
    logNT = np.log(np.maximum(NT, 1.0))
    base_len = np.c_[res, POS, logNT]; sa_len = oof_logit(base_len, Y, G, args.folds)
    base_all = np.c_[res, POS, logNT, UDcol] if has_ud else base_len
    sa_all = oof_logit(base_all, Y, G, args.folds)
    udtag = "+res,pos,len,U_D" if has_ud else "+res,pos,len (no U_D)"
    print(f"\n{'signal':9s} {'AUROC':>6s} {'bucket':>6s} {'resid⊥res':>9s} | "
          f"{'Δ[+res,pos,len]':>15s} {'CI':>16s} | {('Δ['+udtag+']'):>20s} {'CI':>16s}")
    for nm in ["coh_prev", "coh_run", "dir_lam2", "dir_D"]:
        sig = feats[nm]
        a = bdir(auroc(sig, Y)); bk = bucket(sig, Y, NT)
        rsd = residualize_on(sig, res, correct_step, G, args.folds); ar = bdir(auroc(rsd, Y))
        sb1 = oof_logit(np.c_[res, POS, logNT, sig], Y, G, args.folds)
        d1, lo1, hi1 = boot_delta(sa_len, sb1, Y, G, n=args.boot)
        sb2 = oof_logit((np.c_[res, POS, logNT, UDcol, sig] if has_ud else np.c_[res, POS, logNT, sig]),
                        Y, G, args.folds)
        d2, lo2, hi2 = boot_delta(sa_all, sb2, Y, G, n=args.boot)
        f1 = "*" if lo1 > 0 else " "; f2 = "*" if lo2 > 0 else " "
        print(f"{nm:9s} {a:6.3f} {bk:6.3f} {ar:9.3f} | {d1:+15.3f} {f'[{lo1:+.3f},{hi1:+.3f}]':>16s}{f1} | "
              f"{d2:+20.3f} {f'[{lo2:+.3f},{hi2:+.3f}]':>16s}{f2}")

    print("\nread: Δ[+res,pos,len] controls resultant+position+length; the rightmost ALSO controls "
          "per-step U_D (uncertainty). '*' = chain-paired bootstrap CI clears 0. A signal is a REAL "
          "new directional axis only if it survives the RIGHTMOST (all known confounds). dir_lam2 = "
          "second-direction strength (genuine within-step multimodality, not just rank); dir_D is "
          "rank-bounded so it leaks length. anchor_q (cos to question) still needs qvec re-extract.")


if __name__ == "__main__":
    main()
