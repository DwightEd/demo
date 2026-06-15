"""LAST geometric probe -- VERDICT. (v3: + epos cross-fit, within-chain-z FEATURES,
fixed-n control, power print. v2 fixed the reference-direction endogeneity.)

Does geometric DYNAMICS (token-cloud movement, perplexity-blind) add ANY independent
increment over the fixed baseline [static resultant + U_D + U_C + n_tok + pos + density]?

Clean controls (all leaks the reviewers raised, fixed):
  - reference = EXOGENOUS preceding-context direction (v2): no "tokens converge to the
    mean they helped define".
  - epos (within-step position-residualization for the semantic 'conclusion tokens are
    concentrated' curve) is CROSS-FIT: E[align|pos] fit on TRAIN-fold correct steps only,
    applied to test steps -> no "correct steps deflated by a curve they helped fit".
  - within-chain mode z-scores EVERY feature (incl. static resultant) WITHIN chain BEFORE
    the logistic, so baseline + increment live in pure within-chain space (resultant's
    between-chain difficulty is removed from the baseline, not just the output).
  - fixed-n version: align(last k) - align(first k) with k fixed -> removes the n-dependent
    estimation bias of the early/late split (length control beyond linear n_tok).

VERDICT: any L2 within-z increment SIG>0 -> geometry's unique value is dynamics. ns ->
static signal is all geometry offers -> independent modest signal + audit. Print the
first-error count after the >=8-tok/>=2-pre filter: if it collapsed, ns may be low power.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
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


def density(t):
    t = str(t); return 1.0 - sum(c.isalpha() for c in t) / len(t) if t else np.nan


def step_align(rcl, lo, hi, min_pre=2, min_tok=8):
    if lo < min_pre or hi - lo < min_tok:
        return None
    prec = rcl[:lo]
    pu = prec / np.maximum(np.linalg.norm(prec, axis=1, keepdims=True), 1e-9)
    rh = pu.mean(0); rh = rh / (np.linalg.norm(rh) + 1e-9)
    cl = rcl[lo:hi]
    u = cl / np.maximum(np.linalg.norm(cl, axis=1, keepdims=True), 1e-9)
    a = u @ rh; n = len(a)
    return a, np.arange(n) / (n - 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--min_tok", type=int, default=8)
    ap.add_argument("--kfix", type=int, default=3, help="fixed #tokens per half for the length-control version")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no respcloud; re-extract with --store_clouds --cloud_eff_rank")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    RC, SC, SR = z["respcloud"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int); ST = z["steps_text"]
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    K = args.kfix

    a_steps, p_steps = [], []                  # per-step alignment arrays (for cross-fit epos)
    rows = {k: [] for k in ["res", "conv_a", "slope_a", "conv_fix", "U_D", "U_C", "n_tok", "pos", "dens"]}
    velo = {}; Y, G, JI = [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]; txt = list(ST[i]) if i < len(ST) else []
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        res_seq = sc[:, li, cnames.index("resultant")] if "resultant" in cnames else np.full(T, np.nan)
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            rr = step_align(rcl, lo, hi, min_tok=args.min_tok)
            if rr is None:
                continue
            a, p = rr; half = len(a) // 2
            a_steps.append(a); p_steps.append(p)
            rows["res"].append(res_seq[j])
            rows["conv_a"].append(a[half:].mean() - a[:half].mean())
            rows["slope_a"].append(np.polyfit(p, a, 1)[0])
            rows["conv_fix"].append(a[-K:].mean() - a[:K].mean())     # fixed-n length control
            rows["n_tok"].append(hi - lo); rows["pos"].append(j / max(1, T - 1))
            rows["dens"].append(density(txt[j]) if j < len(txt) else np.nan)
            hi2 = min((len(ud) if ud is not None else 0), hi)
            rows["U_D"].append(np.nanmean(ud[lo:hi2]) if (ud is not None and hi2 > lo) else np.nan)
            rows["U_C"].append(np.nanmean(uc[lo:hi2]) if (uc is not None and hi2 > lo) else np.nan)
            dv = res_seq[j] - res_seq[j-1] if j >= 1 and np.isfinite(res_seq[j-1]) else 0.0
            velo.setdefault(i, {})[j] = abs(dv)
            Y.append(y); G.append(i); JI.append(j)
    for k in rows:
        c = np.asarray(rows[k], float); c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        rows[k] = c
    Y = np.asarray(Y, int); G = np.asarray(G, int); JI = np.asarray(JI, int)
    vel = np.array([velo.get(g, {}).get(j, 0.0) for g, j in zip(G, JI)])
    half_idx = [len(a) // 2 for a in a_steps]

    nerr_chain = len(np.unique([g for g, y in zip(G, Y) if y == 1]))
    print(f"file: {args.npz} | layer {args.layer} | kept steps {len(Y)} "
          f"(>= {args.min_tok} tok, >=2 pre) | first-error STEPS {int(Y.sum())} "
          f"(error-chains kept {nerr_chain}) -- POWER CHECK")

    gkf = GroupKFold(args.folds)
    def wcz_feat(X):                            # within-chain z-score every column
        Z = X.copy().astype(float)
        for c in np.unique(G):
            m = G == c
            Z[m] = (Z[m] - Z[m].mean(0)) / (Z[m].std(0) + 1e-9)
        return Z

    def oof(make_feats, within):
        """make_feats(tr, te) -> (Xtr, Xte) so epos can be train-fit per fold."""
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(Y, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            Xtr, Xte = make_feats(tr, te)
            if within:
                full = np.full((len(Y), Xtr.shape[1]), np.nan); full[tr] = Xtr; full[te] = Xte
                full = wcz_feat(full); Xtr, Xte = full[tr], full[te]
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(Xtr, Y[tr]); s[te] = p.predict_proba(Xte)[:, 1]
        return s

    base_cols = np.c_[rows["res"], rows["U_D"], rows["U_C"], rows["n_tok"], rows["pos"], rows["dens"]]
    def feats_base(tr, te): return base_cols[tr], base_cols[te]

    def conv_b_crossfit(tr):
        """epos from TRAIN correct steps; return conv_b for ALL steps using it."""
        centers = np.linspace(0.05, 0.95, 10)
        av, ap = [], []
        for idx in tr:
            if Y[idx] == 0:
                av.append(a_steps[idx]); ap.append(p_steps[idx])
        av = np.concatenate(av); ap = np.concatenate(ap)
        bm = np.array([av[np.abs(ap - c) < 0.05].mean() if (np.abs(ap - c) < 0.05).any() else np.nan
                       for c in centers])
        ef = lambda pp: np.interp(pp, centers, np.nan_to_num(bm, nan=np.nanmean(bm)))
        cb = np.zeros(len(Y))
        for idx in range(len(Y)):
            a, p, h = a_steps[idx], p_steps[idx], half_idx[idx]
            r = a - ef(p); cb[idx] = r[h:].mean() - r[:h].mean()
        return cb

    add = {
        "L1 velocity":        lambda tr, te: (np.c_[base_cols[tr], vel[tr]], np.c_[base_cols[te], vel[te]]),
        "L2a conv_a":         lambda tr, te: (np.c_[base_cols[tr], rows["conv_a"][tr], rows["slope_a"][tr]],
                                              np.c_[base_cols[te], rows["conv_a"][te], rows["slope_a"][te]]),
        "L2b conv_b (epos cf)": None,          # filled below (needs tr)
        "L2c conv_fix (fixed-n)": lambda tr, te: (np.c_[base_cols[tr], rows["conv_fix"][tr]],
                                                  np.c_[base_cols[te], rows["conv_fix"][te]]),
    }

    def boot(sa, sb):
        rng = np.random.default_rng(0); ch = np.unique(G); d = []
        for _ in range(args.boot):
            cb = rng.choice(ch, len(ch), replace=True)
            m = np.concatenate([np.where(G == c)[0] for c in cb])
            d.append(auroc(sa[m], Y[m]) - auroc(sb[m], Y[m]))
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return np.nanmean(d), lo, hi

    print(f"\nstandalone: conv_a {max(auroc(rows['conv_a'],Y),1-auroc(rows['conv_a'],Y)):.3f} | "
          f"conv_fix {max(auroc(rows['conv_fix'],Y),1-auroc(rows['conv_fix'],Y)):.3f} | "
          f"velocity {max(auroc(vel,Y),1-auroc(vel,Y)):.3f}")
    for within in (False, True):
        mode = "within-z feats" if within else "raw feats"
        s_base = oof(feats_base, within)
        print(f"\n[{mode}] baseline AUROC {auroc(s_base, Y):.3f}")
        for nm in ["L1 velocity", "L2a conv_a", "L2b conv_b (epos cf)", "L2c conv_fix (fixed-n)"]:
            if nm.startswith("L2b"):
                def mf(tr, te):
                    cb = conv_b_crossfit(tr)
                    return np.c_[base_cols[tr], cb[tr]], np.c_[base_cols[te], cb[te]]
                s_a = oof(mf, within)
            else:
                s_a = oof(add[nm], within)
            m, lo, hi = boot(s_a, s_base)
            print(f"  {nm:24s} +full {auroc(s_a,Y):.3f}  inc +{m:.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"{'SIG' if lo > 0 else 'ns'}")

    print("\nVERDICT: any L2 within-z-feats increment SIG>0 => geometry's unique value is "
          "DYNAMICS. all ns => static signal is all geometry offers (independent modest "
          "signal + audit). Check the POWER line first -- if first-error STEPS collapsed, ns "
          "is low power, lower --min_tok.")


if __name__ == "__main__":
    main()
