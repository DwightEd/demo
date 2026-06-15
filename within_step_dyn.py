"""LAST geometric probe -- a VERDICT, not a lifeline.

Does geometric DYNAMICS (how the token cloud moves, which perplexity -- a scalar
sequence -- structurally cannot express) add ANY independent increment over the
fixed baseline [static resultant (within-chain z) + U_D + U_C + n_tok + pos + density]?

Ladder (all over the SAME baseline, within-chain z-scored pooled AUROC, chain-paired
bootstrap on the increment):
  L1 step velocity   : |Δresultant_j| + 2nd diff   (predicted ns -- event-study jump
                       is only ~0.5-0.7 within-chain std, so step-to-step is weak)
  L2 within-step a   : convergence = align(late tokens) - align(early tokens)
                       (RAW -- but confounded: late tokens are conclusions/numbers,
                        naturally more concentrated; a semantic, not dynamic, effect)
  L2 within-step b   : same convergence on token alignment RESIDUALIZED on within-step
                       relative position (E[align|pos] fit on correct-chain tokens),
                       which strips the "conclusions are concentrated" semantic curve.

VERDICT: if version b's increment bootstrap-lower-bound > 0 -> geometry's unique value
is DYNAMICS, narrative shifts. If ns -> geometry's value is the static +0.057 it already
has over perplexity (NOT redundant); narrative = independent modest static signal +
audit. BOTH publishable. Either way: this is the last geometric quantity -- accept the
result and write.

Needs a cloud npz (extract --store_clouds --cloud_eff_rank): respcloud (per-token
projected clouds) + stepcloud(resultant) + tok_U_D/U_C.
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


def wcz(s, G):
    z = np.full(len(s), np.nan)
    for c in np.unique(G):
        m = G == c; v = s[m]
        z[m] = (v - np.nanmean(v)) / (np.nanstd(v) + 1e-9)
    return z


def density(t):
    t = str(t); return 1.0 - sum(c.isalpha() for c in t) / len(t) if t else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no respcloud; re-extract with --store_clouds --cloud_eff_rank")
    csl = [int(x) for x in z["cloud_store_layers"]]
    if args.layer not in csl:
        raise SystemExit(f"layer {args.layer} not in cloud_store_layers {csl}")
    cli = csl.index(args.layer)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    RC, SC, SR = z["respcloud"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int); ST = z["steps_text"]
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    # ---- pass 1: collect per-token (alignment, within-step rel pos) on CORRECT steps ----
    def step_align(cl):
        """unit-normalize tokens; align each to the step's mean direction; return a (n,), p (n,)."""
        tn = np.linalg.norm(cl, axis=1)
        u = cl / np.maximum(tn[:, None], 1e-9)
        m = u.mean(0); mh = m / (np.linalg.norm(m) + 1e-9)
        a = u @ mh
        n = len(a); p = np.arange(n) / max(1, n - 1)
        return a, p

    A_pos, A_val = [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rc = np.asarray(RC[i], np.float32); rng = np.asarray(SR[i], int)
        k = int(ges[i]); correct = (k < 0); a0 = int(rng[0, 0]); T = rng.shape[0]
        for j in range(T):
            if not (correct or j < k):     # only y=0 (correct + pre-error) for the nuisance curve
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rc.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < 4:
                continue
            a, p = step_align(rc[lo:hi, cli, :])
            A_pos.append(p); A_val.append(a)
    A_pos = np.concatenate(A_pos); A_val = np.concatenate(A_val)
    # E[align | rel-pos] via 10 bins (the semantic concentration-vs-position curve)
    edges = np.linspace(0, 1, 11); centers = (edges[:-1] + edges[1:]) / 2
    binmean = np.array([A_val[(A_pos >= edges[b]) & (A_pos < edges[b] + 0.1 + (b == 9))].mean()
                        if ((A_pos >= edges[b]) & (A_pos < edges[b] + 0.1 + (b == 9))).any() else np.nan
                        for b in range(10)])

    def epos(p):                            # interpolate the nuisance curve
        return np.interp(p, centers, np.nan_to_num(binmean, nan=np.nanmean(binmean)))

    # ---- pass 2: per-step features ----
    rows = {k: [] for k in ["res", "conv_a", "slope_a", "conv_b", "slope_b",
                            "U_D", "U_C", "n_tok", "pos", "dens"]}
    velo = {}                               # chain -> per-step resultant for velocity
    Y, G, JI = [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rc = np.asarray(RC[i], np.float32); sc = np.asarray(SC[i], float)
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
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rc.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < 4:
                continue
            a, p = step_align(rc[lo:hi, cli, :])
            half = len(a) // 2
            conv_a = a[half:].mean() - a[:half].mean()
            slope_a = np.polyfit(p, a, 1)[0]
            r = a - epos(p)                  # version b: position-residualized alignment
            conv_b = r[half:].mean() - r[:half].mean()
            slope_b = np.polyfit(p, r, 1)[0]
            rows["res"].append(res_seq[j]); rows["conv_a"].append(conv_a); rows["slope_a"].append(slope_a)
            rows["conv_b"].append(conv_b); rows["slope_b"].append(slope_b)
            ntok = hi - lo
            rows["n_tok"].append(ntok); rows["pos"].append(j / max(1, T - 1))
            rows["dens"].append(density(txt[j]) if j < len(txt) else np.nan)
            lo2 = max(0, int(rng[j, 0]) - a0); hi2 = min((len(ud) if ud is not None else 0), int(rng[j, 1]) - a0 + 1)
            rows["U_D"].append(np.nanmean(ud[lo2:hi2]) if (ud is not None and hi2 > lo2) else np.nan)
            rows["U_C"].append(np.nanmean(uc[lo2:hi2]) if (uc is not None and hi2 > lo2) else np.nan)
            dv = res_seq[j] - res_seq[j-1] if j >= 1 and np.isfinite(res_seq[j-1]) else 0.0
            velo.setdefault(i, {})[j] = abs(dv)
            Y.append(y); G.append(i); JI.append(j)
    for k in rows:
        c = np.asarray(rows[k], float); c[~np.isfinite(c)] = np.nanmean(c[np.isfinite(c)]) if np.isfinite(c).any() else 0.0
        rows[k] = c
    Y = np.asarray(Y, int); G = np.asarray(G, int); JI = np.asarray(JI, int)
    vel = np.array([velo.get(g, {}).get(j, 0.0) for g, j in zip(G, JI)])

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\nstandalone AUROC (best dir): "
          f"conv_a {max(auroc(rows['conv_a'],Y),1-auroc(rows['conv_a'],Y)):.3f} | "
          f"conv_b {max(auroc(rows['conv_b'],Y),1-auroc(rows['conv_b'],Y)):.3f} | "
          f"velocity {max(auroc(vel,Y),1-auroc(vel,Y)):.3f}")

    gkf = GroupKFold(args.folds)
    def oof(X):
        s = np.full(len(Y), np.nan)
        for tr, te in gkf.split(X, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            p = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
            p.fit(X[tr], Y[tr]); s[te] = p.predict_proba(X[te])[:, 1]
        return s
    base = np.c_[rows["res"], rows["U_D"], rows["U_C"], rows["n_tok"], rows["pos"], rows["dens"]]
    s_base = oof(base)
    add = {"L1 step-velocity": vel[:, None],
           "L2a within-step RAW (conv_a,slope_a)": np.c_[rows["conv_a"], rows["slope_a"]],
           "L2b within-step POS-RESID (conv_b,slope_b)": np.c_[rows["conv_b"], rows["slope_b"]]}

    def boot(sa, sb, tf):
        rng = np.random.default_rng(0); ch = np.unique(G); d = []
        ta, tb = tf(sa), tf(sb)
        for _ in range(args.boot):
            cb = rng.choice(ch, len(ch), replace=True)
            m = np.concatenate([np.where(G == c)[0] for c in cb])
            d.append(auroc(ta[m], Y[m]) - auroc(tb[m], Y[m]))
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return np.nanmean(d), lo, hi

    print(f"\nbaseline [res(z)+U_D+U_C+n_tok+pos+dens]: raw {auroc(s_base,Y):.3f} | "
          f"within-z {auroc(wcz(s_base,G),Y):.3f}")
    print(f"\n{'ladder level':42s} {'mode':9s} {'+full':>7s} {'increment (CI)':>26s}")
    for nm, ax in add.items():
        s_a = oof(np.c_[base, ax])
        for mode, tf in [("raw", lambda s: s), ("within-z", lambda s: wcz(s, G))]:
            m, lo, hi = boot(s_a, s_base, tf)
            sig = "SIG" if lo > 0 else "ns"
            print(f"{nm:42s} {mode:9s} {auroc(tf(s_a),Y):7.3f}   +{m:.3f} [{lo:+.3f},{hi:+.3f}] {sig}")

    print("\nVERDICT: L2b (pos-residualized within-step) within-z increment SIG>0 => geometry's "
          "unique value is DYNAMICS (perplexity cannot express it) -> narrative shifts to dynamic. "
          "ns => geometry's value is the static +0.057 it already has; dynamics adds nothing -> "
          "narrative = independent modest static signal + audit. If L2a SIG but L2b ns => the "
          "'converge' signal was the semantic conclusion-token confound, not dynamics. STOP HERE "
          "either way -- accept the verdict and write.")


if __name__ == "__main__":
    main()
