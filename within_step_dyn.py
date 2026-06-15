"""LAST geometric probe -- a VERDICT, not a lifeline. (v2: endogeneity bug fixed.)

Does geometric DYNAMICS (how the token cloud moves -- which perplexity, a scalar
sequence, structurally cannot express) add ANY independent increment over the fixed
baseline [static resultant + U_D + U_C + n_tok + pos + density]?

CONVERGENCE is measured against an EXOGENOUS reference: the pooled direction of the
PRECEDING context (response tokens before this step). align(t) = u_t . ref_hat, where
ref_hat does NOT include any of the step's own tokens -> no endogeneity (v1 used the
step's own mean as reference, which baked "later tokens are closer to the mean they
helped define" into the metric). "Convergence" = do this step's tokens align INCREASINGLY
to the already-established context direction (the anchoring intuition).

Ladder (same baseline, raw + within-chain-z pooled AUROC, chain-paired bootstrap):
  L1 step velocity   |Δresultant_j|                       (predicted ns)
  L2a within-step a  conv = align(late) - align(early), slope   (RAW)
  L2b within-step b  same on alignment RESIDUALIZED on within-step rel-pos (strips the
                     "conclusion tokens are concentrated" semantic curve; orthogonal to
                     the endogeneity fix -- both controls needed)

Steps with <2 preceding tokens or <8 own tokens are skipped (no clean reference / the
early/late split of a tiny step is pure noise). Needs a cloud npz (--store_clouds
--cloud_eff_rank): respcloud + stepcloud(resultant) + tok_U_D/U_C.

VERDICT: L2b within-z increment SIG>0 -> geometry's unique value is DYNAMICS. ns ->
geometry's value is the static signal it already has over perplexity (NOT redundant);
narrative = independent modest static signal + audit. Stop and write either way.
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


def step_conv(rcl, lo, hi, min_pre=2, min_tok=8):
    """rcl=(R,k) projected response cloud at one layer. Align step tokens to the
    EXOGENOUS preceding-context direction. Returns (align(n,), relpos(n,)) or None."""
    if lo < min_pre or hi - lo < min_tok:
        return None
    prec = rcl[:lo]
    pu = prec / np.maximum(np.linalg.norm(prec, axis=1, keepdims=True), 1e-9)
    ref = pu.mean(0); rh = ref / (np.linalg.norm(ref) + 1e-9)
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
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no respcloud; re-extract with --store_clouds --cloud_eff_rank")
    csl = [int(x) for x in z["cloud_store_layers"]]
    if args.layer not in csl:
        raise SystemExit(f"layer {args.layer} not in cloud_store_layers {csl}")
    cli = csl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    RC, SC, SR = z["respcloud"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int); ST = z["steps_text"]
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None

    # pass 1: E[align | within-step rel-pos] on correct steps (exogenous-ref alignment)
    A_pos, A_val = [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        k = int(ges[i]); correct = (k < 0); a0 = int(rng[0, 0]); T = rng.shape[0]
        for j in range(T):
            if not (correct or j < k):
                continue
            r = step_conv(rcl, max(0, int(rng[j, 0]) - a0), min(rcl.shape[0], int(rng[j, 1]) - a0 + 1),
                          min_tok=args.min_tok)
            if r is not None:
                A_pos.append(r[1]); A_val.append(r[0])
    A_pos = np.concatenate(A_pos); A_val = np.concatenate(A_val)
    centers = np.linspace(0.05, 0.95, 10)
    binmean = np.array([A_val[np.abs(A_pos - c) < 0.05].mean() if (np.abs(A_pos - c) < 0.05).any()
                        else np.nan for c in centers])
    def epos(p):
        return np.interp(p, centers, np.nan_to_num(binmean, nan=np.nanmean(binmean)))

    # pass 2: per-step features
    rows = {k: [] for k in ["res", "conv_a", "slope_a", "conv_b", "slope_b",
                            "U_D", "U_C", "n_tok", "pos", "dens"]}
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
            rr = step_conv(rcl, lo, hi, min_tok=args.min_tok)
            if rr is None:
                continue
            a, p = rr; half = len(a) // 2
            r = a - epos(p)
            rows["res"].append(res_seq[j])
            rows["conv_a"].append(a[half:].mean() - a[:half].mean())
            rows["slope_a"].append(np.polyfit(p, a, 1)[0])
            rows["conv_b"].append(r[half:].mean() - r[:half].mean())
            rows["slope_b"].append(np.polyfit(p, r, 1)[0])
            rows["n_tok"].append(hi - lo); rows["pos"].append(j / max(1, T - 1))
            rows["dens"].append(density(txt[j]) if j < len(txt) else np.nan)
            lo2 = lo; hi2 = min((len(ud) if ud is not None else 0), hi)
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

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} (>= {args.min_tok} tok, >=2 pre) "
          f"| first-error {int(Y.sum())}")
    print(f"standalone AUROC: conv_a {max(auroc(rows['conv_a'],Y),1-auroc(rows['conv_a'],Y)):.3f} | "
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
           "L2a within-step RAW": np.c_[rows["conv_a"], rows["slope_a"]],
           "L2b within-step POS-RESID": np.c_[rows["conv_b"], rows["slope_b"]]}

    def boot(sa, sb, tf):
        rng = np.random.default_rng(0); ch = np.unique(G); d = []
        ta, tb = tf(sa), tf(sb)
        for _ in range(args.boot):
            cb = rng.choice(ch, len(ch), replace=True)
            m = np.concatenate([np.where(G == c)[0] for c in cb])
            d.append(auroc(ta[m], Y[m]) - auroc(tb[m], Y[m]))
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        return np.nanmean(d), lo, hi

    print(f"\nbaseline [res+U_D+U_C+n_tok+pos+dens]: raw {auroc(s_base,Y):.3f} | "
          f"within-z {auroc(wcz(s_base,G),Y):.3f}")
    print(f"\n{'ladder':26s} {'mode':9s} {'+full':>7s} {'increment (CI)':>26s}")
    for nm, ax in add.items():
        s_a = oof(np.c_[base, ax])
        for mode, tf in [("raw", lambda s: s), ("within-z", lambda s: wcz(s, G))]:
            m, lo, hi = boot(s_a, s_base, tf)
            print(f"{nm:26s} {mode:9s} {auroc(tf(s_a),Y):7.3f}   +{m:.3f} [{lo:+.3f},{hi:+.3f}] "
                  f"{'SIG' if lo > 0 else 'ns'}")

    print("\nVERDICT: L2b within-z increment SIG>0 => geometry's unique value is DYNAMICS "
          "(exogenous-reference convergence, perplexity-blind). ns => static signal is all "
          "geometry offers -> independent modest signal + audit. Stop and write either way.")


if __name__ == "__main__":
    main()
