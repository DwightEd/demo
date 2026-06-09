"""Step 38: representation channel (D/PR/AE) vs uncertainty channel (UD/UC), unified.

WHAT THIS FILE COVERS (per-step SCALAR signal family)
-----------------------------------------------------
Signals (5; UE/epistemic gradient-norm is NOT computed anywhere in the project):
  representation : D  = sv_D            spectral effective rank of the step token cloud
                   PR = sv_pr_<mode>    participation ratio of the step vector
                   AE = sv_ae_<mode>    activation entropy of the step vector
  uncertainty    : UD = sv_out_entropy  distributional aleatoric (next-token entropy)
                   UC = sv_out_committal committal aleatoric p(1-p) of the boundary token

Aggregation:
  per-step  : layer band (deep/mid/all), averaged over layers -> one scalar per step
  per-chain : profile3() = [mean (static level), slope, r2 (dynamic curve-fit shape)]

Evaluation protocols (every cross-channel question is asked under each):
  step  level : pooled (all (chain,step))  AND  within-chain (per-chain corr, averaged)
  chain level : pooled (cross-problem)     AND  within-problem (per-problem mean removed)
  detection   : within-problem paired AUROC, GroupKFold(problem) -> representation /
                uncertainty / fusion, and static-only vs dynamic-only per channel

NOT in this file (other signal families): vector-level displacement / per-dim signed
deviation (27 Mahalanobis, 36 per-dim magnitude) and set-level effective dimension across
a problem's chains (28). Those are scalar-from-vector / cross-chain, a different reduction.

Label: answer-based. Writes results_uncertainty/channels_<mode>_<band>.json.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Which npz key each signal comes from ({mode} filled at load time).
REP_SPEC = [("D", "sv_D"), ("PR", "sv_pr_{mode}"), ("AE", "sv_ae_{mode}")]
UNC_SPEC = [("UD", "sv_out_entropy"), ("UC", "sv_out_committal")]


# ===========================================================================
# Aggregation primitives  (raw arrays -> per-step scalar -> per-chain summary)
# ===========================================================================
def band_cols(L: int, band: str) -> np.ndarray:
    """Layer indices for a depth band of an L-layer stack."""
    if band == "deep": return np.arange(int(L * 0.6), L)
    if band == "mid":  return np.arange(int(L * 0.3), int(L * 0.7))
    return np.arange(L)


def steptrace(M, cols) -> np.ndarray | None:
    """(T, L) per-(step,layer) matrix -> (T,) per-step scalar, averaged over the band."""
    M = np.asarray(M, dtype=np.float64)
    if M.ndim != 2:
        return None
    cc = cols[cols < M.shape[1]]
    with np.errstate(invalid="ignore"):
        return np.nanmean(M[:, cc], axis=1)


def profile3(e) -> np.ndarray | None:
    """Per-chain summary of a per-step series over normalized position [0,1]:
    [mean (static level), slope, r2 (dynamic curve-fit shape)]. None if T<4."""
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    T = len(e)
    if T < 4:
        return None
    pos = np.linspace(0, 1, T); A = np.vstack([pos, np.ones(T)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef
    ssr = ((e - pred) ** 2).sum(); sst = ((e - e.mean()) ** 2).sum() + 1e-12
    return np.array([e.mean(), coef[0], 1 - ssr / sst])


# ===========================================================================
# Correlation primitives  (pooled vs within-{chain,problem})
# ===========================================================================
def spearman_safe(a, b) -> float:
    """Spearman rho with NaN/constant guards; NaN if not estimable."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or np.std(a[m]) < 1e-12 or np.std(b[m]) < 1e-12:
        return np.nan
    return float(spearmanr(a[m], b[m]).correlation)


def within_center(v, pp) -> np.ndarray:
    """Subtract each problem's mean -> the chain-to-chain (difficulty-removed) part."""
    o = np.asarray(v, float).copy()
    for p in np.unique(pp):
        m = pp == p
        o[m] -= np.nanmean(o[m])
    return o


# ===========================================================================
# Detection primitives  (within-problem paired AUROC of a cross-validated probe)
# ===========================================================================
def within_pair_auroc(groups, f, y) -> float:
    """P(score_error > score_correct) over within-problem error-correct pairs; >=0.5 by sym."""
    conc = 0.0; npair = 0
    for idx in groups:
        inc = [f[i] for i in idx if y[i] == 1 and np.isfinite(f[i])]
        cor = [f[i] for i in idx if y[i] == 0 and np.isfinite(f[i])]
        if not inc or not cor:
            continue
        for u in inc:
            for v in cor:
                conc += 1.0 if u > v else (0.5 if u == v else 0.0)
        npair += len(inc) * len(cor)
    a = conc / npair if npair else np.nan
    return max(a, 1 - a) if np.isfinite(a) else np.nan


def group_kfold(g, k, seed):
    """Folds that keep all chains of a problem together (no problem leakage)."""
    u = np.unique(g); r = np.random.default_rng(seed); r.shuffle(u)
    fo = {int(x): i % k for i, x in enumerate(u)}
    f = np.array([fo[int(x)] for x in g])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def within_auroc_cv(F, y, pp, groups, kfold, n_seeds) -> float:
    """GroupKFold logistic probe -> out-of-fold scores -> within-problem paired AUROC,
    averaged over seeds. F: (M, n_features)."""
    vals = []
    for s in range(n_seeds):
        oof = np.full(len(y), np.nan)
        for tr, te in group_kfold(pp, kfold, s):
            if len(np.unique(y[tr])) < 2:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=1000, class_weight="balanced"))
            clf.fit(F[tr], y[tr]); oof[te] = clf.predict_proba(F[te])[:, 1]
        vals.append(within_pair_auroc(groups, oof, y))
    return round(float(np.nanmean(vals)), 4)


# ===========================================================================
# Pipeline stages
# ===========================================================================
def load_signals(d, mode):
    """Return (REP, UNC) dicts {name: object-array-of-per-chain-arrays} for present keys."""
    REP = {n: d[k.format(mode=mode)] for n, k in REP_SPEC if k.format(mode=mode) in d.files}
    UNC = {n: d[k.format(mode=mode)] for n, k in UNC_SPEC if k.format(mode=mode) in d.files}
    if not REP or not UNC:
        raise SystemExit(f"need representation + uncertainty per-step arrays. files={sorted(d.files)}")
    return REP, UNC


def build_step_series(REP, UNC, band, keep):
    """Per chain, reduce each signal to a (T,) per-step series. REP needs the layer band;
    UNC is already per-step. Returns {name: [series or None per chain]} and the band cols."""
    L = next((np.asarray(a[0]).shape[1] for a in REP.values()
              if np.asarray(a[0]).ndim == 2), None)
    cols = band_cols(L, band) if L else None
    ser = {k: [None] * len(keep) for k in list(REP) + list(UNC)}
    for i in range(len(keep)):
        if not keep[i]:
            continue
        for k, arr in REP.items():
            ser[k][i] = steptrace(arr[i], cols)
        for k, arr in UNC.items():
            v = np.asarray(arr[i], dtype=np.float64)
            ser[k][i] = v if v.ndim == 1 else None
    return ser, cols


def step_correlations(ser, rep_keys, unc_keys, keep):
    """STEP level: pooled (all (chain,step)) and within-chain (per-chain corr, averaged)."""
    pooled, within = {}, {}
    for rk in rep_keys:
        for uk in unc_keys:
            A, B, wc = [], [], []
            for i in range(len(keep)):
                if not keep[i]:
                    continue
                a, b = ser[rk][i], ser[uk][i]
                if a is None or b is None:
                    continue
                T = min(len(a), len(b))
                if T < 4:
                    continue
                a, b = a[:T], b[:T]
                A.append(a); B.append(b)
                c = spearman_safe(a, b)
                if np.isfinite(c):
                    wc.append(c)
            key = f"{rk}~{uk}"
            pooled[key] = round(spearman_safe(np.concatenate(A), np.concatenate(B)), 3) if A else None
            within[key] = round(float(np.mean(wc)), 3) if wc else None
    return pooled, within


def chain_features(ser, keys, keep, y, pid):
    """Per chain, profile3() every signal; keep chains where ALL signals are estimable.
    Returns feat {name: (M,3)}, yy (1=error), pp (problem id)."""
    feat = {k: [] for k in keys}; yy, pp = [], []
    for i in range(len(keep)):
        if not keep[i]:
            continue
        ps = {k: (profile3(ser[k][i]) if ser[k][i] is not None else None) for k in keys}
        if any(v is None for v in ps.values()):
            continue
        for k in keys:
            feat[k].append(ps[k])
        yy.append(int(y[i])); pp.append(int(pid[i]))
    return {k: np.array(v) for k, v in feat.items()}, np.array(yy), np.array(pp)


def contrastive_groups(pp, yy):
    """Index groups per problem that contain BOTH an error and a correct chain."""
    prob = {}
    for j, p in enumerate(pp):
        prob.setdefault(int(p), []).append(j)
    return [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]


def chain_correlations(feat, rep_keys, unc_keys, pp):
    """CHAIN level on the mean (static) summary: pooled (cross-problem) and within-problem."""
    pooled, within = {}, {}
    for rk in rep_keys:
        for uk in unc_keys:
            a, b = feat[rk][:, 0], feat[uk][:, 0]
            key = f"{rk}~{uk}"
            pooled[key] = round(spearman_safe(a, b), 3)
            within[key] = round(spearman_safe(within_center(a, pp), within_center(b, pp)), 3)
    return pooled, within


def channel_detection(feat, rep_keys, unc_keys, yy, pp, groups, kfold, n_seeds):
    """Within-problem AUROC of representation-only / uncertainty-only / fusion probes."""
    rep_F = np.column_stack([feat[k] for k in rep_keys])
    unc_F = np.column_stack([feat[k] for k in unc_keys])
    kw = dict(kfold=kfold, n_seeds=n_seeds)
    return {
        "representation_only": within_auroc_cv(rep_F, yy, pp, groups, **kw),
        "uncertainty_only":    within_auroc_cv(unc_F, yy, pp, groups, **kw),
        "fusion":              within_auroc_cv(np.column_stack([rep_F, unc_F]), yy, pp, groups, **kw),
    }


def static_vs_dynamic(feat, rep_keys, unc_keys, yy, pp, groups, kfold, n_seeds):
    """Per channel: static = mean (col 0) only; dynamic = slope+r2 (cols 1:) only."""
    kw = dict(kfold=kfold, n_seeds=n_seeds)
    pick = lambda keys, sl: np.column_stack([feat[k][:, sl] for k in keys])
    return {
        "rep_static":  within_auroc_cv(pick(rep_keys, slice(0, 1)), yy, pp, groups, **kw),
        "rep_dynamic": within_auroc_cv(pick(rep_keys, slice(1, 3)), yy, pp, groups, **kw),
        "unc_static":  within_auroc_cv(pick(unc_keys, slice(0, 1)), yy, pp, groups, **kw),
        "unc_dynamic": within_auroc_cv(pick(unc_keys, slice(1, 3)), yy, pp, groups, **kw),
    }


# ===========================================================================
# Orchestration
# ===========================================================================
def run(d, mode, band, kfold, n_seeds, format_ok):
    REP, UNC = load_signals(d, mode)
    rep_keys, unc_keys = list(REP), list(UNC)
    pid = d["problem_ids"].astype(int)
    y = (d["is_correct"].astype(int) == 0).astype(int)          # 1 = error (answer-based)
    fmt = d["format_ok"].astype(bool) if "format_ok" in d.files else np.ones(len(pid), bool)
    keep = fmt if format_ok else np.ones(len(pid), bool)

    ser, _ = build_step_series(REP, UNC, band, keep)
    step_pooled, step_within = step_correlations(ser, rep_keys, unc_keys, keep)
    feat, yy, pp = chain_features(ser, rep_keys + unc_keys, keep, y, pid)
    groups = contrastive_groups(pp, yy)
    chain_pooled, chain_within = chain_correlations(feat, rep_keys, unc_keys, pp)
    kw = dict(kfold=kfold, n_seeds=n_seeds)

    return {
        "meta": {"band": band, "mode": mode, "N": int(len(yy)), "n_incorrect": int(yy.sum()),
                 "contrastive_problems": len(groups), "rep_channel": rep_keys,
                 "unc_channel": unc_keys, "subset": "format_ok" if format_ok else "all",
                 "label": "answer-based"},
        "step_corr_pooled": step_pooled, "step_corr_within_chain": step_within,
        "chain_corr_pooled": chain_pooled, "chain_corr_within_problem": chain_within,
        "channel_within_auroc": channel_detection(feat, rep_keys, unc_keys, yy, pp, groups, **kw),
        "static_vs_dynamic_within_auroc": static_vs_dynamic(feat, rep_keys, unc_keys, yy, pp, groups, **kw),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--band", default="deep", choices=["deep", "mid", "all"])
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    out = run(d, args.mode, args.band, args.kfold, args.n_seeds, args.format_ok)

    o = args.out or f"results_uncertainty/channels_{args.mode}_{args.band}.json"
    os.makedirs(os.path.dirname(o) or ".", exist_ok=True)
    json.dump(out, open(o, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    m = out["meta"]
    print(f"N={m['N']} incorrect={m['n_incorrect']} contrastive={m['contrastive_problems']} "
          f"rep={m['rep_channel']} unc={m['unc_channel']} band={m['band']}")
    print("step within-chain corr:", json.dumps(out["step_corr_within_chain"], ensure_ascii=False))
    print("chain within-problem corr:", json.dumps(out["chain_corr_within_problem"], ensure_ascii=False))
    print("channel AUROC:", json.dumps(out["channel_within_auroc"], ensure_ascii=False))
    print("static vs dynamic:", json.dumps(out["static_vs_dynamic_within_auroc"], ensure_ascii=False))
    print(f"wrote {o}")


if __name__ == "__main__":
    main()
