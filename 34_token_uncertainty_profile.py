"""Step 34: faithful uncertainty-trace-profile (arXiv 2605.07776) on PER-TOKEN traces.

Uses sv_tok_entropy / sv_tok_committal from 10 --store_token_uncertainty (per-token, raw
model distribution, trace channel). Two aleatoric measures: entropy (distributional),
committal p(1-p). Profile features per chain/horizon: mu_early, mu_mid, mu_late, slope, r2.
Reports cross-problem (random 5-fold = the paper's setting) AND within-problem (GroupKFold
paired = our difficulty control) AUROC, vs the static mean baseline. Writes a JSON with all
results. Label: answer-based. Runs BOTH subsets (all chains / format_ok) in one call.
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def profile(e, n=0):
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    if n: e = e[:n]
    T = len(e)
    if T < 4: return None
    pos = np.linspace(0, 1, T); A = np.vstack([pos, np.ones(T)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None)
    pred = A @ coef; ssr = ((e-pred)**2).sum(); sst = ((e-e.mean())**2).sum()+1e-12
    q = max(1, T//4)
    mid = e[q:T-q].mean() if T-2*q > 0 else e.mean()
    return np.array([e[:q].mean(), mid, e[-q:].mean(), coef[0], 1-ssr/sst]), float(e.mean())


def within_pair_auroc(groups, f, y):
    conc = 0.0; npair = 0
    for idx in groups:
        inc = [f[i] for i in idx if y[i] == 1 and np.isfinite(f[i])]
        cor = [f[i] for i in idx if y[i] == 0 and np.isfinite(f[i])]
        if not inc or not cor: continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc)*len(cor)
    return conc/npair if npair else float("nan")


def gfolds(g, k, s):
    u = np.unique(g); r = np.random.default_rng(s); r.shuffle(u)
    fo = {int(x): i % k for i, x in enumerate(u)}; f = np.array([fo[int(x)] for x in g])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def rfolds(n, k, s):
    idx = np.arange(n); r = np.random.default_rng(s); r.shuffle(idx)
    f = np.empty(n, int); f[idx] = np.arange(n) % k
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def cv(F, y, folds):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(F[tr], y[tr]); oof[te] = clf.predict_proba(F[te])[:, 1]
    return oof


def auroc_both(F, y, pp, groups, kfold, n_seeds, seed):
    N = len(y); cw, cc = [], []
    for s in range(n_seeds):
        og = cv(F, y, gfolds(pp, kfold, seed+s)); orr = cv(F, y, rfolds(N, kfold, seed+s))
        a = within_pair_auroc(groups, og, y); cw.append(max(a, 1-a))
        mm = np.isfinite(orr); cr = roc_auc_score(y[mm], orr[mm]); cc.append(max(cr, 1-cr))
    return {"cross": round(float(np.mean(cc)), 4), "within": round(float(np.mean(cw)), 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--horizons", default="0,300")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results_uncertainty/profile_results.json")
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    if "sv_tok_entropy" not in dd.files:
        raise SystemExit("npz lacks sv_tok_entropy -- run 10 --store_token_uncertainty first.")
    ENT = dd["sv_tok_entropy"]; COM = dd["sv_tok_committal"]
    pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(ENT), bool)
    horizons = [int(x) for x in args.horizons.split(",")]

    out = {"meta": {"input": os.path.basename(args.input), "n_total": int(len(ENT)),
                    "horizons": horizons, "label": "answer-based"}, "results": {}}

    for sub_name, keep in [("all", np.ones(len(ENT), bool)), ("format_ok", fmt)]:
        out["results"][sub_name] = {}
        for h in horizons:
            feat = {"ent": [], "com": [], "both": [], "mean": []}; yy, pp = [], []
            for i in range(len(ENT)):
                if not keep[i]: continue
                pe = profile(ENT[i], h); pc = profile(COM[i], h)
                if pe is None or pc is None: continue
                feat["ent"].append(pe[0]); feat["com"].append(pc[0])
                feat["both"].append(np.concatenate([pe[0], pc[0]])); feat["mean"].append([pe[1], pc[1]])
                yy.append(y[i]); pp.append(pid[i])
            yy = np.array(yy); pp = np.array(pp)
            prob = {}
            for j, p in enumerate(pp): prob.setdefault(int(p), []).append(j)
            groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]
            key = "full" if h == 0 else f"first{h}"
            block = {"N": int(len(yy)), "contrastive_problems": len(groups), "n_incorrect": int(yy.sum())}
            kw = dict(kfold=args.kfold, n_seeds=args.n_seeds, seed=args.seed)
            block["static_mean"] = auroc_both(np.array(feat["mean"]), yy, pp, groups, **kw)
            block["entropy_profile"] = auroc_both(np.array(feat["ent"]), yy, pp, groups, **kw)
            block["committal_profile"] = auroc_both(np.array(feat["com"]), yy, pp, groups, **kw)
            block["both_profile"] = auroc_both(np.array(feat["both"]), yy, pp, groups, **kw)
            if h == 0:
                F = np.array(feat["both"]); nm = ["ent_early", "ent_mid", "ent_late", "ent_slope", "ent_r2",
                                                  "com_early", "com_mid", "com_late", "com_slope", "com_r2"]
                block["feature_means"] = {nm[k]: {"correct": round(float(F[yy == 0, k].mean()), 4),
                                                  "error": round(float(F[yy == 1, k].mean()), 4)}
                                          for k in range(len(nm))}
            out["results"][sub_name][key] = block
            print(f"[{sub_name}/{key}] N={block['N']} contrastive={block['contrastive_problems']}  "
                  f"both: cross={block['both_profile']['cross']} within={block['both_profile']['within']}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"\nwrote {args.out}")
    print(json.dumps(out["results"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
