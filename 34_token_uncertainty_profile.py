"""Step 34: faithful uncertainty-trace-profile (arXiv 2605.07776) on PER-TOKEN traces.

Uses sv_tok_entropy / sv_tok_committal produced by 10 --store_token_uncertainty (per-token,
during actual generation), so this is a faithful reproduction of the profile method for the
two aleatoric measures (entropy = distributional, committal = p(1-p)), trace channel.
(Epistemic gradient measure and the answer channel are NOT included -- they need extra
machinery; with these two measures we already cover the bulk.)

For each measure and horizon (full trace / first-N tokens) compute the 5 profile features
(mu_early, mu_mid, mu_late, slope, r2). Report cross-problem (random 5-fold, the paper's
setting) AND within-problem (GroupKFold paired, our difficulty control) AUROC, vs the static
mean baseline. Label: answer-based.
"""
from __future__ import annotations
import argparse
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--horizons", default="0,300", help="0=full trace; else first-N tokens")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    if "sv_tok_entropy" not in dd.files:
        raise SystemExit("npz lacks sv_tok_entropy -- run 10 --store_token_uncertainty first.")
    ENT = dd["sv_tok_entropy"]; COM = dd["sv_tok_committal"]
    pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(ENT), bool)
    keep = fmt if args.format_ok else np.ones(len(ENT), bool)
    measures = {"entropy": ENT, "committal": COM}

    def run(F, y, pp, groups, name):
        N = len(y); cw, cc = [], []
        for s in range(args.n_seeds):
            og = cv(F, y, gfolds(pp, args.kfold, args.seed+s))
            orr = cv(F, y, rfolds(N, args.kfold, args.seed+s))
            a = within_pair_auroc(groups, og, y); cw.append(max(a, 1-a))
            mm = np.isfinite(orr); cr = roc_auc_score(y[mm], orr[mm]); cc.append(max(cr, 1-cr))
        print(f"  {name:34s} 跨题={np.mean(cc):.3f}  题内={np.mean(cw):.3f}")

    for h in [int(x) for x in args.horizons.split(",")]:
        tag = "全trace" if h == 0 else f"前{h}token"
        # build features for kept chains with valid profiles in BOTH measures
        rows_feat = {"both": [], "ent": [], "com": [], "mean": []}
        yy, pp = [], []
        for i in range(len(ENT)):
            if not keep[i]: continue
            pe = profile(measures["entropy"][i], h); pc = profile(measures["committal"][i], h)
            if pe is None or pc is None: continue
            rows_feat["ent"].append(pe[0]); rows_feat["com"].append(pc[0])
            rows_feat["both"].append(np.concatenate([pe[0], pc[0]]))
            rows_feat["mean"].append([pe[1], pc[1]])
            yy.append(y[i]); pp.append(pid[i])
        yy = np.array(yy); pp = np.array(pp)
        prob = {}
        for j, p in enumerate(pp): prob.setdefault(int(p), []).append(j)
        groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]
        print(f"\n##### 视野={tag}  子集={'仅格式合规' if args.format_ok else '全链'}  "
              f"N={len(yy)} 对照题={len(groups)} #####")
        run(np.array(rows_feat["mean"]), yy, pp, groups, "静态均值(熵+committal)")
        run(np.array(rows_feat["ent"]), yy, pp, groups, "熵 profile(5)")
        run(np.array(rows_feat["com"]), yy, pp, groups, "committal profile(5)")
        run(np.array(rows_feat["both"]), yy, pp, groups, "两measure profile(10)")
        if h == 0:
            F = np.array(rows_feat["both"]); names = ["熵_早", "熵_中", "熵_晚", "熵_slope", "熵_r2",
                                                      "com_早", "com_中", "com_晚", "com_slope", "com_r2"]
            print("  特征均值(正确|错误|差):")
            for k, nm in enumerate(names):
                c = F[yy == 0, k].mean(); e = F[yy == 1, k].mean()
                print(f"    {nm:9s} {c:+.3f} | {e:+.3f} | {e-c:+.3f}")

    print("\n判读: 题内 AUROC 是否 >0.5 且 > 静态均值 = 动态形状带来难度无关的真信号;"
          " 前N token 接近全trace = 早期可检测。")


if __name__ == "__main__":
    main()
