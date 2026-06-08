"""Step 33: verify the uncertainty-trace-profile result (arXiv 2605.07776) on our data.

That paper: summarize each trace by the SHAPE of its uncertainty signal (slope, linearity,
early/mid/late means) and predict correctness -> AUROC up to 0.807. Their split is a random
5-fold (cross-problem, no difficulty control). Claim: correct traces decline more steeply /
more linearly; incorrect traces stay higher.

We have ONE of their three uncertainty measures: per-step next-token entropy (sv_out_entropy)
= distributional-aleatoric, trace channel, per STEP (coarser than per token). We test whether
the SHAPE features of this entropy trace discriminate, and crucially whether they survive our
WITHIN-PROBLEM (difficulty-controlled) protocol -- since their 0.80 is cross-problem.

Profile features per chain (entropy trace e over normalized step position):
  mu_early (first 25%), mu_mid (middle 50%), mu_late (last 25%), slope (linear coef), r2.
Compared: full profile vs static mean(entropy) baseline; cross-problem vs within-problem.
Label: answer-based.
"""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_auc_score


def profile(e):
    e = np.asarray(e, dtype=np.float64); e = e[np.isfinite(e)]
    T = len(e)
    if T < 4: return None
    pos = np.linspace(0, 1, T)
    A = np.vstack([pos, np.ones(T)]).T
    coef, *_ = np.linalg.lstsq(A, e, rcond=None); slope = coef[0]
    pred = A @ coef; ss_res = ((e-pred)**2).sum(); ss_tot = ((e-e.mean())**2).sum()+1e-12
    r2 = 1 - ss_res/ss_tot
    q = max(1, T//4)
    mu_early = e[:q].mean(); mu_late = e[-q:].mean(); mu_mid = e[q:T-q].mean() if T-2*q > 0 else e.mean()
    return np.array([mu_early, mu_mid, mu_late, slope, r2]), float(e.mean())


def within_pair_auroc(idx_groups, feats, y_inc):
    conc = 0.0; npair = 0
    for idx in idx_groups:
        inc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        cor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if not inc or not cor: continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc)*len(cor)
    return (conc/npair if npair else float("nan")), npair


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def random_folds(n, k, seed):
    idx = np.arange(n); rng = np.random.default_rng(seed); rng.shuffle(idx)
    f = np.empty(n, int); f[idx] = np.arange(n) % k
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def cv_auroc(F, y, folds):
    oof = np.full(len(y), np.nan)
    for tr, te in folds:
        if len(np.unique(y[tr])) < 2: continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced"))
        clf.fit(F[tr], y[tr]); oof[te] = clf.predict_proba(F[te])[:, 1]
    return oof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--format_ok", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dd = np.load(args.input, allow_pickle=True)
    if "sv_out_entropy" not in dd.files:
        raise SystemExit("npz lacks sv_out_entropy.")
    OE = dd["sv_out_entropy"]; pid = dd["problem_ids"].astype(int)
    y = (dd["is_correct"].astype(int) == 0).astype(int)              # 1 = wrong (answer-based)
    fmt = dd["format_ok"].astype(bool) if "format_ok" in dd.files else np.ones(len(OE), bool)
    keep = fmt if args.format_ok else np.ones(len(OE), bool)

    feats, means, yy, pp = [], [], [], []
    for i in range(len(OE)):
        if not keep[i]: continue
        pr = profile(OE[i])
        if pr is None: continue
        feats.append(pr[0]); means.append(pr[1]); yy.append(y[i]); pp.append(pid[i])
    F = np.array(feats); m = np.array(means)[:, None]; yy = np.array(yy); pp = np.array(pp)
    N = len(yy)
    prob = {}
    for i, p in enumerate(pp):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(yy[v] == 1) and any(yy[v] == 0)]
    print(f"N={N}  problems={len(prob)}  contrastive={len(idx_groups)}  "
          f"subset={'format_ok' if args.format_ok else 'all'}  incorrect={int(yy.sum())}")

    def evaluate(X, name):
        cwin, ccross = [], []
        for s in range(args.n_seeds):
            og = cv_auroc(X, yy, group_folds(pp, args.kfold, args.seed+s))
            orr = cv_auroc(X, yy, random_folds(N, args.kfold, args.seed+s))
            a = within_pair_auroc(idx_groups, og, yy)[0]; cwin.append(max(a, 1-a))
            mm = np.isfinite(orr); cr = roc_auc_score(yy[mm], orr[mm]); ccross.append(max(cr, 1-cr))
        print(f"  {name:28s} 跨题(随机折)={np.mean(ccross):.3f}   题内(GroupKFold配对)={np.mean(cwin):.3f}")

    print("\n=== 不确定性轨迹 profile vs 静态均值 ===")
    evaluate(m, "静态 mean(entropy) 基线")
    evaluate(F, "形状 profile (5特征)")
    evaluate(F[:, 3:4], "仅 slope(斜率)")
    evaluate(F[:, 4:5], "仅 r2(线性度)")

    # qualitative: feature means correct vs incorrect
    names = ["mu_early", "mu_mid", "mu_late", "slope", "r2"]
    print("\n=== 特征均值(正确 vs 错误)===")
    for j, nm in enumerate(names):
        c = F[yy == 0, j].mean(); e = F[yy == 1, j].mean()
        print(f"  {nm:9s} 正确={c:+.4f}  错误={e:+.4f}  差(错-对)={e-c:+.4f}")
    print("\n论文预测: 错误链 mu 更高、slope 下降更缓(更不陡)、r2 更低(LM)。看是否一致 +")
    print("最关键: profile 的【题内】AUROC 是否 > 静态均值题内 且 明显>0.5(动态是否真带题内信号)。")


if __name__ == "__main__":
    main()
