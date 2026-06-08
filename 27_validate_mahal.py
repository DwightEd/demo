"""Step 27: validate the Mahalanobis 0.83 against three confounds.

The headline mahal_mid within-AUROC = 0.83 (v2 5shot strict) was computed with:
  (i)  mu/sigma fit on ALL chains (correct+error), and
  (ii) IN-SAMPLE (the scored chains are in the fit), and
  (iii) the strict 'error' class includes ~41% format-failures (right answer but no
        '####' marker), so it may detect malformed generation, not reasoning error.

This script recomputes the within-problem paired AUROC under corrected protocols and
prints them side by side, so we know whether 0.83 survives:

  A) in-sample, global mean (reproduces the reported number)
  B) HELD-OUT cross-fit, mu/sigma from TRAIN-fold CORRECT chains only
     (= genuine distance to the *healthy* manifold, no circularity)
  C) B, restricted to format_ok==1 chains only
     (error = emitted a valid answer but got it WRONG = genuine reasoning error)

Diagonal Mahalanobis (sum of per-dim z^2), late-window step_exp vector, band-mean.
Requires a v2 npz with: sv_vec_step_exp, problem_ids, is_correct_strict, format_ok.
"""
from __future__ import annotations
import argparse
import numpy as np


def band_cols(L, band):
    if band == "all": return np.arange(L)
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid": return np.arange(int(L*0.3), int(L*0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


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


def late_vectors(VEC, cols, late_lo, d):
    N = len(VEC); X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)
        T = P.shape[0]; fr = (np.arange(T)/(T-1)) if T > 1 else np.array([0.0])
        m = fr >= late_lo
        if not m.any(): m = fr >= fr.max()
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(P[m], axis=0)
    return X


def contrastive_groups(problem_ids, y, mask):
    prob = {}
    for i in np.where(mask)[0]:
        prob.setdefault(int(problem_ids[i]), []).append(i)
    return [np.array(v) for v in prob.values() if any(y[i] == 1 for i in v) and any(y[i] == 0 for i in v)]


def insample_mahal(X, mask):
    ok = mask & np.isfinite(X).all(1)
    mu = X[ok].mean(0); vr = X[ok].var(0) + 1e-6
    s = np.full(len(X), np.nan)
    s[ok] = ((X[ok]-mu)**2/vr).sum(1)
    return s


def heldout_mahal(X, y, problem_ids, mask, healthy_correct_only, kfold, n_seeds, seed):
    N = len(X); acc = np.zeros(N); cnt = np.zeros(N)
    finite = np.isfinite(X).all(1)
    for s in range(n_seeds):
        for tr, te in group_folds(problem_ids, kfold, seed+s):
            # fit healthy mean/var on TRAIN chains that are (correct, in-mask, finite)
            fit = [i for i in tr if mask[i] and finite[i] and (y[i] == 0 if healthy_correct_only else True)]
            if len(fit) < 20: continue
            F = X[fit]; mu = F.mean(0); vr = F.var(0) + 1e-6
            for i in te:
                if mask[i] and finite[i]:
                    acc[i] += ((X[i]-mu)**2/vr).sum(); cnt[i] += 1
    out = np.full(N, np.nan); m = cnt > 0; out[m] = acc[m]/cnt[m]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="v2 npz (needs is_correct_strict, format_ok)")
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--bands", default="mid,all,deep")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    if "is_correct" not in data.files:
        raise SystemExit("npz lacks is_correct -- need v2 data.")
    ic = data["is_correct"].astype(int)                              # lenient = answer-based
    ics = (data["is_correct_strict"].astype(int) if "is_correct_strict" in data.files else ic)
    fmt = data["format_ok"].astype(bool) if "format_ok" in data.files else np.ones(len(VEC), bool)
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]

    # Two definitions of an ERROR chain (1 = incorrect):
    #   answer  : wrong final ANSWER, regardless of '####' format (format-fail-but-correct
    #             stays CORRECT)  -> the meaningful definition for reasoning errors.
    #   strict  : wrong answer OR missing '####' marker (format failure counted as error)
    #             -> conflates format failure with reasoning error; this is what gave 0.83.
    label_defs = [("answer(只看答案)", (ic == 0).astype(int)),
                  ("strict(需格式)", (ics == 0).astype(int))]

    # precompute late vectors per band once (reused across labels)
    bands = [b.strip() for b in args.bands.split(",")]
    Xb = {b: late_vectors(VEC, band_cols(L, b), args.late_lo, d) for b in bands}
    allmask = np.ones(N, bool)

    for lname, y in label_defs:
        n_err = int(y.sum()); n_fmt_err = int((y & fmt).sum())
        print(f"\n##### 错误链定义 = {lname} #####")
        print(f"  N={N}  incorrect={n_err}  其中 format_ok={n_fmt_err}"
              f"({n_fmt_err/max(1,n_err)*100:.0f}%)  format-fail={n_err-n_fmt_err}")
        print(f"  {'band':5s} {'A in-sample/global':>18s} {'B heldout/healthy':>18s} {'C heldout/healthy/fmt':>22s}")
        for band in bands:
            X = Xb[band]
            sA = insample_mahal(X, allmask)
            igA = contrastive_groups(problem_ids, y, allmask & np.isfinite(X).all(1))
            aA = max(within_pair_auroc(igA, sA, y)[0], 1 - within_pair_auroc(igA, sA, y)[0])
            sB = heldout_mahal(X, y, problem_ids, allmask, True, args.kfold, args.n_seeds, args.seed)
            igB = contrastive_groups(problem_ids, y, np.isfinite(sB))
            aB = max(within_pair_auroc(igB, sB, y)[0], 1 - within_pair_auroc(igB, sB, y)[0])
            sC = heldout_mahal(X, y, problem_ids, fmt, True, args.kfold, args.n_seeds, args.seed)
            igC = contrastive_groups(problem_ids, y, fmt & np.isfinite(sC))
            aC = max(within_pair_auroc(igC, sC, y)[0], 1 - within_pair_auroc(igC, sC, y)[0])
            print(f"  {band:5s} {aA:18.3f} {aB:18.3f} {aC:22.3f}")

    print("\n解读:")
    print("  对比 answer 与 strict 两块:strict 的高数主要来自把格式失败算成错误。")
    print("  A=in-sample全体均值; B=留出+只用正确链估均值(真·到健康流形,无循环);")
    print("  C=B 且仅格式合规链(纯推理错误)。answer 定义下的 B/C 才是诚实的推理错误检测力。")


if __name__ == "__main__":
    main()
