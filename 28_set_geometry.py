"""Step 28: SET-level geometry -- the correct form of the original anchor.

Per-vector participation (PR) reversed under clean labels (error LESS diffuse per chain
in mid layers). But the anchor "error reasoning spreads over more dimensions" is really a
SET-level claim: does the ERROR-chain point cloud span more dimensions / leave the correct
subspace / occupy a different subspace than the CORRECT-chain cloud?

Design (lessons baked in):
  - label: answer-based (answer correct = correct), format ignored. Also reported for the
    format_ok subset (well-formed chains only) since format failures are geometric outliers.
  - per-problem centering: subtract each problem's chain-mean -> removes difficulty/content,
    keeps only within-problem scatter (difficulty-controlled).
  - equal-n: effective dimension grows with sample count, so subsample correct and error to
    the SAME n before measuring; average over repeats.

Measures per band:
  D_correct, D_error : effective dimension = participation ratio of the cloud's covariance
                       spectrum, (sum lambda)^2 / sum lambda^2 (via the n x n Gram). Larger
                       = the cloud spreads over more dimensions.   -> "更发散 / 更高维"
  energy_out         : held-out fraction of a chain's energy OUTSIDE the correct subspace
                       (fit on train correct), error vs correct.   -> "错误是否跑出正确子空间"
  subspace_cos       : mean cosine of principal angles between the correct and error top-k
                       subspaces. 1 = same span, ~0 = different.   -> "是否不同子空间"
"""
from __future__ import annotations
import argparse
import numpy as np


def band_cols(L, band):
    if band == "all": return np.arange(L)
    if band == "deep": return np.arange(int(L*0.6), L)
    if band == "mid": return np.arange(int(L*0.3), int(L*0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


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


def per_problem_center(X, pid, mask):
    Xc = X.copy()
    prob = {}
    for i in np.where(mask)[0]:
        prob.setdefault(int(pid[i]), []).append(i)
    for p, v in prob.items():
        v = np.array(v); Xc[v] = X[v] - X[v].mean(0)
    return Xc


def eff_dim_gram(M):
    # effective dimension = (sum eig)^2 / sum eig^2 of the covariance, via Gram (n x n)
    Mc = M - M.mean(0)
    G = Mc @ Mc.T
    ev = np.linalg.eigvalsh(G); ev = ev[ev > 1e-9]
    if ev.size == 0: return float("nan")
    return float(ev.sum()**2 / (ev**2).sum())


def eff_dim_equal_n(cor, err, R=20, seed=0):
    rng = np.random.default_rng(seed)
    n0 = min(len(cor), len(err))
    if n0 < 10: return float("nan"), float("nan"), n0
    dc, de = [], []
    for _ in range(R):
        ic = rng.choice(len(cor), n0, replace=False); ie = rng.choice(len(err), n0, replace=False)
        dc.append(eff_dim_gram(cor[ic])); de.append(eff_dim_gram(err[ie]))
    return float(np.mean(dc)), float(np.mean(de)), n0


def subspace_basis(M, k):
    Mc = M - M.mean(0)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T   # (d, k) orthonormal columns


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def energy_outside(cor_idx, err_idx, Xc, pid, k, kfold, seed):
    eo_c, eo_e = [], []
    allidx = np.concatenate([cor_idx, err_idx])
    ycor = np.concatenate([np.zeros(len(cor_idx)), np.ones(len(err_idx))])
    g = pid[allidx]
    for tr, te in group_folds(g, kfold, seed):
        tr_cor = allidx[tr][ycor[tr] == 0]
        if len(tr_cor) < k + 5: continue
        U = subspace_basis(Xc[tr_cor], k)               # correct subspace (train)
        for j in te:
            x = Xc[allidx[j]]
            tot = (x**2).sum() + 1e-12
            frac_out = 1.0 - ((U.T @ x)**2).sum() / tot
            (eo_c if ycor[j] == 0 else eo_e).append(frac_out)
    return float(np.mean(eo_c)) if eo_c else float("nan"), float(np.mean(eo_e)) if eo_e else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--bands", default="mid,deep,all")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--k", type=int, default=20, help="subspace dim for energy/angles")
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    d_ = np.load(args.input, allow_pickle=True)
    VEC = d_[f"sv_vec_{args.mode}"]
    pid = d_["problem_ids"].astype(int)
    if "is_correct" not in d_.files:
        raise SystemExit("npz lacks is_correct.")
    y = (d_["is_correct"].astype(int) == 0).astype(int)              # answer-based: 1 = wrong
    fmt = d_["format_ok"].astype(bool) if "format_ok" in d_.files else np.ones(len(VEC), bool)
    N = len(VEC); L = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]

    for subset_name, sub in [("全链(只看答案)", np.ones(N, bool)), ("仅格式合规", fmt)]:
        print(f"\n##### 子集 = {subset_name} #####")
        for band in [b.strip() for b in args.bands.split(",")]:
            cols = band_cols(L, band)
            X = late_vectors(VEC, cols, args.late_lo, d)
            mask = sub & np.isfinite(X).all(1)
            Xc = per_problem_center(X, pid, mask)                    # difficulty-controlled
            cor_idx = np.where(mask & (y == 0))[0]; err_idx = np.where(mask & (y == 1))[0]
            Dc, De, n0 = eff_dim_equal_n(Xc[cor_idx], Xc[err_idx], args.repeats, args.seed)
            eo_c, eo_e = energy_outside(cor_idx, err_idx, Xc, pid, args.k, args.kfold, args.seed)
            Uc = subspace_basis(Xc[cor_idx], args.k); Ue = subspace_basis(Xc[err_idx], args.k)
            cos_pa = float(np.mean(np.linalg.svd(Uc.T @ Ue, compute_uv=False)))
            print(f"  [{band:4s}] n_cor={len(cor_idx)} n_err={len(err_idx)} (equal-n={n0})")
            print(f"         有效维数  正确={Dc:.1f}  错误={De:.1f}  (错误/正确={De/Dc:.2f})" if Dc==Dc else "         有效维数 n/a")
            print(f"         正确子空间外能量(留出)  正确={eo_c:.3f}  错误={eo_e:.3f}")
            print(f"         正确/错误子空间 主夹角平均cos={cos_pa:.3f}  (1=同子空间)")

    print("\n解读:")
    print("  有效维数 错误>正确  => 错误推理在集合层面铺开更多维度(原锚点的集合版成立)")
    print("  错误的子空间外能量 > 正确  => 错误跑到正确推理用不到的维度")
    print("  主夹角 cos 明显<1  => 两者占据不同子空间; 接近1 => 同子空间只是错误更散")


if __name__ == "__main__":
    main()
