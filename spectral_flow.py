#!/usr/bin/env python3
"""Spectral Flow of Reasoning — token-wise / layer-wise spectral-geometry verification.

统一对象: 对每个 token 窗口/步 H (n,d)，κ 是未中心化一阶矩（单位 token 合向量长度），
α (Spectral Geometry of Thought, 2604.15350) / PR / eff_rank 是中心化 Gram 谱的形状泛函。
本脚本验证三条可证伪预测（validate_phase_instability.py 只覆盖了"池化步向量的链级
方向统计"并已将其证伪为长度代理，不覆盖以下任何一条）：

  S1 检测增量: 步级 α_t / Δα_t / 再收缩指数 / 方向有效秩 ρ (uncentered/centered/
     定m子采样三变体, 见 rho_dir/rho_sub) 在 [κ_exp + logN] 之上是否有增量
     (oof_logit + 按题聚类 bootstrap CI —— 这是"指标如何结合"的规范答案: 消融阶梯)。
     二阶矩前科: 链级 stepvec eff_rank=长度代理(已杀), 原始云 cloud_D 净增+0.02(已杀);
     本次测的是"单位化 token + 定 m 子采样"的判决变体。
  S2 再收缩: 错误步的步内再收缩指数 (步末κ − 步初κ) 是否弱于正确步 (长度分桶)。
  S3 流断裂定位 (链内秩检验): 错误链中 gold 首错步的断裂分数在本链所有候选步中的
     排名是否显著优于随机 —— 同链内比较，构造上免疫长度/难度混杂。
  S4 层剖面: 以上信号随层 [10,14,18,22] 的变化。
  S5 相位形状: 滑窗 κ/α 序列的"末段收敛指数" (后1/3均值 − 前1/3均值) 链级区分正确/错误。
  S6 谱级联 (SGoT Finding 5/6 的定量升级, 原文仅 1 模型 5 链无统计检验):
     各层滑窗 α 梯度序列的层间 Pearson 相关 —— (a) ρ(层距) 衰减剖面;
     (b) 错误链 vs 正确链的层间同步差 (bucket 控长度); (c) gold 首错步边界附近的
     "局部失同步"链内秩检验 —— 检验"标点漏拍处即失败处"。--skip_cascade 可跳过。

标签约定: is_correct_strict 1=correct; gold_error_step -1=全对 (见 DATA.md 红线)。
用法:
  python spectral_flow.py --dataset gsm8k
  python spectral_flow.py --dataset math --layers 14 22
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nts.data.loader import _fn                      # noqa: E402
from nts.signals.alpha import spectral_alpha         # noqa: E402
from nts.geom.ntc import participation_ratio         # noqa: E402
from nts.eval.metrics import auroc, bdir, bucket     # noqa: E402
from nts.eval.confound import oof_logit, cluster_boot_increment  # noqa: E402

MIN_TOK_SPECTRUM = 5    # α 需要 >=5 个奇异值
MIN_TOK_HALF = 4        # 再收缩指数每半段至少 4 token
MIN_TOK_BREAK = 8       # 边界窗口每侧至少 8 token


# ---------------------------------------------------------------------------
# per-window statistics
# ---------------------------------------------------------------------------

def _unit_rows(H):
    n = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.maximum(n, 1e-9)


def kappa_mean(H):
    """plain mean resultant length of unit token vectors, in [0,1]."""
    if len(H) < 2:
        return float("nan")
    return float(np.linalg.norm(_unit_rows(H).mean(0)))


def kappa_exp(H):
    """exp-pooled resultant (matches extract_features 'resultant' convention)."""
    n = len(H)
    if n < 2:
        return float("nan")
    w = np.exp(np.arange(n) / max(n - 1, 1))
    w = w / w.sum()
    return float(np.linalg.norm((w[:, None] * _unit_rows(H)).sum(0)))


def _entropy_rank(lam):
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        return float("nan")
    lam = lam / lam.sum()
    return float(np.exp(-(lam * np.log(lam)).sum()))


def rho_dir(H, center=False):
    """方向有效秩: 单位 token 向量散布矩阵谱的熵指数, ρ∈[1, n]。

    uncentered (center=False): S=(1/n)ΣûûT, trace=1, λ1>=κ² —— 谱里含一阶矩(κ)；
    centered  (center=True):  先减合方向再看谱 —— 与 κ 正交的纯二阶矩形状。
    经 Gram 技巧 (n×n) 计算, n=步内 token 数, 开销可忽略。
    注意 ρ<=n 且随 n 有采样噪声增长 —— 判读必须配 bucket(n) + 增量 CI + rho_sub。"""
    if len(H) < 2:
        return float("nan")
    U = _unit_rows(np.asarray(H, np.float64))
    if center:
        U = U - U.mean(0)
    G = U @ U.T / len(U)
    return _entropy_rank(np.linalg.eigvalsh(G))


def rho_sub(H, m=16, draws=3, seed=0):
    """定 m 子采样方向有效秩: 每步都在同样 m 个 token 上评估, 机械 n 依赖
    (rank<=n + 采样噪声) 被构造性消灭 —— 这是二阶矩信号 vs 长度代理的判决变体。"""
    n = len(H)
    if n < m:
        return float("nan")
    rng = np.random.default_rng(seed)
    vals = [rho_dir(H[rng.choice(n, m, replace=False)]) for _ in range(draws)]
    vals = [v for v in vals if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def window_stats(H):
    """(kappa_mean, alpha) for one token window; nan-safe."""
    if len(H) < 2:
        return float("nan"), float("nan")
    k = kappa_mean(H)
    a = spectral_alpha(H) if len(H) >= MIN_TOK_SPECTRUM else float("nan")
    return k, a


# ---------------------------------------------------------------------------
# per-chain feature extraction (one layer column)
# ---------------------------------------------------------------------------

def chain_features(H, rel_ranges, w=32, stride=16):
    """H: (R,d) float32 token hiddens (response only). rel_ranges: (T,2) half-open.

    Returns dict of per-step arrays (T,) + chain-level scalars."""
    T = len(rel_ranges)
    out = {k: np.full(T, np.nan) for k in
           ("alpha", "dalpha", "kap_mean", "kap_exp", "pr", "recon",
            "rho_dir", "rho_dir_c", "rho_sub",
            "break_kap", "break_alpha", "ntok")}
    for t, (lo, hi) in enumerate(rel_ranges):
        lo, hi = int(lo), int(hi)
        seg = H[lo:hi]
        n = len(seg)
        out["ntok"][t] = n
        if n < 2:
            continue
        out["kap_mean"][t] = kappa_mean(seg)
        out["kap_exp"][t] = kappa_exp(seg)
        out["rho_dir"][t] = rho_dir(seg)
        out["rho_dir_c"][t] = rho_dir(seg, center=True)
        out["rho_sub"][t] = rho_sub(seg, seed=t)  # 每步定种子, 可复现
        if n >= MIN_TOK_SPECTRUM:
            out["alpha"][t] = spectral_alpha(seg)
            out["pr"][t] = participation_ratio(seg)
        # within-step re-concentration: late-half kappa minus early-half kappa
        if n >= 2 * MIN_TOK_HALF:
            half = n // 2
            out["recon"][t] = kappa_mean(seg[half:]) - kappa_mean(seg[:half])
        # boundary flow break: window just before step start vs window just after
        if t >= 1:
            before = H[max(0, lo - w):lo]
            after = H[lo:min(len(H), lo + w)]
            if len(before) >= MIN_TOK_BREAK and len(after) >= MIN_TOK_BREAK:
                kb, ab = window_stats(before)
                ka, aa = window_stats(after)
                out["break_kap"][t] = ka - kb
                out["break_alpha"][t] = aa - ab
    out["dalpha"][1:] = out["alpha"][1:] - out["alpha"][:-1]

    # chain-level phase shape from a strided window series
    conv_kap = conv_alpha = float("nan")
    if len(H) >= 3 * w:
        ks, als = [], []
        for s in range(0, len(H) - w + 1, stride):
            k, a = window_stats(H[s:s + w])
            ks.append(k)
            als.append(a)
        ks, als = np.asarray(ks), np.asarray(als)
        third = max(1, len(ks) // 3)

        def _conv(series):
            a, b = series[:third], series[-third:]
            a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            return float(b.mean() - a.mean()) if (len(a) and len(b)) else float("nan")

        conv_kap, conv_alpha = _conv(ks), _conv(als)
    return out, {"conv_kap": conv_kap, "conv_alpha": conv_alpha, "n_tokens": int(len(H))}


# ---------------------------------------------------------------------------
# evaluation helpers
# ---------------------------------------------------------------------------

def cascade_chain(Hs, rel, w=32, stride=8):
    """S6 谱级联: 每层滑窗 α 序列 → 梯度 → 层间相关 + 步边界局部同步。

    Hs: list of (R,d) —— 同一条链在各存储层的 token 隐状态。
    Returns None (链太短) 或 {"pair": {(i,j): ρ}, "sync": (T,) 每步入口边界的局部
    层间同步 (±2w token 邻域内 α 梯度的平均成对相关)}。"""
    R = len(Hs[0])
    if R < 3 * w:
        return None
    starts = np.arange(0, R - w + 1, stride)
    A = np.full((len(starts), len(Hs)), np.nan)
    for li, H in enumerate(Hs):
        for si, s in enumerate(starts):
            A[si, li] = spectral_alpha(H[s:s + w])
    G = np.diff(A, axis=0)                      # (W-1, L) per-window alpha gradients
    centers = starts[1:].astype(float) + w / 2.0 - stride / 2.0
    L = len(Hs)

    def _pair_corrs(M, min_n):
        out = {}
        for i in range(L):
            for j in range(i + 1, L):
                m = np.isfinite(M[:, i]) & np.isfinite(M[:, j])
                if m.sum() >= min_n and np.std(M[m, i]) > 0 and np.std(M[m, j]) > 0:
                    out[(i, j)] = float(np.corrcoef(M[m, i], M[m, j])[0, 1])
                else:
                    out[(i, j)] = float("nan")
        return out

    pair = _pair_corrs(G, min_n=8)
    T = len(rel)
    sync = np.full(T, np.nan)
    for t in range(1, T):
        p = float(rel[t][0])
        sel = np.abs(centers - p) <= 2 * w
        if sel.sum() >= 6:
            cs = [v for v in _pair_corrs(G[sel], min_n=5).values() if np.isfinite(v)]
            if cs:
                sync[t] = float(np.mean(cs))
    return {"pair": pair, "sync": sync}


def run_cascade(crecs, layer_names):
    """Aggregate S6 across chains. crecs: list of dicts with pair/sync/labels."""
    lines = ["===== S6 spectral cascade (cross-layer alpha-gradient sync) ====="]
    res = {}
    # (a) ρ(layer distance) profile, pooled over chains
    dist_vals = {}
    for r in crecs:
        for (i, j), v in r["pair"].items():
            d = abs(layer_names[j] - layer_names[i])
            dist_vals.setdefault(d, []).append(v)
    prof = {}
    for d in sorted(dist_vals):
        v = np.asarray(dist_vals[d], float)
        v = v[np.isfinite(v)]
        if len(v):
            prof[d] = {"mean_rho": float(v.mean()), "n": int(len(v))}
            lines.append(f"  [S6a] rho(layer distance {d:2d}) = {v.mean():+.3f} (n={len(v)})")
    res["rho_by_distance"] = prof
    # (b) chain-level mean sync: correct vs error (bucket by n_tokens)
    msync = np.array([np.nanmean(list(r["pair"].values())) for r in crecs])
    yc = np.array([0 if r["correct"] else 1 for r in crecs])
    ntok = np.array([r["ntok"] for r in crecs], float)
    m = np.isfinite(msync)
    if m.sum() >= 50 and len(np.unique(yc[m])) == 2:
        res["chain_sync"] = {
            "auroc": bdir(auroc(msync[m], yc[m])),
            "bucket_ntok": bucket(msync[m], yc[m], ntok[m]),
            "mean_correct": float(np.mean(msync[m & (yc == 0)])),
            "mean_error": float(np.mean(msync[m & (yc == 1)])),
            "n": int(m.sum())}
        c = res["chain_sync"]
        lines.append(f"  [S6b] chain mean sync: corr {c['mean_correct']:+.3f} err {c['mean_error']:+.3f} "
                     f"| AUROC {c['auroc']:.3f} bucket(ntok) {c['bucket_ntok']:.3f}")
    # (c) within-chain rank of local sync at the gold boundary (sign from detection)
    Ys, Ss = [], []
    for r in crecs:
        okm = r["ok"] & np.isfinite(r["sync"])
        Ys.append(r["y"][okm]); Ss.append(r["sync"][okm])
    Yf = np.concatenate(Ys) if Ys else np.array([])
    Sf = np.concatenate(Ss) if Ss else np.array([])
    sign = 1.0
    if len(Yf) >= 50 and len(np.unique(Yf)) == 2:
        sign = 1.0 if auroc(Sf, Yf) >= 0.5 else -1.0
    err = [r for r in crecs if (not r["correct"]) and r["ges"] >= 0]
    t1, e1, pct, n = within_chain_rank([r["sync"] for r in err], [r["ges"] for r in err], sign)
    res["gold_boundary_sync"] = {"top1": t1, "expected_top1": e1, "mean_pct": pct,
                                 "n": n, "sign": sign}
    if np.isfinite(t1):
        lines.append(f"  [S6c] gold-boundary local (de)sync: top1 {t1:.3f} (exp {e1:.3f}) "
                     f"pct {pct:.3f} n={n} sign={'+' if sign > 0 else '-'}")
    return res, lines


def eval_mask_from(y, correct):
    ok = np.ones(len(y), bool)
    if not correct:
        if (y == 1).any():
            ok[int(np.argmax(y == 1)) + 1:] = False
        else:
            ok[:] = False
    return ok


def within_chain_rank(chains_scores, chains_ges, sign):
    """S3 链内秩检验: 每条错误链上 gold 步分数(乘 sign 后取大为可疑)在有限候选中的名次。

    Returns (top1_rate, expected_top1, mean_percentile, n_used); percentile 0=最可疑。"""
    top1, exp1, pct = [], [], []
    for s, g in zip(chains_scores, chains_ges):
        if g < 0 or g >= len(s):
            continue
        sc = sign * np.asarray(s, float)
        m = np.isfinite(sc)
        if not m[g] or m.sum() < 2:
            continue
        cand = sc[m]
        better = int((cand > sc[g]).sum())
        top1.append(float(better == 0))
        exp1.append(1.0 / m.sum())
        pct.append(better / (m.sum() - 1) if m.sum() > 1 else np.nan)
    if not top1:
        return float("nan"), float("nan"), float("nan"), 0
    return (float(np.mean(top1)), float(np.mean(exp1)),
            float(np.nanmean(pct)), len(top1))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_layer(records, layer, folds=5):
    """records: list of dicts with per-step arrays + labels for this layer."""
    res = {"layer": layer}
    lines = [f"===== layer {layer} ====="]

    # ---- flatten step-level (eval-masked) ----
    step_keys = ["alpha", "dalpha", "kap_mean", "kap_exp", "pr", "recon",
                 "rho_dir", "rho_dir_c", "rho_sub",
                 "break_kap", "break_alpha"]
    flat = {k: [] for k in step_keys}
    Y, OK, LEN, PID, CORR = [], [], [], [], []
    for r in records:
        T = len(r["y"])
        for k in step_keys:
            flat[k].append(r["steps"][k])
        Y.append(r["y"]); OK.append(r["ok"]); LEN.append(r["steps"]["ntok"])
        PID.append(np.full(T, r["pid"])); CORR.append(np.full(T, r["correct"]))
    flat = {k: np.concatenate(v) for k, v in flat.items()}
    Y = np.concatenate(Y); OK = np.concatenate(OK).astype(bool)
    LEN = np.concatenate(LEN); PID = np.concatenate(PID); CORR = np.concatenate(CORR)
    logn = np.log1p(LEN)

    # ---- S1/S2: detection table + increment over [kappa_exp + logN] ----
    lines.append(f"[S1/S2 detection] steps {int(OK.sum())}/{len(Y)} err {int(Y[OK].sum())}")
    det = {}
    base_cols = np.column_stack([flat["kap_exp"], logn])
    for k in step_keys:
        s = flat[k]
        m = OK & np.isfinite(s)
        if m.sum() < 50 or len(np.unique(Y[m])) < 2:
            det[k] = {"auroc": float("nan")}
            continue
        a = bdir(auroc(s[m], Y[m])); b = bucket(s[m], Y[m], LEN[m])
        entry = {"auroc": a, "bucket": b, "n": int(m.sum())}
        if k != "kap_exp":  # increment over the incumbent kappa
            mm = m & np.all(np.isfinite(base_cols), 1)
            if mm.sum() >= 100 and len(np.unique(Y[mm])) == 2:
                sb = oof_logit(base_cols[mm], Y[mm], PID[mm], folds)
                sf = oof_logit(np.column_stack([base_cols[mm], s[mm][:, None]]),
                               Y[mm], PID[mm], folds)
                mean, lo, hi, sig = cluster_boot_increment(sf, sb, Y[mm], PID[mm])
                entry["inc_over_kexp_logn"] = [mean, lo, hi, bool(sig)]
        det[k] = entry
        inc = entry.get("inc_over_kexp_logn")
        inc_s = (f" | +{inc[0]:.3f} [{inc[1]:+.3f},{inc[2]:+.3f}]"
                 f" {'SIG' if inc[3] else 'ns'}" if inc else "")
        lines.append(f"  {k:12s} AUROC {a:.3f} bucket {b:.3f}{inc_s}")
    res["detection"] = det

    # ---- S3: within-chain localization rank ----
    lines.append("[S3 within-chain localization] (top1 vs expected; pct 0=most suspicious)")
    loc = {}
    err_recs = [r for r in records if (not r["correct"]) and r["ges"] >= 0]
    for k in step_keys:
        # sign: direction that detection said is error-like (one bit, noted)
        a = det.get(k, {}).get("auroc", float("nan"))
        s_raw = flat[k]
        m = OK & np.isfinite(s_raw)
        sign = 1.0
        if m.sum() >= 50 and len(np.unique(Y[m])) == 2:
            sign = 1.0 if auroc(s_raw[m], Y[m]) >= 0.5 else -1.0
        scores = [r["steps"][k] for r in err_recs]
        ges = [r["ges"] for r in err_recs]
        t1, e1, pct, n = within_chain_rank(scores, ges, sign)
        loc[k] = {"top1": t1, "expected_top1": e1, "mean_pct": pct, "n": n, "sign": sign}
        if np.isfinite(t1):
            lines.append(f"  {k:12s} top1 {t1:.3f} (exp {e1:.3f}) pct {pct:.3f} n={n}"
                         f" sign={'+' if sign > 0 else '-'}")
    res["localization"] = loc

    # ---- S5: chain-level phase shape ----
    cv_k = np.array([r["chain"]["conv_kap"] for r in records])
    cv_a = np.array([r["chain"]["conv_alpha"] for r in records])
    yc = np.array([0 if r["correct"] else 1 for r in records])
    ntok = np.array([r["chain"]["n_tokens"] for r in records], float)
    phase = {}
    for nm, v in (("conv_kap", cv_k), ("conv_alpha", cv_a)):
        m = np.isfinite(v)
        if m.sum() >= 50 and len(np.unique(yc[m])) == 2:
            phase[nm] = {"auroc": bdir(auroc(v[m], yc[m])),
                         "bucket_ntok": bucket(v[m], yc[m], ntok[m]),
                         "mean_correct": float(np.mean(v[m & (yc == 0)])),
                         "mean_error": float(np.mean(v[m & (yc == 1)])),
                         "n": int(m.sum())}
            p = phase[nm]
            lines.append(f"[S5 phase] {nm:10s} AUROC {p['auroc']:.3f} "
                         f"bucket(ntok) {p['bucket_ntok']:.3f} "
                         f"corr {p['mean_correct']:+.4f} err {p['mean_error']:+.4f}")
    res["phase_shape"] = phase
    return res, lines


def main():
    ap = argparse.ArgumentParser(description="Spectral Flow of Reasoning verification")
    ap.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "math", "omnimath"])
    ap.add_argument("--data_dir", default="/gz-data/research/demo/data")
    ap.add_argument("--layers", type=int, nargs="*", default=None,
                    help="model layers to analyze (default: all stored hidden layers)")
    ap.add_argument("--window", type=int, default=32)
    ap.add_argument("--stride", type=int, default=16)
    ap.add_argument("--cascade_stride", type=int, default=8,
                    help="S6 级联滑窗步长 (α 梯度序列的时间分辨率)")
    ap.add_argument("--skip_cascade", action="store_true", help="跳过 S6 (省 ~一半时间)")
    ap.add_argument("--max_chains", type=int, default=0, help="0 = all (debug cap)")
    ap.add_argument("--output_dir", default="outputs/spectral_flow")
    args = ap.parse_args()

    npz_path = os.path.join(args.data_dir, "features", f"full_{args.dataset}.npz")
    z = np.load(npz_path, allow_pickle=True)
    hidden_dir = os.path.join(args.data_dir, "hidden", args.dataset)
    hlayers = [int(x) for x in z["hidden_layers"]]
    layers = args.layers or hlayers
    cols = {}
    for L in layers:
        if L in hlayers:
            cols[L] = hlayers.index(L)
        else:
            c = int(np.argmin(np.abs(np.array(hlayers) - L)))
            print(f"[warn] layer {L} not stored; using nearest {hlayers[c]}")
            cols[hlayers[c]] = c
    ids = z["ids"]; ges = z["gold_error_step"].astype(int)
    pid = z["problem_ids"].astype(int); ranges = z["step_token_ranges"]
    if "is_correct_strict" in z.files:  # direction sanity anchor (1=correct)
        agree = float(np.mean((z["is_correct_strict"].astype(int) == 1) == (ges < 0)))
        print(f"label check: P(is_correct_strict==1 <=> gold_error_step<0) = {agree:.3f}")

    N = len(ids) if not args.max_chains else min(args.max_chains, len(ids))
    per_layer_records = {L: [] for L in cols}
    cascade_records = []
    n_missing = 0
    for i in range(N):
        shard = os.path.join(hidden_dir, _fn(ids[i]))
        if not os.path.exists(shard):
            n_missing += 1
            continue
        rr = np.asarray(ranges[i]).astype(int)
        T = len(rr)
        if T < 2:
            continue
        a0 = int(rr[0, 0])  # 绝对闭区间 -> 分片相对半开区间
        rel = np.column_stack([np.maximum(0, rr[:, 0] - a0),
                               np.maximum(0, rr[:, 1] - a0 + 1)])
        y = np.array([1 if (ges[i] >= 0 and t == ges[i]) else 0 for t in range(T)])
        correct = bool(ges[i] < 0)
        ok = eval_mask_from(y, correct)
        Hall = np.load(shard, mmap_mode="r")
        Hs = {}
        for L, c in cols.items():
            H = np.asarray(Hall[:, c, :], dtype=np.float32)
            Hs[L] = H
            steps, chain = chain_features(H, rel, w=args.window, stride=args.stride)
            per_layer_records[L].append({
                "steps": steps, "chain": chain, "y": y, "ok": ok,
                "pid": int(pid[i]), "correct": correct, "ges": int(ges[i]),
            })
        if not args.skip_cascade and len(Hs) >= 2:
            layer_sorted = sorted(Hs)
            cas = cascade_chain([Hs[L] for L in layer_sorted], rel,
                                w=args.window, stride=args.cascade_stride)
            if cas is not None:
                cascade_records.append({
                    "pair": cas["pair"], "sync": cas["sync"], "y": y, "ok": ok,
                    "correct": correct, "ges": int(ges[i]), "pid": int(pid[i]),
                    "ntok": int(len(Hs[layer_sorted[0]])),
                })
        if (i + 1) % 50 == 0:
            print(f"  processed {i + 1}/{N} chains")
    if n_missing:
        print(f"[warn] {n_missing}/{N} chains skipped (shard missing)")

    os.makedirs(args.output_dir, exist_ok=True)
    all_res = {"dataset": args.dataset, "window": args.window, "stride": args.stride,
               "n_chains": N - n_missing, "layers": {}}
    for L, recs in per_layer_records.items():
        if not recs:
            continue
        res, lines = run_layer(recs, L)
        all_res["layers"][str(L)] = res
        print("\n" + "\n".join(lines))
    if cascade_records:
        cres, clines = run_cascade(cascade_records, sorted(cols))
        all_res["cascade"] = cres
        print("\n" + "\n".join(clines))
    out_file = os.path.join(args.output_dir, f"{args.dataset}_flow.json")
    with open(out_file, "w", encoding="utf-8") as fh:
        json.dump(all_res, fh, indent=2, ensure_ascii=False,
                  default=lambda o: None if (isinstance(o, float) and not np.isfinite(o)) else
                  (float(o) if isinstance(o, np.floating) else
                   int(o) if isinstance(o, np.integer) else
                   o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print(f"\nsaved: {out_file}")


if __name__ == "__main__":
    main()
