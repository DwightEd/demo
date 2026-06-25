"""Sequence-level vs within-step Gram spectral functionals (the paper 2602.09158 uses the WHOLE-sequence Gram).
  Response-level: whole-response Gram functionals (HS=log-vol, ME=eff-rank, lam1) -> chain correct vs error.
  Step-level via PREFIX-cumulative: dF(t) = F(tokens<=step t) - F(<=step t-1) = step t's contribution to the
    global geometry (large-n, cross-step) vs the isolated within-step F. Pooled + within-chain localization.
Runs on respcloud (whole-response per-token cloud) in *_cloud.npz / full_*.npz."""
from __future__ import annotations
import argparse
import numpy as np


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


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


def bucket(s, y, nt, nb=5):
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) < nb * 2:
        return bdir(auroc(s, y))
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def feats(H, center=False):
    """Paper-faithful (center=False): HS = (1/T) sum log(sigma) = seq-normalized log-volume of the RAW Gram;
    ME = -sum p log p (von Neumann matrix entropy). center=True = centered (covariance) ablation = drops magnitude."""
    if len(H) < 4:
        return (np.nan, np.nan, np.nan)
    M = (H - H.mean(0)) if center else H
    s = np.linalg.svd(M, compute_uv=False); s = s[s > 1e-9]
    if len(s) < 2:
        return (np.nan, np.nan, np.nan)
    lam = s ** 2; p = lam / lam.sum()
    HS = float(np.log(s).sum() / len(H))          # sequence-normalized log-volume (paper HS)
    ME = float(-(p * np.log(p)).sum())            # von Neumann matrix entropy (paper ME)
    return (HS, ME, float(p[0]))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    fn = ["HS", "ME", "lam1"]
    # response-level (whole-response Gram): paper-faithful raw + centered ablation
    RESP_raw = {k: [] for k in fn}; RESP_cen = {k: [] for k in fn}; RLEN = []; RY = []
    # step-level prefix-cumulative dF (eff-rank) + within-step eff-rank, for contrast
    P_dF, P_in, P_nt, P_pos, Y, G = [], [], [], [], [], []
    chains = []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); k = int(ges[i]); correct = k < 0; T = rng.shape[0]
        wr = feats(H, False); wcn = feats(H, True)
        for c, nm in enumerate(fn):
            RESP_raw[nm].append(wr[c]); RESP_cen[nm].append(wcn[c])
        RLEN.append(len(H)); RY.append(int(not correct))
        ends = [min(len(H), int(rng[j, 1]) - a0 + 1) for j in range(T)]
        pre = [feats(H[:e])[1] for e in ends]              # eff-rank of prefix <= step j
        seq = np.full(T, np.nan)
        for j in range(T):
            seq[j] = pre[j] - (pre[j - 1] if j >= 1 else 0.0)   # dF(t)
        instep = np.full(T, np.nan)
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = ends[j]
            if hi - lo >= 4:
                instep[j] = feats(H[lo:hi])[1]
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            P_dF.append(seq[j]); P_in.append(instep[j]); P_nt.append(int(rng[j, 1] - rng[j, 0] + 1)); P_pos.append(j); Y.append(lab); G.append(i)
        if not correct and T >= 4:
            chains.append({"seq": seq, "instep": instep, "k": k, "T": T,
                           "nt": np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)})
    RLEN = np.asarray(RLEN, float); RY = np.asarray(RY, int)
    print(f"{args.npz} | L{args.layer} | chains {len(RY)} err-chains {int(RY.sum())} | steps {len(Y)} err {int(np.sum(Y))}")
    print("  -- RESPONSE-LEVEL (whole-response Gram): chain correct vs error --")
    print(f"     {'metric':9s} {'raw(paper)':>20s} {'centered(no-magnitude)':>24s}")
    for nm in fn:
        vr = np.asarray(RESP_raw[nm], float); vc = np.asarray(RESP_cen[nm], float)
        print(f"     {nm:9s}  AUROC {bdir(auroc(vr, RY)):.3f} bkt {bucket(vr, RY, RLEN):.3f}   "
              f"AUROC {bdir(auroc(vc, RY)):.3f} bkt {bucket(vc, RY, RLEN):.3f}")
    P_dF = np.asarray(P_dF); P_in = np.asarray(P_in); P_nt = np.asarray(P_nt, float); P_pos = np.asarray(P_pos, float); Y = np.asarray(Y, int)
    print("  -- STEP-LEVEL: prefix-cumulative dF(eff-rank) vs within-step eff-rank --")
    print(f"     prefix dF   pooled {bdir(auroc(P_dF, Y)):.3f}  bkt {bucket(P_dF, Y, P_nt):.3f}")
    print(f"     within-step pooled {bdir(auroc(P_in, Y)):.3f}  bkt {bucket(P_in, Y, P_nt):.3f}")

    def wc(arr_key, ctrl):
        locs, w = [], []
        sign = 1.0 if auroc({"seq": P_dF, "instep": P_in}[arr_key], Y) >= 0.5 else -1.0
        for ch in chains:
            v = ch[arr_key]; k = ch["k"]; c = ch[ctrl]; fin = np.isfinite(v) & np.isfinite(c)
            if fin.sum() < 3 or not fin[k]:
                continue
            b = np.polyfit(c[fin], v[fin], 1); res = sign * (v - (b[0] * c + b[1]))
            others = np.array([j for j in range(len(v)) if j != k and fin[j]])
            if len(others) < 2:
                continue
            locs.append(np.mean(res[others] < res[k])); w.append(len(others))
        return np.average(locs, weights=np.asarray(w, float)) if locs else float("nan")

    for ch in chains:
        ch["pos"] = np.arange(ch["T"], dtype=float)
    print(f"     prefix dF   within-chain wc_loc(perp len) {wc('seq','nt'):.3f}   wc_loc(perp pos) {wc('seq','pos'):.3f}")
    print(f"     within-step within-chain wc_loc(perp len) {wc('instep','nt'):.3f}")


if __name__ == "__main__":
    main()
