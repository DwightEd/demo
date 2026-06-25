"""Whole-response Gram functionals, STRICT to the paper 2602.09158 (Eq 1-2), on FULL-dim hidden states.
  G_l = H_l H_l^T  (H_l = m tokens x d, raw, uncentered);  HS = (1/m) log det G_l = (1/m) sum_i log lambda_i (Eq 1);
  ME = -sum_i q_i log q_i, q_i = lambda_i / trace(G_l)  (Eq 2);  lam1 = top q_i (ours).
HS is finite only when G_l is full rank (m <= d) -> requires FULL-dim hidden (data/hidden shards), NOT JL-256.
Response-level (chain correct vs error) raw vs centered; step-level prefix-cumulative dF(ME) vs within-step."""
from __future__ import annotations
import argparse
import numpy as np
import hidden_io


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
    """(HS, ME, lam1) strict to Eq 1-2. HS=nan if G_l rank-deficient (m>d)."""
    m = len(H)
    if m < 4:
        return (np.nan, np.nan, np.nan)
    M = (H - H.mean(0)) if center else H
    s = np.linalg.svd(M, compute_uv=False)        # min(m,d) singular values of H_l
    lam = s ** 2                                   # eigenvalues of G_l = H_l H_l^T (the min(m,d) of them)
    nz = lam[lam > 1e-9]
    if len(nz) < 2:
        return (np.nan, np.nan, np.nan)
    q = nz / nz.sum()                              # q_i = lambda_i / trace(G_l)
    ME = float(-(q * np.log(q)).sum())             # Eq 2
    lam1 = float(q[0])
    HS = float(np.log(lam).sum() / m) if (len(lam) == m and lam.min() > 1e-9) else np.nan   # Eq 1, strict
    return (HS, ME, lam1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    ges = z["gold_error_step"].astype(int)
    if "hidden_stored" in z.files and bool(z["hidden_stored"]):
        hd = str(z["hidden_dir"]); hl = [int(x) for x in z["hidden_layers"]]; lc = hl.index(args.layer); ids = z["ids"]
        def getH(i):
            a = hidden_io.load_chain(hd, ids[i]); return np.asarray(a[:, lc, :], np.float64)
        src = f"FULL-dim hidden shards ({hd})"; N = len(ids)
    else:
        csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer); RC = z["respcloud"]
        def getH(i):
            return None if RC[i] is None else np.asarray(RC[i], np.float64)[:, li, :]
        src = "respcloud (JL-256: HS rank-deficient -> nan)"; N = len(RC)

    fn = ["HS", "ME", "lam1"]
    RESP_raw = {k: [] for k in fn}; RESP_cen = {k: [] for k in fn}; RLEN = []; RY = []
    for i in range(N):
        H = getH(i)
        if H is None or len(H) < 4:
            continue
        k = int(ges[i]); correct = k < 0
        wr = feats(H, False); wc = feats(H, True)
        for c, nm in enumerate(fn):
            RESP_raw[nm].append(wr[c]); RESP_cen[nm].append(wc[c])
        RLEN.append(len(H)); RY.append(int(not correct))
    RLEN = np.asarray(RLEN, float); RY = np.asarray(RY, int)
    print(f"{args.npz} | L{args.layer} | {src} | chains {len(RY)} err {int(RY.sum())}")
    print("  -- RESPONSE-LEVEL whole-response Gram (chain correct vs error) --")
    print(f"     {'metric':6s} {'raw(paper)':>22s} {'centered(no-magnitude)':>24s}")
    for nm in fn:
        vr = np.asarray(RESP_raw[nm], float); vc = np.asarray(RESP_cen[nm], float)
        print(f"     {nm:6s}  AUROC {bdir(auroc(vr, RY)):.3f} bkt {bucket(vr, RY, RLEN):.3f}    AUROC {bdir(auroc(vc, RY)):.3f} bkt {bucket(vc, RY, RLEN):.3f}")


if __name__ == "__main__":
    main()
