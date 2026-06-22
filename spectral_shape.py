"""Decompose the WITHIN-STEP dispersion by the SHAPE of its spectrum, on the CENTERED covariance of
the unit token vectors -- the part resultant is blind to.

resultant reads only an aggregate of the spread: total variance Sum(lambda) = trace(C) = 1 - R^2.
Two clouds with the SAME total variance (same resultant) can have opposite SHAPES:
  isotropic spread (lambda_1 ~ lambda_2 ~ ...): variance spread over many directions = diffuse loss
  split into clusters (lambda_1 >> lambda_2): variance on one axis = oscillation between two answers
These are mechanistically different errors that resultant CONFLATES. If the shape carries error signal
beyond total variance, a scale-invariant shape metric adds an ORTHOGONAL axis (it reads the dimension
resultant discards: direction distribution, not total).

Per step, on the CENTERED unit-vector covariance spectrum {lambda_i} (mean direction removed -> pure
shape, orthogonal to resultant by construction):
  PR        = (Sum lambda)^2 / Sum lambda^2     normalized participation ratio (LOW = split/anisotropic)
  top1      = lambda_1 / Sum lambda             dominant-direction share (HIGH = one axis dominates)
  bimod     = 2-means between/within ratio       direct bimodality test (HIGH = two clusters)
  totvar    = Sum lambda = 1 - R^2               CONTROL: this IS resultant (total variance in disguise)

DECISIVE cell: AUROC of PR / top1 / bimod WITHIN the LOW-resultant stratum (cloud already spread, total
variance high) -- does shape separate error from correct THERE, where resultant is saturated? Plus low
|corr| to resultant and a positive increment over [resultant]. If the normalized shape metrics have
signal in the low-R stratum but totvar (raw) tracks resultant, the signal is SHAPE not total variance.
If all ~0.5 in the low-R stratum, dispersion shape carries nothing beyond resultant.

Needs _cloud.npz: respcloud (clouds_stored) + cloud_store_layers + step_token_ranges + gold_error_step.
"""

from __future__ import annotations
import argparse
import numpy as np

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    HAVE_SK = True
except ImportError:
    HAVE_SK = False


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


def wbuck(s, y, key, nb=5):
    m = np.isfinite(s) & np.isfinite(key); s, y, key = s[m], y[m], key[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(key, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(key, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a[m], b[m])[0, 1]))


def shape_feats(H):
    """ordered token states of ONE step -> (resultant, PR, top1, bimod, totvar) on the CENTERED
    unit-vector covariance spectrum. None if <4 valid tokens."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 4:
        return None
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    R = float(np.linalg.norm(u.mean(0)))                       # within-step resultant (reference)
    Uc = u - u.mean(0)                                         # centered -> remove mean direction
    s = np.linalg.svd(Uc, compute_uv=False)
    lam = (s ** 2) / n; tot = float(lam.sum())                # eigenvalues of centered covariance
    if tot <= 1e-12:
        return None
    PR = float(tot * tot / (np.square(lam).sum() + 1e-18))     # normalized participation ratio
    top1 = float(lam[0] / tot)                                 # dominant-direction share
    # 2-means bimodality: init by sign on top centered PC, a few Lloyd steps, between/within ratio
    Vt = np.linalg.svd(Uc, full_matrices=False)[2]
    a = (Uc @ Vt[0]) >= 0
    for _ in range(5):
        if a.sum() < 1 or a.sum() > n - 1:
            break
        c0 = u[a].mean(0); c1 = u[~a].mean(0)
        a = ((u - c0) ** 2).sum(1) <= ((u - c1) ** 2).sum(1)
    if 1 <= a.sum() <= n - 1:
        c0 = u[a].mean(0); c1 = u[~a].mean(0)
        within = 0.5 * (((u[a] - c0) ** 2).sum(1).mean() + ((u[~a] - c1) ** 2).sum(1).mean())
        bimod = float(((c0 - c1) ** 2).sum() / (within + 1e-9))
    else:
        bimod = 0.0
    return R, PR, top1, bimod, tot


def oof(X, y, grp, folds=5):
    s = np.full(len(y), np.nan); X = X.reshape(-1, 1) if X.ndim == 1 else X
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)

    R_, PR_, T1_, BM_, TV_, Y, NT, G = [], [], [], [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i]); correct = (k < 0)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            f = shape_feats(rcl[lo:hi]) if hi - lo >= 4 else None
            if f is None:
                continue
            R_.append(f[0]); PR_.append(f[1]); T1_.append(f[2]); BM_.append(f[3]); TV_.append(f[4])
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1)); G.append(i)
    R_ = np.asarray(R_); PR_ = np.asarray(PR_); T1_ = np.asarray(T1_); BM_ = np.asarray(BM_)
    TV_ = np.asarray(TV_); Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)

    lowR = R_ <= np.quantile(R_, 1 / 3)                        # cloud already spread (R saturated)
    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())} | "
          f"lowR stratum n={int(lowR.sum())} err={int(Y[lowR].sum())}")
    print(f"\n{'signal':28s} {'AUROC':>7s} {'b(len)':>7s} {'|r|R':>6s} {'lowR AUROC':>11s} "
          f"{'meanCor/meanErr(lowR)':>22s}")
    for nm, v in [("resultant (ref)", -R_), ("totvar=1-R^2 (control)", TV_),
                  ("PR (shape, norm)", PR_), ("top1 share (shape)", T1_), ("bimod 2-means (shape)", BM_)]:
        base = {"resultant (ref)": R_, "totvar=1-R^2 (control)": TV_, "PR (shape, norm)": PR_,
                "top1 share (shape)": T1_, "bimod 2-means (shape)": BM_}[nm]
        mc = float(np.nanmean(base[lowR & (Y == 0)])); me = float(np.nanmean(base[lowR & (Y == 1)]))
        print(f"  {nm:28s} {bdir(auroc(v, Y)):7.3f} {wbuck(v, Y, NT):7.3f} {abscorr(base, R_):6.3f} "
              f"{bdir(auroc(base[lowR], Y[lowR])):11.3f} {mc:>10.3f}/{me:<10.3f}")
    if HAVE_SK:
        r_only = oof(-R_, Y, G)
        best = max([("PR", PR_), ("top1", T1_), ("bimod", BM_)],
                   key=lambda kv: bdir(auroc(kv[1][lowR], Y[lowR])))
        fused = oof(np.c_[-R_, PR_, T1_, BM_], Y, G)
        print(f"\n  increment: AUROC(resultant) {bdir(auroc(r_only, Y)):.3f} -> "
              f"AUROC(resultant + all shape) {bdir(auroc(fused, Y)):.3f}   (best lowR shape: {best[0]})")
    print("\nread: DECISIVE = 'lowR AUROC' of PR/top1/bimod with meanCor != meanErr. In the LOW-resultant "
          "stratum the cloud is already spread (total variance high), so resultant is saturated; if SHAPE "
          "separates error from correct THERE, the spectrum shape (isotropic-diffuse vs bimodal-split) is a "
          "real orthogonal axis (low |r|R confirms) and the increment over resultant is positive. totvar is "
          "the control: it = 1-R^2 (|r|R should be ~1) -> if only totvar has signal and PR/top1/bimod ~0.5 "
          "in the low stratum, the dispersion signal is TOTAL VARIANCE (=resultant), not shape, and shape is "
          "closed for good (centered-covariance version, sharper than the earlier Gram test).")


if __name__ == "__main__":
    main()
