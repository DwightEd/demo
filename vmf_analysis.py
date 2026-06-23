"""Comprehensive directional-statistics (von Mises-Fisher) test of the anchor.

The token unit vectors of a reasoning step live on the unit sphere S^{d-1}. We model them as i.i.d.
vMF(mu, kappa): mu = mean direction (orientation), kappa = concentration. The sufficient statistic is
sum(x_i), whose TWO halves are R-bar = ||mean|| (-> kappa, our resultant) and mu-hat = direction
(-> orientation). This script tests the full story:

  P1  vMF model fit       : is the within-step direction cloud actually vMF-like (unimodal, tangent-
                            isotropic)? If grossly not, the model is the wrong idealization.
  P2  concentration (kappa): R-bar / kappa-hat AUROC for first-error (the validated axis). Show BOTH
                            correct and error are FAR from uniform (R-bar >> 1/sqrt(n)) -> the
                            discriminator is the DEGREE of kappa vs a healthy baseline, NOT a Rayleigh
                            uniformity test (which both classes reject). Empirical null, not chi^2.
  P3  orientation (mu-hat) : the untapped half of the sufficient statistic. 3a goal alignment
                            cos(mu-hat, prompt-direction) if a prompt vector is available; 3b chain
                            deviation cos(mu-hat_j, prefix mean direction). Increment over [R-bar(+len)].
  P4  high-dim null        : empirical R-bar null by token count (concentration of measure: E[R^2]=1/n).

Decision: R-bar saturates the CONCENTRATION axis (sufficiency); if orientation (3a/3b) adds increment
over R-bar, it is the principled second axis (the mu-hat half). Strict EDIS+entropy baseline is a
follow-up join (see detector_shape.py); here orientation is tested over [R-bar + length] only.

Needs _cloud.npz: respcloud (clouds_stored) + cloud_store_layers + step_token_ranges + gold_error_step.
Optional: a prompt/question direction (qvec) for P3a -- auto-detected, else 3b is used.
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


def abscorr(a, b):
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 5 or a[m].std() < 1e-12 or b[m].std() < 1e-12:
        return float("nan")
    return abs(float(np.corrcoef(a[m], b[m])[0, 1]))


def vmf_fit(U, d):
    """unit token vectors U (n x d) -> (R-bar, kappa-hat, mu-hat, tangent_PR).
    R-bar = ||mean||; kappa-hat = Banerjee approx; tangent_PR = participation ratio of the residual
    (tangent) covariance after removing mu-hat (high = isotropic = vMF-like)."""
    n = U.shape[0]
    mean = U.mean(0); R = float(np.linalg.norm(mean))
    if R < 1e-9:
        return R, 0.0, mean, np.nan
    mu = mean / R
    kappa = R * (d - R * R) / (1 - R * R + 1e-9)               # Banerjee et al. MLE approximation
    resid = U - np.outer(U @ mu, mu)                           # tangent components (remove mu)
    C = resid.T @ resid / n
    lam = np.linalg.eigvalsh(C); lam = np.clip(lam, 0, None); tot = lam.sum()
    tpr = float(tot * tot / (np.square(lam).sum() + 1e-18)) if tot > 1e-12 else np.nan
    return R, float(kappa), mu, tpr


def oof(X, y, grp, folds=5):
    X = np.column_stack(X) if isinstance(X, list) else X
    s = np.full(len(y), np.nan)
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
    QV = z["qvec"] if "qvec" in z.files else None              # optional prompt/question direction
    d = int(np.asarray(RC[next(i for i in range(len(RC)) if RC[i] is not None)]).shape[-1])

    R_, K_, TPR_, NT_, DEV_, GOAL_, Y, G = [], [], [], [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        k = int(ges[i]); correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0])
        # prompt direction for this chain (if available)
        p = None
        if QV is not None and QV[i] is not None:
            qv = np.asarray(QV[i], float).ravel()
            if qv.shape[0] == d and np.linalg.norm(qv) > 1e-9:
                p = qv / np.linalg.norm(qv)
        mus = []                                               # per-step mu-hat for chain deviation
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < 4:
                mus.append(None); continue
            H = rcl[lo:hi]; nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
            if ok.sum() < 4:
                mus.append(None); continue
            U = H[ok] / nrm[ok, None]; R, kap, mu, tpr = vmf_fit(U, d); mus.append(mu)
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            prior = [m for m in mus[:j] if m is not None]
            dev = (1.0 - float(mu @ (np.mean(prior, 0) / (np.linalg.norm(np.mean(prior, 0)) + 1e-9)))
                   if prior else np.nan)                       # chain orientation deviation (3b)
            goal = float(mu @ p) if p is not None else np.nan  # goal alignment (3a)
            R_.append(R); K_.append(kap); TPR_.append(tpr); NT_.append(int(ok.sum()))
            DEV_.append(dev); GOAL_.append(goal); Y.append(lab); G.append(i)
    R_ = np.asarray(R_); K_ = np.asarray(K_); TPR_ = np.asarray(TPR_); NT_ = np.asarray(NT_, float)
    DEV_ = np.asarray(DEV_); GOAL_ = np.asarray(GOAL_); Y = np.asarray(Y, int); G = np.asarray(G, int)
    cor, err = Y == 0, Y == 1

    print(f"file: {args.npz} | layer {args.layer} | dim {d} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n[P1 vMF fit] tangent participation ratio (high = isotropic = vMF-like):")
    print(f"  correct {np.nanmean(TPR_[cor]):.1f}   error {np.nanmean(TPR_[err]):.1f}   "
          f"(of max {d}; both high & similar => unimodal-vMF adequate, residual carries little)")
    print(f"\n[P2 concentration = kappa] AUROC (error = LOW):")
    print(f"  R-bar (resultant)   {bdir(auroc(-R_, Y)):.3f}     kappa-hat {bdir(auroc(-K_, Y)):.3f}   "
          f"(should track each other; ~validated 0.7x)")
    floor = float(np.nanmean(1.0 / np.sqrt(NT_)))
    print(f"  uniform floor 1/sqrt(n) ~ {floor:.3f}   |   mean R-bar: correct {np.nanmean(R_[cor]):.3f}  "
          f"error {np.nanmean(R_[err]):.3f}")
    print(f"  -> BOTH classes' R-bar >> uniform floor => Rayleigh(uniform vs concentrated) rejects BOTH; "
          f"the discriminator is kappa DEGREE vs a healthy baseline, not a uniformity test.")
    print(f"\n[P3 orientation = mu-hat, the untapped half]  AUROC | |corr| to R-bar | increment over [R-bar(+len)]")
    for nm, v in [("3b chain deviation", DEV_), ("3a goal align (cos to prompt)", GOAL_)]:
        if not np.isfinite(v).any():
            print(f"  {nm:30s} (unavailable -- no prompt qvec in npz)" if "goal" in nm else
                  f"  {nm:30s} (unavailable)")
            continue
        a = bdir(auroc(v, Y)); rc = abscorr(v, R_)
        inc = ""
        if HAVE_SK:
            base = bdir(auroc(oof([-R_, NT_], Y, G), Y))
            both = bdir(auroc(oof([-R_, NT_, np.nan_to_num(v, nan=np.nanmean(v))], Y, G), Y))
            inc = f"   [R-bar+len {base:.3f} -> +this {both:.3f}  ({both-base:+.3f})]"
        print(f"  {nm:30s} {a:.3f}   |r|R {rc:.3f}{inc}")
    print(f"\n[P4 high-dim null] E[R^2]=1/n => random R-bar ~ {floor:.3f}; use an EMPIRICAL null (correct "
          f"chains), not chi^2_d, since n<<d makes the Rayleigh chi^2 asymptotics unreliable.")
    print("\nread: P1 confirms (or not) that vMF is an adequate idealization (isotropic tangent). P2 is the "
          "validated CONCENTRATION axis (R-bar=kappa), framed correctly as a kappa-degree comparison vs a "
          "healthy baseline (NOT Rayleigh-vs-uniform, which both classes reject). P3 tests the ORIENTATION half "
          "mu-hat: if goal-alignment (3a) adds increment over R-bar with low |r|R, it is the principled second "
          "axis the sufficient statistic predicts -- resultant uses only ||sum x||, discarding the direction. "
          "3a needs a prompt vector (qvec); if absent, locate/extract it -- 3b (chain deviation) is the "
          "respcloud-only proxy. Strict EDIS+entropy baseline for any positive increment is the follow-up join.")


if __name__ == "__main__":
    main()
