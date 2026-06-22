"""BETWEEN-STEP direction coherence: a candidate THIRD orthogonal axis.

Our validated signal (within-step resultant) is a scalar, second-order, order-less, ROTATION-
invariant functional of ONE step's token cloud -- it sees 'is this step's cloud concentrated', not
'does the reasoning direction jump between steps'. A genuinely different error mode: each step is
internally coherent (resultant high every step) but the pooled DIRECTION leaps from step t to t+1
(reasoning fracture / non-sequitur). within-step resultant is blind to it; the angle between
consecutive step directions catches it. Almost surely orthogonal to within-step concentration
(within vs between) and to entropy (pure representation geometry), and it plugs straight into the
cumulative trigger (accumulate cross-step JUMP instead of within-step collapse).

Per step j (mid layer):
  d_j   = normalized exp-pooled UNIT token vectors  (the step's pooled DIRECTION, unit vector)
  R_j   = ||exp-pooled unit vectors||               (within-step resultant; our existing signal)
Common-component control (critical -- this is what killed dir_manifold): remove the GLOBAL shared
anisotropy axis c (top PCA of all step directions) from every d_j before measuring the angle.
  jump_j = 1 - cos( resid(d_{j-1}), resid(d_j) )     (direction change ENTERING step j; j>=1)
Also jump_raw (no common removal) to show the control matters.

Decisive: (1) jump AUROC > ~0.55 with low |corr| to R (orthogonal new axis), (2) increment of jump
over [R] (and it survives length + step-index). If yes -> a third modality for the multi-modal
detector; feed it to the cumulative trigger next. If jump ~0.5, between-step direction carries no
error signal beyond within-step concentration.

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


def stepdir(H):
    """ordered token states of ONE step -> (unit pooled direction d, within-step resultant R)."""
    H = np.asarray(H, np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return None, np.nan
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
    p = (w[:, None] * u).sum(0); R = float(np.linalg.norm(p))
    return (p / (R + 1e-12)), R


def oof(X, y, grp, folds=5):
    s = np.full(len(y), np.nan); X = np.atleast_2d(X.T).T if X.ndim == 1 else X
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

    # pass 1: per-step pooled directions per chain + global common axis
    chains = []
    alldir = []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]; k = int(ges[i])
        dirs, Rs = [], []
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rcl.shape[0], int(rng[j, 1]) - a0 + 1)
            d, R = (stepdir(rcl[lo:hi]) if hi - lo >= 2 else (None, np.nan))
            dirs.append(d); Rs.append(R)
            if d is not None:
                alldir.append(d)
        chains.append(dict(dirs=dirs, Rs=Rs, k=k, T=T, correct=k < 0,
                           nt=[int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)]))
    A = np.asarray(alldir)
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False)
    c = Vt[0]                                                  # global shared anisotropy axis

    def resid(d):
        return d - (d @ c) * c

    JUMP, JRAW, RES, SJ, Y, NT, G = [], [], [], [], [], [], []
    for gi, ch in enumerate(chains):
        for j in range(ch["T"]):
            if ch["correct"] or j < ch["k"]:
                lab = 0
            elif j == ch["k"]:
                lab = 1
            else:
                continue
            dj, dp = ch["dirs"][j], (ch["dirs"][j - 1] if j >= 1 else None)
            if dj is None or dp is None or not np.isfinite(ch["Rs"][j]):
                continue
            rj, rp = resid(dj), resid(dp)
            jump = 1.0 - float(rj @ rp) / (np.linalg.norm(rj) * np.linalg.norm(rp) + 1e-12)
            jraw = 1.0 - float(dj @ dp)
            JUMP.append(jump); JRAW.append(jraw); RES.append(ch["Rs"][j]); SJ.append(j)
            Y.append(lab); NT.append(ch["nt"][j]); G.append(gi)
    JUMP = np.asarray(JUMP); JRAW = np.asarray(JRAW); RES = np.asarray(RES)
    SJ = np.asarray(SJ, float); Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)

    print(f"file: {args.npz} | layer {args.layer} | labeled steps(j>=1) {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'signal':34s} {'AUROC':>7s} {'b(len)':>7s} {'b(stepj)':>8s} {'|r|R':>6s} {'|r|stepj':>8s}")
    for nm, v, base in [("within-step resultant (ours, ref)", -RES, RES),
                        ("cross-step jump (common-removed)", JUMP, JUMP),
                        ("cross-step jump (raw, no control)", JRAW, JRAW)]:
        print(f"  {nm:34s} {bdir(auroc(v, Y)):7.3f} {wbuck(v, Y, NT):7.3f} {wbuck(v, Y, SJ):8.3f} "
              f"{abscorr(base, RES):6.3f} {abscorr(base, SJ):8.3f}")
    print(f"\n  corr(jump, resultant) = {abscorr(JUMP, RES):.3f}  (LOW = orthogonal new axis)")
    if HAVE_SK:
        r_only = oof(-RES, Y, G); fused = oof(np.c_[-RES, JUMP], Y, G)
        print(f"  increment: AUROC(resultant) {bdir(auroc(r_only, Y)):.3f} -> "
              f"AUROC(resultant + jump) {bdir(auroc(fused, Y)):.3f}")
    print("\nread: DECISIVE = cross-step jump (common-removed) AUROC > ~0.55 with LOW corr to resultant, and a "
          "positive increment of [resultant+jump] over resultant. That = a THIRD orthogonal modality (between-"
          "step direction fracture) for the multi-modal detector, and it feeds the cumulative trigger. Compare "
          "common-removed vs raw: if raw ~0.5 but common-removed > 0.55, the global anisotropy was masking it "
          "(the control matters). If even common-removed ~0.5, between-step direction carries no error signal "
          "beyond within-step concentration. Watch b(stepj): jump may scale with position -- the step-index "
          "bucket is the honest read.")


if __name__ == "__main__":
    main()
