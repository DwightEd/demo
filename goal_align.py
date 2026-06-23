"""Goal alignment: the ORIENTATION half of the vMF sufficient statistic (mu-hat), which resultant
(the concentration half, ||sum x||) discards. Does an error step's pooled direction DRIFT away from
the question/prompt direction?

For each step: g = cos( stepvec_j , qvec ) at the mid layer, where stepvec_j is the exp-pooled step
vector and qvec is the exp-pooled prompt-token vector (both full-dim, same pooling). The shared
anisotropy (massive activations) makes everything point the same way, so we remove a GLOBAL common
direction c first, then take the cosine of the residuals (the lesson from manifold-departure).

Decisive (and honest about the prior lesson): goal alignment must add increment over the STRICT
baseline [resultant + EDIS + length + entropy], not just over resultant -- spectral shape died there.
If g adds there with low |corr| to resultant, the orientation axis (mu-hat) is the principled second
axis the sufficient statistic predicts.

Needs ONE npz extracted with --store_step_vectors (stepvec + qvec + sv_layers) AND the usual coh
fields (stepcloud resultant + tok_U_D + step_token_ranges + gold_error_step + layers_used).
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


def edis(H, w=8, tb=1.36, tr=1.33):
    H = np.asarray(H, float); H = H[np.isfinite(H)]
    if len(H) < 3:
        return 0.0
    ww = min(w, max(2, len(H) // 2))
    burst = sum(1 for t in range(len(H) - ww) if H[t + ww] - H[t] > tb) if len(H) > ww else 0
    rebound = 0; rmin = H[0]
    for t in range(1, len(H)):
        if H[t] - rmin > tr:
            rebound += 1
        rmin = min(rmin, H[t])
    return 0.5 * (burst + rebound) * (1.0 + float(H.var()))


def oof(cols, y, grp, folds=5):
    X = np.column_stack(cols); s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return bdir(auroc(s, y))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "stepvec" not in z.files or not bool(z.get("step_vectors_stored", np.array(False))):
        raise SystemExit("npz has no stepvec/qvec. Re-extract with --store_step_vectors.")
    svl = [int(x) for x in z["sv_layers"]]; si = svl.index(args.layer)
    SV = z["stepvec"]; QV = z["qvec"]
    cn = [str(x) for x in z["cloud_feature_names"]]; lyu = [int(x) for x in z["layers_used"]]
    li = lyu.index(args.layer); fi = cn.index("resultant")
    SC = z["stepcloud"]; SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"]

    # pass 1: global common direction c from a sample of step vectors
    acc = []
    for i in range(len(SV)):
        if SV[i] is None:
            continue
        sv = np.asarray(SV[i], np.float32)[:, si, :]
        for row in sv:
            if np.isfinite(row).all() and np.linalg.norm(row) > 1e-6:
                acc.append(row / np.linalg.norm(row))
        if len(acc) > 20000:
            break
    A = np.asarray(acc, np.float64)
    _, _, Vt = np.linalg.svd(A - A.mean(0), full_matrices=False); c = Vt[0]

    def resid_unit(v):
        v = v - (v @ c) * c; n = np.linalg.norm(v)
        return v / n if n > 1e-9 else None

    GOAL, RES, EDS, UNC, LEN, Y, G = [], [], [], [], [], [], []
    for i in range(len(SV)):
        if SV[i] is None or QV[i] is None:
            continue
        sv = np.asarray(SV[i], np.float32)[:, si, :]; qv = np.asarray(QV[i], np.float32)[si]
        if not np.isfinite(qv).all() or np.linalg.norm(qv) < 1e-6:
            continue
        qd = resid_unit(qv.astype(np.float64))
        if qd is None:
            continue
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i])
        correct = (k < 0); T = rng.shape[0]; a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        for j in range(min(T, sv.shape[0])):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            sd = resid_unit(sv[j].astype(np.float64))
            if sd is None:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            GOAL.append(float(sd @ qd)); RES.append(sc[j, li, fi]); EDS.append(edis(ud[lo:hi]))
            UNC.append(float(np.nanmean(ud[lo:hi]))); LEN.append(float(hi - lo))
            Y.append(lab); G.append(i)
    GOAL = np.asarray(GOAL); RES = np.asarray(RES); EDS = np.asarray(EDS); UNC = np.asarray(UNC)
    LEN = np.asarray(LEN); Y = np.asarray(Y, int); G = np.asarray(G, int)

    print(f"file: {args.npz} | layer {args.layer} | steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\ngoal align cos(step_dir, question_dir), common-component removed:")
    print(f"  mean cos: correct {np.nanmean(GOAL[Y==0]):.3f}  error {np.nanmean(GOAL[Y==1]):.3f}   "
          f"(lower for error = drifts off the question direction)")
    print(f"  AUROC {bdir(auroc(GOAL, Y)):.3f}   |corr| to resultant {abscorr(GOAL, RES):.3f}")
    if HAVE_SK:
        gbad = -RES; ucol = UNC
        b1 = oof([gbad, LEN], Y, G); b1g = oof([gbad, LEN, GOAL], Y, G)
        b2 = oof([gbad, EDS, LEN, ucol], Y, G); b2g = oof([gbad, EDS, LEN, ucol, GOAL], Y, G)
        print(f"\n  increment of goal-align:")
        print(f"    over [resultant + length]              {b1:.3f} -> {b1g:.3f}  ({b1g-b1:+.3f})")
        print(f"    over [resultant + EDIS + length + entropy] {b2:.3f} -> {b2g:.3f}  ({b2g-b2:+.3f})  <- STRICT")
    print("\nread: the ORIENTATION axis (mu-hat) the sufficient statistic predicts. If goal-align adds a "
          "positive increment over the STRICT [resultant+EDIS+length+entropy] baseline with low |corr| to "
          "resultant, then 'the step drifts off the question direction' is a real second axis, orthogonal to "
          "concentration -- the mu-hat half resultant discards. If the strict increment ~0 (like spectral "
          "shape), orientation is already covered, and the geometric signal is concentration alone, full stop. "
          "Check the raw mean cos gap (correct vs error) for the mechanism direction.")


if __name__ == "__main__":
    main()
