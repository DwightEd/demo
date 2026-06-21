"""CIM information-volume V and intrinsic-dimension proxy as NON-collinear geometric channels.

resultant = within-STEP token concentration. V (CIM Eq.14) = volume of the across-STEP
TRAJECTORY: V = 0.5 logdet(I + (d/n) Z Z^T), Z = step-vector trajectory matrix. Different
object -> potentially a new axis. RISK: broad-spectrum shrinkage lowers both magnitude and V,
so raw V may just track norm. Fix: compute V on UNIT step vectors (directional trajectory
volume), immune to magnitude -- the candidate orthogonal axis. We also compute an effective-
dimension proxy (participation ratio of the trajectory covariance; robust for few points,
unlike TLE/k-NN ID which is unreliable on 5-20 step chains).

Three tests:
 (A) step-level: windowed V_t / effdim_t -> within-chain causal residual -> AUROC vs resultant,
     and corr(V_t-resid, resultant) -- collinear or not.
 (B) CEILING (decisive): add V_t-resid + effdim_t-resid to the within-chain causal ceiling probe;
     does it lift the ceiling above the resultant-only ceiling? (the real "is V a new signal" test)
 (C) chain-level (ID,V) admissible region: per-chain full-trajectory V & effdim, predict correct/
     error. CIM predicts correct = low-ID + high-V cluster, error = bilateral (high-ID diffuse OR
     low-V collapse). Tests single V, |V-healthy_median| (bilateral), effdim, and the 2D joint --
     if joint >> marginals, hallucination = bilateral departure from the admissible regime.

Needs _cloud.npz: respcloud (step tokens) + cloud_store_layers + stepcloud(resultant) +
is_correct + gold_error_step + step_token_ranges + problem_ids.
"""

from __future__ import annotations
import argparse
import numpy as np

try:
    from sklearn.model_selection import GroupKFold
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.ensemble import GradientBoostingRegressor
except ImportError:
    raise SystemExit("needs scikit-learn")


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
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb; a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def exp_w(n):
    if n <= 1:
        return np.ones(max(n, 1))
    e = np.exp(np.arange(n) / (n - 1)); return e / e.sum()


def step_vec(tok):
    """exp-pooled step vector (raw) and its unit direction. tok: (n,d)."""
    nrm = np.linalg.norm(tok, axis=1); ok = nrm > 1e-9
    if ok.sum() < 1:
        return None, None
    t = tok[ok]; w = exp_w(t.shape[0]); raw = (w[:, None] * t).sum(0)
    r = np.linalg.norm(raw)
    return raw, (raw / r if r > 1e-9 else None)


def vol(Z, d):
    """V = 0.5 logdet(I + (d/n) Z Z^T) over the n x n Gram (n = #step vectors in window)."""
    n = Z.shape[0]
    if n < 2:
        return np.nan
    G = Z @ Z.T; M = np.eye(n) + (d / n) * G
    s, ld = np.linalg.slogdet(M)
    return 0.5 * ld if s > 0 else np.nan


def effdim(Z):
    """participation ratio of the centred trajectory covariance (robust ID proxy)."""
    n = Z.shape[0]
    if n < 2:
        return np.nan
    Zc = Z - Z.mean(0); G = Zc @ Zc.T
    lam = np.clip(np.linalg.eigvalsh(G), 0, None); s = lam.sum()
    return float((s * s) / np.sum(lam * lam)) if s > 1e-12 else np.nan


def causal_resid(y, sd_floor, clip=5.0, eps=1e-6):
    T = len(y); s = np.full(T, np.nan)
    fin = np.where(np.isfinite(y))[0]
    for t in range(T):
        past = [y[k] for k in fin if k < t]
        if len(past) >= 2 and np.isfinite(y[t]):
            h = np.array(past); mu = h.mean(); sd = max(h.std(), sd_floor) + eps
            s[t] = np.clip((y[t] - mu) / sd, -clip, clip)
    return s


def oof_logit(X, y, grp, folds):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def oof_gbm(X, y, grp, folds):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = GradientBoostingClassifier(n_estimators=150, max_depth=3, random_state=0)
        clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--win", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    if "respcloud" not in z.files or not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("need _cloud.npz with stored respcloud")
    csl = [int(x) for x in z["cloud_store_layers"]]; cli = csl.index(args.layer)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else None
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    isc = z["is_correct"].astype(int)
    pid = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(RC))
    SC = z["stepcloud"] if "stepcloud" in z.files else None
    use_res = SC is not None and lyu is not None and args.layer in lyu and "resultant" in cnames
    lic = lyu.index(args.layer) if use_res else None
    fi = cnames.index("resultant") if "resultant" in cnames else None
    d = int(z["cloud_proj_dim"]) if "cloud_proj_dim" in z.files else None

    # per chain: step-vector trajectory (raw + unit), per-step windowed V_dir/V_raw/effdim, resultant
    chains = []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rcl = np.asarray(RC[i], np.float32)[:, cli, :]; rng = np.asarray(SR[i], int)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        if d is None:
            d = rcl.shape[1]
        raws, units = [], []
        for jstep in range(T):
            lo = max(0, int(rng[jstep, 0]) - a0); hi = min(rcl.shape[0], int(rng[jstep, 1]) - a0 + 1)
            rv, uv = step_vec(rcl[lo:hi]) if hi > lo else (None, None)
            raws.append(rv); units.append(uv)
        Vd = np.full(T, np.nan); Vr = np.full(T, np.nan); ED = np.full(T, np.nan)
        for t in range(T):
            w0 = max(0, t - args.win + 1)
            Ru = [units[k] for k in range(w0, t + 1) if units[k] is not None]
            Rr = [raws[k] for k in range(w0, t + 1) if raws[k] is not None]
            if len(Ru) >= 2:
                Zu = np.stack(Ru); Vd[t] = vol(Zu, d); ED[t] = effdim(Zu)
            if len(Rr) >= 2:
                Vr[t] = vol(np.stack(Rr), d)
        res = (np.asarray(SC[i], float)[:, lic, fi] if use_res else np.full(T, np.nan))
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)
        # full-trajectory (chain-level) volumes on unit / raw step vectors
        allu = [u for u in units if u is not None]; allr = [r for r in raws if r is not None]
        Vfull = vol(np.stack(allu), d) if len(allu) >= 2 else np.nan
        EDfull = effdim(np.stack(allu)) if len(allu) >= 2 else np.nan
        chains.append(dict(Vd=Vd, Vr=Vr, ED=ED, res=res, nt=nt, k=int(ges[i]), T=T,
                           correct=int(ges[i]) < 0, pid=int(pid[i]), isc=int(isc[i]),
                           Vfull=Vfull, EDfull=EDfull,
                           resmean=np.nanmean(res) if np.isfinite(res).any() else np.nan))

    def sdfloor(key):
        vals = []
        for c in chains:
            v = c[key][np.isfinite(c[key])]
            if len(v) >= 2:
                vals.append(v.std())
        return 0.5 * float(np.median(vals)) if vals else 1.0
    sf = {k: sdfloor(k) for k in ["Vd", "Vr", "ED", "res"]}
    print(f"file: {args.npz} | layer {args.layer} | win {args.win} | d {d} | chains {len(chains)}")

    # ---- (A) step-level: within-chain causal residual of each channel, labels = first-error ----
    RES, RVD, RVR, RED, Y, NT, G = [], [], [], [], [], [], []
    for ci, c in enumerate(chains):
        rr = causal_resid(c["res"], sf["res"]); rvd = causal_resid(c["Vd"], sf["Vd"])
        rvr = causal_resid(c["Vr"], sf["Vr"]); red = causal_resid(c["ED"], sf["ED"])
        for j in range(c["T"]):
            if c["correct"] or j < c["k"]:
                lab = 0
            elif j == c["k"]:
                lab = 1
            else:
                continue
            if not np.isfinite(rvd[j]):
                continue
            RES.append(rr[j]); RVD.append(rvd[j]); RVR.append(rvr[j]); RED.append(red[j])
            Y.append(lab); NT.append(c["nt"][j]); G.append(ci)
    RES = np.asarray(RES); RVD = np.asarray(RVD); RVR = np.asarray(RVR); RED = np.asarray(RED)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)
    for a in (RES, RVD, RVR, RED):
        a[~np.isfinite(a)] = 0.0

    print(f"\n(A) step-level AUROC (within-chain causal residual) | labeled steps {len(Y)} first-error {int(Y.sum())}")
    for nm, v in [("resultant (baseline)", -RES), ("V_dir (unit-traj vol)", RVD),
                  ("V_raw (raw-traj vol)", RVR), ("effdim (ID proxy)", RED)]:
        cc = np.corrcoef(v, -RES)[0, 1]
        print(f"  {nm:24s} AUROC {bdir(auroc(v, Y)):.3f}  bucket {bucket(v, Y, NT):.3f}  corr(.,res) {cc:+.2f}")

    # ---- (B) ceiling probe: does V/effdim add over resultant within-chain? ----
    base = np.c_[RES]; addV = np.c_[RES, RVD, RVR, RED]
    sa = oof_logit(base, Y, G, args.folds); sb = oof_logit(addV, Y, G, args.folds)
    print(f"\n(B) within-chain CEILING probe (leak-free grouped logistic):")
    print(f"  resultant only           AUROC {bdir(auroc(sa, Y)):.3f}")
    print(f"  + V_dir + V_raw + effdim AUROC {bdir(auroc(sb, Y)):.3f}   (lift = decisive: is CIM volume a NEW axis?)")

    # ---- (C) chain-level (ID, V) admissible region ----
    Vf = np.array([c["Vfull"] for c in chains]); Ef = np.array([c["EDfull"] for c in chains])
    Rm = np.array([c["resmean"] for c in chains]); yc = np.array([1 - c["isc"] for c in chains], int)
    gc = np.array([c["pid"] for c in chains]); m = np.isfinite(Vf) & np.isfinite(Ef)
    Vf, Ef, Rm, yc, gc = Vf[m], Ef[m], Rm[m], yc[m], gc[m]
    Rm[~np.isfinite(Rm)] = np.nanmean(Rm[np.isfinite(Rm)])
    hV = np.median(Vf[yc == 0]); hE = np.median(Ef[yc == 0])           # healthy centre
    print(f"\n(C) chain-level (ID,V) admissible-region | chains {len(yc)} incorrect {int(yc.sum())}")
    for nm, s in [("V_dir (full traj)", Vf), ("|V_dir - healthy| (bilateral)", np.abs(Vf - hV)),
                  ("effdim (full traj)", Ef), ("|effdim - healthy| (bilateral)", np.abs(Ef - hE)),
                  ("resmean (ref)", Rm)]:
        print(f"  {nm:32s} AUROC {bdir(auroc(s, yc)):.3f}")
    jl = oof_logit(np.c_[Vf, Ef], yc, gc, args.folds)
    jg = oof_gbm(np.c_[Vf, Ef], yc, gc, args.folds)
    jgr = oof_gbm(np.c_[Vf, Ef, Rm], yc, gc, args.folds)
    print(f"  joint [V,effdim] logistic        AUROC {bdir(auroc(jl, yc)):.3f}")
    print(f"  joint [V,effdim] grad-boost      AUROC {bdir(auroc(jg, yc)):.3f}   (>> marginals = bilateral 2D regime)")
    print(f"  joint [V,effdim,resmean] gbm     AUROC {bdir(auroc(jgr, yc)):.3f}")

    print("\nread: (A) corr(V_dir,res) low => non-collinear. (B) is the decisive test: if the ceiling "
          "rises clearly above resultant-only, CIM volume is a NEW within-chain axis worth pursuing; "
          "if flat with high corr, V = the same shrinkage signal. (C) if joint [V,effdim] (esp gbm, or "
          "the |.-healthy| bilateral features) >> each marginal, hallucination = bilateral departure "
          "from the admissible low-ID/high-V regime -- explains why single monotone ID/V detectors underfit.")


if __name__ == "__main__":
    main()
