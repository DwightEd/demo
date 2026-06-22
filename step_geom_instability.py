"""GIS = Geometric Instability Score: a MULTIPLICATIVE step-level geometry score, mirroring EDIS's
form -- NOT a logistic stack (which washes out each feature's peak and, empirically, loses to
step-EDIS on the hard configs).

EDIS = 1/2 (S_burst + S_rebound) * (1 + Var): the variance MULTIPLIES the instability count, so the
score spikes only when multiple detectors fire together, each keeping its own discriminative range.
We mirror that in geometry, with components chosen so each captures a DISTINCT failure geometry:

    GIS = 1/2 ( diffusion + multimodality ) * (1 + variance)
      diffusion     = ecdf(1 - resultant)     mean-direction collapse        (burst analog)
      multimodality = ecdf(mode feature)      directions SPLIT into peaks     (rebound analog) <- NEW
      variance      = ecdf(volume feature)    cloud dispersion                (1+Var analog)

The multimodality term is the key new physics: resultant sees only the MEAN direction, so a
bimodal direction cloud (model torn between two reasoning paths) is invisible to it; the eigenvalue
spread / effective-mode count of the direction distribution catches it, and it is length-clean
(shape of the unit-direction distribution, not token count). Each component is rank-CDF normalized
to [0,1] so it keeps its full discriminative ordering and they multiply on a common scale.

Organic (non-logistic) fusion with entropy: GIS * (1 + ecdf(step-EDIS)) -- entropy as one more
multiplicative amplifier, both signals' discriminative power preserved.

Prints the npz feature names first (no guessing), then each component's standalone AUROC so we can
verify orientation, then GIS / product-GIS / GIS*EDIS vs resultant and step-EDIS. pooled + bucket.
"""

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
    m = np.isfinite(s); s, y, nt = s[m], y[m], nt[m]
    if len(s) < 10:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne, ng = int(y[mm].sum()), int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


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


def ecdf(x):
    """empirical CDF rank in [0,1], nan-safe (nan -> 0.5)."""
    x = np.asarray(x, float); out = np.full(len(x), 0.5); m = np.isfinite(x)
    if m.sum() < 2:
        return out
    v = x[m]; o = np.argsort(v, kind="mergesort"); r = np.empty(len(v)); r[o] = np.arange(len(v))
    out[m] = (r + 0.5) / len(v); return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC = z["stepcloud"]; SG = z["stepgeom"] if "stepgeom" in z.files else None
    SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int); UD = z["tok_U_D"]
    print(f"file: {args.npz} | layer {args.layer}")
    print(f"cloud_feature_names: {cn}")
    print(f"geom_feature_names : {gn}")

    cidx = {m: ("c", k) for k, m in enumerate(cn)}
    gidx = {m: ("g", k) for k, m in enumerate(gn)}
    look = {**gidx, **cidx}  # prefer cloud if name clashes

    def role(cands):
        for m in cands:
            if m in look:
                return m
        return None
    f_res = "resultant"
    f_mode = role(["dir_lam2", "cloud_D", "pr", "e90", "e50", "ed_half"])      # multimodality / mode count
    f_vol = role(["cloud_V", "cloud_C", "ae_robust", "ae", "norm", "mean_tok_norm"])  # volume / dispersion
    print(f"selected -> diffusion=1-{f_res} | multimodality={f_mode} | variance={f_vol}\n")

    def getf(name, sc, sg):
        src, k = look[name]
        arr = sc if src == "c" else sg
        return arr[:, li, k] if (arr is not None and arr.ndim == 3) else arr[:, k]

    RES, MODE, VOL, EDS, Y, NT = [], [], [], [], [], []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); sg = np.asarray(SG[i], float) if SG is not None else None
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float)
        res = getf(f_res, sc, sg); mode = getf(f_mode, sc, sg) if f_mode else np.full(T, np.nan)
        vol = getf(f_vol, sc, sg) if f_vol else np.full(T, np.nan)
        for j in range(T):
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                continue
            RES.append(res[j]); MODE.append(mode[j]); VOL.append(vol[j]); EDS.append(edis(ud[lo:hi]))
            Y.append(lab); NT.append(int(rng[j, 1] - rng[j, 0] + 1))
    RES = np.asarray(RES); MODE = np.asarray(MODE); VOL = np.asarray(VOL); EDS = np.asarray(EDS)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)

    # components: higher = worse (error). orient by domain knowledge; verify via standalone AUROC.
    c_diff = ecdf(-RES)            # low resultant = worse
    c_mode = ecdf(MODE)           # more modes / higher spread = worse  (flip below if AUROC<0.5)
    c_vol = ecdf(VOL)             # larger cloud = worse
    if auroc(c_mode, Y) < 0.5:
        c_mode = 1 - c_mode
    if auroc(c_vol, Y) < 0.5:
        c_vol = 1 - c_vol

    GIS = 0.5 * (c_diff + c_mode) * (1.0 + c_vol)                 # EDIS-form multiplicative geometry
    GISp = c_diff * (1.0 + c_mode) * (1.0 + c_vol)               # pure-product variant
    GISe = GIS * (1.0 + ecdf(EDS))                               # organic (multiplicative) entropy fusion

    print(f"steps {len(Y)} | first-error {int(Y.sum())}")
    print(f"\n{'component (standalone)':28s} {'AUROC':>7s} {'bucket':>7s}")
    for nm, v in [("diffusion 1-resultant", c_diff), (f"multimodality {f_mode}", c_mode),
                  (f"variance {f_vol}", c_vol)]:
        print(f"  {nm:28s} {bdir(auroc(v, Y)):7.3f} {bucket(v, Y, NT):7.3f}")
    print(f"\n{'score':28s} {'AUROC':>7s} {'bucket':>7s}")
    for nm, v in [("resultant (single)", -RES), ("step-EDIS (entropy-dyn)", EDS),
                  ("GIS = 1/2(d+m)(1+v)", GIS), ("GIS product d(1+m)(1+v)", GISp),
                  ("GIS x step-EDIS", GISe)]:
        print(f"  {nm:28s} {bdir(auroc(v, Y)):7.3f} {bucket(v, Y, NT):7.3f}")
    print("\nread: GIS is MULTIPLICATIVE (EDIS-form), not a logistic stack -- each geometric component keeps "
          "its own rank-CDF discriminative range and they multiply, so the score fires on the CONJUNCTION "
          "(diffuse AND multimodal AND dispersed = the multi-peak directional collapse of a reasoning error). "
          "WIN = GIS > step-EDIS on the hard configs (omnimath/olympiad) where single resultant and the "
          "logistic ensemble both lost -- the multimodality term is what resultant is blind to. GISxEDIS is "
          "the organic fusion ceiling. Check the multimodality component's standalone AUROC + bucket: if it "
          "is length-confounded (cloud_D), pooled rises but bucket does not -- then we need the directional "
          "eigenvalue spread from the raw token cloud (respcloud) instead of a step-summary proxy.")


if __name__ == "__main__":
    main()
