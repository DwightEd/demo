"""Step 23: a per-trajectory ATTENTION amplifier (not a simple pooled computation).

Everything so far collapses a trajectory to one number by a SIMPLE computation -- mean /
late-mean / max of a per-step signal, then a linear read-out. That averages the failure-
relevant steps together with the many normal steps, DILUTING the signal. The favorable
signals we found are PER-STEP (manifold violation SPE, healthy deviation, participation)
and EMERGE late. What detection needs is a mechanism that, FOR EACH TRAJECTORY, AMPLIFIES
the steps where these signals fire and suppresses the rest.

This is a tiny learned attention head over the step sequence:
    per step t:  feature f_t in R^F  (SPE, healthy-deviation, participation, norm, frac)
    relevance:   e_t = a . f_t
    attention:   alpha = softmax(e)            <- learns WHICH steps to amplify
    context:     c = sum_t alpha_t f_t         <- amplified trajectory representation
    score:       logit = w . c + b
trained by BCE (within-problem GroupKFold). |a| large => sharp amplification of the peak
steps; |a|->0 => uniform = mean pooling, so the SIMPLE pooling is a special case the model
can fall back to -- any gain over mean pooling is genuine amplification.

Compared head-to-head, same per-step features and folds, against:
    mean-pool + logistic   (the simple computation, the thing to beat)
    max-pool + logistic, late-mean   (reference)
All within-problem PAIRED AUROC, healthy reference cross-fit per fold (no leakage).
"""

from __future__ import annotations

import argparse
import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    return np.array([int(x) for x in band.split(",") if x.strip()])


def within_pair_auroc(idx_groups, feats, y_inc):
    conc = 0.0; npair = 0
    for idx in idx_groups:
        inc = [feats[i] for i in idx if y_inc[i] == 1 and np.isfinite(feats[i])]
        cor = [feats[i] for i in idx if y_inc[i] == 0 and np.isfinite(feats[i])]
        if not inc or not cor:
            continue
        for a in inc:
            for b in cor:
                conc += 1.0 if a > b else (0.5 if a == b else 0.0)
        npair += len(inc) * len(cor)
    return (conc / npair if npair else float("nan")), npair


def group_folds(groups, k, seed):
    uniq = np.unique(groups); rng = np.random.default_rng(seed); rng.shuffle(uniq)
    fo = {int(g): i % k for i, g in enumerate(uniq)}
    f = np.array([fo[int(g)] for g in groups])
    return [(np.where(f != j)[0], np.where(f == j)[0]) for j in range(k)]


def healthy_ref(steps, kmax):
    H = np.vstack(steps).astype(np.float32)
    mu = H.mean(0); sd = H.std(0) + 1e-6
    Hc = (H - mu) / sd; cm = Hc.mean(0); Hc -= cm
    C = (Hc.T @ Hc) / max(1, Hc.shape[0] - 1)
    _, ev = np.linalg.eigh(C)
    B = np.ascontiguousarray(ev[:, ::-1][:, :kmax].T)
    return mu, sd, cm, B


def step_features(ps_i, mu, sd, cm, B):
    """Per-step unsupervised signal features for one trajectory: (T, F)."""
    Z = (ps_i - mu) / sd - cm                              # healthy-standardized
    tot = (Z ** 2).sum(1) + 1e-12
    ink = ((Z @ B.T) ** 2).sum(1)
    spe = (tot - ink) / tot                                # manifold violation
    mahal = np.sqrt(tot / Z.shape[1])                      # healthy deviation
    nrm = np.log(np.sqrt((ps_i ** 2).sum(1)) + 1e-6)       # raw norm
    s2 = ps_i ** 2
    pr = (s2.sum(1) ** 2) / ((s2 ** 2).sum(1) + 1e-12) / ps_i.shape[1]   # participation frac
    T = ps_i.shape[0]
    frac = (np.arange(T) / (T - 1)) if T > 1 else np.zeros(1)
    return np.stack([spe, mahal, nrm, pr, frac], axis=1)   # (T, 5)


# ---- attention amplifier (numpy forward + analytic grad, scipy L-BFGS) ----

def _unpack(theta, F):
    return theta[:F], theta[F:2 * F], theta[2 * F], theta[2 * F + 1]   # a, w, b, (unused slot)


def attn_loss_grad(theta, mats, ys, lam):
    F = mats[0].shape[1]
    a = theta[:F]; w = theta[F:2 * F]; b = theta[2 * F]
    ga = np.zeros(F); gw = np.zeros(F); gb = 0.0; loss = 0.0
    for M, y in zip(mats, ys):
        e = M @ a
        e -= e.max()
        ex = np.exp(e); al = ex / ex.sum()                 # softmax attention
        c = M.T @ al                                       # context (F,)
        logit = w @ c + b
        p = 1.0 / (1.0 + np.exp(-logit))
        loss += -(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))
        dlogit = p - y
        gw += dlogit * c; gb += dlogit
        gc = dlogit * w
        g_al = M @ gc                                       # dL/dalpha (T,)
        g_e = al * (g_al - (al @ g_al))                     # softmax jacobian
        ga += M.T @ g_e
    n = len(mats)
    loss = loss / n + lam * (a @ a + w @ w)
    grad = np.concatenate([ga / n + 2 * lam * a, gw / n + 2 * lam * w,
                           [gb / n], [0.0]])
    return loss, grad


def fit_attn(mats, ys, lam, seed, restarts=2):
    F = mats[0].shape[1]
    best = None
    for r in range(restarts):
        rng = np.random.default_rng(seed * 17 + r)
        th0 = rng.standard_normal(2 * F + 2) * 0.1
        res = minimize(attn_loss_grad, th0, args=(mats, ys, lam), jac=True,
                       method="L-BFGS-B", options={"maxiter": 200})
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def predict_attn(theta, mats):
    F = mats[0].shape[1]
    a = theta[:F]; w = theta[F:2 * F]; b = theta[2 * F]
    out = np.empty(len(mats))
    for i, M in enumerate(mats):
        e = M @ a; e -= e.max(); al = np.exp(e); al /= al.sum()
        out[i] = w @ (M.T @ al) + b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="mid")
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--lam", type=float, default=0.01)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="data/amplifier.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)

    ps = [None] * N
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            ps[i] = np.nanmean(V, axis=1)
    valid = np.array([ps[i] is not None and np.isfinite(ps[i]).all() for i in range(N)])

    prob = {}
    for i, p in enumerate(problem_ids):
        prob.setdefault(int(p), []).append(i)
    idx_groups = [np.array(v) for v in prob.values() if any(y[v] == 1) and any(y[v] == 0)]

    print(f"Loaded {N} trajectories over {len(prob)} problems ({len(idx_groups)} "
          f"contrastive); d={d}, band={args.layer_band}, k={args.k}, F=5 step features")

    oof = {m: np.full(N, np.nan) for m in ["attn", "mean", "max", "late"]}
    for s in range(args.n_seeds):
        for tr, te in group_folds(problem_ids, args.kfold, args.seed + s):
            heal = [ps[i] for i in tr if valid[i] and y[i] == 0]
            if len(heal) < 5:
                continue
            mu, sd, cm, B = healthy_ref(heal, min(args.k, d))
            feats = {i: step_features(ps[i], mu, sd, cm, B) for i in range(N) if valid[i]}
            # standardize features on TRAIN steps
            alltr = np.vstack([feats[i] for i in tr if i in feats])
            fsc = StandardScaler().fit(alltr)
            F = {i: fsc.transform(feats[i]) for i in feats}

            tr_v = [i for i in tr if i in F]; te_v = [i for i in te if i in F]
            if len(np.unique(y[tr_v])) < 2:
                continue
            # ---- attention amplifier ----
            theta = fit_attn([F[i] for i in tr_v], y[tr_v].astype(float),
                             args.lam, args.seed + s)
            for i, lg in zip(te_v, predict_attn(theta, [F[i] for i in te_v])):
                oof["attn"][i] = lg
            # ---- simple poolings on the SAME features ----
            for name, pool in [("mean", lambda M: M.mean(0)),
                               ("max", lambda M: M.max(0)),
                               ("late", lambda M: M[max(0, int(M.shape[0] * 0.6)):].mean(0))]:
                Xtr = np.vstack([pool(F[i]) for i in tr_v])
                Xte = np.vstack([pool(F[i]) for i in te_v])
                clf = LogisticRegression(C=1.0, max_iter=2000,
                                         class_weight="balanced").fit(Xtr, y[tr_v])
                pr = clf.decision_function(Xte)
                for i, v in zip(te_v, pr):
                    oof[name][i] = v

    print(f"\n=== per-trajectory amplifier vs simple pooling (within-problem PAIRED AUROC) ===")
    res = {}
    for m in ["mean", "late", "max", "attn"]:
        a = within_pair_auroc(idx_groups, oof[m], y)[0]
        res[m] = max(a, 1 - a) if np.isfinite(a) else float("nan")
        tag = {"mean": "  <- simple baseline", "attn": "  <- learned amplifier"}.get(m, "")
        print(f"  {m:5s} {res[m]:.4f}{tag}")
    print(f"\n  amplifier - mean-pool = {res['attn'] - res['mean']:+.4f}")
    if res["attn"] > max(res["mean"], res["late"], res["max"]) + 0.01:
        print("  -> attention AMPLIFIES: learning which steps to amplify beats pooling "
              "every step equally.")
    else:
        print("  -> no gain over pooling: the signal is spread across steps (no peak to "
              "amplify) / linear pooling already captures it.")

    np.savez(args.output, methods=np.array(list(res.keys()), dtype=object),
             within=np.array([res[m] for m in res]), band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
