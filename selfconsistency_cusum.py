"""Self-consistency detector: health = is this step a natural continuation of THIS chain's
own history, not 'does it look like a typical correct chain'.

For each chain we predict y_t CAUSALLY from its own past y_{0..t-1} and take the standardized
residual. This is within-chain by construction (no cross-chain absolute level) and mode-free
(each chain is its own baseline -> the multimodality of correct reasoning is divided out).

  s_run_t = (y_t - mean_{<t}) / std_{<t}                        running-stat self-consistency
  s_ar_t  = (y_t - [mean_{<t} + a (y_{t-1}-mean_{<t})]) / std_{<t}   + AR(1) memory (a from correct)

Signed: error = resultant DROP -> residual NEGATIVE. Detection score = -s (error positive).

Three questions, answered per config (run each coh.npz separately = difficulty stratification):
  (a) does the causal within-chain residual carry signal? -> step AUROC vs raw resultant
  (b) is going-off-track a UNIFIED pattern or difficulty-dependent? -> event study shape
      (synchronous spike at the error vs precursor rise before it), per config
  detector: reset CUSUM W_t=max(0,W_{t-1}+score_t-k) with a CONFORMAL threshold from held-out
      correct chains -> FPR-guaranteed recall + detection delay (no parametric null assumed).

Needs coh.npz: stepcloud(resultant) + gold_error_step + step_token_ranges (+ layers_used,
cloud_feature_names). No respcloud, no black box -- runs on all four configs.
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
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1)
    num = den = 0.0
    for bb in range(nb):
        m = b == bb
        a = bdir(auroc(s[m], y[m])); ne, ng = int(y[m].sum()), int((y[m] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def causal_resid(y, a, eps=1e-6):
    """signed causal residuals (nan for t<2). returns s_run, s_ar."""
    T = len(y); s_run = np.full(T, np.nan); s_ar = np.full(T, np.nan)
    for t in range(2, T):
        hist = y[:t]; mu = hist.mean(); sd = hist.std() + eps
        s_run[t] = (y[t] - mu) / sd
        s_ar[t] = (y[t] - (mu + a * (y[t - 1] - mu))) / sd
    return s_run, s_ar


def cusum(score, kref):
    """reset CUSUM W_t = max(0, W_{t-1} + score_t - kref). nan scores skipped."""
    W = 0.0; out = np.zeros(len(score))
    for t in range(len(score)):
        if np.isfinite(score[t]):
            W = max(0.0, W + score[t] - kref)
        out[t] = W
    return out


def est_ar(ys):
    """global lag-1 autocorr on correct-chain y series (centered per chain)."""
    num = den = 0.0
    for y in ys:
        d = y - y.mean()
        if len(d) >= 2:
            num += np.sum(d[1:] * d[:-1]); den += np.sum(d[:-1] ** 2)
    return float(np.clip(num / den, -0.95, 0.95)) if den > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kref", type=float, default=0.5)
    ap.add_argument("--alpha", type=float, default=0.05)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    fi = cnames.index("resultant")

    chains = []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int)
        y = sc[:, li, fi]; k = int(ges[i]); T = len(y)
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(min(T, rng.shape[0]))], float)
        if np.isfinite(y).all() and T >= 3 and len(nt) == T:
            chains.append({"y": y, "k": k, "nt": nt, "correct": k < 0})
    a = est_ar([c["y"] for c in chains if c["correct"]])
    print(f"file: {args.npz} | layer {args.layer} | chains {len(chains)} "
          f"(correct {sum(c['correct'] for c in chains)}) | global AR a={a:.3f}")

    # per-step residuals + labels
    RUN, AR, RAW, Y, NT = [], [], [], [], []
    EOFF, ES = [], []                                   # event study: signed s_run vs offset
    for c in chains:
        y = c["y"]; k = c["k"]; correct = c["correct"]; T = len(y)
        sr, sa = causal_resid(y, a)
        for j in range(T):
            if not correct:
                EOFF.append(j - k); ES.append(sr[j])
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            if not np.isfinite(sr[j]):          # j<2: no causal history -> fair same-step compare
                continue
            RUN.append(sr[j]); AR.append(sa[j]); RAW.append(y[j]); Y.append(lab); NT.append(c["nt"][j])
    RUN = np.asarray(RUN); AR = np.asarray(AR); RAW = np.asarray(RAW)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float)
    EOFF = np.asarray(EOFF, int); ES = np.asarray(ES, float)

    print(f"\n(a) step-level AUROC (pooled / length-bucket) -- causal within-chain vs raw")
    for nm, v in [("raw resultant", RAW), ("-s_run (causal within-z)", -RUN), ("-s_ar (causal AR)", -AR)]:
        print(f"  {nm:26s} {bdir(auroc(v, Y)):.3f} / {bucket(v, Y, NT):.3f}")

    print(f"\n(b) event study: mean signed s_run by offset from first error (Δ=0)")
    print(f"  {'Δ=j-k':>6s} {'n':>5s} {'mean s_run':>11s} {'SE':>7s}")
    for dd in range(-4, 4):
        m = (EOFF == dd) & np.isfinite(ES)
        if m.sum() >= 5:
            star = " <-- error" if dd == 0 else ""
            print(f"  {dd:>6d} {int(m.sum()):>5d} {ES[m].mean():>+11.3f} {ES[m].std()/np.sqrt(m.sum()):>7.3f}{star}")
    pre = ES[(EOFF <= -3) & np.isfinite(ES)]; at0 = ES[(EOFF == 0) & np.isfinite(ES)]
    if len(pre) >= 5 and len(at0) >= 5:
        d = at0.mean() - pre.mean(); se = np.sqrt(at0.std()**2/len(at0) + pre.std()**2/len(pre))
        shape = "SYNCHRONOUS dip at error" if abs(d) - 2*se > 0 else "no clear sync dip"
        print(f"  drop s_run(0)-s_run(≤-3) = {d:+.3f} [{d-2*se:+.3f},{d+2*se:+.3f}] -> {shape}")

    # detector: reset CUSUM on score=-s_run, conformal threshold from held-out CORRECT chains
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(chains)); grp = idx
    rec = fpr = dly = 0.0; nerr = ncorr = ndly = 0
    for tr, te in GroupKFold(args.folds).split(idx, idx, grp):
        cal = [t for t in tr if chains[t]["correct"]]
        if len(cal) < 20:
            continue
        maxW_cal = []
        for t in cal:
            sr, _ = causal_resid(chains[t]["y"], a)
            maxW_cal.append(cusum(-sr, args.kref).max())
        h = np.quantile(maxW_cal, 1 - args.alpha)
        for t in te:
            c = chains[t]; sr, _ = causal_resid(c["y"], a); W = cusum(-sr, args.kref)
            flagged = W.max() >= h
            if c["correct"]:
                ncorr += 1; fpr += int(flagged)
            else:
                nerr += 1; rec += int(flagged)
                if flagged:
                    tflag = int(np.argmax(W >= h)); ndly += 1; dly += (tflag - c["k"])
    print(f"\n(detector) reset CUSUM(-s_run, k={args.kref}) + conformal h @ FPR≤{args.alpha}:")
    print(f"  recall {rec/max(nerr,1):.3f} ({int(rec)}/{nerr})  |  empirical FPR {fpr/max(ncorr,1):.3f} "
          f"({int(fpr)}/{ncorr})  |  mean delay {dly/max(ndly,1):+.2f} steps from first error")
    print("\nread: (a) -s_run ~ within-chain ceiling (expect ~0.6-0.7, below pooled raw which keeps "
          "between-chain difficulty/mode). (b) SYNCHRONOUS dip vs precursor tells whether off-track "
          "is detectable the same way across configs -- run all 4 and compare shapes; difficulty-"
          "dependent shape is itself the finding. detector recall is at a GUARANTEED FPR; negative "
          "delay = caught before the labeled error step (precursor), ~0 = synchronous.")


if __name__ == "__main__":
    main()
