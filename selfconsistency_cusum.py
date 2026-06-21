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


def causal_resid(y, a, sd_floor, base_scale, clip=5.0, eps=1e-6):
    """signed causal residuals (nan for t<2). returns s_run, s_ar, s_fix.

    s_run/s_ar use an ADAPTIVE baseline (running mean up to t) -> any slow drift is absorbed
    into the baseline, leaving only the error-step jump. s_fix uses a FIXED baseline (mean of
    the first 2 steps, never updated) scaled by a fixed base_scale -> slow drift ACCUMULATES
    and shows as a pre-error ramp if a precursor exists. Comparing the two event studies tests
    whether 'precursor' is frame-dependent (visible vs a fixed baseline, absorbed by adaptive).
    std floored / winsorized to kill the early-step near-zero-std explosion."""
    T = len(y); s_run = np.full(T, np.nan); s_ar = np.full(T, np.nan); s_fix = np.full(T, np.nan)
    base = y[:2].mean()                                    # fixed anchor = chain's first 2 steps
    for t in range(2, T):
        hist = y[:t]; mu = hist.mean(); sd = max(hist.std(), sd_floor) + eps
        s_run[t] = np.clip((y[t] - mu) / sd, -clip, clip)
        s_ar[t] = np.clip((y[t] - (mu + a * (y[t - 1] - mu))) / sd, -clip, clip)
        s_fix[t] = np.clip((y[t] - base) / base_scale, -clip, clip)
    return s_run, s_ar, s_fix


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
    # std floor = half the typical within-chain std (median over correct chains) -> stops the
    # early-step near-zero-std residual explosion that wrecked the event study.
    sd_floor = 0.5 * float(np.median([c["y"].std() for c in chains if c["correct"] and len(c["y"]) >= 2]))
    base_scale = 2 * sd_floor                              # fixed scale for s_fix (~median chain std)
    print(f"file: {args.npz} | layer {args.layer} | chains {len(chains)} "
          f"(correct {sum(c['correct'] for c in chains)}) | global AR a={a:.3f} | sd_floor {sd_floor:.4f}")

    # per-step residuals + labels (+ causal feature vector for the within-chain ceiling probe)
    RUN, AR, RAW, Y, NT, G = [], [], [], [], [], []
    FEATC = []                                          # leak-free CAUSAL features (uses only y[:t], y[t])
    EOFF, ESr, ESa, ESf = [], [], [], []                # event study: s_run, s_ar, s_fix vs offset
    for ci, c in enumerate(chains):
        y = c["y"]; k = c["k"]; correct = c["correct"]; T = len(y)
        sr, sa, sf = causal_resid(y, a, sd_floor, base_scale)
        for j in range(T):
            if not correct:
                EOFF.append(j - k); ESr.append(sr[j]); ESa.append(sa[j]); ESf.append(sf[j])
            if correct or j < k:
                lab = 0
            elif j == k:
                lab = 1
            else:
                continue
            if not np.isfinite(sr[j]):          # j<2: no causal history -> fair same-step compare
                continue
            hist = y[:j]; mu = hist.mean()
            RUN.append(sr[j]); AR.append(sa[j]); RAW.append(y[j]); Y.append(lab)
            NT.append(c["nt"][j]); G.append(ci)
            FEATC.append([sr[j], sa[j], y[j] - mu, y[j] - y[j - 1], j / max(1, T - 1), hist.std()])
    RUN = np.asarray(RUN); AR = np.asarray(AR); RAW = np.asarray(RAW)
    Y = np.asarray(Y, int); NT = np.asarray(NT, float); G = np.asarray(G, int)
    FEATC = np.asarray(FEATC, float)
    EOFF = np.asarray(EOFF, int); ESr = np.asarray(ESr, float); ESa = np.asarray(ESa, float)
    ESf = np.asarray(ESf, float)

    # within-chain causal CEILING: leak-free grouped logistic on causal features. tells whether
    # running-stat's ~0.68 is the within-chain limit or just a weak predictor (interpretable, no
    # black box). if ceiling ~ -s_run -> running-stat is enough; if >> -> more causal signal exists.
    ceil = np.full(len(Y), np.nan)
    try:
        from sklearn.model_selection import GroupKFold
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        for tr, te in GroupKFold(args.folds).split(FEATC, Y, G):
            if len(np.unique(Y[tr])) < 2:
                continue
            sc = StandardScaler().fit(FEATC[tr])
            lr = LogisticRegression(max_iter=1000, class_weight="balanced").fit(sc.transform(FEATC[tr]), Y[tr])
            ceil[te] = lr.decision_function(sc.transform(FEATC[te]))
    except Exception as e:
        print(f"  [ceiling probe skipped: {e}]")

    print(f"\n(a) step-level AUROC (pooled / length-bucket) -- causal within-chain vs raw & ceiling")
    for nm, v in [("raw resultant (pooled)", RAW), ("-s_run (causal within-z)", -RUN),
                  ("-s_ar (causal AR)", -AR), ("causal probe (WITHIN ceiling)", ceil)]:
        print(f"  {nm:30s} {bdir(auroc(v, Y)):.3f} / {bucket(v, Y, NT):.3f}")

    print(f"\n(b) event study: mean signed residual by offset (Δ=0) -- ADAPTIVE (s_run) vs FIXED (s_fix) baseline")
    print(f"  {'Δ=j-k':>6s} {'n':>5s} {'s_run(adapt)':>13s} {'SE':>7s} {'s_fix(fixed)':>13s} {'SE':>7s}")
    for dd in range(-4, 4):
        m = (EOFF == dd) & np.isfinite(ESr); mf = (EOFF == dd) & np.isfinite(ESf)
        if m.sum() >= 5:
            star = " <-- error" if dd == 0 else ""
            print(f"  {dd:>6d} {int(m.sum()):>5d} {ESr[m].mean():>+13.3f} {ESr[m].std()/np.sqrt(m.sum()):>7.3f} "
                  f"{ESf[mf].mean():>+13.3f} {ESf[mf].std()/np.sqrt(max(mf.sum(),1)):>7.3f}{star}")
    # synchronous dip (drop at 0 vs >=3 before) for each; precursor (already drifting at -1,-2 vs <=-4)
    for tag, ES in [("s_run", ESr), ("s_ar", ESa), ("s_fix", ESf)]:
        pre = ES[(EOFF <= -3) & np.isfinite(ES)]; at0 = ES[(EOFF == 0) & np.isfinite(ES)]
        if len(pre) >= 5 and len(at0) >= 5:
            d = at0.mean() - pre.mean(); se = np.sqrt(at0.std()**2/len(at0) + pre.std()**2/len(pre))
            shape = "SYNCHRONOUS dip" if abs(d) - 2*se > 0 else "no clear sync dip"
            print(f"  drop {tag}(0)-{tag}(≤-3) = {d:+.3f} [{d-2*se:+.3f},{d+2*se:+.3f}] -> {shape}")
    near = ESf[((EOFF == -1) | (EOFF == -2)) & np.isfinite(ESf)]; far = ESf[(EOFF <= -4) & np.isfinite(ESf)]
    if len(near) >= 5 and len(far) >= 5:
        d = near.mean() - far.mean(); se = np.sqrt(near.std()**2/len(near) + far.std()**2/len(far))
        pc = "PRECURSOR drift (fixed baseline)" if abs(d) - 2*se > 0 else "no precursor"
        print(f"  s_fix precursor: s_fix(-1,-2)-s_fix(≤-4) = {d:+.3f} [{d-2*se:+.3f},{d+2*se:+.3f}] -> {pc}")

    # detectors (conformal threshold from held-out CORRECT chains): CUSUM (for sustained shifts)
    # vs MAX single-step -s_run (matched to a transient dip). The event study says the break is
    # transient -> the max detector should beat CUSUM at the same guaranteed FPR.
    from sklearn.model_selection import GroupKFold
    idx = np.arange(len(chains)); grp = idx
    R = {"cusum": dict(rec=0., fpr=0., dly=0., nd=0), "max": dict(rec=0., fpr=0., dly=0., nd=0)}
    nerr = ncorr = 0
    for tr, te in GroupKFold(args.folds).split(idx, idx, grp):
        cal = [t for t in tr if chains[t]["correct"]]
        if len(cal) < 20:
            continue
        statc, stats = [], []
        for t in cal:
            sr, _, _ = causal_resid(chains[t]["y"], a, sd_floor, base_scale)
            statc.append(cusum(-sr, args.kref).max()); stats.append(np.nanmax(-sr))
        hc = np.quantile(statc, 1 - args.alpha); hs = np.quantile(stats, 1 - args.alpha)
        for t in te:
            c = chains[t]; sr, _, _ = causal_resid(c["y"], a, sd_floor, base_scale); W = cusum(-sr, args.kref)
            ms = -sr.copy(); ms[~np.isfinite(ms)] = -np.inf
            fc = W.max() >= hc; fs = np.nanmax(-sr) >= hs
            if c["correct"]:
                ncorr += 1; R["cusum"]["fpr"] += int(fc); R["max"]["fpr"] += int(fs)
            else:
                nerr += 1
                if fc:
                    R["cusum"]["rec"] += 1; R["cusum"]["nd"] += 1
                    R["cusum"]["dly"] += int(np.argmax(W >= hc)) - c["k"]
                if fs:
                    R["max"]["rec"] += 1; R["max"]["nd"] += 1
                    R["max"]["dly"] += int(np.argmax(ms >= hs)) - c["k"]
    print(f"\n(detectors) conformal threshold @ FPR≤{args.alpha}  (error chains {nerr}, correct {ncorr}):")
    for nm, lab in [("cusum", f"reset CUSUM(-s_run,k={args.kref})"), ("max", "max single-step -s_run")]:
        d = R[nm]
        print(f"  {lab:30s} recall {d['rec']/max(nerr,1):.3f}  FPR {d['fpr']/max(ncorr,1):.3f}  "
              f"delay {d['dly']/max(d['nd'],1):+.2f}")
    print("\nread: (a) -s_run ~ within-chain ceiling (expect ~0.6-0.7, below pooled raw which keeps "
          "between-chain difficulty/mode). (b) SYNCHRONOUS dip vs precursor tells whether off-track "
          "is detectable the same way across configs -- run all 4 and compare shapes; difficulty-"
          "dependent shape is itself the finding. detector recall is at a GUARANTEED FPR; negative "
          "delay = caught before the labeled error step (precursor), ~0 = synchronous.")


if __name__ == "__main__":
    main()
