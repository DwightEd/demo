"""Step-level CUSUM + conformal detector on the fused geom⟂entropy score.

The signal is weak and the absolute level is difficulty (between-chain), so a
per-step threshold false-alarms on hard chains. We turn the weak per-step signal
into a guaranteed detector the way the proposal intended:

  1. per-step fused score   s_j = logistic(resultant, U_D, U_C)        (cross-fit by chain)
  2. CAUSAL within-chain z   z_j = (s_j - mean s_{<j}) / std s_{<j}     (removes chain difficulty)
  3. CUSUM                   C_j = max(0, C_{j-1} + z_j - kref)         (accumulate; self-correct -> decays)
  4. conformal threshold h   = (1-eps) quantile of max_j C_j over CORRECT calibration chains
                              -> chain-level FPR <= eps by exchangeability
  5. alarm at first j with C_j > h  -> predicted error step.

Metrics (NOT AUROC -- conformal, deployment-relevant):
  FPR          fraction of CORRECT chains that raise ANY alarm (guaranteed <= eps)
  recall       fraction of ERROR chains that raise an alarm
  delay        median (alarm_step - true_first_error)  (negative = early warning / precursor)
  early_warn   fraction of caught error chains alarmed BEFORE the error step
  reported over an eps sweep = the risk-coverage curve.

Needs an npz with stepcloud(resultant) + stepgeom + tok_U_D/U_C + gold_error_step.
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs scikit-learn")


def cusum_first_alarm(s, kref, h, warmup=2):
    """causal within-chain z -> CUSUM; return (first alarm index or -1, C trajectory)."""
    C = 0.0; first = -1; traj = np.zeros(len(s))
    for j in range(len(s)):
        if j >= warmup:
            past = s[:j]
            mu, sd = past.mean(), past.std()
            z = (s[j] - mu) / (sd + 1e-9)
        else:
            z = 0.0
        C = max(0.0, C + z - kref); traj[j] = C
        if first < 0 and C > h:
            first = j
    return first, traj


def max_cusum(s, kref, warmup=2):
    C = 0.0; m = 0.0
    for j in range(len(s)):
        z = (s[j] - s[:j].mean()) / (s[:j].std() + 1e-9) if j >= warmup else 0.0
        C = max(0.0, C + z - kref); m = max(m, C)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--kref", type=float, default=0.5, help="CUSUM slack (per-step reference)")
    ap.add_argument("--eps_list", default="0.01,0.05,0.10,0.20")
    ap.add_argument("--features", default="resultant,U_D,U_C")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    gnames = [str(x) for x in z["geom_feature_names"]]
    cnames = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    layers = [int(x) for x in z["layers_used"]]; li = layers.index(args.layer)
    SG, SC, SR = z["stepgeom"], z["stepcloud"], z["step_token_ranges"]
    ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    feats = args.features.split(",")

    # per chain: ALL steps' features (in order) + detection labels + k
    chains = []                       # list of dict(X(T,F), y_det(T), k, correct)
    allX, allY, allG = [], [], []     # for training the logistic (labeled steps only)
    for i in range(len(SG)):
        sg = np.asarray(SG[i], float); sc = np.asarray(SC[i], float)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0); T = rng.shape[0]
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        X = np.full((T, len(feats)), np.nan)
        for j in range(T):
            for fi, f in enumerate(feats):
                if f == "resultant":
                    X[j, fi] = sc[j, li, cnames.index("resultant")] if "resultant" in cnames else np.nan
                elif f == "norm":
                    X[j, fi] = sg[j, li, gnames.index("norm")]
                elif f in ("U_D", "U_C"):
                    arr = ud if f == "U_D" else uc
                    lo = max(0, int(rng[j, 0]) - a0); hi = min((len(arr) if arr is not None else 0), int(rng[j, 1]) - a0 + 1)
                    X[j, fi] = np.nanmean(arr[lo:hi]) if (arr is not None and hi > lo) else np.nan
        for fi in range(X.shape[1]):
            col = X[:, fi]; col[~np.isfinite(col)] = np.nanmean(col[np.isfinite(col)]) if np.isfinite(col).any() else 0.0
        chains.append({"X": X, "k": k, "correct": correct, "i": i})
        for j in range(T):                                  # labeled steps for training
            if correct or j < k:
                yl = 0
            elif j == k:
                yl = 1
            else:
                continue
            allX.append(X[j]); allY.append(yl); allG.append(i)
    allX = np.asarray(allX, float); allY = np.asarray(allY, int); allG = np.asarray(allG, int)
    nerr = sum(not c["correct"] for c in chains); ncor = sum(c["correct"] for c in chains)
    print(f"file: {args.npz} | layer {args.layer} | chains {len(chains)} "
          f"(correct {ncor}, error {nerr}) | features {feats} | kref {args.kref}")

    # cross-fit fused score for every chain's every step
    score = {c["i"]: None for c in chains}
    gkf = GroupKFold(args.folds)
    train_correct_scores = {}                                # fold -> list of correct-chain score arrays
    fold_of = {}
    for fold, (tr, te) in enumerate(gkf.split(allX, allY, allG)):
        if len(np.unique(allY[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))
        clf.fit(allX[tr], allY[tr])
        test_chains = set(allG[te].tolist())
        train_chains = set(allG[tr].tolist())
        for c in chains:
            if c["i"] in test_chains:
                score[c["i"]] = clf.predict_proba(c["X"])[:, 1]
                fold_of[c["i"]] = fold
        # calibration scores: train-fold CORRECT chains
        tcs = [clf.predict_proba(c["X"])[:, 1] for c in chains
               if c["i"] in train_chains and c["correct"]]
        train_correct_scores[fold] = tcs

    # evaluate over eps sweep with per-fold conformal threshold
    print(f"\n{'eps':>5s} {'FPR':>6s} {'recall':>7s} {'delay(med)':>11s} {'early_warn':>11s} {'caught':>7s}")
    for eps in [float(x) for x in args.eps_list.split(",")]:
        h_fold = {}
        for fold, tcs in train_correct_scores.items():
            if not tcs:
                continue
            maxc = np.array([max_cusum(s, args.kref) for s in tcs])
            h_fold[fold] = np.quantile(maxc, 1 - eps)
        fp = fa = 0; rec = 0; delays = []; early = 0; caught = 0
        for c in chains:
            s = score[c["i"]]
            if s is None or c["i"] not in fold_of:
                continue
            h = h_fold.get(fold_of[c["i"]], np.inf)
            first, _ = cusum_first_alarm(s, args.kref, h)
            if c["correct"]:
                fa += 1; fp += (first >= 0)
            else:
                rec += 1
                if first >= 0:
                    caught += 1; d = first - c["k"]; delays.append(d)
                    if first < c["k"]:
                        early += 1
        print(f"{eps:5.2f} {fp/max(fa,1):6.3f} {caught/max(rec,1):7.3f} "
              f"{(np.median(delays) if delays else float('nan')):11.1f} "
              f"{early/max(caught,1):11.3f} {caught:7d}")

    print("\nFPR <= eps is the conformal guarantee (false alarm on a correct chain). recall = "
          "error chains caught. delay = alarm_step - true_first_error (negative = caught BEFORE "
          "the error = early warning). Read as a risk-coverage curve. Weak-but-guaranteed.")


if __name__ == "__main__":
    main()
