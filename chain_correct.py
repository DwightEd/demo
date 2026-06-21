"""Chain-level final-answer correctness prediction, keeping ALL step signals.

Aggregating each chain to a single mean throws away the most informative part -- the WORST
step. So we summarize the FULL per-step sequence with order statistics that scan every step
(min / max / quantiles / slope / deepest self-consistency dip / CUSUM peak / fraction of
dipped steps). Predict whether the chain's final answer is correct (is_correct) with
GroupKFold(by problem) logistic + gradient boosting.

At CHAIN level, length/difficulty is a LEGITIMATE predictor (you genuinely want to know if the
answer is right, and harder/longer chains really are more error-prone) -- NOT a confound to
strip as it was at step level. So we also report the length-only baseline to see how much the
geometry adds on top.

Optional CEILING (black-box diagnostic, only if torch present): a small GRU over the raw
per-step sequence. If it beats the interpretable order statistics, the temporal structure
carries extra chain-level signal the summaries missed; if not, the summaries captured it.

Needs coh.npz: stepcloud(resultant) + is_correct + step_token_ranges + problem_ids (+ tok_U_D).
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


def causal_resid(y, sd_floor, clip=5.0, eps=1e-6):
    """signed causal within-chain residual s_run (nan for t<2)."""
    T = len(y); s = np.full(T, np.nan)
    for t in range(2, T):
        h = y[:t]; mu = h.mean(); sd = max(h.std(), sd_floor) + eps
        s[t] = np.clip((y[t] - mu) / sd, -clip, clip)
    return s


def cusum_max(score, kref=0.5):
    W = m = 0.0
    for v in score:
        if np.isfinite(v):
            W = max(0.0, W + v - kref); m = max(m, W)
    return m


def boot_ci(s, y, n=1000, seed=0):
    rng = np.random.default_rng(seed); a0 = bdir(auroc(s, y)); ds = []
    idx = np.arange(len(y))
    for _ in range(n):
        ii = rng.choice(idx, len(idx), replace=True)
        a = auroc(s[ii], y[ii])
        if np.isfinite(a):
            ds.append(max(a, 1 - a))
    lo, hi = np.percentile(ds, [2.5, 97.5]) if ds else (np.nan, np.nan)
    return a0, lo, hi


FEAT_NAMES = ["res_mean", "res_std", "res_min", "res_max", "res_first", "res_last", "res_slope",
              "res_q10", "res_q25", "res_q50", "res_q75", "res_q90",
              "srun_min", "srun_mean", "srun_fracdip", "cusum_max",
              "cum_drop", "cum_drop_norm", "net_drift", "frac_down",
              "n_steps", "log_tot_tok", "ud_mean", "ud_max"]


def chain_features(y, sr, ud, nt):
    q = np.percentile(y, [10, 25, 50, 75, 90])
    slope = np.polyfit(np.arange(len(y)), y, 1)[0] if len(y) >= 2 else 0.0
    f = [y.mean(), y.std(), y.min(), y.max(), y[0], y[-1], slope, *q]
    srf = sr[np.isfinite(sr)]
    f += ([srf.min(), srf.mean(), float((srf < -1).mean())] if len(srf) else [0.0, 0.0, 0.0])
    f += [cusum_max(-sr)]
    # cumulative step-to-step diffusion (drop = resultant fell vs previous step)
    d = np.diff(y)
    cum_drop = float(np.sum(np.maximum(0.0, -d)))
    f += [cum_drop, cum_drop / max(len(d), 1), float(y[0] - y[-1]), float((d < 0).mean()) if len(d) else 0.0]
    f += [float(len(y)), float(np.log(max(nt.sum(), 1)))]
    f += ([np.nanmean(ud), np.nanmax(ud)] if (ud is not None and np.isfinite(ud).any()) else [0.0, 0.0])
    return f


def oof(clf_factory, X, y, grp, folds):
    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(X, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = clf_factory(); clf.fit(X[tr], y[tr]); s[te] = clf.predict_proba(X[te])[:, 1]
    return s


def gru_ceiling(seqs, y, grp, folds, epochs=12):
    """optional black-box ceiling: small GRU over the raw per-step sequence."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        return None
    torch.manual_seed(0)
    F = seqs[0].shape[1]

    class Net(nn.Module):
        def __init__(self):
            super().__init__(); self.g = nn.GRU(F, 32, batch_first=True); self.o = nn.Linear(32, 1)
        def forward(self, x, lens):
            p = nn.utils.rnn.pack_padded_sequence(x, lens, batch_first=True, enforce_sorted=False)
            _, h = self.g(p); return self.o(h[-1]).squeeze(-1)

    def pad(batch):
        lens = [len(s) for s in batch]; L = max(lens)
        out = np.zeros((len(batch), L, F), np.float32)
        for i, s in enumerate(batch):
            out[i, :len(s)] = s
        return torch.tensor(out), torch.tensor(lens)

    s = np.full(len(y), np.nan)
    for tr, te in GroupKFold(folds).split(seqs, y, grp):
        if len(np.unique(y[tr])) < 2:
            continue
        net = Net(); opt = torch.optim.Adam(net.parameters(), lr=1e-2)
        lossf = nn.BCEWithLogitsLoss()
        Xtr, ltr = pad([seqs[i] for i in tr]); ytr = torch.tensor(y[tr].astype(np.float32))
        net.train()
        for _ in range(epochs):
            opt.zero_grad(); out = net(Xtr, ltr); loss = lossf(out, ytr); loss.backward(); opt.step()
        net.eval()
        Xte, lte = pad([seqs[i] for i in te])
        with torch.no_grad():
            s[te] = torch.sigmoid(net(Xte, lte)).numpy()
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--gru", action="store_true", help="also run the GRU sequence ceiling (needs torch)")
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=True)
    cnames = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(args.layer)
    SC, SR = z["stepcloud"], z["step_token_ranges"]
    isc = z["is_correct"].astype(int)
    pid = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(SC))
    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    fi = cnames.index("resultant")

    ys = []
    for i in range(len(SC)):
        ys.append(np.asarray(SC[i], float)[:, li, fi])
    sd_floor = 0.5 * float(np.median([y.std() for y in ys if len(y) >= 2]))

    X, Y, G, SEQ = [], [], [], []
    for i in range(len(SC)):
        y = ys[i]; rng = np.asarray(SR[i], int); T = len(y)
        if not np.isfinite(y).all() or T < 3 or rng.shape[0] != T:
            continue
        nt = np.array([int(rng[j, 1] - rng[j, 0] + 1) for j in range(T)], float)
        a0 = int(rng[0, 0]); ud = np.asarray(UD[i], float) if UD is not None else None
        sr = causal_resid(y, sd_floor)
        X.append(chain_features(y, sr, ud, nt))
        # raw per-step sequence for the GRU ceiling: [resultant, s_run(0 if nan), pos, log n_tok]
        seq = np.zeros((T, 4), np.float32)
        seq[:, 0] = y; seq[:, 1] = np.nan_to_num(sr); seq[:, 2] = np.arange(T) / max(1, T - 1)
        seq[:, 3] = np.log(np.maximum(nt, 1))
        SEQ.append(seq)
        Y.append(1 - int(isc[i]))                       # label 1 = INCORRECT chain (error)
        G.append(int(pid[i]))
    X = np.asarray(X, float); Y = np.asarray(Y, int); G = np.asarray(G, int)
    nimp = ~np.isfinite(X); X[nimp] = 0.0

    print(f"file: {args.npz} | layer {args.layer} | chains {len(Y)} | incorrect {int(Y.sum())} "
          f"({Y.mean()*100:.1f}%) | problems {len(np.unique(G))}")

    # baselines + full feature set
    def lg(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    def gb(): return GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=0)

    li_ = FEAT_NAMES.index("n_steps"); lt_ = FEAT_NAMES.index("log_tot_tok")
    rm_ = FEAT_NAMES.index("res_min"); sd_ = FEAT_NAMES.index("srun_min")
    cd_ = FEAT_NAMES.index("cum_drop_norm"); nd_ = FEAT_NAMES.index("net_drift")
    print(f"\n{'predictor':34s} {'AUROC':>7s} {'95% CI':>17s}")
    for nm, col in [("length only (n_steps,log_tok)", [li_, lt_]),
                    ("res_min only (worst step)", [rm_]),
                    ("srun_min only (deepest dip)", [sd_]),
                    ("cum_drop_norm (total diffusion)", [cd_]),
                    ("net_drift (first-last)", [nd_])]:
        s = oof(lg, X[:, col], Y, G, args.folds)
        a, lo, hi = boot_ci(s, Y)
        print(f"  {nm:32s} {a:7.3f} {f'[{lo:.3f},{hi:.3f}]':>17s}")
    for nm, fac in [("ALL feats (logistic)", lg), ("ALL feats (grad-boost)", gb)]:
        s = oof(fac, X, Y, G, args.folds)
        a, lo, hi = boot_ci(s, Y)
        print(f"  {nm:32s} {a:7.3f} {f'[{lo:.3f},{hi:.3f}]':>17s}")

    if args.gru:
        s = gru_ceiling(SEQ, Y, G, args.folds)
        if s is None:
            print("  GRU ceiling                       (torch not available)")
        else:
            a, lo, hi = boot_ci(s, Y)
            print(f"  {'GRU sequence (CEILING, black-box)':32s} {a:7.3f} {f'[{lo:.3f},{hi:.3f}]':>17s}")

    print("\nread: chain-level CORRECTNESS prediction (label 1 = incorrect). length-only is the "
          "honest baseline (difficulty is legitimate here). ALL-feats shows how much the geometry "
          "adds over length. If the GRU CEILING >> ALL-feats, the order statistics miss temporal "
          "structure; if ~equal, the interpretable summaries already capture the chain signal.")


if __name__ == "__main__":
    main()
