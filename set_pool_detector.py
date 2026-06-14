"""R2 learned-pooling detector: learn the pooling over the within-step token
cloud, instead of the fixed exp-pool + hand-crafted scalar (resultant/coherence).

Motivation (FINDINGS sec.8): on hard configs the single hand-crafted scalar
nearly dies (+0.003 over baseline) while a fused-scalar logistic recovers
(+0.036), and a bigger classifier (GBM) does NOT help -> the bottleneck is the
REPRESENTATION (we pooled the cloud into a scalar), not the classifier. So go
richer: feed the model the actual token cloud and let it learn the pooling.

Needs an npz from `extract_features --store_clouds --cloud_store_layers L
--cloud_proj_dim k` (respcloud: per-chain (R, Lc, k) random-projected clouds).

Model (small, to resist difficulty inflation):
  two permutation-invariant attention-pooling branches over the step's tokens --
    (a) RAW projected tokens     (magnitude-aware, generalizes coherence/norm)
    (b) UNIT-normalized tokens   (pure direction, generalizes resultant)
  each: per-token MLP -> pooling-by-multihead-attention (1 learned seed query)
  -> concat [pool_raw, pool_unit, nuisance(n_tok,pos,density), U_D, U_C] -> MLP -> logit

Protocol = same as fuse_detector: GroupKFold BY CHAIN (no problem leaks ->
no difficulty inflation), out-of-fold AUROC, chain-paired bootstrap on the
increment over the (nuisance + U_D/U_C) baseline. Ablations:
  - baseline      : nuisance + U only (no cloud)         [== fuse_detector baseline]
  - mean-pool     : cloud branches use MEAN pooling (no learned attention)
  - FULL          : learned attention pooling
so we can separate "cloud helps" from "LEARNED pooling helps".
"""

from __future__ import annotations

import argparse
import numpy as np

try:
    import torch
    import torch.nn as nn
    from sklearn.model_selection import GroupKFold
except ImportError:
    raise SystemExit("needs torch + scikit-learn")


def auroc(score, y):
    m = np.isfinite(score); s, yy = score[m], y[m]
    npos, nneg = int((yy == 1).sum()), int((yy == 0).sum())
    if npos == 0 or nneg == 0:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def density(text):
    t = str(text)
    if not t:
        return float("nan")
    return 1.0 - sum(ch.isalpha() for ch in t) / len(t)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------

class AttnPool(nn.Module):
    """Pooling-by-attention: a learned seed query attends over the token set.

    If mode='mean', ignores attention and mean-pools (ablation control)."""

    def __init__(self, k, h, mode="attn"):
        super().__init__()
        self.mode = mode
        self.phi = nn.Sequential(nn.Linear(k, h), nn.GELU(), nn.Linear(h, h))
        self.q = nn.Parameter(torch.randn(h) * 0.02)

    def forward(self, x, mask):                         # x (B,N,k)  mask (B,N) 1=valid
        v = self.phi(x)                                 # (B,N,h)
        if self.mode == "mean":
            w = mask / mask.sum(1, keepdim=True).clamp_min(1e-6)
            return (v * w.unsqueeze(-1)).sum(1)
        a = (v @ self.q) / (v.shape[-1] ** 0.5)         # (B,N)
        a = a.masked_fill(mask < 0.5, float("-inf"))
        a = torch.softmax(a, dim=1)
        return (v * a.unsqueeze(-1)).sum(1)             # (B,h)


class SetDetector(nn.Module):
    def __init__(self, k, n_side, h=64, mode="attn", use_cloud=True):
        super().__init__()
        self.use_cloud = use_cloud
        self.mode = mode
        if use_cloud:
            self.pool_raw = AttnPool(k, h, mode)
            self.pool_unit = AttnPool(k, h, mode)
            head_in = 2 * h + n_side
        else:
            head_in = n_side
        self.head = nn.Sequential(nn.Linear(head_in, h), nn.GELU(),
                                  nn.Linear(h, 1))

    def forward(self, x, mask, side):
        if self.use_cloud:
            xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)   # unit tokens
            feats = torch.cat([self.pool_raw(x, mask), self.pool_unit(xn, mask),
                               side], dim=1)
        else:
            feats = side
        return self.head(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build(npz, layer, with_ud):
    z = np.load(npz, allow_pickle=True)
    if not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no respcloud; re-extract with --store_clouds")
    csl = [int(x) for x in z["cloud_store_layers"]]
    if layer not in csl:
        raise SystemExit(f"layer {layer} not in stored cloud layers {csl}")
    cli = csl.index(layer)
    RC = z["respcloud"]; SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    ST = z["steps_text"]
    UD = z["tok_U_D"] if (with_ud and "tok_U_D" in z.files) else None
    UC = z["tok_U_C"] if (with_ud and "tok_U_C" in z.files) else None

    clouds, side, Y, G = [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rc = np.asarray(RC[i], np.float32)             # (R, Lc, k)
        rng = np.asarray(SR[i], int); k = int(ges[i]); correct = (k < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        txt = list(ST[i]) if i < len(ST) else []
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < k:
                y = 0
            elif j == k:
                y = 1
            else:
                continue
            lo = int(rng[j, 0]) - a0; hi = int(rng[j, 1]) - a0 + 1
            lo, hi = max(0, lo), min(rc.shape[0], hi)
            if hi - lo < 1:
                continue
            cl = rc[lo:hi, cli, :]
            if not np.isfinite(cl).all():
                continue
            ntok = hi - lo
            s = [ntok, j / max(1, T - 1), density(txt[j]) if j < len(txt) else np.nan]
            if ud is not None:
                s += [np.nanmean(ud[lo:hi]) if hi > lo else np.nan,
                      np.nanmean(uc[lo:hi]) if hi > lo else np.nan]
            clouds.append(cl); side.append(s); Y.append(y); G.append(i)
    side = np.asarray(side, float)
    for c in range(side.shape[1]):
        col = side[:, c]; col[~np.isfinite(col)] = np.nanmean(col)
    return clouds, side, np.asarray(Y, int), np.asarray(G, int)


def batches(idx, clouds, side, Y, bs, rng, device, train):
    order = rng.permutation(idx) if train else idx
    for s in range(0, len(order), bs):
        bi = order[s:s + bs]
        nmax = max(clouds[i].shape[0] for i in bi)
        k = clouds[bi[0]].shape[1]
        X = np.zeros((len(bi), nmax, k), np.float32)
        M = np.zeros((len(bi), nmax), np.float32)
        for r, i in enumerate(bi):
            n = clouds[i].shape[0]; X[r, :n] = clouds[i]; M[r, :n] = 1
        yield (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device),
               torch.from_numpy(side[bi]).float().to(device),
               torch.from_numpy(Y[bi]).float().to(device))


def run_oof(clouds, side, Y, G, mode, use_cloud, args, device):
    k = clouds[0].shape[1]; n_side = side.shape[1]
    scores = np.full(len(Y), np.nan)
    gkf = GroupKFold(args.folds)
    for fold, (tr, te) in enumerate(gkf.split(side, Y, G)):
        torch.manual_seed(args.seed + fold)
        rng = np.random.default_rng(args.seed + fold)
        # standardize side features on train
        mu, sd = side[tr].mean(0), side[tr].std(0) + 1e-6
        side_z = (side - mu) / sd
        # global cloud scale on train (numeric stability)
        sc = np.mean([np.sqrt((clouds[i] ** 2).mean()) for i in tr[:2000]]) + 1e-6
        cl_s = [c / sc for c in clouds]
        net = SetDetector(k, n_side, args.hidden, mode, use_cloud).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        lossf = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor(float((Y[tr] == 0).sum() / max(1, (Y[tr] == 1).sum()))
                                    ).to(device))
        net.train()
        for ep in range(args.epochs):
            for X, M, S, yb in batches(tr, cl_s, side_z, Y, args.bs, rng, device, True):
                opt.zero_grad()
                out = net(X, M, S)
                loss = lossf(out, yb)
                loss.backward(); opt.step()
        net.eval()
        # predict OOF (ordered)
        with torch.no_grad():
            ptr = 0
            for X, M, S, yb in batches(np.asarray(te), cl_s, side_z, Y, args.bs,
                                       rng, device, False):
                p = torch.sigmoid(net(X, M, S)).cpu().numpy()
                scores[te[ptr:ptr + len(p)]] = p; ptr += len(p)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--with_ud", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clouds, side, Y, G = build(args.npz, args.layer, args.with_ud)
    print(f"file: {args.npz} | layer {args.layer} | device {device}")
    print(f"steps: {len(Y)} | first-error: {int(Y.sum())} | chains: {len(np.unique(G))} "
          f"| cloud k={clouds[0].shape[1]} | side dims={side.shape[1]}")

    s_base = run_oof(clouds, side, Y, G, "attn", False, args, device)
    s_mean = run_oof(clouds, side, Y, G, "mean", True, args, device)
    s_attn = run_oof(clouds, side, Y, G, "attn", True, args, device)
    a_base, a_mean, a_attn = auroc(s_base, Y), auroc(s_mean, Y), auroc(s_attn, Y)

    print(f"\n{'detector':36s} {'AUROC':>7s}")
    print(f"{'baseline (nuis+U, no cloud)':36s} {a_base:7.3f}")
    print(f"{'+ cloud, MEAN pool':36s} {a_mean:7.3f}")
    print(f"{'+ cloud, LEARNED attn pool (FULL)':36s} {a_attn:7.3f}")

    rng = np.random.default_rng(0); chains = np.unique(G)
    d_ab, d_am = [], []
    for _ in range(args.boot):
        cb = rng.choice(chains, size=len(chains), replace=True)
        mask = np.concatenate([np.where(G == c)[0] for c in cb])
        d_ab.append(auroc(s_attn[mask], Y[mask]) - auroc(s_base[mask], Y[mask]))
        d_am.append(auroc(s_attn[mask], Y[mask]) - auroc(s_mean[mask], Y[mask]))

    def ci(d, name):
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        print(f"  {name:44s} +{np.nanmean(d):.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"{'SIGNIFICANT' if lo > 0 else 'ns'}")

    print("\n=== increments (chain-paired bootstrap) ===")
    ci(d_ab, "FULL - baseline (cloud beyond confound+unc)")
    ci(d_am, "FULL - mean-pool (LEARNED attn beyond cloud)")
    print("\nread: (1)>0 => learned cloud rep adds beyond confound+uncertainty; "
          "compare its AUROC to fuse_detector's FUSED scalar number to see if the "
          "cloud breaks the scalar ceiling. (2)>0 => the ATTENTION (which tokens "
          "matter) is what helps, not just having the cloud.")


if __name__ == "__main__":
    main()
