"""R2 -- Directional-Concentration Set-Encoder (DCSE): learn the pooling over the
within-step token cloud, with an inductive bias toward the dispersion mechanism.

Why not a vanilla DeepSets/PMA: the hand-crafted scalar `resultant` measures token
concentration around ONE direction (the exp-pool axis). FINDINGS sec.8 shows that
single scalar nearly dies on hard configs (omnimath +0.003 over baseline) while a
fused logistic recovers (+0.036) and a bigger classifier (GBM) does NOT help ->
the bottleneck is the REPRESENTATION. DCSE keeps the *mechanism* (directional
concentration) but lets the model learn (i) K anchor directions instead of one,
and (ii) token<->token interaction the scalar cannot see.

                    DCSE : one step's token cloud  ->  first-error logit
  ----------------------------------------------------------------------------------
   step token cloud  X in R^{n x k}   (n tokens; JL-projected hidden states; layer l)
         |
         |-- raw   X --------------------+
         |-- unit  Xhat = X/||X|| -------+        (magnitude  +  direction)
                                         v
                            +-------------------------+
                            |  TokenEmbed  phi: 2k->h |          per-token MLP
                            +------------+------------+
                                         v
                            +-------------------------+
                            |  SAB  (self-attention)  |   tokens attend to EACH OTHER
                            |  token<->token coherence|   -> sees mutual (mis)alignment
                            +------------+------------+        (a scalar cannot)
                                         v
                   +-------------------------------------------+
                   |   MultiProtoPool  (K learned directions)  |
                   |   a_k   = softmax_n( V . Q_k )            |   K anchor directions Q_k
                   |   pool_k= sum_n a_k V         (context)   |
                   |   conc_k= || sum_n a_k Vhat || in [0,1]   |   <- LEARNED multi-dir
                   +---------------------+---------------------+      RESULTANT
                                         v  readout_l = [ mean_k pool_k (h) ,  conc (K) ]
        layer 10 --readout--+
        layer 14 --readout--+   cross-layer concat
                            v
             +-----------------------------------------------+
             | [ readouts || n_tok, pos, density, U_D, U_C ] |   confounds+uncertainty
             |              MLP head  ->  logit              |   live in the head
             +-----------------------------------------------+
  ----------------------------------------------------------------------------------
   conc_k generalizes the hand-crafted resultant (1 fixed dir) to K LEARNED dirs;
   SAB adds the token<->token structure the scalar throws away.
   ablations:  side (no cloud) | mean (phi -> mean pool, no SAB/proto) | FULL (above)

Protocol = fuse_detector: GroupKFold BY CHAIN (no problem leak -> no difficulty
inflation), OOF AUROC, chain-paired bootstrap on increment over (nuisance+U_D/U_C).
Small capacity (h=48, K=4) to resist overfitting. Needs an npz from
`extract_features --store_clouds --cloud_store_layers 10,14 --cloud_proj_dim 256`.
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
    return 1.0 - sum(ch.isalpha() for ch in t) / len(t) if t else float("nan")


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

class SAB(nn.Module):
    """Set-Attention Block: tokens attend to each other (token<->token coherence)."""

    def __init__(self, h, heads=2):
        super().__init__()
        self.att = nn.MultiheadAttention(h, heads, batch_first=True)
        self.ff = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.Linear(h, h))
        self.n1 = nn.LayerNorm(h); self.n2 = nn.LayerNorm(h)

    def forward(self, v, mask):                         # mask (B,N) 1=valid
        kpm = mask < 0.5                                # True = pad (ignored)
        a, _ = self.att(v, v, v, key_padding_mask=kpm, need_weights=False)
        v = self.n1(v + a)
        return self.n2(v + self.ff(v))


class MultiProtoPool(nn.Module):
    """K learned anchor directions -> per-direction context (pool) + concentration.

    conc_k = ||sum_n a_k Vhat|| in [0,1] is a learned, attention-localized RESULTANT:
    how tightly tokens align around direction k. low = diffuse."""

    def __init__(self, h, K):
        super().__init__()
        self.Q = nn.Parameter(torch.randn(K, h) * 0.02)

    def forward(self, v, mask):                         # v (B,N,h)
        sc = torch.einsum("bnh,kh->bkn", v, self.Q) / (v.shape[-1] ** 0.5)
        sc = sc.masked_fill((mask < 0.5).unsqueeze(1), float("-inf"))
        a = torch.softmax(sc, dim=2)                    # (B,K,N)
        pool = torch.einsum("bkn,bnh->bkh", a, v).mean(1)          # (B,h)
        vhat = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        conc = torch.einsum("bkn,bnh->bkh", a, vhat).norm(dim=-1)  # (B,K)
        return torch.cat([pool, conc], dim=1)           # (B, h+K)


class LayerEnc(nn.Module):
    def __init__(self, k, h, K, heads, mode):
        super().__init__()
        self.mode = mode
        self.emb = nn.Sequential(nn.Linear(2 * k, h), nn.GELU())
        self.sab = SAB(h, heads) if mode == "full" else None
        self.pool = MultiProtoPool(h, K) if mode == "full" else None
        self.out_dim = (h + K) if mode == "full" else h

    def forward(self, x, mask):                         # x (B,N,k)
        xn = x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        v = self.emb(torch.cat([x, xn], dim=-1))        # (B,N,h)
        if self.mode == "mean":                         # ablation: plain mean pool
            w = mask / mask.sum(1, keepdim=True).clamp_min(1e-6)
            return (v * w.unsqueeze(-1)).sum(1)
        v = self.sab(v, mask)
        return self.pool(v, mask)


class DCSE(nn.Module):
    def __init__(self, k, Lc, n_side, h=48, K=4, heads=2, mode="full"):
        super().__init__()
        self.mode = mode
        if mode == "side":
            head_in = n_side
            self.encs = None
        else:
            self.encs = nn.ModuleList([LayerEnc(k, h, K, heads, mode) for _ in range(Lc)])
            head_in = sum(e.out_dim for e in self.encs) + n_side
        self.head = nn.Sequential(nn.Linear(head_in, h), nn.GELU(), nn.Linear(h, 1))

    def forward(self, x, mask, side):                   # x (B,N,Lc,k)
        if self.encs is None:
            feats = side
        else:
            r = [enc(x[:, :, l, :], mask) for l, enc in enumerate(self.encs)]
            feats = torch.cat(r + [side], dim=1)
        return self.head(feats).squeeze(-1)


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def build(npz, with_ud):
    z = np.load(npz, allow_pickle=True)
    if not bool(z.get("clouds_stored", np.array(False))):
        raise SystemExit("no respcloud; re-extract with --store_clouds")
    csl = [int(x) for x in z["cloud_store_layers"]]
    RC = z["respcloud"]; SR = z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    ST = z["steps_text"]
    UD = z["tok_U_D"] if (with_ud and "tok_U_D" in z.files) else None
    UC = z["tok_U_C"] if (with_ud and "tok_U_C" in z.files) else None

    clouds, side, Y, G = [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        rc = np.asarray(RC[i], np.float32)             # (R, Lc, k)
        rng = np.asarray(SR[i], int); ke = int(ges[i]); correct = (ke < 0)
        a0 = int(rng[0, 0]); T = rng.shape[0]
        txt = list(ST[i]) if i < len(ST) else []
        ud = np.asarray(UD[i], float) if UD is not None else None
        uc = np.asarray(UC[i], float) if UC is not None else None
        for j in range(T):
            if correct or j < ke:
                y = 0
            elif j == ke:
                y = 1
            else:
                continue
            lo = max(0, int(rng[j, 0]) - a0); hi = min(rc.shape[0], int(rng[j, 1]) - a0 + 1)
            if hi - lo < 1:
                continue
            cl = rc[lo:hi]                              # (n_j, Lc, k)
            if not np.isfinite(cl).all():
                continue
            s = [hi - lo, j / max(1, T - 1), density(txt[j]) if j < len(txt) else np.nan]
            if ud is not None:
                s += [np.nanmean(ud[lo:hi]), np.nanmean(uc[lo:hi])]
            clouds.append(cl); side.append(s); Y.append(y); G.append(i)
    side = np.asarray(side, float)
    for c in range(side.shape[1]):
        col = side[:, c]; col[~np.isfinite(col)] = np.nanmean(col)
    return clouds, side, np.asarray(Y, int), np.asarray(G, int), len(csl), csl


def batches(idx, clouds, side, Y, bs, rng, device, train):
    order = rng.permutation(idx) if train else np.asarray(idx)
    for s in range(0, len(order), bs):
        bi = order[s:s + bs]
        nmax = max(clouds[i].shape[0] for i in bi)
        Lc, k = clouds[bi[0]].shape[1], clouds[bi[0]].shape[2]
        X = np.zeros((len(bi), nmax, Lc, k), np.float32)
        M = np.zeros((len(bi), nmax), np.float32)
        for r, i in enumerate(bi):
            n = clouds[i].shape[0]; X[r, :n] = clouds[i]; M[r, :n] = 1
        yield (torch.from_numpy(X).to(device), torch.from_numpy(M).to(device),
               torch.from_numpy(side[bi]).float().to(device),
               torch.from_numpy(Y[bi]).float().to(device))


def run_oof(clouds, side, Y, G, Lc, mode, args, device):
    k = clouds[0].shape[2]; n_side = side.shape[1]
    scores = np.full(len(Y), np.nan)
    for fold, (tr, te) in enumerate(GroupKFold(args.folds).split(side, Y, G)):
        torch.manual_seed(args.seed + fold)
        rng = np.random.default_rng(args.seed + fold)
        mu, sd = side[tr].mean(0), side[tr].std(0) + 1e-6
        side_z = (side - mu) / sd
        sc = np.mean([np.sqrt((clouds[i] ** 2).mean()) for i in tr[:2000]]) + 1e-6
        cl_s = [c / sc for c in clouds]
        net = DCSE(k, Lc, n_side, args.hidden, args.K, args.heads, mode).to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
        pw = torch.tensor(float((Y[tr] == 0).sum() / max(1, (Y[tr] == 1).sum()))).to(device)
        lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
        net.train()
        for _ in range(args.epochs):
            for X, M, S, yb in batches(tr, cl_s, side_z, Y, args.bs, rng, device, True):
                opt.zero_grad(); lossf(net(X, M, S), yb).backward(); opt.step()
        net.eval()
        with torch.no_grad():
            ptr = 0
            for X, M, S, _ in batches(te, cl_s, side_z, Y, args.bs, rng, device, False):
                p = torch.sigmoid(net(X, M, S)).cpu().numpy()
                scores[te[ptr:ptr + len(p)]] = p; ptr += len(p)
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--with_ud", action="store_true")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--K", type=int, default=4, help="# learned anchor directions")
    ap.add_argument("--heads", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--boot", type=int, default=500)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    clouds, side, Y, G, Lc, csl = build(args.npz, args.with_ud)
    print(f"file: {args.npz} | cloud layers {csl} | device {device}")
    print(f"steps: {len(Y)} | first-error: {int(Y.sum())} | chains: {len(np.unique(G))} "
          f"| k={clouds[0].shape[2]} | side={side.shape[1]} | h={args.hidden} K={args.K}")

    s_base = run_oof(clouds, side, Y, G, Lc, "side", args, device)
    s_mean = run_oof(clouds, side, Y, G, Lc, "mean", args, device)
    s_full = run_oof(clouds, side, Y, G, Lc, "full", args, device)
    a_b, a_m, a_f = auroc(s_base, Y), auroc(s_mean, Y), auroc(s_full, Y)

    print(f"\n{'detector':40s} {'AUROC':>7s}")
    print(f"{'baseline (nuis+U, no cloud)':40s} {a_b:7.3f}")
    print(f"{'+ cloud, MEAN pool (no SAB/proto)':40s} {a_m:7.3f}")
    print(f"{'+ cloud, DCSE (SAB + multi-proto)':40s} {a_f:7.3f}")

    rng = np.random.default_rng(0); chains = np.unique(G)
    d_fb, d_fm = [], []
    for _ in range(args.boot):
        cb = rng.choice(chains, size=len(chains), replace=True)
        mask = np.concatenate([np.where(G == c)[0] for c in cb])
        d_fb.append(auroc(s_full[mask], Y[mask]) - auroc(s_base[mask], Y[mask]))
        d_fm.append(auroc(s_full[mask], Y[mask]) - auroc(s_mean[mask], Y[mask]))

    def ci(d, name):
        d = np.asarray(d); lo, hi = np.nanpercentile(d, [2.5, 97.5])
        print(f"  {name:46s} +{np.nanmean(d):.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"{'SIGNIFICANT' if lo > 0 else 'ns'}")

    print("\n=== increments (chain-paired bootstrap) ===")
    ci(d_fb, "DCSE - baseline (cloud rep beyond confound+unc)")
    ci(d_fm, "DCSE - mean-pool (SAB+multi-proto beyond mean)")
    print("\nread: compare DCSE AUROC to fuse_detector's FUSED scalar -- if DCSE > FUSED "
          "(esp on hard configs where the scalar dies) the learned cloud rep breaks the "
          "scalar ceiling. (2)>0 => the structure (token<->token + K learned dirs) is "
          "what helps, not merely having the cloud.")


if __name__ == "__main__":
    main()
