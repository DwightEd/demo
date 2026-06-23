"""Online chain-error probability via a 2-state Gaussian HMM on per-token EMA-kappa.
Latent z_t in {healthy, error}; forward filter -> P(z_t=error | kappa_{1:t}). Supervised fit
when gold_error_step exists (ProcessBench), else semi-supervised EM (sampled/real-inference).
Reports response-level AUROC (max filtered P) + length bucket + online lead-time."""
from __future__ import annotations
import argparse
import numpy as np
from sklearn.model_selection import GroupKFold


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
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm])); ne = int(y[mm].sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def ema_kappa(U, ok, a):
    N, d = U.shape; out = np.full(N, np.nan); es = np.zeros(d); ew = 0.0
    for t in range(N):
        if ok[t]:
            es = a * es + U[t]; ew = a * ew + 1.0
        if ew > 0:
            out[t] = np.linalg.norm(es / ew)
    return out


def lse(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True); return (m.squeeze(axis) if axis is not None else m.squeeze()) + np.log(np.exp(a - m).sum(axis=axis))


def gll(x, mu, sd):
    return -0.5 * ((x[:, None] - mu[None]) / sd[None]) ** 2 - np.log(sd[None]) - 0.5 * np.log(2 * np.pi)


def filt(x, mu, sd, lA, lpi):
    """Online forward filter -> P(z=error|x_{1:t}) per token."""
    B = gll(x, mu, sd); T = len(x); la = np.empty((T, 2)); la[0] = lpi + B[0]
    for t in range(1, T):
        la[t] = B[t] + lse(la[t - 1][:, None] + lA, axis=0)
    return np.exp(la - lse(la, axis=1)[:, None])[:, 1]


def fb(x, mu, sd, lA, lpi):
    B = gll(x, mu, sd); T = len(x); la = np.empty((T, 2)); lb = np.zeros((T, 2)); la[0] = lpi + B[0]
    for t in range(1, T):
        la[t] = B[t] + lse(la[t - 1][:, None] + lA, axis=0)
    for t in range(T - 2, -1, -1):
        lb[t] = lse(lA + (B[t + 1] + lb[t + 1])[None], axis=1)
    ll = lse(la[-1]); g = np.exp(la + lb - ll)
    xi = np.zeros((2, 2))
    for t in range(T - 1):
        m = la[t][:, None] + lA + (B[t + 1] + lb[t + 1])[None]; xi += np.exp(m - ll)
    return g, xi


def em(xs, lab, iters=20):
    """Semi-supervised EM: correct chains (lab==0) clamped to healthy; error chains (lab==1) free."""
    allx = np.concatenate(xs); q = np.quantile(allx, [0.2, 0.8])
    mu = np.array([q[1], q[0]]); sd = np.array([allx.std()] * 2); A = np.array([[0.97, 0.03], [0.03, 0.97]]); pi = np.array([0.9, 0.1])
    for _ in range(iters):
        gsum = np.zeros(2); gx = np.zeros(2); gxx = np.zeros(2); xis = np.zeros((2, 2)); pis = np.zeros(2)
        for x, lb in zip(xs, lab):
            if lb == 0:
                g = np.zeros((len(x), 2)); g[:, 0] = 1.0; xi = np.zeros((2, 2)); xi[0, 0] = len(x) - 1
            else:
                g, xi = fb(x, mu, sd, np.log(A + 1e-12), np.log(pi + 1e-12))
            gsum += g.sum(0); gx += (g * x[:, None]).sum(0); gxx += (g * (x[:, None] ** 2)).sum(0)
            xis += xi; pis += g[0]
        mu = gx / np.maximum(gsum, 1e-9); sd = np.sqrt(np.maximum(gxx / np.maximum(gsum, 1e-9) - mu ** 2, 1e-6))
        A = xis / np.maximum(xis.sum(1, keepdims=True), 1e-9); pi = pis / pis.sum()
    return mu, sd, A, pi


def sup_fit(xs, states):
    """Closed-form supervised fit from per-token state labels."""
    X = np.concatenate(xs); S = np.concatenate(states)
    mu = np.array([X[S == s].mean() if (S == s).any() else X.mean() for s in (0, 1)])
    sd = np.array([X[S == s].std() + 1e-6 if (S == s).sum() > 1 else X.std() for s in (0, 1)])
    A = np.full((2, 2), 1e-3); pi = np.full(2, 1e-3)
    for st in states:
        pi[st[0]] += 1
        for t in range(len(st) - 1):
            A[st[t], st[t + 1]] += 1
    return mu, sd, A / A.sum(1, keepdims=True), pi / pi.sum()


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("npz"); ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--ema", type=float, default=0.9); args = ap.parse_args()
    z = np.load(args.npz, allow_pickle=True)
    csl = [int(x) for x in z["cloud_store_layers"]]; li = csl.index(args.layer)
    RC, SR = z["respcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    isc = z["is_correct"].astype(int) if "is_correct" in z.files else None
    sup = (ges >= 0).any()
    xs, states, NT, Y, ksteps = [], [], [], [], []
    for i in range(len(RC)):
        if RC[i] is None:
            continue
        H = np.asarray(RC[i], np.float64)[:, li, :]; nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
        U = np.zeros_like(H); U[ok] = H[ok] / nrm[ok, None]; ek = ema_kappa(U, ok, args.ema)
        f = np.isfinite(ek)
        if f.sum() < 3:
            continue
        x = ek[f]; rng = np.asarray(SR[i], int); a0 = int(rng[0, 0]); k = int(ges[i])
        st = np.zeros(len(ek), int); kt = -1
        if sup and k >= 0:
            kt = max(0, int(rng[k, 0]) - a0)
            for j in range(k, rng.shape[0]):
                lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ek), int(rng[j, 1]) - a0 + 1); st[lo:hi] = 1
        xs.append(x); states.append(st[f]); NT.append(int(f.sum()))
        Y.append(int(isc[i] == 0) if (isc is not None and not sup) else int(k >= 0))
        ksteps.append(int(f[:kt].sum()) if kt >= 0 else -1)
    NT = np.asarray(NT, float); Y = np.asarray(Y, int); G = np.arange(len(xs))
    score = np.full(len(xs), np.nan); cross = np.full(len(xs), np.nan)
    for tr, te in GroupKFold(5).split(G[:, None], Y, G):
        if sup:
            mu, sd, A, pi = sup_fit([xs[i] for i in tr], [states[i] for i in tr])
        else:
            mu, sd, A, pi = em([xs[i] for i in tr], [Y[i] for i in tr])
        lA = np.log(A + 1e-12); lpi = np.log(pi + 1e-12)
        for i in te:
            p = filt(xs[i], mu, sd, lA, lpi); score[i] = p.max()
            cr = np.where(p > 0.5)[0]; cross[i] = cr[0] if len(cr) else len(p)
    print(f"{args.npz} | L{args.layer} ema{args.ema} | {'SUPERVISED' if sup else 'EM (semi-sup)'} | resp {len(Y)} err {int(Y.sum())}")
    print(f"  HMM online P(error)  AUROC {bdir(auroc(score, Y)):.3f}  bkt(len) {bucket(score, Y, NT):.3f}")
    if sup:
        ks = np.array(ksteps); m = (Y == 1) & (ks >= 0) & np.isfinite(cross)
        lead = ks[m] - cross[m]; fired = cross[m] < NT[m]
        print(f"  lead-time (err chains): fired {fired.mean():.2f} | crosses AT/BEFORE error {np.mean(lead[fired] >= 0):.2f} | "
              f"median lead {np.median(lead[fired]):.0f} tok")


if __name__ == "__main__":
    main()
