# nts/eval/metrics.py — AUROC / best-direction / length-bucket AUROC (codebase convention)
import numpy as np


def auroc(s, y):
    s = np.asarray(s, float); y = np.asarray(y)
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


def bucket(s, y, nt, nb=6):
    s = np.asarray(s, float); y = np.asarray(y); nt = np.asarray(nt, float)
    m = np.isfinite(s) & np.isfinite(nt); s, y, nt = s[m], y[m], nt[m]
    if len(s) == 0:
        return float("nan")
    e = np.quantile(nt, np.linspace(0, 1, nb + 1)); e[-1] += 1
    b = np.clip(np.digitize(nt, e[1:-1]), 0, nb - 1); num = den = 0.0
    for bb in range(nb):
        mm = b == bb; a = bdir(auroc(s[mm], y[mm]))
        ne = int((y[mm] == 1).sum()); ng = int((y[mm] == 0).sum())
        if np.isfinite(a) and ne and ng:
            num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


def spearman(a, b):
    a = np.asarray(a, float); b = np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    ra = np.argsort(np.argsort(a[m])).astype(float)
    rb = np.argsort(np.argsort(b[m])).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def bimodality_coeff(x):
    """Sarle's bimodality coefficient BC=(skew^2+1)/kurtosis; >0.555 suggests bimodal."""
    x = np.asarray(x, float); x = x[np.isfinite(x)]
    n = len(x)
    if n < 8:
        return float("nan")
    m = x.mean(); s = x.std()
    if s <= 0:
        return float("nan")
    z = (x - m) / s
    g1 = (z ** 3).mean(); g2 = (z ** 4).mean() - 3.0
    denom = g2 + 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float((g1 ** 2 + 1.0) / denom) if denom > 0 else float("nan")
