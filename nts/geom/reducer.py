# nts/geom/reducer.py — drop massive-activation dims + PCA-whiten to m dims, fit on correct steps
import numpy as np


def fit_reducer(corr_steps, m=128, massive_drop=5):
    X = np.asarray(corr_steps, float)
    med = np.median(np.abs(X), 0); massive = np.argsort(med)[::-1][:massive_drop]
    keep = np.setdiff1d(np.arange(X.shape[1]), massive)
    Xk = X[:, keep]; mu = Xk.mean(0)
    _, s, Vt = np.linalg.svd(Xk - mu, full_matrices=False)
    meff = min(m, Vt.shape[0]); comps = Vt[:meff]; scale = s[:meff] / np.sqrt(max(len(Xk) - 1, 1))

    def transform(V):
        Vk = np.asarray(V, float)[:, keep]
        return ((Vk - mu) @ comps.T) / (scale + 1e-8)

    return transform
