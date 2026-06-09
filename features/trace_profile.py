"""Uncertainty-trace-profile summary (Tracing Uncertainty, arXiv:2605.07776, Table 2).

Summarize any per-token (or per-step) series x_1..x_T into 5 features:

  Static                                   Dynamic
    mu_early = mean of the first  25%        m   = slope of OLS linear fit vs f
    mu_mid   = mean of the middle 50%        r2  = goodness-of-fit of that line
    mu_late  = mean of the last   25%

with fractional position f_t = (t-1)/(T-1) in [0,1]:
    early : f < 0.25
    mid   : 0.25 <= f < 0.75
    late  : f >= 0.75

This is applied verbatim to the three uncertainty channels (U_D, U_C, U_E) for
exact reproduction of the paper, and re-used for our per-step geometry series so
the two families are directly comparable.
"""

from __future__ import annotations

import numpy as np

PROFILE_STATS = ("early", "mid", "late", "slope", "r2")


def profile(series, lo: float = 0.25, hi: float = 0.75) -> dict:
    """Return {early, mid, late, slope, r2} for a 1-D series.

    NaNs are dropped (paired with their positions) before fitting. Positions are
    the ORIGINAL fractional locations of the surviving points, so a stride-
    subsampled series (e.g. strided U_E) is summarized at its true positions.
    Degenerate inputs (all-NaN, length < 2, empty bin) yield NaN for the
    affected statistics rather than raising.
    """
    x = np.asarray(series, dtype=np.float64).ravel()
    T = x.shape[0]
    out = {k: float("nan") for k in PROFILE_STATS}
    if T == 0:
        return out

    f_all = np.arange(T, dtype=np.float64) / max(1, T - 1)
    m = np.isfinite(x)
    if not m.any():
        return out
    x = x[m]
    f = f_all[m]

    early = x[f < lo]
    mid = x[(f >= lo) & (f < hi)]
    late = x[f >= hi]
    if early.size:
        out["early"] = float(early.mean())
    if mid.size:
        out["mid"] = float(mid.mean())
    if late.size:
        out["late"] = float(late.mean())

    if x.size >= 2 and np.ptp(f) > 0:
        # OLS slope + R^2 of x on f
        fm = f - f.mean()
        xm = x - x.mean()
        denom = float(np.sum(fm * fm))
        if denom > 0:
            slope = float(np.sum(fm * xm) / denom)
            out["slope"] = slope
            ss_tot = float(np.sum(xm * xm))
            if ss_tot > 0:
                resid = xm - slope * fm
                out["r2"] = float(1.0 - np.sum(resid * resid) / ss_tot)
            else:
                out["r2"] = float("nan")
    return out


def profile_flat(series, prefix: str, lo: float = 0.25, hi: float = 0.75) -> dict:
    """profile() with keys flattened to '<prefix>_<stat>' for tabular storage."""
    return {f"{prefix}_{k}": v for k, v in profile(series, lo, hi).items()}
