#!/usr/bin/env python3
"""Shared audit utilities for the current reasoning-geometry experiments.

This module keeps reusable data loading, scoring, and self-test helpers out of
retired one-off audit scripts.  Current mainline and hypergraph branches should
depend on this file rather than on old experiment entrypoints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


EPS = 1e-9


@dataclass
class Chain:
    idx: int
    group: int
    gold: int
    correct: bool
    features: Dict[str, np.ndarray]
    n_steps: int


def finite_json(obj):
    if isinstance(obj, dict):
        return {str(k): finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [finite_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return finite_json(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    return obj


def auroc(score, y) -> float:
    s = np.asarray(score, float)
    yy = np.asarray(y, int)
    m = np.isfinite(s)
    s, yy = s[m], yy[m]
    p = int((yy == 1).sum())
    n = int((yy == 0).sum())
    if p == 0 or n == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), float)
    ss = s[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return float((ranks[yy == 1].sum() - p * (p + 1) / 2.0) / (p * n))


def bdir(a: float) -> float:
    return float(max(a, 1.0 - a)) if np.isfinite(a) else float("nan")


def safe_mean(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else float("nan")


def safe_std(x) -> float:
    a = np.asarray(x, float)
    a = a[np.isfinite(a)]
    return float(a.std()) if len(a) else float("nan")


def cluster_boot_increment(sf, sb, y, groups, *, n_boot=500, seed=0) -> Dict[str, object]:
    sf = np.asarray(sf, float)
    sb = np.asarray(sb, float)
    y = np.asarray(y, int)
    groups = np.asarray(groups)
    m = np.isfinite(sf) & np.isfinite(sb)
    if m.sum() < 30 or len(np.unique(y[m])) < 2:
        return {"point": float("nan"), "lo": float("nan"), "hi": float("nan"), "sig": False}
    point = auroc(sf[m], y[m]) - auroc(sb[m], y[m])
    rng = np.random.default_rng(seed)
    ug = np.unique(groups[m])
    by = {g: np.where(m & (groups == g))[0] for g in ug}
    vals = []
    for _ in range(n_boot):
        chosen = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([by[g] for g in chosen])
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(auroc(sf[idx], y[idx]) - auroc(sb[idx], y[idx]))
    if not vals:
        return {"point": float(point), "lo": float("nan"), "hi": float("nan"), "sig": False}
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"point": float(point), "lo": float(lo), "hi": float(hi), "sig": bool(lo > 0 or hi < 0)}


def step_ranges(rng: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    rr = np.asarray(rng, int)
    n_tok = rr[:, 1] - rr[:, 0] + 1
    pos = np.arange(len(rr), dtype=float) / max(1, len(rr) - 1)
    return n_tok.astype(float), pos


def layer_index(layers: Sequence[int], layer: int, *, nearest: bool = False) -> Optional[int]:
    layers = [int(x) for x in layers]
    if layer in layers:
        return layers.index(layer)
    if nearest and layers:
        return int(np.argmin([abs(x - layer) for x in layers]))
    return None


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, float)
    return x / max(float(np.linalg.norm(x)), EPS)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(unit(a), unit(b)))


def per_step_token_mean(arr: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None:
        return out
    a = np.asarray(arr, float)
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = min(len(a), int(hi0) - a0 + 1)
        if hi > lo:
            out[t] = float(np.nanmean(a[lo:hi]))
    return out


def per_step_token_var(arr: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None:
        return out
    a = np.asarray(arr, float)
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = min(len(a), int(hi0) - a0 + 1)
        if hi > lo:
            out[t] = float(np.nanvar(a[lo:hi]))
    return out


def per_step_offset_mean(arr: Optional[np.ndarray], offsets: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    """Mean of a strided token trace whose entries are indexed by response offsets."""
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None or offsets is None:
        return out
    a = np.asarray(arr, float).reshape(-1)
    off = np.asarray(offsets, int).reshape(-1)
    m = np.isfinite(a)
    a = a[m]
    off = off[m]
    if len(a) == 0 or len(off) != len(a):
        return out
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = max(lo, int(hi0) - a0)
        keep = (off >= lo) & (off <= hi)
        if keep.any():
            out[t] = float(np.nanmean(a[keep]))
    return out


def per_step_offset_var(arr: Optional[np.ndarray], offsets: Optional[np.ndarray], ranges: np.ndarray) -> np.ndarray:
    """Variance of a strided token trace whose entries are indexed by response offsets."""
    T = len(ranges)
    out = np.full(T, np.nan)
    if arr is None or offsets is None:
        return out
    a = np.asarray(arr, float).reshape(-1)
    off = np.asarray(offsets, int).reshape(-1)
    m = np.isfinite(a)
    a = a[m]
    off = off[m]
    if len(a) == 0 or len(off) != len(a):
        return out
    a0 = int(ranges[0, 0])
    for t, (lo0, hi0) in enumerate(ranges):
        lo = max(0, int(lo0) - a0)
        hi = max(lo, int(hi0) - a0)
        keep = (off >= lo) & (off <= hi)
        if keep.any():
            out[t] = float(np.nanvar(a[keep]))
    return out


def delta(x: np.ndarray, *, reverse: bool = False) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    if len(v) >= 2:
        d = v[1:] - v[:-1]
        out[1:] = -d if reverse else d
    return out


def causal_z(x: np.ndarray, *, warmup: int = 2) -> np.ndarray:
    v = np.asarray(x, float)
    out = np.full(len(v), np.nan)
    for t in range(warmup, len(v)):
        hist = v[:t]
        hist = hist[np.isfinite(hist)]
        if len(hist) >= 2 and np.isfinite(v[t]):
            out[t] = (v[t] - hist.mean()) / (hist.std() + EPS)
    return out


def load_chains(npz_path: str, *, layer: int, max_chains: int = 0) -> Tuple[List[Chain], Dict[str, object]]:
    z = np.load(npz_path, allow_pickle=True)
    ges = z["gold_error_step"].astype(int)
    groups = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(ges))
    SR = z["step_token_ranges"]
    ST = z["steps_text"] if "steps_text" in z.files else None

    SC = z["stepcloud"] if "stepcloud" in z.files else None
    cn = [str(x) for x in z["cloud_feature_names"]] if "cloud_feature_names" in z.files else []
    cloud_layers = [int(x) for x in z["layers_used"]] if "layers_used" in z.files else []
    ci = layer_index(cloud_layers, layer)

    SG = z["stepgeom"] if "stepgeom" in z.files else None
    gn = [str(x) for x in z["geom_feature_names"]] if "geom_feature_names" in z.files else []
    gi = layer_index(cloud_layers, layer) if cloud_layers else None

    SA = z["stepattn"] if "stepattn" in z.files else None
    an = [str(x) for x in z["attn_names"]] if "attn_names" in z.files else []
    has_attn = bool(np.asarray(z["attn_stored"]).item()) if SA is not None and "attn_stored" in z.files else SA is not None

    UD = z["tok_U_D"] if "tok_U_D" in z.files else None
    UC = z["tok_U_C"] if "tok_U_C" in z.files else None
    UE = z["tok_U_E"] if "tok_U_E" in z.files else None
    UEO = z["tok_U_E_offsets"] if "tok_U_E_offsets" in z.files else None

    SV = z["stepvec"] if "stepvec" in z.files else None
    qvec = z["qvec"] if "qvec" in z.files else None
    sv_layers = [int(x) for x in z["sv_layers"]] if "sv_layers" in z.files else []
    svi = layer_index(sv_layers, layer, nearest=True) if SV is not None and sv_layers else None

    N = len(ges) if not max_chains else min(max_chains, len(ges))
    chains: List[Chain] = []
    missing = {"stepcloud": 0, "stepgeom": 0, "stepattn": 0, "stepvec": 0}

    for i in range(N):
        rng = np.asarray(SR[i], int)
        if rng.ndim != 2 or len(rng) == 0:
            continue
        T = len(rng)
        n_tok, pos = step_ranges(rng)
        feats: Dict[str, np.ndarray] = {"n_tok": n_tok, "logN": np.log1p(n_tok), "pos": pos}

        if SC is not None and ci is not None and SC[i] is not None:
            sc = np.asarray(SC[i], float)
            for name in ("resultant", "resultant_unif", "resultant_bulk", "coherence", "cloud_D", "cloud_V", "cloud_C"):
                if name in cn and sc.ndim == 3 and sc.shape[0] >= T:
                    feats[name] = sc[:T, ci, cn.index(name)]
        else:
            missing["stepcloud"] += 1

        if SG is not None and gi is not None and SG[i] is not None:
            sg = np.asarray(SG[i], float)
            for name in ("norm", "pr", "ae", "ed_half", "e50", "e90", "ae_robust", "anom_k5", "anom_k10"):
                if name in gn and sg.ndim == 3 and sg.shape[0] >= T:
                    feats[f"geom_{name}"] = sg[:T, gi, gn.index(name)]
        else:
            missing["stepgeom"] += 1

        if has_attn and SA is not None and ci is not None and SA[i] is not None:
            sa = np.asarray(SA[i], float)
            for name in an:
                if sa.ndim == 3 and sa.shape[0] >= T:
                    feats[f"attn_{name}"] = sa[:T, ci, an.index(name)]
        else:
            missing["stepattn"] += 1

        ud = np.asarray(UD[i], float) if UD is not None and UD[i] is not None else None
        uc = np.asarray(UC[i], float) if UC is not None and UC[i] is not None else None
        ue = np.asarray(UE[i], float) if UE is not None and UE[i] is not None else None
        ueo = np.asarray(UEO[i], int) if UEO is not None and UEO[i] is not None else None
        feats["U_D_mean"] = per_step_token_mean(ud, rng)
        feats["U_D_var"] = per_step_token_var(ud, rng)
        feats["U_C_mean"] = per_step_token_mean(uc, rng)
        feats["U_C_var"] = per_step_token_var(uc, rng)
        if ue is not None and ueo is not None and len(ue) == len(ueo):
            feats["U_E_mean"] = per_step_offset_mean(ue, ueo, rng)
            feats["U_E_var"] = per_step_offset_var(ue, ueo, rng)
        else:
            feats["U_E_mean"] = per_step_token_mean(ue, rng)
            feats["U_E_var"] = per_step_token_var(ue, rng)
        feats["unc_entropy"] = feats["U_D_mean"]
        feats["unc_committal"] = feats["U_C_mean"]
        feats["unc_epistemic"] = feats["U_E_mean"]

        if SV is not None and qvec is not None and svi is not None and SV[i] is not None:
            sv = np.asarray(SV[i], float)
            qv = np.asarray(qvec[i], float) if np.asarray(qvec[i]).ndim == 2 else np.asarray(qvec, float)
            if sv.ndim == 3 and sv.shape[0] >= T:
                dirs = np.array([unit(sv[t, svi]) for t in range(T)])
                feats["q_align"] = np.array([cosine(dirs[t], qv[svi]) for t in range(T)])
                jump = np.full(T, np.nan)
                for t in range(1, T):
                    jump[t] = 1.0 - cosine(dirs[t], dirs[t - 1])
                feats["step_direction_jump"] = jump
                if svi > 0 and sv.shape[1] > svi and qv.ndim == 2 and qv.shape[0] > svi:
                    current = np.asarray(sv[:T, svi], float)
                    previous = np.asarray(sv[:T, svi - 1], float)
                    raw_relative = np.full(T, np.nan)
                    prompt_conditioned = np.full(T, np.nan)
                    depth_rewire = np.full(T, np.nan)
                    q_current = np.asarray(qv[svi], float)
                    q_previous = np.asarray(qv[svi - 1], float)
                    prompt_ready = bool(
                        np.all(np.isfinite(q_current))
                        and np.all(np.isfinite(q_previous))
                    )
                    for t in range(T):
                        h_cur = current[t]
                        h_prev = previous[t]
                        if not (
                            np.all(np.isfinite(h_cur))
                            and np.all(np.isfinite(h_prev))
                        ):
                            continue
                        raw_delta = h_cur - h_prev
                        raw_scale = 0.5 * (
                            np.linalg.norm(h_cur) + np.linalg.norm(h_prev)
                        )
                        raw_relative[t] = float(
                            np.linalg.norm(raw_delta) / max(float(raw_scale), EPS)
                        )
                        depth_rewire[t] = 1.0 - cosine(h_cur, h_prev)
                        if prompt_ready:
                            centered_cur = h_cur - q_current
                            centered_prev = h_prev - q_previous
                            conditioned_delta = centered_cur - centered_prev
                            conditioned_scale = 0.5 * (
                                np.linalg.norm(centered_cur)
                                + np.linalg.norm(centered_prev)
                            )
                            prompt_conditioned[t] = float(
                                np.linalg.norm(conditioned_delta)
                                / max(float(conditioned_scale), EPS)
                            )
                    feats["depth_band_update_relative_norm"] = raw_relative
                    feats["depth_band_prompt_conditioned_norm"] = prompt_conditioned
                    feats["depth_band_state_rewire"] = depth_rewire
        else:
            missing["stepvec"] += 1

        if ST is not None and i < len(ST):
            txt = list(ST[i])
            dens = np.full(T, np.nan)
            for t in range(min(T, len(txt))):
                s = str(txt[t])
                dens[t] = 1.0 - sum(ch.isalpha() for ch in s) / max(1, len(s))
            feats["text_density"] = dens

        base_names = list(feats.keys())
        for name in base_names:
            if name in ("n_tok", "logN", "pos"):
                continue
            v = np.asarray(feats[name], float)
            if name in ("resultant", "coherence", "resultant_unif", "resultant_bulk", "q_align"):
                feats[f"d_{name}_bad"] = delta(v, reverse=True)
                feats[f"cz_{name}_bad"] = causal_z(-v)
            else:
                feats[f"d_{name}"] = delta(v)
                feats[f"cz_{name}"] = causal_z(v)

        if "resultant" in feats and "U_D_mean" in feats:
            r = np.asarray(feats["resultant"], float)
            u = np.asarray(feats["U_D_mean"], float)
            feats["confident_geom_bad"] = (-r) * (-u)
            feats["uncertain_geom_bad"] = (-r) * u
        if "resultant" in feats and "attn_q_frac" in feats:
            feats["flow_geometry_mismatch"] = (-feats["resultant"]) * (-feats["attn_q_frac"])
        if "q_align" in feats and "resultant" in feats:
            feats["coherent_anchor_drift"] = feats["resultant"] * (-feats["q_align"])

        chains.append(
            Chain(
                idx=i,
                group=int(groups[i]),
                gold=int(ges[i]),
                correct=bool(ges[i] < 0),
                features=feats,
                n_steps=T,
            )
        )

    meta = {
        "npz": npz_path,
        "layer": layer,
        "n_chains_seen": N,
        "cloud_layers": cloud_layers,
        "sv_layers": sv_layers,
        "depth_band": (
            [int(sv_layers[svi - 1]), int(sv_layers[svi])]
            if svi is not None and svi > 0 and len(sv_layers) > svi
            else None
        ),
        "cloud_features": cn,
        "geom_features": gn,
        "attn_features": an,
        "has_attention": bool(has_attn),
        "has_stepvec_qvec": bool(SV is not None and qvec is not None and svi is not None),
        "has_uncertainty": {
            "tok_U_D": bool(UD is not None),
            "tok_U_C": bool(UC is not None),
            "tok_U_E": bool(UE is not None),
            "tok_U_E_offsets": bool(UEO is not None),
        },
        "missing": missing,
    }
    return chains, meta


def _object_array(xs: Sequence[object]) -> np.ndarray:
    out = np.empty(len(xs), dtype=object)
    out[:] = list(xs)
    return out


def make_selftest_npz(path: str, *, n_chains: int = 80, layer: int = 14, seed: int = 7) -> None:
    """Build a small synthetic full_*.npz-like file with known failure dynamics."""
    rng = np.random.default_rng(seed)
    layers = np.array([0, layer, layer + 8], dtype=int)
    sv_layers = np.array([layer], dtype=int)
    cloud_names = np.array(
        ["resultant", "resultant_unif", "resultant_bulk", "coherence", "cloud_D", "cloud_V", "cloud_C"],
        dtype=object,
    )
    geom_names = np.array(["norm", "pr", "ae", "ed_half", "e50", "e90", "ae_robust", "anom_k5", "anom_k10"], dtype=object)
    attn_names = np.array(["q_frac", "sink_frac", "attn_entropy"], dtype=object)

    gold, groups = [], []
    step_ranges_all, stepcloud, stepgeom, tok_ud, tok_uc, tok_ue, tok_ue_offsets = [], [], [], [], [], [], []
    stepattn, stepvec, steps_text, qvecs = [], [], [], []
    d = 24

    for i in range(n_chains):
        T = int(rng.integers(5, 9))
        is_error = (i % 5) in (0, 2)
        g = int(rng.integers(2, T - 1)) if is_error else -1
        gold.append(g)
        groups.append(i)

        lens = rng.integers(4, 10, size=T)
        lo = np.cumsum(np.r_[0, lens[:-1]])
        hi = lo + lens - 1
        ranges = np.stack([lo, hi], axis=1).astype(int)
        step_ranges_all.append(ranges)

        kappa = 0.76 + 0.04 * rng.normal(size=T)
        coh = 0.62 + 0.05 * rng.normal(size=T)
        ud_step = 0.22 + 0.04 * rng.normal(size=T)
        uc_step = 0.18 + 0.04 * rng.normal(size=T)
        q_frac = 0.42 + 0.04 * rng.normal(size=T)
        sink = 0.16 + 0.03 * rng.normal(size=T)
        attn_entropy = 1.0 + 0.08 * rng.normal(size=T)
        q_align = 0.78 + 0.04 * rng.normal(size=T)
        if is_error:
            kappa[g] -= 0.28
            coh[g] -= 0.22
            ud_step[g] += 0.38
            uc_step[g] += 0.22
            q_frac[g] -= 0.23
            sink[g] += 0.10
            attn_entropy[g] += 0.34
            q_align[g] -= 0.42
            if g + 1 < T:
                kappa[g + 1 :] -= 0.10
                ud_step[g + 1 :] += 0.12

        sc = np.zeros((T, len(layers), len(cloud_names)), float)
        sg = np.zeros((T, len(layers), len(geom_names)), float)
        sa = np.zeros((T, len(layers), len(attn_names)), float)
        for li, _ly in enumerate(layers):
            jitter = 0.01 * li
            sc[:, li, 0] = kappa - jitter
            sc[:, li, 1] = kappa - 0.03 - jitter
            sc[:, li, 2] = kappa - 0.05 - jitter
            sc[:, li, 3] = coh - jitter
            sc[:, li, 4] = 1.0 / np.clip(kappa, 0.05, 1.0)
            sc[:, li, 5] = 0.2 + 0.1 * (1 - kappa)
            sc[:, li, 6] = 0.3 + 0.1 * (1 - coh)

            sg[:, li, 0] = 8.0 + 0.3 * rng.normal(size=T)
            sg[:, li, 1] = 5.0 + 1.2 * (1 - kappa)
            sg[:, li, 2] = 0.25 + 0.6 * (1 - kappa)
            sg[:, li, 3] = sg[:, li, 2] + 0.02
            sg[:, li, 4] = 0.5 + 0.4 * (1 - kappa)
            sg[:, li, 5] = 0.7 + 0.4 * (1 - kappa)
            sg[:, li, 6] = sg[:, li, 2] + 0.01
            sg[:, li, 7] = 0.2 + 0.5 * (1 - kappa)
            sg[:, li, 8] = 0.3 + 0.5 * (1 - kappa)

            sa[:, li, 0] = q_frac - jitter
            sa[:, li, 1] = sink + jitter
            sa[:, li, 2] = attn_entropy + jitter

        q = unit(rng.normal(size=d))
        qv = q[None, :]
        sv = np.zeros((T, len(sv_layers), d), float)
        prev_dir = q
        for t in range(T):
            if is_error and t == g:
                off = unit(rng.normal(size=d))
                off = unit(off - np.dot(off, q) * q)
                cur = unit(0.35 * q + 0.65 * off + 0.05 * rng.normal(size=d))
            else:
                cur = unit(q + 0.12 * rng.normal(size=d))
            if t > 0 and not (is_error and t == g):
                cur = unit(0.75 * prev_dir + 0.25 * cur)
            sv[t, 0] = cur
            prev_dir = cur

        ud_tok = np.concatenate([rng.normal(ud_step[t], 0.025, size=int(lens[t])) for t in range(T)])
        uc_tok = np.concatenate([rng.normal(uc_step[t], 0.025, size=int(lens[t])) for t in range(T)])
        ue_tok_full = np.concatenate([rng.normal(ud_step[t] * 1.4, 0.035, size=int(lens[t])) for t in range(T)])
        ue_off = np.arange(0, len(ue_tok_full), 2, dtype=np.int32)

        stepcloud.append(sc)
        stepgeom.append(sg)
        stepattn.append(sa)
        stepvec.append(sv)
        qvecs.append(qv)
        tok_ud.append(ud_tok)
        tok_uc.append(uc_tok)
        tok_ue.append(ue_tok_full[ue_off])
        tok_ue_offsets.append(ue_off)
        steps_text.append(np.array([f"step {t}: synthetic reasoning state" for t in range(T)], dtype=object))

    np.savez_compressed(
        path,
        gold_error_step=np.array(gold, dtype=int),
        problem_ids=np.array(groups, dtype=int),
        step_token_ranges=_object_array(step_ranges_all),
        steps_text=_object_array(steps_text),
        stepcloud=_object_array(stepcloud),
        cloud_feature_names=cloud_names,
        layers_used=layers,
        stepgeom=_object_array(stepgeom),
        geom_feature_names=geom_names,
        stepattn=_object_array(stepattn),
        attn_names=attn_names,
        attn_stored=np.array(True),
        tok_U_D=_object_array(tok_ud),
        tok_U_C=_object_array(tok_uc),
        tok_U_E=_object_array(tok_ue),
        tok_U_E_offsets=_object_array(tok_ue_offsets),
        stepvec=_object_array(stepvec),
        qvec=np.asarray(qvecs, float),
        sv_layers=sv_layers,
    )
