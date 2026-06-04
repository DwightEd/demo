"""Step 25: open the failure-probe weight w -- what did it actually learn?

The within-problem failure probe (12) reaches ~0.71 but we never looked at its DIRECTION.
Here we retrain that averaged-vector probe, recover its weight w, and interrogate it:

  un-standardize: the probe lives in StandardScaler space, score = w_std . ((x-mu)/sigma),
      so the direction in RAW activation space is  w_raw = w_std / sigma  (else logit-lens
      and neuron-sparsity are meaningless).

  (A) LOGIT LENS  w_raw @ W_U^T  -> top / bottom vocabulary tokens this direction promotes
      / suppresses (does failure point at hesitation / contradiction / negation words?).
      Only meaningful on a LATE layer (residual aligned with unembedding) -> use deep/last.
      Requires --model (loads the HF unembedding matrix; no forward pass needed).

  (B) GEOMETRY  cos(w_raw, candidate directions): class-mean difference (incorrect-correct),
      mean activation and per-dim sigma (massive-activation / magnitude axes). Tells whether
      the probe is just the mean-shift, or rides the high-magnitude dims, or something else.

  (C) SPARSITY  participation ratio of w_raw^2 (effective # neurons), # neurons holding
      50%/90% of |w| mass -> axis-aligned sparse (tens of neurons) vs distributed.
"""

from __future__ import annotations

import argparse
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def band_indices(L_sub, band):
    if band == "all":
        return np.arange(L_sub)
    if band == "deep":
        return np.arange(int(L_sub * 0.6), L_sub)
    if band == "mid":
        return np.arange(int(L_sub * 0.3), int(L_sub * 0.7))
    if band == "last":
        return np.array([L_sub - 1])
    return np.array([int(x) for x in band.split(",") if x.strip()])


def cosv(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--mode", default="step_exp")
    ap.add_argument("--layer_band", default="deep", help="deep/last cleanest for logit-lens")
    ap.add_argument("--late_lo", type=float, default=0.6)
    ap.add_argument("--C", type=float, default=0.05)
    ap.add_argument("--model", default=None, help="HF model path -> enables logit lens (A)")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--output", default="data/probe_interpret.npz")
    args = ap.parse_args()

    data = np.load(args.input, allow_pickle=True)
    VEC = data[f"sv_vec_{args.mode}"]
    problem_ids = data["problem_ids"].astype(int)
    y = (data["is_correct"].astype(int) == 0).astype(int)
    N = len(VEC)
    L_sub = np.asarray(VEC[0]).shape[1]; d = np.asarray(VEC[0]).shape[2]
    cols = band_indices(L_sub, args.layer_band)
    layers_used = data["layers_used"].astype(int) if "layers_used" in data else np.arange(L_sub)

    X = np.full((N, d), np.nan)
    for i in range(N):
        V = np.asarray(VEC[i], dtype=np.float64)[:, cols, :]
        with np.errstate(invalid="ignore"):
            P = np.nanmean(V, axis=1)
        T = P.shape[0]
        fr = (np.arange(T) / (T - 1)) if T > 1 else np.array([0.0])
        m = fr >= args.late_lo
        if not m.any(): m = fr >= fr.max()
        with np.errstate(invalid="ignore"):
            X[i] = np.nanmean(P[m], axis=0)
    ok = np.isfinite(X).all(1)
    X, y = X[ok], y[ok]
    N = len(y)
    print(f"Loaded {N} solutions; d={d}, band={args.layer_band} -> model layers "
          f"{layers_used[cols].tolist() if len(layers_used) >= L_sub else cols.tolist()}; "
          f"{int(y.sum())} incorrect / {int((1-y).sum())} correct")

    # train the averaged-vector failure probe on ALL data (we want the direction)
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(C=args.C, max_iter=3000, class_weight="balanced").fit(sc.transform(X), y)
    w_std = clf.coef_.ravel()
    w_raw = w_std / sc.scale_                       # direction in RAW activation space
    u_raw = w_raw / np.linalg.norm(w_raw)

    # (B) geometry
    diff_means = X[y == 1].mean(0) - X[y == 0].mean(0)
    mean_act = X.mean(0); sigma = X.std(0)
    print(f"\n=== (B) what direction is w? cos(w_raw, .) ===")
    print(f"  class-mean diff (incorrect-correct) = {cosv(w_raw, diff_means):+.3f}  "
          f"(high -> probe ~ simple mean-shift)")
    print(f"  mean activation (massive-act axis)  = {cosv(w_raw, mean_act):+.3f}")
    print(f"  per-dim sigma (high-variance dims)  = {cosv(w_raw, sigma):+.3f}")

    # (C) sparsity
    aw = np.abs(w_raw); order = np.argsort(aw)[::-1]; cum = np.cumsum(aw[order]) / aw.sum()
    pr_neurons = float((w_raw ** 2).sum() ** 2 / ((w_raw ** 4).sum() + 1e-12))
    n50 = int(np.searchsorted(cum, 0.5) + 1); n90 = int(np.searchsorted(cum, 0.9) + 1)
    print(f"\n=== (C) sparsity of w_raw (d={d}) ===")
    print(f"  participation ratio (effective # neurons) = {pr_neurons:.1f}")
    print(f"  # neurons for 50% of |w| mass = {n50}   for 90% = {n90}")
    print(f"  -> {'axis-aligned SPARSE' if n90 < 100 else 'DISTRIBUTED'} "
          f"({'tens' if n50 < 50 else 'many'} of neurons carry it)")

    top_tokens = bot_tokens = None
    if args.model:
        print(f"\n=== (A) logit lens  w_raw @ W_U  (model {args.model}) ===")
        import importlib.util, os, torch
        spec = importlib.util.spec_from_file_location(
            "ex01", os.path.join(os.path.dirname(os.path.abspath(__file__)), "01_extract_spectral_field.py"))
        ex01 = importlib.util.module_from_spec(spec); spec.loader.exec_module(ex01)
        from transformers import AutoTokenizer, AutoModelForCausalLM
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float32, device_map="cpu")
        W_U = ex01.get_unembedding_matrix(model).float().cpu().numpy()      # (V, d)
        logits = W_U @ u_raw                                                # (V,)
        ti = np.argsort(logits)[::-1][:args.topk]; bi = np.argsort(logits)[:args.topk]
        top_tokens = [tok.decode([int(t)]) for t in ti]
        bot_tokens = [tok.decode([int(t)]) for t in bi]
        print(f"  TOP (failure direction promotes): {top_tokens}")
        print(f"  BOTTOM (suppresses):             {bot_tokens}")
    else:
        print("\n  (A) logit lens skipped (pass --model <path> to enable).")

    np.savez(args.output, w_raw=w_raw.astype(np.float32),
             cos_diffmeans=np.array(cosv(w_raw, diff_means)),
             cos_meanact=np.array(cosv(w_raw, mean_act)),
             cos_sigma=np.array(cosv(w_raw, sigma)),
             pr_neurons=np.array(pr_neurons), n50=np.array(n50), n90=np.array(n90),
             top_tokens=np.array(top_tokens or [], dtype=object),
             bot_tokens=np.array(bot_tokens or [], dtype=object),
             band=np.array(args.layer_band))
    print(f"\nSaved -> {args.output}")


if __name__ == "__main__":
    main()
