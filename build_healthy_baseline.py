"""Build a HEALTHY activation baseline (per-layer, per-dimension mean/std) from
CORRECT reasoning, for the anchor-faithful participation in 10/01.

Anchor (canonical): error reasoning = activation diffuse over MORE dims;
correct reasoning = concentrated in a low-dim non-degenerate subset. To count
"how many dims are ABNORMALLY active", abnormal must be defined RELATIVE to
correct reasoning. So we standardize each step vector per-dimension against this
healthy baseline BEFORE computing participation:  z' = (z - mu_l)/(sigma_l+eps),
then PR/AE(z') = "how many dims deviate from healthy".

This script forwards CORRECT solutions (ProcessBench label==-1, i.e. solutions
whose every step is annotated correct) through the model and accumulates, per
sampled layer, the per-dimension mean and std of the step-range token hidden
states (optionally projected to the reasoning subspace, to match extraction).

Leakage note: this baseline is built from ProcessBench's correct SOLUTION TEXT,
which is DISJOINT from the freshly SAMPLED solutions scored in 10 -> no leakage
between baseline and evaluation. (It is still the same model's activations.)

Output: data/healthy_baseline_<model>.npz with whiten_mu/whiten_sigma (L_sub,d),
whiten_layers, reasoning_subspace_used, d. Pass it to 10 via --whiten_baseline.
"""

from __future__ import annotations

import argparse
import importlib.util
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_ex = _load_local_module("01_extract_spectral_field.py", "extract01")


@torch.no_grad()
def accumulate_chain(model, tokenizer, prompt, response, steps, device,
                     layer_indices, V_R, max_seq_len, acc):
    """Add one correct solution's step-range token hidden states into `acc`
    (dict: li -> {'sum','sumsq','n'}). acc rows are lazily created per layer."""
    ranges = _ex.find_step_token_ranges(tokenizer, prompt, response, steps)
    if len(ranges) < 1:
        return
    enc = tokenizer(prompt + response, return_tensors="pt",
                    truncation=True, max_length=max_seq_len).to(device)
    seq_len = enc["input_ids"].shape[1]
    safe = [(a, b) for (a, b) in ranges if b < seq_len and b - a + 1 >= 2]
    if not safe:
        return
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    for li, l in enumerate(layer_indices):
        H_l = hs[l][0].float().cpu().numpy()                  # (seq_len, d)
        H = np.concatenate([H_l[a:b + 1] for (a, b) in safe], axis=0)
        if V_R is not None:
            H = _ex.project_to_reasoning(H, V_R)              # (n, d_R)
        H = H.astype(np.float64)
        if li not in acc:
            d = H.shape[1]
            acc[li] = {"sum": np.zeros(d), "sumsq": np.zeros(d), "n": 0}
        acc[li]["sum"] += H.sum(axis=0)
        acc[li]["sumsq"] += (H ** 2).sum(axis=0)
        acc[li]["n"] += H.shape[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--dataset", default="data/hf_datasets/ProcessBench")
    ap.add_argument("--subset", default="gsm8k")
    ap.add_argument("--n_baseline", type=int, default=150,
                    help="number of CORRECT solutions to build the baseline from")
    ap.add_argument("--layers", default="all")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    # default OFF (full space) to match the clean multisample default
    ap.add_argument("--reasoning_subspace", action="store_true",
                    help="project to HARP subspace (must MATCH the eval setting).")
    ap.add_argument("--reasoning_mode", default="energy", choices=["energy", "dim_ratio"])
    ap.add_argument("--reasoning_threshold", type=float, default=0.95)
    ap.add_argument("--unembedding_cache", default="data/unembedding_svd.npz")
    ap.add_argument("--output", default="data/healthy_baseline.npz")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device)
    model.eval()

    V_R = None
    if args.reasoning_subspace:
        cache_path = args.unembedding_cache
        if cache_path:
            tag = os.path.basename(args.model.rstrip("/")).replace("/", "_")
            root, ext = os.path.splitext(cache_path)
            cache_path = f"{root}.{tag}{ext}"
        V_R, _ = _ex.prepare_reasoning_subspace(
            model, mode=args.reasoning_mode,
            threshold=args.reasoning_threshold, cache_path=cache_path)

    layer_indices = None if args.layers == "all" else \
        [int(x) for x in args.layers.split(",") if x.strip()]

    print(f"Loading correct (label==-1) solutions from {args.dataset}/{args.subset} ...")
    ds = load_dataset(args.dataset, split=args.subset)
    correct = [ex for ex in ds if int(ex.get("label", -1)) == -1]
    n = min(args.n_baseline, len(correct))
    print(f"  {len(correct)} correct solutions available; using {n}")

    # discover layer_indices on the first usable chain if 'all'
    if layer_indices is None:
        # probe: one forward to count layers
        ex0 = correct[0]
        p0, r0, s0 = _ex.build_prompt_and_response(ex0)
        enc0 = tokenizer(p0 + r0, return_tensors="pt", truncation=True,
                         max_length=args.max_seq_len).to(device)
        with torch.no_grad():
            n_layers = len(model(**enc0, output_hidden_states=True).hidden_states)
        layer_indices = list(range(n_layers))
    print(f"  sampling {len(layer_indices)} layers, "
          f"reasoning_subspace={V_R is not None}")

    acc = {}
    used = 0
    for ex in tqdm(correct[:n], desc="baseline"):
        prompt, response, steps = _ex.build_prompt_and_response(ex)
        if prompt is None:
            continue
        try:
            accumulate_chain(model, tokenizer, prompt, response, steps, device,
                             layer_indices, V_R, args.max_seq_len, acc)
            used += 1
        except Exception as e:
            print(f"  warn: skip a chain: {e}")

    if not acc:
        raise SystemExit("No tokens accumulated; check the dataset.")

    L = len(layer_indices)
    d = acc[0]["sum"].size
    mu = np.full((L, d), np.nan, dtype=np.float64)
    sigma = np.full((L, d), np.nan, dtype=np.float64)
    for li in range(L):
        if li in acc and acc[li]["n"] > 1:
            n_li = acc[li]["n"]
            m = acc[li]["sum"] / n_li
            var = acc[li]["sumsq"] / n_li - m ** 2
            mu[li] = m
            sigma[li] = np.sqrt(np.clip(var, 0.0, None))

    np.savez(args.output,
             whiten_mu=mu.astype(np.float32),
             whiten_sigma=sigma.astype(np.float32),
             whiten_layers=np.asarray(layer_indices, dtype=np.int32),
             reasoning_subspace_used=np.array(V_R is not None),
             d=np.array(d),
             n_chains=np.array(used),
             n_tokens_per_layer=np.array([acc.get(li, {"n": 0})["n"] for li in range(L)]))
    print(f"\nBuilt healthy baseline from {used} correct solutions, "
          f"d={d}, layers={L}.")
    print(f"  median per-dim sigma (layer 0) = {np.nanmedian(sigma[0]):.4f}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
