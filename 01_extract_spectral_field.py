"""Step 1: Extract the per-trajectory (step × layer) spectral field on ProcessBench.

For each reasoning chain we:
  1. Tokenize prompt + response and align each pre-parsed step with its token range.
  2. Run one forward pass with output_hidden_states=True to get hidden states at
     all L+1 layers (embedding + L transformer blocks).
  3. Project the per-token hidden states onto the *reasoning subspace* induced
     by SVD of the unembedding matrix W_U (HARP, Hu et al. ICLR 2026). The
     intuition is that W_U @ h gives the next-token logits, so directions
     aligned with the top singular vectors of W_U carry the semantic
     prediction content; directions in the kernel-like complement carry the
     intermediate computation that does not directly produce the current
     token. Analyzing token-cloud structure inside the reasoning subspace
     isolates the latter.
  4. For every (step j, layer l), reduce the projected token cloud
     H_j^(l) V_R ∈ R^{n_j × d_R} to three scalars: effective rank D, spectral
     energy V, and top concentration C. Stack across steps and layers to get
     three (T, L+1) matrices M_D, M_V, M_C — this is what downstream
     analysis consumes.

The reasoning subspace projection can be turned off via --no_reasoning_subspace
(then the raw hidden states are analyzed as before; this is the v17 baseline).

Usage:
    python 01_extract_spectral_field.py \
        --model /path/to/llama-3.1-8b \
        --dataset Qwen/ProcessBench \
        --subset gsm8k \
        --n_correct 50 \
        --n_error 50 \
        --output data/spectral_field.npz
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from utils import (
    find_step_token_ranges,
    step_layer_spectral_summary,
    compute_unembedding_svd,
    select_reasoning_subspace,
    project_to_reasoning,
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_processbench_subset(dataset_name, subset, n_correct, n_error, seed=42):
    """Stratified sample of ProcessBench by label sign (-1 = all correct)."""
    print(f"Loading {dataset_name} split={subset} ...")
    ds = load_dataset(dataset_name, split=subset)

    correct = [ex for ex in ds if ex.get("label", -1) == -1]
    error = [ex for ex in ds if ex.get("label", -1) >= 0]
    print(f"  dataset: {len(correct)} correct, {len(error)} error")

    rng = np.random.default_rng(seed)
    n_c = min(n_correct, len(correct))
    n_e = min(n_error, len(error))
    correct_idx = rng.choice(len(correct), size=n_c, replace=False)
    error_idx = rng.choice(len(error), size=n_e, replace=False)
    print(f"  sampled: {n_c} correct, {n_e} error")
    return [correct[i] for i in correct_idx], [error[i] for i in error_idx]


def build_prompt_and_response(example):
    """ProcessBench → (prompt, response, steps)."""
    problem = example["problem"]
    steps = example.get("steps", [])
    if not steps:
        return None, None, None
    response = "\n\n".join(steps)
    prompt = f"Problem: {problem}\n\nSolution:\n\n"
    return prompt, response, steps


# ---------------------------------------------------------------------------
# Reasoning subspace from unembedding matrix
# ---------------------------------------------------------------------------

def get_unembedding_matrix(model) -> torch.Tensor:
    """Locate the unembedding (lm_head) weight in a HuggingFace causal-LM.

    Returns a 2D tensor of shape (V, d). Works for tied-weight and untied
    variants by reading model.get_output_embeddings().weight when present.
    """
    out_emb = model.get_output_embeddings()
    if out_emb is None:
        # Fall back to common attribute names.
        for attr in ("lm_head", "output", "embed_out"):
            if hasattr(model, attr):
                out_emb = getattr(model, attr)
                break
    if out_emb is None or not hasattr(out_emb, "weight"):
        raise RuntimeError(
            "Could not locate the unembedding weight on the model. "
            "Pass --no_reasoning_subspace to skip projection."
        )
    return out_emb.weight.detach()


def prepare_reasoning_subspace(model, mode: str, threshold: float,
                               cache_path: str | None):
    """Compute the reasoning subspace basis V_R from W_U.

    Returns:
        V_R: numpy (d, d_R) basis with columns as reasoning directions.
        meta: dict with cutoff information for logging.
    """
    print("Preparing reasoning subspace via unembedding SVD ...")
    W_U = get_unembedding_matrix(model)
    Vt, S = compute_unembedding_svd(W_U, cache_path=cache_path)
    V_R, meta = select_reasoning_subspace(Vt, S, mode=mode, threshold=threshold)
    print(f"  W_U shape: {tuple(W_U.shape)}  (V × d)")
    print(f"  d_total = {meta['d_total']}   d_semantic = {meta['d_semantic']}"
          f"   d_reasoning = {meta['d_reasoning']}")
    print(f"  energy fraction in reasoning subspace = "
          f"{meta['energy_in_reasoning']:.4f}")
    return V_R, meta


# ---------------------------------------------------------------------------
# Spectral field for one trajectory
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_spectral_field(
    model, tokenizer, prompt, response, steps,
    device, layer_indices=None, max_seq_len=4096,
    V_R: np.ndarray | None = None,
    rank_mode: str = "full",
    rank_k: int | None = None,
    rank_threshold: float = 0.95,
):
    """Run one forward pass and reduce each (step, layer) token cloud to (D, V, C).

    If V_R is provided, project each token-cloud onto the reasoning subspace
    before computing the spectral summary.

    The effective rank D is computed by `step_layer_spectral_summary`. Its
    rank_mode argument selects whether to use the full spectrum (default) or
    a truncated form. See `effective_rank_truncated` for the available modes.

    Returns:
        M_D, M_V, M_C: (T, L_sub) float arrays where L_sub = len(layer_indices).
                       Rows in original step order; NaN rows are dropped.
        kept_steps:    indices (in the original 0..T-1) of steps actually kept.
        layers_used:   list of int layer indices that were sampled.
    """
    ranges = find_step_token_ranges(tokenizer, prompt, response, steps)
    if len(ranges) < 3:
        return None, None, None, None, None

    encoding = tokenizer(
        prompt + response,
        return_tensors="pt",
        truncation=True,
        max_length=max_seq_len,
    ).to(device)
    seq_len = encoding["input_ids"].shape[1]

    # Keep only steps whose token range fits inside the truncated sequence.
    safe = [(j, a, b) for j, (a, b) in enumerate(ranges) if b < seq_len and b - a + 1 >= 2]
    if len(safe) < 3:
        return None, None, None, None, None

    outputs = model(**encoding, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d) tensors
    n_layers_total = len(hidden_states)

    if layer_indices is None:
        layer_indices = list(range(n_layers_total))
    layer_indices = [l for l in layer_indices if 0 <= l < n_layers_total]
    L_sub = len(layer_indices)

    T_eff = len(safe)
    M_D = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_V = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_C = np.full((T_eff, L_sub), np.nan, dtype=np.float64)

    for li, l in enumerate(layer_indices):
        H_l = hidden_states[l][0].float().cpu().numpy()  # (seq_len, d)
        for row, (_, a, b) in enumerate(safe):
            H_jl = H_l[a : b + 1]  # (n_j, d)
            if V_R is not None:
                H_jl = project_to_reasoning(H_jl, V_R)  # (n_j, d_R)
            D, V, C = step_layer_spectral_summary(
                H_jl,
                rank_mode=rank_mode,
                rank_k=rank_k,
                rank_threshold=rank_threshold,
            )
            M_D[row, li] = D
            M_V[row, li] = V
            M_C[row, li] = C

    kept_steps = np.array([j for j, _, _ in safe], dtype=np.int32)
    return M_D, M_V, M_C, kept_steps, layer_indices


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--dataset", default="Qwen/ProcessBench")
    parser.add_argument("--subset", default="gsm8k",
                        choices=["gsm8k", "math", "olympiadbench", "omnimath"])
    parser.add_argument("--n_correct", type=int, default=50)
    parser.add_argument("--n_error", type=int, default=50)
    parser.add_argument("--layers", default="all",
                        help='"all" or a comma-separated list of layer indices, '
                             'e.g. "0,8,16,24,30,31". Index 0 = embedding output, '
                             '1..L = transformer block outputs.')
    parser.add_argument("--max_seq_len", type=int, default=4096)
    parser.add_argument("--output", default="data/spectral_field.npz")
    parser.add_argument("--seed", type=int, default=42)
    # Reasoning-subspace projection options.
    parser.add_argument("--no_reasoning_subspace", action="store_true",
                        help="Disable HARP-style unembedding projection.")
    parser.add_argument("--reasoning_mode", default="energy",
                        choices=["energy", "dim_ratio"],
                        help='"energy": top-`threshold` of energy is semantic, '
                             'remainder is reasoning. '
                             '"dim_ratio": bottom-`threshold` × d directions '
                             'are reasoning.')
    parser.add_argument("--reasoning_threshold", type=float, default=0.95,
                        help="Cutoff for the reasoning subspace; meaning "
                             "depends on --reasoning_mode.")
    parser.add_argument("--unembedding_cache",
                        default="data/unembedding_svd.npz",
                        help="Cache file for the W_U SVD.")
    # Effective rank estimator options.
    parser.add_argument("--rank_mode", default="full",
                        choices=["full", "topk", "energy", "kaiser"],
                        help="Effective rank estimator. 'full' is the v17 "
                             "baseline (whole spectrum). 'topk' / 'energy' / "
                             "'kaiser' truncate the spectrum before computing "
                             "the spectral entropy; see "
                             "utils.effective_rank_truncated.")
    parser.add_argument("--rank_topk", type=int, default=10,
                        help="Top-k cutoff for --rank_mode topk.")
    parser.add_argument("--rank_energy_threshold", type=float, default=0.95,
                        help="Cumulative energy threshold for "
                             "--rank_mode energy.")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    print(f"Loading model {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else device,
    )
    model.eval()

    # ---- Reasoning subspace ----
    V_R, V_R_meta = None, None
    if not args.no_reasoning_subspace:
        cache_path = args.unembedding_cache
        # Make cache path model-specific to avoid mixing different W_U.
        if cache_path:
            tag = os.path.basename(args.model.rstrip("/")).replace("/", "_")
            root, ext = os.path.splitext(cache_path)
            cache_path = f"{root}.{tag}{ext}"
        V_R, V_R_meta = prepare_reasoning_subspace(
            model,
            mode=args.reasoning_mode,
            threshold=args.reasoning_threshold,
            cache_path=cache_path,
        )

    layer_indices = None if args.layers == "all" else \
        [int(x) for x in args.layers.split(",") if x.strip()]

    correct_examples, error_examples = load_processbench_subset(
        args.dataset, args.subset, args.n_correct, args.n_error, seed=args.seed
    )

    rank_mode_str = args.rank_mode
    if rank_mode_str == "topk":
        rank_mode_str = f"topk(k={args.rank_topk})"
    elif rank_mode_str == "energy":
        rank_mode_str = f"energy(thr={args.rank_energy_threshold})"
    print(f"Extracting spectral fields (layers={args.layers}, "
          f"reasoning_subspace={V_R is not None}, "
          f"rank_mode={rank_mode_str}) ...")
    rows = []
    skipped = 0
    for ex_set, _tag in [(correct_examples, "correct"), (error_examples, "error")]:
        for ex in tqdm(ex_set, desc=_tag):
            prompt, response, steps = build_prompt_and_response(ex)
            if prompt is None:
                skipped += 1
                continue
            try:
                M_D, M_V, M_C, kept_steps, layers_used = extract_spectral_field(
                    model, tokenizer, prompt, response, steps,
                    device, layer_indices=layer_indices,
                    max_seq_len=args.max_seq_len,
                    V_R=V_R,
                    rank_mode=args.rank_mode,
                    rank_k=args.rank_topk,
                    rank_threshold=args.rank_energy_threshold,
                )
            except Exception as e:
                print(f"  warn: extraction failed: {e}")
                skipped += 1
                continue

            if M_D is None or M_D.shape[0] < 3:
                skipped += 1
                continue

            # Map original label (first-error step in ORIGINAL step indexing) to
            # the kept-step indexing so step-level evaluation is consistent.
            orig_label = int(ex.get("label", -1))
            if orig_label < 0:
                mapped_label = -1
            else:
                kept = kept_steps.tolist()
                mapped_label = kept.index(orig_label) if orig_label in kept else -2
                # -2 means "had an error but it was dropped by truncation"; skip
                if mapped_label == -2:
                    skipped += 1
                    continue

            rows.append({
                "id": str(ex.get("id", len(rows))),
                "label": mapped_label,
                "n_steps": int(M_D.shape[0]),
                "M_D": M_D.astype(np.float32),
                "M_V": M_V.astype(np.float32),
                "M_C": M_C.astype(np.float32),
                "kept_steps": kept_steps,
                "layers_used": np.asarray(layers_used, dtype=np.int32),
            })

    if not rows:
        print("ERROR: no valid trajectories.")
        return

    n_layers_sub = rows[0]["M_D"].shape[1]
    print(f"\nKept {len(rows)} trajectories ({skipped} skipped), "
          f"L_sub = {n_layers_sub} sampled layers.")

    save_dict = dict(
        ids=np.array([r["id"] for r in rows], dtype=object),
        labels=np.array([r["label"] for r in rows], dtype=np.int32),
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        M_D=np.array([r["M_D"] for r in rows], dtype=object),
        M_V=np.array([r["M_V"] for r in rows], dtype=object),
        M_C=np.array([r["M_C"] for r in rows], dtype=object),
        kept_steps=np.array([r["kept_steps"] for r in rows], dtype=object),
        layers_used=rows[0]["layers_used"],
        model_name=np.array(args.model),
        subset=np.array(args.subset),
        reasoning_subspace_used=np.array(V_R is not None),
        rank_mode=np.array(args.rank_mode),
        rank_topk=np.array(args.rank_topk),
        rank_energy_threshold=np.array(args.rank_energy_threshold),
    )
    if V_R_meta is not None:
        for k, v in V_R_meta.items():
            save_dict[f"V_R_{k}"] = np.array(v)

    np.savez(args.output, **save_dict)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
