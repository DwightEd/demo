"""Step 1: Extract the per-trajectory (step × layer) spectral field on ProcessBench.

For each reasoning chain we:
  1. Tokenize prompt and response exactly once with add_special_tokens=False,
     preserve their IDs/offsets, and align each pre-parsed step to that token axis.
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
    -0-=-0---怕【；。 u以提高回家看明年吧v吗像人n'd
        --n_correct 50 \
        --n_error 50 \
        --output data/spectral_field.npz
"""

from __future__ import annotations

import argparse
import json
import os
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from prompt_control_flow.data_contract import DATA_CONTRACT_VERSION
from utils import (
    step_layer_spectral_summary,
    step_layer_cim_summary,
    compute_unembedding_svd,
    select_reasoning_subspace,
    project_to_reasoning,
)
from utils.geometry import cloud_geometry
from utils.step_boundaries import (
    TokenAlignmentError,
    assert_trace_alignment,
    attach_trace_time_axis,
    build_exact_trace_alignment,
    trace_records_to_npz,
    truncate_trace_alignment,
)
from utils.step_vector import step_vector, participation_ratio, activation_entropy


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
    store_geometry: bool = False,
    geom_k: int = 4,
    cim_metrics: bool = False,
    tle_k: int | None = None,
    step_vectors: bool = False,
    sv_modes: tuple[str, ...] = ("last", "mean", "linear", "step_exp"),
    whiten: dict | None = None,
    whiten_eps: float = 1e-6,
    store_vectors: bool = False,
    store_clouds: bool = False,
    cloud_layer_indices: tuple | None = None,
    token_uncertainty: bool = False,
    trace_alignment: dict | None = None,
    question_text: str | None = None,
    return_trace: bool = False,
    store_prompt_hidden: bool = False,
    prompt_hidden_layer_indices: tuple | None = None,
):
    """Run one forward pass and reduce each (step, layer) token cloud to (D, V, C).

    If V_R is provided, project each token-cloud onto the reasoning subspace
    before computing the spectral summary.

    The effective rank D is computed by `step_layer_spectral_summary`. Its
    rank_mode argument selects whether to use the full spectrum (default) or
    a truncated form. See `effective_rank_truncated` for the available modes.

    If store_geometry is True, additionally retain, per (step, layer), the
    token-cloud centroid mu (position) and its top-`geom_k` principal axes
    (orientation) -- the information that the (D, V, C) scalars throw away.
    These power the centroid-drift / orientation-drift analysis in
    05_geometry_analysis.py.

    Returns:
        M_D, M_V, M_C: (T, L_sub) float arrays where L_sub = len(layer_indices).
                       Rows in original step order; NaN rows are dropped.
        kept_steps:    indices (in the original 0..T-1) of steps actually kept.
        layers_used:   list of int layer indices that were sampled.
        GEOM:          None if store_geometry is False; otherwise a dict with
                         "mu":      (T, L_sub, p) centroids
                         "eigvals": (T, L_sub, geom_k) leading eigenvalues
                         "eigvecs": (T, L_sub, p, geom_k) leading axes
                        where p = d_R (projected) or d (raw).
        TRACE:         returned only when ``return_trace=True``. It contains the
                       exact no-special-token model input, offsets, question and
                       response spans, kept-step time axis, and optional selected
                       prompt hidden states.
    """

    def empty_result():
        base = (None, None, None, None, None, None, None, None)
        return base + (None,) if return_trace else base

    if trace_alignment is None:
        trace = build_exact_trace_alignment(
            tokenizer,
            prompt,
            response,
            steps,
            question_text=question_text,
            fail_on_unmatched=True,
        )
    else:
        trace = dict(trace_alignment)
        if trace.get("rendered_prompt") != prompt:
            raise TokenAlignmentError(
                "teacher-forcing prompt differs from the rendered generation prompt"
            )
        if trace.get("response_text") != response:
            raise TokenAlignmentError(
                "teacher-forcing response differs from the exact generated response"
            )
        if list(trace.get("steps_text", [])) != [str(x) for x in steps]:
            raise TokenAlignmentError(
                "teacher-forcing step strings differ from the aligned generation steps"
            )
        assert_trace_alignment(trace)

    trace = truncate_trace_alignment(trace, max_seq_len)
    ranges = list(trace["all_step_token_ranges"])
    if len(ranges) < 3:
        return empty_result()

    # Replay the exact generation token axis. Never re-tokenize prompt+response:
    # doing so can add BOS twice or merge a BPE token across the prompt boundary.
    encoding = {
        "input_ids": torch.tensor(
            [trace["input_ids"]], dtype=torch.long, device=device
        ),
        "attention_mask": torch.tensor(
            [trace["attention_mask"]], dtype=torch.long, device=device
        ),
    }
    seq_len = encoding["input_ids"].shape[1]
    prompt_len = len(trace["prompt_token_ids"])
    if not torch.equal(
        encoding["input_ids"][0, :prompt_len].detach().cpu(),
        torch.as_tensor(trace["prompt_token_ids"], dtype=torch.long),
    ):
        raise TokenAlignmentError("model input no longer has the exact generation prompt prefix")

    # Keep only steps whose token range fits inside the truncated sequence.
    safe = [(j, a, b) for j, (a, b) in enumerate(ranges) if b < seq_len and b - a + 1 >= 2]
    if len(safe) < 3:
        return empty_result()

    outputs = model(**encoding, output_hidden_states=True)
    hidden_states = outputs.hidden_states  # tuple of (1, seq_len, d) tensors
    n_layers_total = len(hidden_states)
    # Logits for per-step output-token entropy (used only if step_vectors).
    logits = outputs.logits[0] if (step_vectors and hasattr(outputs, "logits")) else None

    if layer_indices is None:
        layer_indices = list(range(n_layers_total))
    layer_indices = [l for l in layer_indices if 0 <= l < n_layers_total]
    L_sub = len(layer_indices)

    if store_prompt_hidden:
        prompt_layers = list(
            prompt_hidden_layer_indices
            if prompt_hidden_layer_indices is not None else layer_indices
        )
        if not prompt_layers:
            raise ValueError("store_prompt_hidden requires at least one prompt hidden layer")
        bad_layers = [l for l in prompt_layers if not 0 <= int(l) < n_layers_total]
        if bad_layers:
            raise ValueError(
                f"prompt hidden layers {bad_layers} are outside 0..{n_layers_total - 1}"
            )
        prompt_hidden = [
            hidden_states[int(l)][0, :prompt_len].float().cpu().numpy().astype(np.float16)
            for l in prompt_layers
        ]
        trace["prompt_hidden"] = np.stack(prompt_hidden, axis=1)  # (P, L_prompt, d)
        trace["prompt_hidden_layers"] = np.asarray(prompt_layers, dtype=np.int32)

    T_eff = len(safe)
    M_D = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_V = np.full((T_eff, L_sub), np.nan, dtype=np.float64)
    M_C = np.full((T_eff, L_sub), np.nan, dtype=np.float64)

    # CIM-faithful metric buffers (TLE intrinsic dim, log-det info volume).
    M_Dtle = np.full((T_eff, L_sub), np.nan, dtype=np.float64) if cim_metrics else None
    M_Vld = np.full((T_eff, L_sub), np.nan, dtype=np.float64) if cim_metrics else None

    # Optional geometry buffers. p (feature dim after optional projection) is
    # discovered from the first cloud.
    GEOM = None
    geom_mu = geom_eigvals = geom_eigvecs = None
    p_dim = None

    # Step-vector activation-participation buffers. For each weighting mode we
    # store per-(step, layer) participation ratio (PR) and activation entropy
    # (AE) of the aggregated step vector. Plus a per-step output-token entropy
    # (layer-independent) to test the "more active dims <-> more uncertain" link.
    SV = None
    sv_pr = sv_ae = None
    out_entropy = out_committal = None
    sv_vec = None
    if step_vectors:
        sv_pr = {m: np.full((T_eff, L_sub), np.nan, dtype=np.float32) for m in sv_modes}
        sv_ae = {m: np.full((T_eff, L_sub), np.nan, dtype=np.float32) for m in sv_modes}
        out_entropy = np.full(T_eff, np.nan, dtype=np.float32)
        out_committal = np.full(T_eff, np.nan, dtype=np.float32)
        # optional: keep the raw (un-normalized) step vectors so participation can
        # be re-normalized (raw / healthy-standardized / whitened) in analysis
        # WITHOUT re-running the model. Lazily sized once d is known.
        sv_vec = {m: None for m in sv_modes} if store_vectors else None

    # Raw per-step token clouds (the structure step-vector pooling destroys). Stored
    # BEFORE any reasoning-subspace projection, for a restricted set of layers only
    # (full hidden dim x all tokens is large). cloud_acc[l] accumulates one (n_j, d)
    # array per kept step; concatenated at the end. cloud_sizes records n_j per step
    # so the per-step clouds can be split back on the analysis side.
    cloud_set = set(cloud_layer_indices) if (store_clouds and cloud_layer_indices) else set()
    cloud_set = {l for l in cloud_set if l in layer_indices}
    cloud_acc = {l: [] for l in cloud_set}
    cloud_sizes = [int(b - a + 1) for (_, a, b) in safe] if cloud_set else None

    for li, l in enumerate(layer_indices):
        H_l = hidden_states[l][0].float().cpu().numpy()  # (seq_len, d)
        for row, (_, a, b) in enumerate(safe):
            H_jl = H_l[a : b + 1]  # (n_j, d)  -- raw token cloud
            if l in cloud_set:
                cloud_acc[l].append(H_jl.astype(np.float16))
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

            if cim_metrics:
                D_tle, V_ld = step_layer_cim_summary(H_jl, tle_k=tle_k)
                M_Dtle[row, li] = D_tle
                M_Vld[row, li] = V_ld

            if store_geometry:
                mu, eigvals, eigvecs = cloud_geometry(H_jl, k=geom_k)
                if mu is not None:
                    if p_dim is None:
                        p_dim = mu.shape[0]
                        geom_mu = np.full((T_eff, L_sub, p_dim), np.nan, dtype=np.float32)
                        geom_eigvals = np.full((T_eff, L_sub, geom_k), np.nan, dtype=np.float32)
                        geom_eigvecs = np.full((T_eff, L_sub, p_dim, geom_k), np.nan, dtype=np.float32)
                    geom_mu[row, li] = mu.astype(np.float32)
                    geom_eigvals[row, li] = eigvals.astype(np.float32)
                    geom_eigvecs[row, li] = eigvecs.astype(np.float32)

            if step_vectors:
                # H_jl is the (possibly projected) step token cloud, in token
                # order. Aggregate with each weighting mode, then measure how
                # many dimensions the resulting step vector activates.
                # If `whiten` is given (per-layer healthy mean/std), express the
                # step vector as a per-dimension deviation from healthy reasoning
                # BEFORE counting active dims -> "how many dims are abnormally
                # active vs correct" (the anchor). Use raw (un-L2-normalized)
                # vectors then, since the deviation magnitude is the signal.
                wl = whiten.get(l) if whiten is not None else None
                # store raw (un-normalized) vectors when keeping them for analysis
                l2 = (wl is None) and not store_vectors
                for m in sv_modes:
                    z = step_vector(H_jl, mode=m, l2_normalize=l2)
                    if z is not None:
                        if store_vectors:
                            if sv_vec[m] is None:
                                sv_vec[m] = np.full((T_eff, L_sub, z.shape[0]),
                                                    np.nan, dtype=np.float16)
                            sv_vec[m][row, li] = z.astype(np.float16)
                        z_metric = z
                        if wl is not None:
                            mu_l, sg_l = wl
                            z_metric = (z - mu_l) / (sg_l + whiten_eps)
                        sv_pr[m][row, li] = participation_ratio(z_metric)
                        sv_ae[m][row, li] = activation_entropy(z_metric)

    # Per-step output-token entropy (layer-independent): entropy of the model's
    # next-token distribution at the step's last token position. Tests whether
    # activation participation correlates with predictive uncertainty.
    if step_vectors and logits is not None:
        import torch
        seq_ids = encoding["input_ids"][0]
        for row, (_, a, b) in enumerate(safe):
            lg = logits[b].float()                      # (vocab,) predicting token b+1
            logp = torch.log_softmax(lg, dim=-1)
            p = logp.exp()
            out_entropy[row] = float(-(p * logp).sum().item())
            # committal p(1-p) of the REALIZED next token (the step boundary token)
            if b + 1 < seq_ids.shape[0]:
                ptok = float(p[int(seq_ids[b + 1])].item())
                out_committal[row] = ptok * (1.0 - ptok)

    # Per-TOKEN uncertainty over the response tokens (for uncertainty-trace-profile, 34):
    #   entropy   = H(softmax(logits_{t-1}))            distributional aleatoric
    #   committal = p(1-p), p = prob of the actual token t   committal aleatoric
    # Computed from the SAME forward pass (no extra memory / no output_logits in generate),
    # vectorised over the response range -> one (R, V) tensor per chain, freed immediately.
    tok_ent = tok_com = tok_chosen_logp = tok_margin = tok_top5_mass = None
    if step_vectors and token_uncertainty and logits is not None and len(safe) > 0:
        import torch
        a0 = max(1, int(safe[0][1])); b1 = int(safe[-1][2])
        if b1 >= a0:
            pos = torch.arange(a0, b1 + 1, device=logits.device)
            sub = logits.index_select(0, pos - 1).float()          # (R, V) at predicting positions
            lp = torch.log_softmax(sub, dim=-1); p = lp.exp()
            ent = -(p * lp).sum(-1)                                 # (R,)
            tgt = encoding["input_ids"][0].index_select(0, pos)     # actual next tokens
            chosen_lp = lp.gather(-1, tgt.view(-1, 1)).squeeze(-1)
            ptok = chosen_lp.exp()
            com = ptok * (1 - ptok)
            top_logits = torch.topk(sub, k=2, dim=-1).values
            margin = top_logits[:, 0] - top_logits[:, 1]
            top5_mass = torch.topk(p, k=5, dim=-1).values.sum(dim=-1)
            tok_ent = ent.detach().cpu().numpy().astype(np.float32)
            tok_com = com.detach().cpu().numpy().astype(np.float32)
            tok_chosen_logp = chosen_lp.detach().cpu().numpy().astype(np.float32)
            tok_margin = margin.detach().cpu().numpy().astype(np.float32)
            tok_top5_mass = top5_mass.detach().cpu().numpy().astype(np.float32)
            del sub, lp, p, chosen_lp, top_logits, margin, top5_mass

    if store_geometry and geom_mu is not None:
        GEOM = {"mu": geom_mu, "eigvals": geom_eigvals, "eigvecs": geom_eigvecs}

    kept_steps = np.array([j for j, _, _ in safe], dtype=np.int32)
    trace = attach_trace_time_axis(
        trace,
        kept_steps,
        [(a, b) for _, a, b in safe],
    )
    CIM = None
    if cim_metrics:
        CIM = {"M_Dtle": M_Dtle, "M_Vld": M_Vld}
    # Assemble the raw token-cloud payload (concatenate per-step clouds per layer).
    CLOUDS = None
    if cloud_set and all(len(cloud_acc[l]) == len(safe) for l in cloud_set):
        ls = sorted(cloud_set)
        per_layer = [np.concatenate(cloud_acc[l], axis=0) for l in ls]  # each (n_tot, d)
        CLOUDS = {"clouds": np.stack(per_layer, axis=1),                # (n_tot, L_cloud, d)
                  "sizes": np.asarray(cloud_sizes, dtype=np.int32),
                  "layers": np.asarray(ls, dtype=np.int32)}

    SV = None
    if step_vectors:
        SV = {"pr": sv_pr, "ae": sv_ae, "out_entropy": out_entropy,
              "out_committal": out_committal, "modes": list(sv_modes)}
        if store_vectors:
            SV["vec"] = sv_vec
        if CLOUDS is not None:
            SV["clouds"] = CLOUDS
        if tok_ent is not None:
            SV["tok_entropy"] = tok_ent
            SV["tok_committal"] = tok_com
            SV["tok_chosen_logprob"] = tok_chosen_logp
            SV["tok_logit_margin"] = tok_margin
            SV["tok_top5_mass"] = tok_top5_mass
            SV["tok_token_range"] = np.asarray([a0, b1 + 1], dtype=np.int32)
            trace["token_uncertainty_token_range"] = (int(a0), int(b1 + 1))
    elif CLOUDS is not None:
        # Raw clouds are a first-class ECGH artifact and do not require the
        # legacy step-vector feature family to be enabled.
        SV = {"clouds": CLOUDS}
    result = (M_D, M_V, M_C, kept_steps, layer_indices, GEOM, CIM, SV)
    return result + (trace,) if return_trace else result


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
    # Geometry storage: keep centroid (position) + top-k principal axes
    # (orientation) per (step, layer), which the D/V/C scalars discard.
    parser.add_argument("--store_geometry", action="store_true",
                        help="Additionally store per-(step,layer) centroid mu "
                             "and top-k principal axes for the orientation/"
                             "position drift analysis (05_geometry_analysis.py). "
                             "Increases output size; off by default.")
    parser.add_argument("--geom_k", type=int, default=4,
                        help="Number of leading principal axes to store per "
                             "(step, layer) when --store_geometry is set.")
    # CIM-faithful metrics: TLE intrinsic dimension + log-det information volume.
    parser.add_argument("--cim_metrics", action="store_true",
                        help="Additionally compute and store per-(step,layer) "
                             "CIM-style TLE intrinsic dimension (M_Dtle) and "
                             "log-det information volume (M_Vld). These replace "
                             "the linear effective rank / spectral energy with "
                             "the quantities CIM actually uses.")
    parser.add_argument("--tle_k", type=int, default=None,
                        help="Neighbors for TLE intrinsic dim. Default "
                             "min(n_tokens-1, 20) per cloud.")
    # Step-vector aggregation (Streaming-HD optimal step-time-exponential) +
    # activation participation.
    parser.add_argument("--step_vectors", action="store_true",
                        help="Aggregate each step's token cloud into one vector "
                             "(several weighting modes) and store its activation "
                             "participation (PR, entropy) per (step,layer), plus "
                             "per-step output-token entropy.")
    parser.add_argument("--sv_modes", default="last,mean,linear,step_exp",
                        help="Comma list of step-vector weighting modes to "
                             "compare. step_exp is the Streaming-HD optimum.")
    parser.add_argument("--store_vectors", action="store_true",
                        help="Also store the raw (un-normalized) step vectors "
                             "(fp16) per (step,layer,mode), so participation can be "
                             "re-normalized (raw / healthy-standardized / whitened) "
                             "in analysis WITHOUT re-running the model. Storage ~ "
                              "n_chains x T x L x d x 2 bytes per mode; restrict with "
                              "--sv_modes step_exp to keep it small.")
    parser.add_argument(
        "--store_token_outputs",
        action="store_true",
        help=(
            "Store compact token-aligned output summaries (entropy, realized-token "
            "log-probability, top-1/top-2 margin, and top-5 mass). Requires "
            "--step_vectors; full vocabulary logits are not saved."
        ),
    )
    parser.add_argument("--store_prompt_hidden", action="store_true",
                        help="Store the exact rendered-prompt hidden span (fp16) "
                             "at --prompt_hidden_layers for semantic anchors.")
    parser.add_argument("--prompt_hidden_layers", default="16",
                        help="Comma-separated hidden-state indices stored for the "
                             "prompt span when --store_prompt_hidden is enabled.")
    parser.add_argument("--store_clouds", action="store_true",
                        help="Store raw per-step token clouds (fp16) for the "
                             "selected --cloud_layers. Required by conditional "
                             "Gram/second-moment ECGH audits.")
    parser.add_argument("--cloud_layers", default="16",
                        help="Comma-separated hidden-state indices stored as "
                             "raw token clouds when --store_clouds is enabled.")
    args = parser.parse_args()
    if args.store_token_outputs and not args.step_vectors:
        parser.error("--store_token_outputs requires --step_vectors")

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
    prompt_hidden_layers = tuple(
        int(x) for x in args.prompt_hidden_layers.split(",") if x.strip()
    ) if args.store_prompt_hidden else None
    cloud_layers = tuple(
        int(x) for x in args.cloud_layers.split(",") if x.strip()
    ) if args.store_clouds else None

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
                M_D, M_V, M_C, kept_steps, layers_used, GEOM, CIM, SV, TRACE = extract_spectral_field(
                    model, tokenizer, prompt, response, steps,
                    device, layer_indices=layer_indices,
                    max_seq_len=args.max_seq_len,
                    V_R=V_R,
                    rank_mode=args.rank_mode,
                    rank_k=args.rank_topk,
                    rank_threshold=args.rank_energy_threshold,
                    store_geometry=args.store_geometry,
                    geom_k=args.geom_k,
                    cim_metrics=args.cim_metrics,
                    tle_k=args.tle_k,
                    step_vectors=args.step_vectors,
                    sv_modes=tuple(args.sv_modes.split(",")) if args.step_vectors else (),
                    store_vectors=args.store_vectors,
                    question_text=str(ex.get("problem", "")),
                    return_trace=True,
                    store_prompt_hidden=args.store_prompt_hidden,
                    prompt_hidden_layer_indices=prompt_hidden_layers,
                    store_clouds=args.store_clouds,
                    cloud_layer_indices=cloud_layers,
                    token_uncertainty=args.store_token_outputs,
                )
            except TokenAlignmentError:
                # Trace corruption is a pipeline invariant violation, not a
                # recoverable bad sample. Stop before writing mixed token axes.
                raise
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
                "problem": str(ex.get("problem", "")),
                "generator": str(ex.get("generator", "")),
                "label": mapped_label,
                "final_answer_correct": int(
                    bool(ex.get("final_answer_correct", False))
                ),
                "n_steps": int(M_D.shape[0]),
                "steps_text": np.array([steps[int(k)] for k in kept_steps], dtype=object),
                "M_D": M_D.astype(np.float32),
                "M_V": M_V.astype(np.float32),
                "M_C": M_C.astype(np.float32),
                "kept_steps": kept_steps,
                "layers_used": np.asarray(layers_used, dtype=np.int32),
                "GEOM": GEOM,  # None unless --store_geometry
                "CIM": CIM,    # None unless --cim_metrics
                "SV": SV,      # None unless --step_vectors
                "TRACE": TRACE,
            })

    if not rows:
        print("ERROR: no valid trajectories.")
        return

    n_layers_sub = rows[0]["M_D"].shape[1]
    print(f"\nKept {len(rows)} trajectories ({skipped} skipped), "
          f"L_sub = {n_layers_sub} sampled layers.")

    save_dict = dict(
        ids=np.array([r["id"] for r in rows], dtype=object),
        problems=np.array([r["problem"] for r in rows], dtype=object),
        problem_ids=np.array([r["problem"] for r in rows], dtype=object),
        sample_idx=np.zeros(len(rows), dtype=np.int32),
        labels=np.array([r["label"] for r in rows], dtype=np.int32),
        gold_error_step=np.array([r["label"] for r in rows], dtype=np.int32),
        process_correct=np.array([r["label"] < 0 for r in rows], dtype=np.int32),
        final_answer_correct=np.array(
            [r["final_answer_correct"] for r in rows], dtype=np.int32
        ),
        is_correct=np.array([r["label"] < 0 for r in rows], dtype=np.int32),
        source_generators=np.array([r["generator"] for r in rows], dtype=object),
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        steps_text=np.array([r["steps_text"] for r in rows], dtype=object),
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
        data_contract_version=np.array(DATA_CONTRACT_VERSION),
        trace_semantics=np.array("benchmark_observer_teacher_forcing"),
        prompt_provenance=np.array("fixed_plain_problem_solution_observer_prompt"),
        response_provenance=np.array("processbench_reformatted_steps_double_newline"),
        original_generation_trace_available=np.array(False),
        is_correct_semantics=np.array("process_correct"),
        label_semantics=np.array("gold_error_step_processbench"),
    )
    # Shared exact-generation/teacher-forcing trace schema. This records the
    # actual no-special-token model input, offsets, question/response spans,
    # and the kept-step time axis for every trajectory.
    save_dict.update(trace_records_to_npz([r["TRACE"] for r in rows]))
    model_sampling_meta = {
        "model_name": args.model,
        "model_type": str(getattr(model.config, "model_type", "")),
        "model_revision": str(getattr(model.config, "_commit_hash", "")),
        "tokenizer_name": str(getattr(tokenizer, "name_or_path", args.model)),
        "tokenizer_revision": str(getattr(tokenizer, "init_kwargs", {}).get("revision", "")),
        "torch_dtype": str(dtype),
        "device": str(device),
        "source_mode": "teacher_forced_processbench",
        "trace_semantics": "benchmark_observer_teacher_forcing",
        "prompt_provenance": "fixed_plain_problem_solution_observer_prompt",
        "response_provenance": "processbench_reformatted_steps_double_newline",
        "original_generation_prompt_available": False,
        "original_generation_token_trace_available": False,
        "label_semantics": "gold_error_step_processbench",
        "do_sample": False,
        "seed": int(args.seed),
        "max_seq_len": int(args.max_seq_len),
        "add_special_tokens": False,
    }
    save_dict["model_sampling_metadata_json"] = np.array(
        json.dumps(model_sampling_meta, sort_keys=True, ensure_ascii=False)
    )
    if V_R_meta is not None:
        for k, v in V_R_meta.items():
            save_dict[f"V_R_{k}"] = np.array(v)

    # Geometry payload (object arrays of per-trajectory (T, L, ...) tensors).
    if args.store_geometry:
        have_geom = any(r["GEOM"] is not None for r in rows)
        if have_geom:
            save_dict["geom_stored"] = np.array(True)
            save_dict["geom_k"] = np.array(args.geom_k)
            save_dict["geom_mu"] = np.array(
                [r["GEOM"]["mu"] if r["GEOM"] else None for r in rows], dtype=object)
            save_dict["geom_eigvals"] = np.array(
                [r["GEOM"]["eigvals"] if r["GEOM"] else None for r in rows], dtype=object)
            save_dict["geom_eigvecs"] = np.array(
                [r["GEOM"]["eigvecs"] if r["GEOM"] else None for r in rows], dtype=object)
        else:
            save_dict["geom_stored"] = np.array(False)
    else:
        save_dict["geom_stored"] = np.array(False)

    # CIM-faithful metric payload (TLE intrinsic dim, log-det info volume).
    if args.cim_metrics:
        have_cim = any(r["CIM"] is not None for r in rows)
        if have_cim:
            save_dict["cim_stored"] = np.array(True)
            save_dict["M_Dtle"] = np.array(
                [r["CIM"]["M_Dtle"] if r["CIM"] else None for r in rows], dtype=object)
            save_dict["M_Vld"] = np.array(
                [r["CIM"]["M_Vld"] if r["CIM"] else None for r in rows], dtype=object)
        else:
            save_dict["cim_stored"] = np.array(False)
    else:
        save_dict["cim_stored"] = np.array(False)

    # Step-vector activation-participation payload.
    if args.step_vectors:
        have_sv = any(r["SV"] is not None for r in rows)
        if have_sv:
            modes = rows[0]["SV"]["modes"]
            save_dict["sv_stored"] = np.array(True)
            save_dict["sv_modes"] = np.array(modes, dtype=object)
            # per mode: object array (one (T,L) matrix per chain) of PR and AE
            for m in modes:
                save_dict[f"sv_pr_{m}"] = np.array(
                    [r["SV"]["pr"][m] if r["SV"] else None for r in rows], dtype=object)
                save_dict[f"sv_ae_{m}"] = np.array(
                    [r["SV"]["ae"][m] if r["SV"] else None for r in rows], dtype=object)
            save_dict["sv_out_entropy"] = np.array(
                [r["SV"]["out_entropy"] if r["SV"] else None for r in rows], dtype=object)
            if rows[0]["SV"].get("tok_entropy") is not None:
                save_dict["token_output_summary_schema"] = np.array(
                    "compact_token_output_v1"
                )
                save_dict["sv_tok_entropy"] = np.array(
                    [r["SV"].get("tok_entropy") for r in rows], dtype=object
                )
                save_dict["sv_tok_committal"] = np.array(
                    [r["SV"].get("tok_committal") for r in rows], dtype=object
                )
                save_dict["chosen_token_logprobs"] = np.array(
                    [r["SV"].get("tok_chosen_logprob") for r in rows], dtype=object
                )
                save_dict["top1_top2_margin"] = np.array(
                    [r["SV"].get("tok_logit_margin") for r in rows], dtype=object
                )
                save_dict["top5_mass"] = np.array(
                    [r["SV"].get("tok_top5_mass") for r in rows], dtype=object
                )
            # raw step vectors (fp16), for post-hoc re-normalization in analysis
            if args.store_vectors and rows[0]["SV"].get("vec") is not None:
                save_dict["sv_vectors_stored"] = np.array(True)
                for m in modes:
                    save_dict[f"sv_vec_{m}"] = np.array(
                        [r["SV"]["vec"][m] if (r["SV"] and r["SV"].get("vec"))
                         else None for r in rows], dtype=object)
            else:
                save_dict["sv_vectors_stored"] = np.array(False)
        else:
            save_dict["sv_stored"] = np.array(False)
    else:
        save_dict["sv_stored"] = np.array(False)

    # Anchor-residual Gram geometry needs the raw token cloud and exact sizes,
    # even when the legacy step-vector family is disabled.
    if args.store_clouds:
        have_clouds = any(
            r["SV"] is not None and r["SV"].get("clouds") is not None for r in rows
        )
        save_dict["clouds_stored"] = np.array(have_clouds)
        if have_clouds:
            first = next(r["SV"]["clouds"] for r in rows
                         if r["SV"] is not None and r["SV"].get("clouds") is not None)
            save_dict["cloud_layers"] = first["layers"]
            save_dict["sv_clouds"] = np.array([
                r["SV"]["clouds"]["clouds"]
                if r["SV"] is not None and r["SV"].get("clouds") is not None else None
                for r in rows
            ], dtype=object)
            save_dict["cloud_sizes"] = np.array([
                r["SV"]["clouds"]["sizes"]
                if r["SV"] is not None and r["SV"].get("clouds") is not None else None
                for r in rows
            ], dtype=object)
    else:
        save_dict["clouds_stored"] = np.array(False)

    np.savez(args.output, **save_dict)
    print(f"Saved -> {args.output}"
          + ("  [with geometry]" if args.store_geometry else ""))


if __name__ == "__main__":
    main()
