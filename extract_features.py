"""Teacher-forcing feature extraction (refactor, 2026-06-09).

Extracts, for every chain, by teacher-forcing a fixed (prompt, text):

  PAPER (Tracing Uncertainty, trace channel, PER TOKEN)
    tok_U_D, tok_U_C, tok_U_E (+ offsets)         features/uncertainty.py
  OURS  (raw activation-degree geometry)
    per-step exp-pooled vector  -> stepgeom (T, L, F)   features/geometry.py
    per-token vector            -> tokgeom  (R, L, F)   (optional, fp16)
  SUMMARY
    25/50/25 + slope + r^2 of the 3 paper series -> profile_paper (N, 15)
                                                  features/trace_profile.py

Two data sources (same extractor, same teacher-forcing context
"Problem: {q}\\n\\nSolution:\\n\\n" + text as used by 10):
  --source processbench  : local ProcessBench <subset>.jsonl (gold step labels;
                           `label` = first erroneous step, -1 = correct).
  --source sampled       : an existing 10_sample_and_extract npz (the stored
                           K=12 responses); questions reconstructed by
                           problem_id via load_problems, labels reused.

This RUNS ON GPU (Llama-3.1-8B). U_E is one backward pass per token -- use
--ue_stride / --ue_layers_from to trade fidelity for speed. A CPU --smoke path
on a tiny model checks wiring without a GPU.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.step_boundaries import find_step_token_ranges
from utils.step_vector import step_vector
from utils.spectral import step_layer_spectral_summary, cim_tle_intrinsic_dim

CLOUD_NAMES = ("cloud_D", "cloud_V", "cloud_C", "coherence", "mean_tok_norm")
# cloud_D/V/C = point-cloud eff-rank / energy / concentration;
# coherence = ||exp-pooled vec|| / mean_t ||h_t|| (alignment; low = diffuse/cancelling);
# mean_tok_norm = mean per-token norm (so coherence's denominator is auditable separately)
INTRINSIC_NAMES = ("id_mle", "id_twonn", "cim_V")  # whole-chain CIM: intrinsic dim (D) + information volume (V)
from features import geometry as geo
from features import uncertainty as unc
from features import trace_profile as tp
from features import gradients as grad_mod

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load_local_module(filename, name):
    spec = importlib.util.spec_from_file_location(
        name, os.path.join(SCRIPT_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# reuse 10's deterministic answer matchers + problem loader + prompt building
_s10 = _load_local_module("10_sample_and_extract.py", "sample10")

EXTRACT_PROMPT = "Problem: {q}\n\nSolution:\n\n"   # identical to 10's teacher-forcing context


# ---------------------------------------------------------------------------
# Data sources -> a uniform record stream
# ---------------------------------------------------------------------------

def _pb_record(d, subset, n):
    """Build one record from a ProcessBench row (jsonl dict or HF dataset row)."""
    steps = [s.strip() for s in (d.get("steps") or []) if s and s.strip()]
    if len(steps) < 3:
        return None
    correct = bool(d.get("final_answer_correct", d.get("label", 0) == -1))
    return {
        "id": str(d.get("id", f"{subset}-{n}")),
        "source": "processbench",
        "problem_id": n,
        "sample_idx": 0,
        "question": d["problem"],
        "response": "\n".join(steps),
        "steps_text": steps,
        "is_correct": int(correct),
        "is_correct_strict": int(correct),
        "format_ok": 1,
        "gold_error_step": int(d.get("label", -1)),
        "gold_answer": float("nan"),
        "pred_answer": float("nan"),
    }


def _pb_raw_rows(path, subset):
    """Yield RAW ProcessBench rows (dicts), no filtering, in dataset order.

    Accepts: an explicit .json/.jsonl file; a dir holding <subset>.json or
    <subset>.jsonl; or an HF-saved dataset dir (load_dataset). Whole-file JSON
    array / {"data":[...]} and line-delimited jsonl are both handled.
    """
    fp = None
    if os.path.isfile(path):
        fp = path
    else:
        for cand in (os.path.join(path, f"{subset}.json"),
                     os.path.join(path, f"{subset}.jsonl")):
            if os.path.isfile(cand):
                fp = cand
                break
    if fp is not None:
        with open(fp, encoding="utf-8") as f:
            text = f.read()
        rows = None
        try:
            obj = json.loads(text)
            if isinstance(obj, list):
                rows = obj
            elif isinstance(obj, dict):
                rows = obj.get("data") or obj.get("rows")
        except json.JSONDecodeError:
            rows = None
        if rows is None:
            rows = (json.loads(ln) for ln in text.splitlines() if ln.strip())
        for d in rows:
            yield d
    else:
        if path.endswith((".json", ".jsonl")) or path.endswith((".json/", ".jsonl/")):
            raise SystemExit(
                f"--pb_path '{path}' looks like a file but does NOT exist. "
                f"Check the real filename: ls the ProcessBench dir. "
                f"(gsm8k worked because gsm8k.json exists there; the other configs "
                f"may have different names or not be present.)")
        from datasets import load_dataset
        print(f"  no json/jsonl at {path}; load_dataset({path}, split={subset})")
        for ex in load_dataset(path, split=subset):
            yield ex


def iter_processbench(path, subset, limit=None):
    """Yield extraction records from ProcessBench (steps>=3 kept)."""
    n = 0
    for d in _pb_raw_rows(path, subset):
        rec = _pb_record(d, subset, n)
        if rec is None:
            continue
        yield rec
        n += 1
        if limit and n >= limit:
            return


def reconstruct_questions(path, subset, s10):
    """Rebuild the ordered (question, gold) list EXACTLY as 10.load_problems does
    for ProcessBench, so a stored npz's problem_id maps back to its question.
    Reads the SAME ProcessBench source as iter_processbench (no step filter)."""
    gold_fields = ["answer", "final_answer", "gt_answer", "ground_truth", "gold_answer"]
    probs = {}
    for ex in _pb_raw_rows(path, subset):
        prob = ex.get("problem")
        if not prob:
            continue
        gold = None
        for f in gold_fields:
            if ex.get(f) is not None:
                gold = s10._to_number(str(ex[f]))
                if gold is not None:
                    break
        if gold is None:
            lab = int(ex.get("label", -1))
            fac = ex.get("final_answer_correct", None)
            if lab == -1 or fac is True:
                gold = s10.predicted_answer("\n".join(ex.get("steps", []) or []))
        if gold is not None and prob not in probs:
            probs[prob] = gold
    return list(probs.items())


def iter_sampled(npz_path, problems, limit=None):
    """Yield records from a stored 10_sample_and_extract npz.

    `problems` is load_problems(...) output (list of (question, gold)) so the
    question can be recovered from problem_id, exactly as 10b does.
    """
    z = np.load(npz_path, allow_pickle=True)
    responses = z["responses"]
    steps_all = z["steps_text"] if "steps_text" in z.files else None
    split = str(z["step_split"]) if "step_split" in z.files else "line"
    pids = z["problem_ids"]
    sidx = z["sample_idx"] if "sample_idx" in z.files else np.zeros(len(pids), int)
    isc = z["is_correct"]
    iscs = z["is_correct_strict"] if "is_correct_strict" in z.files else isc
    fok = z["format_ok"] if "format_ok" in z.files else np.ones(len(pids), int)
    gold = z["gold_answers"] if "gold_answers" in z.files else np.full(len(pids), np.nan)
    pred = z["pred_answers"] if "pred_answers" in z.files else np.full(len(pids), np.nan)
    maxpid = int(np.max(pids)) if len(pids) else -1
    if maxpid >= len(problems):
        print(f"  WARN: max problem_id {maxpid} >= #reconstructed problems "
              f"{len(problems)} -- alignment mismatch, those chains are skipped. "
              f"Check the npz 'dataset' field matches --pb_path/--pb_subset.")
    n = 0
    for i in range(len(responses)):
        pid = int(pids[i])
        if pid >= len(problems):
            continue
        resp = str(responses[i])
        steps = (list(steps_all[i]) if steps_all is not None
                 else _s10.split_into_steps(resp, granularity=split))
        if len(steps) < 3:
            continue
        yield {
            "id": f"p{pid}_s{int(sidx[i])}",
            "source": "sampled",
            "problem_id": pid,
            "sample_idx": int(sidx[i]),
            "question": problems[pid][0],
            "response": resp,
            "steps_text": steps,
            "is_correct": int(isc[i]),
            "is_correct_strict": int(iscs[i]),
            "format_ok": int(fok[i]),
            "gold_error_step": -1,
            "gold_answer": float(gold[i]),
            "pred_answer": float(pred[i]),
        }
        n += 1
        if limit and n >= limit:
            break


# ---------------------------------------------------------------------------
# Per-chain extraction
# ---------------------------------------------------------------------------

def extract_chain(model, tokenizer, rec, device, layer_indices,
                  massive_m, want_ue, ue_params, ue_stride,
                  store_token_geom, max_seq_len, store_step_vectors=False,
                  cloud_eff_rank=False, intrinsic_dim=False, sv_layers=None,
                  grad_block=None):
    """Return a dict of arrays for one chain, or None if it cannot be aligned."""
    prompt = EXTRACT_PROMPT.format(q=rec["question"])
    response = rec["response"]
    steps = rec["steps_text"]

    ranges = find_step_token_ranges(tokenizer, prompt, response, steps)
    if len(ranges) < 3:
        return None

    enc = tokenizer(prompt + response, return_tensors="pt",
                    truncation=True, max_length=max_seq_len)
    input_ids = enc["input_ids"][0].to(device)
    attn = enc["attention_mask"][0].to(device)
    seq_len = input_ids.shape[0]

    safe = [(a, b) for (a, b) in ranges if b < seq_len and b - a + 1 >= 2]
    if len(safe) < 3:
        return None
    a0, b1 = safe[0][0], safe[-1][1]
    if b1 <= a0:
        return None

    F = len(geo.GEOM_FEATURE_NAMES)
    L = len(layer_indices)

    # --- hidden states (no grad): per-step exp-pool + per-token geometry ---
    with torch.no_grad():
        out = model(input_ids=input_ids.unsqueeze(0),
                    attention_mask=attn.unsqueeze(0),
                    output_hidden_states=True)
        logits = out.logits[0]
        U_D, U_C = unc._entropy_committal(logits, input_ids, a0, b1)
        hs = [out.hidden_states[l][0].float().cpu().numpy() for l in layer_indices]
    del out, logits

    T = len(safe)
    d = hs[0].shape[1]
    stepgeom = np.full((T, L, F), np.nan, dtype=np.float32)
    R = b1 - a0 + 1
    tokgeom = np.full((R, L, F), np.nan, dtype=np.float16) if store_token_geom else None
    # raw per-step exp-pooled vectors (only for sv_layers, to bound memory/disk)
    sv_set = (list(layer_indices) if sv_layers is None
              else [l for l in layer_indices if l in sv_layers])
    n_sv = len(sv_set)
    stepvec = np.full((T, n_sv, d), np.nan, dtype=np.float16) if store_step_vectors else None
    # exp-pooled vector of the QUESTION/prompt tokens (positions 0..a0-1), per sv layer,
    # used as the "question" baseline for normalization (z_j - q)
    qvec = np.full((n_sv, d), np.nan, dtype=np.float16) if store_step_vectors else None
    # cloud D/V/C + coherence + mean_tok_norm (see CLOUD_NAMES)
    stepcloud = np.full((T, L, len(CLOUD_NAMES)), np.nan, dtype=np.float32) if cloud_eff_rank else None
    # whole-chain nonlinear intrinsic dimension (length-robust), per layer
    chain_id = np.full((L, len(INTRINSIC_NAMES)), np.nan, np.float32) if intrinsic_dim else None

    for li in range(L):
        H_l = hs[li]                                   # (seq, d)
        sv_k = sv_set.index(layer_indices[li]) if (store_step_vectors and
                layer_indices[li] in sv_set) else None
        if sv_k is not None and a0 > 0:                # question/prompt baseline q
            qv = step_vector(H_l[0:a0], mode="step_exp", l2_normalize=False)
            if qv is not None:
                qvec[sv_k] = qv.astype(np.float16)
        for sj, (a, b) in enumerate(safe):
            cloud = H_l[a:b + 1]                        # (n_j, d) token cloud
            z = step_vector(cloud, mode="step_exp", l2_normalize=False)
            if z is not None:
                f = geo.vector_features(z, massive_m=massive_m)
                stepgeom[sj, li] = [f[k] for k in geo.GEOM_FEATURE_NAMES]
                if sv_k is not None:
                    stepvec[sj, sv_k] = z.astype(np.float16)
            if cloud_eff_rank:
                D, V, C = step_layer_spectral_summary(cloud)
                coh = mtn = np.nan
                if z is not None:
                    mtn = float(np.linalg.norm(cloud, axis=1).mean())   # mean per-token norm
                    if mtn > 1e-9:
                        coh = float(np.linalg.norm(z) / mtn)            # pooled / per-token
                stepcloud[sj, li] = (D, V, C, coh, mtn)
        if intrinsic_dim:                              # CIM on the whole-chain last-token trajectory
            whole = H_l[a0:b1 + 1]                      # (R, d)
            chain_id[li, 0] = cim_tle_intrinsic_dim(whole)   # D_stim (kNN-ID)
            chain_id[li, 1] = geo.twonn_dim(whole)           # D_stim (TwoNN)
            chain_id[li, 2] = geo.information_volume(whole)  # V (information volume, Eq.14)
        if store_token_geom:
            for ti, pos in enumerate(range(a0, b1 + 1)):
                f = geo.vector_features(H_l[pos], massive_m=massive_m)
                tokgeom[ti, li] = [f[k] for k in geo.GEOM_FEATURE_NAMES]
    del hs

    # --- epistemic U_E (grad): one backward per (strided) token ---
    U_E = U_E_off = None
    if want_ue:
        U_E, U_E_off = unc.epistemic_grad_norms(
            model, input_ids, attn, a0, b1, ue_params, ue_stride=ue_stride)

    # --- gradient spectral field: per-step, per-layer parameter-gradient norms ---
    gradprof = grad_total = None
    if grad_block is not None:
        bof, n_blocks, gparams = grad_block
        gradprof, grad_total = grad_mod.step_gradient_profile(
            model, input_ids, attn, safe, bof, n_blocks, gparams)

    # --- paper trace-profile summary (15 numbers) ---
    prof = {}
    prof.update(tp.profile_flat(U_D, "UD"))
    prof.update(tp.profile_flat(U_C, "UC"))
    if U_E is not None:
        prof.update(tp.profile_flat(U_E, "UE"))
    else:
        prof.update({f"UE_{s}": float("nan") for s in tp.PROFILE_STATS})

    step_ranges = np.asarray(safe, dtype=np.int32)
    return {
        "tok_U_D": U_D, "tok_U_C": U_C,
        "tok_U_E": (U_E.astype(np.float32) if U_E is not None else None),
        "tok_U_E_offsets": U_E_off,
        "stepgeom": stepgeom,
        "tokgeom": tokgeom,
        "stepvec": stepvec,
        "qvec": qvec,
        "stepcloud": stepcloud,
        "chain_id": chain_id,
        "gradprof": gradprof,
        "grad_total": grad_total,
        "step_token_ranges": step_ranges,
        "n_steps": T, "n_resp_tokens": R,
        "profile": prof,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/gz-data/models/Meta-Llama-3.1-8B-Instruct")
    ap.add_argument("--source", required=True, choices=["processbench", "sampled"])
    # processbench
    ap.add_argument("--pb_path", default="data/processbench")
    ap.add_argument("--pb_subset", default="gsm8k")
    # sampled
    ap.add_argument("--sampled_npz", default=None,
                    help="stored 10_sample_and_extract npz (responses + labels).")
    ap.add_argument("--dataset_format", default="processbench")
    ap.add_argument("--dataset", default="data/hf_datasets/ProcessBench")
    ap.add_argument("--subset", default="gsm8k")
    ap.add_argument("--split", default="test")
    ap.add_argument("--n_problems", type=int, default=300)
    # extraction
    ap.add_argument("--layers", default="8,16,24,31",
                    help='"all" or comma list of hidden_states indices '
                         "(0=embeddings, 1..n=blocks). Stored per layer.")
    ap.add_argument("--massive_m", type=int, default=4,
                    help="#top-magnitude dims removed for the AE_robust feature.")
    ap.add_argument("--no_ue", action="store_true",
                    help="skip epistemic U_E (no backward); keeps U_D/U_C/geometry.")
    ap.add_argument("--ue_stride", type=int, default=1,
                    help="evaluate U_E every Nth response token (1 = every token).")
    ap.add_argument("--ue_layers_from", type=int, default=None,
                    help="only layers >= this carry grad for U_E (speed/memory "
                         "approximation; default None = all params, faithful).")
    ap.add_argument("--no_token_geom", action="store_true",
                    help="do not store per-token geometry (keeps per-step only).")
    ap.add_argument("--store_step_vectors", action="store_true",
                    help="also store the raw per-step exp-pooled vectors (fp16) for "
                         "SPE / baseline-norm / trajectory-dynamics analysis.")
    ap.add_argument("--sv_layers", default="",
                    help="comma list of layers to store step vectors for (subset of "
                         "--layers). Default empty = all --layers. Use e.g. '16' to "
                         "store ONE layer and avoid the OOM that killed the full run.")
    ap.add_argument("--cloud_eff_rank", action="store_true",
                    help="also compute the point-cloud effective rank D + spectral "
                         "energy V + top concentration C per (step, layer) -- the old "
                         "CIM triple, the n<<d cloud-dimension feature.")
    ap.add_argument("--intrinsic_dim", action="store_true",
                    help="also compute the WHOLE-chain nonlinear intrinsic dimension "
                         "(MLE-kNN + TwoNN) per layer -- length-robust, better-"
                         "conditioned than per-step effective rank.")
    ap.add_argument("--grad_profile", action="store_true",
                    help="GRADIENT spectral field: per step, backward on the step NLL, "
                         "per-transformer-block grad norm (gradprof T x n_blocks) + total "
                         "step grad norm (signal 1). Needs ~param-size memory (H100).")
    ap.add_argument("--grad_from_layer", type=int, default=None,
                    help="only blocks >= this carry grad for --grad_profile (memory).")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None, help="cap #chains (debug).")
    ap.add_argument("--output", required=True)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny CPU model (sshleifer/tiny-gpt2) to test wiring.")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    model_name = "sshleifer/tiny-gpt2" if args.smoke else args.model
    print(f"Loading model {model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    device = "cuda" if (torch.cuda.is_available() and not args.smoke) else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None)
    if device == "cpu":
        model.to(device)
    model.eval()

    n_hs = model.config.num_hidden_layers + 1
    layer_indices = (list(range(n_hs)) if args.layers == "all"
                     else [int(x) for x in args.layers.split(",") if x.strip()])
    layer_indices = [l for l in layer_indices if 0 <= l < n_hs]
    if not layer_indices:                       # e.g. tiny smoke model with few layers
        layer_indices = list(range(n_hs))
    print(f"  layers (hidden_states idx): {layer_indices}  of 0..{n_hs - 1}")

    sv_set = (layer_indices if not args.sv_layers
              else [int(x) for x in args.sv_layers.split(",") if int(x) in layer_indices])
    if args.store_step_vectors:
        print(f"  storing step vectors for layers {sv_set}")

    want_ue = not args.no_ue
    ue_params = unc.set_ue_grad_scope(model, args.ue_layers_from) if want_ue else None
    if want_ue:
        n_ue = sum(p.numel() for p in ue_params)
        print(f"  U_E ON: {n_ue/1e6:.1f}M params carry grad, stride={args.ue_stride}, "
              f"layers_from={args.ue_layers_from}")
    else:
        for p in model.parameters():        # free autograd bookkeeping entirely
            p.requires_grad_(False)
        print("  U_E OFF (gradients disabled)")

    grad_block = None
    if args.grad_profile:                   # re-enables grad on the chosen blocks
        grad_block = grad_mod.build_block_map(model, args.grad_from_layer)
        bof, n_blocks, gparams = grad_block
        print(f"  GRAD PROFILE ON: {sum(p.numel() for p in gparams)/1e6:.0f}M grad "
              f"params over {n_blocks} blocks (from_layer={args.grad_from_layer})")

    # record stream
    if args.source == "processbench":
        records = iter_processbench(args.pb_path, args.pb_subset, args.limit)
    else:
        if not args.sampled_npz:
            raise SystemExit("--source sampled needs --sampled_npz")
        problems = reconstruct_questions(args.pb_path, args.pb_subset, _s10)
        print(f"  reconstructed {len(problems)} ProcessBench questions for "
              f"problem_id lookup (from {args.pb_path})")
        records = iter_sampled(args.sampled_npz, problems, args.limit)

    rows, profiles = [], []
    n_seen = n_kept = 0
    for rec in tqdm(list(records), desc="chains"):
        n_seen += 1
        try:
            res = extract_chain(
                model, tokenizer, rec, device, layer_indices,
                args.massive_m, want_ue, ue_params, args.ue_stride,
                store_token_geom=not args.no_token_geom,
                max_seq_len=args.max_seq_len,
                store_step_vectors=args.store_step_vectors,
                cloud_eff_rank=args.cloud_eff_rank,
                intrinsic_dim=args.intrinsic_dim, sv_layers=sv_set,
                grad_block=grad_block)
        except Exception as e:
            print(f"  warn: chain {rec['id']} failed: {e}")
            res = None
        if res is None:
            continue
        n_kept += 1
        rows.append({**rec, **res})
        profiles.append(res["profile"])
        if device == "cuda" and n_kept % 50 == 0:
            torch.cuda.empty_cache()

    if not rows:
        raise SystemExit("No chains extracted -- check source / alignment.")

    prof_cols = list(profiles[0].keys())
    save = dict(
        ids=np.array([r["id"] for r in rows], dtype=object),
        source=np.array([r["source"] for r in rows], dtype=object),
        problem_ids=np.array([r["problem_id"] for r in rows], dtype=np.int32),
        sample_idx=np.array([r["sample_idx"] for r in rows], dtype=np.int32),
        is_correct=np.array([r["is_correct"] for r in rows], dtype=np.int32),
        is_correct_strict=np.array([r["is_correct_strict"] for r in rows], dtype=np.int32),
        format_ok=np.array([r["format_ok"] for r in rows], dtype=np.int32),
        gold_error_step=np.array([r["gold_error_step"] for r in rows], dtype=np.int32),
        gold_answers=np.array([r["gold_answer"] for r in rows], dtype=np.float64),
        pred_answers=np.array([r["pred_answer"] for r in rows], dtype=np.float64),
        n_steps=np.array([r["n_steps"] for r in rows], dtype=np.int32),
        n_resp_tokens=np.array([r["n_resp_tokens"] for r in rows], dtype=np.int32),
        responses=np.array([r["response"] for r in rows], dtype=object),
        steps_text=np.array([r["steps_text"] for r in rows], dtype=object),
        step_token_ranges=np.array([r["step_token_ranges"] for r in rows], dtype=object),
        # per-token paper channels
        tok_U_D=np.array([r["tok_U_D"] for r in rows], dtype=object),
        tok_U_C=np.array([r["tok_U_C"] for r in rows], dtype=object),
        tok_U_E=np.array([r["tok_U_E"] for r in rows], dtype=object),
        tok_U_E_offsets=np.array([r["tok_U_E_offsets"] for r in rows], dtype=object),
        # geometry
        stepgeom=np.array([r["stepgeom"] for r in rows], dtype=object),
        tokgeom=np.array([r["tokgeom"] for r in rows], dtype=object),
        stepvec=np.array([r["stepvec"] for r in rows], dtype=object),
        qvec=np.array([r["qvec"] for r in rows], dtype=object),
        sv_layers=np.array(sv_set, dtype=np.int32),
        step_vectors_stored=np.array(args.store_step_vectors),
        stepcloud=np.array([r["stepcloud"] for r in rows], dtype=object),
        cloud_stored=np.array(args.cloud_eff_rank),
        cloud_feature_names=np.array(CLOUD_NAMES, dtype=object),
        chain_intrinsic=(np.stack([r["chain_id"] for r in rows])
                         if args.intrinsic_dim else np.array(False)),
        intrinsic_stored=np.array(args.intrinsic_dim),
        intrinsic_names=np.array(INTRINSIC_NAMES, dtype=object),
        gradprof=np.array([r["gradprof"] for r in rows], dtype=object),
        grad_total=np.array([r["grad_total"] for r in rows], dtype=object),
        grad_stored=np.array(args.grad_profile),
        grad_layers=np.array(list(range(grad_block[1])) if grad_block else [],
                             dtype=np.int32),
        geom_feature_names=np.array(geo.GEOM_FEATURE_NAMES, dtype=object),
        # paper trace-profile table
        profile_paper=np.array([[p[c] for c in prof_cols] for p in profiles],
                               dtype=np.float64),
        profile_cols=np.array(prof_cols, dtype=object),
        # meta
        layers_used=np.array(layer_indices, dtype=np.int32),
        model_name=np.array(model_name),
        massive_m=np.array(args.massive_m),
        ue_on=np.array(want_ue),
        ue_stride=np.array(args.ue_stride),
        ue_layers_from=np.array(-1 if args.ue_layers_from is None else args.ue_layers_from),
        token_geom_stored=np.array(not args.no_token_geom),
        source_tag=np.array(args.source),
        pb_subset=np.array(args.pb_subset),
    )
    np.savez(args.output, **save)
    print(f"\nKept {n_kept}/{n_seen} chains.")
    print(f"  correct(strict): {int(save['is_correct_strict'].sum())}; "
          f"error: {n_kept - int(save['is_correct_strict'].sum())}")
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
