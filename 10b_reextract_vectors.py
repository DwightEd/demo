"""Step 10b: re-extract per-dimension step VECTORS from an existing run, WITHOUT
re-sampling.

Why: 10 already teacher-forces `extract_spectral_field` over the generated response
(the expensive part is only `model.generate`). If a run was saved WITHOUT
--store_vectors, the raw step vectors (sv_vec_<mode>) are gone and 36 cannot run.
Re-generating wastes the GPU and risks non-identical chains. Instead we read the stored
`responses` + `steps_text`, recover each question by problem_id from load_problems()
(verified against the stored gold), and run the SAME teacher-forcing forward with
store_vectors=True. Chains are bit-identical because we feed the stored text verbatim.

Self-validation: the recomputed sv_pr_step_exp and sv_tok_entropy MUST equal the stored
ones (same forward, same context). We report the max abs diff and abort if it exceeds
--tol -- that would mean the reconstructed context (question / subspace / whiten / layers)
does not match the original run.

Output: a copy of the input npz with sv_vec_step_exp added and sv_vectors_stored=True.
Only step_exp vectors are stored (the mode 33/35/36 use) to keep size down; PR/AE for all
modes are carried over from the input unchanged.
"""
from __future__ import annotations
import argparse, os
from types import SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _load(name, fn):
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPT_DIR, fn))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


m10 = _load("m10", "10_sample_and_extract.py")
_ex = m10._ex


def parse_dataset_field(s):
    """'processbench:data/hf_datasets/ProcessBench/gsm8k' -> (fmt, dataset, subset)."""
    fmt, rest = s.split(":", 1)
    dataset, subset = rest.rsplit("/", 1)
    return fmt, dataset, subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="existing npz (with responses, no sv_vec)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--dataset_format", default=None)
    ap.add_argument("--dataset", default=None)
    ap.add_argument("--subset", default=None)
    ap.add_argument("--split", default="test", help="only for gsm8k format")
    ap.add_argument("--max_seq_len", type=int, default=4096)
    ap.add_argument("--no_reasoning_subspace", action="store_true")
    ap.add_argument("--reasoning_mode", default="energy")
    ap.add_argument("--reasoning_threshold", type=float, default=0.95)
    ap.add_argument("--unembedding_cache", default="data/unembedding_svd.npz")
    ap.add_argument("--whiten_baseline", default=None, help="default: the path stored in npz")
    ap.add_argument("--model", default=None, help="default: model_name stored in npz")
    ap.add_argument("--tol", type=float, default=1e-2, help="max abs diff for PR/tok validation")
    ap.add_argument("--limit", type=int, default=0, help="only process first N rows (debug)")
    args = ap.parse_args()

    d = np.load(args.input, allow_pickle=True)
    for k in ("responses", "steps_text", "problem_ids", "gold_answers", "sv_pr_step_exp"):
        if k not in d.files:
            raise SystemExit(f"input npz lacks {k}; need a run from 10 with responses stored.")
    responses = d["responses"]; steps_text = d["steps_text"]
    pid = d["problem_ids"].astype(int); gold = d["gold_answers"].astype(float)
    stored_pr = d["sv_pr_step_exp"]
    stored_te = d["sv_tok_entropy"] if "sv_tok_entropy" in d.files else None
    layers_used = d["layers_used"].astype(int) if "layers_used" in d.files else None
    N = len(responses)
    if args.limit: N = min(N, args.limit)

    # dataset args (default from npz 'dataset' field) -> recover questions by problem_id
    fmt, dset, sub = (args.dataset_format, args.dataset, args.subset)
    if None in (fmt, dset, sub):
        f0, d0, s0 = parse_dataset_field(str(d["dataset"]))
        fmt = fmt or f0; dset = dset or d0; sub = sub or s0
    model_name = args.model or str(d["model_name"])
    prompt_style = str(d["prompt_style"]) if "prompt_style" in d.files else "custom_zeroshot"
    print(f"dataset: format={fmt} path={dset} subset={sub} | model={model_name} | "
          f"prompt={prompt_style} | rows={N}")

    la = SimpleNamespace(dataset_format=fmt, dataset=dset, subset=sub, split=args.split)
    problems = m10.load_problems(la)
    print(f"  load_problems -> {len(problems)} problems")

    # validate question/gold alignment: stored gold must match problems[pid] gold
    nbad = 0
    for i in range(N):
        p = pid[i]
        if p >= len(problems): nbad += 1; continue
        g = problems[p][1]
        if not (np.isfinite(gold[i]) and g is not None and np.isclose(g, gold[i], atol=1e-4)):
            nbad += 1
    if nbad:
        raise SystemExit(f"gold mismatch on {nbad}/{N} rows -> wrong dataset order/args. "
                         f"Pass --dataset_format/--dataset/--subset/--split explicitly.")
    print("  gold alignment OK (all rows)")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    print(f"Loading model {model_name} ...")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map="auto" if device == "cuda" else device)
    model.eval()

    V_R = None
    if not args.no_reasoning_subspace:
        cp = args.unembedding_cache
        if cp:
            tag = os.path.basename(model_name.rstrip("/")).replace("/", "_")
            root, ext = os.path.splitext(cp); cp = f"{root}.{tag}{ext}"
        V_R, _ = _ex.prepare_reasoning_subspace(
            model, mode=args.reasoning_mode, threshold=args.reasoning_threshold, cache_path=cp)

    whiten = None
    wb_path = args.whiten_baseline or (str(d["whiten_baseline"]) if "whiten_baseline" in d.files else "")
    if wb_path:
        wb = np.load(wb_path, allow_pickle=True)
        if bool(wb["reasoning_subspace_used"]) != (V_R is not None):
            raise SystemExit("whiten_baseline subspace setting != extraction setting; "
                             "set --no_reasoning_subspace to match the baseline.")
        wl = wb["whiten_layers"].astype(int)
        wmu, wsg = wb["whiten_mu"].astype(np.float64), wb["whiten_sigma"].astype(np.float64)
        whiten = {int(l): (wmu[i], wsg[i]) for i, l in enumerate(wl) if np.isfinite(wmu[i]).all()}
        print(f"  whitening ON: {len(whiten)} layers (d={wmu.shape[1]}) from {wb_path}")
    elif bool(d["whitened"]) if "whitened" in d.files else False:
        raise SystemExit("npz was whitened but no whiten_baseline path found; pass --whiten_baseline.")

    li = list(layers_used) if layers_used is not None else None
    vecs = [None]*N
    pr_max = te_max = 0.0; n_fail = n_prval = n_teval = 0
    for i in tqdm(range(N), desc="re-extract"):
        question = problems[pid[i]][0]
        response = str(responses[i]); steps = list(steps_text[i])
        extract_prompt = f"Problem: {question}\n\nSolution:\n\n"
        try:
            M_D, _, _, _, _, _, _, SV = _ex.extract_spectral_field(
                model, tok, extract_prompt, response, steps, device,
                layer_indices=li, max_seq_len=args.max_seq_len,
                V_R=V_R, step_vectors=True, sv_modes=("step_exp",), whiten=whiten,
                store_vectors=True, token_uncertainty=(stored_te is not None))
        except Exception as e:
            print(f"  warn row {i}: {e}"); n_fail += 1; continue
        if SV is None or SV.get("vec") is None:
            n_fail += 1; continue
        vecs[i] = SV["vec"]["step_exp"].astype(np.float16)
        # validate recomputed PR == stored PR
        pr_new = np.asarray(SV["pr"]["step_exp"], float)
        pr_old = np.asarray(stored_pr[i], float)
        if pr_new.shape == pr_old.shape:
            md = np.nanmax(np.abs(pr_new - pr_old)) if pr_new.size else 0.0
            pr_max = max(pr_max, float(md)); n_prval += 1
        if stored_te is not None and "tok_entropy" in SV and SV["tok_entropy"] is not None:
            te_new = np.asarray(SV["tok_entropy"], float); te_old = np.asarray(stored_te[i], float)
            if te_new.shape == te_old.shape and te_new.size:
                te_max = max(te_max, float(np.nanmax(np.abs(te_new - te_old)))); n_teval += 1

    print(f"\n=== validation (recomputed vs stored) ===")
    print(f"  PR  step_exp: max|Δ| = {pr_max:.2e}  over {n_prval} rows")
    print(f"  tok entropy : max|Δ| = {te_max:.2e}  over {n_teval} rows")
    print(f"  extraction failures: {n_fail}/{N}")
    if pr_max > args.tol or (n_teval and te_max > args.tol):
        raise SystemExit(f"VALIDATION FAILED (>{args.tol}); reconstructed context differs from "
                         f"the original run -- do NOT trust the vectors. Check dataset/subspace/"
                         f"whiten/layers/max_seq_len args.")
    print("  VALIDATION PASSED -- vectors come from a faithful forward.")

    save = {k: d[k] for k in d.files}
    save["sv_vec_step_exp"] = np.array(vecs, dtype=object)
    save["sv_vectors_stored"] = np.array(True)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.savez(args.output, **save)
    print(f"\nwrote {args.output}  (+ sv_vec_step_exp for {N - n_fail} rows)")


if __name__ == "__main__":
    main()
