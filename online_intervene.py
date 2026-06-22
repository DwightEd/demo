#!/usr/bin/env python3
"""Step-level online geometric trigger vs entropy (Halo) and EDIS, with leak-free conformal.

HEADLINE (actuator-independent): the within-step directional-collapse geometric trigger catches
reasoning errors that BOTH the entropy trigger (Halo's Observer) AND EDIS dynamics (burst +
peak-valley) structurally miss -- the blind spot of LOW-ENTROPY confident hallucinations.

Pipeline, all at matched HELD-OUT FPR with LEAK-FREE conformal thresholds:
  signal  (per step)        resultant = ||exp-pooled UNIT token vectors|| at mid layer
                            = within-step token directional concentration
  geom    (per step, online) within-chain causal z-score of -resultant, first-crossing
  entropy (per step, online) within-chain causal z-score of step mean entropy, first-crossing  [Halo]
  EDIS    (chain-level)      burst (Eq.5) + peak-valley (Eq.6) instability on the token-entropy
                            trajectory, EDIS = S*(1+Var) (Eq.7), thresholded                    [EDIS, native form]
  BLIND SPOT                wrong & geom fires & NEITHER entropy NOR EDIS fires
  repair  (secondary)       on the blind spot only: repath (high-T different-method) | compress
                            (Halo state-reset) | steer (ablation; see caveat)

Why the split matters: the blind-spot count depends on the firing thresholds, so thresholds are
set ONLY on a CALIB split of correct chains and the realized FPR is measured on a held-out EVAL
split of correct chains. Wrong chains never touch threshold-setting. This removes in-sample leak.

Entry points
  --selftest : runs the leak-prone statistics (residuals / calib-eval split / conformal / quadrants
               / blind-spot entropy) on SYNTHETIC signals with NO model and NO GPU. numpy only.
  (default)  : loads a CoT model, generates, runs the full closed loop on a dataset.

The statistics core `analyze()` is model-free; only the Solver needs torch/transformers/GPU.
"""

from __future__ import annotations
import argparse
import re
import numpy as np


# ============================ signal primitives (model-free) ============================
def resultant(H):
    """H: (n_tok, d) mid-layer hidden states -> within-step directional concentration.
    exp-pooled UNIT vectors: magnitude-free, so this measures DIRECTION agreement, not norm."""
    H = np.asarray(H, np.float64)
    nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return np.nan
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
    return float(np.linalg.norm((w[:, None] * u).sum(0)))


def causal_z(seq, sd_floor, clip=5.0, eps=1e-6):
    """within-chain CAUSAL z-score: z[t] = (s[t]-mean(s[:t]))/max(std(s[:t]),sd_floor).
    sd_floor avoids huge z when early steps are near-constant (spurious early triggers)."""
    s = np.asarray(seq, float); T = len(s); z = np.full(T, np.nan)
    for t in range(2, T):
        h = s[:t]; h = h[np.isfinite(h)]
        if len(h) >= 2 and np.isfinite(s[t]):
            z[t] = np.clip((s[t] - h.mean()) / (max(h.std(), sd_floor) + eps), -clip, clip)
    return z


def edis_burst_pv(ent, w, tau_b, tau_r):
    """EDIS spikes on a token-entropy sequence: burst (Eq.5) and peak-valley (Eq.6)."""
    e = np.asarray(ent, float); T = len(e)
    if T < 2:
        return 0, 0
    burst = int(sum(1 for t in range(max(T - w, 0)) if e[t + w] - e[t] > tau_b)) if T > w else 0
    pv = 0; run_min = e[0]
    for t in range(1, T):
        if e[t] - run_min > tau_r:
            pv += 1
        run_min = min(run_min, e[t])
    return burst, pv


def edis_score(ent, w, tau_b, tau_r):
    """EDIS = S*(1+Var), S = 0.5*(burst + peak-valley)  (Eq.7). Higher = more unstable.
    Kept in its NATIVE chain-level form (its strongest setting); conformal handles the scale."""
    e = np.asarray(ent, float)
    if len(e) < 2:
        return 0.0
    b, pv = edis_burst_pv(e, w, tau_b, tau_r)
    return 0.5 * (b + pv) * (1.0 + float(np.var(e)))


def first_cross(seq, thr):
    """first step index whose value crosses thr (explicit finite mask; nan never fires)."""
    s = np.asarray(seq, float)
    idx = np.where(np.isfinite(s) & (s > thr))[0]
    return int(idx[0]) if len(idx) else None


# ================================ answer checking ================================
def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None
    j = text.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k]
    return None


def extract_answer(text):
    b = extract_boxed(text)
    if b is not None:
        return b.strip()
    m = re.search(r"####\s*(-?[\d,]+\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def _norm(s):
    s = str(s).strip()
    for a, b in [("\\left", ""), ("\\right", ""), (" ", ""), ("$", ""), ("\\!", ""), ("\\,", ""),
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("\\\\", "\\"), ("{}", "")]:
        s = s.replace(a, b)
    return s.rstrip(".")


def correct(pred, gold):
    if pred is None or gold is None:
        return False
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except ValueError:
        return False


# ================================ segmentation ================================
def step_spans(text):
    """STEP-granularity char spans (matches the validation granularity). Prefer 'Step N:'
    markers; fall back to blank-line paragraphs; then sentences. Returns (spans, method)."""
    marks = [m.start() for m in re.finditer(r"(?im)^\s*step\s*\d+\s*[:.\)]", text)]
    if len(marks) >= 2:
        marks.append(len(text))
        return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)], "step"
    paras = list(re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text))
    if len(paras) >= 2:
        return [(p.start(), p.end()) for p in paras], "para"
    spans = []
    for m in re.finditer(r"[^.!?\n]*[.!?\n]+|\S[^.!?\n]*$", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return (spans or [(0, len(text))]), "sent"


# ============================ statistics core (MODEL-FREE) ============================
def analyze(chains, *, fpr=0.2, calib_frac=0.5, edis_w=8, tau_b=1.36, tau_r=1.33, seed=0):
    """Leak-free blind-spot analysis. Each chain dict needs:
       res (per-step resultant), step_ent (per-step mean entropy),
       ent_tok (per-token entropy), is_correct (bool).
    Annotates each chain with zg/ze/edis/fire_*/fg and returns an aggregate result dict.
    Thresholds use ONLY calib correct chains; FPR is measured on held-out eval correct chains."""
    rng = np.random.default_rng(seed)
    cor_idx = [i for i, c in enumerate(chains) if c["is_correct"]]
    wro_idx = [i for i, c in enumerate(chains) if not c["is_correct"]]
    if len(cor_idx) < 4:
        raise SystemExit("need >=4 correct chains to calibrate")
    perm = rng.permutation(len(cor_idx)); k = max(2, int(len(cor_idx) * calib_frac))
    calib = [chains[cor_idx[j]] for j in perm[:k]]
    ev = [chains[cor_idx[j]] for j in perm[k:]]

    def floor(vals):
        v = [x for x in vals if np.isfinite(x)]
        return 0.5 * float(np.median(v)) if v else 1.0
    # scale floors from CALIB correct chains only (leak-free preprocessing)
    sf_g = floor([np.nanstd(c["res"]) for c in calib if np.isfinite(c["res"]).sum() >= 2])
    sf_e = floor([np.nanstd(c["step_ent"]) for c in calib if np.isfinite(c["step_ent"]).sum() >= 2])

    for c in chains:
        c["zg"] = causal_z([-x for x in c["res"]], sf_g)       # high = concentration DROPPED
        c["ze"] = causal_z(c["step_ent"], sf_e)                # high = entropy ROSE
        c["edis"] = edis_score(c["ent_tok"], edis_w, tau_b, tau_r)
        c["mg"] = float(np.nanmax(c["zg"])) if np.isfinite(c["zg"]).any() else -np.inf
        c["me"] = float(np.nanmax(c["ze"])) if np.isfinite(c["ze"]).any() else -np.inf

    q = 1.0 - fpr
    thr_g = float(np.quantile([c["mg"] for c in calib], q))
    thr_e = float(np.quantile([c["me"] for c in calib], q))
    thr_d = float(np.quantile([c["edis"] for c in calib], q))

    for c in chains:
        c["fg"] = first_cross(c["zg"], thr_g)                  # step index (for repair) or None
        c["fire_g"] = c["fg"] is not None                      # geometry: step-level any-cross
        c["fire_e"] = first_cross(c["ze"], thr_e) is not None  # entropy : step-level any-cross
        c["fire_d"] = c["edis"] > thr_d                        # EDIS    : chain-level (native)

    fpr_g = float(np.mean([c["fire_g"] for c in ev])) if ev else float("nan")
    fpr_e = float(np.mean([c["fire_e"] for c in ev])) if ev else float("nan")
    fpr_d = float(np.mean([c["fire_d"] for c in ev])) if ev else float("nan")

    wrong = [chains[i] for i in wro_idx]
    absent = lambda c: float(np.mean(c["ent_tok"])) if len(c["ent_tok"]) else np.nan
    blind = [c for c in wrong if c["fire_g"] and not c["fire_e"] and not c["fire_d"]]
    ecaught = [c for c in wrong if c["fire_e"]]
    return dict(
        n=len(chains), n_correct=len(cor_idx), n_wrong=len(wrong),
        n_calib=len(calib), n_eval=len(ev),
        thr_g=thr_g, thr_e=thr_e, thr_edis=thr_d,
        fpr_g=fpr_g, fpr_e=fpr_e, fpr_edis=fpr_d,
        catch_g=int(sum(c["fire_g"] for c in wrong)),
        catch_e=int(sum(c["fire_e"] for c in wrong)),
        catch_edis=int(sum(c["fire_d"] for c in wrong)),
        geom_not_entropy=int(sum(c["fire_g"] and not c["fire_e"] for c in wrong)),
        geom_not_edis=int(sum(c["fire_g"] and not c["fire_d"] for c in wrong)),
        n_blind=len(blind), blind=blind,
        ent_blind=float(np.nanmean([absent(c) for c in blind])) if blind else float("nan"),
        ent_ecaught=float(np.nanmean([absent(c) for c in ecaught])) if ecaught else float("nan"),
        ent_allwrong=float(np.nanmean([absent(c) for c in wrong])) if wrong else float("nan"),
    )


def print_report(r):
    print(f"\nchains {r['n']}  correct {r['n_correct']} (calib {r['n_calib']} / eval {r['n_eval']})  "
          f"wrong {r['n_wrong']}")
    print(f"held-out FPR (fires on EVAL correct):  entropy {r['fpr_e']:.3f}   "
          f"EDIS {r['fpr_edis']:.3f}   geom {r['fpr_g']:.3f}")
    print(f"\nwrong-solution catches (at matched FPR):")
    print(f"  entropy (Halo)            {r['catch_e']:>4d}")
    print(f"  EDIS (burst+peak-valley)  {r['catch_edis']:>4d}")
    print(f"  geometry                  {r['catch_g']:>4d}")
    print(f"  geom \\ entropy            {r['geom_not_entropy']:>4d}   (geom fires, entropy misses)")
    print(f"  geom \\ EDIS               {r['geom_not_edis']:>4d}   (geom fires, EDIS misses)")
    print(f"  BLIND SPOT (geom \\ both)  {r['n_blind']:>4d}   <- entropy AND EDIS both miss; geometry catches")
    print(f"\nblind-spot 'confident' check (mean absolute token entropy):")
    print(f"  blind spot       {r['ent_blind']:.3f}")
    print(f"  entropy-caught   {r['ent_ecaught']:.3f}   (blind spot should be LOWER = confident errors)")
    print(f"  all wrong        {r['ent_allwrong']:.3f}")


# ================================ self-test (no model, numpy only) ================================
def _synth(seed=0):
    """Synthetic chains exercising every quadrant:
       correct      : flat resultant, flat low entropy.
       wrong 'loud' : resultant dip at error step AND entropy spike+burst -> entropy & EDIS fire.
       wrong 'blind': resultant dip at error step, entropy stays low/flat -> only geometry fires.
       wrong 'quiet': no dip, no spike -> nothing fires (realistic neither)."""
    rng = np.random.default_rng(seed); chains = []

    def mk(is_correct, kind=None):
        T = int(rng.integers(6, 14))
        res = list(rng.normal(0.60, 0.015, T))
        step_ent = list(np.clip(rng.normal(0.25, 0.04, T), 0.01, None))
        tok = []
        for se in step_ent:
            tok += list(np.clip(rng.normal(se, 0.05, 4), 0.0, None))
        if not is_correct:
            kk = int(rng.integers(2, T))
            res[kk] -= 0.09                                   # directional collapse (geom signal)
            if kind == "loud":
                step_ent[kk] += 0.9                           # entropy spike (entropy z fires)
                tok = []
                for i, se in enumerate(step_ent):
                    tok += [0.3, 0.9, 1.6, 2.2] if i == kk else \
                        list(np.clip(rng.normal(se, 0.05, 4), 0.0, None))   # burst -> EDIS fires
            elif kind == "quiet":
                res[kk] += 0.09                               # undo dip -> geometry won't fire
        return dict(is_correct=is_correct, res=res, step_ent=step_ent, ent_tok=np.array(tok))

    for _ in range(140):
        chains.append(mk(True))
    for _ in range(20):
        chains.append(mk(False, "loud"))
    for _ in range(24):
        chains.append(mk(False, "blind"))
    for _ in range(10):
        chains.append(mk(False, "quiet"))
    return chains


def selftest():
    print("=== SELFTEST (synthetic signals, model-free; verifies the leak-prone statistics) ===")
    chains = _synth(0)
    r = analyze(chains, fpr=0.2, calib_frac=0.5, seed=1)
    print_report(r)
    print("\nexpected by construction: ~24 blind-spot (geom-only) wrong chains; entropy & EDIS each "
          "catch ~20 'loud'; held-out FPR ~ 0.2; blind-spot entropy << entropy-caught entropy.")
    ok = True
    if not (r["n_blind"] >= 12):
        print(f"  [FAIL] blind spot too small ({r['n_blind']})"); ok = False
    if not (r["ent_blind"] < r["ent_ecaught"]):
        print(f"  [FAIL] blind spot not lower-entropy ({r['ent_blind']:.3f} vs {r['ent_ecaught']:.3f})"); ok = False
    if not (0.05 <= r["fpr_g"] <= 0.40):
        print(f"  [FAIL] geom held-out FPR off target ({r['fpr_g']:.3f})"); ok = False
    if not (r["catch_edis"] >= 10 and r["catch_e"] >= 10):
        print(f"  [WARN] competitors caught fewer 'loud' than expected"); 
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ================================ model (generation only) ================================
class Solver:
    def __init__(self, model_name, layer, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = dict(torch_dtype=torch.float16, device_map=device, output_hidden_states=True)
        try:                                                  # V100: float16 + sdpa, no flash2
            self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", **kw)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval(); self.layer = layer; self.device = device; self._hook = None

    def prompt(self, q):
        msg = [{"role": "user", "content": q + "\nSolve step by step. Begin EACH step with "
                "'Step N:' (Step 1:, Step 2:, ...). End with the final answer in \\boxed{}."}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt, max_new=512, temp=0.7):
        with self.torch.no_grad():
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                                       temperature=max(temp, 1e-5), top_p=0.9,
                                       pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    def signals(self, prompt, solution):
        """per-step resultant + per-step mean entropy + per-token entropy + step spans +
        per-step re-anchor unit direction + segmentation method. Steps with <2 tokens dropped
        (res/step_ent/sp/vecs stay aligned)."""
        torch = self.torch
        full = prompt + solution
        enc = self.tok(full, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        H = out.hidden_states[self.layer][0].float().cpu().numpy()
        logits = out.logits[0].float()
        ent = (-(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1)).cpu().numpy()
        base = len(prompt)
        sol_tok = [i for i, (o0, o1) in enumerate(offsets) if o0 >= base and o1 > o0]
        ent_tok = np.array([float(ent[i]) for i in sol_tok])
        spans, method = step_spans(solution)
        res, step_ent, sp, vecs = [], [], [], []
        for (cs, ce) in spans:
            a, b = cs + base, ce + base
            tix = [i for i, (o0, o1) in enumerate(offsets) if o0 >= a and o1 <= b and o1 > o0]
            if len(tix) < 2:
                continue
            Hs = H[tix]; res.append(resultant(Hs)); step_ent.append(float(ent[tix].mean())); sp.append((cs, ce))
            nrm = np.linalg.norm(Hs, axis=1); ok = nrm > 1e-9
            if ok.sum() >= 2:
                u = Hs[ok] / nrm[ok, None]; w = np.exp(np.arange(u.shape[0]) / max(u.shape[0] - 1, 1)); w /= w.sum()
                p = (w[:, None] * u).sum(0); vecs.append((p / (np.linalg.norm(p) + 1e-9)).astype(np.float32))
            else:
                vecs.append(np.zeros(H.shape[1], np.float32))
        return dict(res=res, step_ent=step_ent, ent_tok=ent_tok, sp=sp, vecs=vecs, method=method)

    # ---- actuators (all regenerate from the trigger-step cut; see intrusion caveat in main) ----
    def _repath(self, q, prefix, temp):
        stem = self.prompt(q) + prefix + "\n\nThat step looks wrong. Let me redo it with a different method.\n"
        cont = self.generate(stem, temp=temp)
        return extract_answer(prefix + "\n" + cont)

    def _compress(self, q, prefix, temp):
        cmsg = [{"role": "user", "content":
                 f"Problem: {q}\n\nReasoning so far (may contain mistakes):\n{prefix}\n\n"
                 "List ONLY the conclusions that are correct and verified, as short bullets. "
                 "Discard any wrong, uncertain, or circular steps."}]
        summary = self.generate(self.tok.apply_chat_template(cmsg, tokenize=False, add_generation_prompt=True),
                                 max_new=200, temp=0.0)
        nprompt = self.prompt(q + f"\n\nVerified progress so far:\n{summary}\n\n"
                              "Continue ONLY from the verified progress; do not repeat mistakes.")
        return extract_answer(self.generate(nprompt, temp=temp))

    def _steer(self, q, prefix, vec, alpha, temp):
        # ABLATION. A forward hook adds alpha*vec to the layer output during regeneration. Caveat:
        # the hook fires on EVERY continuation token, and adding a MEAN vector TRANSLATES the token
        # cloud rather than RE-CONCENTRATING it -- a first-order edit of a second-order (dispersion)
        # quantity. Expected to underperform repath on these errors; kept to report that negative.
        torch = self.torch
        v = torch.tensor(vec, dtype=torch.float16, device=self.device); v = v / (v.norm() + 1e-6)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * v
            return (h,) + out[1:] if isinstance(out, tuple) else h
        self._hook = self.model.model.layers[self.layer].register_forward_hook(hook)
        try:
            cont = self.generate(self.prompt(q) + prefix, temp=temp)
        finally:
            self._hook.remove(); self._hook = None
        return extract_answer(prefix + "\n" + cont)

    def repair(self, chain, actuator, temp, alpha):
        """intervene at the geometry trigger step chain['fg']; return final-answer-correct bool."""
        si = chain["fg"]
        if si is None:
            return False
        cut = chain["sp"][si][0]; prefix = chain["sol"][:cut]; q, gold = chain["q"], chain["gold"]
        if actuator == "repath":
            return correct(self._repath(q, prefix, temp), gold)
        if actuator == "compress":
            return correct(self._compress(q, prefix, temp), gold)
        if actuator == "steer":
            pre = [chain["vecs"][j] for j in range(si) if np.linalg.norm(chain["vecs"][j]) > 0]
            if not pre:
                return False
            return correct(self._steer(q, prefix, np.mean(pre, 0), alpha, temp), gold)
        raise ValueError(actuator)


# ================================ data ================================
def load_problems(path, n):
    if path:
        import json
        out = []
        for line in open(path, encoding="utf-8"):
            d = json.loads(line); q = d.get("question") or d.get("problem")
            out.append((q, str(d["answer"]).strip()))
        return out[:n]
    try:
        from datasets import load_dataset
        ds = load_dataset("openai/gsm8k", "main", split=f"test[:{n}]")
        return [(d["question"], extract_answer(d["answer"])) for d in ds]
    except Exception as e:
        print(f"[datasets unavailable: {e}; using builtin GSM8K]")
        B = [("Natalia sold clips to 48 friends in April, and half as many in May. How many altogether?", "72"),
             ("Weng earns $12 an hour. Yesterday she babysat 50 minutes. How much did she earn?", "10"),
             ("James writes a 3-page letter to 2 friends twice a week. How many pages a year?", "624"),
             ("A robe takes 2 bolts of blue fiber and half that of white. How many bolts total?", "3"),
             ("Mark has 28 red flowers, 20% fewer yellow, and as many blue as red+yellow. Total flowers?", "129"),
             ("Albert eats 2 large pizzas (16 slices) and 2 small (8 slices). How many slices?", "48")]
        return (B * (n // len(B) + 1))[:n]


# ================================ main ================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the model-free statistics check and exit")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--repath_temp", type=float, default=1.0, help="high temp for repath (escape confident attractor)")
    ap.add_argument("--alpha", type=float, default=6.0, help="steering strength (ablation)")
    ap.add_argument("--repair", default="repath,compress", help="comma list: repath,compress,steer")
    ap.add_argument("--data_jsonl", default=None)
    ap.add_argument("--fpr", type=float, default=0.2, help="target held-out fire rate on correct chains")
    ap.add_argument("--calib_frac", type=float, default=0.5)
    ap.add_argument("--min_steps", type=int, default=3)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest() else 1)

    probs = load_problems(args.data_jsonl, args.n)
    S = Solver(args.model, args.layer)

    chains, methods = [], {"step": 0, "para": 0, "sent": 0}
    for qi, (q, gold) in enumerate(probs):
        prompt = S.prompt(q); sol = S.generate(prompt, temp=args.temp)
        sig = S.signals(prompt, sol)
        if len(sig["res"]) < args.min_steps:
            continue
        methods[sig["method"]] += 1
        chains.append(dict(q=q, prompt=prompt, sol=sol, gold=gold,
                           is_correct=correct(extract_answer(sol), gold),
                           res=np.asarray(sig["res"], float), step_ent=np.asarray(sig["step_ent"], float),
                           ent_tok=sig["ent_tok"], sp=sig["sp"], vecs=sig["vecs"]))
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}/{len(probs)}] kept {len(chains)}  "
                  f"baseline {np.mean([c['is_correct'] for c in chains]):.3f}")

    nbase = float(np.mean([c["is_correct"] for c in chains])) if chains else 0.0
    print(f"\nmodel {args.model} | layer {args.layer} | kept {len(chains)} | baseline pass {nbase:.3f}")
    print(f"segmentation used: step {methods['step']}  para {methods['para']}  sent {methods['sent']}  "
          f"(prefer 'step'; many fallbacks => the validated granularity may not transfer)")

    r = analyze(chains, fpr=args.fpr, calib_frac=args.calib_frac, seed=0)
    print_report(r)

    blind = r["blind"]
    if not blind:
        print("\nno blind-spot errors at this FPR/N -- scale --n (low-baseline benchmark gives more).")
        return
    print(f"\nBLIND-SPOT REPAIR ({len(blind)} errors entropy AND EDIS missed, geometry caught):")
    print("  (pass(after) is NOT the headline -- the blind-spot SIZE above is; this is the repair side)")
    for act in [a.strip() for a in args.repair.split(",") if a.strip()]:
        t = args.repath_temp if act == "repath" else args.temp
        fixed = sum(S.repair(c, act, t, args.alpha) for c in blind)
        tag = {"repath": "high-T different-method", "compress": "Halo state-reset",
               "steer": f"ablation a={args.alpha:g}, mechanism-mismatched"}[act]
        print(f"  {act:9s} fixed {fixed:>3d}/{len(blind)}  ({fixed/len(blind):.2f})   [{tag}]")
    print("\nread: HEADLINE = blind-spot SIZE (geom catches, entropy+EDIS both miss) with the held-out "
          "FPR matched across triggers, AND blind-spot entropy < entropy-caught entropy (genuinely "
          "confident errors). Repair is secondary; hypothesis: repath > compress on confident errors; "
          "steer underperforms (adding a mean translates, not re-concentrates). Intrusion differs across "
          "actuators (compress discards reasoning text; repath/steer continue) -- report, don't hide.")


if __name__ == "__main__":
    main()