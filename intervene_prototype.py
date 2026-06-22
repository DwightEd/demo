"""Generation-time intervention prototype: geometric-trigger vs entropy-trigger resampling.

Closed loop: generate a CoT solution; segment into sentences; per sentence compute the geometric
signal (resultant = directional concentration, mid layer) AND token entropy; at the first TRIGGER
sentence, intervene (truncate + reconsider-and-regenerate, or latent steering) and re-solve.
Compare final pass-rate across triggers {none, geom, entropy} -- the headline is whether the
GEOMETRIC trigger fixes CONFIDENT hallucinations (low-entropy errors) that the entropy trigger
misses.

This is a FIRST prototype to get the loop running on a small N. Actuators:
  --actuator reconsider : truncate at trigger, append a reconsider nudge, regenerate (no hooks).
  --actuator steer      : latent-space steering -- a forward hook adds alpha * (re-anchor direction)
                          to the mid-layer residual during regeneration (the user's latent intervention).

NOTE: confident hallucinations resist plain resampling (low entropy -> same output), so the
reconsider nudge / steering is what breaks the confident-wrong attractor. Run small N first.

Needs: transformers, torch, a CoT model (default Llama-3.1-8B-Instruct), and GSM8K (datasets) or a
fallback handful of problems so it always runs.
"""

from __future__ import annotations
import argparse
import re
import numpy as np

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise SystemExit("needs torch + transformers")


# ----------------------------- signal helpers -----------------------------
def resultant(H):
    """H: (n_tok, d) mid-layer hidden states -> directional concentration (exp-pooled unit vectors)."""
    H = H.astype(np.float64); nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return np.nan
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
    return float(np.linalg.norm((w[:, None] * u).sum(0)))


def sent_spans(text):
    """fallback: split into sentences -> list of (start_char, end_char)."""
    spans = []
    for m in re.finditer(r"[^.!?\n]*[.!?\n]+|\S[^.!?\n]*$", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans or [(0, len(text))]


def step_spans(text):
    """STEP-granularity segmentation to MATCH the validation granularity (not sentences).
    Prefer 'Step N:' markers; else blank-line paragraphs; else sentences."""
    marks = [m.start() for m in re.finditer(r"(?im)^\s*step\s*\d+\s*[:.\)]", text)]
    if len(marks) >= 2:
        marks.append(len(text))
        return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    paras = [m for m in re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text)]
    if len(paras) >= 2:
        return [(p.start(), p.end()) for p in paras]
    return sent_spans(text)


def extract_boxed(text):
    """content of the LAST \\boxed{...} with balanced braces."""
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
    """answer from a generated SOLUTION: prefer \\boxed content, then ####, then last number."""
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
    """robust-ish MATH/GSM8K equality: normalized string match OR numeric match."""
    if pred is None or gold is None:
        return False
    p, g = _norm(pred), _norm(gold)
    if p == g:
        return True
    try:
        return abs(float(p) - float(g)) < 1e-4
    except ValueError:
        return False


# ----------------------------- generation + signals -----------------------------
class Solver:
    def __init__(self, model_name, layer, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                          device_map=device, output_hidden_states=True)
        self.model.eval(); self.layer = layer; self.device = device
        self.steer_vec = None; self._hook = None

    def prompt(self, q):
        msg = [{"role": "user", "content": q + "\nSolve step by step. Begin EACH step with "
                "'Step N:' (Step 1:, Step 2:, ...). End with the final answer in \\boxed{}."}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def generate(self, prompt, max_new=512, temp=0.7):
        ids = self.tok(prompt, return_tensors="pt").to(self.device)
        out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                                  temperature=max(temp, 1e-5), top_p=0.9,
                                  pad_token_id=self.tok.eos_token_id)
        gen = out[0][ids.input_ids.shape[1]:]
        return self.tok.decode(gen, skip_special_tokens=True)

    @torch.no_grad()
    def signals(self, prompt, solution):
        """per-sentence (resultant, mean token entropy) over the solution text."""
        full = prompt + solution
        enc = self.tok(full, return_tensors="pt", return_offsets_mapping=True)
        offsets = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        out = self.model(**enc)
        H = out.hidden_states[self.layer][0].float().cpu().numpy()          # (seq, d)
        logits = out.logits[0].float()                                      # (seq, vocab)
        ent = (-(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1)).cpu().numpy()
        base = len(prompt)
        res, en, sp, vecs = [], [], [], []
        ent_tok = np.array([ent[i] for i, (o0, o1) in enumerate(offsets) if o0 >= base and o1 > o0])  # per-token (EDIS)
        for (cs, ce) in step_spans(solution):              # STEP granularity (matches validation)
            a, b = cs + base, ce + base
            tok_idx = [i for i, (o0, o1) in enumerate(offsets) if o0 >= a and o1 <= b and o1 > o0]
            if len(tok_idx) >= 2:
                Hs = H[tok_idx]; res.append(resultant(Hs)); en.append(float(ent[tok_idx].mean())); sp.append((cs, ce))
                nrm = np.linalg.norm(Hs, axis=1); ok = nrm > 1e-9        # full-dim pooled direction (re-anchor)
                if ok.sum() >= 2:
                    u = Hs[ok] / nrm[ok, None]; w = np.exp(np.arange(u.shape[0]) / max(u.shape[0] - 1, 1)); w /= w.sum()
                    p = (w[:, None] * u).sum(0); vecs.append(p / (np.linalg.norm(p) + 1e-9))
                else:
                    vecs.append(np.zeros(H.shape[1], np.float32))
        return np.array(res), np.array(en), sp, vecs, ent_tok

    @torch.no_grad()
    def compress_reset(self, q, partial, temp):
        """Halo-style actuator: semantic compression of verified progress + history reset, then
        regenerate. Returns the regenerated solution (its \\boxed answer is the new final answer)."""
        cmsg = [{"role": "user", "content":
                 f"Problem: {q}\n\nReasoning so far (may contain mistakes):\n{partial}\n\n"
                 "List ONLY the conclusions that are correct and verified, as short bullets. "
                 "Discard any wrong, uncertain, or circular steps."}]
        cprompt = self.tok.apply_chat_template(cmsg, tokenize=False, add_generation_prompt=True)
        summary = self.generate(cprompt, max_new=200, temp=0.0)
        nprompt = self.prompt(q + f"\n\nVerified progress so far:\n{summary}\n\n"
                              "Continue ONLY from the verified progress; do not repeat mistakes.")
        return self.generate(nprompt, temp=temp)

    def set_steer(self, vec, alpha):
        """register a hook that adds alpha*vec to the mid-layer residual output (latent steering)."""
        self.clear_steer()
        v = torch.tensor(vec, dtype=torch.float16, device=self.device)
        v = v / (v.norm() + 1e-6)

        def hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            h = h + alpha * v
            return (h,) + out[1:] if isinstance(out, tuple) else h
        self._hook = self.model.model.layers[self.layer].register_forward_hook(hook)

    def clear_steer(self):
        if self._hook is not None:
            self._hook.remove(); self._hook = None


def step_resid(res, en, kind):
    """per-STEP causal within-chain residual sequence (nan for t<2). geom = -R (concentration
    drop), entropy = entropy spike; higher = more suspicious. This is the ONLINE step-level signal
    (matches Halo's per-step granularity and exploits geometry's instantaneous dip), vs a chain-
    level max which would drown the spike."""
    s = (-np.asarray(res, float)) if kind == "geom" else np.asarray(en, float)
    out = np.full(len(s), np.nan)
    for t in range(2, len(s)):
        h = s[:t]; out[t] = (s[t] - h.mean()) / (h.std() + 1e-6)
    return out


def first_cross(r, thr):
    """first step index whose residual crosses the threshold (online trigger), else None."""
    idx = np.where(r > thr)[0]
    return int(idx[0]) if len(idx) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--actuator", choices=["compress", "reconsider", "steer"], default="compress")
    ap.add_argument("--alpha", type=float, default=6.0, help="steering strength")
    ap.add_argument("--data_jsonl", default=None, help="local jsonl with question/problem + answer")
    ap.add_argument("--fpr", type=float, default=0.2, help="target fire rate on CORRECT solutions (conformal)")
    args = ap.parse_args()

    # data: --data_jsonl (local: {"question"|"problem", "answer"}), else openai/gsm8k, else builtin
    probs = None
    if args.data_jsonl:
        import json
        probs = []
        for line in open(args.data_jsonl, encoding="utf-8"):
            d = json.loads(line); q = d.get("question") or d.get("problem")
            probs.append((q, str(d["answer"]).strip()))      # gold answer AS-IS (already clean)
        probs = probs[:args.n]
    if probs is None:
        try:
            from datasets import load_dataset
            ds = load_dataset("openai/gsm8k", "main", split=f"test[:{args.n}]")
            probs = [(d["question"], extract_answer(d["answer"])) for d in ds]
        except Exception as e:
            print(f"[datasets unavailable: {e}; using builtin GSM8K problems]")
            BUILTIN = [
                ("Natalia sold clips to 48 friends in April, and half as many in May. How many clips did she sell altogether?", "72"),
                ("Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. How much did she earn?", "10"),
                ("Betty is saving for a $100 wallet. She has half of the money. Her parents give her $15 and her grandparents twice as much as her parents. How much more does she need?", "5"),
                ("James writes a 3-page letter to 2 friends twice a week. How many pages does he write a year?", "624"),
                ("A robe takes 2 bolts of blue fiber and half that of white fiber. How many bolts in total?", "3"),
                ("Mark has a garden with 28 red flowers. There are 20% fewer yellow flowers. There are as many blue as red and yellow combined. How many flowers total?", "129"),
                ("Albert buys 2 large pizzas (16 slices each) and 2 small pizzas (8 slices each). If he eats it all, how many slices does he eat?", "48"),
                ("Ken created a care package. He put 2 pounds of jelly beans, then added enough brownies to triple the weight, then 2 more pounds of jelly beans, then doubled it. Final weight in pounds?", "16"),
                ("Alexis bought items spending 30+46+38+11+18 and had $16 left, plus shoes whose price she forgot. She started with $200. How much were the shoes?", "41"),
                ("Tina makes $18.00 per hour. Over 8 hours she gets paid normally; beyond 8 she gets 1.5x. If she works 10 hours a day for 5 days, how much does she make?", "990"),
                ("A deep-sea monster eats ships every 100 years. Over 300 years it ate 847 people. Each new ship had twice as many people as the last. How many were on the first ship?", "121"),
                ("Marcus has 210 baseball cards. He has 58 more than Carter. How many cards do Marcus and Carter have together?", "362"),
            ]
            probs = (BUILTIN * (args.n // len(BUILTIN) + 1))[:args.n]

    S = Solver(args.model, args.layer)
    # ---- pass 1: baseline generate + per-solution anomaly scores (no intervention yet) ----
    items = []
    for qi, (q, gold) in enumerate(probs):
        prompt = S.prompt(q); sol = S.generate(prompt, temp=args.temp)
        ok = correct(extract_answer(sol), gold)
        res, en, sp, vecs, _ = S.signals(prompt, sol)
        if len(res) < 3:
            continue
        items.append(dict(q=q, prompt=prompt, sol=sol, gold=gold, ok=ok, sp=sp, vecs=vecs,
                          res=np.asarray(res, float), en=np.asarray(en, float)))
        if (qi + 1) % 5 == 0:
            print(f"  [{qi+1}/{len(probs)}] baseline pass {np.mean([it['ok'] for it in items]):.3f}")
    nbase = float(np.mean([it["ok"] for it in items])) if items else 0.0

    # ---- ONLINE step-level trigger: per-step residual, conformal threshold from correct chains ----
    for it in items:
        it["rg"] = step_resid(it["res"], it["en"], "geom"); it["re_"] = step_resid(it["res"], it["en"], "entropy")
        it["mg"] = float(np.nanmax(it["rg"])) if np.isfinite(it["rg"]).any() else -np.inf
        it["me"] = float(np.nanmax(it["re_"])) if np.isfinite(it["re_"]).any() else -np.inf
    cor = [it for it in items if it["ok"]]
    thr_g = float(np.quantile([it["mg"] for it in cor], 1 - args.fpr)) if cor else np.inf
    thr_e = float(np.quantile([it["me"] for it in cor], 1 - args.fpr)) if cor else np.inf
    for it in items:
        it["fg"] = first_cross(it["rg"], thr_g); it["fe"] = first_cross(it["re_"], thr_e)

    def fix_at(it, step_idx, actuator):
        """intervene at a given step index with a given actuator; return final-correct bool."""
        if step_idx is None:
            return it["ok"]
        cut = it["sp"][step_idx][0]; stem = it["prompt"] + it["sol"][:cut]
        if actuator == "compress":
            return correct(extract_answer(S.compress_reset(it["q"], it["sol"][:cut], args.temp)), it["gold"])
        if actuator == "steer":
            pre = [it["vecs"][j] for j in range(step_idx) if np.linalg.norm(it["vecs"][j]) > 0]
            if not pre:
                return it["ok"]
            S.set_steer(np.mean(pre, 0), args.alpha); new = S.generate(stem, temp=args.temp); S.clear_steer()
            return correct(extract_answer(it["sol"][:cut] + new), it["gold"])
        return it["ok"]

    # ---- quadrants on WRONG solutions (online step-level firing) ----
    wrong = [it for it in items if not it["ok"]]
    both = [it for it in wrong if it["fe"] is not None and it["fg"] is not None]
    eonly = [it for it in wrong if it["fe"] is not None and it["fg"] is None]
    gonly = [it for it in wrong if it["fe"] is None and it["fg"] is not None]   # BLIND SPOT
    neither = [it for it in wrong if it["fe"] is None and it["fg"] is None]
    nc = max(len(cor), 1)
    fpr_g = sum(it["fg"] is not None for it in cor) / nc
    fpr_e = sum(it["fe"] is not None for it in cor) / nc

    print(f"\nmodel {args.model} | layer {args.layer} | N {len(items)} | online step-level | "
          f"alpha {args.alpha} | baseline {nbase:.3f}")
    print(f"empirical FPR (fires on correct): entropy {fpr_e:.3f}  geom {fpr_g:.3f}   wrong solutions: {len(wrong)}")
    print(f"\nWRONG-solution quadrants (who fires):")
    print(f"  both fire            {len(both):>4d}")
    print(f"  entropy-only         {len(eonly):>4d}")
    print(f"  GEOM-ONLY (blindspot){len(gonly):>4d}   <- errors entropy STRUCTURALLY misses, geometry catches")
    print(f"  neither              {len(neither):>4d}")

    # ---- on the BLIND SPOT: which actuator repairs it? compress (Halo-style) vs steer (geometric) ----
    bs_comp = sum(fix_at(it, it["fg"], "compress") for it in gonly)
    bs_steer = sum(fix_at(it, it["fg"], "steer") for it in gonly)
    # reference: on entropy-caught errors, does the actuator work at all (sanity)
    ec_comp = sum(fix_at(it, it["fe"], "compress") for it in (eonly[:len(gonly)] or eonly))
    ng = max(len(gonly), 1)
    print(f"\nBLIND SPOT repair ({len(gonly)} errors entropy missed, geometry caught):")
    print(f"  compress (Halo actuator)  fixed {bs_comp:>3d}/{len(gonly)}  ({bs_comp/ng:.2f})")
    print(f"  steer    (geometric, a={args.alpha:g}) fixed {bs_steer:>3d}/{len(gonly)}  ({bs_steer/ng:.2f})")
    print(f"  [ref] compress on entropy-caught: {ec_comp}/{len(eonly[:len(gonly)] or eonly)}")
    print("\nread: HEADLINE = blind-spot size (GEOM-ONLY) > 0 -> geometry catches errors entropy structurally "
          "misses (claim independent of any actuator). Then repair: hypothesis is steer (push representation "
          "back to the chain's healthy concentrated direction) beats compress on these LOW-entropy confident "
          "errors -- 'geometric signal + geometric actuator fixes the geometric blind spot'. Sweep --alpha "
          "{2,4,6,8,12} for steer. Scale --n (more wrong solutions). pass(after) is NOT the headline (the "
          "blind spot is diluted in it); bs-fixed by actuator on the GEOM-ONLY set is.")


if __name__ == "__main__":
    main()
