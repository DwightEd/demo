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
        res, en, sp = [], [], []
        for (cs, ce) in step_spans(solution):              # STEP granularity (matches validation)
            a, b = cs + base, ce + base
            tok_idx = [i for i, (o0, o1) in enumerate(offsets) if o0 >= a and o1 <= b and o1 > o0]
            if len(tok_idx) >= 2:
                res.append(resultant(H[tok_idx])); en.append(float(ent[tok_idx].mean())); sp.append((cs, ce))
        return np.array(res), np.array(en), sp

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


def sol_score(res, en, kind):
    """per-solution anomaly = max over sentences of the within-solution z-deviation, with the
    argmax sentence index. geom: -R (concentration drop); entropy: entropy spike. Higher = more
    suspicious. Returns (score, idx) or (-inf, None) if too short."""
    if len(res) < 3:
        return -np.inf, None
    s = -res if kind == "geom" else en
    best, bi = -np.inf, None
    for t in range(2, len(s)):
        hist = s[:t]; mu, sd = hist.mean(), hist.std() + 1e-6
        a = (s[t] - mu) / sd
        if a > best:
            best, bi = a, t
    return best, bi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--actuator", choices=["reconsider", "steer"], default="reconsider")
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
        res, en, sp = S.signals(prompt, sol)
        if len(res) < 3:
            continue
        gs, gi = sol_score(res, en, "geom"); es, ei = sol_score(res, en, "entropy")
        conf = bool(en.mean() < np.median(en)) if len(en) else False
        items.append(dict(prompt=prompt, sol=sol, gold=gold, ok=ok, sp=sp,
                          gs=gs, gi=gi, es=es, ei=ei, conf=conf))
        if (qi + 1) % 5 == 0:
            print(f"  [{qi+1}/{len(probs)}] baseline pass {np.mean([it['ok'] for it in items]):.3f}")
    nbase = float(np.mean([it["ok"] for it in items])) if items else 0.0

    # ---- conformal calibration: threshold on CORRECT solutions to fire at ~FPR (precision-first) ----
    cor = [it for it in items if it["ok"]]
    thr = {}
    for kind, sk in [("geom", "gs"), ("entropy", "es")]:
        vals = np.array([it[sk] for it in cor]) if cor else np.array([np.inf])
        thr[kind] = float(np.quantile(vals, 1 - args.fpr)) if len(vals) else np.inf

    # ---- pass 2: intervene only where anomaly > calibrated threshold ----
    rows = {k: dict(fired=0, fire_wrong=0, fire_correct=0, after_ok=0, conf_wrong=0, conf_fixed=0)
            for k in ["geom", "entropy"]}
    nn = max(len(items), 1)
    for it in items:
        for kind, sk, ik in [("geom", "gs", "gi"), ("entropy", "es", "ei")]:
            r = rows[kind]; r["conf_wrong"] += int((not it["ok"]) and it["conf"])
            if not (it[sk] > thr[kind] and it[ik] is not None):
                r["after_ok"] += int(it["ok"]); continue                   # no fire -> keep baseline
            r["fired"] += 1; r["fire_wrong"] += int(not it["ok"]); r["fire_correct"] += int(it["ok"])
            cut = it["sp"][it[ik]][0]; stem = it["prompt"] + it["sol"][:cut]
            if args.actuator == "steer":
                S.set_steer(np.ones(S.model.config.hidden_size), args.alpha)
                new = S.generate(stem, temp=args.temp); S.clear_steer()
            else:
                new = S.generate(stem + "\nWait, let me re-check this step carefully.\n", temp=max(args.temp, 0.9))
            aok = correct(extract_answer(it["sol"][:cut] + new), it["gold"])
            r["after_ok"] += int(aok)
            if (not it["ok"]) and it["conf"] and aok:
                r["conf_fixed"] += 1

    print(f"\nmodel {args.model} | layer {args.layer} | N {len(items)} | actuator {args.actuator} | FPR {args.fpr}")
    print(f"baseline pass-rate: {nbase:.3f}")
    print(f"\n{'trigger':9s} {'fired':>6s} {'on-wrong':>9s} {'on-correct':>11s} {'pass(after)':>12s} "
          f"{'conf-wrong':>11s} {'conf-fixed':>11s}")
    for kind in ["geom", "entropy"]:
        r = rows[kind]
        print(f"{kind:9s} {r['fired']:>6d} {r['fire_wrong']:>9d} {r['fire_correct']:>11d} "
              f"{r['after_ok']/nn:>12.3f} {r['conf_wrong']:>11d} {r['conf_fixed']:>11d}")
    print("\nread: threshold conformal-calibrated on CORRECT solutions to fire at ~FPR, so 'on-correct' "
          "stays small. PRECISION = on-wrong/fired -> a good trigger fires mostly on wrong solutions. "
          "pass(after) > baseline = intervention nets positive. conf-fixed = confident hallucinations "
          "repaired (headline). USE A HARD benchmark (low baseline) so there are errors to fix -- GSM8K "
          "(~0.75) breaks more corrects than it fixes. Compare geom vs entropy at the same FPR.")


if __name__ == "__main__":
    main()
