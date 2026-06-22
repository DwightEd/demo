#!/usr/bin/env python3
"""Detection-triggered best-of-N REROUTING: the intervention main line.

Online, during generation, a per-step detector flags a likely-error step; we truncate there, sample
N high-temp continuations (REROUTE, not repair), and pick the final answer by majority vote among the
N. The ONLY thing that differs across methods is the TRIGGER signal -- same conformal FPR, same
best-of-N actuator, same neutral selection -- so any pass@1 gap isolates TRIGGER QUALITY:

  baseline   : no intervention
  Halo       : entropy trigger (step mean-entropy z-score), conformal FPR        [SOTA-style]
  OURS       : geometry (pooled-norm collapse) OR entropy trigger, SAME conformal FPR
  self-cons  : no trigger, but spend the SAME extra-sample budget as best-of-N at the END (control)

Hypothesis: OURS > Halo because the geometric trigger fires on CONFIDENT errors (low entropy) that
the entropy trigger misses -> more of those chains get rerouted -> higher pass@1. Precision-first
operating point (low FPR) so correct chains are rarely disturbed.

Calibration is leak-free: thresholds (and z-score mu/sd) are set ONLY on a CALIB split of problems;
pass@1 is reported on a disjoint TEST split.

Needs a model + a jsonl of {question/problem, answer}. GPU. (Llama-3.1-8B-Instruct, layer 14.)
"""

from __future__ import annotations
import argparse
import json
import re
import numpy as np
from collections import Counter


# ============================ answer checking ============================
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
                 ("\\dfrac", "\\frac"), ("\\tfrac", "\\frac"), ("{}", "")]:
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


def step_spans(text):
    marks = [m.start() for m in re.finditer(r"(?im)^\s*step\s*\d+\s*[:.\)]", text)]
    if len(marks) >= 2:
        marks.append(len(text))
        return [(marks[i], marks[i + 1]) for i in range(len(marks) - 1)]
    paras = list(re.finditer(r"\S[^\n]*(?:\n(?!\s*\n)[^\n]*)*", text))
    return [(p.start(), p.end()) for p in paras] if len(paras) >= 2 else [(0, len(text))]


# ============================ model ============================
class Solver:
    def __init__(self, model_name, layer, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = dict(torch_dtype=torch.float16, device_map=device, output_hidden_states=True)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", **kw)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval(); self.layer = layer; self.device = device

    def prompt(self, q):
        msg = [{"role": "user", "content": q + "\nSolve step by step. Begin EACH step with "
                "'Step N:' (Step 1:, Step 2:, ...). End with the final answer in \\boxed{}."}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def generate(self, prompt, max_new=512, temp=0.7):
        with self.torch.no_grad():
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                                       temperature=max(temp, 1e-5), top_p=0.95,
                                       pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    def step_signals(self, prompt, solution):
        """per-step (pooled_norm geometry, mean entropy) + char spans. One forward pass."""
        torch = self.torch
        enc = self.tok(prompt + solution, return_tensors="pt", return_offsets_mapping=True)
        offs = enc.pop("offset_mapping")[0].tolist()
        enc = {k: v.to(self.device) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc)
        H = out.hidden_states[self.layer][0].float().cpu().numpy()
        logits = out.logits[0].float()
        ent = (-(torch.softmax(logits, -1) * torch.log_softmax(logits, -1)).sum(-1)).cpu().numpy()
        base = len(prompt); pn, en, sp = [], [], []
        for (cs, ce) in step_spans(solution):
            a, b = cs + base, ce + base
            tix = [i for i, (o0, o1) in enumerate(offs) if o0 >= a and o1 <= b and o1 > o0]
            if len(tix) < 2:
                continue
            Hs = H[tix]; n = Hs.shape[0]
            w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
            pn.append(float(np.linalg.norm((w[:, None] * Hs).sum(0))))   # pooled model-length (geometry)
            en.append(float(ent[tix].mean())); sp.append((cs, ce))
        return np.array(pn), np.array(en), sp


# ============================ trigger calibration & reroute ============================
def chain_scores(pn, en, mu_g, sd_g, mu_e, sd_e):
    """per-step z-badness: geometry collapse (-pooled_norm) and entropy. Returns gz, ez arrays."""
    gz = (-pn - mu_g) / (sd_g + 1e-9)
    ez = (en - mu_e) / (sd_e + 1e-9)
    return gz, ez


def first_fire(score, thr):
    idx = np.where(np.isfinite(score) & (score > thr))[0]
    return int(idx[0]) if len(idx) else None


def reroute(S, q, prefix, n, temp):
    """best-of-N from the truncation: sample N high-temp continuations, majority-vote the answer."""
    answers = []
    for _ in range(n):
        cont = S.generate(S.prompt(q) + prefix, temp=temp)
        answers.append(extract_answer(prefix + "\n" + cont))
    votes = Counter(_norm(a) for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--data_jsonl", required=True)
    ap.add_argument("--n_calib", type=int, default=80)
    ap.add_argument("--n_test", type=int, default=200)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--reroute_temp", type=float, default=1.0)
    ap.add_argument("--bestof", type=int, default=5, help="N continuations at a trigger")
    ap.add_argument("--fpr", type=float, default=0.15, help="precision-first chain-FPR on correct calib chains")
    args = ap.parse_args()

    probs = []
    for line in open(args.data_jsonl, encoding="utf-8"):
        d = json.loads(line); q = d.get("question") or d.get("problem")
        probs.append((q, str(d["answer"]).strip()))
    calib = probs[:args.n_calib]; test = probs[args.n_calib:args.n_calib + args.n_test]

    S = Solver(args.model, args.layer)

    # ---- CALIB: baseline gen, collect step stats on CORRECT chains, set matched-FPR thresholds ----
    cal = []
    for qi, (q, gold) in enumerate(calib):
        pr = S.prompt(q); sol = S.generate(pr, temp=args.temp)
        pn, en, sp = S.step_signals(pr, sol)
        if len(pn) < 2:
            continue
        cal.append(dict(pn=pn, en=en, ok=correct(extract_answer(sol), gold)))
    cor = [c for c in cal if c["ok"]]
    allpn = np.concatenate([c["pn"] for c in cor]); allen = np.concatenate([c["en"] for c in cor])
    mu_g, sd_g = float((-allpn).mean()), float((-allpn).std())
    mu_e, sd_e = float(allen.mean()), float(allen.std())
    # chain-level max score on correct chains -> threshold at (1-fpr) quantile (matched FPR per signal)
    halo_peak, ours_peak = [], []
    for c in cor:
        gz, ez = chain_scores(c["pn"], c["en"], mu_g, sd_g, mu_e, sd_e)
        halo_peak.append(float(np.nanmax(ez)))
        ours_peak.append(float(np.nanmax(np.maximum(gz, ez))))
    thr_halo = float(np.quantile(halo_peak, 1 - args.fpr))
    thr_ours = float(np.quantile(ours_peak, 1 - args.fpr))
    print(f"calib: {len(cal)} chains ({len(cor)} correct) | thr_halo {thr_halo:.2f} thr_ours {thr_ours:.2f} "
          f"| matched chain-FPR target {args.fpr}")

    # ---- TEST: baseline + Halo-reroute + OURS-reroute + budget-matched self-consistency ----
    acc = {"baseline": 0, "Halo": 0, "OURS": 0, "self-cons": 0}
    fire = {"Halo": 0, "OURS": 0}; broke = {"Halo": 0, "OURS": 0}; fixed = {"Halo": 0, "OURS": 0}
    n = 0
    for qi, (q, gold) in enumerate(test):
        pr = S.prompt(q); sol = S.generate(pr, temp=args.temp)
        pn, en, sp = S.step_signals(pr, sol)
        if len(pn) < 2:
            continue
        n += 1
        base_ans = extract_answer(sol); base_ok = correct(base_ans, gold)
        acc["baseline"] += int(base_ok)
        gz, ez = chain_scores(pn, en, mu_g, sd_g, mu_e, sd_e)
        triggers = {"Halo": first_fire(ez, thr_halo),
                    "OURS": first_fire(np.maximum(gz, ez), thr_ours)}
        for m, fj in triggers.items():
            if fj is None or fj >= len(sp):
                acc[m] += int(base_ok); continue                # no trigger -> keep baseline answer
            fire[m] += 1
            cut = sp[fj][0]; ans = reroute(S, q, sol[:cut], args.bestof, args.reroute_temp)
            ok = correct(ans, gold); acc[m] += int(ok)
            if base_ok and not ok:
                broke[m] += 1                                   # broke a correct chain (precision cost)
            if (not base_ok) and ok:
                fixed[m] += 1                                   # fixed a wrong chain
        # budget-matched self-consistency: spend bestof extra samples at the END, majority vote with base
        extra = [extract_answer(S.generate(pr, temp=args.temp)) for _ in range(args.bestof)]
        votes = Counter(_norm(a) for a in ([base_ans] + extra) if a is not None)
        acc["self-cons"] += int(correct(votes.most_common(1)[0][0], gold) if votes else False)
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}] n={n} base {acc['baseline']/n:.3f} Halo {acc['Halo']/n:.3f} "
                  f"OURS {acc['OURS']/n:.3f}")

    print(f"\nmodel {args.model} | test {n} | bestof {args.bestof} | reroute_temp {args.reroute_temp} | fpr {args.fpr}")
    print(f"\n{'method':14s} {'pass@1':>7s}")
    for m in ["baseline", "self-cons", "Halo", "OURS"]:
        print(f"  {m:14s} {acc[m]/max(n,1):7.3f}")
    print(f"\ntriggers (test): Halo {fire['Halo']}  OURS {fire['OURS']}   "
          f"(OURS-extra = confident errors Halo's entropy trigger missed)")
    print(f"fixed wrong:     Halo {fixed['Halo']}  OURS {fixed['OURS']}")
    print(f"broke correct:   Halo {broke['Halo']}  OURS {broke['OURS']}  (precision cost; precision-first keeps low)")
    print("\nread: HEADLINE = OURS pass@1 > Halo pass@1 at matched FPR/budget, with OURS firing on MORE wrong "
          "chains (the extra = confident errors the entropy trigger misses) and fixing more, at a comparable "
          "broke-correct cost. self-cons is the budget-matched non-triggered control. If OURS ~ Halo, the "
          "geometric trigger does not help intervention; if OURS > Halo with fixed>broke, the detection edge "
          "translates into a correction edge -- the two-main-line story closes.")


if __name__ == "__main__":
    main()
