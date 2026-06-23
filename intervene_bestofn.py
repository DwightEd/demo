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


def act_bestofn(S, q, prefix, args):
    """best-of-N from the truncation: sample N high-temp continuations, majority-vote (weakest actuator)."""
    answers = []
    for _ in range(args.bestof):
        cont = S.generate(S.prompt(q) + prefix, temp=args.reroute_temp)
        answers.append(extract_answer(prefix + "\n" + cont))
    votes = Counter(_norm(a) for a in answers if a is not None)
    return votes.most_common(1)[0][0] if votes else None


def act_repath(S, q, prefix, args):
    """REPATH: truncate, high-temp regenerate with an explicit DIFFERENT-METHOD instruction -- a
    corrective push (not just resampling), aimed at escaping a confident-wrong attractor."""
    stem = S.prompt(q) + prefix + ("\n\nThe step above is likely a mistake. Discard it and solve this "
                                   "part again from here using a DIFFERENT method or approach.\n")
    cont = S.generate(stem, temp=args.reroute_temp)
    return extract_answer(prefix + "\n" + cont)


def act_compress(S, q, prefix, args):
    """COMPRESS+RESET (Halo-style context surgery): summarize ONLY the verified-correct progress so
    far, discard the rest, regenerate from [question + verified summary]."""
    cmsg = [{"role": "user", "content":
             f"Problem: {q}\n\nReasoning so far (may contain mistakes):\n{prefix}\n\nList ONLY the "
             "conclusions that are correct and verified, as short bullets. Discard wrong/uncertain steps."}]
    summary = S.generate(S.tok.apply_chat_template(cmsg, tokenize=False, add_generation_prompt=True),
                         max_new=200, temp=0.0)
    np2 = S.prompt(q + f"\n\nVerified progress so far:\n{summary}\n\nContinue ONLY from the verified "
                   "progress; do not repeat earlier mistakes.")
    return extract_answer(S.generate(np2, temp=args.temp))


ACTUATORS = {"bestofn": act_bestofn, "repath": act_repath, "compress": act_compress}


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
    ap.add_argument("--actuators", default="bestofn,repath,compress", help="comma list: bestofn,repath,compress")
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

    # ---- TEST: each TRIGGER (Halo/OURS) x each ACTUATOR (bestofn/repath/compress) ----
    acts = [a.strip() for a in args.actuators.split(",") if a.strip()]
    trigs = ["Halo", "OURS"]
    acc = {"baseline": 0, "self-cons": 0}
    pa = {(t, a): 0 for t in trigs for a in acts}              # pass@1 per (trigger, actuator)
    fixed = {(t, a): 0 for t in trigs for a in acts}; broke = {(t, a): 0 for t in trigs for a in acts}
    fire = {t: 0 for t in trigs}; sel = []
    n = 0
    for qi, (q, gold) in enumerate(test):
        pr = S.prompt(q); sol = S.generate(pr, temp=args.temp)
        pn, en, sp = S.step_signals(pr, sol)
        if len(pn) < 2:
            continue
        n += 1
        base_ans = extract_answer(sol); base_ok = correct(base_ans, gold); acc["baseline"] += int(base_ok)
        gz, ez = chain_scores(pn, en, mu_g, sd_g, mu_e, sd_e)
        sel.append((float(np.nanmax(np.maximum(gz, ez))), float(np.nanmax(ez)), int(base_ok)))
        fj = {"Halo": first_fire(ez, thr_halo), "OURS": first_fire(np.maximum(gz, ez), thr_ours)}
        for t in trigs:
            j = fj[t]
            if j is None or j >= len(sp):
                for a in acts:
                    pa[(t, a)] += int(base_ok)                 # no trigger -> keep baseline
                continue
            if t == "OURS":
                fire["OURS"] += 1
            elif t == "Halo":
                fire["Halo"] += 1
            prefix = sol[:sp[j][0]]
            for a in acts:
                ans = ACTUATORS[a](S, q, prefix, args); ok = correct(ans, gold); pa[(t, a)] += int(ok)
                if base_ok and not ok:
                    broke[(t, a)] += 1
                if (not base_ok) and ok:
                    fixed[(t, a)] += 1
        extra = [extract_answer(S.generate(pr, temp=args.temp)) for _ in range(args.bestof)]
        votes = Counter(_norm(a) for a in ([base_ans] + extra) if a is not None)
        acc["self-cons"] += int(correct(votes.most_common(1)[0][0], gold) if votes else False)
        if (qi + 1) % 20 == 0:
            print(f"  [{qi+1}] n={n} base {acc['baseline']/n:.3f}")

    nn = max(n, 1)
    print(f"\nmodel {args.model} | test {n} | actuators {acts} | bestof {args.bestof} | reroute_temp "
          f"{args.reroute_temp} | fpr {args.fpr}")
    print(f"\npass@1:  baseline {acc['baseline']/nn:.3f}   self-cons {acc['self-cons']/nn:.3f}")
    print(f"triggers fired: Halo {fire['Halo']}  OURS {fire['OURS']}  (OURS-extra = confident errors Halo misses)")
    print(f"\n{'trigger x actuator':24s} {'pass@1':>7s} {'fixed':>6s} {'broke':>6s} {'net':>5s}")
    for t in trigs:
        for a in acts:
            f, b = fixed[(t, a)], broke[(t, a)]
            print(f"  {t+' x '+a:24s} {pa[(t,a)]/nn:7.3f} {f:6d} {b:6d} {f-b:+5d}")
    # ---- SELECTIVE PREDICTION / risk-coverage (no reroute -- detection edge -> abstention utility) ----
    # answer the chains the detector ranks LEAST suspicious; abstain on the rest. higher accuracy on the
    # answered set = the detector that better ranks wrong (incl. confident errors) to the abstain pile.
    arr = np.array(sel, float)  # cols: ours_score, halo_score, base_ok
    ok = arr[:, 2]
    print(f"\nselective prediction (answered-set accuracy at coverage; abstain on most-suspicious):")
    print(f"  {'coverage':9s} {'OURS':>7s} {'Halo':>7s} {'entropy=Halo':>13s}")
    order_o = np.argsort(arr[:, 0]); order_h = np.argsort(arr[:, 1])   # ascending badness -> answer first
    for cov in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]:
        k = max(1, int(round(cov * len(ok))))
        ao = float(ok[order_o[:k]].mean()); ah = float(ok[order_h[:k]].mean())
        print(f"  {cov:<9.1f} {ao:7.3f} {ah:7.3f} {'(same as Halo)':>13s}")
    print("\nread: (A) FIX -- the trigger x actuator table. Compare actuators: bestofn (resampling, weakest), "
          "repath (different-method corrective push), compress (Halo-style context surgery). For each, does "
          "OURS-trigger net (fixed-broke) beat Halo-trigger? If a stronger actuator (repath/compress) converts "
          "OURS's extra confident-error triggers into fixes, the FIX story works. If ALL actuators give OURS~Halo "
          "(confident errors un-fixable, per Hidden Error Awareness), fall back to (B). (B) ABSTAIN -- the "
          "selective-prediction table: OURS answered-set accuracy > Halo at low coverage = the detection edge "
          "converts to utility by abstaining on confident errors, no fix needed. Report whichever the data "
          "supports; both are legitimate downstreams in the literature (self-correct/fix vs selective-prediction/abstain).")


if __name__ == "__main__":
    main()
