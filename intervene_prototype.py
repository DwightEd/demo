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
    """split into sentences -> list of (start_char, end_char). simple boundary on . ? ! newline."""
    spans = []; start = 0
    for m in re.finditer(r"[^.!?\n]*[.!?\n]+|\S[^.!?\n]*$", text):
        s, e = m.start(), m.end()
        if text[s:e].strip():
            spans.append((s, e))
    return spans or [(0, len(text))]


def extract_answer(text):
    """GSM8K-style: prefer \\boxed{}, then #### , then last number."""
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        nums = re.findall(r"-?\d[\d,]*\.?\d*", m.group(1))
        if nums:
            return nums[-1].replace(",", "")
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return m.group(1).replace(",", "")
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "") if nums else None


def correct(pred, gold):
    if pred is None:
        return False
    try:
        return abs(float(pred) - float(gold)) < 1e-4
    except ValueError:
        return str(pred).strip() == str(gold).strip()


# ----------------------------- generation + signals -----------------------------
class Solver:
    def __init__(self, model_name, layer, device="cuda"):
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                          device_map=device, output_hidden_states=True)
        self.model.eval(); self.layer = layer; self.device = device
        self.steer_vec = None; self._hook = None

    def prompt(self, q):
        msg = [{"role": "user", "content": q + "\nSolve step by step. End with \\boxed{answer}."}]
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
        for (cs, ce) in sent_spans(solution):
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


def trigger_idx(res, en, kind, k=1.0):
    """first sentence index that trips the trigger. geom = R drops below running mean - k*std;
    entropy = entropy rises above running mean + k*std. returns None if never."""
    if kind == "none" or len(res) < 3:
        return None
    s = -res if kind == "geom" else en                  # both: 'higher = more suspicious'
    for t in range(2, len(s)):
        hist = s[:t]; mu, sd = hist.mean(), hist.std() + 1e-6
        if s[t] > mu + k * sd:
            return t
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.7)
    ap.add_argument("--actuator", choices=["reconsider", "steer"], default="reconsider")
    ap.add_argument("--alpha", type=float, default=6.0, help="steering strength")
    args = ap.parse_args()

    # data: GSM8K test, fallback to a tiny hardcoded set so it always runs
    try:
        from datasets import load_dataset
        ds = load_dataset("gsm8k", "main", split=f"test[:{args.n}]")
        probs = [(d["question"], extract_answer(d["answer"])) for d in ds]
    except Exception as e:
        print(f"[datasets unavailable: {e}; using fallback problems]")
        probs = [("Natalia sold clips to 48 friends in April, and half as many in May. "
                  "How many clips did she sell altogether?", "72"),
                 ("Weng earns $12 an hour for babysitting. Yesterday she babysat 50 minutes. "
                  "How much did she earn?", "10")] * (args.n // 2 + 1)
        probs = probs[:args.n]

    S = Solver(args.model, args.layer)
    rows = {kind: {"n": 0, "base_ok": 0, "after_ok": 0, "fired": 0,
                   "conf_wrong": 0, "conf_fixed": 0} for kind in ["geom", "entropy"]}
    nbase = 0
    for qi, (q, gold) in enumerate(probs):
        prompt = S.prompt(q)
        sol = S.generate(prompt, temp=args.temp)
        base_ok = correct(extract_answer(sol), gold)
        nbase += int(base_ok)
        res, en, sp = S.signals(prompt, sol)
        if len(res) < 3:
            continue
        confident = (en.mean() < np.median(en)) if len(en) else False     # rough: low-entropy solution
        for kind in ["geom", "entropy"]:
            r = rows[kind]; r["n"] += 1; r["base_ok"] += int(base_ok)
            cw = (not base_ok) and confident
            r["conf_wrong"] += int(cw)
            t = trigger_idx(res, en, kind)
            if t is None:
                r["after_ok"] += int(base_ok)                              # no intervention
                continue
            r["fired"] += 1
            cut = sp[t][0]                                                 # truncate at trigger sentence
            stem = prompt + sol[:cut]
            if args.actuator == "steer":
                # re-anchor: steer toward the mean direction of the pre-trigger (healthy) sentences
                # (placeholder direction; refine with a learned correct-vs-error contrast vector)
                S.set_steer(np.ones(S.model.config.hidden_size), args.alpha)
                new = S.generate(stem, temp=args.temp); S.clear_steer()
            else:
                new = S.generate(stem + "\nWait, let me re-check this step carefully.\n", temp=max(args.temp, 0.9))
            after_ok = correct(extract_answer(sol[:cut] + new), gold)
            r["after_ok"] += int(after_ok)
            if cw and after_ok:
                r["conf_fixed"] += 1
        if (qi + 1) % 5 == 0:
            print(f"  [{qi+1}/{len(probs)}] baseline pass {nbase/(qi+1):.3f}")

    print(f"\nmodel {args.model} | layer {args.layer} | N {len(probs)} | actuator {args.actuator}")
    print(f"baseline pass-rate: {nbase/max(len(probs),1):.3f}")
    print(f"\n{'trigger':9s} {'fired':>6s} {'pass(base)':>11s} {'pass(after)':>12s} {'conf-wrong':>11s} {'conf-fixed':>11s}")
    for kind in ["geom", "entropy"]:
        r = rows[kind]; nn = max(r["n"], 1)
        print(f"{kind:9s} {r['fired']:>6d} {r['base_ok']/nn:>11.3f} {r['after_ok']/nn:>12.3f} "
              f"{r['conf_wrong']:>11d} {r['conf_fixed']:>11d}")
    print("\nread: pass(after) > pass(base) means the intervention helps. The headline is conf-fixed: "
          "how many CONFIDENT (low-entropy) wrong solutions each trigger repaired. If geom fixes more "
          "confident hallucinations than entropy at similar fire rate -> the geometric trigger covers "
          "the blind spot. This is a first prototype: run small --n, expect to tune trigger k / actuator.")


if __name__ == "__main__":
    main()
