#!/usr/bin/env python3
"""CUMULATIVE geometric-collapse trigger: accumulate the within-step directional-concentration
anomaly ALONG the reasoning chain (Halo-style integration), fire on sustained collapse, and
intervene -- targeting the low-entropy blind spot that entropy / EDIS structurally miss.

The signal that WORKS (validated ~0.77 step-level) is the WITHIN-STEP directional concentration
`resultant` = ||exp-pooled UNIT token vectors|| at a mid layer. Error steps = the step's token
directions disperse -> resultant drops. This is a static per-step scalar; what was missing was a
DYNAMIC (online, cross-step) way to USE it. Here we accumulate it like Halo accumulates entropy
drift, but on the REPRESENTATION signal rather than the OUTPUT (entropy) signal:

  per step t:   x_t   = within-step resultant (geometry; LOW = collapse)
                r_t   = within-chain CAUSAL residual of -x_t   (HIGH = concentration dropped vs the
                                                                chain's own healthy history)
  accumulate:   Omega_t = max(0, lambda * Omega_{t-1} + (r_t - kappa))     (leaky CUSUM)
  trigger:      first step where Omega_t >= Psi        (Psi = conformal threshold, matched FPR)

Why CUMULATIVE (not single-step max): a one-step dip can be noise; a *sustained* directional
drift accumulates and crosses Psi, while isolated blips decay (leak lambda<1, drift subtract
kappa). This is the dynamic the pooled-per-step view discards and the EDIS-on-entropy view cannot
see on LOW-ENTROPY errors (entropy does not move; representation direction still disperses).

COMPETITORS, same accumulation machinery, matched held-out FPR:
  geom : accumulate residual of -resultant            (REPRESENTATION layer, ours)
  ent  : accumulate residual of step mean entropy     (OUTPUT layer, Halo-style static entropy)
  edis : accumulate per-step EDIS instability         (OUTPUT layer, entropy DYNAMICS)
BLIND SPOT = wrong & geom fires & NEITHER ent NOR edis fires, verified LOW absolute entropy.
On the blind spot: repath (high-T, different-method) repair -- the only actuator that escapes a
confident-wrong attractor (compress re-confirms it; steer translates rather than re-concentrates).

ENTRY POINTS
  --selftest                     : verify the accumulation/trigger/leak-free-conformal/blind-spot
                                   logic on synthetic per-step signals. numpy only, no GPU.
  --npz coh.npz                  : OFFLINE detection only -- compute the three cumulative triggers
                                   on stored features (validated resultant), report blind spot.
  --npz coh.npz --intervene ...  : add ONLINE repath repair on the blind-spot chains (needs model).

OFFLINE npz needs: stepcloud(resultant) + tok_U_D (per-token entropy) + step_token_ranges +
gold_error_step + layers_used + cloud_feature_names. (problem_ids/questions/answers optional, only
required for --intervene.)
"""

from __future__ import annotations
import argparse
import numpy as np


# ============================ metrics ============================
def auroc(s, y):
    m = np.isfinite(s); s, y = s[m], y[m]
    p, n = int((y == 1).sum()), int((y == 0).sum())
    if not p or not n:
        return float("nan")
    o = np.argsort(s, kind="mergesort"); r = np.empty(len(s)); sr = s[o]; i = 0
    while i < len(sr):
        j = i
        while j + 1 < len(sr) and sr[j + 1] == sr[i]:
            j += 1
        r[o[i:j + 1]] = 0.5 * (i + j) + 1; i = j + 1
    return (r[y == 1].sum() - p * (p + 1) / 2) / (p * n)


def bdir(a):
    return max(a, 1 - a) if np.isfinite(a) else a


# ============================ per-step signals ============================
def resultant(H):
    """ordered token states of ONE step -> within-step directional concentration of UNIT vectors
    (magnitude-free, exp-pooled). LOW = directions dispersed = collapse."""
    H = np.asarray(H, np.float64)
    nrm = np.linalg.norm(H, axis=1); ok = nrm > 1e-9
    if ok.sum() < 2:
        return np.nan
    u = H[ok] / nrm[ok, None]; n = u.shape[0]
    w = np.exp(np.arange(n) / max(n - 1, 1)); w /= w.sum()
    return float(np.linalg.norm((w[:, None] * u).sum(0)))


def step_edis(ent, w=4, tau_b=1.0, tau_r=1.0):
    """EDIS instability WITHIN a step's per-token entropy: burst (Eq.5) + peak-valley (Eq.6),
    times (1+var) (Eq.7). The step-level entropy-dynamics competitor."""
    e = np.asarray(ent, float); e = e[np.isfinite(e)]; T = len(e)
    if T < 3:
        return 0.0
    ww = min(w, max(2, T // 2))
    burst = int(sum(1 for t in range(T - ww) if e[t + ww] - e[t] > tau_b)) if T > ww else 0
    pv = 0; rmin = e[0]
    for t in range(1, T):
        if e[t] - rmin > tau_r:
            pv += 1
        rmin = min(rmin, e[t])
    return 0.5 * (burst + pv) * (1.0 + float(e.var()))


# ============================ cumulative trigger (the dynamic) ============================
def causal_resid(seq, sd_floor, clip=5.0, eps=1e-6):
    """within-chain CAUSAL residual: r[t] = (seq[t]-mean(seq[:t]))/max(std(seq[:t]),sd_floor).
    NaN for t<2 (treated as 0 contribution in the accumulator)."""
    s = np.asarray(seq, float); T = len(s); r = np.full(T, np.nan)
    for t in range(2, T):
        h = s[:t]; h = h[np.isfinite(h)]
        if len(h) >= 2 and np.isfinite(s[t]):
            r[t] = np.clip((s[t] - h.mean()) / (max(h.std(), sd_floor) + eps), -clip, clip)
    return r


def accumulate(resid, lam, kappa):
    """leaky CUSUM of a residual sequence: Omega_t = max(0, lam*Omega_{t-1} + (r_t - kappa)).
    Sustained positive drift accumulates; isolated blips decay. Returns the Omega trajectory."""
    r = np.asarray(resid, float)
    omega = np.zeros(len(r)); acc = 0.0
    for t in range(len(r)):
        rt = r[t] if np.isfinite(r[t]) else 0.0
        acc = max(0.0, lam * acc + (rt - kappa))
        omega[t] = acc
    return omega


def first_at(omega, psi):
    idx = np.where(np.asarray(omega, float) >= psi)[0]
    return int(idx[0]) if len(idx) else None


# ============================ leak-free analysis ============================
SIG = ("geom", "ent", "edis")


def analyze(chains, *, fpr=0.2, calib_frac=0.5, lam=0.8, kappa=0.5, seed=0):
    """chains: dicts with per-step arrays res (resultant), step_ent (mean entropy),
    step_edis (within-step EDIS), abs_ent (chain mean token entropy), is_correct.
    Builds the three CUMULATIVE triggers, splits correct chains CALIB/EVAL, sets the
    accumulation threshold Psi per signal on CALIB at matched FPR, measures FPR on EVAL,
    forms the blind spot. Annotates chains with omega_*/psi_*/fire_*/fg. Model-free."""
    rng = np.random.default_rng(seed)
    cor = [c for c in chains if c["is_correct"]]
    wro = [c for c in chains if not c["is_correct"]]
    if len(cor) < 4:
        raise SystemExit("need >=4 correct chains to calibrate")
    perm = rng.permutation(len(cor)); k = max(2, int(len(cor) * calib_frac))
    calib = [cor[perm[j]] for j in range(k)]
    ev = [cor[perm[j]] for j in range(k, len(cor))]

    def floor(vals):
        v = [x for x in vals if np.isfinite(x)]
        return 0.5 * float(np.median(v)) if v else 1.0
    sf = {
        "geom": floor([np.nanstd(c["res"]) for c in calib if np.isfinite(c["res"]).sum() >= 2]),
        "ent":  floor([np.nanstd(c["step_ent"]) for c in calib if np.isfinite(c["step_ent"]).sum() >= 2]),
        "edis": floor([np.nanstd(c["step_edis"]) for c in calib if np.isfinite(c["step_edis"]).sum() >= 2]),
    }

    def raw(c, s):
        if s == "geom":
            return [-x for x in c["res"]]          # collapse -> positive residual
        if s == "ent":
            return c["step_ent"]
        return c["step_edis"]

    for c in chains:
        for s in SIG:
            r = causal_resid(raw(c, s), sf[s])
            om = accumulate(r, lam, kappa)
            c["omega_" + s] = om
            c["peak_" + s] = float(np.max(om)) if len(om) else 0.0

    # MATCHED-FPR operating point: threshold each signal on ALL correct chains at the same quantile,
    # so the realized false-positive rate on correct chains is IDENTICAL across geom/ent/edis. Wrong
    # chains never touch the threshold, so the catch / blind-spot comparison stays leak-free while the
    # comparison is fair (the old calib-only quantile let eval FPR drift, so EDIS could 'catch more'
    # merely by firing more). calib/eval split is kept only for the out-of-sample FPR check below.
    psi = {}
    q = 1.0 - fpr
    for s in SIG:
        psi[s] = float(np.quantile([c["peak_" + s] for c in cor], q))

    for c in chains:
        for s in SIG:
            fa = first_at(c["omega_" + s], psi[s])
            c["fire_" + s] = fa is not None
            if s == "geom":
                c["fg"] = fa                       # geometry trigger step (for repair)

    fpr_e = {s: float(np.mean([c["fire_" + s] for c in cor])) for s in SIG}     # matched by construction
    fpr_oos = {}                                                                # out-of-sample generalization
    for s in SIG:
        th = float(np.quantile([c["peak_" + s] for c in calib], q)) if calib else psi[s]
        fpr_oos[s] = float(np.mean([first_at(c["omega_" + s], th) is not None for c in ev])) if ev else float("nan")

    ab = lambda c: c.get("abs_ent", np.nan)
    blind = [c for c in wro if c["fire_geom"] and not c["fire_ent"] and not c["fire_edis"]]
    ecaught = [c for c in wro if c["fire_ent"] or c["fire_edis"]]
    return dict(
        n=len(chains), n_correct=len(cor), n_wrong=len(wro), n_calib=len(calib), n_eval=len(ev),
        psi=psi, sf=sf, fpr=fpr_e, fpr_oos=fpr_oos,
        catch={s: int(sum(c["fire_" + s] for c in wro)) for s in SIG},
        geom_not_ent=int(sum(c["fire_geom"] and not c["fire_ent"] for c in wro)),
        geom_not_edis=int(sum(c["fire_geom"] and not c["fire_edis"] for c in wro)),
        n_blind=len(blind), blind=blind,
        ent_blind=float(np.nanmean([ab(c) for c in blind])) if blind else float("nan"),
        ent_ecaught=float(np.nanmean([ab(c) for c in ecaught])) if ecaught else float("nan"),
        ent_allwrong=float(np.nanmean([ab(c) for c in wro])) if wro else float("nan"),
    )


def report(r):
    print(f"\nchains {r['n']}  correct {r['n_correct']} (calib {r['n_calib']}/eval {r['n_eval']})  "
          f"wrong {r['n_wrong']}")
    print(f"Psi (matched-FPR accumulation threshold): " + "  ".join(f"{s}={r['psi'][s]:.2f}" for s in SIG))
    print(f"FPR matched on correct chains (identical operating point) | catches on wrong | oos-FPR:")
    for s in SIG:
        oos = r.get("fpr_oos", {}).get(s, float("nan"))
        print(f"  {s:5s} FPR {r['fpr'][s]:.3f}   catches {r['catch'][s]:>3d}/{r['n_wrong']} wrong   "
              f"(held-out FPR {oos:.3f})")
    print(f"\nblind spot (wrong & geom fires & NEITHER ent NOR edis fires): {r['n_blind']}")
    print(f"  geom\\ent {r['geom_not_ent']}   geom\\edis {r['geom_not_edis']}   "
          f"geom\\both {r['n_blind']}  <- representation collapse caught it, both entropy signals missed")
    print(f"\nconfident-error check (mean absolute token entropy):")
    print(f"  blind spot     {r['ent_blind']:.3f}")
    print(f"  ent/edis-caught{r['ent_ecaught']:8.3f}   (blind spot should be LOWER = low-entropy confident)")
    print(f"  all wrong      {r['ent_allwrong']:.3f}")


# ============================ offline feature loading ============================
def load_offline(npz, layer):
    z = np.load(npz, allow_pickle=True)
    cn = [str(x) for x in z["cloud_feature_names"]]
    lyu = [int(x) for x in z["layers_used"]]; li = lyu.index(layer)
    fi = cn.index("resultant")
    SC, SR = z["stepcloud"], z["step_token_ranges"]; ges = z["gold_error_step"].astype(int)
    UD = z["tok_U_D"]
    pid = z["problem_ids"].astype(int) if "problem_ids" in z.files else np.arange(len(SC))
    chains = []
    for i in range(len(SC)):
        sc = np.asarray(SC[i], float); rng = np.asarray(SR[i], int); k = int(ges[i])
        a0 = int(rng[0, 0]); T = rng.shape[0]; ud = np.asarray(UD[i], float)
        res, sent, sedis, toks = [], [], [], []
        for j in range(T):
            lo = max(0, int(rng[j, 0]) - a0); hi = min(len(ud), int(rng[j, 1]) - a0 + 1)
            if hi - lo < 2:
                res.append(np.nan); sent.append(np.nan); sedis.append(np.nan); continue
            uds = ud[lo:hi]
            res.append(float(sc[j, li, fi]))
            sent.append(float(np.nanmean(uds)))
            sedis.append(step_edis(uds))
            toks.append(uds)
        # error label per step: only the FIRST error step is positive; steps after it excluded
        # (handled in analyze via correctness); here we keep full chain + gold_error_step
        all_tok = np.concatenate(toks) if toks else np.array([])
        chains.append(dict(res=np.array(res), step_ent=np.array(sent), step_edis=np.array(sedis),
                           abs_ent=float(np.nanmean(all_tok)) if len(all_tok) else np.nan,
                           is_correct=(k < 0), gold_error_step=k, pid=int(pid[i])))
    return chains


# ============================ online repair (needs model) ============================
class Repairer:
    def __init__(self, model_name, device="cuda"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        kw = dict(torch_dtype=torch.float16, device_map=device)
        try:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, attn_implementation="sdpa", **kw)
        except Exception:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, **kw)
        self.model.eval(); self.device = device

    def prompt(self, q):
        msg = [{"role": "user", "content": q + "\nSolve step by step. Begin EACH step with "
                "'Step N:'. End with the final answer in \\boxed{}."}]
        return self.tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)

    def gen(self, prompt, max_new=512, temp=1.0):
        with self.torch.no_grad():
            ids = self.tok(prompt, return_tensors="pt").to(self.device)
            out = self.model.generate(**ids, max_new_tokens=max_new, do_sample=temp > 0,
                                       temperature=max(temp, 1e-5), top_p=0.95,
                                       pad_token_id=self.tok.eos_token_id)
        return self.tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)

    def repath(self, q, prefix, temp):
        """truncate at the trigger step, HIGH-temp regenerate with an explicit different-method
        instruction -- the only actuator that can escape a confident-wrong attractor."""
        stem = self.prompt(q) + prefix + ("\n\nThe above step is likely wrong. Discard it and "
                                          "solve this part again using a DIFFERENT method.\n")
        return self.gen(stem, temp=temp)


# ============================ self-test (no model) ============================
def _synth(seed=0):
    rng = np.random.default_rng(seed); chains = []

    def mk(is_correct, kind=None):
        T = int(rng.integers(7, 16))
        res = list(rng.normal(0.60, 0.015, T))
        sent = list(np.clip(rng.normal(0.25, 0.04, T), 0.01, None))
        sedis = list(np.abs(rng.normal(0.2, 0.2, T)))
        abs_ent = 0.25
        if not is_correct:
            k = int(rng.integers(2, T - 1))
            # SUSTAINED collapse: dip persists for a few steps (so the accumulator crosses Psi,
            # which an isolated single-step blip would not). This is the dynamic we exploit.
            for d in range(0, min(3, T - k)):
                res[k + d] -= 0.07
            if kind == "loud":                              # entropy ALSO unstable -> ent & edis fire
                for d in range(0, min(3, T - k)):
                    sent[k + d] += 0.7; sedis[k + d] += 2.5
                abs_ent = 0.45
            elif kind == "quiet":                           # undo collapse -> nothing fires
                for d in range(0, min(3, T - k)):
                    res[k + d] += 0.07
        return dict(res=np.array(res), step_ent=np.array(sent), step_edis=np.array(sedis),
                    abs_ent=abs_ent, is_correct=is_correct, gold_error_step=(-1 if is_correct else k))

    for _ in range(140):
        chains.append(mk(True))
    for _ in range(22):
        chains.append(mk(False, "loud"))
    for _ in range(24):
        chains.append(mk(False, "blind"))
    for _ in range(10):
        chains.append(mk(False, "quiet"))
    return chains


def selftest(lam, kappa, fpr):
    print(f"=== SELFTEST (synthetic; verifies CUMULATIVE trigger + leak-free conformal + blind spot) ===")
    print(f"accumulator: lambda={lam} kappa={kappa} | target FPR={fpr}")
    chains = _synth(0)
    r = analyze(chains, fpr=fpr, lam=lam, kappa=kappa, seed=1)
    report(r)
    print("\nby construction: ~24 blind-spot (geom-only, low-entropy) chains; ~22 'loud' caught by "
          "ent/edis; held-out FPR ~ target; blind-spot entropy (~0.25) << ent/edis-caught (~0.45).")
    ok = True
    if r["n_blind"] < 12:
        print(f"  [FAIL] blind spot too small ({r['n_blind']})"); ok = False
    if not (r["ent_blind"] < r["ent_ecaught"]):
        print(f"  [FAIL] blind spot not lower-entropy"); ok = False
    if not (0.05 <= r["fpr"]["geom"] <= 0.40):
        print(f"  [FAIL] geom held-out FPR off target ({r['fpr']['geom']:.3f})"); ok = False
    print("\nSELFTEST", "PASSED" if ok else "FAILED")
    return ok


# ============================ main ============================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default=None, help="offline features (stepcloud resultant + tok_U_D + ...)")
    ap.add_argument("--layer", type=int, default=14)
    ap.add_argument("--fpr", type=float, default=0.2, help="target held-out fire rate on correct chains")
    ap.add_argument("--calib_frac", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=0.8, help="accumulator leak (1=pure CUSUM, <1 forgets blips)")
    ap.add_argument("--kappa", type=float, default=0.5, help="accumulator drift subtract (slack per step)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selftest", action="store_true")
    # online repair (optional; needs model + per-chain question/solution/gold via a jsonl sidecar)
    ap.add_argument("--intervene", action="store_true", help="run repath repair on blind-spot chains")
    ap.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    ap.add_argument("--sidecar_jsonl", default=None,
                    help="lines {pid, question, solution, answer} matching npz problem_ids, for repair")
    ap.add_argument("--repath_temp", type=float, default=1.1)
    args = ap.parse_args()

    if args.selftest:
        raise SystemExit(0 if selftest(args.lam, args.kappa, args.fpr) else 1)
    if args.npz is None:
        raise SystemExit("pass --npz FEATURES.npz  (or --selftest)")

    chains = load_offline(args.npz, args.layer)
    nbase = float(np.mean([c["is_correct"] for c in chains]))
    print(f"file: {args.npz} | layer {args.layer} | chains {len(chains)} | correct-frac {nbase:.3f}")
    r = analyze(chains, fpr=args.fpr, calib_frac=args.calib_frac, lam=args.lam, kappa=args.kappa, seed=args.seed)
    report(r)
    print("\nread: HEADLINE = blind-spot SIZE (cumulative geometry trigger fires; cumulative entropy "
          "AND EDIS triggers both miss) at matched held-out FPR, AND blind-spot entropy < ent/edis-caught "
          "entropy (genuinely confident errors). The accumulation (leaky CUSUM) is the DYNAMIC: sustained "
          "directional collapse crosses Psi, isolated blips decay -- this is how the per-step geometric "
          "signal becomes an online trigger, and why it sees low-entropy errors entropy dynamics cannot.")

    if not args.intervene:
        print("\n(add --intervene --sidecar_jsonl FILE to run repath repair on the blind-spot chains)")
        return
    if not args.sidecar_jsonl:
        raise SystemExit("--intervene needs --sidecar_jsonl with {pid, question, solution, answer}")

    # ---- online repath repair on the blind-spot chains ----
    import json
    side = {}
    for line in open(args.sidecar_jsonl, encoding="utf-8"):
        d = json.loads(line); side[int(d["pid"])] = d
    R = Repairer(args.model)

    def boxed(t):
        i = t.rfind("\\boxed");
        if i < 0:
            return None
        j = t.find("{", i); depth = 0
        for k in range(j, len(t)):
            if t[k] == "{": depth += 1
            elif t[k] == "}":
                depth -= 1
                if depth == 0: return t[j + 1:k]
        return None

    def eq(a, b):
        if a is None or b is None: return False
        na, nb = a.replace(" ", "").rstrip("."), str(b).replace(" ", "").rstrip(".")
        if na == nb: return True
        try: return abs(float(na) - float(nb)) < 1e-4
        except ValueError: return False

    blind = [c for c in r["blind"] if c["pid"] in side and c["fg"] is not None]
    print(f"\nrepath repair on {len(blind)} blind-spot chains (need sidecar text + a 'Step N:' solution):")
    fixed = 0
    for c in blind:
        d = side[c["pid"]]; sol = d["solution"]; gold = d["answer"]; q = d["question"]
        # locate the trigger step's char offset by 'Step N:' markers in the stored solution
        marks = [m.start() for m in __import__("re").finditer(r"(?im)^\s*step\s*\d+\s*[:.\)]", sol)]
        if c["fg"] >= len(marks):
            continue
        cut = marks[c["fg"]]
        regen = R.repath(q, sol[:cut], args.repath_temp)
        fixed += int(eq(boxed(sol[:cut] + "\n" + regen), gold))
    print(f"  repath fixed {fixed}/{len(blind)}  ({(fixed / max(len(blind),1)):.2f})")
    print("  (compare to compress/steer separately; report bootstrap CI; intrusion caveat applies)")


if __name__ == "__main__":
    main()