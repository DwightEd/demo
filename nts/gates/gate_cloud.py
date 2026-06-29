# nts/gates/gate_cloud.py — step-free cloud off-correct-subspace energy, CHAIN-level, within-problem.
# Head-to-head of cloud / alpha (Spectral-Geometry metric) / step-NTS / REMA / kappa, vs is_correct,
# AUROC within each problem (matching the recorded within-problem results). On cross-problem data
# (1 chain/problem) within-problem is undefined -> reports pooled (difficulty-confounded) + a clear note.
import numpy as np
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES
from ..eval.metrics import auroc, bdir

REF = {"probe(within)": 0.71, "SPE(within)": 0.68, "scalar(within)": 0.55}  # results_summary.md


def _chain_level(table, step_scores):
    vals, y, g = [], [], []; off = 0
    for c in table.chains:
        T = len(c.y); s = step_scores[off:off + T]; off += T
        vals.append(np.nanmax(s) if np.isfinite(s).any() else np.nan)
        y.append(0 if c.correct else 1); g.append(c.problem_id)
    return np.array(vals), np.array(y), np.array(g)


def _within_problem_auroc(s, y, g):
    num = den = 0.0; nvalid = 0
    for q in np.unique(g):
        m = g == q
        if 0 < int(y[m].sum()) < int(m.sum()):
            nvalid += 1
            a = bdir(auroc(s[m], y[m])); ne = int(y[m].sum()); ng = int((y[m] == 0).sum())
            if np.isfinite(a):
                num += a * ne * ng; den += ne * ng
    return (num / den if den else float("nan")), nvalid


@GATES.register("gate_cloud")
class GateCloud(BaseGate):
    name = "gate_cloud"

    def run(self, table):
        r = GateResult(self.name)
        r.lines.append(f"gate_cloud | chains={len(table.chains)} | chain-level vs is_correct")
        sigs = [("nts_cloud", "nts_cloud"), ("alpha", "alpha"), ("nts_step(max)", "nts"),
                ("rema(max)", "rema"), ("kappa(max)", "kappa")]
        cloud_metric = float("nan"); cloud_nvalid = 0
        for label, name in sigs:
            try:
                step_s = crossfit_signal(name, table, self.cfg, folds=self.cfg.folds)
            except Exception as e:
                r.lines.append(f"  {label:14s} FAILED: {e}"); continue
            s, y, g = _chain_level(table, step_s)
            aw, nv = _within_problem_auroc(s, y, g); ap = bdir(auroc(s, y))
            r.lines.append(f"  {label:14s} within {aw:.3f} (n_prob={nv}) | pooled {ap:.3f}")
            if name == "nts_cloud":
                cloud_nvalid = nv; cloud_metric = aw if nv >= 10 else ap
        r.lines.append("  recorded baselines: " + ", ".join(f"{k}={v}" for k, v in REF.items()))
        if cloud_nvalid < 10:
            r.lines.append("  NOTE: within-problem N/A (1 chain/problem = cross-problem data). pooled is "
                           "difficulty-confounded; run on self-sampled (gsm8k_v2_5shot) for the within verdict.")
        r.kill = not (np.isfinite(cloud_metric) and cloud_metric > 0.60)
        return r
