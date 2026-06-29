# nts/gates/gate_cloud.py — step-free cloud off-correct-subspace energy, CHAIN-level, within-problem.
# This is the chain-level home for the cloud NTS signal, evaluated the way the existing
# within-problem results are (vs is_correct, AUROC within each problem). It also chain-aggregates
# the step-level signals (max over steps) so cloud + step NTS + REMA + kappa are compared together,
# and prints the recorded baselines from results_summary.md as the bar.
import numpy as np
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES
from ..eval.metrics import auroc, bdir

REF = {"probe(within)": 0.71, "SPE(within)": 0.68, "scalar(within)": 0.55}  # results_summary.md


def _chain_level(table, step_scores):
    """Collapse per-step scores to one value per chain (max); label = chain is wrong."""
    vals, y, g = [], [], []; off = 0
    for c in table.chains:
        T = len(c.y); s = step_scores[off:off + T]; off += T
        vals.append(np.nanmax(s) if np.isfinite(s).any() else np.nan)
        y.append(0 if c.correct else 1); g.append(c.problem_id)
    return np.array(vals), np.array(y), np.array(g)


def _within_problem_auroc(s, y, g):
    num = den = 0.0
    for q in np.unique(g):
        m = g == q
        if 0 < int(y[m].sum()) < int(m.sum()):
            a = bdir(auroc(s[m], y[m])); ne = int(y[m].sum()); ng = int((y[m] == 0).sum())
            if np.isfinite(a):
                num += a * ne * ng; den += ne * ng
    return num / den if den else float("nan")


@GATES.register("gate_cloud")
class GateCloud(BaseGate):
    name = "gate_cloud"

    def run(self, table):
        r = GateResult(self.name)
        r.lines.append(f"gate_cloud | chains={len(table.chains)} | within-problem AUROC vs is_correct")
        sigs = [("nts_cloud", "nts_cloud"), ("nts_step(max)", "nts"), ("rema(max)", "rema"), ("kappa(max)", "kappa")]
        cloud_within = float("nan")
        for label, name in sigs:
            try:
                step_s = crossfit_signal(name, table, self.cfg, folds=self.cfg.folds)
            except Exception as e:
                r.lines.append(f"  {label:16s} FAILED: {e}"); continue
            s, y, g = _chain_level(table, step_s)
            aw = _within_problem_auroc(s, y, g); ap = bdir(auroc(s, y))
            r.lines.append(f"  {label:16s} within {aw:.3f} | pooled {ap:.3f}")
            if name == "nts_cloud":
                cloud_within = aw
        r.lines.append("  recorded baselines: " + ", ".join(f"{k}={v}" for k, v in REF.items()))
        r.kill = not (np.isfinite(cloud_within) and cloud_within > 0.60)
        return r
