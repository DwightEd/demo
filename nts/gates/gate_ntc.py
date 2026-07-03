# nts/gates/gate_ntc.py — the NTC go/no-go (re-analysis, no new inference):
# does the length-normalized saturation ratio phi = PR*tau/T deconfound length better than raw PR,
# and do error steps split into high-phi (diffuse) vs low-phi (collapse/regime-iii) clusters?
# KILL if phi does NOT reduce the length dependence of the signal.
import numpy as np
from .base import BaseGate, GateResult
from ..core.registry import GATES
from ..eval.metrics import auroc, bdir, bucket, spearman, bimodality_coeff
from ..geom.ntc import participation_ratio, autocorr_tau
from ..core.types import step_eval_mask


@GATES.register("gate_ntc")
class GateNTC(BaseGate):
    name = "gate_ntc"

    def run(self, table):
        PR, TAU, PHI, TLEN, Y = [], [], [], [], []
        for c in table.chains:
            if c.hidden_path is None or c.step_ranges is None:
                continue
            ok = step_eval_mask(c)  # unjudged post-first-error steps excluded
            H = np.load(c.hidden_path, mmap_mode="r")[:, c.hidden_col, :]
            for t, (a, b) in enumerate(c.step_ranges):
                if not ok[t]:
                    continue
                seg = np.asarray(H[int(a):int(b)]); n = len(seg)
                pr = participation_ratio(seg); tau = autocorr_tau(seg)
                PR.append(pr); TAU.append(tau); TLEN.append(n)
                PHI.append(pr * tau / n if (n > 0 and np.isfinite(pr) and np.isfinite(tau)) else np.nan)
                Y.append(int(c.y[t]))
        PR = np.array(PR); TAU = np.array(TAU); PHI = np.array(PHI); TLEN = np.array(TLEN, float); Y = np.array(Y)
        logT = np.log(TLEN + 1)
        r = GateResult(self.name)
        r.lines.append(f"gate_ntc | steps {len(Y)} err {int(Y.sum())}")
        for nm, s in [("PR(raw)", PR), ("phi=PR*tau/T", PHI)]:
            r.lines.append(f"  {nm:14s} AUROC {bdir(auroc(s, Y)):.3f} | bucket(T) {bucket(s, Y, TLEN):.3f} "
                           f"| |corr(.,logT)| {abs(spearman(s, logT)):.2f}")
        r.lines.append(f"  corr(tau, T) = {spearman(TAU, TLEN):.2f}  (high => tau tracks length, phi may not fully deconfound)")
        err = PHI[(Y == 1) & np.isfinite(PHI)]
        r.lines.append(f"  phi error-step bimodality coeff {bimodality_coeff(err):.3f}  (>0.555 ~ bimodal: diffuse vs collapse)")
        prc = abs(spearman(PR, logT)); phc = abs(spearman(PHI, logT))
        r.lines.append(f"  length-dependence: PR {prc:.2f} -> phi {phc:.2f}  (want phi << PR)")
        r.kill = not (np.isfinite(phc) and np.isfinite(prc) and phc < prc - 0.05)
        return r
