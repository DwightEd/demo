# nts/gates/gate2_nts_vs_rema.py — ★ NTS residual-normal vs REMA vs raw-kappa, with confound controls
import numpy as np
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES
from ..eval.metrics import auroc, bdir, bucket
from ..eval.confound import residualize, oof_logit, cluster_boot_increment


@GATES.register("gate2_nts_vs_rema")
class Gate2(BaseGate):
    name = "gate2_nts_vs_rema"

    def run(self, table):
        cfg = self.cfg; f = table.flat(); logn = np.log1p(f.length)
        nts = crossfit_signal("nts", table, cfg, folds=cfg.folds)
        rema = crossfit_signal("rema", table, cfg, folds=cfg.folds)
        kap = crossfit_signal("kappa", table, cfg, folds=cfg.folds)  # already -kappa
        conf = np.column_stack([logn, f.speed, f.repetition])
        nts_resid = residualize(nts, conf, f.chain_correct, f.groups, cfg.folds)
        r = GateResult(self.name)

        def block(mask, title):
            y, g = f.y[mask], f.groups[mask]
            r.lines.append(f"  [{title}] steps {int(mask.sum())} err {int(y.sum())}")
            for nm, sc in [("raw-kappa", kap), ("REMA", rema), ("NTS raw", nts), ("NTS resid", nts_resid)]:
                r.lines.append(f"    {nm:12s} AUROC {bdir(auroc(sc[mask], y)):.3f}  bucket {bucket(sc[mask], y, f.length[mask]):.3f}")
            cols = [rema[mask], kap[mask], logn[mask], f.speed[mask], f.repetition[mask]]
            base = oof_logit(np.column_stack(cols), y, g)
            full = oof_logit(np.column_stack(cols + [nts_resid[mask]]), y, g)
            mean, lo, hi, sig = cluster_boot_increment(full, base, y, g)
            r.lines.append(f"    NTS over [REMA+kappa+conf]: +{mean:.3f} [{lo:+.3f},{hi:+.3f}] {'SIG' if sig else 'ns'}")
            return sig

        # Common eval support: (a) unjudged post-first-error steps excluded; (b) all
        # compared signals must be finite on the same steps — NTS/NTS-resid are NaN at
        # t=0 by construction (displacement needs an anchor) while REMA/kappa are not,
        # so without this the head-to-head AUROCs are computed on different populations
        # (and gold_error_step==0 errors silently vanish from NTS only).
        support = np.isfinite(nts) & np.isfinite(rema)
        if not np.all(np.isnan(f.kappa)):
            support &= np.isfinite(kap)
        ok = f.eval_ok & support
        r.lines.append(f"  eval mask: {int(ok.sum())}/{len(f.y)} steps kept "
                       f"(post-error + t=0/common-support masked) err {int(f.y[ok].sum())}/{int(f.y.sum())}")
        full_sig = block(ok, "ALL")
        if np.all(np.isnan(f.kappa)):
            r.lines.append("  [coherent-but-wrong] skipped (kappa/resultant unavailable in this npz)")
            cbw_sig = False
        else:
            kmed = np.median(f.kappa[(f.y == 0) & ok]); cbw = (f.kappa >= kmed) & ok
            cbw_sig = block(cbw, "coherent-but-wrong (kappa>=median)")
        r.kill = not (full_sig or cbw_sig)
        return r
