# nts/gates/gate0_mahal.py — honest Mahalanobis floor (raw / bucket / length-residualized)
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES
from ..eval.metrics import auroc, bdir, bucket
from ..eval.confound import residualize


@GATES.register("gate0_mahal")
class Gate0(BaseGate):
    name = "gate0_mahal"

    def run(self, table):
        f = table.flat()
        mah = crossfit_signal("mahalanobis", table, self.cfg, self.params, self.cfg.folds)
        raw = bdir(auroc(mah, f.y)); bkt = bucket(mah, f.y, f.length)
        rez = bdir(auroc(residualize(mah, f.length[:, None], f.chain_correct, f.groups, self.cfg.folds), f.y))
        honest = min(bkt, rez)
        r = GateResult(self.name)
        r.lines += [f"gate0 mahal | steps {len(f.y)} err {int(f.y.sum())}",
                    f"  raw {raw:.3f} | bucket(len) {bkt:.3f} | len-resid {rez:.3f} | HONEST {honest:.3f}"]
        r.kill = bool(honest <= 0.60)
        return r
