# nts/gates/gate_localize.py — ProcessBench-style FIRST-ERROR LOCALIZATION verdict (P5 落地).
# Motivation: chain-level AUROC is capped (~0.83 in this project, FINDINGS.md) and crowded;
# the proposal's P5 predicts the alarm point aligns with the gold first-error step. This gate
# quantifies that directly: argmax of the OOF step score = predicted first-error step, plus a
# ProcessBench-style F1 where "no error" is predicted via a correct-chain calibrated threshold
# (per-fold quantile, i.e. the comparison population is CORRECT CHAINS, not the chain's own
# history — with T≈4 steps a self-history z-score has no estimable baseline).
# Baselines: REMA, raw-kappa, step token length, analytic random over the same support.
# KILL if NTS exact-hit is not significantly better than the strongest baseline
# (problem-cluster bootstrap CI) or does not beat analytic random.
import numpy as np
from sklearn.model_selection import GroupKFold
from .base import BaseGate, GateResult, crossfit_signal
from ..core.registry import GATES


def _split_by_chain(table, flat_scores):
    out = []; off = 0
    for c in table.chains:
        T = len(c.y); out.append(np.asarray(flat_scores[off:off + T], float)); off += T
    return out


def _predict(scores):
    """(argmax step over finite scores, max score); (-1, nan) if nothing finite."""
    m = np.isfinite(scores)
    if not m.any():
        return -1, float("nan")
    return int(np.argmax(np.where(m, scores, -np.inf))), float(scores[m].max())


def _stats(table, per_chain, e, q=0.9, folds=5):
    """Localization + ProcessBench-style F1 for one signal.

    e: global mask over chains = labeled error chains (gold step in range); chains whose
    signal has no finite step score count as localization MISSES (keeps signals aligned
    for the bootstrap comparison, instead of silently shrinking their eval set)."""
    pid = np.array([c.problem_id for c in table.chains])
    ges = np.array([int(np.argmax(c.y == 1)) if (c.y == 1).any() else -1 for c in table.chains])
    isc = np.array([c.correct for c in table.chains])
    pred = np.full(len(pid), -1); mx = np.full(len(pid), np.nan)
    for i, s in enumerate(per_chain):
        pred[i], mx[i] = _predict(s)
    exact = (pred[e] == ges[e]).astype(float)
    tol1 = (np.abs(pred[e] - ges[e]) <= 1).astype(float)
    early = float(np.mean(pred[e] < ges[e])) if e.any() else float("nan")
    late = float(np.mean(pred[e] > ges[e])) if e.any() else float("nan")
    # per-fold no-error threshold from train-fold CORRECT chains (scores are already OOF)
    haserr = np.zeros(len(pid), bool)
    for tr, te in GroupKFold(folds).split(np.zeros(len(pid)), np.zeros(len(pid)), pid):
        cal = [i for i in tr if isc[i] and np.isfinite(mx[i])]
        if len(cal) < 10:
            continue
        thr = float(np.quantile(mx[cal], q))
        haserr[te] = np.where(np.isfinite(mx[te]), mx[te] > thr, False)
    err_acc = float(np.mean(haserr[e] & (pred[e] == ges[e]))) if e.any() else float("nan")
    cor_acc = float(np.mean(~haserr[isc])) if isc.any() else float("nan")
    f1 = (2 * err_acc * cor_acc / (err_acc + cor_acc)
          if np.isfinite(err_acc) and np.isfinite(cor_acc) and (err_acc + cor_acc) > 0 else 0.0)
    return dict(exact=exact, tol1=tol1, early=early, late=late,
                err_acc=err_acc, cor_acc=cor_acc, f1=f1)


def _boot_diff(a_hits, b_hits, groups, nboot=2000, seed=0):
    """problem-cluster bootstrap CI for mean(a_hits) - mean(b_hits)."""
    rng = np.random.default_rng(seed); gid = np.unique(groups)
    by = {c: np.where(groups == c)[0] for c in gid}; d = []
    for _ in range(nboot):
        take = np.concatenate([by[c] for c in rng.choice(gid, len(gid), replace=True)])
        d.append(np.mean(a_hits[take]) - np.mean(b_hits[take]))
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(np.mean(d)), float(lo), float(hi)


@GATES.register("gate_localize")
class GateLocalize(BaseGate):
    name = "gate_localize"

    def run(self, table):
        cfg = self.cfg; r = GateResult(self.name)
        ges = np.array([int(np.argmax(c.y == 1)) if (c.y == 1).any() else -1 for c in table.chains])
        isc = np.array([c.correct for c in table.chains])
        pid = np.array([c.problem_id for c in table.chains])
        e = (~isc) & (ges >= 0)   # labeled error chains; ges out of range => excluded, stated below
        n_unlab = int(((~isc) & (ges < 0)).sum())

        sigs = {}
        for name in ("nts", "rema", "kappa"):
            sigs[name] = _split_by_chain(table, crossfit_signal(name, table, cfg, folds=cfg.folds))
        # non-model baseline: step token length; t=0 set NaN to match the nts/rema support
        len_pc = []
        for c in table.chains:
            s = np.asarray(c.length, float).copy()
            if len(s):
                s[0] = np.nan
            len_pc.append(s)
        sigs["length"] = len_pc

        stats = {k: _stats(table, v, e, folds=cfg.folds) for k, v in sigs.items()}
        nvalid = np.array([max(len(c.y) - 1, 1) for c in table.chains])   # nts/rema support (t>=1)
        rand_exact = float(np.mean(1.0 / nvalid[e])) if e.any() else float("nan")

        r.lines.append(f"gate_localize | error chains w/ in-range gold step: {int(e.sum())} "
                       f"(dropped {n_unlab} error chains w/o stored gold step) | correct: {int(isc.sum())}")
        r.lines.append(f"  analytic random exact-hit on t>=1 support: {rand_exact:.3f}")
        r.lines.append("  NOTE kappa's support includes t=0 (defined at every step); nts/rema start at t=1.")
        for k, st in stats.items():
            r.lines.append(f"  {k:8s} exact {np.mean(st['exact']):.3f} | +-1 {np.mean(st['tol1']):.3f} "
                           f"| early {st['early']:.2f} late {st['late']:.2f} "
                           f"| PB-F1 {st['f1']:.3f} (err {st['err_acc']:.3f} / corr {st['cor_acc']:.3f})")

        nts_hit = stats["nts"]["exact"]
        best_name = max(("rema", "kappa", "length"), key=lambda k: float(np.mean(stats[k]["exact"])))
        mean, lo, hi = _boot_diff(nts_hit, stats[best_name]["exact"], pid[e])
        beats_rand = bool(np.mean(nts_hit) > rand_exact)
        r.lines.append(f"  NTS exact - {best_name}(best baseline) exact: {mean:+.3f} [{lo:+.3f},{hi:+.3f}] "
                       f"{'SIG' if lo > 0 else 'ns'} | beats analytic random: {beats_rand}")
        r.kill = not (beats_rand and lo > 0)
        return r
