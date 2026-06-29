# nts/gates/gate1_estimability.py — ID curve, tangent stability vs null, normal SNR
import numpy as np
from sklearn.model_selection import GroupKFold
from .base import BaseGate, GateResult
from ..core.registry import GATES
from ..data.loader import load_layer_matrix
from ..geom.reducer import fit_reducer
from ..geom.bank import Bank
from ..geom.tangent import local_tangent, chain_energies
from ..geom.intrinsic_dim import twonn, principal_angle


@GATES.register("gate1_estimability")
class Gate1(BaseGate):
    name = "gate1_estimability"

    def run(self, table):
        cfg = self.cfg; r = GateResult(self.name); npz = self.params.get("npz")
        # (a) per-layer ID curve (needs the npz path; passed via params)
        if npz:
            r.lines.append("  (a) per-layer TwoNN ID (correct steps):")
            z = np.load(npz, allow_pickle=True)
            sv = [int(x) for x in z["sv_layers"]] if "sv_layers" in z.files else [cfg.layer]
            for si, L in enumerate(sv):
                X = load_layer_matrix(npz, si)
                if len(X) > 4000:
                    X = X[np.random.default_rng(0).choice(len(X), 4000, replace=False)]
                r.lines.append(f"      layer {L:3d}  ID={twonn(X):.2f} (n={len(X)})")
        # (b) tangent cross-fold stability: real vs structure-destroyed null
        cc = table.correct_chains(); X = np.concatenate([c.vecs for c in cc], 0)
        transform = fit_reducer(X, cfg.m, cfg.massive_drop); red = transform(X)
        anchors = np.random.default_rng(1).choice(len(red), min(150, len(red)), replace=False)

        def fold_angle(null):
            data = red.copy()
            if null:
                rng = np.random.default_rng(7)
                data = data[rng.permutation(len(data))] @ np.linalg.qr(rng.normal(size=(data.shape[1],) * 2))[0]
            g = np.arange(len(data)) % cfg.folds
            U = {}
            for fold in (0, 1):
                bank = Bank(data[g != fold], cap=cfg.bank_cap)
                U[fold] = [local_tangent(bank.neighbors(data[a], cfg.k)[0], cfg.dloc) for a in anchors]
            return float(np.mean([principal_angle(U[0][j], U[1][j]) for j in range(len(anchors))]))

        real, null = fold_angle(False), fold_angle(True)
        r.lines.append(f"  (b) tangent cross-fold principal angle: real={real:.3f} null={null:.3f}")
        # (c) normal-energy SNR (cross-fit bank by chain)
        chains = table.chains; pid = np.array([c.problem_id for c in chains]); NN, Y = [], []
        for tr, te in GroupKFold(cfg.folds).split(np.zeros(len(chains)), np.zeros(len(chains)), pid):
            ccx = [chains[i] for i in tr if chains[i].correct]
            if not ccx:
                continue
            Xt = np.concatenate([c.vecs for c in ccx], 0); tf = fit_reducer(Xt, cfg.m, cfg.massive_drop)
            bank = Bank(tf(Xt), cap=cfg.bank_cap)
            for i in te:
                _, Nn, _ = chain_energies(tf(chains[i].vecs), bank, cfg.k, cfg.dloc)
                m = np.isfinite(Nn); NN.append(Nn[m]); Y.append(chains[i].y[m])
        NN = np.concatenate(NN); Y = np.concatenate(Y)
        snr = NN[Y == 1].mean() / (NN[Y == 0].mean() + 1e-9) if (Y == 1).any() else float("nan")
        r.lines.append(f"  (c) normal-energy SNR (err/correct): {snr:.3f}")
        r.kill = bool((null <= real + 0.05) or (snr < 1.0))
        return r
