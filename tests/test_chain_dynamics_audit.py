import argparse

import numpy as np

import chain_dynamics_audit as cda
from audit_utils import load_chains


def _args(tmp_path):
    return argparse.Namespace(
        dataset=None,
        data_dir="data",
        layer=14,
        max_chains=0,
        folds=3,
        ridge=1e-3,
        obs=None,
        obs_grid="spread,anchor_loss,uncertainty;spread,anchor_loss,unc_entropy,unc_committal,unc_epistemic",
        min_finite=20,
        recovery_horizon=2,
        high_spread_q=0.70,
        lam=0.8,
        kref=0.25,
        eps_list="0.10,0.20",
        pattern_window=3,
        event_window=2,
        n_boot=20,
        top=12,
        output_dir=str(tmp_path),
        selftest=False,
    )


def test_chain_dynamics_selftest_with_strided_epistemic_uncertainty(tmp_path):
    npz = tmp_path / "chain_dynamics_selftest.npz"
    cda.make_selftest_npz(str(npz), n_chains=54, layer=14, seed=5)

    chains, meta = load_chains(str(npz), layer=14)
    assert meta["has_uncertainty"]["tok_U_E"]
    assert meta["has_uncertainty"]["tok_U_E_offsets"]
    assert sum(np.isfinite(c.features["U_E_mean"]).sum() for c in chains) > 40

    res = cda.run(str(npz), _args(tmp_path))
    cda.assert_selftest(res)

    assert "uncertainty_dynamics" in res["group_oof"]
    dyn_features = res["group_oof"]["uncertainty_dynamics"]["features"]
    assert any(name.startswith("edyn_unc_entropy") for name in dyn_features)
    assert any(name.startswith("edyn_unc_epistemic") for name in dyn_features)
