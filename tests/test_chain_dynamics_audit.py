import argparse

import numpy as np

import chain_dynamics_audit as cda
import mainline_validation_suite as mvs
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
        obs_grid=None,
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

    replication = {row["feature"]: row for row in res["predeclared_replication"]}
    assert "spread" in replication
    assert "d_spread" in replication
    assert "transition_surprise__spread" in replication
    assert replication["spread"]["expected_direction"] == "higher_is_error"
    assert replication["spread"]["raw"]["auroc_high_is_error"] > 0.8
    assert np.isfinite(
        replication["transition_surprise__spread"]["nuisance_residual"][
            "auroc_high_is_error"
        ]
    )

    summary = mvs.summarize_result("selftest", 14, res, max_fpr=0.20)
    aggregate = mvs.aggregate_replication([summary], ["selftest"])
    assert aggregate
    assert all(row["complete"] for row in aggregate)
    assert any(row["feature"] == "spread" for row in aggregate)
    incomplete = mvs.aggregate_replication([summary], ["selftest", "held_out"])
    assert incomplete
    assert not any(row["complete"] for row in incomplete)
    assert not any(row["raw_ci_replication"] for row in incomplete)
    assert not any(row["residual_ci_replication"] for row in incomplete)

    markdown = tmp_path / "replication.md"
    csv_path = tmp_path / "replication.csv"
    mvs.write_markdown(str(markdown), [summary], aggregate)
    mvs.write_replication_csv(str(csv_path), [summary])
    assert "Frozen Cross-Dataset Replication" in markdown.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").startswith("dataset,layer,feature")
