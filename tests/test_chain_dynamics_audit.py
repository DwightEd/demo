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
    assert meta["depth_band"] == [12, 14]
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

    fixed = res["fixed_mechanism_increment"]
    assert fixed["ready"]
    assert fixed["baseline_definition"]["features"] == [
        "logN",
        "pos",
        "spread",
        "anchor_loss",
        "uncertainty",
    ]
    assert {row["component"] for row in fixed["unique_component_value"]} == {
        "spread",
        "anchor",
        "uncertainty",
    }
    additive = {row["signal"]: row for row in fixed["additive_value"]}
    assert "transition.joint_surprise.length_residual" in additive
    assert "temporal.direction_jump.length_residual" in additive
    assert "depth.prompt_conditioned_update.relative_norm" in additive
    assert np.isfinite(
        additive["transition.joint_surprise.length_residual"]["augmented"]["auroc"]
    )
    assert len(
        additive["transition.joint_surprise.length_residual"][
            "standardized_coefficient"
        ]["fold_values"]
    ) == 3

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

    component_rows = mvs.flatten_component_rows([summary])
    additive_rows = mvs.flatten_additive_rows([summary])
    assert len(component_rows) == 3
    assert additive_rows
    component_aggregate = mvs.aggregate_increment_rows(
        component_rows,
        ["selftest"],
        key="component",
    )
    additive_aggregate = mvs.aggregate_increment_rows(
        additive_rows,
        ["selftest"],
        key="signal",
    )
    assert all(row["complete"] for row in component_aggregate)
    assert all(row["complete"] for row in additive_aggregate)
    missing_additive = mvs.aggregate_increment_rows(
        additive_rows,
        ["selftest", "held_out"],
        key="signal",
    )
    assert not any(row["complete"] for row in missing_additive)
    assert not any(row["auroc_ci_replication"] for row in missing_additive)

    markdown = tmp_path / "replication.md"
    csv_path = tmp_path / "replication.csv"
    component_csv = tmp_path / "components.csv"
    additive_csv = tmp_path / "additive.csv"
    mvs.write_markdown(
        str(markdown),
        [summary],
        aggregate,
        component_aggregate,
        additive_aggregate,
    )
    mvs.write_replication_csv(str(csv_path), [summary])
    mvs.write_rows_csv(str(component_csv), component_rows)
    mvs.write_rows_csv(str(additive_csv), additive_rows)
    assert "Frozen Cross-Dataset Replication" in markdown.read_text(encoding="utf-8")
    assert "What Is Inside `anchor_uncertainty`?" in markdown.read_text(encoding="utf-8")
    assert "Fixed Additive Value" in markdown.read_text(encoding="utf-8")
    assert csv_path.read_text(encoding="utf-8").startswith("dataset,layer,feature")
    assert component_csv.read_text(encoding="utf-8").startswith(
        "dataset,layer,component"
    )
    assert additive_csv.read_text(encoding="utf-8").startswith(
        "dataset,layer,signal"
    )


def test_oof_logit_does_not_mutate_or_use_global_imputation() -> None:
    X = np.array(
        [
            [0.0, np.nan],
            [1.0, 0.1],
            [2.0, 0.2],
            [3.0, 0.3],
            [4.0, 1000.0],
            [5.0, np.nan],
        ]
    )
    original = X.copy()
    y = np.array([0, 1, 0, 1, 0, 1])
    groups = np.array([0, 0, 1, 1, 2, 2])
    score, coefficients = cda.oof_logit_details(X, y, groups, folds=3)
    assert np.array_equal(X, original, equal_nan=True)
    assert np.isfinite(score).all()
    assert coefficients.shape == (3, 2)
