from __future__ import annotations

import numpy as np
import pytest

import prompt_control_flow.causal_belief_routing.charts as chart_module

from prompt_control_flow.causal_belief_routing.finite_field import (
    enumerate_vectors,
    in_row_span,
    matrix_rank_mod,
)
from prompt_control_flow.causal_belief_routing.geometry import (
    direct_query_distribution,
    fourier_coordinates,
    query_distribution_from_fourier,
)
from prompt_control_flow.causal_belief_routing.routing import (
    head_residual_writes,
    length_matched_control_mask,
    source_head_pre_output,
)
from prompt_control_flow.causal_belief_routing.routing_schema import (
    EvidenceRoutingTrace,
    ROUTING_SCHEMA,
)
from prompt_control_flow.causal_belief_routing.routing_audit import (
    RoutingAuditConfig,
    run_routing_audit,
)
from prompt_control_flow.causal_belief_routing.patching import (
    apply_source_component_patch,
    frozen_pair_folds,
)
from prompt_control_flow.causal_belief_routing.patch_schema import (
    PATCH_SCHEMA,
    SourcePatchTrace,
)
from prompt_control_flow.causal_belief_routing.patch_audit import (
    SourcePatchAuditConfig,
    run_source_patch_audit,
)
from prompt_control_flow.causal_belief_routing.data import build_alias_observations
from prompt_control_flow.causal_belief_routing.schema import CausalBeliefTrace
from prompt_control_flow.causal_belief_routing.charts import (
    LayerChartBundle,
    RandomProjection,
    RidgeChart,
    build_group_fold_ids,
    fit_ridge_accelerated,
    fit_layer_chart_bundle,
)
from prompt_control_flow.causal_belief_routing.metrics import (
    cluster_bootstrap_mean,
    evaluate_fourier_predictions,
)
from prompt_control_flow.causal_belief_routing.audit import (
    RepresentationAuditConfig,
    run_representation_audit,
)
from prompt_control_flow.causal_belief_routing.extraction import (
    resolve_residue_token_groups,
)
from prompt_control_flow.causal_belief_routing.world import (
    AliasWorldConfig,
    generate_alias_worlds,
    load_alias_worlds_jsonl,
    write_alias_worlds_jsonl,
)


def test_generated_worlds_are_exact_predictive_aliases() -> None:
    cfg = AliasWorldConfig(
        modulus=3,
        num_variables=4,
        common_rank=2,
        template_families=3,
        seed=17,
    )

    worlds = generate_alias_worlds(24, cfg)

    assert len(worlds) == 24
    for world in worlds:
        supports = [world.support_mask(branch) for branch in (0, 1)]
        assert int(supports[0].sum()) == int(supports[1].sum()) == cfg.modulus
        assert not np.array_equal(supports[0], supports[1])

        current = [
            direct_query_distribution(
                world.assignments,
                support,
                world.current_query,
                cfg.modulus,
            )
            for support in supports
        ]
        future = [
            direct_query_distribution(
                world.assignments,
                support,
                world.future_query,
                cfg.modulus,
            )
            for support in supports
        ]

        assert np.allclose(current[0], current[1], atol=1e-12)
        assert np.allclose(current[0], np.full(cfg.modulus, 1.0 / cfg.modulus))
        assert np.max(future[0]) == pytest.approx(1.0)
        assert np.max(future[1]) == pytest.approx(1.0)
        assert int(np.argmax(future[0])) != int(np.argmax(future[1]))


def test_affine_belief_fourier_support_equals_constraint_row_span() -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=23)
    world = generate_alias_worlds(1, cfg)[0]
    frequencies = enumerate_vectors(cfg.modulus, cfg.num_variables)
    common_matrix = np.asarray(
        [constraint.coefficients for constraint in world.common_constraints],
        dtype=np.int64,
    )
    support = world.base_support_mask()

    phi = fourier_coordinates(
        world.assignments,
        support,
        frequencies,
        cfg.modulus,
    )
    expected_nonzero = np.asarray(
        [in_row_span(frequency, common_matrix, cfg.modulus) for frequency in frequencies]
    )

    assert np.array_equal(np.abs(phi) > 1e-9, expected_nonzero)
    assert matrix_rank_mod(common_matrix, cfg.modulus) == cfg.common_rank


def test_fourier_coordinates_recover_current_and_future_query_distributions() -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=29)
    world = generate_alias_worlds(1, cfg)[0]
    frequencies = enumerate_vectors(cfg.modulus, cfg.num_variables)

    for branch in (0, 1):
        support = world.support_mask(branch)
        phi = fourier_coordinates(
            world.assignments,
            support,
            frequencies,
            cfg.modulus,
        )
        for query in (world.current_query, world.future_query):
            direct = direct_query_distribution(
                world.assignments,
                support,
                query,
                cfg.modulus,
            )
            recovered = query_distribution_from_fourier(
                phi,
                frequencies,
                query,
                cfg.modulus,
            )
            assert np.allclose(recovered, direct, atol=1e-10)


def test_branch_constraint_adds_new_fourier_modes() -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=31)
    world = generate_alias_worlds(1, cfg)[0]
    frequencies = enumerate_vectors(cfg.modulus, cfg.num_variables)
    base_phi = fourier_coordinates(
        world.assignments,
        world.base_support_mask(),
        frequencies,
        cfg.modulus,
    )
    branch_phi = fourier_coordinates(
        world.assignments,
        world.support_mask(0),
        frequencies,
        cfg.modulus,
    )

    newly_active = (np.abs(base_phi) <= 1e-9) & (np.abs(branch_phi) > 1e-9)

    assert int(newly_active.sum()) == (cfg.modulus - 1) * (cfg.modulus ** cfg.common_rank)
    assert np.linalg.norm(branch_phi - base_phi) > 0.0


def test_source_head_pre_output_matches_manual_attention_sum() -> None:
    torch = pytest.importorskip("torch")
    attention = torch.tensor(
        [[[[0.10, 0.20, 0.30, 0.40]], [[0.40, 0.30, 0.20, 0.10]]]],
        dtype=torch.float32,
    )
    values = torch.arange(1 * 4 * 2 * 3, dtype=torch.float32).reshape(1, 4, 2, 3)
    source_mask = torch.tensor([[False, True, True, False]])

    contribution, mass = source_head_pre_output(
        attention,
        values,
        target_indices=torch.tensor([0]),
        source_mask=source_mask,
    )
    manual = torch.stack(
        [
            0.20 * values[0, 1, 0] + 0.30 * values[0, 2, 0],
            0.30 * values[0, 1, 1] + 0.20 * values[0, 2, 1],
        ]
    )

    assert contribution.shape == (1, 2, 3)
    assert torch.allclose(contribution[0], manual)
    assert torch.allclose(mass[0], torch.tensor([0.50, 0.50]))

    output_projection = torch.eye(6)
    writes = head_residual_writes(contribution, output_projection)
    assert writes.shape == (1, 2, 6)
    assert torch.allclose(writes[0, 0, :3], contribution[0, 0])
    assert torch.allclose(writes[0, 0, 3:], torch.zeros(3))

    head_input = torch.zeros((1, 4, 6))
    component_delta = torch.arange(6, dtype=torch.float32).reshape(1, 2, 3)
    patched = apply_source_component_patch(
        head_input,
        target_indices=torch.tensor([3]),
        component_delta=component_delta,
        selected_heads=[1],
    )
    assert torch.allclose(patched[0, 3, :3], torch.zeros(3))
    assert torch.allclose(patched[0, 3, 3:], component_delta[0, 1])


def test_length_matched_control_window_preserves_mass_and_avoids_evidence() -> None:
    ranges = np.asarray([[4, 7], [1, 3]])
    control = length_matched_control_mask(
        np.asarray([12, 8]), ranges, padded_length=16
    )

    assert control.shape == (2, 16)
    assert np.array_equal(control.sum(axis=1), np.asarray([3, 2]))
    for row, (start, stop) in enumerate(ranges):
        assert not np.any(control[row, start:stop])
    assert not np.any(control[0, 12:])
    assert not np.any(control[1, 8:])


def test_residue_token_groups_include_distinct_single_token_surface_forms() -> None:
    mapping = {
        "0": [10],
        " 0": [20],
        "\n0": [30, 10],
        "1": [11],
        " 1": [21],
        "\n1": [31, 11],
        "2": [12],
        " 2": [22],
        "\n2": [32, 12],
    }

    class FakeTokenizer:
        def __call__(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return {"input_ids": mapping[text]}

    groups = resolve_residue_token_groups(FakeTokenizer(), 3)

    assert groups == ((10, 20), (11, 21), (12, 22))


def test_alias_world_jsonl_roundtrip_preserves_exact_system(tmp_path) -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=37)
    worlds = generate_alias_worlds(5, cfg)
    path = tmp_path / "aliases.jsonl"

    write_alias_worlds_jsonl(path, worlds, cfg)
    loaded, loaded_cfg = load_alias_worlds_jsonl(path)

    assert loaded_cfg == cfg
    assert loaded == worlds


def test_observation_contract_has_two_queries_per_branch() -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=41)
    worlds = generate_alias_worlds(3, cfg)

    observations, frequencies = build_alias_observations(worlds)

    assert len(observations) == 12
    assert frequencies.shape == (cfg.modulus ** cfg.num_variables, cfg.num_variables)
    for pair_id in range(3):
        local = [row for row in observations if row.pair_id == pair_id]
        assert {(row.branch, row.query_role) for row in local} == {
            (0, "current"),
            (1, "current"),
            (0, "future"),
            (1, "future"),
        }
        current = [row for row in local if row.query_role == "current"]
        assert np.allclose(current[0].exact_query_distribution, current[1].exact_query_distribution)


def test_trace_artifact_roundtrip_preserves_row_alignment(tmp_path) -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=43)
    observations, frequencies = build_alias_observations(generate_alias_worlds(2, cfg))
    rows = len(observations)
    layers = np.asarray([0, 2], dtype=np.int64)
    states = np.arange(rows * 2 * 5, dtype=np.float32).reshape(rows, 2, 5)
    artifact = CausalBeliefTrace.from_observations(
        observations,
        frequencies=frequencies,
        layers=layers,
        states=states,
        residue_logits=np.zeros((rows, cfg.modulus), dtype=np.float32),
        logit_sketch=np.zeros((rows, 7), dtype=np.float32),
        rendered_prompts=[row.user_text for row in observations],
        input_ids=[np.asarray([1, 2, row.row_id + 3]) for row in observations],
        evidence_token_ranges=np.tile(np.asarray([[1, 2]]), (rows, 1)),
        metadata={"model": "dummy"},
    )
    path = tmp_path / "trace.npz"

    artifact.save(path)
    loaded = CausalBeliefTrace.load(path)

    assert loaded.states.shape == (rows, 2, 5)
    assert loaded.metadata["model"] == "dummy"
    assert np.array_equal(loaded.pair_ids, artifact.pair_ids)
    assert np.array_equal(loaded.layers, layers)


def test_group_folds_keep_alias_pair_together() -> None:
    groups = np.repeat(np.arange(15), 2)
    fold_ids = build_group_fold_ids(groups, num_folds=5, seed=47)

    assert set(fold_ids.tolist()) == set(range(5))
    for group in np.unique(groups):
        assert len(np.unique(fold_ids[groups == group])) == 1


def test_projected_ridge_chart_recovers_linear_coordinates() -> None:
    rng = np.random.default_rng(53)
    features = rng.normal(size=(160, 20)).astype(np.float32)
    truth = rng.normal(size=(20, 8)).astype(np.float32)
    targets = features @ truth + 0.01 * rng.normal(size=(160, 8))
    projection = RandomProjection(input_dim=20, output_dim=20, seed=59)
    chart = RidgeChart.fit(
        projection.transform(features),
        targets,
        alpha=1e-3,
    )

    predictions = chart.predict(projection.transform(features))

    assert np.mean((predictions - targets) ** 2) < 1e-3


def test_accelerated_ridge_dispatches_non_cpu_backend(monkeypatch) -> None:
    marker = RidgeChart(
        feature_mean=np.zeros(2, dtype=np.float32),
        feature_scale=np.ones(2, dtype=np.float32),
        target_mean=np.zeros(1, dtype=np.float32),
        weight=np.zeros((2, 1), dtype=np.float32),
        alpha=1.0,
    )
    calls = []

    def fake_fit(features, targets, *, alpha, compute_device):
        calls.append((features.shape, targets.shape, alpha, compute_device))
        return marker

    monkeypatch.setattr(chart_module, "_fit_ridge_torch", fake_fit)
    result = fit_ridge_accelerated(
        np.zeros((4, 2)),
        np.zeros((4, 1)),
        alpha=3.0,
        compute_device="cuda",
    )

    assert result is marker
    assert calls == [((4, 2), (4, 1), 3.0, "cuda")]


def test_ridge_dual_form_handles_more_features_than_rows() -> None:
    rng = np.random.default_rng(60)
    features = rng.normal(size=(24, 80))
    targets = features[:, :5] @ rng.normal(size=(5, 3))

    model = RidgeChart.fit(features, targets, alpha=1e-2)
    predictions = model.predict(features)

    assert np.mean((predictions - targets) ** 2) < 1e-4


def test_layer_chart_bundle_roundtrip_preserves_pair_fold_mapping(tmp_path) -> None:
    rng = np.random.default_rng(61)
    pair_ids = np.repeat(np.arange(20), 2)
    states = rng.normal(size=(40, 3, 12)).astype(np.float32)
    truth = rng.normal(size=(3, 12, 6)).astype(np.float32)
    targets = np.einsum("nlh,lho->nlo", states, truth).mean(axis=1)

    bundle, predictions, fold_ids = fit_layer_chart_bundle(
        states,
        targets,
        pair_ids,
        np.asarray([4, 8, 12]),
        num_folds=4,
        projection_dim=12,
        projection_seed=67,
        alpha=1e-3,
        split_seed=71,
    )
    path = tmp_path / "charts.npz"
    bundle.save(path)
    loaded = LayerChartBundle.load(path)

    assert predictions.shape == (40, 3, 6)
    assert fold_ids.shape == (40,)
    assert loaded.fold_for_pair(7) == bundle.fold_for_pair(7)
    assert np.array_equal(loaded.layers, np.asarray([4, 8, 12]))


def test_fourier_prediction_metrics_measure_future_query_information() -> None:
    cfg = AliasWorldConfig(modulus=3, num_variables=4, common_rank=2, seed=73)
    observations, frequencies = build_alias_observations(generate_alias_worlds(8, cfg))
    current = [row for row in observations if row.query_role == "current"]
    future = {
        (row.pair_id, row.branch): row
        for row in observations
        if row.query_role == "future"
    }
    targets = np.stack([row.fourier_coordinates for row in current])
    queries = np.stack(
        [future[(row.pair_id, row.branch)].query_vector for row in current]
    )
    labels = np.asarray(
        [
            np.argmax(future[(row.pair_id, row.branch)].exact_query_distribution)
            for row in current
        ]
    )

    perfect = evaluate_fourier_predictions(
        targets,
        targets,
        frequencies,
        queries,
        labels,
        modulus=cfg.modulus,
    )
    collapsed = evaluate_fourier_predictions(
        np.zeros_like(targets),
        targets,
        frequencies,
        queries,
        labels,
        modulus=cfg.modulus,
    )

    assert perfect["future_accuracy"] == pytest.approx(1.0)
    assert perfect["future_nll_nats"] < 1e-8
    assert collapsed["future_nll_nats"] == pytest.approx(np.log(3.0))
    assert perfect["fourier_mse"] < collapsed["fourier_mse"]


def test_cluster_bootstrap_mean_resamples_whole_alias_pairs() -> None:
    values = np.asarray([1.0, 3.0, 2.0, 4.0])
    pair_ids = np.asarray([10, 10, 20, 20])

    summary = cluster_bootstrap_mean(values, pair_ids, draws=200, seed=79)

    assert summary["point"] == pytest.approx(2.5)
    assert summary["groups"] == 2
    assert summary["ci_low"] <= summary["point"] <= summary["ci_high"]


def test_representation_audit_detects_hidden_future_under_output_alias(tmp_path) -> None:
    cfg = AliasWorldConfig(
        modulus=2,
        num_variables=3,
        common_rank=1,
        template_families=2,
        seed=83,
    )
    observations, frequencies = build_alias_observations(generate_alias_worlds(60, cfg))
    rows = len(observations)
    targets = np.stack([row.fourier_coordinates for row in observations]).astype(
        np.float32
    )
    states = np.stack([targets, 0.75 * targets], axis=1)
    residue_logits = np.zeros((rows, cfg.modulus), dtype=np.float32)
    for index, row in enumerate(observations):
        if row.query_role == "future":
            residue_logits[index, np.argmax(row.exact_query_distribution)] = 8.0
    artifact = CausalBeliefTrace.from_observations(
        observations,
        frequencies=frequencies,
        layers=np.asarray([8, 16]),
        states=states,
        residue_logits=residue_logits,
        logit_sketch=np.zeros((rows, 6), dtype=np.float32),
        rendered_prompts=[row.user_text for row in observations],
        input_ids=[np.arange(8 + row.template_family) for row in observations],
        evidence_token_ranges=np.tile(np.asarray([[2, 5]]), (rows, 1)),
        metadata={"model": "synthetic", "modulus": cfg.modulus},
    )
    trace_path = tmp_path / "trace.npz"
    artifact.save(trace_path)

    report = run_representation_audit(
        trace_path,
        tmp_path / "audit",
        RepresentationAuditConfig(
            folds=3,
            projection_dim=targets.shape[1],
            ridge_alpha=1e-3,
            bootstrap=100,
            seed=89,
        ),
    )

    assert report["alias_checks"]["model_current_js_median"] == pytest.approx(0.0)
    assert report["prediction_metrics"]["hidden_all_layers"]["future_accuracy"] > 0.95
    assert (
        report["conditional_information"]["joint_over_output_plus_controls"]["ci_low"]
        > 0.0
    )
    assert report["decision_gate"]["representation_supported"] is True
    assert (tmp_path / "audit" / "layer_charts.npz").exists()


def test_routing_audit_crossfits_head_selection_and_passes_known_signal(tmp_path) -> None:
    rng = np.random.default_rng(97)
    pair_ids = np.repeat(np.arange(40), 2)
    branches = np.tile(np.asarray([0, 1]), 40)
    shape = (len(pair_ids), 2, 4)
    evidence_true = 0.05 * rng.normal(size=shape)
    evidence_opposite = 0.05 * rng.normal(size=shape)
    evidence_true[:, 1, 2] += 0.9
    evidence_opposite[:, 1, 2] -= 0.3
    trace = EvidenceRoutingTrace(
        row_indices=np.arange(len(pair_ids)),
        pair_ids=pair_ids,
        branches=branches,
        layers=np.asarray([8, 16]),
        evidence_mass=np.full(shape, 0.4, dtype=np.float32),
        control_mass=np.full(shape, 0.4, dtype=np.float32),
        evidence_alignment_true=evidence_true.astype(np.float32),
        evidence_alignment_opposite=evidence_opposite.astype(np.float32),
        control_alignment_true=np.zeros(shape, dtype=np.float32),
        control_alignment_opposite=np.zeros(shape, dtype=np.float32),
        evidence_write_norm=np.ones(shape, dtype=np.float32),
        control_write_norm=np.ones(shape, dtype=np.float32),
        layer_alignment_true=np.full((len(pair_ids), 2), 0.2, dtype=np.float32),
        layer_alignment_opposite=np.zeros((len(pair_ids), 2), dtype=np.float32),
        metadata={
            "schema": ROUTING_SCHEMA,
            "representation_gate": {"ready_for_routing_analysis": True},
        },
    )
    path = tmp_path / "routing.npz"
    trace.save(path)

    report = run_routing_audit(
        path,
        tmp_path / "routing_audit",
        RoutingAuditConfig(folds=4, top_heads=1, bootstrap=100, seed=101),
    )

    assert report["tests"]["routed_update_above_zero"]["ci_low"] > 0.0
    assert report["tests"]["evidence_over_length_matched_control"]["ci_low"] > 0.0
    assert report["decision_gate"]["routing_supported"] is True
    loaded = EvidenceRoutingTrace.load(path)
    assert np.array_equal(loaded.layers, np.asarray([8, 16]))


def test_source_patch_audit_requires_donor_shift_beyond_control(tmp_path) -> None:
    rng = np.random.default_rng(103)
    pair_ids = np.repeat(np.arange(40), 2)
    directions = len(pair_ids)
    evidence_shift = 0.8 + 0.05 * rng.normal(size=directions)
    control_shift = 0.02 * rng.normal(size=directions)
    trace = SourcePatchTrace(
        pair_ids=pair_ids,
        recipient_branches=np.tile(np.asarray([0, 1]), 40),
        donor_branches=np.tile(np.asarray([1, 0]), 40),
        fold_ids=np.repeat(np.arange(40) % 4, 2),
        selected_head_counts=np.full(directions, 4),
        replay_js=np.full(directions, 1e-5),
        evidence_logodds_shift=evidence_shift.astype(np.float32),
        control_logodds_shift=control_shift.astype(np.float32),
        random_head_logodds_shift=control_shift.astype(np.float32),
        evidence_donor_probability_shift=(0.2 * evidence_shift).astype(np.float32),
        control_donor_probability_shift=(0.2 * control_shift).astype(np.float32),
        random_head_donor_probability_shift=(0.2 * control_shift).astype(np.float32),
        metadata={
            "schema": PATCH_SCHEMA,
            "coverage": 1.0,
            "routing_gate": {"ready_for_causal_patching": True},
        },
    )
    path = tmp_path / "patches.npz"
    trace.save(path)

    report = run_source_patch_audit(
        path,
        tmp_path / "patch_audit",
        SourcePatchAuditConfig(bootstrap=100, seed=107),
    )

    assert report["tests"]["evidence_patch_moves_toward_donor_logodds"]["ci_low"] > 0
    assert report["tests"]["evidence_beats_control_logodds"]["ci_low"] > 0
    assert report["decision_gate"]["causal_routing_supported"] is True
    loaded = SourcePatchTrace.load(path)
    assert len(loaded.pair_ids) == directions


def test_patch_pilot_preserves_full_routing_fold_assignments() -> None:
    all_pairs = np.arange(80)
    selected = all_pairs[:17]
    full_folds = build_group_fold_ids(all_pairs, num_folds=5, seed=109)

    pilot_folds = frozen_pair_folds(
        all_pairs,
        selected,
        folds=5,
        seed=109,
    )

    assert np.array_equal(pilot_folds, full_folds[: len(selected)])
