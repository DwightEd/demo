from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from functional_divergence.hidden_state_geometry.component_contract import (
    ComponentStepArtifact,
)
from functional_divergence.hidden_state_geometry.contracts import ChainSample
from functional_divergence.hidden_state_geometry.method import FoldInput
from functional_divergence.hidden_state_geometry.methods import (
    component_resolved_hazard as hazard_module,
)
from functional_divergence.hidden_state_geometry.methods import load_builtin_methods
from functional_divergence.hidden_state_geometry.registry import (
    create_method,
    method_spec,
)
from functional_divergence.hidden_state_geometry.tasks import (
    build_post_step_task,
    build_whole_chain_task,
)
from sklearn.metrics import roc_auc_score

ComponentResolvedHazardConfig = hazard_module.ComponentResolvedHazardConfig
_attention_source_buckets = hazard_module._attention_source_buckets


EXPECTED_ARMS = {
    "nuisance",
    "output_only",
    "hidden_post",
    "attention_only",
    "mlp_only",
    "propagation_only",
    "components_all",
    "components_plus_hidden",
    "all_minus_attention",
    "all_minus_mlp",
    "all_minus_propagation",
}


EXPECTED_CONTRASTS = {
    ("output_summary_given_nuisance_nll", "nuisance", "output_only"),
    (
        "components_all_vs_capacity_matched_hidden_nll",
        "hidden_post",
        "components_all",
    ),
    (
        "components_given_capacity_matched_hidden_nll",
        "hidden_post",
        "components_plus_hidden",
    ),
    ("attention_innovation_given_output_nll", "output_only", "attention_only"),
    ("mlp_innovation_given_output_nll", "output_only", "mlp_only"),
    (
        "propagation_innovation_given_output_nll",
        "output_only",
        "propagation_only",
    ),
    (
        "attention_full_vs_all_minus_nll",
        "all_minus_attention",
        "components_all",
    ),
    ("mlp_full_vs_all_minus_nll", "all_minus_mlp", "components_all"),
    (
        "propagation_full_vs_all_minus_nll",
        "all_minus_propagation",
        "components_all",
    ),
}


def _metadata(chain_id: int, dataset: str) -> dict[str, str]:
    return {
        "schema": "component_step_v1",
        "sample_id": f"chain-{chain_id}",
        "dataset": dataset,
        "model_name": "meta-llama/Llama-3.1-8B-Instruct",
        "model_revision": "rev-model",
        "tokenizer_name": "meta-llama/Llama-3.1-8B-Instruct",
        "tokenizer_revision": "rev-tokenizer",
        "extractor_commit": "abc123",
        "source_trace_sha256": "0" * 64,
        "generation_config_sha256": "1" * 64,
    }


def _step_labels(n_steps: int, gold: int) -> np.ndarray:
    if gold == -1:
        return np.zeros(n_steps, dtype=np.int8)
    labels = np.full(n_steps, -1, dtype=np.int8)
    labels[:gold] = 0
    labels[gold] = 1
    return labels


def _source_grid(n_steps: int) -> tuple[np.ndarray, np.ndarray]:
    source_step_id = np.full((n_steps, n_steps + 1), -32768, dtype=np.int16)
    source_mask = np.zeros((n_steps, n_steps + 1), dtype=bool)
    for step in range(n_steps):
        ids = [-1, *range(step + 1)]
        source_step_id[step, : len(ids)] = np.asarray(ids, dtype=np.int16)
        source_mask[step, : len(ids)] = True
    return source_step_id, source_mask


def _artifact(
    *,
    chain_id: int,
    dataset: str,
    gold: int,
    n_steps: int = 4,
    attention_signal_scale: float = 0.0,
    attention_noise: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ComponentStepArtifact:
    hidden_size = 6
    attention_layers = 2
    mlp_layers = 2
    residual_layers = 2
    input_ids = np.arange(10, 10 + n_steps + 1, dtype=np.int32)
    boundaries = np.arange(1, n_steps + 2, dtype=np.int32)
    source_step_id, source_mask = _source_grid(n_steps)
    attn_msg = np.zeros(
        (n_steps, attention_layers, n_steps + 1, hidden_size),
        dtype=np.float32,
    )
    if rng is not None and attention_noise > 0.0:
        attn_msg += rng.normal(scale=attention_noise, size=attn_msg.shape).astype(
            np.float32
        )
        attn_msg *= source_mask[:, None, :, None]
    if gold >= 0 and attention_signal_scale:
        current_source = gold + 1
        direction = np.zeros((attention_layers, hidden_size), dtype=np.float32)
        direction[:, 0] = attention_signal_scale
        direction[:, 1] = -0.75 * attention_signal_scale
        attn_msg[gold, :, current_source, :] += direction
    resid = np.zeros((n_steps + 1, residual_layers, hidden_size), dtype=np.float32)
    mlp = np.zeros((n_steps, mlp_layers, hidden_size), dtype=np.float32)
    return ComponentStepArtifact(
        input_ids=input_ids,
        step_token_start=boundaries[:-1],
        step_token_end=boundaries[1:],
        boundary_token_end=boundaries,
        first_error_step=gold,
        step_label=_step_labels(n_steps, gold),
        source_step_id=source_step_id,
        source_mask=source_mask,
        resid_boundary=resid.astype(np.float16),
        attn_msg_resid_by_source=attn_msg.astype(np.float16),
        attn_out_step=np.sum(attn_msg, axis=2).astype(np.float16),
        mlp_out_step=mlp.astype(np.float16),
        residual_layers=np.asarray([8, 12], dtype=np.int16),
        attention_layers=np.asarray([8, 12], dtype=np.int16),
        mlp_layers=np.asarray([8, 12], dtype=np.int16),
        metadata=_metadata(chain_id, dataset),
    )


def _sample(
    tmp_path,
    *,
    chain_id: int,
    dataset: str,
    gold: int,
    component_path=None,
) -> ChainSample:
    state_path = tmp_path / f"chain_{chain_id}.npy"
    np.save(state_path, np.zeros((5, 2, 6), dtype=np.float32))
    return ChainSample(
        chain_id=chain_id,
        manifest_row=chain_id,
        problem_group=f"problem-{chain_id}",
        dataset=dataset,
        generator="Llama-3.1-8B",
        observer_model="Llama-3.1-8B",
        state_path=state_path,
        state_count=5,
        response_start=1,
        step_ranges=np.asarray([[1 + step, 1 + step] for step in range(4)]),
        layer_ids=np.asarray([8, 12], dtype=np.int64),
        output_steps=np.zeros((4, 2), dtype=np.float32),
        output_feature_names=("token_entropy", "token_nll"),
        first_error_step=gold,
        component_path=component_path,
    )


def _write_sample(
    tmp_path,
    *,
    chain_id: int,
    dataset: str,
    gold: int,
    attention_signal_scale: float = 0.0,
    rng: np.random.Generator | None = None,
) -> ChainSample:
    component_path = tmp_path / f"{dataset}_{chain_id}.component_step_v1.npz"
    sample = _sample(
        tmp_path,
        chain_id=chain_id,
        dataset=dataset,
        gold=gold,
        component_path=component_path,
    )
    _artifact(
        chain_id=chain_id,
        dataset=dataset,
        gold=gold,
        attention_signal_scale=attention_signal_scale,
        attention_noise=0.02,
        rng=rng,
    ).save(component_path)
    return sample


def _attention_fold(tmp_path) -> tuple[FoldInput, np.ndarray]:
    rng = np.random.default_rng(7)
    samples = []
    chain_id = 0
    for domain in ("train_a", "train_b", "train_c", "test"):
        for block in range(7):
            gold = block % 4
            samples.append(
                _write_sample(
                    tmp_path,
                    chain_id=chain_id,
                    dataset=domain,
                    gold=gold,
                    attention_signal_scale=8.0,
                    rng=rng,
                )
            )
            chain_id += 1
            samples.append(
                _write_sample(
                    tmp_path,
                    chain_id=chain_id,
                    dataset=domain,
                    gold=-1,
                    attention_signal_scale=0.0,
                    rng=rng,
                )
            )
            chain_id += 1
    task = build_post_step_task(tuple(samples))
    train = np.flatnonzero(task.domains != "test")
    test = np.flatnonzero(task.domains == "test")
    return (
        FoldInput(
            task_name=task.name,
            train_examples=tuple(task.examples[index] for index in train),
            train_labels=task.labels[train],
            train_groups=task.groups[train],
            test_examples=tuple(task.examples[index] for index in test),
            seed=23,
        ),
        task.labels[test],
    )


def test_component_resolved_hazard_is_registered_with_expected_arms_and_contrasts():
    load_builtin_methods()

    specification = method_spec("component_resolved_hazard")
    method = create_method(
        "component_resolved_hazard",
        {
            "attention_rank": 1,
            "mlp_rank": 1,
            "propagation_rank": 1,
            "pre_context_rank": 1,
            "l2": 0.01,
        },
    )

    assert isinstance(method.config, ComponentResolvedHazardConfig)
    assert set(specification.arm_definitions) == EXPECTED_ARMS
    assert {
        (item.name, item.baseline, item.candidate) for item in specification.contrasts
    } == EXPECTED_CONTRASTS
    assert all(
        "causal" not in item.description.lower() for item in specification.contrasts
    )
    assert all(
        "diagnostic" in item.description.lower()
        or "predictive increment" in item.description.lower()
        for item in specification.contrasts
    )


def test_component_resolved_hazard_rejects_non_post_step_tasks(tmp_path):
    sample = _write_sample(
        tmp_path,
        chain_id=0,
        dataset="train",
        gold=1,
        attention_signal_scale=1.0,
        rng=np.random.default_rng(0),
    )
    task = build_whole_chain_task(
        (sample, replace(sample, chain_id=1, first_error_step=-1))
    )
    fold = FoldInput(
        task_name=task.name,
        train_examples=task.examples,
        train_labels=task.labels,
        train_groups=task.groups,
        test_examples=task.examples,
        seed=1,
    )

    with pytest.raises(ValueError, match="post_step"):
        create_method("component_resolved_hazard", None).fit_predict(fold)


def test_attention_source_buckets_are_exact_at_step_zero_and_later():
    artifact = _artifact(chain_id=5, dataset="gsm8k", gold=-1, n_steps=4)
    messages = np.zeros_like(artifact.attn_msg_resid_by_source, dtype=np.float32)
    messages[0, :, 0, :] = 1.0
    messages[0, :, 1, :] = 2.0
    messages[3, :, 0, :] = 10.0
    messages[3, :, 1, :] = 20.0
    messages[3, :, 2, :] = 30.0
    messages[3, :, 3, :] = 40.0
    messages[3, :, 4, :] = 50.0
    artifact = ComponentStepArtifact(
        **{
            **artifact.__dict__,
            "attn_msg_resid_by_source": messages.astype(np.float16),
            "attn_out_step": np.sum(messages, axis=2).astype(np.float16),
        }
    )

    step0 = _attention_source_buckets(artifact, 0)
    step3 = _attention_source_buckets(artifact, 3)

    assert step0.shape == (4, 2, 6)
    assert np.all(step0[0] == 1.0)
    assert np.all(step0[1] == 0.0)
    assert np.all(step0[2] == 0.0)
    assert np.all(step0[3] == 2.0)
    assert np.all(step3[0] == 10.0)
    assert np.all(step3[1] == 20.0 + 30.0)
    assert np.all(step3[2] == 40.0)
    assert np.all(step3[3] == 50.0)


def test_capacity_matched_dimensions_and_diagnostics_are_honest(tmp_path):
    fold, _ = _attention_fold(tmp_path)
    method = create_method(
        "component_resolved_hazard",
        {
            "attention_rank": 1,
            "mlp_rank": 1,
            "propagation_rank": 1,
            "pre_context_rank": 1,
            "normal_ridge_alpha": 1.0,
            "l2": 0.01,
            "max_iter": 1000,
        },
    )

    result = method.fit_predict(fold)

    dimensions = result.diagnostics["arm_feature_dimensions"]
    assert dimensions["components_all"] == dimensions["hidden_post"]
    assert dimensions["all_minus_attention"] == dimensions["components_all"]
    assert dimensions["all_minus_mlp"] == dimensions["components_all"]
    assert dimensions["all_minus_propagation"] == dimensions["components_all"]
    assert result.diagnostics["target_timing"] == "post_step_after_current_step"
    assert result.diagnostics["no_post_error_policy"] == "censored_by_post_step_task"
    assert result.diagnostics["source_bucket_definitions"] == {
        "prompt": "source_step_id == -1",
        "earlier_steps": "source_step_id < current_step - 1",
        "previous_step": "source_step_id == current_step - 1",
        "current_step": "source_step_id == current_step",
    }
    assert result.diagnostics["ranks"] == {
        "attention": 1,
        "mlp": 1,
        "propagation": 1,
        "pre_context": 1,
        "hidden_post": 3,
        "total_component": 3,
    }
    assert result.diagnostics["transform_fit_scope"] == "outer_training_fold_only"
    assert (
        "observational predictive decomposition"
        in result.diagnostics["claim_limitation"]
    )
    assert (
        "activation patching required for causal attribution"
        in result.diagnostics["claim_limitation"]
    )
    assert result.diagnostics["normal_bank_rows"] == int(np.sum(fold.train_labels == 0))
    assert result.factors["training_weights"].shape == fold.train_labels.shape
    assert "attention_projection.components" in result.factors
    assert "attention_normal_map.coefficients" in result.factors


def test_attention_message_innovation_beats_output_and_all_minus_attention(tmp_path):
    fold, test_labels = _attention_fold(tmp_path)
    method = create_method(
        "component_resolved_hazard",
        {
            "attention_rank": 1,
            "mlp_rank": 1,
            "propagation_rank": 1,
            "pre_context_rank": 1,
            "normal_ridge_alpha": 1.0,
            "l2": 0.001,
            "max_iter": 1000,
        },
    )

    result = method.fit_predict(fold)

    attention_auc = roc_auc_score(test_labels, result.probabilities["attention_only"])
    components_auc = roc_auc_score(test_labels, result.probabilities["components_all"])
    output_auc = roc_auc_score(test_labels, result.probabilities["output_only"])
    hidden_auc = roc_auc_score(test_labels, result.probabilities["hidden_post"])
    minus_attention_auc = roc_auc_score(
        test_labels,
        result.probabilities["all_minus_attention"],
    )
    assert attention_auc > 0.9
    assert components_auc > 0.9
    assert output_auc < 0.75
    assert hidden_auc < 0.75
    assert attention_auc > output_auc + 0.2
    assert components_auc > minus_attention_auc + 0.2
    assert all(
        probability.shape == test_labels.shape
        for probability in result.probabilities.values()
    )
    assert all(
        np.isfinite(probability).all() for probability in result.probabilities.values()
    )


def test_missing_component_file_fails_explicitly(tmp_path):
    samples = []
    for index, gold in enumerate((1, -1, 1, -1)):
        sample = _sample(
            tmp_path,
            chain_id=index,
            dataset="train",
            gold=gold,
            component_path=tmp_path / f"missing_{index}.component_step_v1.npz",
        )
        samples.append(sample)
    task = build_post_step_task(tuple(samples))
    fold = FoldInput(
        task_name=task.name,
        train_examples=task.examples,
        train_labels=task.labels,
        train_groups=task.groups,
        test_examples=task.examples,
        seed=1,
    )

    with pytest.raises(FileNotFoundError, match="missing_0.component_step_v1.npz"):
        create_method("component_resolved_hazard", None).fit_predict(fold)
