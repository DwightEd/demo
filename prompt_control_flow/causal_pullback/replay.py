from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch

from .schema import (
    BASELINE_STEP_FEATURE_NAMES,
    DIRECTION_NAMES,
    CausalPullbackConfig,
    FieldWitnesses,
)


EPS = 1e-8


class ReplayAlignmentError(RuntimeError):
    pass


@dataclass(frozen=True)
class PerturbationVariant:
    direction: int
    transition: int
    sign: int
    fraction: float


@dataclass
class PullbackReplayResult:
    replay_cosine: np.ndarray
    baseline_step_features: np.ndarray
    witness_norms: np.ndarray
    fisher_transfer: np.ndarray
    chosen_logprob_transfer: np.ndarray
    entropy_transfer: np.ndarray
    primary_half_fisher_transfer: np.ndarray
    perturbation_scale: np.ndarray
    metadata: dict[str, Any]


@dataclass
class ReplayStepStateResult:
    """Step states reconstructed on the observer's teacher-forcing trace."""

    step_states: np.ndarray
    source_cosine: np.ndarray
    metadata: dict[str, Any]


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as exc:  # pragma: no cover
        raise ValueError("model has no parameters") from exc


def _autocast_context(device: torch.device, model):
    dtype = next(model.parameters()).dtype
    enabled = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def locate_causal_components(model):
    prefix = str(getattr(model, "base_model_prefix", "") or "")
    backbone = getattr(model, prefix, None) if prefix else None
    if backbone is None or backbone is model:
        backbone = getattr(model, "model", None)
    if backbone is None or backbone is model:
        backbone = getattr(model, "transformer", None)
    if backbone is None or backbone is model:
        raise TypeError("cannot locate causal-LM backbone")
    layers = getattr(backbone, "layers", None)
    if layers is None and getattr(backbone, "decoder", None) is not None:
        layers = getattr(backbone.decoder, "layers", None)
    if layers is None:
        layers = getattr(backbone, "h", None)
    if layers is None:
        raise TypeError("cannot locate decoder layer sequence")
    output_head = model.get_output_embeddings()
    if output_head is None:
        raise TypeError("model.get_output_embeddings() returned None")
    return backbone, layers, output_head


def _step_exp_pool(hidden: torch.Tensor, ranges: Sequence[tuple[int, int]]) -> torch.Tensor:
    rows = []
    for start, stop in ranges:
        cloud = hidden[int(start) : int(stop) + 1]
        if cloud.shape[0] == 1:
            rows.append(cloud[0])
            continue
        position = torch.linspace(
            0.0,
            1.0,
            int(cloud.shape[0]),
            dtype=torch.float32,
            device=cloud.device,
        )
        weight = torch.softmax(position, dim=0).to(dtype=cloud.dtype)
        rows.append(torch.sum(cloud * weight[:, None], dim=0))
    return torch.stack(rows)


def _cosine_rows(first: torch.Tensor, second: torch.Tensor) -> np.ndarray:
    if first.ndim != 2 or second.ndim != 2 or first.shape != second.shape:
        raise ReplayAlignmentError(
            "replay/stored step-state shape mismatch: "
            f"replay={tuple(first.shape)}, stored={tuple(second.shape)}. "
            "Residual interventions require raw sv_clouds in the model hidden "
            "dimension; projected sv_vec_* coordinates are invalid."
        )
    first = first.float()
    second = second.float()
    numerator = torch.sum(first * second, dim=-1)
    denominator = first.norm(dim=-1) * second.norm(dim=-1)
    value = numerator / denominator.clamp_min(EPS)
    return value.detach().cpu().numpy().astype(np.float32)


def _canonical_hidden_layout(
    hidden: torch.Tensor,
    *,
    batch: int,
    sequence: int,
    width: int,
    context: str,
) -> tuple[torch.Tensor, str]:
    """Return hidden states as [batch, sequence, width] with an explicit layout."""

    if hidden.ndim != 3 or int(hidden.shape[0]) != int(batch):
        raise ReplayAlignmentError(
            f"{context}: expected rank-3 hidden batch {batch}, got {tuple(hidden.shape)}"
        )
    if tuple(hidden.shape[1:]) == (int(sequence), int(width)):
        return hidden, "batch_sequence_hidden"
    if tuple(hidden.shape[1:]) == (int(width), int(sequence)):
        return hidden.transpose(1, 2), "batch_hidden_sequence"
    raise ReplayAlignmentError(
        f"{context}: cannot identify hidden layout {tuple(hidden.shape)} using "
        f"sequence={sequence}, hidden_width={width}"
    )


def _prediction_axis(
    input_ids: torch.Tensor,
    ranges: Sequence[tuple[int, int]],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    positions = []
    steps = []
    for step, (start, stop) in enumerate(ranges):
        if int(start) <= 0:
            raise ValueError("causal scoring requires every target token position > 0")
        local = torch.arange(
            int(start), int(stop) + 1, dtype=torch.long, device=input_ids.device
        )
        positions.append(local)
        steps.append(torch.full_like(local, int(step)))
    target_positions = torch.cat(positions)
    return target_positions - 1, input_ids[target_positions], torch.cat(steps)


def _apply_output_head(model, output_head, hidden: torch.Tensor) -> torch.Tensor:
    logits = output_head(hidden)
    bias = getattr(model, "final_logits_bias", None)
    if bias is not None:
        logits = logits + bias.to(device=logits.device, dtype=logits.dtype)
    return logits


def _baseline_log_probs(
    model,
    output_head,
    final_hidden: torch.Tensor,
    prediction_positions: torch.Tensor,
    *,
    token_chunk: int,
) -> torch.Tensor:
    chunks = []
    for start in range(0, len(prediction_positions), int(token_chunk)):
        stop = min(len(prediction_positions), start + int(token_chunk))
        selected = final_hidden.index_select(
            0, prediction_positions[start:stop]
        )
        logits = _apply_output_head(model, output_head, selected)
        chunks.append(torch.log_softmax(logits.float(), dim=-1))
        del selected, logits
    return torch.cat(chunks, dim=0)


def _scatter_step_mean(
    values: torch.Tensor,
    token_steps: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    if values.ndim == 1:
        values = values[None, :]
    output = torch.zeros(
        (values.shape[0], int(n_steps)),
        dtype=values.dtype,
        device=values.device,
    )
    index = token_steps[None, :].expand(values.shape[0], -1)
    output.scatter_add_(1, index, values)
    count = torch.bincount(token_steps, minlength=int(n_steps)).to(values.dtype)
    return output / count.clamp_min(1.0)[None, :]


def _baseline_step_features(
    baseline_logp: torch.Tensor,
    target_ids: torch.Tensor,
    token_steps: torch.Tensor,
    n_steps: int,
) -> np.ndarray:
    probability = baseline_logp.exp()
    entropy = -torch.sum(probability * baseline_logp, dim=-1)
    chosen_nll = -baseline_logp.gather(1, target_ids[:, None]).squeeze(1)
    top = torch.topk(baseline_logp, k=2, dim=-1).values
    margin = top[:, 0] - top[:, 1]
    top1 = top[:, 0].exp()
    token = torch.stack([entropy, chosen_nll, margin, top1], dim=-1)
    if token.shape[1] != len(BASELINE_STEP_FEATURE_NAMES):
        raise RuntimeError("baseline feature schema drift")
    features = []
    for column in range(token.shape[1]):
        features.append(
            _scatter_step_mean(token[:, column], token_steps, n_steps)[0]
        )
    return torch.stack(features, dim=-1).detach().cpu().numpy().astype(np.float32)


def _compare_variant_hidden(
    model,
    output_head,
    final_hidden: torch.Tensor,
    prediction_positions: torch.Tensor,
    target_ids: torch.Tensor,
    token_steps: torch.Tensor,
    baseline_logp: torch.Tensor,
    *,
    n_steps: int,
    token_chunk: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = int(final_hidden.shape[0])
    kl_sum = torch.zeros((batch, n_steps), dtype=torch.float32, device=final_hidden.device)
    chosen_sum = torch.zeros_like(kl_sum)
    entropy_sum = torch.zeros_like(kl_sum)
    counts = torch.zeros(n_steps, dtype=torch.float32, device=final_hidden.device)
    for start in range(0, len(prediction_positions), int(token_chunk)):
        stop = min(len(prediction_positions), start + int(token_chunk))
        positions = prediction_positions[start:stop]
        selected = final_hidden.index_select(1, positions)
        logits = _apply_output_head(model, output_head, selected)
        logq = torch.log_softmax(logits.float(), dim=-1)
        logp = baseline_logp[start:stop]
        probability = logp.exp()
        kl = torch.sum(
            probability[None, :, :] * (logp[None, :, :] - logq), dim=-1
        )
        ids = target_ids[start:stop]
        chosen = logq.gather(
            2, ids[None, :, None].expand(batch, -1, 1)
        ).squeeze(-1)
        entropy = -torch.sum(logq.exp() * logq, dim=-1)
        local_steps = token_steps[start:stop]
        index = local_steps[None, :].expand(batch, -1)
        kl_sum.scatter_add_(1, index, kl)
        chosen_sum.scatter_add_(1, index, chosen)
        entropy_sum.scatter_add_(1, index, entropy)
        counts.scatter_add_(
            0,
            local_steps,
            torch.ones_like(local_steps, dtype=torch.float32),
        )
        del selected, logits, logq, probability, kl, chosen, entropy
    denominator = counts.clamp_min(1.0)[None, :]
    return tuple(
        value.detach().cpu().numpy().astype(np.float32)
        for value in (kl_sum / denominator, chosen_sum / denominator, entropy_sum / denominator)
    )


def _run_perturbed_backbone(
    backbone,
    layer_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    variants: Sequence[PerturbationVariant],
    directions: torch.Tensor,
    source_positions: torch.Tensor,
    base_scale: torch.Tensor,
    *,
    model,
    hidden_width: int,
) -> torch.Tensor:
    batch = len(variants)
    variant_direction = torch.stack(
        [directions[v.direction, v.transition] for v in variants]
    )
    variant_position = torch.stack(
        [source_positions[v.transition] for v in variants]
    ).long()
    variant_amplitude = torch.stack(
        [
            base_scale[v.transition]
            * float(v.sign)
            * float(v.fraction)
            for v in variants
        ]
    )

    def hook(_module, _inputs, output):
        raw_hidden = output[0] if isinstance(output, tuple) else output
        hidden, layout = _canonical_hidden_layout(
            raw_hidden,
            batch=batch,
            sequence=int(input_ids.numel()),
            width=int(hidden_width),
            context="perturbed decoder hook",
        )
        hidden = hidden.clone()
        row = torch.arange(batch, device=hidden.device)
        hidden[row, variant_position.to(hidden.device)] += (
            variant_direction.to(hidden.device, dtype=hidden.dtype)
            * variant_amplitude.to(hidden.device, dtype=hidden.dtype)[:, None]
        )
        restored = (
            hidden.transpose(1, 2)
            if layout == "batch_hidden_sequence"
            else hidden
        )
        if isinstance(output, tuple):
            return (restored,) + output[1:]
        return restored

    handle = layer_module.register_forward_hook(hook)
    try:
        with _autocast_context(input_ids.device, model):
            output = backbone(
                input_ids=input_ids[None, :].expand(batch, -1),
                attention_mask=attention_mask[None, :].expand(batch, -1),
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
        final_hidden, _ = _canonical_hidden_layout(
            output.last_hidden_state,
            batch=batch,
            sequence=int(input_ids.numel()),
            width=int(hidden_width),
            context="perturbed backbone output",
        )
        return final_hidden
    finally:
        handle.remove()


def _run_baseline_backbone(
    backbone,
    layer_module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    model,
    hidden_width: int,
) -> tuple[Any, torch.Tensor, str, torch.Tensor, str]:
    captured: list[tuple[torch.Tensor, str]] = []

    def capture(_module, _inputs, output):
        raw_hidden = output[0] if isinstance(output, tuple) else output
        captured.append(
            _canonical_hidden_layout(
                raw_hidden,
                batch=1,
                sequence=int(input_ids.numel()),
                width=int(hidden_width),
                context="baseline decoder hook",
            )
        )

    handle = layer_module.register_forward_hook(capture)
    try:
        with _autocast_context(input_ids.device, model):
            output = backbone(
                input_ids=input_ids[None, :],
                attention_mask=attention_mask[None, :],
                use_cache=False,
                output_hidden_states=False,
                return_dict=True,
            )
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError(
            f"target decoder hook fired {len(captured)} times during baseline replay"
        )
    final_hidden, final_layout = _canonical_hidden_layout(
        output.last_hidden_state,
        batch=1,
        sequence=int(input_ids.numel()),
        width=int(hidden_width),
        context="baseline backbone output",
    )
    captured_hidden, captured_layout = captured[0]
    return output, captured_hidden, captured_layout, final_hidden, final_layout


@torch.inference_mode()
def replay_step_states(
    model,
    trace: dict[str, Any],
    source_step_states: np.ndarray,
    cfg: CausalPullbackConfig,
) -> ReplayStepStateResult:
    """Reconstruct pooled step states without treating legacy states as coordinates.

    Legacy artifacts can omit exact token IDs. Re-tokenizing their text may then
    produce a nearby, but not identical, hidden trajectory. This function reports
    that source drift while returning states that live on the exact observer trace
    used by subsequent interventions.
    """

    cfg.validate()
    device = _model_device(model)
    input_ids = torch.as_tensor(trace["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        trace["attention_mask"], dtype=torch.long, device=device
    )
    ranges = [(int(a), int(b)) for a, b in trace["step_token_ranges"]]
    source_array = np.asarray(source_step_states)
    if source_array.ndim != 2:
        raise ReplayAlignmentError(
            f"source step states must be [step, hidden], got {source_array.shape}"
        )
    if len(ranges) != int(source_array.shape[0]):
        raise ReplayAlignmentError(
            f"replay has {len(ranges)} steps but source trajectory has "
            f"{source_array.shape[0]}"
        )
    hidden_width = int(source_array.shape[-1])
    backbone, layers, _ = locate_causal_components(model)
    if cfg.layer > len(layers):
        raise ValueError(
            f"hidden-state layer {cfg.layer} exceeds decoder depth {len(layers)}"
        )
    layer_module = layers[cfg.layer - 1]
    (
        baseline,
        captured_layer,
        captured_layout,
        baseline_final,
        final_layout,
    ) = _run_baseline_backbone(
        backbone,
        layer_module,
        input_ids,
        attention_mask,
        model=model,
        hidden_width=hidden_width,
    )
    replay_states = _step_exp_pool(captured_layer[0], ranges)
    source = torch.as_tensor(
        source_array, dtype=replay_states.dtype, device=device
    )
    source_cosine = _cosine_rows(replay_states, source)
    result = ReplayStepStateResult(
        step_states=replay_states.float().cpu().numpy().astype(np.float32),
        source_cosine=source_cosine,
        metadata={
            "median_source_replay_cosine": float(np.nanmedian(source_cosine)),
            "minimum_source_replay_cosine": float(np.nanmin(source_cosine)),
            "captured_layer_layout": captured_layout,
            "final_hidden_layout": final_layout,
            "hidden_width": hidden_width,
            "sequence_tokens": int(input_ids.numel()),
        },
    )
    del baseline, baseline_final, captured_layer, replay_states, source
    return result


@torch.inference_mode()
def compute_causal_pullback(
    model,
    trace: dict[str, Any],
    stored_step_states: np.ndarray,
    witnesses: FieldWitnesses,
    cfg: CausalPullbackConfig,
) -> PullbackReplayResult:
    """Estimate a causal step-to-output Fisher observability operator.

    A field-normal direction is injected at the source step endpoint.  Central
    finite differences measure its influence on every future output step.  KL
    curvature estimates the categorical-Fisher pullback without materializing
    a vocabulary-sized Jacobian.
    """

    cfg.validate()
    device = _model_device(model)
    input_ids = torch.as_tensor(trace["input_ids"], dtype=torch.long, device=device)
    attention_mask = torch.as_tensor(
        trace["attention_mask"], dtype=torch.long, device=device
    )
    ranges = [(int(a), int(b)) for a, b in trace["step_token_ranges"]]
    n_steps = len(ranges)
    stored_array = np.asarray(stored_step_states)
    if stored_array.ndim != 2:
        raise ReplayAlignmentError(
            f"stored step states must be [step, hidden], got {stored_array.shape}"
        )
    if n_steps != int(stored_array.shape[0]):
        raise ReplayAlignmentError(
            f"replay has {n_steps} steps but stored trajectory has "
            f"{stored_array.shape[0]}"
        )
    hidden_width = int(stored_array.shape[-1])
    witnesses.validate(hidden_width)
    if witnesses.n_transitions != n_steps - 1:
        raise ReplayAlignmentError("witness transition count does not match replay steps")

    backbone, layers, output_head = locate_causal_components(model)
    if cfg.layer > len(layers):
        raise ValueError(
            f"hidden-state layer {cfg.layer} exceeds decoder depth {len(layers)}"
        )
    layer_module = layers[cfg.layer - 1]
    (
        baseline,
        captured_layer,
        captured_layout,
        baseline_final,
        final_layout,
    ) = _run_baseline_backbone(
        backbone,
        layer_module,
        input_ids,
        attention_mask,
        model=model,
        hidden_width=hidden_width,
    )
    layer_hidden = captured_layer[0]
    final_hidden = baseline_final[0]
    replay_step_states = _step_exp_pool(layer_hidden, ranges)
    stored = torch.as_tensor(
        stored_step_states, dtype=replay_step_states.dtype, device=device
    )
    replay_cosine = _cosine_rows(replay_step_states, stored)
    median_cosine = float(np.nanmedian(replay_cosine))
    if median_cosine < cfg.replay_cosine_threshold:
        raise ReplayAlignmentError(
            f"median replay/stored step-vector cosine {median_cosine:.6f} is below "
            f"threshold {cfg.replay_cosine_threshold:.6f}"
        )

    prediction_positions, target_ids, token_steps = _prediction_axis(input_ids, ranges)
    baseline_logp = _baseline_log_probs(
        model,
        output_head,
        final_hidden,
        prediction_positions,
        token_chunk=cfg.logit_token_chunk,
    )
    baseline_features = _baseline_step_features(
        baseline_logp, target_ids, token_steps, n_steps
    )

    direction_array = np.stack(
        [
            witnesses.field_direction,
            witnesses.shuffle_direction,
            witnesses.random_direction,
        ],
        axis=0,
    )
    directions = torch.as_tensor(direction_array, dtype=torch.float32, device=device)
    # Transition ``t`` is observed only after step ``t + 1`` has completed.
    # Injecting it at step ``t`` would leak the future displacement into the
    # past, so the causal intervention lives at the transition destination.
    source_positions = torch.as_tensor(
        [ranges[transition + 1][1] for transition in range(n_steps - 1)],
        dtype=torch.long,
        device=device,
    )
    source_hidden = layer_hidden.index_select(0, source_positions)
    base_scale = cfg.epsilon_fraction * source_hidden.float().norm(dim=-1)

    variants: list[PerturbationVariant] = []
    # The final observed transition has no downstream reasoning step on which
    # to measure transport. It remains in the stored field statistics, while
    # its causal-operator row is deliberately all missing.
    for transition in range(max(0, n_steps - 2)):
        for direction in range(len(DIRECTION_NAMES)):
            vector_norm = float(torch.linalg.vector_norm(directions[direction, transition]))
            if vector_norm <= EPS:
                continue
            variants.extend(
                [
                    PerturbationVariant(direction, transition, +1, 1.0),
                    PerturbationVariant(direction, transition, -1, 1.0),
                ]
            )
        if cfg.linearity_half_step and float(
            torch.linalg.vector_norm(directions[0, transition])
        ) > EPS:
            variants.extend(
                [
                    PerturbationVariant(0, transition, +1, 0.5),
                    PerturbationVariant(0, transition, -1, 0.5),
                ]
            )

    measurements: dict[tuple[int, int, int, float], tuple[np.ndarray, ...]] = {}
    for start in range(0, len(variants), cfg.variant_batch_size):
        local = variants[start : start + cfg.variant_batch_size]
        perturbed_hidden = _run_perturbed_backbone(
            backbone,
            layer_module,
            input_ids,
            attention_mask,
            local,
            directions,
            source_positions,
            base_scale,
            model=model,
            hidden_width=hidden_width,
        )
        compared = _compare_variant_hidden(
            model,
            output_head,
            perturbed_hidden,
            prediction_positions,
            target_ids,
            token_steps,
            baseline_logp,
            n_steps=n_steps,
            token_chunk=cfg.logit_token_chunk,
        )
        for row, variant in enumerate(local):
            measurements[
                (variant.direction, variant.transition, variant.sign, variant.fraction)
            ] = tuple(value[row] for value in compared)
        del perturbed_hidden

    shape = (len(DIRECTION_NAMES), n_steps - 1, n_steps)
    fisher = np.full(shape, np.nan, dtype=np.float32)
    chosen = np.full(shape, np.nan, dtype=np.float32)
    entropy = np.full(shape, np.nan, dtype=np.float32)
    half_fisher = np.full((n_steps - 1, n_steps), np.nan, dtype=np.float32)
    acausal_kl = []
    for direction in range(len(DIRECTION_NAMES)):
        for transition in range(n_steps - 1):
            plus = measurements.get((direction, transition, +1, 1.0))
            minus = measurements.get((direction, transition, -1, 1.0))
            if plus is None or minus is None:
                continue
            alpha = float(base_scale[transition].item())
            if alpha <= EPS:
                continue
            local_fisher = (plus[0] + minus[0]) / (alpha * alpha)
            local_chosen = (plus[1] - minus[1]) / (2.0 * alpha)
            local_entropy = (plus[2] - minus[2]) / (2.0 * alpha)
            source_step = transition + 1
            acausal_kl.extend(local_fisher[: source_step + 1].tolist())
            local_fisher[: source_step + 1] = np.nan
            local_chosen[: source_step + 1] = np.nan
            local_entropy[: source_step + 1] = np.nan
            fisher[direction, transition] = local_fisher
            chosen[direction, transition] = local_chosen
            entropy[direction, transition] = local_entropy
            if direction == 0 and cfg.linearity_half_step:
                half_plus = measurements.get((0, transition, +1, 0.5))
                half_minus = measurements.get((0, transition, -1, 0.5))
                if half_plus is not None and half_minus is not None:
                    half_alpha = 0.5 * alpha
                    value = (half_plus[0] + half_minus[0]) / (
                        half_alpha * half_alpha
                    )
                    value[: source_step + 1] = np.nan
                    half_fisher[transition] = value

    witness_norms = np.stack(
        [
            witnesses.field_witness_norm,
            witnesses.shuffle_witness_norm,
            witnesses.field_witness_norm,
        ],
        axis=0,
    ).astype(np.float32)
    metadata = {
        "median_replay_cosine": median_cosine,
        "minimum_replay_cosine": float(np.nanmin(replay_cosine)),
        "response_token_count": int(len(prediction_positions)),
        "variant_count": len(variants),
        "maximum_acausal_fisher_leakage": (
            float(np.nanmax(np.abs(acausal_kl))) if acausal_kl else 0.0
        ),
        "fisher_estimator": "central_kl_curvature",
        "injection_location": "observed_transition_destination_endpoint_after_decoder_layer",
        "causal_timing": (
            "transition_t_is_observed_at_step_t_plus_1_and_scores_only_steps_after_t_plus_1"
        ),
        "source_hidden_state_index": int(cfg.layer),
        "captured_layer_layout": captured_layout,
        "final_hidden_layout": final_layout,
        "intervention_hidden_width": hidden_width,
        "alignment_state_source": "caller_supplied_replay_native_step_states",
    }
    del (
        baseline,
        baseline_final,
        captured_layer,
        layer_hidden,
        final_hidden,
        baseline_logp,
    )
    return PullbackReplayResult(
        replay_cosine=replay_cosine,
        baseline_step_features=baseline_features,
        witness_norms=witness_norms,
        fisher_transfer=fisher,
        chosen_logprob_transfer=chosen,
        entropy_transfer=entropy,
        primary_half_fisher_transfer=half_fisher,
        perturbation_scale=base_scale.detach().cpu().numpy().astype(np.float32),
        metadata=metadata,
    )
