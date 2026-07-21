"""Memory-bounded exact attention capture for decoder-only HF models."""

from __future__ import annotations

import inspect
from typing import Sequence


def find_decoder_layers(model):
    """Find the ordered decoder blocks for common decoder-only architectures."""

    paths = (
        ("layers",),
        ("model", "layers"),
        ("transformer", "h"),
        ("model", "transformer", "h"),
        ("gpt_neox", "layers"),
        ("model", "gpt_neox", "layers"),
        ("decoder", "layers"),
        ("model", "decoder", "layers"),
    )
    for path in paths:
        value = model
        for part in path:
            value = getattr(value, part, None)
            if value is None:
                break
        if value is None:
            continue
        try:
            count = len(value)
        except TypeError:
            continue
        if count > 0 and all(
            hasattr(value[index], "register_forward_pre_hook")
            for index in range(count)
        ):
            return value, ".".join(path)
    return None, None


def resolve_attention_modules(model, attention_layers: Sequence[int]):
    """Resolve requested self-attention modules and their auditable model path."""

    if len(attention_layers) == 0:
        raise ValueError("attention_layers cannot be empty")
    decoder_layers, decoder_path = find_decoder_layers(model)
    if decoder_layers is None:
        return None, None
    if any(not 0 <= int(index) < len(decoder_layers) for index in attention_layers):
        raise ValueError("attention layer is outside the discovered decoder stack")
    modules = {
        int(layer_index): getattr(decoder_layers[int(layer_index)], "self_attn", None)
        for layer_index in attention_layers
    }
    if any(
        module is None
        or not hasattr(module, "register_forward_pre_hook")
        or not hasattr(module, "register_forward_hook")
        for module in modules.values()
    ):
        return None, None
    return modules, f"{decoder_path}.self_attn"


def _replace_forward_argument(module, args, kwargs, name: str, value):
    """Replace a named forward argument whether the caller used args or kwargs."""

    updated_args = list(args)
    updated_kwargs = dict(kwargs)
    if name in updated_kwargs:
        updated_kwargs[name] = value
        return tuple(updated_args), updated_kwargs
    try:
        parameters = list(inspect.signature(module.forward).parameters.values())
    except (TypeError, ValueError):
        parameters = []
    positional_names = [
        parameter.name
        for parameter in parameters
        if parameter.kind
        in (parameter.POSITIONAL_ONLY, parameter.POSITIONAL_OR_KEYWORD)
    ]
    if name in positional_names:
        position = positional_names.index(name)
        if position < len(updated_args):
            updated_args[position] = value
            return tuple(updated_args), updated_kwargs
    accepts_var_kwargs = any(
        parameter.kind == parameter.VAR_KEYWORD for parameter in parameters
    )
    if accepts_var_kwargs or name in {parameter.name for parameter in parameters}:
        updated_kwargs[name] = value
        return tuple(updated_args), updated_kwargs
    raise RuntimeError(
        f"attention module {type(module).__name__} has no {name!r} forward argument; "
        "selected-layer attention capture is unsupported for this architecture"
    )


def full_forward_with_selected_attention(
    model,
    torch,
    *,
    input_ids,
    attention_mask,
    attention_layers: Sequence[int],
    attention_heads: Sequence[int],
    storage_torch_dtype,
    want_hidden: bool,
):
    """Run one exact forward while retaining only requested attention blocks."""

    if len(attention_heads) == 0 or any(int(head) < 0 for head in attention_heads):
        raise ValueError("attention_heads must contain non-negative indices")
    attention_modules, attention_path = resolve_attention_modules(
        model, attention_layers
    )
    if attention_modules is None:
        return None, None, None

    captured = {}
    handles = []

    def pre_hook(module, args, kwargs):
        return _replace_forward_argument(
            module, args, kwargs, "output_attentions", True
        )

    def make_post_hook(layer_index: int):
        def post_hook(module, args, kwargs, output):
            del module, args, kwargs
            if not isinstance(output, (tuple, list)) or len(output) < 2:
                raise RuntimeError(
                    f"attention module at layer {layer_index} did not return weights "
                    "after output_attentions=True"
                )
            weights = output[1]
            if weights is None or getattr(weights, "ndim", 0) != 4:
                raise RuntimeError(
                    f"attention module at layer {layer_index} returned malformed weights"
                )
            if int(weights.shape[0]) != 1:
                raise RuntimeError("attention extraction expects batch size one")
            if max(int(head) for head in attention_heads) >= int(weights.shape[1]):
                raise RuntimeError(
                    f"requested attention head is outside layer {layer_index} output"
                )
            captured[layer_index] = (
                weights[0, list(attention_heads)]
                .detach()
                .to("cpu", dtype=storage_torch_dtype)
            )

        return post_hook

    try:
        for layer_index in attention_layers:
            attention_module = attention_modules[int(layer_index)]
            handles.append(
                attention_module.register_forward_pre_hook(
                    pre_hook, with_kwargs=True
                )
            )
            handles.append(
                attention_module.register_forward_hook(
                    make_post_hook(int(layer_index)), with_kwargs=True
                )
            )
        with torch.inference_mode():
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=False,
                output_hidden_states=want_hidden,
                use_cache=False,
                return_dict=True,
            )
    finally:
        for handle in handles:
            handle.remove()

    missing = [int(layer) for layer in attention_layers if int(layer) not in captured]
    if missing:
        raise RuntimeError(f"selected attention hooks did not run for layers {missing}")
    ordered_attention = [captured[int(layer)] for layer in attention_layers]
    selected_attention = (
        ordered_attention[0].unsqueeze(0)
        if len(ordered_attention) == 1
        else torch.stack(ordered_attention, dim=0)
    ).numpy()
    return output, selected_attention, attention_path
