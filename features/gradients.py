"""Gradient spectral field -- per-step, per-layer parameter-gradient norms.

For reasoning step j with token range [a,b], the step loss is the summed NLL
    L_j = - sum_{t in [a,b]} log p(y_t | y_<t),
and one backward pass gives the gradient w.r.t. every parameter. We aggregate
the squared gradient norm BY TRANSFORMER BLOCK:
    g_j^(l) = sum over params in block l of ||grad||^2      (the layer profile)
    ||g_j||^2 = sum over all params                          (signal 1, step grad norm)

Why gradients (vs activations): the activation layer profile is trivially
cross-layer-correlated through the residual-stream identity path. Parameter
gradients of different blocks share NO identity path -- block l's gradient is
that block's responsibility for the step -- so cross-layer gradient structure is
pure functional coupling. The "spectral field" hypothesis is mechanistically far
cleaner here, and layer l* of an anomalous gradient literally means "the layer
that most needs correction".

Cost: grad-enabled forward + one backward per step (retain_graph). Full-model
gradients for an 8B model are ~param-size (needs ~32GB+ / H100). Restrict with
`grad_from_layer` for smaller GPUs (a partial profile).
"""

from __future__ import annotations

import re

import numpy as np
import torch


def build_block_map(model, grad_from_layer=None):
    """Return (block_of_param: list aligned with grad-params, n_blocks, params).

    Sets requires_grad: all blocks (default) or only blocks >= grad_from_layer
    (+ final norm / lm_head) to bound memory. params = the grad-carrying params.
    """
    nb = model.config.num_hidden_layers
    block_of, params = [], []
    for name, p in model.named_parameters():
        m = re.search(r"\.layers\.(\d+)\.", name)
        blk = int(m.group(1)) if m else -1
        on = True if grad_from_layer is None else (
            blk >= grad_from_layer if blk >= 0
            else ("lm_head" in name or name.endswith("norm.weight") or "model.norm" in name))
        p.requires_grad_(on)
        if on:
            params.append(p); block_of.append(blk)
    return np.asarray(block_of, dtype=np.int64), nb, params


def step_gradient_profile(model, input_ids, attn, safe, block_of, n_blocks, params):
    """Return (gradprof (T, n_blocks) float32, grad_total (T,) float64).

    safe = list of (a, b) inclusive token ranges per kept step. One grad-enabled
    forward, one backward per step on the step's summed NLL.
    """
    enc = {"input_ids": input_ids.unsqueeze(0), "attention_mask": attn.unsqueeze(0)}
    out = model(**enc)
    logits = out.logits[0]                                   # (seq, vocab), grad-enabled
    T = len(safe)
    gp = np.zeros((T, n_blocks), dtype=np.float64)
    gtot = np.zeros(T, dtype=np.float64)
    for j, (a, b) in enumerate(safe):
        pos = torch.arange(a, b + 1, device=logits.device)
        lp = torch.log_softmax(logits.index_select(0, pos - 1).float(), dim=-1)
        tgt = input_ids.index_select(0, pos)
        L_j = -lp.gather(-1, tgt.view(-1, 1)).sum()         # summed NLL over the step
        model.zero_grad(set_to_none=True)
        L_j.backward(retain_graph=(j < T - 1))
        blk = np.zeros(n_blocks, dtype=np.float64); tot = 0.0
        for prm, bi in zip(params, block_of):
            if prm.grad is not None:
                s = float(prm.grad.detach().double().pow(2).sum().item())
                tot += s
                if bi >= 0:
                    blk[bi] += s
        gp[j] = blk; gtot[j] = tot
    model.zero_grad(set_to_none=True)
    del out, logits
    return gp.astype(np.float32), gtot
