"""Paper "Tracing Uncertainty in Language Model Reasoning" (arXiv:2605.07776),
TRACE channel only (target = next token), estimated at every token position.

Three uncertainty types (subsection 3.1), with the isotropic gradient-norm
estimator of Gruenefeld et al. (arXiv:2603.29466, Eq. 4 / 8):

  U_D(t) = -sum_v p(v|v_<t) log p(v|v_<t)          full-vocab predictive entropy
  U_C(t) = p(v_t|v_<t) * (1 - p(v_t|v_<t))         committal (Bernoulli variance)
  U_E(t) = || d/d_theta  p(v_t|v_<t) ||^2          epistemic = squared grad norm
                                                    over ALL params (Cov[theta]=I)

U_D and U_C come free from the forward logits (one pass). U_E requires one
BACKWARD pass per token position (the scalar p(v_t|v_<t) back-propagated to the
parameters), which is the expensive part; `ue_stride` subsamples token positions
and `ue_layers_from` optionally restricts which layers carry gradient (a
documented speed/memory approximation -- default None = all parameters, faithful
to the paper).

The answer channel (target = final answer y_hat) is intentionally NOT computed.
"""

from __future__ import annotations

import numpy as np
import torch


@torch.no_grad()
def _entropy_committal(logits, input_ids, a0: int, b1: int):
    """U_D (entropy) and U_C (committal) for response token positions a0..b1.

    logits: (seq, vocab) for the full (prompt+response) context.
    Token at position `pos` is predicted from logits[pos-1]; we evaluate every
    pos in [a0, b1]. Returns (U_D, U_C) float32 arrays of length (b1-a0+1).
    """
    pos = torch.arange(a0, b1 + 1, device=logits.device)
    sub = logits.index_select(0, pos - 1).float()           # (R, V)
    lp = torch.log_softmax(sub, dim=-1)
    p = lp.exp()
    U_D = -(p * lp).sum(-1)                                  # (R,)
    tgt = input_ids.index_select(0, pos)                    # actual next tokens
    p_tok = p.gather(-1, tgt.view(-1, 1)).squeeze(-1)
    U_C = p_tok * (1.0 - p_tok)
    return (U_D.detach().cpu().numpy().astype(np.float32),
            U_C.detach().cpu().numpy().astype(np.float32))


def set_ue_grad_scope(model, ue_layers_from):
    """Restrict which parameters carry gradient for the U_E backward.

    None  -> all parameters require grad (faithful to the paper, slowest).
    int L -> only parameters whose name contains '.layers.{i}.' with i >= L
             (plus the final norm / lm_head) require grad. Documented
             approximation that cuts backward time and the per-token grad-norm
             accumulation. Returns the list of params that require grad.
    """
    if ue_layers_from is None:
        for p in model.parameters():
            p.requires_grad_(True)
        return [p for p in model.parameters() if p.requires_grad]

    import re
    keep = []
    for name, p in model.named_parameters():
        m = re.search(r"\.layers\.(\d+)\.", name)
        if m is not None:
            on = int(m.group(1)) >= ue_layers_from
        else:
            # embeddings off; final norm / lm_head on
            on = ("lm_head" in name) or name.endswith("norm.weight") or ("model.norm" in name)
        p.requires_grad_(on)
        if on:
            keep.append(p)
    return keep


def epistemic_grad_norms(model, input_ids, attention_mask, a0: int, b1: int,
                         ue_params, ue_stride: int = 1):
    """U_E for response positions a0..b1 (optionally strided).

    One grad-enabled forward, then one backward per evaluated position on the
    scalar p(v_t|v_<t), accumulating the squared L2 norm of the gradient over
    `ue_params`. Returns (values, offsets) where `offsets` are the 0-based
    indices (relative to a0) at which U_E was evaluated, so trace_profile can
    place them at their true fractional positions.
    """
    enc = {"input_ids": input_ids.unsqueeze(0), "attention_mask": attention_mask.unsqueeze(0)}
    out = model(**enc)
    logits = out.logits[0]                                  # (seq, vocab), grad-enabled
    positions = list(range(a0, b1 + 1, max(1, ue_stride)))
    vals = np.full(len(positions), np.nan, dtype=np.float64)
    for i, pos in enumerate(positions):
        lp = torch.log_softmax(logits[pos - 1].float(), dim=-1)
        p_tok = lp[int(input_ids[pos])].exp()              # p(v_t | v_<t), scalar w/ grad
        model.zero_grad(set_to_none=True)
        retain = i < len(positions) - 1
        p_tok.backward(retain_graph=retain)
        total = 0.0
        for prm in ue_params:
            if prm.grad is not None:
                total += float(prm.grad.detach().double().pow(2).sum().item())
        vals[i] = total
    model.zero_grad(set_to_none=True)
    offsets = np.asarray([pos - a0 for pos in positions], dtype=np.int32)
    return vals, offsets


def trace_uncertainty(model, input_ids, attention_mask, a0: int, b1: int,
                      want_ue: bool = True, ue_params=None, ue_stride: int = 1):
    """Full trace-channel uncertainty for one teacher-forced chain.

    Returns dict with:
      U_D, U_C            : (R,) float32, R = b1-a0+1 response tokens
      U_E                 : (n_eval,) float64 or None
      U_E_offsets         : (n_eval,) int32 offsets relative to a0, or None
    """
    res = {"U_D": None, "U_C": None, "U_E": None, "U_E_offsets": None}
    if b1 < a0:
        return res

    if want_ue:
        # grad-enabled forward feeds both U_E (graph) and U_D/U_C (detached).
        enc = {"input_ids": input_ids.unsqueeze(0),
               "attention_mask": attention_mask.unsqueeze(0)}
        out = model(**enc)
        logits = out.logits[0]
        res["U_D"], res["U_C"] = _entropy_committal(
            logits.detach(), input_ids, a0, b1)
        positions = list(range(a0, b1 + 1, max(1, ue_stride)))
        vals = np.full(len(positions), np.nan, dtype=np.float64)
        for i, pos in enumerate(positions):
            lp = torch.log_softmax(logits[pos - 1].float(), dim=-1)
            p_tok = lp[int(input_ids[pos])].exp()
            model.zero_grad(set_to_none=True)
            p_tok.backward(retain_graph=(i < len(positions) - 1))
            total = 0.0
            for prm in ue_params:
                if prm.grad is not None:
                    total += float(prm.grad.detach().double().pow(2).sum().item())
            vals[i] = total
        model.zero_grad(set_to_none=True)
        res["U_E"] = vals
        res["U_E_offsets"] = np.asarray([p - a0 for p in positions], dtype=np.int32)
        del out, logits
    else:
        with torch.no_grad():
            enc = {"input_ids": input_ids.unsqueeze(0),
                   "attention_mask": attention_mask.unsqueeze(0)}
            logits = model(**enc).logits[0]
            res["U_D"], res["U_C"] = _entropy_committal(logits, input_ids, a0, b1)
    return res
