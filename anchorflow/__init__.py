"""AnchorFlow: causal prompt-anchor geometry for reasoning traces."""

from .anchor_repr import (
    AnchorBank,
    build_anchor_bank,
    char_span_to_token_span,
    prompt_span_vectors,
    select_prompt_hidden_layer,
)
from .hazard import (
    DiscreteHazardReadout,
    HazardTargets,
    discrete_hazard_nll,
    grouped_oof_hazard,
    make_first_error_hazard_targets,
)
from .intervention import (
    ReplayPlan,
    TextReplayPlan,
    apply_micro_replay,
    build_micro_replay,
    build_text_micro_replay,
)
from .lookback import compact_attention_lookback, compact_hidden_lookback
from .phase import (
    calibrate_chain_fpr_threshold,
    causal_boundary_events,
    causal_change_scores,
)
from .volume import (
    anchor_residual_cloud,
    anchor_subspace,
    conditional_gram_geometry,
    gram_features,
    gram_spectrum,
)

__all__ = [
    "AnchorBank",
    "DiscreteHazardReadout",
    "HazardTargets",
    "ReplayPlan",
    "TextReplayPlan",
    "anchor_residual_cloud",
    "anchor_subspace",
    "apply_micro_replay",
    "build_anchor_bank",
    "build_micro_replay",
    "build_text_micro_replay",
    "calibrate_chain_fpr_threshold",
    "causal_boundary_events",
    "causal_change_scores",
    "char_span_to_token_span",
    "compact_attention_lookback",
    "compact_hidden_lookback",
    "conditional_gram_geometry",
    "discrete_hazard_nll",
    "gram_features",
    "gram_spectrum",
    "grouped_oof_hazard",
    "make_first_error_hazard_targets",
    "prompt_span_vectors",
    "select_prompt_hidden_layer",
]
