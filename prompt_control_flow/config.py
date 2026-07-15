from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from .replay_protocols import PROCESSBENCH_OBSERVER_CHAT_V1


@dataclass(frozen=True)
class ExtractionConfig:
    """Configuration for prompt-control residual-flow extraction."""

    layers: Tuple[int, ...] = (8, 10, 12, 14, 16, 18, 20, 22)
    subspace_k: int = 16
    prefix_k: int = 16
    random_seed: int = 13
    max_seq_len: int = 4096
    replay_protocol: str = PROCESSBENCH_OBSERVER_CHAT_V1
    response_separator: str = "\n\n"
    min_step_tokens: int = 1
    center_subspaces: bool = True
    include_entropy: bool = True
    store_step_vectors: bool = False
    store_step_state_vectors: bool = False
    store_flat_step_state_vectors: bool = True
    store_response_token_states: bool = False
    dtype: str = "auto"
    device: str = "auto"
    full_attention_token_threshold: int = 1200
    icr_top_k: int = 20
    icr_top_p: float | None = None


class MetricNames:
    PROMPT_FRAC = "prompt_frac"
    PREFIX_FRAC = "prefix_frac"
    RANDOM_FRAC = "random_frac"
    OFF_PROMPT = "off_prompt"
    QUESTION_FRAC = "question_frac"
    OFF_QUESTION = "off_question"
    PROMPT_CONTROL_RATIO = "prompt_control_ratio"
    PREFIX_LOCK_RATIO = "prefix_lock_ratio"
    TOKEN_ENTROPY = "token_entropy"
    TOKEN_ENTROPY_MAX = "token_entropy_max"
    TOKEN_ENTROPY_FIRST = "token_entropy_first"
    TOKEN_ENTROPY_LAST = "token_entropy_last"
    TOKEN_NLL = "token_nll"
    TOKEN_NLL_MAX = "token_nll_max"
    TOKEN_NLL_FIRST = "token_nll_first"
    TOKEN_NLL_LAST = "token_nll_last"
    TOKEN_CHOSEN_LOGPROB = "token_chosen_logprob"
    TOKEN_CHOSEN_LOGPROB_MIN = "token_chosen_logprob_min"
    TOKEN_MARGIN = "token_top1_top2_margin"
    TOKEN_MARGIN_MIN = "token_top1_top2_margin_min"
    TOKEN_TOPK_MASS = "token_top10_mass"
    TOKEN_TOPK_MASS_MIN = "token_top10_mass_min"
    STEP_LEN = "step_len"
    REL_POS = "rel_pos"
    ICR_MEAN = "icr_mean"
    ICR_MAX = "icr_max"
    ICR_TOP20_MEAN = "icr_top20_mean"
    GEOM_BOUNDARY_PROJ = "geom_boundary_proj"
    GEOM_HEALTHY_RESIDUAL = "geom_healthy_residual"
    GEOM_LID = "geom_lid"
    GEOM_KNN_ERROR_FRAC = "geom_knn_error_frac"
    GEOM_KNN_LABEL_ENTROPY = "geom_knn_label_entropy"
    GEOM_LOCAL_SPEC_ENTROPY = "geom_local_spec_entropy"
    GEOM_LAYER_NBR_INSTABILITY = "geom_layer_nbr_instability"
    GEOM_COMPARTMENT_SCORE = "geom_compartment_score"


STEP_METRIC_NAMES = (
    MetricNames.PROMPT_FRAC,
    MetricNames.PREFIX_FRAC,
    MetricNames.RANDOM_FRAC,
    MetricNames.OFF_PROMPT,
    MetricNames.QUESTION_FRAC,
    MetricNames.OFF_QUESTION,
    MetricNames.PROMPT_CONTROL_RATIO,
    MetricNames.PREFIX_LOCK_RATIO,
    MetricNames.TOKEN_ENTROPY,
    MetricNames.TOKEN_ENTROPY_MAX,
    MetricNames.TOKEN_ENTROPY_FIRST,
    MetricNames.TOKEN_ENTROPY_LAST,
    MetricNames.TOKEN_NLL,
    MetricNames.TOKEN_NLL_MAX,
    MetricNames.TOKEN_NLL_FIRST,
    MetricNames.TOKEN_NLL_LAST,
    MetricNames.TOKEN_CHOSEN_LOGPROB,
    MetricNames.TOKEN_CHOSEN_LOGPROB_MIN,
    MetricNames.TOKEN_MARGIN,
    MetricNames.TOKEN_MARGIN_MIN,
    MetricNames.TOKEN_TOPK_MASS,
    MetricNames.TOKEN_TOPK_MASS_MIN,
    MetricNames.STEP_LEN,
    MetricNames.REL_POS,
    MetricNames.ICR_MEAN,
    MetricNames.ICR_MAX,
    MetricNames.ICR_TOP20_MEAN,
    MetricNames.GEOM_BOUNDARY_PROJ,
    MetricNames.GEOM_HEALTHY_RESIDUAL,
    MetricNames.GEOM_LID,
    MetricNames.GEOM_KNN_ERROR_FRAC,
    MetricNames.GEOM_KNN_LABEL_ENTROPY,
    MetricNames.GEOM_LOCAL_SPEC_ENTROPY,
    MetricNames.GEOM_LAYER_NBR_INSTABILITY,
    MetricNames.GEOM_COMPARTMENT_SCORE,
)
