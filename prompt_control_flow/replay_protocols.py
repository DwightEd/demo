from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Sequence


PROCESSBENCH_OBSERVER_CHAT_V1 = "processbench_observer_chat_v1"
PROCESSBENCH_OBSERVER_PLAIN_V1 = "processbench_observer_plain_v1"
EXACT_ARTIFACT_REPLAY = "exact_artifact_replay"
STORED_RENDERED_PROMPT_REPLAY = "stored_rendered_prompt_replay"

SUPPORTED_OBSERVER_PROTOCOLS = (
    PROCESSBENCH_OBSERVER_CHAT_V1,
    PROCESSBENCH_OBSERVER_PLAIN_V1,
)

PROCESSBENCH_SOLVER_INSTRUCTION = (
    "Solve the following problem carefully. Show the reasoning step by step, "
    "and do not skip intermediate calculations."
)


@dataclass(frozen=True)
class RenderedReplayPrompt:
    """Rendered observer context and its provenance.

    ``rendered_prompt`` is the exact text tokenized before the fixed candidate
    continuation. ``question_char_span`` is half-open on that rendered text.
    """

    protocol: str
    rendered_prompt: str
    messages_json: str
    question_char_span: tuple[int, int]
    provenance: str
    rendered_prompt_sha256: str


def normalize_problem_text(problem: str) -> str:
    """Normalize only for grouping; never use this text for model replay."""

    return " ".join(str(problem).split())


def stable_problem_group_id(problem: str) -> str:
    normalized = normalize_problem_text(problem)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return f"problem_sha256:{digest}"


def build_observer_response(
    steps: Sequence[str],
    *,
    separator: str = "\n\n",
) -> str:
    values = [str(step) for step in steps]
    if not values or any(not value.strip() for value in values):
        raise ValueError("reasoning steps must be non-empty strings")
    return str(separator).join(values)


def render_processbench_observer_prompt(
    tokenizer,
    problem: str,
    *,
    protocol: str = PROCESSBENCH_OBSERVER_CHAT_V1,
) -> RenderedReplayPrompt:
    """Render a frozen ProcessBench observer prompt.

    The chat protocol is the primary observer estimand for instruction-tuned
    models. The plain protocol exists only to reproduce legacy artifacts.
    Neither protocol claims to recover the unknown original-generator prompt.
    """

    question = str(problem)
    if not question.strip():
        raise ValueError("ProcessBench problem text is empty")

    if protocol == PROCESSBENCH_OBSERVER_CHAT_V1:
        content = f"{PROCESSBENCH_SOLVER_INSTRUCTION}\n\nProblem: {question}"
        messages = [{"role": "user", "content": content}]
        rendered = str(
            tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        )
        provenance = "frozen_solver_chat_template_observer_prompt"
    elif protocol == PROCESSBENCH_OBSERVER_PLAIN_V1:
        rendered = f"Problem: {question}\n\nSolution:\n\n"
        messages = []
        provenance = "legacy_fixed_plain_problem_solution_observer_prompt"
    else:
        raise ValueError(
            f"unsupported observer replay protocol {protocol!r}; "
            f"choose one of {SUPPORTED_OBSERVER_PROTOCOLS}"
        )

    question_start = rendered.rfind(question)
    if question_start < 0:
        raise ValueError("rendered observer prompt does not contain the target problem")
    question_span = (question_start, question_start + len(question))
    messages_json = json.dumps(messages, ensure_ascii=False, sort_keys=True)
    prompt_hash = hashlib.sha256(rendered.encode("utf-8")).hexdigest()
    return RenderedReplayPrompt(
        protocol=str(protocol),
        rendered_prompt=rendered,
        messages_json=messages_json,
        question_char_span=question_span,
        provenance=provenance,
        rendered_prompt_sha256=prompt_hash,
    )
