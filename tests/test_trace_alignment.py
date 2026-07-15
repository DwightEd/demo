"""CPU-only tests for the exact-generation trace schema."""

from __future__ import annotations

import unittest

import numpy as np

from utils.step_boundaries import (
    TRACE_SCHEMA_VERSION,
    TokenAlignmentError,
    attach_trace_time_axis,
    build_exact_trace_alignment,
    trace_records_to_npz,
    trim_trailing_generation_tokens,
    truncate_trace_alignment,
)


class MockTokenizer:
    """A deterministic character tokenizer with HF-like return fields."""

    @staticmethod
    def ids(text):
        return [ord(ch) + 17 for ch in text]

    def __call__(self, text, *, add_special_tokens, return_offsets_mapping=False, **_):
        if add_special_tokens:
            raise AssertionError("trace tokenization must disable special tokens")
        out = {
            "input_ids": self.ids(text),
            "attention_mask": [1] * len(text),
        }
        if return_offsets_mapping:
            out["offset_mapping"] = [(i, i + 1) for i in range(len(text))]
        return out


class TraceAlignmentTest(unittest.TestCase):
    def setUp(self):
        self.tok = MockTokenizer()
        self.prompt = "<chat>Question: 2+2?\nAssistant:"
        self.response = "\nfirst step\nsecond step"
        self.steps = ["first step", "second step"]

    def build(self):
        return build_exact_trace_alignment(
            self.tok,
            self.prompt,
            self.response,
            self.steps,
            prompt_token_ids=self.tok.ids(self.prompt),
            prompt_attention_mask=[1] * len(self.prompt),
            response_token_ids=self.tok.ids(self.response),
            response_attention_mask=[1] * len(self.response),
            question_text="2+2?",
        )

    def test_exact_prompt_response_and_step_axis(self):
        trace = self.build()
        self.assertEqual(trace["trace_schema_version"], TRACE_SCHEMA_VERSION)
        self.assertEqual(trace["input_ids"][:len(self.prompt)], self.tok.ids(self.prompt))
        self.assertEqual(trace["response_token_range"],
                         (len(self.prompt), len(self.prompt) + len(self.response)))
        q0 = self.prompt.index("2+2?")
        self.assertEqual(trace["question_char_span"], (q0, q0 + 4))
        self.assertEqual(trace["question_token_range"], (q0, q0 + 4))

        expected = []
        cursor = 0
        for step in self.steps:
            start = self.response.index(step, cursor)
            a = len(self.prompt) + start
            expected.append((a, a + len(step) - 1))
            cursor = start + len(step)
        self.assertEqual(trace["all_step_token_ranges"], expected)

        trace = attach_trace_time_axis(trace, [0, 1], expected)
        payload = trace_records_to_npz([trace])
        self.assertEqual(str(payload["trace_schema_version"]), TRACE_SCHEMA_VERSION)
        np.testing.assert_array_equal(payload["kept_steps"][0], [0, 1])
        np.testing.assert_array_equal(payload["step_token_ranges"][0], expected)
        self.assertEqual(payload["time_axis_kind"].item(), "kept_step_index")
        self.assertFalse(bool(payload["trace_token_add_special_tokens"]))
        self.assertEqual(
            payload["hidden_state_token_semantics"].item(),
            "h_i_after_reading_token_i",
        )
        self.assertEqual(
            payload["logit_prediction_semantics"].item(),
            "logits_i_predict_token_i_plus_1",
        )
        self.assertEqual(int(payload["step_prediction_position_shift"]), -1)

    def test_prompt_token_mismatch_fails_fast(self):
        wrong = self.tok.ids(self.prompt)
        wrong[-1] += 1
        with self.assertRaisesRegex(TokenAlignmentError, "prompt token mismatch"):
            build_exact_trace_alignment(
                self.tok,
                self.prompt,
                self.response,
                self.steps,
                prompt_token_ids=wrong,
                response_token_ids=self.tok.ids(self.response),
                question_text="2+2?",
            )

    def test_response_token_mismatch_fails_fast(self):
        wrong = self.tok.ids(self.response)[:-1]
        with self.assertRaisesRegex(TokenAlignmentError, "response token mismatch"):
            build_exact_trace_alignment(
                self.tok,
                self.prompt,
                self.response,
                self.steps,
                prompt_token_ids=self.tok.ids(self.prompt),
                response_token_ids=wrong,
                question_text="2+2?",
            )

    def test_unmatched_step_fails_fast(self):
        with self.assertRaisesRegex(TokenAlignmentError, "not a verbatim response substring"):
            build_exact_trace_alignment(
                self.tok,
                self.prompt,
                self.response,
                ["missing step"],
                question_text="2+2?",
            )

    def test_truncation_preserves_exact_prompt_prefix(self):
        trace = self.build()
        first = trace["all_step_token_ranges"][0]
        limit = first[1] + 1
        trace = truncate_trace_alignment(trace, limit)
        trace = attach_trace_time_axis(trace, [0], [first])
        self.assertTrue(trace["model_input_truncated"])
        self.assertEqual(trace["input_ids"][:len(self.prompt)], self.tok.ids(self.prompt))
        self.assertEqual(trace["response_token_range"], (len(self.prompt), limit))
        self.assertEqual(trace["full_input_ids"],
                         self.tok.ids(self.prompt) + self.tok.ids(self.response))
        self.assertGreater(len(trace["full_token_offsets"]), len(trace["token_offsets"]))

    def test_terminal_generation_tokens_are_preserved_separately(self):
        visible, terminal = trim_trailing_generation_tokens(
            [10, 11, 2, 2, 0], pad_token_id=0, eos_token_id=2
        )
        self.assertEqual(visible, [10, 11])
        self.assertEqual(terminal, [2, 2, 0])
        trace = self.build()
        trace = attach_trace_time_axis(
            trace, [0, 1], trace["all_step_token_ranges"]
        )
        trace["generated_token_ids"] = visible + terminal
        trace["generation_terminal_token_ids"] = terminal
        payload = trace_records_to_npz([trace])
        self.assertTrue(bool(payload["generated_token_ids_stored"]))
        np.testing.assert_array_equal(payload["generated_token_ids"][0], [10, 11, 2, 2, 0])
        np.testing.assert_array_equal(payload["generation_terminal_token_ids"][0], terminal)

    def test_selected_prompt_hidden_span_uses_shared_schema(self):
        trace = self.build()
        trace = attach_trace_time_axis(
            trace, [0, 1], trace["all_step_token_ranges"]
        )
        trace["prompt_hidden_layers"] = np.asarray([2, 5], dtype=np.int32)
        trace["prompt_hidden"] = np.zeros(
            (len(trace["prompt_token_ids"]), 2, 4), dtype=np.float16
        )
        payload = trace_records_to_npz([trace])
        self.assertTrue(bool(payload["prompt_hidden_stored"]))
        np.testing.assert_array_equal(payload["prompt_hidden_layers"], [2, 5])
        self.assertEqual(payload["prompt_hidden"][0].shape,
                         (len(trace["prompt_token_ids"]), 2, 4))


if __name__ == "__main__":
    unittest.main()
