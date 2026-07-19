# Independent Code Review

Date: 2026-07-19

Verdict after initial review: **Conditional Pass** for synthetic validation and controlled offline pilot.

## Confirmed

- Weighted debiased pair-cosine matched explicit pair enumeration exactly.
- Weighted centered scatter and effective-rank implementation are correct.
- Every trailing window uses only positions at or before the current token.
- Post-token/next-token-shift semantics are explicit.
- Sparse layer ids, missing layer metadata, insufficient nonzero directions, and empty traces fail closed.
- Existing per-chain shard fields (`response_token_state_files`, counts, layers, `chain_idx`) are covered.

## Findings fixed during review

1. Direct inputs previously inferred consecutive layers with `arange`; they now require layer metadata.
2. `min_tokens` previously counted window positions; it now counts nonzero valid directions.
3. `mean_write_norm` is now unconditional and includes zero writes in every branch.
4. Snapshot provenance now defaults to `unverified` and is rejected for block-write analysis unless the
   caller explicitly requests exploratory use.
5. Field metadata now separates causal statistics from the acausal retrospective normalized phase.

## Remaining blocking gate for real-model claims

The extractor must record `response_token_state_snapshot_kind=raw_residual_stream`, select consecutive
depths, and avoid treating a terminal final-norm state as a block output. M2 should additionally require
small reconstruction error between adjacent-state deltas and hooked block residual writes.

Verification after fixes: `9 passed`; CLI selftest passed; `compileall` passed.
