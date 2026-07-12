# Related-Work Guardrails

This file records why Prompt-Controlled Residual Flow must be evaluated as a
narrow diagnostic, not presented as a generic new subspace method.

## Nearby Work

1. **HARP: Hallucination Detection via Reasoning Subspace Projection**
   - Uses SVD-derived reasoning subspaces and hidden-state projections for
     hallucination detection.
   - Guardrail: do not claim that "project hidden states into a reasoning
     subspace" is novel.

2. **ICR Probe: Tracking Hidden State Dynamics for Reliable Hallucination
   Detection in LLMs**
   - Studies hidden-state update dynamics and residual-stream contribution.
   - Guardrail: do not claim residual-update dynamics alone as the contribution.

3. **Hallucinations as Orthogonal Noise / DCO**
   - Frames hallucinations as components orthogonal to a context/semantic
     manifold and intervenes on attention-head outputs.
   - Guardrail: our method should not simply rebrand context-orthogonal noise.

4. **GeoFaith**
   - Uses spatio-temporal latent geometry and entropy dynamics for faithful CoT.
   - Guardrail: do not use generic latent geometry plus entropy as the main
     novelty claim.

5. **Where Does Reasoning Break?**
   - Uses hidden-state trajectory transport geometry for first-error
     localization.
   - Guardrail: first-error transport/localization is already a crowded target;
     response-level and control-source shift diagnostics need explicit evidence.

## Narrow Claim Worth Testing

The claim here is not "SVD works" or "geometry detects errors."  The claim is:

Correct reasoning remains materially controlled by the original problem prompt,
whereas some wrong reasoning becomes increasingly controlled by the generated
prefix/template dynamics.  This should appear as a change in residual-stream
writes, not merely as a longer chain, later step, larger spread, or higher
entropy.

## Required Evidence Before Any Claim

1. `prompt_frac` and `prompt_control_ratio` separate correct and incorrect
   response trajectories better than `step_len`, `rel_pos`, and `random_frac`.
2. Error chains show a visible trajectory-level shift, not only one late spike.
3. Matched-rank random subspaces do not reproduce the same result.
4. On same-problem multisample data, incorrect samples score worse than correct
   samples for the same problem.
5. Case cards should show whether failure is prompt-control loss, prefix lock,
   or a coherent wrong basin.

If these checks fail, this direction is a diagnostic negative result, not a
paper method.
