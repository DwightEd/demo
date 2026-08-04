# Conditional Innovation Hazard

## Research question

The method tests whether a reasoning prefix contains a transferable precursor of
the first erroneous step. It does not classify a chain from a few hand-written
geometry scores. Instead, it learns how a correct prefix normally moves between
two residual-stream layers, then asks whether the full deviation from that
learned flow improves held-domain first-error risk prediction.

The primary estimand is the held-domain predictive-risk difference

```text
NLL(output + nuisance) - NLL(output + nuisance + innovation)
```

The claim is deliberately limited to prospective association. Hidden states are
teacher-forced observations, and a predictive increment is not proof that an
attention route or an MLP fact caused the error.

## Temporal target

For candidate step `t`, a row may use only information available after step
`t-1`:

- residual state at the end of step `t-1`;
- output summaries from steps `0..t-1`;
- boundary position and completed-prefix lengths.

The label is one only when `t` is the annotated first error. Rows after the first
error are censored. Fully correct chains contribute right-censored negative risk
rows. Step-zero errors are left-truncated because the stored response-token
shards do not contain the prompt-end state.

This target detects *risk before the next step*. It does not inspect the tokens
inside the step that is about to be generated.

## Fold-local model

Every outer leave-one-dataset-out fold performs the following operations using
outer-training data only:

1. Select negative at-risk rows, which are prefixes that have remained correct
   through the observed boundary.
2. Learn separate weighted rank-`q` coordinates for source and destination
   layer states.
3. Fit the normal conditional map

   ```text
   z_destination = D [z_source, nuisance, past_output] + b + residual
   ```

4. Estimate and shrink the normal residual covariance, then whiten the complete
   `q`-dimensional residual. No residual norm, angle, rank, or dispersion scalar
   replaces this vector.
5. Fit a regularized discrete first-error hazard to six predeclared arms.

The two crucial capacity-matched arms are:

```text
output_plus_hidden      = controls + past output + q-dimensional destination state
output_plus_innovation  = controls + past output + q-dimensional whitened residual
```

If innovation does not beat the equal-rank destination state, the conditional
flow construction has not earned its added interpretation.

## What this says about routing

The source-to-destination pair is one directed edge in a layer graph. Its
innovation measures an unexpected net residual update across that edge. With
hidden states alone it cannot separate attention routing from MLP computation.
That attribution requires component-level attention/MLP updates or an
intervention, and is intentionally not claimed by this stage.

A larger graph is a gated extension, not the starting point. If the single-edge
innovation is reproducible, add several predeclared layer edges and test whether
message-passing or sparse graph coupling improves held-domain NLL over the same
edge innovations. If the single-edge signal fails, a graph network mainly adds
capacity and does not repair the hypothesis.

## Execution path

The existing project remains the only runnable project. Its direct path is:

```text
hidden_state_geometry/cli.py
  -> experiment.py (data, task, outer LODO, artifacts)
  -> methods/innovation_hazard.py::InnovationHazard.fit_predict()
  -> evaluation.py (held-domain metrics and problem-cluster bootstrap)
```

Remote foreground commands:

```bash
bash run_hidden_geometry_remote.sh innovation-smoke
bash run_hidden_geometry_remote.sh innovation-full
```

Both commands use only stored ProcessBench residual states and aligned
entropy/NLL summaries. They do not load the language model or require a GPU.

## Continue or stop

Continue to component attribution only if the full four-domain result shows:

1. `output_plus_innovation` improves held-domain NLL over `output_only` with a
   positive problem-cluster confidence interval;
2. the improvement is not explained by the equal-rank `output_plus_hidden` arm;
3. the sign is not carried by only one dataset;
4. rank, layer-pair, and shrinkage sensitivity preserve the conclusion.

Only after that gate is it worth extracting attention/MLP component updates,
adding a layer-time graph, or measuring exact output susceptibility and doing
activation patching. Without the gate, those extensions are too flexible for
the current evidence.
