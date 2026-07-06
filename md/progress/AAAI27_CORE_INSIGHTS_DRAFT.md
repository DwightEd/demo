# Core Insights Draft for a Reasoning-Monitor Paper

Date: 2026-07-06

This note is a first paper-facing synthesis.  It combines validated results,
negative-result boundaries, and the next target experiments into a coherent
NeurIPS/ICLR/AAAI-style narrative.  It is not a final manuscript; it is the
argument spine from which the abstract, introduction, Figure 1, and experiment
section should be written.

## Working Title Options

1. **Reasoning Errors Are Local Failures of Hidden-State Anchoring**
2. **Beyond Hidden-State Probes: Confound-Controlled Physiology of Reasoning Errors**
3. **When Reasoning Loses Its Anchor: Token-Level Geometry for Online Error Monitoring**
4. **Reasoning Fails Where Hidden-State Flow Loses Directional Consensus**

The strongest title is probably option 1 if the massive-activation/anchor
decomposition succeeds.  If that branch fails, option 2 is safer and more
honest: the paper becomes a rigorous audit of what internal geometry can and
cannot detect.

## One-Sentence Contribution

**Draft contribution sentence:**

> We show that reasoning errors in long chain-of-thought generation produce a
> weak but reproducible loss of within-step hidden-state directional consensus,
> then turn this observation into a confound-controlled online monitoring
> protocol that separates genuine reasoning physiology from length, difficulty,
> entropy, and anisotropy artifacts.

**Stronger version if massive-activation results pass:**

> We identify reasoning errors as local failures of hidden-state anchoring:
> incorrect reasoning steps weaken the massive-activation/sink dimensions that
> normally stabilize token representations, producing a measurable loss of
> directional consensus that is complementary to output uncertainty.

## Core Insight

The central insight should not be "we found another geometry metric."  The
field already has many geometry metrics, and our own experiments show that most
simple spectral, Gram, HMM, and path-shape variants do not add signal once
same-problem controls are enforced.

The core insight should be:

```text
Reasoning-error geometry is real, but small, conditional, and easily
confounded.  Its value is as a physiological signal of local anchoring failure,
not as a standalone correctness classifier.
```

This gives the paper a stronger shape than a metric paper.  The narrative is:

1. **Discover:** Wrong reasoning steps lose directional consensus in middle
   hidden states.
2. **Debias:** Much apparent geometry signal disappears under same-problem,
   length, difficulty, entropy, and static-spread controls.
3. **Localize:** The surviving signal is synchronous/local, not a long-range
   early-warning drift.
4. **Mechanize:** The next explanatory hypothesis is anchoring: massive
   activation / attention-sink dimensions may provide stabilizing directions
   whose failure produces the observed kappa drop.
5. **Deploy:** A real-time monitor must be boundary-free and token-causal, using
   step labels only for offline evaluation.

## Figure 1 Concept

**Figure 1 should be the paper in one page.**

Panel A: A generated reasoning trace with a first wrong step.  Show hidden
token vectors within each step as directional clouds.  Correct steps have a
tighter consensus direction; the wrong step loses consensus.

Panel B: The same trace under online token windows.  Plot:

- direction concentration `R_t`;
- direction breadth `1 - R_t`;
- effective rank `eff_rank_t`;
- optional entropy baseline.

Panel C: Confound controls.  Show that cross-problem AUROC is inflated, while
same-problem paired ranking is stricter.  Include length/difficulty as a
visible nuisance axis.

Panel D: Mechanistic decomposition.  Split hidden norm and concentration into:

```text
massive dimensions vs residual dimensions
```

If massive dimensions explain the error-step norm/concentration drop, Figure 1
becomes a mechanistic story rather than another probe.

**Caption center sentence:**

> Reasoning errors are visible not as a global collapse of the chain, but as a
> local failure to maintain directional consensus in the hidden states of the
> tokens that realize the faulty step.

## Abstract Draft

Version for current evidence:

> Long chain-of-thought reasoning exposes language models to local failures that
> are often invisible from final-answer confidence alone.  We study whether such
> failures leave an intrinsic signature in the model's hidden dynamics.  Across
> ProcessBench-style step labels and same-question multi-sampling, we find that
> incorrect reasoning steps exhibit a reproducible loss of directional
> consensus among middle-layer token representations.  This signal is real but
> fragile: under leak-free same-problem controls, many plausible extensions
> including spectral tails, Gram dynamics, latent HMMs, path kernels, and
> arithmetic regex ledgers fail to improve over simple spread and entropy
> baselines.  We therefore reframe hidden geometry as a physiological marker
> rather than a standalone verifier, and introduce a boundary-free token-stream
> audit that tests whether concentration, breadth, and effective-rank dynamics
> remain useful under online, length-controlled evaluation.  Our results provide
> a confound-controlled map of when internal geometry helps, when it fails, and
> what mechanisms must be tested next for deployable reasoning monitors.

Version if token-stream and massive-anchor results pass:

> Long chain-of-thought reasoning fails locally: a model can continue producing
> fluent, confident text after the internal computation supporting a step has
> lost its anchor.  We show that these failures produce a measurable weakening
> of directional consensus among middle-layer token representations, partly
> explained by reduced energy in massive-activation dimensions that normally act
> as stabilizing sink directions.  Building on this mechanism, we introduce a
> boundary-free token-stream monitor that computes causal concentration,
> breadth, and effective-rank traces without requiring parsed reasoning steps.
> Under same-question multi-sample controls, our monitor separates genuine
> reasoning physiology from length, difficulty, entropy, and anisotropy
> artifacts, while fixed-FPR alarms quantify localization delay.  The resulting
> framework moves hidden-state geometry from post-hoc probing toward online
> intervention: it identifies when the model's reasoning flow loses the
> representational anchor needed to sustain valid inference.

## Introduction Spine

### Paragraph 1: Problem

**Center sentence:**

> Long reasoning traces fail at the level of process, but most reliability
> methods observe only the final answer or output uncertainty.

Key points:

- Long CoT creates many opportunities for a local step to go wrong.
- Final-answer correctness is a delayed, coarse label.
- Entropy/logit confidence is useful but misses coherent wrong reasoning.
- A deployable monitor needs an internal, token-causal signal.

### Paragraph 2: Existing Geometry Is Promising but Overclaimed

**Center sentence:**

> Recent work suggests that reasoning has geometric structure, but geometric
> compression alone is not a detector of correctness.

Use:

- `Reasoning emerges from constrained inference manifolds...` argues that
  inference dynamics self-organize into low-dimensional manifolds, but also
  emphasizes that compression alone is insufficient for reliable reasoning.
- `Effective Reasoning Chains Reduce Intrinsic Dimensionality` shows that
  effective reasoning strategies correlate with lower intrinsic dimensionality.
- `What do Geometric Hallucination Detection Metrics Actually Measure?` warns
  that geometry metrics can measure domain, relevance, coherence, or confidence
  rather than truth itself.

### Paragraph 3: Our Anchor Observation

**Center sentence:**

> The robust signal we find is local and directional: when a reasoning step is
> wrong, the token representations within that step lose directional consensus.

Key results to cite from project:

- Error steps show lower kappa / higher spread in middle-layer hidden states.
- Error rate is monotonic over fixed-length kappa bins in prior ProcessBench
  validation.
- Geometry is complementary to output entropy; fusion improves over entropy or
  geometry alone in the full GSM8K setting.
- The signal is weak under hard same-problem controls, so the paper must not
  sell it as a universal detector.

### Paragraph 4: What We Ruled Out

**Center sentence:**

> The negative results are central: most obvious ways of making the geometry
> more sophisticated do not survive same-problem controls.

Use specific results:

- Scalar HSMM / latent EM: AUROC around 0.538 vs static spread around 0.682.
- Path kernel / functional shape: best witness below static spread.
- Direct token Gram / spectral tail: baseline 0.685 vs best Gram group 0.660;
  spectral tail increment negative.
- Regex arithmetic ledger: below geometry baselines on both custom and 5-shot.

This paragraph makes the paper more credible.  It tells reviewers we did not
just tune metrics until one worked.

### Paragraph 5: Reframing

**Center sentence:**

> These results suggest a different role for hidden geometry: it is a
> physiological symptom of local anchoring failure, not a complete verifier of
> reasoning correctness.

This is where the massive-activation hypothesis enters:

- Deep-layer anisotropy can inflate cosine similarity.
- Massive activation dimensions can dominate middle-layer norms.
- Therefore, "wrong step has lower norm/concentration" may mean "the step did
  not establish the anchor/sink computation that usually stabilizes tokens."
- This is concrete and falsifiable by ablating massive dimensions.

### Paragraph 6: Contributions

**Draft contribution bullets:**

1. We validate a local, direction-level signature of reasoning errors: incorrect
   steps exhibit reduced hidden-token directional consensus.
2. We provide a confound-controlled audit showing which geometry families fail
   under same-problem, length, entropy, and static-spread controls.
3. We introduce a token-causal monitoring protocol that removes the unrealistic
   assumption of pre-segmented reasoning steps and reports fixed-FPR delay.
4. We propose and test an anchoring decomposition that separates massive
   activation dimensions from residual hidden geometry.

If the anchoring experiment is still pending, write contribution 4 as:

> We identify massive-activation anchoring as a falsifiable mechanism for the
> observed norm/concentration drop and provide the experimental protocol needed
> to test it.

## Method Section Spine

### Method 1: Directional Consensus

**Center sentence:**

> We measure whether tokens in a local reasoning region point toward a shared
> hidden-state direction.

Definitions:

```text
u_i^l = h_i^l / ||h_i^l||
R = ||sum_i w_i u_i^l|| / sum_i w_i
spread = 1 - R
```

For step-level data, `i` ranges over tokens in a step.  For online monitoring,
`i` ranges over causal token windows.

### Method 2: Boundary-Free Token Stream

**Center sentence:**

> Because real generation does not provide reasoning-step boundaries, the
> deployable monitor computes all primary scores from causal token windows.

Definitions:

```text
R_t(W,l) = ||sum_{i=t-W+1}^{t} exp(-beta(t-i)) u_i^l|| / sum_i exp(-beta(t-i))
spread_t(W,l) = 1 - R_t(W,l)
```

Report:

- paired AUROC;
- fixed-FPR recall;
- alarm delay;
- endpoint fraction;
- increment over length/entropy/static baselines.

### Method 3: Flow Shape

**Center sentence:**

> We test whether reasoning traces exhibit a stereotyped expand-then-compress
> morphology, and whether errors interrupt the compression phase.

Signals:

```text
eff_rank_t(W) = exp(H(spectrum(H_{t-W:t} H_{t-W:t}^T)))
alpha_t(W)    = log-log spectral slope
hump_present  = interior peak + early rise + late fall
```

This is exploratory until the remote shape results are in.

### Method 4: Massive-Activation Anchoring

**Center sentence:**

> We decompose the consensus signal into massive-activation dimensions and the
> residual hidden subspace to test whether error steps fail by losing anchor
> energy rather than by becoming uniformly noisy.

Protocol:

```text
M_l = top dimensions by global/token percentile activation magnitude
massive_energy_t = ||h_t[M_l]||^2
residual_energy_t = ||h_t[not M_l]||^2
R_massive, R_residual, R_without_massive
```

Falsifiable outcomes:

- If `R_without_massive` keeps the signal, the signal is genuinely distributed.
- If only `R_massive` carries the signal, the kappa effect is an anchor/sink
  mechanism.
- If neither carries signal under same-problem controls, norm/geometry was a
  difficulty proxy.

## Experiments Spine

### Experiment 1: Does directional consensus detect wrong steps?

**Center sentence:**

> Incorrect steps show systematically lower directional consensus than correct
> steps, but the effect size depends strongly on task difficulty and control
> regime.

Report:

- ProcessBench step-level result.
- Fixed-length bins: error rate decreases monotonically with kappa.
- Cross-fit/leak-free AUROC and bootstrap CI.

### Experiment 2: Is the signal orthogonal to entropy?

**Center sentence:**

> Directional consensus and output uncertainty capture different failure modes.

Report:

- correlation with entropy;
- fusion gain over entropy baseline;
- confident-wrong subset.

### Experiment 3: What does not work?

**Center sentence:**

> More complex geometry does not automatically mean more reliable detection.

Report as a single table:

| Branch | Result | Decision |
|---|---|---|
| HSMM/EM over scalar channels | below static spread | retire |
| Path kernels | below static spread | retire |
| Token Gram/spectral tail | negative increment | retire |
| Regex premise ledger | below baseline | retire as local regex |

This table is paper-worthy because it prevents reviewers from asking why we
did not try the obvious variants.

### Experiment 4: Does online token monitoring preserve the signal?

**Center sentence:**

> A deployable monitor must work without step segmentation and without
> accumulating late-chain artifacts.

Report:

- `token_stream_geometry_audit.py`;
- no-alpha fast kappa result;
- alpha/effective-rank shape result;
- fixed-FPR delay.

### Experiment 5: Does massive anchoring explain the signal?

**Center sentence:**

> The key mechanistic test is whether the wrong-step norm and kappa drop are
> concentrated in massive-activation dimensions.

Report:

- massive dimension identification;
- energy decomposition;
- kappa with/without massive dims;
- same-problem paired increment.

## Related Work Map

### Reasoning Geometry

- `Reasoning emerges from constrained inference manifolds in large language
  models` (arXiv 2026): reasoning dynamics self-organize into low-dimensional
  manifolds, but compression alone is not sufficient for stable reasoning.
- `Effective Reasoning Chains Reduce Intrinsic Dimensionality` (ICML spotlight
  2026): effective reasoning strategies reduce intrinsic dimensionality and
  correlate with generalization.
- `LLM Reasoning as Trajectories` (arXiv 2026): treats CoT as structured
  trajectories through representation space.

### Flow and Intervention

- `Reasoning Fails Where Step Flow Breaks` (ACL 2026): Step-Saliency reveals
  Shallow Lock-in and Deep Decay, and StepFlow improves performance by repairing
  information flow.

### Geometric Reliability Metrics

- `What do Geometric Hallucination Detection Metrics Actually Measure?` (ICML
  R2-FM Workshop 2025): geometry metrics can capture different hallucination
  phenotypes and are sensitive to domain shifts.

### Massive Activations and Anchors

- `Massive Activations in Large Language Models` (COLM 2024): rare very large
  activations act as indispensable bias terms and concentrate attention.
- `A Single Layer to Explain Them All: Understanding Massive Activations in
  Large Language Models` (arXiv 2026): identifies a Massive Emergence Layer and
  links massive activations to representation rigidity and attention sinks.

## Claims We Should Avoid

Do not claim:

- hidden geometry alone predicts correctness;
- kappa is novel;
- spectral alpha solves online error detection;
- early warning exists before checking delay and endpoint fraction;
- regex arithmetic is a premise verifier;
- same-problem oracle supports are deployable.

Safer claims:

- directional consensus is a reproducible physiological signal;
- the signal is complementary to entropy;
- many geometric variants fail under strict controls;
- online token-stream evaluation is the correct deployment setting;
- massive-activation anchoring is the most promising mechanism-level next test.

## Candidate Paper Opening

> A long reasoning trace can fail before the final answer is visibly wrong.  In
> a multi-step mathematical solution, a model may choose the wrong operation,
> bind a number to the wrong premise, or stop using earlier context, while still
> producing fluent and confident text.  Detecting such failures requires a
> signal that is local, intrinsic, and available during generation.

> We study one such signal: the directional consensus of hidden states within
> the tokens that realize a reasoning step.  The intuition is simple.  When a
> model carries out a valid local computation, the generated tokens should be
> constrained by a shared internal state of the computation.  When that local
> computation fails, the token states may remain fluent at the output layer but
> lose directional agreement in the middle layers.

> This intuition is only partly correct.  We find that incorrect reasoning steps
> do exhibit lower hidden-state directional consensus, and this signal is
> complementary to output entropy.  However, the effect is small under
> same-question controls, and many attractive extensions do not improve it:
> latent HMMs, path kernels, token Gram spectra, spectral tails, and regex
> arithmetic ledgers all fail to beat simple spread/entropy baselines.  These
> failures are not incidental; they define the contribution of the paper.  They
> show that hidden geometry is not a verifier of truth, but a physiological
> marker of local anchoring.

## Immediate Next Writing Tasks

1. Run `token_stream_geometry_audit.py` with `--save_profiles` and record
   whether effective-rank traces exhibit real rise-then-fall morphology.
2. Implement massive-dimension decomposition and test whether wrong-step norm
   drops are concentrated in anchor/sink dimensions.
3. Convert the negative-result matrix into one compact main-paper table.
4. Draft Figure 1 from the token-stream profile JSONL once remote results arrive.
5. Start a proper `.bib` only after fetching verified BibTeX programmatically.

## Verified Source Links

- Reasoning emerges from constrained inference manifolds in large language
  models, arXiv 2026: https://arxiv.org/abs/2605.08142
- Reasoning Fails Where Step Flow Breaks, ACL 2026: https://arxiv.org/abs/2604.06695
- Effective Reasoning Chains Reduce Intrinsic Dimensionality, ICML spotlight
  2026: https://arxiv.org/abs/2602.09276
- What do Geometric Hallucination Detection Metrics Actually Measure?, ICML
  R2-FM Workshop 2025: https://arxiv.org/abs/2602.09158
- Massive Activations in Large Language Models, COLM 2024:
  https://arxiv.org/abs/2402.17762
- A Single Layer to Explain Them All: Understanding Massive Activations in
  Large Language Models, arXiv 2026: https://arxiv.org/abs/2605.08504

