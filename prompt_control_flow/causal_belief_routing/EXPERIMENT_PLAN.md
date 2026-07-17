# Claim-driven Experiment Plan

## Paper target

Working title: **Hidden Futures, Routed Evidence: Causal Belief Geometry in
Pretrained Transformers**.

Dominant contribution: a predictive-alias protocol with analytic Fourier belief
geometry that distinguishes residual belief state from current logits and then
tests the QK/OV path causally.

## Experiment block 1: geometry existence

### Inputs

- 2,000 training and 500 held-out affine alias pairs;
- modulus \(p=3\), four variables, two common constraints, one branch constraint;
- three independent paraphrase templates;
- all residual depths or a preregistered sparse depth grid.

### Models

- primary observer: Meta-Llama-3.1-8B-Instruct;
- replication: one different family or one smaller same-family model;
- optional small trained transformer as a positive-control ceiling.

### Primary endpoints

- current branch Jensen-Shannon divergence over residue logits;
- future-query accuracy;
- cross-fitted future NLL from output-only, layer-only, all-layer, and joint
  Fourier charts;
- conditional usable bits of all-layer hidden geometry over current logits;
- pairwise Fourier-distance preservation.

### Nulls

- shuffled belief labels within template and support size;
- randomized finite-field query directions;
- Fourier-label permutation inside training folds;
- branch text with an equal-length but logically irrelevant residue.

### Decision

Continue only if the hidden chart has a positive cluster-bootstrap information
increment over actual current logits and the exact-output alias remains intact.

## Experiment block 2: routing and mediation

### Extraction

- eager attention only for short controlled prompts;
- selected residual states through boundary hooks;
- selected V-projection outputs and final-token attention rows;
- no persisted full \([L,H,N,N]\) attention tensors;
- save only evidence attention mass, pre-\(W_O\) source contributions, and
  chart-projected Fourier writes.

### Primary endpoints

- evidence OV-write alignment with \(\Delta\Phi^\star\);
- true-versus-opposite-branch alignment difference;
- evidence-versus-length-matched-token alignment difference;
- donor-to-recipient future-logit mediation effect;
- fraction of the total branch causal effect mediated by preregistered heads.

### Ablations

- QK routing held fixed while V content is swapped (follow-up ablation);
- V content fixed while attention weights are swapped (follow-up ablation);
- whole-head versus evidence-source-only patch;
- MLP-output patch and residual-state patch as upper-bound controls;
- head rankings selected on training folds and frozen on test folds.

## Experiment block 3: ProcessBench transfer

### Question

Does a confidence update become unreliable when it lacks an evidence-aligned
residual write, even after the output distribution itself is observed?

### Inputs

- exact ProcessBench observer traces already extracted for GSM8K, MATH,
  OlympiadBench, and Omni-MATH;
- a new attention pass only for the preregistered layer-head paths and only on a
  confirmatory subset;
- first-error labels and response correctness.

### Models

\[
M_Z=P(Y\mid Z_{1:t},C),
\qquad
M_{Z+R}=P(Y\mid Z_{1:t},R_{1:t},C),
\]

where \(Z\) contains entropy, margin, chosen-token probability, top-k mass, and
their history; \(R\) contains only routing-mechanism scores frozen in block 2;
\(C\) contains length, position, dataset, and lexical controls.

### Primary endpoints

- cluster-cross-fitted usable bits of \(M_{Z+R}\) over \(M_Z\);
- response AUROC/AUPRC increment;
- first-error localization under matched step length;
- cross-dataset replication without retuning the mechanism score.

## Run order

1. Unit and schema tests.
2. Generate 20 alias pairs and inspect exact invariants.
3. Extract 20 pairs with all layers, no attention.
4. Fit Fourier charts and run nulls.
5. If Gate B passes, extract attention for 20 pairs.
6. Run ten donor-recipient patches and verify directionality.
7. Scale block 1 to 2,500 pairs.
8. Freeze layers/heads and scale block 2.
9. Only after Gates B-D pass, run ProcessBench transfer.

## Reproducibility contract

- pair IDs are the grouping unit in every split and bootstrap;
- every artifact records model path, revision, tokenizer, chat template hash,
  selected layers, dtype, seed, and extraction protocol;
- failures are written to a skip report and coverage is always reported;
- GPU extraction and CPU/GPU analysis are separate commands;
- every claim table includes the output-only and exact matched nulls.
