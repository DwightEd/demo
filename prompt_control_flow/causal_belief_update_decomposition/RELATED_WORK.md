# Related-work Boundary and Search Protocol

## What is already established

1. **Residual belief geometry.** [Shai et al., *Transformers Represent Belief
   State Geometry in their Residual Stream*](https://arxiv.org/abs/2405.15943)
   (2024/2025), use exact HMM mixed
   states and affine regression to show that residual activations preserve
   belief-simplex geometry. Their RRXOR experiment is especially important:
   distinct beliefs with the same next-token distribution can be distributed
   across layers.
2. **Attention-constrained updates.** [Piotrowski et al., *Constrained Belief
   Updates Explain Geometric Structures in Transformer Representations*](https://proceedings.mlr.press/v267/piotrowski25a.html)
   (ICML 2025), predict attention patterns, OV vectors, and intermediate geometry from
   architecture-constrained belief updates in trained HMM transformers.
3. **Bayesian wind tunnels.** [Agarwal et al., *The Bayesian Geometry of
   Transformer Attention*](https://arxiv.org/abs/2512.22471) (2025/2026
   preprint), report orthogonal key bases,
   progressive QK alignment, and entropy-parameterized value geometry in small
   controlled transformers.
4. **Unsupervised simplex discovery.** [Levinson, *Finding Belief Geometries with
   Sparse Autoencoders*](https://arxiv.org/abs/2604.02685) (2026 preprint),
   combines SAE features, subspace
   clustering, and simplex fitting, but explicitly treats causal evidence in
   natural LLMs as preliminary.

## Unclaimed gap

The present project does **not** claim that belief geometry or constrained
attention updates are new. It tests a narrower missing link:

- a frozen pretrained instruction LLM rather than a model trained on the toy
  generator;
- exact predictive aliases that hold the current target distribution fixed;
- an analytic Fourier chart for relative update directions;
- source-token-specific OV decomposition;
- donor-recipient path interventions;
- transfer from the controlled mechanism to reasoning-error detection beyond
  logits.

## Search query families used

The literature scan deliberately moved from representation to mechanism to
causality:

```text
"belief state geometry" transformer "residual stream"
"mixed state presentation" transformer activations future prediction
"next-token degeneracy" belief state transformer layers
"constrained belief updates" attention OV vectors geometry
"Bayesian geometry" transformer attention residual stream
attention routing value manifold posterior entropy transformer
QK OV circuit belief update activation patching
path patching attention source token residual contribution
finite field Fourier representation transformer modular arithmetic
affine constraints posterior Fourier characters neural representation
pretrained LLM exact posterior constraint reasoning hidden state
```

Searches were restricted to arXiv, PMLR/OpenReview, and official paper/code
pages for technical claims. Broad terms such as `LLM manifold` were used only
for discovery and never as evidence for the method.
