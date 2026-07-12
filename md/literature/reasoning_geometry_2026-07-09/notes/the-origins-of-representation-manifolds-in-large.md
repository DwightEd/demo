# The Origins of Representation Manifolds in Large Language Models

- **Local PDF filename**: `The Origins of Representation Manifolds in Large.pdf`
- **Slug**: `the-origins-of-representation-manifolds-in-large`
- **Pages**: 16
- **Approx Words**: 8135
- **Auto Tags**: geometry
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.625705

## Keyword Profile

- `manifold`: 43
- `dimension`: 23
- `geometry`: 17
- `topolog`: 10
- `geometric`: 8
- `latent`: 3
- `probe`: 3
- `entropy`: 2
- `riemann`: 1

## Abstract / Opening Summary

There is a large ongoing scientific effort in mechanistic interpretability to map embeddings and internal representations of AI systems into human-understandable concepts. A key element of this effort is the linear representation hypothesis, which posits that neural representations are sparse linear combinations of ‘almost- orthogonal’ direction vectors, reflecting the presence or absence of different fea- tures. This model underpins the use of sparse autoencoders to recover features from representations. Moving towards a fuller model of features, in which neural representations could encode not just the presence but also a potentially continuous and multidimensional value for a feature, has been a subject of intense recent discourse. We describe why and how a feature might be represented as a mani- fold, demonstrating in particular that cosine similarity in representation space may encode the intrinsic geometry of a feature through shortest, on-manifold paths, potentially answering the question of how distance in representation space and relatedness in concept space could be connected. The critical assumptions and predictions of the theory are validated on text embeddings and token activations of large language models. 1

## Method / Algorithms Extract

methodology of sparse autoencoders (SAEs) (Elhage et al., 2022; Bricken et al., 2023) employs ideas from sparse coding (Elad, 2010) to estimate a dictionary of these directions from representations. This model and methodology reflect a radical goal of breaking representations down into basic, irreducible, atomic concepts which are meaningfully only described as present or absent (Cunningham et al., 2023; Bricken et al., 2023; Templeton et al., 2024). Commonly cited examples are features such as floppy_ears, Eiffel_Tower, or is_Arabic, the presence of which it would presumably be useful for an algorithm to infer (corresponding e.g. to cat/dog classification, the topic of a question, the language of a query). It is generally accepted that this breakdown of representation space into purely atomic features does not tell the whole story (Smith, 2024; Mendel, 2024; Bussmann et al., 2024; Olah, 2024; Engels Preprint. Under review. ===== PAGE 2 / 16 ===== et al., 2025). There is overwhelming empirical evidence that neural networks represent complex features in structures which unfold across multiple directions in potentially continuous, nonlinear ways: examples of curves (Hanna et al., 2023; Chang et al., 2022), swiss-roll-like manifolds (Cai et al., 2021), loops (Engels et al., 2025; Gorton, 2024), tori (Chang et al., 2022), hierarchical trees (Park et al., 2024) in real language models; topologically circular representations of numbers in toy models trained to perform modular arithmetic (Liu et al., 2022; Nanda et al., 2023a; Zhong et al., 2023; He et al., 2024) or simulated angular data (Olah and Batson, 2024), fractal geometry in simulated hidden Markov models (Shai et al., 2024); and broader phenomenology from local finite-state-automata (Bricken et al., 2023), to spatial ‘brain-like’ modularity (Li et al., 2025), to behaviour, such as deception (Templeton et al., 2024). SAEs are not made defunct by these discoveries, and in fact have often facilitated them through recombination of SAE directions (Bussmann et al., 2025; Engels et al., 2025). The LRH has been extended to allow this more flexible interpretation of the output of SAEs: Definition 1 (Multidimensional linear representation hypothesis). There exists a collection of features labeled f ∈F and associated subspaces Vf ∈RD such that the functional relationship between an input x ∈X and its representation Ψ(x) is Ψ(x) = X f∈F(x) ρf(x)vf(x), vf(x) ∈Vf and ∥vf(x)∥2 = 1, (1) where ρf(x) is a non-negative scaling denoting the presence of the feature f in x, and F(x) = {f : ρf(x) > 0} is the set of features ...

## Experiments / Evidence Extract

experiments are shown in the first row of Figure 3. The second (indirect) approach is to test Theorem 1: geodesic distance on Mf (shortest path length) should be linear in the geodesic distance on Zf, up to noise (the slope being p −2g′(0)). We estimate geodesics on Mf by constructing the K-nearest-neighbours graph over the representations, and reporting weighted graph distance, k chosen as small as possible subject to the graph being connected. We quantify the strength of isometry using Pearson’s correlation ρ, which would be 1 if the distances were in a perfect proportional relationship. These experiments are shown in the second row of Figure 3. Across our experiments, we have found that a low-dimensional projection tends to be necessary for the representations to plausibly show isometry with a simple metric space. For our text embeddings, we find that projecting onto the first few (uncentered) principal components works well. The routine “low-rank” explanation that the remaining components are mostly noise seems disputable; these components often show clear structure. Our best explanation is that semantic similarity is much richer than the rudimentary metric spaces to which they are being compared. It is likely that we could achieve a deeper understanding of semantic similarity through improved metric space design. In the example of years, the process of extracting feature representation via an SAE automatically yields low-dimensional representations, so no PCA is applied in this case. Recall that we conjectured the following metric space for the years example: Zyears = [1900, 1999], dyear(x, y) = |x −y|. Although we found a rank correlation near 1, indicating homeomorphism, the evidence of the tests above is against isometry. The clearest indication in this direction is possibly provided by the right panel of Figure 4, which does not show a regular linear relationship, the colours suggesting that distances between more recent years are expanded on the manifold. In light of this, we consider a modified representation Zyears = {log(2019 −year) : year ∈ [1900, 1999]}, dyear(x, y) = |x −y|, 2019 being the year GPT-2 was released (Radford et al., 2019). Observe that the rank correlation as computed in Section 2.3 remains unchanged: the two representations are homeomorphic, and cannot be distinguished on purely topological criteria. The tests are now in much s...

## Conclusion / Discussion Extract

discussion on how features are geometrically represented in language models, we really ought to pin down what exactly we mean by “a feature”. Definition 2. A feature, labeled f, is a metric space (Zf, df). 2 ===== PAGE 3 / 16 ===== Figure 1: Representation manifolds in large language models: colours, years and dates. The first and third example show text embeddings obtained from OpenAI’s text-embedding-large-3 model from prompts relating to English names for colours and dates of the year, respectivly. The second example shows token activations from layer 7 of GPT2-small, which were studied in Engels et al. (2025). The token activations were processed via an SAE to extract a feature corresponding to years of the twentieth century as in Engels et al. (2025), and normalized to have norm one. For each example, we perform principal component analysis (PCA) to reduce the dimension to three and display the resulting point clouds from two perspectives. The embeddings of English names for colours are displayed in their respective colour value. Years are coloured from blue (1900) through green to yellow (1999), and dates are coloured from white (1st Janurary) through blue to black (1st July) through red and back to white. A metric space is simply a set equipped with a distance, and we find that it provides a simple yet highly expressive formal mathematical framework for discussing the abstract notion of a feature or concept. In particular, it allows us to readily talk about: 1. Atomic features: Zf a singleton set. 2. Hierarchical features: Zf a discrete set, df a tree distance. 3. Continuous features: Zf an interval (equipped with e.g. df(x, y) = |x −y|), a cir- cle (equipped with e.g. arc-length distance), multi-dimensional (equipped with e.g. the Euclidean distance), etc. We find that this formalism strikes a balance between the less expressive Euclidean and hyperspherical models often assumed in the learning theory literature (e.g. Zimmermann et al., 2021; Hyvärinen et al., 2024; Reizinger et al., 2025), and the more complicated and less accessible models which are often assumed in the disentanglement literature, such as Riemannian manifolds equipped with group structu...

## Problem

TBD in close-reading pass.

## Core Hypothesis

TBD in close-reading pass.

## Relation To Our Project

- Hidden-state geometry:
- Manifold / Riemannian / topology:
- Temporal dynamics / online detection:
- Faithful CoT / process faithfulness:
- Error awareness / self-correction:
- Length/position/confidence proxy risk:

## What Gap Remains For Us

TBD in synthesis pass.

## Useful Quotes / Exact Pointers

TBD: add short excerpts with page markers from `../texts/the-origins-of-representation-manifolds-in-large.txt`.
