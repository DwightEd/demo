# Muon Dynamics as a Spectral Wasserstein Flow

- **Local PDF filename**: `Muon Dynamics as a Spectral Wasserstein Flow.pdf`
- **Slug**: `muon-dynamics-as-a-spectral-wasserstein-flow`
- **Pages**: 42
- **Approx Words**: 20210
- **Auto Tags**: geometry;dynamics
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.620657

## Keyword Profile

- `flow`: 74
- `spectral`: 71
- `geometry`: 38
- `entropy`: 23
- `dimension`: 20
- `phase`: 8
- `topolog`: 5
- `geometric`: 3
- `curvature`: 1

## Abstract / Opening Summary

Gradient normalization stabilizes deep-learning optimization, and spectral normalizations are especially natural for matrix-shaped parameter blocks; Muon is the motivating example. We study an idealized deterministic, continuous-time, vanishing-momentum version of this idea in the mean-field regime, where wide models are represented by probability measures on parameter space. Starting from normalized matrix flows, we introduce Spectral Wasserstein distances indexed by norms γ on positive semidefinite matrices: the trace norm gives classical W2, the operator norm gives the Muon geometry, and Schatten norms interpolate between them. We develop the static Kantorovich formulation, a max-min robust- cost representation, Gaussian reductions extending the Bures formula, and for monotone norms, prove equivalence with a Benamou–Brenier formulation. This yields a gradient-flow interpretation of the mean-field normalized training dynamics. We illustrate these findings by numerical experiments on MMD flows, Gaussian reductions, two-layer ReLU models, and shallow attention. 1

## Method / Algorithms Extract

methods. The recent framework of Pethick et al. [2025] is particularly relevant because it treats norm-constrained linear minimization oracles as a general language for normalized gradient methods and includes spectral normalizations as special cases. Earlier works such as Cutkosky and Mehta [2020] and Murray et al. [2019] show how normalized gradient methods change the optimization dynamics even in nonconvex settings. For deep architectures, matrix-aware normalizations are especially natural: Shampoo is an early influential example of tensor/matrix-aware preconditioning [Gupta et al., 2018], and Muon has become a leading example of spectral normalization in large-scale training [Jordan et al., 2024, Liu et al., 2025]. This connects naturally with the mean-field description of wide neural networks through probability measures on parameter space, which underlies the landscape analysis of two-layer networks by Mei et al. [2018], the optimal-transport convergence analysis of over-parameterized models by Chizat and Bach [2018], and the metric-gradient-flow viewpoint developed in Ambrosio et al. [2008]. Our work keeps this mean-field perspective but changes the underlying metric from Euclidean Wasserstein geometry to matrix-aware Spectral Wasserstein geometries. Generalized optimal distances. On the transport side, our work is closest in spirit to generalized coupling costs and weak transport, as developed for instance by Gozlan et al. [2017], Backhoff-Veraguas et al. [2019], and Backhoff-Veraguas and Pammer [2022]. It is also related to covariance-dependent transport costs [Burger et al., 2025], to matrix-valued optimal transport [Ning et al., 2015, Chen et al., 2018], and to the Bures–Wasserstein geometry of covariance matrices [Bhatia et al., 2019]. Another useful perspective is to robustify the Wasserstein distance by optimizing over the cost itself. The closest antecedent is the 1 arXiv:2604.04891v2 [math.OC] 8 May 2026 ===== PAGE 2 / 42 ===== subspace robust Wasserstein distance of Paty and Cuturi [2019], which corresponds to the special case of our spectral Wasserstein distance Wγ obtained when γ is a Ky Fan norm [Fan, 1951]. Maximizing over costs is also connected to metric learning in Wasserstein discriminant analysis [Flamary et al., 2018], ground metric learning [Cuturi and Avis, 2014], and congestion models for transportation networks [Carlier et al., 2008]. The opposite direction, minimizing over costs, appears for instance, in Sebbouh et al. [2024]; this leads to a concave minimization problem useful for Gromov–Wasserstein-type structure. By con...

## Experiments / Evidence Extract

experiments on MMD flows, Gaussian reductions, two-layer ReLU models, and shallow attention. 1 Introduction Spectrally normalized optimizers such as Muon replace the Euclidean geometry of gradient descent by a matrix-aware geometry adapted to block-shaped parameters. This paper presents a mathematical framework for these optimizers in the setting of very wide, possibly infinitely wide, layers by introducing a Spectral Wasserstein geometry that models their dynamics as mean-field gradient flows. Normalized gradient descent and mean field dynamics. On the optimization side, a growing literature studies normalized first-order methods. The recent framework of Pethick et al. [2025] is particularly relevant because it treats norm-constrained linear minimization oracles as a general language for normalized gradient methods and includes spectral normalizations as special cases. Earlier works such as Cutkosky and Mehta [2020] and Murray et al. [2019] show how normalized gradient methods change the optimization dynamics even in nonconvex settings. For deep architectures, matrix-aware normalizations are especially natural: Shampoo is an early influential example of tensor/matrix-aware preconditioning [Gupta et al., 2018], and Muon has become a leading example of spectral normalization in large-scale training [Jordan et al., 2024, Liu et al., 2025]. This connects naturally with the mean-field description of wide neural networks through probability measures on parameter space, which underlies the landscape analysis of two-layer networks by Mei et al. [2018], the optimal-transport convergence analysis of over-parameterized models by Chizat and Bach [2018], and the metric-gradient-flow viewpoint developed in Ambrosio et al. [2008]. Our work keeps this mean-field perspective but changes the underlying metric from Euclidean Wasserstein geometry to matrix-aware Spectral Wasserstein geometries. Generalized optimal distances. On the transport side, our work is closest in spirit to generalized coupling costs and weak transport, as developed for instance by Gozlan et al. [2017], Backhoff-Veraguas et al. [2019], and Backhoff-Veraguas and Pammer [2022]. It is also related to covariance-dependent transport costs [Burger et al., 2025], to matrix-valued optimal transport [Ning et al., 2015, Chen et al., 2018], and to the Bures–Wasserstein geometry of covariance matrices [Bhatia et al....

## Conclusion / Discussion Extract

The main message of this paper is that matrix-normalized optimizers for mean-field neural models are naturally encoded by Spectral Wasserstein distances. In this view, matrix normalization is not merely a preconditioning device, but the local steepest-descent geometry generated by a spectral transport cost. This clarifies the transport side, through coupling-based costs, geodesics, and Gaussian reductions, and the 9 ===== PAGE 10 / 42 ===== optimization side, through a continuum interpretation of normalized matrix updates. The resulting picture is that the choice of spectral gauge shapes which directions of motion are emphasized or suppressed in the mean-field dynamics, thereby providing a geometric language for comparing matrix-normalized optimizers. A major open question is the rigorous study of the Spectral Wasserstein gradient-flow PDE: global-in-time existence, stability, and the avoidance of local minima for shallow neural-network and MMD losses. Acknowledgement The author thanks Tony Silveti-Falls, from whom he learned about the intricate details of Muon and with whom he had insightful discussions about Newton–Schulz approximation. This work was supported by the European Research Council (ERC project WOLF) and the French government under the management of Agence Nationale de la Recherche as part of the “France 2030” program, reference ANR-23-IACL-0008 (PRAIRIE-PSAI). References Luigi Ambrosio, Nicola Gigli, and Giuseppe Savaré. Gradient Flows in Metric Spaces and in the Space of Probability Measures. Lectures in Mathematics ETH Zürich. Birkhäuser Basel, 2 edition, 2008. Julio Daniel Backhoff-Veraguas and Gudmund Pammer. Applications of weak transport theory. Bernoulli, 28 (1):370–394, 2022. Julio Daniel Backhoff-Veraguas, Mathias Beiglböck, and Gudmund Pammer. Existence, duality, and cyclical monotonicity for weak transport costs. Calculus of Variations and Partial Differential Equations, 58(6): 203, 2019. Jean-David Benamou and Yann Brenier. A computational fluid mechanics solution to the Monge–Kantorovich mass transfer problem. Numerische Mathematik, 84(3):375–393, 2000. Rajendra Bhatia, Tanvi Jain, and Yongdo Lim. On the Bures–Wasserstein distance betw...

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

TBD: add short excerpts with page markers from `../texts/muon-dynamics-as-a-spectral-wasserstein-flow.txt`.
