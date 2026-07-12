# A theory of multineuronal dimensionality, dynamics and measurement

- **Local PDF filename**: `Atheory of multineuronal dimensionality, dynamics and measurement.pdf`
- **Slug**: `atheory-of-multineuronal-dimensionality-dynamics-and-measurement`
- **Pages**: 50
- **Approx Words**: 32411
- **Auto Tags**: geometry;dynamics;emergence
- **Verification Status**: local-pdf-unverified-metadata
- **Read Status**: full-text-extracted; preliminary-section-read; needs exact quote pass
- **Last Updated**: 2026-07-09T01:56:48.604228

## Keyword Profile

- `dimension`: 409
- `manifold`: 98
- `trajectory`: 31
- `geometry`: 15
- `geometric`: 13
- `entropy`: 8
- `curvature`: 7
- `latent`: 4
- `phase`: 1
- `transition`: 1
- `causal`: 1

## Abstract / Opening Summary

19 In many experiments, neuroscientists tightly control behavior, record many trials, and obtain trial-averaged 20 ﬁring rates from hundreds of neurons in circuits containing billions of behaviorally relevant neurons. Di- 21 mensionality reduction methods reveal a striking simplicity underlying such multi-neuronal data: they can 22 be reduced to a low-dimensional space, and the resulting neural trajectories in this space yield a remarkably 23 insightful dynamical portrait of circuit computation. This simplicity raises profound and timely conceptual 24 questions. What are its origins and its implications for the complexity of neural dynamics? How would the 25 situation change if we recorded more neurons? When, if at all, can we trust dynamical portraits obtained 26 from measuring an inﬁnitesimal fraction of task relevant neurons? We present a theory that answers these 27 questions, and test it using physiological recordings from reaching monkeys. This theory reveals conceptual 28 insights into how task complexity governs both neural dimensionality and accurate recovery of dynamic 29 portraits, thereby providing quantitative guidelines for future large-scale experimental design. 30 ∗sganguli@stanford.edu 1 . CC-BY-NC-ND 4.0 International license under a not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available The copyright holder for this preprint (which was this version posted November 12, 2017. ; https://doi.org/10.1101/214262 doi: bioRxiv preprint ===== PAGE 2 / 50 ===== 1

## Method / Algorithms Extract

methods reveal a striking simplicity underlying such multi-neuronal data: they can 22 be reduced to a low-dimensional space, and the resulting neural trajectories in this space yield a remarkably 23 insightful dynamical portrait of circuit computation. This simplicity raises profound and timely conceptual 24 questions. What are its origins and its implications for the complexity of neural dynamics? How would the 25 situation change if we recorded more neurons? When, if at all, can we trust dynamical portraits obtained 26 from measuring an inﬁnitesimal fraction of task relevant neurons? We present a theory that answers these 27 questions, and test it using physiological recordings from reaching monkeys. This theory reveals conceptual 28 insights into how task complexity governs both neural dimensionality and accurate recovery of dynamic 29 portraits, thereby providing quantitative guidelines for future large-scale experimental design. 30 ∗sganguli@stanford.edu 1 . CC-BY-NC-ND 4.0 International license under a not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available The copyright holder for this preprint (which was this version posted November 12, 2017. ; https://doi.org/10.1101/214262 doi: bioRxiv preprint ===== PAGE 2 / 50 ===== 1 Introduction 31 In this work, we aim to address a major conceptual elephant residing within almost all studies in mod- 32 ern systems neurophysiology. Namely, how can we record on the order of hundreds of neurons in regions 33 deep within the brain, far from the sensory and motor peripheries, like mammalian hippocampus, or pre- 34 frontal, parietal, or motor cortices, and obtain scientiﬁcally interpretable results that relate neural activity 35 to behavior and cognition? Our apparent success at this endeavor seems absolutely remarkable, consid- 36 ering such circuits mediating complex sensory, motor and cognitive behaviors contain O(106) to O(109) 37 neurons [Shepherd, 2004] - 4 to 7 orders of magnitude more than we currently record. Or alternatively, we 38 could be completely misleading ourselves: perhaps we should not trust scientiﬁc conclusions drawn from 39 statistical analyses of so few neurons, as such conclusions might become qualitatively different as we record 40 more. Without an adequate theory of neural measurement, it is impossible to quantitatively adjudicate where 41 systems neuroscience currently stands between these two extreme scenarios of success and failure. 42 One potential solution is an experimental one: simply wait un...

## Experiments / Evidence Extract

experiments, neuroscientists tightly control behavior, record many trials, and obtain trial-averaged 20 ﬁring rates from hundreds of neurons in circuits containing billions of behaviorally relevant neurons. Di- 21 mensionality reduction methods reveal a striking simplicity underlying such multi-neuronal data: they can 22 be reduced to a low-dimensional space, and the resulting neural trajectories in this space yield a remarkably 23 insightful dynamical portrait of circuit computation. This simplicity raises profound and timely conceptual 24 questions. What are its origins and its implications for the complexity of neural dynamics? How would the 25 situation change if we recorded more neurons? When, if at all, can we trust dynamical portraits obtained 26 from measuring an inﬁnitesimal fraction of task relevant neurons? We present a theory that answers these 27 questions, and test it using physiological recordings from reaching monkeys. This theory reveals conceptual 28 insights into how task complexity governs both neural dimensionality and accurate recovery of dynamic 29 portraits, thereby providing quantitative guidelines for future large-scale experimental design. 30 ∗sganguli@stanford.edu 1 . CC-BY-NC-ND 4.0 International license under a not certified by peer review) is the author/funder, who has granted bioRxiv a license to display the preprint in perpetuity. It is made available The copyright holder for this preprint (which was this version posted November 12, 2017. ; https://doi.org/10.1101/214262 doi: bioRxiv preprint ===== PAGE 2 / 50 ===== 1 Introduction 31 In this work, we aim to address a major conceptual elephant residing within almost all studies in mod- 32 ern systems neurophysiology. Namely, how can we record on the order of hundreds of neurons in regions 33 deep within the brain, far from the sensory and motor peripheries, like mammalian hippocampus, or pre- 34 frontal, parietal, or motor cortices, and obtain scientiﬁcally interpretable results that relate neural activity 35 to behavior and cognition? Our apparent success at this endeavor seems absolutely remarkable, consid- 36 ering such circuits mediating complex sensory, motor and cognitive behaviors contain O(106) to O(109) 37 neurons [Shepherd, 2004] - 4 to 7 orders of magnitude more than we currently record. Or alternatively, we 38 could be completely misleading ourselves: perhaps we sh...

## Conclusion / Discussion Extract

395 6.1 An intuitive summary of our theory 396 Overall, we have generated a quantitative theory of trial averaged neural dimensionality, dynamics, and 397 measurement that can impact both the interpretation of past experiments, and the design of future ones. Our 398 theory provides both quantitative and conceptual insights into the underlying nature of two major order of 399 magnitude discrepancies dominating almost all experiments in systems neuroscience: (1) the dimensionality 400 of neural state space dynamics is often orders of magnitude smaller than the number of recorded neurons 401 (e.g. Fig. 2), and (2) the number of recorded neurons is orders of magnitude smaller than the total num- 402 ber of relevant neurons in a circuit, yet we nevertheless claim to make scientiﬁc conclusions from such 403 inﬁnitesimally small numbers of recorded neurons. This latter discrepancy is indeed troubling, as it calls 404 into question whether or not systems neuroscience has been a success or a failure, even within the relatively 405 circumscribed goal of correctly recovering trial-averaged neural state space dynamics in such an undersam- 406 pled measurement regime. To address this fundamental ambiguity, our theory identiﬁes and weaves together 407 diverse aspects of experimental design and neural dynamics, including the number of recorded neurons, the 408 total number of neurons in a relevant circuit, the number of task parameters, the volume of the manifold of 409 task parameters, and the smoothness of neural dynamics, into quantitative scaling laws determining bounds 410 on the dimensionality and accuracy of neural state space dynamics recovered from large scale recordings. 411 In particular, we address both order of magnitude discrepancies by taking a geometric viewpoint in 412 which trial-averaged neural data is fundamentally an embedding of a task manifold into neural ﬁring rate 413 space (Fig 3EF), yielding a neural state space dynamical portrait of circuit computation during the task. We 414 explain the ﬁrst order of magnitude discrepancy by carefully considering how the complexity of the task, 415 as measured by the volume of the task manifold, and the smoothness ...

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

TBD: add short excerpts with page markers from `../texts/atheory-of-multineuronal-dimensionality-dynamics-and-measurement.txt`.
