# Geometry-First Ideas Beyond GeoFaith: Reasoning Spectral Fields

Date: 2026-07-09

## Hard Constraint For This Round

Do not start from prefix-locking, constraint edits, or the previous spread / kappa / transition-z line. The method must be geometrically native: the main object should be a manifold field, spectral field, curvature field, transport connection, Hodge decomposition, geodesic deviation, or topological obstruction. Scalar scores may exist only as final summaries of a richer geometric object.

## One-Paragraph Paper Story

We argue that chain-of-thought failure is not merely a pointwise abnormality in hidden states and not merely a high-curvature path. A reasoning trace forms a **hidden reasoning surface** over depth and time: layers transform the current state vertically, while generated steps move it horizontally. Faithful reasoning is a spectrally organized surface: its hidden-state energy progressively condenses into low-frequency solution modes, and transport along depth and step directions is approximately integrable. At the first real reasoning failure, this surface develops a **spectral phase defect**: high-frequency energy leaks into local modes, depth-time transport stops commuting, and nearby semantically equivalent trajectories form caustics. We detect these defects with a Reasoning Spectral Field rather than scalar spread, giving step-level and response-level error detection while explaining why smooth but wrong chains can evade existing geometry methods.

## Flagship Method: RSF, Reasoning Spectral Field

### Core Hypothesis

Correct reasoning is not simply "less dispersed." It behaves like **spectral annealing** on a learned hidden-state manifold:

\[
\mathcal{E}_{t}(k_{\mathrm{high}})\downarrow,\qquad
\mathcal{E}_{t}(k_{\mathrm{solution}})\uparrow
\]

where \(k_{\mathrm{high}}\) are high-frequency local variation modes and \(k_{\mathrm{solution}}\) are low-frequency modes associated with stable solution basins.

Wrong reasoning has one of two signatures:

1. **Spectral leakage**: energy bursts into high-frequency local modes before or at the first error.
2. **Wrong-mode locking**: energy condenses smoothly, but into a basin/mode family associated with wrong continuations rather than the correct solution manifold.

This directly addresses the weakness of spread: a hard correct step can move a lot, and a wrong step can be smooth. The question becomes not "how much does it move?" but "which manifold frequencies and modes carry the movement?"

## Geometry Object

Let \(h_{\ell,t}\in\mathbb{R}^{d}\) be the hidden state at layer \(\ell\) and reasoning step/token \(t\). Build a hidden-state manifold \(\mathcal{M}\) from training traces using a local kernel:

\[
K(h_i,h_j)=
\exp\left(-\frac{\|W(h_i-h_j)\|_2^2}{\tau_i\tau_j}\right)
\]

where \(W\) is a whitening or layer-normalized projection and \(\tau_i\) is a local bandwidth. Define the graph Laplacian:

\[
L = I - D^{-1/2}KD^{-1/2}
\]

Compute eigenfunctions:

\[
L\phi_k=\lambda_k\phi_k
\]

For a new hidden state, use Nyström extension:

\[
\phi_k(h)=
\frac{1}{\lambda_k}
\sum_{j}K(h,h_j)\phi_k(h_j)
\]

The reasoning trace becomes a spectral field:

\[
a_{\ell,t,k}=\phi_k(h_{\ell,t})
\]

\[
\mathcal{E}_{\ell,t}(k)=|a_{\ell,t,k}|^2
\]

The model input is not a scalar. It is the tensor:

\[
\mathcal{E}\in\mathbb{R}^{L_{\mathrm{layers}}\times T_{\mathrm{steps}}\times K_{\mathrm{modes}}}
\]

This can be treated like a spectrogram of reasoning.

## Key Geometric Features

### 1. Spectral Annealing

Correct reasoning should move energy from local/noisy modes toward coherent low-frequency modes:

\[
A_{\mathrm{low}}(t)=
\sum_{k\le K_0}\mathcal{E}_{t}(k)
\]

\[
A_{\mathrm{high}}(t)=
\sum_{k>K_0}\mathcal{E}_{t}(k)
\]

But these are not final detectors. The useful object is the full trajectory:

\[
t\mapsto
\left[
\mathcal{E}_{t}(1),
\mathcal{E}_{t}(2),
\dots,
\mathcal{E}_{t}(K)
\right]
\]

The detector should learn the shape of energy transfer, not threshold a mean.

### 2. Spectral Flux

Define frequency-wise flow:

\[
J_t(k)=\mathcal{E}_{t+1}(k)-\mathcal{E}_{t}(k)
\]

Define cumulative high-frequency leakage:

\[
\Pi_t(K_0)=
\sum_{k>K_0}
\left[
\mathcal{E}_{t+1}(k)-\mathcal{E}_{t}(k)
\right]
\]

Failure is not just high \(A_{\mathrm{high}}\). It is an anomalous flux pattern: sudden upward transfer, backscatter, or failure to anneal after an uncertainty burst.

### 3. Wrong-Mode Locking

To catch smooth wrong chains, learn class-conditional or outcome-conditional spectral templates:

\[
\bar{\mathcal{E}}^{+}_{s}(k)
=
\mathbb{E}[\mathcal{E}_{t}(k)\mid \text{correct},\ \mathrm{phase}(t)=s]
\]

\[
\bar{\mathcal{E}}^{-}_{s}(k)
=
\mathbb{E}[\mathcal{E}_{t}(k)\mid \text{wrong},\ \mathrm{phase}(t)=s]
\]

Then detect whether the current chain is spectrally approaching the correct solution family or a wrong attractor:

\[
D^{+}_{t}=
\mathrm{OT}\left(\mathcal{E}_{t}(\cdot),
\bar{\mathcal{E}}^{+}_{s(t)}(\cdot)\right)
\]

\[
D^{-}_{t}=
\mathrm{OT}\left(\mathcal{E}_{t}(\cdot),
\bar{\mathcal{E}}^{-}_{s(t)}(\cdot)\right)
\]

\[
\mathrm{WrongLock}(t)=
D^{+}_{t}-D^{-}_{t}
\]

This makes smooth confident wrong reasoning visible as convergence to the wrong spectral basin, rather than as high spread.

## Stronger Geometry: Depth-Time Curvature Field

GeoFaith mainly treats the reasoning path as a trajectory. The stronger view is that hidden reasoning is a **surface** over layer and step:

\[
(\ell,t)\mapsto h_{\ell,t}
\]

There are two transport directions:

1. Depth transport: layer \(\ell\rightarrow \ell+1\).
2. Step transport: reasoning step \(t\rightarrow t+1\).

If reasoning is coherent, depth-then-step and step-then-depth should approximately agree. First errors can be framed as non-commutativity defects.

### Local Tangent Frames

For each \((\ell,t)\), estimate a local tangent basis \(U_{\ell,t}\) using local PCA on neighbors of \(h_{\ell,t}\) in \(\mathcal{M}\).

Depth transport:

\[
T^{d}_{\ell,t}
=
U_{\ell+1,t}^{\top}U_{\ell,t}
\]

Step transport:

\[
T^{s}_{\ell,t}
=
U_{\ell,t+1}^{\top}U_{\ell,t}
\]

### Curvature Defect

Compare the two routes around a small layer-step square:

\[
\Omega_{\ell,t}
=
T^{s}_{\ell+1,t}T^{d}_{\ell,t}
-
T^{d}_{\ell,t+1}T^{s}_{\ell,t}
\]

The scalar norm:

\[
\|\Omega_{\ell,t}\|_F
\]

is only a visualization. The real signal is the field:

\[
\Omega\in
\mathbb{R}^{L_{\mathrm{layers}}\times T_{\mathrm{steps}}\times r\times r}
\]

where \(r\) is the local tangent dimension.

### Hypothesis

Correct reasoning has low, structured curvature defects: transformations through depth and transformations through reasoning time are compatible.

Wrong reasoning creates **phase slips**:

\[
\Omega_{\ell,t}\ \text{spikes before or at first error}
\]

especially in middle-to-late layers where semantic computation crystallizes.

This is a stronger geometry claim than "wrong trajectories are more scattered." It says reasoning failure is a loss of integrability in the hidden reasoning surface.

## Combined Flagship: RSF-C, Spectral Curvature Field

The main model should combine:

\[
\mathcal{E}_{\ell,t}(k)
\quad\text{and}\quad
\Omega_{\ell,t}
\]

into a field-level detector:

\[
g_{\theta}
\left(
\mathcal{E}_{1:L,1:t,1:K},
\Omega_{1:L,1:t}
\right)
\rightarrow
\lambda_t
\]

where \(\lambda_t\) is the hazard of a first reasoning failure at step \(t\).

Response risk uses survival aggregation:

\[
P(\mathrm{error\ by\ }T)
=
1-\prod_{t=1}^{T}(1-\lambda_t)
\]

This avoids mean/max dilution and preserves local defects.

## Why This Can Beat GeoFaith

### GeoFaith

GeoFaith says faithful and unfaithful reasoning occupy different spatio-temporal latent geometry and uses geometry plus uncertainty to detect faithfulness.

### RSF-C Difference

RSF-C says the actual object is not a path but a **spectral-curvature field** over depth and time:

\[
\text{CoT path}
\quad\Rightarrow\quad
\text{hidden reasoning surface}
\quad\Rightarrow\quad
\text{spectral energy + curvature defects}
\]

It can catch:

1. Long but correct hard steps: high motion, but healthy spectral flux.
2. Smooth but wrong steps: low motion, but wrong-mode spectral locking.
3. Early hidden inconsistency: curvature defect appears before textual error.
4. Overthinking loops: Hodge-curl component rises without solution-mode annealing.
5. Shortcut reasoning: premature low-frequency collapse into a shallow wrong basin.

## Even More Ambitious Extension: Hodge Reasoning Flow

Represent hidden updates as a vector field on \(\mathcal{M}\):

\[
v_t=h_{t+1}-h_t
\]

Use graph Hodge decomposition:

\[
v
=
\nabla \phi
+ \delta \psi
+ h_{\mathrm{harm}}
\]

where:

1. \(\nabla\phi\): gradient flow, interpreted as progress toward a solution basin.
2. \(\delta\psi\): curl flow, interpreted as circular or self-referential reasoning.
3. \(h_{\mathrm{harm}}\): harmonic flow, interpreted as global ambiguity or unresolved constraints.

### Hypothesis

Correct reasoning is gradient-dominated near the answer:

\[
\frac{\|\nabla\phi\|^2}
{\|v\|^2}
\uparrow
\]

Wrong or unfaithful reasoning shows rising curl or harmonic residue:

\[
\frac{\|\delta\psi\|^2+\|h_{\mathrm{harm}}\|^2}
{\|v\|^2}
\uparrow
\]

This gives a genuinely geometric explanation of why some chains "look busy" but make no progress: they are circulating on the manifold rather than descending toward the solution basin.

## Geodesic-Deviation Variant: Reasoning Caustics

Sample semantically equivalent variants of the same problem, or multiple correct paraphrases. Each produces a nearby trajectory:

\[
\gamma_i(t)
\]

The separation vector between nearby trajectories is a Jacobi-like field:

\[
J_{ij}(t)=\gamma_i(t)-\gamma_j(t)
\]

On a Riemannian manifold, geodesic deviation is governed by:

\[
\nabla_{\dot{\gamma}}^2J
+R(J,\dot{\gamma})\dot{\gamma}
=0
\]

We do not need to fully estimate \(R\). We can estimate empirical deviation:

\[
\mathrm{Dev}(t)=
\mathbb{E}_{i,j}
\left[
\|J_{ij}(t)\|_2
\right]
\]

and, more importantly, deviation acceleration:

\[
\mathrm{AccDev}(t)=
\mathrm{Dev}(t+1)-2\mathrm{Dev}(t)+\mathrm{Dev}(t-1)
\]

### Hypothesis

Faithful reasoning contracts paraphrase bundles toward the same solution channel. Wrong reasoning creates **caustics**: initially nearby trajectories suddenly separate into different basins.

This is useful because it does not depend on raw length. It asks whether equivalent prompts remain geometrically equivalent through the reasoning process.

## Potential Paper Titles

1. **Reasoning Spectral Fields: Detecting Chain-of-Thought Failures as Hidden-Manifold Phase Defects**
2. **Beyond Trajectories: Spectral Curvature Fields for Faithful Chain-of-Thought Detection**
3. **When Reasoning Surfaces Tear: Hidden-State Spectral Geometry for Error Localization**
4. **HodgeFlow: Decomposing Hidden Reasoning Dynamics into Progress, Circulation, and Ambiguity**
5. **Reasoning Caustics: Geodesic Deviation Reveals Chain-of-Thought Failure**

## Recommended Main Direction

The best flagship is:

> **Reasoning Spectral Curvature Fields**

because it has all of the following:

1. It is geometry-first, not a repackaged control or prefix idea.
2. It naturally uses full hidden trajectories across layers and steps.
3. It yields visualizations that can look genuinely new: spectral heatmaps, layer-step curvature fields, phase-slip maps.
4. It explains why old scalar geometry fails: scalars collapse the field.
5. It gives a strong "beyond GeoFaith" claim: GeoFaith studies spatio-temporal geometry; RSF-C studies spectral-curvature defects of the reasoning surface.

## Minimal Implementation Plan

### Stage 1: Data Tensor Construction

For each response, extract:

\[
H_i\in\mathbb{R}^{L\times T_i\times d}
\]

where \(L\) is number of layers, \(T_i\) is number of reasoning steps or selected token checkpoints, and \(d\) is hidden dimension.

Normalize within layer:

\[
\tilde{h}_{\ell,t}
=
\frac{h_{\ell,t}-\mu_{\ell}}{\sigma_{\ell}+\epsilon}
\]

Use length-matched and phase-normalized indexing:

\[
s(t)=\frac{t}{T_i}
\]

to avoid repeating the previous length-proxy mistake.

### Stage 2: Manifold Eigenbasis

Build kNN graph from sampled hidden states, stratified by layer and phase. Compute graph Laplacian eigenvectors. Use Nyström extension for held-out states.

Keep the first \(K\) modes:

\[
K\in\{32,64,128\}
\]

### Stage 3: Spectral Field

Convert every chain to:

\[
\mathcal{E}_i\in\mathbb{R}^{L\times T_i\times K}
\]

Train a small field model:

1. 2D CNN over \(T\times K\) for each layer.
2. Cross-layer attention.
3. Survival head for first-error hazard.

### Stage 4: Curvature Field

Estimate local tangent frames \(U_{\ell,t}\). Compute \(\Omega_{\ell,t}\) and feed:

\[
\|\Omega_{\ell,t}\|_F,\quad
\mathrm{eig}(\Omega_{\ell,t}^{\top}\Omega_{\ell,t})
\]

plus optionally the raw small matrix if dimension allows.

### Stage 5: Hodge Flow Add-On

Build a transition graph over hidden states. Edges are step transitions. Decompose flow into gradient, curl, and harmonic components. Add these as a second field.

## Essential Experiments

### Detection

1. First-error AUROC/AUPRC.
2. Response-level AUROC/AUPRC with survival aggregation.
3. Length-matched and step-position-matched evaluation.
4. Hard-step matched evaluation.
5. Smooth-wrong subset.
6. Long-correct subset.

### Mechanism

1. Show spectral phase defect precedes first textual error.
2. Show wrong-mode locking handles cases where spread is low.
3. Show curvature defects concentrate in semantically meaningful middle/late layers.
4. Show Hodge-curl rises in circular/overthinking cases.
5. Patch hidden states around high-curvature defects and test if future error probability changes.

### Against GeoFaith

Report:

1. Overall AUC.
2. First-error localization.
3. Response-level risk.
4. Smooth-wrong recall.
5. Length-matched AUC.
6. Cross-task transfer.
7. Visualization quality and interpretability.

The key win condition should not be only "higher AUC." It should be:

> RSF-C explains failure modes where GeoFaith-style geometry is ambiguous: high-motion correct computation and smooth confident wrong reasoning.

## Ideas To Keep But Not Lead With

### 1. Morse-Smale Reasoning Landscape

Learn an energy landscape \(V(h)\) on \(\mathcal{M}\). Correct reasoning follows gradient descent toward a correct basin; wrong reasoning crosses a separatrix into a wrong basin.

\[
\frac{d}{dt}V(h_t)<0
\]

is healthy only if the basin is correct. This can explain premature convergence.

Risk: harder to estimate robustly and may look like metric learning if not carefully geometric.

### 2. Topological Persistence Of Reasoning Fields

Compute persistent homology over hidden-state point clouds within a chain or across paraphrase bundles. Reasoning failure corresponds to creation or destruction of holes/loops in the representation field.

Risk: visual and interesting, but reviewers may ask why topology directly predicts correctness.

### 3. Schrödinger Bridge Reasoning

Model correct reasoning as an entropic optimal transport bridge from question manifold to answer manifold. Wrong reasoning has high bridge action or follows a bridge to a wrong answer manifold.

Risk: close to existing transport-style papers unless combined with spectral fields.

## Bottom Line

If we want a geometry-first method with a real chance to feel new, the best bet is:

\[
\boxed{
\text{Reasoning failure is a spectral phase defect on a hidden reasoning surface.}
}
\]

That statement is much stronger and more memorable than:

\[
\text{wrong reasoning is more dispersed.}
\]

It also gives concrete objects to compute:

\[
\mathcal{E}_{\ell,t}(k)
\quad
\Omega_{\ell,t}
\quad
v=\nabla\phi+\delta\psi+h_{\mathrm{harm}}
\]

These are not toy scalar proxies. They are field-level geometric structures from which detection, localization, visualization, and mechanism experiments can all be derived.
