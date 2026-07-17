# What Residual-Stream Geometry Means Here

## The observed object

For a prompt prefix ending at token position \(t\), let

\[
h_{\ell,t}\in\mathbb R^d
\]

be the residual state after depth \(\ell\). A single prompt therefore produces
a layer-by-token field, not one point cloud:

\[
\mathcal H=\{h_{\ell,t}:\ell=0,\ldots,L,\ t=1,\ldots,T\}.
\]

Euclidean statistics on \(\mathcal H\) are representation statistics. They
become computational geometry only after specifying which latent variable the
states should represent and which update the model should implement.

## Four distinct meanings of geometry

### 1. Extrinsic activation geometry

Distances, angles, singular values, effective rank, LID, curvature, and spread
are calculated directly in \(\mathbb R^d\). They can describe condensation or
anisotropy, but they do not identify the represented computation. They are also
sensitive to token count, layer norm, lexical content, and basis choice.

### 2. Latent belief geometry

Suppose a prompt determines an exact posterior \(b_t(x)\). A representation map
\(g_\ell\) would satisfy

\[
h_{\ell,t}\approx g_\ell(b_t).
\]

The question is then whether held-out states preserve neighborhoods,
distinctions, and query consequences of \(b_t\). In this project the posterior
is uniform on an affine subset of \(\mathbb F_p^n\), and its coordinates are the
finite Fourier characters

\[
\Phi(b_t)=\left(\mathbb E_{x\sim b_t}
e^{2\pi i k^\top x/p}\right)_{k\in\mathbb F_p^n}.
\]

This chart is known before looking at model activations. It is not chosen to
separate successful and failed examples.

### 3. Update geometry

Evidence changes the exact belief by

\[
\Delta\Phi_t^\star=\Phi(b_t)-\Phi(b_{t-1}).
\]

For a cross-fitted local affine chart \(D_\ell\), a residual write
\(\Delta h_{\ell,t}\) has the induced belief-coordinate direction

\[
\delta\widehat\Phi_{\ell,t}=J_{D_\ell}\Delta h_{\ell,t}.
\]

The meaningful directional test is not whether \(\Delta h\) is large or curved.
It is whether \(\delta\widehat\Phi\) aligns with the analytically required
\(\Delta\Phi_t^\star\) and not with a matched wrong update.

### 4. Routing geometry

For evidence-token set \(E\), attention head \(a\) routes the source-specific
value component

\[
u_{\ell a t}^{E}=\sum_{s\in E}\alpha_{\ell a t s}v_{\ell a s},
\qquad
w_{\ell a t}^{E}=W_{O,\ell}^{(a)}u_{\ell a t}^{E}.
\]

Projecting \(w_{\ell a t}^{E}\) through the held-out belief chart tests whether
an evidence route actually writes the required belief update. Attention mass
alone cannot establish this because a heavily attended token may contribute a
small, canceling, or irrelevant value vector.

## Primary quantities and what they establish

| quantity | definition | evidential role |
|---|---|---|
| Fourier \(R^2\) | held-out reconstruction of \(\Phi(b_t)\) | representation fidelity |
| paired target accuracy | own branch posterior is closer than opposite branch | alias distinction |
| future-query NLL | Fourier inversion on an unseen query direction | task-relevant belief content |
| conditional usable bits | future NLL reduction over logits plus controls | information beyond output |
| update alignment margin | cosine to true update minus cosine to opposite update | directional mechanism |
| routed update score | evidence attention mass times update margin | source-specific mediation |
| donor patch log-odds shift | patched donor-vs-recipient answer margin | causal effect |

## Why predictive aliasing is necessary

Without an alias, a hidden-state decoder can recover the answer simply because
the answer already appears in the current logits. The controlled pair enforces

\[
P(Y_{\mathrm{current}}\mid C_0)
=P(Y_{\mathrm{current}}\mid C_1),
\]

while

\[
P(Y_{\mathrm{future}}\mid C_0)
\ne P(Y_{\mathrm{future}}\mid C_1).
\]

Therefore a positive held-out future-information increment is evidence for a
latent distinction, not merely another readout of current confidence.

## What is not claimed

- A low-dimensional PCA plot is not evidence of a manifold.
- Lower spread is not automatically higher certainty or correctness.
- A linear chart is a measurement instrument, not the proposed mechanism.
- Attention weights are not residual contributions.
- Decodability is not causality.
- Passing the controlled task does not prove that all natural-language
  reasoning follows Bayesian updates.

The ProcessBench transfer stage is allowed only after representation, routing,
and causal gates pass in the controlled system.
