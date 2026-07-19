# Experiment Tracker

Updated: 2026-07-19 (Asia/Shanghai)

| ID | Claim / gate | Status | Evidence | Next action |
|---|---|---|---|---|
| M0-1 | Coherent-to-diffuse transition raises directional dispersion | passed-synthetic | deterministic CLI selftest | add low-rank and magnitude-only counterexamples |
| M0-2 | Multi-scale fields are token-causal | passed | future-perturbation regression test | extend invariant test to every saved causal field |
| M0-3 | Weighted dispersion formulas are numerically correct | passed-review | brute-force pair calculation; covariance identity test | retain as invariant |
| M0-4 | Sparse or unknown depths fail closed | passed | layer/provenance tests | retain as schema gate |
| M1-1 | Existing extraction manifests can be loaded | passed-local | streaming relative-shard loader and end-to-end legacy CLI test | run 20-chain server smoke test |
| M1-1b | Sparse selected depths are labeled honestly | passed-local | intervals `[8,10],...`; output kind is sparse pilot | do not interpret as single-block writes |
| M1-2 | Snapshots are raw residual-stream states | blocked-input | existing schema does not record provenance | update extractor, re-extract consecutive depths, verify hook reconstruction |
| M1-3 | Real activations show stable label-free evolution | pending | no real run yet | run only after M1-2 |
| M2-1 | Attention/MLP writes reconstruct block delta | pending | component conflict metric only | add architecture-specific hooks and reconstruction threshold |

Current conclusion: the numerical M0 is ready. Real-model mechanism claims are deliberately blocked until
the extraction provenance and consecutive-depth contract are satisfied.
