"""Feature-extraction package (refactor, 2026-06-09).

Three families, all extracted by teacher-forcing a fixed (prompt, text):

  uncertainty.py   - paper "Tracing Uncertainty" trace channel, PER TOKEN:
                     U_D (full-vocab entropy), U_C (committal p(1-p)),
                     U_E (squared gradient norm, isotropic; one backward / token).
  geometry.py      - our own activation-degree features, RAW (un-standardized):
                     norm + Renyi effective-dim family {PR, AE, ed_half, E90}
                     + AE_robust + anom_count. Computed on the per-step
                     exp-pooled vector (Lu et al. 2601.02170 Eq.6) AND per token.
  trace_profile.py - paper Table 2 summary of any per-token/per-step series:
                     mu_early (first 25%), mu_mid (mid 50%), mu_late (last 25%),
                     slope m, linear fit r^2.

Reuses utils/{step_vector,step_boundaries,spectral}.py and 10's label matchers.
"""
