# Latent VAE + Separatrix Implementation Plan

Date: 2026-07-09

## Core Hypothesis

正确推理不是简单的低 spread 或短链代理，而是在 hidden-state 空间中靠近一个可学习的健康推理 chart；错误步骤不是任意发散，而是靠近该 chart 的 separatrix 或离开 chart 的高不确定区域。

形式上，先用 VAE 学一个无监督 chart:

$$
q_\phi(z\mid x)=\mathcal{N}(z;\mu_\phi(x), \operatorname{diag}(\sigma_\phi^2(x)))
$$

$$
p_\psi(x\mid z)=\mathcal{N}(x;\mu_\psi(z), \operatorname{diag}(\sigma_\psi^2(z)))
$$

再在同一个 latent chart 上学习错误风险:

$$
\lambda_t = \sigma(w^\top \mu_\phi(x_t)+b)
$$

最终联合目标为:

$$
\mathcal{L}
=
\mathcal{L}_{\mathrm{survival}}
+ \alpha_E \mathcal{L}_{\mathrm{energy}}
+ \alpha_C \mathcal{L}_{\mathrm{contrast}}
+ \alpha_N \mathcal{L}_{\mathrm{nuisance}}
+ \alpha_V
\left(
\mathcal{L}_{\mathrm{rec}}
+ \beta_t \mathcal{L}_{\mathrm{KL}}
\right)
$$

其中:

$$
\beta_t=\beta_{\max}\min\left(1,\frac{t}{T_{\mathrm{warm}}}\right)
$$

## Why This Replaces the Old Toy Signal

旧的 spread/kappa 类变量容易变成长度、步骤难度、语义切分方式的代理。新的实现不再把某个手工标量当作几何本体，而是让模型先学习一个 latent chart，然后报告三类证据:

1. `hazard`: 在 learned chart 上是否接近 first-error separatrix。
2. `vae_rec_nll`: 当前 hidden state 是否难以被健康 chart 重构。
3. `vae_kl / vae_dec_unc / vae_post_unc`: 当前状态在 latent posterior 和 decoder 中是否不稳定。

## Code Entry Points

Main combined implementation:

```bash
python latent_separatrix_audit.py data/features/full_gsm8k.npz \
  --output_dir outputs/latent_vae_separatrix_full_gsm8k \
  --tag full_gsm8k \
  --vae_weight 0.1 \
  --vae_beta_max 0.5 \
  --vae_warmup_epochs 10 \
  --epochs 40 \
  --batch_size 32 \
  --eval_batch_size 64
```

Pure unsupervised VAE audit:

```bash
python latent_vae_audit.py data/features/full_gsm8k.npz \
  --output_dir outputs/latent_vae_full_gsm8k \
  --tag full_gsm8k \
  --latent_dim 32 \
  --pca_dim 256 \
  --epochs 120 \
  --batch_size 1024
```

Do not pass `--amp` for first validation runs. The code now computes losses in fp32, but VAE Gaussian losses and contrastive logits are still easier to debug without mixed precision.

## Output Checks

The combined script writes:

- `{tag}_latent_separatrix_summary.json`
- `{tag}_latent_separatrix_summary.md`
- `{tag}_latent_separatrix_rows.csv`
- `{tag}_latent_separatrix_chains.csv`
- `{tag}_latent_separatrix_latents.npz`

Key fields to inspect:

- step-level: `hazard`, `energy`, `latent_norm`, `vae_rec_nll`, `vae_kl`, `vae_dec_unc`, `vae_post_unc`
- response-level: `survival_score`, `max_hazard`, `mean_hazard`, `max_vae_rec_nll`, `mean_vae_rec_nll`
- model-level: `controls`, `latent`, `controls+latent`

## Validation Logic

The result is useful only if at least one of the following holds out-of-fold:

1. `latent` beats `controls` on first-error detection.
2. `controls+latent` beats `controls`.
3. `hazard` ranks the gold first-error step top-1 within chain.
4. VAE-only signals such as `vae_rec_nll` or `vae_kl` separate errors after controlling for position and step length.

If only `step_len` or `pos` wins, the method should be treated as a failed geometry hypothesis for that dataset/config.

