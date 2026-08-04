# Round 1 Refinement

## Problem Anchor

- **Bottom-line problem**：判断 pretrained decoder-only Transformer 是否在残差流中实现可计量的近似信念更新，并把更新分解到 evidence routing、OV content write 与 MLP correction，而不是继续依赖长度污染的几何标量。
- **Must-solve bottleneck**：当前实现只用 attention mass 与 OV-write cosine alignment 形成 routing score；它没有计量写入覆盖了多少解析目标更新，也没有分离 attention output 与 MLP output，因此不能区分路由失败、内容写入失败和局部更新算子失败。
- **Non-goals**：不声称所有自然语言推理都是 Bayesian；不把一般 residual point cloud 称为光滑流形；不把 decodability 当作 causality；本阶段不训练新模型或构造通用错误检测器。
- **Constraints**：复用 frozen pretrained LLM、predictive-alias finite-field world、cross-fitted Fourier charts 和已有 source patch pipeline；新增逻辑必须能在短 prompt 上选择性提取，不能保存完整 `[L,H,N,N]` attention；本地无 Python/GPU，运行验证由远程环境完成。
- **Success condition**：在 held-out alias pairs 上，真实 block residual delta 可由 attention output 与 MLP output 重构；这些分量在 cross-fitted analytic belief coordinates 中得到有量纲的 target progress 与 relative target error；预注册层上可以判断 MLP 是修正、破坏还是与 attention 交互，并为后续 factorial patching 提供明确门槛。

## Anchor Check

- 原始瓶颈保持不变：从通用几何相关性转向可识别的计算更新。
- 新实现只补最缺失的 block decomposition，不扩展为 detector 或知识定位系统。

## Simplicity Check

- 主贡献仍是 **Causal Belief Update Decomposition (CBUD)**。
- 保留既有 routing v1 artifact，新增独立 block-update contract，避免迁移复杂度。
- 本阶段不实现 MLP causal patch；只生成是否值得进入 factorial patch 的 observational gate。

## Changes Made

1. 将 MLP `correction supported` 更名为 `mlp_update_signature`，明确因果边界。
2. 增加 block reconstruction p95 fail-closed 条件。
3. audit 强制预注册 primary layer。
4. 更新量的主计量改为 target progress 与 target error，cosine margin 仅保留方向判别。

## Revised Proposal

实施一个独立 `BeliefUpdateTrace`：对每个 current-query row 和选定 raw block depth 捕获 attention output、MLP output、block input 与 block output；只保存最后可见 token 的派生标量。用该 row 所属 held-out fold 的 `LayerChartBundle` Jacobian 投影三个写入，记录：

- true/opposite alignment；
- signed target progress；
- relative target error；
- residual-space write norms；
- `||block_delta-(attention+MLP)||/||block_delta||`。

audit 在显式 primary layer 上做 pair-cluster bootstrap，报告 attention update signature、MLP incremental signature 与 decomposition validity。只有 representation gate、reconstruction gate、MLP margin gain 和 target-error reduction同时通过，才标记 `ready_for_factorial_patching=true`；此标记不是 MLP 因果结论。
