# Research Proposal: Causal Belief Update Decomposition

项目的单一主张是：在具有精确后验的 predictive-alias 任务中，可以把 frozen Transformer 每个 block 的实际 residual delta 分解为 attention 与 MLP 写入，并通过训练折外的 analytic Fourier belief chart 计量它们对目标更新的有符号进展、方向判别和剩余误差。

实现复用现有 exact posterior、LayerChartBundle、pair folds 和 bootstrap，不训练新组件。新增独立 block-update artifact，保存最后可见 token 上的 attention、MLP 与 actual block delta 的派生计量，并用 reconstruction error 验证 hook 语义。audit 强制预注册 primary layer；其他层只作描述。

核心量为：

\[
\operatorname{progress}(u)=\frac{\langle J_\ell u,\Delta\Phi^*\rangle}{\|\Delta\Phi^*\|^2},
\qquad
\operatorname{error}(u)=\frac{\|\Delta\Phi^*-J_\ell u\|}{\|\Delta\Phi^*\|}.
\]

现有 `attention mass × cosine margin` 只保留为 routing/head-selection diagnostic。MLP 的观测性证据由 full-block 相对 attention-only 的 margin gain 与 target-error reduction 给出；只有之后的 attention-only、MLP-only、joint factorial patch 才能建立因果 correction 主张。

项目公开名称与 Python package 路径统一为 **Causal Belief Update Decomposition (CBUD)** / `causal_belief_update_decomposition`。已有 representation trace/chart 的 schema 和数据目录仍保留旧标识，以便复用远端产物。
