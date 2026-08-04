# Research Proposal: Causal Belief Update Decomposition

## Problem Anchor

- **Bottom-line problem**：判断 pretrained decoder-only Transformer 是否在残差流中实现可计量的近似信念更新，并把更新分解到 evidence routing、OV content write 与 MLP correction，而不是继续依赖长度污染的几何标量。
- **Must-solve bottleneck**：当前实现只用 attention mass 与 OV-write cosine alignment 形成 routing score；它没有计量写入覆盖了多少解析目标更新，也没有分离 attention output 与 MLP output，因此不能区分路由失败、内容写入失败和局部更新算子失败。
- **Non-goals**：不声称所有自然语言推理都是 Bayesian；不把一般 residual point cloud 称为光滑流形；不把 decodability 当作 causality；本阶段不训练新模型或构造通用错误检测器。
- **Constraints**：复用 frozen pretrained LLM、predictive-alias finite-field world、cross-fitted Fourier charts 和已有 source patch pipeline；新增逻辑必须能在短 prompt 上选择性提取，不能保存完整 `[L,H,N,N]` attention；本地无 Python/GPU，运行验证由远程环境完成。
- **Success condition**：在 held-out alias pairs 上，真实 block residual delta 可由 attention output 与 MLP output 重构；这些分量在 cross-fitted analytic belief coordinates 中得到有量纲的 target progress 与 relative target error；预注册层上可以判断 MLP 是修正、破坏还是与 attention 交互，并为后续 factorial patching 提供明确门槛。

## Technical Gap

现有项目的方法链是：精确后验与 predictive alias -> cross-fitted residual-to-Fourier chart -> evidence-token source OV decomposition -> donor/recipient source patch。这个链条已经避免了长度和当前 logits 的主要混杂，但 routing audit 的主分数

\[
\text{attention mass}\times(\cos(w,\Delta\Phi^*)-\cos(w,\Delta\Phi^{opp}))
\]

仍是方向性启发量。它不能回答写入是否在 Fourier 坐标中完成了 10%、100% 或反向的目标更新；也无法判断 MLP 对 attention 后状态做了什么。把更多几何特征堆进分类器不会修复这一机制缺口。

## Method Thesis

- **One-sentence thesis**：在受控 predictive-alias 任务中，将每个 Transformer block 的实际 residual delta 分解为 attention 与 MLP 写入，并通过训练折外的解析 belief chart 计量每个分量对精确信念更新的 progress、direction margin 与 residual error。
- **Smallest adequate intervention**：新增一个只读 block-component capture、一个稳定 artifact contract 和一个预注册层 audit；复用已有模型 replay、charts、pair folds 与 bootstrap。
- **Frontier fit**：这是对 frozen foundation model 的 mechanistic causal measurement，不需要添加新的生成、微调或 probe 模块。

## Contribution Focus

- **Dominant contribution**：Causal Belief Update Decomposition（CBUD），即把 source routing 与 block-local update 在同一个精确信念坐标系中计量。
- **Supporting contribution**：以 MLP correction gain 和 target-error reduction 为门槛，决定是否值得运行 attention/MLP factorial patch。
- **Explicit non-contributions**：一般 Bayesian cognition、全局光滑 manifold、自然任务在线 detector、参数知识定位。

## Proposed Method

### Complexity Budget

- **Frozen/reused**：LLM、tokenizer、finite-field generator、Fourier target、LayerChartBundle、pair-group cross-fitting、cluster bootstrap。
- **New trainable components**：0。
- **New code**：component capture、update metric functions、NPZ schema、audit 与两条 CLI。

### System Overview

```text
exact posterior transition ΔΦ*
            |
saved prompts + frozen LLM ----> block hooks
                                  |-- attention output A_l
                                  |-- MLP output F_l
                                  `-- actual block delta B_l
                                            |
                         held-out fold chart J_l
                                            |
                  δΦ_A, δΦ_F, δΦ_B in one analytic chart
                                            |
           progress / alignment margin / target error / reconstruction
```

### Core Measurement

对分量写入 \(u\) 及目标更新 \(\Delta\Phi^*\)，定义：

\[
\operatorname{progress}(u)=
\frac{\langle J_\ell u,\Delta\Phi^*\rangle}
{\|\Delta\Phi^*\|^2},
\]

\[
\operatorname{target\_error}(u)=
\frac{\|\Delta\Phi^*-J_\ell u\|}
{\|\Delta\Phi^*\|},
\]

\[
\operatorname{margin}(u)=
\cos(J_\ell u,\Delta\Phi^*)-
\cos(J_\ell u,\Delta\Phi^{opp}).
\]

其中 progress 是有符号的目标覆盖量，target error 同时惩罚方向和幅度，margin 用来区分 matched opposite update。MLP correction 用

\[
G_{\mathrm{margin}}=\operatorname{margin}(B)-\operatorname{margin}(A),
\qquad
G_{\mathrm{error}}=\operatorname{target\_error}(A)-
\operatorname{target\_error}(B)
\]

计量。实际 block delta 必须满足 \(B\approx A+F\)，否则 artifact fail closed。

### Integration

新增 extractor 使用 current-query observations 和现有 LayerChartBundle。它只捕获选定 block 的最后可见 token 分量，不持久化 sequence histories。audit 强制指定 `primary_layer`；其他层只作描述，避免事后挑层。

### Failure Modes and Diagnostics

- **Chart 不可靠**：沿用 representation gate，失败时禁止正式 extraction。
- **Hook 语义不匹配**：报告 `||B-(A+F)||/||B||`，超过阈值即禁止机制结论。
- **MLP 写入大但无目标意义**：progress、margin 与 target error 联合报告，不以 residual norm 代替 belief update。
- **单层分量不能代表完整 posterior transition**：逐层报告，不跨 chart 直接求和；结论限定为 block-local contribution。
- **自然任务没有精确后验**：只有 controlled gate 通过后才冻结机制分数并迁移。

### Novelty and Elegance

现有 belief-geometry 工作说明 residual state 可表示 posterior，现有项目也已实现 source-specific OV routing；缺失的是在 frozen instruction LLM 上把 attention 与 MLP 的真实 block writes 放入同一 held-out analytic chart，并以可重构、有幅度的更新量而非 attention-weighted cosine 进行机制审计。新增机制不训练模型，也不增加并行论文主张。

## Claim-Driven Validation Sketch

### Claim 1: block components can be measured as belief-update writes

- **Minimal experiment**：在 predictive aliases 上提取 \(A,F,B\)，审计 reconstruction error 与 held-out progress/error。
- **Baselines/ablations**：opposite branch update、permuted chart labels、wrong layer chart。
- **Metric**：reconstruction p95、primary-layer margin、progress、target error。
- **Expected evidence**：reconstruction p95 低于预注册阈值，真实目标优于 opposite/null。

### Claim 2: MLP performs a state-dependent correction beyond routed attention

- **Minimal experiment**：在预注册层比较 attention-only 与 full-block 更新。
- **Baselines/ablations**：MLP output zeroing/patching、length-matched non-evidence source patch、random-layer MLP patch。
- **Metric**：cluster-bootstrap \(G_{margin}\) 与 \(G_{error}\)，随后 donor-recipient factorial patch 的 future-logit shift。
- **Expected evidence**：两个 correction 指标的下置信界大于零，并由 MLP patch 因果复现。

## Experiment Handoff Inputs

- **Must prove**：component reconstruction；cross-fitted target progress；MLP correction 是否存在。
- **Must run**：opposite-update null；pre-registered layer；attention-only、MLP-only、joint patch。
- **Highest risk**：Fourier chart 的局部 Jacobian可能只适合状态差分而不适合大幅单层写入；MLP 功能可能分散到多层而非单一校正层。

## Compute & Timeline Estimate

- **GPU**：短 prompt、只保存 boundary components；预计与一次普通 forward replay 同阶，显著低于 full-attention routing pass。
- **Data**：复用 200-pair pilot；正式门槛再扩到 2,500 pairs。
- **Implementation**：纯逻辑、schema/audit、extractor/CLI 三个增量。
