# 实现与研究计划

## 研究问题

不预设“一个推理 step 必须存在”。要检验的是：token 级 residual write 的方向分布是否会从
低维、持续的流形转入多方向竞争状态，以及这种转变是否在错误答案、扰动不稳或未来错误
之前出现。

## 当前边界

- 数据轴：`token × consecutive depth × hidden`。
- 来源门：只有 extractor 标记为 `raw_residual_stream` 的快照才允许支持 block-write 结论；
  M2 还需用 hook 重构误差进一步验证。
- 估计轴：`token × block × causal scale`，默认窗口 `4/8/16/32`。
- 主指标：debiased pair dispersion；几何佐证：scatter trace，以及条件于散布幅度的
  effective rank（禁止脱离 trace 单独解释 rank）。
- 对照量：write norm 与 residual arc length，防止把“能量变大”错称为“方向变乱”。
- 禁止项：手工 step、CoT 句号切分、用首次错误位置参与指标构造、把全轨迹 phase 当在线量。

## 里程碑

1. **M0 合成可证伪测试（本目录已实现）**
   - 连贯方向 → 随机方向的合成变点。
   - 检验离散度和有效秩升高。
   - 修改未来 token 不得改变历史窗口结果。
2. **M1 离线真实激活审计（CLI 已实现）**
   - 读取现有 per-chain token-state shards。
   - 导出每条轨迹 `.npz` 和汇总 JSON。
   - 首批只做描述统计，不以答案标签调参。
3. **M2 组件级残差流**
   - 在 pre-norm block 中分别 hook attention write 与 MLP write。
   - 校验 `block_delta ≈ attention_write + mlp_write`，误差超阈值即拒绝分析。
   - 计算 antagonism 与 cancellation，区分“多方向探索”和“组件互相抵消”。
4. **M3 无监督演化事件**
   - 在多尺度场上做 change-point / persistent event detection。
   - 事件只由激活定义；首次错误 token 仅作事后对齐和检验。
5. **M4 因果与泛化**
   - 同题多采样、prompt 扰动、正确/错误匹配对照。
   - 训练集确定阈值，跨任务/模型冻结评估。
   - 与 entropy、margin、norm、长度、token identity 等基线比较增量解释力。

## 首轮验收门槛

- 数值恒等式 `tr(Cov)=1-R^2` 最大误差 `<1e-8`（float64 合成数据）。
- 合成 diffuse 区间的 pair dispersion 至少高于 coherent 区间 `0.35`。
- 任意未来扰动不改变当前及过去 token 的统计量。
- 真实数据上先报告 effect size、bootstrap CI 与 seed/trace 聚类稳健性，再讨论预测意义。

## 尚未声称

当前实现不证明离散度导致错误，也不证明存在统一的“推理阶段”。它只建立一个无 step、
可复算、能被反例推翻的测量层，为后续因果实验提供对象。
