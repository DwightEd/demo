# ProcessBench 首错边界图监测实验计划

**方法名**：Residual Depth Graph Monitor（RDGM，残差深度图监测器）  
**冻结时间**：2026-08-05 22:25:35（Asia/Shanghai）  
**版本化原件**：[EXPERIMENT_PLAN_20260805_222535.md](EXPERIMENT_PLAN_20260805_222535.md)  
**状态**：代码已构建并通过本地测试；远端真实数据尚未运行。

## 核心任务

在步骤 `t` 的第一个 token 尚未生成时，用 `q_t=a_t-1` 处完整的 `[layer, hidden]` 残差状态预测该步骤是否为首错步骤。

风险集严格定义为：错误链保留 `0..t*`，正确链保留全部步骤，首错后的行全部删除，step 0 保留。输出历史只允许使用已完成的 `0..t-1` 步。

## 实验模型

| 模型 | 输入 | 作用 |
|---|---|---|
| `nuisance` | 可见长度、位置、过去步长 | 测量长度/难度污染 |
| `output_history` | nuisance + 过去 entropy/NLL | 强输出基线 |
| `layer_set` | 完整 `[L,D]`，无层序 pooling | hidden 信息但无图结构 |
| `depth_graph_shuffled` | 完整 `[L,D]`，随机重接邻接 | 容量/错误邻接对照 |
| `depth_graph` | 完整 `[L,D]`，真实深度链消息传递 | 主模型 |

主模型不先计算方向一致性、秩、范数、PCA/Ridge innovation 等手工指标。hidden 各维通过共享可学习投影进入层深图。

## 划分与评估

- 外层四域 LODO；
- 内层按 domain 内 problem/sibling group 划分验证集；
- 标准化仅拟合训练行；
- 训练权重按 domain→problem→row 平衡；
- 验证正确链冻结 false-alarm threshold；
- 主指标：链内 Top-1、MRR、正确链/步骤误报；
- 辅助指标：问题组平衡 NLL、AUROC、AUPRC；
- 推断：domain→problem 配对 bootstrap。

## 主要检验

```text
depth_graph - output_history
depth_graph - layer_set
depth_graph - depth_graph_shuffled
```

若图模型只胜输出基线而不胜 `layer_set`/`shuffled`，结论只能是完整 hidden tensor 有预测信息，不能声称图结构有效。若 shuffled 不差于真实邻接，层深路由假设被否证。

本实验的结论范围固定为预测关联；Attention/FFN 根因必须由 CFEA 受控反事实干预另行验证。

## 运行

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
bash run_hidden_geometry_remote.sh causal-monitor-full
```

完整数据合同、门槛、否证规则与后续研究顺序见版本化原件。
