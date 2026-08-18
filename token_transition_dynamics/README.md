# Token Transition Dynamics

这个实验直接读取逐 token 的原始 residual-stream hidden states。保存的层可以是
`[8, 10, 12, ..., 22]` 这样的稀疏观测层；程序不会再把 8→10 误解成相邻层状态转移。
每一个保存层都被独立地看成一条沿 token 时间演化的轨迹。

## 实验测什么

对每个固定长度的 trailing token window、每个保存层，直接在原始 hidden 维度中计算：

- `tle_intrinsic_dimension`：Tight Local intrinsic dimension；
- `information_volume`：`0.5 logdet(I + d/T ZZ^T)`；
- `velocity_innovation_ratio`：相邻 token 速度变化相对速度能量；
- `path_efficiency`：端点位移与实际轨迹长度之比。

前两项只依赖窗口点云，打乱 token 顺序不变；后两项显式保留 token 顺序。实现不做
PCA、不做 KMeans、不池化 step，也不训练 MLP。information volume 使用 determinant
lemma 在 `T×T` Gram matrix 上精确计算，不会构造 `4096×4096` 矩阵。

基线只使用 train split 中完全正确的推理链，按 domain、保存层和相对位置 bin 拟合
median/MAD。测试分数是偏离正确轨迹 norm 的 squared robust z distance。首错 step
覆盖的 token window endpoint 是正类；首错之前是负类；首错之后完全排除。固定窗口长度
控制长度混淆，相对位置分层控制位置混淆。

TLE 数值实现遵循 scikit-dimension 的 BSD-3-Clause 实现；实验设计中的 TLE 与
information-volume 定义对应论文
[Reasoning emerges from constrained inference manifolds in large language models](https://arxiv.org/abs/2605.08142)。
两个有序动力学量是本项目增加的检验，因为论文中的点云量本身不能识别 token 顺序。

## 数据

默认读取：

```text
/share/home/tm902089733300000/a903202310/lys/data/ProcessBench/reasoning_error_detection/llama31_8b/<domain>/selected/trace.raw_residual_stream.npz
```

manifest 指向 `[response_token, stored_layer, hidden_dim]` 的 `.npy` shard，并提供
`step_token_ranges`、`gold_error_step` 和保存层编号。程序只 mmap 原始 shard；不会读取
`step_scores` 或其他派生特征。

## 远端运行

已经进入远端 `demo` 目录后，不需要 SSH。先做快速数据检查，再跑 smoke 或 full：

```bash
bash token_transition_dynamics/run_remote.sh preflight
bash token_transition_dynamics/run_remote.sh smoke
bash token_transition_dynamics/run_remote.sh full
```

也可以覆盖路径和 Python：

```bash
DATA_ROOT=/your/data PYTHON_BIN=/your/env/bin/python \
  bash token_transition_dynamics/run_remote.sh full
```

结果写入：

```text
outputs/token_transition_dynamics/smoke/results.json
outputs/token_transition_dynamics/smoke/summary.txt
outputs/token_transition_dynamics/full/results.json
outputs/token_transition_dynamics/full/summary.txt
```

`early_error_records_skipped` 会报告由于首错发生早于完整窗口而无法评估的错误链。
结果属于预测关联证据，不直接证明 LLM 内部存在某种因果流形机制。
