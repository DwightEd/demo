# Token Transition Dynamics

这个子项目直接读取逐 token、逐层的原始 residual-stream hidden state，检验错误是否表现为偏离“全正确推理”所学习到的局部转移规律。它不读取 `step_scores`，也不把一个 step 池化成一个向量。

## 验证对象

- 单位：一个 response token 在每一对相邻 hidden layers 之间的转移。
- 表征：只用训练集全正确链拟合 PCA。
- 动力学：每个相邻层对独立拟合软混合线性向量场。
- 四个实验臂：`Euclidean/Spherical × K=1/K=2`。
- 监督：错误标签不参与 PCA 或动力学训练。测试时，first-error step 覆盖的 token 是区间正类；错误后的 token 不进入评估。
- 指标：逐 token AUROC/AUPRC（同时保留每个 layer transition 的结果）、problem-group bootstrap 95% CI、错误链内的宏平均定位 AUROC、全正确 calibration NLL。跨几何不直接比较不同坐标系的原始 NLL，而比较各自相对 position-only baseline 的 NLL gain。结果还保存 K=2 在每个相邻层对学到的 mixture weights，便于发现簇塌缩。

这里的 K=2 是“数据是否需要两个局部转移场”的最小检验，不预设任何语义簇。Spherical arm 使用单位球上的 log map；它并不预设激活一定构成球面。

`verdict` 只给出 exploratory point gate，并固定保留 `claim_supported=false`：单次点估计不能证明存在流形或机制，还需要 paired-delta 不确定性、跨 seed/模型复现和干预实验。

## 数据要求

默认读取：

```text
<data-root>/<domain>/selected/trace.raw_residual_stream.npz
```

manifest 必须指向形状为 `[response_token, stored_layer, hidden_dim]` 的 `.npy` shards，并包含 `step_token_ranges` 与 `gold_error_step`。存储的层必须连续；稀疏层快照会直接报错，因为它不能支持逐层动力学解释。

## 远端一键运行

在 `demo` 目录执行，不需要再次 SSH：

```bash
git pull
bash token_transition_dynamics/run_remote.sh preflight
bash token_transition_dynamics/run_remote.sh smoke
bash token_transition_dynamics/run_remote.sh full
```

默认数据根目录是：

```text
/share/home/tm902089733300000/a903202310/lys/data/ProcessBench/reasoning_error_detection/llama31_8b
```

覆盖路径或 Python 环境：

```bash
DATA_ROOT=/your/data PYTHON_BIN=/your/env/bin/python \
  bash token_transition_dynamics/run_remote.sh smoke
```

结果分别保存到：

```text
outputs/token_transition_dynamics/smoke/results.json
outputs/token_transition_dynamics/full/results.json
```

这一步只需要 CPU 和内存映射读取，不需要 GPU。`smoke` 从每个 domain 的整个 manifest 等距抽取 128 条记录；若抽样后缺少 train/calibration/test 所需队列，程序会明确报错，此时先运行 `preflight`，再直接运行 `full`。
