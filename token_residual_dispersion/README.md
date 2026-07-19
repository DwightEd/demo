# Token-Causal Residual Dispersion

这是一个不依赖 CoT step、句子边界或语义分段的残差流分析原型。分析单位是模型真实的
token 位置，核心对象是 block residual write 在因果滑动窗口中的方向分布。

## 定义

若提取到 token `t` 在相邻深度的状态 `h[t,l-1]` 与 `h[t,l]`，则 block write 为：

`delta[t,l] = h[t,l] - h[t,l-1]`。

对窗口 `W_t={max(0,t-W+1),...,t}` 中的单位方向 `u[i,l]`，代码计算：

- resultant `R = ||sum_i w_i u_i||`；
- 去自相似后的 pair dispersion `1 - c_bar`；
- 中心化方向散布的 effective rank；
- `tr(Cov) = 1 - R^2` 数值恒等式误差；
- 沿 token 的变化量，以及原始/归一化残差弧长坐标。

输出轴始终是 `[token, block, scale]`。位置 `t` 的量只使用 `<=t` 的激活，因此它是
**post-token diagnostic**；用于预测下一个 token 时应整体后移一位。归一化弧长 phase 使用
整条轨迹总长度，只用于回顾性可视化，不能伪装成在线特征。

## 快速运行

```powershell
cd D:\projects\research\demo
python -m token_residual_dispersion.cli --selftest
python -m token_residual_dispersion.cli `
  --input path\to\features.npz `
  --output-dir outputs\token_residual_dispersion `
  --windows 4,8,16,32
```

正式分析要求 manifest 写明
`response_token_state_snapshot_kind=raw_residual_stream`。这是为了避免某些架构的最后一个
hidden state 已经过 final norm，却被相邻差分误称为 block write。旧 manifest 可用
`--allow-unverified-snapshots` 做探索性 pilot，但这种输出不能支撑 block-write 机制结论。

旧版 `selected` manifest 保存了逐 token、稀疏深度 `8,10,...,22` 的状态。可以使用
`--legacy-sparse-pilot` 直接分析，但输出会标记为
`sparse_multi_block_depth_interval_delta_pilot`：例如 `h[10]-h[8]` 只是跨两个 block 的
depth-interval delta，不会被命名为单 block residual write。

服务器上先运行 20 条链的四子集 pilot：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
MAX_TRACES=20 bash token_residual_dispersion/run_existing_selected_pilot.sh \
  data/exact/processbench_observer_llama31 \
  outputs/token_residual_dispersion_sparse_pilot20
```

确认四个 `audit_summary.json` 后，去掉 `MAX_TRACES=20` 并使用新的输出目录运行全量。
脚本以逐链流式方式读取 shard；`--rank-stride 4` 只降低 effective-rank 的时间采样密度，
pair dispersion 仍在每个 token、每个深度区间、每个窗口上计算。

输入可以是单条 `[token, depth, hidden]` `.npy`（必须同时传 `--layers`），也可以是现有 extraction manifest；后者
通过 `response_token_state_files` 加载 per-chain `.npy` shards。为了保证名称真实，只有连续
layer/depth 快照才允许被解释为单个 block write；缺失 layer metadata 或稀疏层都会直接报错。

注意 effective rank 描述的是“已经存在的散布分布在多少个方向”，并不描述散布幅度；当
scatter trace 很小时，微小的各向同性噪声也可能有较高 rank。因此判断“杂乱”必须联合看
pair dispersion/scatter trace 与 effective rank，不能只看 rank。

当前版本是 NumPy-only 的 M0/M1：先验证“离散度是否稳定、是否因果、是否有独立于范数的
信息”。attention/MLP 分解的数值定义已提供，但真正的 component hook 留到 M2。

代码中的 cumulative arc length 更准确地说是累计 residual-write activity；归一化 phase
使用完整轨迹，只能回顾性使用。

详见 [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md)。
