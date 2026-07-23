# Full-tensor ridge hidden-state probe

## 它是不是把几个标量放进 probe？

不是。两个静态比较臂确实使用少量标量：

- `nuisance`：3 个长度/位置控制量；
- `output_only`：上述控制量加已保存的 entropy/NLL 时间摘要。

hidden 臂读取真实 residual-stream shard，并执行：

```text
[reasoning step, stored layer, hidden=4096]
    -> outer/inner-train-only PCA
    -> [time-or-history basis, layer basis, PCA channel]
    -> flatten every tensor cell
    -> ridge logistic probe
```

默认配置下：

- `whole_chain` 的 hidden 输入为 `3 × 3 × 16 = 144` 个张量单元；
- `strict_prefix` 的 hidden 输入为 `2 × 3 × 16 = 96` 个张量单元。

它们是坐标化 residual trajectory 的全部低维张量单元，不是手工计算的“发散度、谱半径”等几个标量。旧方法再将该张量的系数强制写成一个
`u(time) ⊗ v(layer) ⊗ w(channel)`，只有约 20 个有效自由度；新方法为每个单元独立学习一个线性系数，因此可以表示多秩、非可分的时层模式。

## 方法与验证边界

每个 outer leave-one-dataset-out fold 内：

1. 只用 outer-train 的原始 hidden states 拟合 chain-balanced PCA；
2. 编码完整 outer-train 与 held-domain；
3. 再对 outer-train 做 inner leave-one-training-domain-out；
4. 每个 inner fold 都重新拟合 PCA、缺失值填补/标准化器和 ridge probe；
5. 以 held-inner-domain 的 problem-group-balanced NLL 选择 L2；各 inner domain 等权；
6. 用选定 L2 在完整 outer-train 重拟合四个 arm，预测 outer held-domain。

四个 arm 各自选择 L2，因此这里比较的是“经过同一内层协议调优后的嵌套特征模型类”，不是强迫两个 arm
共享同一个正则强度。显式传入单元素 `l2_grid` 时不需要内层选择，诊断会标为
`outer_lodo_fixed_l2_full_tensor_ridge`，不会伪称执行过 inner LODO。

训练权重先让每个训练域等权，再让域内每个 problem group 等权。任一优化、类别或分组契约失败都会直接报错，没有 baseline 替代、缺包替代或其他退化实现。

四个 arm 是：

```text
nuisance
output_only
hidden_only
output_plus_hidden
```

主比较为：

```text
NLL(output_only) - NLL(output_plus_hidden)
```

正值表示完整 hidden tensor 在 entropy/NLL 摘要之外提供 held-domain 预测增量。它仍然只是跨域判别关联，不等于 Jacobian/Fisher 功能性或 activation-patching 因果证据。

这个版本仍有三个明确边界：

- whole-chain 的时间轴是低频 DCT，strict-prefix 是 `[current, history mean]`，尚不是保留全部有序历史的原始 `t × layer` 模型；
- 没有容量匹配的 time/layer shuffle arm，因此 hidden 显著时也不能直接归因于层序、旋转或动态谱；
- 每折保存的 `hidden_tensor_coefficient` 位于该折自己的 whitened PCA 坐标，不能跨折逐单元直接平均。

`fold_diagnostics[*].selected_at_grid_edge` 会报告 `lower`、`upper`、`none` 或 `fixed`。若关键 hidden arm
落在上下边界，应扩展 L2 网格后重跑，不能把边界解读成稳定阴性或阳性结论。

## 代码组织

```text
hidden_state_geometry/features.py
    共享的 fold-local PCA、按链投影缓存与 time-layer 编码

hidden_state_geometry/methods/raw_functional_probe.py
    原 rank-one 方法；复用共享编码组件

hidden_state_geometry/methods/full_tensor_ridge.py
    独立 full-tensor 方法、inner LODO、L2 选择与模型因素导出
```

以后新增方法仍只需在 `methods/` 中新增插件文件并在 `methods/__init__.py` 导入，不需要复制数据加载、任务构造、LODO、bootstrap 或 artifact writer。

## 远端前台运行

```bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

PROJECT_ROOT=/share/home/tm902089733300000/a903202310/lys/research/demo/reasoning_activation_divergence
cd "$PROJECT_ROOT"

git pull --ff-only
"$CONDA_PREFIX/bin/python" -m pip install -e '.[test]'

PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  bash run_hidden_geometry_remote.sh ridge-smoke

PYTHON_BIN="$CONDA_PREFIX/bin/python" \
  bash run_hidden_geometry_remote.sh ridge-full
```

两条命令都是普通前台进程，会打印 PCA、投影、inner-domain ridge、final arms 和 bootstrap 进度。默认真实数据位置为：

```text
/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/processbench_observer_llama31_full
```

如位置不同，显式设置 `DATA_ROOT=/absolute/path`。结果分别写入：

```text
outputs/hidden_state_geometry/ridge_smoke_<timestamp>/
outputs/hidden_state_geometry/ridge_full_<timestamp>/
```

优先检查：

```text
results.json
  tasks.whole_chain.summary.increments.hidden_given_output_summary_nll
  tasks.strict_prefix.summary.increments.hidden_given_output_summary_nll
  tasks.<task>.fold_diagnostics[*].selected_l2
  tasks.<task>.fold_diagnostics[*].selected_at_grid_edge
  tasks.<task>.fold_diagnostics[*].inner_cv_scores

model_factors.npz
  <task>.fold_<k>.output_plus_hidden.hidden_tensor_coefficient
```
