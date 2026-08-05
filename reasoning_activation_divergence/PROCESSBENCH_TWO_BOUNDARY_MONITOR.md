# Prefix-conditioned Two-Boundary Innovation Hazard（PTIH）

PTIH 是 CFEA 中面向自然 ProcessBench 的检测分支。它不要求 `onset_pairs_v1.jsonl`，也不重新加载 Llama；它直接读取现有 causal pre-step 全层 hidden states。

它只检验预测关联：相邻两个合法前缀边界之间的有向更新是否预示下一步成为首错步。Attention 路由或 FFN 根因仍需受控干预验证。

## 数据流

```text
<domain>/geometry/trace.npz
  -> step_pre_state_memmap_path
  -> (H_{t-1}, H_t), each [layer, hidden]
  -> error risk set: 0..first_error_step
  -> correct risk set: all steps
  -> grouped inner validation + outer LODO
  -> nuisance / output / static / directionless bag / directed innovation
  -> chain localization + false alarm + paired bootstrap
```

步骤 0 令 `H_{-1}=H_0`，所以创新量为零。主要推断只使用 `t>=1`，全风险集作为部署视角的次要结果。

## 模型

- `nuisance`：步骤、token 位置、历史步长；
- `output_history`：nuisance 加过去步骤 entropy/NLL；
- `static_layer_set`：当前 `H_t`，第二输入固定为零；
- `two_boundary_bag`：两边界均值和绝对差，删除时间方向；
- `two_boundary_innovation`：当前 `H_t` 和有符号增量 `H_t-H_{t-1}`。

三个 hidden 实验臂使用同一个共享层投影、mean/max pooling 和相同分类头，参数量严格一致。这里没有手工几何标量，也不预设层邻接图。

## 实现入口

- `monitor_data.py`：读取 pre-step memmap，构造无未来泄漏风险集和同链边界历史；
- `monitor_models.py`：三个同容量两边界模型；
- `monitor_training.py`：train-only normalization、domain→problem→chain→row 平衡和 early stopping；
- `monitor_experiment.py`：四域 LODO、阈值冻结、全风险集/`t>=1` 配对 bootstrap；
- `main.py train-monitor`：唯一运行入口。

## 直接运行

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
bash run_hidden_geometry_remote.sh causal-monitor-full
```

脚本在前台运行，失败时保留 traceback 并等待确认。结果写入：

```text
outputs/hidden_state_geometry/causal_monitor_<smoke|full>_<timestamp>/
├── config.json
├── predictions.jsonl
└── summary.json
```

首先查看 `summary.json` 的：

```text
domain_macro
temporal_paired_contrasts.two_boundary_innovation_vs_static_layer_set
temporal_paired_contrasts.two_boundary_innovation_vs_two_boundary_bag
claim_decision
```

只有 innovation 在 `t>=1` 上同时稳定胜过 static 和 bag，且正确链误报不恶化，才支持“有向边界更新包含独立首错风险信息”。完整冻结计划见 [EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)。
