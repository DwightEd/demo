# Residual Depth Graph Monitor（RDGM）

RDGM 是 CFEA 中面向自然 ProcessBench 的检测分支。它不要求 `onset_pairs_v1.jsonl`，也不加载 Llama 权重重新提取；它直接读取已经存在的 causal pre-step 全层状态。

## 数据流

```text
<domain>/geometry/trace.npz
  -> step_pre_state_memmap_path
  -> X_t = [layer, hidden] at token(step_start - 1)
  -> error risk set: 0..first_error_step
  -> correct risk set: all steps
  -> grouped inner validation + outer LODO
  -> nuisance / output / layer-set / 3 shuffled topologies / depth-graph
  -> chain localization + false alarm + paired bootstrap
```

核心实现：

- `monitor_data.py`：读取 pre-step memmap，构造无未来泄漏风险集；
- `monitor_models.py`：完整层×hidden 输入的深度图、无序层对照；
- `monitor_training.py`：train-only normalization、group-balanced loss、early stopping；
- `monitor_experiment.py`：四域 LODO、阈值冻结、指标和配对 bootstrap；
- `main.py train-monitor`：唯一运行入口。

## 直接运行

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
bash run_hidden_geometry_remote.sh causal-monitor-full
```

脚本在前台运行。`fit state normalization`、`LODO folds` 和每个实验臂的 epoch 都有进度显示。结果写入带时间戳的新目录：

```text
outputs/hidden_state_geometry/causal_monitor_<smoke|full>_<timestamp>/
├── config.json
├── predictions.jsonl
└── summary.json
```

## 如何判断

先看 `summary.json`：

```text
domain_macro.depth_graph
paired_contrasts.depth_graph_vs_output_history
paired_contrasts.depth_graph_vs_layer_set
paired_contrasts.depth_graph_vs_depth_graph_shuffled
```

每个 shuffled topology 都禁止保留原深度链的无向邻接边，并以独立 topology seed 训练；报告同时保存每个 seed 和 ensemble 的比较。只有图模型稳定胜过 `layer_set`、shuffled ensemble 及各 topology seed，才能说真实层深结构提供了预测增量。该结果仍不是 Attention 路由或 FFN 更新的因果证据。

完整冻结计划见 [EXPERIMENT_PLAN.md](refine-logs/EXPERIMENT_PLAN.md)。
