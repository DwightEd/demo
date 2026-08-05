# RDGM Experiment Tracker

**当前版本**：[EXPERIMENT_TRACKER_20260805_222535.md](EXPERIMENT_TRACKER_20260805_222535.md)

| Run | 内容 | 状态 |
|---|---|---|
| RDGM-00 | pre-step 数据合同与无未来泄漏测试 | complete（本地） |
| RDGM-01 | depth graph / layer set / 零原边多拓扑 shuffled adjacency 测试 | complete（本地） |
| RDGM-02 | grouped validation、train-only normalization、五实验臂训练 | complete（本地） |
| RDGM-03 | 首错定位、正确链阈值、配对 bootstrap | complete（本地） |
| RDGM-04 | 每域 32 chain smoke | pending（远端） |
| RDGM-05 | smoke 完整性与稳定性审核 | gated |
| RDGM-06 | 全量四域 LODO | gated |
| RDGM-07 | result-to-claim | gated |

下一步：

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
```

自然 ProcessBench 没有受控因果 pair；这不妨碍检测实验，但 Attention/FFN 根因仍不可识别。
