# RDGM Experiment Tracker

**计划版本**：`EXPERIMENT_PLAN_20260805_222535.md`  
**更新日期**：2026-08-05

| Run | 目标 | 输入 | 主要输出 | 状态 |
|---|---|---|---|---|
| RDGM-00 | pre-step 数据契约测试 | 合成 geometry trace/memmap | step 0、风险集、无未来泄漏 | complete（本地） |
| RDGM-01 | 模型结构测试 | 合成 `[B,L,D]` | depth adjacency、set invariance、零原边多拓扑 shuffle | complete（本地） |
| RDGM-02 | 训练与分组测试 | 合成三域数据 | grouped validation、train-only normalization、5 arms | complete（本地） |
| RDGM-03 | 评估测试 | 合成首错链 | threshold、Top-1/MRR、paired bootstrap | complete（本地） |
| RDGM-04 | 真实四域 smoke | 每域 32 chain | `causal_monitor_smoke_<tag>/summary.json` | pending（远端） |
| RDGM-05 | smoke 审核 | RDGM-04 artifacts | 时间边界、类别/领域覆盖、训练稳定性 | gated by RDGM-04 |
| RDGM-06 | 真实四域 full LODO | 全部 3400 chains | `causal_monitor_full_<tag>/summary.json` | gated by RDGM-05 |
| RDGM-07 | result-to-claim | RDGM-06 summary/predictions | detection/graph claim status | gated by RDGM-06 |
| RDGM-08 | 组件受控干预 | controlled causal pairs | Attention/FFN root-cause evidence | separate; unavailable on natural ProcessBench |

## 当前实现审计

- 自然 ProcessBench 数据：3400 chains（2221 error / 1179 correct）；
- `target_correction=0`、`controlled_root=0` 不阻止 RDGM 检测实验；
- RDGM 读取 `<domain>/geometry/trace.npz` 与 `trace.states.pre.*.npy`；
- 不读取 post-step component artifacts；
- 所有远端命令前台运行并显示 tqdm；
- 当前未产生任何真实数据结果，不能填写效果数值。

## 下一命令

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
```
