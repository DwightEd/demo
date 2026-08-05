# PTIH Experiment Tracker

| Run | 内容 | 状态 |
|---|---|---|
| RDGM-NEG | 固定深度图与 shuffled adjacency smoke | 完成；未发现稳定图结构增量，主线停止 |
| PTIH-00 | 冻结主张、同容量对照与停止条件 | 完成 |
| PTIH-01 | 同链两边界数据接口与无未来泄漏测试 | 完成 |
| PTIH-02 | static / bag / innovation 等容量模型 | 完成；语法编译通过 |
| PTIH-03 | LODO 训练、`t>=1` 主检验与结果持久化 | 完成；待远端 pytest |
| PTIH-04 | 每域 32 chains smoke | 待远端运行 |
| PTIH-05 | 全量四领域实验 | 由 smoke 完整性门控 |
| PTIH-06 | result-to-claim | 由全量结果门控 |

下一条运行命令：

```bash
bash run_hidden_geometry_remote.sh causal-monitor-smoke
```
