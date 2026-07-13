# Layer-Time Geometry Experiment Tracker

更新时间：2026-07-13 10:03（Asia/Shanghai）
运行 ID：demo-method-redesign-20260713

| ID | Claim | 状态 | 产物/证据 | 下一动作 |
|---|---|---|---|---|
| S0 | schema/invariance | passed | 全库 `127 passed`；trace/teacher-forcing/layer-time tests | 加真实 tokenizer property test |
| S1 | separable synthetic | pending | 未运行正式生成器 | 增加可分离二维场 fixture |
| S2 | interaction synthetic | partial | Wilson gauge invariance 与 ID-shuffle transport break 已测试 | 增加已知非零曲率生成器 |
| E0 | claim-driven evaluator | passed-synthetic | first-error event、位置残差化、同题 paired、conditional Wilson tests | 在 G0 artifact 上运行 |
| G0 | real GPU exact smoke | blocked-external | 本地无 torch/transformers/GPU | 远端跑 64 problems × 4 samples |
| G1 | reference sensitivity | pending | max_reference 已参数化 | G0 后跑 256/512/all |
| G2 | JL sensitivity | pending | shared JL/exact 已参数化 | G0 后跑 0/32/64/128 |
| G3 | k/rank/reliability sensitivity | pending | fixed q、LID ablation、transport threshold 已参数化 | G0 后冻结 config |
| M1 | same-problem main | pending | problem-group OOF 已实现 | internal multisample 上确证 |
| P1 | first-error main | pending | gold labels 保留 | ProcessBench 上 event study |
| X1 | cross-task | pending | - | M1/P1 通过后运行 |
| X2 | cross-model | pending | - | 固定 config 后运行 |
| N1 | phase shuffle | implemented-unrun | `--null_mode phase_shuffle` | G0 后写独立 artifact |
| N2 | reference-ID shuffle | passed-synthetic | transport/curvature 显著被破坏 | G0 后写独立 artifact |

## 当前结果分析

- 旧方法先拼接层再计算一个 LID，无法回答整层演化；新 schema 已保留显式 layer 轴。
- 旧跨层邻域使用独立随机投影，会制造伪重连；新实现共享 JL，identical-layer depth rewiring 为 0。
- 新 field 对 correctness/first-error relabel 完全不变，说明几何构造本身 label-free。
- V2 已将 LID/rank front 与固定秩 connection 分离，并新增 Wilson eigenangle curvature、transport residual 与 reliability-gated curvature。
- 专用 evaluator 已替代仅看 pooled AUROC 的旧入口，能够直接检验局部事件和同题差异。
- 目前只有合成与 schema 证据；没有任何真实数据结果支持错误附近 holonomy 更高。

## 后续研究方向

1. 先完成 G0，确认真实 tokenizer、显存、存储和 OOF coverage。
2. 再完成 S1/S2 与 G1-G3，冻结几何超参。
3. 只在稳定性通过后跑 M1/P1；不先看测试集挑 layer band。
4. 若 Wilson event 在条件化 transport residual 后消失，停止 holonomy 主线，退回 neighborhood bifurcation 的描述性研究。

## 优化建议

- 以 token progress/arclength 替代 normalized-step phase 做稳健性比较。
- reference interpolation 与 kNN 可按 phase cache，并行各 fold。
- 全层 fp16 states 使用分片写盘；审计仍保留 float32 field。
- 增加 transport bootstrap confidence 和 low-rank singularity coverage 报告。
