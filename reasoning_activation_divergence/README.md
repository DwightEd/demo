# Reasoning Activation Divergence

本项目只把行对齐的真实 response-token residual stream 作为实验输入。本地
`geometry_audit.npz` 标量代理结果不属于当前证据链。

## 方法

状态保持为 `[sample, time, layer, hidden]`。每个 component-grouped
cross-validation fold 只使用训练集中的全正确样本拟合：

1. sklearn randomized SVD 共享坐标系；
2. sklearn Ridge 深度与 token-time 仿射算子；
3. held-out 径向变化、深度残差、时间残差和 plaquette 路径分歧；
4. 特征值相位、proper polar rotation、orientation reversal、谱半径、条件数、
   有效秩和 Henrici 非正规性；
5. 匹配对 AUROC、bootstrap 区间、sign-flip 检验和方法差值。

这些是存储 residual states 上的经验局部算子，不是 autograd Jacobian，也不是
model-native logits Fisher。

## 工程结构

```text
config.py       参数与数据来源配置
domain.py       provenance、cohort、dataset、result 领域对象
source.py       audited manifest 与 mmap shard repository
matching.py     first-error/control 匹配窗口 builder
analysis.py     joint token-times-layer operator analyzer
statistics.py   唯一的配对统计实现
reporting.py    JSON、CSV、figure artifact writer
runner.py       单一实验应用服务
progress.py     tqdm/测试进度接口
raw_residual_experiment.py  CLI 与兼容入口
```

manifest 元数据不会作为大字典贯穿调用链。`SourceProvenance` 与
`CohortSummary` 在内存中保持类型化，只在 `ArtifactWriter` 输出边界序列化。

## 数据门禁

- generator 必须按 manifest 行显式过滤；
- 标签、step/token ranges 和 shard path 使用同一个行掩码；
- snapshot 必须声明 `raw_residual_stream`；
- first-error 与 fully-correct 两类都必须存在；
- reused row/problem group 必须留在同一个 fold；
- sklearn、tqdm 等依赖缺失时直接失败，不存在退化实现。

设计与迁移边界见 [REFACTOR_PLAN.md](REFACTOR_PLAN.md)，研究方法见
[REAL_RAW_RESIDUAL_METHOD.md](REAL_RAW_RESIDUAL_METHOD.md)，远端前台运行见
[RUN_RAW_REMOTE.md](RUN_RAW_REMOTE.md)。

## 验证

```bash
python -m pip install -e './reasoning_activation_divergence[test]'
python -m pytest reasoning_activation_divergence/tests -q
```
