# Round 1 Review

> 外部 `gpt-5.5` reviewer 经两次调用仍未在十分钟硬超时内返回。本文件是明确标注的内部五轴 fallback，不冒充外部评审。

| Dimension | Score | Assessment |
|---|---:|---|
| Problem Fidelity | 9 | 直接解决长度污染指标无法识别计算机制的问题。 |
| Method Specificity | 8 | capture、chart、metric 与 artifact 接口清楚；factorial patch 尚未实现。 |
| Contribution Quality | 8 | 一个主贡献明确，但需避免把观测性 MLP gain 写成因果 correction。 |
| Frontier Leverage | 8 | 对 frozen instruction LLM 做 source/component causal measurement，路线合适。 |
| Feasibility | 8 | 复用现有 replay 与 chart；短 prompt selective hooks 可行。 |
| Validation Focus | 7 | 三个核心量足够，但需要 reconstruction 与预注册层 fail-closed。 |
| Venue Readiness | 7 | controlled result 可形成机制论文；自然推理迁移和 factorial patch 决定最终强度。 |

**OVERALL SCORE**: 8.05 / 10

**Verdict**: REVISE

## Required fixes

1. 把 `attention mass × cosine margin` 明确降为 routing/head-selection diagnostic；真实更新量使用 signed target progress 与 relative target error。
2. 实际 block delta 必须与 attention output + MLP output 做 reconstruction 检验，超过阈值时 fail closed。
3. audit 必须要求显式 `primary_layer`，其他层仅描述。
4. 在没有 MLP intervention 前使用 `mlp_update_signature`，不得使用 causal `mlp_correction_supported`。
5. 不跨层直接求和，因为每层 chart 虽映射到同一 analytic target，但局部线性误差不同。

## Simplification Opportunities

- 不修改既有 routing artifact v1；新增独立 block-update artifact，避免破坏远程旧数据。
- 不在本次实现 factorial patch；先用明确 gate 决定它是否值得运行。
- 不新增 probe 或 trainable component。

## Modernization Opportunities

NONE。当前 frozen-model mechanistic intervention 路线已经合适，不需要额外训练框架。

## Drift Warning

NONE。
