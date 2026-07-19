# Experiment Audit Report

**Date**: 2026-07-20
**Auditor**: GPT-5.6-Sol ultra（fresh same-family agent，read-only，provisional）
**Project**: faithful attention-row hypergraph
**Audited scope**: `hypergraph/attention/**`、`tests/test_attention_hypergraph.py`

## Overall Verdict: WARN

## Integrity Status: warn

代码完整性检查没有 P0/P1 缺陷，可以提交为“实现与真实服务器 pilot 入口”。WARN 的唯一
根因是尚未在用户给出的 Meta-Llama-3.1-8B-Instruct checkpoint 和两张 24GB GPU 上运行
pilot，因此不能声称真实吞吐、显存或模型上的 cached-query 等价性，更不能声称检测效果提升。

## Checks

### A. Ground Truth Provenance: PASS

- ProcessBench `gold_step` 必须由数据提供；缺失时直接拒绝，不会伪造 `-1`。
- step 标签不会扩张成伪 token 标签；partial token labels 也不会伪造 negative response。
- 跨 token/step/response 粒度冲突和字符串布尔标签均 fail closed。

证据：`extract.py:173-224`、`construction.py:457-548`、`data.py` 的 trace canonicalization。

### B. Score Normalization: PASS

- AUROC/AUPR/F1 使用原始 held-out prediction；不存在除以自身 max/min/mean 的指标美化。
- 非有限 prediction/metric 会报错，JSON 只在序列化阶段把非有限诊断值转为 `null`。
- `per_graph_zscore` 是显式 feature preprocessing，不是 score normalization；对 prefix 任务默认拒绝。

证据：`train.py:1576` 及 token/step/response objective 协议检查。

### C. Result File Existence: PASS

仓库没有伪造的真实 Llama/GPU 结果，也没有效果提升数字。README 明确说明真实 checkpoint
pilot 尚未执行。因此“实现已完成”有代码与测试证据，“效果好/加速已实测”当前均不作声明。

### D. Dead Code and Release Gates: PASS

- trace、manifest、shard audit 文件均原子写入；成功样本后立即 checkpoint manifest。
- cached trace 只有在 selector=`threshold`、无 top-k、训练阈值等于门禁阈值时可用于正式训练。
- shard cohort 检查输入 SHA256、modulo membership、重复行、缺 shard/row 和 failed/skipped row。
- 抽取方法 fingerprint 已绑定 `attention/*.py` 与 `utils/step_boundaries.py` 的 SHA256。

证据：`extract.py:589`、`extract.py:733`、`extract.py:1012`、`shards.py:85`、`train.py:828`。

### E. Scope Assessment: WARN

- 普通 Python：`32 passed, 5 skipped`；跳过项均为缺少 PyTorch。
- research/PyTorch 环境：`37 passed`。
- 独立 release suite：`50 passed, 6 skipped`；Ruff、compileall、diff-check 与三个 CLI help 通过。
- 独立 CPU Transformers Llama smoke 的 full-vs-cached attention/activation 未出现 topology flip。
- Linux 双卡 wrapper、用户的 8B 权重、真实长度分布、RAM/scratch 预算仍未实测。

### F. Evaluation Type

- 计划中的 ProcessBench supervision：`real_gt`。
- 单元测试中的 fake model/cache：`simulation_only`，只验证张量契约。
- observer teacher forcing：`real_gt` 标签上的 counterfactual observer representation，不能解释为原生成器机制。

## Action Items

1. 先跑 4-sample 双卡 pilot，保留两份 log、manifest 与 `shard_audit.json`。
2. 要求零 failed/skipped row，且不使用任何 `--allow-*` diagnostic bypass。
3. 在代表性短/中/长序列上另跑少量 `--query-chunk-size 0` 完整前向基准。
4. 再做 length-only、degree/lag-matched rewiring、pairwise 与容量匹配 no-graph 控制。
5. 效果结论使用 problem-level paired bootstrap 和多 seed，而不是单次点估计。

## Claim Impact

- C1 “忠实实现原 attention-row HyperCHARM 默认图/模型”：`supported`（provisional same-family audit）。
- C2 “双 24GB GPU 上能稳定完成 8B 抽取”：`needs_real_pilot`。
- C3 “cached-query 与完整前向在真实 8B 上等价”：`needs_real_pilot`。
- C4 “超图提高幻觉检测/首错定位”：`unsupported_before_experiments`。
- C5 “attention 揭示幻觉因果机制”：`unsupported`；当前只允许 routing proxy/diagnostic 表述。

`review_independence=same-family`
`acceptance_status=provisional`
