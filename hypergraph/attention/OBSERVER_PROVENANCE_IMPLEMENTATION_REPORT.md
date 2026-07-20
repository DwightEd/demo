# Attention Hypergraph：Observer Provenance 与同生成器重放实现报告

生成时间：2026-07-21 01:58:42 +08:00

关联计划：`OBSERVER_PROVENANCE_AND_MATCHED_GENERATOR_PLAN.md`

## 1. 结论

本轮没有改动“一个 token 一个节点、attention row 构造 receiver-centered 超边”的主方法定义，修复的是此前导致 ProcessBench 训练立即失败的数据协议和运行入口。

原失败的直接原因是：全部回答都由同一个 Llama observer 前向得到 attention，但旧 cohort fingerprint 仍把每条回答的 `generator_model` 当成表示模型身份。同一数据集包含多个 generator，于是 preflight 只做结构检查时通过，训练的整批兼容性检查随后立即拒绝。

修复后的语义分为两条实验：

1. 主实验：只选择数据中 `Llama-3.1-8B-Instruct` 来源回答，再由本地 `Meta-Llama-3.1-8B-Instruct` teacher-forcing 前向。它是 **generator-tag-matched reconstructed observer**，不是原生成时轨迹，也没有证明 checkpoint 权重完全相同。
2. 辅助实验：所有 generator 的冻结回答都由同一个 observer 前向。它研究统一表示空间中的跨来源可检测性，必须报告 generator 分层结果，不能作为原 generator 内部机制证据。

## 2. 已实现内容

| 模块 | 实现状态 | 核心约束 |
|---|---|---|
| M1 provenance 拆分 | 完成 | source generator 仍逐样本保存；只有验证过的 observer replay 才从 representation fingerprint 排除 source generator |
| M2 generator cohort | 完成 | exact、case-insensitive dataset tag 筛选；筛选先于 limit；主 wrapper 只允许明确的 Meta-Llama/Llama 别名 |
| M3 shared cohort gate | 完成 | 带 objective 的 inspect 与 train 共用同一 loader、scope audit、representation/axis/replay gate |
| M4 cache/run gate | 完成 | trace extraction hash 与 training hash 分离；legacy request 不重写；run gate 绑定 effective config 与 preflight SHA，cross summary 记录并核验 run-gate SHA |
| 两卡入口 | 完成 | 默认 model-parallel 抽取；fold 训练轮流调度到两张卡；matched/all-generator 使用不同目录 |
| 分层分析 | 完成 | 输出 generator×label 表、prediction generator 字段和各 generator 的 held-out fold/seed 指标 |
| 长度审计 | 部分完成 | 已记录长度/step/首错位置分布；length-only baseline 与 class-conditional length analysis 留给正式实验矩阵 |

## 3. 关键安全不变量

- observer representation compatibility 仍绑定 observer model、tokenizer、prompt protocol、replay fidelity、dtype、提取方法和实际 layer/head axis。
- same-generator 或缺失/未知 replay 状态仍 fail closed，不会借 observer 规则放宽。
- 完整 extraction scope 在 generator 过滤和 limit 之前审计。
- 多 trace 根目录上的训练端 `--limit` 会被拒绝，避免 shard0/shard1 遍历顺序改变样本集合；data-parallel wrapper 在分 shard 前应用 limit。
- preflight 先写到 run 目录外的临时文件；只有 run config gate 通过后才原子替换正式 `preflight.json`，不会污染已有结果证据。
- aggregate 必须得到精确的 folds×seeds 文件集合，且总体 held-out AUROC/AUPR/accuracy 必须全部有限；跨数据集宏平均不允许缺数据集。
- all-dataset 汇总重新计算 preflight SHA，并与每个 dataset run gate 中绑定的 SHA 对照；随后检查 representation、axis、graph 和当前 validation/training code 一致性。legacy/v2 request 类型分别留痕，但只要 trace 内的真实 representation provenance 一致，不因 sidecar schema 不同而伪造不兼容。
- trace、`pipeline_request.json`、`shard_audit.json` 和 matched cohort report 必须一起保存。

## 4. 代码范围

本轮相关实现位于：

- `hypergraph/attention/data.py`
- `hypergraph/attention/extract.py`
- `hypergraph/attention/trace_contract.py`
- `hypergraph/attention/pipeline_guard.py`
- `hypergraph/attention/cohort.py`
- `hypergraph/attention/train.py`
- `hypergraph/attention/scripts/run_single_layer_response_pipeline.sh`
- `hypergraph/attention/scripts/run_all_processbench_response_pipeline.sh`
- `tests/test_attention_hypergraph.py`

操作与方法说明位于本目录的 `README.md` 和 `ALL_PROCESSBENCH_PIPELINE.md`。

## 5. 本地验证

- attention/hypergraph 相关测试：`63 passed, 6 skipped`
- 目标 attention 测试：`45 passed, 5 skipped`
- Ruff：通过
- Python compile：通过
- 两个 shell 文件中的全部 Python heredoc：编译通过
- Git for Windows Bash 对 LF 规范化后的两个 wrapper 执行 `bash -n`：通过
- Git index 中两个 shell blob：纯 LF（CR=0），并直接由 staged blob 执行 `bash -n` 通过
- `git diff --check`：通过

仓库加入 `*.sh text eol=lf`，staged blob 已确认为纯 LF；Linux 服务器 pull 后仍应再次执行 `bash -n` 作为部署环境确认。

## 6. 仍不能声称的内容

- 不能把 reconstructed observer attention 称为原生成模型的因果内部轨迹。
- 本地 checkpoint 路径没有权重内容 manifest；模型 tag 匹配不等于 exact-weight 匹配。
- cohort preflight 不等于所有 fold 都有双类，也不保证优化成功。
- 记录长度分布不能消除长度混杂。正式结果前必须加入 length-only baseline、训练集冻结的 length bins，以及 class-conditional score–length 分析。
- 尚未在服务器上重新运行 GPU 主实验，本报告只证明代码路径和确定性门禁通过本地验证。
