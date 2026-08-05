# Reasoning Activation Divergence

当前推荐方法是 **CFEA（Controlled First-Error Attribution and Repair）**。它把首错检测、错误传播、修复和机制根因分开，避免把一个预测探针的分数误写成 Attention/FFN 的因果解释。

完整方案见 [GPT56_CFEA_DESIGN.md](refine-logs/GPT56_CFEA_DESIGN.md)。旧的 `hidden_state_geometry` 与 `component_resolved_hazard` 保留用于复现实验历史，不再是机制归因的推荐入口。

自然 ProcessBench 的首错检测现由 **PTIH（Prefix-conditioned Two-Boundary Innovation Hazard）** 承担：它读取相邻两个步骤生成前的完整 `[layer, hidden]` 状态，用同容量静态和无方向对照检验有向更新是否提供独立风险信息。实现和运行方式见 [PROCESSBENCH_TWO_BOUNDARY_MONITOR.md](PROCESSBENCH_TWO_BOUNDARY_MONITOR.md)。

## CFEA 的执行路径

入口是：

```bash
python -m functional_divergence.causal_first_error_attribution.main <command>
```

主流程为：

```text
audit
  -> extract recipient/counterfactual decision graphs
  -> intervene with Attention x FFN x pre-state patches
  -> summarize claim scope
  -> evaluate first-error localization separately
```

各命令的职责：

- `audit`：不加载模型，核验数据究竟支持自然 repair 还是受控 root-cause 分析。
- `extract`：在目标 token 前物理截断 prefix，提取 layer/head/source-token residual-message graph。
- `intervene`：仅对 `controlled_root` pair 做 Attention×FFN 四格 patch 与 pre-state patch。
- `summarize`：汇总已保存干预；null controls 缺失时只报告 `controls_pending`。
- `evaluate-monitor`：按链计算首错 Top-1、MRR 和正确链/步骤 false alarm。
- `train-monitor`：直接训练并评估 ProcessBench pre-step 两边界创新风险模型，不需要 causal pair。

## 文件职责

```text
src/functional_divergence/causal_first_error_attribution/
├── main.py           CLI 参数解析与线性调度
├── contracts.py      pair 与 onset-trace 数据契约
├── audit.py          无模型可识别性审计
├── pairs.py          pair 读取与首分歧边界
├── trace_data.py     从 trace.npz 读取并裁剪决策 prefix
├── replay.py         residual-message graph 提取
├── extraction.py     多域/多 pair 前台提取流程
├── interventions.py Attention、FFN、pre-state rerun patch
├── experiment.py    受控干预保存与汇总
├── analysis.py       分组 bootstrap 与 claim gate
├── evaluation.py     LODO 分组与链内首错定位
├── monitor_data.py   causal pre-step 风险集与 memmap 读取
├── monitor_models.py static / bag / directed-update 同容量模型
├── monitor_training.py 分组训练、归一化与 early stopping
└── monitor_experiment.py 四域 LODO、阈值和配对 bootstrap
```

核心类与公开方法：

- `PairAuditor(...).run()`
- `OnsetTraceExtraction(config).run(model)`
- `InterventionExperiment(config).run(model)`
- `CausalInterventionRunner(layer=...).run(...)`

参数只从 CLI 进入，经 dataclass 配置传给工作流类。模型只在 `extract` 或 `intervene` 已确认存在有效 pair 后加载。

## 数据布局

当前服务器上已经核验的根目录：

```text
/share/home/tm902089733300000/a903202310/lys/research/demo/data/exact/
└── processbench_observer_llama31_full/
    ├── gsm8k/selected/
    ├── math/selected/
    ├── olympiadbench/selected/
    ├── omnimath/selected/
    └── controlled_math/selected/       # 生成后通过 CAUSAL_DOMAINS 加入
```

每个域的 CFEA 输入为：

```text
selected/
├── trace.npz
└── causal_first_error_v1/
    └── onset_pairs_v1.jsonl
```

`target_correction` 只支持 target reference / mediation / repair；只有 `controlled_root` 可以进入 root-cause candidate。纯受控域不要求 ProcessBench 的 `gold_error_step`。

## 远端前台运行

先审计，不加载模型：

```bash
bash run_hidden_geometry_remote.sh causal-audit
```

有已验证 pair 后运行 smoke：

```bash
bash run_hidden_geometry_remote.sh causal-extract-smoke
bash run_hidden_geometry_remote.sh causal-intervene-smoke
```

加入受控域：

```bash
CAUSAL_DOMAINS=controlled_math bash run_hidden_geometry_remote.sh causal-full
```

脚本在前台运行并显示 `tqdm` 进度，不使用 `nohup`、`screen` 或 `tmux`。

## 结论门槛

图分数是描述性候选，不是因果效应。干预改变 margin 也不能自动升级成根因。正式路由/FFN 结论还必须超过 random donor、wrong source/head/layer、norm-matched noise 等对照，并产生 token/step rescue，再按 domain → problem/template 分组 bootstrap。

## 验证

```bash
python -m pytest tests/causal_first_error_attribution tests/test_remote_runner.py -q
```
