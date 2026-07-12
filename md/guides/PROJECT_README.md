# ECGH：证据耦合的推理错误定位与干预研究工程

本仓库当前主线是 **Evidence-Coupled Geometry Hazard（ECGH）**：在实际生成上下文中，追踪 response 与题设证据锚点的连接、锚点不能解释的二阶几何变化、首次错误 hazard，以及等计算量的局部修复。

旧的 `(Step × Layer) spectral field`、scalar HMM、tube、qvec AnchorFlow 和 hypergraph 脚本仍保留为历史基线，但不再是 Quick Start，也不应单独支撑 paper-ready 主张。

## 方法对象

```text
exact generation trace
  -> semantic prompt-span anchors
  -> compact causal lookback
  -> anchor-residual Gram geometry
  -> boundary-free causal event
  -> first-error survival hazard
  -> micro-replay / repath / abstention
```

对窗口 hidden cloud \(H_t\) 和 prompt anchor bank \(A_t\)，先投影掉 anchor 子空间：

\[
R_t=H_t(I-P_{A_t}),
\qquad
G_t=\frac{1}{n_t}R_tR_t^\top.
\]

首次错误不再用独立 token BCE，而用离散 hazard：

\[
q_t=P(T=t\mid T\ge t,\mathcal H_{\le t}),
\qquad
S_t=\prod_{s\le t}(1-q_s).
\]

正确链是右删失样本，错误发生后的位置不进入风险集。

详细设计与实验门槛：

- [方法重设计报告](../../research-reports/IDEA_DISCOVERY_LATEST.md)
- [claim-driven 实验路线图](../../refine-logs/EXPERIMENT_PLAN_LATEST.md)
- [《The Phenomenology of Hallucinations》机制拆解](../../research-reports/PHENOMENOLOGY_LATEST.md)
- [artifact manifest](../../MANIFEST.md)

## 1. 本地安装与确定性检查

```bash
python -m pip install -r requirements-dev.txt
python ecgh_pipeline.py doctor
python ecgh_pipeline.py selftest
python -m pytest -q
```

`doctor` 会区分 CPU、本地 GPU、HyperGNN、真实数据和干预原型状态。`selftest` 是 CPU-only 端到端合成门：semantic anchor、compact lookback、anchor-residual Gram、causal event、right-censored hazard 和 micro-replay 必须全部连通。

NTS/Hydra 配置入口可单独检查：

```bash
python scripts/run_gate.py --help
```

## 2. GPU：生成与 exact trace 抽取

推荐先跑小样本 schema gate，再扩数据：

```bash
python 10_sample_and_extract.py \
  --model /path/to/Llama-3.1-8B-Instruct \
  --dataset_format processbench \
  --dataset /path/to/ProcessBench \
  --subset gsm8k \
  --n_problems 32 \
  --k_samples 4 \
  --prompt_style lm_eval_5shot \
  --layers 8,16,24,32 \
  --store_prompt_hidden \
  --prompt_hidden_layers 16 \
  --store_clouds \
  --cloud_layers 16 \
  --store_token_uncertainty \
  --output data/gsm8k_ecgh_smoke.npz
```

抽取契约是 `exact_generation_trace_v1`：

- generation 和 teacher-forcing 复用同一 rendered prompt、prompt IDs 和 generated response IDs；
- 不重新 tokenize `prompt + response`；
- 保存 prompt/response IDs、mask、offsets、question span、完整与 kept step ranges、time-axis metadata；
- 可保存选定层 prompt hidden 与 raw step clouds；
- 任一 token、offset、prefix 或 step-range 错位都会抛出 `TokenAlignmentError`，不会静默写出 artifact。

抽取后先做结构审计：

```bash
python ecgh_pipeline.py trace-audit \
  --input data/gsm8k_ecgh_smoke.npz \
  --layer 16 \
  --output outputs/trace_audit/gsm8k_smoke.json
```

只有 `semantic_anchor_ready` 与 `conditional_geometry_ready` 达到预期，才扩到 128×8 或全量。

若需要直接验证 first-error hazard，可先在带 ProcessBench 首错标签的 teacher-forced 轨迹上开发：

```bash
python 01_extract_spectral_field.py \
  --model /path/to/Llama-3.1-8B-Instruct \
  --dataset /path/to/ProcessBench \
  --subset gsm8k \
  --n_correct 50 \
  --n_error 50 \
  --layers 8,16,24,32 \
  --store_prompt_hidden \
  --prompt_hidden_layers 16 \
  --store_clouds \
  --cloud_layers 16 \
  --step_vectors \
  --output data/processbench_gsm8k_ecgh.npz

python ecgh_pipeline.py trace-audit \
  --input data/processbench_gsm8k_ecgh.npz \
  --layer 16 \
  --output outputs/trace_audit/processbench_gsm8k.json
```

该路径的 metadata 会明确标记 `teacher_forced_processbench`；它适合开发定位器，但不能冒充真实在线生成。`10_sample_and_extract.py` 的新生成样本只有 chain correctness，若要训练 first-error hazard，还需独立 step verifier / 人工首错标注。

## 3. 二阶矩非参数 gate

在考虑 Matrix-HMM/Wishart state model 前，先运行透明 gate：

```bash
python second_moment_dynamics_audit.py \
  --input data/gsm8k_ecgh_smoke.npz \
  --layer 16 \
  --policy answer_format_ok \
  --output_dir outputs/second_moment_dynamics
```

该审计使用 direct raw/centered token-matrix Gram、problem-grouped OOF、fold 内预处理与 problem-cluster bootstrap，主判据是 `baseline + Gram` 相对 spread/entropy/length baseline 的同题增量。项目现有真实数据结果为负：Gram/spectral-tail 低于强 baseline；在新 exact trace 上复验前，它是已失败的旧主线而不是默认创新点。只有增量置信区间稳定大于零，才考虑 matrix-state 模型。

## 4. temporal audit

```bash
python multisample_temporal_rupture_audit.py \
  --input data/gsm8k_ecgh_smoke.npz \
  --policies answer_format_ok \
  --output_dir outputs/multisample_temporal_rupture
```

token entropy/committal 只有通过 `cloud_sizes` 或显式 token-to-step ranges 池化后，才可与 step channel 融合。所有多通道序列必须具有完全相同的 step 数；不再按 `minT` 截断硬拼。

## 5. 旧/上界流程

| 流程 | 定位 | 当前使用方式 |
|---|---|---|
| `scripts/run_gate.py` | NTS 数学证据门 | CPU 可运行；真实数据另需对应 features/hidden |
| `anchorflow_anchor_audit.py` | qvec fallback 历史基线 | 只作 baseline，不称 semantic anchor |
| `second_moment_dynamics_audit.py` | grouped-OOF direct Gram gate | 当前真实结果为负；只作复验/条件专家门槛 |
| `hypergraph_token_hgn.py` | 结构化神经上界 | 需 Torch/PyG/GPU；不是默认 reader |
| `online_intervene.py` | 旧离线 repair 原型 | 非真实 streaming，不用于净收益主结论 |
| 旧 spectral/tube/HMM 脚本 | 负结果与历史基线 | 保留复现，不再作为根入口 |

HyperGNN synthetic upper bound（GPU/PyG 环境）：

```bash
python hypergraph_token_hgn.py --selftest --device cuda
```

## 6. 评价协议

- 按 `problem_id` GroupKFold；同一道题的所有 sample 不跨 fold。
- 插补、标准化、残差化、阈值与超参只在训练 fold 拟合。
- 主指标：same-problem pair-micro AUROC、problem-macro AUROC、固定正确链 FPR 下 recall/delay、Brier/ECE。
- 置信区间按 problem cluster bootstrap。
- 真实 anchor 必须比较 qvec、random vector、span shuffle、kind shuffle。
- 干预必须在所有触发链上报告 wrong→right、correct→wrong、额外 tokens、coverage 与净收益。

## 7. Go / no-go

1. token/offset/range 对齐不是 100%：停止下游。
2. semantic anchor 不优于 qvec 与 span-shuffle：删除语义主张。
3. conditional Gram 对强基线无同题 OOF 增量：不做 Matrix-HMM。
4. boundary-free event 只在链尾触发：删除 local rupture 主张。
5. 等预算 intervention 净收益不为正：只保留检测/机制，不声称闭环。

## 8. 配套 research skills

ARIS Codex 套件安装在项目根目录：

```text
D:\projects\research\.agents\skills       # 80 个 SKILL.md
D:\projects\research\.agents\skills\shared-references
D:\projects\research\.aris\tools
D:\projects\research\.aris\installed-skills-codex.txt
```

Codex 通常在新一轮会话中重新发现项目本地 skills。
