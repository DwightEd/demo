# CCT-HG: Causal Constraint Transport Hypergraph

本项目不再实现基于 attention 阈值的旧 HyperCHARM 复现。唯一主线是一个面向
ProcessBench 的首错诊断器：**Causal Constraint Transport Hypergraph (CCT-HG)**。

## 核心假设

正确推理依赖题目约束向当前决策位置的输出有效输运。首个错误不是简单的 hidden state
弯曲或发散，而是同时出现：

1. 来自 prompt 约束的有效支持下降；
2. 残差更新逃离 prompt 写入张成的局部切空间；
3. 逃逸分量进入会改变当前输出概率的方向；
4. 多个 source 的联合消融产生不可由单 source 效应相加解释的协同效应。

对 source $j$、决策位置 $t$、层 $l$ 和 head $h$，输出有效贡献定义为

\[
c_{j\to t}^{l,h}
=a_{tj}^{l,h}
\left\langle
W_O^{l,h}W_V^{l,h}\operatorname{LN}(h_j^l),
g_t^l
\right\rangle,
\]

其中 $g_t^l$ 是观测 token 的对数概率相对于该层残差状态的梯度。它明确分离 attention
routing、OV 搬运内容和输出敏感方向。

由 prompt-origin 写入张成局部子空间 $T_t^l$，对目标层 attention residual update
$u_t^l$ 定义有界的输出敏感横向逃逸：

\[
e_t^l=
\frac{
\left|\left\langle(I-\Pi_{T_t^l})u_t^l,g_t^l\right\rangle\right|
}{
\lVert u_t^l\rVert_2\lVert g_t^l\rVert_2+\epsilon
}.
\]

该归一化避免平行与横向 logit 效应相消时分母接近零，且满足 $0\le e_t^l\le 1$。

source 集合只有在联合干预表现出非加性时才成为超边：

\[
\operatorname{Syn}(S\to t)
=\Delta\ell(S\to t)-\sum_{j\in S}\Delta\ell(j\to t).
\]

否则构图器退化为普通有向 pair edge，因此“高阶关系存在”是可证伪的实验结论，不是预设。

## 工程结构

```text
hypergraph/attention/
  cct/
    contracts.py       固定形状的数据契约
    processbench.py    ProcessBench 读取、渲染与 token 对齐
    hf_backend.py      Llama-family OV/梯度/干预抽取
    contribution.py    输出有效贡献
    geometry.py        prompt 条件化切空间与横向逃逸
    hypergraph.py      干预校准的 pair/hyperedge 构图
    pipeline.py        唯一机制组装流水线
    data.py            无 pickle、无自由 metadata 的 trace 存储
    model.py           仅更新 receiver 的有向超图网络
    hazard.py          首错离散生存目标
    training.py        训练集拟合归一化、训练、早停与测试
    cli.py             extract / inspect / train / benchmark
  evaluation.py        AUROC、AUPRC、校准、MCC 与定位指标
  splitting.py         problem-disjoint 固定划分
```

数学核心不接收任意 `metadata` 字典。模型、层、问题 ID、首错位置和 token 数均为显式字段；
JSON 只出现在 CLI 报告边界。

## 紧凑表示

直接保存完整 OV 写入需要 $O(HSD)$ 内存。抽取器先用所有 step 的输出梯度和残差更新建立
输出相关正交基，再保存

```text
content_effect  [heads, steps, sources]
source_writes   [steps, sources, projected_rank]
```

这保留 CCT-HG 使用的内积和横向逃逸，同时将主存储降为

\[
O(HQS+QSR),
\]

其中 $Q$ 是步骤数，通常远小于 token 数 $S$。所有可批量线性代数均在 GPU 上完成。

节点内容不使用每条样本各自旋转的局部坐标，也不保存 4096 维完整 hidden。抽取器为同一
cohort 使用固定 `projection_seed` 的 64 维 Johnson-Lindenstrauss sketch；它只是统一的
内容压缩层，`hidden_only` 控制使用完全相同的 sketch，因此不能被当作本文的机制贡献。

## 运行

从 `demo` 目录执行。抽取器不做人为长度截断；超过模型原生上下文长度时会明确失败。
普通模型前向默认使用 SDPA，只 hook 目标层并重建步骤 query 对应的 attention 行，不保存
任何层的完整 $N\times N$ attention。

```bash
python -m hypergraph.attention.cct extract \
  --input data/hf_datasets/ProcessBench/omnimath.json \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --output data/cct_traces/omnimath_layer14 \
  --layer 14 \
  --top-sources 3 \
  --dtype bfloat16
```

两张 GPU 可直接启动两个互斥 shard，不需要项目级 shell 包装：

```bash
CUDA_VISIBLE_DEVICES=0 python -m hypergraph.attention.cct extract \
  --input data/hf_datasets/ProcessBench/omnimath.json \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --output data/cct_traces/omnimath_layer14 \
  --layer 14 --num-shards 2 --shard-index 0

CUDA_VISIBLE_DEVICES=1 python -m hypergraph.attention.cct extract \
  --input data/hf_datasets/ProcessBench/omnimath.json \
  --model /path/to/Meta-Llama-3.1-8B-Instruct \
  --output data/cct_traces/omnimath_layer14 \
  --layer 14 --num-shards 2 --shard-index 1
```

两个进程按全局样本序号取模分片，写入同一个 trace 目录但文件名互不重叠；各自保存独立的
配置与失败清单。

```bash
python -m hypergraph.attention.cct inspect \
  --traces data/cct_traces/omnimath_layer14
```

```bash
python -m hypergraph.attention.cct train \
  --traces data/cct_traces/omnimath_layer14 \
  --output results/cct_hg/omnimath_layer14 \
  --epochs 100 \
  --batch-size 8 \
  --device cuda
```

正式实验应直接运行同划分控制矩阵：

```bash
python -m hypergraph.attention.cct benchmark \
  --traces data/cct_traces/omnimath_layer14 \
  --output results/cct_hg/omnimath_layer14_benchmark \
  --epochs 100 \
  --device cuda
```

该命令先拟合只看长度与 step 数的 `nuisance_only`，再依次训练 `full`、`hidden_only`、`no_edge`、`pairwise`、
`causal_cardinality_rewire` 和 `no_geometry`，并生成统一的
`benchmark.json`。

训练结果固定写入：

```text
model.pt
normalizer.npz
config.json
split.json
history.csv
metrics.json
predictions_validation.csv
predictions_test.csv
```

`metrics.json` 同时报告 response AUROC/AUPRC、balanced accuracy、MCC、Brier、ECE，以及
首错定位 top-1、mean rank 和 MRR，并按 `problem_id` 整组 bootstrap 给出 AUROC/AUPRC
置信区间。两个 prediction CSV 保存每条 held-out 样本的 response 概率、预测错误步和完整
step-risk 序列。测试集只在 validation 早停完成后评估。

训练将多条变长超图拼成互不连通的 disjoint batch，一次 GPU 前向后再按轨迹切片计算
等权生存损失；`--batch-size` 控制每批 response 数，不改变每条 response 的权重。

## 验证要求

正式结果必须包含以下对照，否则不能声称超图或几何机制有效：

- hidden-only / no-edge；
- pairwise-only；
- 保持 receiver、因果方向与边基数的 rewired topology；
- additive intervention（禁止产生超边）；
- length、step count、relative position 控制；
- 同题多采样与 problem-level bootstrap。

单一小测试集或 `accuracy@0.5` 不构成有效性证据。
