# Attention-row HyperCHARM：忠实迁移与可控创新

这个子目录重新实现原 `hypergraph-hallucination` 的核心方法，而不是继续把
hidden-kNN 当作原 attention 超图：**完整 prompt+response 序列中的一个 token
就是一个节点；一个 response query 在某个 layer/head 上超过阈值的历史 token
集合，加上 query 自身，构成一条超边。**

默认的构图、消息传递与单图训练超参数忠实保留之前的方法；step/response 目标、
严格 split 与 prefix-directional receiver 模式属于场景适配或协议修复。可创新的细节全部是显式开关，并且写入构图
manifest、训练配置和 checkpoint；因此“原方法能否迁移”和“某项改进是否有效”
可以分别回答。

## 当前实现

| 文件 | 职责 |
|---|---|
| `schema.py` | NumPy 图结构和忠实默认配置 |
| `extract.py` | 从 ProcessBench JSON/JSONL 校验原生成 token 轴并抽取 dense attention |
| `data.py` | `.npz/.pt` trace 适配、完整 token 轴与标签粒度校验 |
| `construction.py` | attention-row 阈值超边与严格 schema 校验 |
| `model.py` | 原式 HyperCHARM 消息传递及显式创新开关；不依赖 PyG |
| `objectives.py` | 严格区分 token、first-error step、response 三种监督 |
| `train.py` | inspect、批量构图、group-aware/explicit-split 训练评估 |

## 原代码事实与本项目的对齐协议

设完整序列 token 为 \(v_0,\ldots,v_{N-1}\)，response 从 \(r\) 开始。对于
原 `processed_hypergraph.py` 对 response receiver \(i\) 和实际被循环处理的第一个
flattened head 构造：

\[
S_i=\{j<i:A_{i,j}>0.05\}.
\]

若 \(S_i=\varnothing\)，原代码从所有正注意力历史 token 中取最多 16 个最大值作为
fallback；若 fallback 后仍少于 2 个 source，则丢弃该 row。最后构造
\(e_i=S_i\cup\{i\}\)。因此原实现不是“纯 threshold + 一个 source 即保留”。

必须区分节点特征与超边特征：

- 节点特征 `x` 是 self-attention diagonal，原文件形状为
  `[num_tokens, num_stored_layers * num_heads]`；本单层实验抽取一层全部 32 个 head，
  所以节点维数为 32，不是 3；
- 三个数 `[attention_mean, attention_max, normalized_head_id]` 是每条**超边**的属性；
- 原文件在外层 head 循环末尾有无条件 `break`，所以拓扑实际上只来自第一个 flattened
  head，而节点仍保留所有已存 layer/head 的 diagonal。正式结果把这一事实记录为
  `TOPOLOGY_HEADS=0`；all-head 图必须作为修正版消融，不能冒充原代码复现。

原对齐 response wrapper 的冻结配置是：

```text
threshold=0.05                 source_selection=threshold_fallback_topk
top_k=16                       min_sources=2
include_center=true            source_scope=all_past
topology_heads=0               node_feature_heads=all extracted heads
incidence_weight_mode=uniform  propagation_mode=symmetric
edge_attr_mode=faithful (3-D)  node_feature_mode=attention_diagonal
preprocessing=per_graph_zscore model_layers=2
pooling=mean                   objective=response_bce
split_mode=fixed_holdout       split_seed=17
val_ratio=0.1                  test_ratio=0.2
```

ProcessBench 没有原项目 RAGTruth 的外部 train/test 目录，因此这里只能采用其最接近且可审计的
协议：按 `problem_id` 一次性固定 train/validation/test，validation 选择 epoch，test 只评估
一次。它是原训练协议的 ProcessBench 适配，不是原论文外部测试集的字面复现。

通用 `train.py` 仍保留纯 threshold、all-head 和 group-CV 作为显式诊断/消融开关；这些不是
上述 wrapper 的默认主结果。其余共同约束是：

- 所有 prompt 与 response 历史 token 均可成为 source；
- receiver/center 是超边成员；
- 节点特征是所有 layer/head 的 self-attention diagonal；
- incidence 均匀聚合；
- 超边消息对称广播给所有成员；
- `he_mark=[prompt_cross,response_only]` 告诉消息 MLP 该超边是否含 prompt source；
- 主复现按原单图脚本对每张图的 node/edge feature 分列标准化；
- 两层 HyperCHARM、residual、128 hidden dimensions，消息 MLP 含 LayerNorm；
- 原 token 任务直接输出逐 token logit，不做 pooling；新增的 step/response 读出默认使用
  mean，以降低 max 随长度增长的偏差。

`target_alignment=same_index_post_emission` 是冻结的 schema 字段，不是 CLI 开关。
`min_sources` 只计算历史 source，不包含 receiver；原代码在加入 receiver 前执行第一次
`m_len < MIN_MEMBERS_IN_HE` 检查，所以应为 2。

原仓库的两个训练入口在预处理上并不一致：单图脚本执行逐图 clamp + 列 z-score，
multi-batch 脚本则注释掉了这两行。这里把 `per_graph_zscore` 冻结为活动单图主复现
preset；`--preprocessing none` 只复现 multi-batch 的这一项预处理选择，不能单独称为
完整 multi-batch 复现。

threshold 也不是原仓库所有历史文件唯一一致的常量，但当前本地
`processed_hypergraph.py` 与 attribute 脚本都写的是 `0.05`。原对齐 wrapper 因而冻结
`tau=0.05 + per_graph_zscore + classifier LayerNorm + Xavier init`。multi-batch 入口除
`tau=0.05`、无预处理、无 classifier norm/init 外，还使用真实 PyG batch size 2、
lr=5e-4、hidden={64,128}、layers={3,4}、20 epochs、warmup=0.02、
patience=3、pos-weight cap=8；当前逐图梯度累积不等价于该 PyG 节点级 batch。因此这里只
提供这些结构/预处理开关，不声称已经完整复现 multi-batch preset。
该脚本虽在 search space 写 `dropout=0.05`，消息层 dropout 调用已注释，classifier 又
硬编码为 `0.1`，所以 `0.05` 是 dead hp，不能当成有效模型设置。

还有一个必须冻结的 target alignment：原代码用 attention row `i` 和 self-diagonal `i`
分类 token `i`。标准 causal LM 的 row `i` 已经读入 token `i`，其 logits 通常预测
token `i+1`；因此 A1-F 是继承原版的 **same-index post-emission/post-hoc detector**，
不是 token `i` 生成前的机制预测器。receiver-only 只消除图上的 future→past 回流，不能
修复这一位移。它可用于“token 已输出后立即标记”，step pooling 则是在该 step 已观测后
定位。若要讨论生成机制，必须另做 `row i-1 → label i` 的 next-token-aligned 实验，并
单独处理首个 response token、中心 incidence 与 source/receiver 双角色特征；本轮未把
这个尚未定义清楚的变体混入忠实基线。

## 数据契约：旧 features 不能直接复现 attention 方法

构图要求每条样本保存：

```text
attention     float [layers, heads, N, N]
token_ids     int   [N]
response_idx  int   scalar，完整序列中第一个 response token 的位置
attention_layers int [layers]，可选；保存的 row 对应原模型 block id
attention_heads  int [heads]，可选；保存的 row 对应原模型 head id
num_model_layers / num_model_heads，可选；用于保持子集抽取后的原始 layer-head id
```

可选监督/特征：

```text
activation      float [N,F]、[L,N,D] 或 [N,L,D]
token_y         float [N] 或 [N-response_idx]，仅限精确 token span 标签
token_label_mask bool [N] 或 [N-response_idx]，可选；标出真正有精确监督的位置
step_ranges     int [steps,2]，完整序列绝对、半开区间 [start,end)
gold_step       int，-1 表示整条推理正确
step_loss_mask  bool [steps]，仅表示与标签无关的缺失监督，不能由 gold_step 生成
response_y      float scalar，1 表示存在幻觉/错误
problem_id      group-aware split 使用的题目 id
split           train/validation/test（若采用显式划分）
```

兼容旧字段别名包括 `attentions`、`input_ids`、`response_start`、
`hallucination_labels`、`gold_error_step` 等。response-only token 标签会在 prompt
位置填入 `-100`，相应 `token_label_mask=False`，不会误当成负例。若未提供 mask，
adapter 只从 response 中的真实 `0/1` 标签推导；prompt 永远不进入 token loss。
ProcessBench 抽取同样不会把缺失 `gold_step` 静默填成 `-1`：首错标签必须明确提供；多个
别名、字符串布尔值以及与首错标签冲突的 response 标签都会被拒绝。旧 `risk_mask` 也不再
作为 step 有效性掩码，因为它常由 gold step 派生；只能使用与标签无关的 `step_loss_mask`。

当前 `demo/data/features/full_*.npz` 主要保存 pooled prompt vector、step/token
hidden 派生量，**没有完整的 token-token、layer-head attention tensor，也无法从
hidden state 反推出 attention。** 因而必须从相同模型、tokenizer、chat template
和序列截断配置重新抽取 attention。生成阶段应保存 rendered prompt、原 response
文本、prompt/response token IDs 和模型 revision；复放时以保存的 token IDs 为真值，
分段重分词只用于证明 character/step offsets 对应同一 token 轴。若二者不一致就拒绝，
不会把另一套重分词结果静默当成原生成轨迹。

当保存了 prompt token IDs 时，抽取器会分别尝试是否添加 tokenizer special tokens，
并只接受与原 ID 完全相同的那一种；没有原 ID 的 observer 模式下，chat template
默认不重复添加 BOS/control tokens。生成尾部 EOS/pad 会从可见 response 轴剥离并单独
保存，不能混入 step pooling。若只抽取 layer/head 子集，
NPZ 会同时保存原模型 axis id 和总层/头数；构图不会把原 `layer=20, head=7` 静默重编号
成 `layer=0, head=0`。

推荐每文件一条 dense trace。adapter 也支持一个 `.pt` 中的 mapping 列表或
`[B,L,H,N,N]` dense batch，但不建议把大量不等长对象塞进单个 object-NPZ。
`.npz` object array 使用 pickle 读取，只应处理可信的本地研究数据。

严格的同权重、同 token 轴 eager teacher-forcing trace 要求每条输入含
`generator`/`generator_model`/`model`、
`rendered_prompt`、`response_text`、`prompt_token_ids` 和 `response_token_ids`。这会复放
相同 token 轴；严格模式还要求数据中的 generator commit 与回放模型 commit 相同。
远端模型给出 `--model_commit_hash` 时，model 与 tokenizer 都会从该 revision 加载，并与
加载后解析出的 commit 交叉核对。参数必须是至少 7 位的十六进制 commit，而不是会移动的
branch/tag。本地 checkpoint 的 CLI hash 或 `config._commit_hash` 只是一项声明，不能证明
目录中的权重字节确实来自该 commit，因此不能进入 verified replay：

```powershell
cd D:\projects\research\demo
python -m hypergraph.attention.extract `
  --input D:\path\to\processbench.json `
  --output_dir data\attention\processbench `
  --model <same-hugging-face-model-id> `
  --model_commit_hash <40-hex-generator-commit> `
  --replay_mode same_generator --dtype float16
```

普通 ProcessBench 若没有上述原生成字段，只能做明确标记的 counterfactual observer
teacher forcing，而且必须写入另一个目录、作为另一个实验报告：

```powershell
python -m hypergraph.attention.extract `
  --input D:\path\to\processbench.json `
  --output_dir data\attention\processbench_observer `
  --model D:\path\to\observer_model `
  --replay_mode observer --prompt_style plain --dtype float16
```

“模型名称相同”本身不再足以声称同权重/同 token trace：缺失任一原 prompt/response/token 字段时，
默认严格模式直接拒绝。observer trace 的训练还必须显式传
`--allow-observer-traces`，防止它与同生成器 trace 结果混报。

如果只有精确 token 轴、没有可核对的双方权重 revision，只能同时在抽取和训练阶段
显式传 `--allow-unverified-generator-weights`，并将其命名为 unverified-weight
diagnostic；它不能与 verified replay 合并，也不能支持“权重匹配”的结论。即使双方
commit 与 token 轴匹配，teacher forcing 也没有复现原采样 kernel、KV-cache 路径或生成
时 dtype；在 attention 阈值边稳定性审计前，只能称 weight/token-matched trace，不能称
完整生成机制复放。trace 会记录 `model_commit_source`；任何本地 model metadata 或 CLI hash
都只作为声明，抽取器强制归入 unverified diagnostic，并要求抽取/训练两阶段显式开关。
仅 tokenizer metadata 也不能证明模型权重 revision。

抽取器禁止截断后继续使用原 step 标签，并会在 dense attention-output 大小下界超过
`--max_attention_gib` 时停止；这不是 eager forward 峰值显存估计，低于阈值仍可能 OOM。
只有另外核实模型、activation 与临时张量显存预算后才使用
`--allow_large_attention`。`--activation_layer` 是可选创新，忠实 attention-only
基线不需要保存 hidden。hidden-node 模式默认令 `activation_layer=layer+1`，例如零基 block
14 对应 `hidden_states[15]`，即该 block 的输出。trace 默认以 float32 保存 attention；缓存兼容性门禁仍在
`0.01` 检查数值翻转以复用既有 exact-full-forward trace，而正式构图在 `tau=0.05`
重新选择成员。只有经过边稳定性审计后才使用
`--storage_dtype float16` 节省磁盘。

## 双 24GB GPU 加速抽取

加速组件位于 `scripts/extract_dual_gpu.sh`，包含四项不改变 attention-row 方法定义的优化：

- `AutoModel` 代替带未使用 LM head 的 `AutoModelForCausalLM`；
- 两个互斥、可合并的数据 shard，每张 GPU 一个进程；
- KV-cache query chunk，将模型返回 attention 的显存峰值从完整 (N\times N) 行块降为
  `chunk_size × N`；
- 每个适用真实样本前缀自动执行完整前向与 chunk 前向门禁（包括启用的 activation）；
  数值误差超限或在 `tau=0.01` 处出现任何超边成员翻转都会停止。

trace 的方法 fingerprint 还包含 attention 抽取/契约源码 SHA256、PyTorch/Transformers/CUDA
版本、GPU 名称/计算能力、eager attention 实现和实际 resolved device map。两张卡型号、源码或
软件栈不同而可能改变阈值附近数值时，两个 shard 会被训练入口拒绝合并；机器路径、archive
压缩和输入 shard 范围只进入运行/作用域 provenance，不会伪装成方法差异。

先在服务器项目根目录安装仅用于加速分发的依赖，并确认两张卡：

```bash
cd <服务器上的 demo 目录>
python -m pip install -r hypergraph/attention/requirements-accelerated.txt
python - <<'PY'
import torch, transformers, accelerate
print("torch", torch.__version__, "transformers", transformers.__version__)
print("cuda", torch.version.cuda, "gpus", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    print(i, torch.cuda.get_device_name(i), torch.cuda.get_device_properties(i).total_memory / 2**30)
PY
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
```

推荐先跑 4 条 observer pilot。`INPUT` 必须换成真实 JSON/JSONL；使用一个从未写入过的
`OUTPUT_ROOT`，因为抽取器会拒绝把新 trace 混入旧目录：

```bash
export INPUT=/absolute/path/to/processbench.json
export MODEL=/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct
export OUTPUT_ROOT="$PWD/outputs/attention_traces/llama31_8b_pilot4"
LIMIT=4 MODE=model_parallel QUERY_CHUNK_SIZE=0 \
  bash hypergraph/attention/scripts/extract_dual_gpu.sh
tail -n 50 "$OUTPUT_ROOT"/logs/balanced.log
```

pilot 通过后，用新的输出目录跑完整数据；不设置 `LIMIT`：

```bash
export OUTPUT_ROOT="$PWD/outputs/attention_traces/llama31_8b_full"
MODE=model_parallel QUERY_CHUNK_SIZE=0 \
  bash hypergraph/attention/scripts/extract_dual_gpu.sh
```

严格主流程默认使用一个进程把模型平衡切到两卡。它不提供样本级 2 倍并行，目的是让
完整序列 attention 前向有足够显存；下面展示如何显式调整两张卡和 CPU 的内存预算：

```bash
export OUTPUT_ROOT="$PWD/outputs/attention_traces/llama31_8b_balanced_pilot4"
LIMIT=4 MODE=model_parallel QUERY_CHUNK_SIZE=0 GPU_MEMORY=22GiB CPU_MEMORY=64GiB \
  bash hypergraph/attention/scripts/extract_dual_gpu.sh
```

`QUERY_CHUNK_SIZE>0` 的 cached-query 路径只保留作诊断实验，不属于严格主结果。真实
Llama-3.1-8B pilot 中，该路径相对完整前向出现 `max_abs=0.0488`，并在 `0.01` 阈值处
造成 1606 次超边成员翻转；因此不能通过放宽容差或关闭等价性门禁用于正式构图。

## 严格单层 response 检测：全流程入口

`scripts/run_single_layer_response_pipeline.sh` 串联单层 attention 抽取、互补 shard
审计、构图预检、一次固定的按题目隔离 train/validation/test，以及最终 held-out response
评估。它不再运行五折，也不再把 fold 均值称为最终测试指标。它固定以下任务语义：

- prompt 与 response token 都作为节点；
- 只有 response token 作为超边 receiver；
- 默认 `SOURCE_SCOPE=all_past`，因此 prompt 和更早的 response token 都可作为 source；
- 使用 `response_bce`，对 response token logits 做 mean pooling 后判断整条回答是否错误；
- 抽取和训练同时限定同一个零基 Transformer block，因此是真正的单层输入，而不只是筛选构图层。

主实验先选出由 `Llama-3.1-8B-Instruct` 生成的回答，再训练超图检测器：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
bash hypergraph/attention/scripts/run_single_layer_response_pipeline.sh \
  --layer 14 \
  --dataset omnimath \
  --generator-model Llama-3.1-8B-Instruct
```

若完整的 audited trace 已经存在，脚本直接从其中筛选目标 generator，不重复前向；若完整缓存
不存在，脚本先在 `outputs/attention_cohorts/` 物化带原始行号报告的 matched-generator JSON，
再只对这些样本前向。去掉 `--generator-model` 才是单独命名的 all-generator observer 辅助实验。
主入口会要求 generator tag 与 observer 目录名（仅允许明确的 `Meta-Llama-*`/`Llama-*` 别名）
一致，但本地模型目录没有可验证的权重内容摘要。因此该实验应称为 **generator-tag-matched
reconstructed observer**，不能称为 exact-weight same-generator replay。
这两种 observer teacher-forcing 都不能表述为恢复了原生成时的因果轨迹；ProcessBench 没有保存
精确 checkpoint revision、原始 rendered prompt 和原始 token IDs。

测试第 11 层只需改为 `--layer 11`。先跑 4 条 pilot 时增加 `--limit 4`；固定主结果只接受
一个模型 seed，例如 `--seed 17`。默认读取
`<demo>/data/hf_datasets/ProcessBench/<dataset>.json` 和
`/share/home/tm902089733300000/a903202310/lys/models/Meta-Llama-3.1-8B-Instruct`。
例如 OmniMath 的默认输入是
`/share/home/tm902089733300000/a903202310/lys/research/demo/data/hf_datasets/ProcessBench/omnimath.json`，
不是 `data/` 目录本身，也不是 `data/features/full_omnimath.npz`。用 `--input`、`--model`
可覆盖。所有可配置项及默认值通过下面的命令查看：

```bash
bash hypergraph/attention/scripts/run_single_layer_response_pipeline.sh --help
```

默认 no-cap attention-only 输出分别位于
`outputs/attention_traces/<dataset>_llama31_layer<layer>_nocap` 与
`outputs/attention_hypergraph/<dataset>_response_layer<layer>_matched_Llama-3.1-8B-Instruct_node_attention_nocap_fixed_original`；
all-generator observer 使用 `_observer_all` 后缀。若必须新抽 matched cohort，trace 目录也使用
对应 `_matched_...` 后缀。每个数据集直接写出：

- `fixed_seed17/results.json`：训练、验证和一次测试的完整结果；
- `aggregate_results.json`：根目录最终测试摘要，主字段为 `final_test`；
- `predictions_test.csv`：每条固定测试样本的 held-out 预测；
- `split_manifest.json`：三份 problem/trace ID、比例和分布审计。

四数据集入口最后生成逐数据集固定测试 AUROC/AUPRC 与 unweighted macro；不再读取
`fold0_seed17/predictions_test.csv`，也不会生成 `pooled_oof_test`。
已完成且通过 manifest 门禁的抽取和训练会被复用；不完整目录默认
fail closed，避免新旧配置静默混合。默认 symmetric propagation 与 per-graph z-score 对应
**整条回答生成完成后的离线 response 检测**，脚本会显式传入许可并写入结果配置；若要做
prefix-causal 版本，运行前设置
`PROPAGATION_MODE=receiver PREPROCESSING=none`。

把 block 输出 hidden state 加入节点内容时使用：

```bash
NODE_FEATURE_MODE=diagonal_plus_activation MAX_SEQ_LEN=0 \
bash hypergraph/attention/scripts/run_single_layer_response_pipeline.sh \
  --layer 14 --dataset omnimath \
  --generator-model Llama-3.1-8B-Instruct
```

此时节点为 `[32-D attention diagonal; hidden_states[15]]`；设
`NODE_FEATURE_MODE=activation_only` 则只使用 hidden。两种模式仍由 attention 构图，
`he_attr` 始终是 mean/max/flattened-head 三维。trace 与结果分别增加
`_hidden_hs15` 和 `_node_attention_hidden_hs15`（或 `_node_hidden_hs15`）后缀，因而不会与
attention-only 缓存混用。`MAX_SEQ_LEN=0` 还会写入独立 `_nocap` 目录；旧的 2048-cap
缓存不会被静默视为无上限缓存。

这里仍保留完整 dense trace，所以节省的是 GPU 峰值而不是最终的 (O(LHN^2)) CPU/磁盘量。
Llama-3.1-8B 的 32 层 × 32 头、2048 token、float32 trace 下界约为 16 GiB/样本；“完整数据”
命令只有在 pilot 的真实长度分布、主机 RAM 与 scratch 磁盘预算都通过后才能运行。
抽取时的 `--attention_layers`/`--attention_heads` 子集会同时裁剪超边候选和 node diagonal
通道/输入维度，是 **topology + node-feature + storage 联合变体**；不能称为仅拓扑消融。
训练时的 `--selected-layers`/`--selected-heads` 才只筛构图 rows、保留 trace 内全部 diagonal
通道。`STORAGE_DTYPE=float16` 也必须先检查 `tau=0.05` 周围的成员稳定性；这些设置均不能
静默冒充 all-layer/all-head float32 忠实基线。
前缀门禁只证明被检查的真实短前缀与指定阈值，没有证明所有长上下文、所有 threshold sweep
都与完整前向一致；manifest 因而记录为 `prefix_pass`，且 chunk size、软件与硬件环境进入方法
fingerprint。要求最严格的 A1-F 复现仍应另跑 `--query-chunk-size 0` 小规模基准，并将加速轨迹
作为单独 track 比较。

缓存分块门禁只绑定纯 `threshold` 且无训练期 `top-k` 的构图配置，并且训练阈值必须与抽取门禁
阈值完全一致。原对齐 wrapper 强制使用 exact full forward，因此可以执行
`threshold_fallback_topk`。`top_k_only`、`cumulative_mass`、threshold 后再 top-k 或其他 threshold sweep
会被训练入口拒绝；这些创新要么重新设计相应的成员等价门禁，要么用完整前向 trace。仅用于
故障定位时可显式传入 `--allow-unverified-chunk-topology`，但该结果必须单独命名且不能用于效果
结论。

两个 shard 可作为两个输入一起检查和训练。普通 ProcessBench 是 observer trace，训练时必须
显式允许并单独命名；超图网络本身很小，优先让不同 fold/seed 分别占卡，而不是使用 DDP：

```bash
python -m hypergraph.attention.train inspect \
  "$OUTPUT_ROOT/shard0" "$OUTPUT_ROOT/shard1" \
  --objective response_bce --allow-observer-traces \
  --generator-model Llama-3.1-8B-Instruct \
  --limit 20 --output outputs/attention_preflight.json

CUDA_VISIBLE_DEVICES=0 python -m hypergraph.attention.train train \
  "$OUTPUT_ROOT/shard0" "$OUTPUT_ROOT/shard1" \
  --objective step_bce --split-mode group_cv --folds 5 --fold-index 0 \
  --propagation-mode receiver --preprocessing none --allow-observer-traces \
  --output outputs/attention_step_fold0_seed17 &
CUDA_VISIBLE_DEVICES=1 python -m hypergraph.attention.train train \
  "$OUTPUT_ROOT/shard0" "$OUTPUT_ROOT/shard1" \
  --objective step_bce --split-mode group_cv --folds 5 --fold-index 1 \
  --propagation-mode receiver --preprocessing none --allow-observer-traces \
  --output outputs/attention_step_fold1_seed17 &
wait
```

脚本完成后会自动运行 `python -m hypergraph.attention.shards` 并写出
`$OUTPUT_ROOT/shard_audit.json`。`train` 会从 trace 内的签名元数据再次检查输入 SHA256、modulo
分片成员、重复原始行、缺失 shard/样本和失败行；不完整作用域默认 fail closed。
带 `--objective` 的 `inspect` 与 `train` 共享严格 cohort gate，会再次检查完整 extraction scope、
representation fingerprint、observer 许可、layer/head axis、generator/标签分布；不带 objective
的 `inspect` 和 `build` 只做结构检查。若未用双卡脚本，仍需先手动执行
`python -m hypergraph.attention.shards <trace-dir> ...`。只有诊断任务才使用
`--allow-incomplete-extraction-scope`，合并不同输入数据集则还需显式
`--allow-multiple-input-datasets`。该 preflight 只表示 supervised cohort/provenance/graph gate
通过，不承诺每个 fold 都有双类、训练一定完成、跨数据集兼容或长度混杂已经消除。
`--limit` 与多个 trace 根目录同时使用会因 shard 遍历顺序歧义而被拒绝；生产 wrapper 会先在
原始行顺序上应用 limit，再分 shard，因此训练端无需二次截断。

若数据确实保存了原生成器的 rendered prompt、response 和两段 token IDs，可改成
`REPLAY_MODE=same_generator`。但当前给出的模型是本地目录，本地 commit metadata 不能验证
权重内容；这种运行还必须在抽取和训练两端显式使用
`--allow-unverified-generator-weights`，并作为 unverified-weight diagnostic 报告。

先检查一小批样本：

```powershell
cd D:\projects\research\demo
python -m hypergraph.attention.train inspect D:\path\to\attention_traces `
  --objective response_bce --allow-observer-traces `
  --generator-model Llama-3.1-8B-Instruct `
  --limit 20 --output outputs\attention_preflight.json
```

`inspect` 同时执行数据对齐、causal-attention 检查、真实构图和 schema 校验，并报告
节点数、超边数、incidence 数、标签覆盖和 fallback group 数。默认拒绝未来 attention
质量超过 `1e-5` 的 decoder trace；只有明确分析非因果模型时才使用
`--no-require-causal`。

抽取器要求 `output_dir` 为空；它不会把新的 `--limit`、模型或模板结果混入旧 NPZ。
每条 trace 同时保存 model/template/dtype/layer-head、replay mode/fidelity、source generator、
prompt 来源和 prompt/response hash。训练把 representation compatibility 与 source provenance
分开：经状态机验证的 `observer_counterfactual` 模式允许来源 generator 不同，但 observer 模型、模板、抽取方法和实际 axis
必须一致；same-generator 模式仍严格绑定 generator identity/commit。输入文件 SHA256、原始行号
和可复算的 scope fingerprint 单独保存，因此互补 shard
可合并而数据范围仍可追溯。内容 hash 逐样本保留用于审计，但不要求彼此相同。旧文件缺少
完整 provenance 时默认拒绝，只有明确的 legacy 诊断才可传
`--allow-missing-provenance`。
复现实验时必须把 trace 目录、同级 `pipeline_request.json`、`shard_audit.json` 和（若存在）
matched cohort 的 `.report.json` 作为一个审计单元保存；只复制裸 `.npz` 会丢失 wrapper 级
数据选择与 producer-code 绑定。正式发布前还应为本地 checkpoint、tokenizer、config 建立
只读内容 manifest，因为目录路径和 nominal model tag 不能检测权重被替换。

## 三种任务不能混用标签

| 场景 | CLI objective | 必需标签 | 损失/读出 |
|---|---|---|---|
| RAGTruth 精确 span | `token_bce` | 真实 `token_y` | response token BCE |
| ProcessBench 首错定位 | `step_bce` | `step_ranges + gold_step` | token logits → step mean → first-error risk-set BCE |
| 整条 QA 是否幻觉 | `response_bce` | `response_y` | response token mean → response BCE |

`gold_step` **不会**被扩张成“一整个错误 step 的 token 全为 1”。这是必要的协议
修复，不是方法创新：step 标注没有提供首个错误 token，强行扩张会制造伪标签，且
step 前部的 causal hidden/attention 不可能预见后部才发生的错误。

对 first-error step，risk set 包含正确链全部 step，或错误链从开头到首错 step；
首错后的 step 属于错误后果，不被伪装成干净负例。step localization 额外报告严格的
`unique_top1` 和 tie-aware reciprocal rank。risk set 只用于训练和 step-level
AUROC/AUPR；定位排名始终在全部有效 step 上计算，绝不使用 gold step 裁掉后续高分。
此外同时报告全 trace 的 `max(step probability)` 检测 AUROC/AUPR；first-crossing
threshold 只在 validation 上选择，test 报告含正确链误报在内的 first-error exact
accuracy、response F1 和 false-alarm rate，并导出每个 step 的 probability。

## 构图与训练

只构图、不训练：

```powershell
python -m hypergraph.attention.train build D:\path\to\attention_traces `
  --output outputs\faithful_attention_graphs
```

输出是每条 trace 一个 framework-neutral graph NPZ，以及 `manifest.json`；图内保留
`trace_id/group_id/split`、exact token mask 和完整 construction config，避免构图后
无法恢复 problem-disjoint split 或阈值来源。已有目标文件默认不覆盖；确定要重建时
显式传入 `--overwrite`。

使用数据自带的 `split=train/validation/test`：

```powershell
python -m hypergraph.attention.train train D:\path\to\attention_traces `
  --objective token_bce --propagation-mode receiver --preprocessing none `
  --split-mode explicit `
  --output outputs\faithful_token_seed17
```

按题目分组的首错 step 训练：

```powershell
python -m hypergraph.attention.train train D:\path\to\attention_traces `
  --objective step_bce --propagation-mode receiver --preprocessing none `
  --split-mode group_cv --folds 5 --fold-index 0 `
  --output outputs\faithful_step_fold0
```

token/首错定位默认拒绝两类完整-response 泄漏：对称传播会把未来 receiver 的超边消息
回写给较早 source；逐图 z-score 又会用未来/错后 token 的均值方差变换早期特征。
risk-set mask 只能屏蔽 loss，不能消除 forward 泄漏。prefix-consistent 的事后定位必须同时使用
`--propagation-mode receiver --preprocessing none`。若只为了复现原版全响应离线比较，
必须显式加入 `--allow-offline-full-context`（旧别名
`--allow-offline-symmetric-step` 仍可用），并在结果中标记为 offline。

group CV 默认要求真实 `problem_id/question_id/group_id`，不允许把每条 trace 当成
独立 group。`--allow-trace-as-group` 只适合单样本题目的诊断，不应作为同题多采样
实验结论。分组分配会近似平衡 response 类别、首错相对位置和 response 长度；划分后仍
逐分区检查两类覆盖，并输出长度/step/首错位置分布。若使用官方显式 split，训练、验证、
测试之间仍会检查 group 是否泄漏。只要 trace 中存在任何官方 split metadata，显式
`group_cv` 也默认拒绝，避免把原 test 重分到 train；仅诊断性重分才可使用醒目的
`--allow-resplit-official-data`。

训练输出包括：

- `config.json`：loader、graph、model、objective 和 split 的完整配置；
- `checkpoint.pt`：最佳 validation epoch 的真实权重；
- `history.json`：每轮真实 loss/validation 指标；
- `results.json`：train/validation/test 指标；
- `predictions_*.csv`：held-out 审计所需的 trace/group 分数。

`config.json`/`results.json` 的 resolved model 信息还记录 total/trainable parameter count，
用于识别 feature 维度变化带来的容量差异。

当前模型只依赖 PyTorch，不要求 PyG。没有 PyTorch 时 `inspect` 与 `build` 仍可用，
`train` 会直接报出依赖缺失，不会生成假的实验结果。

## 当前框架内可改的内容

下面按“离原方法的距离”排列。每次只改一个维度，并始终保留忠实基线。

| 模块 | 忠实默认 | 可调/创新项 | 主要假设与风险 |
|---|---|---|---|
| attention 拓扑选择 | 原代码事实为 head 0；节点保留全部抽取 head | `--selected-layers`、`--selected-heads all` | all-head 是修正版消融；必须只用 validation 选配置，不能看 test 挑 head |
| 超边稀疏化 | `threshold_fallback_topk, tau=0.05, top_k=16, min_sources=2` | 纯 threshold；`top_k_only`；`cumulative_mass` | 后三者改变原拓扑，需独立命名 |
| source 范围 | `all_past` | `prompt_only`、`response_only` | 分离题设 grounding 与推理内部依赖 |
| incidence 权重 | `uniform` | `attention`=raw weighted-sum；`normalized_attention`=convex weighted-mean | 前者保留选中 mass、后者只保留相对强度；都可能被 attention sink 支配 |
| 超边属性 | 原版 mean/max/flattened-head 三维 | `--edge-attr-mode extended` | 加入分离 layer/head 与长度归一化成员数；必须作为显式创新 |
| 传播方向 | `symmetric` | `receiver` | receiver-only 更符合有向 query→key 语义并避免 source 被反向更新 |
| 节点特征 | `attention_diagonal` | `--node-feature-mode activation_only/diagonal_plus_activation` | hidden 提供状态内容，但必须严格 token 对齐；只改变内容特征，不替代 attention 拓扑；三种输入维度并不天然参数匹配 |
| 消息函数 | source 独立编码 | `--receiver-source-interaction` | 使用 `[source,receiver,diff,product]` 学习关系，而非只做平滑 |
| 预处理 | 离线忠实 preset 为 `per_graph_zscore` | `--preprocessing none` | prefix-consistent token/step 事后定位必须为 none；逐图统计本身含未来上下文 |
| 正则化 | residual + MLP LayerNorm | dropout、weight decay、`--no-mlp-norm` | 数据少时控制过拟合 |
| 网络深度 | 2 层 | `--model-layers 0/1/2/...` | `0` 是 feature-only/no-message 基线；层数过深会过平滑 |
| 关系算子 | hypergraph joint set | `--message-operator pairwise`（需 receiver + faithful attrs + preprocessing none） | 相同 incidence/参数量和原始属性尺度，先以单对 attention 属性逐 query-key 解码再聚合；用于检验是否真有高阶集合增益 |
| 粒度 | 标签匹配 objective | 三个独立 objective | 不允许 step→token 伪标签 |
| 读出 | 原 token logit 直接监督 | step/response mean；normalized `logmeanexp` | 后两者是任务适配；检查稀疏高风险 token 时仍需控制长度偏差 |

“超图看 token 间相关性”和“传播方式可调”并不矛盾：attention-row 决定哪些 token
属于同一超边，即图拓扑；uniform/attention weighting 和 symmetric/receiver 更新决定
这些相关性怎样被模型使用。这里没有另造一套 hidden 主拓扑。

固定阈值也有明确的数据偏差：上下文越长，attention mass 分摊给更多 key，超过
`tau` 的成员数可能随 receiver 位置系统变化。因此 threshold sweep 不能只报最终
AUROC，还必须报告 edge density/member count 对序列长度和相对位置的曲线。当前实现的
`--source-selection top_k_only --top-k K` 可保留每行最强 source，
`--source-selection cumulative_mass --cumulative-mass 0.8` 可保留候选历史 attention
质量的固定比例；二者不能替换 A1 的固定阈值忠实基线。extended
edge attributes 中的成员数也采用长度归一化形式，并保持显式 opt-in。

`top_k_only` 是固定度数的长度控制诊断：在 exact-zero/underflow ties 中仍会稳定选出 K
个 source；配合 uniform incidence 时，这些零 attention source 也会成为等权成员。因此
它不能被解释为 attention-supported 语义拓扑。需要这一语义时应使用 threshold(+top-k)
或另行实现并命名 positive-only top-k。

当前 `selected-layers/heads` 只筛选构图用的 attention rows；忠实节点特征仍包含所有
layer/head 的 self-attention diagonal。若也裁剪节点特征，那是另一项独立消融，不能与
拓扑筛选一起静默发生。

固定阈值还会引入位置/长度混杂：attention row 归一化后，候选历史 token 越多，单个
权重通常越小，`A>tau` 的边密度会随 receiver 位置变化；步骤越长，mean/max pooling
和可参与传播的超边数也会变化。因此实验必须同时报告超边覆盖率/度数随 token 位置的
曲线和按 step 长度分层指标。`top-k` 或累计 attention-mass 稀疏化值得测试，但它们是
控制这一混杂的创新项，不是原版复现。
当前代码记录 response token 数、step 数、首错相对位置和分区分布，但这些记录本身不能消除
混杂。正式主实验前仍须先冻结 length-only baseline、仅用训练集定义长度 bins，并报告每类内
score--length 关系；否则超图分数可能只是回答长度或步骤长度的代理。

## 推荐实验矩阵

每个可比 track 内使用完全相同的数据、problem-disjoint split、seed、objective 与 early
stopping；F/C/R 不跨 objective 直接比较分数。

| 编号 | 图/模型 | 目的 |
|---|---|---|
| A0-F | A1-F + `--model-layers 0` | 原 exact-token 协议下的节点特征 MLP |
| A1-F | `token_bce` + 默认 symmetric/zscore + `--allow-offline-full-context` | 原版 exact-token、完整 response 离线 HyperCHARM |
| A1-R | `response_bce` 使用默认 symmetric/zscore | 整条 QA 检测的场景适配，不冒充原 token 训练目标 |
| A0-C | A1-C + `--model-layers 0` | prefix-consistent 事后节点特征对照；判断图传播是否真正增量 |
| A1-C | `--propagation-mode receiver --preprocessing none` | token/step 任务的无未来回流、same-index post-hoc attention-row 基线 |
| A2 | 在同一 F 或 C 基线上改 `--incidence-weight-mode attention/normalized_attention` | 测试 attention 强度是否比二值成员关系有用 |
| A3 | A1-C + `--receiver-source-interaction` | 测试 source-receiver 配对而非均值平滑 |
| A4 | 在匹配基线上加 `--use-activation` | attention topology + activation node feature；同时记录容量变化 |
| A5 | A1-C/A3 的 `prompt_only`、`response_only` | 区分 grounding 边与 reasoning-history 边 |
| A6 | fixed threshold vs `top_k_only` vs `cumulative_mass` | 检验收益是否只是修正长度导致的零边/密度漂移 |
| A7-P | A1-C 仅增加 `--message-operator pairwise` | 同参数、同 incidence、同原始尺度的 pairwise query-key 对照；检验超边集合是否优于普通有向 attention 图 |

节点内容还要做正交对照：`attention_diagonal`、`activation_only`、
`diagonal_plus_activation` 都分别配合 `--model-layers 0` 和真实消息传递。若 attention
超图没有超过 feature-only 与 pairwise 对照，就不能把结果归因于“超图高阶关系”。另需
加入只看 response/step 长度和相对位置的轻量分类器；若它已达到相近分数，说明主模型
很可能在学习长度 hazard，而不是幻觉机制。

三种 node feature 的原始维度不同，因此当前训练结果会记录 total/trainable parameter
count，但这仍不是严格的容量匹配。若 hidden/combined 更好，只能先解释为“输入内容与
容量的联合差异”；要声称 hidden 内容本身有效，应再加入相同维度的冻结随机投影或
fold 内拟合的 bottleneck，并保持分类器与消息层参数量一致。

不要一开始把 A2–A6 全部叠加，否则即使分数变化也无法解释来自哪里。至少同时报告：

- response AUROC/AUPR，或 step risk-set AUROC/AUPR；
- 首错 step 的 unique top-1 与 MRR；
- 按 step/response token 长度分层的结果；
- feature-only 相对匹配的 A1-F/A1-C 的 held-out 增量；
- 多 seed 与 problem-level bootstrap CI。

效果 claim 使用 paired problem-bootstrap：相对 no-graph、rewired 和 pairwise 的 95% CI
下界都必须超过实验前登记的 practical margin（最低要求为 0）；单个 seed 或正的点估计
不构成“图有效/超图有效”。

默认 response 入口已实现单个固定、problem-disjoint 的 train/validation/test 划分和单 seed
训练；CPU/PyTorch 路径及模拟 cache 路径已通过作用域测试，真实 Llama checkpoint 的显存和
吞吐仍须由服务器运行验证。通用 `train.py` 继续保留 group-CV 作为稳健性诊断，但它不属于
原对齐主入口，也不能与固定测试结果混合汇总。正式结果应在冻结测试集上补充
problem-level bootstrap；尚未完成的真实数据实验不构成效果提升声明。

## JSON 配置

为了固定消融，可传入 JSON。键可以平铺，也可以放在 `loader`、`graph`、`model`、
`training`、`data` section；显式 CLI 参数优先。例如：

```json
{
  "graph": {
    "threshold": 0.05,
    "top_k": 16,
    "source_selection": "threshold_fallback_topk",
    "min_sources": 2,
    "source_scope": "all_past",
    "incidence_weight_mode": "uniform",
    "propagation_mode": "receiver"
  },
  "model": {
    "hidden_dim": 128,
    "model_layers": 2,
    "receiver_source_interaction": false
  },
  "training": {
    "pooling": "mean",
    "seed": 17
  },
  "data": {
    "preprocessing": "none"
  }
}
```

```powershell
python -m hypergraph.attention.train train D:\path\to\attention_traces `
  --config configs\faithful.json --objective step_bce `
  --output outputs\faithful_step_fold0
```

`inputs`、`objective`、`output`、`command` 和 `config` 禁止由 JSON 改写，必须显式写在
命令行，避免旧配置悄悄换数据集或任务。JSON 值按对应 CLI 的精确类型与 choices 校验；
例如整数不能写成 `1.5`，布尔值不能写成字符串 `"false"`，重复键也会拒绝。
