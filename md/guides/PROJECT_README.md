# Layer–Time Representation Geometry

当前项目主线只研究一个对象：推理表征在“网络深度 × 推理时间”二维网格上的局部几何演化。

我们不再把 prompt anchor、uncertainty、hazard、intervention 或若干 detector 拼成主方法。旧的 `representation_geometry.py`、`spectral_chain_dynamics.py`、VAE、HMM、Gram 和 AnchorFlow 仍保留为历史基线，但不进入当前方法定义。

## 唯一主对象

对链 $i$、推理步 $t$、hidden-state 深度 $\ell$，保存而不是压平：

\[
z_{i,t,\ell}
=
\frac{1}{|S_{i,t}|}
\sum_{p\in S_{i,t}}h_{i,p}^{(\ell)}
\in\mathbb R^D,
\qquad
Z_i\in\mathbb R^{T_i\times L\times D}.
\]

每个 held-out 状态只与训练 fold 中、相同归一化推理进度的完整参考链比较。局部 kNN 同时给出 LID、邻域谱系和局部切空间；沿 layer 与 time 两条轴拟合局部 connection，最后在最小 layer–time 方格上比较两条路径：

\[
P_{\ell t}=R^t_{\ell+1,t}R^\ell_{\ell,t},
\qquad
P_{t\ell}=R^\ell_{\ell,t+1}R^t_{\ell,t},
\]

\[
\kappa_{i,\ell,t}
=
\frac{\|P_{\ell t}-P_{t\ell}\|_F}{2\sqrt q}.
\]

这里 LID 决定局部 fiber rank，邻域身份定义 paired local chart，holonomy $\kappa$ 是唯一核心交互量；它们不是彼此无关的特征栈。完整定义见 [METHOD_LAYER_TIME_GEOMETRY.md](../../prompt_control_flow/METHOD_LAYER_TIME_GEOMETRY.md)。

## 两阶段主流程

### A. 从文本链重新抽取全层 mean state

`--geometry_only` 会关闭 prompt-flow 与 uncertainty，只保存 `[step, layer, hidden]`。`--layers all` 的含义是所有 transformer block 的 post-block hidden states；exact artifact 若已含 prompt IDs、response IDs、offsets 与 step ranges，会直接回放，绝不重新 tokenize。

exact token IDs 默认还会核对 artifact 的 source model/tokenizer 名称；不匹配会停止。`--allow_model_mismatch` 只允许用于显式的不安全 ablation。

```bash
python -m prompt_control_flow.cli.extract_mechanisms \
  --input data/gsm8k_exact_multisample.npz \
  --model /path/to/Llama-3.1-8B-Instruct \
  --output outputs/layer_time/gsm8k_whole_layer_states.npz \
  --geometry_only \
  --min_success_fraction 0.99 \
  --max_seq_len 4096 \
  --device cuda \
  --dtype bfloat16
```

该命令边抽取边写 `gsm8k_whole_layer_states.states.<run-id>.npy`（fp16 memmap）；轻量 NPZ 保存相对路径、chain/step 索引、层号和 provenance。两个文件必须一起移动。state 使用 run-unique 文件名，manifest 最后原子提交，避免中断时让旧 manifest 指向半个新 tensor；确认新 run 后可清理不再被 manifest 引用的旧 state 文件。

### B. 构造 problem-grouped OOF layer–time field

```bash
python -m prompt_control_flow.cli.audit_layer_time_geometry \
  --input outputs/layer_time/gsm8k_whole_layer_states.npz \
  --output outputs/layer_time/gsm8k_ltg.npz \
  --output_dir outputs/layer_time/gsm8k_ltg_audit \
  --folds 5 \
  --knn_k 20 \
  --tangent_k 24 \
  --tangent_rank 6 \
  --projection_dim 64 \
  --max_reference 256 \
  --phase_grid_size 11
```

第二阶段只依赖 NumPy，可在 CPU 上运行。reference、归一化和局部 transport 只使用训练 problem；JL 由固定 seed data-independently 生成，并跨层、跨 outer fold 共享。错误标签仅在输出后的验证阶段使用。

## 直接读取 exact `sv_vec_mean`

`10_sample_and_extract.py` 与 `01_extract_spectral_field.py` 的全层 raw mean vectors 可直接进入第二阶段，但正式主线必须满足：

```bash
--layers all --no_reasoning_subspace --sv_modes mean --store_vectors
```

这两个 writer 的 depth 0 是 embedding output；adapter 会显式删除 depth 0，使其与 `geometry_only` 的 post-block depths 1..N 对齐，并记录 `layer_time_embedding_depth_dropped`。

`sv_vec_mean` adapter 只构造 run-unique 的临时 layered fp16 memmap，不再经过旧 spectral canonicalizer 的 $LD$ flattened 副本；field artifact 写完后临时文件会被删除，输入目录不会被覆盖。

示例：

```bash
python 10_sample_and_extract.py \
  --model /path/to/Llama-3.1-8B-Instruct \
  --dataset_format processbench \
  --dataset /path/to/ProcessBench \
  --subset gsm8k \
  --n_problems 64 \
  --k_samples 8 \
  --layers all \
  --no_reasoning_subspace \
  --sv_modes mean \
  --store_vectors \
  --output data/gsm8k_ltg_smoke.npz

python -m prompt_control_flow.cli.audit_layer_time_geometry \
  --input data/gsm8k_ltg_smoke.npz \
  --output outputs/layer_time/gsm8k_ltg_smoke.npz
```

已有 canonical `full_*.npz` 只有稀疏层且通常是 `step_exp` pooling，默认会被连续层与 mean-pooling 两道 guard 拒绝。只有做历史 ablation 时才同时显式添加：

```text
--allow_sparse_layers --allow_legacy_pooling
```

该结果不得称为 whole-layer 主结果。

## 输出 schema

主输出保留 `float32` 完整场：

```text
layer_time_geometry_field
  shape = [chain, max_step, layer, observable]

layer_time_geometry_field_names =
  lid
  depth_neighbor_rewire
  time_neighbor_rewire
  depth_tangent_drift
  time_tangent_drift
  plaquette_holonomy
  rank_singularity
```

同时记录：

```text
layer_time_geometry_layers
layer_time_geometry_fold
layer_time_geometry_reference_sizes
layer_time_geometry_lid_coverage
layer_time_geometry_connection_coverage
layer_time_geometry_holonomy_coverage
layer_time_geometry_reference_policy
layer_time_geometry_pooling_kind
layer_time_geometry_representation_kind
```

`ltg_*` step/chain reductions只用于预注册验证和兼容现有 evaluator，不是方法对象，也不做任意加权总分。

第二阶段默认不把数 GB 的 raw state tensor 再复制进 field artifact；输出保存 source path/size 与采样 provenance。只有显式 `--keep_state_vectors` 才会嵌入原 tensor。

## 必须通过的门

1. exact token replay、offset 与 step-range 必须处在同一 token 轴；一处错位即停止。
2. `state_representation_kind == hidden_state`，不能偷偷使用 reasoning-subspace projection。
3. `state_pooling_kind == arithmetic_mean_over_step_tokens`。
4. 正式主结果要求层号连续；稀疏层只作 pilot。
5. 同一 `problem_id` 的全部 sample 必须在同一 fold。
6. OOF LID/connection/holonomy coverage 默认至少为 0.99/0.95/0.90；任何 fold 参考链少于 k 会硬失败。
7. identical-layer 不变量要求 depth rewiring (=0)，tangent/holonomy 只允许数值误差。
8. 结论必须通过 $k$、JL 维度、tangent-rank、reference cap 与 phase-grid sensitivity；不能在同一测试集挑最佳层或最佳超参。

## 资源估算

全层 hidden tensor 的未压缩 fp16 大小约为：

\[
2\times N_{\text{step}}\times L\times D\ \text{bytes}.
\]

例如 $24{,}000$ 个 step、$L=32$、$D=4096$ 时约 $6.3$ GB。`geometry_only` 直接写 fp16 memmap，不保留全量 float32 list，也不保存 flattened 副本。审计阶段以零拷贝方式打开 memmap，仅对 fold/query 分块转 float32 并共享 JL 到 64 维；reference cap 按训练链而不是训练 step 计数。

## 本地验证

```bash
python -m pip install -r requirements-dev.txt
pytest -q tests/test_trace_alignment.py \
          tests/test_teacher_forcing_trace.py \
          tests/test_layer_time_geometry.py
python -m prompt_control_flow.cli.extract_mechanisms --help
python -m prompt_control_flow.cli.audit_layer_time_geometry --help
```

当前 CPU 合成链、schema、group split、label invariance、identical-layer 与 gauge-invariance 测试已连通；真实 7B/8B 全层 GPU smoke 仍必须在远端完成后，才能声称真实数据流程通过。

## 研究产物

- [方法重设计报告](../../research-reports/IDEA_DISCOVERY_LATEST.md)
- [claim-driven 实验计划](../../refine-logs/EXPERIMENT_PLAN_LATEST.md)
- [紧凑实验跟踪器](../../refine-logs/EXPERIMENT_TRACKER.md)
- [《The Phenomenology of Hallucinations》详解](../../research-reports/PHENOMENOLOGY_LATEST.md)
- [artifact manifest](../../MANIFEST.md)

## 配套 skills

已安装到项目根目录：

```text
D:\projects\research\.agents\skills              # 80 个 SKILL.md
D:\projects\research\.agents\skills\shared-references
D:\projects\research\.aris\tools
D:\projects\research\.aris\installed-skills-codex.txt
```
