# Reliability-Calibrated Layer-Time Connection Geometry

> **Status (2026-07-13): compatibility/negative baseline, not the mainline
> method.** The layer-depth and reasoning-time edges are not established as
> commuting perturbations of one base manifold, and the reliable Wilson score
> had insufficient coverage. See
> [METHOD_REASONING_FLOW_SIGNATURES.md](METHOD_REASONING_FLOW_SIGNATURES.md)
> for the replacement hypothesis and implementation.

## 1. Research question

我们只问一个问题：正确与错误推理在完整网络深度和推理时间上的局部表征几何，是否具有不同的二维 interaction curvature？

方法不是多个 detector 的拼接。唯一对象是 held-out 推理链在 layer-time 网格上的局部 tangent bundle：kNN 给出局部 chart 与参考身份，局部谱估计给出 rank front，固定秩 connection 描述 chart 之间的输运，reliability-calibrated Wilson loop 描述二维路径依赖。

V2 的关键修正是：**LID 不再决定主 holonomy 的 fiber rank**。局部维度变化和 connection curvature 是两个必须分离检验的性质。正式主结果在预注册固定秩 (q\in\{2,4,6\}) 上计算；LID-adaptive rank 仅作为历史消融。

## 2. State field

对推理链 (i) 的第 (t) 个文本步骤，在 hidden-state 深度 (ell) 上对该步骤的 exact token span (S_{i,t}) 做算术平均：

\[
z_{i,t,\ell}
=
\frac{1}{|S_{i,t}|}
\sum_{p\in S_{i,t}}h^{(\ell)}_{i,p}
\in\mathbb R^D.
\]

磁盘主 schema 保留：

\[
Z_i\in\mathbb R^{T_i\times L\times D},
\]

而不是先变成 (T_i\times(LD))。正式主线只接受 raw hidden state、算术 mean pooling、exact token span 和连续层深。

## 3. Leakage-safe reference grid

外层 fold 按 `problem_id` 切分，同一题的所有 sampled responses 不可跨 fold。几何构造不使用 correctness 或 first-error label。

变长链首先以归一化进度表示：

\[
\rho_{i,t}=\frac{t}{T_i-1}.
\]

对 held-out 查询 ((i,t))，每条训练参考链 (j) 在线性插值后只贡献同一进度 (ho_{i,t}) 的一个状态：

\[
\widetilde z_{j,\rho,\ell}
=
(1-\alpha)z_{j,a,\ell}+\alpha z_{j,a+1,\ell}.
\]

所以 cell point cloud 是：

\[
X_{\rho,\ell}
=
\{\widetilde z_{j,\rho,\ell}:j\in\mathcal R_{\mathrm{train}}\}.
\]

邻居 ID 永远是同一批训练链 ID。跨层或跨时间比较邻域时，参考宇宙不会改变。

训练 fold 内先做每层中心化和单一 RMS scale。高维状态使用跨 outer folds 与所有层共享的 data-independent Johnson-Lindenstrauss 投影：

\[
x_{i,t,\ell}
=
\frac{z_{i,t,\ell}-\mu_\ell}{s_\ell}J.
\]

正式结果必须比较 64/128 维、多 seed 与 exact ambient distance。线性 phase interpolation 仍是待检验假设；nearest observed phase、token progress 和 arclength alignment 必须作为稳健性对照。

## 4. Local chart, LID, and neighborhood lineage

在 phase-matched train cloud 中定义查询点的 (k) 近邻：

\[
\mathcal N_k(i,t,\ell)
=
\operatorname{kNN}\left(x_{i,t,\ell},X_{\rho_{i,t},\ell}\right).
\]

设有序半径为 (r_1\le\cdots\le r_k)，局部内在维度采用 kNN MLE：

\[
\widehat d_{i,t,\ell}
=
\left[
\frac{1}{k-1}
\sum_{m=1}^{k-1}
\log\frac{r_k}{r_m}
\right]^{-1}.
\]

邻域沿 depth/time 两个轴的重连率为：

\[
\Omega^\ell_{i,t,\ell}
=
1-
\frac{|\mathcal N_k(i,t,\ell-1)\cap\mathcal N_k(i,t,\ell)|}{k},
\]

\[
\Omega^t_{i,t,\ell}
=
1-
\frac{|\mathcal N_k(i,t-1,\ell)\cap\mathcal N_k(i,t,\ell)|}{k}.
\]

这些量描述同一个 reference-chain neighborhood 的 lineage，不使用错误标签。

## 5. Rank front and fixed-rank tangent fiber

对近邻云局部中心化并做 SVD，得到切基 (B_{i,t,\ell})。局部维度场独立定义为：

\[
d^{\mathrm{front}}_{i,t,\ell}
=
\min\left(
\operatorname{round}(\widehat d_{i,t,\ell}),
\operatorname{rank}_{\mathrm{num}}
\right).
\]

它只用于报告 `fiber_rank` 与 `rank_singularity`。主 connection 使用固定秩：

\[
q_{\mathrm{conn}}=q_0,
\qquad q_0\in\{2,4,6\}.
\]

只有当相邻 cell 的数值秩均不小于 (q_0) 时才计算该 edge。这样 rank change 不会被截断到共同最小维数后变成曲率。

相邻 cell 使用相同参考链身份的局部坐标。令共同参考 ID 集为 (mathcal U)：

\[
C_s=(X_s[\mathcal U]-\bar X_s)B_s,
\qquad
C_t=(X_t[\mathcal U]-\bar X_t)B_t.
\]

局部正交 connection 由 paired Procrustes 得到：

\[
R_{s\rightarrow t}
=
\arg\min_{R^\top R=I}
\left\|
\frac{C_s}{\|C_s\|_F}R-
\frac{C_t}{\|C_t\|_F}
\right\|_F.
\]

归一化 Procrustes residual (arepsilon_e) 是 connection 的测量误差估计。它不与其他几何量加权成任意总分，但决定 plaquette 是否进入 reliability-gated 主分析。

## 6. Reliability-calibrated Wilson curvature

对方格左上角 ((t-1,\ell-1)) 到右下角 ((t,\ell))，两条路径为：

\[
P_{\ell t}=R^t_{i,t,\ell}R^\ell_{i,t-1,\ell},
\qquad
P_{t\ell}=R^\ell_{i,t,\ell}R^t_{i,t,\ell-1}.
\]

旧版兼容量是外在 Frobenius 路径差：

\[
\kappa^F_{i,t,\ell}
=
\frac{\|P_{\ell t}-P_{t\ell}\|_F}{2\sqrt q}.
\]

V2 主量使用相对 Wilson loop：

\[
W_{i,t,\ell}=P_{\ell t}^{\top}P_{t\ell}.
\]

若其特征值写为 (e^{\mathrm{i}\theta_r})，定义群上的内禀曲率：

\[
\kappa^W_{i,t,\ell}
=
\frac{1}{\pi}
\sqrt{\frac{1}{q}\sum_{r=1}^{q}\theta_r^2}
\in[0,1].
\]

任意 cell 的正交 gauge 变化只会让 (W) 发生共轭变换，因此特征角保持不变。四条 edge residual 的最大值定义为：

\[
\varepsilon_{i,t,\ell}
=
\max_{e\in\partial p}\varepsilon_e.
\]

主分析使用预注册阈值 (arepsilon_{i,t,\ell}\le\tau) 的 `plaquette_reliable_wilson`。原始 Wilson curvature、transport residual 和 reliable coverage 必须同时汇报。若 curvature 与 residual 高度相关，则不能解释为 layer-time interaction。

## 7. Primary hypothesis and claim boundary

主假设只有一个：

> 错误出现附近更容易形成局部 layer-time connection curvature 集中带；该效应不能由位置/长度、LID/rank front、单轴 neighborhood rewiring 或 connection 拟合噪声解释。

主证据不是一个更高 AUROC，而是以下预注册结构同时成立：

1. reliability-gated Wilson curvature 在 first-error 对齐后形成局部峰，而不是只随相对位置单调增加；
2. 在控制 LID、rank、rewiring、transport residual、位置与长度后，该局部事件仍存在；
3. 同题 correct/wrong 响应以及同前缀最小错误对照产生同方向差异；
4. `phase_shuffle` 与 `reference_id_shuffle` 等结构负控破坏该局部带；
5. 结果对 (k)、JL 维数、reference cap、phase alignment、固定 (q) 与 reliability threshold 稳定；
6. 跨任务与跨模型的 effect direction 一致。

当前实现支持 label-free field construction 与 out-of-fold diagnosis，不支持因果主张。time connection 使用当前步与前一步，因此 cell ((t,\ell)) 只在步骤 (t) 完成后可得；没有读取 (t+1)。

## 8. Evaluation protocol

`layer_time_evaluate.py` 提供以下专用验证，不再只依赖 pooled AUROC：

- first-error aligned event curve；
- problem fixed effects + `rel_pos` + `log1p(step_len)` residualization；
- 同题 wrong-correct paired response difference 与 pair-micro AUROC；
- Wilson curvature 与 transport residual 的 Spearman correlation；
- conditioning on LID、fiber rank、rewiring、transport residual 后的 Wilson event；
- 完整 per-layer event map，避免 scalar reduction 隐藏 layer band。

`claim_gate` 只是描述性 go/no-go 前置条件，不是因果或论文接收判决。

## 9. Implementation map

- `metrics.py`: `[step, layer, hidden]` mean-state extraction。
- `teacher_forcing.py`: exact artifact token replay 与 fallback token-axis guard。
- `data.py`: exact prompt/token/range 与 kept-step schema。
- `extraction.py`: geometry-only、全 post-block 深度、非 flattened 主存储。
- `layer_time_geometry.py`: OOF reference、rank front、fixed-rank connection、transport reliability 与 Wilson curvature。
- `layer_time_evaluate.py`: event study、nuisance residualization、same-problem pairing 与 claim gate。
- bulk JL projection 与 kNN 支持 Torch/CUDA；变长小型 connection SVD 保留 CPU 路径。
- `cli/audit_layer_time_geometry.py`: 构建 field 并自动运行专用验证。
- `cli/evaluate_layer_time_geometry.py`: 对已有 field 重跑统计，不重新抽 hidden state。
- `tests/test_layer_time_geometry.py`: group、label-free、identical-layer、gauge 与 reference-ID null。
- `tests/test_layer_time_evaluate.py`: local event、同题配对与位置残差化。

## 10. Current limitations

1. normalized-step phase 可能不等价于认知进度，必须与 token progress、nearest observed phase 和 arclength alignment 比较。
2. mean pooling 忽略 step 内 token cloud；当前版本只是同一几何对象的 step-resolution 版本。
3. LID MLE 在参考链少、距离集中或重复点时不稳定，因此它只定义 rank front，不定义主 connection rank。
4. reliability threshold 必须在开发集冻结并报告 sweep，不能在测试集选择最佳阈值。
5. Wilson curvature 是 population-referenced interaction geometry，不是 Transformer 内部真实执行的两条计算路径。
6. representation patching 之前不能声称 curvature 导致错误。
7. 当前仅完成 CPU synthetic/schema 验证；真实全层 GPU smoke 与科学主张均待验证。
