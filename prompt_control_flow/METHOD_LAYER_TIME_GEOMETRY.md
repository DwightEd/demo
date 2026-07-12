# Phase-Matched Layer–Time Tangent-Bundle Geometry

## 1. Research question

我们只问一个问题：正确与错误推理在完整网络深度和推理时间上的局部表征几何，是否具有不同的演化与二维路径相容性？

方法不是多个 detector 的拼接。唯一对象是 held-out 推理链在 layer–time 网格上的局部 tangent bundle；LID 决定 fiber rank，kNN 给出局部 chart 与参考身份，connection 描述 chart 之间的输运，holonomy 描述二维路径依赖。

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

而不是先变成 (T_i\times (LD))。正式主线只接受 raw hidden state、算术 mean pooling 和连续层深。

## 3. Leakage-safe reference grid

外层 fold 按 `problem_id` 切分；同一题的所有采样响应不可跨 fold。几何构造不使用 correctness 或 first-error label。

变长链以归一化进度表示：

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

训练 fold 内先做每层中心化和单一 RMS scale；单一 scale 保留层内角度与相对距离。高维状态使用由全局 seed 生成、跨所有 outer folds 与所有层共享的 data-independent Johnson–Lindenstrauss 投影 $J\in\mathbb R^{D\times d'}$：

\[
x_{i,t,\ell}
=
\frac{z_{i,t,\ell}-\mu_\ell}{s_\ell}J.
\]

共享 $J$ 是必要不变量；每层或每 fold 独立随机投影会制造不可比较的邻域重连。artifact 保存 projection digest，主结果仍必须与 64/128 维、多 seed 及 exact ambient 距离比较。

## 4. Local chart, LID, and topology

在 phase-matched train cloud 中定义查询点的 (k) 近邻：

\[
\mathcal N_k(i,t,\ell)
=
\operatorname{kNN}
\left(x_{i,t,\ell},X_{\rho_{i,t},\ell}\right).
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

邻域拓扑沿两个轴的重连率为：

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

## 5. LID-adaptive local tangent fiber

对近邻云局部中心化并做 SVD，得到最多 (q_{\max}) 维切基 (B_{i,t,\ell})。fiber rank 由 LID 决定：

\[
q_{i,t,\ell}
=
\min\left(
q_{\max},
\operatorname{round}(\widehat d_{i,t,\ell}),
\operatorname{rank}_{\mathrm{num}}
\right).
\]

相邻 cell 使用共同参考链身份的局部坐标。以 layer edge 为例，令共同/并集近邻 ID 为 (mathcal U)：

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

同一定义分别用于 depth connection (R^\ell) 和 time connection (R^t)。归一化残差只报告 transport confidence，不与 LID 或 topology 加权成总分。

## 6. Plaquette holonomy

对方格左上角 ((t-1,\ell-1)) 到右下角 ((t,\ell))，有两条路径：

\[
P_{\ell t}
=
R^t_{i,t,\ell}
R^\ell_{i,t-1,\ell},
\]

\[
P_{t\ell}
=
R^\ell_{i,t,\ell}
R^t_{i,t,\ell-1}.
\]

四角使用共同 rank (q=\min q_{\mathrm{corner}})。核心量是：

\[
\kappa_{i,t,\ell}
=
\frac{\|P_{\ell t}-P_{t\ell}\|_F}{2\sqrt q}
\in[0,1].
\]

它衡量“先经过一层网络再继续一步推理”和“先继续一步推理再经过一层网络”的局部 tangent transport 是否相容。任意 cell 的切基旋转只会在路径两端做共同正交变换，因此 Frobenius 路径差保持不变。

若四角 (q) 不同，同时输出：

\[
S_{i,t,\ell}
=
\mathbf 1\{|\{q_{\mathrm{corner}}\}|>1\},
\]

将 rank change 明确记录为 singularity，而不是强行假设固定维度。

## 7. Primary hypothesis and claim boundary

主假设只有一个：

> 稳定推理在 layer–time 局部 tangent bundle 上更接近可积；错误出现附近更容易出现 LID/rank front、reference-neighborhood bifurcation，以及局部 holonomy curvature 集中带。

主证据不是一个更高 AUROC，而是以下预注册结构同时成立：

1. (kappa) 在 first-error 对齐后形成局部峰，而不是只随相对位置单调增加；
2. 该峰在同题 OOF、跨任务和跨模型上方向一致；
3. identical-layer、time-shuffle、reference-ID shuffle、shared-vs-independent projection 等负控满足预期；
4. 结果对 (k)、JL 维数、reference cap、phase grid 和 (q_{\max}) 稳定；
5. LID、rewiring 与 holonomy 的完整二维图能解释峰来自 rank singularity、邻域分叉还是两轴非交换，而不是 post-hoc 加权。

当前实现支持 label-free 描述与 out-of-fold 诊断，不支持因果或在线预警主张。time connection 使用当前步与前一步，因此 cell ((t,\ell)) 只在步骤 (t) 完成后可得；没有读取 (t+1)。

## 8. Implementation map

- `metrics.py`: `[step, layer, hidden]` mean-state extraction。
- `teacher_forcing.py`: exact artifact 直接 token replay；fallback 统一 no-special-token 轴。
- `data.py`: exact prompt/token/range 与 kept-step schema 读取。
- `extraction.py`: geometry-only、全 post-block 深度、非 flattened 主存储。
- `layer_time_geometry.py`: phase-matched OOF reference、LID、neighborhood lineage、local connection、rank singularity 与 holonomy。
- `cli/audit_layer_time_geometry.py`: 主审计入口。
- `tests/test_layer_time_geometry.py`: group、label-free、identical-layer 与 gauge invariance。
- `tests/test_teacher_forcing_trace.py`: BOS 轴错位与 exact replay 回归。

## 9. Current limitations

1. phase 使用 normalized step progress；必须再与 token progress/arclength 对齐做敏感性分析。
2. mean pooling 忽略 step 内 token cloud；token-cloud 版本属于后续同一几何对象的更细分辨率扩展，不与当前版本混合。
3. LID MLE 在参考链少、距离集中或重复点时不稳定；正式实验必须 bootstrap 和 (k)-sweep。
4. holonomy 是描述性几何证据；在做 representation patching 之前不能声称因果。
5. 目前仅完成 CPU synthetic/schema 验证，真实全层 GPU smoke 尚待执行。
