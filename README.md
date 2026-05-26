# Demo: LLM 推理轨迹流形分析（CIM-Style）

## 目标

在 ProcessBench 的少量样本上，按 **CIM (Constrained Inference Manifolds, arXiv:2605.08142) 的设计**，对每条推理轨迹的**步级几何动力学量**做可视化分析，验证以下假设：

> 正确推理与含首错推理在隐状态轨迹的**步级几何动力学量演化曲线**上呈现可分离模式（注：不是单步分布对比）。

## CIM 设计要点（本 demo 严格沿用）

- **轨迹定义** $\mathcal{T}_\ell(x) = \{h_{\ell,1}(x), \ldots, h_{\ell,T(x)}(x)\}$ —— 每生成一个 token 取 last-token-position 的第 $\ell$ 层 hidden state
- **预处理** —— 每条轨迹做 mean-centering：$\tilde h_{\ell,t}(x) = h_{\ell,t}(x) - \bar h_\ell(x)$
- **内禀维度** —— TWO-NN 估计器（Facco et al. 2017）；本 demo 补充 Bias-corrected PR（Chun 2025, arXiv:2509.26560）作为小样本鲁棒替代
- **信息体积** —— $V_\ell(x) = \frac{1}{2}\log\det\!\left(I + \frac{d_\ell}{T(x)} Z_\ell(x) Z_\ell(x)^\top\right)$，其中 $Z_\ell$ 是中心化轨迹矩阵
- **三组对照** —— token-shuffle / non-cognitive prompt / truncated trajectory（**留作后续 demo**，本期先看 Figure 1）

## 与 CIM 的区别（本研究 niche）

| 维度 | CIM | 本 demo |
|---|---|---|
| 粒度 | 模型层面（多 prompt pool） | **单条轨迹** |
| 信号来源 | aggregated over many prompts | **single trace's step-level evolution** |
| 监督 | unsupervised model diagnosis | 按 ProcessBench 步级 first-error label 分组 |
| 时间维 | trajectory as static cloud | **沿步轴追踪几何动力学量演化** |
| 关注量 | $D_{\text{stim}}$, $V$, $H$ 三标量 | 6 个步级量：$D_j$, $V_j$, $u_j$, $\theta_j$, $\rho_j$, $\Delta D_j$ |

## 8 个步级几何动力学量

| 量 | 公式 | 启示来源 |
|---|---|---|
| $D_j$ | bias-corrected PR on 前 $w$ 步 | CIM 第二条件（流形维度）|
| $D_j^{\text{TWO-NN}}$ | TWO-NN on 前 $w$ 步 | CIM 默认（小样本可能 noisy，用作对照）|
| $V_j$ | CIM-2 logdet on 前 $w$ 步 | CIM 第三条件（信息体积） |
| $u_j$ | $\|r_j - r_{j-1}\|_2$ | 步间位移幅度 |
| $\theta_j$ | $\angle(T_j, T_{j-1})$ subspace angle | 局部切空间旋转 |
| $\rho_j$ | $\|P_{T_j}(r_j - r_{j-1})\|^2 / u_j^2$ | **流形自洽度**（步间转移落在当前切空间内的占比）|
| $\Delta D_j$ | $D_j - D_{j-1}$ | 流形维度变化率 |
| $\kappa_j$ | $\|r_{j+1} - 2 r_j + r_{j-1}\|$ | **Curved Inference 离散曲率**（Manson 2025, arXiv:2507.21107）—— 已在 LLM residual stream 上验证与 correctness 相关 |

## 文件结构

```
demo/
├── README.md                    # 本文件
├── requirements.txt             # 依赖
├── 01_extract_hidden_states.py  # 加载模型 + ProcessBench，提取每条轨迹 step-level hidden states
├── 02_compute_metrics.py        # 计算 6 个步级几何动力学量
├── 03_plot_figure_1.py          # 生成 Figure 1（正确 vs 含首错对比，6 子图）
├── utils/
│   ├── __init__.py
│   ├── intrinsic_dim.py         # TWO-NN, PR, Bias-corrected PR
│   ├── info_volume.py           # CIM-2 logdet 信息体积
│   ├── tangent_space.py         # 局部 PCA, subspace angle, 流形自洽度
│   └── step_boundaries.py       # 步代表 token 识别（基于换行的简单切分）
├── data/
│   └── (放置缓存的 hidden states 与 ProcessBench 子集)
└── output/
    └── (放置 Figure 1 与统计报告)
```

## 使用流程

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 提取 hidden states (需要 GPU，约 1-2 小时)
python 01_extract_hidden_states.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --dataset Qwen/ProcessBench \
    --subset math \
    --n_correct 25 \
    --n_error 25 \
    --layer -1 \
    --output data/hidden_states.npz

# 3. 计算几何动力学量（CPU 即可）
python 02_compute_metrics.py \
    --input data/hidden_states.npz \
    --window 5 \
    --output data/metrics.npz

# 4. 生成 Figure 1
python 03_plot_figure_1.py \
    --input data/metrics.npz \
    --output output/figure_1.png
```

## 预期产出

**Figure 1**（2×3 子图）：
- 子图 1: $D_j$ 演化曲线对比（正确 vs 含首错，含首错按首错步 τ 对齐）
- 子图 2: $V_j$ 演化曲线对比
- 子图 3: $u_j$ 演化曲线对比
- 子图 4: $\theta_j$ 演化曲线对比
- 子图 5: $\rho_j$ 演化曲线对比
- 子图 6: $\Delta D_j$ 演化曲线对比

**Figure 1 判定**：
- 如果至少有 2-3 个量在 τ 附近呈现明显分离 → CIM-启示假设在 single-trajectory 步级上成立 → 围绕这些量设计后续方法
- 如果全部曲线重合 → 假设失败 → 需调整（target layer / window size / HARP 投影等）

## 注意

- 本 demo 完全 training-free，没有探针
- 目的是先看现象、不是先做方法
- 所有数值方法尽量与 CIM 原文保持一致（用 TWO-NN 作 baseline、PR 作小样本替代、CIM-2 logdet 直接用）
