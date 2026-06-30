# 方向2完整实现：Cross-Step Geometric Coherence

## 📚 论文框架

**Title:** "Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning"

### 核心研究假设

> **错误推理是几何轨迹的相变，不是单点异常**

- **正确推理**：步骤间的表示空间连续、平滑演化，几何轨迹保持低维流形结构
- **错误推理**：几何轨迹发生"相变"——要么断裂（信息流中断），要么扭曲（进入错误子空间）

### 关键洞察

单步骤的κ/eff_rank无法区分"合法发散"vs"错误发散"，但步骤间的几何轨迹形状可以：
- 正确推理 = 平滑曲线
- 错误推理 = 突变/分叉

---

## 🏗️ 方法架构：三层分析

### Layer 1: Step-wise Geometry (每步的几何特征)

```
┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Step 1   │  │ Step 2   │  │ Step 3   │  │  ...     │
│ κ, eff_R │  │ κ, eff_R │  │ κ, eff_R │  │           │
└──────────┘  └──────────┘  └──────────┘  └──────────┘
```

数据结构：
```python
@dataclass
class StepGeometry:
    step_id: int
    layer: int
    kappa: float           # 一阶矩：方向集中度
    eff_rank: float        # 二阶矩：有效秩
    spectrum: np.ndarray   # 谱形状向量
```

---

### Layer 2: Trajectory Geometry (步骤间几何关系)

三个核心指标：

#### 1. Local Smoothness（局部平滑度）

```python
def local_smoothness(geometry_sequence):
    """相邻步骤的几何相似度

    正确推理：smoothness高（渐进变化）
    错误推理：smoothness低（突变）
    """
    for i in range(len(geometry_sequence)-1):
        kappa_sim = 1 - abs(geom_i.kappa - geom_ip1.kappa)
        eff_rank_sim = 1 - abs(geom_i.eff_rank - geom_ip1.eff_rank)
        spectrum_sim = jensen_shannon(geom_i.spectrum, geom_ip1.spectrum)
        smoothness.append(0.4*kappa_sim + 0.3*eff_rank_sim + 0.3*spectrum_sim)
    return np.mean(smoothness)
```

#### 2. Global Coherence（全局连贯性）

```python
def global_coherence(geometry_sequence):
    """首尾步骤的关联强度

    正确推理：首尾连贯（始终围绕问题展开）
    错误推理：首尾断裂（后期遗忘问题）
    """
    early_spectrum = np.mean([g.spectrum for g in geometry_sequence[:3]], axis=0)
    late_spectrum = np.mean([g.spectrum for g in geometry_sequence[-3:]], axis=0)
    return 1 - jensen_shannon(early_spectrum, late_spectrum)
```

#### 3. Spectral Evolution Stability（谱演化稳定性）

```python
def spectral_evolution_stability(geometry_sequence):
    """谱形状沿步骤的演化稳定性

    正确推理：谱形状渐进演化
    错误推理：谱形状跳变
    """
    spectra = np.array([g.spectrum for g in geometry_sequence])
    diffs = np.diff(spectra, axis=0)
    variance = np.var(np.linalg.norm(diffs, axis=1))
    return 1 / (1 + variance)
```

---

### Layer 3: Phase Transition Detection (相变检测)

#### 模式1：Shallow Lock-in = 浅层平滑度突降

```python
def detect_shallow_lockin_trajectory(chain, shallow_layers=[10, 14, 18]):
    """检测浅层的信息流锁定

    Lock-in特征：在浅层，步骤间的几何相似度突然集中在当前步骤
    （当前步骤不再与之前步骤相关，而是自强化）
    """
    # 计算每一步与之前所有步骤的平均相似度
    coherence_profile = []
    for i in range(1, len(geom_sequence)):
        sim_to_past = [geometric_sim(geom_sequence[i], geom_sequence[j])
                      for j in range(i)]
        coherence_profile.append(np.mean(sim_to_past))

    # 检测突然下降
    if detect_sudden_drop(coherence_profile):
        return True, layer
```

#### 模式2：Deep Decay = 深层谱演化失稳

```python
def detect_deep_decay_trajectory(chain, deep_layers=[22, 26, 30]):
    """检测深层的信息衰减

    Decay特征：在深层，谱形状的演化不再平滑，而是混乱
    """
    stability = spectral_evolution_stability(geom_sequence)
    late_entropy = np.mean([entropy(s) for s in geom_sequence[-3:]])

    # Decay信号：稳定性低 AND 晚期谱熵低（信息丢失）
    if stability < 0.5 and late_entropy < 1.0:
        return True, layer
```

---

## 🔬 实验验证设计

### H1: 轨迹平滑度区分error vs correct

- **检验**: Mann-Whitney U test on smoothness scores
- **预期**: error < correct, p < 0.001

### H2: Shallow Lock-in模式在error中更频繁

- **检验**: Fisher's exact test
- **预期**: OR > 2, p < 0.01

### H3: Deep Decay模式在error中更频繁

- **检验**: Fisher's exact test
- **预期**: OR > 2, p < 0.01

### H4: 基于轨迹的检测器优于基于单步几何的检测器

- **检验**: Paired t-test on per-chain AUROC
- **预期**: trajectory > step-wise, p < 0.05

---

## 📝 论文组织

### Abstract结构

1. **问题**：现有方法看单点或单向注意力，忽略了步骤间的几何演化
2. **假设**：错误推理是几何轨迹的相变
3. **方法**：三层分析（step-wise → trajectory → phase transition）
4. **发现**：识别出两种相变模式（Shallow Lock-in, Deep Decay）
5. **贡献**：提供步骤级的几何诊断，无监督、可解释

### Main Sections

1. **Introduction** - 推理轨迹视角的必要性
2. **Related Work** - 与两篇论文的对话
3. **Method** - 三层架构
4. **Experiments** - 验证四个假设
5. **Analysis** - 相变模式的可视化
6. **Discussion** - 何时检测有效，何时失效

---

## 🚀 使用方法

### 在远程服务器上运行

```bash
# 正确的项目路径
cd /gz-data/research/demo/
git pull

# 运行验证
python trajectory_phase_transition.py /gz-data/research/demo/data/features/full_omnimath.npz --output_dir ./trajectory_results

# 或运行多个数据集
for dataset in gsm8k math omnimath; do
    python trajectory_phase_transition.py /gz-data/research/demo/data/features/full_${dataset}.npz --output_dir ./trajectory_results
done
```

---

## 📊 数据格式

输入：`full_*.npz` 文件

- `stepcloud`: (N, T, 33, 9) - 每个步骤每层的几何特征
  - T = 步骤数
  - 33 = 层数
  - 9 = 特征数 (n_tokens, kappa, eff_rank, lam1, gap, ...)
- `problem_ids`: (N,) - 问题ID
- `is_correct_strict`: (N,) - 正确性标签 (0=correct, 1=error)
- `sv_layers`: (33,) - 层索引

数据位置（在box上）：`/gz-data/research/demo/data/features/`

---

## 📈 输出结果

`trajectory_validation_results.json`:

```json
{
  "metadata": {
    "subset": "omnimath",
    "n_chains": 1000,
    "n_correct": 600,
    "n_error": 400
  },
  "results": {
    "L14_H1": {
      "hypothesis": "H1",
      "metric": "smoothness",
      "error_mean": 0.65,
      "correct_mean": 0.78,
      "cohens_d": 0.45,
      "p_value": 0.0001,
      "interpretation": "error=0.650 < correct=0.780: d=0.450, p=0.0001"
    },
    "L14_H2": {
      "hypothesis": "H2",
      "metric": "shallow_lockin",
      "error_mean": 0.35,
      "correct_mean": 0.15,
      "cohens_d": NaN,
      "p_value": 0.005,
      "interpretation": "OR=3.12, p=0.0050"
    },
    ...
  }
}
```

---

## 🔗 与现有工作的对比

| 维度 | Spectral Geometry | Step Flow | 我们的工作 |
|------|------------------|-----------|-----------|
| 分析单元 | 单点（整个回答的谱） | 步骤间的注意力流 | 步骤间的几何轨迹 |
| 核心发现 | 谱分布预测正确性 | 信息流断裂模式 | 几何轨迹的平滑性=推理健康度 |
| 方法 | 谱统计 | 梯度加权注意力 | 轨迹的几何拓扑分析 |
| 局限 | 不看步骤演化 | 需要backward pass | 无监督、可解释、步骤级 |

**我们的novelty**：
- 将两篇论文的思想结合：谱方法（Spectral Geometry）+ 步骤演化（Step Flow）
- 提出几何轨迹视角：推理是一条曲线，错误是曲线的"扭结"

---

## 📁 文件结构

```
demo/
├── trajectory_phase_transition.py    # 主实现文件
├── run_phase_transition.sh           # 服务器执行脚本
├── TRAJECTORY_PHASE_README.md        # 本文档
└── results/                          # 输出目录
    ├── gsm8k/
    │   └── trajectory_validation_results.json
    ├── math/
    │   └── trajectory_validation_results.json
    └── omnimath/
        └── trajectory_validation_results.json
```

---

## 🎯 下一步计划

1. **运行验证实验** - 在omnimath上确认H1-H4
2. **分析结果** - 统计显著性、效应量
3. **可视化** - 绘制轨迹图、相变检测图
4. **撰写论文** - 按照上述组织结构
