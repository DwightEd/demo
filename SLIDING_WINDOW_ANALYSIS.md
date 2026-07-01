# 滑动窗口 vs Step-based 方法对比分析

## 一、Spectral Geometry 的滑动窗口方法

### 1.1 为什么使用滑动窗口？

**论文**: "The Spectral Geometry of Thought: Phase Transitions, Instruction Reversal, Token-Level Dynamics, and Perfect Correctness Prediction in How Transformers Reason"

**核心参数**:
- **窗口大小 w = 10**: 每个窗口包含10个连续token
- **步长 stride = 5**: 每次滑动5个token（50%重叠）

**优势**:
1. **统一粒度**: 所有窗口都是10个token，几何特征可直接比较
2. **高分辨率**: 可以捕捉推理过程中的细微变化
3. **无依赖**: 不需要step标注，适用于任何推理链
4. **token-level动态**: 真正反映每个token位置的几何状态

### 1.2 滑动窗口的实现逻辑

```python
WINDOW_SIZE = 10
STRIDE = 5

# 对于长度为R的序列
for start in range(0, R - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    H_window = hidden[start:end, layer_idx, :]  # 固定10个token
    
    # 计算该窗口的几何特征
    geom = compute_window_geometry(H_window)
```

**输出**: 每个窗口一个几何特征，形成一个几何轨迹

### 1.3 为什么效果好？

1. **固定大小的一致性**
   - 所有窗口都是10个token，scatter matrix的形状一致
   - 特征值分布可比，不会因step长度不同而引入偏差

2. **高时间分辨率**
   -Stride=5意味着每5个token就有一个观测点
   - 可以捕捉推理过程中的快速变化

3. **完整特征值分解**
   - 使用真正的 eigh 分解（不是对角近似）
   - 捕获真实的协方差结构

4. **Token-level 谱轨迹**
   - 每个token位置都有谱特征
   - 可以分析"per-token spectral trajectory"

---

## 二、我们的 Step-based 方法

### 2.1 数据来源

**ProcessBench 数据集**提供了预标注的steps:
```python
# NPZ文件中的step_token_ranges
step_token_ranges = [
    [(0, 45), (45, 120), (120, 180), ...],  # Chain 0
    [(0, 50), (50, 110), (110, 175), ...],  # Chain 1
    ...
]
```

**Step定义**: 通过 `utils/step_boundaries.py` 中的 `find_step_token_ranges()` 生成
- 使用预解析的step文本（如"Step 1: ..."）
- 通过 tokenizer.offset_mapping 精确定位token范围

### 2.2 Step-based 的优势

1. **语义完整性**
   - 每个step对应一个完整的推理单元
   - 更符合人类的推理逻辑

2. **标注质量**
   - 使用人工/自动标注的step边界
   - 比简单的固定窗口更有意义

### 2.3 Step-based 的劣势

1. **长度不一致**
   ```
   Step 0: 45 tokens
   Step 1: 75 tokens
   Step 2: 60 tokens
   ```
   - 不同step的几何特征不可直接比较
   - scatter matrix的形状不同 (45×45 vs 75×75)

2. **分辨率低**
   - 如果一个推理链只有5个step，只有5个数据点
   - 滑动窗口可以有 100+ 个数据点

3. **边界敏感性**
   - step边界定义的微小变化会影响结果
   - 依赖于step标注的质量

---

## 三、NPZ 数据结构详解

### 3.1 完整字段列表

```python
data = np.load('full_omnimath.npz', allow_pickle=True)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `problem_ids` | ndarray (N,) | 问题ID |
| `is_correct_strict` | ndarray (N,) | 正确性 (0=正确, 1=错误) |
| `step_token_ranges` | ndarray (N,) object | **Step的token范围** |
| `stepcloud` | ndarray (N,) object | Step级别的特征云 |
| `stepgeom` | ndarray (N,) object | Step级别的几何特征 |
| `gold_error_step` | ndarray (N,) | 发生错误的step索引 |

### 3.2 step_token_ranges 格式

```python
# 每个元素是一个列表，包含该链的所有step边界
step_token_ranges[0] = [(0, 45), (45, 120), (120, 180), ...]

# 格式: List[Tuple[int, int]]
# 每个 tuple: (start_token_idx, end_token_idx)
# 含义: 该step占据的token范围
```

**示例**:
```python
# 推理链: "Let x=5\nThen y=2x\nSo y=10"
# token化后: [10, 15, 20, 30, 40, 50, ...]
# step_token_ranges可能为:
[(0, 3), (3, 6), (6, 9)]
#    ↑      ↑      ↑
#  "Let x=5" "Then y=2x" "So y=10"
```

### 3.3 已有的 Step-level 几何特征

**stepgeom 字段**: 
- 已经包含step级别的几何特征
- 可能是通过其他方法预计算的
- **需要检查具体格式和计算方式**

---

## 四、为什么我们应该用已有的 Step 数据？

### 4.1 数据质量

**ProcessBench 的 step 标注**:
- 基于**真实的推理结构**
- 每个 step 是一个语义完整的推理单元
- 比"任意10个token"更有意义

### 4.2 与我们的研究目标一致

**我们的目标**: 分析"推理轨迹的几何相变"
- **step是推理的自然单元**
- step之间的转换才是真正的"推理转变"

**对比**:
- 滑动窗口: 分析"任意10个token"的几何
- Step-based: 分析"推理步骤"的几何

### 4.3 实际数据已存在

**NPZ文件已有**:
- `step_token_ranges`: 精确的step边界
- `stepgeom`: 可能已有几何特征
- `gold_error_step`: 哪个step出错了

**这是真实的标注数据，不应该浪费！**

---

## 五、当前问题的根源

### 5.1 不是方法问题，是计算错误

**问题不在"用step还是用滑动窗口"**，而在于：

1. **特征值计算错误**
   ```python
   # 错误：对角元近似
   S_diag = np.sum(H_norm ** 2, axis=0) / n_tokens
   eigenvalues = np.sort(S_diag)[::-1][:10]
   
   # 正确：完整分解
   S = (H_norm.T @ H_norm) / n_tokens
   eigvals = eigh(S, eigvals_only=True)
   ```

2. **缓存数据损坏**
   - 现有缓存基于错误的计算
   - 需要清除并重新计算

### 5.2 为什么 Spectral Geometry 效果好？

**不是因为用了滑动窗口**，而是因为：
1. **计算正确**: 使用完整的特征值分解
2. **特征有效**: κ, rank, entropy 都是基于真实eigenvalues
3. **分析细致**: token-level 的高分辨率分析

**如果我们修复计算错误，step-based 方法也能有效！**

---

## 六、建议的实现方案

### 方案A: 修复现有计算（推荐）

```python
# 保持使用 step_token_ranges
# 但修复 compute_step_geometry_ultra_fast

def compute_step_geometry_fast(H, step_id, layer_id):
    """修复版：使用完整特征值分解"""
    n_tokens, d = H.shape
    
    # 归一化
    H_norm = H / (np.linalg.norm(H, axis=1, keepdims=True) + eps)
    
    # 一阶矩
    mu = H_norm.mean(axis=0)
    kappa = float(np.linalg.norm(mu))
    
    # Scatter matrix (完整)
    S = (H_norm.T @ H_norm) / n_tokens  # (d, d)
    
    # 特征值分解 (完整)
    try:
        from scipy.linalg import eigh
        eigvals = eigh(S, eigvals_only=True)
        eigvals = np.sort(eigvals)[::-1]
        eigvals = eigvals / (eigvals.sum() + eps)
    except:
        # 降级方案
        return None
    
    # 计算特征
    lam = eigvals[eigvals > eps]
    eff_rank = float(np.exp(-np.sum(lam * np.log(lam + eps))))
    spectral_entropy = float(-np.sum(eigvals * np.log(eigvals + eps)))
    
    return {
        'step_id': step_id,
        'layer': layer_id,
        'n_tokens': n_tokens,
        'kappa': kappa,
        'eff_rank': eff_rank,
        'spectral_entropy': spectral_entropy,
        'eigenvalues': eigvals[:10],
    }
```

### 方案B: 检查 stepgeom 字段

```python
# 检查NPZ中已有的stepgeom
data = np.load('full_omnimath.npz', allow_pickle=True)
stepgeom = data['stepgeom']

# 看看是否已经有正确的几何特征
# 如果有，可以直接使用
```

### 方案C: 混合方法

```python
# 使用 step_token_ranges 作为主边界
# 但在每个step内部使用滑动窗口
# 这样既有语义完整性，又有高分辨率
```

---

## 七、下一步行动

1. **先检查 NPZ 的 stepgeom 字段**
   ```bash
   python check_npz_structure.py
   ```

2. **如果 stepgeom 已有正确数据**
   - 直接使用
   - 不需要重新计算

3. **如果 stepgeom 不完整或错误**
   - 修复 `compute_step_geometry_ultra_fast`
   - 使用完整特征值分解
   - 清除缓存，重新计算

4. **保留 step_token_ranges**
   - 这是真实的推理结构
   - 比滑动窗口更有意义
