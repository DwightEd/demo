# Step 区分方式和缓存格式说明

## 一、Step 区分方式

### 1.1 Step 来源

我们项目的 step 边界来自**数据集预标注的 `steps` 列表**（ProcessBench 格式）。

### 1.2 Step Token Ranges 生成方法

**代码位置**: `utils/step_boundaries.py`

**核心逻辑**:
```python
def find_step_token_ranges(tokenizer, prompt_text, response_text, steps_text):
    """
    输入:
    - steps_text: 预解析的step字符串列表 (如 ["Step 1: ...", "Step 2: ..."])
    
    输出:
    - ranges: [(start_token_idx, end_token_idx), ...]
    """
    # 1. 拼接完整文本
    full_text = prompt_text + response_text
    
    # 2. 获取tokenizer的offset_mapping
    encoding = tokenizer(full_text, return_offsets_mapping=True)
    offsets = encoding["offset_mapping"]  # [(char_start, char_end), ...]
    
    # 3. 对每个step文本，找到其在response中的位置
    for step in steps_text:
        idx = response_text.find(step, cursor)
        start_char = len(prompt_text) + idx
        end_char = start_char + len(step)
        
        # 4. 根据字符位置找到对应的token范围
        for t, (a, b) in enumerate(offsets):
            if b <= start_char: continue
            if a >= end_char: break
            # ... 找到token边界
        
        ranges.append((start_tok, end_tok))
    
    return ranges
```

**示例**:
```python
# 输入
steps_text = ["Let x = 5", "Then y = 2x", "So y = 10"]
prompt = "Solve: 2x = 10"
response = "Let x = 5\nThen y = 2x\nSo y = 10"

# 输出 (假设)
step_token_ranges = [
    (5, 12),   # Step 1: tokens [5, 12]
    (13, 20),  # Step 2: tokens [13, 20]
    (21, 27),  # Step 3: tokens [21, 27]
]
```

### 1.3 NPZ 文件中的 step_token_ranges

```python
# NPZ文件结构
data = np.load('full_omnimath.npz', allow_pickle=True)
step_token_ranges = data['step_token_ranges']  # shape: (n_chains,)
# 每个元素: List[Tuple[int, int]] 或 None
```

---

## 二、缓存格式详解

### 2.1 缓存文件结构

**保存位置**: `data/hidden/cache/{subset}/chain_{chain_id}.pkl`

**保存内容**: 完整的 `ReasoningTrajectory` 对象

### 2.2 ReasoningTrajectory 数据结构

```python
@dataclass
class ReasoningTrajectory:
    chain_id: int           # 链索引
    problem_id: int         # 问题ID
    is_correct: bool        # 是否正确
    n_steps: int           # step数量
    step_ranges: list       # [(start, end), ...] token范围
    steps: dict            # {layer_id: {step_id: StepGeometry}}
```

**示例**:
```python
ReasoningTrajectory(
    chain_id=0,
    problem_id=1234,
    is_correct=True,
    n_steps=5,
    step_ranges=[(0, 45), (45, 120), (120, 180), (180, 240), (240, 280)],
    steps={
        10: {0: StepGeometry(...), 1: StepGeometry(...), ...},
        14: {0: StepGeometry(...), 1: StepGeometry(...), ...},
        18: {...},
        22: {...}
    }
)
```

### 2.3 StepGeometry 数据结构

```python
@dataclass
class StepGeometry:
    step_id: int                    # step索引
    layer: int                      # 层索引 (10/14/18/22)
    n_tokens: int                  # 该step的token数
    kappa: float                    # 方向集中度 ||mean(û)||
    eff_rank: float                 # 有效秩
    spectral_entropy: float         # 谱熵
    norm: float                     # 平均token范数
    eigenvalues: np.ndarray         # 特征值 (当前实现: 对角元近似)
    principal_directions: np.ndarray # 主成分向量 (cache模式为空)
```

**字段详解**:

| 字段 | 类型 | 说明 | 当前实现 |
|------|------|------|----------|
| `step_id` | int | Step索引 (0, 1, 2, ...) | ✓ |
| `layer` | int | 层ID (10/14/18/22) | ✓ |
| `n_tokens` | int | 该step的token数量 | ✓ |
| `kappa` | float | 方向集中度, 范围[0,1] | ✓ 正确 |
| `eff_rank` | float | 有效秩, 范围[1,d] | ⚠️ 对角近似 |
| `spectral_entropy` | float | 谱熵, 范围[0,log(d)] | ⚠️ 基于错误eigenvalues |
| `norm` | float | 平均token范数 | ✓ |
| `eigenvalues` | np.ndarray | 特征值 (前10个) | ❌ 对角元近似 |
| `principal_directions` | np.ndarray | 主成分向量 | ❌ 空数组 |

### 2.4 缓存生成流程

```python
# 1. 从NPZ加载step_token_ranges
ranges = step_token_ranges[idx]  # [(start, end), ...]

# 2. 对每个step和每个layer计算几何特征
for layer_idx, layer_id in enumerate([10, 14, 18, 22]):
    for step_id, (start, end) in enumerate(ranges):
        # 提取hidden states
        H = hidden[start:end, layer_idx, :]
        
        # 计算几何特征 (compute_step_geometry_ultra_fast)
        geom = compute_step_geometry_ultra_fast(H, step_id, layer_id)
        
        # 保存
        layer_steps[step_id] = StepGeometry(**geom)

# 3. 保存为pickle
save_cached_features(cache_dir, idx, traj)
```

---

## 三、当前实现的问题

### 3.1 Step 定义的问题

**当前**: 使用预定义的 `step_token_ranges`（可变长度）
- Step 0: tokens [0, 45] (45 tokens)
- Step 1: tokens [45, 120] (75 tokens)
- Step 2: tokens [120, 180] (60 tokens)

**问题**:
1. 不是真正的滑动窗口
2. 每个step长度不同，几何特征不可比
3. 与 Spectral Geometry 论文方法不一致

### 3.2 特征值计算的问题

**当前** (`compute_step_geometry_ultra_fast`):
```python
S_diag = np.sum(H_norm ** 2, axis=0) / n_tokens  # 对角元
eigenvalues = np.sort(S_diag)[::-1][:10]  # 当作特征值
```

**问题**:
1. 对角元 ≠ 特征值
2. 丢失协方差信息
3. 导致所有下游指标失效

### 3.3 缓存中的数据状态

**现有缓存** (`data/hidden/cache/omnimath/chain_*.pkl`):
```
ReasoningTrajectory
├── step_ranges: [(start, end), ...]  # 可变长度step
└── steps: {layer: {step_id: StepGeometry}}
    └── StepGeometry
        ├── kappa: ✓ 正确
        ├── eff_rank: ⚠️ 基于错误eigenvalues
        ├── spectral_entropy: ⚠️ 基于错误eigenvalues
        ├── eigenvalues: ❌ 对角元近似
        └── principal_directions: ❌ 空数组
```

**结论**: 现有缓存数据**不可用**，需要清除并重新计算。

---

## 四、修复建议

### 4.1 优先级

| 优先级 | 问题 | 修复方案 |
|--------|------|----------|
| P0 | 对角近似错误 | 使用完整 eigh 分解 |
| P1 | 缓存数据错误 | 清除缓存，重新计算 |
| P2 | Step定义不一致 | 考虑使用滑动窗口 |

### 4.2 修复方案

**方案A**: 修复特征值计算（必须）
```python
# 替换 compute_step_geometry_ultra_fast
def compute_step_geometry_fast(H, step_id, layer_id):
    S = (H_norm.T @ H_norm) / n_tokens  # Scatter matrix
    eigvals = eigh(S, eigvals_only=True)  # 完整分解
    eigvals = np.sort(eigvals)[::-1]
    # ... 计算 eff_rank, spectral_entropy
```

**方案B**: 使用真正的滑动窗口（可选）
```python
WINDOW_SIZE = 10
STRIDE = 5

for start in range(0, R - WINDOW_SIZE + 1, STRIDE):
    H = hidden[start:start+WINDOW_SIZE, layer_idx, :]
    geom = compute_step_geometry_fast(H, window_id, layer_id)
```

**方案C**: 保留现有step定义，修复计算（折中）
- 继续使用 `step_token_ranges`
- 但使用完整的特征值分解
- 清除所有旧缓存

---

## 五、数据集字段说明

### NPZ 文件结构

```python
data = np.load('full_omnimath.npz', allow_pickle=True)
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `problem_ids` | ndarray (n,) | 问题ID |
| `is_correct_strict` | ndarray (n,) | 严格正确性标签 (0=正确, 1=错误) |
| `step_token_ranges` | ndarray (n,) | 每个链的step token范围 |
| `stepcloud` | ndarray (n,) | Step级别的特征云 |
| `stepgeom` | ndarray (n,) | Step级别的几何特征 |

### Hidden 文件

```
data/hidden/omnimath/{subset}-{chain_id}.npy
```

**格式**: `(R, 4, 4096)`
- `R`: 响应的总token数
- `4`: 4个层的hidden states (layers 10, 14, 18, 22)
- `4096`: hidden dimension
