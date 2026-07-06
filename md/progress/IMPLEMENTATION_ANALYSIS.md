# 实现分析：为什么结果与预期不符

## 核心问题总结

### 问题1: 数据分段方式错误

**当前实现** (`data_loading_cache.py` 第216-220行):
```python
# 使用预定义的 step_token_ranges 分段
for step_id, (start, end) in enumerate(ranges):  # ranges = step_token_ranges[idx]
    if end <= start:
        continue
    H = hidden[start:end, layer_idx, :].copy()  # 直接使用整个step的token
    geom = compute_step_geometry_ultra_fast(H, step_id, layer_id)
```

- 每个 step 的 token 数量是可变的（如 91 个 token）
- 不是真正的滑动窗口

**应该是** (`data_loading_sliding.py` 第89-96行):
```python
WINDOW_SIZE = 10  # 固定窗口大小
STRIDE = 5        # 固定步长

for start in range(0, R - WINDOW_SIZE + 1, STRIDE):
    end = start + WINDOW_SIZE
    H_window = hidden[start:end, layer_idx, :]  # 固定10个token
    geom = compute_window_geometry(H_window, len(windows), layer_id)
```

---

### 问题2: 对角近似完全错误

**当前实现** (`data_loading_cache.py` 第34-44行):
```python
# 用scatter matrix的对角元作为近似的eigenvalues
S_diag = np.sum(H_norm ** 2, axis=0) / n_tokens  # 对角元

# 归一化
if S_diag.sum() > eps:
    S_diag = S_diag / S_diag.sum()
else:
    S_diag = np.ones(4096) / 4096

# 降序排列作为近似谱
eigenvalues = np.sort(S_diag)[::-1][:10]  # 前10个
```

**为什么这是错误的**:

scatter matrix 的对角元 ≠ 特征值

例如：
```
矩阵 A = [[0.5, 0.5],
         [0.5, 0.5]]

对角元 = [0.5, 0.5]
真实特征值 = [1.0, 0.0]  (完全不同!)
```

对角元只代表每个维度上的平均能量，不反映协方差结构。而 scatter matrix 的特征值才代表主方向上的方差分布。

**应该是** (`data_loading.py` 第153-170行):
```python
# 完整的 scatter matrix
S = (H_norm.T @ H_norm) / n_tokens  # (d, d)

# 真正的特征值分解
eigvals = eigh(S, eigvals_only=True)
eigvals = np.sort(eigvals)[::-1]  # 降序
eigvals = eigvals / (eigvals.sum() + eps)  # 归一化
```

---

### 问题3: 相似度计算降级

**当前行为** (`trajectory_geometry.py` 第370-379行):
```python
# 检查是否有 principal_directions
if has_size1 and has_size2:
    sim = geometric_sim(g1, g2)  # 使用完整几何相似度
else:
    sim = geometric_sim_scalar_only(g1, g2)  # 降级到标量相似度
```

由于 cache 模式没有计算 principal_directions，所有相似度计算都降级到 `geometric_sim_scalar_only`，丢失了：
- 主方向旋转 (principal_direction_rotation)
- 子空间漂移 (subspace_drift)

**geometric_sim_scalar_only** (第317-343行):
```python
def geometric_sim_scalar_only(g1: StepGeometry, g2: StepGeometry):
    # κ相似度
    kappa_sim = 1.0 - min(abs(g1.kappa - g2.kappa), 1.0)

    # eff_rank相似度
    max_er = max(g1.eff_rank, g2.eff_rank, 1.0)
    eff_rank_sim = 1.0 - abs(g1.eff_rank - g2.eff_rank) / max_er

    # 谱相似度
    if _has_non_empty_array(g1, 'eigenvalues') and _has_non_empty_array(g2, 'eigenvalues'):
        if g1.eigenvalues.size >= 2 and g2.eigenvalues.size >= 2:
            s1 = g1.eigenvalues[:10]
            s2 = g2.eigenvalues[:10]
            s1 = s1 / (s1.sum() + 1e-12)
            s2 = s2 / (s2.sum() + 1e-12)
            spectrum_sim = 1.0 - jensenshannon(s1, s2)
        else:
            spectrum_sim = 0.5
    else:
        spectrum_sim = 0.5

    # 等权组合
    return (kappa_sim + eff_rank_sim + spectrum_sim) / 3.0
```

但是，由于 eigenvalues 是错误的（对角近似），spectrum_sim 也是错误的。

---

### 问题4: 指标设计问题

**smoothness 计算逻辑**:
```python
smoothness = mean(similarities between consecutive steps)
```

如果所有 similarities 都在 0.8-0.9 范围内（当前数据），那么：
- 正确和错误的 smoothness 差异会很小（如 0.846 vs 0.847）
- Cohen's d 会接近 0（当前是 -0.026）

**为什么差异这么小？**

可能的原因：
1. 当前数据本身就没有明显的"突变"模式
2. 指标设计没有捕捉到真正的错误信号
3. 错误的 eigenvalues 导致所有计算都失效

---

## 与 Spectral Geometry 论文的差异

### 论文的方法：
1. **真正的滑动窗口**: w=10, stride=5
2. **完整的特征值分解**: 不是对角近似
3. **Token-level 分析**: 每个 token 都有谱特征
4. **动态演化分析**: focus on spectral shape evolution

### 当前实现：
1. ✗ 使用 step_token_ranges（可变长度）
2. ✗ 对角近似（完全错误）
3. ✗ Step-level 而非 token-level
4. ✗ 错误的 eigenvalues 导致所有下游分析失效

---

## 诊断结果解读

```
smoothness:
  Correct: mean=0.8462
  Error:   mean=0.8475
  Cohen's d: -0.026  (几乎为0，无区分能力)
```

**Cohen's d 解释**:
- |d| < 0.2: 小效应（当前情况）
- 0.2 ≤ |d| < 0.5: 中等效应
- |d| ≥ 0.5: 大效应

当前 d = -0.026，说明该指标完全无法区分正确和错误。

```
kappa:
  Correct: mean=0.5633
  Error:   mean=0.5701

eff_rank:
  Correct: mean=82.24
  Error:   mean=76.09

spectral_entropy:
  Correct: mean=1.6177
  Error:   mean=1.5799
```

这些差异很小，且方向不一致（kappa 是错误组更高，eff_rank 是正确组更高）。

---

## 修复建议

### 方案A: 修复当前实现（不推荐）

需要同时修复：
1. 切换到真正的滑动窗口
2. 使用完整的特征值分解
3. 重新计算所有缓存

### 方案B: 使用滑动窗口模式（推荐）

```bash
# 使用 data_loading_sliding.py
python main.py \
    --mode sliding \
    --window_size 10 \
    --stride 5
```

但是当前 main.py 不支持 sliding 模式，需要添加。

### 方案C: 检查原始假设

可能需要重新考虑：
1. 这些指标是否真的能区分正确/错误？
2. 数据集是否包含足够的信号？
3. 是否需要其他指标（如 orientation, qvec）？

---

## 下一步行动

1. **添加真正的滑动窗口支持到 main.py**
2. **使用完整的特征值分解**（不是对角近似）
3. **重新计算并分析**
4. **如果仍无显著差异，考虑调整假设或指标**
