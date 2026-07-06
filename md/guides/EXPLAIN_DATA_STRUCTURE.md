# 缓存数据结构说明

## 数据组织结构

```
/gz-data/research/demo/data/hidden/cache/omnimath/
├── chain_0.pkl      # 第0条推理链
├── chain_1.pkl      # 第1条推理链
├── ...
└── chain_997.pkl    # 第997条推理链
```

## 单个chain的数据结构

每个`chain_*.pkl`文件包含一个`ReasoningTrajectory`对象：

```
ReasoningTrajectory {
    chain_id: 0                          # 链ID
    problem_id: 123                      # 问题ID
    is_correct: True                     # 最终答案是否正确
    n_steps: 5                           # 总step数
    step_ranges: [(0,45), (45,120), ...] # 每个step的token范围

    steps: {                             # 每层的几何特征
        14: {                           # Layer 14
            0: StepGeometry{...},       # Step 0的特征
            1: StepGeometry{...},       # Step 1的特征
            2: StepGeometry{...},       # Step 2的特征
            ...
        },
        18: { ... },                    # Layer 18
        22: { ... },                    # Layer 22
    }
}
```

## 每个StepGeometry包含

```
StepGeometry {
    step_id: 0                    # Step ID
    layer: 14                     # 层ID
    n_tokens: 45                  # 该step的token数
    kappa: 0.82                  # 向量集中度（一阶矩）
    eff_rank: 2.3                # 有效秩（二阶矩）
    spectral_entropy: 3.1        # 谱熵
    norm: 0.15                   # 向量范数
    eigenvalues: [0.4, 0.2, ...] # 前10个特征值
}
```

## 总结

- **每个pkl文件 = 1条推理链**
- **每条链 = 多个step**（比如5个step）
- **每个step = 4层的几何特征**（layer 10, 14, 18, 22）

所以998个pkl文件 × 平均5个step × 4层 ≈ **20,000个step几何特征**
