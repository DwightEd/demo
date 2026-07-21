# Reasoning Activation Divergence

> 原始残差流的远端入口见 [RUN_RAW_REMOTE.md](RUN_RAW_REMOTE.md)。正式实验使用
> `functional_divergence.raw_residual_experiment` 直接读取 per-chain hidden
> shards；下面的 `geometry_audit.npz` 内容仅保留为早期代理实验记录。

本子项目研究推理首错附近的激活变化是否不仅是径向放大，还包含有功能的方向变化。当前包含两条互补管线：

1. 跨事件 affine transport residual 与任务错误探针 pullback Fisher；
2. 联合 (t\times layer) affine operator field，用深度边、时间边及 plaquette 路径不一致衡量二维动态。

## 联合 time × layer 方法

输入状态保持 `[sample, time, layer, feature]`，不再把 layer 展平。每个交叉验证折只使用正确控制样本，并在所有 time/layer cell 上拟合一个共享 PCA/尺度坐标系。随后拟合：

- 深度算子 (D_{t,\ell}: z_{t,\ell}\to z_{t,\ell+1})；
- 时间算子 (T_{t,\ell}: z_{t,\ell}\to z_{t+1,\ell})；
- 观测路径差异
  \(\|T_{t,\ell+1}(z_{t,\ell+1})-D_{t+1,\ell}(z_{t+1,\ell})\|\)；
- 每个 cell 的复特征相位、proper polar rotation、方向反转率、奇异值有效秩、条件数与 Henrici 非正规度；
- plaquette 线性非对易度
  \(\|D_{t,\ell}T_{t,\ell+1}-T_{t,\ell}D_{t+1,\ell}\|_F\)。

复用同一轨迹的匹配对先合并为连通分量，再整组进入训练或测试，避免泄漏。旋转指标将 det < 0 的反射单列，避免把反射误报为旋转。

## 证据边界

本地 `geometry_audit.npz` 只有标量几何摘要，并非 residual-stream hidden states。因此 `results_layer_time/` 是派生坐标中的管线/代理实验，不能证明模型隐藏态发生旋转。真正的机制实验需要加载原模型与 KV context，通过 JVP 得到投影 Jacobian

\[
\widetilde J_{t,\ell}=B_{out}^{\top}J_{t,\ell}B_{in},
\]

再复用相同的谱与 plaquette 分析。`project_jvp_operator` 已提供这个投影接口。

## 当前 pilot 结果

固定 seed 17、5 折、offsets `[-1,0]`：

| 尺度 | 匹配对 | Plaquette AUROC (95% CI) | 相对径向 AUROC 增量 (95% CI) |
|---|---:|---:|---:|
| step | 108 | 0.546 [0.444, 0.639] | 0.037 [-0.083, 0.167] |
| token | 170 | 0.529 [0.453, 0.606] | 0.059 [-0.041, 0.165] |

两个尺度都没有通过预注册成功条件。当前结论是“实现可运行，但派生几何数据不支持二维旋转/非对易与首错相关的主张”。

## 复现

```powershell
$env:PYTHONPATH='src'
$taskPython='D:\Apps\Program Files\anaconda3\python.exe'

& $taskPython -m pytest -q

& $taskPython -m functional_divergence.layer_time_experiment `
  '..\outputs\first_error_geometry\full_gsm8k\step\geometry_audit.npz' `
  '..\outputs\first_error_geometry\full_gsm8k\token\geometry_audit.npz' `
  --output-dir results_layer_time --offsets=-1,0 `
  --rank 2 --folds 5 --bootstrap 2000 --seed 17 --ridge-alpha 1.0
```

主要产物：

- `results_layer_time/results.json`：统计结果、逐 cell 动态谱和 plaquette 场；
- `results_layer_time/pair_scores.csv`：逐匹配对的 out-of-fold 分数；
- `results_layer_time/metric_comparison.png`：径向、单轴与联合指标比较；
- `refine-logs/EXPERIMENT_PLAN.md`：claims、失败条件和 raw-state 升级路线；
- `refine-logs/EXPERIMENT_RESULTS.md`：结果解读与边界。
