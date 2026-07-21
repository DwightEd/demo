# Reasoning Activation Divergence

本目录现在只保留真实 response-token residual stream 实验。此前基于本地
`geometry_audit.npz` 标量指标得到的结果、图表、claims 和叙事报告已删除，
不得再作为本研究的实验证据。

## 数据边界

正式实验必须同时满足：

1. 回答由 `Meta-Llama-3.1-8B-Instruct` 生成；
2. 中间状态与该回答保持同一 manifest 行和 shard 路径；
3. manifest 声明
   `response_token_state_snapshot_kind=raw_residual_stream`；
4. 状态是中间 post-block residual depths，而不是 final-normalized output；
5. 首错和完全正确样本均存在，才能进行匹配对比较。

`--response-generator llama3.1-8b` 会使用同一个行掩码同步过滤标签、
problem id、step/token ranges、state shard path 和 token count，避免把某个
模型的回答与另一个模型的状态错配。

## 当前方法

真实状态保持为 `[sample,time,layer,hidden]`。每个交叉验证折只用训练集的
完全正确样本拟合共享 randomized-SVD 坐标系，以及 sklearn `Ridge` 深度和
token-time 仿射算子。held-out 首错/正确匹配对比较：

- 径向边变化；
- 深度算子残差；
- 时间算子残差；
- depth→time 与 time→depth 的 plaquette 路径分歧；
- 特征值相位、proper polar rotation、反射、有效秩、条件数和 Henrici
  非正规性。

这些是已保存真实 residual states 上的经验局部算子，不是 autograd
Jacobian，也不是 model-native logits Fisher。

详细方法和逐函数说明见 [REAL_RAW_RESIDUAL_METHOD.md](REAL_RAW_RESIDUAL_METHOD.md)。
远端预检与运行见 [RUN_RAW_REMOTE.md](RUN_RAW_REMOTE.md)。

## 验证

```bash
python -m pip install -e 'reasoning_activation_divergence[test]'
python -m pytest reasoning_activation_divergence/tests -q
bash reasoning_activation_divergence/run_raw_remote.sh exact-pilot
```

正式输出只应出现在仓库根目录：

```text
outputs/raw_layer_time/exact_pilot/<subset>/
outputs/raw_layer_time/exact_full/<subset>/
```
