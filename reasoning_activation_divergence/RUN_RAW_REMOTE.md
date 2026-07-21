# 远端真实残差流实验

## 数据范围

主实验读取：

```text
data/exact/processbench_observer_llama31_full/<subset>/selected/trace.raw_residual_stream.npz
```

四个 `<subset>` 为 `gsm8k`、`math`、`olympiadbench`、`omnimath`。程序只保留
`response_generator` 匹配 `llama3.1-8b` 的记录，并对标签、token ranges、shard
路径使用同一个行掩码。manifest 必须显式声明
`response_token_state_snapshot_kind=raw_residual_stream`，否则立即失败。

当前 selected 规模为：GSM8K 400、MATH 1000、OlympiadBench 1000、OmniMath
1000；其中 Llama-3.1-8B generator cohort 分别为 61、139、164、162，共 526 条。

## 为什么安装过 sklearn 仍可能报缺包

发布包名是 `scikit-learn`，Python 导入名是 `sklearn`。conda 安装会持久写入
对应环境的 `site-packages`，不会在退出 shell 后消失。报错几乎总是因为运行实验的
`python` 不是安装包时使用的 `research` 环境解释器。

先激活并验证：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

which python
python -c 'import sys, sklearn; print(sys.executable); print(sklearn.__version__); print(sklearn.__file__)'
python -m pip show scikit-learn
```

`sys.executable` 应指向类似 `.../anaconda3/envs/research/bin/python`，而
`sklearn.__file__` 应位于同一个环境目录。始终使用 `python -m pip`，避免裸
`pip` 与 `python` 来自不同环境。

若该环境确实没有项目依赖，在已激活的 `research` 环境中执行：

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
python -m pip install -e './reasoning_activation_divergence[test]'
```

这会安装真实实现所需的 `scikit-learn` 和 `tqdm`。项目没有任何缺包退化实现。

## 前台运行全部四个子数据集

```bash
cd /share/home/tm902089733300000/a903202310/lys/research/demo
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate research

PYTHON_BIN="$CONDA_PREFIX/bin/python" \
bash reasoning_activation_divergence/run_raw_remote.sh exact-full
```

这是普通前台进程：不使用 `screen`、`nohup` 或 `&`。终端会依次显示：

- 当前 Python、conda 环境、sklearn 版本与安装路径；
- 当前子数据集和 preflight JSON；
- matched-pair shard 加载进度条；
- cross-validation fold 进度条；
- statistics 和 artifact-writing 阶段提示；
- 最终结果 JSON。

中断终端或按 `Ctrl-C` 会停止实验。输出分别位于：

```text
outputs/raw_layer_time/exact_full/gsm8k/
outputs/raw_layer_time/exact_full/math/
outputs/raw_layer_time/exact_full/olympiadbench/
outputs/raw_layer_time/exact_full/omnimath/
```

## 先跑低成本真实数据 pilot

如果需要先验证环境和进度显示：

```bash
PYTHON_BIN="$CONDA_PREFIX/bin/python" \
bash reasoning_activation_divergence/run_raw_remote.sh exact-pilot
```

pilot 仍来自 full audited manifest 中的真实 Llama cohort，只限制最多 20 个匹配对；
它不是代理指标数据，也不是另一套一类 pilot manifest。
