#!/bin/bash
# 完整GPU加速几何特征计算流程
# 从安装到执行到汇报

set -e  # 遇到错误就退出

echo "========================================"
echo "GPU加速几何特征计算 - 完整流程"
echo "========================================"

# 1. 安装CuPy（使用清华镜像）
echo ""
echo "[1/6] 安装CuPy..."
pip install cupy-cuda12x -i https://pypi.tuna.tsinghua.edu.cn/simple --timeout 1000 || {
    echo "CuPy安装失败，将使用CPU版本"
    USE_CPU=1
}

if [ -z "$USE_CPU" ]; then
    # 验证CuPy安装
    python -c "import cupy; print('CuPy版本:', cupy.__version__); print('CUDA可用:', cupy.cuda.is_available())"
fi

# 2. 进入工作目录
echo ""
echo "[2/6] 进入工作目录..."
cd /gz-data/research/demo/

# 3. 拉取最新代码
echo ""
echo "[3/6] 拉取最新代码..."
git pull origin main

# 4. 清除旧缓存（强制重新计算）
echo ""
echo "[4/6] 清除旧缓存..."
python data_loading_gpu.py --clear

# 5. 运行GPU加速计算
echo ""
echo "[5/6] 开始GPU加速计算..."
echo "预计时间: 1-2小时（取决于数据量和GPU性能）"

START_TIME=$(date +%s)

if [ -z "$USE_CPU" ]; then
    # GPU版本
    python data_loading_gpu.py --force --workers 8
else
    # CPU版本（降级）
    echo "使用CPU优化版本..."
    python data_loading_fast.py
fi

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
echo "计算完成! 用时: $((ELAPSED / 60)) 分 $((ELAPSED % 60)) 秒"

# 6. 运行分析和统计
echo ""
echo "[6/6] 生成计算报告..."
echo "========================================"
echo "计算报告"
echo "========================================"

python << 'EOF'
import numpy as np
from pathlib import Path

# 加载缓存数据检查结果
cache_dir = Path("../data/cache/omnimath")

if cache_dir.exists():
    import pickle
    import os

    # 统计缓存文件
    cache_files = list(cache_dir.glob("chain_*.pkl"))
    print(f"✓ 缓存文件数量: {len(cache_files)}")

    # 抽样检查
    if cache_files:
        sample_file = cache_files[0]
        with open(sample_file, 'rb') as f:
            traj = pickle.load(f)

        print(f"✓ 示例轨迹:")
        print(f"  - chain_id: {traj.chain_id}")
        print(f"  - is_correct: {traj.is_correct}")
        print(f"  - n_steps: {traj.n_steps}")

        if traj.has_layer(14):
            geoms = traj.get_geometry_sequence(14)
            if geoms:
                g = geoms[0]
                print(f"  - 示例几何特征 (layer=14, step=0):")
                print(f"    kappa: {g.kappa:.4f}")
                print(f"    eff_rank: {g.eff_rank:.2f}")
                print(f"    spectral_entropy: {g.spectral_entropy:.4f}")
                print(f"    eigenvalues: {g.eigenvalues[:3]}")

    print("\n✓ 所有缓存文件正常，计算成功!")
else:
    print("✗ 缓存目录不存在，计算可能失败")
EOF

echo ""
echo "========================================"
echo "完整流程执行完毕!"
echo "========================================"
