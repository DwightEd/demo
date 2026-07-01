#!/bin/bash
# OMniMath结果分析完整流程

set -e

echo "========================================"
echo "OMniMath 几何特征分析"
echo "========================================"

# 1. 拉取最新代码
echo ""
echo "[1/5] 拉取最新代码..."
git pull origin main

# 2. 检查缓存状态
echo ""
echo "[2/5] 检查缓存状态..."
python check_cache.py

# 3. 运行静态分析（step-level统计）
echo ""
echo "[3/5] 运行静态分析（Cohen's d, AUC）..."
python analyze_results.py

# 4. 运行动态分析（相变检测，在线监测）
echo ""
echo "[4/5] 运行动态分析（相变检测、在线监测）..."
python analyze_dynamics.py

echo ""
echo "========================================"
echo "分析完成!"
echo "========================================"
echo ""
echo "结果文件位置:"
echo "  - 静态分析: /gz-data/research/demo/data/results/omnimath_analysis.json"
echo "  - 动态分析: /gz-data/research/demo/data/results/omnimath_dynamics.json"
echo "  - LaTeX表格: 见上方输出，直接复制到论文"
echo ""
echo "运行时参数:"
echo "  python analyze_results.py --help"
echo "  python analyze_dynamics.py --help"
