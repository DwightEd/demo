#!/bin/bash
# OMniMath结果分析完整流程

set -e

echo "========================================"
echo "OMniMath 几何特征分析"
echo "========================================"

# 1. 拉取最新代码
echo ""
echo "[1/4] 拉取最新代码..."
git pull origin main

# 2. 检查缓存状态
echo ""
echo "[2/4] 检查缓存状态..."
python check_cache.py

# 3. 运行分析
echo ""
echo "[3/4] 运行统计分析..."
python analyze_results.py

echo ""
echo "========================================"
echo "分析完成!"
echo "========================================"
echo ""
echo "结果文件位置:"
echo "  - JSON: /gz-data/research/demo/data/results/omnimath_analysis.json"
echo "  - LaTeX表格: 见上方输出，直接复制到论文"
echo ""
echo "运行时参数:"
echo "  python analyze_results.py --help"
