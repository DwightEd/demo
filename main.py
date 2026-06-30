#!/usr/bin/env python3
"""主入口：Trajectory of Thought 几何相变检测

Usage:
    python main.py \\
        --npz_path /gz-data/research/demo/data/features/full_omnimath.npz \\
        --hidden_dir /gz-data/research/demo/data/hidden/omnimath/ \\
        --output_dir ./trajectory_results \\
        --layers 14 \\
        --n_bootstrap 5000
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime
import numpy as np

from data_loading import (
    load_all_trajectories,
    get_trajectory_with_min_steps,
    print_trajectory_stats
)
try:
    from data_loading_fast import load_all_trajectories_fast
    HAS_FAST = True
except ImportError:
    HAS_FAST = False

try:
    from data_loading_minimal import load_all_trajectories_minimal
    HAS_MINIMAL = True
except ImportError:
    HAS_MINIMAL = False

try:
    from data_loading_cache import load_all_trajectories_cached, clear_cache
    HAS_CACHE = True
except ImportError:
    HAS_CACHE = False

from trajectory_geometry import (
    compute_all_metrics,
    print_metrics_summary,
    TrajectoryMetrics
)
from phase_transition import (
    batch_detect_lockin,
    batch_detect_decay,
    compute_lockin_statistics,
    compute_decay_statistics,
    print_detection_summary
)
from validation import (
    run_all_validations,
    print_validation_summary,
    save_validation_results
)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Trajectory of Thought: Geometric Phase Transitions in Multi-Step Reasoning'
    )

    # 数据路径
    parser.add_argument('--npz_path', type=str, required=True,
                       help='Path to full_*.npz file')
    parser.add_argument('--hidden_dir', type=str, required=True,
                       help='Path to hidden shards directory')
    parser.add_argument('--output_dir', type=str, default='./trajectory_results',
                       help='Output directory for results')

    # 分析参数
    parser.add_argument('--layers', type=int, nargs='+', default=[14],
                       help='Layers to analyze (default: [14])')
    parser.add_argument('--min_steps', type=int, default=3,
                       help='Minimum steps required per trajectory (default: 3)')
    parser.add_argument('--n_bootstrap', type=int, default=5000,
                       help='Number of bootstrap iterations (default: 5000)')
    parser.add_argument('--n_top_components', type=int, default=10,
                       help='Number of principal components to compute (default: 10)')
    parser.add_argument('--n_workers', type=int, default=16,
                       help='Number of parallel workers for loading (default: 16, 0 for serial)')
    parser.add_argument('--mode', type=str, default='minimal',
                       choices=['minimal', 'fast', 'full', 'cache'],
                       help='Loading mode: minimal (fastest), fast (sparse eig), full (exact), cache (persistent)')
    parser.add_argument('--clear_cache', action='store_true',
                       help='Clear cache and recompute')

    # 检测参数
    parser.add_argument('--drop_threshold', type=float, default=0.15,
                       help='Drop threshold for shallow lock-in detection (default: 0.15)')
    parser.add_argument('--stability_threshold', type=float, default=0.5,
                       help='Stability threshold for deep decay detection (default: 0.5)')
    parser.add_argument('--entropy_threshold', type=float, default=0.7,
                       help='Entropy threshold for deep decay detection (default: 0.7)')

    # 其他
    parser.add_argument('--verbose', action='store_true', default=True,
                       help='Show progress bars (default: True)')
    parser.add_argument('--skip_validation', action='store_true',
                       help='Skip H1-H4 validation')
    parser.add_argument('--detect_method', type=str, default='standard',
                       choices=['standard', 'adaptive'],
                       help='Detection method: standard or adaptive (default: standard)')

    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 80)
    print("Trajectory of Thought: Geometric Phase Transitions")
    print("=" * 80)
    print(f"NPZ: {args.npz_path}")
    print(f"Hidden dir: {args.hidden_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Layers: {args.layers}")
    print(f"Min steps: {args.min_steps}")
    print("=" * 80)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    print("\n[1/5] Loading trajectories...")
    print(f"Mode: {args.mode}")

    if args.clear_cache and HAS_CACHE:
        clear_cache(args.npz_path, args.hidden_dir)
        print("Cache cleared, recomputing...")

    if args.mode == 'cache' and HAS_CACHE:
        print(f"Using CACHED loading (persistent storage)...")
        trajectories, metadata = load_all_trajectories_cached(
            npz_path=args.npz_path,
            hidden_dir=args.hidden_dir,
            force_recompute=args.clear_cache,
            verbose=args.verbose,
        )
    elif args.mode == 'minimal' and HAS_MINIMAL and args.n_workers > 0:
        print(f"Using MINIMAL loading with {args.n_workers} workers (fastest, approximate)...")
        trajectories, metadata = load_all_trajectories_minimal(
            npz_path=args.npz_path,
            hidden_dir=args.hidden_dir,
            n_workers=args.n_workers,
            verbose=args.verbose,
        )
    elif args.mode == 'fast' and HAS_FAST and args.n_workers > 0:
        print(f"Using FAST loading with {args.n_workers} workers (sparse eigendecomposition)...")
        trajectories, metadata = load_all_trajectories_fast(
            npz_path=args.npz_path,
            hidden_dir=args.hidden_dir,
            n_top=args.n_top_components + 5,
            n_workers=args.n_workers,
            verbose=args.verbose,
        )
    else:
        print(f"Using FULL loading (exact eigendecomposition)...")
        trajectories, metadata = load_all_trajectories(
            npz_path=args.npz_path,
            hidden_dir=args.hidden_dir,
            n_top_components=args.n_top_components,
            verbose=args.verbose,
        )

    print_trajectory_stats(trajectories, metadata, layer=args.layers[0])

    # 过滤轨迹
    print(f"\n[2/5] Filtering trajectories (min_steps={args.min_steps})...")
    filtered = get_trajectory_with_min_steps(
        trajectories,
        min_steps=args.min_steps,
        layer=args.layers[0]
    )
    print(f"Filtered to {len(filtered)} trajectories")

    if len(filtered) == 0:
        print("ERROR: No trajectories left after filtering!")
        sys.exit(1)

    # 为每一层运行分析
    all_results = {}

    for layer in args.layers:
        print(f"\n{'=' * 80}")
        print(f"Analyzing Layer {layer}")
        print(f"{'=' * 80}")

        # [3/5] 计算轨迹指标
        print(f"\n[3/5] Computing trajectory metrics (L{layer})...")
        metrics_list = compute_all_metrics(filtered, layer=layer, verbose=args.verbose)
        is_correct_list = [t.is_correct for t in filtered]
        print_metrics_summary(metrics_list, is_correct_list)

        # [4/5] 相变检测
        print(f"\n[4/5] Detecting phase transitions (L{layer})...")
        lockin_results = batch_detect_lockin(
            filtered,
            layer=layer,
            method=args.detect_method,
            verbose=args.verbose
        )
        decay_results = batch_detect_decay(
            filtered,
            layer=layer,
            verbose=args.verbose
        )

        lockin_stats = compute_lockin_statistics(lockin_results, is_correct_list)
        decay_stats = compute_decay_statistics(decay_results, is_correct_list)
        print_detection_summary(lockin_stats, decay_stats)

        # 保存检测统计
        layer_results = {
            'metadata': {
                'layer': layer,
                'n_trajectories': len(filtered),
                'n_correct': sum(is_correct_list),
                'n_error': sum(~np.array(is_correct_list)),
            },
            'lockin_stats': lockin_stats,
            'decay_stats': decay_stats,
        }
        all_results[f'L{layer}'] = layer_results

    # [5/5] 验证实验
    if not args.skip_validation:
        print(f"\n[5/5] Running validation experiments...")
        validation_results = run_all_validations(
            filtered,
            layers=args.layers,
            n_bootstrap=args.n_bootstrap,
            verbose=args.verbose,
        )
        print_validation_summary(validation_results)
        all_results['validation'] = {k: v.__dict__ for k, v in validation_results.items()}
    else:
        print("\n[5/5] Skipping validation experiments")
        all_results['validation'] = {}

    # 保存结果
    output_path = output_dir / f"results_{metadata['subset']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=lambda x: float(x) if isinstance(x, (np.float32, np.float64)) else x)

    print(f"\n{'=' * 80}")
    print(f"Results saved to: {output_path}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
