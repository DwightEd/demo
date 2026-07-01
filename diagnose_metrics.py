#!/usr/bin/env python3
"""诊断脚本：分析指标分布，帮助调整阈值"""

import numpy as np
from data_loading_cache import load_all_trajectories_cached, get_trajectory_with_min_steps
from trajectory_geometry import compute_all_metrics, extract_scalar_sequence
from phase_transition import batch_detect_lockin, batch_detect_decay
import json

def main():
    npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"
    hidden_dir = "/gz-data/research/demo/data/hidden/omnimath/"
    layer = 14

    print("Loading data...")
    trajectories, metadata = load_all_trajectories_cached(
        npz_path=npz_path,
        hidden_dir=hidden_dir,
        verbose=False,
    )

    filtered = get_trajectory_with_min_steps(trajectories, min_steps=3, layer=layer)
    print(f"Filtered to {len(filtered)} trajectories\n")

    is_correct = np.array([t.is_correct for t in filtered])
    n_correct = is_correct.sum()
    n_error = (~is_correct).sum()

    # 计算指标
    metrics_list = compute_all_metrics(filtered, layer=layer, verbose=False)

    # ========== 分析核心指标 ==========
    print("=" * 80)
    print("核心指标分布分析")
    print("=" * 80)

    for metric_name in ['smoothness', 'coherence', 'stability', 'kappa_smoothness']:
        values_correct = []
        values_error = []

        for m, ic in zip(metrics_list, is_correct):
            val = getattr(m, metric_name, np.nan)
            if np.isfinite(val):
                if ic:
                    values_correct.append(val)
                else:
                    values_error.append(val)

        if values_correct and values_error:
            vc = np.array(values_correct)
            ve = np.array(values_error)

            print(f"\n{metric_name}:")
            print(f"  Correct (n={len(vc)}): mean={vc.mean():.4f}, std={vc.std():.4f}, "
                  f"min={vc.min():.4f}, max={vc.max():.4f}")
            print(f"  Error   (n={len(ve)}): mean={ve.mean():.4f}, std={ve.std():.4f}, "
                  f"min={ve.min():.4f}, max={ve.max():.4f}")

            # Cohen's d
            pooled_std = np.sqrt(((len(vc)-1)*vc.var() + (len(ve)-1)*ve.var()) / (len(vc)+len(ve)-2))
            d = (vc.mean() - ve.mean()) / pooled_std if pooled_std > 0 else 0
            print(f"  Cohen's d: {d:.3f}")

    # ========== 分析标量序列 ==========
    print("\n" + "=" * 80)
    print("标量序列分析 (各轨迹内)")
    print("=" * 80)

    for scalar_name in ['kappa', 'eff_rank', 'spectral_entropy']:
        all_values_correct = []
        all_values_error = []

        for traj, ic in zip(filtered, is_correct):
            geom_seq = traj.get_geometry_sequence(layer)
            if scalar_name == 'kappa':
                vals = [g.kappa for g in geom_seq if np.isfinite(g.kappa)]
            elif scalar_name == 'eff_rank':
                vals = [g.eff_rank for g in geom_seq if np.isfinite(g.eff_rank)]
            else:
                vals = [g.spectral_entropy for g in geom_seq if np.isfinite(g.spectral_entropy)]

            if vals:
                if ic:
                    all_values_correct.extend(vals)
                else:
                    all_values_error.extend(vals)

        if all_values_correct and all_values_error:
            vc = np.array(all_values_correct)
            ve = np.array(all_values_error)

            print(f"\n{scalar_name}:")
            print(f"  Correct (n={len(vc)}): mean={vc.mean():.4f}, std={vc.std():.4f}, "
                  f"median={np.median(vc):.4f}")
            print(f"  Error   (n={len(ve)}): mean={ve.mean():.4f}, std={ve.std():.4f}, "
                  f"median={np.median(ve):.4f}")

    # ========== 分析不同阈值下的检测率 ==========
    print("\n" + "=" * 80)
    print("不同阈值下的检测率分析")
    print("=" * 80)

    lockin_results = batch_detect_lockin(filtered, layer=layer, verbose=False)
    decay_results = batch_detect_decay(filtered, layer=layer, verbose=False)

    # 收集实际值
    drop_magnitudes = [r.drop_magnitude for r in lockin_results if r.detected]
    stabilities = [r.stability for r in decay_results if np.isfinite(r.stability)]
    late_entropies = [r.late_entropy for r in decay_results if np.isfinite(r.late_entropy)]

    if drop_magnitudes:
        print(f"\nLock-in drop_magnitude (detected cases, n={len(drop_magnitudes)}):")
        print(f"  mean={np.mean(drop_magnitudes):.4f}, max={np.max(drop_magnitudes):.4f}")

    if stabilities:
        print(f"\nDecay stability (all finite values, n={len(stabilities)}):")
        print(f"  mean={np.mean(stabilities):.4f}, min={np.min(stabilities):.4f}, max={np.max(stabilities):.4f}")
        print(f"  25%: {np.percentile(stabilities, 25):.4f}")
        print(f"  50%: {np.percentile(stabilities, 50):.4f}")
        print(f"  75%: {np.percentile(stabilities, 75):.4f}")

    if late_entropies:
        print(f"\nDecay late_entropy (all finite values, n={len(late_entropies)}):")
        print(f"  mean={np.mean(late_entropies):.4f}, min={np.min(late_entropies):.4f}, max={np.max(late_entropies):.4f}")

    # ========== 建议阈值 ==========
    print("\n" + "=" * 80)
    print("建议的阈值调整")
    print("=" * 80)

    if stabilities:
        # 当前阈值是 0.5，但数据都在 0.999
        # 建议用低分位数
        p25 = np.percentile(stabilities, 25)
        p10 = np.percentile(stabilities, 10)
        print(f"\nstability_threshold 建议:")
        print(f"  当前值: 0.5")
        print(f"  25%分位数: {p25:.4f} (检出 {100*np.sum(np.array(stabilities) < p25)/len(stabilities):.1f}%)")
        print(f"  10%分位数: {p10:.4f} (检出 {100*np.sum(np.array(stabilities) < p10)/len(stabilities):.1f}%)")

    if late_entropies:
        p25 = np.percentile(late_entropies, 25)
        p10 = np.percentile(late_entropies, 10)
        print(f"\nentropy_threshold 建议:")
        print(f"  当前值: 0.7")
        print(f"  25%分位数: {p25:.4f}")
        print(f"  10%分位数: {p10:.4f}")

    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()
