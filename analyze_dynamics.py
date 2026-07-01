#!/usr/bin/env python3
"""动态监测分析：step-level几何轨迹分析

分析内容：
1. Step-wise几何轨迹（kappa/eff_rank/entropy随step变化）
2. 相变检测（特征突然变化）
3. 错误step前的预警信号
4. 因果z-score（online触发）
"""

import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
from scipy import stats
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
import json
import argparse

HIDDEN_LAYERS = [10, 14, 18, 22]


@dataclass
class StepTrajectory:
    """单条链的step轨迹"""
    chain_id: int
    is_correct: bool
    gold_error_step: Optional[int] = None
    step_kappa: Dict[int, List[float]] = field(default_factory=dict)  # {layer: [kappa]}
    step_eff_rank: Dict[int, List[float]] = field(default_factory=dict)
    step_entropy: Dict[int, List[float]] = field(default_factory=dict)
    step_ids: List[int] = field(default_factory=list)


def causal_z_score(seq: np.ndarray, sd_floor: float = 0.01, clip: float = 5.0) -> np.ndarray:
    """因果z-score：z[t] = (s[t]-mean(s[:t]))/std(s[:t])

    只使用历史数据，满足online约束
    """
    s = np.asarray(seq, float)
    T = len(s)
    z = np.full(T, np.nan)

    for t in range(2, T):
        history = s[:t]
        history = history[np.isfinite(history)]

        if len(history) >= 2 and np.isfinite(s[t]):
            std = max(history.std(), sd_floor)
            z[t] = np.clip((s[t] - history.mean()) / std, -clip, clip)

    return z


def detect_phase_transition(seq: np.ndarray, threshold: float = 2.0) -> List[int]:
    """检测相变：z-score超过阈值"""
    z = causal_z_score(seq)
    return list(np.where(np.abs(z) > threshold)[0])


def load_trajectories_with_dynamics(cache_dir: Path, npz_path: str) -> Tuple[List[StepTrajectory], Dict]:
    """加载轨迹并提取动态特征"""
    # 加载NPZ获取gold_error_step
    data = np.load(npz_path, allow_pickle=True)
    gold_error_steps = data.get('gold_error_step', None)

    trajectories = []
    cache_files = sorted(cache_dir.glob("chain_*.pkl"),
                         key=lambda p: int(p.stem.split('_')[1]))

    print(f"Loading {len(cache_files)} trajectories with dynamics...")
    for cf in tqdm(cache_files):
        try:
            with open(cf, 'rb') as f:
                traj = pickle.load(f)

            chain_id = traj.chain_id

            # 提取每层的step序列
            step_kappa = {}
            step_eff_rank = {}
            step_entropy = {}
            step_ids = []

            # 获取step ID列表
            if traj.has_layer(14):
                geoms = traj.get_geometry_sequence(14)
                step_ids = [g.step_id for g in geoms]

            for layer in HIDDEN_LAYERS:
                if traj.has_layer(layer):
                    geoms = traj.get_geometry_sequence(layer)
                    step_kappa[layer] = [g.kappa for g in geoms]
                    step_eff_rank[layer] = [g.eff_rank for g in geoms]
                    step_entropy[layer] = [g.spectral_entropy for g in geoms]

            # 获取gold_error_step
            gold_err = None
            if gold_error_steps is not None:
                try:
                    ges = gold_error_steps[chain_id]
                    if ges is not None and ges >= 0:
                        gold_err = int(ges)
                except:
                    pass

            trajectories.append(StepTrajectory(
                chain_id=chain_id,
                is_correct=traj.is_correct,
                gold_error_step=gold_err,
                step_kappa=step_kappa,
                step_eff_rank=step_eff_rank,
                step_entropy=step_entropy,
                step_ids=step_ids,
            ))

        except Exception as e:
            print(f"Failed to load {cf}: {e}")

    metadata = {
        'n_trajectories': len(trajectories),
        'n_correct': sum(1 for t in trajectories if t.is_correct),
        'n_error': sum(1 for t in trajectories if not t.is_correct),
        'n_with_gold_error': sum(1 for t in trajectories if t.gold_error_step is not None),
    }

    return trajectories, metadata


def analyze_trajectory_dynamics(trajectories: List[StepTrajectory],
                                 layer: int = 14) -> Dict:
    """分析轨迹动态特征"""

    results = {
        'layer': layer,
        'kappa_dynamics': {},
        'eff_rank_dynamics': {},
        'entropy_dynamics': {},
        'error_prediction': {},
    }

    # 收集所有轨迹
    correct_kappa_traj = []
    error_kappa_traj = []

    for traj in trajectories:
        if layer not in traj.step_kappa:
            continue

        kappa_seq = np.array(traj.step_kappa[layer])
        if len(kappa_seq) < 3:
            continue

        if traj.is_correct:
            correct_kappa_traj.append(kappa_seq)
        else:
            error_kappa_traj.append(kappa_seq)

    # Kappa动态统计
    if correct_kappa_traj and error_kappa_traj:
        # 计算每条轨迹的z-score
        correct_z_triggers = []
        error_z_triggers = []

        for seq in correct_kappa_traj:
            z = causal_z_score(seq)
            triggers = detect_phase_transition(z, threshold=2.0)
            correct_z_triggers.append(len(triggers))

        for seq in error_kappa_traj:
            z = causal_z_score(seq)
            triggers = detect_phase_transition(z, threshold=2.0)
            error_z_triggers.append(len(triggers))

        results['kappa_dynamics'] = {
            'n_correct_traj': len(correct_kappa_traj),
            'n_error_traj': len(error_kappa_traj),
            'correct_mean_triggers': float(np.mean(correct_z_triggers)) if correct_z_triggers else 0,
            'error_mean_triggers': float(np.mean(error_z_triggers)) if error_z_triggers else 0,
            'correct_has_triggers': sum(1 for t in correct_z_triggers if t > 0),
            'error_has_triggers': sum(1 for t in error_z_triggers if t > 0),
        }

    # 错误预测分析
    error_with_gold = [t for t in trajectories if not t.is_correct and t.gold_error_step is not None]
    if error_with_gold and layer in trajectories[0].step_kappa:
        # 分析gold_error_step前的特征变化
        pre_error_kappa_drops = []

        for traj in error_with_gold:
            if layer not in traj.step_kappa:
                continue

            kappa_seq = traj.step_kappa[layer]
            err_step = traj.gold_error_step

            if err_step > 0 and err_step < len(kappa_seq):
                # 检查错误step前的kappa是否下降
                pre_err = kappa_seq[err_step - 1]
                at_err = kappa_seq[err_step]

                if np.isfinite(pre_err) and np.isfinite(at_err):
                    pre_error_kappa_drops.append(at_err - pre_err)  # 负值表示下降

        if pre_error_kappa_drops:
            results['error_prediction'] = {
                'n_analyzed': len(pre_error_kappa_drops),
                'mean_pre_error_change': float(np.mean(pre_error_kappa_drops)),
                'pct_kappa_drop': float(np.sum(1 for x in pre_error_kappa_drops if x < 0) / len(pre_error_kappa_drops) * 100),
                'mean_drop_magnitude': float(np.mean([abs(x) for x in pre_error_kappa_drops if x < 0])) if any(x < 0 for x in pre_error_kappa_drops) else 0,
            }

    return results


def print_dynamics_report(all_results: List[Dict], metadata: Dict):
    """打印动态分析报告"""
    print("\n" + "="*80)
    print("动态监测分析报告")
    print("="*80 + "\n")

    print(f"总轨迹数: {metadata['n_trajectories']}")
    print(f"正确: {metadata['n_correct']}, 错误: {metadata['n_error']}")
    print(f"有gold_error_step标注: {metadata['n_with_gold_error']}")

    print("\n" + "-"*80)
    print("Kappa动态分析（相变检测）")
    print("-"*80 + "\n")

    for res in all_results:
        layer = res['layer']
        dyn = res.get('kappa_dynamics', {})

        if dyn:
            print(f"Layer {layer}:")
            print(f"  正确轨迹平均触发次数: {dyn['correct_mean_triggers']:.2f}")
            print(f"  错误轨迹平均触发次数: {dyn['error_mean_triggers']:.2f}")
            print(f"  正确轨迹有触发: {dyn['correct_has_triggers']}/{dyn['n_correct_traj']} ({dyn['correct_has_triggers']/dyn['n_correct_traj']*100:.1f}%)")
            print(f"  错误轨迹有触发: {dyn['error_has_triggers']}/{dyn['n_error_traj']} ({dyn['error_has_triggers']/dyn['n_error_traj']*100:.1f}%)")

            # 判断错误轨迹是否更容易触发
            if dyn['error_mean_triggers'] > dyn['correct_mean_triggers'] * 1.5:
                print(f"  ✓ 错误轨迹显著更多相变")
            else:
                print(f"  ✗ 相变频率差异不明显")

    print("\n" + "-"*80)
    print("错误预测分析（Gold Error Step前的特征变化）")
    print("-"*80 + "\n")

    for res in all_results:
        pred = res.get('error_prediction', {})

        if pred and pred['n_analyzed'] > 0:
            print(f"Layer {res['layer']}:")
            print(f"  分析样本数: {pred['n_analyzed']}")
            print(f"  错误step前kappa平均变化: {pred['mean_pre_error_change']:.4f}")
            print(f"  kappa下降比例: {pred['pct_kappa_drop']:.1f}%")
            print(f"  平均下降幅度: {pred['mean_drop_magnitude']:.4f}")

            if pred['pct_kappa_drop'] > 60:
                print(f"  ✓ 大多数错误前有kappa下降信号")
            else:
                print(f"  ✗ kappa下降信号不明显")


def print_online_monitoring_example(trajectories: List[StepTrajectory], layer: int = 14):
    """打印在线监测示例"""
    print("\n" + "="*80)
    print("在线监测示例（Layer 14）")
    print("="*80 + "\n")

    # 找一个错误轨迹作为示例
    error_traj = None
    for traj in trajectories:
        if not traj.is_correct and layer in traj.step_kappa and len(traj.step_kappa[layer]) >= 3:
            error_traj = traj
            break

    if error_traj is None:
        print("没有找到合适的错误轨迹示例")
        return

    kappa_seq = np.array(error_traj.step_kappa[layer])
    z_scores = causal_z_score(kappa_seq)

    print(f"Chain ID: {error_traj.chain_id}")
    print(f"Gold Error Step: {error_traj.gold_error_step}")
    print(f"\n{'Step':<8} {'Kappa':<12} {'Z-Score':<12} {'Status':<20}")
    print("-" * 60)

    for i, (k, z) in enumerate(zip(kappa_seq, z_scores)):
        status = "Normal"
        if np.isfinite(z) and abs(z) > 2.0:
            status = "⚠️ PHASE CHANGE"
        if error_traj.gold_error_step == i:
            status = "❌ ERROR STEP"

        print(f"{i:<8} {k:<12.4f} {z if np.isfinite(z) else float('nan'):<12.2f} {status:<20}")

    # 打印触发阈值建议
    print("\n" + "-"*60)
    print("在线监测配置建议:")
    print(f"  建议阈值: z < -1.5 (kappa下降)")
    print(f"  响应时间: 提前 {error_traj.gold_error_step - np.where(z_scores < -1.5)[0][-1] if len(np.where(z_scores < -1.5)[0]) > 0 else 'N/A'} steps")


def main():
    parser = argparse.ArgumentParser(description='动态监测分析')
    parser.add_argument('--cache-dir', type=str,
                        default='/gz-data/research/demo/data/hidden/cache/omnimath')
    parser.add_argument('--npz-path', type=str,
                        default='/gz-data/research/demo/data/features/full_omnimath.npz')
    parser.add_argument('--output', type=str,
                        default='/gz-data/research/demo/data/results/omnimath_dynamics.json')

    args = parser.parse_args()

    cache_dir = Path(args.cache_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 加载轨迹
    trajectories, metadata = load_trajectories_with_dynamics(cache_dir, args.npz_path)
    print(f"成功加载 {len(trajectories)} 条轨迹")

    # 分析每一层
    all_results = []
    for layer in HIDDEN_LAYERS:
        print(f"\n分析 Layer {layer} 动态...")
        results = analyze_trajectory_dynamics(trajectories, layer=layer)
        all_results.append(results)

    # 打印报告
    print_dynamics_report(all_results, metadata)
    print_online_monitoring_example(trajectories)

    # 保存JSON
    output = {
        'metadata': metadata,
        'layer_results': all_results,
    }
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n结果已保存到: {output_path}")


if __name__ == "__main__":
    main()
