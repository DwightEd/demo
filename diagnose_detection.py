#!/usr/bin/env python3
"""诊断H2/H3检测失败的原因

检查：
1. 每个链有多少步骤
2. 可用的层有哪些
3. 实际的coherence和stability值分布
4. 检测条件为什么没有触发
"""

import numpy as np
from trajectory_phase_transition import (
    load_full_npz,
    compute_trajectory_metrics,
    detect_shallow_lockin_trajectory,
    detect_deep_decay_trajectory,
    geometric_sim,
)

npz_path = "/gz-data/research/demo/data/features/full_omnimath.npz"

print("=" * 80)
print("Diagnosing Detection Failure")
print("=" * 80)

trajectories, metadata = load_full_npz(npz_path, use_hidden_shards=False)

print(f"\nMetadata:")
print(f"  Subset: {metadata['subset']}")
print(f"  Total chains: {metadata['n_chains']}")
print(f"  Correct: {metadata['n_correct']}")
print(f"  Error: {metadata['n_error']}")
print(f"  Available layers: {metadata['sv_layers']}")

# 统计步骤数
step_counts = []
layer_coverage = {}
for traj in trajectories[:100]:  # 检查前100个
    n_steps = len(set(s.step_id for s in traj.steps))
    step_counts.append(n_steps)

    for layer in metadata['sv_layers']:
        if traj.has_layer(layer):
            layer_coverage[layer] = layer_coverage.get(layer, 0) + 1

print(f"\nStep count distribution (first 100 chains):")
print(f"  Min: {min(step_counts)}, Max: {max(step_counts)}, Mean: {np.mean(step_counts):.1f}")

print(f"\nLayer coverage (how many chains have data for each layer):")
for layer in sorted(layer_coverage.keys()):
    print(f"  Layer {layer}: {layer_coverage[layer]}/100")

# 检查layer 14的详细情况
print("\n" + "=" * 80)
print("Layer 14 Detailed Analysis")
print("=" * 80)

layer = 14
traj_with_layer = [t for t in trajectories if t.has_layer(layer)]
print(f"\nChains with layer {layer}: {len(traj_with_layer)}")

if len(traj_with_layer) == 0:
    print("ERROR: No chains have layer 14!")
    exit(1)

# 检查步骤数分布
step_counts_l14 = []
for traj in traj_with_layer[:50]:
    geom_seq = traj.get_geometry_sequence(layer)
    step_counts_l14.append(len(geom_seq))

print(f"Step counts at layer 14 (first 50 chains):")
print(f"  Min: {min(step_counts_l14)}, Max: {max(step_counts_l14)}, Mean: {np.mean(step_counts_l14):.1f}")

# 计算几个样本的详细指标
print("\n" + "=" * 80)
print("Sample Chain Analysis (5 error chains)")
print("=" * 80)

error_trajs = [t for t in traj_with_layer if not t.is_correct][:5]

for i, traj in enumerate(error_trajs):
    geom_seq = traj.get_geometry_sequence(layer)
    print(f"\n--- Error chain {i} (ID={traj.chain_id}, steps={len(geom_seq)}) ---")

    if len(geom_seq) < 2:
        print("  SKIP: < 2 steps")
        continue

    # 打印每个步骤的几何特征
    for j, geom in enumerate(geom_seq):
        print(f"  Step {j}: κ={geom.kappa:.3f}, eff_R={geom.eff_rank:.2f}, "
              f"spectrum=[{', '.join(f'{x:.2f}' for x in geom.spectrum[:3])}]")

    # 计算coherence profile
    if len(geom_seq) >= 3:
        coherence_profile = []
        for k in range(1, len(geom_seq)):
            sims = []
            for j in range(k):
                sim = geometric_sim(geom_seq[k], geom_seq[j])
                if np.isfinite(sim):
                    sims.append(sim)
            if sims:
                coherence_profile.append(np.mean(sims))

        print(f"  Coherence profile: {[f'{x:.3f}' for x in coherence_profile]}")

        if coherence_profile:
            print(f"    Min: {min(coherence_profile):.3f}, Max: {max(coherence_profile):.3f}")
            print(f"    Range: {max(coherence_profile) - min(coherence_profile):.3f}")

    # 运行检测
    lockin_result = detect_shallow_lockin_trajectory(traj, layers_to_check=[layer])
    print(f"  Lock-in detection: {lockin_result['detected']}")
    if lockin_result['detected']:
        print(f"    Layer: {lockin_result['layer']}")
        print(f"    Step: {lockin_result['lockin_step']}")
        print(f"    Drop: {lockin_result['drop_magnitude']:.3f}")

    decay_result = detect_deep_decay_trajectory(traj, layers_to_check=[layer])
    print(f"  Decay detection: {decay_result['detected']}")
    if decay_result['detected']:
        print(f"    Layer: {decay_result['layer']}")
        print(f"    Stability: {decay_result['stability']:.3f}")
        print(f"    Late entropy: {decay_result['late_entropy']:.3f}")
    else:
        if 'stability_values' in decay_result and decay_result['stability_values']:
            print(f"    Stability values: {[f'{x:.3f}' for x in decay_result['stability_values']]}")

# 统计所有链的检测情况
print("\n" + "=" * 80)
print("Detection Statistics (All chains)")
print("=" * 80)

lockin_count = 0
decay_count = 0
for traj in trajectories:
    lockin_result = detect_shallow_lockin_trajectory(traj, layers_to_check=[layer])
    if lockin_result['detected']:
        lockin_count += 1

    decay_result = detect_deep_decay_trajectory(traj, layers_to_check=[layer])
    if decay_result['detected']:
        decay_count += 1

print(f"\nTotal chains: {len(trajectories)}")
print(f"Shallow Lock-in detected: {lockin_count} ({lockin_count/len(trajectories)*100:.1f}%)")
print(f"Deep Decay detected: {decay_count} ({decay_count/len(trajectories)*100:.1f}%)")

# 检查正确vs错误的检测率
correct_trajs = [t for t in trajectories if t.is_correct and t.has_layer(layer)]
error_trajs = [t for t in trajectories if not t.is_correct and t.has_layer(layer)]

lockin_correct = sum(1 for t in correct_trajs if detect_shallow_lockin_trajectory(t, layers_to_check=[layer])['detected'])
lockin_error = sum(1 for t in error_trajs if detect_shallow_lockin_trajectory(t, layers_to_check=[layer])['detected'])

decay_correct = sum(1 for t in correct_trajs if detect_deep_decay_trajectory(t, layers_to_check=[layer])['detected'])
decay_error = sum(1 for t in error_trajs if detect_deep_decay_trajectory(t, layers_to_check=[layer])['detected'])

print(f"\nBy correctness:")
print(f"  Correct ({len(correct_trajs)}): Lock-in={lockin_correct}, Decay={decay_correct}")
print(f"  Error ({len(error_trajs)}): Lock-in={lockin_error}, Decay={decay_error}")

print("\n" + "=" * 80)
