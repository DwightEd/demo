#!/usr/bin/env python3
"""相变不稳定性指标验证

验证 compute_phase_instability_metrics 函数在区分正确/错误推理中的有效性。
使用 full_*.npz 数据中的 stepvec 和 stepcloud 字段。

数据来源 (DATA.md):
    data/features/full_gsm8k.npz, full_math.npz, full_omnimath.npz
    - stepvec: (T, 8, 4096) pooled step vectors at 8 sv-layers
    - stepcloud: (T, 33, 9) cloud features including 'resultant' (kappa)
    - is_correct_strict: correctness labels
    - problem_ids: problem identifiers

使用方法:
    python validate_phase_instability.py --dataset gsm8k --output_dir ./results

本版本相对原版的主要变更（详见各函数 docstring）:
    1. [bug修复] JSON 序列化: numpy 标量 (float32/bool_ 等) 不能被 json 模块直接
       序列化，导致脚本在计算完所有链之后、写文件的最后一步崩溃、结果全部丢失。
       现在通过 sanitize_for_json() 递归清洗 + 自定义 Encoder 双重兜底修复。
    2. [性能] effective_rank 的特征分解从 O(d^3) 优化为 O(min(T,d)^3)，
       利用 (U @ U.T) 与 (U.T @ U) 共享非零特征值这一性质。当 T<<d（本数据集
       d=4096）时，经实测可带来 4-5 个数量级的加速（详见随附验证报告）。
    3. [校验] 为每个函数补充了输入合法性检查（形状/NaN/Inf/退化向量/标签合法性/
       字段完整性等），并将原来"静默使用错误数据继续算"的几处隐患改为显式跳过
       并记录原因，运行结束时输出汇总，避免结果被无声污染。
    4. 已知问题 (combined_instability_score 中 `rank/max(rank,1)` 恒为1、
       geometric_deviation_score 的理论值域与实际值域不匹配) 按原样保留，
       不属于本次修改范围，详见函数内注释。
    5. [bug修复][致命] 标签方向: full_*.npz 中 is_correct_strict 约定为 1=correct
       (写入端 extract_features._pb_record: correct = final_answer_correct 或
       label==-1)。原实现按 0=correct 解读，所有 AUROC / Cohen's d 方向整体反转
       ——此前跑出的"全指标 AUC 0.4~0.5"实为 0.5~0.6 的镜像。已修复，并在
       加载时用 gold_error_step 做方向一致性断言。
    6. [新增] --vector_mode {raw,center,delta}: stepvec 是步内 token 云 exp-pool
       的"位置"向量 h_t (extract_features.py:376-383)，同链各步共享巨大的公共
       分量（表征各向异性 + massive activations），未中心化直接做方向统计会使
       所有链的 concentration 饱和在 1 附近、无判别力。center=减链内均值，
       delta=一阶差分 Δh_t (推荐，与 NTS 的位移几何一致)。raw 保留作对照。
    7. [新增] 长度混杂基线: 报告 AUROC(T alone) 与各指标对 T 的 Spearman 相关。
       effective_rank 上界是 min(T,d) 且错误链更长，任何指标的 AUROC 都必须
       与长度基线同表解读（项目内 FINDINGS.md 的教训: 长度是首要混杂）。
"""

from __future__ import annotations

import argparse
import json
import os
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 参与 AUROC / 统计检验的连续型指标名单（唯一数据源，避免多处重复定义后漂移）
CONTINUOUS_METRIC_KEYS = [
    'concentration',
    'effective_rank',
    'geometric_deviation_score',
    'combined_instability_score',
]

# 加载 npz 时要求必须存在的字段
REQUIRED_NPZ_KEYS = ['problem_ids', 'is_correct_strict']

# 单侧样本量低于此值时，对统计检验结果给出"功效不足"提示
MIN_SAMPLE_SIZE_WARNING = 10


# --------------------------------------------------------------------------
# JSON 序列化工具（修复 "float32 is not serializable"）
# --------------------------------------------------------------------------

class NumpyJSONEncoder(json.JSONEncoder):
    """兜底 Encoder：处理 sanitize_for_json 可能遗漏的 numpy 类型。

    主修复路径是 sanitize_for_json（递归预处理整个输出字典），这个 Encoder
    是第二道防线——万一未来新增字段忘了走 sanitize，也不至于在写文件的最后
    一步整体失败、丢掉本次运行的全部计算结果。
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            val = float(obj)
            return None if (np.isnan(val) or np.isinf(val)) else val
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def sanitize_for_json(obj: Any) -> Any:
    """递归地将对象转换为标准 JSON 可安全序列化的原生 Python 类型。

    根本原因: 本文件中的指标 (concentration / effective_rank / 各 score /
    combined_anomaly_flag 等) 都是 numpy 运算的直接产物，类型是
    np.float32/np.float64/np.bool_ 等，而不是 Python 内置的 float/bool。
    json.dump 只认识内置类型，遇到 numpy 标量会抛出
    `TypeError: Object of type float32 is not JSON serializable`。

    同时把 float('nan') / float('inf') 转成 None：标准 JSON 没有 NaN/Inf
    字面量，Python 的 json 模块默认会输出非标准的 `NaN` token
    (allow_nan=True)，其它语言/工具的 JSON 解析器读到会报错，这里主动转
    成合法的 null，更安全。
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        obj = float(obj)
    if isinstance(obj, float):
        return None if (np.isnan(obj) or np.isinf(obj)) else obj
    return obj


# --------------------------------------------------------------------------
# 核心指标计算
# --------------------------------------------------------------------------

class ChainSkipped(ValueError):
    """表示某条链因数据问题被跳过，供批处理层捕获并做原因聚合统计。

    Attributes:
        reason_code: 稳定的短分类标签（不含每条链变化的具体数值），用于跨链聚合
        detail: 完整的人类可读说明，用于调试单条链问题
    """

    def __init__(self, reason_code: str, detail: str):
        self.reason_code = reason_code
        self.detail = detail
        super().__init__(detail)


def compute_phase_instability_metrics(
    unit_vectors: np.ndarray,
    kappa_threshold: float = 0.5,
    rank_threshold: float = 3.0,
) -> dict:
    """统一相变检测。

    Args:
        unit_vectors: (T, d) 单位向量集合
        kappa_threshold: 方案B中判定"方向分散"的 concentration 阈值（经验值，未校准）
        rank_threshold: 方案B中判定"维度爆炸"的 effective_rank 阈值（经验值，未校准）

    Returns:
        dict，字段见函数末尾，均为原生 Python 类型（float/bool/int），可直接 json.dump

    Raises:
        ValueError: 输入为 None/空/维度不对/含 NaN-Inf，或退化到无法计算的程度

    Notes（已知问题，本次未修改，仅补充说明，供解读结果时参考）:
        - geometric_deviation_score (方案A): expected_effective_rank 值域是
          (0, 1]，而 effective_rank 的值域是 [1, min(T,d)]，量级通常不匹配，
          该指标在 effective_rank > 1（几乎总是如此）时几乎退化成 effective_rank
          本身，对"偏离几何关系"的刻画能力存疑。
        - combined_instability_score (方案C): `effective_rank / max(effective_rank, 1)`
          当 effective_rank >= 1 时恒为 1，导致该项对 effective_rank 不敏感，
          整个表达式退化为 (1 - concentration)。这是此前已指出的 bug，按用户
          要求原样保留，用于和其它候选指标做对比。
    """
    if unit_vectors is None:
        raise ValueError("unit_vectors 为 None")

    unit_vectors = np.asarray(unit_vectors, dtype=np.float64)

    if unit_vectors.ndim != 2:
        raise ValueError(f"unit_vectors 应为二维数组 (T, d)，实际维度为 {unit_vectors.ndim}")

    n_vectors, d = unit_vectors.shape

    if n_vectors == 0:
        raise ValueError("unit_vectors 为空 (T=0)")

    if not np.all(np.isfinite(unit_vectors)):
        raise ValueError("unit_vectors 中含 NaN/Inf；请先用 normalize_vectors 清洗")

    # 轻量健全性检查：函数假设输入是单位向量，若明显不是则提醒一次
    # （消息本身不含具体数值，确保 Python 的默认警告去重能生效，不会刷屏）
    input_norms = np.linalg.norm(unit_vectors, axis=-1)
    if not np.allclose(input_norms, 1.0, atol=1e-3):
        warnings.warn(
            "compute_phase_instability_metrics: 部分输入向量范数显著偏离 1，"
            "本函数假设输入为单位向量，结果可能不可靠。",
            RuntimeWarning,
        )

    # 1. 方向集中度 (mean resultant length)
    mean_vector = unit_vectors.mean(axis=0)
    mean_resultant_length = float(np.linalg.norm(mean_vector))
    if mean_resultant_length > 1.0 + 1e-6:
        warnings.warn(
            f"mean_resultant_length={mean_resultant_length:.6f} 超过理论上界 1，已裁剪。",
            RuntimeWarning,
        )
    mean_resultant_length = min(mean_resultant_length, 1.0)

    # 2. 有效秩：Gram 矩阵技巧
    #    (U @ U.T) 是 (T,T)，(U.T @ U) 是 (d,d)，两者非零特征值完全相同
    #    （标准线性代数事实，来自 U 的 SVD）。经验证两种算法结果一致
    #    （最大误差 ~1e-12，属浮点噪声），但当 T << d 时前者的特征分解成本
    #    是 O(T^3)，后者是 O(d^3)。本数据集 d=4096，T 通常是几到几十的推理
    #    步数，实测 (T=30,d=4096) 场景下加速比可达约 10000 倍——这直接决定
    #    了脚本能否在合理时间内跑完整个数据集，而不只是数值上的小优化。
    if n_vectors <= d:
        gram = (unit_vectors @ unit_vectors.T) / n_vectors  # (T, T)
    else:
        gram = (unit_vectors.T @ unit_vectors) / n_vectors  # (d, d)

    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = eigenvalues[eigenvalues > 0]

    if eigenvalues.size == 0:
        raise ValueError("特征分解后没有正特征值（输入可能全为退化向量），无法计算 effective_rank")

    eigenvalues = eigenvalues / eigenvalues.sum()
    effective_rank = float(np.exp(-(eigenvalues * np.log(eigenvalues + 1e-12)).sum()))

    max_possible_rank = float(min(n_vectors, d))
    if effective_rank > max_possible_rank + 1e-6:
        warnings.warn(
            f"effective_rank={effective_rank:.4f} 超过理论上界 {max_possible_rank}，已裁剪。",
            RuntimeWarning,
        )
    effective_rank = min(effective_rank, max_possible_rank)

    # 3. 三种候选联合指标（供对比，未收敛为单一指标）

    # 方案A：几何偏差
    expected_effective_rank = 1.0 / (1.0 + mean_resultant_length**2)
    geometric_deviation_score = float(abs(effective_rank - expected_effective_rank))

    # 方案B：布尔异常判定
    low_concentration_flag = bool(mean_resultant_length < kappa_threshold)
    rank_explosion_flag = bool(effective_rank > rank_threshold)
    combined_anomaly_flag = bool(low_concentration_flag and rank_explosion_flag)

    # 方案C：连续联合分数（EDIS风格）—— 已知bug按原样保留，见函数 docstring
    combined_instability_score = float(
        (1 - mean_resultant_length) * (effective_rank / max(effective_rank, 1))
    )

    return {
        'concentration': mean_resultant_length,
        'effective_rank': effective_rank,
        'geometric_deviation_score': geometric_deviation_score,    # 方案A
        'combined_anomaly_flag': combined_anomaly_flag,            # 方案B
        'combined_instability_score': combined_instability_score,  # 方案C
        'n_vectors': int(n_vectors),
        'vector_dim': int(d),
    }


def load_full_npz(npz_path: str) -> dict:
    """加载 full_*.npz 数据文件。

    Returns:
        dict，字段见下方 return 语句

    Raises:
        ValueError: 文件无法加载，或缺少必要字段，或字段长度不一致
    """
    print(f"Loading data from {npz_path}...")

    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        raise ValueError(f"无法加载 npz 文件 {npz_path}: {e}") from e

    missing_keys = [k for k in REQUIRED_NPZ_KEYS if k not in data.files]
    if missing_keys:
        raise ValueError(
            f"npz 缺少必要字段: {missing_keys}；实际字段: {list(data.files)}"
        )

    problem_ids = data['problem_ids']
    labels = data['is_correct_strict']
    N = len(problem_ids)

    if len(labels) != N:
        raise ValueError(
            f"problem_ids (长度{N}) 与 is_correct_strict (长度{len(labels)}) 不一致"
        )

    unexpected_labels = set(np.unique(labels).tolist()) - {0, 1}
    if unexpected_labels:
        warnings.warn(
            f"is_correct_strict 中含预期外取值: {unexpected_labels}（预期只有 0/1），"
            "这些样本会在逐链处理阶段被跳过。",
            RuntimeWarning,
        )

    # 标签方向断言：写入端约定 1=correct，且 correct 应与 gold_error_step<0 基本一致
    # （ProcessBench: gold_error_step=-1 表示全对）。若一致率低于 0.5 说明方向反了。
    if 'gold_error_step' in data.files:
        ges = data['gold_error_step'].astype(int)
        if len(ges) == N:
            agree = float(np.mean((labels == 1) == (ges < 0)))
            print(f"  label check: P(is_correct_strict==1 <=> gold_error_step<0) = {agree:.3f}")
            if agree < 0.5:
                warnings.warn(
                    "is_correct_strict 与 gold_error_step 方向相反——标签约定疑似反转，"
                    "请核对写入端 extract_features._pb_record（应为 1=correct）。",
                    RuntimeWarning,
                )

    stepvec = data.get('stepvec', None)
    stepcloud = data.get('stepcloud', None)

    if stepvec is None and stepcloud is None:
        warnings.warn("npz 中 stepvec 和 stepcloud 均缺失，后续指标计算将无法进行。", RuntimeWarning)
    if stepvec is not None and len(stepvec) != N:
        warnings.warn(
            f"stepvec 长度({len(stepvec)}) 与 problem_ids 长度({N}) 不一致，索引可能错位。",
            RuntimeWarning,
        )

    cloud_feature_names = data.get('cloud_feature_names', None)
    if cloud_feature_names is not None:
        cloud_feature_names = [str(n) for n in cloud_feature_names]

    # sv_layers: 实际的层号（stepvec 第二维对应的层）
    sv_layers = data.get('sv_layers', None)
    if sv_layers is not None:
        sv_layers = [int(l) for l in sv_layers]

    print(f"  Loaded {N} chains")
    if sv_layers is not None:
        print(f"  sv_layers (actual layer numbers): {sv_layers}")
        print(f"  Use --layer_idx 0-{len(sv_layers)-1} to select layer")
    if stepvec is not None and len(stepvec) > 0:
        sample_idx = 0
        for idx in range(min(10, len(stepvec))):
            if stepvec[idx] is not None and hasattr(stepvec[idx], 'shape'):
                sample_idx = idx
                break
        sample = stepvec[sample_idx]
        if sample is not None:
            print(f"  stepvec[{sample_idx}] shape: {sample.shape}")
    if stepcloud is not None and len(stepcloud) > 0:
        sample_idx = 0
        for idx in range(min(10, len(stepcloud))):
            if stepcloud[idx] is not None and hasattr(stepcloud[idx], 'shape'):
                sample_idx = idx
                break
        sample = stepcloud[sample_idx]
        if sample is not None:
            print(f"  stepcloud[{sample_idx}] shape: {sample.shape}")
            print(f"  cloud features: {cloud_feature_names}")

    return {
        'stepvec': stepvec,
        'stepcloud': stepcloud,
        'cloud_feature_names': cloud_feature_names,
        'sv_layers': sv_layers,
        'labels': labels,
        'problem_ids': problem_ids,
        'N': N,
    }


def normalize_vectors(vectors: np.ndarray, eps: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    """将向量归一化为单位向量，并标记退化向量（零范数 / 含 NaN-Inf）。

    Args:
        vectors: (..., d) 待归一化向量
        eps: 判定"零范数"的阈值

    Returns:
        (unit_vectors, valid_mask):
            unit_vectors: (..., d)，退化位置被显式置为全零向量（而不是让 NaN 泄漏出去）
            valid_mask:   (...,) bool，True 表示该向量被成功归一化为真正的单位向量

    Raises:
        ValueError: 输入为空数组，或处理后仍存在非有限值（安全网，理论上不会触发）
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    if vectors.size == 0:
        raise ValueError("输入为空数组")

    # 先隔离含 NaN/Inf 的行，避免其在范数计算中污染其它维度
    finite_mask = np.all(np.isfinite(vectors), axis=-1)
    safe_input = np.where(finite_mask[..., None], vectors, 0.0)

    norms = np.linalg.norm(safe_input, axis=-1, keepdims=True)
    valid_mask = finite_mask & (norms.ravel() > eps)

    norms_safe = np.where(norms > eps, norms, 1.0)
    unit_vectors = np.where(valid_mask[..., None], safe_input / norms_safe, 0.0)

    if not np.all(np.isfinite(unit_vectors)):
        raise ValueError("归一化后仍出现 NaN/Inf（不应发生，请检查数据）")

    return unit_vectors, valid_mask


def compute_chain_phase_metrics(
    stepvec: np.ndarray,
    layer_idx: int = 0,
    min_steps: int = 2,
    drop_degenerate: bool = True,
    kappa_threshold: float = 0.5,
    rank_threshold: float = 3.0,
    vector_mode: str = 'raw',
) -> dict:
    """计算单条链的相变不稳定性指标。

    Args:
        stepvec: (T, n_layers, d) 单条链的 step vectors
        layer_idx: 要分析的层索引
        min_steps: 参与计算所需的最少有效步骤数；T=1 时方向"分散度"和"有效秩"
            在数学上退化为常数(1)，不具备统计意义，默认至少要 2 步
        drop_degenerate: 是否在计算前剔除退化(零范数/NaN)向量。若为 False，
            退化向量会被当成零向量参与均值/Gram矩阵计算，导致 concentration
            系统性偏低（零向量拉低均值合向量长度，但不贡献有效方向），
            这是原始实现未处理的隐藏偏差来源。
        kappa_threshold, rank_threshold: 透传给方案B的判定阈值
        vector_mode: 'raw' = 原始池化位置向量 h_t（各向异性饱和，仅作对照）；
            'center' = 减去链内均值，去掉同链各步共享的公共分量；
            'delta' = 一阶差分 Δh_t = h_{t+1} − h_t，步间位移方向
            （推荐，与 NTS 的位移几何一致；有效向量数变为 T−1）

    Returns:
        指标字典（在 compute_phase_instability_metrics 基础上附加
        n_steps_total / n_steps_used / n_degenerate_vectors）

    Raises:
        ChainSkipped: 数据不满足计算条件，reason_code 可用于批量统计
    """
    if stepvec is None:
        raise ChainSkipped("stepvec_none", "stepvec 为 None")

    stepvec = np.asarray(stepvec)

    if stepvec.size == 0:
        raise ChainSkipped("stepvec_empty", "stepvec 为空数组")

    if stepvec.ndim != 3:
        raise ChainSkipped(
            "stepvec_bad_ndim",
            f"stepvec 维度应为 3 (T, n_layers, d)，实际为 {stepvec.ndim}, shape={stepvec.shape}",
        )

    T, n_layers, d = stepvec.shape

    if not (0 <= layer_idx < n_layers):
        raise ChainSkipped(
            "layer_idx_out_of_range",
            f"layer_idx={layer_idx} 超出该链范围 [0, {n_layers - 1}] (n_layers={n_layers})"
            "——已跳过，而非静默回退到 layer 0（原实现的隐藏 bug）",
        )

    vectors = np.asarray(stepvec[:, layer_idx, :], dtype=np.float64)

    if vector_mode == 'center':
        vectors = vectors - vectors.mean(axis=0, keepdims=True)
    elif vector_mode == 'delta':
        if len(vectors) < 2:
            raise ChainSkipped("too_few_steps_for_delta",
                               f"delta 模式需要 T>=2，实际 T={len(vectors)}")
        vectors = np.diff(vectors, axis=0)
    elif vector_mode != 'raw':
        raise ValueError(f"未知 vector_mode: {vector_mode!r}（可选 raw/center/delta）")

    try:
        unit_vectors, valid_mask = normalize_vectors(vectors)
    except ValueError as e:
        raise ChainSkipped("normalize_failed", str(e)) from e

    n_degenerate = int((~valid_mask).sum())
    if drop_degenerate and n_degenerate > 0:
        unit_vectors = unit_vectors[valid_mask]

    n_used = len(unit_vectors)
    if n_used < min_steps:
        extra = f"（原始 T={T}，含 {n_degenerate} 个退化向量）" if n_degenerate else ""
        raise ChainSkipped(
            "too_few_valid_steps",
            f"有效步骤数 {n_used} < min_steps={min_steps}{extra}",
        )

    try:
        metrics = compute_phase_instability_metrics(
            unit_vectors,
            kappa_threshold=kappa_threshold,
            rank_threshold=rank_threshold,
        )
    except ValueError as e:
        raise ChainSkipped("metrics_computation_failed", str(e)) from e

    metrics['n_steps_total'] = int(T)
    metrics['n_steps_used'] = int(n_used)
    metrics['n_degenerate_vectors'] = n_degenerate

    return metrics


def compute_per_chain_metrics(
    data: dict,
    layer_idx: int = 0,
    min_steps: int = 2,
    drop_degenerate: bool = True,
    kappa_threshold: float = 0.5,
    rank_threshold: float = 3.0,
    vector_mode: str = 'raw',
) -> list:
    """为所有链计算相变指标，附带完整的跳过原因统计。

    Args:
        data: load_full_npz 返回的数据字典
        layer_idx / min_steps / drop_degenerate / kappa_threshold / rank_threshold /
            vector_mode: 透传给 compute_chain_phase_metrics

    Returns:
        list of dict，每个包含一条链的指标和标签
    """
    results = []

    if data['stepvec'] is None:
        print("No stepvec data available.")
        return results

    stepvec = data['stepvec']
    labels = data['labels']
    N = data['N']

    skip_reasons: Counter = Counter()
    examples_shown: Counter = Counter()
    chains_with_degenerate = 0
    total_degenerate_vectors = 0

    def _log_skip(reason_code: str, detail: str, idx: int):
        skip_reasons[reason_code] += 1
        if examples_shown[reason_code] < 3:  # 每类原因只打印前3个具体例子，避免刷屏
            tqdm.write(f"    [跳过 #{idx}] {reason_code}: {detail}")
            examples_shown[reason_code] += 1

    for i in tqdm(range(N), desc="Computing phase metrics"):
        chain_stepvec = stepvec[i]

        try:
            metrics = compute_chain_phase_metrics(
                chain_stepvec,
                layer_idx=layer_idx,
                min_steps=min_steps,
                drop_degenerate=drop_degenerate,
                kappa_threshold=kappa_threshold,
                rank_threshold=rank_threshold,
                vector_mode=vector_mode,
            )
        except ChainSkipped as e:
            _log_skip(e.reason_code, e.detail, i)
            continue

        if metrics['n_degenerate_vectors'] > 0:
            chains_with_degenerate += 1
            total_degenerate_vectors += metrics['n_degenerate_vectors']

        label_val = labels[i]
        if label_val not in (0, 1):
            _log_skip("unexpected_label_value", f"label={label_val!r}", i)
            continue

        raw_pid = data['problem_ids'][i]
        try:
            pid: Any = int(raw_pid)
        except (ValueError, TypeError):
            pid = str(raw_pid)  # problem_id 不一定是纯数字，容错处理

        results.append({
            'chain_idx': i,
            'problem_id': pid,
            'is_correct': bool(label_val == 1),  # npz约定: 1=correct (extract_features._pb_record)
            'metrics': metrics,
        })

    n_skipped = sum(skip_reasons.values())
    if n_skipped > 0:
        print(f"\n  共跳过 {n_skipped}/{N} 条链，跳过原因统计:")
        for reason, count in skip_reasons.most_common():
            print(f"    - {reason}: {count} 条")

    if chains_with_degenerate > 0:
        action = "已剔除后再计算" if drop_degenerate else "未剔除（可能造成 concentration 系统性偏低）"
        print(
            f"\n  提示: {chains_with_degenerate}/{N} 条链中共发现 {total_degenerate_vectors} "
            f"个退化(零范数/NaN)向量，{action}。可用 --keep_degenerate_vectors 切换该行为。"
        )

    return results


# --------------------------------------------------------------------------
# 统计检验
# --------------------------------------------------------------------------

def compute_length_confound(results: list) -> dict:
    """长度混杂诊断: AUROC(T alone) + 各指标与 T 的 Spearman 相关。

    动机: 错误链往往更长，而 effective_rank 的上界是 min(T,d)、
    concentration 也有机械的 T 依赖。若某指标的 AUROC 没有明显超过
    AUROC(T alone)，或与 T 强相关，其"信号"很可能只是长度的代理
    （本项目 FINDINGS.md 已证明长度是首要混杂，步级长度单独 AUROC≈0.71）。
    """
    if not results:
        return {}

    y_true = np.array([0 if r['is_correct'] else 1 for r in results])
    T_used = np.array([r['metrics']['n_steps_used'] for r in results], dtype=np.float64)

    out: dict = {}
    if len(np.unique(y_true)) >= 2 and len(np.unique(T_used)) >= 2:
        out['auroc_T_alone'] = float(roc_auc_score(y_true, T_used))
    else:
        out['auroc_T_alone'] = float('nan')

    out['spearman_vs_T'] = {}
    for key in CONTINUOUS_METRIC_KEYS:
        vals = np.array([r['metrics'][key] for r in results], dtype=np.float64)
        m = np.isfinite(vals) & np.isfinite(T_used)
        if m.sum() >= 3:
            rho = stats.spearmanr(vals[m], T_used[m]).statistic
            out['spearman_vs_T'][key] = float(rho) if np.isfinite(rho) else float('nan')
        else:
            out['spearman_vs_T'][key] = float('nan')
    return out


def compute_auroc_scores(results: list) -> dict:
    """计算各指标的 AUROC (预测 error 的能力)。"""
    if not results:
        return {}

    y_true = np.array([0 if r['is_correct'] else 1 for r in results])

    if len(np.unique(y_true)) < 2:
        warnings.warn("compute_auroc_scores: 所有样本标签相同，无法计算 AUROC。", RuntimeWarning)
        return {key: {'auroc': np.nan, 'n_valid': len(results)} for key in CONTINUOUS_METRIC_KEYS}

    auroc_results = {}

    for key in CONTINUOUS_METRIC_KEYS:
        scores = np.array([r['metrics'][key] for r in results], dtype=np.float64)

        valid_mask = np.isfinite(scores)
        y_valid = y_true[valid_mask]
        scores_valid = scores[valid_mask]

        n_dropped = len(scores) - len(scores_valid)
        if n_dropped > 0:
            warnings.warn(
                f"compute_auroc_scores: 指标 '{key}' 中有 {n_dropped} 个非有限值被剔除。",
                RuntimeWarning,
            )

        if len(np.unique(y_valid)) < 2:
            auroc_results[key] = {'auroc': np.nan, 'n_valid': len(scores_valid)}
            continue

        try:
            auroc = roc_auc_score(y_valid, scores_valid)
            auroc_results[key] = {'auroc': float(auroc), 'n_valid': len(scores_valid)}
        except Exception as e:
            auroc_results[key] = {'auroc': np.nan, 'n_valid': len(scores_valid), 'error': str(e)}

    anomaly_pred = np.array([r['metrics']['combined_anomaly_flag'] for r in results], dtype=float)
    valid_mask = np.isfinite(anomaly_pred)
    y_valid = y_true[valid_mask]
    pred_valid = anomaly_pred[valid_mask]

    if len(np.unique(y_valid)) >= 2:
        try:
            auroc = roc_auc_score(y_valid, pred_valid)
            auroc_results['combined_anomaly_flag'] = {'auroc': float(auroc), 'n_valid': len(pred_valid)}
        except Exception:
            auroc_results['combined_anomaly_flag'] = {'auroc': np.nan, 'n_valid': len(pred_valid)}
    else:
        auroc_results['combined_anomaly_flag'] = {'auroc': np.nan, 'n_valid': len(pred_valid)}

    return auroc_results


def run_statistical_tests(results: list) -> dict:
    """对相变指标进行统计检验，区分 error vs correct。"""
    correct_results = [r for r in results if r['is_correct']]
    error_results = [r for r in results if not r['is_correct']]

    n_correct = len(correct_results)
    n_error = len(error_results)

    print(f"\nStatistical testing:")
    print(f"  Correct: {n_correct}, Error: {n_error}")

    if n_correct == 0 or n_error == 0:
        print("  Insufficient data for statistical testing.")
        return {}

    if n_correct < MIN_SAMPLE_SIZE_WARNING or n_error < MIN_SAMPLE_SIZE_WARNING:
        warnings.warn(
            f"run_statistical_tests: 样本量偏小 (correct={n_correct}, error={n_error})，"
            "p值/AUROC估计可能不稳定，结果仅供参考。",
            RuntimeWarning,
        )

    auroc_results = compute_auroc_scores(results)
    test_results = {}

    for key in CONTINUOUS_METRIC_KEYS:
        correct_vals = np.array([r['metrics'][key] for r in correct_results], dtype=np.float64)
        error_vals = np.array([r['metrics'][key] for r in error_results], dtype=np.float64)

        correct_vals = correct_vals[np.isfinite(correct_vals)]
        error_vals = error_vals[np.isfinite(error_vals)]

        if len(correct_vals) == 0 or len(error_vals) == 0:
            test_results[key] = {
                'n_correct': 0, 'n_error': 0,
                'correct_mean': np.nan, 'error_mean': np.nan, 'mean_diff': np.nan,
                'statistic': np.nan, 'p_value': np.nan, 'cohens_d': np.nan,
                'auroc': np.nan, 'significant': False,
            }
            continue

        try:
            stat, pval = stats.mannwhitneyu(error_vals, correct_vals, alternative='two-sided')
        except Exception:
            stat, pval = np.nan, np.nan

        n_c, n_e = len(correct_vals), len(error_vals)
        if n_c + n_e > 2:
            pooled_std = np.sqrt(
                ((n_c - 1) * correct_vals.var(ddof=1) + (n_e - 1) * error_vals.var(ddof=1))
                / (n_c + n_e - 2)
            )
        else:
            pooled_std = np.nan

        if np.isfinite(pooled_std) and pooled_std > 0:
            cohens_d = (error_vals.mean() - correct_vals.mean()) / pooled_std
        else:
            # 标准差为0或样本不足：效应量在数学上未定义，用 NaN 而非 0，
            # 避免被误读为"确认没有差异"
            cohens_d = np.nan

        auroc = auroc_results.get(key, {}).get('auroc', np.nan)

        test_results[key] = {
            'n_correct': n_c,
            'n_error': n_e,
            'correct_mean': float(correct_vals.mean()),
            'error_mean': float(error_vals.mean()),
            'mean_diff': float(error_vals.mean() - correct_vals.mean()),
            'correct_p10_p50_p90': [float(x) for x in np.percentile(correct_vals, [10, 50, 90])],
            'error_p10_p50_p90': [float(x) for x in np.percentile(error_vals, [10, 50, 90])],
            'statistic': float(stat) if np.isfinite(stat) else np.nan,
            'p_value': float(pval) if np.isfinite(pval) else np.nan,
            'cohens_d': float(cohens_d) if np.isfinite(cohens_d) else np.nan,
            'auroc': float(auroc) if np.isfinite(auroc) else np.nan,
            'significant': bool(pval < 0.05) if np.isfinite(pval) else False,
        }

        direction = ">" if error_vals.mean() > correct_vals.mean() else "<"
        sig_str = "*" if test_results[key]['significant'] else ""
        print(f"  {key:30s}: error{direction}correct | "
              f"error={error_vals.mean():.4f}, correct={correct_vals.mean():.4f} | "
              f"d={cohens_d:.3f}, p={pval:.4f}, auroc={auroc:.3f}{sig_str}")

    correct_anomaly = sum(1 for r in correct_results if r['metrics']['combined_anomaly_flag'])
    error_anomaly = sum(1 for r in error_results if r['metrics']['combined_anomaly_flag'])
    correct_no_anomaly = n_correct - correct_anomaly
    error_no_anomaly = n_error - error_anomaly

    try:
        oddsratio, pval = stats.fisher_exact(
            [[error_anomaly, error_no_anomaly], [correct_anomaly, correct_no_anomaly]],
            alternative='greater',
        )
    except Exception:
        oddsratio, pval = np.nan, np.nan

    auroc = auroc_results.get('combined_anomaly_flag', {}).get('auroc', np.nan)

    test_results['combined_anomaly_flag'] = {
        'n_correct': n_correct,
        'n_error': n_error,
        'correct_anomaly_rate': float(correct_anomaly / n_correct) if n_correct > 0 else np.nan,
        'error_anomaly_rate': float(error_anomaly / n_error) if n_error > 0 else np.nan,
        'correct_anomaly': correct_anomaly,
        'error_anomaly': error_anomaly,
        'odds_ratio': float(oddsratio) if np.isfinite(oddsratio) else np.nan,
        'p_value': float(pval) if np.isfinite(pval) else np.nan,
        'auroc': float(auroc) if np.isfinite(auroc) else np.nan,
        'significant': bool(pval < 0.05) if np.isfinite(pval) else False,
    }

    sig_str = "*" if test_results['combined_anomaly_flag']['significant'] else ""
    print(f"  {'combined_anomaly_flag':30s}: error={error_anomaly}/{n_error} "
          f"({error_anomaly/n_error*100:.1f}%), correct={correct_anomaly}/{n_correct} "
          f"({correct_anomaly/n_correct*100:.1f}%) | OR={oddsratio:.2f}, p={pval:.4f}, "
          f"auroc={auroc:.3f}{sig_str}")

    return test_results


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='验证相变不稳定性指标')
    parser.add_argument('--dataset', choices=['gsm8k', 'math', 'omnimath'],
                         default='gsm8k', help='数据集名称')
    parser.add_argument('--data_dir', default='/gz-data/research/demo/data/features',
                         help='数据目录路径')
    parser.add_argument('--output_dir', default='./results/phase_instability',
                         help='输出目录')
    parser.add_argument('--layer_idx', type=int, default=3,
                         help='stepvec 中的层索引 (0-7，对应 sv_layers 数组中的位置；'
                              '运行时会显示实际层号)')
    parser.add_argument('--min_steps', type=int, default=2,
                         help='参与统计所需的最少有效步骤数（默认2，数学下限）')
    parser.add_argument('--kappa_threshold', type=float, default=0.5,
                         help='方案B: 判定"方向分散"的 concentration 阈值')
    parser.add_argument('--rank_threshold', type=float, default=3.0,
                         help='方案B: 判定"维度爆炸"的 effective_rank 阈值')
    parser.add_argument('--keep_degenerate_vectors', action='store_true',
                         help='保留退化(零范数/NaN)向量参与计算（默认剔除；保留会使 '
                              'concentration 系统性偏低，仅用于复现旧行为/对比）')
    parser.add_argument('--vector_mode', choices=['raw', 'center', 'delta'],
                         default='delta',
                         help='raw=原始池化位置向量（各向异性饱和，仅作对照）; '
                              'center=减链内均值; delta=一阶差分位移 Δh_t（默认，推荐）')

    args = parser.parse_args()

    npz_path = os.path.join(args.data_dir, f'full_{args.dataset}.npz')

    print("=" * 70)
    print("相变不稳定性指标验证")
    print("=" * 70)
    print(f"Dataset: {args.dataset}")
    print(f"NPZ path: {npz_path}")
    print(f"Layer index: {args.layer_idx}")
    print(f"Min steps: {args.min_steps}")
    print(f"Vector mode: {args.vector_mode}")
    print(f"退化向量处理: {'保留' if args.keep_degenerate_vectors else '剔除'}")
    print()

    if not os.path.exists(npz_path):
        print(f"错误: 文件不存在: {npz_path}")
        print(f"\n提示: 根据配置，数据应在远程服务器上:")
        print(f"  /gz-data/research/demo/data/features/full_{args.dataset}.npz")
        return

    # 尽早校验输出目录可写，避免算完几小时后才发现无法保存结果
    try:
        os.makedirs(args.output_dir, exist_ok=True)
    except OSError as e:
        print(f"错误: 无法创建输出目录 {args.output_dir}: {e}")
        return

    try:
        data = load_full_npz(npz_path)
    except ValueError as e:
        print(f"错误: 加载数据失败: {e}")
        return

    print("\nComputing phase instability metrics...")
    results = compute_per_chain_metrics(
        data,
        layer_idx=args.layer_idx,
        min_steps=args.min_steps,
        drop_degenerate=not args.keep_degenerate_vectors,
        kappa_threshold=args.kappa_threshold,
        rank_threshold=args.rank_threshold,
        vector_mode=args.vector_mode,
    )

    if len(results) == 0:
        print("No results computed. Exiting.")
        return

    print(f"Computed metrics for {len(results)} chains")

    test_results = run_statistical_tests(results)
    length_confound = compute_length_confound(results)

    output_file = os.path.join(
        args.output_dir,
        f'{args.dataset}_layer{args.layer_idx}_{args.vector_mode}_results.json',
    )

    output = {
        'dataset': args.dataset,
        'layer_idx': args.layer_idx,
        'min_steps': args.min_steps,
        'kappa_threshold': args.kappa_threshold,
        'rank_threshold': args.rank_threshold,
        'drop_degenerate': not args.keep_degenerate_vectors,
        'vector_mode': args.vector_mode,
        'label_convention': 'is_correct_strict: 1=correct (extract_features._pb_record)',
        'npz_path': npz_path,
        'n_chains': len(results),
        'n_correct': len([r for r in results if r['is_correct']]),
        'n_error': len([r for r in results if not r['is_correct']]),
        'test_results': test_results,
        'length_confound': length_confound,
        'per_chain_results': results,
    }

    # --- 核心bug修复处 ---
    # numpy 标量(float32/float64/bool_等)不能被 json 模块直接序列化；
    # 递归清洗为原生类型，并将 NaN/Inf 转为 null。
    output_clean = sanitize_for_json(output)

    try:
        with open(output_file, 'w') as f:
            json.dump(output_clean, f, indent=2, cls=NumpyJSONEncoder, allow_nan=False)
        print(f"\n结果保存至: {output_file}")
    except (TypeError, ValueError) as e:
        # 兜底：sanitize 理论上已覆盖所有已知字段，这里是双重保险。
        # 完整结果序列化失败时，至少保住汇总统计，不让几小时计算白跑。
        print(f"\n警告: 完整结果序列化失败 ({e})，尝试仅保存汇总统计...")
        summary_file = output_file.replace('.json', '_summary_only.json')
        summary_output = {k: v for k, v in output_clean.items() if k != 'per_chain_results'}
        try:
            with open(summary_file, 'w') as f:
                json.dump(summary_output, f, indent=2, cls=NumpyJSONEncoder, allow_nan=False)
            print(f"  汇总统计已保存至: {summary_file}（per_chain_results 因序列化失败被跳过）")
        except (TypeError, ValueError) as e2:
            print(f"  错误: 汇总统计也序列化失败: {e2}")
            return

    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)

    auroc_sorted = sorted(
        [(k, v.get('auroc', np.nan)) for k, v in test_results.items() if 'auroc' in v],
        key=lambda x: x[1] if np.isfinite(x[1]) else -1,
        reverse=True,
    )

    print("\nAUROC 排序 (预测 error 的能力):")
    for key, auroc in auroc_sorted:
        if np.isfinite(auroc):
            sig_str = "*" if test_results[key].get('significant', False) else ""
            print(f"  - {key:30s}: {auroc:.4f}{sig_str}")

    if length_confound:
        t_auroc = length_confound.get('auroc_T_alone', float('nan'))
        print(f"\n长度混杂基线 (必读，与上表同表解读):")
        if np.isfinite(t_auroc):
            print(f"  - {'AUROC(T alone)':30s}: {t_auroc:.4f}   <- 任何指标至少要打过它")
        for k, v in length_confound.get('spearman_vs_T', {}).items():
            if np.isfinite(v):
                flag = "  <- 与T强相关，疑似长度代理" if abs(v) > 0.6 else ""
                print(f"  - Spearman({k[:22]:22s}, T): {v:+.3f}{flag}")

    significant_metrics = [k for k, v in test_results.items() if v.get('significant', False)]

    if significant_metrics:
        print(f"\n显著区分 error vs correct 的指标 (p < 0.05):")
        for key in significant_metrics:
            val = test_results[key]
            auroc_str = f", AUROC = {val.get('auroc', np.nan):.3f}" if np.isfinite(val.get('auroc', np.nan)) else ""
            if 'cohens_d' in val:
                print(f"  - {key}: Cohen's d = {val['cohens_d']:.3f}, p = {val['p_value']:.4f}{auroc_str}")
            else:
                print(f"  - {key}: OR = {val['odds_ratio']:.2f}, p = {val['p_value']:.4f}{auroc_str}")
    else:
        print("\n没有指标达到显著性水平 (p < 0.05)")

    print("\n说明:")
    print("  - concentration: 方向集中度 (mean resultant length)")
    print("  - effective_rank: 有效秩 (基于方向张量的熵)")
    print("  - geometric_deviation_score: 几何偏差 (方案A)")
    print("  - combined_anomaly_flag: 布尔异常判定 (方案B)")
    print("  - combined_instability_score: 连续联合分数 (方案C, 含已知bug，原样保留)")
    print("  - AUROC: Area Under ROC Curve, 0.5=随机, 1.0=完美区分 error vs correct")
    print("  - * 表示 p < 0.05 显著")


if __name__ == '__main__':
    main()