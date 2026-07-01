#!/usr/bin/env python3
"""测试修复后的几何特征计算"""
import numpy as np
import sys
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

def test_compute_step_geometry():
    """测试compute_step_geometry_ultra_fast修复版"""
    from data_loading_cache import compute_step_geometry_ultra_fast

    print("测试修复后的compute_step_geometry_ultra_fast...")

    # 创建测试数据
    np.random.seed(42)
    n_tokens = 50
    d = 4096

    # 模拟hidden states
    H = np.random.randn(n_tokens, d).astype(np.float32)

    # 调用函数
    result = compute_step_geometry_ultra_fast(H, step_id=0, layer_id=14, n_top=10)

    if result is None:
        print("❌ 函数返回None")
        return False

    print("✓ 函数返回非None")

    # 检查所有字段
    required_fields = ['step_id', 'layer', 'n_tokens', 'kappa', 'eff_rank',
                      'spectral_entropy', 'eigenvalues']
    for field in required_fields:
        if field not in result:
            print(f"❌ 缺少字段: {field}")
            return False

    print("✓ 所有必需字段存在")

    # 检查值
    print(f"\n检查结果:")
    print(f"  step_id: {result['step_id']} (预期: 0)")
    print(f"  layer: {result['layer']} (预期: 14)")
    print(f"  n_tokens: {result['n_tokens']} (预期: {n_tokens})")
    print(f"  kappa: {result['kappa']:.4f} (范围: [0,1])")
    print(f"  eff_rank: {result['eff_rank']:.2f}")
    print(f"  spectral_entropy: {result['spectral_entropy']:.4f}")
    print(f"  eigenvalues shape: {result['eigenvalues'].shape}")
    print(f"  eigenvalues sum: {result['eigenvalues'].sum():.6f} (应该≈1)")

    # 验证
    assert result['step_id'] == 0, "step_id错误"
    assert result['layer'] == 14, "layer错误"
    assert result['n_tokens'] == n_tokens, "n_tokens错误"

    # kappa范围
    assert 0 <= result['kappa'] <= 1, f"kappa超出范围: {result['kappa']}"

    # eigenvalues
    eig = result['eigenvalues']
    assert len(eig) == 10, f"eigenvalues长度错误: {len(eig)}"
    assert np.all(eig >= 0), "存在负特征值"
    assert eig.sum() > 0.9 and eig.sum() <= 1.1, f"eigenvalues sum异常: {eig.sum()}"

    # eff_rank
    assert result['eff_rank'] > 0, f"eff_rank应该>0: {result['eff_rank']}"
    assert result['eff_rank'] <= n_tokens, f"eff_rank应该<=n_tokens: {result['eff_rank']}"

    # spectral_entropy (非NaN且非负)
    assert np.isfinite(result['spectral_entropy']), f"spectral_entropy是NaN"
    assert result['spectral_entropy'] >= 0, f"spectral_entropy应该>=0: {result['spectral_entropy']}"

    print("\n✓ 所有检查通过!")
    return True

def test_compare_with_old():
    """对比旧版（对角近似）和新版（完整分解）的差异"""
    from data_loading_cache import compute_step_geometry_ultra_fast

    print("\n对比测试...")

    np.random.seed(42)
    H = np.random.randn(50, 4096).astype(np.float32)

    # 新版（完整分解）
    result_new = compute_step_geometry_ultra_fast(H, 0, 14, n_top=10)

    print("新版特征值:", result_new['eigenvalues'][:3])
    print("新版eff_rank:", result_new['eff_rank'])
    print("新版spectral_entropy:", result_new['spectral_entropy'])

    # 检查eigenvalues的分布
    eig = result_new['eigenvalues']
    print(f"\n特征值分布:")
    print(f"  最大值: {eig[0]:.6f}")
    print(f"  最小值: {eig[-1]:.6f}")
    print(f"  标准差: {eig.std():.6f}")

    return True

if __name__ == "__main__":
    try:
        success = test_compute_step_geometry()
        if success:
            test_compare_with_old()
            print("\n" + "="*50)
            print("✓ 所有测试通过!")
            print("="*50)
        else:
            print("\n" + "="*50)
            print("❌ 测试失败!")
            print("="*50)
    except Exception as e:
        print(f"\n❌ 测试出错: {e}")
        import traceback
        traceback.print_exc()
