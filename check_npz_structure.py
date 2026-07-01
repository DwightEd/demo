#!/usr/bin/env python3
"""检查NPZ文件结构"""
import numpy as np
from pathlib import Path

npz_path = Path('data/features/full_omnimath.npz')
if not npz_path.exists():
    print(f"NPZ文件不存在: {npz_path}")
    exit(1)

data = np.load(npz_path, allow_pickle=True)

print('='*60)
print('NPZ文件完整字段列表')
print('='*60)
for key in data.keys():
    val = data[key]
    print(f'\n{key}:')
    print(f'  Type: {type(val)}')
    print(f'  Shape/Dtype: {val.shape if hasattr(val, "shape") else val.dtype}')
    print(f'  Size: {len(val)}')

    # 如果是object类型，检查第一个元素
    if hasattr(val, 'dtype') and val.dtype == object:
        print(f'  Object dtype, 检查前3个元素...')
        for i in range(min(3, len(val))):
            item = val[i]
            print(f'    [{i}] type: {type(item)}')
            if item is not None:
                if hasattr(item, '__len__'):
                    print(f'        length: {len(item)}')
                    if len(item) > 0:
                        first_elem = item[0]
                        print(f'        [0] type: {type(first_elem)}')
                        print(f'        [0] value: {first_elem}')
                        if len(item) > 1:
                            second_elem = item[1]
                            print(f'        [1] type: {type(second_elem)}')
                            print(f'        [1] value: {second_elem}')

# 详细检查step_token_ranges
print('\n' + '='*60)
print('详细检查 step_token_ranges')
print('='*60)
step_token_ranges = data['step_token_ranges']
print(f'Total chains: {len(step_token_ranges)}')

# 统计
valid_ranges = [r for r in step_token_ranges if r is not None]
print(f'Valid ranges: {len(valid_ranges)}')
print(f'None ranges: {len(step_token_ranges) - len(valid_ranges)}')

if valid_ranges:
    step_counts = [len(r) for r in valid_ranges]
    print(f'Step counts - min: {min(step_counts)}, max: {max(step_counts)}, mean: {np.mean(step_counts):.2f}')

    # 看前5个有效样本
    print('\n前5个有效样本的step_token_ranges:')
    for i, ranges in enumerate([r for r in step_token_ranges if r is not None][:5]):
        print(f'  Sample {i}: {ranges}')
        for j, (start, end) in enumerate(ranges[:3]):
            print(f'    Step {j}: tokens [{start}, {end}), length={end-start}')
        if len(ranges) > 3:
            print(f'    ... ({len(ranges)} total steps)')
