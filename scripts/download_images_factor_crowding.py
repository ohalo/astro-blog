#!/usr/bin/env python3
"""
为因子拥挤度文章下载/生成配图
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-crowding'
os.makedirs(output_dir, exist_ok=True)

# 图1: 因子拥挤度传导机制示意图
fig, ax = plt.subplots(figsize=(12, 8))
ax.text(0.5, 0.9, '因子拥挤度传导机制', ha='center', fontsize=20, fontweight='bold')

# 绘制流程图
stages = [
    (0.5, 0.75, '资金涌入\n因子被发现'),
    (0.5, 0.60, '更多资金跟随\n因子溢价扩散'),
    (0.5, 0.45, '估值透支\n价格偏离基本面'),
    (0.5, 0.30, '流动性边际恶化\n交易摩擦增加'),
    (0.5, 0.15, '拥挤交易踩踏\n因子失效')
]

for i, (x, y, text) in enumerate(stages):
    ax.annotate('', xy=(x, y-0.1), xytext=(x, y+0.05),
                arrowprops=dict(arrowstyle='->', lw=2, color='red'))
    ax.text(x, y, text, ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black'),
            fontsize=14)

ax.axis('off')
plt.tight_layout()
plt.savefig(f'{output_dir}/crowding_mechanism.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 拥挤度指数（CCI）与因子收益的关系
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2024-12-31', freq='D')[:504]
cci = np.random.beta(2, 5, len(dates)) * 100  # 模拟CCI指数
factor_ret = -0.005 * (cci / 100) + np.random.normal(0, 0.02, len(dates))  # 模拟因子收益

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax1.plot(dates, cci, linewidth=2, color='red', label='拥挤度指数(CCI)')
ax1.axhline(y=80, color='darkred', linestyle='--', label='拥挤阈值(80)')
ax1.fill_between(dates, 80, cci, where=(cci >= 80), alpha=0.3, color='red')
ax1.set_ylabel('拥挤度指数', fontsize=14)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(dates, np.cumprod(1 + factor_ret), linewidth=2, color='blue', label='因子累计收益')
ax2.scatter(dates[cci > 80], np.cumprod(1 + factor_ret)[cci > 80],
            color='red', s=50, label='高拥挤度时期', zorder=5)
ax2.set_ylabel('累计收益', fontsize=14)
ax2.set_xlabel('日期', fontsize=14)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.suptitle('因子拥挤度指数(CCI)与因子收益的关系', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/cci_vs_returns.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 图片已保存到 {output_dir}")
print(f"   - crowding_mechanism.png")
print(f"   - cci_vs_returns.png")
