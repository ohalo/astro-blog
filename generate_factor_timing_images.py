#!/usr/bin/env python3
"""
生成因子择时文章的配图
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 输出目录
output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-timing'
os.makedirs(output_dir, exist_ok=True)

# 生成图1: 因子暴露动态调仓示意图
fig, axes = plt.subplots(2, 1, figsize=(12, 10))
fig.suptitle('因子暴露动态调仓示意图', fontsize=16, fontweight='bold')

# 子图1: 市值因子暴露时序变化
ax1 = axes[0]
dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
np.random.seed(42)
base_exposure = 0.5 + 0.3 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
timing_signal = np.random.normal(0, 0.1, len(dates))
dynamic_exposure = base_exposure + timing_signal

ax1.plot(dates, base_exposure, label='基准暴露 (固定)', linewidth=2, alpha=0.7)
ax1.plot(dates, dynamic_exposure, label='动态暴露 (择时)', linewidth=2)
ax1.fill_between(dates, 0, dynamic_exposure, alpha=0.3)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
ax1.set_ylabel('市值因子暴露')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)

# 子图2: 因子收益对比
ax2 = axes[1]
strategies = ['固定暴露', '动态择时', '基准(沪深300)']
returns = [0.08, 0.15, 0.06]
volatility = [0.16, 0.18, 0.20]
sharpe = [0.50, 0.83, 0.30]

x = np.arange(len(strategies))
width = 0.25

bars1 = ax2.bar(x - width, returns, width, label='年化收益', alpha=0.8)
bars2 = ax2.bar(x, volatility, width, label='年化波动', alpha=0.8)
bars3 = ax2.bar(x + width, sharpe, width, label='夏普比率', alpha=0.8)

ax2.set_ylabel('比率')
ax2.set_title('策略表现对比')
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

plt.tight_layout()
plt.savefig(f'{output_dir}/cover.png', dpi=300, bbox_inches='tight')
print(f"✓ 生成封面图: {output_dir}/cover.png")

# 生成图2: 宏观因子择时信号
fig2, ax = plt.subplots(figsize=(12, 6))

# 模拟宏观指标
dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
np.random.seed(123)

# CPI
cpi = 100 + np.cumsum(np.random.normal(0.2, 0.5, len(dates)))
# PMI
pmi = 50 + 5 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.normal(0, 1, len(dates))
# 利率
interest_rate = 3 + 2 * np.sin(np.linspace(0, 3*np.pi, len(dates))) + np.random.normal(0, 0.3, len(dates))

ax.plot(dates, cpi / 100, label='CPI (标准化)', linewidth=2, alpha=0.8)
ax.plot(dates, pmi / 50, label='PMI (标准化)', linewidth=2, alpha=0.8)
ax.plot(dates, interest_rate / 3, label='利率 (标准化)', linewidth=2, alpha=0.8)

# 添加择时信号区域
signal_periods = [(datetime(2020, 3, 1), datetime(2020, 6, 30)),
                  (datetime(2021, 6, 1), datetime(2021, 9, 30)),
                  (datetime(2022, 3, 1), datetime(2022, 8, 31)),
                  (datetime(2023, 6, 1), datetime(2023, 10, 31))]

for start, end in signal_periods:
    ax.axvspan(start, end, alpha=0.2, color='red', label='高暴露期' if start == signal_periods[0][0] else '')

ax.set_xlabel('日期')
ax.set_ylabel('标准化数值')
ax.set_title('宏观因子择时信号', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/macro-timing-signals.png', dpi=300, bbox_inches='tight')
print(f"✓ 生成宏观择时信号图: {output_dir}/macro-timing-signals.png")

# 生成图3: 因子相关性热图
fig3, ax = plt.subplots(figsize=(10, 8))

# 因子相关性矩阵
factors = ['市值', '价值', '动量', '质量', '低波', '成长']
correlation_matrix = np.array([
    [1.00, -0.15, 0.25, 0.10, -0.20, 0.30],
    [-0.15, 1.00, -0.10, 0.35, 0.15, -0.05],
    [0.25, -0.10, 1.00, 0.05, -0.30, 0.20],
    [0.10, 0.35, 0.05, 1.00, 0.25, 0.15],
    [-0.20, 0.15, -0.30, 0.25, 1.00, -0.10],
    [0.30, -0.05, 0.20, 0.15, -0.10, 1.00]
])

im = ax.imshow(correlation_matrix, cmap='RdBu_r', vmin=-1, vmax=1)

# 添加数值标注
for i in range(len(factors)):
    for j in range(len(factors)):
        text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                      ha='center', va='center', color='black', fontweight='bold')

ax.set_xticks(range(len(factors)))
ax.set_yticks(range(len(factors)))
ax.set_xticklabels(factors)
ax.set_yticklabels(factors)
ax.set_title('因子相关性矩阵', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='相关系数')

plt.tight_layout()
plt.savefig(f'{output_dir}/factor-correlation.png', dpi=300, bbox_inches='tight')
print(f"✓ 生成因子相关性图: {output_dir}/factor-correlation.png")

print("\n✅ 所有配图生成完成!")
print(f"输出目录: {output_dir}")
