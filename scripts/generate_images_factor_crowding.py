#!/usr/bin/env python3
"""
为因子拥挤度文章生成配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-crowding', exist_ok=True)

# 图1: 因子拥挤度生命周期
fig, ax = plt.subplots(figsize=(12, 6))

stages = ['低拥挤度\n(因子发现期)', '中等拥挤度\n(因子普及期)', 
          '高拥挤度\n(因子饱和期)', '拥挤释放\n(因子失效/反转)']
x_pos = np.arange(len(stages))
crowding_scores = [20, 50, 85, 40]

bars = ax.bar(x_pos, crowding_scores, color=['green', 'yellow', 'red', 'orange'], 
              alpha=0.7, edgecolor='black', linewidth=2)

ax.axhline(y=50, color='red', linestyle='--', linewidth=2, label='警戒线 (50)')
ax.axhline(y=70, color='darkred', linestyle='--', linewidth=2, label='危险线 (70)')

ax.set_xticks(x_pos)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel('拥挤度评分', fontsize=12, fontweight='bold')
ax.set_title('因子拥挤度的生命周期', fontsize=14, fontweight='bold', pad=20)
ax.set_ylim(0, 100)
ax.legend(loc='upper right', fontsize=10)
ax.grid(axis='y', alpha=0.3)

# 添加数值标签
for bar, score in zip(bars, crowding_scores):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{score}', ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图2: ETF资金流入与因子收益的关系
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

# 生成模拟数据
dates = pd.date_range('2020-01-01', '2026-06-15', freq='ME')  # ME = Month End
n_months = len(dates)

np.random.seed(42)
etf_inflow = np.cumsum(np.random.randn(n_months) * 0.02 + 0.01)
factor_return = np.cumsum(np.random.randn(n_months) * 0.03 + 0.005 - 
                          0.1 * (etf_inflow > 0.15).astype(float))

ax1.plot(dates, etf_inflow * 100, 'b-', linewidth=2.5, label='ETF资金流入 (%)')
ax1.axhline(y=10, color='orange', linestyle='--', linewidth=1.5, label='拥挤警戒线')
ax1.fill_between(dates, 0, etf_inflow * 100, alpha=0.3, color='blue')
ax1.set_ylabel('ETF资金流入 (%)', fontsize=11, fontweight='bold')
ax1.set_title('因子ETF资金流入', fontsize=13, fontweight='bold')
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.plot(dates, factor_return * 100, 'g-', linewidth=2.5, label='因子累计收益 (%)')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.fill_between(dates, 0, factor_return * 100, 
                  where=(factor_return > 0), alpha=0.3, color='green', label='正收益')
ax2.fill_between(dates, 0, factor_return * 100, 
                  where=(factor_return <= 0), alpha=0.3, color='red', label='负收益')
ax2.set_ylabel('因子累计收益 (%)', fontsize=11, fontweight='bold')
ax2.set_xlabel('日期', fontsize=11, fontweight='bold')
ax2.set_title('因子收益表现', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/etf_flow_vs_return.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

# 图3: 因子相关性突变监测
fig, ax = plt.subplots(figsize=(12, 6))

# 模拟相关性数据
dates_corr = pd.date_range('2023-01-01', '2026-06-15', freq='ME')  # ME = Month End
n = len(dates_corr)

# 相关性在某一时刻开始上升（拥挤信号）
base_corr = 0.3
correlation = base_corr + np.linspace(0, 0.4, n) + np.random.randn(n) * 0.05

ax.plot(dates_corr, correlation, 'purple', linewidth=2.5, label='因子间平均相关性')
ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='高拥挤阈值 (0.5)')
ax.fill_between(dates_corr, 0.3, correlation, alpha=0.3, color='purple')

# 标记拥挤区域
high_crowding_start = dates_corr[correlation > 0.5][0]
ax.axvline(x=high_crowding_start, color='red', linewidth=2, linestyle=':', 
           label=f'拥挤开始: {high_crowding_start.strftime("%Y-%m")}')

ax.set_xlabel('日期', fontsize=12, fontweight='bold')
ax.set_ylabel('平均配对相关性', fontsize=12, fontweight='bold')
ax.set_title('因子相关性突变监测（拥挤度警示指标）', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/correlation_breakdown.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 因子拥挤度文章配图生成完成！")
print("   - cover.jpg: 封面图（拥挤度生命周期）")
print("   - etf_flow_vs_return.png: ETF资金流与收益关系")
print("   - correlation_breakdown.png: 因子相关性突变")
