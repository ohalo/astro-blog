#!/usr/bin/env python3
"""
为因子拥挤度文章生成配图
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-crowding', exist_ok=True)

# 图1：因子拥挤度监测仪表盘
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('因子拥挤度多维度监测仪表盘', fontsize=16, fontweight='bold')

# 子图1：综合拥挤度得分趋势
dates = pd.date_range('2025-01-01', '2026-06-19', freq='D')
composite_score = 0.3 + 0.4 * np.sin(np.linspace(0, 4*np.pi, len(dates))) + np.random.uniform(-0.05, 0.05, len(dates))
composite_score = np.clip(composite_score, 0, 1)

axes[0, 0].plot(dates, composite_score, linewidth=2, color='#2E86AB')
axes[0, 0].axhline(y=0.4, color='green', linestyle='--', linewidth=2, label='正常阈值')
axes[0, 0].axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='危险阈值')
axes[0, 0].fill_between(dates, 0, composite_score, alpha=0.3, color='#2E86AB')
axes[0, 0].set_title('综合拥挤度得分', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('得分 (0-1)', fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：各维度得分热力图
dimensions = ['资金流', '离散度', '换手率', '自相关']
scores_matrix = np.random.uniform(0.2, 0.8, (6, 4))
im = axes[0, 1].imshow(scores_matrix, cmap='RdYlGn_r', aspect='auto')
axes[0, 1].set_xticks(range(len(dimensions)))
axes[0, 1].set_xticklabels(dimensions, fontsize=9)
axes[0, 1].set_yticks(range(6))
axes[0, 1].set_yticklabels([f'T-{i}' for i in range(5, -1, -1)], fontsize=9)
axes[0, 1].set_title('各维度得分热力图', fontsize=12, fontweight='bold')
plt.colorbar(im, ax=axes[0, 1], fraction=0.046, pad=0.04)

# 子图3：因子收益率离散度
dispersion = 0.5 + 0.3 * np.cos(np.linspace(0, 3*np.pi, len(dates))) + np.random.uniform(-0.05, 0.05, len(dates))
dispersion = np.clip(dispersion, 0.1, 1.0)

axes[1, 0].plot(dates, dispersion, linewidth=2, color='#A23B72')
axes[1, 0].axhline(y=0.3, color='orange', linestyle='--', linewidth=2, label='拥挤阈值')
axes[1, 0].set_title('因子收益率离散度', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('离散度', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：预警信号时间线
alert_dates = ['2025-03-15', '2025-07-20', '2025-11-10', '2026-03-05']
alert_levels = ['WARNING', 'DANGER', 'WARNING', 'DANGER']
colors = ['orange', 'red', 'orange', 'red']

for date, level, color in zip(alert_dates, alert_levels, colors):
    axes[1, 1].axvline(x=pd.Timestamp(date), color=color, linewidth=3, alpha=0.7)
    axes[1, 1].text(pd.Timestamp(date), 0.5, level, rotation=90, 
                      verticalalignment='center', fontsize=9, color=color, fontweight='bold')

axes[1, 1].set_xlim(pd.Timestamp('2025-01-01'), pd.Timestamp('2026-06-19'))
axes[1, 1].set_ylim(0, 1)
axes[1, 1].set_title('预警信号时间线', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('预警级别', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/dashboard.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2：2019年动量因子崩盘示意图
fig, ax = plt.subplots(figsize=(14, 8))

# 模拟动量因子指数表现
dates_mtum = pd.date_range('2019-01-01', '2020-12-31', freq='D')
mtum_returns = np.random.normal(0.0008, 0.015, len(dates_mtum))
mtum_returns[200:220] = -0.02  # 模拟崩盘期
mtum_cumulative = (1 + pd.Series(mtum_returns)).cumprod()

ax.plot(dates_mtum, mtum_cumulative, linewidth=3, color='#2E86AB', label='动量因子指数')
ax.axvspan(pd.Timestamp('2019-08-01'), pd.Timestamp('2019-09-30'), 
           alpha=0.3, color='red', label='动量崩盘期')
ax.axhline(y=1.0, color='black', linestyle='-', linewidth=1)
ax.set_title('2019年动量因子崩盘示意图', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('累计收益 (基准=1.0)', fontsize=12)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

# 添加注释
ax.annotate('崩盘开始\n收益回撤-10%', 
            xy=(pd.Timestamp('2019-08-15'), 1.15),
            xytext=(pd.Timestamp('2019-06-01'), 1.25),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=11, color='red', fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/momentum_crash.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3：因子拥挤度与收益的负相关关系
fig, ax = plt.subplots(figsize=(12, 8))

# 模拟数据
np.random.seed(42)
n_samples = 100
crowding = np.random.uniform(0, 1, n_samples)
expected_return = 0.1 - 0.08 * crowding + np.random.normal(0, 0.02, n_samples)

ax.scatter(crowding, expected_return, alpha=0.6, s=50, color='#A23B72', edgecolors='black', linewidth=0.5)
ax.set_xlabel('因子拥挤度得分', fontsize=12)
ax.set_ylabel('预期年化收益', fontsize=12)
ax.set_title('因子拥挤度与预期收益的负相关关系', fontsize=14, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3)

# 添加拟合线
z = np.polyfit(crowding, expected_return, 1)
p = np.poly1d(z)
ax.plot(np.sort(crowding), p(np.sort(crowding)), color='red', linewidth=2, linestyle='--', label=f'拟合线: y = {z[0]:.3f}x + {z[1]:.3f}')
ax.legend(fontsize=11)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_return_relation.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4：动态因子配置权重变化
fig, ax = plt.subplots(figsize=(14, 8))

dates_alloc = pd.date_range('2025-01-01', '2026-06-19', freq='D')
n_factors = 5
factor_names = ['Momentum', 'Value', 'Size', 'Quality', 'LowVol']

# 模拟权重变化
weights = np.random.dirichlet(np.ones(n_factors), len(dates_alloc))
weights = pd.DataFrame(weights, index=dates_alloc, columns=factor_names)

# 模拟拥挤度上升时权重调整
crowding_signal = 0.3 + 0.5 * (dates_alloc > pd.Timestamp('2025-09-01')).astype(int) + np.random.uniform(-0.1, 0.1, len(dates_alloc))
crowding_signal = np.clip(crowding_signal, 0, 1)

for i, factor in enumerate(factor_names):
    ax.plot(dates_alloc, weights[factor], linewidth=2, label=factor, alpha=0.7)

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('因子权重', fontsize=12)
ax.set_title('动态因子配置权重变化（考虑拥挤度）', fontsize=14, fontweight='bold', pad=20)
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/dynamic_allocation.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 因子拥挤度文章配图生成完成！")
print("生成文件：")
print("  1. dashboard.png - 多维度监测仪表盘")
print("  2. momentum_crash.png - 2019年动量因子崩盘示意图")
print("  3. crowding_return_relation.png - 拥挤度与收益关系")
print("  4. dynamic_allocation.png - 动态因子配置")
