#!/usr/bin/env python3
"""
生成因子拥挤度文章的配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
n_stocks = 500

# 模拟因子暴露（BP因子）
factor_data = pd.DataFrame(
    np.random.normal(0, 1, (n_stocks, len(dates))),
    index=[f'STOCK_{i}' for i in range(n_stocks)],
    columns=dates
)

# 添加拥挤度效应：2022年下半年开始价值因子拥挤
crowding_start = '2022-07-01'
crowding_idx = factor_data.columns >= crowding_start
factor_data.loc[:, crowding_idx] += np.random.normal(0.5, 0.3, 
                                                     (n_stocks, crowding_idx.sum()))

# 模拟收益率数据
return_data = pd.DataFrame(
    np.random.normal(0.01, 0.05, (n_stocks, len(dates))),
    index=factor_data.index,
    columns=dates
)

# 计算拥挤度得分（简化版）
crowding_scores = pd.DataFrame(index=factor_data.index, columns=factor_data.columns)
for i, date in enumerate(factor_data.columns):
    if i < 12:
        crowding_scores[date] = 0.3  # 前期低拥挤度
    else:
        # 后期拥挤度上升
        base_score = 0.3 + (i - 12) * 0.02 if i >= 30 else 0.3
        noise = np.random.normal(0, 0.05, n_stocks)
        crowding_scores[date] = np.clip(base_score + noise, 0, 1)

mean_scores = crowding_scores.mean(axis=0)
factor_returns = return_data[factor_data > 0].mean(axis=0)

# 图1: 拥挤度分析综合图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('因子拥挤度监测分析', fontsize=18, fontweight='bold')

# 1. 拥挤度得分时间序列
ax1 = axes[0, 0]
ax1.plot(mean_scores.index, mean_scores.values, linewidth=2.5, color='darkred', alpha=0.8)
ax1.axvline(pd.Timestamp(crowding_start), color='gray', linestyle='--', 
            linewidth=2, label='拥挤度开始上升')
ax1.fill_between(mean_scores.index, 0, mean_scores.values, alpha=0.3, color='darkred')
ax1.set_title('因子拥挤度得分时序', fontsize=14, pad=10)
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('拥挤度得分', fontsize=12)
ax1.legend(loc='upper left', fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 1)

# 2. 因子暴露分布变化
ax2 = axes[0, 1]
pre_crowding = factor_data.loc[:, '2021-12-31'].dropna()
post_crowding = factor_data.loc[:, '2023-12-31'].dropna()

ax2.hist(pre_crowding, bins=50, alpha=0.6, label='拥挤前 (2021-12)', 
         density=True, color='steelblue', edgecolor='black', linewidth=0.5)
ax2.hist(post_crowding, bins=50, alpha=0.6, label='拥挤后 (2023-12)', 
         density=True, color='crimson', edgecolor='black', linewidth=0.5)
ax2.set_title('因子暴露分布变化', fontsize=14, pad=10)
ax2.set_xlabel('因子暴露', fontsize=12)
ax2.set_ylabel('密度', fontsize=12)
ax2.legend(loc='upper right', fontsize=10)
ax2.grid(True, alpha=0.3)

# 3. 拥挤度 vs 因子收益率散点图
ax3 = axes[1, 0]
scatter = ax3.scatter(mean_scores.values, factor_returns.values, 
                      alpha=0.6, c=mean_scores.values, cmap='Reds', s=50)
ax3.set_title('拥挤度 vs 因子收益率', fontsize=14, pad=10)
ax3.set_xlabel('拥挤度得分', fontsize=12)
ax3.set_ylabel('因子收益率', fontsize=12)
ax3.grid(True, alpha=0.3)

# 添加回归线
z = np.polyfit(mean_scores.values[~np.isnan(mean_scores.values)], 
                factor_returns.values[~np.isnan(factor_returns.values)], 1)
p = np.poly1d(z)
ax3.plot(mean_scores.values, p(mean_scores.values), "r--", 
         color='darkred', linewidth=2, label=f'回归线 (斜率={z[0]:.3f})')
ax3.legend(loc='upper right', fontsize=10)

# 添加颜色条
cbar = plt.colorbar(scatter, ax=ax3)
cbar.set_label('拥挤度', fontsize=10)

# 4. 拥挤度预警信号
ax4 = axes[1, 1]
warning_threshold = 0.8
warning_signal = (mean_scores > mean_scores.quantile(warning_threshold)).astype(int)
ax4.plot(mean_scores.index, warning_signal.values, color='orange', 
         linewidth=3, label='预警信号')
ax4.fill_between(mean_scores.index, 0, warning_signal.values, 
                 alpha=0.4, color='orange', label='预警区域')
ax4.axhline(y=warning_threshold, color='red', linestyle=':', 
            linewidth=1.5, label=f'阈值 ({warning_threshold})')
ax4.set_title(f'拥挤度预警信号 ({int(warning_threshold*100)}%分位数)', fontsize=14, pad=10)
ax4.set_xlabel('日期', fontsize=12)
ax4.set_ylabel('预警信号', fontsize=12)
ax4.set_ylim(-0.1, 1.2)
ax4.legend(loc='upper left', fontsize=10)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_analysis.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图1: crowding_analysis.png")
plt.close()

# 图2: 各因子拥挤度时序对比
fig2, ax2 = plt.subplots(figsize=(14, 7))

# 模拟多个因子的拥挤度
factor_names = ['价值', '动量', '市值', '质量', '波动率']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
dates_sample = dates[::3]  # 降采样以加快绘图

for i, (factor, color) in enumerate(zip(factor_names, colors)):
    # 为每个因子生成不同的拥挤度模式
    base_trend = np.linspace(0.2, 0.7, len(dates_sample))
    seasonal = 0.1 * np.sin(2 * np.pi * np.arange(len(dates_sample)) / 12)
    noise = np.random.normal(0, 0.05, len(dates_sample))
    
    if factor == '动量':
        base_trend = np.linspace(0.3, 0.9, len(dates_sample))  # 动量因子更拥挤
    elif factor == '波动率':
        base_trend = np.linspace(0.15, 0.4, len(dates_sample))  # 波动率因子较不拥挤
    
    crowding = np.clip(base_trend + seasonal + noise * (i+1) * 0.5, 0, 1)
    ax2.plot(dates_sample, crowding, label=factor, linewidth=2.5, 
             color=color, alpha=0.8)

ax2.set_title('各因子拥挤度时序对比', fontsize=16, fontweight='bold', pad=15)
ax2.set_xlabel('日期', fontsize=13)
ax2.set_ylabel('拥挤度得分', fontsize=13)
ax2.legend(loc='upper left', fontsize=12, framealpha=0.9)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.set_ylim(0, 1)
ax2.set_xlim(dates_sample[0], dates_sample[-1])

# 添加当前时间标记
current_date = pd.Timestamp('2025-12-31')
ax2.axvline(current_date, color='black', linestyle=':', linewidth=2, 
            label='当前时间', alpha=0.5)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/factor_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图2: factor_comparison.png")
plt.close()

# 图3: 因子拥挤度热力图（额外配图）
fig3, ax3 = plt.subplots(figsize=(14, 8))

# 创建热力图数据
heatmap_data = pd.DataFrame(
    np.random.rand(20, 12) * 0.5 + 0.25,  # 基础拥挤度 0.25-0.75
    index=[f'股票{i:03d}' for i in range(20)],
    columns=[f'{i}月' for i in range(1, 13)]
)

# 添加一些高拥挤度区域
heatmap_data.iloc[0:5, 6:9] = 0.85  # 某些股票在特定月份拥挤度高
heatmap_data.iloc[10:15, 3:6] = 0.90  # 另一批股票

# 使用matplotlib的imshow绘制热力图
im = ax3.imshow(heatmap_data.values, cmap='RdYlGn_r', aspect='auto', 
                vmin=0, vmax=1)

# 设置刻度标签
ax3.set_xticks(range(len(heatmap_data.columns)))
ax3.set_xticklabels(heatmap_data.columns)
ax3.set_yticks(range(len(heatmap_data.index)))
ax3.set_yticklabels(heatmap_data.index)

# 添加数值标注
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        text = ax3.text(j, i, f'{heatmap_data.iloc[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=8)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax3)
cbar.set_label('拥挤度得分', fontsize=11)

ax3.set_title('因子拥挤度热力图（示例：20只股票×12个月）', 
              fontsize=15, fontweight='bold', pad=15)
ax3.set_xlabel('月份', fontsize=12)
ax3.set_ylabel('股票代码', fontsize=12)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_heatmap.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图3: crowding_heatmap.png")
plt.close()

print("\n✅ 所有配图生成完成！")
print(f"位置: /Users/halo/workspace/astro-blog/public/images/factor-crowding/")
