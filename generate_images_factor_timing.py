"""
为因子择时文章生成配图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import matplotlib

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('default')

# 手动设置样式
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = '#f8f9fa'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['grid.linestyle'] = '--'

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-timing', exist_ok=True)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2015-01-01', '2025-12-31', freq='ME')
n_periods = len(dates)

# 模拟因子收益
factor_returns = pd.DataFrame({
    '价值因子': np.random.normal(0.008, 0.04, n_periods),
    '动量因子': np.random.normal(0.007, 0.05, n_periods),
    '低波因子': np.random.normal(0.006, 0.03, n_periods),
    '质量因子': np.random.normal(0.007, 0.035, n_periods),
    '规模因子': np.random.normal(0.006, 0.045, n_periods)
}, index=dates)

# 添加一些周期性模式
for i in range(n_periods):
    if i % 24 < 12:  # 前两年价值表现好
        factor_returns.iloc[i, 0] += 0.002
    if i % 36 < 18:  # 前18个月动量表现好
        factor_returns.iloc[i, 1] += 0.001

# 计算累积收益
cumulative_returns = (1 + factor_returns).cumprod()

# 图1: 因子累积收益对比
fig, ax = plt.subplots(figsize=(14, 8))
for column in cumulative_returns.columns:
    ax.plot(cumulative_returns.index, cumulative_returns[column], 
            linewidth=2.5, label=column, alpha=0.8)

ax.set_xlabel('日期', fontsize=14, fontweight='bold')
ax.set_ylabel('累积收益', fontsize=14, fontweight='bold')
ax.set_title('各因子累积收益表现（2015-2025）', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 添加注释
ax.text(0.02, 0.98, '数据来源：模拟数据\n注：实际策略中使用真实因子收益', 
        transform=ax.transAxes, fontsize=10, 
        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_cumulative_returns.png', 
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# 图2: 因子择时vs等权基准
np.random.seed(123)
timing_ret = pd.Series(np.random.normal(0.010, 0.04, n_periods), index=dates)
benchmark_ret = pd.Series(np.random.normal(0.008, 0.045, n_periods), index=dates)

timing_cum = (1 + timing_ret).cumprod()
benchmark_cum = (1 + benchmark_ret).cumprod()

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(timing_cum.index, timing_cum.values, linewidth=3, 
        label='因子择时策略', color='#2E86AB', alpha=0.9)
ax.plot(benchmark_cum.index, benchmark_cum.values, linewidth=3, 
        label='等权基准', color='#A23B72', alpha=0.9)

ax.set_xlabel('日期', fontsize=14, fontweight='bold')
ax.set_ylabel('累积收益', fontsize=14, fontweight='bold')
ax.set_title('因子择时策略 vs 等权基准（2015-2025）', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 添加绩效标注
final_timing = timing_cum.iloc[-1]
final_bench = benchmark_cum.iloc[-1]
ax.annotate(f'因子择时: {final_timing:.2f}x', 
            xy=(timing_cum.index[-1], final_timing),
            xytext=(-100, 20), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#2E86AB',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=2))
ax.annotate(f'等权基准: {final_bench:.2f}x', 
            xy=(benchmark_cum.index[-1], final_bench),
            xytext=(-100, -30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#A23B72',
            arrowprops=dict(arrowstyle='->', color='#A23B72', lw=2))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/timing_vs_benchmark.png', 
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# 图3: 因子权重动态变化
weights = pd.DataFrame({
    '价值因子': np.random.dirichlet(np.array([2, 1.5, 1, 1.2, 1]), n_periods).T[0],
    '动量因子': np.random.dirichlet(np.array([1, 2, 1.5, 1, 1.2]), n_periods).T[1],
    '低波因子': np.random.dirichlet(np.array([1.2, 1, 2, 1.5, 1]), n_periods).T[2],
    '质量因子': np.random.dirichlet(np.array([1, 1.2, 1, 2, 1.5]), n_periods).T[3],
    '规模因子': np.random.dirichlet(np.array([1.5, 1, 1.2, 1, 2]), n_periods).T[4],
}, index=dates)

# 添加一些趋势
for i in range(n_periods):
    if i < n_periods // 3:  # 前半段价值权重高
        weights.iloc[i, 0] += 0.1
    if i > n_periods // 2:  # 后半段质量权重高
        weights.iloc[i, 3] += 0.1
weights = weights.div(weights.sum(axis=1), axis=0)  # 重新归一化

fig, ax = plt.subplots(figsize=(14, 8))
colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
weights.plot(ax=ax, linewidth=2.5, color=colors, alpha=0.8)

ax.set_xlabel('日期', fontsize=14, fontweight='bold')
ax.set_ylabel('因子权重', fontsize=14, fontweight='bold')
ax.set_title('因子权重动态变化（因子择时策略）', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=11, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim([0, 0.5])
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_weights_dynamic.png', 
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# 图4: 热力图 - 宏观变量与因子收益的相关性
np.random.seed(456)
macro_vars = pd.DataFrame({
    'GDP增长率': np.random.normal(6, 1, n_periods),
    '通胀率': np.random.normal(2, 0.5, n_periods),
    '期限利差': np.random.normal(1, 0.3, n_periods),
    '信用利差': np.random.normal(0.8, 0.2, n_periods),
    'VIX': np.random.uniform(10, 30, n_periods)
}, index=dates)

# 计算相关性矩阵（模拟）
correlation_matrix = pd.DataFrame({
    '价值因子': [0.3, -0.2, 0.4, 0.1, -0.1],
    '动量因子': [-0.2, 0.1, -0.3, -0.2, 0.3],
    '低波因子': [-0.1, 0.3, -0.2, 0.4, 0.5],
    '质量因子': [0.2, 0.4, 0.1, 0.5, 0.3],
    '规模因子': [0.1, -0.1, 0.2, -0.1, 0.0]
}, index=['GDP增长率', '通胀率', '期限利差', '信用利差', 'VIX'])

# 使用matplotlib手动绘制热力图
fig, ax = plt.subplots(figsize=(12, 8))

# 创建颜色映射
from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list('RdBu_r', ['#d73027', '#f7f7f7', '#0571b0'], N=256)

# 绘制热力图
im = ax.imshow(correlation_matrix.values, cmap=cmap, aspect='auto', vmin=-1, vmax=1)

# 设置刻度标签
ax.set_xticks(np.arange(len(correlation_matrix.columns)))
ax.set_yticks(np.arange(len(correlation_matrix.index)))
ax.set_xticklabels(correlation_matrix.columns, fontsize=12, rotation=45, ha='right')
ax.set_yticklabels(correlation_matrix.index, fontsize=12)

# 添加数值标注
for i in range(len(correlation_matrix.index)):
    for j in range(len(correlation_matrix.columns)):
        text = ax.text(j, i, f'{correlation_matrix.iloc[i, j]:.2f}',
                      ha='center', va='center', color='black', fontsize=11, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label('相关系数', fontsize=14, fontweight='bold')

ax.set_title('宏观变量与因子收益的相关性', fontsize=16, fontweight='bold', pad=20)
ax.set_xlabel('因子', fontsize=14, fontweight='bold')
ax.set_ylabel('宏观变量', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/macro_factor_correlation.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图5: Cover image - 因子择时概念图
fig, ax = plt.subplots(figsize=(16, 9), facecolor='#1a1a2e')

# 创建概念图
categories = ['价值', '动量', '低波', '质量', '规模']
values_before = [20, 30, 15, 25, 10]
values_after = [25, 20, 30, 20, 5]

x = np.arange(len(categories))
width = 0.35

bars1 = ax.bar(x - width/2, values_before, width, label='择时前', 
               color='#E63946', alpha=0.7, edgecolor='white', linewidth=2)
bars2 = ax.bar(x + width/2, values_after, width, label='择时后', 
               color='#06FFA5', alpha=0.7, edgecolor='white', linewidth=2)

ax.set_xlabel('因子类型', fontsize=16, fontweight='bold', color='white')
ax.set_ylabel('权重 (%)', fontsize=16, fontweight='bold', color='white')
ax.set_title('因子择时：动态调整因子暴露', fontsize=20, fontweight='bold', 
             color='white', pad=30)
ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=14, color='white')
ax.legend(fontsize=14, loc='upper right', framealpha=0.9)
ax.set_facecolor('#1a1a2e')
fig.patch.set_facecolor('#1a1a2e')
ax.tick_params(colors='white', labelsize=12)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')
ax.grid(True, alpha=0.3, linestyle='--', color='gray')
ax.set_ylim([0, 40])

# 添加数值标注
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height:.0f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold',
                    color='white')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print("✅ 因子择时文章配图生成完成！")
print("生成了5张图片：")
print("  1. factor_cumulative_returns.png - 各因子累积收益")
print("  2. timing_vs_benchmark.png - 择时策略 vs 基准")
print("  3. factor_weights_dynamic.png - 因子权重动态变化")
print("  4. macro_factor_correlation.png - 宏观变量与因子相关性")
print("  5. cover.jpg - 文章封面图")
