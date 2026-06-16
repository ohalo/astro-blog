#!/usr/bin/env python3
"""
为因子拥挤度文章生成配图
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-crowding'
os.makedirs(output_dir, exist_ok=True)

# 图1: 因子拥挤度综合得分示例
fig, ax = plt.subplots(figsize=(12, 6))
dates = pd.date_range('2024-01-01', '2025-12-31', freq='D')
np.random.seed(42)
base_score = 0.3 + 0.1 * np.sin(np.linspace(0, 4*np.pi, len(dates)))
crowding_score = base_score + 0.2 * ((dates.month > 6) & (dates.month < 10))  # 模拟年中拥挤
crowding_score = crowding_score + np.random.randn(len(dates)) * 0.05
crowding_score = np.clip(crowding_score, 0, 1)

ax.plot(dates, crowding_score, linewidth=2, color='#2E86AB')
ax.axhline(y=0.6, color='orange', linestyle='--', linewidth=2, label='黄色预警 (0.6)')
ax.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='红色预警 (0.8)')
ax.fill_between(dates, 0.6, 0.8, alpha=0.2, color='orange')
ax.fill_between(dates, 0.8, 1.0, alpha=0.2, color='red')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('拥挤度得分', fontsize=12)
ax.set_title('因子拥挤度监测示例：综合得分时间序列', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/crowding_score_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 因子暴露集中度可视化 (HHI指数)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：正常分布的因子暴露
np.random.seed(42)
normal_exposure = np.random.normal(0, 1, 1000)
ax1.hist(normal_exposure, bins=50, alpha=0.7, color='#2E86AB', edgecolor='black')
ax1.set_title('正常状态：因子暴露分布分散', fontsize=12, fontweight='bold')
ax1.set_xlabel('因子暴露', fontsize=10)
ax1.set_ylabel('频数', fontsize=10)
ax1.axvline(x=np.mean(normal_exposure), color='red', linestyle='--', linewidth=2)

# 右图：拥挤状态的因子暴露 (偏度更高)
crowded_exposure = np.concatenate([
    np.random.normal(0, 0.5, 700),  # 70%股票集中
    np.random.normal(3, 1, 300)       # 30%股票极端暴露
])
ax2.hist(crowded_exposure, bins=50, alpha=0.7, color='#A23B72', edgecolor='black')
ax2.set_title('拥挤状态：因子暴露集中', fontsize=12, fontweight='bold')
ax2.set_xlabel('因子暴露', fontsize=10)
ax2.set_ylabel('频数', fontsize=10)
ax2.axvline(x=np.mean(crowded_exposure), color='red', linestyle='--', linewidth=2)

plt.tight_layout()
plt.savefig(f'{output_dir}/factor_exposure_concentration.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3: 因子收益率自相关性对比
fig, ax = plt.subplots(figsize=(10, 6))

lags = np.arange(1, 11)
normal_autocorr = np.exp(-lags / 5) * 0.3 + np.random.randn(10) * 0.05  # 正常：快速衰减
crowded_autocorr = 0.5 * np.ones(10) + np.random.randn(10) * 0.05        # 拥挤：高自相关

ax.plot(lags, normal_autocorr, marker='o', linewidth=2, markersize=8, 
        label='正常状态 (低自相关)', color='#2E86AB')
ax.plot(lags, crowded_autocorr, marker='s', linewidth=2, markersize=8, 
        label='拥挤状态 (高自相关)', color='#A23B72')

ax.set_xlabel('滞后阶数', fontsize=12)
ax.set_ylabel('自相关系数', fontsize=12)
ax.set_title('因子拥挤度信号：收益率自相关性对比', fontsize=14, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([-0.2, 0.8])

plt.tight_layout()
plt.savefig(f'{output_dir}/autocorrelation_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4: ETF资金净流入与因子收益
fig, ax1 = plt.subplots(figsize=(12, 6))

dates = pd.date_range('2024-01-01', '2025-06-30', freq='ME')
np.random.seed(42)

# 模拟ETF净流入 (百万元)
etf_inflow = 100 * np.random.randn(len(dates)) + 50 * (dates.month > 6)  # 年中涌入
etf_inflow = etf_inflow.cumsum()

# 模拟因子收益
factor_return = 0.002 * etf_inflow / 100 + 0.001 * np.random.randn(len(dates))
factor_cumret = (1 + factor_return).cumprod()

color = 'tab:blue'
ax1.set_xlabel('日期', fontsize=12)
ax1.set_ylabel('ETF净流入 (百万元)', color=color, fontsize=12)
ax1.plot(dates, etf_inflow, color=color, linewidth=2, label='ETF净流入')
ax1.tick_params(axis='y', labelcolor=color)
ax1.grid(True, alpha=0.3)

ax2 = ax1.twinx()
color = 'tab:red'
ax2.set_ylabel('因子累计收益', color=color, fontsize=12)
ax2.plot(dates, factor_cumret, color=color, linewidth=2, linestyle='--', label='因子收益')
ax2.tick_params(axis='y', labelcolor=color)

fig.suptitle('资金流与因子收益的关系：拥挤前的资金涌入', fontsize=14, fontweight='bold')
fig.tight_layout()
plt.savefig(f'{output_dir}/etf_flow_vs_return.png', dpi=300, bbox_inches='tight')
plt.close()

# 图5: 封面图 - 因子拥挤度概念图
fig, ax = plt.subplots(figsize=(12, 8))

# 创建概念图：展示"拥挤"vs"分散"
categories = ['低拥挤\n(分散)', '中度拥挤', '高拥挤\n(极度集中)']
values = [30, 60, 90]
colors = ['#2E86AB', '#F18F01', '#C73E1D']

bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2, width=0.6)

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val}', ha='center', va='bottom', fontsize=16, fontweight='bold')

# 添加解释文本
ax.text(0, 95, '✓ 因子溢价稳定', fontsize=11, color='green')
ax.text(1, 125, '⚠ 开始监测', fontsize=11, color='orange')
ax.text(2, 155, '✗ 立即降仓', fontsize=11, color='red')

ax.set_ylabel('拥挤度得分', fontsize=12)
ax.set_title('因子拥挤度监测与预警系统', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0, 170])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/cover.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ 已生成5张配图到 {output_dir}")
print("  - crowding_score_timeseries.png")
print("  - factor_exposure_concentration.png")
print("  - autocorrelation_comparison.png")
print("  - etf_flow_vs_return.png")
print("  - cover.png")
