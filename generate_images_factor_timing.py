#!/usr/bin/env python3
"""
生成因子择时文章的配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-timing', exist_ok=True)

# 生成日期范围（2015-2025）
dates = pd.date_range(start='2015-01-01', end='2025-12-31', freq='D')
dates = dates[dates.dayofweek < 5]  # 只保留工作日

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成因子收益率数据（模拟真实因子表现）
n_days = len(dates)

# 价值因子（HML）- 有周期性
hml = np.random.normal(0.0003, 0.012, n_days)
hml = hml + 0.001 * np.sin(2 * np.pi * np.arange(n_days) / 252)  # 添加周期性

# 动量因子（UMD）- 趋势性强
umd = np.random.normal(0.0004, 0.015, n_days)
umd = umd + 0.0002 * np.cumsum(np.random.choice([-1, 1], n_days))  # 添加趋势

# 低波因子（BAB）- 负与市场相关性
bab = np.random.normal(0.0002, 0.008, n_days)
market_ret = np.random.normal(0.0003, 0.012, n_days)
bab = bab - 0.3 * market_ret  # 与市场负相关

# 质量因子（QMJ）- 稳健
qmj = np.random.normal(0.0003, 0.010, n_days)

# 构建 DataFrame
factor_rets = pd.DataFrame({
    'HML': hml,
    'UMD': umd,
    'BAB': bab,
    'QMJ': qmj
}, index=dates)

# 生成动态权重（模拟因子择时策略）
weights = pd.DataFrame(index=dates, columns=factor_rets.columns)

# 初始等权
weights.iloc[0] = [0.25, 0.25, 0.25, 0.25]

# 模拟动态权重变化
for i in range(1, len(dates)):
    # 根据"市场状态"调整权重
    if i < n_days // 3:  # 第一阶段：牛市，超配动量
        weights.iloc[i] = [0.15, 0.45, 0.20, 0.20]
    elif i < 2 * n_days // 3:  # 第二阶段：震荡市，超配低波和质量
        weights.iloc[i] = [0.20, 0.20, 0.35, 0.25]
    else:  # 第三阶段：熊市/恢复期，超配价值
        weights.iloc[i] = [0.40, 0.15, 0.20, 0.25]

# 添加一些随机扰动
weights = weights + np.random.normal(0, 0.02, weights.shape)
weights = weights.clip(0.05, 0.55)  # 限制权重范围
weights = weights.div(weights.sum(axis=1), axis=0)  # 归一化

# 计算策略收益
strategy_returns = (weights.shift(1).fillna(0.25) * factor_rets).sum(axis=1)
equal_weight_returns = factor_rets.mean(axis=1)

# 计算累计收益
cumulative_strategy = (1 + strategy_returns).cumprod()
cumulative_benchmark = (1 + equal_weight_returns).cumprod()

print(f"生成 {len(dates)} 个交易日的数据")
print(f"策略最终累计收益: {cumulative_strategy.iloc[-1]:.2f}")
print(f"基准最终累计收益: {cumulative_benchmark.iloc[-1]:.2f}")

# 图1：因子择时策略框架示意图
fig, ax = plt.subplots(figsize=(14, 8))

# 创建流程图
boxes = {
    '宏观指标': (0.1, 0.8, 0.15, 0.1),
    '市场状态': (0.1, 0.6, 0.15, 0.1),
    '技术信号': (0.1, 0.4, 0.15, 0.1),
    '机器学习\n模型': (0.4, 0.6, 0.2, 0.15),
    '因子权重\n输出': (0.7, 0.6, 0.15, 0.1),
    '组合构建': (0.9, 0.6, 0.15, 0.1),
}

for text, (x, y, w, h) in boxes.items():
    ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, color='lightblue', ec='black', linewidth=2))
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', fontsize=12, weight='bold')

# 添加箭头
arrows = [
    ((0.25, 0.85), (0.4, 0.65)),
    ((0.25, 0.65), (0.4, 0.65)),
    ((0.25, 0.45), (0.4, 0.65)),
    ((0.6, 0.65), (0.7, 0.65)),
    ((0.85, 0.65), (0.9, 0.65)),
]

for (x1, y1), (x2, y2) in arrows:
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax.set_xlim(0, 1.1)
ax.set_ylim(0.3, 1.0)
ax.axis('off')
ax.set_title('因子择时策略框架', fontsize=16, weight='bold', pad=20)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_timing_framework.png', dpi=300, bbox_inches='tight')
print("✓ 生成图1：因子择时框架")
plt.close()

# 图2：累计收益对比
fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(dates, cumulative_strategy.values, label='因子择时策略', linewidth=2.5, color='#2E86AB')
ax.plot(dates, cumulative_benchmark.values, label='等权基准', linewidth=2.5, color='#A23B72', linestyle='--')

ax.set_xlabel('日期', fontsize=12, weight='bold')
ax.set_ylabel('累计收益 (净值)', fontsize=12, weight='bold')
ax.set_title('因子择时策略 vs 等权基准：累计收益对比 (2015-2025)', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle=':')

# 添加绩效标注
annual_ret_strategy = strategy_returns.mean() * 252
annual_ret_benchmark = equal_weight_returns.mean() * 252
sharpe_strategy = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
sharpe_benchmark = equal_weight_returns.mean() / equal_weight_returns.std() * np.sqrt(252)

textstr = f'因子择时策略:\n  年化收益: {annual_ret_strategy*100:.1f}%\n  夏普比率: {sharpe_strategy:.2f}\n\n等权基准:\n  年化收益: {annual_ret_benchmark*100:.1f}%\n  夏普比率: {sharpe_benchmark:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_timing_performance.png', dpi=300, bbox_inches='tight')
print("✓ 生成图2：累计收益对比")
plt.close()

# 图3：因子权重动态变化
fig, ax = plt.subplots(figsize=(14, 7))

colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261']
weight_data = weights.apply(pd.to_numeric)

for i, col in enumerate(weight_data.columns):
    ax.plot(dates, weight_data[col].values, label=col, linewidth=2, color=colors[i])

ax.set_xlabel('日期', fontsize=12, weight='bold')
ax.set_ylabel('因子权重', fontsize=12, weight='bold')
ax.set_title('动态因子权重随时间的变化 (2015-2025)', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper right')
ax.grid(True, alpha=0.3, linestyle=':')
ax.set_ylim(0, 0.6)

# 添加阶段标注
ax.axvspan(dates[0], dates[n_days//3], alpha=0.2, color='green', label='牛市阶段')
ax.axvspan(dates[n_days//3], dates[2*n_days//3], alpha=0.2, color='yellow', label='震荡市阶段')
ax.axvspan(dates[2*n_days//3], dates[-1], alpha=0.2, color='red', label='熊市阶段')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_weights_evolution.png', dpi=300, bbox_inches='tight')
print("✓ 生成图3：因子权重动态变化")
plt.close()

# 图4：各因子单独表现
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
axes = axes.flatten()

for i, col in enumerate(factor_rets.columns):
    ax = axes[i]
    cumulative = (1 + factor_rets[col]).cumprod()
    ax.plot(dates, cumulative.values, linewidth=2, color=colors[i])
    ax.set_title(f'{col} 因子累计收益', fontsize=12, weight='bold')
    ax.set_xlabel('日期', fontsize=10)
    ax.set_ylabel('累计收益', fontsize=10)
    ax.grid(True, alpha=0.3, linestyle=':')
    
    # 添加年化收益标注
    annual_ret = factor_rets[col].mean() * 252
    ax.text(0.05, 0.95, f'年化收益: {annual_ret*100:.1f}%', 
            transform=ax.transAxes, fontsize=9, 
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.suptitle('各因子单独表现 (2015-2025)', fontsize=16, weight='bold')
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_individual_performance.png', dpi=300, bbox_inches='tight')
print("✓ 生成图4：各因子单独表现")
plt.close()

print("\n✅ 所有配图已生成完成！")
print(f"图片保存位置: /Users/halo/workspace/astro-blog/public/images/factor-timing/")
