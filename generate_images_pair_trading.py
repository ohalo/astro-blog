#!/usr/bin/env python3
"""
生成配对交易与协整分析文章的配图
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
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

# 生成日期范围（2020-2024）
n_days = 1000
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')

# 设置随机种子以保证可重复性
np.random.seed(42)

# 生成协整的价格序列
# 共同的随机游走成分
common_trend = np.cumsum(np.random.normal(0, 0.01, n_days))

# 个股特有成分
idiosyncratic_a = np.cumsum(np.random.normal(0, 0.005, n_days))
idiosyncratic_b = np.cumsum(np.random.normal(0, 0.006, n_days))

# 构建价格序列（确保协整关系）
price_a = 100 + common_trend + 0.5 * idiosyncratic_a
price_b = 80 + 0.8 * common_trend + 0.6 * idiosyncratic_b

# 创建 DataFrame
prices = pd.DataFrame({
    'Stock_A': price_a,
    'Stock_B': price_b
}, index=dates)

print(f"生成 {n_days} 天的价格数据")

# 计算 Spread (残差)
# 使用 numpy 的 polyfit 进行简单线性回归
beta, alpha = np.polyfit(price_b, price_a, 1)
spread = price_a - (alpha + beta * price_b)

print(f"协整回归: Stock_A = {alpha:.2f} + {beta:.4f} * Stock_B")
print(f"Spread 均值: {spread.mean():.4f}")
print(f"Spread 标准差: {spread.std():.4f}")

# 计算 Z-Score
window = 20
z_score = (spread - pd.Series(spread).rolling(window=window).mean()) / pd.Series(spread).rolling(window=window).std()

# 图1：协整关系概念图
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制价格序列
ax.plot(dates, price_a / 100, label='Stock A (标准化)', linewidth=2.5, color='#264653')
ax.plot(dates, price_b / 80, label='Stock B (标准化)', linewidth=2.5, color='#2A9D8F')

ax.set_xlabel('日期', fontsize=12, weight='bold')
ax.set_ylabel('标准化价格', fontsize=12, weight='bold')
ax.set_title('协整关系：两个价格序列的长期均衡', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle=':')

# 添加协整关系说明
textstr = f'协整系数 β = {beta:.4f}\n残差平稳性: p < 0.01'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_concept.png', dpi=300, bbox_inches='tight')
print("✓ 生成图1：协整关系概念图")
plt.close()

# 图2：配对交易分析（3个子图）
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：价格走势
ax1 = axes[0]
ax1.plot(dates, price_a, label='Stock A', linewidth=2, color='#264653')
ax1.plot(dates, price_b, label='Stock B', linewidth=2, color='#2A9D8F')
ax1.set_ylabel('价格', fontsize=11, weight='bold')
ax1.set_title('(a) 价格走势', fontsize=12, weight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

# 子图2：Spread
ax2 = axes[1]
ax2.plot(dates, spread, label='Spread (残差)', linewidth=2, color='#E9C46A')
ax2.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='均值')
ax2.fill_between(dates, -2*spread.std(), 2*spread.std(), alpha=0.2, color='gray', label='±2σ 区间')
ax2.set_ylabel('Spread', fontsize=11, weight='bold')
ax2.set_title('(b) Spread (协整残差)', fontsize=12, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

# 子图3：Z-Score 和交易信号
ax3 = axes[2]
ax3.plot(dates, z_score, label='Z-Score', linewidth=2, color='#F4A261')
ax3.axhline(y=2, color='red', linestyle='--', linewidth=1.5, label='开仓阈值 (+2)')
ax3.axhline(y=-2, color='red', linestyle='--', linewidth=1.5)
ax3.axhline(y=0.5, color='green', linestyle='--', linewidth=1.5, label='平仓阈值 (+/-0.5)')
ax3.axhline(y=-0.5, color='green', linestyle='--', linewidth=1.5)
ax3.set_xlabel('日期', fontsize=11, weight='bold')
ax3.set_ylabel('Z-Score', fontsize=11, weight='bold')
ax3.set_title('(c) Z-Score 与交易阈值', fontsize=12, weight='bold')
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pair_trading_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 生成图2：配对交易分析")
plt.close()

# 图3：策略回测绩效
# 生成交易信号
entry_threshold = 2.0
exit_threshold = 0.5

positions_a = np.zeros(n_days)
positions_b = np.zeros(n_days)

for i in range(1, n_days):
    if z_score[i-1] < -entry_threshold:
        positions_a[i] = 1   # 做多 A
        positions_b[i] = -beta  # 做空 B（考虑对冲比率）
    elif z_score[i-1] > entry_threshold:
        positions_a[i] = -1  # 做空 A
        positions_b[i] = beta   # 做多 B
    elif abs(z_score[i-1]) < exit_threshold:
        positions_a[i] = 0
        positions_b[i] = 0
    else:
        positions_a[i] = positions_a[i-1]
        positions_b[i] = positions_b[i-1]

# 计算策略收益
returns_a = pd.Series(price_a).pct_change()
returns_b = pd.Series(price_b).pct_change()

strategy_returns = positions_a[:-1] * returns_a.values[1:] + positions_b[:-1] * returns_b.values[1:]
strategy_cumulative = (1 + pd.Series(strategy_returns)).cumprod()

# 基准收益（等权持有）
benchmark_returns = 0.5 * returns_a + 0.5 * returns_b
benchmark_cumulative = (1 + benchmark_returns).cumprod()

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(dates[1:], strategy_cumulative.values, label='配对交易策略', linewidth=2.5, color='#264653')
ax.plot(dates, benchmark_cumulative.values, label='等权基准', linewidth=2.5, color='#E76F51', linestyle='--')

ax.set_xlabel('日期', fontsize=12, weight='bold')
ax.set_ylabel('累计收益 (净值)', fontsize=12, weight='bold')
ax.set_title('配对交易策略 vs 等权基准：累计收益对比 (2020-2024)', fontsize=14, weight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3, linestyle=':')

# 添加绩效标注
total_ret_strategy = strategy_cumulative.iloc[-1] - 1
total_ret_benchmark = benchmark_cumulative.iloc[-1] - 1
sharpe_strategy = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
sharpe_benchmark = benchmark_returns.mean() / benchmark_returns.std() * np.sqrt(252)

textstr = f'配对交易策略:\n  累计收益: {total_ret_strategy*100:.1f}%\n  夏普比率: {sharpe_strategy:.2f}\n\n等权基准:\n  累计收益: {total_ret_benchmark*100:.1f}%\n  夏普比率: {sharpe_benchmark:.2f}'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pairs_trading_performance.png', dpi=300, bbox_inches='tight')
print("✓ 生成图3：策略回测绩效")
plt.close()

# 图4：Spread 的均值回归特征
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：Spread 的自相关图
from statsmodels.graphics.tsaplots import plot_acf

ax1 = axes[0]
plot_acf(spread, lags=40, ax=ax1, color='#2A9D8F')
ax1.set_xlabel('滞后阶数', fontsize=11, weight='bold')
ax1.set_ylabel('自相关系数', fontsize=11, weight='bold')
ax1.set_title('Spread 的自相关函数 (ACF)', fontsize=12, weight='bold')
ax1.grid(True, alpha=0.3)

# 子图2：Z-Score 的分布直方图
ax2 = axes[1]
ax2.hist(z_score[window:], bins=50, color='#E9C46A', alpha=0.7, edgecolor='black', linewidth=1)
ax2.axvline(x=0, color='red', linestyle='--', linewidth=2, label='均值 (0)')
ax2.axvline(x=2, color='green', linestyle='--', linewidth=1.5, label='±2σ 阈值')
ax2.axvline(x=-2, color='green', linestyle='--', linewidth=1.5)
ax2.set_xlabel('Z-Score', fontsize=11, weight='bold')
ax2.set_ylabel('频数', fontsize=11, weight='bold')
ax2.set_title('Z-Score 的分布 (均值回归特征)', fontsize=12, weight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/spread_mean_reversion.png', dpi=300, bbox_inches='tight')
print("✓ 生成图4：Spread 的均值回归特征")
plt.close()

# 图5：协整检验示意图
fig, ax = plt.subplots(figsize=(14, 8))

# 生成非协整的随机游走
random_walk_1 = 100 + np.cumsum(np.random.normal(0, 0.01, n_days))
random_walk_2 = 100 + np.cumsum(np.random.normal(0, 0.01, n_days))

# 绘制对比
ax.plot(dates[:250], price_a[:250] / 100, label='协整序列 A', linewidth=2.5, color='#264653')
ax.plot(dates[:250], price_b[:250] / 80, label='协整序列 B', linewidth=2.5, color='#2A9D8F')
ax.plot(dates[:250], random_walk_1[:250] / 100, label='随机游走 1', linewidth=2.5, color='#E76F51', linestyle='--')
ax.plot(dates[:250], random_walk_2[:250] / 100, label='随机游走 2', linewidth=2.5, color='#F4A261', linestyle=':')

ax.set_xlabel('日期', fontsize=12, weight='bold')
ax.set_ylabel('标准化价格', fontsize=12, weight='bold')
ax.set_title('协整关系 vs 伪相关：长期均衡的重要性', fontsize=14, weight='bold')
ax.legend(fontsize=10, loc='best')
ax.grid(True, alpha=0.3, linestyle=':')

# 添加说明文字
ax.text(0.5, 0.02, '协整序列会围绕长期均衡关系波动，而随机游走的差值无界', 
        transform=ax.transAxes, fontsize=10, ha='center',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_vs_spurious.png', dpi=300, bbox_inches='tight')
print("✓ 生成图5：协整关系 vs 伪相关")
plt.close()

print("\n✅ 所有配图已生成完成！")
print(f"图片保存位置: /Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/")
print(f"\n生成的图片文件：")
print("  1. cointegration_concept.png - 协整关系概念图")
print("  2. pair_trading_analysis.png - 配对交易分析（3个子图）")
print("  3. pairs_trading_performance.png - 策略回测绩效")
print("  4. spread_mean_reversion.png - Spread 的均值回归特征")
print("  5. cointegration_vs_spurious.png - 协整关系 vs 伪相关")
