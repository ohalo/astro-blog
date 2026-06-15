#!/usr/bin/env python3
"""
为第二篇文章生成配图：配对交易与协整分析
只使用已安装的包：numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建模拟数据
np.random.seed(42)

# 图1: 股票价格序列对比
fig, ax = plt.subplots(figsize=(14, 6))

# 生成模拟的价格数据
n_days = 1000
t = np.arange(n_days)

# 两只协整股票的价格
beta = 1.5
P2 = 50 + np.cumsum(np.random.normal(0, 0.5, n_days))  # 股票B
epsilon = np.random.normal(0, 2, n_days)  # 平稳残差
P1 = 60 + beta * (P2 - 50) / 50 + epsilon  # 股票A

ax.plot(t, P1, label='Stock A (e.g., Coca-Cola)', linewidth=2, color='#FF6B6B')
ax.plot(t, P2, label='Stock B (e.g., PepsiCo)', linewidth=2, color='#4ECDC4')
ax.set_title('Stock Price Series: Pair Trading Candidate', fontsize=16, fontweight='bold')
ax.set_xlabel('Trading Days')
ax.set_ylabel('Price ($)')
ax.legend(loc='best', fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/price_series.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 1 generated: price_series.png")
plt.close()

# 图2: 回测结果
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
fig.suptitle('Pair Trading Strategy: Backtest Results', fontsize=16, fontweight='bold')

# 生成模拟的价差和Z-Score
spread = np.random.normal(0, 1, n_days)
spread = np.cumsum(spread) * 0.1  # 带漂移的随机游走（模拟非平稳）
spread_mean = np.convolve(spread, np.ones(252)/252, mode='same')[:n_days]
spread_std = np.std(spread[-252:])  # 使用最后252天的标准差
z_score = (spread - spread_mean) / spread_std

# 1. 价差与Z-Score
ax1 = axes[0]
ax1.plot(t, spread, label='Spread', color='blue', alpha=0.7, linewidth=1.5)
ax1_twin = ax1.twinx()
ax1_twin.plot(t, z_score, label='Z-Score', color='red', alpha=0.7, linewidth=1.5)
ax1.set_ylabel('Spread', color='blue', fontsize=12)
ax1_twin.set_ylabel('Z-Score', color='red', fontsize=12)
ax1.set_title('(a) Spread and Z-Score', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1_twin.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1_twin.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax1_twin.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# 2. 组合价值
ax2 = axes[1]
portfolio_value = 100000 * (1 + np.cumsum(np.random.normal(0.0005, 0.005, n_days)))
ax2.plot(t, portfolio_value, label='Portfolio Value', linewidth=2, color='green')
ax2.set_title('(b) Portfolio Value Over Time', fontsize=14, fontweight='bold')
ax2.set_ylabel('Value ($)', fontsize=12)
ax2.grid(True, alpha=0.3)

# 3. 累计收益
ax3 = axes[2]
strategy_returns = np.random.normal(0.0005, 0.005, n_days)
cumulative_returns = np.cumprod(1 + strategy_returns)
benchmark_returns = np.random.normal(0.0003, 0.01, n_days)
benchmark_cumulative = np.cumprod(1 + benchmark_returns)

ax3.plot(t, cumulative_returns, label='Pair Trading Strategy', linewidth=2, color='green')
ax3.plot(t, benchmark_cumulative, label='Buy & Hold Benchmark', linewidth=2, 
         color='gray', alpha=0.7)
ax3.set_title('(c) Cumulative Returns: Strategy vs Benchmark', fontsize=14, fontweight='bold')
ax3.set_xlabel('Trading Days', fontsize=12)
ax3.set_ylabel('Cumulative Return', fontsize=12)
ax3.legend(loc='best', fontsize=11)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest_results.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 2 generated: backtest_results.png")
plt.close()

# 图3: 参数优化热力图
fig, ax = plt.subplots(figsize=(10, 6))

# 模拟参数优化结果
entry_z_values = [1.5, 2.0, 2.5]
exit_z_values = [0.5, 1.0, 1.5]
sharpe_matrix = np.array([
    [1.2, 1.5, 1.4],
    [1.6, 1.8, 1.7],
    [1.3, 1.5, 1.6]
])

im = ax.imshow(sharpe_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest')
ax.set_xticks(range(len(exit_z_values)))
ax.set_yticks(range(len(entry_z_values)))
ax.set_xticklabels(exit_z_values)
ax.set_yticklabels(entry_z_values)
ax.set_xlabel('Exit Z-Score', fontsize=12)
ax.set_ylabel('Entry Z-Score', fontsize=12)
ax.set_title('Parameter Optimization: Sharpe Ratio Heatmap', fontsize=14, fontweight='bold')

# 添加数值标签
for i in range(len(entry_z_values)):
    for j in range(len(exit_z_values)):
        ax.text(j, i, f'{sharpe_matrix[i, j]:.2f}',
               ha='center', va='center', color='black', fontsize=11, fontweight='bold')

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Sharpe Ratio', rotation=270, labelpad=20, fontsize=11)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/parameter_optimization.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 3 generated: parameter_optimization.png")
plt.close()

# 额外图: 协整关系示意图
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Cointegration Analysis: Key Concepts', fontsize=16, fontweight='bold')

# 1. 两个非平稳序列
ax1 = axes[0, 0]
series1 = np.cumsum(np.random.normal(0, 1, 500))
series2 = 1.5 * series1 + np.random.normal(0, 5, 500)
ax1.plot(series1, label='Stock A Price', linewidth=2)
ax1.plot(series2, label='Stock B Price', linewidth=2)
ax1.set_title('(a) Two Non-Stationary Series', fontsize=13, fontweight='bold')
ax1.set_xlabel('Time')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. 价差（非平稳）
ax2 = axes[0, 1]
spread_nonstationary = series1 - series2
ax2.plot(spread_nonstationary, linewidth=2, color='red')
ax2.set_title('(b) Spread (Non-Stationary)', fontsize=13, fontweight='bold')
ax2.set_xlabel('Time')
ax2.set_ylabel('Spread')
ax2.grid(True, alpha=0.3)

# 3. 协整的价差（平稳）
ax3 = axes[1, 0]
spread_stationary = np.random.normal(0, 1, 500)
ax3.plot(spread_stationary, linewidth=2, color='green')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axhline(y=2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.axhline(y=-2, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax3.set_title('(c) Spread (Stationary = Cointegrated)', fontsize=13, fontweight='bold')
ax3.set_xlabel('Time')
ax3.set_ylabel('Spread')
ax3.grid(True, alpha=0.3)

# 4. 直方图：价差分布
ax4 = axes[1, 1]
ax4.hist(spread_stationary, bins=30, edgecolor='black', alpha=0.7, color='green')
ax4.set_title('(d) Spread Distribution (Normal)', fontsize=13, fontweight='bold')
ax4.set_xlabel('Spread')
ax4.set_ylabel('Frequency')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_concepts.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 4 generated: cointegration_concepts.png")
plt.close()

print("\n✅ All images for Post 2 generated successfully!")
print("Location: /Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/")
