#!/usr/bin/env python3
"""
为配对交易与协整分析文章生成配图
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

output_dir = '/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration'
os.makedirs(output_dir, exist_ok=True)

# 图1: 协整 vs 相关性对比
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 左图：高相关但非协整 (两个独立随机游走)
np.random.seed(42)
n = 500
rw1 = np.cumsum(np.random.randn(n))
rw2 = np.cumsum(np.random.randn(n))  # 独立的随机游走

ax1.plot(range(n), rw1, linewidth=2, label='股票A', color='#2E86AB')
ax1.plot(range(n), rw2, linewidth=2, label='股票B', color='#A23B72')
ax1.set_title('高相关 ≠ 协整\n(两个独立随机游走)', fontsize=12, fontweight='bold')
ax1.set_xlabel('时间', fontsize=10)
ax1.set_ylabel('价格', fontsize=10)
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：协整关系 (Y = 0.5*X + 平稳残差)
x = np.cumsum(0.01 * np.random.randn(n))  # 随机游走
residual = 0.5 * np.sin(np.linspace(0, 4*np.pi, n)) + 0.1 * np.random.randn(n)
y = 0.5 * x + residual  # 协整关系

ax2.plot(range(n), x, linewidth=2, label='股票X', color='#2E86AB')
ax2.plot(range(n), y, linewidth=2, label='股票Y', color='#A23B72')
ax2.set_title('协整关系\n(价差围绕均值波动)', fontsize=12, fontweight='bold')
ax2.set_xlabel('时间', fontsize=10)
ax2.set_ylabel('价格', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/cointegration_vs_correlation.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 价差均值回归可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# 生成协整配对的价格和价差
np.random.seed(42)
n = 500
x = 100 + np.cumsum(0.0005 * np.random.randn(n))  # 股票X价格
residual = 2 * np.sin(np.linspace(0, 8*np.pi, n)) + 0.5 * np.random.randn(n)
y = 0.8 * x + residual  # 股票Y价格
spread = y - 0.8 * x  # 价差

dates = pd.date_range('2024-01-01', periods=n, freq='D')

# 上图：两只股票价格
ax1.plot(dates, x, linewidth=2, label='股票X (600519.SH)', color='#2E86AB')
ax1.plot(dates, y, linewidth=2, label='股票Y (000858.SZ)', color='#A23B72')
ax1.set_ylabel('价格', fontsize=12)
ax1.set_title('配对交易示例：茅台 vs 五粮液', fontsize=14, fontweight='bold')
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

# 下图：价差及阈值
ax2.plot(dates, spread, linewidth=2, color='#F18F01', label='价差')
ax2.axhline(y=spread.mean(), color='black', linestyle='-', linewidth=1.5, label='均值')
ax2.axhline(y=spread.mean() + 2*spread.std(), color='red', linestyle='--', 
             linewidth=2, label='+2σ (做空信号)')
ax2.axhline(y=spread.mean() - 2*spread.std(), color='green', linestyle='--', 
             linewidth=2, label='-2σ (做多信号)')
ax2.axhline(y=spread.mean() + 0.5*spread.std(), color='gray', linestyle=':', 
             linewidth=1.5, label='+0.5σ (平仓)')
ax2.axhline(y=spread.mean() - 0.5*spread.std(), color='gray', linestyle=':', 
             linewidth=1.5)

ax2.set_xlabel('日期', fontsize=12)
ax2.set_ylabel('价差', fontsize=12)
ax2.legend(loc='upper right')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/spread_mean_reversion.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3: Z-Score交易信号示意图
fig, ax = plt.subplots(figsize=(12, 6))

# 生成Z-Score序列
np.random.seed(42)
n = 300
z_score = np.cumsum(0.05 * np.random.randn(n))  # 模拟Z-Score随机游走
dates = pd.date_range('2024-07-01', periods=n, freq='D')

# 绘制Z-Score
ax.plot(dates, z_score, linewidth=2, color='#2E86AB', label='Z-Score')

# 添加阈值线
ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2, label='做空阈值 (+2σ)')
ax.axhline(y=-2.0, color='green', linestyle='--', linewidth=2, label='做多阈值 (-2σ)')
ax.axhline(y=0.5, color='gray', linestyle=':', linewidth=1.5, label='平仓阈值 (+0.5σ)')
ax.axhline(y=-0.5, color='gray', linestyle=':', linewidth=1.5)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

# 标记交易信号
enter_long = (z_score < -2.0) & (np.concatenate([[False], z_score[:-1] >= -2.0]))
enter_short = (z_score > 2.0) & (np.concatenate([[False], z_score[:-1] <= 2.0]))
exit_signal = (np.abs(z_score) < 0.5) & (np.concatenate([[False], np.abs(z_score[:-1]) >= 0.5]))

ax.scatter(dates[enter_long], z_score[enter_long], 
            color='green', s=100, marker='^', label='做多信号', zorder=5)
ax.scatter(dates[enter_short], z_score[enter_short], 
            color='red', s=100, marker='v', label='做空信号', zorder=5)
ax.scatter(dates[exit_signal], z_score[exit_signal], 
            color='gray', s=100, marker='o', label='平仓信号', zorder=5)

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('Z-Score', fontsize=12)
ax.set_title('配对交易信号：基于Z-Score的阈值法', fontsize=14, fontweight='bold')
ax.legend(loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/zscore_trading_signals.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4: 配对交易累积收益曲线
fig, ax = plt.subplots(figsize=(12, 6))

# 模拟策略收益
np.random.seed(42)
n = 500
daily_return = 0.0003 + 0.005 * np.random.randn(n)  # 日收益约0.03%，波动0.5%
cumulative_return = (1 + daily_return).cumprod()

dates = pd.date_range('2024-01-01', periods=n, freq='D')

ax.plot(dates, cumulative_return, linewidth=2.5, color='#2E86AB', label='配对交易策略')

# 添加基准 (假设买入持有)
benchmark_return = (1 + 0.0002 + 0.008 * np.random.randn(n)).cumprod()
ax.plot(dates, benchmark_return, linewidth=2, color='gray', linestyle='--', label='基准 (买入持有)')

ax.set_xlabel('日期', fontsize=12)
ax.set_ylabel('累计收益', fontsize=12)
ax.set_title('配对交易策略vs基准：累计收益曲线', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加关键指标标注
total_return = (cumulative_return[-1] - 1) * 100
sharpe = daily_return.mean() / daily_return.std() * np.sqrt(252)
ax.text(0.02, 0.98, f'总收益: {total_return:.1f}%\n夏普比率: {sharpe:.2f}', 
         transform=ax.transAxes, fontsize=11, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/cumulative_returns.png', dpi=300, bbox_inches='tight')
plt.close()

# 图5: 封面图 - 配对交易概念
fig, ax = plt.subplots(figsize=(12, 8))

# 创建概念图：展示"协整配对"+"均值回归"
categories = ['无协整\n(价格发散)', '弱协整\n(缓慢回归)', '强协整\n(快速回归)']
values = [30, 60, 90]
colors = ['#C73E1D', '#F18F01', '#2E86AB']

bars = ax.bar(categories, values, color=colors, edgecolor='black', linewidth=2, width=0.6)

# 添加数值标签
for bar, val in zip(bars, values):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height + 2,
            f'{val}', ha='center', va='bottom', fontsize=16, fontweight='bold')

# 添加解释文本
ax.text(0, 95, '✗ 不适合配对', fontsize=11, color='red')
ax.text(1, 125, '⚠ 谨慎使用', fontsize=11, color='orange')
ax.text(2, 155, '✓ 优质配对', fontsize=11, color='green')

ax.set_ylabel('协整得分', fontsize=12)
ax.set_title('配对交易：协整关系质量评估', fontsize=16, fontweight='bold', pad=20)
ax.set_ylim([0, 170])
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{output_dir}/cover.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✓ 已生成5张配图到 {output_dir}")
print("  - cointegration_vs_correlation.png")
print("  - spread_mean_reversion.png")
print("  - zscore_trading_signals.png")
print("  - cumulative_returns.png")
print("  - cover.png")
