"""
生成配对交易文章的配图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1: hero.png - 配对交易概念图
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#0a0e27')

# 创建背景
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 添加标题
ax.text(5, 9, 'Pairs Trading', fontsize=48, fontweight='bold', 
        ha='center', va='center', color='#00d4ff')

ax.text(5, 8, '配对交易与协整分析', fontsize=28, 
        ha='center', va='center', color='#7b68ee')

# 绘制两个股票的价格走势（示意）
t = np.linspace(0, 10, 100)
price_a = 5 + 2 * np.sin(t) + 0.5 * np.random.randn(100)
price_b = 5 + 2 * np.sin(t + 0.5) + 0.5 * np.random.randn(100)

ax.plot(t[:50], price_a[:50], color='#00d4ff', linewidth=3, label='Stock A', alpha=0.8)
ax.plot(t[:50], price_b[:50], color='#7b68ee', linewidth=3, label='Stock B', alpha=0.8)

# 绘制价差（Spread）
spread = price_a - price_b
ax.plot(t[:50], spread[:50] + 2, color='#ff6b6b', linewidth=2, label='Spread', alpha=0.9)

# 添加均值线
ax.axhline(y=2, color='white', linestyle='--', linewidth=1, alpha=0.5)

# 添加图例
ax.text(0.5, 7, 'Stock A: 工商银行', fontsize=14, color='#00d4ff', 
        bbox=dict(boxstyle='round', facecolor='#0a0e27', alpha=0.8))
ax.text(0.5, 6.5, 'Stock B: 建设银行', fontsize=14, color='#7b68ee',
        bbox=dict(boxstyle='round', facecolor='#0a0e27', alpha=0.8))
ax.text(0.5, 6, 'Spread: 均值回归', fontsize=14, color='#ff6b6b',
        bbox=dict(boxstyle='round', facecolor='#0a0e27', alpha=0.8))

ax.set_title('Pairs Trading Strategy', fontsize=20, fontweight='bold', pad=20, color='white')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/hero.png', 
            dpi=150, bbox_inches='tight', facecolor='#0a0e27')
plt.close()

print("✓ 生成 hero.png")

# 图2: equity-curve.png - 配对交易策略收益曲线
fig, ax = plt.subplots(figsize=(14, 8))

# 生成模拟的权益曲线
np.random.seed(42)
n_days = 1500  # 约6年交易日
dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')

# 生成配对策略的收益（均值回归特性）
returns = np.random.normal(0.0005, 0.005, n_days)  # 日收益：均值0.05%，标准差0.5%

# 添加一些回撤期间（模拟协整关系破裂）
drawdown_periods = [
    (200, 250),  # 2020年Q4
    (600, 650),  # 2022年Q2
    (1000, 1050)  # 2024年Q1
]

for start, end in drawdown_periods:
    returns[start:end] = np.random.normal(-0.001, 0.008, end - start)

# 计算累计收益
cumulative_returns = pd.Series(np.cumprod(1 + returns), index=dates)
benchmark_returns = pd.Series(np.cumprod(1 + np.random.normal(0.0003, 0.006, n_days)), index=dates)  # 沪深300作为基准

# 绘制权益曲线
ax.plot(dates, cumulative_returns, linewidth=2, color='#00d4ff', label='Pairs Trading', alpha=0.9)
ax.plot(dates, benchmark_returns, linewidth=2, color='#7b68ee', label='Benchmark (HS300)', alpha=0.7)

# 添加回撤标注
for start, end in drawdown_periods:
    ax.axvspan(dates[start], dates[end], alpha=0.2, color='red', label='Drawdown' if start == 200 else '')

# 添加关键指标
total_return = (cumulative_returns.iloc[-1] - 1) * 100
sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252)
max_dd = (cumulative_returns / cumulative_returns.expanding().max() - 1).min() * 100

ax.text(0.02, 0.98, f'Total Return: {total_return:.1f}%\nSharpe Ratio: {sharpe:.2f}\nMax Drawdown: {max_dd:.1f}%', 
        transform=ax.transAxes, fontsize=14, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Cumulative Return', fontsize=14)
ax.set_title('Pairs Trading Strategy: Equity Curve (2020-2025)', fontsize=18, fontweight='bold', pad=20)
ax.legend(fontsize=12, loc='best')
ax.grid(True, alpha=0.3)

# 设置y轴为百分比格式
from matplotlib.ticker import PercentFormatter
ax.yaxis.set_major_formatter(PercentFormatter(1.0))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/equity-curve.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("✓ 生成 equity-curve.png")

print("\n所有配图生成完成！")
