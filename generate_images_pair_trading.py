#!/usr/bin/env python3
"""
生成配对交易与协整分析文章的配图
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = '/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration'
os.makedirs(output_dir, exist_ok=True)

# 生成示意图1: 配对交易价差与交易信号
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
fig.suptitle('配对交易：价差分析与交易信号', fontsize=14, fontweight='bold')

# 模拟两只协整股票的价格
np.random.seed(42)
n_days = 252

# 股票 A 的价格（随机游走）
stock_a = 100 * np.cumprod(1 + np.random.normal(0.0005, 0.015, n_days))

# 股票 B 的价格（与 A 协整，但加入均值回归的价差）
spread_mean = 10
spread = spread_mean + np.random.normal(0, 2, n_days)
spread = spread + 0.05 * (spread_mean - spread)  # 均值回归

stock_b = stock_a * 0.8 + spread

dates = pd.date_range('2024-01-01', periods=n_days, freq='D')

# 图1：两只股票价格
ax1.plot(dates, stock_a, label='股票 A', linewidth=2, alpha=0.7)
ax1.plot(dates, stock_b, label='股票 B', linewidth=2, alpha=0.7)
ax1.set_ylabel('价格')
ax1.set_title('(a) 配对股票价格走势')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 图2：价差（Spread）
spread_actual = stock_b - stock_a * 0.8
ax2.plot(dates, spread_actual, label='实际价差', linewidth=2, color='green')
ax2.axhline(y=spread_mean, color='red', linestyle='--', label='长期均值')
ax2.axhline(y=spread_mean + 2, color='orange', linestyle=':', label='+2 STD')
ax2.axhline(y=spread_mean - 2, color='orange', linestyle=':', label='-2 STD')
ax2.set_ylabel('价差')
ax2.set_title('(b) 价差（Spread）走势')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 图3：Z-Score 与交易信号
z_score = (spread_actual - spread_actual.mean()) / spread_actual.std()
ax3.plot(dates, z_score, label='Z-Score', linewidth=2, color='purple')
ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='入场阈值')
ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, label='平仓线')
ax3.fill_between(dates, 2, z_score, where=(z_score > 2), alpha=0.3, color='red', label='做空信号')
ax3.fill_between(dates, -2, z_score, where=(z_score < -2), alpha=0.3, color='green', label='做多信号')
ax3.set_ylabel('Z-Score')
ax3.set_xlabel('日期')
ax3.set_title('(c) Z-Score 与交易信号')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/pair-trading-overview.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成示意图2: 协整检验示意图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('协整关系检验', fontsize=14, fontweight='bold')

# 左图：协整序列（价差平稳）
np.random.seed(123)
n = 500
t = np.arange(n)

# 随机游走
random_walk1 = np.cumsum(np.random.normal(0, 1, n))
random_walk2 = 0.8 * random_walk1 + np.random.normal(0, 0.5, n)

# 价差
spread_example = random_walk2 - random_walk1 * 0.8

ax1.plot(t, random_walk1, label='序列 X', linewidth=2)
ax1.plot(t, random_walk2, label='序列 Y', linewidth=2)
ax1.set_xlabel('时间')
ax1.set_ylabel('值')
ax1.set_title('(a) 协整序列（价差平稳）')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 右图：非协整序列（价差发散）
np.random.seed(456)
random_walk3 = np.cumsum(np.random.normal(0, 1, n))
random_walk4 = np.cumsum(np.random.normal(0, 1, n))  # 独立的随机游走

ax2.plot(t, random_walk3, label='序列 X', linewidth=2)
ax2.plot(t, random_walk4, label='序列 Y', linewidth=2)
ax2.set_xlabel('时间')
ax2.set_ylabel('值')
ax2.set_title('(b) 非协整序列（价差发散）')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/cointegration-test.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成示意图3: 配对交易累积收益
fig, ax = plt.subplots(figsize=(12, 6))

# 模拟配对交易策略收益
np.random.seed(789)
n_days = 504  # 2年

# 假设我们运行配对交易策略
# 做多 Z-score < -2 的价差，做空 Z-score > 2 的价差
z_score_sim = np.random.normal(0, 1, n_days)
z_score_sim = z_score_sim + 0.9 * np.roll(z_score_sim, 1)  # 增加持续性

# 交易信号
position = np.zeros(n_days)
position[z_score_sim < -2] = 1   # 做多价差
position[z_score_sim > 2] = -1    # 做空价差
position[(z_score_sim > -0.5) & (z_score_sim < 0.5)] = 0  # 平仓

# 策略收益（简化：假设每次交易平均赚 2%）
strategy_returns = position * np.random.normal(0.002, 0.01, n_days)
strategy_returns = pd.Series(strategy_returns)

# 基准收益（市场）
market_returns = np.random.normal(0.0003, 0.01, n_days)
market_cumret = np.cumprod(1 + market_returns)

# 策略累积收益
strategy_cumret = np.cumprod(1 + strategy_returns)

dates_sim = pd.date_range('2024-01-01', periods=n_days, freq='D')

ax.plot(dates_sim, strategy_cumret, label='配对交易策略', linewidth=2)
ax.plot(dates_sim, market_cumret, label='市场基准', linewidth=2, alpha=0.7)
ax.set_ylabel('累积净值')
ax.set_xlabel('日期')
ax.set_title('配对交易策略累积收益 vs 市场基准')
ax.legend()
ax.grid(True, alpha=0.3)

# 添加标注：交易次数
n_trades = (position != 0).sum()
ax.text(0.02, 0.98, f'交易次数: {n_trades}', transform=ax.transAxes,
        fontsize=10, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(f'{output_dir}/pair-trading-performance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 配对交易配图已生成：")
print(f"  - {output_dir}/pair-trading-overview.png")
print(f"  - {output_dir}/cointegration-test.png")
print(f"  - {output_dir}/pair-trading-performance.png")
