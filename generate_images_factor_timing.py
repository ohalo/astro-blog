#!/usr/bin/env python3
"""
生成因子择时文章的配图
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
output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-timing'
os.makedirs(output_dir, exist_ok=True)

# 生成示意图1: 因子轮动示意图
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('因子择时：动态调整因子暴露示意图', fontsize=14, fontweight='bold')

# 模拟因子收益时序
np.random.seed(42)
dates = pd.date_range('2023-01-01', '2025-12-31', freq='ME')
n_months = len(dates)

# 价值因子和动量因子的收益（负相关）
value_returns = np.random.normal(0.01, 0.05, n_months)
momentum_returns = -0.3 * value_returns + np.random.normal(0.01, 0.03, n_months)

# 累积收益
value_cum = np.cumprod(1 + value_returns)
momentum_cum = np.cumprod(1 + momentum_returns)

ax1.plot(dates, value_cum, label='价值因子', linewidth=2)
ax1.plot(dates, momentum_cum, label='动量因子', linewidth=2)
ax1.set_ylabel('累积净值')
ax1.set_title('(a) 价值因子 vs 动量因子累积收益')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 模拟动态权重调整
value_weight = np.zeros(n_months)
momentum_weight = np.zeros(n_months)

# 简单的择时策略：根据过去3个月收益调整权重
for i in range(3, n_months):
    value_recent = np.mean(value_returns[i-3:i])
    momentum_recent = np.mean(momentum_returns[i-3:i])
    
    if value_recent > momentum_recent:
        value_weight[i] = 0.7
        momentum_weight[i] = 0.3
    else:
        value_weight[i] = 0.3
        momentum_weight[i] = 0.7

ax2.plot(dates[3:], value_weight[3:], label='价值因子权重', linewidth=2)
ax2.plot(dates[3:], momentum_weight[3:], label='动量因子权重', linewidth=2)
ax2.set_ylabel('因子权重')
ax2.set_xlabel('日期')
ax2.set_title('(b) 动态因子权重调整')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0, 1])

plt.tight_layout()
plt.savefig(f'{output_dir}/factor-timing-overview.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成示意图2: 因子择时策略收益对比
fig, ax = plt.subplots(figsize=(12, 6))

# 模拟不同策略的累积收益
np.random.seed(123)
n_days = 252 * 3  # 3年
dates_daily = pd.date_range('2023-01-01', periods=n_days, freq='D')

# 市场收益
market_returns = np.random.normal(0.0003, 0.01, n_days)
market_cum = np.cumprod(1 + market_returns)

# 固定权重组合
fixed_returns = 0.5 * np.random.normal(0.0004, 0.008, n_days) + 0.5 * np.random.normal(0.0003, 0.009, n_days)
fixed_cum = np.cumprod(1 + fixed_returns)

# 动态择时组合（在因子表现好时提高权重）
timing_returns = np.zeros(n_days)
for i in range(20, n_days):
    if i % 63 < 31:  # 约一半时间使用因子1
        timing_returns[i] = np.random.normal(0.0005, 0.007)
    else:
        timing_returns[i] = np.random.normal(0.0004, 0.008)

timing_cum = np.cumprod(1 + timing_returns)

ax.plot(dates_daily, market_cum, label='市场基准', linewidth=2, alpha=0.7)
ax.plot(dates_daily, fixed_cum, label='固定权重组合', linewidth=2, alpha=0.7)
ax.plot(dates_daily, timing_cum, label='动态因子择时', linewidth=2, alpha=0.7)

ax.set_ylabel('累积净值')
ax.set_xlabel('日期')
ax.set_title('因子择时策略 vs 固定权重 vs 市场基准')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/factor-timing-performance.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 因子择时配图已生成：")
print(f"  - {output_dir}/factor-timing-overview.png")
print(f"  - {output_dir}/factor-timing-performance.png")
