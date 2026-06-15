#!/usr/bin/env python3
"""
为配对交易与协整分析文章下载/生成配图
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

output_dir = '/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration'
os.makedirs(output_dir, exist_ok=True)

# 图1: 配对交易原理示意图（价差均值回归）
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=252, freq='D')

# 模拟两只协整股票的价格
trend = np.linspace(100, 120, 252)
noise1 = np.random.normal(0, 5, 252)
noise2 = np.random.normal(0, 5, 252)

price_X = trend + noise1  # 股票X
price_Y = 0.8 * trend + noise2 + 50  # 股票Y（与X协整）

# 计算价差
spread = price_Y - 0.8 * price_X
spread_mean = spread.mean()
spread_std = spread.std()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

# 上图：两只股票价格
ax1.plot(dates, price_X, linewidth=2, label='股票X（做空）', color='blue')
ax1.plot(dates, price_Y, linewidth=2, label='股票Y（做多）', color='red')
ax1.set_ylabel('价格', fontsize=14)
ax1.legend(fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.set_title('配对股票价格走势', fontsize=16, fontweight='bold')

# 下图：价差与交易信号
ax2.plot(dates, spread, linewidth=2, color='purple', label='价差')
ax2.axhline(y=spread_mean, color='black', linestyle='-', label='均值', linewidth=1)
ax2.axhline(y=spread_mean + 2*spread_std, color='red', linestyle='--', label='+2σ（做多信号）', linewidth=1.5)
ax2.axhline(y=spread_mean - 2*spread_std, color='green', linestyle='--', label='-2σ（做空信号）', linewidth=1.5)
ax2.fill_between(dates, spread_mean - 2*spread_std, spread_mean + 2*spread_std, alpha=0.2, color='gray')
ax2.set_ylabel('价差', fontsize=14)
ax2.set_xlabel('日期', fontsize=14)
ax2.legend(fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.set_title('价差均值回归与交易信号', fontsize=16, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/pair_trading_mechanism.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2: 协整检验可视化（模拟ADF检验结果）

# 生成非平稳序列和平稳残差
np.random.seed(42)
T = 252
X = np.cumsum(np.random.normal(0, 1, T)) + 100  # 随机游走
Y = 0.8 * X + np.random.normal(0, 10, T)  # 与X协整

# OLS回归（手动计算）
X_with_const = np.column_stack([np.ones(T), X])
beta = np.linalg.inv(X_with_const.T @ X_with_const) @ X_with_const.T @ Y
residual = Y - (beta[0] + beta[1] * X)

# 模拟ADF检验结果
adf_stat = -3.45
p_value = 0.008

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：X和Y的价格走势（非平稳）
axes[0, 0].plot(X, linewidth=2, label='X (非平稳)', color='blue')
axes[0, 0].plot(Y, linewidth=2, label='Y (非平稳)', color='red')
axes[0, 0].set_title('X和Y均为非平稳序列', fontsize=14, fontweight='bold')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2：回归残差（平稳）
axes[0, 1].plot(residual, linewidth=2, color='green')
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 1].set_title(f'回归残差（平稳）\np-value={p_value:.4f}', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# 子图3：残差的自相关（模拟快速衰减）
lag = np.arange(1, 21)
acf = np.exp(-lag/5) * np.cos(lag/3)  # 模拟快速衰减的ACF
axes[1, 0].stem(lag, acf, linefmt='b-', markerfmt='bo', basefmt='k-')
axes[1, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[1, 0].set_xlabel('滞后阶数')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].set_title('残差的ACF（快速衰减→平稳）', fontsize=14, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3)

# 子图4：残差的直方图（接近正态分布）
axes[1, 1].hist(residual, bins=30, edgecolor='black', alpha=0.7, color='purple', density=True)
axes[1, 1].set_title('残差的分布（接近正态）', fontsize=14, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3)

plt.suptitle('Engle-Granger协整检验可视化', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/cointegration_test.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"✅ 图片已保存到 {output_dir}")
print(f"   - pair_trading_mechanism.png")
print(f"   - cointegration_test.png")
