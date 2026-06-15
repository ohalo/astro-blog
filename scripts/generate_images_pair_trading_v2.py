#!/usr/bin/env python3
"""
为配对交易与协整分析文章生成配图（无外部依赖版本）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

# 生成模拟的配对交易数据
np.random.seed(42)
n = 500
t = np.arange(n)

# 生成协整序列（模拟）
X = 50 + np.cumsum(np.random.randn(n) * 0.5 + 0.01)  # 随机游走 + 向上漂移
beta = 1.5
alpha = 2.0
epsilon = np.random.randn(n) * 2  # 平稳残差
Y = alpha + beta * X + epsilon  # Y和X协整

dates = pd.date_range('2022-01-01', periods=n, freq='B')  # 工作日

# 图1: 配对交易原理图（封面）
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(dates, X, 'b-', linewidth=2.5, label='资产 X (如：可口可乐 KO)', alpha=0.8)
ax.plot(dates, Y, 'r-', linewidth=2.5, label='资产 Y (如：百事可乐 PEP)', alpha=0.8)

# 标记偏离和回归
divergence_start = 100
divergence_end = 200
ax.axvspan(dates[divergence_start], dates[divergence_end], 
           alpha=0.2, color='yellow', label='偏离期（套利机会）')
ax.text(dates[divergence_start + 20], np.max(Y) * 0.92, 
        '价差扩大\n做空Y，做多X', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
        fontsize=10, ha='center', weight='bold')

convergence_start = 350
convergence_end = 450
ax.axvspan(dates[convergence_start], dates[convergence_end], 
           alpha=0.2, color='green', label='回归期（平仓获利）')
ax.text(dates[convergence_start + 20], np.max(Y) * 0.85,
        '价差回归\n平仓获利',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='green', alpha=0.7),
        fontsize=10, ha='center', weight='bold')

ax.set_xlabel('日期', fontsize=12, fontweight='bold')
ax.set_ylabel('价格 ($)', fontsize=12, fontweight='bold')
ax.set_title('配对交易原理：利用价差的均值回归获利', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='upper left', fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 生成图1: cover.jpg")

# 图2: 协整关系示意图
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：协整序列（Y vs X）散点图
axes[0, 0].scatter(X, Y, c=t, cmap='viridis', s=20, alpha=0.6)
x_line = np.linspace(X.min(), X.max(), 100)
y_line = alpha + beta * x_line
axes[0, 0].plot(x_line, y_line, 'r-', linewidth=2.5, label=f'Y = {alpha:.1f} + {beta:.1f}X')
axes[0, 0].set_xlabel('X 价格', fontsize=11, fontweight='bold')
axes[0, 0].set_ylabel('Y 价格', fontsize=11, fontweight='bold')
axes[0, 0].set_title('协整关系：Y vs X\n(线性关系稳定)', fontsize=12, fontweight='bold')
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：残差平稳（协整的证据）
residuals = Y - (alpha + beta * X)
axes[0, 1].plot(dates, residuals, 'g-', linewidth=2)
axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[0, 1].fill_between(dates, 0, residuals, alpha=0.3, where=(residuals > 0), color='green')
axes[0, 1].fill_between(dates, 0, residuals, alpha=0.3, where=(residuals <= 0), color='red')
axes[0, 1].set_xlabel('日期', fontsize=11, fontweight='bold')
axes[0, 1].set_ylabel('残差', fontsize=11, fontweight='bold')
axes[0, 1].set_title('协整检验：残差平稳\n(均值回归特征)', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axhline(y=2*residuals.std(), color='red', linestyle='--', alpha=0.5, label='±2σ')
axes[0, 1].axhline(y=-2*residuals.std(), color='red', linestyle='--', alpha=0.5)
axes[0, 1].legend(fontsize=9)

# 子图3：两个随机游走（伪回归示例）
np.random.seed(123)
Z1 = 50 + np.cumsum(np.random.randn(n) * 2)
Z2 = 50 + np.cumsum(np.random.randn(n) * 2)
axes[1, 0].scatter(Z1, Z2, c=t, cmap='viridis', s=20, alpha=0.6)
axes[1, 0].set_xlabel('Z1', fontsize=11, fontweight='bold')
axes[1, 0].set_ylabel('Z2', fontsize=11, fontweight='bold')
axes[1, 0].set_title('伪回归：两个独立的随机游走\n(无经济意义)', fontsize=12, fontweight='bold', color='darkred')
axes[1, 0].grid(True, alpha=0.3)

# 子图4：伪回归的残差（非平稳）
Z_beta = np.polyfit(Z1, Z2, 1)[0]
Z_alpha = np.mean(Z2) - Z_beta * np.mean(Z1)
Z_residuals = Z2 - (Z_alpha + Z_beta * Z1)
axes[1, 1].plot(range(n), Z_residuals, 'r-', linewidth=2)
axes[1, 1].axhline(y=0, color='black', linestyle='-', linewidth=1.5)
axes[1, 1].set_xlabel('时间', fontsize=11, fontweight='bold')
axes[1, 1].set_ylabel('残差', fontsize=11, fontweight='bold')
axes[1, 1].set_title('伪回归：残差非平稳\n(无均值回归)', fontsize=12, fontweight='bold', color='darkred')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_vs_spurious.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 生成图2: cointegration_vs_spurious.png")

# 图3: Z得分交易信号
fig, ax = plt.subplots(figsize=(14, 7))

# 计算Z得分（使用滚动窗口）
window = 63
spread = Y - (alpha + beta * X)
rolling_mean = pd.Series(spread).rolling(window).mean()
rolling_std = pd.Series(spread).rolling(window).std()
z_score = ((spread - rolling_mean) / rolling_std).fillna(0)

ax.plot(dates, z_score, 'b-', linewidth=2.5, label='Z 得分 (价差标准化)', alpha=0.8)
ax.axhline(y=2.0, color='red', linestyle='--', linewidth=2.5, label='入场阈值 (+2.0)')
ax.axhline(y=-2.0, color='red', linestyle='--', linewidth=2.5)
ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2.5, label='出场阈值 (+0.5)')
ax.axhline(y=-0.5, color='orange', linestyle='--', linewidth=2.5)
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1.5, alpha=0.8)

# 标记交易信号
entry_long = z_score < -2.0
entry_short = z_score > 2.0
exit_signal = (z_score.abs() < 0.5) & (z_score.abs().shift(1) >= 0.5)

ax.scatter(dates[entry_long], z_score[entry_long], 
           color='green', marker='^', s=120, label='做多信号 (买低)', zorder=5, alpha=0.9)
ax.scatter(dates[entry_short], z_score[entry_short], 
           color='red', marker='v', s=120, label='做空信号 (卖高)', zorder=5, alpha=0.9)
ax.scatter(dates[exit_signal], z_score[exit_signal], 
           color='blue', marker='o', s=120, label='平仓信号', zorder=5, alpha=0.9)

ax.set_xlabel('日期', fontsize=12, fontweight='bold')
ax.set_ylabel('Z 得分', fontsize=12, fontweight='bold')
ax.set_title('配对交易信号生成：基于Z得分的均值回归策略', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([-4, 4])

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/z_score_signals.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 生成图3: z_score_signals.png")

# 图4: 累计收益曲线（模拟回测）
fig, ax = plt.subplots(figsize=(12, 6))

# 模拟策略收益（基于Z得分信号）
np.random.seed(42)
n_trades = n
returns = np.zeros(n_trades)

# 简单模拟：当Z得分绝对值大时，未来收益为正
for i in range(1, n_trades):
    if abs(z_score.iloc[i-1]) > 2.0:
        # 入场后，价差回归带来收益
        returns[i] = -z_score.iloc[i-1] * 0.01 + np.random.randn() * 0.005
    else:
        returns[i] = np.random.randn() * 0.001

cumulative = np.cumprod(1 + returns)

ax.plot(dates[:len(cumulative)], cumulative, 'g-', linewidth=2.5, label='配对交易策略')
ax.axhline(y=1, color='gray', linestyle='-', linewidth=1.5, alpha=0.8)
ax.fill_between(dates[:len(cumulative)], 1, cumulative, 
                alpha=0.3, where=(cumulative >= 1), color='green', label='盈利期')
ax.fill_between(dates[:len(cumulative)], 1, cumulative, 
                alpha=0.3, where=(cumulative < 1), color='red', label='亏损期')

# 标记关键事件（使用numpy计算）
cummax = np.maximum.accumulate(cumulative)
max_dd_idx = np.argmin(cumulative / cummax - 1)
ax.scatter(dates[max_dd_idx], cumulative[max_dd_idx], 
           color='red', marker='v', s=150, label='最大回撤', zorder=5, alpha=0.8)

# 添加注释
max_dd = np.min(cumulative / cummax - 1)
ax.annotate(f'最大回撤: {max_dd:.2%}', 
            xy=(dates[max_dd_idx], cumulative[max_dd_idx]),
            xytext=(20, -40), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='red', alpha=0.8),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='white', weight='bold')

# 计算最大回撤（使用numpy）
cummax = np.maximum.accumulate(cumulative)
max_dd_idx = np.argmin(cumulative / cummax - 1)

# 添加绩效标签
total_return = cumulative[-1] - 1
sharpe = returns.mean() / returns.std() * np.sqrt(252)
ax.text(0.02, 0.98, f'总收益: {total_return:.2%}\n夏普比率: {sharpe:.2f}', 
        transform=ax.transAxes, fontsize=11, verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), weight='bold')

ax.set_xlabel('日期', fontsize=12, fontweight='bold')
ax.set_ylabel('累计收益 (净值)', fontsize=12, fontweight='bold')
ax.set_title('配对交易策略累计收益（模拟回测 2022-2026）', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim([cumulative.min() * 0.95, cumulative.max() * 1.05])

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cumulative_returns.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 生成图4: cumulative_returns.png")

print("\n" + "="*60)
print("✅ 配对交易文章配图生成完成！")
print("="*60)
print("   - cover.jpg: 封面图（配对交易原理）")
print("   - cointegration_vs_spurious.png: 协整 vs 伪回归对比")
print("   - z_score_signals.png: Z得分交易信号")
print("   - cumulative_returns.png: 累计收益曲线")
print("="*60)
