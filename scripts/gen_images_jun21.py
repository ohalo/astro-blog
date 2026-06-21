#!/usr/bin/env python3
"""生成2026-06-21两篇量化文章配图"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from matplotlib import font_manager
import os

# 中文字体配置
matplotlib.rcParams['axes.unicode_minus'] = False
try:
    font_paths = font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
    noto = [f for f in font_paths if 'Noto' in f and 'CJK' in f]
    pingfang = [f for f in font_paths if 'PingFang' in f or 'pingfang' in f.lower()]
    if pingfang:
        plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti']
    elif noto:
        plt.rcParams['font.sans-serif'] = ['Noto Sans CJK SC']
    else:
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
except:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']

np.random.seed(20240621)

# ========================
# 文章1: 多因子模型风险分解
# ========================
out1 = "/Users/halo/workspace/astro-blog/public/images/2026-06-21-multi-factor-risk-decomposition"
os.makedirs(out1, exist_ok=True)

# 图1: 因子风险贡献条形图
fig, ax = plt.subplots(figsize=(10, 6))
factors = ['市场因子(β)', '规模因子(SMB)', '价值因子(HML)', '动量因子(MOM)', '质量因子(QMJ)', '低波因子(BAB)']
risk_contrib = np.array([0.35, 0.12, 0.18, 0.10, 0.15, 0.10])
colors = ['#5470c6', '#91cc75', '#fac858', '#ee6666', '#73c0de', '#3ba272']
bars = ax.bar(factors, risk_contrib * 100, color=colors)
ax.set_ylabel('风险贡献 (%)', fontsize=12)
ax.set_title('多因子模型 — 各因子风险贡献分解', fontsize=14, fontweight='bold')
ax.set_ylim(0, 40)
ax.tick_params(axis='x', rotation=30)
for bar, val in zip(bars, risk_contrib):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{val*100:.1f}%', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{out1}/risk_contribution.png", dpi=150, bbox_inches='tight')
plt.close()

# 图2: 因子收益率时序（模拟）
fig, axes = plt.subplots(3, 2, figsize=(14, 10))
axes = axes.flatten()
factor_names = factors
for i, name in enumerate(factor_names):
    returns = np.random.normal(0.0005 if i == 0 else 0.0003, 
                              0.008 if i == 0 else 0.005, 
                              252)
    cum_ret = (1 + pd.Series(returns)).cumprod()
    axes[i].plot(cum_ret.values, color=colors[i], linewidth=1.5)
    axes[i].set_title(name, fontsize=11)
    axes[i].set_xlabel('交易日')
    axes[i].set_ylabel('累计净值')
    axes[i].grid(alpha=0.3)
plt.suptitle('六大因子累计净值曲线（模拟数据）', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(f"{out1}/factor_cumulative_returns.png", dpi=150, bbox_inches='tight')
plt.close()

# 图3: 因子相关系数矩阵热图
fig, ax = plt.subplots(figsize=(9, 7))
np.random.seed(42)
n = len(factors)
corr = np.eye(n)
for i in range(n):
    for j in range(i+1, n):
        c = np.random.uniform(-0.3, 0.5)
        corr[i, j] = c
        corr[j, i] = c
corr[0, 1] = 0.15  # 市场-规模
corr[0, 2] = -0.10 # 市场-价值
corr[1, 2] = -0.25 # 规模-价值
corr[3, 4] = 0.35  # 动量-质量
corr = (corr + corr.T) / 2
np.fill_diagonal(corr, 1.0)

sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
            xticklabels=factors, yticklabels=factors,
            square=True, ax=ax, cbar_kws={'label': '相关系数'})
ax.set_title('因子收益率相关系数矩阵', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(f"{out1}/factor_correlation_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 文章1配图已生成: {out1}")

# ========================
# 文章2: 配对交易与协整分析
# ========================
out2 = "/Users/halo/workspace/astro-blog/public/images/2026-06-21-pair-trading-cointegration"
os.makedirs(out2, exist_ok=True)

# 图1: 配对股票价差与信号
np.random.seed(99)
n = 252
t = np.arange(n)

# 模拟两只协整股票
x = np.cumsum(np.random.normal(0, 1, n)) * 0.5 + 50
y = 1.2 * x + np.random.normal(0, 2, n) + 10

spread = y - 1.2 * x
rolling_mean = pd.Series(spread).rolling(20).mean()
rolling_std = pd.Series(spread).rolling(20).std()
z_score = (spread - rolling_mean) / rolling_std

fig, axes = plt.subplots(3, 1, figsize=(12, 10))

# 子图1: 配对数格
axes[0].plot(t, x, label='股票A (中国平安)', color='#5470c6', linewidth=1.5)
axes[0].plot(t, y, label='股票B (中国人寿)', color='#ee6666', linewidth=1.5, alpha=0.8)
axes[0].set_title('配对股票：中国平安 vs 中国人寿（模拟价格）', fontsize=12, fontweight='bold')
axes[0].legend()
axes[0].grid(alpha=0.3)

# 子图2: 价差
axes[1].plot(t, spread, color='#91cc75', linewidth=1.5, label='价差')
axes[1].plot(t, rolling_mean, color='#fac858', linewidth=1.2, linestyle='--', label='20日均值')
axes[1].fill_between(t, rolling_mean - 2*rolling_std, rolling_mean + 2*rolling_std, 
                      alpha=0.2, color='#fac858', label='±2σ区间')
axes[1].set_title('价差序列与均值回归带', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

# 子图3: Z-score与交易信号
axes[2].plot(t, z_score, color='#73c0de', linewidth=1.5, label='Z-score')
axes[2].axhline(y=2, color='#ee6666', linestyle='--', linewidth=1.2, label='上轨(+2σ)')
axes[2].axhline(y=-2, color='#91cc75', linestyle='--', linewidth=1.2, label='下轨(-2σ)')
axes[2].axhline(y=0, color='gray', linestyle='-', linewidth=0.8, alpha=0.5)
# 标记交易信号
buy_signals = np.where(z_score < -2)[0]
sell_signals = np.where(z_score > 2)[0]
if len(buy_signals) > 0:
    axes[2].scatter(buy_signals, z_score[buy_signals], color='red', s=30, 
                    marker='^', label='买入信号', zorder=5)
if len(sell_signals) > 0:
    axes[2].scatter(sell_signals, z_score[sell_signals], color='green', s=30,
                    marker='v', label='卖出信号', zorder=5)
axes[2].set_title('Z-score 交易信号图', fontsize=12)
axes[2].set_xlabel('交易日')
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out2}/pairs_trading_signals.png", dpi=150, bbox_inches='tight')
plt.close()

# 图2: 协整检验 ADF 测试结果可视化
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# 左图: 价差序列 ADF 检验（模拟）
np.random.seed(123)
spread_random_walk = np.cumsum(np.random.normal(0, 1, 252))  # 非平稳
spread_stationary = np.random.normal(0, 1, 252).cumsum()
spread_stationary = spread_stationary - pd.Series(spread_stationary).rolling(50, min_periods=1).mean()

axes[0].plot(spread_random_walk, color='#ee6666', linewidth=1.5, label='随机游走（非协整）')
axes[0].set_title('ADF检验：非平稳序列（p-value > 0.05）', fontsize=11)
axes[0].set_xlabel('交易日')
axes[0].set_ylabel('序列值')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(spread[:200], color='#91cc75', linewidth=1.5, label='平稳价差（协整）')
axes[1].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
axes[1].set_title('ADF检验：平稳序列（p-value < 0.01）', fontsize=11)
axes[1].set_xlabel('交易日')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f"{out2}/cointegration_adf_test.png", dpi=150, bbox_inches='tight')
plt.close()

# 图3: 配对交易累计收益
fig, ax = plt.subplots(figsize=(11, 5))
np.random.seed(789)
n = 504
# 模拟配对策略收益
daily_pnl = np.random.normal(0.0008, 0.004, n)
cum_pnl = (1 + pd.Series(daily_pnl)).cumprod()

ax.plot(cum_pnl.values, color='#5470c6', linewidth=2, label='配对策略累计净值')
ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
ax.fill_between(range(n), 1, cum_pnl.values, 
                where=(cum_pnl.values >= 1), alpha=0.2, color='green')
ax.fill_between(range(n), 1, cum_pnl.values,
                where=(cum_pnl.values < 1), alpha=0.2, color='red')
ax.set_title('配对交易策略累计净值（模拟 2 年）', fontsize=13, fontweight='bold')
ax.set_xlabel('交易日')
ax.set_ylabel('累计净值')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{out2}/pairs_cumulative_return.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 文章2配图已生成: {out2}")
print("🎉 所有配图生成完毕！")
