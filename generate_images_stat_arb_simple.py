#!/usr/bin/env python3
"""
为统计套利文章生成配图（简化版，无外部依赖）
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion', exist_ok=True)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2026-06-19', freq='D')
n_days = len(dates)

# 图1：配对交易原理示意图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Statistical Arbitrage: Pairs Trading', fontsize=16, fontweight='bold')

# 子图1：两个协整资产的价格走势
trend = np.linspace(0, 50, n_days)
noise1 = np.cumsum(np.random.normal(0, 1, n_days))
noise2 = np.cumsum(np.random.normal(0, 1, n_days))
price1 = 100 + trend + noise1
price2 = 50 + 0.5 * trend + 0.5 * noise2

axes[0, 0].plot(dates, price1, linewidth=2, color='#2E86AB', label='Asset A')
axes[0, 0].plot(dates, price2, linewidth=2, color='#A23B72', label='Asset B')
axes[0, 0].set_title('Cointegrated Price Series', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Price', fontsize=10)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：价差序列与交易信号
spread = price1 - 2 * price2
spread_mean = pd.Series(spread).rolling(60).mean()
spread_std = pd.Series(spread).rolling(60).std()

axes[0, 1].plot(dates, spread, linewidth=2, color='green', label='Spread')
axes[0, 1].axhline(y=spread_mean.iloc[-1], color='black', linestyle='-', linewidth=1, label='Mean')
axes[0, 1].axhline(y=spread_mean.iloc[-1] + 2*spread_std.iloc[-1], 
                     color='red', linestyle='--', linewidth=2, label='+2σ')
axes[0, 1].axhline(y=spread_mean.iloc[-1] - 2*spread_std.iloc[-1], 
                     color='red', linestyle='--', linewidth=2, label='-2σ')
axes[0, 1].fill_between(dates, 
                          spread_mean.iloc[-1] - 2*spread_std.iloc[-1],
                          spread_mean.iloc[-1] + 2*spread_std.iloc[-1],
                          alpha=0.2, color='gray')
axes[0, 1].set_title('Spread with Trading Bands', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Spread', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：z-score分布
z_score = (spread - spread_mean) / spread_std

axes[1, 0].hist(z_score.dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].axvline(x=2, color='red', linestyle='--', linewidth=2, label='Entry Threshold')
axes[1, 0].axvline(x=-2, color='red', linestyle='--', linewidth=2)
axes[1, 0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='Exit Threshold')
axes[1, 0].axvline(x=-0.5, color='green', linestyle='--', linewidth=2)
axes[1, 0].set_title('z-score Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('z-score', fontsize=10)
axes[1, 0].set_ylabel('Frequency', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：累计收益曲线（模拟回测结果）
cum_returns = pd.Series(1 + np.cumsum(np.random.normal(0.0005, 0.005, n_days)))
benchmark = pd.Series(1 + np.cumsum(np.random.normal(0.0003, 0.008, n_days)))

axes[1, 1].plot(dates, cum_returns.values, linewidth=2, color='blue', label='Pairs Strategy')
axes[1, 1].plot(dates, benchmark.values, linewidth=2, color='gray', linestyle='--', label='Benchmark')
axes[1, 1].axhline(y=1.0, color='black', linestyle='-', linewidth=1)
axes[1, 1].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('Cumulative Return', fontsize=10)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/principle.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2：协整检验可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Cointegration Test & Parameters', fontsize=16, fontweight='bold')

# 子图1：OLS回归残差
residuals = spread - spread_mean

axes[0, 0].plot(dates, residuals, linewidth=2, color='purple')
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_title('OLS Residuals', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Residual', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：残差自相关（手动计算）
acf_vals = [1.0]
for lag in range(1, 41):
    if len(residuals.dropna()) > lag:
        res_clean = residuals.dropna().values
        acf = np.corrcoef(res_clean[lag:], res_clean[:-lag])[0, 1]
        acf_vals.append(acf if not np.isnan(acf) else 0)
    else:
        acf_vals.append(0)

axes[0, 1].bar(range(len(acf_vals)), acf_vals, alpha=0.7, color='blue')
axes[0, 1].axhline(y=0, color='black', linewidth=1)
axes[0, 1].axhline(y=1.96/np.sqrt(len(residuals.dropna())), color='red', 
                   linestyle='--', linewidth=1)
axes[0, 1].axhline(y=-1.96/np.sqrt(len(residuals.dropna())), color='red', 
                   linestyle='--', linewidth=1)
axes[0, 1].set_title('Residual Autocorrelation', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Lag', fontsize=10)
axes[0, 1].set_ylabel('ACF', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：平稳性对比
random_walk = np.cumsum(np.random.normal(0, 1, 500))
stationary = 0.5 * random_walk + 0.5 * np.random.normal(0, 1, 500)

axes[1, 0].plot(range(500), random_walk, linewidth=2, color='red', alpha=0.7, label='Non-stationary')
axes[1, 0].plot(range(500), stationary, linewidth=2, color='blue', alpha=0.7, label='Stationary')
axes[1, 0].set_title('Stationarity Test', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Time', fontsize=10)
axes[1, 0].set_ylabel('Value', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：半衰期分布（模拟）
half_lives = np.random.gamma(10, 2, 100)

axes[1, 1].hist(half_lives, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].axvline(x=np.mean(half_lives), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean={np.mean(half_lives):.1f} days')
axes[1, 1].set_title('Half-life Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Half-life (days)', fontsize=10)
axes[1, 1].set_ylabel('Frequency', fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cointegration_test.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3：风险控制
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Risk Control & Position Management', fontsize=16, fontweight='bold')

# 子图1：协整稳定性监测
stability_scores = 0.05 + 0.03 * np.sin(np.linspace(0, 4*np.pi, n_days)) + np.random.uniform(-0.01, 0.01, n_days)
stability_scores = np.clip(stability_scores, 0.01, 0.1)
dates_stability = pd.date_range('2025-01-01', periods=n_days, freq='D')

axes[0, 0].plot(dates_stability, stability_scores, linewidth=2, color='orange')
axes[0, 0].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='Significance')
axes[0, 0].fill_between(dates_stability, 0, stability_scores, 
                          where=(stability_scores > 0.05), alpha=0.3, color='red')
axes[0, 0].set_title('Cointegration Stability', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('p-value', fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：动态仓位调整
volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 2*np.pi, n_days)) + np.random.uniform(-0.005, 0.005, n_days)
volatility = np.clip(volatility, 0.01, 0.05)
target_vol = 0.20 / np.sqrt(252)
position_size = target_vol / volatility

axes[0, 1].plot(dates_stability, position_size, linewidth=2, color='green')
axes[0, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Full Position')
axes[0, 1].fill_between(dates_stability, 0, position_size, alpha=0.3, color='green')
axes[0, 1].set_title('Dynamic Position Sizing', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Position %', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：凯利公式仓位
win_rates = np.linspace(0.4, 0.6, 100)
win_loss_ratios = np.linspace(0.8, 1.5, 100)
X, Y = np.meshgrid(win_rates, win_loss_ratios)
Z = (X * Y - (1 - X)) / Y
Z = np.clip(Z, 0, 2)

im = axes[1, 0].imshow(Z, extent=[0.4, 0.6, 0.8, 1.5], origin='lower', cmap='RdYlGn', aspect='auto')
axes[1, 0].set_title('Kelly Criterion', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Win Rate', fontsize=10)
axes[1, 0].set_ylabel('Win/Loss Ratio', fontsize=10)
plt.colorbar(im, ax=axes[1, 0], fraction=0.046, pad=0.04, label='Position %')

# 子图4：最大回撤
cum_returns_risk = pd.Series(1 + np.cumsum(np.random.normal(0.0008, 0.01, n_days)))
drawdown = cum_returns_risk / cum_returns_risk.cummax() - 1

axes[1, 1].fill_between(range(n_days), 0, drawdown.values, 
                         alpha=0.5, color='red')
axes[1, 1].axhline(y=-0.05, color='orange', linestyle='--', linewidth=2, label='Stop-loss -5%')
axes[1, 1].set_title('Maximum Drawdown', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Trading Days', fontsize=10)
axes[1, 1].set_ylabel('Drawdown', fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/risk_control.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4：回测结果可视化（简化版）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Backtest Results', fontsize=16, fontweight='bold')

# 模拟回测数据
np.random.seed(42)
n_days = 252
daily_returns = np.random.normal(0.0008, 0.01, n_days)
cum_returns = (1 + pd.Series(daily_returns)).cumprod()
benchmark_returns = np.random.normal(0.0005, 0.012, n_days)
benchmark_cum = (1 + pd.Series(benchmark_returns)).cumprod()

# 子图1：累计收益
axes[0, 0].plot(range(n_days), cum_returns.values, linewidth=2, color='blue', label='Strategy')
axes[0, 0].plot(range(n_days), benchmark_cum.values, linewidth=2, color='gray', linestyle='--', label='Benchmark')
axes[0, 0].axhline(y=1.0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_title('Cumulative Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Cumulative Return', fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：回撤
drawdown = cum_returns / cum_returns.cummax() - 1

axes[0, 1].fill_between(range(n_days), 0, drawdown.values, alpha=0.5, color='red')
axes[0, 1].set_title('Drawdown Curve', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Drawdown', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：日收益分布
axes[1, 0].hist(daily_returns, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1, 0].axvline(x=np.mean(daily_returns), color='red', linestyle='--', linewidth=2, 
                      label=f'Mean={np.mean(daily_returns):.4f}')
axes[1, 0].set_title('Daily Returns Distribution', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Daily Return', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：月度收益
# 直接使用模拟的月度收益
monthly = np.random.normal(0.02, 0.05, 12)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

axes[1, 1].bar(range(12), monthly, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axhline(y=0, color='black', linewidth=1)
axes[1, 1].set_title('Monthly Returns', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Month', fontsize=10)
axes[1, 1].set_ylabel('Return', fontsize=10)
axes[1, 1].set_xticks(range(12))
axes[1, 1].set_xticklabels(months, rotation=45, ha='right', fontsize=8)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/backtest_results.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ Statistical arbitrage article images generated!")
print("Generated files:")
print("  1. principle.png - Pairs trading principle")
print("  2. cointegration_test.png - Cointegration test")
print("  3. risk_control.png - Risk control")
print("  4. backtest_results.png - Backtest results")
