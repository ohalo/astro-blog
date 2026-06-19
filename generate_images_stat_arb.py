#!/usr/bin/env python3
"""
为统计套利文章生成配图
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

# 图1：配对交易原理示意图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('统计套利：配对交易原理与实战', fontsize=16, fontweight='bold')

# 子图1：两个协整资产的价格走势
np.random.seed(42)
dates = pd.date_range('2024-01-01', '2026-06-19', freq='D')
n_days = len(dates)

# 生成协整价格序列
trend = np.linspace(0, 50, n_days)
noise1 = np.cumsum(np.random.normal(0, 1, n_days))
noise2 = np.cumsum(np.random.normal(0, 1, n_days))
price1 = 100 + trend + noise1
price2 = 50 + 0.5 * trend + 0.5 * noise2

axes[0, 0].plot(dates, price1, linewidth=2, color='#2E86AB', label='资产A')
axes[0, 0].plot(dates, price2, linewidth=2, color='#A23B72', label='资产B')
axes[0, 0].set_title('协整资产价格走势', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('价格', fontsize=10)
axes[0, 0].legend(fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：价差序列与交易信号
spread = price1 - 2 * price2
spread_mean = pd.Series(spread).rolling(60).mean()
spread_std = pd.Series(spread).rolling(60).std()
z_score = (spread - spread_mean) / spread_std

axes[0, 1].plot(dates, spread, linewidth=2, color='green', label='价差')
axes[0, 1].axhline(y=spread_mean.iloc[-1], color='black', linestyle='-', linewidth=1, label='均值')
axes[0, 1].axhline(y=spread_mean.iloc[-1] + 2*spread_std.iloc[-1], 
                     color='red', linestyle='--', linewidth=2, label='+2σ')
axes[0, 1].axhline(y=spread_mean.iloc[-1] - 2*spread_std.iloc[-1], 
                     color='red', linestyle='--', linewidth=2, label='-2σ')
axes[0, 1].fill_between(dates, 
                          spread_mean.iloc[-1] - 2*spread_std.iloc[-1],
                          spread_mean.iloc[-1] + 2*spread_std.iloc[-1],
                          alpha=0.2, color='gray')
axes[0, 1].set_title('价差序列与交易信号', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('价差', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：z-score分布
axes[1, 0].hist(z_score.dropna(), bins=50, alpha=0.7, color='orange', edgecolor='black')
axes[1, 0].axvline(x=2, color='red', linestyle='--', linewidth=2, label='入场阈值')
axes[1, 0].axvline(x=-2, color='red', linestyle='--', linewidth=2)
axes[1, 0].axvline(x=0.5, color='green', linestyle='--', linewidth=2, label='出场阈值')
axes[1, 0].axvline(x=-0.5, color='green', linestyle='--', linewidth=2)
axes[1, 0].set_title('z-score分布', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('z-score', fontsize=10)
axes[1, 0].set_ylabel('频次', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：累计收益曲线（模拟回测结果）
np.random.seed(42)
cum_returns = pd.Series(1 + np.cumsum(np.random.normal(0.0005, 0.005, n_days)))
benchmark = pd.Series(1 + np.cumsum(np.random.normal(0.0003, 0.008, n_days)))

axes[1, 1].plot(dates, cum_returns.values, linewidth=2, color='blue', label='配对策略')
axes[1, 1].plot(dates, benchmark.values, linewidth=2, color='gray', linestyle='--', label='基准')
axes[1, 1].axhline(y=1.0, color='black', linestyle='-', linewidth=1)
axes[1, 1].set_title('累计收益对比', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('累计收益', fontsize=10)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/principle.png', dpi=300, bbox_inches='tight')
plt.close()

# 图2：协整检验可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('协整检验与参数估计', fontsize=16, fontweight='bold')

# 子图1：残差序列
residuals = spread - spread_mean
axes[0, 0].plot(dates, residuals, linewidth=2, color='purple')
axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0, 0].set_title('OLS回归残差序列', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('残差', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：残差自相关图（手动计算）
acf_vals = [1.0]  # lag 0
residuals_clean = residuals[~np.isnan(residuals)]
for lag in range(1, 41):
    if len(residuals_clean) > lag:
        acf = np.corrcoef(residuals_clean[lag:], residuals_clean[:-lag])[0, 1]
        acf_vals.append(acf)
    else:
        acf_vals.append(0)
axes[0, 1].bar(range(41), acf_vals, alpha=0.7, color='blue')
axes[0, 1].axhline(y=0, color='black', linewidth=1)
axes[0, 1].axhline(y=1.96/np.sqrt(len(residuals_clean)), color='red', 
                   linestyle='--', linewidth=1, label='95%置信区间')
axes[0, 1].axhline(y=-1.96/np.sqrt(len(residuals_clean)), color='red', 
                   linestyle='--', linewidth=1)
axes[0, 1].set_title('残差自相关函数（ACF）', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('滞后期', fontsize=10)
axes[0, 1].set_ylabel('自相关系数', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：ADF检验（模拟）
np.random.seed(42)
n_samples = 1000
random_walk = np.cumsum(np.random.normal(0, 1, n_samples))
stationary_series = 0.5 * random_walk + 0.5 * np.random.normal(0, 1, n_samples)

axes[1, 0].plot(range(n_samples), random_walk, linewidth=2, color='red', alpha=0.7, label='非平稳序列')
axes[1, 0].plot(range(n_samples), stationary_series, linewidth=2, color='blue', alpha=0.7, label='平稳序列')
axes[1, 0].set_title('ADF检验：平稳性对比', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('时间', fontsize=10)
axes[1, 0].set_ylabel('值', fontsize=10)
axes[1, 0].legend(fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# 子图4：半衰期分布（模拟多个配对）
half_lives = np.random.gamma(10, 2, 100)  # 模拟100个配对的半衰期
axes[1, 1].hist(half_lives, bins=30, alpha=0.7, color='green', edgecolor='black')
axes[1, 1].axvline(x=np.mean(half_lives), color='red', linestyle='--', linewidth=2, 
                      label=f'平均半衰期={np.mean(half_lives):.1f}天')
axes[1, 1].set_title('配对半衰期分布', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('半衰期（交易日）', fontsize=10)
axes[1, 1].set_ylabel('频次', fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cointegration_test.png', dpi=300, bbox_inches='tight')
plt.close()

# 图3：多资产统计套利（PCA）
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('多资产统计套利：PCA降维', fontsize=16, fontweight='bold')

# 生成模拟数据
np.random.seed(42)
n_assets = 10
n_days = 252
returns = np.random.normal(0.0005, 0.02, (n_days, n_assets))
asset_names = [f'Asset_{i+1}' for i in range(n_assets)]

# 子图1：原始收益率相关矩阵热力图
corr_matrix = np.corrcoef(returns.T)
im1 = axes[0, 0].imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
axes[0, 0].set_title('原始收益率相关矩阵', fontsize=12, fontweight='bold')
axes[0, 0].set_xticks(range(n_assets))
axes[0, 0].set_xticklabels(asset_names, rotation=45, ha='right', fontsize=8)
axes[0, 0].set_yticks(range(n_assets))
axes[0, 0].set_yticklabels(asset_names, fontsize=8)
plt.colorbar(im1, ax=axes[0, 0], fraction=0.046, pad=0.04)

# 子图2：PCA解释方差比
from sklearn.decomposition import PCA
pca = PCA()
pca.fit(returns)
cumvar = np.cumsum(pca.explained_variance_ratio_)

axes[0, 1].plot(range(1, n_assets+1), cumvar, 'bo-', linewidth=2, markersize=8)
axes[0, 1].axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='90%方差')
axes[0, 1].set_title('PCA解释方差比', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('主成分个数', fontsize=10)
axes[0, 1].set_ylabel('累计解释方差比', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：残差相关矩阵（去除主成分后）
returns_pca = pca.transform(returns)
returns_reconstructed = pca.inverse_transform(returns_pca)
residuals_pca = returns - returns_reconstructed
residuals_corr = np.corrcoef(residuals_pca.T)

im2 = axes[1, 0].imshow(residuals_corr, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
axes[1, 0].set_title('残差相关矩阵（套利空间）', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(range(n_assets))
axes[1, 0].set_xticklabels(asset_names, rotation=45, ha='right', fontsize=8)
axes[1, 0].set_yticks(range(n_assets))
axes[1, 0].set_yticklabels(asset_names, fontsize=8)
plt.colorbar(im2, ax=axes[1, 0], fraction=0.046, pad=0.04)

# 子图4：残差收益率分布
flattened_residuals = residuals_pca.flatten()
axes[1, 1].hist(flattened_residuals, bins=50, alpha=0.7, color='purple', edgecolor='black')
axes[1, 1].axvline(x=0, color='black', linestyle='-', linewidth=2)
axes[1, 1].set_title('残差收益率分布', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('残差收益率', fontsize=10)
axes[1, 1].set_ylabel('频次', fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/pca_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 图4：风险控制与仓位管理
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('风险控制与仓位管理', fontsize=16, fontweight='bold')

# 子图1：协整稳定性监测（模拟）
stability_scores = 0.05 + 0.03 * np.sin(np.linspace(0, 4*np.pi, n_days)) + np.random.uniform(-0.01, 0.01, n_days)
stability_scores = np.clip(stability_scores, 0.01, 0.1)
dates_stability = pd.date_range('2025-01-01', periods=n_days, freq='D')

axes[0, 0].plot(dates_stability, stability_scores, linewidth=2, color='orange')
axes[0, 0].axhline(y=0.05, color='red', linestyle='--', linewidth=2, label='显著性阈值')
axes[0, 0].fill_between(dates_stability, 0, stability_scores, 
                          where=(stability_scores > 0.05), alpha=0.3, color='red', label='不稳定期')
axes[0, 0].set_title('协整稳定性监测', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('p-value', fontsize=10)
axes[0, 0].legend(fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# 子图2：动态仓位调整
volatility = 0.02 + 0.01 * np.sin(np.linspace(0, 2*np.pi, n_days)) + np.random.uniform(-0.005, 0.005, n_days)
volatility = np.clip(volatility, 0.01, 0.05)
target_vol = 0.20 / np.sqrt(252)
position_size = target_vol / volatility

axes[0, 1].plot(dates_stability, position_size, linewidth=2, color='green')
axes[0, 1].axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='满仓')
axes[0, 1].fill_between(dates_stability, 0, position_size, alpha=0.3, color='green')
axes[0, 1].set_title('动态仓位调整（目标波动率）', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('仓位比例', fontsize=10)
axes[0, 1].legend(fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：凯利公式仓位建议（模拟不同胜率和盈亏比）
win_rates = np.linspace(0.4, 0.6, 100)
win_loss_ratios = np.linspace(0.8, 1.5, 100)
X, Y = np.meshgrid(win_rates, win_loss_ratios)
Z = (X * Y - (1 - X)) / Y
Z = np.clip(Z, 0, 2)

im3 = axes[1, 0].imshow(Z, extent=[0.4, 0.6, 0.8, 1.5], origin='lower', cmap='RdYlGn', aspect='auto')
axes[1, 0].set_title('凯利公式仓位建议', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('胜率', fontsize=10)
axes[1, 0].set_ylabel('盈亏比', fontsize=10)
plt.colorbar(im3, ax=axes[1, 0], fraction=0.046, pad=0.04, label='仓位比例')

# 子图4：最大回撤控制（模拟）
cum_returns_risk = pd.Series(1 + np.cumsum(np.random.normal(0.0008, 0.01, n_days)))
drawdown = cum_returns_risk / cum_returns_risk.cummax() - 1

axes[1, 1].fill_between(range(n_days), 0, drawdown.values, 
                         alpha=0.5, color='red', label='回撤')
axes[1, 1].axhline(y=-0.05, color='orange', linestyle='--', linewidth=2, label='止损线-5%')
axes[1, 1].set_title('最大回撤控制', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('交易日', fontsize=10)
axes[1, 1].set_ylabel('回撤', fontsize=10)
axes[1, 1].legend(fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/risk_control.png', dpi=300, bbox_inches='tight')
plt.close()

print("✅ 统计套利文章配图生成完成！")
print("生成文件：")
print("  1. principle.png - 配对交易原理示意图")
print("  2. cointegration_test.png - 协整检验可视化")
print("  3. pca_analysis.png - PCA多资产分析")
print("  4. risk_control.png - 风险控制与仓位管理")
print("  5. backtest_results.png - 回测结果可视化（由策略类生成）")
