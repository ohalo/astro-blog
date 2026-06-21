#!/usr/bin/env python3
"""
为PCA统计套利文章生成配图（使用模拟数据）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ========== 生成模拟数据 ==========
print("生成模拟股票数据...")

np.random.seed(42)
n_days = 500
n_stocks = 10

# 模拟因子：市场因子 + 2个行业因子
market_factor = np.random.normal(0.0005, 0.01, n_days)
sector1_factor = np.random.normal(0, 0.008, n_days)
sector2_factor = np.random.normal(0, 0.006, n_days)

# 股票对因子的暴露
betas = np.array([
    [1.2, 0.8, 0.1],  # Stock 1: 高市场暴露，高行业1暴露
    [1.0, 0.9, -0.2], # Stock 2
    [1.1, 0.7, 0.3],  # Stock 3
    [0.9, -0.1, 0.8], # Stock 4: 高行业2暴露
    [1.3, -0.2, 0.9], # Stock 5
    [1.0, 0.5, -0.1], # Stock 6
    [0.8, 0.6, 0.2],  # Stock 7
    [1.1, -0.3, 0.7], # Stock 8
    [0.9, 0.4, -0.4], # Stock 9
    [1.2, -0.1, 0.5], # Stock 10
])

# 生成股票收益率
returns_matrix = np.zeros((n_days, n_stocks))
for i in range(n_stocks):
    returns_matrix[:, i] = (
        betas[i, 0] * market_factor +
        betas[i, 1] * sector1_factor +
        betas[i, 2] * sector2_factor +
        np.random.normal(0, 0.004, n_days)  # 特异性收益
    )

# 转换为DataFrame
dates = pd.date_range('2023-01-01', periods=n_days, freq='B')
tickers = ['TECH1', 'TECH2', 'TECH3', 'FIN1', 'FIN2', 'TECH4', 'TECH5', 'FIN3', 'TECH6', 'FIN4']
returns = pd.DataFrame(returns_matrix, index=dates, columns=tickers)

print(f"模拟数据形状: {returns.shape}")
print(f"时间范围: {returns.index[0].date()} 到 {returns.index[-1].date()}")

# ========== 2. PCA分析 ==========
print("\n执行PCA分析...")
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

pca = PCA(n_components=n_stocks)
pca.fit(returns_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print(f"\n前5个主成分解释方差:")
for i in range(5):
    print(f"  PC{i+1}: {explained_variance[i]*100:.2f}% (累计: {cumulative_variance[i]*100:.2f}%)")

# ========== 3. 生成图1: 碎石图 ==========
print("\n生成图1: PCA方差解释图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 单个主成分解释方差
axes[0].bar(range(1, len(explained_variance) + 1), explained_variance * 100, 
            color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
axes[0].set_xlabel('Principal Component', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Explained Variance (%)', fontsize=12, fontweight='bold')
axes[0].set_title('PCA Explained Variance Distribution', fontsize=14, fontweight='bold', pad=15)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[0].set_xticks(range(1, len(explained_variance) + 1))
axes[0].set_xlim(0.5, len(explained_variance) + 0.5)

# 累计解释方差
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 
             'bo-', linewidth=2.5, markersize=8, markerfacecolor='lightblue')
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2.5, alpha=0.7, label='80% Threshold')
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=2.5, alpha=0.7, label='95% Threshold')
axes[1].set_xlabel('Number of Components', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Cumulative Variance (%)', fontsize=12, fontweight='bold')
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold', pad=15)
axes[1].legend(fontsize=11, loc='lower right', framealpha=0.9)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1].set_xticks(range(1, len(cumulative_variance) + 1))
axes[1].set_xlim(0.5, len(cumulative_variance) + 0.5)
axes[1].set_ylim(0, 105)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pca_variance.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: pca_variance.png")

# ========== 4. 生成图2: 载荷热力图 ==========
print("生成图2: 主成分载荷热力图...")
loadings = pca.components_
loadings_df = pd.DataFrame(
    loadings.T, 
    index=tickers,
    columns=[f'PC{i+1}' for i in range(loadings.shape[0])]
)

plt.figure(figsize=(12, 6))
sns.heatmap(loadings_df.iloc[:, :5], 
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r', 
            center=0,
            cbar_kws={'label': 'Factor Loading', 'shrink': 0.8},
            annot_kws={'size': 10, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='gray')
plt.title('Stock Loadings on First 5 Principal Components', 
          fontsize=14, fontweight='bold', pad=20)
plt.xlabel('Principal Components', fontsize=12, fontweight='bold')
plt.ylabel('Stocks', fontsize=12, fontweight='bold')
plt.xticks(rotation=0, fontweight='bold')
plt.yticks(rotation=0, fontweight='bold')
plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pca_loadings.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: pca_loadings.png")

# ========== 5. 计算残差 ==========
print("\n计算残差序列...")
pc_scores = pca.transform(returns_scaled)

# 只用前3个主成分重构（将其他成分置零）
pc_scores_reduced = pc_scores.copy()
pc_scores_reduced[:, 3:] = 0

# 重构
returns_reconstructed = pca.inverse_transform(pc_scores_reduced)
residuals = returns_scaled - returns_reconstructed
residuals_df = pd.DataFrame(residuals, index=returns.index, columns=tickers)

print(f"残差统计:")
print(f"  均值范围: [{residuals_df.mean().min():.6f}, {residuals_df.mean().max():.6f}]")
print(f"  标准差范围: [{residuals_df.std().min():.6f}, {residuals_df.std().max():.6f}]")

# ========== 6. 生成图3: 残差时间序列 ==========
print("生成图3: 残差时间序列...")
fig, axes = plt.subplots(5, 2, figsize=(16, 18))
axes = axes.flatten()

for i, ticker in enumerate(tickers):
    axes[i].plot(residuals_df.index, residuals_df[ticker], linewidth=1.2, color='steelblue', alpha=0.8)
    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
    axes[i].axhline(y=2*residuals_df[ticker].std(), color='orange', linestyle=':', alpha=0.5, linewidth=1)
    axes[i].axhline(y=-2*residuals_df[ticker].std(), color='orange', linestyle=':', alpha=0.5, linewidth=1)
    axes[i].set_title(f'{ticker} Residual Series', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Date', fontsize=9)
    axes[i].set_ylabel('Residual (σ)', fontsize=9)
    axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
    axes[i].set_xlim(residuals_df.index[0], residuals_df.index[-1])
    # 只显示部分日期标签，避免重叠
    axes[i].tick_params(axis='x', rotation=45, labelsize=8)

plt.tight_layout(pad=2.0, h_pad=3.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/residuals_timeseries.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: residuals_timeseries.png")

# ========== 7. 模拟配对交易回测 ==========
print("\n模拟配对交易回测...")
# 选择两个最相关的股票
corr_matrix = residuals_df.corr()
print(f"\n残差相关系数矩阵（前5个股票）:")
print(corr_matrix.iloc[:5, :5].round(3))

# 找到相关系数最高的配对
max_corr = 0
best_pair = None
for i in range(n_stocks):
    for j in range(i+1, n_stocks):
        corr = abs(corr_matrix.iloc[i, j])
        if corr > max_corr:
            max_corr = corr
            best_pair = (tickers[i], tickers[j])

stock1, stock2 = best_pair
print(f"\n最优配对: {stock1} - {stock2} (相关系数: {max_corr:.3f})")

# 计算价差和Z分数
spread = residuals_df[stock1] - residuals_df[stock2]
z_score = (spread - spread.mean()) / spread.std()

# 生成交易信号
positions = pd.DataFrame(index=spread.index, columns=['position'])
positions['position'] = 0

for i in range(1, len(z_score)):
    if z_score.iloc[i-1] > 2.0 and positions['position'].iloc[i-1] == 0:
        positions.iloc[i, 0] = -1  # 做空stock1，做多stock2
    elif z_score.iloc[i-1] < -2.0 and positions['position'].iloc[i-1] == 0:
        positions.iloc[i, 0] = 1   # 做多stock1，做空stock2
    elif abs(z_score.iloc[i-1]) < 0.5 and positions['position'].iloc[i-1] != 0:
        positions.iloc[i, 0] = 0   # 平仓
    else:
        positions.iloc[i, 0] = positions['position'].iloc[i-1]

# 计算策略收益（使用原始收益率）
strategy_returns = positions['position'].shift(1) * (returns[stock1] - returns[stock2])
strategy_returns = strategy_returns.fillna(0)
cumulative_returns = (1 + strategy_returns).cumprod()

total_return = (cumulative_returns.iloc[-1] - 1) * 100
sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
max_dd = ((cumulative_returns / cumulative_returns.cummax()) - 1).min() * 100

print(f"\n回测结果:")
print(f"  总收益率: {total_return:.2f}%")
print(f"  夏普比率: {sharpe:.2f}")
print(f"  最大回撤: {max_dd:.2f}%")
print(f"  交易次数: {(positions['position'] != positions['position'].shift(1)).sum() // 2}")

# ========== 8. 生成图4: 配对交易回测 ==========
print("\n生成图4: 配对交易回测结果...")
fig, axes = plt.subplots(3, 1, figsize=(14, 11))

# Z分数
axes[0].plot(z_score.index, z_score, linewidth=1.5, color='darkblue', alpha=0.8)
axes[0].axhline(y=2, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Entry Threshold (+2σ)')
axes[0].axhline(y=-2, color='green', linestyle='--', linewidth=2, alpha=0.7, label='Entry Threshold (-2σ)')
axes[0].axhline(y=0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Exit Threshold (+/-0.5σ)')
axes[0].axhline(y=-0.5, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
axes[0].fill_between(z_score.index, 2, z_score, 
                      where=(z_score > 2), alpha=0.3, color='red', label='Short Zone')
axes[0].fill_between(z_score.index, -2, z_score,
                      where=(z_score < -2), alpha=0.3, color='green', label='Long Zone')
axes[0].set_title(f'Z-Score: {stock1} - {stock2}', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(loc='upper right', fontsize=9, framealpha=0.9)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[0].set_xlim(z_score.index[0], z_score.index[-1])
axes[0].tick_params(axis='x', rotation=45, labelsize=8)

# 仓位
axes[1].plot(positions.index, positions['position'], linewidth=2, color='purple', alpha=0.8)
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
axes[1].fill_between(positions.index, 0, positions['position'], 
                      where=(positions['position'] > 0), alpha=0.3, color='green')
axes[1].fill_between(positions.index, 0, positions['position'], 
                      where=(positions['position'] < 0), alpha=0.3, color='red')
axes[1].set_title('Position Changes', fontsize=13, fontweight='bold', pad=10)
axes[1].set_ylabel('Position (1: Long, -1: Short)', fontsize=10, fontweight='bold')
axes[1].set_yticks([-1, 0, 1])
axes[1].set_yticklabels(['-1 (Short)', '0 (Flat)', '1 (Long)'])
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1].set_xlim(positions.index[0], positions.index[-1])
axes[1].tick_params(axis='x', rotation=45, labelsize=8)

# 累计收益
axes[2].plot(cumulative_returns.index, cumulative_returns, linewidth=2.5, color='darkgreen', alpha=0.8)
axes[2].axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
axes[2].fill_between(cumulative_returns.index, 1, cumulative_returns, 
                      where=(cumulative_returns > 1), alpha=0.3, color='green')
axes[2].fill_between(cumulative_returns.index, 1, cumulative_returns,
                      where=(cumulative_returns < 1), alpha=0.3, color='red')
axes[2].set_title('Strategy Cumulative Returns', fontsize=13, fontweight='bold', pad=10)
axes[2].set_ylabel('Cumulative Return (Multiple)', fontsize=10, fontweight='bold')
axes[2].set_xlabel('Date', fontsize=10, fontweight='bold')
axes[2].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[2].set_xlim(cumulative_returns.index[0], cumulative_returns.index[-1])
axes[2].tick_params(axis='x', rotation=45, labelsize=8)

# 添加性能标注
textstr = f'Total Return: {total_return:.1f}%\nSharpe: {sharpe:.2f}\nMax DD: {max_dd:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[2].text(0.02, 0.98, textstr, transform=axes[2].transAxes, fontsize=9,
                verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout(pad=2.0, h_pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pair_trading_backtest.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: pair_trading_backtest.png")

# ========== 9. 生成图5: 残差分布直方图 ==========
print("生成图5: 残差分布直方图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 残差分布
n_bins = 50
axes[0].hist(residuals_df[stock1], bins=n_bins, alpha=0.6, color='steelblue', 
             edgecolor='black', linewidth=1.2, density=True, label=stock1)
axes[0].hist(residuals_df[stock2], bins=n_bins, alpha=0.6, color='orange', 
             edgecolor='black', linewidth=1.2, density=True, label=stock2)
axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2.5, alpha=0.7)
axes[0].set_xlabel('Residual Value (σ)', fontsize=12, fontweight='bold')
axes[0].set_ylabel('Probability Density', fontsize=12, fontweight='bold')
axes[0].set_title(f'Residual Distribution: {stock1} vs {stock2}', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=11, loc='upper right', framealpha=0.9)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 添加正态分布拟合曲线
from scipy import stats
x1 = np.linspace(residuals_df[stock1].min(), residuals_df[stock1].max(), 100)
y1 = stats.norm.pdf(x1, residuals_df[stock1].mean(), residuals_df[stock1].std())
axes[0].plot(x1, y1, '--', linewidth=2, alpha=0.7, color='blue', label='Normal Fit')

x2 = np.linspace(residuals_df[stock2].min(), residuals_df[stock2].max(), 100)
y2 = stats.norm.pdf(x2, residuals_df[stock2].mean(), residuals_df[stock2].std())
axes[0].plot(x2, y2, '--', linewidth=2, alpha=0.7, color='orange', label='Normal Fit')

# Spread分布
axes[1].hist(spread, bins=n_bins, alpha=0.7, color='green', 
             edgecolor='black', linewidth=1.2, density=True)
axes[1].axvline(x=spread.mean(), color='red', linestyle='--', linewidth=2.5, 
                alpha=0.8, label=f'Mean: {spread.mean():.3f}')
axes[1].axvline(x=spread.mean() + 2*spread.std(), color='orange', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'+2σ: {spread.mean() + 2*spread.std():.3f}')
axes[1].axvline(x=spread.mean() - 2*spread.std(), color='orange', linestyle='--', 
                linewidth=2, alpha=0.8, label=f'-2σ: {spread.mean() - 2*spread.std():.3f}')
axes[1].set_xlabel('Spread (Residual1 - Residual2)', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Probability Density', fontsize=12, fontweight='bold')
axes[1].set_title(f'Spread Distribution: {stock1} - {stock2}', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=9, loc='upper right', framealpha=0.9)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# 添加正态分布拟合
x_spread = np.linspace(spread.min(), spread.max(), 100)
y_spread = stats.norm.pdf(x_spread, spread.mean(), spread.std())
axes[1].plot(x_spread, y_spread, '--', linewidth=2, alpha=0.7, color='green', label='Normal Fit')
axes[1].legend(fontsize=9, loc='upper right', framealpha=0.9)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/spread_distribution.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: spread_distribution.png")

# ========== 10. 生成图6: 配对股票累计收益对比 ==========
print("生成图6: 配对股票累计收益对比...")
fig, ax = plt.subplots(figsize=(12, 6))

# 计算累计收益
cumret1 = (1 + returns[stock1]).cumprod()
cumret2 = (1 + returns[stock2]).cumprod()

ax.plot(cumret1.index, cumret1, linewidth=2.5, color='steelblue', alpha=0.8, label=f'{stock1} (Final: {cumret1.iloc[-1]:.2f}x)')
ax.plot(cumret2.index, cumret2, linewidth=2.5, color='orange', alpha=0.8, label=f'{stock2} (Final: {cumret2.iloc[-1]:.2f}x)')
ax.axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1.5)

ax.set_xlabel('Date', fontsize=12, fontweight='bold')
ax.set_ylabel('Cumulative Return (Multiple)', fontsize=12, fontweight='bold')
ax.set_title(f'Cumulative Returns: {stock1} vs {stock2}', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.tick_params(axis='x', rotation=45, labelsize=9)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/cumulative_returns_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: cumulative_returns_comparison.png")

# ========== 总结 ==========
print("\n" + "="*60)
print("所有配图生成完成！")
print("="*60)
print(f"图片保存位置: /Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/")
print("\n生成的图片:")
print("  1. pca_variance.png - PCA方差解释图")
print("  2. pca_loadings.png - 主成分载荷热力图")
print("  3. residuals_timeseries.png - 残差时间序列")
print("  4. pair_trading_backtest.png - 配对交易回测")
print("  5. spread_distribution.png - 残差分布直方图")
print("  6. cumulative_returns_comparison.png - 累计收益对比")
print("\n" + "="*60)
