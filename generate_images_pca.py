#!/usr/bin/env python3
"""
为PCA统计套利文章生成配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ========== 1. 数据获取 ==========
print("正在下载股票数据...")
tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'CRM', 'ORCL', 'ADBE']
start_date = '2023-01-01'
end_date = '2024-12-31'

data = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True)['Close']
returns = np.log(data / data.shift(1)).dropna()

print(f"数据形状: {returns.shape}")

# ========== 2. PCA分析 ==========
print("执行PCA分析...")
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

pca = PCA(n_components=len(tickers))
pca.fit(returns_scaled)

explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

# ========== 3. 生成图1: 碎石图 ==========
print("生成图1: PCA方差解释图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 单个主成分解释方差
axes[0].bar(range(1, len(explained_variance) + 1), explained_variance * 100, 
            color='steelblue', alpha=0.8, edgecolor='black')
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance (%)', fontsize=12)
axes[0].set_title('PCA Explained Variance Distribution', fontsize=14, fontweight='bold')
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].set_xticks(range(1, len(explained_variance) + 1))

# 累计解释方差
axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, 
             'bo-', linewidth=2, markersize=6)
axes[1].axhline(y=80, color='red', linestyle='--', linewidth=2, label='80% Threshold')
axes[1].axhline(y=95, color='green', linestyle='--', linewidth=2, label='95% Threshold')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Variance (%)', fontsize=12)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, linestyle='--')
axes[1].set_xticks(range(1, len(cumulative_variance) + 1))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pca_variance.png', 
            dpi=300, bbox_inches='tight')
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

plt.figure(figsize=(12, 8))
sns.heatmap(loadings_df.iloc[:, :5], 
            annot=True, 
            fmt='.3f', 
            cmap='RdBu_r', 
            center=0,
            cbar_kws={'label': 'Factor Loading'})
plt.title('Stock Loadings on First 5 Principal Components', fontsize=14, fontweight='bold')
plt.xlabel('Principal Components', fontsize=12)
plt.ylabel('Stocks', fontsize=12)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pca_loadings.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ 已保存: pca_loadings.png")

# ========== 5. 计算残差 ==========
print("计算残差序列...")
pc_scores = pca.transform(returns_scaled)
returns_reconstructed = pca.inverse_transform(pc_scores[:, :3])  # 只用前3个主成分
residuals = returns_scaled - returns_reconstructed
residuals_df = pd.DataFrame(residuals, index=returns.index, columns=tickers)

# ========== 6. 生成图3: 残差时间序列 ==========
print("生成图3: 残差时间序列...")
fig, axes = plt.subplots(5, 2, figsize=(16, 20))
axes = axes.flatten()

for i, ticker in enumerate(tickers[:10]):
    axes[i].plot(residuals_df.index, residuals_df[ticker], linewidth=1, color='steelblue')
    axes[i].axhline(y=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    axes[i].set_title(f'{ticker} Residual Series', fontsize=11, fontweight='bold')
    axes[i].set_xlabel('Date', fontsize=10)
    axes[i].set_ylabel('Residual', fontsize=10)
    axes[i].grid(True, alpha=0.3, linestyle='--')
    axes[i].set_xlim(residuals_df.index[0], residuals_df.index[-1])

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/residuals_timeseries.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("  ✓ 已保存: residuals_timeseries.png")

# ========== 7. 寻找协整对并回测 ==========
print("寻找协整对...")
coint_pairs = []
n = len(residuals_df.columns)

for i in range(n):
    for j in range(i+1, n):
        stock1 = residuals_df.columns[i]
        stock2 = residuals_df.columns[j]
        score, p_value, _ = coint(residuals_df[stock1], residuals_df[stock2])
        if p_value < 0.05:
            coint_pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'p_value': p_value
            })

if len(coint_pairs) > 0:
    coint_pairs_df = pd.DataFrame(coint_pairs)
    best_pair = coint_pairs_df.sort_values('p_value').iloc[0]
    stock1, stock2 = best_pair['stock1'], best_pair['stock2']
    
    print(f"最优协整对: {stock1} - {stock2}")
    
    # 计算Z分数和策略收益
    spread = residuals_df[stock1] - residuals_df[stock2]
    z_score = (spread - spread.mean()) / spread.std()
    
    # 生成交易信号
    positions = pd.DataFrame(index=spread.index, columns=['position'])
    positions['position'] = 0
    
    for i in range(1, len(z_score)):
        if z_score.iloc[i-1] > 2.0 and positions['position'].iloc[i-1] == 0:
            positions.iloc[i, 0] = -1
        elif z_score.iloc[i-1] < -2.0 and positions['position'].iloc[i-1] == 0:
            positions.iloc[i, 0] = 1
        elif abs(z_score.iloc[i-1]) < 0.5 and positions['position'].iloc[i-1] != 0:
            positions.iloc[i, 0] = 0
        else:
            positions.iloc[i, 0] = positions['position'].iloc[i-1]
    
    # 计算策略收益
    returns1 = returns[stock1]
    returns2 = returns[stock2]
    strategy_returns = positions['position'].shift(1) * (returns1 - returns2)
    strategy_returns = strategy_returns.fillna(0)
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # ========== 8. 生成图4: 配对交易回测 ==========
    print("生成图4: 配对交易回测结果...")
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # Z分数
    axes[0].plot(z_score.index, z_score, linewidth=1.5, color='darkblue')
    axes[0].axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold (+2σ)')
    axes[0].axhline(y=-2, color='green', linestyle='--', alpha=0.7, label='Entry Threshold (-2σ)')
    axes[0].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Exit Threshold (+/-0.5σ)')
    axes[0].axhline(y=-0.5, color='orange', linestyle='--', alpha=0.7)
    axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[0].set_title(f'Z-Score: {stock1} - {stock2}', fontsize=13, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xlim(z_score.index[0], z_score.index[-1])
    
    # 仓位
    axes[1].plot(positions.index, positions['position'], linewidth=1.5, color='purple')
    axes[1].set_title('Position Changes', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Position (1: Long, -1: Short)', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_yticks([-1, 0, 1])
    axes[1].set_xlim(positions.index[0], positions.index[-1])
    
    # 累计收益
    axes[2].plot(cumulative_returns.index, cumulative_returns, linewidth=2, color='darkgreen')
    axes[2].axhline(y=1, color='black', linestyle='--', alpha=0.5)
    axes[2].set_title('Strategy Cumulative Returns', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Cumulative Return (Multiple)', fontsize=10)
    axes[2].set_xlabel('Date', fontsize=10)
    axes[2].grid(True, alpha=0.3, linestyle='--')
    axes[2].set_xlim(cumulative_returns.index[0], cumulative_returns.index[-1])
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pair_trading_backtest.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ 已保存: pair_trading_backtest.png")
    
    # ========== 9. 额外生成图5: 配对股票残差分布 ==========
    print("生成图5: 残差分布直方图...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 残差分布
    axes[0].hist(residuals_df[stock1], bins=50, alpha=0.7, color='steelblue', edgecolor='black', label=stock1)
    axes[0].hist(residuals_df[stock2], bins=50, alpha=0.7, color='orange', edgecolor='black', label=stock2)
    axes[0].axvline(x=0, color='red', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Residual Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Residual Distribution: {stock1} vs {stock2}', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # Spread分布
    axes[1].hist(spread, bins=50, alpha=0.7, color='green', edgecolor='black')
    axes[1].axvline(x=spread.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {spread.mean():.4f}')
    axes[1].axvline(x=spread.mean() + 2*spread.std(), color='orange', linestyle='--', linewidth=2, label=f'+2σ: {spread.mean() + 2*spread.std():.4f}')
    axes[1].axvline(x=spread.mean() - 2*spread.std(), color='orange', linestyle='--', linewidth=2, label=f'-2σ: {spread.mean() - 2*spread.std():.4f}')
    axes[1].set_xlabel('Spread (Residual1 - Residual2)', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'Spread Distribution: {stock1} - {stock2}', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/spread_distribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ 已保存: spread_distribution.png")
    
else:
    print("未找到协整对，跳过配对交易可视化")

print("\n" + "="*50)
print("所有配图生成完成！")
print("="*50)
print(f"图片保存位置: /Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/")
print("生成的图片:")
print("  1. pca_variance.png - PCA方差解释图")
print("  2. pca_loadings.png - 主成分载荷热力图")
print("  3. residuals_timeseries.png - 残差时间序列")
print("  4. pair_trading_backtest.png - 配对交易回测")
print("  5. spread_distribution.png - 残差分布直方图")
