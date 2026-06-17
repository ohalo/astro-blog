#!/usr/bin/env python3
"""
为PCA与统计套利文章生成配图（无sklearn依赖）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage', exist_ok=True)

# 生成示例数据
np.random.seed(42)
n_days = 1000
n_stocks = 10

dates = pd.date_range('2020-01-01', periods=n_days, freq='B')
stock_names = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM']

# 生成相关的收益率数据
market_factor = np.random.normal(0.0005, 0.01, n_days)
tech_factor = np.random.normal(0.0002, 0.008, n_days)

returns_data = {}
for i, stock in enumerate(stock_names):
    idiosyncratic = np.random.normal(0, 0.015, n_days)
    beta_market = 0.8 + 0.4 * np.random.rand()
    beta_tech = 0.6 + 0.3 * np.random.rand()
    returns = (beta_market * market_factor + 
               beta_tech * tech_factor + 
               idiosyncratic)
    returns_data[stock] = returns

returns_df = pd.DataFrame(returns_data, index=dates)

print("✓ 示例数据已生成")

# 手动实现PCA
def manual_pca(data, n_components=None):
    """
    手动实现PCA
    
    Parameters:
    -----------
    data : array
        数据矩阵（样本 × 特征）
    n_components : int or None
        主成分数量
    
    Returns:
    --------
    result : dict
        PCA结果
    """
    # 标准化
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    data_scaled = (data - mean) / std
    
    # 计算协方差矩阵
    cov_matrix = np.cov(data_scaled.T)
    
    # 特征值分解
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # 降序排列
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 选择主成分
    if n_components is not None:
        eigenvectors = eigenvectors[:, :n_components]
        eigenvalues = eigenvalues[:n_components]
    
    # 计算解释方差比例
    explained_variance_ratio = eigenvalues / np.sum(np.linalg.eigvalsh(cov_matrix))
    
    # 计算主成分得分
    components = eigenvectors.T
    scores = np.dot(data_scaled, eigenvectors)
    
    result = {
        'components': components,
        'explained_variance_ratio': explained_variance_ratio,
        'scores': scores,
        'mean': mean,
        'std': std
    }
    
    return result

# 执行PCA
returns_array = returns_df.values
pca_result = manual_pca(returns_array, n_components=None)

print("✓ PCA分析完成")

# 图1: PCA分析结果 - 封面图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('PCA Analysis on Stock Returns', fontsize=18, fontweight='bold')

# 子图1：解释方差比例
ax1 = axes[0, 0]
n_comps = len(pca_result['explained_variance_ratio'])
ax1.bar(range(1, min(15, n_comps) + 1), 
        pca_result['explained_variance_ratio'][:15], 
        alpha=0.7, color='steelblue', edgecolor='black')
ax1.set_xlabel('Principal Component', fontsize=12)
ax1.set_ylabel('Explained Variance Ratio', fontsize=12)
ax1.set_title('Individual Explained Variance', fontsize=14)
ax1.grid(True, alpha=0.3, axis='y')

# 子图2：累积解释方差
ax2 = axes[0, 1]
cum_variance = np.cumsum(pca_result['explained_variance_ratio'])
ax2.plot(range(1, min(15, n_comps) + 1), 
         cum_variance[:15], 
         marker='o', linewidth=2.5, color='darkorange', markersize=8)
ax2.axhline(y=0.8, color='red', linestyle='--', linewidth=2, label='80% Variance', alpha=0.7)
ax2.axhline(y=0.9, color='green', linestyle='--', linewidth=2, label='90% Variance', alpha=0.7)
ax2.set_xlabel('Number of Components', fontsize=12)
ax2.set_ylabel('Cumulative Explained Variance', fontsize=12)
ax2.set_title('Cumulative Explained Variance', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 子图3：第一主成分载荷
ax3 = axes[1, 0]
pc1_loadings = pca_result['components'][0]
n_top = min(10, len(stock_names))
top_assets_idx = np.argsort(np.abs(pc1_loadings))[::-1][:n_top]
y_pos = np.arange(n_top)
ax3.barh(y_pos, pc1_loadings[top_assets_idx][::-1], 
          alpha=0.7, color='green', edgecolor='black')
ax3.set_yticks(y_pos)
ax3.set_yticklabels([stock_names[i] for i in top_assets_idx][::-1])
ax3.set_xlabel('Loading (Weight)', fontsize=12)
ax3.set_title('PC1 Loadings (Top 15 Assets)', fontsize=14)
ax3.grid(True, alpha=0.3, axis='x')
ax3.axvline(x=0, color='black', linewidth=0.8)

# 子图4：第二主成分载荷
ax4 = axes[1, 1]
pc2_loadings = pca_result['components'][1]
n_top = min(10, len(stock_names))
top_assets_idx2 = np.argsort(np.abs(pc2_loadings))[::-1][:n_top]
ax4.barh(y_pos, pc2_loadings[top_assets_idx2][::-1], 
          alpha=0.7, color='purple', edgecolor='black')
ax4.set_yticks(y_pos)
ax4.set_yticklabels([stock_names[i] for i in top_assets_idx2][::-1])
ax4.set_xlabel('Loading (Weight)', fontsize=12)
ax4.set_title('PC2 Loadings (Top 15 Assets)', fontsize=14)
ax4.grid(True, alpha=0.3, axis='x')
ax4.axvline(x=0, color='black', linewidth=0.8)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 封面图已生成: cover.jpg")

# 图2: 因子模型示意图
fig, ax = plt.subplots(figsize=(14, 8))

# 模拟因子收益
n_factors = 3
factor_returns = pd.DataFrame({
    'Market_Factor': np.random.normal(0.0005, 0.01, n_days),
    'Tech_Factor': np.random.normal(0.0002, 0.008, n_days),
    'Value_Factor': np.random.normal(0.0001, 0.006, n_days)
}, index=dates)

# 绘制因子收益
colors = ['#2c3e50', '#e74c3c', '#3498db']
for i, col in enumerate(factor_returns.columns):
    cum_ret = (1 + factor_returns[col]).cumprod() - 1
    ax.plot(dates, cum_ret * 100, linewidth=2.5, 
            label=col.replace('_', ' '), color=colors[i], alpha=0.8)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.set_title('Factor Model: Extracted Factors (PCA)', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/factor_model.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 因子模型图已生成: factor_model.png")

# 图3: 配对交易策略示意图
# 生成模拟的配对交易数据
np.random.seed(123)
n_days_pair = 500
dates_pair = pd.date_range('2021-01-01', periods=n_days_pair, freq='B')

# 生成协整的残差序列
spread = np.random.normal(0, 1, n_days_pair)
# 添加均值回归特性
for t in range(1, n_days_pair):
    spread[t] = 0.95 * spread[t-1] + np.random.normal(0, 0.1)

spread_series = pd.Series(spread, index=dates_pair)
z_score = (spread - pd.Series(spread).rolling(63).mean()) / pd.Series(spread).rolling(63).std()

# 生成交易信号
# 确保索引对齐
signals = pd.Series(0, index=dates_pair)
z_score_series = pd.Series(z_score, index=dates_pair)

signals.loc[z_score_series > 2.0] = -1
signals.loc[z_score_series < -2.0] = 1
signals.loc[np.abs(z_score_series) < 0.5] = 0

# 计算策略收益
price_diff = pd.Series(spread).diff()
strategy_returns = signals.shift(1) * price_diff
cumulative_returns = (1 + strategy_returns).cumprod() - 1

fig, axes = plt.subplots(3, 1, figsize=(16, 14))
fig.suptitle('Pairs Trading Strategy: Residuals-Based', fontsize=18, fontweight='bold')

# 子图1：价差和z-score
ax1 = axes[0]
ax1_twin = ax1.twinx()

ax1.plot(spread_series.index, spread_series, 
         linewidth=2, color='blue', label='Spread', alpha=0.8)
ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
ax1.set_ylabel('Spread', fontsize=12, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.grid(True, alpha=0.3)

ax1_twin.plot(z_score.index, z_score, 
               linewidth=1.5, color='red', alpha=0.7, label='Z-Score')
ax1_twin.axhline(y=2, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1_twin.axhline(y=-2, color='red', linestyle='--', alpha=0.7, linewidth=2)
ax1_twin.axhline(y=0, color='green', linestyle='--', alpha=0.7, linewidth=2)
ax1_twin.set_ylabel('Z-Score', fontsize=12, color='red')
ax1_twin.tick_params(axis='y', labelcolor='red')

ax1.set_title('Spread and Z-Score', fontsize=14)

# 子图2：交易信号
ax2 = axes[1]
ax2.plot(z_score.index, z_score, 
         linewidth=1.5, color='gray', alpha=0.5, label='Z-Score')

long_signals = signals == 1
short_signals = signals == -1

ax2.scatter(z_score.index[long_signals], 
           z_score[long_signals], 
           color='green', s=60, label='Long', zorder=5, alpha=0.8)
ax2.scatter(z_score.index[short_signals], 
           z_score[short_signals], 
           color='red', s=60, label='Short', zorder=5, alpha=0.8)

ax2.axhline(y=2, color='red', linestyle='--', alpha=0.7, label='Entry Threshold')
ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.7)
ax2.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, label='Exit Threshold')
ax2.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7)

ax2.set_ylabel('Z-Score', fontsize=12)
ax2.set_title('Trading Signals', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 子图3：累积收益
ax3 = axes[2]
ax3.plot(cumulative_returns.index, cumulative_returns * 100, 
         linewidth=3, color='darkgreen', label='Strategy', alpha=0.9)
ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
ax3.fill_between(cumulative_returns.index, 
                cumulative_returns * 100, 
                alpha=0.3, color='darkgreen')
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Cumulative Return (%)', fontsize=12)
ax3.set_title('Cumulative Returns', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)

plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/pairs_trading.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 配对交易策略图已生成: pairs_trading.png")

# 图4: 统计套利组合表现
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Statistical Arbitrage Portfolio Performance', fontsize=18, fontweight='bold')

# 模拟5个配对的表现
n_pairs = 4
pair_names = ['AAPL-MSFT', 'GOOGL-AMZN', 'META-NVDA', 'TSLA-AMD']

for i, pair in enumerate(pair_names):
    # 生成模拟的累积收益
    np.random.seed(i * 10)
    pair_ret = np.random.normal(0.0003, 0.005, n_days)
    # 添加均值回归特性
    for t in range(1, n_days):
        pair_ret[t] = 0.9 * pair_ret[t-1] + np.random.normal(0.0003, 0.002)
    
    pair_cum_ret = np.cumprod(1 + pair_ret) - 1
    
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    ax.plot(dates, pair_cum_ret * 100, linewidth=2.5, 
            color=colors[i % len(colors)], alpha=0.9)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.set_title(pair, fontsize=13, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加绩效指标
    total_ret = pair_cum_ret.iloc[-1]
    sharpe = (pair_ret.mean() / pair_ret.std()) * np.sqrt(252)
    ax.text(0.02, 0.98, f'Total: {total_ret:.1%}\nSharpe: {sharpe:.2f}', 
            transform=ax.transAxes, fontsize=10,
            verticalalignment='top', 
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/portfolio_performance.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 组合表现图已生成: portfolio_performance.png")

# 图5: PCA vs 传统配对的对比
fig, ax = plt.subplots(figsize=(14, 8))

# 模拟策略收益对比
np.random.seed(42)
n_days_compare = 1000
dates_compare = pd.date_range('2020-01-01', periods=n_days_compare, freq='B')

# 策略1：传统配对交易（无PCA）
traditional_ret = np.random.normal(0.0002, 0.008, n_days_compare)
traditional_ret[500:520] = np.random.normal(-0.02, 0.015, 20)  # 添加一段回撤
traditional_cum = np.cumprod(1 + traditional_ret) - 1

# 策略2：基于PCA的统计套利
pca_ret = np.random.normal(0.0004, 0.006, n_days_compare)
pca_cum = np.cumprod(1 + pca_ret) - 1

# 策略3：买入持有基准
benchmark_ret = np.random.normal(0.0003, 0.012, n_days_compare)
benchmark_cum = np.cumprod(1 + benchmark_ret) - 1

ax.plot(dates_compare, traditional_cum * 100, 
        linewidth=2.5, color='orange', label='Traditional Pairs Trading', alpha=0.8)
ax.plot(dates_compare, pca_cum * 100, 
        linewidth=2.5, color='green', label='PCA-Based Stat Arb', alpha=0.8)
ax.plot(dates_compare, benchmark_cum * 100, 
        linewidth=2, color='gray', label='Buy & Hold Benchmark', alpha=0.6, linestyle='--')

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Cumulative Return (%)', fontsize=12)
ax.set_title('Performance Comparison: PCA vs Traditional Methods', fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

# 添加绩效指标框
textstr = 'Sharpe Ratio:\n'
textstr += f'Traditional: {traditional_ret.mean()/traditional_ret.std()*np.sqrt(252):.2f}\n'
textstr += f'PCA-Based: {pca_ret.mean()/pca_ret.std()*np.sqrt(252):.2f}\n'
textstr += f'Benchmark: {benchmark_ret.mean()/benchmark_ret.std()*np.sqrt(252):.2f}'

props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.97, textstr, transform=ax.transAxes, fontsize=11,
        verticalalignment='top', bbox=props)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage/performance_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 性能对比图已生成: performance_comparison.png")

print("\n" + "="*60)
print("所有配图已成功生成！")
print("="*60)
print("\n生成的图片文件：")
print("  1. cover.jpg - PCA分析结果（封面）")
print("  2. factor_model.png - 因子模型示意图")
print("  3. pairs_trading.png - 配对交易策略")
print("  4. portfolio_performance.png - 组合表现")
print("  5. performance_comparison.png - 性能对比")
print("="*60)
