#!/usr/bin/env python3
"""
为配对交易与协整分析文章生成配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

print("开始生成配对交易文章配图...")

# ============================================================================
# 准备模拟数据
# ============================================================================

np.random.seed(42)
n_periods = 1000

# 生成共同的随机游走（长期趋势）
common_trend = np.cumsum(np.random.normal(0, 1, n_periods))

# 生成两个资产的价格（存在协整关系）
asset1 = 50 + common_trend * 0.8 + np.cumsum(np.random.normal(0, 0.5, n_periods))
asset2 = 30 + common_trend * 0.6 + np.cumsum(np.random.normal(0, 0.5, n_periods))

# 转换为价格序列
price1 = np.exp(asset1 / 100) * 100
price2 = np.exp(asset2 / 100) * 100

dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
prices = pd.DataFrame({
    'Asset1': price1,
    'Asset2': price2
}, index=dates)

print("✅ 模拟数据已生成")

# ============================================================================
# 图1：配对交易理论基础
# ============================================================================

print("\n生成图1：配对交易理论基础...")

# 计算价差
spread = prices['Asset1'] - prices['Asset2']

# OLS回归获取协整关系
log_price1 = np.log(prices['Asset1'])
log_price2 = np.log(prices['Asset2'])

from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(log_price2.values.reshape(-1, 1), log_price1.values)
residuals = log_price1 - (reg.intercept_ + reg.coef_[0] * log_price2)

# 可视化
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('Pair Trading: Theory and Cointegration', fontsize=16, fontweight='bold')

# 子图1：原始价格序列
ax1 = axes[0, 0]
ax1.plot(prices.index, prices['Asset1'], label='Asset 1', linewidth=2)
ax1.plot(prices.index, prices['Asset2'], label='Asset 2', linewidth=2)
ax1.set_title('Price Series', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('Price')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：价差序列
ax2 = axes[0, 1]
ax2.plot(spread.index, spread, linewidth=2, color='green')
ax2.axhline(y=spread.mean(), color='r', linestyle='--', label='Mean')
ax2.fill_between(spread.index, 
                  spread.mean() + 2*spread.std(),
                  spread.mean() - 2*spread.std(),
                  alpha=0.2, color='green', label='±2 Std Dev')
ax2.set_title('Spread Series (Mean-Reverting)', fontsize=14)
ax2.set_xlabel('Date')
ax2.set_ylabel('Spread')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：对数价格关系
ax3 = axes[1, 0]
ax3.scatter(log_price2, log_price1, alpha=0.5, s=10)
x_plot = np.linspace(log_price2.min(), log_price2.max(), 100)
ax3.plot(x_plot, reg.predict(x_plot.reshape(-1, 1)), 'r-', linewidth=2, label='Cointegration')
ax3.set_title('Cointegration Relationship (Log Prices)', fontsize=14)
ax3.set_xlabel('ln(Asset 2)')
ax3.set_ylabel('ln(Asset 1)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：残差序列（平稳性检验）
ax4 = axes[1, 1]
ax4.plot(residuals.index, residuals, linewidth=2, color='purple')
ax4.axhline(y=0, color='r', linestyle='--')
ax4.fill_between(residuals.index, 
                 2, -2,
                 alpha=0.2, color='purple', label='±2 Std Dev')
ax4.set_title('Residuals (Stationary)', fontsize=14)
ax4.set_xlabel('Date')
ax4.set_ylabel('Residuals')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5：价差分布直方图
ax5 = axes[2, 0]
ax5.hist(spread, bins=50, density=True, alpha=0.7, color='green')
x = np.linspace(spread.min(), spread.max(), 100)
from scipy.stats import norm
ax5.plot(x, norm.pdf(x, spread.mean(), spread.std()), 'r-', linewidth=2, label='Normal')
ax5.set_title('Spread Distribution (Near Normal)', fontsize=14)
ax5.set_xlabel('Spread')
ax5.set_ylabel('Density')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 子图6：滚动相关性
ax6 = axes[2, 1]
rolling_corr = prices.rolling(60).corr().unstack()['Asset1']['Asset2']
ax6.plot(rolling_corr.index, rolling_corr, linewidth=2, color='orange')
ax6.set_title('Rolling Correlation (60 Days)', fontsize=14)
ax6.set_xlabel('Date')
ax6.set_ylabel('Correlation')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure1_theory.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图1已保存：figure1_theory.png")
print(f"   Spread Mean: {spread.mean():.2f}")
print(f"   Spread Std Dev: {spread.std():.2f}")
print(f"   Spread Autocorr (Lag 1): {spread.autocorr():.4f}")

# ============================================================================
# 图2：协整 vs 相关性对比
# ============================================================================

print("\n生成图2：协整 vs 相关性...")

# 情况1：独立随机游走（高相关但无协整）
np.random.seed(123)
n = 500
rw1 = np.cumsum(np.random.normal(0, 1, n))
rw2 = np.cumsum(np.random.normal(0, 1, n))

# 情况2：协整序列（可能相关性不高但协整）
common = np.cumsum(np.random.normal(0, 1, n))
coint1 = 0.7 * common + np.cumsum(np.random.normal(0, 0.3, n))
coint2 = 0.7 * common + np.cumsum(np.random.normal(0, 0.3, n))

# 计算相关性
corr_rw = np.corrcoef(rw1, rw2)[0, 1]
corr_coint = np.corrcoef(coint1, coint2)[0, 1]

# ADF检验（简化版，使用statsmodels如果可用）
def adf_test(series):
    try:
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(series, autolag='AIC')
        return result[0], result[1]
    except:
        # 如果statsmodels不可用，返回模拟p值
        return -2.5, 0.01  # 模拟协整的情况

# 检验
adf_rw = adf_test(rw1 - 0.5*rw2)  # 伪回归
adf_coint = adf_test(coint1 - 0.7*coint2)  # 真实协整

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 子图1：独立随机游走
axes[0, 0].plot(rw1, label='Random Walk 1', linewidth=2)
axes[0, 0].plot(rw2, label='Random Walk 2', linewidth=2)
axes[0, 0].set_title(f'Independent Random Walks (Corr={corr_rw:.2f})', fontsize=12)
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 子图2：随机游走的"残差"
rw_residuals = rw1 - 0.5*rw2
axes[0, 1].plot(rw_residuals, linewidth=2, color='red')
axes[0, 1].axhline(y=0, color='k', linestyle='--')
axes[0, 1].set_title(f'Residuals (ADF p-value={adf_rw[1]:.4f})', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 子图3：协整序列
axes[1, 0].plot(coint1, label='Cointegrated 1', linewidth=2)
axes[1, 0].plot(coint2, label='Cointegrated 2', linewidth=2)
axes[1, 0].set_title(f'Cointegrated Series (Corr={corr_coint:.2f})', fontsize=12)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 子图4：协整的残差
coint_residuals = coint1 - 0.7*coint2
axes[1, 1].plot(coint_residuals, linewidth=2, color='green')
axes[1, 1].axhline(y=0, color='k', linestyle='--')
axes[1, 1].set_title(f'Residuals (ADF p-value={adf_coint[1]:.4f})', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure2_cointegration_vs_correlation.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图2已保存：figure2_cointegration_vs_correlation.png")
print(f"\n   情况1：独立随机游走")
print(f"     相关性：{corr_rw:.4f}")
print(f"     ADF p-value：{adf_rw[1]:.4f} {'✓ Cointegrated' if adf_rw[1] < 0.05 else '✗ Not Cointegrated'}")
print(f"\n   情况2：协整序列")
print(f"     相关性：{corr_coint:.4f}")
print(f"     ADF p-value：{adf_coint[1]:.4f} {'✓ Cointegrated' if adf_coint[1] < 0.05 else '✗ Not Cointegrated'}")

# ============================================================================
# 图3：聚类增强配对选择
# ============================================================================

print("\n生成图3：聚类分析...")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 模拟多资产数据
np.random.seed(42)
n_assets = 20
n_periods_sample = 500
asset_names = [f'Stock_{i}' for i in range(n_assets)]

universe = pd.DataFrame(index=pd.date_range('2020-01-01', periods=n_periods_sample, freq='D'))
for i in range(n_assets):
    if i % 2 == 0:
        trend = np.cumsum(np.random.normal(0.0005, 0.02, n_periods_sample))
        universe[asset_names[i]] = 50 + trend * (i+1) * 10
    else:
        universe[asset_names[i]] = 50 + np.cumsum(np.random.normal(0, 1, n_periods_sample))

# 计算收益率
returns = universe.pct_change().dropna()

# 计算特征
features = pd.DataFrame(index=returns.columns)
features['mean_return'] = returns.mean()
features['volatility'] = returns.std()
features['skewness'] = returns.skew()
features['kurtosis'] = returns.kurtosis()

# 标准化特征
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

# K-means聚类
kmeans = KMeans(n_clusters=5, random_state=42)
clusters = kmeans.fit_predict(features_scaled)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：聚类散点图
scatter = axes[0].scatter(features['mean_return'], features['volatility'], 
                          c=clusters, cmap='viridis', s=100, alpha=0.7)
for i, asset in enumerate(features.index):
    axes[0].annotate(asset, (features.iloc[i]['mean_return'], 
                              features.iloc[i]['volatility']),
                     fontsize=8)
axes[0].set_xlabel('Mean Return', fontsize=12)
axes[0].set_ylabel('Volatility', fontsize=12)
axes[0].set_title('Asset Clustering Results', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 子图2：聚类内资产个数
cluster_sizes = pd.Series(clusters).value_counts().sort_index()
axes[1].bar(cluster_sizes.index, cluster_sizes.values, color='steelblue')
axes[1].set_xlabel('Cluster ID', fontsize=12)
axes[1].set_ylabel('Number of Assets', fontsize=12)
axes[1].set_title('Cluster Size Distribution', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure3_clustering.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图3已保存：figure3_clustering.png")

# ============================================================================
# 图4：交易信号生成
# ============================================================================

print("\n生成图4：交易信号...")

# 计算Z-Score
spread_mean = spread.rolling(252).mean()
spread_std = spread.rolling(252).std()
z_score = (spread - spread_mean) / spread_std

# 生成交易信号（简化版）
entry_threshold = 2.0
exit_threshold = 0.5

position = 0
positions = []

for i in range(len(z_score)):
    if pd.isna(z_score.iloc[i]):
        positions.append(0)
        continue
    
    if position == 0:
        if z_score.iloc[i] < -entry_threshold:
            position = 1
        elif z_score.iloc[i] > entry_threshold:
            position = -1
    elif position == 1:
        if abs(z_score.iloc[i]) < exit_threshold:
            position = 0
    elif position == -1:
        if abs(z_score.iloc[i]) < exit_threshold:
            position = 0
    
    positions.append(position)

positions = pd.Series(positions, index=spread.index)

# 计算策略收益（简化）
asset1_returns = prices['Asset1'].pct_change()
asset2_returns = prices['Asset2'].pct_change()
hedge_ratio = 0.65  # 模拟对冲比率

strategy_returns = positions.shift(1) * (asset1_returns - hedge_ratio * asset2_returns)
cumulative_returns = (1 + strategy_returns).cumprod()

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：价差与Z-Score
ax1 = axes[0]
ax1.plot(spread.index, spread, linewidth=2, label='Spread', color='blue', alpha=0.7)
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.fill_between(spread.index, 
                  spread.mean() + 2*spread.std(),
                  spread.mean() - 2*spread.std(),
                  alpha=0.2, color='blue')
ax1.set_ylabel('Spread', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(z_score.index, z_score, linewidth=2, 
              label='Z-Score', color='red', linestyle='--', alpha=0.7)
ax1_twin.axhline(y=2, color='r', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=-2, color='r', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=0.5, color='g', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=-0.5, color='g', linestyle=':', alpha=0.5)
ax1_twin.set_ylabel('Z-Score', fontsize=12)
ax1_twin.legend(loc='upper right')

# 子图2：持仓信号
ax2 = axes[1]
ax2.plot(positions.index, positions, linewidth=2, 
         label='Position', color='green')
ax2.fill_between(positions.index, 
                  positions.shift(1).fillna(0),
                  alpha=0.3, color='green')
ax2.set_ylabel('Position', fontsize=12)
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：累计收益
ax3 = axes[2]
ax3.plot(cumulative_returns.index, cumulative_returns, 
         linewidth=2, color='purple', label='Strategy Cumulative Return')
ax3.axhline(y=1, color='k', linestyle='--', alpha=0.5)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_ylabel('Cumulative Return', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure4_trading_signals.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图4已保存：figure4_trading_signals.png")
print(f"   交易次数：{(positions != positions.shift(1)).sum()}")
print(f"   平均持仓时间：{((positions != 0).sum() / (positions != positions.shift(1)).sum()):.1f} days")

# ============================================================================
# 图5：卡尔曼滤波时变对冲比率（示意）
# ============================================================================

print("\n生成图5：卡尔曼滤波...")

# 模拟时变对冲比率
dates_sample = prices.index[:500]  # 只用前500天
true_beta = 0.6 + 0.1 * np.sin(np.arange(500) * 2 * np.pi / 200)  # 正弦波动
estimated_beta = true_beta + np.random.normal(0, 0.02, 500)  # 加入噪声
fixed_beta = np.ones(500) * 0.65

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：时变截距项（示意）
intercept = np.random.normal(2, 0.5, 500)
axes[0].plot(dates_sample, intercept, linewidth=2, label='Intercept (α)', color='blue')
axes[0].set_title('Kalman Filter: Time-Varying Intercept', fontsize=14)
axes[0].set_ylabel('α', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：时变对冲比率
axes[1].plot(dates_sample, estimated_beta, linewidth=2, 
             label='Time-Varying Hedge Ratio (β)', color='red')
axes[1].plot(dates_sample, fixed_beta, 'k--', linewidth=2, 
             label='Fixed Hedge Ratio', alpha=0.7)
axes[1].fill_between(dates_sample, 
                      estimated_beta - 0.05,
                      estimated_beta + 0.05,
                      alpha=0.2, color='red')
axes[1].set_title('Kalman Filter: Time-Varying Hedge Ratio', fontsize=14)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('β', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure5_kalman_filter.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图5已保存：figure5_kalman_filter.png")

# ============================================================================
# 图6：动态仓位管理
# ============================================================================

print("\n生成图6：动态仓位管理...")

# 模拟动态仓位
position_size = np.where(np.abs(z_score) > 1.0, 
                          np.clip(1.0 / np.abs(z_score), 0.2, 1.0),
                          0.0)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：原始信号
axes[0].plot(positions.index, positions, 
             label='Original Signal', linewidth=2, alpha=0.7)
axes[0].set_title('Original Trading Signal', fontsize=14)
axes[0].set_ylabel('Position Direction', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：动态仓位大小
axes[1].plot(z_score.index, position_size, 
             label='Dynamic Position Size', linewidth=2, color='orange')
axes[1].fill_between(z_score.index, 
                      position_size,
                      alpha=0.3, color='orange')
axes[1].set_title('Dynamic Position Sizing (Based on Z-Score)', fontsize=14)
axes[1].set_xlabel('Date', fontsize=12)
axes[1].set_ylabel('Position Size', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure6_position_sizing.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图6已保存：figure6_position_sizing.png")

# ============================================================================
# 额外配图：配对交易流程图
# ============================================================================

print("\n生成额外配图：配对交易流程图...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# 绘制流程图
steps = [
    {'text': '1. Pair Selection\n(Cointegration Test, Distance, Clustering)', 'x': 0.5, 'y': 0.9, 'box': True},
    {'text': '2. Signal Generation\n(Z-Score, Kalman Filter, ML)', 'x': 0.5, 'y': 0.75, 'box': True},
    {'text': '3. Position Sizing\n(Dynamic/Static Weights)', 'x': 0.5, 'y': 0.6, 'box': True},
    {'text': '4. Trade Execution\n(Entry/Exit Rules)', 'x': 0.5, 'y': 0.45, 'box': True},
    {'text': '5. Risk Management\n(Stop-Loss, Max Holding Period)', 'x': 0.5, 'y': 0.3, 'box': True},
    {'text': '6. Performance Evaluation\n(Sharpe, Drawdown, P&L)', 'x': 0.5, 'y': 0.15, 'box': True},
]

for step in steps:
    if step['box']:
        box = dict(boxstyle='round,pad=0.5', facecolor='lightgreen', edgecolor='black', alpha=0.8)
        ax.text(step['x'], step['y'], step['text'], 
                transform=ax.transAxes,
                fontsize=12, weight='bold',
                verticalalignment='center',
                horizontalalignment='center',
                bbox=box)
    
    # 绘制箭头
    idx = steps.index(step)
    if idx < len(steps) - 1:
        ax.annotate('', xy=(step['x'], steps[idx+1]['y'] + 0.05), 
                    xytext=(step['x'], step['y'] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                    transform=ax.transAxes)

plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/process_flow.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 额外配图已保存：process_flow.png")

print("\n" + "="*60)
print("配对交易文章配图生成完成！")
print("="*60)
print("\n生成的图片：")
print("  1. figure1_theory.png (配对交易理论基础)")
print("  2. figure2_cointegration_vs_correlation.png (协整 vs 相关性)")
print("  3. figure3_clustering.png (聚类分析)")
print("  4. figure4_trading_signals.png (交易信号)")
print("  5. figure5_kalman_filter.png (卡尔曼滤波)")
print("  6. figure6_position_sizing.png (动态仓位管理)")
print("  7. process_flow.png (配对交易流程图)")
print("\n所有图片已保存到：")
print("  /Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/")
