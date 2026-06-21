"""
为PCA统计套利文章生成配图（使用合成数据）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
save_dir = '/Users/halo/workspace/astro-blog/public/images/pca-statistical-arbitrage'
os.makedirs(save_dir, exist_ok=True)

# 生成合成数据（模拟8只股票的收益率）
np.random.seed(42)
n_days = 500
n_stocks = 8

# 生成共同因子（3个主成分）
factors = np.random.randn(n_days, 3) * [0.02, 0.015, 0.01]

# 生成因子暴露度
loadings = np.random.randn(n_stocks, 3) * [1.0, 0.8, 0.6]
loadings[0:2, 0] = 1.5  # 前2只股票对第1因子暴露度高
loadings[2:4, 1] = 1.2  # 中间2只股票对第2因子暴露度高
loadings[4:6, 2] = 1.0  # 后2只股票对第3因子暴露度高

# 生成收益率：共同因子 + 特质收益率
common_returns = loadings @ factors.T
idio_returns = np.random.randn(n_days, n_stocks) * 0.005
returns_np = common_returns.T + idio_returns

# 转换为DataFrame
dates = pd.date_range('2023-01-01', periods=n_days, freq='D')
stock_names = ['Stock_A', 'Stock_B', 'Stock_C', 'Stock_D', 
               'Stock_E', 'Stock_F', 'Stock_G', 'Stock_H']
returns = pd.DataFrame(returns_np, index=dates, columns=stock_names)

# 标准化数据
scaler = StandardScaler()
returns_scaled = scaler.fit_transform(returns)

# PCA分析
pca = PCA()
pca.fit(returns_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

# 图1: 碎石图和累积解释方差
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(range(1, len(explained_variance_ratio) + 1), 
             explained_variance_ratio[:8], 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Principal Component', fontsize=12)
axes[0].set_ylabel('Explained Variance Ratio', fontsize=12)
axes[0].set_title('Scree Plot', fontsize=14, fontweight='bold')
axes[0].set_xticks(range(1, 9))
axes[0].grid(True, alpha=0.3)

axes[1].plot(range(1, len(cumulative_variance_ratio) + 1), 
             cumulative_variance_ratio[:8], 'ro-', linewidth=2, markersize=8)
axes[1].axhline(y=0.8, color='g', linestyle='--', linewidth=2, label='80% Variance')
axes[1].axhline(y=0.9, color='b', linestyle='--', linewidth=2, label='90% Variance')
axes[1].set_xlabel('Number of Components', fontsize=12)
axes[1].set_ylabel('Cumulative Variance Ratio', fontsize=12)
axes[1].set_title('Cumulative Explained Variance', fontsize=14, fontweight='bold')
axes[1].set_xticks(range(1, 9))
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/pca_variance_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片1: pca_variance_analysis.png")
plt.close()

# 使用选定数量的主成分进行分解
n_components = 3
pca_selected = PCA(n_components=n_components)
factors_extracted = pca_selected.fit_transform(returns_scaled)
loadings_extracted = pca_selected.components_.T
reconstructed = pca_selected.inverse_transform(factors_extracted)
residuals = returns_scaled - reconstructed

# 图2: 收益率分解示例
ticker_idx = 0
ticker_name = returns.columns[ticker_idx]

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

axes[0].plot(returns.index[100:200], returns_scaled[100:200, ticker_idx], 
             'b-', linewidth=1.5)
axes[0].set_title(f'{ticker_name} - Standardized Returns (100 days)', 
                  fontsize=12, fontweight='bold')
axes[0].set_ylabel('Returns', fontsize=10)
axes[0].grid(True, alpha=0.3)

axes[1].plot(returns.index[100:200], reconstructed[100:200, ticker_idx], 
             'r-', linewidth=1.5)
axes[1].set_title('Common Factor Component', fontsize=12, fontweight='bold')
axes[1].set_ylabel('Returns', fontsize=10)
axes[1].grid(True, alpha=0.3)

axes[2].plot(returns.index[100:200], residuals[100:200, ticker_idx], 
             'g-', linewidth=1.5)
axes[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
axes[2].set_title('Residuals (Idiosyncratic Returns)', fontsize=12, fontweight='bold')
axes[2].set_ylabel('Returns', fontsize=10)
axes[2].set_xlabel('Date', fontsize=10)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/returns_decomposition.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片2: returns_decomposition.png")
plt.close()

# 计算策略收益（基于残差的均值回归）
def calculate_z_score(data, window=20):
    df = pd.DataFrame(data)
    mean = df.rolling(window=window).mean()
    std = df.rolling(window=window).std()
    return ((data - mean) / std).values

z_score = calculate_z_score(residuals, window=20)
signals = np.zeros_like(z_score)
signals[z_score < -2] = 1   # 做多
signals[z_score > 2] = -1    # 做空

# 简化版策略收益计算
strategy_returns_list = []
for i in range(1, len(signals)):
    ret = np.sum(signals[i-1] * returns_scaled[i])
    strategy_returns_list.append(ret)

strategy_returns = pd.Series(strategy_returns_list, index=returns.index[1:])
cumulative_returns = (1 + strategy_returns).cumprod()

# 图3: 策略表现
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 累积收益
axes[0, 0].plot(cumulative_returns.index, cumulative_returns.values, 
                 'b-', linewidth=2)
axes[0, 0].set_title('Strategy Cumulative Returns', fontsize=12, fontweight='bold')
axes[0, 0].set_ylabel('Cumulative Returns', fontsize=10)
axes[0, 0].grid(True, alpha=0.3)

# 滚动夏普比率
rolling_sharpe = strategy_returns.rolling(window=60).mean() / \
                  strategy_returns.rolling(window=60).std() * np.sqrt(252)
axes[0, 1].plot(rolling_sharpe.index, rolling_sharpe.values, 
                 'g-', linewidth=1.5)
axes[0, 1].set_title('Rolling Sharpe Ratio (60-day)', fontsize=12, fontweight='bold')
axes[0, 1].set_ylabel('Sharpe Ratio', fontsize=10)
axes[0, 1].grid(True, alpha=0.3)

# 回撤
cummax = cumulative_returns.cummax()
drawdown = (cumulative_returns - cummax) / cummax
axes[1, 0].fill_between(drawdown.index, drawdown.values, 0, 
                         alpha=0.3, color='red')
axes[1, 0].plot(drawdown.index, drawdown.values, 'r-', linewidth=1.5)
axes[1, 0].set_title('Drawdown Curve', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Drawdown', fontsize=10)
axes[1, 0].set_xlabel('Date', fontsize=10)
axes[1, 0].grid(True, alpha=0.3)

# 残差分布
flattened_residuals = residuals.flatten()
axes[1, 1].hist(flattened_residuals, bins=50, density=True, 
                  alpha=0.7, color='purple', edgecolor='black')

x = np.linspace(flattened_residuals.min(), flattened_residuals.max(), 100)
axes[1, 1].plot(x, stats.norm.pdf(x, flattened_residuals.mean(), 
                                   flattened_residuals.std()), 
                 'r-', linewidth=2, label='Normal Distribution')
axes[1, 1].set_title('Residuals Distribution', fontsize=12, fontweight='bold')
axes[1, 1].set_xlabel('Residual Value', fontsize=10)
axes[1, 1].set_ylabel('Density', fontsize=10)
axes[1, 1].legend(fontsize=10)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/strategy_performance.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片3: strategy_performance.png")
plt.close()

# 图4: 因子暴露度热图
factor_exposure = pd.DataFrame(loadings_extracted, 
                              index=returns.columns,
                              columns=[f'PC{i+1}' for i in range(n_components)])

plt.figure(figsize=(10, 6))
sns.heatmap(factor_exposure.T, cmap='RdBu_r', center=0, 
            xticklabels=True, yticklabels=True, 
            cbar_kws={'label': 'Factor Exposure'}, annot=True, fmt='.2f')
plt.title('Stock Exposure to Principal Components', fontsize=14, fontweight='bold')
plt.xlabel('Stocks', fontsize=12)
plt.ylabel('Principal Components', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig(f'{save_dir}/factor_exposure_heatmap.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片4: factor_exposure_heatmap.png")
plt.close()

# 图5: 封面图 - PCA概念示意图
fig, ax = plt.subplots(figsize=(12, 8))

# 生成2D数据点
np.random.seed(42)
n_points = 200
data_2d = np.random.randn(n_points, 2) @ np.array([[2, 1], [1, 2]]) + np.array([0, 0])

# 计算PCA
pca_2d = PCA(n_components=2)
pca_2d.fit(data_2d)
pc1 = pca_2d.components_[0]
pc2 = pca_2d.components_[1]

# 绘制数据点
ax.scatter(data_2d[:, 0], data_2d[:, 1], alpha=0.6, c='blue', s=50, label='Data Points')

# 绘制主成分方向
mean_point = data_2d.mean(axis=0)
ax.arrow(mean_point[0], mean_point[1], 
         pc1[0] * 3, pc1[1] * 3, 
         head_width=0.2, head_length=0.3, fc='red', ec='red', 
         linewidth=3, label='PC1 (Direction of Max Variance)')
ax.arrow(mean_point[0], mean_point[1], 
         pc2[0] * 2, pc2[1] * 2, 
         head_width=0.2, head_length=0.3, fc='green', ec='green', 
         linewidth=3, label='PC2 (Orthogonal to PC1)')

ax.set_xlabel('Feature 1', fontsize=14)
ax.set_ylabel('Feature 2', fontsize=14)
ax.set_title('PCA Concept: Finding Directions of Maximum Variance', 
             fontsize=16, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_aspect('equal')

plt.tight_layout()
plt.savefig(f'{save_dir}/cover.jpg', dpi=300, bbox_inches='tight')
print("✓ 生成封面图: cover.jpg")
plt.close()

print(f"\n✅ 所有配图已生成完成！")
print(f"图片保存位置: {save_dir}/")
print(f"共生成 5 张图片")
