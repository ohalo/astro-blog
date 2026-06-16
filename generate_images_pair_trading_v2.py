"""
生成配对交易与协整分析相关配图（简化版）
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

# 模拟数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
n_days = len(dates)

# 模拟两只协整股票价格
stock_a = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_days)))
beta = 0.8
stock_b = beta * stock_a + 20 + np.random.normal(0, 5, n_days)
stock_b = stock_b * 1.5

# 计算残差
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(stock_b, stock_a)
residuals = stock_a - (intercept + slope * stock_b)
z_score = (residuals - residuals.mean()) / residuals.std()

# 图1: 协整关系示意图
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

# 价格序列
ax1.plot(dates, stock_a, linewidth=2, label='Stock A', color='#3498DB')
ax1.plot(dates, stock_b, linewidth=2, label='Stock B', color='#E74C3C', linestyle='--')
ax1.set_title('Price Series', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 散点图与回归线
ax2.scatter(stock_b, stock_a, alpha=0.5, s=10, color='#3498DB')
fitted_line = intercept + slope * stock_b
ax2.plot(stock_b, fitted_line, 'r-', linewidth=2)
ax2.set_xlabel('Stock B Price')
ax2.set_ylabel('Stock A Price')
ax2.set_title(f'Cointegration (R²={r_value**2:.3f})', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)

# 残差序列
ax3.plot(dates, residuals, linewidth=1.5, color='#9B59B6')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.fill_between(dates, -2*residuals.std(), 2*residuals.std(), alpha=0.2, color='red')
ax3.set_title('Spread (Residuals)', fontsize=14, fontweight='bold')
ax3.grid(True, alpha=0.3)

# 残差分布
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#9B59B6', density=True)
x = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2)
ax4.set_title('Spread Distribution', fontsize=14, fontweight='bold')
ax4.grid(True, alpha=0.3)

plt.suptitle('Cointegration Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图2: Z-Score交易信号
fig, ax = plt.subplots(figsize=(14, 7))
ax.plot(dates, z_score, linewidth=1.5, color='#2C3E50')
ax.axhline(y=2, color='#E74C3C', linestyle='--', linewidth=2)
ax.axhline(y=-2, color='#E74C3C', linestyle='--', linewidth=2)
ax.axhline(y=0.5, color='#27AE60', linestyle=':', linewidth=2)
ax.axhline(y=-0.5, color='#27AE60', linestyle=':', linewidth=2)
ax.fill_between(dates, 2, z_score, where=(z_score > 2), alpha=0.3, color='red')
ax.fill_between(dates, -2, z_score, where=(z_score < -2), alpha=0.3, color='green')
ax.set_title('Pair Trading Signals: Z-Score', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Z-Score', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/trading_signals.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图3: 流程图（文字版）
fig, ax = plt.subplots(figsize=(14, 10))
steps = [
    '1. Stock Selection\n(Industry, Market Cap)',
    '2. Cointegration Test\n(ADF, Johansen)',
    '3. Calculate Spread\n(Z-Score)',
    '4. Set Thresholds\n(Entry/Exit)',
    '5. Generate Signals\n(Long/Short)',
    '6. Risk Management\n(Stop-loss, Position)'
]
y_positions = [0.9, 0.75, 0.6, 0.45, 0.3, 0.15]
for step, y in zip(steps, y_positions):
    box = plt.Rectangle((0.3, y-0.05), 0.4, 0.08, 
                       edgecolor='#2C3E50', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(box)
    ax.text(0.5, y, step, ha='center', va='center', fontsize=11, fontweight='bold')
    if y > 0.15:
        ax.annotate('', xy=(0.5, y-0.05), xytext=(0.5, y-0.13),
                   arrowprops=dict(arrowstyle='->', lw=2.5, color='#3498DB'))
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Pairs Trading Workflow', fontsize=18, fontweight='bold', pad=30)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/workflow.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图4: 相关性热力图
correlation_matrix = np.array([
    [1.00, 0.85, 0.15, -0.10],
    [0.85, 1.00, 0.20, 0.05],
    [0.15, 0.20, 1.00, 0.65],
    [-0.10, 0.05, 0.65, 1.00]
])
factor_names = ['Stock A', 'Stock B', 'Stock C', 'Stock D']
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(correlation_matrix, cmap='RdYlBu_r', vmin=-1, vmax=1)
ax.set_xticks(range(len(factor_names)))
ax.set_yticks(range(len(factor_names)))
ax.set_xticklabels(factor_names, fontsize=12)
ax.set_yticklabels(factor_names, fontsize=12)
ax.set_title('Stock Price Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
for i in range(len(factor_names)):
    for j in range(len(factor_names)):
        text = ax.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=11)
plt.colorbar(im, ax=ax, shrink=0.8)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/correlation_matrix.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 已生成4张配图:")
print("  1. cointegration_analysis.png - 协整分析示意图")
print("  2. trading_signals.png - Z-Score交易信号")
print("  3. workflow.png - 配对交易流程图")
print("  4. correlation_matrix.png - 相关性热力图")
