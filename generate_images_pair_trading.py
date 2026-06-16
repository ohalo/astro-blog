"""
生成配对交易与协整分析相关配图
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

# 模拟数据
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='D')
n_days = len(dates)

# 模拟两只协整股票价格
# 股票A：基准价格
stock_a = 100 * np.exp(np.cumsum(np.random.normal(0.0005, 0.015, n_days)))

# 股票B：与A协整，但加入噪声
beta = 0.8
stock_b = beta * stock_a + 20 + np.random.normal(0, 5, n_days)
stock_b = stock_b * 1.5  # 调整价格水平

# 图1: 协整关系示意图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：原始价格序列
ax1 = axes[0, 0]
ax1.plot(dates, stock_a, label='Stock A', color='#3498DB')
ax1.plot(dates, stock_b, label='Stock B', color='#E74C3C', linestyle='--')
ax1.set_title('Price Series', fontsize=14, fontweight='bold')
ax1.set_ylabel('Price ($)', fontsize=12)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 子图2：协整回归
ax2 = axes[0, 1]
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(stock_b, stock_a)
fitted_line = intercept + slope * stock_b
ax2.scatter(stock_b, stock_a, alpha=0.5, s=10, color='#3498DB')
ax2.plot(stock_b, fitted_line, 'r-', label=f'Fitted: y={slope:.3f}x+{intercept:.1f}')
ax2.set_xlabel('Stock B Price', fontsize=12)
ax2.set_ylabel('Stock A Price', fontsize=12)
ax2.set_title(f'Cointegration Regression (R²={r_value**2:.3f})', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：残差（价差）序列
ax3 = axes[1, 0]
residuals = stock_a - (intercept + slope * stock_b)
ax3.plot(dates, residuals, color='#9B59B6')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.axhline(y=np.mean(residuals) + 2*np.std(residuals), color='red', linestyle='--', label='±2σ')
ax3.axhline(y=np.mean(residuals) - 2*np.std(residuals), color='red', linestyle='--', linewidth=1.5)
ax3.fill_between(dates, np.mean(residuals) - 2*np.std(residuals), 
                 np.mean(residuals) + 2*np.std(residuals), alpha=0.2, color='red')
ax3.set_title('Spread (Residuals)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Spread', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：残差分布
ax4 = axes[1, 1]
ax4.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='#9B59B6', density=True)

# 叠加正态分布
from scipy.stats import norm
x = np.linspace(residuals.min(), residuals.max(), 100)
ax4.plot(x, norm.pdf(x, residuals.mean(), residuals.std()), 'r-', label=f'Normal(μ={residuals.mean():.2f}, σ={residuals.std():.2f})')

ax4.set_title('Spread Distribution', fontsize=14, fontweight='bold')
ax4.set_xlabel('Spread', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.suptitle('Cointegration Analysis: Stock A vs Stock B', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图2: Z-Score交易信号示意图
fig, ax = plt.subplots(figsize=(14, 7))

# 计算Z-Score
z_score = (residuals - residuals.mean()) / residuals.std()

# 绘制Z-Score
ax.plot(dates, z_score, color='#2C3E50', label='Z-Score')

# 添加阈值线
ax.axhline(y=2, color='#E74C3C', linestyle='--', label='Entry (+2σ)')
ax.axhline(y=-2, color='#E74C3C', linestyle='--', label='Entry (-2σ)')
ax.axhline(y=0.5, color='#27AE60', linestyle=':', label='Exit (+0.5σ)')
ax.axhline(y=-0.5, color='#27AE60', linestyle=':', label='Exit (-0.5σ)')
ax.axhline(y=0, color='gray', linestyle='-', linewidth=1)

# 标注交易区域
ax.fill_between(dates, 2, z_score, where=(z_score > 2), alpha=0.3, color='red', label='Short A, Long B')
ax.fill_between(dates, -2, z_score, where=(z_score < -2), alpha=0.3, color='green', label='Long A, Short B')

ax.set_title('Pair Trading Signals: Z-Score of Spread', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Z-Score', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/trading_signals.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图3: 累积收益对比图
fig, ax = plt.subplots(figsize=(14, 7))

# 模拟策略收益
# 简单假设：当|Z-Score|>2时持有，<0.5时平仓
position = np.zeros(n_days)
for i in range(1, n_days):
    if z_score[i-1] > 2:  # 做空A，做多B
        position[i] = -1
    elif z_score[i-1] < -2:  # 做多A，做空B
        position[i] = 1
    elif abs(z_score[i-1]) < 0.5:  # 平仓
        position[i] = 0
    else:
        position[i] = position[i-1]  # 保持仓位

# 计算策略收益（简化版）
strategy_returns = position * (np.diff(stock_a) - slope * np.diff(stock_b)) / stock_a[:-1]
strategy_cumulative = np.cumprod(1 + strategy_returns)

# 买入持有收益
bh_returns = stock_a / stock_a[0]

# 绘制
ax.plot(dates[1:], strategy_cumulative, color='#3498DB', 
         label='Pairs Trading Strategy')
ax.plot(dates, bh_returns, color='#E74C3C', linestyle='--', 
         label='Buy & Hold (Stock A)', linewidth=2.5)

ax.set_title('Cumulative Returns: Pairs Trading vs Buy & Hold', fontsize=16, fontweight='bold', pad=20)
ax.set_ylabel('Cumulative Returns', fontsize=13, fontweight='bold')
ax.set_xlabel('Date', fontsize=13, fontweight='bold')
ax.legend(fontsize=12, loc='upper left')
ax.grid(True, alpha=0.3)
ax.yaxis.set_ticks_position('both')

# 添加绩效标注
total_return = (strategy_cumulative[-1] - 1) * 100
ax.text(0.02, 0.98, f'Strategy Total Return: {total_return:.1f}%', 
         transform=ax.transAxes, fontsize=11,
         verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cumulative_returns.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图4: 配对交易流程图
fig, ax = plt.subplots(figsize=(14, 10))

# 定义流程步骤
steps = [
    ('Step 1\nStock Selection', 0.5, 0.95),
    ('Step 2\nCointegration Test', 0.5, 0.78),
    ('Step 3\nCalculate Spread', 0.5, 0.61),
    ('Step 4\nSet Thresholds', 0.5, 0.44),
    ('Step 5\nGenerate Signals', 0.5, 0.27),
    ('Step 6\nExecute Trades', 0.5, 0.10)
]

# 绘制方框和箭头
box_width = 0.6
box_height = 0.08
for i, (text, x, y) in enumerate(steps):
    # 绘制方框
    box = plt.Rectangle((x - box_width/2, y - box_height/2), 
                         box_width, box_height,
                         edgecolor='#2C3E50', facecolor='#ECF0F1', linewidth=2)
    ax.add_patch(box)
    
    # 添加文字
    ax.text(x, y, text, ha='center', va='center', fontsize=13, 
             fontweight='bold', color='#2C3E50')
    
    # 绘制箭头（除了最后一步）
    if i < len(steps) - 1:
        arrow_start_y = y - box_height/2
        arrow_end_y = steps[i+1][2] + box_height/2
        ax.annotate('', xy=(x, arrow_end_y), xytext=(x, arrow_start_y),
                     arrowprops=dict(arrowstyle='->', lw=2.5, color='#3498DB'))

# 添加侧边说明
notes = [
    ('• Fundamental analysis\n• Industry classification\n• Market cap similarity', 0.85, 0.95),
    ('• ADF test on residuals\n• Johansen test (multivariate)\n• p-value < 0.05', 0.85, 0.78),
    ('• OLS: Stock_A = α + β·Stock_B\n• Spread = Residuals\n• Calculate Z-Score', 0.85, 0.61),
    ('• Entry: Z = ±2\n• Exit: Z = ±0.5\n• Stop-loss: Z = ±3', 0.85, 0.44),
    ('• Long Spread when Z < -2\n• Short Spread when Z > 2\n• Close when |Z| < 0.5', 0.85, 0.27),
    ('• Execute both legs\n• Account for transaction costs\n• Risk management', 0.85, 0.10)
]

for note, x, y in notes:
    ax.text(x, y, note, ha='left', va='center', fontsize=9.5,
             style='italic', 
             bbox=dict(boxstyle='round', facecolor='#FEF9E7', alpha=0.8))

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
ax.set_title('Pairs Trading Workflow', fontsize=18, fontweight='bold', pad=30)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/workflow.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 已生成4张配图:")
print("  1. cointegration_analysis.png - 协整分析示意图")
print("  2. trading_signals.png - Z-Score交易信号")
print("  3. cumulative_returns.png - 累积收益对比")
print("  4. workflow.png - 配对交易流程图")

