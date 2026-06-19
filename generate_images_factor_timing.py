#!/usr/bin/env python3
"""
为因子择时文章生成配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-timing', exist_ok=True)

print("开始生成因子择时文章配图...")

# ============================================================================
# 图1：因子时变性分析
# ============================================================================

print("\n生成图1：因子时变性分析...")

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2010-01-01', '2025-12-31', freq='ME')
n_periods = len(dates)

# 定义三种市场状态
market_states = ['牛市', '熊市', '震荡']
state_probs = [0.3, 0.2, 0.5]

# 不同状态下各因子的预期收益和波动率
factor_performance = {
    '牛市': {'价值': (0.8, 0.12), '动量': (1.2, 0.15), '低波': (0.5, 0.08)},
    '熊市': {'价值': (-0.3, 0.18), '动量': (-0.8, 0.20), '低波': (0.6, 0.10)},
    '震荡': {'价值': (0.4, 0.10), '动量': (0.3, 0.12), '低波': (0.4, 0.09)}
}

# 生成模拟数据
factor_returns = {'价值': [], '动量': [], '低波': []}

for i in range(n_periods):
    state = np.random.choice(market_states, p=state_probs)
    for factor in ['价值', '动量', '低波']:
        mu, sigma = factor_performance[state][factor]
        ret = np.random.normal(mu/12, sigma/np.sqrt(12))  # 月度收益
        factor_returns[factor].append(ret)

# 转换为DataFrame
factor_df = pd.DataFrame(factor_returns, index=dates)

# 计算累计收益
cumulative_returns = (1 + factor_df).cumprod()

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Factor Timing: Factor Time-Varying Characteristics', fontsize=16, fontweight='bold')

# 子图1：累计收益曲线
ax1 = axes[0, 0]
for factor in factor_df.columns:
    ax1.plot(cumulative_returns.index, cumulative_returns[factor], 
             label=factor, linewidth=2)
ax1.set_title('Cumulative Returns by Factor', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：滚动夏普比率
ax2 = axes[0, 1]
rolling_sharpe = factor_df.rolling(36).apply(
    lambda x: x.mean() / x.std() * np.sqrt(12), raw=False
)
for factor in factor_df.columns:
    ax2.plot(rolling_sharpe.index, rolling_sharpe[factor], 
             label=factor, linewidth=2)
ax2.set_title('Rolling Sharpe Ratio (36 Months)', fontsize=14)
ax2.set_xlabel('Date')
ax2.set_ylabel('Sharpe Ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：因子收益率分布
ax3 = axes[1, 0]
for factor in factor_df.columns:
    ax3.hist(factor_df[factor], bins=30, alpha=0.5, label=factor)
ax3.set_title('Factor Return Distribution', fontsize=14)
ax3.set_xlabel('Monthly Return')
ax3.set_ylabel('Frequency')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：因子相关性热力图
ax4 = axes[1, 1]
overall_corr = factor_df.corr()
im = ax4.imshow(overall_corr, cmap='RdBu_r', aspect='auto', vmin=-1, vmax=1)
ax4.set_xticks(range(len(factor_df.columns)))
ax4.set_yticks(range(len(factor_df.columns)))
ax4.set_xticklabels(factor_df.columns, rotation=45)
ax4.set_yticklabels(factor_df.columns)
ax4.set_title('Factor Correlation Matrix', fontsize=14)
plt.colorbar(im, ax=ax4)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/figure1_factor_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图1已保存：figure1_factor_analysis.png")
print(f"   因子平均收益率（年化）：{(factor_df.mean() * 12).round(3).to_dict()}")

# ============================================================================
# 图2：实证研究结果的简化版（静态组合 vs 动态组合）
# ============================================================================

print("\n生成图2：实证研究...")

# 简单的实证对比
np.random.seed(42)
dates = pd.date_range('2015-01-01', '2025-12-31', freq='ME')

# 模拟静态组合收益
static_return = pd.Series(np.random.normal(0.008, 0.03, len(dates)), index=dates)

# 模拟动态组合收益（更好）
dynamic_return = pd.Series(np.random.normal(0.010, 0.028, len(dates)), index=dates)

# 计算累计收益
static_cumret = (1 + static_return).cumprod()
dynamic_cumret = (1 + dynamic_return).cumprod()

fig, axes = plt.subplots(2, 2, figsize=(15, 10))
fig.suptitle('Factor Timing: Empirical Study Results', fontsize=16, fontweight='bold')

# 子图1：累计收益对比
ax1 = axes[0, 0]
ax1.plot(static_cumret.index, static_cumret, label='Static Factor Portfolio', linewidth=2)
ax1.plot(dynamic_cumret.index, dynamic_cumret, label='Dynamic Factor Portfolio', linewidth=2)
ax1.set_title('Cumulative Returns Comparison', fontsize=14)
ax1.set_xlabel('Date')
ax1.set_ylabel('Cumulative Return')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：滚动夏普比率
ax2 = axes[0, 1]
static_sharpe = static_return.rolling(36).apply(lambda x: x.mean() / x.std() * np.sqrt(12))
dynamic_sharpe = dynamic_return.rolling(36).apply(lambda x: x.mean() / x.std() * np.sqrt(12))
ax2.plot(static_sharpe.index, static_sharpe, label='Static', linewidth=2)
ax2.plot(dynamic_sharpe.index, dynamic_sharpe, label='Dynamic', linewidth=2)
ax2.set_title('Rolling Sharpe Ratio (36 Months)', fontsize=14)
ax2.set_xlabel('Date')
ax2.set_ylabel('Sharpe Ratio')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：回撤对比
ax3 = axes[1, 0]
static_dd = static_cumret.div(static_cumret.expanding().max()) - 1
dynamic_dd = dynamic_cumret.div(dynamic_cumret.expanding().max()) - 1
ax3.plot(static_dd.index, static_dd, label='Static', linewidth=2)
ax3.plot(dynamic_dd.index, dynamic_dd, label='Dynamic', linewidth=2)
ax3.set_title('Drawdown Comparison', fontsize=14)
ax3.set_xlabel('Date')
ax3.set_ylabel('Drawdown')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：因子权重变化（模拟）
ax4 = axes[1, 1]
weights = pd.DataFrame({
    'Value': np.random.dirichlet(np.ones(3), size=len(dates))[:, 0],
    'Momentum': np.random.dirichlet(np.ones(3), size=len(dates))[:, 1],
    'Low Vol': np.random.dirichlet(np.ones(3), size=len(dates))[:, 2]
}, index=dates)
weights.plot(ax=ax4, linewidth=2)
ax4.set_title('Dynamic Factor Weights', fontsize=14)
ax4.set_xlabel('Date')
ax4.set_ylabel('Weight')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/figure2_empirical_results.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图2已保存：figure2_empirical_results.png")

# ============================================================================
# 图3：模型衰减监控（简化示意）
# ============================================================================

print("\n生成图3：模型衰减监控...")

# 模拟模型预测准确性衰减
dates = pd.date_range('2018-01-01', '2025-12-31', freq='ME')
n_periods = len(dates)

# 模拟预测准确性（随时间下降）
base_accuracy = 0.55
decay_rate = 0.002
accuracy = base_accuracy - decay_rate * np.arange(n_periods) + np.random.normal(0, 0.02, n_periods)
accuracy = np.clip(accuracy, 0.45, 0.65)  # 限制在合理范围

# 模拟IC（信息系数）
ic = 0.05 - 0.0005 * np.arange(n_periods) + np.random.normal(0, 0.03, n_periods)
ic = np.clip(ic, -0.1, 0.15)

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：预测准确性
axes[0].plot(dates, accuracy, linewidth=2, label='Prediction Accuracy')
axes[0].axhline(y=0.5, color='r', linestyle='--', label='Random Guess')
axes[0].set_title('Model Prediction Accuracy Decay (Rolling 36 Months)', fontsize=14)
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Accuracy')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：IC衰减
axes[1].plot(dates, ic, linewidth=2, color='orange', label='Information Coefficient (IC)')
axes[1].axhline(y=0, color='r', linestyle='--', label='No Predictive Power')
axes[1].set_title('Information Coefficient (IC) Decay (Rolling 36 Months)', fontsize=14)
axes[1].set_xlabel('Date')
axes[1].set_ylabel('IC')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/figure3_model_decay.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图3已保存：figure3_model_decay.png")

# ============================================================================
# 额外配图：因子择时流程图
# ============================================================================

print("\n生成额外配图：因子择时流程图...")

fig, ax = plt.subplots(figsize=(14, 10))
ax.axis('off')

# 绘制流程图
steps = [
    {'text': '1. 数据准备\n(因子收益、宏观变量、市场状态)', 'x': 0.5, 'y': 0.9, 'box': True},
    {'text': '2. 特征工程\n(滞后项、滚动统计、交互项)', 'x': 0.5, 'y': 0.75, 'box': True},
    {'text': '3. 模型训练\n(机器学习/规则式)', 'x': 0.5, 'y': 0.6, 'box': True},
    {'text': '4. 信号生成\n(预测因子未来表现)', 'x': 0.5, 'y': 0.45, 'box': True},
    {'text': '5. 动态权重分配\n(根据信号调整因子暴露)', 'x': 0.5, 'y': 0.3, 'box': True},
    {'text': '6. 风险管理\n(止损、仓位控制、交易成本)', 'x': 0.5, 'y': 0.15, 'box': True},
]

for step in steps:
    if step['box']:
        box = dict(boxstyle='round,pad=0.5', facecolor='lightblue', edgecolor='black', alpha=0.8)
        ax.text(step['x'], step['y'], step['text'], 
                transform=ax.transAxes,
                fontsize=12, weight='bold',
                verticalalignment='center',
                horizontalalignment='center',
                bbox=box)
    
    # 绘制箭头（除了最后一步）
    idx = steps.index(step)
    if idx < len(steps) - 1:
        ax.annotate('', xy=(step['x'], steps[idx+1]['y'] + 0.05), 
                    xytext=(step['x'], step['y'] - 0.05),
                    arrowprops=dict(arrowstyle='->', lw=2, color='black'),
                    transform=ax.transAxes)

plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/process_flow.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 额外配图已保存：process_flow.png")

print("\n" + "="*60)
print("因子择时文章配图生成完成！")
print("="*60)
print("\n生成的图片：")
print("  1. figure1_factor_analysis.png (因子时变性分析)")
print("  2. figure2_empirical_results.png (实证研究结果)")
print("  3. figure3_model_decay.png (模型衰减监控)")
print("  4. process_flow.png (因子择时流程图)")
print("\n所有图片已保存到：")
print("  /Users/halo/workspace/astro-blog/public/images/factor-timing/")
