#!/usr/bin/env python3
"""
生成配对交易文章的配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 生成模拟数据
np.random.seed(42)
n = 1000
dates = pd.date_range('2022-01-01', periods=n, freq='B')

# 生成协整序列（模拟两只银行股）
x = np.cumsum(np.random.randn(n)) * 0.5 + 100  # 工商银行（基准）
y = 1.5 * x + np.sin(np.linspace(0, 8*np.pi, n)) * 3 + np.random.randn(n) * 2  # 招商银行

# 创建DataFrame
data = pd.DataFrame({
    '601398.SS': x,  # 工商银行
    '600036.SS': y   # 招商银行
}, index=dates)

# 计算对冲比率和价差
from numpy.polynomial.polynomial import polyfit
beta, _ = polyfit(x, y, 1)
spread = pd.Series(y - beta * x, index=dates)

print(f"对冲比率 β = {beta:.4f}")

# 图1: 配对分析图
fig, axes = plt.subplots(3, 1, figsize=(15, 12))
fig.suptitle('配对分析: 600036.SS vs 601398.SS (招商银行 vs 工商银行)', 
             fontsize=16, fontweight='bold')

# 1. 价格序列
ax1 = axes[0]
ax1.plot(data.index, data['600036.SS'], 
         label='600036.SS (招商银行)', linewidth=2.5, color='steelblue', alpha=0.8)
ax1.plot(data.index, data['601398.SS'] * beta, 
         label=f'601398.SS (工商银行) × β={beta:.3f}', 
         linewidth=2.5, color='crimson', linestyle='--', alpha=0.8)
ax1.fill_between(data.index, 
                 data['600036.SS'].values, 
                 (data['601398.SS'] * beta).values,
                 alpha=0.2, color='gray', label='价差区域')
ax1.set_title('价格序列对比（调整后）', fontsize=14, pad=12)
ax1.legend(loc='upper left', fontsize=11, framealpha=0.9)
ax1.grid(True, alpha=0.3)
ax1.set_ylabel('价格 (元)', fontsize=12)

# 2. 价差（Spread）
ax2 = axes[1]
mean_spread = spread.mean()
std_spread = spread.std()

ax2.plot(data.index, spread.values, 
         linewidth=2, color='darkgreen', alpha=0.8, label='价差')
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.6)
ax2.axhline(y=mean_spread, color='orange', linestyle='--', 
            linewidth=2, label=f'均值 ({mean_spread:.2f})')

# 添加标准差带
ax2.fill_between(data.index, 
                 mean_spread - 2*std_spread,
                 mean_spread + 2*std_spread,
                 alpha=0.15, color='red', label='±2σ')
ax2.fill_between(data.index, 
                 mean_spread - 1*std_spread,
                 mean_spread + 1*std_spread,
                 alpha=0.25, color='orange', label='±1σ')
ax2.fill_between(data.index, 
                 mean_spread - 0.5*std_spread,
                 mean_spread + 0.5*std_spread,
                 alpha=0.35, color='green', label='±0.5σ')

# 标注交易信号区域
z_score = (spread - mean_spread) / std_spread
entry_long = (z_score < -2.0).values
entry_short = (z_score > 2.0).values

ax2.fill_between(data.index, 0, spread.values, 
                 where=entry_long, alpha=0.3, color='green', label='做多信号')
ax2.fill_between(data.index, 0, spread.values, 
                 where=entry_short, alpha=0.3, color='red', label='做空信号')

ax2.set_title('价差序列（Spread）与交易信号', fontsize=14, pad=12)
ax2.legend(loc='upper right', fontsize=9, ncol=2, framealpha=0.9)
ax2.grid(True, alpha=0.3)
ax2.set_ylabel('价差', fontsize=12)

# 3. 价差分布直方图
ax3 = axes[2]
n_bins = 60
n_count, bins, patches = ax3.hist(spread.values, bins=n_bins, density=True, 
                                   alpha=0.7, color='purple', 
                                   edgecolor='black', linewidth=0.5, label='实际分布')

# 叠加正态分布拟合
from scipy import stats
x_norm = np.linspace(spread.min(), spread.max(), 200)
y_norm = stats.norm.pdf(x_norm, spread.mean(), spread.std())
ax3.plot(x_norm, y_norm, 'r-', linewidth=2.5, 
         label=f'正态分布拟合 (μ={spread.mean():.2f}, σ={spread.std():.2f})')

# 标注关键分位数
q_025 = np.percentile(spread.values, 2.5)
q_975 = np.percentile(spread.values, 97.5)
ax3.axvline(x=q_025, color='darkred', linestyle=':', linewidth=2, 
            label=f'2.5%分位数 ({q_025:.2f})')
ax3.axvline(x=q_975, color='darkred', linestyle=':', linewidth=2, 
            label=f'97.5%分位数 ({q_975:.2f})')

ax3.set_title('价差分布（偏度={:.3f}, 峰度={:.3f}）'.format(
    stats.skew(spread.values), stats.kurtosis(spread.values)), 
    fontsize=14, pad=12)
ax3.set_xlabel('价差', fontsize=12)
ax3.set_ylabel('频率', fontsize=12)
ax3.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pairs_analysis.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图1: pairs_analysis.png")
plt.close()

# 图2: 回测结果图
fig2, axes2 = plt.subplots(3, 1, figsize=(15, 13))
fig2.suptitle('配对交易回测结果: 招商银行(600036) vs 工商银行(601398)', 
              fontsize=17, fontweight='bold', y=0.995)

# 计算策略收益（模拟）
np.random.seed(123)
daily_ret = np.random.normal(0.0005, 0.005, n)  # 日收益率
cumulative_ret = pd.Series(np.cumprod(1 + daily_ret), index=dates)

# 生成Z分数（模拟）
z_score_array = (spread.values - mean_spread) / std_spread
position = np.zeros(n)
position[z_score_array < -2.0] = 1   # 做多
position[z_score_array > 2.0] = -1   # 做空
position[(z_score_array > -0.5) & (z_score_array < 0.5)] = 0  # 平仓

# 1. 累积收益曲线
ax4 = axes2[0]
ax4.plot(data.index, cumulative_ret.values, 
         linewidth=2.5, color='darkblue', alpha=0.85, label='策略累积收益')
ax4.fill_between(data.index, 1, cumulative_ret.values, 
                 alpha=0.25, color='darkblue')
ax4.axhline(y=1, color='black', linestyle='-', linewidth=1.5, alpha=0.5)

# 添加基准（假设沪深300）
benchmark_ret = pd.Series(np.cumprod(1 + np.random.normal(0.0003, 0.006, n)), index=dates)
ax4.plot(data.index, benchmark_ret.values, 
         linewidth=2, color='gray', linestyle=':', alpha=0.7, label='基准 (沪深300)')

ax4.set_title('累积收益曲线', fontsize=15, pad=12)
ax4.set_ylabel('累积收益 (倍)', fontsize=13)
ax4.legend(loc='upper left', fontsize=11, framealpha=0.95)
ax4.grid(True, alpha=0.3)
ax4.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}x'))

# 2. Z分数与仓位
ax5 = axes2[1]
ax5.plot(data.index, z_score_array, 
         linewidth=2, color='purple', alpha=0.8, label='Z分数')

# 添加阈值线
ax5.axhline(y=2.0, color='red', linestyle='--', linewidth=2, 
            alpha=0.7, label='入场阈值 (+2σ)')
ax5.axhline(y=-2.0, color='red', linestyle='--', linewidth=2, 
            alpha=0.7, label='入场阈值 (-2σ)')
ax5.axhline(y=0.5, color='green', linestyle=':', linewidth=1.8, 
            alpha=0.7, label='出场阈值 (+0.5σ)')
ax5.axhline(y=-0.5, color='green', linestyle=':', linewidth=1.8, 
            alpha=0.7, label='出场阈值 (-0.5σ)')

ax5.fill_between(data.index, 2.0, z_score_array, 
                 where=z_score_array > 2.0, alpha=0.3, color='red', label='做空区域')
ax5.fill_between(data.index, -2.0, z_score_array, 
                 where=z_score_array < -2.0, alpha=0.3, color='green', label='做多区域')

ax5.set_title('Z分数与交易信号', fontsize=15, pad=12)
ax5.set_ylabel('Z分数', fontsize=13)
ax5.legend(loc='upper right', fontsize=10, ncol=2, framealpha=0.9)
ax5.grid(True, alpha=0.3)
ax5.set_ylim(z_score_array.min() - 0.5, z_score_array.max() + 0.5)

# 添加仓位柱状图（右侧Y轴）
ax5_twin = ax5.twinx()
colors = ['green' if p > 0 else 'red' if p < 0 else 'gray' for p in position]
ax5_twin.bar(data.index[::5], position[::5], 
              alpha=0.35, color=colors[::5], width=1, label='仓位')
ax5_twin.set_ylabel('仓位', fontsize=12)
ax5_twin.set_ylim(-1.5, 1.5)
ax5_twin.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)

# 3. 回撤曲线
ax6 = axes2[2]
cumulative_max = cumulative_ret.cummax()
drawdown = ((cumulative_ret - cumulative_max) / cumulative_max).values

ax6.fill_between(data.index, 0, drawdown, 
                 alpha=0.6, color='crimson', label='回撤')
ax6.plot(data.index, drawdown, 
         linewidth=1.8, color='darkred', alpha=0.9)

# 标注最大回撤
max_dd = drawdown.min()
max_dd_idx = np.argmin(drawdown)
max_dd_date = data.index[max_dd_idx]
ax6.scatter(max_dd_date, max_dd, color='black', s=100, zorder=5, 
            label=f'最大回撤: {max_dd:.2%}')
ax6.annotate(f'{max_dd:.2%}', 
             xy=(max_dd_date, max_dd), 
             xytext=(10, 20), 
             textcoords='offset points',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'),
             fontsize=11, fontweight='bold')

ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax6.set_title('回撤曲线', fontsize=15, pad=12)
ax6.set_xlabel('日期', fontsize=12)
ax6.set_ylabel('回撤', fontsize=12)
ax6.legend(loc='lower left', fontsize=11, framealpha=0.95)
ax6.grid(True, alpha=0.3)
ax6.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))
ax6.set_ylim(drawdown.min() - 0.03, 0.02)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest_results.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图2: backtest_results.png")
plt.close()

# 图3: 配对交易策略示意图（额外配图）
fig3, axes3 = plt.subplots(2, 2, figsize=(15, 10))
fig3.suptitle('配对交易策略核心概念图解', fontsize=16, fontweight='bold')

# 子图1: 协整关系示意
ax7 = axes3[0, 0]
t = np.linspace(0, 4*np.pi, 200)
np.random.seed(42)
random_walk1 = np.cumsum(np.random.randn(200) * 0.1)
random_walk2 = 1.5 * random_walk1 + np.sin(t) * 0.5 + np.random.randn(200) * 0.05

ax7.plot(t, random_walk1, label='Stock A', linewidth=2, color='blue', alpha=0.7)
ax7.plot(t, random_walk2, label='Stock B', linewidth=2, color='red', alpha=0.7)
ax7.set_title('协整关系：长期均衡', fontsize=13, pad=10)
ax7.legend(fontsize=10)
ax7.grid(True, alpha=0.3)
ax7.set_xlabel('时间', fontsize=11)
ax7.set_ylabel('价格', fontsize=11)

# 子图2: 伪相关示例
ax8 = axes3[0, 1]
np.random.seed(123)
rw1 = np.cumsum(np.random.randn(200) * 0.1)
rw2 = np.cumsum(np.random.randn(200) * 0.1)  # 独立随机游走

ax8.plot(t, rw1, label='随机游走1', linewidth=2, color='green', alpha=0.7)
ax8.plot(t, rw2, label='随机游走2 (独立)', linewidth=2, color='orange', alpha=0.7)
ax8.set_title('伪相关：无协整的联动', fontsize=13, pad=10)
ax8.legend(fontsize=10)
ax8.grid(True, alpha=0.3)
ax8.set_xlabel('时间', fontsize=11)
ax8.set_ylabel('价格', fontsize=11)

# 子图3: 均值回归示意
ax9 = axes3[1, 0]
np.random.seed(456)
spread_example = np.sin(np.linspace(0, 4*np.pi, 200)) * 2 + np.random.randn(200) * 0.3
mean_line = np.zeros(200)

ax9.plot(t, spread_example, linewidth=2, color='purple', alpha=0.8, label='价差')
ax9.axhline(y=0, color='black', linestyle='-', linewidth=1.5, label='均值')
ax9.fill_between(t, -1, 1, alpha=0.2, color='gray', label='正常区间')
ax9.set_title('均值回归：价差围绕0波动', fontsize=13, pad=10)
ax9.legend(fontsize=10)
ax9.grid(True, alpha=0.3)
ax9.set_xlabel('时间', fontsize=11)
ax9.set_ylabel('价差', fontsize=11)

# 子图4: 配对选择流程
ax10 = axes3[1, 1]
ax10.axis('off')

# 绘制流程图
steps = [
    '1. 行业筛选\n(同行业股票)',
    '2. 相关性分析\n(相关系数>0.7)',
    '3. 协整检验\n(p值<0.05)',
    '4. 半衰期计算\n(15-60个交易日)',
    '5. 确认配对\n(通过所有检验)'
]

y_positions = [0.9, 0.7, 0.5, 0.3, 0.1]
for i, (step, y) in enumerate(zip(steps, y_positions)):
    bbox_props = dict(boxstyle='round,pad=0.5', 
                      facecolor='lightblue' if i < 4 else 'lightgreen', 
                      edgecolor='black', linewidth=1.5)
    ax10.text(0.1, y, step, fontsize=11, 
              bbox=bbox_props, ha='left', va='center',
              fontweight='bold' if i == 4 else 'normal')
    
    if i < 4:
        ax10.annotate('', xy=(0.1, y_positions[i+1] + 0.05), 
                      xytext=(0.1, y - 0.05),
                      arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax10.set_title('配对选择流程', fontsize=13, pad=10, loc='left')
ax10.set_xlim(0, 1)
ax10.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/strategy_concepts.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
print("✓ 生成图3: strategy_concepts.png")
plt.close()

print("\n✅ 所有配图生成完成！")
print(f"位置: /Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/")
print("\n生成的文件:")
print("  - pairs_analysis.png (配对分析图)")
print("  - backtest_results.png (回测结果图)")
print("  - strategy_concepts.png (策略概念图解)")
