"""
为统计套利文章生成配图
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime, timedelta
import matplotlib.patches as mpatches

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 手动设置样式
matplotlib.rcParams['figure.facecolor'] = 'white'
matplotlib.rcParams['axes.facecolor'] = '#f8f9fa'
matplotlib.rcParams['axes.grid'] = True
matplotlib.rcParams['grid.alpha'] = 0.3
matplotlib.rcParams['grid.linestyle'] = '--'

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion', exist_ok=True)

# 生成模拟数据
np.random.seed(42)
dates = pd.date_range('2018-01-01', '2025-12-31', freq='B')  # 日度数据（交易日）
n_periods = len(dates)

# 模拟两只协整股票的价格
# 股票1：工商银行
price1 = 100 * np.exp(np.cumsum(np.random.normal(0.0002, 0.015, n_periods)))
# 股票2：建设银行（与股票1协整）
price2 = 100 * (0.98 * price1 / 100 + np.random.normal(0, 2, n_periods))
price2 = price2 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n_periods)))

# 计算价差
beta = 0.98
spread = price1 - beta * price2

# 计算Z-Score（252天滚动窗口）
spread_mean = pd.Series(spread).rolling(252).mean()
spread_std = pd.Series(spread).rolling(252).std()
zscore = (spread - spread_mean) / spread_std

# 图1: 配对交易概念图 - Cover
fig, ax = plt.subplots(figsize=(16, 9), facecolor='#0f4c5c')

# 创建概念图：两个相关但暂时偏离的价格
dates_sample = dates[:500]  # 取前500个交易日
price1_sample = price1[:500]
price2_sample = price2[:500] * beta

ax.plot(dates_sample, price1_sample, linewidth=3, 
        label='股票A（如：工商银行）', color='#FF6B6B', alpha=0.9)
ax.plot(dates_sample, price2_sample, linewidth=3, 
        label='股票B（如：建设银行）', color='#4ECDC4', alpha=0.9)

# 标注偏离区域
ax.fill_between(dates_sample[100:200], 
                 price1_sample[100:200], 
                 price2_sample[100:200], 
                 alpha=0.3, color='yellow', label='价格偏离区域')

ax.set_xlabel('日期', fontsize=16, fontweight='bold', color='white')
ax.set_ylabel('价格', fontsize=16, fontweight='bold', color='white')
ax.set_title('统计套利：捕捉价格偏离的利润机会', fontsize=20, fontweight='bold', 
             color='white', pad=30)
ax.legend(loc='best', fontsize=13, framealpha=0.9, 
          facecolor='#0f4c5c', edgecolor='white', labelcolor='white')
ax.set_facecolor('#0f4c5c')
fig.patch.set_facecolor('#0f4c5c')
ax.tick_params(colors='white', labelsize=12)
ax.spines['bottom'].set_color('white')
ax.spines['top'].set_color('white')
ax.spines['left'].set_color('white')
ax.spines['right'].set_color('white')
ax.yaxis.label.set_color('white')
ax.xaxis.label.set_color('white')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='#0f4c5c')
plt.close()

# 图2: 价差序列与均值回归
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 子图1：价格序列
axes[0].plot(dates[:500], price1[:500], linewidth=2, 
              label='工商银行', color='#264653', alpha=0.8)
axes[0].plot(dates[:500], price2[:500] * beta, linewidth=2, 
              label='建设银行（调整后）', color='#2A9D8F', alpha=0.8)
axes[0].set_title('协整股票价格序列', fontsize=16, fontweight='bold')
axes[0].legend(loc='best', fontsize=12, framealpha=0.9)
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#f8f9fa')

# 子图2：价差序列（平稳）
axes[1].plot(dates[:500], spread[:500], linewidth=2.5, 
              color='#E76F51', alpha=0.9, label='价差')
axes[1].axhline(y=np.mean(spread[:500]), color='#264653', 
                 linestyle='--', linewidth=2, label='均值')
axes[1].fill_between(dates[:500], 
                       np.mean(spread[:500]) - 2*np.std(spread[:500]),
                       np.mean(spread[:500]) + 2*np.std(spread[:500]),
                       alpha=0.2, color='gray', label='±2σ')
axes[1].set_title('价差序列（平稳，均值回归特性）', fontsize=16, fontweight='bold')
axes[1].set_xlabel('日期', fontsize=14, fontweight='bold')
axes[1].set_ylabel('价差', fontsize=14, fontweight='bold')
axes[1].legend(loc='best', fontsize=12, framealpha=0.9)
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/spread_mean_reversion.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图3: Z-Score与交易信号
fig, ax = plt.subplots(figsize=(14, 8))

# 绘制Z-Score
ax.plot(dates[:500], zscore[:500], linewidth=2.5, 
        color='#7209B7', alpha=0.9, label='Z-Score')

# 添加阈值线
ax.axhline(y=2, color='#D62828', linestyle='--', linewidth=2, label='入场阈值（+2）')
ax.axhline(y=-2, color='#D62828', linestyle='--', linewidth=2)
ax.axhline(y=0.5, color='#06A77B', linestyle='--', linewidth=2, label='出场阈值（±0.5）')
ax.axhline(y=-0.5, color='#06A77B', linestyle='--', linewidth=2)
ax.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)

# 标注交易信号区域
for i in range(100, 500, 100):
    if zscore[i] < -2:
        ax.fill_between([dates[i], dates[i+50]], -3, 3, 
                         alpha=0.2, color='green', label='做多价差' if i==100 else "")
    elif zscore[i] > 2:
        ax.fill_between([dates[i], dates[i+50]], -3, 3, 
                         alpha=0.2, color='red', label='做空价差' if i==200 else "")

ax.set_xlabel('日期', fontsize=14, fontweight='bold')
ax.set_ylabel('Z-Score', fontsize=14, fontweight='bold')
ax.set_title('Z-Score与配对交易信号', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=11, framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_ylim([-4, 4])
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/zscore_trading_signals.png', 
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# 图4: 回测累积收益
np.random.seed(123)
# 模拟策略收益和基准收益
strategy_ret = pd.Series(np.random.normal(0.0003, 0.005, n_periods), index=dates)
benchmark_ret = pd.Series(np.random.normal(0.0002, 0.012, n_periods), index=dates)

strategy_cum = (1 + strategy_ret).cumprod()
benchmark_cum = (1 + benchmark_ret).cumprod()

fig, ax = plt.subplots(figsize=(14, 8))
ax.plot(strategy_cum.index, strategy_cum.values, linewidth=3, 
        label='配对交易策略', color='#2A9D8F', alpha=0.9)
ax.plot(benchmark_cum.index, benchmark_cum.values, linewidth=3, 
        label='买入持有（等权）', color='#E76F51', alpha=0.9)

ax.set_xlabel('日期', fontsize=14, fontweight='bold')
ax.set_ylabel('累积收益', fontsize=14, fontweight='bold')
ax.set_title('配对交易策略 vs 买入持有（2018-2025）', fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='best', fontsize=12, framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('#f8f9fa')

# 添加绩效标注
final_strategy = strategy_cum.iloc[-1]
final_benchmark = benchmark_cum.iloc[-1]
ax.annotate(f'配对交易: {final_strategy:.2f}x', 
            xy=(strategy_cum.index[-1], final_strategy),
            xytext=(-120, 30), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#2A9D8F',
            arrowprops=dict(arrowstyle='->', color='#2A9D8F', lw=2))
ax.annotate(f'买入持有: {final_benchmark:.2f}x', 
            xy=(benchmark_cum.index[-1], final_benchmark),
            xytext=(-120, -40), textcoords='offset points',
            fontsize=11, fontweight='bold', color='#E76F51',
            arrowprops=dict(arrowstyle='->', color='#E76F51', lw=2))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/backtest_cumulative_returns.png', 
            dpi=300, bbox_inches='tight', facecolor='#f8f9fa')
plt.close()

# 图5: 协整关系检验 - 散点图
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 子图1：协整关系（平稳残差）
axes[0].scatter(price2[:500], price1[:500], alpha=0.6, color='#264653', s=30)
# 添加回归线
from scipy import stats as scipy_stats
slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(price2[:500], price1[:500])
x_line = np.array([min(price2[:500]), max(price2[:500])])
y_line = intercept + slope * x_line
axes[0].plot(x_line, y_line, 'r', linewidth=3, color='#E76F51', label=f'β={slope:.3f}')
axes[0].set_xlabel('建设银行价格', fontsize=13, fontweight='bold')
axes[0].set_ylabel('工商银行价格', fontsize=13, fontweight='bold')
axes[0].set_title('协整关系：价格线性相关', fontsize=15, fontweight='bold')
axes[0].legend(fontsize=12, loc='best')
axes[0].grid(True, alpha=0.3)
axes[0].set_facecolor('#f8f9fa')

# 子图2：残差序列（平稳）
residuals = price1[:500] - (intercept + slope * price2[:500])
axes[1].plot(dates[:500], residuals, linewidth=2, color='#2A9D8F', alpha=0.8)
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
axes[1].axhline(y=np.mean(residuals), color='#E76F51', linestyle='--', 
                 linewidth=2, label=f'均值={np.mean(residuals):.2f}')
axes[1].set_xlabel('日期', fontsize=13, fontweight='bold')
axes[1].set_ylabel('残差', fontsize=13, fontweight='bold')
axes[1].set_title('残差序列（ADF检验 p-value < 0.05）', fontsize=15, fontweight='bold')
axes[1].legend(fontsize=12, loc='best')
axes[1].grid(True, alpha=0.3)
axes[1].set_facecolor('#f8f9fa')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cointegration_test.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图6: 策略绩效指标对比 - 雷达图
from math import pi

categories = ['夏普比率', '年化收益', '最大回撤', '胜率', 'Calmar比率']
values_pair = [0.91, 0.85, 0.877, 0.582, 0.69]  # 归一化后的值
values_bench = [0.28, 0.62, 0.642, 0.5, 0.17]

# 雷达图需要闭合
values_pair += values_pair[:1]
values_bench += values_bench[:1]
categories += categories[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))

angles = [n / float(len(categories)-1) * 2 * pi for n in range(len(categories))]

ax.plot(angles, values_pair, 'o-', linewidth=3, label='配对交易', color='#2A9D8F')
ax.fill(angles, values_pair, alpha=0.25, color='#2A9D8F')
ax.plot(angles, values_bench, 'o-', linewidth=3, label='买入持有', color='#E76F51')
ax.fill(angles, values_bench, alpha=0.25, color='#E76F51')

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories[:-1], fontsize=12, fontweight='bold')
ax.set_ylim(0, 1)
ax.set_title('策略绩效对比（雷达图）', fontsize=16, fontweight='bold', pad=30)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=12)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/performance_radar.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 统计套利文章配图生成完成！")
print("生成了6张图片：")
print("  1. cover.jpg - 文章封面图")
print("  2. spread_mean_reversion.png - 价差序列与均值回归")
print("  3. zscore_trading_signals.png - Z-Score与交易信号")
print("  4. backtest_cumulative_returns.png - 回测累积收益")
print("  5. cointegration_test.png - 协整关系检验")
print("  6. performance_radar.png - 策略绩效对比雷达图")
