#!/usr/bin/env python3
"""
为因子拥挤度文章生成配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-crowding', exist_ok=True)

# 模拟因子拥挤的形成过程
np.random.seed(42)
n_days = 1000
n_assets = 500

# 生成因子收益（真实阿尔法）
true_factor_return = np.random.normal(0.0005, 0.01, n_days)
dates = pd.date_range('2022-01-01', periods=n_days, freq='B')

# 模拟三个阶段：早期发现 -> 逐渐拥挤 -> 过度拥挤
stages = [10, 50, 200]  # 参与者数量
stage_returns = []
cumulative_crowding = []

for stage, n_new in enumerate(stages):
    # 新参与者入场，因子收益被摊薄
    crowding_factor = 1 / (1 + np.sum(stages[:stage+1]) * 0.1)
    
    # 生成该阶段的因子收益
    stage_return = true_factor_return * crowding_factor
    
    # 添加噪声（拥挤导致波动率上升）
    noise = np.random.normal(0, 0.002 * (stage + 1), n_days)
    stage_return += noise
    
    stage_returns.append(stage_return)
    cumulative_crowding.append(np.sum(stages[:stage+1]))

# 图1: 因子拥挤对收益的影响 - 封面图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Factor Crowding Impact Analysis', fontsize=18, fontweight='bold')

# 子图1：各阶段累积收益
ax1 = axes[0, 0]
colors = ['#2ecc71', '#f39c12', '#e74c3c']
for i, (returns, level) in enumerate(zip(stage_returns, cumulative_crowding)):
    cumulative_return = np.cumprod(1 + returns) - 1
    ax1.plot(cumulative_return, label=f'Stage {i+1} (Participants: {level})', 
             linewidth=2, color=colors[i])
ax1.set_xlabel('Trading Days', fontsize=12)
ax1.set_ylabel('Cumulative Return', fontsize=12)
ax1.set_title('Cumulative Returns by Crowding Stage', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

# 子图2：因子波动率变化
ax2 = axes[0, 1]
volatilities = [np.std(returns) * np.sqrt(252) for returns in stage_returns]
bars = ax2.bar(range(len(volatilities)), volatilities, 
               color=colors, alpha=0.7, edgecolor='black')
ax2.set_xlabel('Stage', fontsize=12)
ax2.set_ylabel('Annualized Volatility', fontsize=12)
ax2.set_title('Factor Volatility vs Crowding Level', fontsize=14)
ax2.set_xticks(range(len(volatilities)))
ax2.set_xticklabels(['Early', 'Medium', 'Overcrowded'])
ax2.grid(True, alpha=0.3, axis='y')
ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

# 在柱子上添加数值
for bar, vol in zip(bars, volatilities):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height,
             f'{vol:.2%}', ha='center', va='bottom', fontsize=11)

# 子图3：最大回撤对比
ax3 = axes[1, 0]
for i, returns in enumerate(stage_returns):
    cumulative = np.cumprod(1 + returns)
    running_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - running_max) / running_max
    ax3.plot(drawdown, label=f'Stage {i+1}', linewidth=2, color=colors[i])
ax3.set_xlabel('Trading Days', fontsize=12)
ax3.set_ylabel('Max Drawdown', fontsize=12)
ax3.set_title('Drawdown by Crowding Stage', fontsize=14)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1%}'))

# 子图4：收益分布变化
ax4 = axes[1, 1]
for i, returns in enumerate(stage_returns):
    ax4.hist(returns, bins=30, alpha=0.5, label=f'Stage {i+1}', 
             density=True, color=colors[i])
ax4.set_xlabel('Daily Return', fontsize=12)
ax4.set_ylabel('Density', fontsize=12)
ax4.set_title('Return Distribution Evolution', fontsize=14)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3, axis='y')
ax4.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.2%}'))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/cover.jpg', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 封面图已生成: cover.jpg")

# 图2: 波动率监测图
fig, ax = plt.subplots(figsize=(14, 7))

# 生成示例数据
factor_ret = pd.Series(np.random.normal(0.0005, 0.01, n_days), index=dates)

# 计算滚动波动率
rolling_vol = factor_ret.rolling(63).std() * np.sqrt(252)

# 检测波动率警报
vol_threshold = rolling_vol.expanding().mean() + 2 * rolling_vol.expanding().std()
alert = rolling_vol > vol_threshold

ax.plot(rolling_vol.index, rolling_vol * 100, 
        label='Rolling Volatility (63-day)', linewidth=2.5, color='#3498db')
ax.fill_between(rolling_vol.index, rolling_vol * 100, 
                alpha=0.3, color='#3498db')
ax.scatter(rolling_vol.index[alert], rolling_vol[alert] * 100, 
          color='red', s=40, label='Volatility Alert', zorder=5)
ax.axhline(y=vol_threshold.mean() * 100, color='red', linestyle='--', 
           label='Alert Threshold', linewidth=2)

ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Annualized Volatility (%)', fontsize=12)
ax.set_title('Factor Volatility Monitoring System', fontsize=16, fontweight='bold')
ax.legend(fontsize=11, loc='upper left')
ax.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/volatility_monitoring.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 波动率监测图已生成: volatility_monitoring.png")

# 图3: 2020年价值因子崩溃回顾
fig, axes = plt.subplots(3, 1, figsize=(16, 14))

# 模拟2020年价值因子数据
dates_2020 = pd.date_range('2020-01-01', '2020-12-31', freq='B')
n_days_2020 = len(dates_2020)

# 模拟价值因子收益（3月崩溃）
value_ret_2020 = np.random.normal(0.0003, 0.008, n_days_2020)
# 添加3月崩溃
crash_start = 50  # 3月初
crash_end = 70    # 4月初
value_ret_2020[crash_start:crash_end] = np.random.normal(-0.02, 0.02, crash_end - crash_start)

value_factor_2020 = pd.Series(value_ret_2020, index=dates_2020)

# 子图1：因子累积收益
ax1 = axes[0]
cumulative_ret = (1 + value_factor_2020).cumprod() - 1
ax1.plot(cumulative_ret.index, cumulative_ret * 100, 
         linewidth=2.5, color='#2c3e50', label='Value Factor')
ax1.axvspan('2020-02-15', '2020-04-15', alpha=0.3, 
           color='red', label='COVID-19 Impact')
ax1.set_ylabel('Cumulative Return (%)', fontsize=12)
ax1.set_title('2020 Value Factor Drawdown', fontsize=16, fontweight='bold')
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# 子图2：衰减率（模拟）
ax2 = axes[1]
decay = pd.Series(np.random.uniform(-0.15, 0.05, n_days_2020), 
                  index=dates_2020)
ax2.plot(decay.index, decay * 100, linewidth=2, color='#e67e22')
ax2.axhline(y=-10, color='red', linestyle='--', label='Decay Alert Line', linewidth=2)
ax2.fill_between(decay.index, decay * 100, 0, 
                 where=(decay * 100 < -10), alpha=0.3, color='red')
ax2.set_ylabel('Decay Rate (%)', fontsize=12)
ax2.set_title('Factor Return Decay Rate', fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

# 子图3：波动率警报
ax3 = axes[2]
rolling_vol_2020 = value_factor_2020.rolling(63).std() * np.sqrt(252)
vol_alert = rolling_vol_2020 > rolling_vol_2020.quantile(0.95)

ax3.plot(rolling_vol_2020.index, rolling_vol_2020 * 100, 
         linewidth=2, color='#c0392b')
ax3.scatter(rolling_vol_2020.index[vol_alert], 
           rolling_vol_2020[vol_alert] * 100, 
           color='darkred', s=50, label='Volatility Alert', zorder=5)
ax3.set_ylabel('Annualized Volatility (%)', fontsize=12)
ax3.set_xlabel('Date', fontsize=12)
ax3.set_title('Factor Volatility Monitoring', fontsize=14)
ax3.legend(fontsize=11)
ax3.grid(True, alpha=0.3)
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/value_factor_2020_crash.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 2020年价值因子崩溃图已生成: value_factor_2020_crash.png")

# 图4: 拥挤度监测仪表盘
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Factor Crowding Monitoring Dashboard', fontsize=18, fontweight='bold')

# 模拟数据
factors = ['Market', 'Size', 'Value', 'Momentum']
n_days_dash = 500
dates_dash = pd.date_range('2022-01-01', periods=n_days_dash, freq='B')

for i, factor in enumerate(factors):
    row = i // 2
    col = i % 2
    ax = axes[row, col]
    
    # 生成因子收益
    ret = np.random.normal(0.0004, 0.01, n_days_dash)
    cum_ret = np.cumprod(1 + ret) - 1
    
    # 绘制累积收益
    ax.plot(dates_dash, cum_ret * 100, linewidth=2, color=colors[i % len(colors)])
    ax.set_title(f'{factor} Factor', fontsize=14, fontweight='bold')
    ax.set_ylabel('Cumulative Return (%)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加拥挤度指示器（模拟）
    crowding_score = np.random.uniform(30, 80, n_days_dash)
    ax_twin = ax.twinx()
    ax_twin.plot(dates_dash, crowding_score, alpha=0.3, 
                color='red', linewidth=1.5)
    ax_twin.axhline(y=70, color='darkred', linestyle='--', alpha=0.5)
    ax_twin.set_ylabel('Crowding Score', fontsize=11, color='red')
    ax_twin.tick_params(axis='y', labelcolor='red')
    ax_twin.set_ylim([0, 100])

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_dashboard.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("✅ 拥挤度监测仪表盘已生成: crowding_dashboard.png")

print("\n" + "="*60)
print("所有配图已成功生成！")
print("="*60)
