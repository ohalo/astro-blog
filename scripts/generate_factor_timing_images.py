"""
生成因子择时文章的配图
"""
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 图1: hero.png - 因子择时概念图
fig, ax = plt.subplots(figsize=(12, 8), facecolor='#1a1a2e')

# 创建背景渐变效果
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.axis('off')

# 添加标题
ax.text(5, 9, 'Factor Timing', fontsize=48, fontweight='bold', 
        ha='center', va='center', color='#e94560')

ax.text(5, 8, '动态调整因子暴露', fontsize=28, 
        ha='center', va='center', color='#0f3460')

# 绘制因子轮动示意图
factors = ['Value', 'Momentum', 'Quality', 'Low Vol', 'Size']
colors = ['#e94560', '#0f3460', '#533483', '#e8a87c', '#41b3a3']

# 创建圆形布局
angles = np.linspace(0, 2*np.pi, len(factors), endpoint=False)
x = 5 + 3 * np.cos(angles)
y = 5 + 3 * np.sin(angles)

for i, (factor, color, xi, yi) in enumerate(zip(factors, colors, x, y)):
    # 绘制圆形
    circle = plt.Circle((xi, yi), 0.8, color=color, alpha=0.7)
    ax.add_patch(circle)
    
    # 添加因子名称
    ax.text(xi, yi, factor, fontsize=16, fontweight='bold', 
            ha='center', va='center', color='white')

# 添加中心文本
ax.text(5, 5, 'Timing', fontsize=24, fontweight='bold', 
        ha='center', va='center', color='#1a1a2e',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='none', alpha=0.8))

# 添加箭头表示轮动
for i in range(len(factors)):
    start_angle = angles[i]
    end_angle = angles[(i+1) % len(factors)]
    
    arrow_x = [5 + 2.5 * np.cos(start_angle), 5 + 2.5 * np.cos(end_angle)]
    arrow_y = [5 + 2.5 * np.sin(start_angle), 5 + 2.5 * np.sin(end_angle)]
    
    ax.annotate('', xy=(arrow_x[1], arrow_y[1]), xytext=(arrow_x[0], arrow_y[0]),
                arrowprops=dict(arrowstyle='->', color='white', lw=2, alpha=0.6))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/hero.png', 
            dpi=150, bbox_inches='tight', facecolor='#1a1a2e')
plt.close()

print("✓ 生成 hero.png")

# 图2: factor-rotation.png - 因子收益轮动热力图
fig, ax = plt.subplots(figsize=(14, 8))

# 模拟因子月度收益数据
np.random.seed(42)
dates = pd.date_range(start='2020-01', end='2025-12', freq='ME')
factors = ['Value', 'Momentum', 'Quality', 'Low_Vol', 'Size', 'Market']

# 生成具有周期性的因子收益
n_months = len(dates)
returns = np.zeros((n_months, len(factors)))

# Value因子：与经济周期相关
value_cycle = np.sin(np.linspace(0, 4*np.pi, n_months)) * 0.02
returns[:, 0] = value_cycle + np.random.normal(0, 0.01, n_months)

# Momentum因子：趋势市表现好
momentum_trend = np.concatenate([
    np.linspace(0.01, 0.03, n_months//3),
    np.linspace(0.03, -0.01, n_months//3),
    np.linspace(-0.01, 0.02, n_months - 2*(n_months//3))
])
returns[:, 1] = momentum_trend + np.random.normal(0, 0.015, n_months)

# Quality因子：防御性强
quality_defensive = -0.5 * (returns[:, 0] + returns[:, 1]) + 0.005
returns[:, 2] = quality_defensive + np.random.normal(0, 0.008, n_months)

# Low Vol因子：市场波动时表现好
market_vol = np.abs(np.random.normal(0, 0.015, n_months))
returns[:, 3] = 0.01 - 0.3 * market_vol + np.random.normal(0, 0.01, n_months)

# Size因子：小盘股溢价
size_premium = np.random.normal(0.002, 0.02, n_months)
returns[:, 4] = size_premium

# Market因子：市场整体收益
returns[:, 5] = np.random.normal(0.008, 0.04, n_months)

# 创建DataFrame
returns_df = pd.DataFrame(returns, index=dates, columns=factors)

# 绘制热力图
im = ax.imshow(returns_df.T, aspect='auto', cmap='RdYlGn', 
               vmin=-0.05, vmax=0.05)

# 设置坐标轴
ax.set_xticks(range(0, len(dates), 12))
ax.set_xticklabels([d.strftime('%Y') for d in dates[::12]], rotation=45)
ax.set_yticks(range(len(factors)))
ax.set_yticklabels(factors)

# 添加颜色条
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Monthly Return', rotation=270, labelpad=20)

# 添加标题
ax.set_title('Factor Return Rotation (2020-2025)', fontsize=20, fontweight='bold', pad=20)

# 添加注释
ax.text(0.02, 0.98, 'Red = Negative Return | Green = Positive Return\nDarker = Larger Magnitude', 
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor-rotation.png', 
            dpi=150, bbox_inches='tight')
plt.close()

print("✓ 生成 factor-rotation.png")

print("\n所有配图生成完成！")
