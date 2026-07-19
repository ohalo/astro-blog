#!/usr/bin/env python3
"""Generate images for factor-mining-pipeline article"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-mining-pipeline'

# === Image 1: Factor pipeline overview ===
fig, ax = plt.subplots(1, 1, figsize=(12, 4))
ax.set_xlim(0, 10)
ax.set_ylim(0, 4)
ax.axis('off')

stages = [
    ("数据准备", 0.5, "清洗 / 复权\n缺值 / 异常值"),
    ("因子构造", 2.5, "经济直觉 →\n数学公式"),
    ("因子检验", 4.5, "IC / 分组\n行业中性"),
    ("组合构建", 6.5, "多空 vs 纯多\n权重 / 容量"),
    ("实盘上线", 8.5, "纸交 / 监控\n灰度发布"),
]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7']

for i, (name, x, desc) in enumerate(stages):
    # Stage box
    rect = mpatches.FancyBboxPatch((x-0.8, 2), 1.6, 1.5, boxstyle="round,pad=0.1",
                                     facecolor=colors[i], edgecolor='#333', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x, 3.2, name, ha='center', va='center', fontsize=11, fontweight='bold', color='#222')
    ax.text(x, 2.6, desc, ha='center', va='center', fontsize=8, color='#444')

    # Arrow
    if i < len(stages) - 1:
        ax.annotate('', xy=(stages[i+1][1]-0.9, 2.75), xytext=(x+0.9, 2.75),
                    arrowprops=dict(arrowstyle='->', color='#555', lw=2.5))

ax.set_title('因子挖掘全生命周期流程图', fontsize=16, fontweight='bold', pad=15, color='#222')
fig.tight_layout()
fig.savefig(f'{output_dir}/factor-pipeline-overview.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("✅ factor-pipeline-overview.jpg")

# === Image 2: Factor IC decay over time ===
np.random.seed(42)
time = np.arange(0, 24)
ic_base = 0.05 * np.exp(-time / 12)
ic_series = ic_base + 0.01 * np.random.randn(24)

fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
ax.plot(time, ic_series, 'o-', color='#4ECDC4', linewidth=2, markersize=6, label='月度 IC')
ax.axhline(y=0, color='#FF6B6B', linestyle='--', linewidth=1, alpha=0.7)
ax.fill_between(time, 0, ic_series, alpha=0.15, color='#4ECDC4')
ax.axhline(y=ic_series[:6].mean(), color='#45B7D1', linestyle=':', linewidth=1.5, label=f'前6月IC均值: {ic_series[:6].mean():.3f}')
ax.axhline(y=ic_series[-6:].mean(), color='#E17055', linestyle=':', linewidth=1.5, label=f'后6月IC均值: {ic_series[-6:].mean():.3f}')

ax.set_xlabel('月份', fontsize=12)
ax.set_ylabel('Rank IC', fontsize=12)
ax.set_title('因子 IC 衰减趋势：信号随时间消退', fontsize=14, fontweight='bold')
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.5, 23.5)
fig.tight_layout()
fig.savefig(f'{output_dir}/factor-ic-decay.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("✅ factor-ic-decay.jpg")
