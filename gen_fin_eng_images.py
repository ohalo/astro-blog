#!/usr/bin/env python3
"""Generate images for financial-engineer-skills article"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

output_dir = '/Users/halo/workspace/astro-blog/public/images/financial-engineer-skills'

# === Image 1: Skill radar / full-stack map ===
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
ax.set_xlim(0, 10)
ax.set_ylim(0, 7)
ax.axis('off')

# Central node
center = (5, 3.5)
circle = plt.Circle(center, 0.4, color='#2D3436', ec='#FFD700', lw=3, zorder=10)
ax.add_patch(circle)
ax.text(center[0], center[1], '金融\n工程师', ha='center', va='center', fontsize=11,
        fontweight='bold', color='white')

# Skill nodes
skills = [
    (1.5, 5.5, '编程\nPython / C++ / Rust', '#FF6B6B'),
    (8.5, 5.5, '数学与统计\n随机微积分 / 时序分析', '#4ECDC4'),
    (1.5, 1.5, '系统设计\n数据管道 / 回测引擎', '#45B7D1'),
    (8.5, 1.5, '金融知识\n定价模型 / 市场结构', '#96CEB4'),
    (5, 6.2, 'DevOps\nCI/CD / 监控 / 灾备', '#FFEAA7'),
]

for x, y, label, color in skills:
    rect = mpatches.FancyBboxPatch((x-1.3, y-0.7), 2.6, 1.4, boxstyle="round,pad=0.15",
                                     facecolor=color, edgecolor='#333', linewidth=1.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x, y+0.15, label, ha='center', va='center', fontsize=10, fontweight='bold', color='#222')
    # Connect to center
    ax.plot([center[0], x], [center[1], y], '-', color='#B2BEC3', lw=1.5, alpha=0.6, zorder=0)

ax.set_title('金融工程师技能全栈图', fontsize=17, fontweight='bold', pad=20, color='#222')
fig.tight_layout()
fig.savefig(f'{output_dir}/fin-eng-skill-map.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("✅ fin-eng-skill-map.jpg")

# === Image 2: Backtest engine architecture ===
fig, ax = plt.subplots(1, 1, figsize=(12, 5))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Data layer
for i, (x, label, color) in enumerate([
    (2, '历史行情\n(OLAP)', '#FF6B6B'),
    (6, '实时行情\n(Tick)', '#4ECDC4'),
    (10, '基本面/另类\n(ETL)', '#45B7D1'),
]):
    rect = mpatches.FancyBboxPatch((x-1.2, 4.5), 2.4, 1.2, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='#555', lw=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, 5.1, label, ha='center', va='center', fontsize=9, fontweight='bold', color='#222')

ax.text(6, 5.95, '数据层', ha='center', fontsize=12, fontweight='bold', color='#333')

# Arrows data -> engine
for x in [2, 6, 10]:
    ax.annotate('', xy=(6, 3.8), xytext=(x, 4.4),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

# Event-driven engine
rect = mpatches.FancyBboxPatch((3, 2.5), 6, 1.5, boxstyle="round,pad=0.15",
                                 facecolor='#2D3436', edgecolor='#FFD700', lw=2.5, alpha=0.9)
ax.add_patch(rect)
ax.text(6, 3.25, '事件驱动回测引擎\nEvent-Driven Backtesting Engine', ha='center', va='center',
        fontsize=12, fontweight='bold', color='white')

ax.text(6, 4.2, '回测引擎层', ha='center', fontsize=12, fontweight='bold', color='#333')

# Arrows engine -> components
for x in [2, 6, 10]:
    ax.annotate('', xy=(x, 1.95), xytext=(6, 2.4),
                arrowprops=dict(arrowstyle='->', color='#888', lw=1.5))

# Output components
for i, (x, label, color) in enumerate([
    (2, '信号生成\n(Alpha)', '#96CEB4'),
    (6, '订单管理\n(OMS)', '#FFEAA7'),
    (10, '风控检查\n(Risk)', '#DDA0DD'),
]):
    rect = mpatches.FancyBboxPatch((x-1.2, 0.8), 2.4, 1.2, boxstyle="round,pad=0.1",
                                     facecolor=color, edgecolor='#555', lw=1.5, alpha=0.8)
    ax.add_patch(rect)
    ax.text(x, 1.4, label, ha='center', va='center', fontsize=9, fontweight='bold', color='#222')

ax.text(6, 2.2, '输出层', ha='center', fontsize=12, fontweight='bold', color='#333')

ax.set_title('事件驱动回测引擎架构', fontsize=15, fontweight='bold', pad=15, color='#222')
fig.tight_layout()
fig.savefig(f'{output_dir}/backtest-engine-arch.jpg', dpi=150, bbox_inches='tight')
plt.close()
print("✅ backtest-engine-arch.jpg")
