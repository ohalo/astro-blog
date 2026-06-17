#!/usr/bin/env python3
"""Generate images for quant-researcher-research-directions article"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = '/Users/halo/workspace/astro-blog/public/images/quant-researcher-research-directions'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set Chinese font
plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== Image 1: Quant Researcher Skill Radar =====
fig, ax = plt.subplots(figsize=(10, 8), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

categories = ['数学统计', '机器学习', '编程实现', '金融知识', '数据工程', '研究思维']
values = [90, 85, 80, 75, 70, 88]
N = len(categories)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
values += values[:1]

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, color='white', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], color='#94a3b8', fontsize=9)
ax.fill(angles, values, color='#3b82f6', alpha=0.3)
ax.plot(angles, values, color='#60a5fa', linewidth=2.5, marker='o', markersize=8, markerfacecolor='#1d4ed8')

ax.grid(color='#334155', linewidth=0.8)
for spine in ax.spines.values():
    spine.set_edgecolor('#334155')

plt.title('量化研究员核心能力雷达图', color='white', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/skill_radar.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Image 1 saved: skill_radar.png')

# ===== Image 2: Research Direction Flowchart =====
fig, ax = plt.subplots(figsize=(12, 7))
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis('off')

# Define boxes as (x, y, w, h, color, text)
boxes = [
    (4.5, 5.8, 3, 0.8, '#1d4ed8', '量化研究员\n研究方向'),
    (0.5, 3.8, 2.5, 1.2, '#047857', 'Alpha 研究\n• 因子挖掘\n• 信号构建\n• 另类数据'),
    (3.5, 3.8, 2.5, 1.2, '#b45309', '风险研究\n• 因子模型\n• 风险预算\n• 压力测试'),
    (6.5, 3.8, 2.5, 1.2, '#7c3aed', '执行研究\n• 交易成本\n• 市场微观结构\n• 最优执行'),
    (9.5, 3.8, 2.5, 1.2, '#be123c', '资产配置\n• 战略配置\n• 战术轮动\n• 多资产建模'),
]

for x, y, w, h, color, text in boxes:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1", 
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h/2, text, ha='center', va='center', color='white', 
            fontsize=10, fontweight='bold')

# Arrows
for start_x in [1.75, 4.75, 7.75, 10.75]:
    ax.annotate('', xy=(5.25, 4.6), xytext=(start_x, 3.65),
                arrowprops=dict(arrowstyle='->', color='#475569', lw=1.5, connectionstyle='arc3,rad=0.1'))

plt.title('量化研究员四大研究方向', color='white', fontsize=16, fontweight='bold', pad=10)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/research_directions.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Image 2 saved: research_directions.png')

print('All images generated!')
