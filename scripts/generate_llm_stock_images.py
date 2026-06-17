#!/usr/bin/env python3
"""Generate images for training-llm-stock-prediction article"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
import os

OUTPUT_DIR = '/Users/halo/workspace/astro-blog/public/images/training-llm-stock-prediction'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.rcParams['font.sans-serif'] = ['PingFang SC', 'Heiti SC', 'STHeiti', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ===== Image 1: LLM Training Pipeline for Finance =====
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#0f172a')
ax.set_xlim(0, 14)
ax.set_ylim(0, 6)
ax.axis('off')

stages = [
    (0.3, 2.5, 2.2, 2.2, '#1e40af', 'Phase 1\n预训练', '海量金融文本\n• 年报/公告\n• 财经新闻\n• 研报/路演'),
    (3, 2.5, 2.2, 2.2, '#047857', 'Phase 2\n金融微调', '领域适配\n• 金融问答对\n• 情感分析标注\n• NER实体识别'),
    (5.7, 2.5, 2.2, 2.2, '#b45309', 'Phase 3\n指令对齐', '预测能力注入\n• 价格序列理解\n• 技术指标解读\n• 基本面分析'),
    (8.4, 2.5, 2.2, 2.2, '#7c3aed', 'Phase 4\nRLHF优化', '偏好对齐\n• 人类交易员反馈\n• 回测结果排序\n• 安全约束对齐'),
    (11.1, 2.5, 2.2, 2.2, '#be123c', 'Phase 5\n部署推理', '实时预测\n• 多模态输入\n• 实时数据流\n• 在线学习'),
]

for x, y, w, h, color, title, desc in stages:
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
    ax.add_patch(rect)
    ax.text(x + w/2, y + h - 0.5, title, ha='center', va='top', color='white', fontsize=11, fontweight='bold')
    ax.text(x + w/2, y + 0.5, desc, ha='center', va='bottom', color='#e2e8f0', fontsize=8, linespacing=1.5)

# Arrows between stages
for i in range(4):
    x1 = stages[i][0] + stages[i][2]
    x2 = stages[i+1][0]
    y_mid = stages[i][1] + stages[i][3]/2
    ax.annotate('', xy=(x2 - 0.05, y_mid), xytext=(x1 + 0.05, y_mid),
                arrowprops=dict(arrowstyle='->', color='#60a5fa', lw=2.5))

plt.title('训练大模型成为股市预测专家：五阶段训练流水线', color='white', fontsize=15, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/training_pipeline.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Image 1 saved: training_pipeline.png')

# ===== Image 2: LLM vs Traditional Methods Comparison =====
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0f172a')
ax.set_facecolor('#1e293b')

methods = ['传统时序\nARIMA/GARCH', '机器学习\nXGBoost/LGBM', '深度学习\nLSTM/Transformer', '大语言模型\nGPT/LLaMA', '多模态LLM\n文本+图表']
accuracy = [52, 58, 65, 62, 68]
sharpe = [0.8, 1.2, 1.6, 1.4, 1.9]
multimodal = [0, 1, 2, 4, 5]
explainability = [5, 3, 2, 5, 5]

x = np.arange(len(methods))
width = 0.2

bars1 = ax.bar(x - 1.5*width, [v*1.0 for v in accuracy], width, label='预测准确率(%)', color='#3b82f6', alpha=0.85)
bars2 = ax.bar(x - 0.5*width, [v*1.0 for v in sharpe], width, label='夏普比率', color='#10b981', alpha=0.85)
bars3 = ax.bar(x + 0.5*width, multimodal, width, label='多模态能力(0-5)', color='#f59e0b', alpha=0.85)
bars4 = ax.bar(x + 1.5*width, explainability, width, label='可解释性(0-5)', color='#ef4444', alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(methods, color='white', fontsize=9)
ax.set_ylabel('得分', color='white', fontsize=11)
ax.tick_params(axis='y', colors='white')
ax.set_ylim(0, 80)
ax.legend(loc='upper left', fontsize=8, facecolor='#334155', edgecolor='#475569', labelcolor='white')
ax.grid(axis='y', color='#334155', linewidth=0.5, alpha=0.5)
for spine in ax.spines.values():
    spine.set_edgecolor('#475569')

plt.title('不同方法在股市预测中的多维度对比', color='white', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/method_comparison.png', dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
plt.close()
print('Image 2 saved: method_comparison.png')

print('All images generated!')
