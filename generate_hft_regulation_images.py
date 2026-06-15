#!/usr/bin/env python3
"""
生成高频交易监管与合规文章的配图
"""
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# 创建输出目录
output_dir = '/Users/halo/workspace/astro-blog/public/images/hft-regulation-compliance'
os.makedirs(output_dir, exist_ok=True)

# 生成图1：全球高频交易监管框架对比
regions = ['美国', '欧盟', '中国', '日本', '英国', '加拿大', '澳大利亚']
regulation_levels = [85, 78, 82, 65, 80, 70, 68]  # 监管强度评分（0-100）
transparency_levels = [75, 72, 68, 70, 76, 65, 62]  # 透明度要求评分

x = np.arange(len(regions))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
bars1 = ax.bar(x - width/2, regulation_levels, width, label='监管强度', color='#2E86AB', alpha=0.8)
bars2 = ax.bar(x + width/2, transparency_levels, width, label='透明度要求', color='#A23B72', alpha=0.8)

ax.set_xlabel('国家/地区', fontsize=12)
ax.set_ylabel('评分', fontsize=12)
ax.set_title('全球高频交易监管框架对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(regions)
ax.legend()
ax.set_ylim(0, 100)

# 添加数值标签
def autolabel(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{height}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

autolabel(bars1)
autolabel(bars2)

plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{output_dir}/global_hft_regulation.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成图2：高频交易订单流与合规检查点
fig, ax = plt.subplots(figsize=(14, 8))

# 定义流程节点
nodes = [
    (1, 8, '订单生成\n(策略引擎)'),
    (3, 8, '风险控制\n(预交易检查)'),
    (5, 8, '订单路由\n(智能路由)'),
    (7, 8, '交易所\n(撮合引擎)'),
    (7, 5, '成交回报\n(实时)'),
    (5, 5, '合规监控\n(实时)'),
    (3, 5, '报告生成\n(定时)'),
    (1, 5, '监管报送\n(定时/实时)')
]

# 绘制节点
for (x, y, label) in nodes:
    box_color = '#2E86AB' if '风险' in label or '合规' in label or '监控' in label else '#A23B72'
    rect = plt.Rectangle((x-0.5, y-0.4), 1, 0.8, linewidth=2, 
                        edgecolor=box_color, facecolor=box_color, alpha=0.3)
    ax.add_patch(rect)
    ax.text(x, y, label, ha='center', va='center', fontsize=10, fontweight='bold')

# 绘制箭头（流程）
arrows = [
    ((1.5, 8), (2.5, 8)),  # 订单生成 -> 风险控制
    ((3.5, 8), (4.5, 8)),  # 风险控制 -> 订单路由
    ((5.5, 8), (6.5, 8)),  # 订单路由 -> 交易所
    ((7, 7.5), (7, 5.5)),  # 交易所 -> 成交回报
    ((6.5, 5), (5.5, 5)),  # 成交回报 -> 合规监控
    ((4.5, 5), (3.5, 5)),  # 合规监控 -> 报告生成
    ((2.5, 5), (1.5, 5)),  # 报告生成 -> 监管报送
    ((5, 7.5), (5, 5.5)),  # 订单路由 -> 合规监控（虚线，监控）
]

for (start, end) in arrows[:4]:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='black', lw=2))

for (start, end) in arrows[4:]:
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color='gray', lw=2, linestyle='--'))

ax.set_xlim(0, 8)
ax.set_ylim(4, 9)
ax.set_xlabel('流程阶段', fontsize=12)
ax.set_ylabel('处理层级', fontsize=12)
ax.set_title('高频交易订单流与合规检查点', fontsize=14, fontweight='bold')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# 添加说明文本框
textstr = '实线: 订单流\n虚线: 监控流\n蓝色框: 合规相关'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='bottom', bbox=props)

plt.tight_layout()
plt.savefig(f'{output_dir}/hft_order_flow_compliance.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成图3：高频交易违规类型分布（模拟数据）
violation_types = ['幌骗(Spoofing)', '分层挂单(Layering)', '动量点燃(Momentum Ignition)', 
                   '抢先交易(Front Running)', '交叉交易(Cross Trading)', '其他']
violation_counts = [35, 28, 18, 12, 5, 2]  # 占比（%）

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#FF6B6B', '#FFA07A', '#FFD700', '#98FB98', '#87CEEB', '#D3D3D3']
wedges, texts, autotexts = ax.pie(violation_counts, labels=violation_types, 
                                  colors=colors, autopct='%1.1f%%',
                                  startangle=90, textprops={'fontsize': 10})

# 突出显示前三种违规类型
wedges[0].set_edgecolor('red')
wedges[0].set_linewidth(2)
wedges[1].set_edgecolor('red')
wedges[1].set_linewidth(2)

ax.set_title('高频交易违规类型分布（2020-2025）', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_dir}/hft_violation_types.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 已生成3张配图到 {output_dir}")
print("   - global_hft_regulation.png")
print("   - hft_order_flow_compliance.png")
print("   - hft_violation_types.png")
