"""
生成因子拥挤度相关配图
"""
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建输出目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-crowding', exist_ok=True)

# 图1: 因子拥挤度生命周期
fig, ax = plt.subplots(figsize=(14, 6))

stages = ['低拥挤度\n(因子发现)', '因子发现期\n(业绩吸引)', '中度拥挤\n(资金流入)', 
          '高度拥挤\n(溢价衰减)', '因子崩盘\n(去拥挤化)', '恢复期\n(新周期)']
x = np.arange(len(stages))
y = [20, 35, 60, 85, 40, 55]

ax.plot(x, y, marker='o', linewidth=3, markersize=10, color='#E74C3C')
ax.fill_between(x, y, alpha=0.3, color='#E74C3C')
ax.set_xticks(x)
ax.set_xticklabels(stages, fontsize=11)
ax.set_ylabel('拥挤度评分', fontsize=12, fontweight='bold')
ax.set_title('因子拥挤度生命周期', fontsize=16, fontweight='bold', pad=20)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_ylim(0, 100)

# 添加关键节点标注
for i, (stage, score) in enumerate(zip(stages, y)):
    ax.annotate(f'{score}', xy=(i, score), xytext=(0, 10),
                textcoords='offset points', ha='center', fontsize=10,
                bbox=dict(boxstyle='round,pad=0.5', fc='yellow', alpha=0.7))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/lifecycle.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图2: 因子相关性热力图（正常 vs 拥挤）
from matplotlib.colors import TwoSlopeNorm

np.random.seed(42)
factors = ['价值', '动量', '低波', '质量', '规模']

# 正常期相关性
corr_normal = np.array([
    [1.00, 0.15, -0.10, 0.25, 0.20],
    [0.15, 1.00, 0.10, -0.05, 0.30],
    [-0.10, 0.10, 1.00, 0.15, -0.20],
    [0.25, -0.05, 0.15, 1.00, 0.10],
    [0.20, 0.30, -0.20, 0.10, 1.00]
])

# 拥挤期相关性（相关性异常升高）
corr_crowded = np.array([
    [1.00, 0.65, 0.45, 0.55, 0.50],
    [0.65, 1.00, 0.50, 0.40, 0.60],
    [0.45, 0.50, 1.00, 0.35, 0.25],
    [0.55, 0.40, 0.35, 1.00, 0.45],
    [0.50, 0.60, 0.25, 0.45, 1.00]
])

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 正常期
im1 = ax1.imshow(corr_normal, cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
ax1.set_xticks(range(len(factors)))
ax1.set_yticks(range(len(factors)))
ax1.set_xticklabels(factors, fontsize=11)
ax1.set_yticklabels(factors, fontsize=11)
ax1.set_title('正常期因子相关性', fontsize=14, fontweight='bold', pad=15)

# 添加数值标注
for i in range(len(factors)):
    for j in range(len(factors)):
        text = ax1.text(j, i, f'{corr_normal[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=10)

# 拥挤期
im2 = ax2.imshow(corr_crowded, cmap='RdYlBu_r', norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1))
ax2.set_xticks(range(len(factors)))
ax2.set_yticks(range(len(factors)))
ax2.set_xticklabels(factors, fontsize=11)
ax2.set_yticklabels(factors, fontsize=11)
ax2.set_title('拥挤期因子相关性', fontsize=14, fontweight='bold', pad=15)

# 添加数值标注
for i in range(len(factors)):
    for j in range(len(factors)):
        text = ax2.text(j, i, f'{corr_crowded[i, j]:.2f}',
                       ha='center', va='center', color='black', fontsize=10)

# 添加颜色条
cbar = plt.colorbar(im2, ax=[ax1, ax2], location='right', shrink=0.6)
cbar.set_label('相关系数', fontsize=12)

plt.suptitle('因子相关性对比：正常期 vs 拥挤期', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/correlation_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图3: ETF资金流向与因子收益
fig, ax1 = plt.subplots(figsize=(14, 7))

dates = pd.date_range('2020-01-01', '2024-12-31', freq='ME')
n_months = len(dates)

# 模拟ETF资金流向（累计）
etf_flows = np.cumsum(np.random.normal(50, 20, n_months))
etf_flows = pd.Series(etf_flows, index=dates)

# 模拟因子收益
factor_return = pd.Series(np.random.normal(0.01, 0.03, n_months), index=dates)
cumulative_return = (1 + factor_return).cumprod()

# 绘制资金流向
ax1.plot(dates, etf_flows, color='#3498DB', linewidth=2.5, label='ETF资金流入（百万美元）')
ax1.set_ylabel('ETF资金流入', fontsize=12, fontweight='bold', color='#3498DB')
ax1.tick_params(axis='y', labelcolor='#3498DB')
ax1.grid(True, alpha=0.3)

# 创建第二个Y轴
ax2 = ax1.twinx()
ax2.plot(dates, cumulative_return, color='#E74C3C', linewidth=2.5, 
         linestyle='--', label='因子累积收益')
ax2.set_ylabel('累积收益（归一化）', fontsize=12, fontweight='bold', color='#E74C3C')
ax2.tick_params(axis='y', labelcolor='#E74C3C')

# 添加关键事件标注
ax1.axvline(pd.Timestamp('2021-06-01'), color='orange', linestyle=':', 
            linewidth=2, label='拥挤度警告')
ax1.text(pd.Timestamp('2021-06-15'), etf_flows.max()*0.8, '拥挤度上升\n资金流入放缓', 
         bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7), fontsize=10)

plt.title('ETF资金流向与因子收益关系', fontsize=16, fontweight='bold', pad=20)
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.95), fontsize=11)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/etf_flow_returns.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 图4: 拥挤度预警系统示意图
fig, ax = plt.subplots(figsize=(14, 7))

levels = ['green', 'yellow', 'orange', 'red']
colors = ['#2ECC71', '#F1C40F', '#E67E22', '#E74C3C']
thresholds = [0, 40, 70, 85, 100]
descriptions = ['安全\n(0-40)', '关注\n(40-70)', '警告\n(70-85)', '危险\n(85-100)']

# 绘制色带
for i in range(len(levels)):
    ax.barh(0, thresholds[i+1]-thresholds[i], left=thresholds[i], 
            height=0.5, color=colors[i], edgecolor='white', linewidth=2)
    ax.text((thresholds[i] + thresholds[i+1])/2, 0, descriptions[i], 
            ha='center', va='center', fontsize=12, fontweight='bold', color='white')

# 模拟当前评分指示器
current_score = 78
ax.annotate('', xy=(current_score, 0), xytext=(current_score, -0.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=3))
ax.text(current_score, -0.8, f'当前评分: {current_score}\n(橙色警告)', 
        ha='center', fontsize=11, bbox=dict(boxstyle='round', facecolor='orange', alpha=0.7))

ax.set_xlim(0, 100)
ax.set_ylim(-1.5, 1)
ax.set_xlabel('拥挤度评分', fontsize=13, fontweight='bold')
ax.set_title('因子拥挤度预警系统', fontsize=16, fontweight='bold', pad=20)
ax.set_yticks([])
ax.grid(True, axis='x', alpha=0.3, linestyle='--')

# 添加操作建议框
suggestions = """
操作建议（橙色警告）:
  • 因子高度拥挤，强烈建议降权或暂停使用
  • 检查持仓集中度，避免流动性风险
  • 考虑切换到替代因子
"""
ax.text(102, 0, suggestions, fontsize=10, va='center',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/alert_system.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 已生成4张配图:")
print("  1. lifecycle.png - 因子拥挤度生命周期")
print("  2. correlation_comparison.png - 因子相关性对比")
print("  3. etf_flow_returns.png - ETF资金流向与因子收益")
print("  4. alert_system.png - 拥挤度预警系统")
