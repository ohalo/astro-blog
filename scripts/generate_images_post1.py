#!/usr/bin/env python3
"""
为第一篇文章生成配图：多因子模型风险分解
只使用已安装的包：numpy, matplotlib, scipy
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 创建模拟数据
np.random.seed(42)

# 模拟风险分解数据
factors = ['Market', 'SMB', 'HML', 'UMD']
factor_risk_contrib = [35.2, 18.5, 22.3, 24.0]  # 风险贡献百分比
factor_exposures = [1.05, -0.25, 0.38, 0.42]  # 因子暴露

# 图1: 风险分解可视化
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Multi-Factor Risk Decomposition Analysis', fontsize=16, fontweight='bold')

# 1. 风险构成饼图
ax1 = axes[0, 0]
risks = ['Factor Risk', 'Specific Risk']
values = [0.65, 0.35]  # 65%因子风险，35%特异性风险
colors = ['#FF6B6B', '#4ECDC4']

wedges, texts, autotexts = ax1.pie(values, labels=risks, autopct='%1.1f%%',
                                   colors=colors, startangle=90)
ax1.set_title('Risk Composition', fontsize=14, fontweight='bold')

# 2. 因子风险贡献柱状图
ax2 = axes[0, 1]
colors_bar = ['#FF6B6B' if e > 0 else '#4ECDC4' for e in factor_exposures]
bars = ax2.bar(factors, factor_risk_contrib, color=colors_bar)
ax2.set_title('Factor Risk Contribution (%)', fontsize=14, fontweight='bold')
ax2.set_ylabel('Risk Contribution (%)')
ax2.tick_params(axis='x', rotation=45)

# 添加数值标签
for bar, value in zip(bars, factor_risk_contrib):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f'{value:.1f}%', ha='center', va='bottom')

# 3. 因子暴露柱状图
ax3 = axes[1, 0]
bars = ax3.bar(factors, factor_exposures, color=colors_bar)
ax3.set_title('Factor Exposures (Beta)', fontsize=14, fontweight='bold')
ax3.set_ylabel('Beta')
ax3.tick_params(axis='x', rotation=45)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_ylim(-0.5, 1.2)

# 添加数值标签
for bar, value in zip(bars, factor_exposures):
    ax3.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + (0.02 if value > 0 else -0.05),
            f'{value:.3f}', ha='center', va='bottom' if value > 0 else 'top')

# 4. 风险贡献vs暴露散点图
ax4 = axes[1, 1]
ax4.scatter(factor_exposures, factor_risk_contrib, s=200, alpha=0.6, c=colors_bar)
ax4.set_title('Risk Contribution vs Factor Exposure', fontsize=14, fontweight='bold')
ax4.set_xlabel('Factor Exposure (Beta)')
ax4.set_ylabel('Risk Contribution (%)')
ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.5)
ax4.set_xlim(-0.5, 1.2)
ax4.set_ylim(-5, 40)

# 添加因子标签
for factor, x, y in zip(factors, factor_exposures, factor_risk_contrib):
    ax4.annotate(factor, (x, y), xytext=(5, 5), textcoords='offset points',
                fontsize=10, alpha=0.7)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/risk_decomposition.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 1 generated: risk_decomposition.png")
plt.close()

# 图2: 收益归因可视化
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Return Attribution Analysis', fontsize=16, fontweight='bold')

# 1. 收益构成饼图
ax1 = axes[0]
returns = ['Factor Return', 'Selection Return']
values = [0.08, 0.03]  # 8%因子收益，3%个股选择收益
colors = ['#FF6B6B', '#4ECDC4']

wedges, texts, autotexts = ax1.pie(values, labels=returns, autopct='%1.2f%%',
                                   colors=colors, startangle=90)
ax1.set_title('Return Attribution', fontsize=14, fontweight='bold')

# 2. 因子收益柱状图
ax2 = axes[1]
factor_rets = [0.045, -0.008, 0.023, 0.020]  # 各因子收益
colors_bar = ['#FF6B6B' if r > 0 else '#4ECDC4' for r in factor_rets]

bars = ax2.bar(factors, factor_rets, color=colors_bar)
ax2.set_title('Factor Return Breakdown', fontsize=14, fontweight='bold')
ax2.set_ylabel('Return')
ax2.tick_params(axis='x', rotation=45)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_ylim(-0.015, 0.055)

# 添加数值标签
for bar, value in zip(bars, factor_rets):
    ax2.text(bar.get_x() + bar.get_width()/2, 
            bar.get_height() + (0.002 if value > 0 else -0.004),
            f'{value:.2%}', ha='center', va='bottom' if value > 0 else 'top')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/return_attribution.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 2 generated: return_attribution.png")
plt.close()

# 图3: 额外配图 - 因子收益率时序图
fig, ax = plt.subplots(figsize=(14, 6))

# 生成模拟的因子收益率时序
np.random.seed(42)
n_days = 500
factor_returns_np = {
    'Market': np.random.normal(0.0005, 0.01, n_days),
    'SMB': np.random.normal(0.0002, 0.005, n_days),
    'HML': np.random.normal(0.0003, 0.006, n_days),
    'UMD': np.random.normal(0.0004, 0.008, n_days),
}

# 计算累计收益
cumulative_returns = {}
for key in factor_returns_np:
    cumulative_returns[key] = np.cumprod(1 + factor_returns_np[key])

# 绘制
x = np.arange(n_days)
for i, (key, values) in enumerate(cumulative_returns.items()):
    ax.plot(x, values, label=key, linewidth=2, color=plt.cm.Set1(i))

ax.set_title('Factor Cumulative Returns', fontsize=16, fontweight='bold')
ax.set_xlabel('Trading Days')
ax.set_ylabel('Cumulative Return')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/factor_cumulative_returns.png',
            dpi=300, bbox_inches='tight')
print("✓ Figure 3 generated: factor_cumulative_returns.png")
plt.close()

print("\n✅ All images for Post 1 generated successfully!")
print("Location: /Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/")
