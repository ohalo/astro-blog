#!/usr/bin/env python3
"""Generate figures for Barra factor model article"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches

# Set style
plt.rcParams.update({'font.size': 11})

# ===== Figure 1: Barra Factor Attribution Framework =====
fig, ax = plt.subplots(figsize=(12, 7))

categories = ['Size', 'Beta', 'Momentum', 'Volatility', 'Quality',
              'Growth', 'Value', 'Leverage', 'Liquidity', 'Sentiment']

contributions = [0.052, 0.428, -0.187, -0.093, 0.245,
                  -0.034, 0.312, 0.021, -0.056, 0.089]

total_factor = sum(contributions)
alpha = 0.223
total = total_factor + alpha

colors = ['#e74c3c' if x < 0 else '#2ecc71' if x > 0.1 else '#3498db' for x in contributions]

bars = ax.barh(categories, contributions, color=colors, edgecolor='white', height=0.6)
ax.axvline(x=0, color='black', linewidth=0.8)

# Add value labels
for i, (v, c) in enumerate(zip(contributions, categories)):
    ax.text(v + 0.01 if v >= 0 else v - 0.08, i, f'{v:+.3f}', va='center', fontsize=9, fontweight='bold')

ax.set_xlabel('Factor Contribution to Active Return', fontsize=12)
ax.set_title('Barra Multi-Factor Performance Attribution\nPortfolio Active Return Decomposition', fontsize=14, fontweight='bold')

# Add summary boxes
summary_text = f'Total Factor Return: {total_factor:+.3f}\nSpecific Alpha: {alpha:+.3f}\nTotal Active Return: {total:+.3f}'
ax.text(0.98, 0.02, summary_text, transform=ax.transAxes, fontsize=11,
        bbox=dict(boxstyle='round', facecolor='#f8f9fa', edgecolor='#dee2e6'),
        verticalalignment='bottom', horizontalalignment='right', fontfamily='monospace')

legend_elements = [
    mpatches.Patch(color='#2ecc71', label='Positive Contribution'),
    mpatches.Patch(color='#e74c3c', label='Negative Contribution'),
    mpatches.Patch(color='#3498db', label='Small Contribution')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_xlim(-0.35, 0.65)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/2026-07-26-barra-factor-model/barra-attribution.png', dpi=150, bbox_inches='tight')
plt.close()

# ===== Figure 2: Factor Attribution Breakdown Chart (pie-style sunburst alternative) =====
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: Waterfall-style contribution bars
ax1 = axes[0]
items = ['Sector\nAllocation', 'Style\nFactors', 'Interaction\nEffect', 'Selection\n(Alpha)', 'Total\nActive']
values = [0.082, sum(contributions) - 0.052, -0.031, alpha, total]
waterfall_colors = ['#3498db', '#2ecc71', '#e74c3c', '#9b59b6', '#2c3e50']

bars_wf = ax1.bar(items, values, color=waterfall_colors, edgecolor='white', width=0.5)
for bar, val in zip(bars_wf, values):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02 if val >= 0 else bar.get_height() - 0.06,
             f'{val:+.3f}', ha='center', va='bottom' if val >= 0 else 'top', fontweight='bold', fontsize=11)

ax1.axhline(y=0, color='black', linewidth=0.8)
ax1.set_title('Performance Attribution Breakdown', fontsize=13, fontweight='bold')
ax1.set_ylabel('Active Return Contribution', fontsize=11)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Right: Cumulative factor risk decomposition (pie chart)
ax2 = axes[1]
risk_shares = [8.2, 28.5, 12.1, 15.3, 7.8, 9.6, 11.2, 3.4, 2.1, 1.8]
risk_labels = ['Size', 'Beta', 'Momentum', 'Volatility', 'Quality',
               'Growth', 'Value', 'Leverage', 'Liquidity', 'Sentiment']

wedges, texts, autotexts = ax2.pie(risk_shares, labels=risk_labels, autopct='%1.1f%%',
                                     colors=plt.cm.tab10(np.linspace(0, 1, 10)),
                                     startangle=90, pctdistance=0.75)
for t in autotexts:
    t.set_fontsize(7)
for t in texts:
    t.set_fontsize(8)

ax2.set_title('Factor Risk Contribution\n(% of Total Factor Risk)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/2026-07-26-barra-factor-model/factor-attribution-chart.png', dpi=150, bbox_inches='tight')
plt.close()

print("✅ Generated 2 figures for barra-factor-model")
