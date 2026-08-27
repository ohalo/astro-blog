import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ===== Image 1: CDS-bond basis mechanics: spread comparison =====
fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
entities = ['Germany', 'France', 'Italy', 'Spain', 'Portugal', 'Greece', 'Brazil', 'Turkey', 'South Africa']
bond_spread = np.array([0.25, 0.35, 1.20, 0.90, 1.80, 3.50, 2.80, 4.20, 3.00])
cds_spread = np.array([0.15, 0.25, 1.05, 0.75, 1.55, 3.10, 2.20, 3.60, 2.40])
basis = bond_spread - cds_spread

x = np.arange(len(entities))
width = 0.35
bars1 = ax.bar(x - width/2, bond_spread, width, label='Sovereign Bond Spread (%)', color='darkred', alpha=0.8)
bars2 = ax.bar(x + width/2, cds_spread, width, label='CDS Spread (%)', color='navy', alpha=0.8)
ax.set_xticks(x)
ax.set_xticklabels(entities, rotation=30, ha='right')
ax.set_ylabel('Spread (%)', fontsize=11)
ax.set_title('Sovereign CDS-Bond Basis: Protection Often Cheaper than Cash Bonds', fontsize=12)
ax.legend()
ax.grid(True, axis='y', alpha=0.3)

# annotate basis values on top
for i, b in enumerate(basis):
    ax.text(i, max(bond_spread[i], cds_spread[i]) + 0.1, f'Basis {b:.2f}', ha='center', fontsize=8, color='green')

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/sovereign-cds-basis/cds_bond_basis_bar.png')
plt.close()

# ===== Image 2: Time series of CDS-bond basis =====
np.random.seed(3)
days = 252 * 4
basis_ts = 0.40 + 0.15 * np.sin(np.linspace(0, 4*np.pi, days)) + np.random.normal(0, 0.08, days)
# add a crisis widening
crisis_start = 600
crisis_end = 750
basis_ts[crisis_start:crisis_end] += np.linspace(0, 0.80, crisis_end - crisis_start)
basis_ts[crisis_end:crisis_end+100] -= np.linspace(0.80, 0.10, 100)

dates = pd.date_range('2020-01-01', periods=days, freq='B')
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(dates, basis_ts, color='purple', lw=2, label='CDS-Bond Basis (%)')
ax.axhline(0, color='gray', linestyle='--', alpha=0.7)
ax.axhline(np.mean(basis_ts), color='green', linestyle=':', alpha=0.7, label=f'Mean={np.mean(basis_ts):.2f}%')
ax.fill_between(dates, basis_ts, 0, where=(basis_ts > 0), color='green', alpha=0.15)
ax.fill_between(dates, basis_ts, 0, where=(basis_ts <= 0), color='red', alpha=0.15)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Basis (%)', fontsize=11)
ax.set_title('CDS-Bond Basis Over Time: Persistent Positive Premium', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/sovereign-cds-basis/cds_basis_timeseries.png')
plt.close()

# ===== Image 3: Arbitrage payoff: long bond + buy CDS =====
np.random.seed(5)
T = 1000
cash_bond_premium = 0.004 + np.random.normal(0, 0.001, T)  # positive carry
cds_premium_cost = 0.0015 + np.random.normal(0, 0.0005, T)
net_carry = cash_bond_premium - cds_premium_cost
# default event at some point
default_t = 700
net_carry[default_t:] += 0.010  # default payoff from CDS
net_carry[default_t-50:default_t] -= 0.003  # pre-default bond cheapening

cum_arb = np.cumprod(1 + net_carry)
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(cum_arb, color='darkgreen', lw=2, label='Long Bond + Buy CDS (replicating risk-free)')
ax.axvline(default_t, color='red', linestyle=':', alpha=0.7, label='Credit Event')
ax.set_xlabel('Trading Days', fontsize=11)
ax.set_ylabel('Cumulative Wealth', fontsize=11)
ax.set_title('Synthetic Risk-Free Arbitrage: Why Positive Basis is Not Free Lunch', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/sovereign-cds-basis/bond_cds_arbitrage.png')
plt.close()

print('Images saved for sovereign-cds-basis.')
