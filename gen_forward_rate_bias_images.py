import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# ===== Image 1: Forward rate bias scatter & regression =====
np.random.seed(42)
countries = ['AUD', 'NZD', 'GBP', 'USD', 'EUR', 'JPY', 'CHF', 'SEK', 'NOK', 'CAD']
n = len(countries)
# interest rate differential vs spot depreciation (annualized, %)
ir_diff = np.array([2.8, 2.5, 0.6, 0.0, -0.2, -1.2, -1.0, -0.5, -0.3, 0.1])
spot_dep = -0.6 * ir_diff + np.random.normal(0, 0.8, n)  # negative coeff => forward rate bias

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
ax.scatter(ir_diff, spot_dep, s=120, c='steelblue', edgecolor='k', zorder=3)
for i, c in enumerate(countries):
    ax.annotate(c, (ir_diff[i], spot_dep[i]), textcoords="offset points", xytext=(5, 5), fontsize=9)
# regression line
x_fit = np.linspace(ir_diff.min() - 0.5, ir_diff.max() + 0.5, 100)
coeff = np.polyfit(ir_diff, spot_dep, 1)
ax.plot(x_fit, np.polyval(coeff, x_fit), 'r--', lw=2, label=f'OLS slope={coeff[0]:.2f}')
ax.axhline(0, color='gray', linewidth=0.8)
ax.axvline(0, color='gray', linewidth=0.8)
ax.set_xlabel('Interest Rate Differential (%)', fontsize=11)
ax.set_ylabel('Spot Depreciation (%)', fontsize=11)
ax.set_title('Forward Rate Bias: Higher Yielders Depreciate Less Than UIP Predicts', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/forward-rate-bias-carry-trade/forward_bias_scatter.png')
plt.close()

# ===== Image 2: Cumulative carry trade returns =====
np.random.seed(7)
days = 252 * 5
# high-yield basket (long) vs low-yield basket (short)
# carry component plus small excess return (forward rate bias premium)
ret_high = np.random.normal(0.03 / 252, 0.008, days)
ret_low = np.random.normal(0.005 / 252, 0.006, days)
carry_premium = (0.025 / 252) * np.ones(days)
# add occasional crash for realism
crash_day = 900
ret_high[crash_day:crash_day+20] += np.linspace(-0.12, -0.02, 20)

strat_ret = ret_high - ret_low + carry_premium
# UIP-implied: carry exactly offset by expected spot move => zero excess return
uip_ret = np.random.normal(0, 0.005 / 252, days)

cum_strat = np.cumprod(1 + strat_ret)
cum_uip = np.cumprod(1 + uip_ret)

dates = pd.date_range('2021-01-01', periods=days, freq='B')
fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
ax.plot(dates, cum_strat, label='Carry Trade Strategy (High minus Low + carry)', lw=2, color='darkgreen')
ax.plot(dates, cum_uip, label='UIP Fair-Value Benchmark (zero excess return)', lw=2, color='gray', linestyle='--')
ax.axvline(dates[crash_day], color='red', linestyle=':', alpha=0.7, label='Currency Crash')
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Cumulative Wealth', fontsize=11)
ax.set_title('Carry Trade Cumulative Returns vs UIP Null', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/forward-rate-bias-carry-trade/carry_trade_cumreturns.png')
plt.close()

# ===== Image 3: Distribution of carry trade returns with crash risk =====
np.random.seed(11)
simulations = 10000
annual_returns = np.random.normal(0.04, 0.05, simulations)
# add fat-left tail
crash = np.random.choice([0, -0.25], size=simulations, p=[0.92, 0.08])
annual_returns += crash

fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
counts, bins, patches = ax.hist(annual_returns, bins=80, density=True, alpha=0.7, color='steelblue', edgecolor='k')
ax.axvline(np.median(annual_returns), color='darkgreen', lw=2, linestyle='-', label=f'Median={np.median(annual_returns):.1%}')
ax.axvline(np.percentile(annual_returns, 5), color='red', lw=2, linestyle='--', label=f'5th percentile={np.percentile(annual_returns,5):.1%}')
ax.set_xlabel('Annual Excess Return', fontsize=11)
ax.set_ylabel('Density', fontsize=11)
ax.set_title('Carry Trade Return Distribution: Positive Median, Fat Left Tail', fontsize=12)
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/forward-rate-bias-carry-trade/carry_return_distribution.png')
plt.close()

print('Images saved for forward-rate-bias-carry-trade.')
