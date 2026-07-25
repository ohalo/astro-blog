#!/usr/bin/env python3
"""VPIN (Easley, Lopez de Prado, O'Hara 2012) simulation: volume buckets, flow toxicity, crash early warning."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/vpin-flow-toxicity"
os.makedirs(OUT, exist_ok=True)

# ---------- simulate intraday-ish tick stream over ~2 years with 3 toxicity episodes + 1 crash ----------
# time in "minutes", ~500 days * 240 min
n_days = 500
mins_per_day = 240
T = n_days * mins_per_day
t = np.arange(T)

# base order flow imbalance process: OU around 0, plus toxicity episodes where informed selling dominates
imb = np.zeros(T)
phi = 0.97
sigma_imb = 0.10
# toxicity episodes: informed flow ramps up BEFORE price crash
episodes = [(int(T*0.30), int(T*0.30)+8*mins_per_day, -0.25),   # bearish toxic, mild selloff
            (int(T*0.55), int(T*0.55)+6*mins_per_day, 0.22),    # bullish toxic (runup)
            (int(T*0.80), int(T*0.80)+4*mins_per_day, -0.45)]   # heavy toxic -> flash-crash style drop
drift = np.zeros(T)
for s, e, lvl in episodes:
    ramp = np.linspace(0, lvl, e - s)
    drift[s:e] = ramp
eps = rng.normal(0, sigma_imb, T)
for i in range(1, T):
    imb[i] = phi * imb[i-1] + (1-phi) * drift[i] * 6 + eps[i] * np.sqrt(1-phi**2)
imb = np.clip(imb + drift, -0.9, 0.9)

# volume per minute: lognormal, spikes during episodes (toxicity attracts volume)
base_vol = rng.lognormal(mean=np.log(1000), sigma=0.5, size=T)
vol_mult = 1 + 2.5 * np.abs(drift) / 0.45
volume = base_vol * vol_mult

# price: impact of imbalance + noise; crash materializes near end of episode 3
ret = 0.00004 * imb * (volume / 1000) ** 0.5 + rng.normal(0, 0.0006, T)
# add crash: sharp drop in the last day of episode 3
c_s, c_e, _ = episodes[2]
crash_start = c_e - int(0.5 * mins_per_day)
ret[crash_start:crash_start + 30] -= 0.0035  # ~10% drop in 30 min
price = 100 * np.exp(np.cumsum(ret))

buy_vol = volume * (1 + imb) / 2
sell_vol = volume * (1 - imb) / 2

# ---------- VPIN: volume buckets ----------
V_bucket = volume.sum() / (n_days * 50)  # ~50 buckets per day
cum_v = np.cumsum(volume)
bucket_id = (cum_v // V_bucket).astype(int)
n_buckets = bucket_id.max() + 1
bb = np.bincount(bucket_id, weights=buy_vol, minlength=n_buckets)
sb = np.bincount(bucket_id, weights=sell_vol, minlength=n_buckets)
bucket_endtime = np.zeros(n_buckets, dtype=int)
np.maximum.at(bucket_endtime, bucket_id, t)

W = 50  # rolling window in buckets (~1 day)
oi = np.abs(bb - sb)
tot = bb + sb
vpin = np.convolve(oi, np.ones(W), "full")[:n_buckets] / np.convolve(tot, np.ones(W), "full")[:n_buckets]
vpin[:W] = np.nan

# CDF-transform (rolling percentile over trailing 5000 buckets)
from scipy.stats import rankdata
vpin_cdf = np.full(n_buckets, np.nan)
look = 5000
for i in range(W + 200, n_buckets):
    lo = max(W, i - look)
    window = vpin[lo:i+1]
    vpin_cdf[i] = (window < vpin[i]).mean()

# map buckets to time for plotting
bt = bucket_endtime / mins_per_day  # in days
pt = t / mins_per_day

# ---------- fig1: price + VPIN over full sample ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
axes[0].plot(pt[::30], price[::30], lw=0.9, c="#1f77b4")
for s, e, lvl in episodes:
    axes[0].axvspan(s/mins_per_day, e/mins_per_day, color=("#d62728" if lvl < 0 else "#2ca02c"), alpha=0.10)
axes[0].axvline(crash_start/mins_per_day, color="#d62728", ls="--", lw=1, label="闪崩起点")
axes[0].set_ylabel("价格"); axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].set_title("合成 tick 流：价格、毒性事件区间（阴影）与 VPIN")
axes[1].plot(bt, vpin, lw=0.8, c="#ff7f0e")
axes[1].axhline(np.nanquantile(vpin, 0.95), color="k", ls=":", lw=1, label="95% 分位")
axes[1].set_ylabel("VPIN"); axes[1].set_xlabel("交易日")
axes[1].legend(); axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/price-vpin.png", dpi=110); plt.close(fig)

# ---------- fig2: zoom on crash: VPIN rises before crash ----------
z0 = crash_start - 6*mins_per_day
z1 = crash_start + 2*mins_per_day
mask_p = (t >= z0) & (t <= z1)
mask_b = (bucket_endtime >= z0) & (bucket_endtime <= z1)
fig, ax1 = plt.subplots(figsize=(9.5, 5.5))
ax1.plot(pt[mask_p], price[mask_p], lw=1.0, c="#1f77b4", label="价格")
ax1.axvline(crash_start/mins_per_day, color="#d62728", ls="--", lw=1.2, label="闪崩起点")
ax1.set_xlabel("交易日"); ax1.set_ylabel("价格", color="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(bt[mask_b], vpin_cdf[mask_b], lw=1.4, c="#ff7f0e", label="VPIN CDF")
ax2.axhline(0.9, color="k", ls=":", lw=1)
ax2.set_ylabel("VPIN CDF（滚动分位）", color="#ff7f0e")
ax2.set_ylim(0, 1.05)
# first time cdf > 0.9 in the zoom window
idx = np.where(mask_b & (vpin_cdf > 0.9))[0]
lead = None
if len(idx):
    first_warn = bucket_endtime[idx[0]]
    lead = (crash_start - first_warn) / mins_per_day
    ax1.axvline(first_warn/mins_per_day, color="#2ca02c", ls="--", lw=1.2, label=f"VPIN 预警（提前 {lead:.1f} 天）")
ax1.legend(loc="lower left")
ax1.set_title("闪崩前后放大：VPIN CDF 在暴跌前越过 0.9 预警线")
ax1.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/crash-zoom.png", dpi=110); plt.close(fig)
print("lead days:", lead)

# ---------- fig3: event study - future realized vol / drawdown conditioned on VPIN CDF ----------
# for each bucket, compute forward 1-day max drawdown and realized vol of price
fwd_min = 1 * mins_per_day
fdd = np.full(n_buckets, np.nan)
fvol = np.full(n_buckets, np.nan)
for i in range(W + 200, n_buckets):
    s = bucket_endtime[i]
    e = min(s + fwd_min, T)
    if e - s < 60: continue
    p = price[s:e]
    run_max = np.maximum.accumulate(p)
    fdd[i] = ((p / run_max) - 1).min()
    fvol[i] = np.std(np.diff(np.log(p))) * np.sqrt(240 * 252)
valid = ~np.isnan(vpin_cdf) & ~np.isnan(fdd)
bins = [0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0001]
labels_b = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-0.95", ">0.95"]
mean_dd, mean_vol, cnts = [], [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    m = valid & (vpin_cdf >= lo) & (vpin_cdf < hi)
    mean_dd.append(np.nanmean(fdd[m]) * 100)
    mean_vol.append(np.nanmean(fvol[m]) * 100)
    cnts.append(m.sum())
print("bucket counts:", cnts)
print("mean fwd 1d maxDD (%):", np.round(mean_dd, 3))
print("mean fwd vol (%):", np.round(mean_vol, 1))

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
axes[0].bar(labels_b, mean_dd, color=["#aec7e8"]*4 + ["#ff9896", "#d62728"])
axes[0].set_ylabel("未来 1 日最大回撤（%）")
axes[0].set_xlabel("VPIN CDF 区间")
axes[0].set_title("VPIN 越高，未来 1 日回撤越深")
axes[0].grid(alpha=0.3, axis="y")
axes[1].bar(labels_b, mean_vol, color=["#aec7e8"]*4 + ["#ff9896", "#d62728"])
axes[1].set_ylabel("未来 1 日年化波动（%)")
axes[1].set_xlabel("VPIN CDF 区间")
axes[1].set_title("VPIN 与未来已实现波动")
axes[1].grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(f"{OUT}/event-study.png", dpi=110); plt.close(fig)

# ---------- stats: hedging strategy: cut exposure when vpin_cdf > 0.9 ----------
# daily returns of buy&hold vs vpin-filtered
day_ret = np.zeros(n_days)
for d in range(n_days):
    s, e = d*mins_per_day, (d+1)*mins_per_day
    day_ret[d] = price[e-1]/price[s] - 1 if d > 0 else price[e-1]/price[0] - 1
# signal at day level: max vpin_cdf during previous day
sig = np.zeros(n_days)
bd = bucket_endtime // mins_per_day
for d in range(1, n_days):
    m = bd == (d-1)
    if m.any():
        v = vpin_cdf[m]
        sig[d] = np.nanmax(v) if not np.all(np.isnan(v)) else 0
pos = (sig <= 0.9).astype(float)
strat = day_ret * pos
bh_nav = np.cumprod(1 + day_ret)
st_nav = np.cumprod(1 + strat)
def mdd(nav):
    rm = np.maximum.accumulate(nav)
    return ((nav/rm) - 1).min()
def sharpe(r):
    return r.mean()/r.std(ddof=1)*np.sqrt(252)
print(f"B&H: ann={(bh_nav[-1])**(252/n_days)-1:.3%}, sharpe={sharpe(day_ret):.2f}, mdd={mdd(bh_nav):.2%}")
print(f"VPIN-filter: ann={(st_nav[-1])**(252/n_days)-1:.3%}, sharpe={sharpe(strat):.2f}, mdd={mdd(st_nav):.2%}, days out={100*(1-pos.mean()):.1f}%")

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(np.arange(n_days), bh_nav, lw=1.3, label=f"买入持有（MDD {mdd(bh_nav):.1%}）")
ax.plot(np.arange(n_days), st_nav, lw=1.3, label=f"VPIN>0.9 减仓（MDD {mdd(st_nav):.1%}）")
ax.set_xlabel("交易日"); ax.set_ylabel("累计净值")
ax.set_title("VPIN 过滤 vs 买入持有")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/vpin-filter-nav.png", dpi=110); plt.close(fig)
