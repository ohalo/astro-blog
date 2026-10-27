#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""回购利率与股指期货基差：资金成本定价升贴水的模拟研究"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json, os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/repo-rate-index-futures-basis"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 模拟市场环境：8 年日频 ----------
T = 8 * 252
dates = pd.bdate_range("2018-01-02", periods=T)

# 指数现货：几何布朗 + 机制切换波动
mu_spot, base_vol = 0.06, 0.18
vol_regime = np.ones(T) * base_vol
# 两段危机：高波动
crisis1 = slice(int(T*0.28), int(T*0.31))
crisis2 = slice(int(T*0.68), int(T*0.72))
vol_regime[crisis1] = 0.45
vol_regime[crisis2] = 0.38
spot_ret = mu_spot/252 + vol_regime/np.sqrt(252) * rng.standard_normal(T)
spot = 4000 * np.exp(np.cumsum(spot_ret))

# 回购利率（7天回购，年化）：均值回复 + 季末/年末冲高 + 危机期资金紧张
r_mean = 0.022
r = np.zeros(T); r[0] = r_mean
kappa, sigma_r = 0.05, 0.0018
for t in range(1, T):
    r[t] = r[t-1] + kappa*(r_mean - r[t-1]) + sigma_r*rng.standard_normal()
# 季末冲高：每季度最后 5 个交易日 +80~200bp 尖峰
month = dates.month.values; day_in_q = np.zeros(T, bool)
for i in range(T-1):
    if dates[i].quarter != dates[i+1].quarter:
        day_in_q[max(0,i-4):i+1] = True
spike = day_in_q * (0.008 + 0.012*rng.random(T))
r_repo = np.clip(r + spike, 0.005, None)
# 危机期资金紧张
r_repo[crisis1] += 0.010
r_repo[crisis2] += 0.006

# 分红率：季节性（A股集中在 5-8 月除息），年化口径
div_annual = 0.024
div_season = np.where(np.isin(month, [5,6,7,8]), 2.4, 0.35)
div_season = div_season / div_season.mean() * div_annual  # 年化瞬时分红率

# 期货合约：滚动当季合约，到期日为每季第三个周五附近，用固定剩余期限近似再精确化
# 构造连续的"剩余到期天数"序列（当季合约，到期滚到下一季）
ttm = np.zeros(T)
expiry_idx = []
i = 0
while i < T:
    j = min(i + 63, T)  # 一个季度约 63 个交易日
    expiry_idx.append(j-1)
    ttm[i:j] = (np.arange(j-1, i-1, -1) - 0) [::-1]
    ttm[i:j] = (j-1) - np.arange(i, j)
    i = j
ttm_years = ttm / 252

# 理论基差（持有成本模型）: F = S * exp((r - d) * ttm) => basis = F - S
# 预期分红：到期前的累计分红率（用季节性分红积分近似）
exp_div = np.zeros(T)
for t in range(T):
    horizon = int(ttm[t])
    if horizon > 0:
        exp_div[t] = div_season[t:t+horizon].mean() if t+horizon <= T else div_season[t:].mean()
    else:
        exp_div[t] = div_season[t]
fair_basis = spot * (np.exp((r_repo - exp_div) * ttm_years) - 1)

# 市场基差 = 理论基差 + 噪音 + 情绪项（危机期贴水加深：对冲需求压价）
sentiment = np.zeros(T)
sentiment[crisis1] = -0.012; sentiment[crisis2] = -0.008
# 平滑情绪
sent_s = pd.Series(sentiment).rolling(10, min_periods=1).mean().values
noise = 0.0015 * rng.standard_normal(T)
noise = pd.Series(noise).rolling(3, min_periods=1).mean().values
mkt_basis = fair_basis + spot * (sent_s + noise) * np.sqrt(np.maximum(ttm_years, 1/252))
futures = spot + mkt_basis

# 年化基差率（消除到期日效应）
with np.errstate(divide="ignore", invalid="ignore"):
    ann_basis = np.where(ttm_years > 2/252, (futures/spot - 1) / ttm_years, np.nan)
ann_fair = np.where(ttm_years > 2/252, (fair_basis/spot) / ttm_years, np.nan)

df = pd.DataFrame({"spot": spot, "fut": futures, "repo": r_repo, "divr": exp_div,
                   "ttm": ttm, "ann_basis": ann_basis, "ann_fair": ann_fair}, index=dates)
df["carry"] = df.repo - df.divr  # 理论年化基差 ≈ r - d

# ---------- 统计：基差与资金成本的关系 ----------
valid = df.dropna(subset=["ann_basis"])
corr_all = np.corrcoef(valid.ann_basis, valid.carry)[0,1]
# 排除危机期
mask_c = np.ones(T, bool); mask_c[crisis1] = False; mask_c[crisis2] = False
valid_nc = df[mask_c].dropna(subset=["ann_basis"])
corr_nc = np.corrcoef(valid_nc.ann_basis, valid_nc.carry)[0,1]
# 回归斜率
beta_all = np.polyfit(valid.carry, valid.ann_basis, 1)
beta_nc = np.polyfit(valid_nc.carry, valid_nc.ann_basis, 1)

# ---------- 2. 期现套利带 ----------
# 成本：现货冲击+佣金 单边 15bp，期货 2bp，融资即回购利率（正向）；反向需融券，费率 8%/年
c_spot, c_fut = 0.0015, 0.0002
short_fee = 0.08
upper_band = (df.repo - df.divr) + (c_spot + c_fut) * 2 / np.maximum(ttm_years, 5/252)  # 年化
lower_band = (df.repo - df.divr - short_fee) - (c_spot + c_fut) * 2 / np.maximum(ttm_years, 5/252)
df["upper"] = upper_band; df["lower"] = lower_band
viol_up = (df.ann_basis > df.upper).mean()
viol_dn = (df.ann_basis < df.lower).mean()

# ---------- 3. 基差动量/贴水收割策略：贴水深时做多期货（收基差回归） ----------
# 信号：年化基差偏离其理论值的 z 分数（60日窗）
dev = (df.ann_basis - df.ann_fair).ffill()
z = (dev - dev.rolling(120, min_periods=60).mean()) / dev.rolling(120, min_periods=60).std()
# 信号改用绝对偏离：年化基差比理论锚低 2% 以上开多（收敛到期收贴水），回到 -0.5% 内平仓
zv = np.where(np.isnan(dev.values), np.nan, np.where(dev.values < -0.02, -2.0, np.where(dev.values > -0.005, 0.0, np.nan)))
zv = pd.Series(zv, index=dates).ffill().values  # 中间地带保持原状态
pos_raw = np.zeros(T)
state = 0.0
for t in range(T):
    if np.isnan(zv[t]):
        pos_raw[t] = state; continue
    if state == 0.0 and zv[t] < -1.5:
        state = 1.0
    elif state > 0 and zv[t] >= 0.0:
        state = 0.0
    pos_raw[t] = state
pos = pd.Series(pos_raw, index=dates).shift(1).fillna(0)  # signal on t, trade t+1
# 对冲版：多期货空现货（纯基差收敛）。PnL = 基差点数变化 / 现货，
# 但合约换月日基差从 ~0 跳到新合约水平，属账面跳变而非损益，必须剔除
b_pts = pd.Series(mkt_basis, index=dates)
roll_day = pd.Series(np.r_[False, np.diff(ttm) > 0], index=dates)  # ttm 变大 = 换月
b_chg = b_pts.diff().where(~roll_day, 0.0).fillna(0)
basis_pnl = pos * b_chg / pd.Series(spot, index=dates).shift(1)
tc = pos.diff().abs().fillna(0) * (c_fut + 0.0003)
basis_pnl_net = basis_pnl - tc
eq = (1 + basis_pnl_net).cumprod()
ann_ret = eq.iloc[-1] ** (252/len(eq)) - 1
ann_vol = basis_pnl_net.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
mdd = (eq / eq.cummax() - 1).min()
n_trades = int((pos.diff().abs() > 0).sum())

# 空仓持有贴水收割（被动 long futures vs long spot 的年化差）
avg_ann_basis = np.nanmean(df.ann_basis)
avg_carry = df.carry.mean()
crisis_basis = np.nanmean(df.ann_basis.values[~mask_c])
normal_basis = np.nanmean(df.ann_basis.values[mask_c])

# ---------- 4. 图 ----------
# 图1 cover: 现货/期货 + 年化基差率
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                         gridspec_kw={"height_ratios": [2, 1.4]})
axes[0].plot(dates, spot, lw=1, label="指数现货", color="#1f77b4")
axes[0].plot(dates, futures, lw=0.7, alpha=0.7, label="当季期货", color="#d62728")
axes[0].set_ylabel("点位"); axes[0].legend(loc="upper left"); axes[0].set_title("指数现货与当季期货（模拟 8 年）")
axes[1].plot(dates, df.ann_basis*100, lw=0.7, color="#2ca02c", label="年化基差率(市场)")
axes[1].plot(dates, df.carry*100, lw=1.2, color="#ff7f0e", label="回购利率 − 分红率(理论锚)")
axes[1].axhline(0, color="gray", lw=0.6)
for cs in (crisis1, crisis2):
    axes[1].axvspan(dates[cs.start], dates[cs.stop-1], color="red", alpha=0.10)
axes[1].set_ylabel("%"); axes[1].legend(loc="lower left", ncol=2)
axes[1].set_title("年化基差率 vs 资金成本锚（红色区间为危机期）")
plt.tight_layout(); plt.savefig(f"{OUT}/cover.png", dpi=110); plt.close()

# 图2 散点：carry vs ann_basis
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].scatter(valid.carry*100, valid.ann_basis*100, s=4, alpha=0.3, color="#1f77b4")
xs = np.linspace(valid.carry.min(), valid.carry.max(), 50)
axes[0].plot(xs*100, np.polyval(beta_all, xs)*100, color="#d62728", lw=2,
             label=f"斜率 {beta_all[0]:.2f}, ρ={corr_all:.3f}")
axes[0].set_xlabel("回购利率 − 分红率 (%)"); axes[0].set_ylabel("年化基差率 (%)")
axes[0].set_title("全样本：基差 vs 资金成本"); axes[0].legend()
axes[1].scatter(valid_nc.carry*100, valid_nc.ann_basis*100, s=4, alpha=0.3, color="#2ca02c")
axes[1].plot(xs*100, np.polyval(beta_nc, xs)*100, color="#d62728", lw=2,
             label=f"斜率 {beta_nc[0]:.2f}, ρ={corr_nc:.3f}")
axes[1].set_xlabel("回购利率 − 分红率 (%)"); axes[1].set_ylabel("年化基差率 (%)")
axes[1].set_title("剔除危机期后"); axes[1].legend()
plt.tight_layout(); plt.savefig(f"{OUT}/scatter_carry.png", dpi=110); plt.close()

# 图3 套利带
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(dates, df.ann_basis*100, lw=0.6, color="#1f77b4", label="年化基差率")
ax.plot(dates, df.upper*100, lw=1, color="#d62728", ls="--", label="正向套利上界")
ax.plot(dates, df.lower*100, lw=1, color="#2ca02c", ls="--", label="反向套利下界(含融券费)")
ax.fill_between(dates, df.lower*100, df.upper*100, color="gray", alpha=0.08)
ax.set_ylabel("%"); ax.legend(loc="lower left", ncol=3)
ax.set_title(f"期现套利无套利带：上破占比 {viol_up:.1%}，下破占比 {viol_dn:.1%}")
plt.tight_layout(); plt.savefig(f"{OUT}/arb_band.png", dpi=110); plt.close()

# 图4 策略
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
axes[0].plot(eq.index, eq.values, lw=1.2, color="#1f77b4")
axes[0].set_title(f"基差回归策略净值（对冲版）\nSharpe {sharpe:.2f}，年化 {ann_ret:.1%}，最大回撤 {mdd:.1%}")
axes[1].hist(dev.dropna()*100, bins=60, color="#ff7f0e", alpha=0.8)
axes[1].axvline(0, color="k", lw=0.8)
axes[1].set_xlabel("年化基差偏离理论值 (%)"); axes[1].set_title("基差对理论锚的偏离分布")
plt.tight_layout(); plt.savefig(f"{OUT}/strategy.png", dpi=110); plt.close()

# 图5 分红季节性对期限结构的影响
fig, ax = plt.subplots(figsize=(11, 4.2))
monthly = pd.DataFrame({"m": month, "b": df.ann_basis.values, "d": div_season}).groupby("m").mean()
ax2 = ax.twinx()
ax.bar(monthly.index - 0.18, monthly.b*100, width=0.36, color="#1f77b4", label="平均年化基差率")
ax2.bar(monthly.index + 0.18, monthly.d*100, width=0.36, color="#ff7f0e", label="平均年化分红率")
ax.set_xlabel("月份"); ax.set_ylabel("基差率 %", color="#1f77b4"); ax2.set_ylabel("分红率 %", color="#ff7f0e")
ax.set_title("分红季节性：5-8 月除息高峰压深基差")
ax.set_xticks(range(1,13))
lines1, lab1 = ax.get_legend_handles_labels(); lines2, lab2 = ax2.get_legend_handles_labels()
ax.legend(lines1+lines2, lab1+lab2, loc="lower left")
plt.tight_layout(); plt.savefig(f"{OUT}/dividend_season.png", dpi=110); plt.close()

stats = {
    "corr_all": round(float(corr_all), 3), "corr_nc": round(float(corr_nc), 3),
    "beta_all": round(float(beta_all[0]), 3), "beta_nc": round(float(beta_nc[0]), 3),
    "viol_up": round(float(viol_up), 4), "viol_dn": round(float(viol_dn), 4),
    "avg_ann_basis": round(float(avg_ann_basis), 4), "avg_carry": round(float(avg_carry), 4),
    "crisis_basis": round(float(crisis_basis), 4), "normal_basis": round(float(normal_basis), 4),
    "sharpe": round(float(sharpe), 3), "ann_ret": round(float(ann_ret), 4),
    "ann_vol": round(float(ann_vol), 4), "mdd": round(float(mdd), 4), "n_trades": n_trades,
    "avg_repo": round(float(df.repo.mean()), 4), "avg_div": round(float(df.divr.mean()), 4),
    "spike_repo_max": round(float(df.repo.max()), 4),
}
with open(f"{OUT}/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
