#!/usr/bin/env python3
"""生成 volume-clock-sampling 与 dollar-bars-information-sampling 两篇文章的配图
核心模拟设计：subordination（Clark 1973）——每笔交易收益 iid 正态，
日历时间下的波动聚集与肥尾完全来自交易活动强度的时变与持续性。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

ROOT = Path("/Users/halo/workspace/astro-blog/public/images")
D1 = ROOT / "volume-clock-sampling"
D2 = ROOT / "dollar-bars-information-sampling"
D1.mkdir(parents=True, exist_ok=True)
D2.mkdir(parents=True, exist_ok=True)

# ============================================================
# 模拟：120 个交易日
# - 日级活动强度 log-AR(1)（持续性 → 日历时间下的波动聚集）
# - 日内 U 型 + 随机新闻爆发
# - 每笔收益 iid N(0, s²) —— 交易时钟下收益天然正态
# ============================================================
def simulate(n_days=120, base_ticks=1200, seed=42):
    rng = np.random.default_rng(seed)
    all_p, all_v, all_t, all_day = [], [], [], []
    price = 100.0
    log_act = 0.0
    s = 0.0009  # 每笔收益标准差
    for d in range(n_days):
        log_act = 0.92 * log_act + rng.normal(0, 0.35)  # 持续性活动区制
        day_mult = np.exp(log_act)
        grid = np.linspace(0, 1, 241)
        u_shape = np.clip(1.6 - 4.8 * grid * (1 - grid), 0.35, None)
        burst = np.zeros_like(grid)
        if rng.random() < 0.3:
            c = rng.uniform(0.15, 0.85)
            burst = 4.0 * np.exp(-((grid - c) ** 2) / (2 * 0.02 ** 2))
        lam = (u_shape + burst)
        lam = lam / lam.mean() * day_mult
        ticks_per_min = rng.poisson(lam * base_ticks / 241)
        for m, k in enumerate(ticks_per_min):
            if k == 0:
                continue
            r = rng.normal(0, s, k)
            for rr in r:
                price *= (1 + rr)
                all_p.append(price)
                all_v.append(max(int(rng.lognormal(4.5, 1.0)), 1))
                all_t.append(d + m / 241)
                all_day.append(d)
    return (np.array(all_t), np.array(all_p), np.array(all_v),
            np.array(all_day, dtype=int))

t, p, v, day = simulate()
n_days = 120
print(f"ticks: {len(p)}")

def time_bar_closes(t, p, per_day, n_days):
    edges = np.linspace(0, n_days, n_days * per_day + 1)[1:]
    idx = np.clip(np.searchsorted(t, edges, side="right") - 1, 0, len(p) - 1)
    idx = np.unique(idx)
    return p[idx]

def thr_bar_closes(p, metric, thr, t=None):
    closes, times = [], []
    cum = 0.0
    for i in range(len(p)):
        cum += metric[i]
        if cum >= thr:
            closes.append(p[i])
            if t is not None:
                times.append(t[i])
            cum = 0.0
    return (np.array(closes), np.array(times)) if t is not None else np.array(closes)

PER_DAY = 8
N_BARS = n_days * PER_DAY
tb_close = time_bar_closes(t, p, PER_DAY, n_days)
vb_close, vb_time = thr_bar_closes(p, v.astype(float), v.sum() / N_BARS, t)
tk_close = thr_bar_closes(p, np.ones(len(p)), len(p) / N_BARS)
db_close = thr_bar_closes(p, v * p, (v * p).sum() / N_BARS)

tb_ret = np.diff(np.log(tb_close))
vb_ret = np.diff(np.log(vb_close))
tk_ret = np.diff(np.log(tk_close))
db_ret = np.diff(np.log(db_close))
print(f"bars: time={len(tb_ret)} vol={len(vb_ret)} tick={len(tk_ret)} dollar={len(db_ret)}")

def acf(x, nlags=20):
    x = x - x.mean()
    d = np.sum(x ** 2)
    return np.array([1.0 if k == 0 else np.sum(x[:-k] * x[k:]) / d
                     for k in range(nlags + 1)])

# ---------- 图 1-1: 日内活动 + 两种时钟的采样点 ----------
d_sel = None
for dd in range(n_days):
    mask = day == dd
    if mask.sum() < 500:
        continue
    per_min_vol, _ = np.histogram(t[mask], bins=60, weights=v[mask])
    med = np.median(per_min_vol[per_min_vol > 0])
    if med > 0 and per_min_vol.max() > 5 * med:
        d_sel = dd
        break
if d_sel is None:
    d_sel = 5
mask = day == d_sel
tt = (t[mask] - d_sel) * 4 + 9.5
fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1.6, 1]})
bins = np.linspace(9.5, 13.5, 61)
vol_hist, edges_ = np.histogram(tt, bins=bins, weights=v[mask])
axes[0].bar((edges_[:-1] + edges_[1:]) / 2, vol_hist / 1e3,
            width=np.diff(edges_) * 0.9, color="#4C72B0", alpha=0.8)
axes[0].set_ylabel("每 4 分钟成交量（千股）")
axes[0].set_title("某交易日的日内活动：U 型形态 + 一次新闻爆发", fontsize=12)

tb_day = np.linspace(9.5, 13.5, PER_DAY + 1)[1:]
vb_mask = (vb_time >= d_sel) & (vb_time < d_sel + 1)
vb_day = (vb_time[vb_mask] - d_sel) * 4 + 9.5
axes[1].scatter(tb_day, np.ones_like(tb_day), marker="|", s=600, color="#C44E52",
                label=f"时间 Bar 收盘点（{len(tb_day)} 根，等间隔）")
axes[1].scatter(vb_day, np.ones_like(vb_day) * 0.4, marker="|", s=600, color="#55A868",
                label=f"成交量 Bar 收盘点（{len(vb_day)} 根，追着活动走）")
axes[1].set_ylim(0, 1.5)
axes[1].set_yticks([])
axes[1].set_xlabel("日内时间")
axes[1].legend(loc="upper left", fontsize=9)
axes[1].set_title("同一天：时间时钟均匀切分，成交量时钟在爆发期自动加密采样", fontsize=11)
plt.tight_layout()
plt.savefig(D1 / "vc-intraday-clock.png", bbox_inches="tight")
plt.close()

# ---------- 图 1-2: 收益分布对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
stats_txt = {}
for ax, r, name, c in [(axes[0], tb_ret, "时间 Bar", "#C44E52"),
                        (axes[1], vb_ret, "成交量 Bar", "#55A868")]:
    r_std = (r - r.mean()) / r.std()
    ax.hist(r_std, bins=61, density=True, color=c, alpha=0.75,
            edgecolor="white", linewidth=0.3)
    x = np.linspace(-6, 6, 300)
    ax.plot(x, stats.norm.pdf(x), "k--", lw=1.4, label="标准正态")
    k = stats.kurtosis(r)
    jb = stats.jarque_bera(r).statistic
    stats_txt[name] = (k, jb)
    ax.set_title(f"{name} 收益分布\n超额峰度 {k:.2f} ｜ JB 统计量 {jb:.0f}", fontsize=11)
    ax.set_xlim(-6, 6)
    ax.set_yscale("log")
    ax.set_ylim(1e-4, 1)
    ax.legend(fontsize=9)
plt.suptitle("同一段逐笔数据、相同 bar 数量（对数纵轴看尾部）：成交量时钟下收益接近正态",
             y=1.02, fontsize=12)
plt.tight_layout()
plt.savefig(D1 / "vc-return-distribution.png", bbox_inches="tight")
plt.close()
print("fig1-2 stats:", {k: (round(a, 2), round(b)) for k, (a, b) in stats_txt.items()})

# ---------- 图 1-3: 平方收益自相关 ----------
acf_tb = acf(tb_ret ** 2, 20)
acf_vb = acf(vb_ret ** 2, 20)
fig, ax = plt.subplots(figsize=(9, 4.2))
w = 0.38
lags = np.arange(1, 21)
ax.bar(lags - w / 2, acf_tb[1:], width=w, color="#C44E52", label="时间 Bar 平方收益 ACF")
ax.bar(lags + w / 2, acf_vb[1:], width=w, color="#55A868", label="成交量 Bar 平方收益 ACF")
ci = 1.96 / np.sqrt(len(tb_ret))
ax.axhline(ci, color="gray", ls=":", lw=1)
ax.axhline(-ci, color="gray", ls=":", lw=1, label="95% 置信带")
ax.set_xlabel("滞后阶数")
ax.set_ylabel("自相关")
ax.set_title("平方收益自相关：时间时钟下显著的波动聚集，在成交量时钟下基本消失", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(D1 / "vc-acf-squared.png", bbox_inches="tight")
plt.close()
print(f"acf1: tb={acf_tb[1]:.3f} vb={acf_vb[1]:.3f}; acf mean(1-10): tb={acf_tb[1:11].mean():.3f} vb={acf_vb[1:11].mean():.3f}")

# ============================================================
# 文章 2 图
# ============================================================
# ---------- 图 2-2: 四种 bar 的收益统计（同一平稳模拟） ----------
names = ["时间 Bar", "Tick Bar", "成交量 Bar", "美元 Bar"]
rets = [tb_ret, tk_ret, vb_ret, db_ret]
colors = ["#C44E52", "#8172B2", "#55A868", "#4C72B0"]
kurts = [stats.kurtosis(r) for r in rets]
jbs = [stats.jarque_bera(r).statistic for r in rets]
acf1s = [acf(r ** 2, 10)[1:11].mean() for r in rets]

fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.0))
x = np.arange(4)
axes[0].bar(x, kurts, color=colors)
axes[0].set_title("超额峰度（越低越接近正态）", fontsize=11)
axes[1].bar(x, jbs, color=colors)
axes[1].set_yscale("log")
axes[1].set_title("Jarque-Bera 统计量（对数轴）", fontsize=11)
axes[2].bar(x, acf1s, color=colors)
axes[2].axhline(1.96 / np.sqrt(len(tb_ret)), color="gray", ls=":", lw=1)
axes[2].set_title("平方收益 ACF（1-10 阶均值）\n（波动聚集残留）", fontsize=11)
for ax in axes:
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9, rotation=15)
plt.suptitle("四种采样时钟下的收益统计性质（同一逐笔序列、相同 bar 总数）", y=1.03, fontsize=12)
plt.tight_layout()
plt.savefig(D2 / "db-stat-compare.png", bbox_inches="tight")
plt.close()
print(f"kurts: {dict(zip(names, [round(k, 2) for k in kurts]))}")
print(f"jbs: {dict(zip(names, [round(j) for j in jbs]))}")
print(f"acf1-10: {dict(zip(names, [round(a, 3) for a in acf1s]))}")

# ---------- 长期模拟：价格 ~3 倍 + 活动增长（阈值漂移演示） ----------
def simulate_long(n_days=1000, seed=7):
    rng = np.random.default_rng(seed)
    prices, vols, days = [], [], []
    price = 20.0
    for d in range(n_days):
        n_ticks = rng.poisson(300 * (1 + 1.5 * d / n_days))
        drift = 0.0011
        vol_regime = 2.2 if 420 <= d < 540 else 1.0
        sigma = 0.016 * vol_regime / np.sqrt(max(n_ticks, 1))
        r = rng.normal(drift / max(n_ticks, 1), sigma, n_ticks)
        for rr in r:
            price *= (1 + rr)
            prices.append(price)
            vols.append(max(int(rng.lognormal(4.0, 1.0)), 1))
            days.append(d)
    return np.array(days, dtype=int), np.array(prices), np.array(vols)

ld, lp, lv = simulate_long()
print(f"long sim: ticks={len(lp)}, price {lp[0]:.1f} -> {lp[-1]:.1f}")

def count_bars_per_day(ld, metric, thr, n_days=1000):
    counts = np.zeros(n_days)
    cum = 0.0
    for i in range(len(metric)):
        cum += metric[i]
        if cum >= thr:
            counts[ld[i]] += 1
            cum = 0.0
    return counts

cal = ld < 100
tick_thr = cal.sum() / (100 * 6)
vol_thr = lv[cal].sum() / (100 * 6)
dol_thr = (lv[cal] * lp[cal]).sum() / (100 * 6)
cnt_tick = count_bars_per_day(ld, np.ones(len(lp)), tick_thr)
cnt_vol = count_bars_per_day(ld, lv.astype(float), vol_thr)
cnt_dol = count_bars_per_day(ld, lv * lp, dol_thr)

def smooth(x, w=20):
    return np.convolve(x, np.ones(w) / w, mode="valid")

# ---------- 图 2-1: 固定阈值下每日 bar 数漂移 ----------
fig, ax = plt.subplots(figsize=(9.5, 4.6))
ax.plot(smooth(cnt_tick), color="#8172B2", lw=1.6, label="Tick Bar（每 N 笔）")
ax.plot(smooth(cnt_vol), color="#55A868", lw=1.6, label="成交量 Bar（每 N 股）")
ax.plot(smooth(cnt_dol), color="#C44E52", lw=1.8, label="美元 Bar（每 N 元成交额）")
ax.axhline(6, color="gray", ls=":", lw=1, label="校准目标：每天 6 根")
ax.axvspan(420, 540, color="orange", alpha=0.12)
ax.set_xlabel("交易日")
ax.set_ylabel("每日 bar 数（20 日平滑）")
ax.set_title("价格 ~3 倍上涨 + 活动增长的市场：固定阈值下每日 bar 数的漂移", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(D2 / "db-bars-per-day.png", bbox_inches="tight")
plt.close()
print(f"末期日均 bar 数: tick={cnt_tick[-100:].mean():.1f} vol={cnt_vol[-100:].mean():.1f} dollar={cnt_dol[-100:].mean():.1f}")

# ---------- 图 2-3: 价格路径 + 美元 bar 采样密度（EWMA 动态阈值 vs 固定） ----------
day_close = np.array([lp[ld == dd][-1] for dd in range(1000)])
daily_dollar = np.array([np.sum(lv[ld == dd] * lp[ld == dd]) for dd in range(1000)])

# EWMA 动态阈值：目标每天 6 根，阈值 = EWMA(日成交额)/6
ewma = np.zeros(1000)
ewma[0] = daily_dollar[0]
alpha = 2 / (50 + 1)
for i in range(1, 1000):
    ewma[i] = alpha * daily_dollar[i - 1] + (1 - alpha) * ewma[i - 1]

cnt_dol_dyn = np.zeros(1000)
cum = 0.0
for i in range(len(lp)):
    dthr = ewma[ld[i]] / 6
    cum += lv[i] * lp[i]
    if cum >= dthr:
        cnt_dol_dyn[ld[i]] += 1
        cum = 0.0

fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.2), sharex=True,
                         gridspec_kw={"height_ratios": [1.4, 1]})
axes[0].plot(day_close, color="#4C72B0", lw=1.2)
axes[0].axvspan(420, 540, color="orange", alpha=0.15, label="高波动区制")
axes[0].set_ylabel("价格")
axes[0].set_title("模拟市场：4 年价格约 3 倍，中段一次高波动区制", fontsize=12)
axes[0].legend(fontsize=9)

axes[1].plot(smooth(cnt_dol), color="#C44E52", lw=1.4, alpha=0.8,
             label="固定阈值美元 Bar：随价格上涨持续加密（漂移）")
axes[1].plot(smooth(cnt_dol_dyn), color="#4C72B0", lw=1.6,
             label="EWMA 动态阈值美元 Bar：日常稳定，高波动期自动加密")
axes[1].axhline(6, color="gray", ls=":", lw=1)
axes[1].axvspan(420, 540, color="orange", alpha=0.15)
axes[1].set_xlabel("交易日")
axes[1].set_ylabel("每日 bar 数（20 日平滑）")
axes[1].legend(fontsize=9)
axes[1].set_title("动态阈值把趋势性漂移滤掉，只保留「信息爆发时加密采样」这个想要的性质", fontsize=11)
plt.tight_layout()
plt.savefig(D2 / "db-price-density.png", bbox_inches="tight")
plt.close()
print(f"动态阈值末期日均 bar 数: {cnt_dol_dyn[-100:].mean():.1f}, 高波动段: {cnt_dol_dyn[420:540].mean():.1f}, 平稳段: {cnt_dol_dyn[100:400].mean():.1f}")
print("ALL DONE")
