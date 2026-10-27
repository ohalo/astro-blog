#!/usr/bin/env python3
"""Dollar Bars 实验：时间棒 vs 成交量棒 vs 美元棒的统计性质对比"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)
OUT = "/Users/halo/workspace/astro-blog/public/images/dollar-bars-sampling"
os.makedirs(OUT, exist_ok=True)

# ---------- 1. 合成 tick 级市场：活跃度成簇（Hawkes 式自激强度） ----------
# 模拟 60 个交易日，每日 240 分钟。分钟级"信息到达强度"自激。
DAYS, MIN_PER_DAY = 60, 240
T = DAYS * MIN_PER_DAY
lam = np.zeros(T)
base, alpha, decay = 0.3, 0.55, 0.08
excite = 0.0
n_events = np.zeros(T, dtype=int)
for t in range(T):
    # 日内 U 型季节性
    tod = t % MIN_PER_DAY
    seasonal = 1.0 + 0.9 * ((tod / MIN_PER_DAY - 0.5) ** 2 * 4)
    lam[t] = min((base + excite) * seasonal, 60.0)
    n_events[t] = rng.poisson(lam[t])
    excite = excite * (1 - decay) + alpha * decay * n_events[t]

# 每分钟价格变化：事件数越多，方差越大（信息驱动波动）
minute_ret = np.zeros(T)
for t in range(T):
    if n_events[t] > 0:
        minute_ret[t] = 0.0006 * np.sqrt(n_events[t]) * rng.standard_normal()
price = 50 * np.exp(np.cumsum(minute_ret))

# 每分钟成交量：与事件数正相关 + 噪声
volume = (n_events * rng.lognormal(8.5, 0.5, T)).astype(float) + rng.lognormal(6.5, 0.6, T)
dollar = volume * price

print(f"总分钟数 {T}, 事件强度均值 {lam.mean():.2f}, 分钟成交额中位数 {np.median(dollar):,.0f}")

# ---------- 2. 三种采样 ----------
def time_bars(ret, k):
    """每 k 分钟一根"""
    n = len(ret) // k
    return np.array([ret[i*k:(i+1)*k].sum() for i in range(n)])

def threshold_bars(ret, meter, thr):
    """累计 meter 超过 thr 就切一根"""
    out, acc, r_acc = [], 0.0, 0.0
    for i in range(len(ret)):
        acc += meter[i]; r_acc += ret[i]
        if acc >= thr:
            out.append(r_acc); acc, r_acc = 0.0, 0.0
    return np.array(out)

TARGET_BARS = 1200
tb = time_bars(minute_ret, T // TARGET_BARS)
vol_thr = volume.sum() / TARGET_BARS
vb = threshold_bars(minute_ret, volume, vol_thr)
dol_thr = dollar.sum() / TARGET_BARS
db = threshold_bars(minute_ret, dollar, dol_thr)
print(f"bar 数: time={len(tb)} volume={len(vb)} dollar={len(db)}")

def norm_stats(x):
    x = x[np.abs(x) > 0]
    z = (x - x.mean()) / x.std()
    k = stats.kurtosis(z)          # 超额峰度
    jb = stats.jarque_bera(z).statistic
    # 自相关(|r|)：波动聚集度
    a = np.abs(z)
    ac = np.corrcoef(a[:-1], a[1:])[0, 1]
    return k, jb, ac

k_t, jb_t, ac_t = norm_stats(tb)
k_v, jb_v, ac_v = norm_stats(vb)
k_d, jb_d, ac_d = norm_stats(db)
print(f"超额峰度: time={k_t:.2f} volume={k_v:.2f} dollar={k_d:.2f}")
print(f"Jarque-Bera: time={jb_t:.0f} volume={jb_v:.0f} dollar={jb_d:.0f}")
print(f"|r|自相关: time={ac_t:.3f} volume={ac_v:.3f} dollar={ac_d:.3f}")

# ---------- 图1：活跃度成簇的市场 ----------
fig, ax = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
ax[0].plot(price, lw=0.6, color="#334155")
ax[0].set_title("合成价格（60 个交易日，分钟级）")
ax[1].plot(lam, lw=0.5, color="#dc2626")
ax[1].set_title("信息到达强度 λ(t)：自激 + 日内 U 型")
ax[2].plot(dollar / 1e6, lw=0.4, color="#0ea5e9")
ax[2].set_title("分钟成交额（百万）：活跃度高度成簇")
ax[2].set_xlabel("分钟")
plt.tight_layout(); plt.savefig(f"{OUT}/market-activity.png", dpi=110); plt.close()

# ---------- 图2：三种 bar 的收益分布 vs 正态 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax_, x, name, kk in zip(axes, [tb, vb, db], ["时间棒", "成交量棒", "美元棒"], [k_t, k_v, k_d]):
    x = x[np.abs(x) > 0]
    z = (x - x.mean()) / x.std()
    ax_.hist(z, bins=60, density=True, alpha=0.75, color="#0ea5e9")
    grid = np.linspace(-5, 5, 300)
    ax_.plot(grid, stats.norm.pdf(grid), "--", color="#dc2626", lw=1.4, label="标准正态")
    ax_.set_title(f"{name}\n超额峰度 {kk:.2f}")
    ax_.set_xlim(-5, 5); ax_.legend(fontsize=8)
plt.tight_layout(); plt.savefig(f"{OUT}/return-distributions.png", dpi=110); plt.close()

# ---------- 图3：QQ 图对比 ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax_, x, name in zip(axes, [tb, db], ["时间棒", "美元棒"]):
    x = x[np.abs(x) > 0]
    z = np.sort((x - x.mean()) / x.std())
    q = stats.norm.ppf((np.arange(len(z)) + 0.5) / len(z))
    ax_.scatter(q, z, s=4, alpha=0.5, color="#0ea5e9")
    lim = max(abs(q[0]), abs(z[0]), abs(z[-1])) * 1.05
    ax_.plot([-lim, lim], [-lim, lim], "--", color="#dc2626", lw=1)
    ax_.set_title(f"{name} QQ 图"); ax_.set_xlabel("理论分位"); ax_.set_ylabel("样本分位")
plt.tight_layout(); plt.savefig(f"{OUT}/qq-plots.png", dpi=110); plt.close()

# ---------- 图4：每根 bar 覆盖的分钟数分布（美元棒的自适应性） ----------
def bar_durations(meter, thr):
    out, acc, cnt = [], 0.0, 0
    for i in range(len(meter)):
        acc += meter[i]; cnt += 1
        if acc >= thr:
            out.append(cnt); acc, cnt = 0.0, 0
    return np.array(out)

dur_d = bar_durations(dollar, dol_thr)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].hist(dur_d, bins=50, color="#0ea5e9", alpha=0.8)
axes[0].axvline(T // TARGET_BARS, color="#dc2626", ls="--", label=f"时间棒固定 {T//TARGET_BARS} 分钟")
axes[0].set_xlabel("每根美元棒覆盖的分钟数"); axes[0].set_ylabel("bar 数")
axes[0].set_title("美元棒时长分布：活跃时快切，清淡时慢切"); axes[0].legend()
# bar 时长 vs 当期强度
starts = np.cumsum(dur_d)[:-1]
lam_at = lam[np.minimum(starts, T - 1)]
axes[1].scatter(lam_at, dur_d[1:], s=6, alpha=0.4, color="#334155")
axes[1].set_xlabel("bar 起点处的信息强度 λ"); axes[1].set_ylabel("该 bar 覆盖分钟数")
axes[1].set_title("信息强度越高，美元棒切得越快")
plt.tight_layout(); plt.savefig(f"{OUT}/bar-durations.png", dpi=110); plt.close()

# ---------- 统计汇总 ----------
print("\n=== 汇总 ===")
print(f"{'指标':<14}{'时间棒':>10}{'成交量棒':>10}{'美元棒':>10}")
print(f"{'bar 数':<14}{len(tb):>10}{len(vb):>10}{len(db):>10}")
print(f"{'超额峰度':<14}{k_t:>10.2f}{k_v:>10.2f}{k_d:>10.2f}")
print(f"{'Jarque-Bera':<14}{jb_t:>10.0f}{jb_v:>10.0f}{jb_d:>10.0f}")
print(f"{'|r|一阶自相关':<14}{ac_t:>10.3f}{ac_v:>10.3f}{ac_d:>10.3f}")
print("图已生成:", os.listdir(OUT))
