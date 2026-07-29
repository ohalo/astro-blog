#!/usr/bin/env python3
"""CUSUM 事件过滤器实验图表生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from scipy import stats

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "public/images/cusum-filter-event-sampling"
os.makedirs(OUT, exist_ok=True)

# ---------- 模拟日线价格：漂移区制切换 + 噪声 ----------
n = 1500
# 三段区制：横盘 / 上升趋势 / 高波动下跌
mu = np.zeros(n)
mu[500:850] = 0.0012   # 趋势段
mu[1050:1250] = -0.0015
sigma = np.full(n, 0.010)
sigma[1050:1250] = 0.022
ret = mu + sigma * rng.standard_normal(n)
price = 100 * np.exp(np.cumsum(ret))
logp = np.log(price)

# ---------- 对称 CUSUM 过滤器 ----------
def cusum_filter(logp, h):
    """h: 阈值（可以是标量或数组）"""
    events = []
    s_pos, s_neg = 0.0, 0.0
    diff = np.diff(logp)
    h_arr = np.full(len(diff), h) if np.isscalar(h) else h[1:]
    for i in range(len(diff)):
        s_pos = max(0.0, s_pos + diff[i])
        s_neg = min(0.0, s_neg + diff[i])
        if s_pos >= h_arr[i]:
            events.append(i + 1); s_pos = 0.0
        elif s_neg <= -h_arr[i]:
            events.append(i + 1); s_neg = 0.0
    return np.array(events)

# 固定阈值 vs 波动率自适应阈值
h_fixed = 0.03
ew_vol = np.zeros(n)
lam = 0.94
v = ret[:50].var()
for i in range(n):
    v = lam * v + (1 - lam) * ret[i] ** 2
    ew_vol[i] = np.sqrt(v)
h_adapt = 2.5 * ew_vol

ev_fixed = cusum_filter(logp, h_fixed)
ev_adapt = cusum_filter(logp, h_adapt)

# ---------- 图1：价格 + 两种过滤器的事件点 ----------
fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for ax, ev, name, color in [(axes[0], ev_fixed, f"固定阈值 h={h_fixed} ({len(ev_fixed)} 事件)", "#1f77b4"),
                            (axes[1], ev_adapt, f"波动率自适应阈值 h=2.5σ ({len(ev_adapt)} 事件)", "#d62728")]:
    ax.plot(price, lw=0.8, color="#444")
    ax.scatter(ev, price[ev], color=color, s=22, zorder=3)
    ax.axvspan(500, 850, color="green", alpha=0.08)
    ax.axvspan(1050, 1250, color="red", alpha=0.08)
    ax.set_title(name)
    ax.set_ylabel("价格")
axes[1].set_xlabel("交易日（绿=趋势段，红=高波动段）")
plt.tight_layout()
plt.savefig(f"{OUT}/cusum-events-price.png", dpi=110)
plt.close()

# ---------- 图2：事件密度对比 ----------
win = 100
edges = np.arange(0, n + win, win)
d_fixed, _ = np.histogram(ev_fixed, bins=edges)
d_adapt, _ = np.histogram(ev_adapt, bins=edges)
centers = edges[:-1] + win / 2

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(centers, d_fixed, "o-", color="#1f77b4", label="固定阈值")
ax.plot(centers, d_adapt, "s-", color="#d62728", label="自适应阈值")
ax.axvspan(500, 850, color="green", alpha=0.08, label="趋势段")
ax.axvspan(1050, 1250, color="red", alpha=0.08, label="高波动段")
ax.set_xlabel("交易日")
ax.set_ylabel("每 100 日事件数")
ax.set_title("固定阈值在高波动段事件爆炸，自适应阈值密度均匀")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/cusum-event-density.png", dpi=110)
plt.close()

# ---------- 图3：标签冗余度 —— 逐日采样 vs CUSUM 事件采样的有效样本量 ----------
# AFML 平均唯一性：每个标签覆盖 [t, t+H]，concurrency c_t = 覆盖 t 的标签数
# uniqueness_i = mean(1/c_t)，有效样本量 = sum(uniqueness)
H = 20
def avg_uniqueness(idx, n, H):
    idx = idx[idx < n - H]
    conc = np.zeros(n)
    for t0 in idx:
        conc[t0:t0 + H] += 1
    u = np.array([np.mean(1.0 / conc[t0:t0 + H]) for t0 in idx])
    return idx, u

daily_idx = np.arange(50, n - H)
_, u_daily = avg_uniqueness(daily_idx, n, H)
ev_idx, u_ev = avg_uniqueness(ev_adapt, n, H)

eff_daily = u_daily.sum()
eff_ev = u_ev.sum()

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
ax = axes[0]
bars = ax.bar(["逐日采样", "CUSUM 事件采样"], [len(u_daily), len(u_ev)],
              color=["#7f7f7f", "#d62728"], alpha=0.7)
ax.bar(["逐日采样", "CUSUM 事件采样"], [eff_daily, eff_ev],
       color=["#333", "#8b0000"], alpha=0.9, width=0.4,
       label="有效独立样本量")
for x, (tot, eff) in enumerate([(len(u_daily), eff_daily), (len(u_ev), eff_ev)]):
    ax.text(x, tot + 15, f"名义 {tot}", ha="center")
    ax.text(x, eff + 15, f"有效 {eff:.0f}", ha="center", color="w" if eff > 100 else "k")
ax.set_ylabel("样本数")
ax.set_title("名义样本量 vs 有效独立样本量（H=20）")
ax.legend()

ax = axes[1]
ax.hist(u_daily, bins=30, alpha=0.6, color="#7f7f7f", density=True, label=f"逐日（均值 {u_daily.mean():.3f}）")
ax.hist(u_ev, bins=30, alpha=0.6, color="#d62728", density=True, label=f"CUSUM（均值 {u_ev.mean():.3f}）")
ax.set_xlabel("单标签平均唯一性")
ax.set_ylabel("密度")
ax.set_title("CUSUM 样本的唯一性分布整体右移")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/cusum-label-uniqueness.png", dpi=110)
plt.close()

print(f"fixed events={len(ev_fixed)}, adaptive={len(ev_adapt)}")
print(f"density hv fixed={d_fixed[10:13].mean():.1f} quiet={d_fixed[:5].mean():.1f}; "
      f"adaptive hv={d_adapt[10:13].mean():.1f} quiet={d_adapt[:5].mean():.1f}")
print(f"daily: N={len(u_daily)} eff={eff_daily:.1f} u={u_daily.mean():.3f}")
print(f"cusum: N={len(u_ev)} eff={eff_ev:.1f} u={u_ev.mean():.3f}")
print(f"ratio: cusum keeps {eff_ev/eff_daily*100:.0f}% of effective info with {len(u_ev)/len(u_daily)*100:.0f}% of samples")
