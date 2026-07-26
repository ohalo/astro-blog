# -*- coding: utf-8 -*-
"""BVC 批量成交分类配图生成 v2"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/bulk-volume-classification"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(7)

# ---------------- 模拟逐笔成交 ----------------
N = 200_000
HALF = 0.03          # 半价差
IMPACT = 0.008       # 每笔订单流的永久冲击
SIG_MID = 0.005      # 中间价噪声
P_NOISE = 0.01       # 成交价微观噪声
PERSIST = 0.65       # 订单流持续性

def simulate(n, seed):
    r = np.random.default_rng(seed)
    Q = np.where(r.random(n) < 0.5, 1, -1)
    for i in range(1, n):
        if r.random() < PERSIST:
            Q[i] = Q[i - 1]
    mid = 100 + np.cumsum(r.normal(0, SIG_MID, n) + IMPACT * Q)
    price = mid + Q * HALF + r.normal(0, P_NOISE, n)
    vol = r.lognormal(4.0, 1.0, n)
    return Q, price, vol

Q, price, vol = simulate(N, 7)

def make_bars(price, vol, Q, n_bars):
    cum = np.cumsum(vol)
    bv = cum[-1] / n_bars
    edges = np.searchsorted(cum, np.arange(bv, cum[-1], bv))
    out, start = [], 0
    for e in edges:
        if e <= start:
            continue
        seg = slice(start, e)
        v = vol[seg]
        out.append((price[e - 1] - price[start], v[Q[seg] == 1].sum() / v.sum()))
        start = e
    return np.array(out)

# ---------------- 图1：BVC 映射函数 + 估计 vs 真实 ----------------
bars = make_bars(price, vol, Q, 800)
dP, true_frac = bars[:, 0], bars[:, 1]
est_norm = stats.norm.cdf(dP / dP.std())

fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
ax = axes[0]
zz = np.linspace(-3.5, 3.5, 400)
ax.plot(zz, stats.norm.cdf(zz), lw=2.2, color="#1f77b4", label="正态 CDF")
ax.plot(zz, stats.t.cdf(zz, df=3), lw=2.2, color="#d62728", ls="--", label="t 分布 CDF (df=3)")
ax.axhline(0.5, color="gray", lw=0.8, ls=":")
ax.axvline(0, color="gray", lw=0.8, ls=":")
ax.set_xlabel("标准化价格变化 z = ΔP/σ")
ax.set_ylabel("判定为买方主动的成交量占比")
ax.set_title("BVC 核心映射：bar 涨得越多 → 买量占比越高")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.scatter(est_norm, true_frac, s=12, alpha=0.4, color="#1f77b4", edgecolors="none")
ax.plot([0, 1], [0, 1], color="black", lw=1, ls="--", label="45° 线")
r1 = np.corrcoef(est_norm, true_frac)[0, 1]
ax.set_xlabel("BVC 估计买量占比")
ax.set_ylabel("真实买量占比")
ax.set_title(f"每根成交量 bar：BVC 估计 vs 真实（r = {r1:.2f}）")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/bvc-cdf-mapping.png", dpi=130)
plt.close(fig)
print("图1 r =", round(r1, 3))

# ---------------- 图2：时间戳乱序（碎片化）下 BVC vs Tick 规则 ----------------
def tick_frac_per_bar(p_seg, v_seg):
    sign = np.empty(len(p_seg))
    last = 1.0
    prev = p_seg[0]
    for i, x in enumerate(p_seg):
        if x > prev + 1e-12:
            last = 1.0
        elif x < prev - 1e-12:
            last = -1.0
        sign[i] = last
        prev = x
    return v_seg[sign == 1].sum() / v_seg.sum()

def scramble(arrs, window, seed):
    """把逐笔序列按 window 大小分块，块内随机打乱（模拟时间戳粒度/多场所归并乱序）"""
    r = np.random.default_rng(seed)
    n = len(arrs[0])
    idx = np.arange(n)
    if window > 1:
        for s in range(0, n, window):
            e = min(s + window, n)
            perm = r.permutation(e - s)
            idx[s:e] = idx[s:e][perm]
    return [a[idx] for a in arrs]

windows = [1, 5, 20, 50, 200]
NB = 400
corr_bvc, corr_tick = [], []
Qs, ps, vs = simulate(120_000, 21)
for w in windows:
    p2, v2, q2 = scramble([ps, vs, Qs], w, seed=100 + w)
    cum = np.cumsum(v2)
    bv = cum[-1] / NB
    edges = np.searchsorted(cum, np.arange(bv, cum[-1], bv))
    dP_l, true_l, tick_l = [], [], []
    start = 0
    for e in edges:
        if e <= start:
            continue
        seg = slice(start, e)
        true_l.append(v2[seg][q2[seg] == 1].sum() / v2[seg].sum())
        tick_l.append(tick_frac_per_bar(p2[seg], v2[seg]))
        dP_l.append(p2[e - 1] - p2[start])
        start = e
    dP_l = np.array(dP_l); true_l = np.array(true_l); tick_l = np.array(tick_l)
    est = stats.norm.cdf(dP_l / dP_l.std())
    corr_bvc.append(np.corrcoef(est, true_l)[0, 1])
    corr_tick.append(np.corrcoef(tick_l, true_l)[0, 1])

fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(windows))
w_ = 0.36
ax.bar(x - w_/2, corr_bvc, w_, color="#1f77b4", label="BVC（只用 bar 端点价格）")
ax.bar(x + w_/2, corr_tick, w_, color="#ff7f0e", label="Tick 规则（依赖逐笔顺序）")
ax.set_xticks(x)
ax.set_xticklabels(["无乱序", "5 笔", "20 笔", "50 笔", "200 笔"])
ax.set_xlabel("时间戳乱序窗口（块内逐笔顺序随机打乱）")
ax.set_ylabel("与真实买量占比的相关系数")
ax.set_title("碎片化/时间戳乱序加剧时：Tick 规则崩塌，BVC 几乎无感")
ax.legend()
ax.grid(alpha=0.3, axis="y")
for i, (a, b) in enumerate(zip(corr_bvc, corr_tick)):
    ax.text(i - w_/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=9)
    ax.text(i + w_/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/bvc-vs-tick-scramble.png", dpi=130)
plt.close(fig)
print("图2 bvc:", [round(a, 3) for a in corr_bvc])
print("图2 tick:", [round(a, 3) for a in corr_tick])

# ---------------- 图3：bar 大小权衡 ----------------
n_bars_list = [100, 200, 400, 800, 1600, 3200]
r_list, mae_list = [], []
for nb in n_bars_list:
    b = make_bars(price, vol, Q, nb)
    dPb, tb = b[:, 0], b[:, 1]
    est = stats.norm.cdf(dPb / dPb.std())
    r_list.append(np.corrcoef(est, tb)[0, 1])
    mae_list.append(np.abs(est - tb).mean())

fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(n_bars_list, r_list, "o-", color="#1f77b4", lw=2, label="相关系数（左轴）")
ax1.set_xscale("log")
ax1.set_xlabel("bar 数量（同一段数据切得越多 → 每根 bar 越小）")
ax1.set_ylabel("与真实买量占比的相关系数", color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax1.grid(alpha=0.3)
ax2 = ax1.twinx()
ax2.plot(n_bars_list, mae_list, "s--", color="#d62728", lw=2, label="平均绝对误差（右轴）")
ax2.set_ylabel("平均绝对误差", color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax1.set_title("bar 大小权衡：切太细噪声主导、切太粗把反向流平均掉")
l1, la1 = ax1.get_legend_handles_labels()
l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="center right")
fig.tight_layout()
fig.savefig(f"{OUT}/bvc-bar-size-tradeoff.png", dpi=130)
plt.close(fig)
print("图3 r:", [round(r, 3) for r in r_list])
print("图3 mae:", [round(m, 4) for m in mae_list])
