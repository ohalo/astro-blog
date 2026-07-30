#!/usr/bin/env python3
"""Imbalance Bars experiment.

Synthetic tick-level market with informed-trading episodes (persistent one-sided
signed flow). Compare Time Bars / Dollar Bars / Tick-Imbalance Bars (TIB) &
Dollar-Imbalance Bars (DIB) on:
  1) sampling density around informed episodes (do imbalance bars 'wake up'?)
  2) detection latency of the onset of informed flow
  3) return distribution stats
Outputs PNGs + printed metrics.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(7)
OUT = "/Users/halo/workspace/astro-blog/public/images/microstructure-bars-imbalance"
os.makedirs(OUT, exist_ok=True)

# ---------------- 1. synthetic tick market ----------------
# ticks arrive continuously; each tick has sign b in {+1,-1} and size v.
# Normal regime: P(b=+1)=0.5, informed episodes: P(b=+1)=p_inf (one side) for a stretch.
N = 120_000
p = np.full(N, 0.5)
episodes = []  # (start, end, side)
t = 5_000
while t < N - 8_000:
    gap = int(rng.integers(6_000, 18_000))
    t += gap
    L = int(rng.integers(1_500, 4_000))
    side = rng.choice([1, -1])
    p_inf = 0.5 + side * rng.uniform(0.10, 0.18)
    p[t : t + L] = p_inf
    episodes.append((t, min(t + L, N), side))
    t += L
print(f"episodes: {len(episodes)}")

b = np.where(rng.random(N) < p, 1, -1)
# sizes: lognormal, informed flow slightly larger
size = rng.lognormal(mean=0.0, sigma=0.6, size=N)
informed_mask = p != 0.5
size[informed_mask] *= 1.3

# price: permanent impact of signed flow + noise
impact = 0.00004
noise = 0.0003
logp = np.cumsum(impact * b * np.sqrt(size) + noise * rng.standard_normal(N))
price = 100 * np.exp(logp)
dollar = price * size

# time index: ticks arrive at ~constant rate for simplicity; 200 ticks = 1 "minute"
TICKS_PER_MIN = 200

# ---------------- 2. bar constructors ----------------
def time_bars(step_ticks):
    edges = np.arange(0, N, step_ticks)
    return edges[1:]

def dollar_bars(threshold):
    edges = []
    acc = 0.0
    for i in range(N):
        acc += dollar[i]
        if acc >= threshold:
            edges.append(i)
            acc = 0.0
    return np.array(edges)

def imbalance_bars(flow, threshold):
    """Fixed-threshold imbalance bars: close bar when |cum signed flow| >= threshold.

    Note: the AFML EWMA version (E_T * |E_b|) is notoriously unstable (positive
    feedback collapse to 1-tick bars). Fixed threshold, numerically calibrated
    to a target bar count, keeps the economics and stays robust.
    """
    edges = []
    theta = 0.0
    for i in range(N):
        theta += flow[i]
        if abs(theta) >= threshold:
            edges.append(i)
            theta = 0.0
    return np.array(edges)

def calibrate_threshold(flow, n_target, lo, hi, iters=40):
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        nb = len(imbalance_bars(flow, mid))
        if nb > n_target:
            lo = mid
        else:
            hi = mid
        if abs(nb - n_target) <= max(3, n_target // 100):
            break
    return 0.5 * (lo + hi)

tb = time_bars(TICKS_PER_MIN)          # 1-min time bars
n_target = len(tb)
# calibrate dollar threshold to match bar count
thr_d = dollar.sum() / n_target
db = dollar_bars(thr_d)
flow_t = b.astype(float)
flow_d = b * dollar
thr_tib = calibrate_threshold(flow_t, n_target, 1.0, 200.0)
thr_dib = calibrate_threshold(flow_d, n_target, dollar.mean(), 500 * dollar.mean())
tib = imbalance_bars(flow_t, thr_tib)
dib = imbalance_bars(flow_d, thr_dib)
print(f"thresholds: TIB={thr_tib:.1f} ticks, DIB={thr_dib:.0f} dollars")
print(f"bars: time={len(tb)} dollar={len(db)} TIB={len(tib)} DIB={len(dib)}")

# ---------------- 3. sampling density around episodes ----------------
def bar_rate_in(edges, mask):
    """bars per 1000 ticks inside mask vs outside."""
    inside = mask[edges].sum() / mask.sum() * 1000
    outside = (~mask[edges]).sum() / (~mask).sum() * 1000
    return inside, outside

for name, e in [("时间棒", tb), ("美元棒", db), ("tick失衡棒", tib), ("美元失衡棒", dib)]:
    inside, outside = bar_rate_in(e, informed_mask)
    print(f"{name}: bar-rate informed={inside:.2f}/1k ticks, normal={outside:.2f}/1k, ratio={inside/outside:.2f}")

# ---------------- 4. detection latency ----------------
# latency = ticks from episode start until K consecutive same-direction bars
def detect_latency(edges, K=3):
    lats = []
    # bar signs = sign of bar return
    ret = np.diff(logp[edges])
    sgn = np.sign(ret)
    for (s, e_end, side) in episodes:
        # find bars after s
        j0 = np.searchsorted(edges[1:], s)
        run = 0
        lat = None
        for j in range(j0, len(sgn)):
            if edges[j + 1] > e_end + 4000:
                break
            if sgn[j] == side:
                run += 1
                if run >= K:
                    lat = edges[j + 1] - s
                    break
            else:
                run = 0
        lats.append(lat if lat is not None else np.nan)
    return np.array(lats, dtype=float)

lat_tb = detect_latency(tb)
lat_db = detect_latency(db)
lat_tib = detect_latency(tib)
lat_dib = detect_latency(dib)
def latsum(x, name):
    ok = ~np.isnan(x)
    print(f"{name}: detected {ok.sum()}/{len(x)}, median latency={np.nanmedian(x):.0f} ticks (~{np.nanmedian(x)/TICKS_PER_MIN:.1f} min)")
    return np.nanmedian(x)
m_tb = latsum(lat_tb, "时间棒")
m_db = latsum(lat_db, "美元棒")
m_tib = latsum(lat_tib, "tick失衡棒")
m_dib = latsum(lat_dib, "美元失衡棒")

# ---------------- 5. distribution stats ----------------
from scipy import stats as st
def dist_stats(edges, name):
    r = np.diff(logp[edges])
    r = (r - r.mean()) / r.std()
    kurt = st.kurtosis(r)
    jb = st.jarque_bera(r).statistic
    ac = np.corrcoef(np.abs(r[:-1]), np.abs(r[1:]))[0, 1]
    print(f"{name}: n={len(r)}, excess kurt={kurt:.2f}, JB={jb:.0f}, |r| autocorr={ac:.3f}")
    return kurt, jb, ac
k_tb = dist_stats(tb, "时间棒")
k_db = dist_stats(db, "美元棒")
k_tib = dist_stats(tib, "tick失衡棒")
k_dib = dist_stats(dib, "美元失衡棒")

# ---------------- plots ----------------
# P1: price + episodes + bar positions (zoom on one episode)
s0, e0, side0 = episodes[1]
z0, z1 = s0 - 3000, min(e0 + 3000, N)
fig, axes = plt.subplots(3, 1, figsize=(12, 8.5), sharex=True, gridspec_kw={"height_ratios": [2, 1, 1]})
xs = np.arange(z0, z1)
axes[0].plot(xs / TICKS_PER_MIN, price[z0:z1], color="k", lw=0.8)
axes[0].axvspan(s0 / TICKS_PER_MIN, e0 / TICKS_PER_MIN, color="#ffdddd", label="知情交易时段")
axes[0].legend(loc="upper left")
axes[0].set_title(f"一段知情交易时段（单边 {'买' if side0>0 else '卖'}压持续 {e0-s0} ticks）")
axes[0].set_ylabel("价格")
for ax, e_arr, name, c in [(axes[1], tb, "时间棒", "#ff7f0e"), (axes[2], dib, "美元失衡棒", "#1f77b4")]:
    ee = e_arr[(e_arr >= z0) & (e_arr < z1)]
    ax.vlines(ee / TICKS_PER_MIN, 0, 1, color=c, lw=0.9)
    ax.axvspan(s0 / TICKS_PER_MIN, e0 / TICKS_PER_MIN, color="#ffdddd")
    ax.set_yticks([])
    ax.set_ylabel(name)
axes[2].set_xlabel("时间（分钟）")
axes[1].set_title("bar 切分位置：时间棒均匀打点，失衡棒在知情时段密集开火")
plt.tight_layout()
plt.savefig(f"{OUT}/episode-sampling.png", dpi=110)
plt.close()

# P2: bar rate ratio bars
fig, ax = plt.subplots(figsize=(9, 4.2))
names = ["时间棒", "美元棒", "tick 失衡棒", "美元失衡棒"]
ratios = []
for e in [tb, db, tib, dib]:
    i_, o_ = bar_rate_in(e, informed_mask)
    ratios.append(i_ / o_)
bars = ax.bar(names, ratios, color=["#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4"], width=0.55)
for bb, v in zip(bars, ratios):
    ax.text(bb.get_x() + bb.get_width() / 2, v + 0.03, f"{v:.2f}×", ha="center")
ax.axhline(1, color="k", lw=0.8, ls="--")
ax.set_ylabel("知情时段 / 平静时段 采样密度比")
ax.set_title("失衡棒在知情交易时段的采样密度倍数")
plt.tight_layout()
plt.savefig(f"{OUT}/bar-rate-ratio.png", dpi=110)
plt.close()

# P3: detection latency boxplot
fig, ax = plt.subplots(figsize=(9, 4.5))
data = [lat_tb[~np.isnan(lat_tb)] / TICKS_PER_MIN, lat_db[~np.isnan(lat_db)] / TICKS_PER_MIN,
        lat_tib[~np.isnan(lat_tib)] / TICKS_PER_MIN, lat_dib[~np.isnan(lat_dib)] / TICKS_PER_MIN]
bp = ax.boxplot(data, tick_labels=names, patch_artist=True)
for patch, c in zip(bp["boxes"], ["#ff7f0e", "#9467bd", "#2ca02c", "#1f77b4"]):
    patch.set_facecolor(c); patch.set_alpha(0.6)
ax.set_ylabel("检测延迟（分钟）")
ax.set_title("知情流开始 → 连续3根同向bar 的检测延迟")
plt.tight_layout()
plt.savefig(f"{OUT}/detection-latency.png", dpi=110)
plt.close()

# P4: bar duration distribution (ticks per bar) for DIB
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
dur_dib = np.diff(dib)
axes[0].hist(dur_dib, bins=60, color="#1f77b4", alpha=0.85)
axes[0].set_title("美元失衡棒：每根 bar 的时长分布（ticks）")
axes[0].set_xlabel("ticks / bar")
# theta accumulation illustration around an episode
s0, e0, side0 = episodes[2]
zz0, zz1 = s0 - 1500, s0 + 2500
theta_demo = np.cumsum(b[zz0:zz1] * dollar[zz0:zz1])
axes[1].plot((np.arange(zz0, zz1) - s0) / TICKS_PER_MIN, theta_demo, color="#1f77b4", lw=1)
axes[1].axvline(0, color="#d62728", ls="--", label="知情流开始")
axes[1].set_title("累计签名美元流 θ：知情流开始后一路单边")
axes[1].set_xlabel("相对知情流开始的分钟数")
axes[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT}/theta-durations.png", dpi=110)
plt.close()

print("PLOTS DONE")
