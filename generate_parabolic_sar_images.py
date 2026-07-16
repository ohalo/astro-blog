#!/usr/bin/env python3
"""
为文章「抛物线 SAR 与追踪止损：用加速因子把出场点焊在价格上」(parabolic-sar-trailing)
生成真实配图（自洽合成，非占位图）。

模型：Wilder(1978) Parabolic SAR
  SAR_{i+1} = SAR_i + AF * (EP - SAR_i)，趋势反转时翻转
图表：
  1. sar_dots_price.png       价格 + SAR 点（多头绿点在下、空头红点在上）
  2. sar_af_acceleration.png  加速因子 AF 随新极值递增，SAR 追近价格
  3. sar_vs_fixed_stop.png    SAR 追踪止损 vs 固定百分比止损 净值
  4. sar_af_step_scan.png     不同 AF 步长下年化收益/Sharpe/交易次数
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "parabolic-sar-trailing"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

np.random.seed(2026)

# ---------- 合成价格：带趋势的 OHLC ----------
def synth_ohlc(n=700):
    rng = np.random.default_rng(2026)
    price = 100.0
    closes = []
    regime_len = 0
    drift = 0.0
    vol = 0.01
    for i in range(n):
        if regime_len <= 0:
            regime_len = rng.integers(50, 130)
            drift = rng.choice([1, -1, 0], p=[0.4, 0.3, 0.3]) * rng.uniform(0.0006, 0.0022)
            vol = rng.uniform(0.007, 0.013)
        r = rng.normal(drift, vol)
        price *= np.exp(r)
        closes.append(price)
        regime_len -= 1
    close = np.array(closes)
    intr = rng.uniform(0.004, 0.011, size=n)
    high = close * (1 + intr)
    low = close * (1 - intr)
    openp = np.concatenate([[close[0]], close[:-1]])
    return openp, high, low, close

openp, high, low, close = synth_ohlc()
n = len(close)


# ---------- Parabolic SAR ----------
def parabolic_sar(high, low, af_step=0.02, af_max=0.20):
    n = len(high)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)  # 1=long, -1=short
    af = af_step
    # 初始化
    up = True
    ep = high[0]
    sar[0] = low[0]
    trend[0] = 1
    for i in range(1, n):
        prev_sar = sar[i - 1]
        if up:
            cur = prev_sar + af * (ep - prev_sar)
            cur = min(cur, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if high[i] > ep:
                ep = high[i]; af = min(af + af_step, af_max)
            if low[i] < cur:
                up = False; cur = ep; ep = low[i]; af = af_step
            sar[i] = cur; trend[i] = 1 if up else -1
        else:
            cur = prev_sar + af * (ep - prev_sar)
            cur = max(cur, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if low[i] < ep:
                ep = low[i]; af = min(af + af_step, af_max)
            if high[i] > cur:
                up = True; cur = ep; ep = high[i]; af = af_step
            sar[i] = cur; trend[i] = 1 if up else -1
    return sar, trend

sar, trend = parabolic_sar(high, low, 0.02, 0.20)

# ---------- 图1：SAR 点 + 价格 ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
seg = slice(200, 420)
x = np.arange(n)[seg]
ax.plot(x, close[seg], color="#1f2937", lw=1.3, label="收盘价", zorder=2)
long_mask = trend[seg] == 1
short_mask = trend[seg] == -1
ax.scatter(x[long_mask], sar[seg][long_mask], s=12, color="#16a34a",
           label="SAR（多头，点在价下方）", zorder=3)
ax.scatter(x[short_mask], sar[seg][short_mask], s=12, color="#dc2626",
           label="SAR（空头，点在价上方）", zorder=3)
ax.set_title("抛物线 SAR：止损点随趋势移动，反转时穿价翻边", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
ax.legend(loc="upper left"); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "sar_dots_price.png"), dpi=130)
plt.close()

# ---------- 图2：AF 演化，展示加速 ----------
def sar_with_af(high, low, af_step=0.02, af_max=0.20):
    n = len(high)
    sar = np.zeros(n); afs = np.zeros(n); trend = np.zeros(n, int)
    af = af_step; up = True; ep = high[0]; sar[0] = low[0]; afs[0] = af; trend[0] = 1
    for i in range(1, n):
        prev = sar[i - 1]
        if up:
            cur = prev + af * (ep - prev)
            cur = min(cur, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if high[i] > ep: ep = high[i]; af = min(af + af_step, af_max)
            if low[i] < cur: up = False; cur = ep; ep = low[i]; af = af_step
        else:
            cur = prev + af * (ep - prev)
            cur = max(cur, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if low[i] < ep: ep = low[i]; af = min(af + af_step, af_max)
            if high[i] > cur: up = True; cur = ep; ep = high[i]; af = af_step
        sar[i] = cur; afs[i] = af; trend[i] = 1 if up else -1
    return sar, afs, trend

sar2, afs, trend2 = sar_with_af(high, low)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
seg2 = slice(240, 360)
xx = np.arange(n)[seg2]
ax1.plot(xx, close[seg2], color="#1f2937", lw=1.3, label="收盘价")
ax1.plot(xx, sar2[seg2], color="#2563eb", lw=1.0, marker=".", ms=4, label="SAR 追踪线")
ax1.fill_between(xx, close[seg2], sar2[seg2], where=(close[seg2] > sar2[seg2]),
                 color="#16a34a", alpha=0.10)
ax1.set_ylabel("价格"); ax1.legend(loc="upper left"); ax1.grid(alpha=0.25)
ax1.set_title("加速因子 AF 让 SAR 越涨越快地贴近价格", fontsize=13, fontweight="bold")
ax2.plot(xx, afs[seg2], color="#f59e0b", lw=1.5, label="加速因子 AF")
ax2.axhline(0.20, color="#9ca3af", ls="--", lw=1.0, label="AF 上限 0.20")
ax2.set_ylabel("AF"); ax2.set_xlabel("交易日")
ax2.legend(loc="upper left"); ax2.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "sar_af_acceleration.png"), dpi=130)
plt.close()

# ---------- 回测：SAR 追踪止损 vs 固定百分比止损（只做多趋势跟随） ----------
ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])

def backtest_sar(sar, trend):
    pos = np.zeros(n); cur = 0
    for i in range(1, n):
        # trend 在 i 确定，i+1 执行
        if trend[i - 1] == 1:
            cur = 1
        else:
            cur = 0
        pos[i] = cur
    sr = pos * ret
    eq = np.cumprod(1 + sr)
    trades = int(np.sum(np.abs(np.diff(pos)) > 0))
    return eq, sr, pos, trades

def backtest_fixed_stop(stop_pct=0.08):
    """入场用 20 日突破，止损用固定百分比追踪。"""
    pos = np.zeros(n); cur = 0; entry_peak = 0
    win = 20
    for i in range(1, n):
        if cur == 0:
            lo = max(0, i - win - 1)
            if i > win and close[i - 1] >= np.max(close[lo:i - 1]):
                cur = 1; entry_peak = close[i - 1]
        else:
            entry_peak = max(entry_peak, close[i - 1])
            if close[i - 1] < entry_peak * (1 - stop_pct):
                cur = 0
        pos[i] = cur
    sr = pos * ret
    eq = np.cumprod(1 + sr)
    trades = int(np.sum(np.abs(np.diff(pos)) > 0))
    return eq, sr, pos, trades

eq_sar, sr_sar, pos_sar, tr_sar = backtest_sar(sar, trend)
eq_fix, sr_fix, pos_fix, tr_fix = backtest_fixed_stop(0.08)
eq_bh = np.cumprod(1 + ret)

def stats(sr):
    ann = np.mean(sr) * 252
    sharpe = np.mean(sr) / (np.std(sr) + 1e-12) * np.sqrt(252)
    eq = np.cumprod(1 + sr)
    mdd = np.min(eq / np.maximum.accumulate(eq) - 1)
    return ann, sharpe, mdd

a_s, sh_s, mdd_s = stats(sr_sar)
a_x, sh_x, mdd_x = stats(sr_fix)
a_b, sh_b, mdd_b = stats(ret)

fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(eq_sar, color="#2563eb", lw=1.8, label="抛物线 SAR 追踪止损")
ax.plot(eq_fix, color="#f59e0b", lw=1.4, label="固定 8% 追踪止损")
ax.plot(eq_bh, color="#6b7280", lw=1.3, ls="--", label="买入持有")
ax.set_title("SAR 追踪止损 vs 固定百分比止损", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("净值（初始=1）")
ax.legend(loc="upper left"); ax.grid(alpha=0.25)
txt = (f"SAR:   年化{a_s*100:.1f}% / Sharpe {sh_s:.2f} / 回撤{mdd_s*100:.1f}% / 交易{tr_sar}\n"
       f"固定8%: 年化{a_x*100:.1f}% / Sharpe {sh_x:.2f} / 回撤{mdd_x*100:.1f}% / 交易{tr_fix}\n"
       f"买入持有: 年化{a_b*100:.1f}% / Sharpe {sh_b:.2f} / 回撤{mdd_b*100:.1f}%")
ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round", fc="#f8fafc", ec="#cbd5e1"))
plt.tight_layout()
plt.savefig(os.path.join(D, "sar_vs_fixed_stop.png"), dpi=130)
plt.close()

# ---------- 图4：AF 步长扫描 ----------
steps = [0.005, 0.01, 0.015, 0.02, 0.03, 0.04, 0.05]
anns, sharpes, trades_list = [], [], []
for st in steps:
    s_, t_ = parabolic_sar(high, low, st, 0.20)
    eq, sr, pos, tr = backtest_sar(s_, t_)
    a, sh, _ = stats(sr)
    anns.append(a * 100); sharpes.append(sh); trades_list.append(tr)

fig, ax1 = plt.subplots(figsize=(11, 5))
ax1.plot(steps, anns, "o-", color="#2563eb", lw=1.8, label="年化收益(%)")
ax1.set_xlabel("AF 步长"); ax1.set_ylabel("年化收益(%)", color="#2563eb")
ax1.tick_params(axis="y", labelcolor="#2563eb"); ax1.grid(alpha=0.25)
ax2 = ax1.twinx()
ax2.plot(steps, sharpes, "s--", color="#dc2626", lw=1.6, label="Sharpe")
ax2.set_ylabel("Sharpe", color="#dc2626"); ax2.tick_params(axis="y", labelcolor="#dc2626")
ax1.axvline(0.02, color="#9ca3af", ls=":", lw=1.2)
for x_, y_, t_ in zip(steps, anns, trades_list):
    ax1.annotate(f"{t_}笔", (x_, y_), textcoords="offset points", xytext=(0, 8),
                 fontsize=7, ha="center", color="#6b7280")
ax1.set_title("AF 步长敏感性：步长越大越贴价（交易更频繁）", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "sar_af_step_scan.png"), dpi=130)
plt.close()

print("SAR images done:", os.listdir(D))
print(f"SAR:   ann={a_s*100:.1f}% sharpe={sh_s:.2f} mdd={mdd_s*100:.1f}% trades={tr_sar}")
print(f"Fixed: ann={a_x*100:.1f}% sharpe={sh_x:.2f} mdd={mdd_x*100:.1f}% trades={tr_fix}")
print(f"BH:    ann={a_b*100:.1f}% sharpe={sh_b:.2f} mdd={mdd_b*100:.1f}%")
