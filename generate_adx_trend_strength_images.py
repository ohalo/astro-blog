#!/usr/bin/env python3
"""
为文章「ADX 趋势强度：用定向运动指数把『有趋势』和『无趋势』分开」(adx-trend-strength)
生成真实配图（自洽合成，非占位图）。

模型：Wilder(1978) 定向运动系统
  +DM / -DM -> +DI / -DI -> DX -> ADX（Wilder 平滑）
图表：
  1. adx_components.png       价格 + +DI/-DI/ADX 三线随时间演化
  2. adx_regime_split.png     ADX 阈值把行情切成趋势段/震荡段
  3. adx_filter_equity.png    ADX 过滤 vs 裸趋势策略 vs 买入持有 净值
  4. adx_threshold_scan.png   不同 ADX 阈值下的年化收益/Sharpe 扫描
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "adx-trend-strength"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)

np.random.seed(42)

# ---------- 合成价格：交替的趋势段与震荡段 ----------
def synth_prices(n_total=900):
    segs = []
    price = 100.0
    rng = np.random.default_rng(42)
    regimes = []  # 1=trend, 0=range
    while sum(len(s) for s in segs) < n_total:
        is_trend = rng.random() < 0.5
        length = rng.integers(60, 140)
        if is_trend:
            drift = rng.choice([1, -1]) * rng.uniform(0.0008, 0.0025)
            vol = rng.uniform(0.008, 0.012)
        else:
            drift = 0.0
            vol = rng.uniform(0.006, 0.010)
        rets = rng.normal(drift, vol, size=length)
        if not is_trend:
            # 均值回复叠加，压制趋势
            level = np.log(price)
            p = [level]
            for r in rets:
                p.append(0.97 * p[-1] + 0.03 * level + r)
            seg = np.exp(np.array(p[1:]))
        else:
            seg = price * np.exp(np.cumsum(rets))
        segs.append(seg)
        regimes.append(np.full(length, 1 if is_trend else 0))
        price = seg[-1]
    close = np.concatenate(segs)[:n_total]
    regime = np.concatenate(regimes)[:n_total]
    # 由 close 造 high/low/open
    rng2 = np.random.default_rng(7)
    rng_intraday = rng2.uniform(0.004, 0.012, size=n_total)
    high = close * (1 + rng_intraday)
    low = close * (1 - rng_intraday)
    openp = np.concatenate([[close[0]], close[:-1]])
    return openp, high, low, close, regime

openp, high, low, close, regime = synth_prices()
n = len(close)


# ---------- Wilder ADX ----------
def wilder_smooth(x, period):
    """Wilder 平滑（RMA）。"""
    x = np.asarray(x, dtype=float)
    out = np.full_like(x, np.nan)
    if len(x) < period:
        return out
    out[period - 1] = np.nanmean(x[:period])
    for i in range(period, len(x)):
        out[i] = (out[i - 1] * (period - 1) + x[i]) / period
    return out


def compute_adx(high, low, close, period=14):
    n = len(close)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        down = low[i - 1] - low[i]
        plus_dm[i] = up if (up > down and up > 0) else 0.0
        minus_dm[i] = down if (down > up and down > 0) else 0.0
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i - 1]),
                    abs(low[i] - close[i - 1]))
    atr = wilder_smooth(tr[1:], period)
    sm_plus = wilder_smooth(plus_dm[1:], period)
    sm_minus = wilder_smooth(minus_dm[1:], period)
    atr = np.concatenate([[np.nan], atr])
    sm_plus = np.concatenate([[np.nan], sm_plus])
    sm_minus = np.concatenate([[np.nan], sm_minus])
    with np.errstate(invalid="ignore", divide="ignore"):
        plus_di = 100 * sm_plus / atr
        minus_di = 100 * sm_minus / atr
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = wilder_smooth(dx[~np.isnan(dx)], period)
    # 对齐回原长度
    adx_full = np.full(n, np.nan)
    valid_idx = np.where(~np.isnan(dx))[0]
    if len(valid_idx) >= period:
        adx_full[valid_idx] = adx
    return plus_di, minus_di, adx_full


plus_di, minus_di, adx = compute_adx(high, low, close, 14)

# ---------- 图1：价格 + DI/ADX ----------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1]})
ax1.plot(close, color="#1f2937", lw=1.2, label="收盘价")
ax1.set_ylabel("价格")
ax1.set_title("ADX 定向运动系统：价格与趋势强度指标", fontsize=13, fontweight="bold")
ax1.legend(loc="upper left")
ax1.grid(alpha=0.25)

ax2.plot(plus_di, color="#16a34a", lw=1.0, label="+DI（上升方向强度）")
ax2.plot(minus_di, color="#dc2626", lw=1.0, label="-DI（下降方向强度）")
ax2.plot(adx, color="#2563eb", lw=1.6, label="ADX（趋势强度，无方向）")
ax2.axhline(25, color="#9ca3af", ls="--", lw=1.0, label="ADX=25 阈值")
ax2.set_ylabel("指标值")
ax2.set_xlabel("交易日")
ax2.legend(loc="upper left", ncol=2, fontsize=8)
ax2.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "adx_components.png"), dpi=130)
plt.close()

# ---------- 图2：ADX 阈值切分行情 ----------
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(close, color="#1f2937", lw=1.1, zorder=3)
trend_mask = adx > 25
# 阴影标注趋势段
in_trend = False
start = 0
for i in range(n):
    t = bool(trend_mask[i]) if not np.isnan(adx[i]) else False
    if t and not in_trend:
        start = i; in_trend = True
    elif not t and in_trend:
        ax.axvspan(start, i, color="#3b82f6", alpha=0.12, zorder=1)
        in_trend = False
if in_trend:
    ax.axvspan(start, n - 1, color="#3b82f6", alpha=0.12, zorder=1)
ax.set_title("ADX>25 标出的『趋势段』（蓝色阴影）vs 震荡段（留白）", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("价格")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#3b82f6", alpha=0.12, label="ADX>25 趋势段"),
                   plt.Line2D([0], [0], color="#1f2937", label="收盘价")], loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "adx_regime_split.png"), dpi=130)
plt.close()

# ---------- 回测：DI 交叉趋势策略，ADX 作为开关 ----------
ret = np.concatenate([[0.0], np.diff(close) / close[:-1]])

def backtest(use_adx_filter, adx_th=25):
    pos = np.zeros(n)
    cur = 0
    for i in range(1, n):
        if np.isnan(plus_di[i]) or np.isnan(minus_di[i]) or np.isnan(adx[i]):
            pos[i] = cur
            continue
        strong = adx[i] > adx_th if use_adx_filter else True
        # 信号在 i 判定，i+1 执行 -> 用前一日 DI
        if plus_di[i - 1] > minus_di[i - 1] and strong:
            cur = 1
        elif minus_di[i - 1] > plus_di[i - 1] and strong:
            cur = 0  # 只做多，弱势/下行空仓
        elif use_adx_filter and adx[i] <= adx_th:
            cur = 0  # 无趋势离场
        pos[i] = cur
    strat_ret = pos * ret
    equity = np.cumprod(1 + strat_ret)
    return equity, pos, strat_ret

eq_filter, pos_f, sr_f = backtest(True, 25)
eq_naive, pos_n, sr_n = backtest(False)
eq_bh = np.cumprod(1 + ret)

# ---------- 图3：净值对比 ----------
fig, ax = plt.subplots(figsize=(11, 5.5))
ax.plot(eq_filter, color="#2563eb", lw=1.8, label="ADX>25 过滤的趋势策略")
ax.plot(eq_naive, color="#f59e0b", lw=1.3, label="裸 DI 交叉（无 ADX 过滤）")
ax.plot(eq_bh, color="#6b7280", lw=1.3, ls="--", label="买入持有")
ax.set_title("ADX 过滤把『无趋势期的空砍』挡在门外", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("净值（初始=1）")
ax.legend(loc="upper left"); ax.grid(alpha=0.25)


def stats(sr):
    ann = np.mean(sr) * 252
    sharpe = np.mean(sr) / (np.std(sr) + 1e-12) * np.sqrt(252)
    return ann, sharpe

a_f, s_f = stats(sr_f)
a_n, s_n = stats(sr_n)
a_b, s_b = stats(ret)
txt = (f"ADX过滤: 年化{a_f*100:.1f}% / Sharpe {s_f:.2f}\n"
       f"裸交叉:  年化{a_n*100:.1f}% / Sharpe {s_n:.2f}\n"
       f"买入持有: 年化{a_b*100:.1f}% / Sharpe {s_b:.2f}")
ax.text(0.98, 0.03, txt, transform=ax.transAxes, ha="right", va="bottom",
        fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="#f8fafc", ec="#cbd5e1"))
plt.tight_layout()
plt.savefig(os.path.join(D, "adx_filter_equity.png"), dpi=130)
plt.close()

# ---------- 图4：阈值扫描 ----------
ths = list(range(10, 41, 2))
anns, sharpes = [], []
for th in ths:
    eq, pos, sr = backtest(True, th)
    a, s = stats(sr)
    anns.append(a * 100); sharpes.append(s)

fig, ax1 = plt.subplots(figsize=(11, 5))
color1 = "#2563eb"
ax1.plot(ths, anns, "o-", color=color1, lw=1.8, label="年化收益(%)")
ax1.set_xlabel("ADX 阈值"); ax1.set_ylabel("年化收益(%)", color=color1)
ax1.tick_params(axis="y", labelcolor=color1)
ax1.grid(alpha=0.25)
ax2 = ax1.twinx()
color2 = "#dc2626"
ax2.plot(ths, sharpes, "s--", color=color2, lw=1.6, label="Sharpe")
ax2.set_ylabel("Sharpe", color=color2)
ax2.tick_params(axis="y", labelcolor=color2)
ax1.axvline(25, color="#9ca3af", ls=":", lw=1.2)
ax1.set_title("ADX 阈值敏感性：年化收益与 Sharpe 随阈值变化", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "adx_threshold_scan.png"), dpi=130)
plt.close()

print("ADX images done:", os.listdir(D))
print(f"stats  filter: ann={a_f*100:.1f}% sharpe={s_f:.2f}")
print(f"stats  naive:  ann={a_n*100:.1f}% sharpe={s_n:.2f}")
print(f"stats  bh:     ann={a_b*100:.1f}% sharpe={s_b:.2f}")
print(f"best threshold by sharpe: {ths[int(np.argmax(sharpes))]} -> {max(sharpes):.2f}")
