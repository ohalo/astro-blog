#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""DCCA 去趋势互相关分析 配图生成"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for name in ["PingFang SC", "Heiti TC", "Arial Unicode MS", "STSong"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/dcca-cross-correlation"
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GRAY, GREEN, ORANGE = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60", "#e67e22"
rng = np.random.default_rng(20260726)


def integrate(x):
    return np.cumsum(x - x.mean())


def dcca(x, y, scales):
    """计算两条序列在多个尺度下的 DCCA 协方差 F2_xy 与去趋势方差 F2_xx, F2_yy。"""
    X, Y = integrate(x), integrate(y)
    N = len(X)
    Fxy, Fxx, Fyy = [], [], []
    for s in scales:
        nseg = N // s
        cov_xy, var_x, var_y = [], [], []
        t = np.arange(s)
        A = np.vstack([t, np.ones(s)]).T
        for v in range(nseg):
            xs = X[v * s:(v + 1) * s]
            ys = Y[v * s:(v + 1) * s]
            # 线性去趋势
            bx = np.linalg.lstsq(A, xs, rcond=None)[0]
            by = np.linalg.lstsq(A, ys, rcond=None)[0]
            rx = xs - A @ bx
            ry = ys - A @ by
            cov_xy.append(np.mean(rx * ry))
            var_x.append(np.mean(rx * rx))
            var_y.append(np.mean(ry * ry))
        Fxy.append(np.mean(cov_xy))
        Fxx.append(np.mean(var_x))
        Fyy.append(np.mean(var_y))
    return np.array(Fxy), np.array(Fxx), np.array(Fyy)


# ============ 图1：两条非平稳价格序列（共享趋势 + 各自噪声）============
N = 4000
# 共同的长期慢趋势（非平稳）
common = np.cumsum(rng.normal(0, 1, N)) * 0.4
# 两个市场各自的特质游走
idio_a = np.cumsum(rng.normal(0, 1, N)) * 0.9
idio_b = np.cumsum(rng.normal(0, 1, N)) * 0.9
price_a = 100 + common + idio_a
price_b = 100 + common + idio_b

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(price_a, color=BLUE, lw=1.1, label="市场 A（非平稳价格）")
ax.plot(price_b, color=RED, lw=1.1, label="市场 B（非平稳价格）")
ax.set_title("两条共享慢趋势的非平稳价格序列", fontsize=14, fontweight="bold")
ax.set_xlabel("时间")
ax.set_ylabel("价格")
ax.legend(loc="best", framealpha=0.9)
ax.grid(alpha=0.25)
ax.text(0.02, 0.04, "皮尔逊相关会被共同趋势夸大——分不清是真联动还是伪相关",
        transform=ax.transAxes, fontsize=9, color=GRAY,
        bbox=dict(boxstyle="round", fc="white", ec=GRAY, alpha=0.8))
fig.tight_layout()
fig.savefig(f"{OUT}/dcca-nonstationary-series.png", dpi=130)
plt.close(fig)

# ============ 图2：DCCA 系数 vs 尺度（三种依赖场景）============
scales = np.unique(np.floor(np.logspace(np.log10(10), np.log10(N // 8), 22)).astype(int))
ra = np.diff(price_a)
rb = np.diff(price_b)

# 场景一：真实共享趋势（上面的序列）
Fxy, Fxx, Fyy = dcca(ra, rb, scales)
rho_shared = Fxy / np.sqrt(Fxx * Fyy)

# 场景二：两条完全独立的游走差分
ia = rng.normal(0, 1, N - 1)
ib = rng.normal(0, 1, N - 1)
Fxy2, Fxx2, Fyy2 = dcca(ia, ib, scales)
rho_indep = Fxy2 / np.sqrt(Fxx2 * Fyy2)

# 场景三：仅在大尺度耦合（短期独立、长期同向）
long_common = np.repeat(rng.normal(0, 1, (N - 1) // 40 + 1), 40)[:N - 1]
ca = rng.normal(0, 1, N - 1) + 0.0 * long_common
cb = rng.normal(0, 1, N - 1)
ca_scale = rng.normal(0, 1, N - 1) + 1.2 * long_common
cb_scale = rng.normal(0, 1, N - 1) + 1.2 * long_common
Fxy3, Fxx3, Fyy3 = dcca(ca_scale, cb_scale, scales)
rho_scale = Fxy3 / np.sqrt(Fxx3 * Fyy3)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.semilogx(scales, rho_shared, "o-", color=BLUE, lw=1.6, ms=5, label="共享趋势：全尺度正相关")
ax.semilogx(scales, rho_scale, "s-", color=GREEN, lw=1.6, ms=5, label="仅大尺度耦合：短期弱、长期强")
ax.semilogx(scales, rho_indep, "^-", color=GRAY, lw=1.6, ms=5, label="独立序列：围绕 0 波动")
ax.axhline(0, color="black", lw=0.8, ls="--")
ax.set_title("DCCA 相关系数随时间尺度的变化", fontsize=14, fontweight="bold")
ax.set_xlabel("时间尺度 s（对数轴）")
ax.set_ylabel(r"DCCA 系数 $\rho_{DCCA}(s)$")
ax.set_ylim(-0.4, 1.05)
ax.legend(loc="center left", framealpha=0.9)
ax.grid(alpha=0.25, which="both")
fig.tight_layout()
fig.savefig(f"{OUT}/dcca-coefficient-scale.png", dpi=130)
plt.close(fig)

# ============ 图3：DCCA vs 皮尔逊相关（对非平稳的鲁棒性对比）============
# 逐步增加共同趋势强度，观察两种度量对"伪相关"的反应
trend_strengths = np.linspace(0, 1.5, 12)
pearson_vals, dcca_vals = [], []
mid_scale_idx = np.argmin(np.abs(scales - 60))
for g in trend_strengths:
    ct = np.cumsum(rng.normal(0, 1, N)) * g
    pa = ct + np.cumsum(rng.normal(0, 1, N)) * 0.9
    pb = ct + np.cumsum(rng.normal(0, 1, N)) * 0.9
    # 皮尔逊：直接对非平稳"价格"求相关（错误做法，演示陷阱）
    pearson_vals.append(np.corrcoef(pa, pb)[0, 1])
    # DCCA：对差分做尺度=60 的去趋势互相关
    fx, fxx, fyy = dcca(np.diff(pa), np.diff(pb), [scales[mid_scale_idx]])
    dcca_vals.append((fx / np.sqrt(fxx * fyy))[0])

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(trend_strengths, pearson_vals, "o-", color=RED, lw=1.7, ms=5,
        label="皮尔逊相关（直接对价格）")
ax.plot(trend_strengths, dcca_vals, "s-", color=BLUE, lw=1.7, ms=5,
        label="DCCA 系数（尺度 s=60）")
ax.set_title("共同趋势越强，皮尔逊被夸大越多，DCCA 保持稳健", fontsize=13.5, fontweight="bold")
ax.set_xlabel("共同趋势强度")
ax.set_ylabel("相关度量")
ax.set_ylim(-0.1, 1.05)
ax.legend(loc="center right", framealpha=0.9)
ax.grid(alpha=0.25)
ax.text(0.03, 0.5, "两条真实相互独立的特质序列\n只因共享趋势就被皮尔逊判成高相关",
        transform=ax.transAxes, fontsize=9, color=GRAY,
        bbox=dict(boxstyle="round", fc="#fff5f5", ec=RED, alpha=0.85))
fig.tight_layout()
fig.savefig(f"{OUT}/dcca-vs-pearson.png", dpi=130)
plt.close(fig)

# ============ 图4：滚动 DCCA 系数捕捉危机期联动跃升 ============
M = 3000
regime = np.ones(M) * 0.15
regime[1200:1700] = 0.85   # 危机期：跨市场联动骤升
shared = np.cumsum(rng.normal(0, 1, M))
xa = np.zeros(M)
xb = np.zeros(M)
for t in range(M):
    c = regime[t]
    common_shock = rng.normal(0, 1)
    xa[t] = c * common_shock + np.sqrt(1 - c**2) * rng.normal(0, 1)
    xb[t] = c * common_shock + np.sqrt(1 - c**2) * rng.normal(0, 1)

win = 250
s_fixed = 40
roll = np.full(M, np.nan)
for t in range(win, M):
    seg_a = xa[t - win:t]
    seg_b = xb[t - win:t]
    fx, fxx, fyy = dcca(seg_a, seg_b, [s_fixed])
    roll[t] = (fx / np.sqrt(fxx * fyy))[0]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                               gridspec_kw={"height_ratios": [1, 1.3]})
ax1.axvspan(1200, 1700, color=RED, alpha=0.12)
ax1.plot(np.cumsum(xa), color=BLUE, lw=0.8, label="市场 A")
ax1.plot(np.cumsum(xb), color=ORANGE, lw=0.8, label="市场 B")
ax1.set_ylabel("累积序列")
ax1.legend(loc="upper left", fontsize=9, framealpha=0.9)
ax1.set_title("滚动 DCCA 系数捕捉危机期的跨市场联动跃升", fontsize=13.5, fontweight="bold")
ax1.grid(alpha=0.2)

ax2.axvspan(1200, 1700, color=RED, alpha=0.12, label="危机期（真实高联动）")
ax2.plot(roll, color=GREEN, lw=1.4)
ax2.axhline(0.15, color=GRAY, ls="--", lw=1, label="平静期基线 ≈0.15")
ax2.axhline(0.85, color=RED, ls=":", lw=1, label="危机期真值 ≈0.85")
ax2.set_ylabel(r"滚动 $\rho_{DCCA}$ (s=40)")
ax2.set_xlabel("时间")
ax2.set_ylim(-0.2, 1.0)
ax2.legend(loc="upper left", fontsize=8.5, framealpha=0.9, ncol=2)
ax2.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(f"{OUT}/dcca-rolling-regime.png", dpi=130)
plt.close(fig)

print("DCCA images done ->", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
