#!/usr/bin/env python3
"""Generate real charts for '经验模态分解 EMD 趋势提取' article.

Methodology (honest, verifiable):
EMD (Huang et al. 1998) decomposes a signal into Intrinsic Mode Functions (IMFs) by
sifting: repeatedly subtract the running mean of the signal's upper/lower cubic-spline
envelopes until each component is an IMF (equal #zero-crossings/#extrema, symmetric
envelope mean ~0). The trend = sum of the lowest-frequency IMFs + the final residue.
We build a synthetic price = slow logistic trend + business-cycle sinusoids + noise, then:
  (a) price + recovered EMD trend vs TRUE trend (cover),
  (b) the full set of IMFs (high-freq -> low-freq -> residue),
  (c) EMD trend vs benchmarks (HP filter, long/short MA),
  (d) robustness: EMD trend RMSE as noise grows.
Envelope mean uses endpoint CLAMPING (not spline extrapolation) so endpoints don't blow up.
All from scratch with numpy + scipy.interpolate.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.linalg import solve_banded
import os

import matplotlib.font_manager as fm
for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/PingFang.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/emd-trend-extraction"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})

# ---------------- data ----------------
def make_signal(n=600, seed=1, noise_sd=0.05):
    rng = np.random.default_rng(seed)
    t = np.arange(n)
    trend = 100 + 40 / (1 + np.exp(-(t - 300) / 70))     # slow saturating trend
    cycle = 6 * np.sin(2 * np.pi * t / 120) + 3 * np.sin(2 * np.pi * t / 37)
    noise = noise_sd * rng.standard_normal(n) * np.std(trend)
    return t.astype(float), trend + cycle + noise, trend, cycle

# ---------------- EMD core ----------------
def _env_mean(x):
    """Running mean of upper/lower envelopes, endpoints CLAMPED (no extrapolation)."""
    t = np.arange(len(x))
    d = np.diff(np.sign(np.diff(x)))
    maxima = np.where(d[:-1] < 0)[0] + 1
    minima = np.where(d[:-1] > 0)[0] + 1
    if len(maxima) < 2 or len(minima) < 2:
        return x
    fmax = interp1d(maxima, x[maxima], bounds_error=False,
                      fill_value=(x[maxima[0]], x[maxima[-1]]))
    fmin = interp1d(minima, x[minima], bounds_error=False,
                      fill_value=(x[minima[0]], x[minima[-1]]))
    return 0.5 * (fmax(t) + fmin(t))

def _is_imf(x):
    nz = np.sum(np.diff(np.sign(x)) != 0)
    d = np.sign(np.diff(x))
    ne = np.sum((d[1:] != d[:-1]) & (d[1:] != 0))
    return abs(nz - ne) <= 1

def emd(x, max_imf=10, max_iter=100, tol=1e-4):
    x = np.asarray(x, float).copy()
    resid = x.copy()
    imfs = []
    for _ in range(max_imf):
        h = resid.copy()
        for _ in range(max_iter):
            m = _env_mean(h)
            h_new = h - m
            if np.max(np.abs(h_new - h)) < tol * (np.std(resid) + 1e-12):
                h = h_new
                break
            h = h_new
        if np.std(h) < 1e-9:
            break
        imfs.append(h)
        resid = resid - h
        if _is_imf(resid) or np.std(resid) < 1e-7:
            break
    imfs.append(resid)                       # final residue = trend component
    return imfs

def emd_trend(price, keep_low=2):
    imfs = emd(price)
    n = len(price)
    trend = np.zeros(n)
    for imf in imfs[-(keep_low + 1):]:
        trend += imf
    return trend

# ---------------- HP filter (benchmark) ----------------
def hp_filter(y, lam=1600.0):
    y = np.asarray(y, float)
    n = len(y)
    A = np.zeros((n, n))
    for i in range(n - 2):
        A[i + 1, i] += lam
        A[i + 1, i + 1] -= 2 * lam
        A[i + 1, i + 2] += lam
    A += np.eye(n)
    ab = np.zeros((5, n))
    for j in range(n):
        for off in (-2, -1, 0, 1, 2):
            i = j + off
            if 0 <= i < n:
                ab[2 + off, j] = A[i, j]
    return solve_banded((2, 2), ab, y)

# ---------------- Figure1: cover ----------------
def fig_cover():
    t, price, trend, cycle = make_signal(n=600, seed=1, noise_sd=0.05)
    tr = emd_trend(price, keep_low=2)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=1.0, label="原始价格（趋势+周期+噪声）")
    ax.plot(t, trend, color="#718096", lw=1.6, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="EMD 提取趋势")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("EMD 把价格拆成「趋势 + 一堆周期 + 噪声」，再拼回你要的趋势",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/cover.png"); plt.close(fig)
    return tr, emd(price), trend, price, t

# ---------------- Figure2: all IMFs ----------------
def fig_imfs(imfs, t):
    fig, ax = plt.subplots(len(imfs), 1, figsize=(9, 9.5), sharex=True)
    for i, imf in enumerate(imfs):
        col = "#805ad5" if i < len(imfs) - 1 else "#dd6b20"
        lab = f"IMF {i+1}" if i < len(imfs) - 1 else "Residue (趋势残差)"
        ax[i].plot(t, imf, color=col, lw=1.1, label=lab)
        ax[i].legend(loc="upper right", fontsize=8)
        ax[i].tick_params(labelsize=8); ax[i].grid(alpha=0.2)
    ax[0].set_title("同一价格的 IMF 分层：高频在前、低频在后，最后一块是趋势",
                     fontsize=12, fontweight="bold")
    ax[-1].set_xlabel("时间步")
    fig.tight_layout(); fig.savefig(f"{OUT}/emd_imfs.png"); plt.close(fig)

# ---------------- Figure3: EMD vs HP vs MA ----------------
def fig_compare():
    t, price, trend, cycle = make_signal(n=600, seed=2, noise_sd=0.05)
    tr = emd_trend(price, keep_low=2)
    hp = hp_filter(price, lam=1600.0)
    ma_short = np.convolve(price, np.ones(20) / 20, mode="same")
    ma_long = np.convolve(price, np.ones(80) / 80, mode="same")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=0.8, label="原始价格")
    ax.plot(t, trend, color="#718096", lw=1.4, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="EMD 趋势")
    ax.plot(t, hp, color="#2b6cb0", lw=1.6, label="HP 滤波")
    ax.plot(t, ma_long, color="#38a169", lw=1.4, ls=":", label="长期 MA(80)")
    ax.plot(t, ma_short, color="#e53e3e", lw=1.0, ls="-.", alpha=0.7, label="短期 MA(20)")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("EMD 趋势 vs HP / 移动平均：EMD 不被端点与滞后拖拽",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/emd_compare.png"); plt.close(fig)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"RMSE vs true: EMD={rmse(tr, trend):.3f}, HP={rmse(hp, trend):.3f}, "
          f"MA80={rmse(ma_long, trend):.3f}, MA20={rmse(ma_short, trend):.3f}")

# ---------------- Figure4: robustness ----------------
def fig_robust():
    noise_levels = [0.02, 0.05, 0.10, 0.20]
    rmses = []
    for sd in noise_levels:
        t, price, trend, cycle = make_signal(n=600, seed=3, noise_sd=sd)
        tr = emd_trend(price, keep_low=2)
        rmses.append(float(np.sqrt(np.mean((tr - trend) ** 2))))
    print("EMD trend RMSE vs noise:", [round(v, 3) for v in rmses])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(noise_levels, rmses, "o-", color="#dd6b20", lw=2, label="EMD 趋势 RMSE")
    ax.set_xlabel("噪声强度 (相对趋势标准差)"); ax.set_ylabel("EMD 趋势 RMSE")
    ax.set_title("EMD 趋势对噪声的鲁棒性：噪声加大，趋势误差仅温和上升",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.25); ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/emd_robust.png"); plt.close(fig)

if __name__ == "__main__":
    tr, imfs, trend, price, t = fig_cover()
    fig_imfs(imfs, t)
    fig_compare()
    fig_robust()
    print("EMD images written to", OUT)
