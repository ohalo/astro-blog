#!/usr/bin/env python3
"""Generate real charts for '变分模态分解 VMD' article.

Methodology (honest, verifiable):
VMD (Dragomiretskiy & Zosso 2014) casts decomposition as a constrained
variational problem: find K modes u_k with center frequencies w_k that
collectively reconstruct the signal, each mode band-limited around w_k.
Solved by ADMM. We use the reference implementation `vmdpy` (the canonical
VMD) so the math is faithful and the modes sum back to the signal.

Synthetic price = slow logistic trend + two business-cycle sinusoids
(period 120 & 37) + noise, then:
  (a) cover: noisy price + VMD trend (lowest center-freq mode) vs TRUE trend,
  (b) the full set of K=5 modes (high-freq -> low-freq, last = trend),
  (c) VMD trend vs benchmarks (HP filter, long/short MA) with RMSE printed,
  (d) left: recovered dominant periods (via FFT peak) vs injected 120/37;
      right: VMD trend RMSE vs noise level.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import solve_banded
import os
import matplotlib.font_manager as fm
import vmdpy

for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/vmd-decomposition"
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
    c1 = 6 * np.sin(2 * np.pi * t / 120)                 # period 120
    c2 = 3 * np.sin(2 * np.pi * t / 37)                  # period 37
    cycle = c1 + c2
    noise = noise_sd * rng.standard_normal(n) * np.std(trend)
    return t.astype(float), trend + cycle + noise, trend, c1, c2

# ---------------- VMD (via vmdpy, normalized) ----------------
def vmd_modes(price, alpha=2000.0, K=5, DC=1):
    f = np.asarray(price, float)
    f_mean = f.mean()
    f0 = f - f_mean
    sstd = f0.std()
    if sstd < 1e-9:
        sstd = 1.0
    f_n = f0 / sstd                                   # vmdpy expects ~unit variance
    modes, _, omega_traj = vmdpy.VMD(f_n, alpha=alpha, tau=0.0, K=K,
                                     DC=DC, init=1, tol=1e-7)
    modes = modes * sstd                              # rescale back to price units
    return modes, f_mean

def vmd_trend(price, alpha=2000.0, K=5):
    modes, f_mean = vmd_modes(price, alpha=alpha, K=K)
    # trend = DC mode (vmdpy DC=1 puts it first); otherwise lowest-frequency mode.
    if K >= 1 and True:
        tr = modes[0] + f_mean
    return tr, modes, f_mean

def dom_period(m, n):
    """Dominant period of a mode via FFT peak (excludes DC)."""
    sp = np.abs(np.fft.fft(m))
    freqs = np.fft.fftfreq(n)
    sp[0] = 0
    dom = freqs[np.argmax(sp)]
    if abs(dom) < 1e-9:
        return np.inf
    return abs(1.0 / dom)

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
    t, price, trend, c1, c2 = make_signal(n=600, seed=1, noise_sd=0.05)
    tr, modes, _ = vmd_trend(price, alpha=2000.0, K=5)
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=1.0, label="原始价格（趋势+周期+噪声）")
    ax.plot(t, trend, color="#718096", lw=1.6, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="VMD 提取趋势（最低中心频率模态）")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("VMD 把价格拆成 K 个「窄带模态」，最低频那块就是趋势",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/cover.png"); plt.close(fig)
    return modes, t, price, trend

# ---------------- Figure2: all modes ----------------
def fig_modes(modes, t, price):
    K = len(modes)
    n = len(price)
    fig, ax = plt.subplots(K, 1, figsize=(9, 10.5), sharex=True)
    for i in range(K):
        per = dom_period(modes[i], n)
        if np.isfinite(per):
            lab = f"Mode {i+1}  (主导周期≈{per:.0f} 步)"
        else:
            lab = f"Mode {i+1}  (趋势 / 直流分量)"
        col = "#2b6cb0" if i > 0 else "#dd6b20"
        ax[i].plot(t, modes[i], color=col, lw=1.1, label=lab)
        ax[i].legend(loc="upper right", fontsize=8)
        ax[i].tick_params(labelsize=8); ax[i].grid(alpha=0.2)
    ax[0].set_title("同一价格的 VMD 模态分层：趋势在最底、周期在中、高频噪声在最顶",
                    fontsize=12, fontweight="bold")
    ax[-1].set_xlabel("时间步")
    fig.tight_layout(); fig.savefig(f"{OUT}/vmd_modes.png"); plt.close(fig)

# ---------------- Figure3: VMD vs HP vs MA ----------------
def fig_compare():
    t, price, trend, c1, c2 = make_signal(n=600, seed=2, noise_sd=0.05)
    tr, _, _ = vmd_trend(price, alpha=2000.0, K=5)
    hp = hp_filter(price, lam=1600.0)
    ma_short = np.convolve(price, np.ones(20) / 20, mode="same")
    ma_long = np.convolve(price, np.ones(80) / 80, mode="same")
    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.plot(t, price, color="#a0aec0", lw=0.8, label="原始价格")
    ax.plot(t, trend, color="#718096", lw=1.4, ls="--", label="真实趋势")
    ax.plot(t, tr, color="#dd6b20", lw=2.0, label="VMD 趋势")
    ax.plot(t, hp, color="#2b6cb0", lw=1.6, label="HP 滤波")
    ax.plot(t, ma_long, color="#38a169", lw=1.4, ls=":", label="长期 MA(80)")
    ax.plot(t, ma_short, color="#e53e3e", lw=1.0, ls="-.", alpha=0.7, label="短期 MA(20)")
    ax.set_xlabel("时间步"); ax.set_ylabel("价格")
    ax.set_title("VMD 趋势 vs HP / 移动平均：端点稳、低频模态干净",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.25)
    fig.tight_layout(); fig.savefig(f"{OUT}/vmd_compare.png"); plt.close(fig)
    def rmse(a, b):
        return float(np.sqrt(np.mean((a - b) ** 2)))
    print(f"[VMD] RMSE vs true: VMD={rmse(tr, trend):.3f}, "
          f"HP={rmse(hp, trend):.3f}, MA80={rmse(ma_long, trend):.3f}, "
          f"MA20={rmse(ma_short, trend):.3f}")

# ---------------- Figure4: freq recovery + noise robustness ----------------
def fig_freq_robust():
    true_periods = [120, 37]
    t, price, trend, c1, c2 = make_signal(n=600, seed=3, noise_sd=0.02)
    modes, _ = vmd_modes(price, alpha=2000.0, K=5, DC=1)
    n = len(price)
    # cycle modes = the two non-trend modes with dominant period in (10, 200)
    cand = []
    for i in range(1, len(modes)):
        per = dom_period(modes[i], n)
        if 10 <= per <= 200:
            cand.append(per)
    cand = sorted(cand)
    rec_periods = cand[:2] if len(cand) >= 2 else cand

    noise_levels = [0.02, 0.05, 0.10, 0.20]
    rmses = []
    for sd in noise_levels:
        tt, pp, tr2, _, _ = make_signal(n=600, seed=4, noise_sd=sd)
        trm, _, _ = vmd_trend(pp, alpha=2000.0, K=5)
        rmses.append(float(np.sqrt(np.mean((trm - tr2) ** 2))))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.6))
    x = np.arange(2)
    ax1.bar(x - 0.18, true_periods, width=0.36, color="#a0aec0", label="注入周期")
    ax1.bar(x + 0.18, rec_periods, width=0.36, color="#dd6b20", label="VMD 恢复周期")
    ax1.set_xticks(x); ax1.set_xticklabels(["周期 1", "周期 2"])
    ax1.set_ylabel("周期（步）"); ax1.set_title("VMD 把「隐藏的周期」直接估出来",
                                                 fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.25)
    ax2.plot(noise_levels, rmses, "o-", color="#dd6b20", lw=2, label="VMD 趋势 RMSE")
    ax2.set_xlabel("噪声强度 (相对趋势标准差)"); ax2.set_ylabel("VMD 趋势 RMSE")
    ax2.set_title("VMD 趋势对噪声的鲁棒性", fontsize=11, fontweight="bold")
    ax2.grid(alpha=0.25); ax2.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(f"{OUT}/vmd_freq_robust.png"); plt.close(fig)
    print(f"[VMD] true periods: {true_periods}")
    print(f"[VMD] recovered cycle periods: {[round(v,1) for v in rec_periods]}")
    print(f"[VMD] trend RMSE vs noise: {[round(v,3) for v in rmses]}")

if __name__ == "__main__":
    modes, t, price, trend = fig_cover()
    fig_modes(modes, t, price)
    fig_compare()
    fig_freq_robust()
    _, price0, trend0, _, _ = make_signal(n=600, seed=1, noise_sd=0.05)
    tr0, _, _ = vmd_trend(price0, alpha=2000.0, K=5)
    print(f"[VMD] cover trend RMSE vs true: "
          f"{np.sqrt(np.mean((tr0 - trend0)**2)):.3f}")
    print("VMD images written to", OUT)
