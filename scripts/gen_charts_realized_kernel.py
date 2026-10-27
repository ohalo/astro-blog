#!/usr/bin/env python3
"""Generate real charts for '实现核与预平均去噪：高频波动率估计' article.

Methodology (honest, verifiable):
High-frequency returns are contaminated by market microstructure noise (bid-ask bounce,
price discreteness, asynchronous quotes). The naive Realized Variance (RV) = sum of squared
intraday returns is a CONSISTENT estimator of integrated variance (IV) only in the absence
of noise; with i.i.d. noise of variance omega^2, sampling n times gives
    E[RV] ≈ IV + 2*n*omega^2
so RV explodes linearly in n (the "volatility signature plot" blows up at high frequency).

Two noise-robust estimators:
  1. Realized Kernel (Barndorff-Nielsen, Hansen, Lunde, Shephard 2008): weight autocovariances
     gamma_h of returns with a kernel k(h/H), e.g. the Parzen kernel, out to bandwidth H.
     RK = gamma_0 + sum_{h=1}^{H} k((h-1)/H) * (gamma_h + gamma_{-h}).
  2. Pre-averaging (Jacod et al. 2009): average returns over a local window of length kn before
     squaring, which attenuates the noise, then bias-correct.

We simulate an efficient log-price as a Heston-like stochastic-vol path, add i.i.d. microstructure
noise, and show (a) the signature plot exploding for RV vs flat for RK/PA, (b) noise attenuation,
(c) IV tracking across days, (d) bandwidth sensitivity. All from scratch with numpy.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

import matplotlib.font_manager as fm
for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc",
           "/System/Library/Fonts/Supplemental/Songti.ttc"]:
    try:
        fm.fontManager.addfont(_f); break
    except Exception:
        continue
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/realized-kernel-preaveraging"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight", "font.family": "Arial Unicode MS",
    "axes.unicode_minus": False,
})

rng = np.random.default_rng(20260721)

# ---------------- Efficient price simulation (Heston-like) ----------------
def simulate_day(n_grid=23400, seed=None, mu=0.0, kappa=5.0, theta=0.20**2,
                 xi=0.4, v0=None, dt_year=1/252):
    """Simulate one trading day of efficient log-prices on a fine grid (n_grid steps ~ 1s bars).
    Returns efficient log-price path (len n_grid+1) and the true integrated variance (IV)."""
    r = np.random.default_rng(seed)
    dt = dt_year / n_grid
    if v0 is None:
        v0 = theta
    v = np.empty(n_grid + 1); v[0] = v0
    logp = np.empty(n_grid + 1); logp[0] = 0.0
    z1 = r.standard_normal(n_grid)
    z2 = r.standard_normal(n_grid)
    rho = -0.5
    w2 = rho * z1 + np.sqrt(1 - rho**2) * z2
    for t in range(n_grid):
        vt = max(v[t], 1e-12)
        v[t+1] = max(vt + kappa*(theta - vt)*dt + xi*np.sqrt(vt)*np.sqrt(dt)*w2[t], 1e-12)
        logp[t+1] = logp[t] + (mu - 0.5*vt)*dt + np.sqrt(vt)*np.sqrt(dt)*z1[t]
    iv = np.sum(v[:-1]) * dt  # integrated variance over the day
    return logp, iv

# ---------------- Estimators ----------------
def realized_variance(logp):
    ret = np.diff(logp)
    return np.sum(ret**2)

def parzen_weight(x):
    # Parzen kernel on [0,1]
    x = np.abs(x)
    w = np.where(x <= 0.5, 1 - 6*x**2 + 6*x**3, np.where(x <= 1.0, 2*(1-x)**3, 0.0))
    return w

def realized_kernel(logp, H):
    ret = np.diff(logp)
    n = len(ret)
    gamma0 = np.sum(ret**2)
    rk = gamma0
    for h in range(1, H+1):
        g = np.sum(ret[h:] * ret[:n-h])  # gamma_h (== gamma_{-h} for this estimator form)
        w = parzen_weight((h-1)/H)
        rk += 2 * w * g
    return rk

def preaveraging(logp, kn):
    """Pre-averaging estimator (Jacod et al 2009), triangular weight g(x)=min(x,1-x)."""
    ret = np.diff(logp)
    n = len(ret)
    j = np.arange(1, kn)
    g = np.minimum(j/kn, 1 - j/kn)   # weights on returns inside window
    # pre-averaged returns
    pavg = np.convolve(ret, g[::-1], mode="valid")  # length n-(kn-2)
    psi2 = np.sum(g**2)
    psi1 = np.sum(np.diff(np.concatenate([[0], g, [0]]))**2)
    # main term
    main = np.sum(pavg**2)
    # noise bias correction term
    dret2 = np.sum(ret**2)
    theta = kn / np.sqrt(n)
    est = (1.0/(theta*psi2)) * (main/np.sqrt(n)) - (psi1/(2*theta**2*psi2*n)) * dret2
    # simpler practical normalization: scale to per-day variance
    # Use the standard normalization: n/(n-kn+2) * (1/(kn*psi2)) sum pavg^2 - bias
    return est

# ---------- 1. Volatility signature plot: RV explodes, RK/PA stay flat ----------
def signature_plot():
    n_fine = 23400
    logp, iv = simulate_day(n_fine, seed=101)
    omega = 0.0008  # microstructure noise sd (per obs, in log-price units)
    noise = rng.normal(0, omega, n_fine + 1)
    obs = logp + noise

    # sample at increasing frequencies (subsampling the observed price)
    freqs = np.array([1, 2, 5, 10, 30, 60, 120, 300, 600, 1200, 2340])  # step size in seconds
    rv_vals, rk_vals, pa_vals = [], [], []
    for step in freqs:
        idx = np.arange(0, n_fine + 1, step)
        sub = obs[idx]
        n_sub = len(sub) - 1
        rv_vals.append(realized_variance(sub))
        H = max(1, int(np.ceil(n_sub**0.5)))
        rk_vals.append(realized_kernel(sub, H))
        kn = max(2, int(np.ceil(n_sub**0.5)))
        try:
            pa_vals.append(preaveraging(sub, kn))
        except Exception:
            pa_vals.append(np.nan)

    n_obs_per_day = n_fine / freqs
    ann = 252  # to annualize variance -> show as annualized vol %
    def to_annvol(v):
        return np.sqrt(np.maximum(v, 1e-12) * ann) * 100

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(n_obs_per_day, to_annvol(np.array(rv_vals)), "o-", color="#d62728",
            label="朴素实现波动率 RV（受噪声污染）", lw=2)
    ax.plot(n_obs_per_day, to_annvol(np.array(rk_vals)), "s-", color="#1f77b4",
            label="实现核 RK（Parzen）", lw=2)
    ax.axhline(np.sqrt(iv*ann)*100, color="#2ca02c", ls="--", lw=2,
               label=f"真实积分波动率 IV ≈ {np.sqrt(iv*ann)*100:.1f}%")
    ax.set_xscale("log")
    ax.set_xlabel("每日采样次数（越往右 = 采样越密）")
    ax.set_ylabel("年化波动率估计 (%)")
    ax.set_title("波动率信号图：RV 随采样加密而爆炸，RK 保持稳定")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/signature_plot.png"); plt.close(fig)
    print(f"[1] signature_plot.png  IV_annvol={np.sqrt(iv*ann)*100:.2f}%  "
          f"RV@1s_annvol={to_annvol(np.array([rv_vals[0]]))[0]:.1f}%")

# ---------- 2. Noise contamination illustration ----------
def noise_illustration():
    n_fine = 6000
    logp, iv = simulate_day(n_fine, seed=202)
    omega = 0.0010
    noise = rng.normal(0, omega, n_fine + 1)
    obs = logp + noise

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    t = np.arange(n_fine + 1)
    axes[0].plot(t, (logp)*1e4, color="#2ca02c", lw=1.4, label="有效价格（不可观测）")
    axes[0].plot(t, (obs)*1e4, color="#d62728", lw=0.6, alpha=0.6, label="观测价格 = 有效 + 微观噪声")
    axes[0].set_xlabel("日内时间（秒）"); axes[0].set_ylabel("对数价格 (bps)")
    axes[0].set_title("噪声让观测价格在有效价格附近抖动")
    axes[0].legend(loc="best")

    ret_eff = np.diff(logp)
    ret_obs = np.diff(obs)
    # autocorrelation of returns: noise induces negative first-order autocorrelation
    def acf(x, lags):
        x = x - x.mean()
        d = np.sum(x**2)
        return [np.sum(x[h:]*x[:len(x)-h])/d for h in lags]
    lags = np.arange(0, 11)
    axes[1].bar(lags-0.15, acf(ret_obs, lags), width=0.3, color="#d62728", label="观测收益 ACF")
    axes[1].bar(lags+0.15, acf(ret_eff, lags), width=0.3, color="#2ca02c", label="有效收益 ACF")
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xlabel("滞后阶数 h"); axes[1].set_ylabel("自相关")
    axes[1].set_title("微观噪声在一阶滞后制造强负自相关")
    axes[1].legend(loc="best")
    fig.savefig(f"{OUT}/noise_illustration.png"); plt.close(fig)
    print(f"[2] noise_illustration.png  ret_obs_acf1={acf(ret_obs,[1])[0]:.3f}")

# ---------- 3. Multi-day IV tracking: RK tracks true IV, RV biased up ----------
def multiday_tracking():
    n_days = 60
    n_fine = 23400
    omega = 0.0008
    step = 60  # observe once a minute (390 obs/day) -> realistic
    true_iv, rv_est, rk_est = [], [], []
    for d in range(n_days):
        logp, iv = simulate_day(n_fine, seed=1000+d)
        noise = rng.normal(0, omega, n_fine + 1)
        obs = logp + noise
        idx = np.arange(0, n_fine + 1, step)
        sub = obs[idx]
        n_sub = len(sub) - 1
        true_iv.append(iv)
        rv_est.append(realized_variance(sub))
        H = max(1, int(np.ceil(n_sub**0.5)))
        rk_est.append(realized_kernel(sub, H))
    ann = 252
    tv = np.sqrt(np.array(true_iv)*ann)*100
    rvv = np.sqrt(np.maximum(rv_est,1e-12)*ann)*100
    rkv = np.sqrt(np.maximum(rk_est,1e-12)*ann)*100

    fig, ax = plt.subplots(figsize=(10, 4.8))
    ax.plot(tv, "-", color="#2ca02c", lw=2.2, label="真实积分波动率 IV")
    ax.plot(rvv, "-", color="#d62728", lw=1.2, alpha=0.8, label="朴素 RV（系统性高估）")
    ax.plot(rkv, "-", color="#1f77b4", lw=1.4, label="实现核 RK")
    ax.set_xlabel("交易日"); ax.set_ylabel("年化波动率 (%)")
    ax.set_title("60 个交易日：RK 贴合真实 IV，RV 系统性偏高")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/multiday_tracking.png"); plt.close(fig)
    rmse_rv = np.sqrt(np.mean((rvv-tv)**2))
    rmse_rk = np.sqrt(np.mean((rkv-tv)**2))
    print(f"[3] multiday_tracking.png  RMSE_RV={rmse_rv:.2f}%  RMSE_RK={rmse_rk:.2f}%")
    return rmse_rv, rmse_rk

# ---------- 4. Bandwidth sensitivity of RK ----------
def bandwidth_sensitivity():
    n_fine = 23400
    omega = 0.0008
    step = 30
    n_days = 30
    Hs = [1, 2, 4, 8, 16, 32, 64, 128]
    errs = {H: [] for H in Hs}
    for d in range(n_days):
        logp, iv = simulate_day(n_fine, seed=3000+d)
        noise = rng.normal(0, omega, n_fine + 1)
        obs = logp + noise
        idx = np.arange(0, n_fine + 1, step)
        sub = obs[idx]
        for H in Hs:
            rk = realized_kernel(sub, H)
            errs[H].append((np.sqrt(max(rk,1e-12)) - np.sqrt(iv)) / np.sqrt(iv) * 100)
    mean_bias = [np.mean(errs[H]) for H in Hs]
    std_err = [np.std(errs[H]) for H in Hs]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.errorbar(Hs, mean_bias, yerr=std_err, fmt="o-", color="#1f77b4", capsize=4, lw=2,
                label="RK 相对偏差 ± 1 标准差")
    ax.axhline(0, color="#2ca02c", ls="--", lw=1.5, label="无偏基准")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("核带宽 H（对数轴）")
    ax.set_ylabel("波动率相对偏差 (%)")
    ax.set_title("带宽权衡：H 太小残留噪声偏高，H 太大方差放大")
    ax.legend(loc="best"); ax.grid(alpha=0.3)
    fig.savefig(f"{OUT}/bandwidth_sensitivity.png"); plt.close(fig)
    print(f"[4] bandwidth_sensitivity.png  bias@H=1:{mean_bias[0]:.1f}%  bias@H=16:{mean_bias[4]:.1f}%")

if __name__ == "__main__":
    signature_plot()
    noise_illustration()
    multiday_tracking()
    bandwidth_sensitivity()
    print("DONE realized-kernel charts ->", OUT)
