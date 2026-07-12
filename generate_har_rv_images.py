#!/usr/bin/env python3
"""
为文章「HAR-RV 已实现波动率模型：用分钟频数据预测明日波动」(har-rv-model)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 用「慢随机游走 + 快 AR(1)」两层构造带长记忆的日度方差，模拟波动率聚集与缓慢衰减；
  - 每天用 M=78 个 5 分钟收益还原已实现方差 RV_t = Σ r²，使其期望 ≈ 当日方差；
  - HAR-RV: RV_{t+1} = β0 + β1·RV_t + β2·mean(RV_{t-5:t-1}) + β3·mean(RV_{t-22:t-1}) + ε；
  - 与 naive(=RV_t) 及 AR(1) 基准比 OOS R²。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "har-rv-model")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "rv": "#2F4B7C", "lat": "#C44E52", "har": "#55A868",
     "naive": "#E8A33D", "mk": "#8172B3", "line": "#333333"}

# ============================================================
# 1) 合成数据：带长记忆的日度方差 + 日内 5 分钟还原 RV
# ============================================================
def build_data(T=2000, M=78, seed=20260712):
    rng = np.random.default_rng(seed)
    # 慢层：近似单位根 → 长记忆；快层：AR(1)
    slow = np.zeros(T)
    for t in range(1, T):
        slow[t] = slow[t - 1] + 0.012 * rng.normal()
    fast = np.zeros(T)
    for t in range(1, T):
        fast[t] = 0.5 * fast[t - 1] + 0.5 * rng.normal()
    logvar = np.log(0.0004) + 0.9 * slow + 0.3 * fast   # 日波动率≈1%~2%
    sig = np.sqrt(np.exp(logvar))

    rv = np.empty(T)
    daily_ret = np.empty(T)
    noise_floor = 1e-7
    for t in range(T):
        intra = rng.normal(0.0, sig[t] / np.sqrt(M), size=M)
        rv[t] = np.sum(intra ** 2) + noise_floor * abs(rng.normal())
        daily_ret[t] = np.sum(intra)
    return rv, daily_ret, sig ** 2

# ============================================================
# 2) HAR-RV 特征构造 + OLS
# ============================================================
def har_matrix(rv, win_w=5, win_m=22):
    n = len(rv)
    rows, y = [], []
    for t in range(1, n):                     # 目标 RV_t，特征用过去
        if t - 1 - win_m < 0:
            continue
        daily = rv[t - 1]
        weekly = rv[t - 1 - win_w:t - 1].mean()
        monthly = rv[t - 1 - win_m:t - 1].mean()
        rows.append([1.0, daily, weekly, monthly])
        y.append(rv[t])
    return np.array(rows), np.array(y)

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    s2 = resid @ resid / (len(y) - X.shape[1])
    return beta, resid, s2

def ar1_fit(y):
    yy = y[1:]; X = np.column_stack([np.ones_like(y[:-1]), y[:-1]])
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    return beta

# ============================================================
# 3) 图一：RV 序列的聚集 + 潜在方差
# ============================================================
def fig_rv_series(rv, sig2):
    last = 500
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.bar(np.arange(last), rv[-last:], width=0.8, color=C["rv"], alpha=0.75,
           label="realized variance $RV_t$")
    ax.plot(np.arange(last), sig2[-last:], color=C["lat"], lw=1.8, alpha=0.9,
            label=r"latent daily variance $\sigma_t^2$")
    ax.set_title("Realized variance clusters; HAR-RV reads it at three horizons", fontsize=11.5)
    ax.set_xlabel("day")
    ax.set_ylabel("variance")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "har_rv_series.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 4) 图二：RV 的自相关（长记忆慢衰减）
# ============================================================
def acf(x, max_lag=60):
    x = x - x.mean()
    n = len(x)
    denom = (x * x).sum()
    return np.array([(x[:-k] * x[k:]).sum() / denom for k in range(1, max_lag + 1)])

def fig_acf(rv, har_resid):
    a = acf(rv, 60)
    b = acf(har_resid, 60)
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    ax.plot(np.arange(1, 61), a, color=C["rv"], lw=2.2, label="ACF of $RV_t$ (long memory)")
    ax.plot(np.arange(1, 61), b, color=C["har"], lw=2.0, label="ACF of HAR-RV residuals")
    ax.axhline(0, color="#777", lw=0.8)
    ax.set_title("HAR-RV eats the slow-decaying autocorrelation", fontsize=11.5)
    ax.set_xlabel("lag (days)"); ax.set_ylabel("autocorrelation")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "har_acf.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 5) 图三：样本外预测对比（HAR-RV vs naive）
# ============================================================
def fig_forecast(test_actual, fc_har, fc_naive):
    L = 250
    idx = np.arange(L)
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    ax.plot(idx, test_actual[-L:], color=C["line"], lw=1.3, alpha=0.85, label="actual $RV_{t+1}$")
    ax.plot(idx, fc_har[-L:], color=C["har"], lw=1.6, label="HAR-RV forecast")
    ax.plot(idx, fc_naive[-L:], color=C["naive"], lw=1.3, ls="--", label="naive ($=RV_t$)")
    ax.set_title("Out-of-sample: HAR-RV tracks tomorrow's variance better", fontsize=11.5)
    ax.set_xlabel("test day"); ax.set_ylabel("variance")
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(True, color=C["grid"], lw=0.6); ax.set_axisbelow(True)
    fig.tight_layout()
    p = os.path.join(D, "har_forecast.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

# ============================================================
# 6) 图四：OOS R² 对比
# ============================================================
def fig_oos_r2(r2_har, r2_ar1):
    fig, ax = plt.subplots(figsize=(6.6, 4.0))
    names = ["naive\n(RV_t)", "AR(1)\non RV", "HAR-RV"]
    vals = [0.0, r2_ar1, r2_har]
    cols = [C["naive"], C["mk"], C["har"]]
    bars = ax.bar(names, vals, color=cols, alpha=0.9)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.2f}",
                ha="center", va="bottom", fontsize=9.5)
    ax.set_title("Out-of-sample $R^2$ vs naive benchmark", fontsize=11.5)
    ax.set_ylabel("OOS $R^2$")
    ax.grid(True, color=C["grid"], lw=0.6, axis="y"); ax.set_axisbelow(True)
    ax.set_ylim(0, max(vals) * 1.25 + 0.05)
    fig.tight_layout()
    p = os.path.join(D, "har_oos_r2.png")
    fig.savefig(p, dpi=130); plt.close(fig)
    return p

if __name__ == "__main__":
    rv, daily_ret, sig2 = build_data(T=2000, M=78)
    X, y = har_matrix(rv)
    n = len(y)
    cut = int(n * 0.7)
    beta, resid, _ = ols(X[:cut], y[:cut])
    # 训练集系数
    print("HAR-RV in-sample coefs: β0=%.3e β_d=%.3f β_w=%.3f β_m=%.3f"
          % (beta[0], beta[1], beta[2], beta[3]))
    # 样本外预测
    Xt, yt = X[cut:], y[cut:]
    fc_har = Xt @ beta
    fc_naive = Xt[:, 1]            # RV_t
    # AR(1) 基准（用训练集估）
    b0, b1 = ar1_fit(y[:cut])
    fc_ar1 = b0 + b1 * y[cut - 1: -1]
    if len(fc_ar1) > len(yt):
        fc_ar1 = fc_ar1[:len(yt)]

    def oos_r2(actual, fc):
        num = ((actual - fc) ** 2).sum()
        den = ((actual - fc_naive) ** 2).sum()
        return 1 - num / den

    r2_har = oos_r2(yt, fc_har)
    r2_ar1 = oos_r2(yt, fc_ar1)
    print(f"OOS R²: HAR-RV={r2_har:.3f}  AR(1)={r2_ar1:.3f}  naive=0.000")

    # 训练集拟合优度
    ss_res = (resid ** 2).sum(); ss_tot = ((y[:cut] - y[:cut].mean()) ** 2).sum()
    print(f"HAR-RV in-sample R²={1 - ss_res / ss_tot:.3f}")

    p1 = fig_rv_series(rv, sig2)
    p2 = fig_acf(rv, resid)
    p3 = fig_forecast(yt, fc_har, fc_naive)
    p4 = fig_oos_r2(r2_har, r2_ar1)
    print("saved:", p1); print("saved:", p2); print("saved:", p3); print("saved:", p4)
