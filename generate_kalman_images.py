#!/usr/bin/env python3
"""
为文章「卡尔曼滤波在动态对冲中的应用」(kalman-dynamic-hedging) 生成真实配图。
全部用 matplotlib + numpy 从原理渲染，非占位图。

图表：
  1. kalman_beta.png        动态对冲比(β)随时间演化：真实 / Kalman / 滚动OLS / 静态OLS
  2. kalman_residual.png    对冲残差(持仓净值)对比：静态OLS vs Kalman 动态对冲
  3. kalman_pnl_dist.png    对冲后组合 PnL 分布：动态对冲显著降低方差与尾部
  4. kalman_fairvalue.png   Kalman 对含噪「公平价值」的跟踪 + 95% 置信带
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
D = os.path.join(BASE, "kalman-dynamic-hedging")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)


# ============================================================
# 1) 合成一对协整标的：x = 标的(如指数)，y = 其可套利对手(如期货/ETF)
#    真实对冲比 β_t 缓慢时变（regime 漂移）
# ============================================================
T = 1500
t = np.arange(T)
# x 用带漂移的随机游走模拟价格
x = 100 + np.cumsum(np.random.normal(0.02, 1.0, T))
# 真实时变对冲比：基线 + 正弦慢漂移 + 阶跃（regime 切换）
beta_true = 0.85 + 0.25 * np.sin(t / 220.0) + 0.30 * (t > 950)
mu_true = 5.0 + 0.02 * t + 1.5 * (t > 700)
# 观测噪声
noise = np.random.normal(0, 1.2, T)
y = beta_true * x + mu_true + noise


# ============================================================
# 2) 从零实现卡尔曼滤波：状态 θ_t=[β_t, μ_t]，随机游走演化
# ============================================================
def kalman_filter(y, x, R=1.4, Qb=5e-6, Qm=5e-5):
    n = len(y)
    theta = np.zeros((n, 2))      # [beta, mu]
    P = np.zeros((n, 2, 2))
    theta[0] = [0.5, 0.0]
    P[0] = np.eye(2) * 10.0
    Q = np.diag([Qb, Qm])
    for k in range(1, n):
        theta_pred = theta[k - 1].copy()
        P_pred = P[k - 1] + Q
        H = np.array([x[k], 1.0])
        S = H @ P_pred @ H + R
        K = P_pred @ H / S
        innov = y[k] - H @ theta_pred
        theta[k] = theta_pred + K * innov
        P[k] = (np.eye(2) - np.outer(K, H)) @ P_pred
    return theta[:, 0], theta[:, 1]


beta_kf, mu_kf = kalman_filter(y, x)


# ============================================================
# 3) 对照组：静态 OLS 全样本 + 滚动 OLS(窗口=120)
# ============================================================
def rolling_ols(y, x, W=120):
    n = len(y)
    beta = np.full(n, np.nan)
    for k in range(W, n):
        yy = y[k - W:k]; xx = x[k - W:k]
        b = np.polyfit(xx, yy, 1)
        beta[k] = b[0]
    # 前 W 段用全样本近似填充，避免空窗
    b0 = np.polyfit(x[:n], y[:n], 1)[0]
    beta[:W] = b0
    return beta

beta_roll = rolling_ols(y, x, 120)
b_static, a_static = np.polyfit(x, y, 1)
beta_static = np.full(T, b_static)
mu_static = np.full(T, a_static)


# ============================================================
# 图1：动态对冲比演化
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(t, beta_true, color="#2ca02c", lw=2.0, label="真实 β_t（含 regime 切换）", alpha=0.85)
ax.plot(t, beta_kf, color="#1f77b4", lw=1.6, label="卡尔曼滤波 β̂_t（动态对冲）")
ax.plot(t, beta_roll, color="#ff7f0e", lw=1.2, label="滚动OLS β（窗口120）", alpha=0.8)
ax.axhline(b_static, color="#d62728", ls="--", lw=1.6, label=f"静态OLS β={b_static:.3f}（固定）")
ax.set_xlabel("交易日 t", fontsize=11)
ax.set_ylabel("对冲比 β（每单位 x 需卖空的 y 数量）", fontsize=11)
ax.set_title("动态对冲比 β_t 随时间演化：固定对冲比会系统性偏离", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kalman_beta.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图2：对冲残差（持有「多 y + 空 βx」中性组合）净值对比
# ============================================================
res_static = y - beta_static * x - mu_static
res_kf = y - beta_kf * x - mu_kf
# 用累计残差近似对冲组合的净值轨迹
cum_static = np.cumsum(res_static)
cum_kf = np.cumsum(res_kf)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(t, cum_static, color="#d62728", lw=1.3, label=f"静态OLS对冲残差累计（σ={res_static.std():.2f}）")
ax.plot(t, cum_kf, color="#1f77b4", lw=1.3, label=f"卡尔曼动态对冲残差累计（σ={res_kf.std():.2f}）")
ax.set_xlabel("交易日 t", fontsize=11)
ax.set_ylabel("对冲组合累计残差", fontsize=11)
ax.set_title("对冲残差轨迹：动态对冲把波动压得更扁、更稳", fontsize=13.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kalman_residual.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图3：对冲后组合 PnL 分布（日度残差）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.6))
bins = np.linspace(-8, 8, 70)
ax.hist(res_static, bins=bins, alpha=0.5, density=True, color="#d62728",
        label=f"静态OLS对冲 (σ={res_static.std():.2f})")
ax.hist(res_kf, bins=bins, alpha=0.5, density=True, color="#1f77b4",
        label=f"卡尔曼动态对冲 (σ={res_kf.std():.2f})")
ax.set_xlabel("对冲组合日度残差（PnL）", fontsize=11)
ax.set_ylabel("密度", fontsize=11)
ax.set_title("对冲后 PnL 分布：动态对冲显著收窄方差、削减尾部", fontsize=13, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kalman_pnl_dist.png"), dpi=150, bbox_inches="tight")
plt.close()


# ============================================================
# 图4：Kalman 对含噪公平价值的跟踪 + 95% 置信带（滤波器直观演示）
# ============================================================
# 构造一个被噪声淹没的「真实公平价值」信号
true_fv = 50 + 8 * np.sin(t / 90.0) + 0.01 * t
obs = true_fv + np.random.normal(0, 3.0, T)   # 噪声观测

def kalman_level(obs, R=9.0, Q=0.05):
    n = len(obs)
    xhat = np.zeros(n); P = np.zeros(n)
    xhat[0] = obs[0]; P[0] = 10.0
    band = np.zeros(n)
    for k in range(1, n):
        x_pred = xhat[k - 1]; P_pred = P[k - 1] + Q
        S = P_pred + R
        K = P_pred / S
        xhat[k] = x_pred + K * (obs[k] - x_pred)
        P[k] = (1 - K) * P_pred
        band[k] = 1.96 * np.sqrt(P[k])
    return xhat, band

fv_hat, fv_band = kalman_level(obs)
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(t, obs, color="#bbbbbb", lw=0.8, alpha=0.6, label="噪声观测值")
ax.plot(t, true_fv, color="#2ca02c", lw=1.8, label="真实公平价值")
ax.plot(t, fv_hat, color="#1f77b4", lw=1.8, label="Kalman 估计")
ax.fill_between(t, fv_hat - fv_band, fv_hat + fv_band, color="#1f77b4", alpha=0.18,
                label="95% 置信带")
ax.set_xlabel("时间 t", fontsize=11)
ax.set_ylabel("公平价值", fontsize=11)
ax.set_title("卡尔曼滤波对含噪信号的跟踪：自动在「跟得住」与「抗噪声」间权衡", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "kalman_fairvalue.png"), dpi=150, bbox_inches="tight")
plt.close()


print("✅ 卡尔曼滤波配图生成完成：", sorted(os.listdir(D)))
print(f"   残差σ: 静态OLS={res_static.std():.3f}  Kalman={res_kf.std():.3f}  方差下降={1-(res_kf.std()/res_static.std())**2:.1%}")
