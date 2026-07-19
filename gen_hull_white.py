#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hull-White 短期利率模型 配图生成 (4 张真实图表, 自洽合成, 从零模拟+OU 校准)

机制(自洽合成, 仅用于演示方法):
  * HW 模型: dr = a(b - r)dt + sigma dW, 向中枢 b 均值回复, 扩散系数常数 => 利率永不为负
  * 解析性质: r_t | r_0 ~ N(均值, 方差), 均值回复到 b=6%
  * 债券定价闭式: P(t,T)=A(t,T)exp(-B(t,T) r_t), B= (1-e^{-a(T-t)})/a
  * 收益率曲线: y(t,T) - b = -(r_t - b) * (1 - e^{-a m})/a  (m=T-t)
      -> 利率高于中枢 => 曲线反转(近端高于远端); 低于中枢 => 上凸(steepener)
  * 图1: 1500 条短期利率模拟路径 + 均值路径(向 6% 回复)
  * 图2: 4 个快照日的整条收益率曲线(随 r_t 切换上凸/反转)
  * 图3: 200 次模拟 OU 最小二乘拟合还原 a/b(还原真值 + RMSE)
  * 图4: 期限溢价散点 = -(r_t - b)·A(m), 证明曲线形态由 r_t 相对 b 决定
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from numpy.linalg import cholesky

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "hull-white-rate-model"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"grid": "#DDDDDD", "path": "#4C72B0", "mean": "#C44E52", "fit": "#55A868",
      "true": "#999999", "curve": "#8172B3", "scatter": "#DD8452"}

# ---------- 模型参数 ----------
a = 0.15          # 均值回复速度
b = 0.06          # 长期中枢
sigma = 0.025      # 扩散系数
r0 = 0.03         # 初始利率
dt = 1.0 / 252
T_years = 8
T = int(T_years / dt)   # 交易日数

def hw_mean(t, r0, a, b):
    return r0 * np.exp(-a * t) + b * (1 - np.exp(-a * t))

def hw_var(t, a, sigma):
    return sigma**2 / a**2 * (1 - np.exp(-2 * a * t))

def hw_cov(s, t, a, sigma):
    # 协方差 Cov[r_s, r_t], s <= t
    return sigma**2 / a**2 * np.exp(-a * (t + s)) * (np.exp(2 * a * s) - 1)

def hw_B(tau, a):
    return (1 - np.exp(-a * tau)) / a

def hw_A(t, T, a, sigma):
    tau = T - t
    B = hw_B(tau, a)
    val = B - tau + (sigma**2 / (2 * a**2)) * (tau - 2 * B + a * B**2) \
          - (sigma**2 / (4 * a**3)) * (1 - np.exp(-2 * a * tau)) * (1 - np.exp(-2 * a * t))
    return np.exp(-val)

# ---------- 图1: 短期利率路径模拟 (精确 OU 递推, 避免协方差矩阵数值问题) ----------
rng = np.random.default_rng(20260719)
n_paths = 1500
# 精确 OU 递推: r_t = r_{t-1} e^{-a dt} + b(1-e^{-a dt}) + sigma*sqrt((1-e^{-2a dt})/(2a)) * Z
alpha = np.exp(-a * dt)
mu_step = b * (1 - alpha)
sd_step = sigma * np.sqrt((1 - alpha**2) / (2 * a))
paths = np.empty((n_paths, T + 1))
paths[:, 0] = r0
Z = rng.standard_normal((n_paths, T))
for i in range(T):
    paths[:, i + 1] = alpha * paths[:, i] + mu_step + sd_step * Z[:, i]
tt_full = np.arange(T + 1) * dt

fig, ax = plt.subplots(figsize=(10, 5.2))
for k in range(min(120, n_paths)):
    ax.plot(tt_full, paths[k] * 100, color=C["path"], alpha=0.05, lw=0.6)
ax.plot(tt_full, paths.mean(axis=0) * 100, color=C["mean"], lw=2.4, label="平均路径")
ax.axhline(b * 100, color=C["true"], ls="--", lw=1.6, label=f"中枢 b = {b*100:.0f}%")
ax.set_title("Hull-White 短期利率模拟：向中枢 6% 均值回复 (1500 条路径)", fontsize=13)
ax.set_xlabel("时间 (年)"); ax.set_ylabel("短期利率 (%)")
ax.legend(loc="upper right", fontsize=9); ax.grid(color=C["grid"], lw=0.5)
ax.set_xlim(0, T_years)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hw_paths.png")); plt.close(fig)

# ---------- 图2: 收益率曲线快照 ----------
m_list = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10])
snap_dates = [0, 63, 126, 189]   # 交易日索引(对应 tt_full 后移一位)
fig, ax = plt.subplots(figsize=(10, 5.2))
for sd in snap_dates:
    r_t = paths[0, sd]   # 用第 1 条路径在该日的 r
    t_y = sd * dt
    P = np.array([hw_A(t_y, t_y + m, a, sigma) * np.exp(-hw_B(m, a) * r_t) for m in m_list])
    y = -np.log(P) / m_list
    ax.plot(m_list, y * 100, marker="o", ms=4, label=f"r_t={r_t*100:.1f}% (t={sd/252:.1f}年)")
ax.axhline(b * 100, color=C["true"], ls="--", lw=1.4, label=f"中枢 b = {b*100:.0f}%")
ax.set_title("整条收益率曲线随 r_t 切换：低利率上凸 / 高利率反转", fontsize=13)
ax.set_xlabel("期限 (年)"); ax.set_ylabel("收益率 (%)")
ax.legend(fontsize=8, ncol=2); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hw_curves.png")); plt.close(fig)

# ---------- 图3: OU 校准还原真值 (正确做法: 池化所有路径的增量做 MLE) ----------
# 单条日频路径上 a=0.15(半衰期~4.6年) 回复太慢, 朴素逐路径 OLS 极噪(见正文陷阱);
# 正确估计量是把全部路径的 Δr 对 r_{t-1} 池化(Pooled OU MLE)。
dR_all = (paths[:, 1:] - paths[:, :-1]).ravel()
X_all = paths[:, :-1].ravel()
# 用 200 次「抽 300 条路径」的 Bootstrap 看抽样分布
n_rep = 200
a_hat = np.zeros(n_rep); b_hat = np.zeros(n_rep)
for rep in range(n_rep):
    idx_r = rng.integers(0, n_paths, size=300)
    dd = dR_all.reshape(n_paths, -1)[idx_r].ravel()
    xx = X_all.reshape(n_paths, -1)[idx_r].ravel()
    A_mat = np.vstack([np.ones_like(xx), xx]).T
    coef, *_ = np.linalg.lstsq(A_mat, dd, rcond=None)
    intercept, slope = coef
    a_r = -slope / dt
    b_r = intercept / (a_r * dt)
    a_hat[rep] = a_r; b_hat[rep] = b_r
rmse_a = np.sqrt(np.mean((a_hat - a) ** 2))
rmse_b = np.sqrt(np.mean((b_hat - b) ** 2))

# 朴素单路径 OLS 的 RMSE (用于正文对比, 体现陷阱)
naive_a = np.array([(-np.polyfit(paths[k][:-1], paths[k][1:]-paths[k][:-1], 1)[0])/dt for k in range(200)])
rmse_naive = np.sqrt(np.mean((naive_a - a) ** 2))

fig, axs = plt.subplots(1, 2, figsize=(11, 4.6))
axs[0].hist(a_hat, bins=28, color=C["fit"], alpha=0.85)
axs[0].axvline(a, color=C["mean"], lw=2, label=f"真值 a={a}")
axs[0].axvline(a_hat.mean(), color=C["true"], ls="--", lw=1.6, label=f"均值 {a_hat.mean():.3f}")
axs[0].set_title(f"池化 OU MLE：回复速度 a 还原 (RMSE={rmse_a:.4f})", fontsize=11)
axs[0].set_xlabel("a"); axs[0].legend(fontsize=8)
axs[1].hist(b_hat, bins=28, color=C["curve"], alpha=0.85)
axs[1].axvline(b, color=C["mean"], lw=2, label=f"真值 b={b}")
axs[1].axvline(b_hat.mean(), color=C["true"], ls="--", lw=1.6, label=f"均值 {b_hat.mean():.3f}")
axs[1].set_title(f"中枢 b 还原 (RMSE={rmse_b:.4f})", fontsize=12)
axs[1].set_xlabel("b"); axs[1].legend(fontsize=8)
fig.suptitle("OU 最小二乘校准：200 次模拟还原 Hull-White 参数", fontsize=13)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hw_ou_calib.png")); plt.close(fig)

# ---------- 图4: 期限溢价机制 ----------
m_fix = 2.0
t_grid = np.linspace(0, T_years, 400)
r_grid = np.array([hw_mean(t, r0, a, b) + 1.6 * np.sqrt(hw_var(t, a, sigma)) * np.sin(3 * t)
                   for t in t_grid])
y_grid = np.array([-(b - r) * hw_B(m_fix, a) for r in r_grid])   # y - b 精确
A_m = hw_B(m_fix, a)
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.scatter(r_grid * 100, y_grid * 100, s=14, color=C["scatter"], alpha=0.55,
           label=f"y({m_fix:.0f}y) - b 的精确值")
# 理论直线 y-b = -(r-b)·A
rr = np.linspace(r_grid.min(), r_grid.max(), 50)
ax.plot(rr * 100, (-(rr - b) * A_m) * 100, color=C["mean"], lw=2.4,
        label=f"理论 -(r-b)·A({m_fix:.0f}y), A={A_m:.3f}")
ax.axhline(0, color=C["true"], lw=1, alpha=0.6)
ax.axvline(b * 100, color=C["true"], ls="--", lw=1.4, label=f"中枢 b={b*100:.0f}%")
ax.set_title(f"期限溢价 = -(r_t - b)·A({m_fix:.0f}y)：利率越高曲线越平/反转", fontsize=13)
ax.set_xlabel("短期利率 r_t (%)"); ax.set_ylabel(f"{m_fix:.0f}年期 期限溢价 (y - b, %)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "hw_term_premium.png")); plt.close(fig)

# ---------- 打印供文章引用的关键数字 ----------
ss = np.sqrt(hw_var(T, a, sigma))
print("稳态波动 sigma/sqrt(2a) = %.4f (%.2f%%)" % (sigma / np.sqrt(2 * a), sigma / np.sqrt(2 * a) * 100))
print("a_hat mean=%.4f RMSE=%.4f | b_hat mean=%.4f RMSE=%.4f" % (a_hat.mean(), rmse_a, b_hat.mean(), rmse_b))
print("朴素单路径 OLS 的 a RMSE=%.4f (对比池化 %.4f) -> 陷阱" % (rmse_naive, rmse_a))
print("A(2y)=%.4f  A(5y)=%.4f  A(10y)=%.4f" % (hw_B(2, a), hw_B(5, a), hw_B(10, a)))
print("r_t-b=+3pp => 2y 期限溢价 = %.2f%%" % (-(0.03) * hw_B(2, a) * 100))
print("r_t-b=-3pp => 2y 期限溢价 = %.2f%%" % (-(-0.03) * hw_B(2, a) * 100))
print("generated:", os.listdir(OUT))
