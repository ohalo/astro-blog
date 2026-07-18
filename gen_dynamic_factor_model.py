#!/usr/bin/env python3
"""
为文章「动态因子模型(DFM)：用少数共同因子把宏观面板压缩成可预测信号」
(dynamic-factor-model-dfm) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（自洽合成，仅用于演示方法；真实落地见文末路径）：
  * N=8 个宏观指标,每个 = 共同因子 f_t 的载荷 + 各自自相关 + 特质噪声
        x_{i,t} = lambda_i * f_t + phi_i * x_{i,t-1} + e_{i,t}
    共同因子 f_t 是 AR(1)：f_t = a*f_{t-1} + eps_f
  * 两步法(Stock-Watson):(1) 各序列对滞后因子回归得载荷 λ;(2) 用 λ 把面板合成因子 f_t
  * 预测:用估计因子 f_t 与 AR(1) 外推 f_{t+h},再映射回各指标
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "dynamic-factor-model-dfm")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "x": "#4C72B0", "y": "#DD8452", "base": "#55A868",
      "band": "#C44E52", "eq": "#2F4B7C", "bh": "#CCB974", "mk": "#8172B3",
      "null": "#999999"}


def generate_panel(N=8, T=240, a=0.90, seed=0):
    r = np.random.default_rng(seed)
    # 共同因子 AR(1)
    f = np.zeros(T)
    f[0] = r.normal()
    for t in range(1, T):
        f[t] = a * f[t - 1] + r.normal() * np.sqrt(1 - a ** 2)
    # 各序列: 载荷 * f + 自相关 + 特质噪声
    lam = r.uniform(0.6, 1.4, size=N)           # 对共同因子的敏感度
    phi = r.uniform(0.3, 0.7, size=N)           # 自身惯性
    x = np.zeros((N, T))
    for i in range(N):
        x[i, 0] = r.normal()
        for t in range(1, T):
            x[i, t] = lam[i] * f[t] + phi[i] * x[i, t - 1] + r.normal() * 1.3
    # 标准化
    x = (x - x.mean(axis=1, keepdims=True)) / x.std(axis=1, keepdims=True)
    return x, f, lam


def estimate_factor(x):
    """两步法(Stock-Watson 2002):
    step1: 用 x_{i,t} 对 x_{i,t-1} 回归的残差,再对这些残差在 t 截面回归出因子
    step2: 用估计因子反估载荷 lambda,再合成更干净的因子(迭代 1 次)
    """
    N, T = x.shape
    xlag = x[:, :-1]
    xnow = x[:, 1:]
    # 去各自自相关,得到「纯共同 + 噪声」残差
    resid = np.zeros_like(xnow)
    for i in range(N):
        b = np.polyfit(xlag[i], xnow[i], 1)[0]
        resid[i] = xnow[i] - b * xlag[i]
    # 截面第一主成分 = 共同因子载荷(变量在行,协方差 N×N)
    from numpy.linalg import svd
    cov = resid @ resid.T / T
    U, S, Vt = svd(cov, full_matrices=False)
    lam_est = U[:, 0] * np.sign(U[0, 0])           # 载荷方向对齐(长度 N)
    f_est = resid.T @ lam_est / (lam_est @ lam_est)
    # step2 用载荷再合成一次(更干净)
    lam2 = xnow @ f_est / (f_est @ f_est)
    f2 = xnow.T @ lam2 / (lam2 @ lam2)
    return f_est, f2, lam_est


# ---------- 生成数据 ----------
x, f_true, lam = generate_panel(N=8, T=240, seed=7)
f_est, f2, lam_est = estimate_factor(x)
# 相关性:估计因子与真实因子
corr_f = np.corrcoef(f_est, f_true[1:])[0, 1]
corr_f2 = np.corrcoef(f2, f_true[1:])[0, 1]
print("估计因子 vs 真因子 相关=%.3f  两步修正后=%.3f" % (corr_f, corr_f2))

# ---------- 图 1：原始面板 8 条宏观序列 ----------
fig, ax = plt.subplots(figsize=(10, 5))
for i in range(8):
    ax.plot(x[i], lw=1.0, alpha=0.7, label="指标 %d" % (i + 1))
ax.set_ylabel("标准化值", fontsize=11)
ax.set_xlabel("时间(月)", fontsize=11)
ax.set_title("8 条宏观指标：各自的噪声把共同趋势淹没在面板里", fontsize=12)
ax.legend(ncol=4, fontsize=8, loc="upper right")
ax.grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "macro_panel_8_series.png"), dpi=130)
plt.close(fig)

# ---------- 图 2：共同因子 (真实 vs DFM 估计) ----------
fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax[0].plot(f_true, color=C["eq"], lw=1.5, label="真实共同因子 f_t")
ax[0].set_ylabel("因子值", fontsize=11)
ax[0].set_title("动态因子模型：把 8 条指标压成 1 条可预测因子", fontsize=12)
ax[0].legend(fontsize=9)
ax[0].grid(True, color=C["grid"])
ax[1].plot(f_true[1:], color=C["null"], lw=1.2, ls="--", label="真实 f_t")
ax[1].plot(f2, color=C["base"], lw=1.4, label="DFM 估计 f_t (相关=%.2f)" % corr_f2)
ax[1].set_ylabel("因子值", fontsize=11)
ax[1].set_xlabel("时间(月)", fontsize=11)
ax[1].legend(fontsize=9)
ax[1].grid(True, color=C["grid"])
plt.tight_layout()
fig.savefig(os.path.join(D, "common_factor_estimated.png"), dpi=130)
plt.close(fig)

# ---------- 图 3：捕获共同因子的能力（各原始指标 vs DFM 因子，与真因子相关性） ----------
# 模拟研究:已知 DGP,看谁更贴近真实共同因子 f_t
corr_raw = [np.corrcoef(x[i], f_true)[0, 1] for i in range(x.shape[0])]
corr_dfm = np.corrcoef(f2, f_true[1:])[0, 1]
print("各指标 vs 真因子 相关:", np.round(corr_raw, 3))
print("DFM 因子 vs 真因子 相关: %.3f" % corr_dfm)

fig, ax = plt.subplots(figsize=(10, 4.8))
labels = ["指标 %d" % (i + 1) for i in range(len(corr_raw))] + ["DFM 因子"]
vals = corr_raw + [corr_dfm]
colors = [C["null"]] * len(corr_raw) + [C["base"]]
bars = ax.bar(labels, vals, color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.01, "%.2f" % v, ha="center", fontsize=9)
ax.set_ylabel("与真实共同因子的相关系数", fontsize=11)
ax.set_title("捕获共同信号：DFM 因子比任何单一原始指标更贴近真因子", fontsize=12)
ax.set_ylim(0, 1.05)
ax.grid(True, color=C["grid"], axis="y")
plt.tight_layout()
fig.savefig(os.path.join(D, "dfm_factor_capture.png"), dpi=130)
plt.close(fig)

print("saved 3 figures to", D)
