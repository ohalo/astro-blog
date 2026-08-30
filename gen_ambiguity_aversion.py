#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「模糊厌恶与最大最小投资组合」文章配图 + 核心数值。
数据均为 numpy 合成（固定 seed 可复现）。
核心模型：Garlappi-Uppal-Wang (2007) 模糊厌恶组合 = 切线组合 与 全局最小方差组合的凸组合。
配图保存到 public/images/ambiguity-aversion-maxmin-portfolio/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang TC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

SEED = 20260830
rng = np.random.default_rng(SEED)
OUT = "public/images/ambiguity-aversion-maxmin-portfolio"
os.makedirs(OUT, exist_ok=True)

# ============ 1. 合成数据：40 资产，2 个隐含公因子 + 特异性噪声（日度尺度） ============
N, T = 40, 1500
F = rng.standard_normal((T, 2)) * 0.012            # 因子日收益 ~1.2%
B = rng.standard_normal((N, 2)) * 0.6             # 因子载荷
mu_true = 0.0003 + 0.0004 * B[:, 0] - 0.0003 * B[:, 1]   # 真实日度预期超额收益
eps = rng.standard_normal((T, N)) * 0.010         # 特异性日度噪声 ~1%
R = mu_true[None, :] + F @ B.T + eps              # T x N 收益矩阵

mu_hat = R.mean(0)
Sigma = np.cov(R.T)
Sigma_inv = np.linalg.inv(Sigma)
ones = np.ones(N)

# ============ 2. 三类权重 ============
# (a) 切线组合（最大夏普）
w_tan = Sigma_inv @ mu_hat
w_tan = np.maximum(w_tan, 0)
w_tan /= w_tan.sum()

# (b) 全局最小方差组合（GMV，完全不看收益观点）
w_gmv = Sigma_inv @ ones
w_gmv = np.maximum(w_gmv, 0)
w_gmv /= w_gmv.sum()

# (c) 最大最小 / 模糊厌恶组合：w = a*w_tan + (1-a)*w_gmv，a = 1/(1+delta)
delta = 1.2                                       # 模糊厌恶系数
a = 1.0 / (1.0 + delta)
w_mm = a * w_tan + (1 - a) * w_gmv

# ============ 3. OOS 测试：真实 mu 相对估计值偏移（模拟估计误差） ============
delta_mu = rng.standard_normal(N) * 0.0002
mu_oos = mu_true + delta_mu
R_oos = mu_oos[None, :] + F @ B.T + rng.standard_normal((T, N)) * 0.010

def metrics(w, Rmat):
    r_p = Rmat @ w
    cum = np.cumprod(1.0 + r_p)
    peak = np.maximum.accumulate(cum)
    mdd = (cum / peak - 1.0).min()
    return cum[-1], mdd, r_p.mean() / (r_p.std() + 1e-12) * np.sqrt(252)

mv_f, mv_d, mv_s = metrics(w_tan, R_oos)
gmv_f, gmv_d, gmv_s = metrics(w_gmv, R_oos)
mm_f, mm_d, mm_s = metrics(w_mm, R_oos)

print("=== 核心统计（用于正文）===")
print(f"切线组合   终值={mv_f:.4f}  回撤={mv_d*100:.2f}%  年化Sharpe={mv_s:.3f}")
print(f"GMV组合    终值={gmv_f:.4f}  回撤={gmv_d*100:.2f}%  年化Sharpe={gmv_s:.3f}")
print(f"最大最小   终值={mm_f:.4f}  回撤={mm_d*100:.2f}%  年化Sharpe={mm_s:.3f}")
print(f"权重: 切线max={w_tan.max():.4f}  非零={int((w_tan>1e-5).sum())} / 40")
print(f"权重: GMV max={w_gmv.max():.4f}  非零={int((w_gmv>1e-5).sum())} / 40")
print(f"权重: MM  max={w_mm.max():.4f}  非零={int((w_mm>1e-5).sum())} / 40")
print(f"切线 vs MM 权重相关={np.corrcoef(w_tan,w_mm)[0,1]:.4f}")
print(f"模糊厌恶系数 delta={delta} -> a(切线占比)={a:.3f}")

# ============ 图 1：累积财富 ============
W_tan = np.cumprod(1.0 + R_oos @ w_tan)
W_gmv = np.cumprod(1.0 + R_oos @ w_gmv)
W_mm  = np.cumprod(1.0 + R_oos @ w_mm)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(W_tan, label=f"切线组合 (回撤 {mv_d*100:.1f}%)", lw=1.8, color="#d1495b")
ax.plot(W_gmv, label=f"全局最小方差 (回撤 {gmv_d*100:.1f}%)", lw=1.6, color="#6a4c93", ls="--")
ax.plot(W_mm, label=f"最大最小稳健 (回撤 {mm_d*100:.1f}%)", lw=1.8, color="#1b6ca8")
ax.set_title("OOS 累积财富：最大最小组合收敛于『切线 + GMV』之间", fontsize=13)
ax.set_xlabel("交易日（OOS 测试样本）")
ax.set_ylabel("累积财富（起始 = 1.0）")
ax.legend(fontsize=9, loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/wealth_curves.png")
plt.close(fig)

# ============ 图 2：权重对比 ============
idx = np.argsort(w_tan)[::-1][:12]
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(idx))
ax.bar(x - 0.2, w_tan[idx], width=0.4, label="切线组合权重", color="#d1495b")
ax.bar(x + 0.2, w_mm[idx], width=0.4, label="最大最小权重", color="#1b6ca8")
ax.set_title(f"权重分配：模糊厌恶把极端权重向 GMV 拉回 (a={a:.2f})", fontsize=12.5)
ax.set_xlabel("按切线权重排序的前 12 只资产")
ax.set_ylabel("权重")
ax.legend(fontsize=9)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(f"{OUT}/weight_compare.png")
plt.close(fig)

# ============ 图 3：模糊厌恶系数 delta -> 组合权重分布（赫芬达尔） ============
deltas = np.linspace(0.0, 4.0, 21)
hh_mm = []
for d in deltas:
    aa = 1.0 / (1.0 + d)
    w = aa * w_tan + (1 - aa) * w_gmv
    hh_mm.append(np.sum(w**2))
hh_mm = np.array(hh_mm)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(deltas, hh_mm, marker="o", color="#edae49", lw=2, label="最大最小组合 HH 指数")
ax.axhline(np.sum(w_tan**2), color="#d1495b", ls="--", lw=1.2, label="切线组合 HH")
ax.axhline(np.sum(w_gmv**2), color="#1b6ca8", ls="--", lw=1.2, label="GMV 组合 HH")
ax.set_title("模糊厌恶越强，权重越分散（赫芬达尔指数下降）", fontsize=12.5)
ax.set_xlabel("模糊厌恶系数 δ")
ax.set_ylabel("权重集中度 HHI = Σ w²")
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ambiguity_radius.png")
plt.close(fig)

print(f"图片已保存到 {OUT}")
