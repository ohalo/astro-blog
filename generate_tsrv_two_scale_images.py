#!/usr/bin/env python3
"""
为文章「TSRV 二次变差：用子采样把微观结构噪声剔出波动率」(tsrv-two-scale-volatility)
生成真实配图。模拟：高频对数价格 + iid 微观结构噪声；
用 Zhang-Mykland-Ait-Sahalia(2005) 两尺度（Two-Scale Realized Volatility）估计去噪。

图1 tsrv_noisy_price.png        真实连续路径 vs 加噪观测（噪声可见）
图2 tsrv_rvk_vs_invK.png        RV_K vs 1/K 线性外推，截距=TSRV，参考线=真值IV
图3 tsrv_estimator_compare.png  RV_all / TSRV / IV_true 对比（单情景）
图4 tsrv_noise_robustness.png   不同噪声尺度下 RV_all 与 TSRV 的偏差稳健性
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

SLUG = "tsrv-two-scale-volatility"
BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, SLUG)
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

# ===== 1) 模拟高频路径 + 微观结构噪声 =====
n = 4000
sigma = 0.30
dt = 1.0 / n
Z = rng.standard_normal(n)
X = np.cumsum(sigma * np.sqrt(dt) * Z)          # 真实对数价格（扩散部分）
IV_true = float(np.sum(np.diff(X) ** 2))         # 无噪声积分方差（真值基准）
eta = 0.002
Y = X + rng.normal(0.0, eta, n)                  # 观测对数价格（含微观结构噪声）

# ===== 2) 子采样已实现方差 =====
def rv_subsampled(Y, K):
    m = len(Y); total = 0.0
    for k in range(K):
        idx = np.arange(k, m, K)
        if len(idx) > 1:
            total += np.sum(np.diff(Y[idx]) ** 2)
    return total / K

K_list = [1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200]
rvk = np.array([rv_subsampled(Y, K) for K in K_list])
rv_all_val = rvk[0]
invK = 1.0 / np.array(K_list, dtype=float)

# TSRV 两尺度校正 (K>=2): TSRV(K) = (K*RV_K - RV_all)/(K-1)
# 取适中 K（此处 K=12）作标杆：过大 K 方差爆炸、过小 K 去噪不足
K_REF = 12
TSRV_REF = (K_REF * rv_subsampled(Y, K_REF) - rv_all_val) / (K_REF - 1)

Ks = K_list[1:]
rvk_s = rvk[1:]
TSRV = np.array([(K * rk - rv_all_val) / (K - 1) for K, rk in zip(Ks, rvk_s)])

# 线性外推：RV_K = a + b*(1/K)，截距 a ≈ IV
coef = np.polyfit(invK, rvk, 1)
a_ext, b_ext = coef[1], coef[0]
tsrv_extrap = a_ext

print(f"IV_true={IV_true:.6f}  RV_all={rv_all_val:.6f} (bias {100*(rv_all_val-IV_true)/IV_true:.1f}%)")
print(f"TSRV(K={K_REF})={TSRV_REF:.6f} (bias {100*(TSRV_REF-IV_true)/IV_true:.2f}%)")
print(f"Linear extrapolate a={tsrv_extrap:.6f}")

# ===== 图1：真实 vs 加噪观测 =====
fig, ax = plt.subplots(figsize=(9, 4.2))
i0, i1 = 100, 400
ax.plot(range(i0, i1), X[i0:i1], lw=1.4, color="#1f77b4", label="真实对数价格 X_t（无噪声扩散）")
ax.plot(range(i0, i1), Y[i0:i1], lw=0.8, color="#d62728", alpha=0.75, label="观测价格 Y_t（含微观结构噪声）")
ax.set_title("高频观测被微观结构噪声包围：真实路径 vs 加噪观测", fontsize=12)
ax.set_xlabel("高频时间步"); ax.set_ylabel("对数价格")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "tsrv_noisy_price.png"), dpi=130); plt.close(fig)

# ===== 图2：RV_K vs 1/K 外推 =====
fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(invK, rvk, "o-", color="#2ca02c", label="RV_K（子采样已实现方差）")
xline = np.linspace(0, invK[0], 50)
ax.plot(xline, a_ext + b_ext * xline, "--", color="#7f7f7f",
        label=f"线性外推 → 截距={tsrv_extrap:.4f}")
ax.axhline(IV_true, color="#1f77b4", lw=1.4, ls=":", label=f"真实积分方差 IV={IV_true:.4f}")
ax.scatter([0], [tsrv_extrap], color="#ff7f0e", zorder=5, s=80, label="TSRV 校正值")
ax.set_xlabel("1/K（子样本间隔的倒数）"); ax.set_ylabel("已实现方差估计")
ax.set_title("RV_K 随子采样变粗而收敛：外推到 K→∞ 即 TSRV", fontsize=12)
ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "tsrv_rvk_vs_invK.png"), dpi=130); plt.close(fig)

# ===== 图3：单情景对比 =====
fig, ax = plt.subplots(figsize=(8, 4.4))
labels = ["RV_all\n(全样本)", "TSRV\n(两尺度校正)", "IV_true\n(真值)"]
vals = [rv_all_val, TSRV[-1], IV_true]
colors = ["#d62728", "#ff7f0e", "#1f77b4"]
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("积分方差估计")
ax.set_title(f"单情景对比：全样本 RV 高估 {100*(rv_all_val-IV_true)/IV_true:.0f}%，TSRV(K={K_REF}) 修复至 {100*(TSRV_REF-IV_true)/IV_true:+.0f}%",
             fontsize=11)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "tsrv_estimator_compare.png"), dpi=130); plt.close(fig)

# ===== 图4：噪声稳健性 =====
eta_list = np.array([0.001, 0.002, 0.004, 0.008, 0.016])
bias_all, bias_tsrv = [], []
for et in eta_list:
    Yt = X + rng.normal(0, et, n)
    rva = rv_subsampled(Yt, 1)
    tsv = (K_REF * rv_subsampled(Yt, K_REF) - rva) / (K_REF - 1)
    bias_all.append(100 * (rva - IV_true) / IV_true)
    bias_tsrv.append(100 * (tsv - IV_true) / IV_true)
fig, ax = plt.subplots(figsize=(9, 4.4))
ax.plot(eta_list, bias_all, "o-", color="#d62728", label="RV_all 偏差（全样本）")
ax.plot(eta_list, bias_tsrv, "s-", color="#2ca02c", label="TSRV 偏差（两尺度校正）")
ax.axhline(0, color="gray", lw=1, ls=":")
ax.set_xlabel("微观结构噪声标准差 η"); ax.set_ylabel("相对真值的偏差 (%)")
ax.set_title("噪声越大，全样本 RV 越失真；TSRV 几乎不受 η 影响", fontsize=12)
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "tsrv_noise_robustness.png"), dpi=130); plt.close(fig)

print("DONE", sorted(os.listdir(D)))
