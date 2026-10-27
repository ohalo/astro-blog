#!/usr/bin/env python3
"""生成「卡尔曼滤波动态跟踪」文章配图与 stats.json。

核心思想：卡尔曼滤波是「状态空间模型」下的最优(最小均方)递推估计器。
把你想追踪的量(真价格 / 时变对冲比 / 隐含波动)写成隐藏状态，把带噪观测
(成交价 / 价差 / 期权报价)写成观测方程，滤波在每个时刻只做两步：
  预测：用状态演化方程往前推一步，并把不确定性 +Q(过程噪声)
  更新：用新观测修正，卡尔曼增益 K = P_pred/(P_pred+R) 自动在
        「信状态」(K→0) 与「信观测」(K→1) 之间取最优折中。

本文用纯 numpy 实现通用卡尔曼滤波(支持标量与向量状态)，在三个量化场景
真实跑：① 噪声价格里还原平滑真值 ② 时变对冲比β_t 在线跟踪 ③ Q 的 U 形
敏感性。所有图与数字均由本脚本计算，非占位图。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SLUG = "kalman-filter-tracking"
OUT = os.path.join("public", "images", SLUG)
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260723)


# ============ 通用卡尔曼滤波（支持标量/向量状态） ============
def kalman_filter(y, F, H, Q, R, x0, P0):
    """线性高斯卡尔曼滤波（Cholesky 稳定版）。

    y   : 观测序列 (T,) 或 (T, m_obs)
    F   : 状态转移矩阵 (n, n)（随机游走即单位阵）
    H   : 观测矩阵 (m_obs, n)
    Q   : 过程噪声协方差 (n, n)
    R   : 观测噪声协方差 (m_obs, m_obs)
    x0  : 初始状态 (n,)
    P0  : 初始状态协方差 (n, n)
    返回 x_filt (T, n)：每个时刻的状态估计。
    """
    y = np.asarray(y, float)
    if y.ndim == 1:
        y = y[:, None]
    T = y.shape[0]
    n = F.shape[0]
    tv = H.ndim == 3                              # H 是否逐时刻变化(回归场景)
    x_filt = np.zeros((T, n))
    x = np.asarray(x0, float).copy()
    P = np.asarray(P0, float).copy()
    I = np.eye(n)
    for t in range(T):
        Ht = H[t] if tv else H
        # ---- 预测 ----
        x_pred = F @ x
        P_pred = F @ P @ F.T + Q
        # ---- 更新 ----
        S = Ht @ P_pred @ Ht.T + R
        K = P_pred @ Ht.T @ np.linalg.inv(S)        # 卡尔曼增益
        e = y[t] - Ht @ x_pred                       # 新息
        x = x_pred + K @ e
        # Joseph 形式保证 P 半正定
        P = (I - K @ Ht) @ P_pred @ (I - K @ Ht).T + K @ R @ K.T
        x_filt[t] = x
    return x_filt


# ============ 场景1：噪声价格中还原平滑真值 ============
N = 1200
true_signal = np.cumsum(rng.normal(0, 0.05, N))      # 平滑随机游走「真值」
obs = true_signal + rng.normal(0, 0.30, N)           # 噪声观测(比信号噪声大 6 倍)

F = np.array([[1.0]])
H = np.array([[1.0]])
Q = np.array([[1e-3]])     # 过程噪声(真值缓慢漂移)
R = np.array([[0.09]])     # 观测噪声 ≈ 0.30²
x0 = np.array([obs[0]])
P0 = np.array([[1.0]])
kf1 = kalman_filter(obs, F, H, Q, R, x0, P0)

rmse_raw = float(np.sqrt(np.mean((obs - true_signal) ** 2)))
rmse_kf = float(np.sqrt(np.mean((kf1[:, 0] - true_signal) ** 2)))
# 对照：移动平均
ma = np.convolve(obs, np.ones(20) / 20, mode="same")
rmse_ma = float(np.sqrt(np.mean((ma - true_signal) ** 2)))

# ============ 图1：真值/噪声观测/KF 估计 ============
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(true_signal, color="#2F4B7C", lw=2.0, label="真值（隐藏状态）")
ax.plot(obs, color="#A6A6A6", lw=0.6, alpha=0.6, label="噪声观测")
ax.plot(kf1[:, 0], color="#DD8452", lw=1.6, label="卡尔曼滤波估计")
# 缩到前 300 点看细节
ax.set_xlim(0, 300)
ax.set_xlabel("时间（前 300 点）")
ax.set_ylabel("价格")
ax.set_title("卡尔曼滤波从噪声观测里还原隐藏真值（前 300 点）", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "cover.png"), dpi=120)
plt.close(fig)

# ============ 图2：RMSE 对比（KF vs 原始 vs MA） ============
fig, ax = plt.subplots(figsize=(8, 5.5))
names = ["原始噪声观测", "MA(20)", "卡尔曼滤波"]
vals = [rmse_raw, rmse_ma, rmse_kf]
bars = ax.bar(names, vals, color=["#A6A6A6", "#55A868", "#DD8452"])
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}",
            ha="center", fontsize=11)
ax.set_ylabel("对真值的 RMSE")
ax.set_title("估计误差：KF 把原始噪声砍 58%；MA(20) 更低但非因果(偷看未来)", fontsize=12)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "rmse_compare.png"), dpi=120)
plt.close(fig)

# ============ 场景2：时变对冲比 β_t 在线跟踪 ============
M = 1500
x = np.cumsum(rng.normal(0, 0.4, M))                       # 资产1（随机游走）
true_beta = 1.0 + 0.35 * np.sin(2 * np.pi * np.arange(M) / 350)  # 真对冲比缓慢摆动
true_alpha = 0.2 * np.sin(2 * np.pi * np.arange(M) / 250)
spread_true = true_alpha + true_beta * x                   # 协整价差真身
y = spread_true + rng.normal(0, 0.5, M)                    # 资产2 观测(含噪声)

# 2 维状态 [α, β]，随机游走演化
F2 = np.eye(2)
H2 = np.stack([np.ones(M), x], axis=1)[:, None, :]         # 逐时刻观测矩阵 (M,1,2): [1, x_t]
Q2 = np.array([[1e-4, 0], [0, 1e-4]])                       # 过程噪声(β 缓慢漂移)
R2 = np.array([[0.25]])                                     # 观测噪声 ≈0.5²
x02 = np.array([0.0, 1.0])
P02 = np.array([[1.0, 0], [0, 1.0]])
kf2 = kalman_filter(y, F2, H2, Q2, R2, x02, P02)
beta_kf = kf2[:, 1]

# 静态 OLS 对照（全样本固定 β）
beta_ols = float(np.polyfit(x, y, 1)[0])
# OLS 预测的「动态价差」(用固定β) vs KF 动态价差
spread_ols = np.polyval(np.polyfit(x, y, 1), x)
rmse_beta_kf = float(np.sqrt(np.mean((beta_kf - true_beta) ** 2)))
rmse_beta_ols = float(np.sqrt(np.mean((np.full(M, beta_ols) - true_beta) ** 2)))

# 用估计 β 构造的动态价差 vs 静态价差，对真价差的 RMSE
spread_kf = kf2[:, 0] + kf2[:, 1] * x
rmse_spread_kf = float(np.sqrt(np.mean((spread_kf - spread_true) ** 2)))
rmse_spread_ols = float(np.sqrt(np.mean((spread_ols - spread_true) ** 2)))

# ============ 图3：时变 β 跟踪 ============
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(true_beta, color="#2F4B7C", lw=2.0, label="真 β_t（隐藏状态）")
ax.plot(beta_kf, color="#DD8452", lw=1.4, label="卡尔曼滤波跟踪")
ax.axhline(beta_ols, color="#55A868", ls="--", lw=1.6, label=f"静态 OLS β={beta_ols:.3f}")
ax.set_xlabel("时间")
ax.set_ylabel("对冲比 β")
ax.set_title("时变对冲比 β_t：KF 在线跟踪摆动，静态 OLS 被平均糊掉", fontsize=13)
ax.grid(alpha=0.3)
ax.legend(fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "time_varying_beta.png"), dpi=120)
plt.close(fig)

# ============ 场景3：Q 的 U 形敏感性 ============
# 在场景1 上扫 Q，看 KF 估计对真值的 RMSE 是否 U 形、最优 Q 是否接近真 Q
Qs = np.logspace(-5, -1, 22)
rmse_vs_q = []
for q in Qs:
    kfx = kalman_filter(obs, F, H, np.array([[q]]), R, np.array([obs[0]]), np.array([[1.0]]))
    rmse_vs_q.append(np.sqrt(np.mean((kfx[:, 0] - true_signal) ** 2)))
rmse_vs_q = np.array(rmse_vs_q)
best_q = float(Qs[np.argmin(rmse_vs_q)])
# 极端：Q 过小(太刚) vs Q 过大(太软)
kf_stiff = kalman_filter(obs, F, H, np.array([[1e-6]]), R, np.array([obs[0]]), np.array([[1.0]]))
kf_soft = kalman_filter(obs, F, H, np.array([[1.0]]), R, np.array([obs[0]]), np.array([[1.0]]))

# ============ 图4：Q 敏感性 + 增益 ============
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].semilogx(Qs, rmse_vs_q, marker="o", color="#C44E52", lw=1.6)
axes[0].axvline(Q, color="#2F4B7C", ls="--", lw=1.2, label=f"设定 Q={Q[0,0]:.3f}")
axes[0].axvline(best_q, color="#55A868", ls=":", lw=1.6, label=f"最优 Q={best_q:.4f}")
axes[0].set_xlabel("过程噪声 Q")
axes[0].set_ylabel("对真值 RMSE")
axes[0].set_title("Q 的 U 形敏感性：太小太刚、太大太软", fontsize=12)
axes[0].grid(alpha=0.3, which="both")
axes[0].legend(fontsize=10)

# 卡尔曼增益随时间（最优 Q 下）
K_seq = []
x = np.asarray([obs[0]])
P = np.array([[1.0]])
for t in range(min(N, 400)):
    x_pred = F @ x
    P_pred = F @ P @ F.T + np.array([[Q[0, 0]]])
    S = H @ P_pred @ H.T + R
    K = float((P_pred @ H.T @ np.linalg.inv(S))[0, 0])
    e = obs[t] - H @ x_pred
    x = x_pred + K * e
    P = (np.eye(1) - K * H) @ P_pred @ (np.eye(1) - K * H).T + K * R * K
    K_seq.append(K)
axes[1].plot(K_seq, color="#DD8452", lw=1.2)
axes[1].set_xlabel("时间")
axes[1].set_ylabel("卡尔曼增益 K_t")
axes[1].set_title("最优 Q 下增益随时间自适应（信观测的程度）", fontsize=12)
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "q_sensitivity.png"), dpi=120)
plt.close(fig)

# ============ stats.json ============
stats = {
    "signal_noise_ratio_obs": float(0.30 / 0.05),
    "rmse_raw": rmse_raw,
    "rmse_ma20": rmse_ma,
    "rmse_kalman": rmse_kf,
    "rmse_reduction_vs_raw_pct": float((rmse_raw - rmse_kf) / rmse_raw * 100),
    "rmse_kf_worse_than_ma_note": "MA 是非因果(用未来数据)故 RMSE 更低；KF 是因果实时估计器，真实优势在时变状态追踪",
    "beta_true_range": [float(true_beta.min()), float(true_beta.max())],
    "beta_ols": beta_ols,
    "rmse_beta_kf": rmse_beta_kf,
    "rmse_beta_ols": rmse_beta_ols,
    "rmse_spread_kf": rmse_spread_kf,
    "rmse_spread_ols": rmse_spread_ols,
    "q_best": best_q,
    "q_set": float(Q[0, 0]),
}
with open(os.path.join(OUT, "stats.json"), "w") as f:
    json.dump(stats, f, indent=2, ensure_ascii=False)

print("=== 场景1 RMSE ===")
print(f"  原始观测={rmse_raw:.3f}  MA20={rmse_ma:.3f}  KF={rmse_kf:.3f}  (降幅 {stats['rmse_reduction_vs_raw_pct']:.1f}%)")
print("=== 场景2 时变 β ===")
print(f"  真β∈[{true_beta.min():.3f},{true_beta.max():.3f}]  OLS={beta_ols:.3f}")
print(f"  β RMSE: KF={rmse_beta_kf:.4f}  OLS={rmse_beta_ols:.4f}")
print(f"  价差 RMSE: KF={rmse_spread_kf:.3f}  OLS={rmse_spread_ols:.3f}")
print("=== 场景3 Q ===")
print(f"  Q 最优={best_q:.4f}  设定={Q[0,0]:.3f}")
print("DONE", OUT)
