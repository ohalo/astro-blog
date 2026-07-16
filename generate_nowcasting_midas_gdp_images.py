#!/usr/bin/env python3
"""
为文章「Nowcasting 混频实时预测：用高频指标盯住低频 GDP」(nowcasting-midas-gdp) 生成真实配图。

核心逻辑（Giannone-Reichlin-Small 风格 nowcasting，混频状态空间）：
  - 低频目标：季度 GDP 增速 g_q，由不可观测的季度共同因子 f_q 驱动（f_q = phi*f_{q-1} + eta）。
  - 高频指示变量：月度工业生产、周度初请失业金(取负)、日度金融压力指数，
    各自在「所在季度」加载到同一因子 f_q（ragged-edge：季内信息逐步到位）。
  - 用序贯卡尔曼滤波：因子先验来自上一季度；季内每来一条高频观测就做一次测量更新，
    nowcast（对 f_q 的估计）随信息累积而收敛，季末 GDP 公布再做一次终更新。
  - 关键 DGP：所有高频变量都由同一个低频因子 f_q 驱动 → 混频信息对低频目标有真实领先/解释力。
  全部为合成数据，非占位图。
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
D = os.path.join(BASE, "nowcasting-midas-gdp")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260716)

# ---------- 1) 构造混频 DGP ----------
Q = 120                                   # 季度数（30 年）
phi, Q_eta = 0.55, 0.30                    # 因子 AR(1)
var_g = 0.20                              # GDP 观测噪声

# 真实低频因子
f_true = np.zeros(Q)
f_true[0] = rng.normal(0, 1)
for q in range(1, Q):
    f_true[q] = phi * f_true[q - 1] + rng.normal(0, np.sqrt(Q_eta))
g_obs = f_true + rng.normal(0, np.sqrt(var_g), Q)   # 季度 GDP 增速（季末公布）

# 混频指示变量：都加载到所在季度因子 f_q
n_m, n_w, n_d = 3, 13, 65                 # 每季 月/周/日 观测数
lam_m, lam_w, lam_d = 0.85, -0.65, 0.45   # 载荷（周度失业金取负→经济好则为负）
var_m, var_w, var_d = 0.25, 0.40, 0.55

def gen_hf(lam, var, n):
    out = np.zeros((Q, n))
    for q in range(Q):
        out[q] = lam * f_true[q] + rng.normal(0, np.sqrt(var), n)
    return out

M = gen_hf(lam_m, var_m, n_m)             # 月度工业生产
W = gen_hf(lam_w, var_w, n_w)             # 周度初请失业金(取负)
Dd = gen_hf(lam_d, var_d, n_d)            # 日度金融压力指数

# ---------- 2) 序贯卡尔曼 nowcasting ----------
def run_nowcast(record_vintages=False):
    """返回：最终 nowcast 序列 m_final，以及(可选)各 vintage 的 nowcast。"""
    m, P = 0.0, 1.0
    m_final = np.zeros(Q)
    vint_now = {"start": [], "m1": [], "m2": [], "end": []}  # 记录不同信息 vintage
    for q in range(Q):
        # 预测步：季度状态转移
        m = phi * m
        P = phi * phi * P + Q_eta
        # 季内序贯更新（按时间顺序：先月、再周、再日）
        # 月（每季 n_m 个，用完一次，不重复）
        for k in range(n_m):
            y = M[q, k]; H = lam_m; R = var_m
            S = H * H * P + R
            Kk = P * H / S
            m = m + Kk * (y - H * m); P = P - Kk * H * P
        if record_vintages:
            vint_now["m1"].append(m)
        # 周（全部）
        for j in range(n_w):
            y = W[q, j]; H = lam_w; R = var_w
            S = H * H * P + R
            Kk = P * H / S
            m = m + Kk * (y - H * m); P = P - Kk * H * P
        if record_vintages:
            vint_now["m2"].append(m)
        # 日（全部）
        for i in range(n_d):
            y = Dd[q, i]; H = lam_d; R = var_d
            S = H * H * P + R
            Kk = P * H / S
            m = m + Kk * (y - H * m); P = P - Kk * H * P
        if record_vintages:
            vint_now["end"].append(m)
        # 季末 GDP 公布：终更新
        y = g_obs[q]; H = 1.0; R = var_g
        S = H * H * P + R
        Kk = P * H / S
        m = m + Kk * (y - H * m); P = P - Kk * H * P
        m_final[q] = m
    if record_vintages:
        return m_final, vint_now
    return m_final, None

m_final, vint = run_nowcast(record_vintages=True)

# 信息 vintage 的 RMSE（与真实因子比较）
def rmse(a, b):
    return float(np.sqrt(np.mean((np.array(a) - np.array(b)) ** 2)))
rmse_start = rmse([phi * 0.0] * Q, f_true)   # 近似：start 用先验（这里用 end 近似对比，改用下方真实计算）
# 直接计算各 vintage RMSE：start 用"仅上一季滤波态"预测，这里用简化：用 end 序列本身代表终态
rmse_end = rmse(vint["end"], f_true)
rmse_m2 = rmse(vint["m2"], f_true)
rmse_m1 = rmse(vint["m1"], f_true)
# start vintage：用纯先验（phi * 上一季 nowcast）重跑一次
m_start = np.zeros(Q)
prev = 0.0
for q in range(Q):
    prev = phi * prev
    m_start[q] = prev
rmse_start = rmse(m_start, f_true)

print("=== Nowcasting 诊断 ===")
print(f"因子 AR(1): phi={phi} Q_eta={Q_eta}")
print(f"各 vintage 对真实因子 f_q 的 RMSE:")
print(f"  季初(仅先验)        : {rmse_start:.3f}")
print(f"  第1月后(月频到位)   : {rmse_m1:.3f}")
print(f"  第2月后            : {rmse_m2:.3f}")
print(f"  季末(高频全到位)    : {rmse_end:.3f}")
print(f"  nowcast 与真实 GDP 相关: {np.corrcoef(m_final, g_obs)[0,1]:.3f}")
print(f"生成图片: {os.listdir(D)}")

# ---------- 绘图 ----------
plt.rcParams["figure.dpi"] = 150

# 图1：不同信息 vintage 的 nowcast RMSE（随信息累积下降）
fig, ax = plt.subplots(figsize=(11, 5.2))
vint_names = ["季初\n(仅先验)", "第1月后\n(月频到位)", "第2月后", "季末\n(高频全到位)"]
vint_vals = [rmse_start, rmse_m1, rmse_m2, rmse_end]
colors = ["#999999", "#6baed6", "#3182bd", "#c44e52"]
bars = ax.bar(vint_names, vint_vals, color=colors, alpha=0.88)
for b, v in zip(bars, vint_vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("对真实因子的 RMSE", fontsize=11)
ax.set_title("实时 Nowcasting：信息越累积，对季度 GDP 因子的估计越准",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "nowcast_rmse_vintage.png"), bbox_inches="tight")
plt.close()

# 图2：样本季度内 nowcast 收敛路径（取 Q=80 附近一个季度）
q0 = 80
m_q, P_q = 0.0, 1.0
steps, means, lows, highs = [], [], [], []
# 重放到 q0 前
for q in range(q0):
    m_q = phi * m_q; P_q = phi * phi * P_q + Q_eta
    for k in range(n_m):
        y = M[q, k]; H = lam_m; R = var_m
        S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    for j in range(n_w):
        y = W[q, j]; H = lam_w; R = var_w
        S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    for i in range(n_d):
        y = Dd[q, i]; H = lam_d; R = var_d
        S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    y = g_obs[q]; H = 1.0; R = var_g
    S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
# 在 q0 内逐步更新，记录
m_q = phi * m_q; P_q = phi * phi * P_q + Q_eta
steps.append(0); means.append(m_q); lows.append(m_q - 1.96*np.sqrt(P_q)); highs.append(m_q + 1.96*np.sqrt(P_q))
for k in range(n_m):
    y = M[q0, k]; H = lam_m; R = var_m
    S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    steps.append(len(steps)); means.append(m_q); lows.append(m_q-1.96*np.sqrt(P_q)); highs.append(m_q+1.96*np.sqrt(P_q))
for j in range(n_w):
    y = W[q0, j]; H = lam_w; R = var_w
    S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    steps.append(len(steps)); means.append(m_q); lows.append(m_q-1.96*np.sqrt(P_q)); highs.append(m_q+1.96*np.sqrt(P_q))
for i in range(n_d):
    y = Dd[q0, i]; H = lam_d; R = var_d
    S = H*H*P_q + R; Kk = P_q*H/S; m_q += Kk*(y-H*m_q); P_q -= Kk*H*P_q
    steps.append(len(steps)); means.append(m_q); lows.append(m_q-1.96*np.sqrt(P_q)); highs.append(m_q+1.96*np.sqrt(P_q))
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.fill_between(steps, lows, highs, color="#c44e52", alpha=0.12, label="95% 置信带")
ax.plot(steps, means, color="#c44e52", lw=2.2, marker="o", ms=3, label="Nowcast（对 f_q 的估计）")
ax.axhline(f_true[q0], color="#2ca02c", lw=2.0, ls="--", label=f"真实因子 f_q={f_true[q0]:.2f}")
ax.set_xlabel("季内信息累积步数（月→周→日）", fontsize=11)
ax.set_ylabel("GDP 因子估计", fontsize=11)
ax.set_title(f"样本季度 Q={q0}：nowcast 随高频信息到位而收敛到真实值",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "nowcast_path_sample_quarter.png"), bbox_inches="tight")
plt.close()

# 图3：最终 nowcast vs 实际 GDP（散点 + 相关）
fig, ax = plt.subplots(figsize=(6.2, 6.0))
ax.scatter(g_obs, m_final, color="#4c72b0", alpha=0.6, s=22)
# 拟合线
z = np.polyfit(g_obs, m_final, 1)
xs = np.linspace(g_obs.min(), g_obs.max(), 50)
ax.plot(xs, np.polyval(z, xs), color="#c44e52", lw=2)
ax.set_xlabel("实际 GDP 增速 g_q", fontsize=11)
ax.set_ylabel("Nowcast（季末滤波估计）", fontsize=11)
ax.set_title(f"Nowcast vs 实际 GDP\n相关={np.corrcoef(m_final, g_obs)[0,1]:.3f}",
             fontsize=12.5, fontweight="bold")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "nowcast_vs_gdp_scatter.png"), bbox_inches="tight")
plt.close()

# 图4：混频指示变量面板（对齐到季度 GDP）
fig, axes = plt.subplots(4, 1, figsize=(11, 8.5), sharex=True)
qq = np.arange(Q)
axes[0].plot(qq, g_obs, color="#c44e52", lw=1.8, marker="o", ms=3)
axes[0].set_ylabel("GDP 增速", fontsize=10); axes[0].set_title("混频 Nowcasting 数据面板：低频 GDP 由高频指标共同驱动", fontsize=12.5, fontweight="bold")
# 月度：取每季第1月代表
axes[1].plot(qq, M[:, 0], color="#4c72b0", lw=1.2, alpha=0.85)
axes[1].set_ylabel("月产(工业)", fontsize=10)
axes[2].plot(qq, W[:, 0], color="#6a3d9a", lw=1.2, alpha=0.85)
axes[2].set_ylabel("周失业金(负)", fontsize=10)
axes[3].plot(qq, Dd[:, 0], color="#55a868", lw=1.2, alpha=0.85)
axes[3].set_ylabel("日金融压力", fontsize=10); axes[3].set_xlabel("季度", fontsize=11)
for a in axes:
    a.grid(True, alpha=0.22)
plt.tight_layout()
plt.savefig(os.path.join(D, "nowcasting_indicators_panel.png"), bbox_inches="tight")
plt.close()

print("✅ 图片已生成:", sorted(os.listdir(D)))
