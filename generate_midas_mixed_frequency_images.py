#!/usr/bin/env python3
"""
为文章「MIDAS 混频回归：用高频数据预测低频变量」(midas-mixed-frequency) 生成真实配图。

核心逻辑（Mixed-Data Sampling, Ghysels-Kobracki-Kounanos 2001）：
  低频目标 y_t（如月度通胀/GDP/收益率）可由高频解释变量 x 的「混频滞后」预测：
    y_t = β0 + β1 · Σ_{k=1}^{K} w_k(θ) · x_{t−k} + ε_t
  其中高频 x 按交易日对齐，权重 w_k(θ) 用 Beta 多项式把数十个高频滞后压缩成少量参数：
    w_k(θ) = (k/K)^{θ1−1} · (1−k/K)^{θ2−1} / Σ
  关键 DGP：日频 x 在「第 q 月」由「下个月因子 f_{q+1}」驱动——即日频流量 LEADS 月度目标，
  这是 MIDAS 真正有价值的场景：高频数据既超前、又带日内噪声，Beta 权重把噪声平滑掉并抓住领先信号。
  U-MIDAS 则直接用无约束滞后。本文演示 Beta-MIDAS 拟合权重 + 样本外预测优势（vs AR(1) / 朴素）。
  全部为合成数据，非占位图。
"""
import os
import numpy as np
from scipy.optimize import minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "midas-mixed-frequency")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260715)

# ---------- 1) 构造混频数据（日频 LEADS 月度）----------
T = 120                         # 月度数
m = 21                          # 每月交易日
K = 2 * m                       # 用近 2 个月（42 个）日频滞后预测当月
Nday = T * m

# 月度潜在因子 f_t：AR(1)（中等持续性）
f = np.zeros(T + 1)
f[0] = rng.normal(0, 1)
for t in range(1, T + 1):
    f[t] = 0.5 * f[t - 1] + rng.normal(0, 1)

# 日频 x：第 q 月的日频数据由「下个月因子 f_{q+1}」驱动（领先关系）+ 日内噪声
x = np.zeros(Nday)
day_state = 0.0
for q in range(T):                       # q = 0..T-1 月
    f_next = f[q + 1]                    # 该月日频携带下月因子 → 领先月度 y
    for d in range(m):
        idx = q * m + d
        day_state = 0.5 * day_state + rng.normal(0, 0.4)   # 日内短记忆
        x[idx] = f_next + 0.4 * day_state + rng.normal(0, 0.7)

# 月度目标 y_t：由当月因子 f_t 驱动（f_t 已被「上月日频 x」提前暴露）
beta0, beta1 = 0.3, 0.8
y = beta0 + beta1 * f[:T] + rng.normal(0, 0.30, T)

# ---------- 2) Beta-MIDAS 权重与混频估计 ----------
def beta_weights(theta, K):
    k = np.arange(1, K + 1)
    raw = (k / K) ** (theta[0] - 1) * (1 - k / K) ** (theta[1] - 1)
    return raw / raw.sum()

def midas_z(x, t, m, K, theta):
    # 预测第 t 月 y_t：用截至「第 t-1 月结束」的最近 K 个日频观测（即月份 t-2, t-1）
    end = t * m
    seg = x[end - K:end]
    if len(seg) < K:
        seg = np.concatenate([np.full(K - len(seg), seg[0] if len(seg) else 0.0), seg])
    w = beta_weights(theta, K)
    return np.dot(w, seg)

def midas_rss(params, x, y, m, K, T_use):
    beta0, beta1, th1, th2 = params
    z = np.array([midas_z(x, t, m, K, (th1, th2)) for t in range(T_use)])
    pred = beta0 + beta1 * z
    return np.sum((y - pred) ** 2)

# 仅用 t=2..T-1（前面需要 2 个月日频历史）
T_use = T
bnds = [(None, None), (None, None), (1.01, 20), (1.01, 20)]
p0 = [0.3, 0.8, 3.0, 3.0]
res = minimize(midas_rss, p0, args=(x, y, m, K, T_use), bounds=bnds, method="L-BFGS-B")
b0_h, b1_h, th1_h, th2_h = res.x
z_hat = np.array([midas_z(x, t, m, K, (th1_h, th2_h)) for t in range(T_use)])
y_fit = b0_h + b1_h * z_hat
rss_midas = midas_rss(res.x, x, y, m, K, T_use)
tss = np.sum((y - y.mean()) ** 2)
r2_in = 1 - rss_midas / tss

# 权重多项式形状
w_hat = beta_weights((th1_h, th2_h), K)
days_axis = np.arange(1, K + 1)

# ---------- 3) 样本外对比：递归 MIDAS vs AR(1) vs 朴素 ----------
def oos_forecast(train_T):
    sub_x = x[:train_T * m]
    sub_y = y[:train_T]
    if train_T < 24:
        return None
    r = minimize(midas_rss, p0, args=(sub_x, sub_y, m, K, train_T),
                 bounds=bnds, method="L-BFGS-B")
    b0e, b1e, te1, te2 = r.x
    ze = midas_z(x, train_T, m, K, (te1, te2))      # 用截至 train_T-1 月的日频预测 train_T
    yhat_midas = b0e + b1e * ze
    yy = sub_y
    Xa = np.vstack([np.ones(train_T - 1), yy[:-1]]).T
    co = np.linalg.lstsq(Xa, yy[1:], rcond=None)[0]
    yhat_ar = co[0] + co[1] * yy[-1]
    yhat_naive = yy[-1]
    return yhat_midas, yhat_ar, yhat_naive

y_true_oos, y_midas_oos, y_ar_oos, y_naive_oos = [], [], [], []
for t in range(24, T):
    fc = oos_forecast(t)
    if fc is None:
        continue
    y_true_oos.append(y[t]); y_midas_oos.append(fc[0]); y_ar_oos.append(fc[1]); y_naive_oos.append(fc[2])
y_true_oos = np.array(y_true_oos); y_midas_oos = np.array(y_midas_oos)
y_ar_oos = np.array(y_ar_oos); y_naive_oos = np.array(y_naive_oos)

def oos_r2(true, pred):
    base = np.sum((true - true.mean()) ** 2)
    return 1 - np.sum((true - pred) ** 2) / base
r2_midas = oos_r2(y_true_oos, y_midas_oos)
r2_ar = oos_r2(y_true_oos, y_ar_oos)
r2_naive = oos_r2(y_true_oos, y_naive_oos)

# ---------- 绘图 ----------
# 图1：日频 x（细线）与月度 y（粗线）混频叠加
fig, ax1 = plt.subplots(figsize=(11, 5.5))
days = np.arange(Nday)
ax1.plot(days, x, color="#bbbbbb", lw=0.6, alpha=0.8, label="日频 x（高频噪声，领先 y 一个月）")
ax1.set_xlabel("交易日", fontsize=11)
ax1.set_ylabel("日频 x", fontsize=11, color="#666")
ax2 = ax1.twinx()
mt = np.arange(T) * m + m // 2
ax2.plot(mt, y, color="#c44e52", lw=2.4, marker="o", ms=3, label="月度 y（低频目标）")
ax2.set_ylabel("月度 y", fontsize=11, color="#c44e52")
ax1.set_title("混频数据：日频流量领先月度目标一个月，并叠加日内噪声",
              fontsize=12.5, fontweight="bold")
l1, lb1 = ax1.get_legend_handles_labels(); l2, lb2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, lb1 + lb2, loc="upper left", fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "midas_mixed_freq_data.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图2：Beta-MIDAS 权重多项式（近期日频权重更大）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.bar(days_axis, w_hat, color="#4c72b0", alpha=0.85, width=0.9)
ax.set_xlabel("滞后日数 k（距预测月越近 k 越小）", fontsize=11)
ax.set_ylabel("权重 w_k(θ)", fontsize=11)
ax.set_title(f"Beta-MIDAS 权重多项式：θ=({th1_h:.2f},{th2_h:.2f})，近期日频主导",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "midas_beta_weights.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图3：样本内拟合 —— MIDAS 用日频还原月度 y
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(y, color="#bbbbbb", lw=2.2, marker="o", ms=3, label="实际 y（月度）")
ax.plot(y_fit, color="#c44e52", lw=2.0, label=f"MIDAS 拟合（日频聚合）  R²={r2_in:.3f}")
ax.set_xlabel("月份", fontsize=11)
ax.set_ylabel("y", fontsize=11)
ax.set_title("Beta-MIDAS：仅用日频 x 的混频加权，还原月度 y 的走势",
             fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "midas_in_sample_fit.png"), dpi=150, bbox_inches="tight")
plt.close()

# 图4：样本外 R² 对比（MIDAS vs AR(1) vs 朴素）
fig, ax = plt.subplots(figsize=(11, 5.0))
names = ["Beta-MIDAS\n(日频混频)", "AR(1)\n(仅 y 历史)", "朴素\n(上月值)"]
vals = [r2_midas, r2_ar, r2_naive]
colors = ["#c44e52", "#4c72b0", "#999999"]
bars = ax.bar(names, vals, color=colors, alpha=0.85)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.01 if v >= 0 else -0.03), f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.axhline(0, color="#333", lw=1)
ax.set_ylabel("样本外 R²", fontsize=11)
ax.set_title("样本外预测力：混频 MIDAS 显著优于单频基准",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "midas_oos_r2.png"), dpi=150, bbox_inches="tight")
plt.close()

print("=== MIDAS 诊断 ===")
print(f"真实: β0={beta0} β1={beta1} | 估计: β0={b0_h:.3f} β1={b1_h:.3f} θ=({th1_h:.2f},{th2_h:.2f})")
print(f"样本内 R² = {r2_in:.3f}")
print(f"样本外 R²: MIDAS={r2_midas:.3f}  AR(1)={r2_ar:.3f}  朴素={r2_naive:.3f}")
print(f"生成图片: {sorted(os.listdir(D))}")
