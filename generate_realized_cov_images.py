#!/usr/bin/env python3
"""
为文章「已实现协方差矩阵：用高频数据拼出日内风险全景」(realized-covariance-highfreq)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

数据模型：5 只资产，单因子 + 特异性 + 价格层微观结构噪声（噪声注入在"价格"上，
所以高频差分回报里会带 MA(1) 噪声项 —— 这正是已实现方差在极高频被高估、且
相关系数在极细采样下塌向 0（Epps 效应）的根源）。

图表：
  1. realized_cov_heatmap.png   5 资产已实现相关矩阵（高频聚合）—— 日内风险全景
  2. noise_ushape.png            已实现方差 vs 采样频率：微观结构噪声造成的 U 形偏差
  3. epps_effect.png            相关系数 vs 采样频率：朴素 RC 的 Epps 塌缩 vs 已实现核恢复
  4. har_forecast.png           HAR-RV：用高频已实现波动预测次日的日度波动

数值校验：已实现核(RK)的方差应贴近"真实总方差 TV"，朴素 1 分钟 RC 显著高估；
          相关矩阵应呈现单因子结构（对角线强、因子资产间相关高）。
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
D = os.path.join(BASE, "realized-covariance-highfreq")
os.makedirs(D, exist_ok=True)

C = {"blue": "#4C72B0", "red": "#C44E52", "green": "#55A868",
     "purple": "#8172B3", "orange": "#CCB974", "grid": "#DDDDDD", "band": "#EAEAF2"}

# ============================================================
# 1. 真实协方差结构：单因子模型  Sigma = beta*beta' + diag(delta^2)
# ============================================================
np.random.seed(7)
K = 5
beta = np.array([1.00, 0.80, 1.20, 0.55, 0.95])      # 因子载荷（资产对共同因子的暴露）
delta = np.array([0.30, 0.42, 0.28, 0.50, 0.38])    # 特异性波动
Sigma_true = np.outer(beta, beta) + np.diag(delta ** 2)

# 日内设置：每天 240 根 1 分钟 bar，共 60 个交易日
N_PERDAY = 240
N_DAYS = 60
N_TOTAL = N_PERDAY * N_DAYS
SIGMA_1MIN = Sigma_true / N_PERDAY                 # 1 分钟真实回报协方差（日方差/240）

# 微观结构噪声：注入在"价格"上，u_t ~ N(0, eta^2 I)，独立同分布
# 让 m=1 时的噪声偏差约等于 1.5 倍真实总方差，制造明显 U 形
TV_avg = N_DAYS * np.mean(np.diag(Sigma_true))     # 全样本真实总方差（平均到资产）
ETA2 = 1.5 * TV_avg / (2.0 * N_TOTAL)
ETA = np.sqrt(ETA2)

print("=== 模型参数 ===")
print("Sigma_true 对角线(日方差):", np.round(np.diag(Sigma_true), 4))
print("TV_avg(全样本真实总方差, 平均资产):", round(TV_avg, 3))
print("ETA2(噪声方差):", round(ETA2, 6), " ETA:", round(ETA, 5))

# ============================================================
# 2. 模拟真实价格路径 + 观测价格（含噪声）
# ============================================================
def simulate():
    rng = np.random.default_rng(2026)
    # 真实 1 分钟回报：N(0, SIGMA_1MIN)
    true_ret = rng.multivariate_normal(np.zeros(K), SIGMA_1MIN, size=N_TOTAL)
    p_true = np.cumsum(true_ret, axis=0)                       # 真实对数价格
    noise = rng.normal(0, ETA, size=(N_TOTAL, K))              # 价格层噪声
    p_obs = p_true + noise                                    # 观测价格
    obs_ret = np.diff(p_obs, axis=0)                          # 观测 1 分钟回报（含 MA(1) 噪声）
    return p_obs, obs_ret

P_OBS, OBS_RET = simulate()

# ============================================================
# 3. 朴素已实现协方差 / 相关（按频率 m 聚合；这里先全样本）
# ============================================================
def realized_cov(returns):
    """朴素已实现协方差 = Σ_t r_t r_t'（returns: (T, K)）"""
    return returns.T @ returns

def corr_from_cov(cov):
    d = np.sqrt(np.diag(cov))
    return cov / np.outer(d, d)

# 全样本 1 分钟朴素 RC
RC_1MIN = realized_cov(OBS_RET)
CORR_1MIN = corr_from_cov(RC_1MIN)

# ============================================================
# 4. 已实现核估计（Barndorff-Nielsen et al. 2008），逐元素 Bartlett 权重
#    修正微观结构噪声偏差与异步交易偏差
# ============================================================
def realized_kernel(returns, c=3.7):
    T, k = returns.shape
    H = int(np.floor(c * T ** (1 / 3)))            # 带宽 ~ c * T^{1/3}
    H = min(H, T - 1)
    # 各阶自协方差 γ_h = Σ_t r_t r_{t-h}'
    g = np.zeros((H + 1, k, k))
    g[0] = returns.T @ returns
    for h in range(1, H + 1):
        a = returns[h:]
        b = returns[:-h]
        g[h] = a.T @ b
    # Bartlett 权重 k_h = 1 - h/(H+1)
    w = 1.0 - np.arange(H + 1) / (H + 1)
    RK = g[0].copy()
    for h in range(1, H + 1):
        RK += w[h] * (g[h] + g[h].T)
    return RK, H

RK_ALL, H_RK = realized_kernel(OBS_RET)
CORR_RK = corr_from_cov(RK_ALL)

print("\n=== 已实现核带宽 H =", H_RK, "===")
print("朴素 1min RC 平均方差:", round(np.mean(np.diag(RC_1MIN)), 3),
      " | RK 平均方差:", round(np.mean(np.diag(RK_ALL)), 3),
      " | 真实 TV_avg:", round(TV_avg, 3))
print("朴素 1min RC 平均相关:", round(np.mean(CORR_1MIN[np.triu_indices(K, 1)]), 4),
      " | RK 平均相关:", round(np.mean(CORR_RK[np.triu_indices(K, 1)]), 4),
      " | 真实平均相关:", round(np.mean(((Sigma_true/np.outer(np.sqrt(np.diag(Sigma_true)),np.sqrt(np.diag(Sigma_true)))))[np.triu_indices(K,1)]), 4))

# ============================================================
# 图 1：已实现相关矩阵热图（高频聚合全景）
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
labels = [f"资产{i+1}" for i in range(K)]
corr_true = Sigma_true / np.outer(np.sqrt(np.diag(Sigma_true)), np.sqrt(np.diag(Sigma_true)))
for ax, M, title in [(axes[0], corr_true, "真实相关矩阵（模型设定）"),
                     (axes[1], CORR_RK, "已实现相关矩阵（高频 + 已实现核）")]:
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(K)); ax.set_yticks(range(K))
    ax.set_xticklabels(labels); ax.set_yticklabels(labels)
    for i in range(K):
        for j in range(K):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="black", fontsize=9)
    ax.set_title(title, fontsize=11)
fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
fig.suptitle("用高频数据拼出的日内风险全景：5 资产已实现相关结构", fontsize=13)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig(os.path.join(D, "realized_cov_heatmap.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：已实现方差 vs 采样频率（U 形噪声偏差）
# ============================================================
freqs = [1, 2, 3, 5, 10, 15, 30, 60, 120, 240]
rv_by_freq = []
for m in freqs:
    # 用整段观测价格做跨度为 m 分钟的差分
    coarse = P_OBS[m:] - P_OBS[:-m]                 # (T-m, K)
    rv = np.sum(coarse ** 2, axis=0) / (len(coarse) / N_TOTAL)  # 归一到全样本尺度
    rv_by_freq.append(rv)
rv_by_freq = np.array(rv_by_freq)
rv_avg = rv_by_freq.mean(axis=1)                    # 资产平均已实现方差

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(freqs, rv_avg, "o-", color=C["blue"], lw=1.8, ms=7, label="朴素已实现方差(各频率)")
ax.axhline(TV_avg, color=C["red"], ls="--", lw=2, label=f"真实总方差 TV={TV_avg:.1f}")
ax.fill_between(freqs, TV_avg * 0.9, TV_avg * 1.1, color=C["band"], alpha=0.6, label="真实值邻域")
ax.set_xscale("log")
ax.set_xlabel("采样频率（分钟 / bar）")
ax.set_ylabel("已实现方差（全样本，资产平均）")
ax.set_title("微观结构噪声的 U 形偏差：极高频被噪声高估，极粗频被采样方差拖高")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
ax.annotate("噪声偏差主导\n(MA(1) 噪声)", xy=(1, rv_avg[0]), xytext=(2.2, rv_avg[0] * 0.78),
            fontsize=9, color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"]))
plt.tight_layout()
plt.savefig(os.path.join(D, "noise_ushape.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：Epps 效应 —— 相关系数 vs 采样频率
# ============================================================
pair = (0, 2)                                      # 资产1-资产3（高因子暴露对）
corr_naive, corr_rk = [], []
for m in freqs:
    coarse = P_OBS[m:] - P_OBS[:-m]
    cov = coarse.T @ coarse
    d = np.sqrt(np.diag(cov))
    cmat = cov / np.outer(d, d)
    corr_naive.append(cmat[pair])
    # 该频率下也做已实现核（用 coarse 序列）
    rk, _ = realized_kernel(coarse)
    dk = np.sqrt(np.diag(rk))
    corr_rk.append((rk / np.outer(dk, dk))[pair])
corr_naive = np.array(corr_naive)
corr_rk = np.array(corr_rk)
true_pair = corr_true[pair]

fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(freqs, corr_naive, "o-", color=C["red"], lw=1.8, ms=7, label="朴素已实现相关 (RC)")
ax.plot(freqs, corr_rk, "s--", color=C["green"], lw=1.8, ms=6, label="已实现核相关 (RK)")
ax.axhline(true_pair, color=C["blue"], ls=":", lw=2, label=f"真实相关={true_pair:.2f}")
ax.set_xscale("log")
ax.set_xlabel("采样频率（分钟 / bar）")
ax.set_ylabel(f"相关系数 (资产{pair[0]+1}-资产{pair[1]+1})")
ax.set_title("Epps 效应：采样越细，朴素相关越塌向 0；已实现核把它救回真实值")
ax.legend(loc="lower right", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
ax.set_ylim(-0.05, 1.05)
plt.tight_layout()
plt.savefig(os.path.join(D, "epps_effect.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：HAR-RV —— 用高频已实现波动预测次日日度波动
#    RV_t = b0 + b1*RV_{t-1} + b2*(近5日均值) + b3*(近22日均值) + e
# ============================================================
daily_rv = np.array([np.sum(OBS_RET[d * N_PERDAY:(d + 1) * N_PERDAY] ** 2, axis=0)
                     for d in range(N_DAYS)])       # (N_DAYS, K)
rv0 = daily_rv[:, 0]                                               # 用资产1的日度 RV
L = len(rv0)
y, X = [], []
for t in range(22, L):
    y.append(rv0[t])
    d_lag = rv0[t - 1]
    w_lag = rv0[t - 5:t].mean()
    m_lag = rv0[t - 22:t].mean()
    X.append([1.0, d_lag, w_lag, m_lag])
y = np.array(y); X = np.array(X)
beta_har, *_ = np.linalg.lstsq(X, y, rcond=None)
fitted = X @ beta_har
r2 = 1 - np.sum((y - fitted) ** 2) / np.sum((y - y.mean()) ** 2)

fig, ax = plt.subplots(figsize=(10, 5.2))
tt = np.arange(22, L)
ax.plot(tt, y, color=C["blue"], lw=1.2, alpha=0.7, label="真实日度已实现波动 RV_t")
ax.plot(tt, fitted, color=C["red"], lw=1.6, label=f"HAR 拟合 (R²={r2:.2f})")
ax.set_xlabel("交易日")
ax.set_ylabel("日度已实现方差")
ax.set_title("HAR-RV：用日/周/月三个时间尺度的已实现波动预测次日波动")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "har_forecast.png"), dpi=130)
plt.close()

print("\nHAR 系数 [截距, 日, 周, 月]:", np.round(beta_har, 5), " 样本内 R²:", round(r2, 3))
print("图片已保存到:", D)
