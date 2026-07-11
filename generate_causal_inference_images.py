#!/usr/bin/env python3
"""
为文章「因果推断与双重差分(DID)：从相关性到策略可解释性」(causal-inference-quant)
生成真实配图（matplotlib，非占位图，全部可复现）。

模拟设定：
  - 90 只股票，120 个交易日，第 60 天发生一个政策/事件冲击；
  - 30 只为处理组（受事件影响），60 只控制组（不受影响）；
  - 个体固定效应 α_i + 时间固定效应 γ_t（共同市场趋势，保证平行趋势）
    + 处理效应 τ（仅处理组在事件后获得）+ 噪声；
  - 用双重差分(DID)估计 τ，集群 bootstrap 给 95% 置信区间；
  - 事件研究给出处理组−控制组的累计异常收益(CAR)。
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
D = os.path.join(BASE, "causal-inference-quant")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260711)
N = 90
T = 120
t0 = 60                      # 事件发生在第 t0 天
n_treat = 30
tau_true = 0.0040            # 处理效应：处理组事件后每天多 0.40% 异常收益

treated = np.zeros(N, dtype=bool)
treated[:n_treat] = True

# 个体固定效应
alpha = rng.normal(0.0, 0.004, N)
# 共同时间趋势（处理组与控制组共享 -> 平行趋势）
gamma = 0.0010 * np.sin(np.arange(T) / 9.0) + 0.00002 * np.arange(T)
post = (np.arange(T) >= t0).astype(float)

# 生成面板数据 y_{it}（单位：日异常收益）
Y = np.zeros((N, T))
for i in range(N):
    eps = rng.normal(0.0, 0.010, T)
    Y[i] = alpha[i] + gamma + tau_true * treated[i] * post + eps

# ---- DID 估计量（显式公式）----
post_bool = post == 1

def did_estimate(Y, treated, post_bool):
    t_p = Y[np.ix_(treated, post_bool)].mean()
    t_pre = Y[np.ix_(treated, ~post_bool)].mean()
    c_p = Y[np.ix_(~treated, post_bool)].mean()
    c_pre = Y[np.ix_(~treated, ~post_bool)].mean()
    return (t_p - t_pre) - (c_p - c_pre)

tau_hat = did_estimate(Y, treated, post_bool)

# ---- 集群 bootstrap 置信区间（以股票为聚类单位）----
B = 2000
boot = np.empty(B)
idx = np.arange(N)
for b in range(B):
    samp = rng.choice(idx, size=N, replace=True)
    boot[b] = did_estimate(Y[samp], treated[samp], post_bool)
ci_lo, ci_hi = np.percentile(boot, [2.5, 97.5])

# ---- 平行趋势检验：事件前处理组与控制组的斜率差 ----
pre_mask = post == 0
# 各自在事件前的样本均值随时间（这里简化处理：比较事件前两段斜率）
pre_t = np.arange(t0)
treat_pre_mean = Y[treated][:, pre_mask].mean(axis=0)
ctrl_pre_mean = Y[~treated][:, pre_mask].mean(axis=0)
slope_treat = np.polyfit(pre_t, treat_pre_mean, 1)[0]
slope_ctrl = np.polyfit(pre_t, ctrl_pre_mean, 1)[0]
pre_trend_gap = abs(slope_treat - slope_ctrl)

# ---- 事件研究：处理组 - 控制组的日度差（去除共同时间效应）----
diff = Y[treated].mean(axis=0) - Y[~treated].mean(axis=0)
win = 20
car = np.cumsum(diff)
car_win = car[(t0 - win):(t0 + win + 1)]
days = np.arange(-win, win + 1)

# ================= 图 1：平行趋势 =================
fig, ax = plt.subplots(figsize=(8.4, 5.0))
treat_mean = Y[treated].mean(axis=0)
ctrl_mean = Y[~treated].mean(axis=0)
ax.plot(np.arange(T), treat_mean * 100, "-", color="#d62728", lw=2.2, label="处理组（受事件影响）")
ax.plot(np.arange(T), ctrl_mean * 100, "-", color="#1f77b4", lw=2.2, label="控制组（不受影响）")
ax.axvline(t0, color="#333", ls="--", lw=1.4)
ax.text(t0 + 1, ax.get_ylim()[1] * 0.96, "事件日", fontsize=10)
ax.axvspan(0, t0, color="#888", alpha=0.07)
ax.axvspan(t0, T, color="#d62728", alpha=0.05)
ax.set_xlabel("交易日", fontsize=12)
ax.set_ylabel("平均日异常收益 (%)", fontsize=12)
ax.set_title("平行趋势：事件前两组走势一致，事件后才分叉（DID 的核心前提）", fontsize=12.5)
ax.legend(fontsize=10, loc="lower right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_parallel_trends.png"), dpi=130); plt.close(fig)

# ================= 图 2：DID 估计量 + bootstrap 分布 =================
cell_means = [
    Y[np.ix_(treated, ~post_bool)].mean() * 100,
    Y[np.ix_(treated, post_bool)].mean() * 100,
    Y[np.ix_(~treated, ~post_bool)].mean() * 100,
    Y[np.ix_(~treated, post_bool)].mean() * 100,
]
labels = ["处理·前", "处理·后", "控制·前", "控制·后"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
colors = ["#d62728", "#d62728", "#1f77b4", "#1f77b4"]
a1.bar(labels, cell_means, color=colors, alpha=0.85)
for j, v in enumerate(cell_means):
    a1.text(j, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
a1.set_ylabel("平均日异常收益 (%)", fontsize=11)
a1.set_title(f"四格均值与 DID = {tau_hat*100:.3f}%", fontsize=11.5)
a1.annotate("",
            xy=(1, cell_means[1]), xytext=(0, cell_means[0]),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.6))
a1.annotate("",
            xy=(3, cell_means[3]), xytext=(2, cell_means[2]),
            arrowprops=dict(arrowstyle="->", color="#1f77b4", lw=1.6))
a1.grid(axis="y", alpha=0.3)

a2.hist(boot * 100, bins=50, color="#2ca02c", alpha=0.8)
a2.axvline(tau_hat * 100, color="#d62728", lw=2, label=f"DID 估计 {tau_hat*100:.3f}%")
a2.axvline(tau_true * 100, color="#333", ls="--", lw=1.6, label=f"真实 τ {tau_true*100:.2f}%")
a2.axvline(ci_lo * 100, color="#ff7f0e", ls=":", lw=1.6)
a2.axvline(ci_hi * 100, color="#ff7f0e", ls=":", lw=1.6, label=f"95% CI [{ci_lo*100:.3f}, {ci_hi*100:.3f}]")
a2.set_xlabel("DID 估计量 (%)", fontsize=11)
a2.set_ylabel("bootstrap 频次", fontsize=11)
a2.set_title("集群 bootstrap 2000 次", fontsize=11.5)
a2.legend(fontsize=8.5); a2.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_did_estimator.png"), dpi=130); plt.close(fig)

# ================= 图 3：事件研究 CAR =================
fig, ax = plt.subplots(figsize=(8.4, 5.0))
ax.plot(days, car_win * 100, "-o", color="#9467bd", ms=4, lw=2, label="处理组−控制组 CAR")
ax.axvline(0, color="#333", ls="--", lw=1.4)
ax.text(0.5, ax.get_ylim()[1] * 0.85, "事件日", fontsize=10)
ax.axhline(0, color="#888", lw=1.0)
# 事件后理论累计 = tau_true * 窗口长度
ax.plot([0, win], [0, tau_true * win * 100], ls=":", color="#2ca02c", lw=1.6,
        label=f"理论斜率 τ={tau_true*100:.2f}%/日")
ax.set_xlabel("相对事件日的交易日", fontsize=12)
ax.set_ylabel("累计异常收益 (%)", fontsize=12)
ax.set_title("事件研究：事件前 CAR 平稳，事件后稳步抬升", fontsize=12.5)
ax.legend(fontsize=9.5, loc="upper left"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_event_study.png"), dpi=130); plt.close(fig)

# 保存数值
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    f.write(f"N={N} T={T} n_treat={n_treat} tau_true={tau_true}\n")
    f.write(f"tau_hat={tau_hat:.6f}\n")
    f.write(f"ci=[{ci_lo:.6f},{ci_hi:.6f}]\n")
    f.write(f"pre_trend_gap_slope={pre_trend_gap:.6e}\n")
    f.write(f"car_final={car_win[-1]:.6f}\n")
    f.write(f"cell_means(percent)={','.join(f'{v:.4f}' for v in cell_means)}\n")

print("✅ causal-inference 配图生成完成")
print(f"tau_true={tau_true*100:.3f}%  tau_hat={tau_hat*100:.3f}%  CI=[{ci_lo*100:.3f}%,{ci_hi*100:.3f}%]")
print(f"pre-trend slope gap={pre_trend_gap:.2e} (应接近0)  CAR(final)={car_win[-1]*100:.3f}%")
