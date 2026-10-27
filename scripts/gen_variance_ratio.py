#!/usr/bin/env python3
"""
为文章「方差比检验市场有效性」(variance-ratio-test) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 三条对比序列：
      (a) 随机游走（有效市场 → VR(k) 应≈1）
      (b) 均值回复（反趋势 → VR(k) < 1）
      (c) 趋势/动量（正自相关 → VR(k) > 1）
  - 用 Lo & MacKinlay (1988) 方差比 VR(k) = Var(k期收益)/k / Var(1期收益)，
    并用其同方差检验统计量 M_r(k) = (VR-1)/sqrt(theta(k)) 做 z 检验。
  - 给出 VR(k) 随 k 变化的柱状图（三 regime 对比）。
  - 给出一条「类市场」序列的滚动 VR(5) + 95% 置信带，演示不误杀有效市场。
  - 用 Monte-Carlo 模拟随机游走零假设，得到 VR(10) 的经验分布与经验 p 值。
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
D = os.path.join(BASE, "variance-ratio-test")
os.makedirs(D, exist_ok=True)

C = {
    "grid": "#DDDDDD",
    "pos": "#2F4B7C",
    "neg": "#C44E52",
    "ls": "#55A868",
    "mk": "#8172B3",
    "gold": "#E1A100",
    "blue": "#4C72B0",
}

# ============================================================
# 1) 合成三类序列
# ============================================================
def make_rw(n=2000, seed=1):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, n)
    return np.cumsum(r), r

def make_mean_rev(n=2000, seed=2, rho=0.30):
    """收益序列 AR(1) 系数取负 → 负自相关 → 价格均值回复 → VR(k)<1。"""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    r[0] = rng.normal(0, 0.01)
    for t in range(1, n):
        r[t] = -rho * r[t - 1] + rng.normal(0, 0.01)
    return np.cumsum(r), r

def make_trend(n=2000, seed=3, rho=0.25):
    """收益序列 AR(1) 系数取正 → 正自相关（动量）→ VR(k)>1。"""
    rng = np.random.default_rng(seed)
    r = np.zeros(n)
    r[0] = rng.normal(0, 0.01)
    for t in range(1, n):
        r[t] = rho * r[t - 1] + rng.normal(0, 0.01)
    return np.cumsum(r), r

P_rw, r_rw = make_rw()
P_mr, r_mr = make_mean_rev(rho=0.30)
P_td, r_td = make_trend(rho=0.25)

# ============================================================
# 2) VR(k) 计算 + Lo-MacKinlay 同方差检验统计量
# ============================================================
def vr_stat(r, k):
    n = len(r)
    mu = r.mean()
    var1 = np.sum((r - mu) ** 2) / n
    nk = n // k
    yk = r[: nk * k].reshape(nk, k).sum(axis=1)
    var_k = np.sum((yk - yk.mean()) ** 2) / (nk - 1)
    VR = var_k / (k * max(var1, 1e-12))
    theta = (2 * (2 * k - 1) * (k - 1)) / (3 * k * n)
    M = (VR - 1) / np.sqrt(theta)
    return VR, M

Ks = [2, 4, 5, 10]
VR_rw = [vr_stat(r_rw, k)[0] for k in Ks]
VR_mr = [vr_stat(r_mr, k)[0] for k in Ks]
VR_td = [vr_stat(r_td, k)[0] for k in Ks]

# ============================================================
# 图一 (cover)：三序列价格路径
# ============================================================
fig, axs = plt.subplots(3, 1, figsize=(9, 6.6), sharex=False)
tt = np.arange(len(P_rw))
axs[0].plot(tt, P_rw, color=C["pos"], lw=0.9)
axs[0].set_ylabel("价格"); axs[0].set_title("随机游走（有效市场）→ VR(k)≈1")
axs[0].grid(True, color=C["grid"])
axs[1].plot(tt, P_mr, color=C["ls"], lw=0.9)
axs[1].set_ylabel("价格"); axs[1].set_title("均值回复（反趋势）→ VR(k)<1")
axs[1].grid(True, color=C["grid"])
axs[2].plot(tt, P_td, color=C["neg"], lw=0.9)
axs[2].set_ylabel("价格"); axs[2].set_xlabel("时间 t")
axs[2].set_title("趋势/动量（正自相关）→ VR(k)>1")
axs[2].grid(True, color=C["grid"])
fig.suptitle("市场有效性检验：三类序列，方差比对它们「一测便知」", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图二：VR(k) 柱状图（三 regime）
# ============================================================
x = np.arange(len(Ks))
w = 0.26
fig, ax = plt.subplots(figsize=(8.5, 5))
b1 = ax.bar(x - w, VR_rw, w, color=C["pos"], label="随机游走 (有效)")
b2 = ax.bar(x, VR_mr, w, color=C["ls"], label="均值回复")
b3 = ax.bar(x + w, VR_td, w, color=C["neg"], label="趋势/动量")
ax.axhline(1.0, color="black", lw=1.2, ls="--", label="VR=1（有效市场边界）")
ax.set_xticks(x); ax.set_xticklabels([f"k={k}" for k in Ks])
ax.set_ylabel("方差比 VR(k)")
ax.set_title("VR(k) 随持有期 k 的变化：有效市场恒在 1 附近")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], axis="y")
for bars in (b1, b2, b3):
    for rect in bars:
        ax.text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.01,
                f"{rect.get_height():.2f}", ha="center", fontsize=7)
fig.tight_layout()
fig.savefig(os.path.join(D, "vr_bars.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图三：滚动 VR(5) + 95% 置信带（类市场序列，不应误杀）
# ============================================================
def rolling_vr(r, k=5, W=250):
    out = np.full(len(r), np.nan)
    for t in range(W, len(r)):
        out[t] = vr_stat(r[t - W:t], k)[0]
    return out

rvr = rolling_vr(r_rw, k=5, W=250)
# 95% 带：用 theta(k) 构造 (近似)
n = 250
k = 5
theta = (2 * (2 * k - 1) * (k - 1)) / (3 * k * n)
lo, hi = 1 - 1.96 * np.sqrt(theta), 1 + 1.96 * np.sqrt(theta)
fig, ax = plt.subplots(figsize=(9, 4.6))
tt = np.arange(len(rvr))
mask = ~np.isnan(rvr)
ax.plot(tt[mask], rvr[mask], color=C["gold"], lw=1.0, label="滚动 VR(5)")
ax.axhline(1.0, color="black", lw=1.2, ls="--", label="VR=1")
ax.axhspan(lo, hi, color=C["ls"], alpha=0.18, label="95% 置信带")
ax.set_xlabel("时间 t"); ax.set_ylabel("VR(5)")
ax.set_title("随机游走序列的滚动 VR(5)：多数时间在 1 附近、落于置信带内（不误杀有效市场）")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "vr_rolling.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图四：Monte-Carlo 零假设下 VR(10) 分布 + 经验 p 值
# ============================================================
rng = np.random.default_rng(20260721)
N_MC = 4000
vr_null = np.zeros(N_MC)
for i in range(N_MC):
    rr = rng.normal(0, 0.01, 2000)
    vr_null[i] = vr_stat(rr, 10)[0]
# 用随机游走序列自身的 VR(10) 作为「待检验序列」
vr_obs, M_obs = vr_stat(r_rw, 10)
# 双尾经验 p 值
p_val = np.mean(np.abs(vr_null - 1) >= np.abs(vr_obs - 1))

fig, ax = plt.subplots(figsize=(8.5, 4.8))
ax.hist(vr_null, bins=60, color=C["blue"], alpha=0.85, edgecolor="white", density=True)
ax.axvline(1.0, color="black", lw=1.5, ls="--", label="零假设中心 VR=1")
ax.axvline(vr_obs, color=C["neg"], lw=1.8, label=f"待检验序列 VR(10)={vr_obs:.3f}")
ax.set_xlabel("VR(10)"); ax.set_ylabel("密度")
ax.set_title(f"随机游走零假设下 VR(10) 经验分布（MC={N_MC}）\n经验 p 值={p_val:.3f}（不拒绝：该序列近似有效市场）")
ax.legend(fontsize=8); ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "vr_distribution.png"), dpi=130)
plt.close(fig)

print(f"VR images done. VR_rw={VR_rw}, VR_mr={VR_mr}, VR_td={VR_td}")
print(f"vr_obs={vr_obs:.4f}, M_obs={M_obs:.3f}, MC p={p_val:.3f}")
