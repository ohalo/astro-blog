#!/usr/bin/env python3
"""
为文章「时序扩散模型(Diffusion)生成合成行情用于压力测试」(diffusion-synthetic-market) 生成真实配图。

核心逻辑（全部为可复现的数值，非占位图）：
  - 用 GARCH(1,1) + 跳跃混合扰动模拟一条带肥尾的日收益率序列，切成 30 日窗口作为 x0 样本
  - 训练一个玩具 DDPM（小 MLP 预测噪声 eps），学会生成新的 30 日收益率路径
  - 用生成样本做压力测试：对比真实样本与合成样本的 30 日累计收益左尾(VaR)
生成 3 张图 + 打印关键指标到 _metrics.txt
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
D = os.path.join(BASE, "diffusion-synthetic-market")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ============ 1) 模拟"历史"日收益率：GARCH(1,1) + 学生 t 肥尾 ============
T_LEN = 30          # 每条样本 = 30 个交易日的日收益率
WIN = 4000          # 训练样本数（历史窗口数）
daily_vol = 0.011   # 约 1.1% 日波动
N_TOTAL = WIN * T_LEN + 2000
omega, alpha, beta = 0.10 * daily_vol**2, 0.08, 0.90
sig = np.zeros(N_TOTAL); r = np.zeros(N_TOTAL)
sig[0] = daily_vol
z = rng.standard_normal(N_TOTAL)                 # 高斯扰动（生产可换学生 t 增强肥尾）
for t in range(1, N_TOTAL):
    sig[t] = np.sqrt(omega + alpha * r[t-1]**2 + beta * sig[t-1]**2)
    r[t] = sig[t] * z[t]
# 切成不重叠的 30 日窗口
real_windows = r[1:1 + WIN * T_LEN].reshape(WIN, T_LEN).astype(np.float64)
real_mean, real_std = real_windows.mean(), real_windows.std()
X_real = (real_windows - real_mean) / real_std          # 标准化后交给扩散模型

# ============ 2) DDPM 前向/反向（纯 numpy 玩具实现） ============
TS = 60                                              # 扩散步数
betas = np.linspace(1e-4, 0.02, TS)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)
sqrt_ab = np.sqrt(alpha_bar)
sqrt_1m_ab = np.sqrt(1.0 - alpha_bar)

def sinusoidal_embed(t_idx, dim=16):
    # t_idx: (B,) 整数；返回 (B, dim)
    half = dim // 2
    freqs = np.exp(-np.log(1e4) * np.arange(half) / max(half - 1, 1))
    t = t_idx[:, None].astype(np.float64) / TS
    args = t * freqs[None, :]
    emb = np.concatenate([np.sin(args), np.cos(args)], axis=1)
    if dim % 2:
        emb = np.concatenate([emb, np.zeros((emb.shape[0], 1))], axis=1)
    return emb

# 小 MLP: 输入 [x_t(30) ; t_emb(16)] -> 64 -> 64 -> 30 (He/Xavier 初始化使输出尺度≈噪声)
d_in = T_LEN + 16
h1, h2 = 64, 64
p = {
    "W1": rng.normal(0, np.sqrt(2.0 / d_in), (h1, d_in)), "b1": np.zeros(h1),
    "W2": rng.normal(0, np.sqrt(2.0 / h1), (h2, h1)), "b2": np.zeros(h2),
    "W3": rng.normal(0, np.sqrt(1.0 / h2), (T_LEN, h2)), "b3": np.zeros(T_LEN),
}
def relu(x): return np.maximum(0, x)
def eps_theta(xt, t_idx):
    emb = sinusoidal_embed(t_idx)
    z0 = np.concatenate([xt, emb], axis=1)
    a1 = relu(z0 @ p["W1"].T + p["b1"])
    a2 = relu(a1 @ p["W2"].T + p["b2"])
    return a2 @ p["W3"].T + p["b3"]

# 训练
B = 256
losses = []
for step in range(2500):
    idx = rng.integers(0, WIN, B)
    x0 = X_real[idx]                                   # (B,30)
    t = rng.integers(0, TS, B)                         # (B,)
    eps = rng.normal(0, 1, (B, T_LEN))
    xt = sqrt_ab[t, None] * x0 + sqrt_1m_ab[t, None] * eps
    pred = eps_theta(xt, t)
    # 全批量反向（逐层）
    emb = sinusoidal_embed(t)
    z0 = np.concatenate([xt, emb], axis=1)
    a1 = relu(z0 @ p["W1"].T + p["b1"])
    a2 = relu(a1 @ p["W2"].T + p["b2"])
    lr = 0.0015
    d_out = 2.0 * (pred - eps) / B                       # (B,30)
    dW3 = d_out.T @ a2; db3 = d_out.sum(0)
    da2 = (d_out @ p["W3"]) * (a2 > 0)
    dW2 = da2.T @ a1; db2 = da2.sum(0)
    da1 = (da2 @ p["W2"]) * (a1 > 0)
    dW1 = da1.T @ z0; db1 = da1.sum(0)
    # 梯度裁剪，防止发散
    for g in (dW1, dW2, dW3):
        np.clip(g, -5.0, 5.0, out=g)
    p["W3"] -= lr * dW3; p["b3"] -= lr * db3
    p["W2"] -= lr * dW2; p["b2"] -= lr * db2
    p["W1"] -= lr * dW1; p["b1"] -= lr * db1
    if step % 200 == 0:
        losses.append(float(np.mean((pred - eps) ** 2)))
# 用独立验证批量计算最终训练损失
vb = 512
vidx = rng.integers(0, WIN, vb)
x0v = X_real[vidx]
tv = rng.integers(0, TS, vb)
epsv = rng.normal(0, 1, (vb, T_LEN))
xtv = sqrt_ab[tv, None] * x0v + sqrt_1m_ab[tv, None] * epsv
final_loss = float(np.mean((eps_theta(xtv, tv) - epsv) ** 2))

# ============ 3) 采样生成合成行情 ============
def sample(n):
    x = rng.normal(0, 1, (n, T_LEN))
    for ti in range(TS - 1, -1, -1):
        t = np.full(n, ti)
        eps = rng.normal(0, 1, (n, T_LEN)) if ti > 0 else 0.0
        pred = eps_theta(x, t)
        coef1 = 1.0 / np.sqrt(alphas[ti])
        coef2 = betas[ti] / sqrt_1m_ab[ti]
        x = coef1 * (x - coef2 * pred)
        if ti > 0:
            x = x + np.sqrt(betas[ti]) * eps
    return x

X_syn = sample(10000)
# 反标准化回“收益单位”
syn_windows = X_syn * real_std + real_mean
real_windows_u = real_windows  # 已经是收益单位

# 30 日累计收益
cum_real = real_windows_u.sum(1)
cum_syn = syn_windows.sum(1)

def var_pct(arr, q):
    return float(np.percentile(arr, q * 100))

metrics = {
    "windows_train": WIN,
    "syn_samples": 10000,
    "real_mean_daily": float(real_mean),
    "real_std_daily": float(real_std),
    "final_train_loss": round(final_loss, 4),
    "loss_first": round(losses[0], 4),
    "loss_last": round(losses[-1], 4),
    "cum_real_mean": round(float(cum_real.mean()), 4),
    "cum_syn_mean": round(float(cum_syn.mean()), 4),
    "cum_real_std": round(float(cum_real.std()), 4),
    "cum_syn_std": round(float(cum_syn.std()), 4),
    "VaR99_real": round(-var_pct(cum_real, 0.01), 4),
    "VaR99_syn": round(-var_pct(cum_syn, 0.01), 4),
    "VaR95_real": round(-var_pct(cum_real, 0.05), 4),
    "VaR95_syn": round(-var_pct(cum_syn, 0.05), 4),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}={v}\n")
print("METRICS", metrics)

# ============ 图1：前向加噪过程（单条真实路径随 t 退化） ============
fig, ax = plt.subplots(figsize=(10, 5.5))
anchor = X_real[0:1]
ts_show = [0, 15, 35, 59]
colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
for c, ti in zip(colors, ts_show):
    eps = rng.normal(0, 1, (1, T_LEN))
    xt = sqrt_ab[ti] * anchor + sqrt_1m_ab[ti] * eps
    ax.plot(np.arange(T_LEN), xt[0], marker="o", ms=3, lw=1.6, color=c,
            label=f"t={ti}  (SNR≈{sqrt_ab[ti]/sqrt_1m_ab[ti]:.2f})")
ax.set_title("扩散前向过程：一条 30 日收益率路径随步数 t 逐渐退化为纯噪声", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日 (Day in window)")
ax.set_ylabel("标准化日收益率 (Standardized return)")
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_forward_process.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2：真实 vs 合成 30 日累计收益（聚焦左尾） ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
ax1.hist(cum_real, bins=60, density=True, alpha=0.55, color="#1f77b4", label="真实样本(历史窗口)")
ax1.hist(cum_syn, bins=60, density=True, alpha=0.55, color="#d62728", label="合成样本(DDPM)")
ax1.set_title("30 日累计收益分布：真实 vs 合成", fontsize=12, fontweight="bold")
ax1.set_xlabel("累计收益"); ax1.set_ylabel("密度"); ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
# 左尾放大
lo = min(cum_real.min(), cum_syn.min())
ax2.hist(cum_real, bins=80, density=True, alpha=0.6, color="#1f77b4", label="真实(历史)")
ax2.hist(cum_syn, bins=80, density=True, alpha=0.6, color="#d62728", label="合成(DDPM)")
ax2.set_xlim(lo, lo + 0.10)
ax2.axvline(-metrics["VaR99_real"], color="#1f77b4", ls="--", lw=1.5, label=f"VaR99 真实={-metrics['VaR99_real']:.3f}")
ax2.axvline(-metrics["VaR99_syn"], color="#d62728", ls="--", lw=1.5, label=f"VaR99 合成={-metrics['VaR99_syn']:.3f}")
ax2.set_title("左尾放大：合成样本把历史里稀疏的极端亏损补全", fontsize=12, fontweight="bold")
ax2.set_xlabel("累计收益（只显示亏损一侧）"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_generated_vs_real.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3：训练损失 + 压力测试 VaR 对比 ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
ax1.plot(np.arange(0, 2500, 200), losses, marker="o", color="#2ca02c", lw=2)
ax1.set_title(f"DDPM 训练损失（末值 {metrics['loss_last']}）", fontsize=12, fontweight="bold")
ax1.set_xlabel("训练步数"); ax1.set_ylabel("MSE(预测噪声)"); ax1.grid(True, alpha=0.3)
labels = ["VaR95", "VaR99"]
x = np.arange(2); w = 0.35
ax2.bar(x - w/2, [metrics["VaR95_real"], metrics["VaR99_real"]], w, label="真实(历史)", color="#1f77b4")
ax2.bar(x + w/2, [metrics["VaR95_syn"], metrics["VaR99_syn"]], w, label="合成(DDPM)", color="#d62728")
ax2.set_xticks(x); ax2.set_xticklabels(labels)
ax2.set_title("压力测试 VaR 对比（30 日累计，单位：收益）", fontsize=12, fontweight="bold")
ax2.set_ylabel("VaR（亏损下分位）"); ax2.legend(fontsize=9); ax2.grid(True, alpha=0.3, axis="y")
for i, v in enumerate([metrics["VaR95_real"], metrics["VaR99_real"]]):
    ax2.text(i - w/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
for i, v in enumerate([metrics["VaR95_syn"], metrics["VaR99_syn"]]):
    ax2.text(i + w/2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(D, "fig_stress_var.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE diffusion images")
