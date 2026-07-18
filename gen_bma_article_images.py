#!/usr/bin/env python3
"""
为文章「贝叶斯模型平均：把『哪个模型对』变成权重而非赌注」
生成真实配图。所有图表均由文中方法真实计算生成。

机制（合成、仅用于演示方法；真实落地见文末路径）：
  - 真实预期收益 mu_i ~ N(0,1) 横截面
  - M=5 个候选模型，各自对 mu 做不同强度收缩 a_k（a 越小越贴近真值）
  - 训练期回报 Y = mu + 噪声；用 BIC 给每个模型算后验权重
  - BMA 预测 = Σ w_k · f_k
  - 在测试期比 MSPE：单模型最优 vs BMA；蒙特卡洛 200 次算 BMA 胜率
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "bayesian-model-averaging-asset")
os.makedirs(D, exist_ok=True)

C = {"blue": "#2F4B7C", "red": "#C44E52", "green": "#55A868",
     "purple": "#8172B3", "orange": "#FF7F0E", "grey": "#888888"}

N = 12        # 资产数
T = 120       # 训练期观测
Tt = 40       # 测试期观测
M = 5         # 候选模型数

rng = np.random.default_rng(20260719)

# 每个模型对真值的收缩强度（0=完全贴合真值，1=完全丢弃）
a = np.array([0.05, 0.20, 0.40, 0.60, 0.85])
model_names = [f"M{k+1}\n(a={a[k]:.2f})" for k in range(M)]


def build(mu, seed):
    r = np.random.default_rng(seed)
    g = np.zeros((M, N))
    for k in range(M):
        inform = 1.0 - a[k]
        g[k] = inform * mu + np.sqrt(1 - inform ** 2) * r.normal(0, 1, size=N)
    return g


def bic_weights_per_asset(F, Ytr):
    Tn, Nn = Ytr.shape
    W = np.zeros((Nn, M))
    for i in range(Nn):
        w = []
        for k in range(M):
            resid = Ytr[:, i] - F[k, i]
            ssr = np.sum(resid ** 2)
            sigma2 = ssr / Tn if Tn > 0 else 1.0
            sigma2 = max(sigma2, 1e-12)
            bic = Tn * np.log(sigma2) + np.log(Tn)   # 1 参数
            w.append(-0.5 * bic)
        w = np.array(w)
        w -= w.max()
        p = np.exp(w)
        W[i] = p / p.sum()
    return W


def run_once(seed):
    r = np.random.default_rng(seed)
    mu = r.normal(0, 1, size=N)
    F = build(mu, seed + 1)
    Ytr = mu[None, :] + r.normal(0, 1.2, size=(T, N))
    Yte = mu[None, :] + r.normal(0, 1.2, size=(Tt, N))
    W = bic_weights_per_asset(F, Ytr)
    w_global = W.mean(axis=0)
    w_global /= w_global.sum()
    pred_bma = np.einsum('ik,ki->i', W, F)   # 逐资产: Σ_k W[i,k]·F[k,i]
    pred_single = F[np.argmin([np.sum((Yte - F[k][None, :]) ** 2) for k in range(M)])]
    mspe_single = np.mean((Yte - pred_single[None, :]) ** 2)
    mspe_bma = np.mean((Yte - pred_bma) ** 2)
    return w_global, F, mu, Yte, pred_bma, pred_single, mspe_single, mspe_bma


w_glob, F, mu, Yte, pred_bma, pred_single, mspe_s, mspe_b = run_once(20260719)
print("全局后验权重:", np.round(w_glob, 3))
print(f"测试期 MSPE: 单模型最优={mspe_s:.3f}  BMA={mspe_b:.3f}  提升={1-mspe_b/mspe_s:.1%}")

# ---------- 图1：逐资产模型权重热力图 ----------
Wpa = bic_weights_per_asset(F, mu[None, :] + rng.normal(0, 1.2, size=(T, N)))
fig, ax = plt.subplots(figsize=(7.2, 5.0))
im = ax.imshow(Wpa.T, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(N)); ax.set_xticklabels([f"A{i+1}" for i in range(N)], fontsize=9)
ax.set_yticks(range(M)); ax.set_yticklabels(model_names, fontsize=9)
ax.set_xlabel("资产")
ax.set_ylabel("候选模型")
ax.set_title("逐资产后验模型权重 (BMA)：不同资产『信哪个模型』并不一致", fontsize=11)
for i in range(N):
    for k in range(M):
        ax.text(i, k, f"{Wpa[i, k]:.2f}", ha="center", va="center",
                color="white" if Wpa[i, k] > 0.5 else "black", fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="后验权重")
plt.tight_layout()
plt.savefig(os.path.join(D, "bma_weights_heatmap.png"), dpi=130)
plt.close()

# ---------- 图2：预测对比（6 个资产：真值 / 最优单模型 / BMA） ----------
sel = list(range(6))
xpos = np.arange(len(sel))
w = 0.27
fig, ax = plt.subplots(figsize=(7.6, 4.8))
ax.bar(xpos - w, mu[sel], w, label="真实 μ", color=C["grey"])
ax.bar(xpos, pred_single[sel], w, label="最优单模型", color=C["red"])
ax.bar(xpos + w, pred_bma[sel], w, label="BMA 加权", color=C["blue"])
ax.set_xticks(xpos); ax.set_xticklabels([f"A{i+1}" for i in sel])
ax.set_xlabel("资产"); ax.set_ylabel("预期收益预测")
ax.set_title("预测对比：BMA 不会盲信某一模型，离真值更近", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "bma_forecast_compare.png"), dpi=130)
plt.close()

# ---------- 图3：蒙特卡洛胜率（BMA vs 单模型最优） ----------
R = 200
ratios = []
for rep in range(R):
    _, _, _, _, _, _, ms, mb = run_once(1000 + rep)
    ratios.append(ms / mb)
ratios = np.array(ratios)
winrate = np.mean(ratios >= 1.0)
print(f"蒙特卡洛 {R} 次: BMA MSPE 不劣于最优单模型比例 = {winrate:.1%}")
fig, ax = plt.subplots(figsize=(6.4, 4.6))
ax.boxplot(ratios, vert=True, widths=0.5, patch_artist=True,
           boxprops=dict(facecolor=C["green"], alpha=0.6),
           medianprops=dict(color="black"))
ax.axhline(1.0, color=C["red"], ls="--", lw=1.2, label="1.0 (打平)")
ax.set_ylabel("MSPE(最优单模型) / MSPE(BMA)")
ax.set_title(f"200 次蒙特卡洛：BMA 在 {winrate:.0%} 情形下不劣于单模型最优", fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(D, "bma_mspe_boxplot.png"), dpi=130)
plt.close()

print("images saved to", D)
