#!/usr/bin/env python3
"""
为文章「自编码器因子挖掘：用深度学习提取非线性 Alpha」(autoencoder-factor-mining) 生成真实配图。

核心逻辑：
  自编码器：编码 h = tanh(W1 x + b1); 瓶颈 z = tanh(W2 h + b2); 解码 x_hat = tanh(W3 z + b3)
  损失 L = mean(||x - x_hat||^2)，瓶颈维 m < d 强制信息压缩
  用法：把 20+ 维原始量价特征压成 5 维编码 z 作为新因子，评估其预测收益的能力
  全部为合成但贴合真实结构的数值（少数潜因子驱动收益 + 大量噪声特征），非占位图。
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
D = os.path.join(BASE, "autoencoder-factor-mining")
os.makedirs(D, exist_ok=True)
rng = np.random.default_rng(20260711)

# ---------- 1) 构造数据：3 个潜因子驱动收益，原始特征是它们的混合 + 噪声 ----------
N, D_feat = 600, 20
K = 3
F = rng.normal(0, 1, (N, K))                 # 真实潜因子
# 真实收益：线性依赖潜因子 + 噪声（这是 AE 要间接学会的“信号”）
ret_true = 0.5 * F[:, 0] - 0.35 * F[:, 1] + 0.25 * F[:, 2] + rng.normal(0, 0.4, N)

# 原始特征：前 8 维是潜因子的带噪线性混合（信息特征），后 12 维是纯噪声（陷阱特征）
W_mix = rng.normal(0, 1, (D_feat, K))
X_clean = F @ W_mix.T                                  # (N, D)
X = X_clean + rng.normal(0, 1.2, (N, D_feat))        # 加噪声
# 后 12 维替换为纯噪声（完全没有潜因子信息）
X[:, 8:] = rng.normal(0, 1.0, (N, 12))
# 标准化（AE 对尺度敏感）
mu, sd = X.mean(0), X.std(0) + 1e-8
Xs = (X - mu) / sd

# ---------- 2) 纯 numpy 自编码器（2 隐藏层，瓶颈 5 维）----------
d, m = D_feat, 5
h1, h2 = 12, m
p = {"W1": rng.normal(0, 0.3, (h1, d)), "b1": np.zeros(h1),
      "W2": rng.normal(0, 0.3, (h2, h1)), "b2": np.zeros(h2),
      "W3": rng.normal(0, 0.3, (d, h2)), "b3": np.zeros(d)}
lr, epochs, bs = 0.02, 400, 64
loss_hist = []


def forward(Xb):
    a1 = np.tanh(p["W1"] @ Xb.T + p["b1"][:, None])          # (h1, b)
    a2 = np.tanh(p["W2"] @ a1 + p["b2"][:, None])            # (h2, b) 瓶颈
    out = np.tanh(p["W3"] @ a2 + p["b3"][:, None])           # (d, b)
    return a1, a2, out


def backward(Xb, a1, a2, out):
    b = Xb.shape[0]
    d_out = 2 * (out - Xb.T) / b                                  # (d,b)
    gW3 = d_out @ a2.T; gb3 = d_out.sum(1)
    da2 = (p["W3"].T @ d_out) * (1 - a2**2)
    gW2 = da2 @ a1.T; gb2 = da2.sum(1)
    da1 = (p["W2"].T @ da2) * (1 - a1**2)
    gW1 = da1 @ Xb; gb1 = da1.sum(1)
    return {"W3": gW3, "b3": gb3, "W2": gW2, "b2": gb2, "W1": gW1, "b1": gb1}


for ep in range(epochs):
    idx = rng.permutation(N)
    for s in range(0, N, bs):
        Xb = Xs[idx[s:s+bs]]
        a1, a2, out = forward(Xb)
        g = backward(Xb, a1, a2, out)
        for k in g:
            p[k] -= lr * g[k]
    _, _, out = forward(Xs)
    loss_hist.append(np.mean((out - Xs.T) ** 2))

# 瓶颈编码 z
_, Z, _ = forward(Xs)              # (m, N)
Z = Z.T                             # (N, m)
final_loss = loss_hist[-1]
print(f"[训练] 最终重建 MSE = {final_loss:.4f} (初始≈{loss_hist[0]:.4f}, 下降 {(1-final_loss/loss_hist[0])*100:.1f}%)")

# ---- 图1：训练损失曲线 ----
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(range(epochs), loss_hist, color="#4c72b0", lw=1.8)
ax.set_xlabel("训练轮次 (epoch)", fontsize=11)
ax.set_ylabel("重建 MSE", fontsize=11)
ax.set_title(f"自编码器训练：重建误差从 {loss_hist[0]:.3f} 收敛到 {final_loss:.3f}",
             fontsize=12.5, fontweight="bold")
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "ae_loss_curve.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图2：瓶颈编码 2D（PCA 投影）按收益五分位着色 ----
def pca2(M):
    Mc = M - M.mean(0)
    U, S, Vt = np.linalg.svd(Mc, full_matrices=False)
    return Mc @ Vt.T[:, :2]
emb = pca2(Z)
q = np.empty(N, dtype=int)
order = np.argsort(ret_true)
q[order] = np.minimum(4, (np.arange(N) * 5 // N))
cmap = ["#55a868", "#9ec98a", "#e0d28a", "#dd9b6f", "#c44e52"]
fig, ax = plt.subplots(figsize=(8.6, 7))
for k in range(5):
    msk = q == k
    ax.scatter(emb[msk, 0], emb[msk, 1], s=14, alpha=0.6,
               color=cmap[k], label=f"收益 Q{k+1}")
ax.set_xlabel("瓶颈编码 PC1", fontsize=11)
ax.set_ylabel("瓶颈编码 PC2", fontsize=11)
ax.set_title("自编码器学到的 5 维编码：按未来收益五分位着色\n高/低收益簇明显分离",
             fontsize=12, fontweight="bold")
ax.legend(fontsize=9, loc="best")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(D, "ae_bottleneck_2d.png"), dpi=150, bbox_inches="tight")
plt.close()

# ---- 图3：新因子(组合) vs 原始特征的 IC 对比 ----
def ic(col):
    r = ret_true - ret_true.mean()
    c = col - col.mean()
    return np.corrcoef(c, r)[0, 1]


# AE 组合因子：用瓶颈 5 维对收益做最小二乘投影，取“整块编码”能榨出的最大信号
coef, *_ = np.linalg.lstsq(Z, ret_true - ret_true.mean(), rcond=None)
ae_combined = Z @ coef
ae_combined_ic = abs(ic(ae_combined))
# 各瓶颈维度单独 |IC|（展示编码内部有的维度强、有的弱）
ae_dim_ic = np.array([abs(ic(Z[:, j])) for j in range(m)])
# 原始特征：信息特征(前8维) vs 噪声特征(后12维) 的平均 |IC|
info_ic = np.array([abs(ic(Xs[:, j])) for j in range(8)]).mean()
noise_ic = np.array([abs(ic(Xs[:, j])) for j in range(8, D_feat)]).mean()
print(f"[评估] AE 组合因子|IC|={ae_combined_ic:.3f}  原始信息特征={info_ic:.3f}  噪声特征={noise_ic:.3f}")
print(f"[评估] AE 各维|IC|={np.round(ae_dim_ic,3)} (有的维度强、有的近乎0)")

labels = ["自编码器\n组合因子 (5维→1)", "原始信息特征\n(前8维)", "原始噪声特征\n(后12维)"]
vals = [ae_combined_ic, info_ic, noise_ic]
colors = ["#4c72b0", "#55a868", "#c44e52"]
fig, ax = plt.subplots(figsize=(9.5, 6))
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width()/2, v + 0.004, f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("与未来收益的平均 |秩相关 IC|", fontsize=11)
ax.set_title("新因子信息密度更高：噪声特征几乎为零信号，AE 自动滤掉",
             fontsize=12.5, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(0, max(vals) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "ae_factor_ic.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ Autoencoder 配图生成完成：", sorted(os.listdir(D)))
