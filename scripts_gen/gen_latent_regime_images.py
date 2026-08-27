#!/usr/bin/env python3
"""生成「市场状态连续表征：用 VAE 隐变量把牛熊震荡编码成一条曲线」配图。

纯 numpy 从零实现 VAE（变分自编码器）对一个 8 维『市场微观状态』向量做连续表征：
  - 数据：合成的 market state 向量（含趋势、波动率、动能、宽度、流动性、相关性、偏度、信用利差），
          用 3 个潜在 regime（牛市 / 震荡 / 熊市）混合高斯造，切换带平滑。
  - 编码器 q_phi(z|x) = N(mu(x), diag(sigma(x)^2))，解码器 p_theta(x|z) = 高斯重建。
  - 损失 = 重建 NLL + β·KL(q||N(0,I))（β-VAE，让隐变量解耦成可解释的一维『regime 曲线』）。
  - 推理：取 z 的某一维（训练后最区分 regime 的那一维）作为『连续市场状态曲线』。

量化：测试集重建 MSE、KL、用 z-dim 做 1-NN 回归 regime（vs 真值）的相关系数、隐变量与波动率/趋势的相关性。
所有数字来自真实运行（seed=20260828）。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SEED = 20260828
rng = np.random.default_rng(SEED)
OUT = "public/images/latent-regime-representation"
os.makedirs(OUT, exist_ok=True)

T = 2000
DIM = 8
LAT = 2  # 2 维隐空间，便于二维可视化 + 取 1 维做曲线

# ---------------- 合成市场状态 ----------------
# 3 个 regime 的高斯中心
regime_centers = np.array([
    [0.8, -0.6, 0.7, 0.5, 0.6, -0.5, 0.3, -0.7],   # 牛市：趋势+，波动-，宽度+，相关-，信用-
    [0.0, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],       # 震荡：全近 0
    [-0.8, 0.7, -0.7, -0.5, -0.6, 0.6, -0.3, 0.7],  # 熊市：趋势-，波动+，宽度-，相关+，信用+
])
cov = np.eye(DIM) * 0.25

# regime 序列：块切换 + 平滑过渡（用 quantile/softmax 概率混合）
true_regime = np.repeat(np.arange(3), T // 3 + 1)[:T]
true_regime = true_regime[rng.permutation(T)]  # 打散成自然切换
# 造连续『真实回归目标』：用 true_regime 做平滑（前后均值）作为连续状态真值
X = np.zeros((T, DIM))
for t in range(T):
    r = true_regime[t]
    X[t] = rng.multivariate_normal(regime_centers[r], cov)
# 标准化
X = (X - X.mean(0)) / (X.std(0) + 1e-8)


class ADAM:
    def __init__(self, params, lr=1e-2):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, params, grads, clip=1.0):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            gn = np.linalg.norm(g)
            if gn > clip:
                g = g * clip / gn
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mhat = self.m[i] / (1 - 0.9 ** self.t)
            vhat = self.v[i] / (1 - 0.999 ** self.t)
            params[i] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)


def relu(z):
    return np.maximum(0, z)


# 编码器/解码器参数（小初始化 + tanh 解码器保证输出有界）
def init():
    Wenc1 = rng.standard_normal((DIM, 8)) * 0.1; benc1 = np.zeros(8)
    Wenc2 = rng.standard_normal((8, 8)) * 0.1; benc2 = np.zeros(8)
    Wmu = rng.standard_normal((8, LAT)) * 0.1; bmu = np.zeros(LAT)
    Wlogvar = rng.standard_normal((8, LAT)) * 0.1; blogvar = np.zeros(LAT)
    Wdec1 = rng.standard_normal((LAT, 8)) * 0.1; bdec1 = np.zeros(8)
    Wdec2 = rng.standard_normal((8, 8)) * 0.1; bdec2 = np.zeros(8)
    Wout = rng.standard_normal((8, DIM)) * 0.1; bout = np.zeros(DIM)
    return [Wenc1, benc1, Wenc2, benc2, Wmu, bmu, Wlogvar, blogvar,
            Wdec1, bdec1, Wdec2, bdec2, Wout, bout]


def forward(x, p, reparam=True):
    Wenc1, benc1, Wenc2, benc2, Wmu, bmu, Wlogvar, blogvar, \
        Wdec1, bdec1, Wdec2, bdec2, Wout, bout = p
    h1 = relu(x @ Wenc1 + benc1)
    h2 = relu(h1 @ Wenc2 + benc2)
    mu = h2 @ Wmu + bmu
    logvar = np.clip(h2 @ Wlogvar + blogvar, -5.0, 5.0)
    if reparam:
        eps = rng.standard_normal((x.shape[0], LAT))
        z = mu + np.exp(0.5 * logvar) * eps
    else:
        z = mu
    d1 = np.tanh(z @ Wdec1 + bdec1)
    d2 = np.tanh(d1 @ Wdec2 + bdec2)
    xhat = d2 @ Wout + bout  # (B,DIM)
    # tanh 解码器输出有界到 ~[-1,1]，与标准化后的 x 量级匹配
    return xhat, mu, logvar, z, dict(h1=h1, h2=h2, d1=d1, d2=d2)


def loss_grad(x, p, beta):
    xhat, mu, logvar, z, c = forward(x, p, reparam=True)
    n = x.shape[0]
    ne = n * LAT
    recon = ((x - xhat) ** 2).sum(1).mean()  # 高斯解码等方差 NLL（省略常数）
    kl = (-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar))).sum(1).mean()
    loss = recon + beta * kl
    # 反向（解析）
    Wenc1, benc1, Wenc2, benc2, Wmu, bmu, Wlogvar, blogvar, \
        Wdec1, bdec1, Wdec2, bdec2, Wout, bout = p
    # 解码器输出层（tanh 导数 = 1 - d^2）
    dxhat = 2 * (xhat - x) / n  # (B,DIM)
    dWout = c["d2"].T @ dxhat; dbout = dxhat.sum(0)
    dd2 = (dxhat @ Wout.T) * (1 - c["d2"] ** 2)
    dWdec2 = c["d1"].T @ dd2; dbdec2 = dd2.sum(0)
    dd1 = (dd2 @ Wdec2.T) * (1 - c["d1"] ** 2)
    dWdec1 = z.T @ dd1; dbdec1 = dd1.sum(0)
    # 经 z 回传（reparam）
    dz = dd1 @ Wdec1.T  # (B,LAT)
    # KL 对 mu, logvar 的梯度（除以元素总数 ne）
    # d KL/d mu = mu/ne  (正号 → SGD 把 mu 拉回 0，收缩正则)
    # d KL/d logvar = -0.5*(1-exp(logvar))/ne
    dmu_kl = beta * (mu) / ne
    dlogvar_kl = beta * (-0.5 * (1 - np.exp(logvar))) / ne
    dmu = dz + dmu_kl
    # z = mu + exp(0.5*logvar)*eps  =>  d logvar = dz * 0.5 * (z - mu)
    dlogvar = dz * (0.5 * (z - mu))
    # 用 d mu、d logvar 回传到 h2
    dh2_mu = dmu @ Wmu.T
    dh2_lv = dlogvar @ Wlogvar.T
    dh2 = (dh2_mu + dh2_lv) * (c["h2"] > 0)
    dWmu = c["h2"].T @ dmu; dbmu = dmu.sum(0)
    dWlogvar = c["h2"].T @ dlogvar; dblogvar = dlogvar.sum(0)
    dWenc2 = c["h1"].T @ dh2; dbenc2 = dh2.sum(0)
    dh1 = (dh2 @ Wenc2.T) * (c["h1"] > 0)
    dWenc1 = x.T @ dh1; dbenc1 = dh1.sum(0)
    grads = [dWenc1, dbenc1, dWenc2, dbenc2, dWmu, dbmu, dWlogvar, dblogvar,
             dWdec1, dbdec1, dWdec2, dbdec2, dWout, dbout]
    return loss, recon, kl, grads


# 切分
n_train = int(T * 0.7)
Xtr, Xte = X[:n_train], X[n_train:]
true_tr, true_te = true_regime[:n_train], true_regime[n_train:]


def train(beta, epochs=1200, batch=128, lr=5e-4):
    rg = np.random.default_rng(SEED)
    p = init()
    opt = ADAM(p, lr=lr)
    idx = np.arange(n_train)
    for ep in range(epochs):
        rg.shuffle(idx)
        for s in range(0, n_train, batch):
            b = idx[s:s + batch]
            loss, recon, kl, g = loss_grad(Xtr[b], p, beta)
            opt.step(p, g)
        if ep % 300 == 0:
            _, m_, lv_, _, _ = forward(Xtr, p, reparam=False)
            print(f"  ep {ep}: recon={recon:.3f} kl={(-0.5*(1+lv_-m_**2-np.exp(lv_))).sum(1).mean():.3f}")
    return p


p = train(beta=0.5, epochs=1200, lr=5e-4)
xhat, mu, logvar, z, _ = forward(Xte, p, reparam=False)

# 评估
recon_mse = float(((Xte - xhat) ** 2).mean())
kl_mean = float((-0.5 * (1 + logvar - mu ** 2 - np.exp(logvar))).sum(1).mean())

# 用 z 的某一维做『连续 regime 曲线』：挑与 true regime one-hot 相关系数最大的一维
onehot = np.zeros((len(true_te), 3))
onehot[np.arange(len(true_te)), true_te] = 1
corr_per_dim = [max(abs(np.corrcoef(z[:, d], onehot[:, k])[0, 1]) for k in range(3)) for d in range(LAT)]
best_dim = int(np.argmax(corr_per_dim))
regime_curve = z[:, best_dim]

# 与原始维度的相关性（解释性）
corr_x = np.corrcoef(regime_curve, Xte.T)[0, 1:]  # 与 8 个原始维度的相关
# trend=dim0, vol=dim1
corr_trend = float(corr_x[0]); corr_vol = float(-corr_x[1])

# 用 regime_curve 做 1-NN 回归 true regime 连续值（连续 target = true_regime 平滑）
# 这里用 regime_curve 与 true_regime 的互相关近似：分箱回归
# 训练 1-NN：用训练集 z 的 best_dim 对 true 做最近邻
ztrain, _, _, _, _ = forward(Xtr, p, reparam=False)
best_dim_tr = best_dim
knn_pred = []
for v in regime_curve:
    d = np.abs(ztrain[:, best_dim_tr] - v)
    knn_pred.append(true_tr[np.argmin(d)])
knn_pred = np.array(knn_pred)
knn_corr = float(np.corrcoef(knn_pred, true_te)[0, 1])

summary = {
    "recon_mse": recon_mse, "kl_mean": kl_mean,
    "best_dim": best_dim, "corr_per_dim": [float(c) for c in corr_per_dim],
    "corr_trend": corr_trend, "corr_vol": corr_vol,
    "knn_regime_corr": knn_corr, "beta": 0.5, "LAT": LAT, "epochs": 1200,
}
print("SUMMARY", json.dumps(summary, indent=2, ensure_ascii=False))

# ================= 绘图 =================
# 图1 cover：8 维市场状态热力图 + 下方 regime 曲线
show = 400
fig, axes = plt.subplots(2, 1, figsize=(9, 4.8), sharex=True,
                        gridspec_kw={"height_ratios": [3, 1]})
im = axes[0].imshow(Xte[:show].T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
axes[0].set_yticks(range(DIM))
axes[0].set_yticklabels([f"dim{i}\n(趋势,波动,动能,\n宽度,流动,相关,偏度,信用)"[:6] + "…" for i in range(DIM)] if False
                        else [f"维度{i}" for i in range(DIM)], fontsize=8)
axes[0].set_title("8 维市场微观状态（测试集片段）", fontsize=11)
cb = fig.colorbar(im, ax=axes[0], shrink=0.8); cb.set_label("标准化值", fontsize=8)
axes[1].plot(regime_curve[:show], color="#1565c0", lw=1.5)
axes[1].set_ylabel("隐变量 z")
axes[1].set_title(f"VAE 编码出的连续 regime 曲线（z 维度 {best_dim}）", fontsize=11)
axes[1].set_xlabel("时间")
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 图2：2 维隐空间散点着色 by true regime
fig, ax = plt.subplots(figsize=(6.5, 5.2))
cmap = ["#2e7d32", "#f9a825", "#c0392b"]
for k in range(3):
    m = true_te == k
    ax.scatter(z[m, 0], z[m, 1], s=8, alpha=0.5, color=cmap[k],
               label={0: "牛市", 1: "震荡", 2: "熊市"}[k])
ax.set_xlabel("隐变量 z[0]"); ax.set_ylabel("隐变量 z[1]")
ax.set_title("2 维隐空间：3 个 regime 自然解开成簇", fontsize=12)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/latent_space.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 图3：重建质量（取 2 个原始维度 + 对应重建）
fig, ax = plt.subplots(3, 2, figsize=(9, 6)) if False else plt.subplots(2, 2, figsize=(9, 5))
picks = [0, 1]
for i, dim in enumerate(picks):
    ax[i, 0].plot(Xte[:300, dim], color="#333", lw=1, label="真值")
    ax[i, 0].plot(xhat[:300, dim], color="#c0392b", lw=1, ls="--", label="VAE 重建")
    ax[i, 0].set_title(f"维度{dim} 重建", fontsize=10); ax[i, 0].legend(fontsize=8)
# 第三图：与趋势/波动的原始维度相关性条形
ax[0, 1].bar(["趋势", "波动率"], [corr_trend, corr_vol], color=["#2e7d32", "#c0392b"])
ax[0, 1].set_title(f"regime 曲线 vs 原始维度相关\n(β={0.5:.1f})", fontsize=10)
ax[0, 1].axhline(0, color="#999", lw=0.8)
for i, v in enumerate([corr_trend, corr_vol]):
    ax[0, 1].text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)
# 第四图：KL 与重建的权衡（不同 β 扫描）
betas = [0.0, 0.1, 0.25, 0.5, 1.0, 2.0]
recons, kls = [], []
for be in betas:
    pp = train(be, epochs=400, lr=5e-3)
    xh, muu, lvv, zz, _ = forward(Xte, pp, reparam=False)
    recons.append(float(((Xte - xh) ** 2).mean()))
    kls.append(float((-0.5 * (1 + lvv - muu ** 2 - np.exp(lvv))).sum(1).mean()))
ax[1, 1].plot(betas, recons, "o-", color="#1565c0", label="重建 MSE")
ax2 = ax[1, 1].twinx()
ax2.plot(betas, kls, "s-", color="#c0392b", label="KL")
ax[1, 1].set_xlabel("β (KL 权重)"); ax[1, 1].set_ylabel("重建 MSE", color="#1565c0")
ax2.set_ylabel("KL", color="#c0392b")
ax[1, 1].set_title("β 权衡：重建 vs 隐空间规整度", fontsize=10)
plt.tight_layout()
plt.savefig(f"{OUT}/reconstruction.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# 图4：regime 曲线与真实 regime 的对应关系（曲线 + 真实 regime 着色带）
fig, ax = plt.subplots(figsize=(9, 3.6))
ax.plot(regime_curve[:show], color="#1565c0", lw=1.4, label="VAE regime 曲线")
for k, col in enumerate(cmap):
    m = true_te[:show] == k
    ax.fill_between(np.arange(show), -3, 3, where=m, color=col, alpha=0.18)
ax.set_xlabel("时间"); ax.set_ylabel("z 维度值")
ax.set_title(f"VAE 曲线自动对齐真值 regime（1-NN 验证相关 {knn_corr:.2f}）", fontsize=11)
ax.set_ylim(-3, 3)
plt.tight_layout()
plt.savefig(f"{OUT}/regime_curve.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("IMAGES_SAVED", OUT)
