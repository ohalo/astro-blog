#!/usr/bin/env python3
"""
为文章「变分自编码器 VAE 因子挖掘：把不确定性也一起学出来」(vae-factor-mining) 生成真实配图。

核心逻辑：
  VAE: 编码器 q_phi(z|x) = N(mu(x), diag(sigma^2(x))); 瓶颈 z ~ q_phi
       解码器 p_theta(x|z) 重建 x
  损失 ELBO = E_q[log p(x|z)] - KL(q_phi(z|x) || N(0,I))
  用法：把 20+ 维原始量价特征压成 5 维潜变量 z 作为新因子；
        相比普通 AE 的额外红利是「每个样本有一个编码分布」——可以做不确定性感知。
  全部为合成但贴合真实结构的数值（少数潜因子驱动收益 + 大量噪声特征），非占位图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in ["/Library/Fonts/Arial Unicode.ttf",
           "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try:
        fm.fontManager.addfont(_f)
    except Exception:
        pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "vae-factor-mining")
os.makedirs(D, exist_ok=True)
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
    "axes.unicode_minus": False, "font.family": "Arial Unicode MS",
})

rng = np.random.default_rng(20260722)

# ---------- 1) 构造数据：3 个潜因子驱动收益，原始特征是它们的混合 + 噪声 ----------
N, D_feat = 800, 24
K = 3
F = rng.normal(0, 1, (N, K))                 # 真实潜因子
# 真实收益：线性依赖潜因子 + 噪声（VAE 要间接学会的"信号"）
ret_true = 0.5 * F[:, 0] - 0.35 * F[:, 1] + 0.25 * F[:, 2] + rng.normal(0, 0.4, N)

W_mix = rng.normal(0, 1, (D_feat, K))
X_clean = F @ W_mix.T                                  # (N, D)
X = X_clean + rng.normal(0, 1.2, (N, D_feat))          # 加噪声
X[:, 8:] = rng.normal(0, 1.0, (N, D_feat - 8))         # 后 16 维为纯噪声
mu, sd = X.mean(0), X.std(0) + 1e-8
Xs = (X - mu) / sd


# ---------- 2) 纯 numpy VAE（重参数化技巧）----------
# 编码器: 两层 -> (mu, logvar)；瓶颈 z = mu + eps * exp(0.5*logvar)
# 解码器: 两层 -> 重建 x_hat (线性输出)
def init_params(d, enc_h, z_dim, dec_h):
    return {
        "We1": rng.normal(0, 0.3, (enc_h, d)), "be1": np.zeros(enc_h),
        "We2m": rng.normal(0, 0.3, (z_dim, enc_h)), "be2m": np.zeros(z_dim),
        "We2l": rng.normal(0, 0.3, (z_dim, enc_h)), "be2l": np.zeros(z_dim),
        "Wd1": rng.normal(0, 0.3, (dec_h, z_dim)), "bd1": np.zeros(dec_h),
        "Wd2": rng.normal(0, 0.3, (d, dec_h)), "bd2": np.zeros(d),
    }


def forward(p, Xb):
    e1 = np.tanh(p["We1"] @ Xb.T + p["be1"][:, None])              # (enc_h, b)
    mu = p["We2m"] @ e1 + p["be2m"][:, None]                      # (z, b)
    logvar = p["We2l"] @ e1 + p["be2l"][:, None]                  # (z, b)
    eps = rng.standard_normal(mu.shape)
    z = mu + eps * np.exp(0.5 * logvar)                           # 重参数化
    d1 = np.tanh(p["Wd1"] @ z + p["bd1"][:, None])
    out = p["Wd2"] @ d1 + p["bd2"][:, None]                       # 重建 (d, b)
    return e1, mu, logvar, z, out


def loss_and_backward(p, Xb):
    e1, mu, logvar, z, out = forward(p, Xb)
    b = Xb.shape[0]
    # 重建损失 (MSE) + KL
    recon = np.mean((out - Xb.T) ** 2)
    kl = np.mean(np.sum(-0.5 * (1 + logvar - mu**2 - np.exp(logvar)), axis=0))
    elbo = recon + kl
    # 反向（对 ELBO 近似梯度，仅用于驱动优化）
    # 解码器第一层激活 d1 需从 z 重算
    d1 = np.tanh(p["Wd1"] @ z + p["bd1"][:, None])              # (dec_h, b)
    d_out = 2 * (out - Xb.T) / b                                  # (d,b)
    gd2 = d_out @ d1.T; gbd2 = d_out.sum(1)
    dd1 = (p["Wd2"].T @ d_out) * (1 - d1**2)
    gd1 = dd1 @ z.T; gbd1 = dd1.sum(1)
    # z 处梯度来自解码 + KL
    dz = p["Wd1"].T @ dd1 + mu / b                                # KL 项对 z 均值部分的近似
    ge2m = dz @ e1.T; gbe2m = dz.sum(1)
    # logvar 梯度: KL 项 -0.5*(1 - exp(logvar)) -> 对 logvar 偏导 = 0.5*(exp(logvar) - 1)
    dlv = 0.5 * (np.exp(logvar) - 1.0)
    ge2l = dlv @ e1.T; gbe2l = dlv.sum(1)
    de1 = (p["We2m"].T @ dz + p["We2l"].T @ dlv) * (1 - e1**2)
    ge1 = de1 @ Xb; gbe1 = de1.sum(1)
    grads = {"Wd2": gd2, "bd2": gbd2, "Wd1": gd1, "bd1": gbd1,
             "We2m": ge2m, "be2m": gbe2m, "We2l": ge2l, "be2l": gbe2l,
             "We1": ge1, "be1": gbe1}
    return elbo, recon, kl, grads


p = init_params(D_feat, 16, 5, 16)
lr, epochs, bs = 0.015, 500, 128
elbo_hist, recon_hist, kl_hist = [], [], []
for ep in range(epochs):
    idx = rng.permutation(N)
    for s in range(0, N, bs):
        Xb = Xs[idx[s:s + bs]]
        elbo, recon, kl, g = loss_and_backward(p, Xb)
        for k in g:
            p[k] -= lr * g[k]
    _, recon0, kl0, _ = loss_and_backward(p, Xs)
    elbo_hist.append(recon0 + kl0)
    recon_hist.append(recon0)
    kl_hist.append(kl0)

# 推断瓶颈编码的均值 mu（确定性编码）与 logvar（不确定性）
e1, MU, LOGVAR, _, _ = forward(p, Xs)
Z_mu = MU.T                                              # (N, z)
Z_var = np.exp(LOGVAR.T)                                # (N, z) 各维方差
final_elbo = elbo_hist[-1]
print(f"[训练] 最终 ELBO={final_elbo:.4f}  重建={recon_hist[-1]:.4f}  KL={kl_hist[-1]:.4f}")
print(f"        KL 占比 = {kl_hist[-1]/final_elbo*100:.1f}% (>0 说明编码器确实在学分布, 非退化退化到单点)")


# ---------- 3) 配图 ----------
# 图1(cover): 训练 ELBO 分解为重建 + KL 两条曲线
fig, ax = plt.subplots(figsize=(10, 5.6))
ax.plot(range(epochs), recon_hist, color="#4c72b0", lw=1.8, label="重建损失")
ax.plot(range(epochs), kl_hist, color="#c44e52", lw=1.8, label="KL 散度")
ax.plot(range(epochs), elbo_hist, color="#333333", lw=1.4, ls="--", label="ELBO (总)")
ax.set_xlabel("训练轮次 (epoch)", fontsize=11)
ax.set_ylabel("损失", fontsize=11)
ax.set_title(f"VAE 训练：重建与 KL 两条损失同步收敛 (ELBO 终值 {final_elbo:.3f})",
             fontsize=12.5, fontweight="bold")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vae_elbo.png"), dpi=150)
plt.close()

# 图2: 编码器输出——瓶颈 5 维的 (mu) 散点 + 每点不确定性(误差棒颜色)
# 取两个强因子维度做 2D 投影
fig, ax = plt.subplots(figsize=(8.6, 7))
unc = Z_var.sum(1)                                       # 每样本总不确定性
sc = ax.scatter(Z_mu[:, 0], Z_mu[:, 1], s=18, c=unc, cmap="plasma",
               alpha=0.75, edgecolor="none")
cb = plt.colorbar(sc, ax=ax)
cb.set_label("编码不确定性 (Σ logvar 反向)", fontsize=10)
ax.set_xlabel("瓶颈维度 1 (均值 μ)", fontsize=11)
ax.set_ylabel("瓶颈维度 2 (均值 μ)", fontsize=11)
ax.set_title("VAE 学到的 5 维潜编码：颜色=样本不确定性\n同样的特征, 有的点编码很确定、有的很散",
             fontsize=12, fontweight="bold")
ax.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(D, "vae_bottleneck_unc.png"), dpi=150)
plt.close()

# 图3: 不确定性感知选股——高 IC 个股中, 按不确定性把"不可信信号"过滤掉后 IC 提升
def ic(col):
    r = ret_true - ret_true.mean()
    c = col - col.mean()
    return np.corrcoef(c, r)[0, 1]


coef, *_ = np.linalg.lstsq(Z_mu, ret_true - ret_true.mean(), rcond=None)
vae_factor = Z_mu @ coef
vae_ic = abs(ic(vae_factor))
# 按不确定性分桶: 低不确定(可信) vs 高不确定(可疑)
order = np.argsort(unc)
n = len(unc)
lo = order[:n // 2]
hi = order[n // 2:]
ic_lo = abs(np.corrcoef(vae_factor[lo], ret_true[lo])[0, 1])
ic_hi = abs(np.corrcoef(vae_factor[hi], ret_true[hi])[0, 1])
print(f"[评估] VAE 全样本|IC|={vae_ic:.3f}  低不确定子集|IC|={ic_lo:.3f}  高不确定子集|IC|={ic_hi:.3f}")
labels = ["VAE 因子\n(全样本)", "低不确定子集\n(可信信号)", "高不确定子集\n(可疑信号)"]
vals = [vae_ic, ic_lo, ic_hi]
colors = ["#4c72b0", "#55a868", "#dd8452"]
fig, ax = plt.subplots(figsize=(9.5, 6))
bars = ax.bar(labels, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.set_ylabel("与未来收益的 |秩相关 IC|", fontsize=11)
ax.set_title("编码不确定性本身不是信号过滤器 (本例高不确定子集 IC 反而略高)",
             fontsize=12, fontweight="bold")
ax.grid(True, axis="y", alpha=0.25)
ax.set_ylim(0, max(vals) * 1.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "vae_uncertainty_select.png"), dpi=150)
plt.close()

# 图0(cover): VAE 架构示意 —— 编码器输出 (mu,sigma) 分布, 采样得 z, 解码重建
fig, ax = plt.subplots(figsize=(11, 5))
rng_c = np.random.default_rng(7)
x_demo = rng_c.normal(0, 1, 600)
# 编码器输出的一个样本分布 N(mu,sigma)
mu_d, sig_d = 0.3, 0.55
xs = np.linspace(-3, 3, 300)
zdist = np.exp(-0.5 * ((xs - mu_d) / sig_d) ** 2) / (sig_d * np.sqrt(2 * np.pi))
ax.plot(xs, zdist, color="#4c72b0", lw=2.5, label="编码器输出 q(z|x)=N(μ,σ²)")
ax.axvline(mu_d, color="#333", ls="--", lw=1.2, label=f"均值 μ={mu_d:.2f}")
ax.fill_between(xs, 0, zdist, color="#4c72b0", alpha=0.15)
ax.set_xlim(-3, 3); ax.set_yticks([])
ax.set_xlabel("瓶颈编码 z 的取值", fontsize=11)
ax.set_title("变分自编码器: 每个样本不再得到一个点, 而是一个分布\nσ² 大=编码器没把握, σ² 小=编码很确定",
             fontsize=13, fontweight="bold")
ax.legend(fontsize=10, loc="upper right")
ax.grid(True, axis="x", alpha=0.2)
plt.tight_layout()
plt.savefig(os.path.join(D, "cover.png"), dpi=150)
plt.close()

print("\n✅ VAE 配图生成完成：", sorted(os.listdir(D)))
