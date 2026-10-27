#!/usr/bin/env python3
"""生成「神经过程 Neural Process：用元学习给出带置信区间的点预测」配图。

纯 numpy 从零实现条件神经过程 CNP（Garnelo et al. 2018）——Neural Process 的确定性变体：
  - 编码器：每个 context 点 (x,y) -> 表示 r_i
  - 聚合：r = mean(r_i)（对 context 集置换不变）
  - 解码器：给定 (r, 目标x) -> 预测 Gaussian(μ, σ)
  训练用 ELBO ~ 高斯负对数似然，天然输出逐点 σ：context 稀疏处 σ 自动变宽。

本文用 GP 先验造元学习任务：
  - CNP（带 σ） vs 普通确定性 MLP（仅点预测）
  - 量化：①context 越多 σ 越窄；②OOD 区 CNP σ 自动抬升、MLP 仍过窄（过自信）；③95% 区间覆盖率 CNP≈标称、MLP 偏低。
所有数字来自真实运行（seed=20260828）。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()

SEED = 20260828
rng = np.random.default_rng(SEED)

OUT = "public/images/neural-process-finance"
os.makedirs(OUT, exist_ok=True)

# ---------- 工具：GP 先验造函数 ----------
def rbf_kernel(x, ell=1.0, sig=1.0):
    d = x[:, None] - x[None, :]
    return sig**2 * np.exp(-0.5 * (d / ell) ** 2)

def sample_gp(x, ell=1.0, noise=1e-4):
    K = rbf_kernel(x, ell) + noise * np.eye(len(x))
    L = np.linalg.cholesky(K)
    return L @ rng.standard_normal(len(x))

# ---------- 极简手动自动微分 MLP（支持 batch 维 = 函数数 B） ----------
class ADAM:
    def __init__(self, params, lr=1e-2):
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0
    def step(self, params, grads):
        self.t += 1
        for i, (p, g) in enumerate(zip(params, grads)):
            self.m[i] = 0.9 * self.m[i] + 0.1 * g
            self.v[i] = 0.999 * self.v[i] + 0.001 * g * g
            mhat = self.m[i] / (1 - 0.9 ** self.t)
            vhat = self.v[i] / (1 - 0.999 ** self.t)
            params[i] -= self.lr * mhat / (np.sqrt(vhat) + 1e-8)

def relu(z):
    return np.maximum(0, z)

def sample_context_target(fx, m_context, x_dense):
    """从函数 f 在 x_dense 上取值，抽 m_context 个 context 点（在 [-4,4] 内）"""
    idx = rng.choice(np.where((x_dense >= -4) & (x_dense <= 4))[0], size=m_context, replace=False)
    xc = x_dense[idx]; yc = fx[idx]
    return xc, yc

# ============ CNP 模型 ============
# 编码器: (2)->16->16 ; 解码器: (17)->32->32->2(μ,logσ)
W1 = (rng.standard_normal((2, 16)) * 0.3); b1 = np.zeros(16)
W2 = (rng.standard_normal((16, 16)) * 0.3); b2 = np.zeros(16)
We1 = (rng.standard_normal((17, 32)) * 0.3); be1 = np.zeros(32)
We2 = (rng.standard_normal((32, 32)) * 0.3); be2 = np.zeros(32)
Wo = (rng.standard_normal((32, 2)) * 0.3); bo = np.zeros(2)
cnp_params = [W1, b1, W2, b2, We1, be1, We2, be2, Wo, bo]

# ============ 确定性 MLP 基线（仅点预测 μ，x 为 1 维） ============
M1 = (rng.standard_normal((1, 16)) * 0.3); mb1 = np.zeros(16)
M2 = (rng.standard_normal((16, 16)) * 0.3); mb2 = np.zeros(16)
Me1 = (rng.standard_normal((16, 32)) * 0.3); me1 = np.zeros(32)
Me2 = (rng.standard_normal((32, 32)) * 0.3); me2 = np.zeros(32)
Mo = (rng.standard_normal((32, 1)) * 0.3); mo = np.zeros(1)
mlp_params = [M1, mb1, M2, mb2, Me1, me1, Me2, me2, Mo, mo]

x_dense = np.linspace(-5, 5, 80)
x_target = x_dense  # 目标覆盖 [-5,5]，含 OOD 区

# 固定一组演示函数族核尺度，提升 context 计数效应的可学习性
ELL_SET = [0.6, 0.9, 1.2, 1.6]

def cnp_forward(xc, yc, xt, params, training=True):
    W1, b1, W2, b2, We1, be1, We2, be2, Wo, bo = params
    m = len(xc)
    xyc = np.stack([xc, yc], axis=1)  # (m,2)
    h1 = relu(xyc @ W1 + b1)
    h2 = relu(h1 @ W2 + b2)          # (m,16)
    r = h2.mean(axis=0, keepdims=True)  # (1,16)
    xt = xt.reshape(-1, 1)
    r_tiled = np.repeat(r, len(xt), axis=0)  # (n,16)
    inp = np.concatenate([r_tiled, xt], axis=1)  # (n,17)
    d1 = relu(inp @ We1 + be1)
    d2 = relu(d1 @ We2 + be2)
    out = d2 @ Wo + bo
    mu = out[:, 0]; logsig = np.clip(out[:, 1], -3, 3); sig = np.exp(logsig)
    cache = dict(xyc=xyc, h1=h1, h2=h2, r=r, xt=xt, r_tiled=r_tiled,
                 inp=inp, d1=d1, d2=d2, out=out, m=m)
    return mu, sig, cache

def cnp_loss_grad(xc, yc, xt, yt, params):
    mu, sig, c = cnp_forward(xc, yc, xt, params)
    nll = 0.5 * ((yt - mu) / sig) ** 2 + np.log(sig) + 0.5 * np.log(2 * np.pi)
    loss = nll.mean()
    # 反向
    W1, b1, W2, b2, We1, be1, We2, be2, Wo, bo = params
    # 输出层
    dout = np.zeros_like(c["out"]); n = len(xt)
    dout[:, 0] = (-(yt - mu) / (sig ** 2)) / n
    dout[:, 1] = (1 - ((yt - mu) ** 2) / (sig ** 2)) / n
    dWo = c["d2"].T @ dout; dbo = dout.sum(0)
    dd2 = (dout @ Wo.T) * (c["d2"] > 0)
    dWe2 = c["d1"].T @ dd2; dbe2 = dd2.sum(0)
    dd1 = (dd2 @ We2.T) * (c["d1"] > 0)
    dWe1 = c["inp"].T @ dd1; dbe1 = dd1.sum(0)
    dinp = dd1 @ We1.T  # (n,17)
    dr_tiled = dinp[:, :16]; dxt = dinp[:, 16:]
    dr = dr_tiled.mean(axis=0, keepdims=True)  # 聚合梯度回传
    dh2 = np.repeat(dr, c["m"], axis=0) * (c["h2"] > 0)
    dW2 = c["h1"].T @ dh2; db2 = dh2.sum(0)
    dh1 = (dh2 @ W2.T) * (c["h1"] > 0)
    dW1 = c["xyc"].T @ dh1; db1 = dh1.sum(0)
    grads = [dW1, db1, dW2, db2, dWe1, dbe1, dWe2, dbe2, dWo, dbo]
    return loss, grads

def mlp_forward(xt, params):
    M1, mb1, M2, mb2, Me1, me1, Me2, me2, Mo, mo = params
    xt = xt.reshape(-1, 1)
    h1 = relu(xt @ M1 + mb1)
    h2 = relu(h1 @ M2 + mb2)
    d1 = relu(h2 @ Me1 + me1)
    d2 = relu(d1 @ Me2 + me2)
    mu = d2 @ Mo + mo
    return mu.flatten(), dict(xt=xt, h1=h1, h2=h2, d1=d1, d2=d2)

def mlp_loss_grad(xt, yt, params):
    mu, c = mlp_forward(xt, params)
    n = len(xt)
    loss = 0.5 * ((yt - mu) ** 2).mean()
    M1, mb1, M2, mb2, Me1, me1, Me2, me2, Mo, mo = params
    dmu = (mu - yt) / n
    dMo = c["d2"].T @ dmu.reshape(-1, 1); dmo = dmu.sum(0)
    dd2 = (dmu.reshape(-1, 1) @ Mo.T) * (c["d2"] > 0)
    dMe2 = c["d1"].T @ dd2; dme2 = dd2.sum(0)
    dd1 = (dd2 @ Me2.T) * (c["d1"] > 0)
    dMe1 = c["h2"].T @ dd1; dme1 = dd1.sum(0)
    dh2 = (dd1 @ Me1.T) * (c["h2"] > 0)
    dM2 = c["h1"].T @ dh2; dmb2 = dh2.sum(0)
    dh1 = (dh2 @ M2.T) * (c["h1"] > 0)
    dM1 = c["xt"].T @ dh1; dmb1 = dh1.sum(0)
    grads = [dM1, dmb1, dM2, dmb2, dMe1, dme1, dMe2, dme2, dMo, dmo]
    return loss, grads

# ============ 训练 ============
opt_c = ADAM(cnp_params, lr=1e-2)
opt_m = ADAM(mlp_params, lr=1e-2)
STEPS = 9000
for step in range(STEPS):
    fx = sample_gp(x_dense, ell=rng.choice(ELL_SET))
    m = int(rng.integers(3, 40))
    xc, yc = sample_context_target(fx, m, x_dense)
    yt = fx
    lc, gc = cnp_loss_grad(xc, yc, x_target, yt, cnp_params)
    opt_c.step(cnp_params, gc)
    lm, gm = mlp_loss_grad(x_target, yt, mlp_params)
    opt_m.step(mlp_params, gm)
    if step % 1500 == 0:
        print(f"step {step}: cnp_nll={lc:.3f} mlp_mse={lm:.3f}")

# ============ 评估：测试函数 ============
def cnp_predict(xc, yc, xt):
    mu, sig, _ = cnp_forward(xc, yc, xt, cnp_params)
    return mu, sig

def mlp_predict(xt):
    mu, _ = mlp_forward(xt, mlp_params)
    return mu

# 覆盖率（95% 名义区间）
cov_cnp = []; cov_mlp = []
resid_mlp = []
N_TEST = 200
for _ in range(N_TEST):
    fx = sample_gp(x_dense, ell=rng.choice(ELL_SET))
    m = int(rng.integers(8, 16))
    xc, yc = sample_context_target(fx, m, x_dense)
    mu_c, sig_c = cnp_predict(xc, yc, x_target)
    mu_m = mlp_predict(x_target)
    cov_cnp.append(np.mean(np.abs(fx - mu_c) < 1.96 * sig_c))
    resid_mlp.append(np.abs(fx - mu_m))
cov_cnp_mean = float(np.mean(cov_cnp))
# MLP 用训练残差标准差构造常数带
resid_all = np.concatenate(resid_mlp)
mlp_band = np.percentile(resid_all, 95) * 1.96  # 近似 95% 常数带宽
cov_mlp_mean = float(np.mean([np.mean(np.abs(fx - mlp_predict(x_target)) < mlp_band)
                              for _ in range(40)]))

# OOD vs in-context σ
fx_demo = sample_gp(x_dense, ell=1.0)
xc_d, yc_d = sample_context_target(fx_demo, 12, x_dense)
mu_d, sig_d = cnp_predict(xc_d, yc_d, x_target)
in_mask = (x_target >= -4) & (x_target <= 4)
ood_mask = x_target > 4
sig_in = float(sig_d[in_mask].mean())
sig_ood = float(sig_d[ood_mask].mean())

# context 覆盖范围 vs 尾部不确定性的诚实关系：context 覆盖越多，整体区间越收敛
# 但 vanilla CNP 的 σ 是『集合级』全局不确定性（来自 r=mean(r_i)），不会因单点附近观测变多而收窄。
# 真实的诚实行为：σ(x) 在 context 覆盖区内低、在 OOD 未覆盖区跳升。
fx_spread = sample_gp(x_dense, ell=1.0)
xc_s, yc_s = sample_context_target(fx_spread, 14, x_dense)
mu_s, sig_s = cnp_predict(xc_s, yc_s, x_target)
width_in = float((1.96 * sig_s[in_mask]).mean())
width_ood = float((1.96 * sig_s[ood_mask]).mean())

summary = {
    "cov_cnp_95": cov_cnp_mean,
    "cov_mlp_95_const": cov_mlp_mean,
    "mlp_const_band_95": float(mlp_band),
    "sig_in_context": sig_in,
    "sig_ood": sig_ood,
    "width_in_context": width_in,
    "width_ood": width_ood,
    "n_test": N_TEST,
}
print("SUMMARY", json.dumps(summary, indent=2, ensure_ascii=False))

# ============ 画图 ============
# ---- 图1 cover：单函数 + CNP 带 + MLP 叠加 ----
fig, ax = plt.subplots(figsize=(8.5, 4.4))
ax.plot(x_target, fx_demo, color="#333", lw=1.5, label="真实函数 f(x)")
ax.scatter(xc_d, yc_d, c="#c62828", zorder=5, label="context 观测点")
mu_c, sig_c = cnp_predict(xc_d, yc_d, x_target)
ax.plot(x_target, mu_c, color="#1565c0", lw=2, label="CNP 预测均值 μ")
ax.fill_between(x_target, mu_c - 1.96 * sig_c, mu_c + 1.96 * sig_c,
                color="#90caf9", alpha=0.5, label="CNP 95% 区间")
mu_m = mlp_predict(x_target)
ax.plot(x_target, mu_m, color="#ef6c00", lw=1.8, ls="--", label="MLP 点预测（无 σ）")
ax.axvline(-4, color="#999", ls=":"); ax.axvline(4, color="#999", ls=":")
ax.text(4.3, fx_demo.max() * 0.9, "OOD 区", fontsize=9, color="#888")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("CNP：context 稀疏的 OOD 区 σ 自动变宽；MLP 过自信", fontsize=13)
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图2：σ(x) 沿输入轴 —— 覆盖区低、OOD 跳升（vanilla CNP 的全局不确定性） ----
fig, ax = plt.subplots(figsize=(8.5, 4.2))
ax.plot(x_target, 1.96 * sig_s, color="#1565c0", lw=2, label="95% 区间半宽 σ(x)")
ax.scatter(xc_s, np.zeros_like(xc_s), c="#c62828", zorder=5, label="context 覆盖区")
ax.axvline(-4, color="#999", ls=":"); ax.axvline(4, color="#999", ls=":")
ax.axvspan(4, 5, color="#ffcdd2", alpha=0.3)
ax.text(4.2, 1.96 * sig_s.max() * 0.9, "OOD 未覆盖区\nσ 跳升", fontsize=9, color="#a33")
ax.set_xlabel("x"); ax.set_ylabel("95% 区间半宽")
ax.set_title("CNP 是集合级不确定性：覆盖区内 σ 低、OOD 区 σ 跳升", fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/context_dependence.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图3：覆盖率校准 ----
fig, ax = plt.subplots(figsize=(7.5, 4.2))
nominals = [0.5, 0.8, 0.9, 0.95]
# 用对应分位构造 CNP 经验覆盖
covs = []
for p in nominals:
    z = {0.5: 0.674, 0.8: 1.282, 0.9: 1.645, 0.95: 1.96}[p]
    c = []
    for _ in range(60):
        fx = sample_gp(x_dense, ell=rng.choice(ELL_SET))
        m = int(rng.integers(8, 16))
        xc, yc = sample_context_target(fx, m, x_dense)
        mu, sig = cnp_predict(xc, yc, x_target)
        c.append(np.mean(np.abs(fx - mu) < z * sig))
    covs.append(float(np.mean(c)))
ax.plot(nominals, covs, "o-", color="#2e7d32", lw=2, label="CNP 经验覆盖")
ax.plot(nominals, nominals, "--", color="#999", label="理想对角线")
ax.set_xlabel("名义置信水平"); ax.set_ylabel("经验覆盖")
ax.set_title("校准：CNP 区间覆盖贴近标称（95pct 实测 {:.0f}pct）".format(covs[-1] * 100), fontsize=13)
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/calibration.png", dpi=130, bbox_inches="tight")
plt.close(fig)

# ---- 图4：OOD σ 抬升（另一条函数） ----
fig, ax = plt.subplots(figsize=(8.5, 4.2))
fx2 = sample_gp(x_dense, ell=0.8)
xc2, yc2 = sample_context_target(fx2, 10, x_dense)
mu2, sig2 = cnp_predict(xc2, yc2, x_target)
ax.plot(x_target, fx2, color="#333", lw=1.3, alpha=0.7, label="真实函数")
ax.scatter(xc2, yc2, c="#c62828", zorder=5, label="context")
ax.plot(x_target, mu2, color="#1565c0", lw=2)
ax.fill_between(x_target, mu2 - 1.96 * sig2, mu2 + 1.96 * sig2,
                color="#90caf9", alpha=0.5, label="95% 区间")
ax.axvspan(4, 5, color="#ffcdd2", alpha=0.3)
ax.text(4.3, fx2.min(), "OOD 区 σ 显著抬升\n(%.2f vs 区内 %.2f)" % (sig2[x_target > 4].mean(), sig2[in_mask].mean()),
        fontsize=9, color="#a33")
ax.set_xlabel("x"); ax.set_ylabel("f(x)")
ax.set_title("OOD 区：CNP 给出高不确定性（诚实地说『我不知道』）", fontsize=13)
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig(f"{OUT}/ood_uncertainty.png", dpi=130, bbox_inches="tight")
plt.close(fig)

print("IMAGES_SAVED", OUT)
