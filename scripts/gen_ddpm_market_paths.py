#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""扩散模型(DDPM)生成市场路径：用去噪扩散生成符合历史收益分布的价格轨迹
E1 合成"历史"收益分布（高斯混合，带肥尾）作为训练靶
E2 手写 2 层 MLP 噪声预测器 + DDPM 前向/反向过程，现场训练
E3 反向采样生成新路径，与靶分布做边际/尾部/自相关对拍
E4 KS 检验 + 分位数对齐，量化"像不像历史"
输出 5 张图 + _results.json 到 public/images/{slug}/
"""
import os, json, math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]; break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["figure.facecolor"] = "white"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.3

SLUG = "diffusion-model-market-path-generation"
OUT = f"/Users/halo123456/workspace/astro-blog/public/images/{SLUG}"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"
SEED = 20261028
rng = np.random.default_rng(SEED)
R = {}

# ---------------- 靶分布：高斯混合日收益（肥尾） ----------------
L = 24            # 轨迹长度（24 个交易日收益）
N = 4000          # 训练样本数
# 85% 平静日 σ=0.8% μ=0.05%；15% 极端日 σ=3.0% μ=0.0%  -> 合成肥尾
p_calm, mu_calm, s_calm = 0.85, 0.0005, 0.008
mu_jump, s_jump = 0.0, 0.030
comp = rng.random(N * L) < p_calm
target = np.where(comp, rng.normal(mu_calm, s_calm, N * L), rng.normal(mu_jump, s_jump, N * L)).reshape(N, L)
rng2 = np.random.default_rng(SEED + 1)
target_test = rng2.normal(0, 1, 0)  # placeholder

def moments(x):
    x = np.asarray(x, float)
    m = x.mean(); s = x.std(ddof=1)
    sk = float(((x - m) ** 3).mean() / s ** 3) if s > 0 else 0.0
    ku = float(((x - m) ** 4).mean() / s ** 4 - 3.0) if s > 0 else 0.0
    return float(m), float(s), sk, ku

tm = moments(target.ravel())
R["target_mean"], R["target_std"] = round(tm[0], 6), round(tm[1], 6)
R["target_skew"], R["target_kurt"] = round(tm[2], 4), round(tm[3], 4)
# 自相关 lag1（应为 ~0）
ac1 = float(np.corrcoef(target[:, :-1].ravel(), target[:, 1:].ravel())[0, 1])
R["target_ac1"] = round(ac1, 4)
R["target_p99"] = round(float(np.percentile(target.ravel(), 99)), 5)

# 标准化到零均值单位方差（扩散模型要求数据近似单位方差，否则 q(x_T) 边际与 N(0,I) 不符）
mu_all, sig_all = float(target.mean()), float(target.std())
target_z = (target - mu_all) / sig_all

# ---------------- 手写 2 层 MLP（numpy 反向传播） ----------------
H1 = H2 = 160
IN = L + 1   # x_t(L 维) + 时间步 t(1 维)
W1 = rng.normal(0, math.sqrt(2.0 / IN), (H1, IN))
b1 = np.zeros(H1)
W2 = rng.normal(0, math.sqrt(2.0 / H2), (H2, H1))
b2 = np.zeros(H2)
W3 = rng.normal(0, math.sqrt(2.0 / L), (L, H2))
b3 = np.zeros(L)

def mlp_forward(x):
    z1 = W1 @ x + b1; a1 = np.maximum(z1, 0)
    z2 = W2 @ a1 + b2; a2 = np.maximum(z2, 0)
    out = W3 @ a2 + b3
    return a1, a2, out

def predict(X):  # X: (L+1, B)
    z1 = W1 @ X + b1[:, None]; a1 = np.maximum(z1, 0)
    z2 = W2 @ a1 + b2[:, None]; a2 = np.maximum(z2, 0)
    return W3 @ a2 + b3[:, None]

# ---------------- DDPM 调度 ----------------
T = 50
beta = np.linspace(1e-4, 0.025, T)
alpha = 1.0 - beta
alphabar = np.cumprod(alpha)
sqrt_ab = np.sqrt(alphabar)
sqrt_1mab = np.sqrt(1.0 - alphabar)

# ---------------- 训练 ----------------
B = 256
epochs = 3000
lr = 0.003
losses = []
for ep in range(epochs):
    perm = rng.permutation(N)
    ep_loss = 0.0
    nb = 0
    for i in range(0, N, B):
        idx = perm[i:i + B]
        x0 = target_z[idx].T                     # (L, B) 标准化后
        t = rng.integers(1, T + 1, size=len(idx)) # (B,)
        eps = rng.normal(0, 1, (L, len(idx)))
        ab = alphabar[t - 1]
        xt = np.sqrt(ab)[None, :] * x0 + np.sqrt(1 - ab)[None, :] * eps
        tt = (t / T).astype(float)[None, :]       # (1, B)
        X = np.vstack([xt, tt])                   # (L+1, B)
        eps_hat = predict(X)
        err = eps_hat - eps
        loss = (err ** 2).mean()
        ep_loss += float(loss) * len(idx); nb += len(idx)
        # 反向传播（对批量均值）：重算前向以便取隐藏层激活
        dout = err / len(idx)
        z1 = W1 @ X + b1[:, None]; a1 = np.maximum(z1, 0)
        z2 = W2 @ a1 + b2[:, None]; a2 = np.maximum(z2, 0)
        dout2 = dout
        dW3 = dout2 @ a2.T
        db3 = dout2.sum(1)
        da2 = (W3.T @ dout2) * (a2 > 0)
        dW2 = da2 @ a1.T
        db2 = da2.sum(1)
        da1 = (W2.T @ da2) * (a1 > 0)
        dW1 = da1 @ X.T
        db1 = da1.sum(1)
        W3 -= lr * dW3; b3 -= lr * db3
        W2 -= lr * dW2; b2 -= lr * db2
        W1 -= lr * dW1; b1 -= lr * db1
    losses.append(ep_loss / nb)
R["loss_start"], R["loss_end"] = round(float(losses[0]), 6), round(float(losses[-1]), 6)
R["loss_reduction"] = round((losses[0] - losses[-1]) / losses[0] * 100, 1)
R["epochs"] = epochs

# ---------------- 反向采样 ----------------
def sample(n=200, with_path=False):
    x = rng.normal(0, 1, (L, n))
    traj = [x.copy()] if with_path else None
    for t in range(T, 0, -1):
        ab = alphabar[t - 1]
        at = alpha[t - 1]
        s1 = np.sqrt(ab); s2 = np.sqrt(1 - ab)
        tt = np.full((1, n), t / T)
        X = np.vstack([x, tt])
        eps_hat = predict(X)
        coeff = (1 - at) / s2
        x = (x - coeff * eps_hat) / math.sqrt(at)
        if t > 1:
            x = x + np.sqrt(beta[t - 1]) * rng.normal(0, 1, (L, n))
        if with_path and t % 8 == 0:
            traj.append(x.copy())
    return x, traj

gen_z, _ = sample(N, with_path=False)
gen = gen_z * sig_all + mu_all          # 反标准化回原始收益尺度
gm = moments(gen.ravel())
R["gen_mean"], R["gen_std"] = round(gm[0], 6), round(gm[1], 6)
R["gen_skew"], R["gen_kurt"] = round(gm[2], 4), round(gm[3], 4)
gac1 = float(np.corrcoef(gen[:, :-1].ravel(), gen[:, 1:].ravel())[0, 1])
R["gen_ac1"] = round(gac1, 4)
R["gen_p99"] = round(float(np.percentile(gen.ravel(), 99)), 5)
# KS 检验
ks = stats.ks_2samp(target.ravel(), gen.ravel())
R["ks_stat"], R["ks_p"] = round(float(ks.statistic), 4), round(float(ks.pvalue), 4)
# 分位数对齐（5,50,95,99）
qs = [1, 5, 50, 95, 99]
R["quantiles"] = {f"q{q}": [round(float(np.percentile(target.ravel(), q)), 5),
                             round(float(np.percentile(gen.ravel(), q)), 5)] for q in qs}
# 相对误差（std / kurt / p99）
R["err_std"] = round(abs(R["gen_std"] - R["target_std"]) / R["target_std"] * 100, 1)
R["err_kurt"] = round(abs(R["gen_kurt"] - R["target_kurt"]) / abs(R["target_kurt"]) * 100, 1)
R["err_p99"] = round(abs(R["gen_p99"] - R["target_p99"]) / abs(R["target_p99"]) * 100, 1)

# ---------------- 图 1：训练损失 ----------------
fig, ax = plt.subplots(figsize=(12, 4.6))
xs = range(0, epochs, max(1, epochs // 100))
ax.plot(list(xs), [losses[i] for i in xs], color=BLUE, lw=1.4)
ax.set_title(f"DDPM 训练：噪声预测 MSE {R['loss_start']:.4f} → {R['loss_end']:.4f}（降 {R['loss_reduction']}%，{epochs} epoch，手写 2 层 MLP）")
ax.set_xlabel("epoch"); ax.set_ylabel("批量 MSE")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_training_loss.png"); plt.close(fig)

# ---------------- 图 2：前向扩散（加噪过程，标准化尺度） ----------------
demo = target_z[0].copy()
fig, ax = plt.subplots(figsize=(12, 4.6))
ax.plot(np.cumsum(demo), color=BLACK, lw=1.0, label="x₀ 原始轨迹（标准化后累计）")
for frac, c in [(0.25, GREEN), (0.5, BLUE), (0.75, ORANGE), (1.0, RED)]:
    ti = int(frac * T) - 1
    ab = alphabar[ti]
    xt = math.sqrt(ab) * demo + math.sqrt(1 - ab) * rng.normal(0, 1, L)
    ax.plot(np.cumsum(xt), color=c, lw=0.9, alpha=0.85, label=f"t={ti+1} (SNR={ab:.2f})")
ax.set_title("前向扩散：同一轨迹从清晰（x₀）逐步被高斯噪声淹没到 x_T≈纯噪声（标准化尺度）")
ax.set_xlabel("交易日"); ax.set_ylabel("累计（标准化）"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2_forward_diffusion.png"); plt.close(fig)

# ---------------- 图 3：反向采样生成轨迹 ----------------
g3, p3 = sample(3, with_path=True)
g3 = g3 * sig_all + mu_all       # 反标准化到原始收益尺度
fig, ax = plt.subplots(figsize=(12, 4.6))
for k in range(3):
    ax.plot(np.cumsum(g3[:, k]), color=[GREEN, BLUE, ORANGE][k], lw=1.0, label=f"生成路径 {k+1}")
ax.axhline(0, color=GRAY, lw=0.8, ls="--")
ax.set_title(f"反向去噪采样：从纯噪声 x_T 逐步生成 3 条新价格轨迹（每条即一条『合成历史』）")
ax.set_xlabel("交易日"); ax.set_ylabel("累计收益"); ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig3_reverse_samples.png"); plt.close(fig)

# ---------------- 图 4：边际分布对拍 ----------------
fig, ax = plt.subplots(figsize=(12, 4.6))
bins = np.linspace(-0.10, 0.10, 60)
ax.hist(target.ravel(), bins=bins, density=True, alpha=0.45, color=BLUE, label=f"历史靶（σ={R['target_std']*100:.2f}%, 超额峰度={R['target_kurt']:.1f}）")
ax.hist(gen.ravel(), bins=bins, density=True, histtype="step", lw=2.0, color=RED, label=f"DDPM 生成（σ={R['gen_std']*100:.2f}%, 超额峰度={R['gen_kurt']:.1f}）")
ax.set_title(f"边际收益分布对拍：DDPM 还原肥尾（KS D={R['ks_stat']}, p={R['ks_p']}）")
ax.set_xlabel("日收益"); ax.set_ylabel("密度"); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig(f"{OUT}/fig4_marginal_dist.png"); plt.close(fig)

# ---------------- 图 5：统计量 + 分位数对齐 ----------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
metrics = ["均值", "标准差", "偏度", "超额峰度", "lag1自相关", "99%分位"]
tv = [R["target_mean"]*1e4, R["target_std"]*1e4, R["target_skew"], R["target_kurt"], R["target_ac1"], R["target_p99"]*1e4]
gv = [R["gen_mean"]*1e4, R["gen_std"]*1e4, R["gen_skew"], R["gen_kurt"], R["gen_ac1"], R["gen_p99"]*1e4]
x = np.arange(len(metrics)); w = 0.38
ax[0].bar(x - w/2, tv, w, color=BLUE, alpha=0.8, label="历史靶")
ax[0].bar(x + w/2, gv, w, color=RED, alpha=0.6, edgecolor=BLACK, label="DDPM 生成")
ax[0].set_xticks(x); ax[0].set_xticklabels(metrics, rotation=30, fontsize=8)
ax[0].set_title("六维统计量对拍（数值×1e4，除偏度/峰度/自相关）"); ax[0].legend(fontsize=8)
qt = [f"q{q}" for q in qs]
qtv = [R["quantiles"][k][0]*1e4 for k in qt]; qgv = [R["quantiles"][k][1]*1e4 for k in qt]
xq = np.arange(len(qt)); wq = 0.38
ax[1].bar(xq - wq/2, qtv, wq, color=BLUE, alpha=0.8, label="历史靶")
ax[1].bar(xq + wq/2, qgv, wq, color=RED, alpha=0.6, edgecolor=BLACK, label="DDPM 生成")
ax[1].set_xticks(xq); ax[1].set_xticklabels(qt, fontsize=8)
ax[1].set_title("分位数对齐（×1e4，越小越接近）"); ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(f"{OUT}/fig5_stats_quantiles.png"); plt.close(fig)

with open(f"{OUT}/_results.json", "w") as f:
    json.dump(R, f, ensure_ascii=False, indent=1)
print(json.dumps(R, ensure_ascii=False, indent=1))
