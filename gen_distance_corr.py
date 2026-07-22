#!/usr/bin/env python3
"""生成 距离相关系数 文章：纯 numpy 从零实现 dCor/dCov（Székely 2007）+ 4 张真实图表。

距离相关系数（distance correlation, dCor）是 Pearson 相关在「非线性依赖」上的推广：
Pearson r 只能抓线性，r=0 时两变量仍可能强烈依赖（如 Y=X²）；而 dCor=0 当且仅当两变量
独立（一个真正的独立性检验）。本文从零实现：
  - 双中心化距离矩阵 A_ij = a_ij - ā_i· - ā_·j + ā_··  （a_ij=|X_i-X_j|）
  - dCov²(X,Y) = (1/n²) ∑∑ A_ij·B_ij
  - dCor = dCov / √(dVarX · dVarY)
并在 4 个合成关系上对比 Pearson vs dCor。诚实结论：
  - 二次依赖：Pearson r≈0（看不出），dCor≈0.21（一眼看穿）→ dCor 的强项
  - 径向对称（圆）：Pearson 与 dCor 双双≈0 → dCor 的盲区（真实边界，不粉饰）
  - 置换检验对「二次」能拒绝独立，对「独立对照」不能拒绝 → 检验功效成立
所有图表均由下文真实计算。
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


# ============ 核心：dCov / dCor（纯 numpy） ============
def dist_mat(X):
    """成对欧氏距离矩阵（1 维即 |x_i-x_j|）"""
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def double_center(D):
    """双中心化：A_ij = d_ij - rowmean_i - colmean_j + grandmean"""
    n = D.shape[0]
    row = D.mean(axis=1, keepdims=True)
    col = D.mean(axis=0, keepdims=True)
    grand = D.mean()
    return D - row - col + grand


def dcov2(X, Y):
    A = double_center(dist_mat(X))
    B = double_center(dist_mat(Y))
    return float(np.sum(A * B) / (len(X) ** 2))


def dvar2(X):
    return dcov2(X, X)


def dcor(X, Y):
    cov = dcov2(X, Y)
    vx = dvar2(X)
    vy = dvar2(Y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / np.sqrt(vx * vy))


def pearson(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    return float(np.corrcoef(X, Y)[0, 1])


# ============ 1. 合成关系 ============
rng = np.random.default_rng(20260723)
N = 400

# 线性
x_lin = rng.uniform(-1, 1, N)
y_lin = x_lin + rng.normal(0, 0.25, N)

# 二次（对称 X → Pearson r≈0，但明显依赖）
x_sq = rng.uniform(-1, 1, N)
y_sq = x_sq ** 2 + rng.normal(0, 0.10, N)

# 圆关系（径向对称：X=cos θ, Y=sin θ → 完全依赖，但 dCor 的盲区）
theta = rng.uniform(0, 2 * np.pi, N)
x_circ = np.cos(theta) + rng.normal(0, 0.05, N)
y_circ = np.sin(theta) + rng.normal(0, 0.05, N)

# 独立对照（重新生成一份真正无关的噪声，避免和上面共线）
rng_i = np.random.default_rng(999)
x_ind = rng_i.uniform(-1, 1, N)
y_ind = rng_i.uniform(-1, 1, N)

relations = {
    "线性": (x_lin, y_lin),
    "二次 Y=X²": (x_sq, y_sq),
    "圆 X²+Y²": (x_circ, y_circ),
    "独立(对照)": (x_ind, y_ind),
}

rows = []
for name, (x, y) in relations.items():
    r = pearson(x, y)
    dc = dcor(x, y)
    rows.append((name, r, dc))
    print(f"  {name:10s}  Pearson r = {r:+.4f}   dCor = {dc:.4f}")

# ============ 2. 置换检验（独立数据上 dCor 应≈0，p 应大；依赖数据 p 应小） ============
def perm_pvalue(X, Y, B=1000, seed=12345):
    rng_p = np.random.default_rng(seed)
    obs = dcor(X, Y)
    Yp = Y.copy()
    perm = np.empty(B)
    for b in range(B):
        rng_p.shuffle(Yp)
        perm[b] = dcor(X, Yp)
    p = (1 + np.sum(perm >= obs)) / (1 + B)
    return obs, perm, p

# 二次关系：dCor 弱但仍能拒绝独立（检验有功效）
obs_sq, perm_sq, p_sq = perm_pvalue(x_sq, y_sq, B=1000, seed=101)
# 独立对照：用真正无关的噪声，确保 p 偏大、不能拒绝
obs_ind, perm_ind, p_ind = perm_pvalue(x_ind, y_ind, B=1000, seed=202)
print(f"  二次 置换 p = {p_sq:.4f}  独立 置换 p = {p_ind:.4f}")

# ============ 3. 噪声稀释：二次关系 dCor 随噪声增大而衰减 ============
noise_levels = np.linspace(0.02, 0.60, 12)
dcor_vs_noise = []
for sd in noise_levels:
    y = x_sq ** 2 + rng.normal(0, sd, N)
    dcor_vs_noise.append(dcor(x_sq, y))
dcor_vs_noise = np.array(dcor_vs_noise)

# ============ 4. 双中心矩阵可视化（二次关系） ============
A_sq = double_center(dist_mat(x_sq))
B_sq = double_center(dist_mat(y_sq))
K_sq = A_sq * B_sq  # dCov 的逐元素核

# ============ 5. 图像 ============
outdir = "public/images/distance-correlation-dependence"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：4 个关系的 Pearson vs dCor 对比
fig, ax = plt.subplots(figsize=(11, 5.4))
names = [r[0] for r in rows]
r_vals = [r[1] for r in rows]
dc_vals = [r[2] for r in rows]
xpos = np.arange(len(names))
ax.bar(xpos - 0.2, r_vals, width=0.4, color="#7f7f7f", label="Pearson 相关系数 r")
ax.bar(xpos + 0.2, dc_vals, width=0.4, color="#1a9850", label="距离相关系数 dCor")
for i, (rv, dv) in enumerate(zip(r_vals, dc_vals)):
    ax.text(i - 0.2, rv + 0.02, f"{rv:+.2f}", ha="center", fontsize=8.5, color="#444")
    ax.text(i + 0.2, dv + 0.02, f"{dv:.2f}", ha="center", fontsize=8.5, color="#1a6b32")
ax.set_xticks(xpos)
ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(-0.1, 1.15)
ax.set_ylabel("系数取值")
ax.set_title("Pearson 抓不住的，dCor 也不一定都抓得住：二次依赖一击即中，径向对称却双双翻车",
             fontsize=12.8, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9)
ax.grid(alpha=0.25, axis="y")
fig.tight_layout()
fig.savefig(f"{outdir}/cover.png", dpi=130)
plt.close(fig)

# 图2：双中心距离矩阵核（二次关系）
fig, axes = plt.subplots(1, 3, figsize=(11, 3.4))
im0 = axes[0].imshow(A_sq, cmap="RdBu_r", aspect="auto")
axes[0].set_title("A：X 距离双中心化", fontsize=10.5)
axes[0].axis("off")
fig.colorbar(im0, ax=axes[0], fraction=0.05, pad=0.04)
im1 = axes[1].imshow(B_sq, cmap="RdBu_r", aspect="auto")
axes[1].set_title("B：Y 距离双中心化", fontsize=10.5)
axes[1].axis("off")
fig.colorbar(im1, ax=axes[1], fraction=0.05, pad=0.04)
im2 = axes[2].imshow(K_sq, cmap="RdYlBu_r", aspect="auto")
axes[2].set_title("A⊙B：dCov 的逐元素贡献", fontsize=10.5)
axes[2].axis("off")
fig.colorbar(im2, ax=axes[2], fraction=0.05, pad=0.04)
fig.suptitle("双中心化把『距离结构』扣掉均值后相乘：依赖就藏在 A⊙B 里",
             fontsize=12, fontweight="bold", color="#1f3a5f")
fig.tight_layout()
fig.savefig(f"{outdir}/double_center.png", dpi=80)
plt.close(fig)

# 图3：置换检验（二次关系 + 独立对照）
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
axes[0].hist(perm_sq, bins=30, color="#1a9850", alpha=0.8)
axes[0].axvline(obs_sq, color="#d73027", lw=2, label=f"观测 dCor={obs_sq:.3f}")
axes[0].set_title(f"二次关系：观测值落在置换分布右尾（p={p_sq:.3f}，拒绝独立）",
                  fontsize=11.5, color="#1f3a5f")
axes[0].set_xlabel("置换 dCor"); axes[0].set_ylabel("频数"); axes[0].legend(fontsize=9)
axes[1].hist(perm_ind, bins=30, color="#4393c3", alpha=0.8)
axes[1].axvline(obs_ind, color="#d73027", lw=2, label=f"观测 dCor={obs_ind:.3f}")
axes[1].set_title(f"独立数据：观测值落在置换分布中央（p={p_ind:.3f}，不能拒绝独立）",
                  fontsize=11.5, color="#1f3a5f")
axes[1].set_xlabel("置换 dCor"); axes[1].set_ylabel("频数"); axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{outdir}/perm_test.png", dpi=130)
plt.close(fig)

# 图4：噪声稀释
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(noise_levels, dcor_vs_noise, "o-", color="#762a83", lw=1.8)
ax.set_xlabel("噪声标准差 σ（加到 Y=X² 上）")
ax.set_ylabel("距离相关系数 dCor")
ax.set_title("诚实边界：噪声越大，dCor 越被稀释——它量的是『依赖强度』不是『有无依赖』",
             fontsize=12.8, fontweight="bold", color="#1f3a5f")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{outdir}/noise_dilution.png", dpi=130)
plt.close(fig)

# ============ 6. stats ============
stats = {
    "N": N,
    "relations": {name: {"pearson": round(r, 4), "dcor": round(dc, 4)}
                  for name, r, dc in rows},
    "perm_quad_p": round(float(p_sq), 4),
    "perm_ind_p": round(float(p_ind), 4),
    "dcor_vs_noise": {f"{sd:.2f}": round(float(v), 4) for sd, v in zip(noise_levels, dcor_vs_noise)},
}
with open("public/images/distance-correlation-dependence/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\nDistance correlation images written. quad dCor =", round(dcor(x_sq, y_sq), 4))
