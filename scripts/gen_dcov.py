#!/usr/bin/env python3
"""生成 DCov 距离协方差 文章：用「特征映射的核」把依赖关系写成距离。

本文与「距离相关系数(dCor)」是同一棵树的两个视角：
  dCor 文章强调「dCor=0 ⟺ 独立」的独立性检验；
  本文强调「dCov 就是 RKHS 里的协方差 / 能量距离是分布间的距离」——
  把依赖关系写成距离、写成核内积，并给出两个可直接落地的量化工具：
    1) HSIC（Hilbert-Schmidt 独立性准则）：用 RBF 核把 dCov 思想搬进核空间；
    2) 能量距离（energy distance）：一个真正的「分布间距离」度量，用来做
       两样本检验——识别行情处于「平静态」还是「危机态」。
所有图表均由下文真实计算（numpy + matplotlib），非占位图。
"""
import numpy as np
import os
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from collections import Counter

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti SC"]
plt.rcParams["axes.unicode_minus"] = False
fm._load_fontmanager()


# ============ 核心：dCov / dCor（纯 numpy，距离视角） ============
def dist_mat(X):
    """成对欧氏距离矩阵（1 维即 |x_i-x_j|）"""
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    diff = X[:, None, :] - X[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def double_center(D):
    """双中心化：A_ij = d_ij - rowmean_i - colmean_j + grandmean"""
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
    vx, vy = dvar2(X), dvar2(Y)
    if vx <= 0 or vy <= 0:
        return 0.0
    return float(cov / np.sqrt(vx * vy))


def pearson(X, Y):
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    return float(np.corrcoef(X, Y)[0, 1])


# ============ 核视角：HSIC（RBF 核，把 dCov 搬进 RKHS） ============
def sq_dists(X):
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    return np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)


def rbf_kernel(X, gamma):
    return np.exp(-gamma * sq_dists(X))


def hsic(X, Y, gamma=None):
    """Hilbert-Schmidt 独立性准则：HSIC = (1/n²) tr(K H L H)

    H = I - (1/n) 11ᵀ 为中心化矩阵。gamma 用中位数启发式（median heuristic）。
    HSIC=0 ⟺ 在对应 RKHS 里独立；它是 dCov 在「核特征空间」里的表亲。
    """
    X = np.asarray(X, float).reshape(-1, 1) if np.ndim(X) == 1 else np.asarray(X, float)
    Y = np.asarray(Y, float).reshape(-1, 1) if np.ndim(Y) == 1 else np.asarray(Y, float)
    n = len(X)
    dX = np.sqrt(sq_dists(X)); dY = np.sqrt(sq_dists(Y))
    dX[dX == 0] = np.nan; dY[dY == 0] = np.nan
    gx = 1.0 / (2 * np.nanmedian(dX) ** 2)
    gy = 1.0 / (2 * np.nanmedian(dY) ** 2)
    gamma = (gx + gy) / 2 if gamma is None else gamma
    K = rbf_kernel(X, gamma)
    L = rbf_kernel(Y, gamma)
    H = np.eye(n) - np.ones((n, n)) / n
    return float(np.trace(K @ H @ L @ H) / (n * n))


# ============ 能量距离：一个真正的「分布间距离」 ============
def energy_distance(a, b):
    """Székely(2002) 能量距离：D(P,Q)=E|X-X'|+E|Y-Y'|-2E|X-Y| 的样本版。
    D=0 当且仅当 P=Q，因此它是分布之间的真度量（可用于两样本检验）。"""
    a = np.asarray(a, float); b = np.asarray(b, float)
    na, nb = len(a), len(b)
    Daa = np.abs(a[:, None] - a[None, :])
    Dbb = np.abs(b[:, None] - b[None, :])
    Dab = np.abs(a[:, None] - b[None, :])
    # 只用上三角避免重复（i<j），与 Dab 的全配对均值口径一致
    e_xx = 2 * np.sum(np.triu(Daa, 1)) / (na * (na - 1))
    e_yy = 2 * np.sum(np.triu(Dbb, 1)) / (nb * (nb - 1))
    e_xy = np.mean(Dab)
    # Székely(2002) 能量距离平方：D² = 2E|X-Y| - E|X-X'| - E|Y-Y'| ≥ 0
    return float(2 * e_xy - e_xx - e_yy)


# ============ 1. 四类关系：dCor vs HSIC ============
rng = np.random.default_rng(20260723)
N = 400

x_lin = rng.uniform(-1, 1, N); y_lin = x_lin + rng.normal(0, 0.25, N)
x_sq = rng.uniform(-1, 1, N);  y_sq = x_sq ** 2 + rng.normal(0, 0.10, N)
theta = rng.uniform(0, 2 * np.pi, N)
x_circ = np.cos(theta) + rng.normal(0, 0.05, N)
y_circ = np.sin(theta) + rng.normal(0, 0.05, N)
rng_i = np.random.default_rng(999)
x_ind = rng_i.uniform(-1, 1, N); y_ind = rng_i.uniform(-1, 1, N)

relations = {
    "线性": (x_lin, y_lin),
    "二次 Y=X²": (x_sq, y_sq),
    "圆 X²+Y²": (x_circ, y_circ),
    "独立(对照)": (x_ind, y_ind),
}
rows = []
for name, (x, y) in relations.items():
    r = pearson(x, y); dc = dcor(x, y); h = hsic(x, y)
    rows.append((name, r, dc, h))
    print(f"  {name:10s}  Pearson r = {r:+.4f}   dCor = {dc:.4f}   HSIC = {h:.4f}")

# ============ 2. 核矩阵可视化（二次关系）：HSIC 的「交互核」 K H L H ============
dX_sq = np.sqrt(sq_dists(x_sq)); dY_sq = np.sqrt(sq_dists(y_sq))
dX_sq[dX_sq == 0] = np.nan; dY_sq[dY_sq == 0] = np.nan
gamma_sq = (1.0 / (2 * np.nanmedian(dX_sq) ** 2) + 1.0 / (2 * np.nanmedian(dY_sq) ** 2)) / 2
K_sq = rbf_kernel(x_sq, gamma_sq)
L_sq = rbf_kernel(y_sq, gamma_sq)
n = len(x_sq)
H = np.eye(n) - np.ones((n, n)) / n
KHLH = K_sq @ H @ L_sq @ H


# ============ 3. 能量距离做两样本检验：平静态 vs 危机态 ============
# 固定参考样本：平静态（低波动、零漂移）/ 危机态（高波动、负漂移）
rng_r = np.random.default_rng(4242)
calm_ref = rng_r.normal(0.0003, 0.005, 300)
crisis_ref = rng_r.normal(-0.0020, 0.025, 300)

# 构造一段含 regime 切换的日收益序列：平静-危机-平静-危机
rng_s = np.random.default_rng(777)
def block(kind, length):
    if kind == "calm":
        return rng_s.normal(0.0003, 0.005, length)
    else:
        return rng_s.normal(-0.0020, 0.025, length)
series = np.concatenate([block("calm", 150), block("crisis", 150),
                         block("calm", 150), block("crisis", 150)])
true_regime = (["calm"] * 150 + ["crisis"] * 150 +
               ["calm"] * 150 + ["crisis"] * 150)

W = 60
d_to_calm = []; d_to_crisis = []
edges = range(W, len(series) + 1)
for t in edges:
    win = series[t - W:t]
    d_to_calm.append(energy_distance(win, calm_ref))
    d_to_crisis.append(energy_distance(win, crisis_ref))
d_to_calm = np.array(d_to_calm); d_to_crisis = np.array(d_to_crisis)
# 用「更靠近谁」做分类，与真实 regime 对比
pred = np.where(d_to_crisis < d_to_calm, "crisis", "calm")
acc = np.mean(np.array(pred) == np.array(true_regime[W - 1:]))
# 把 crisis 区间的平均能量距离对比打印出来
calm_win_idx = [i for i, rg in enumerate(true_regime[W - 1:]) if rg == "calm"]
crisis_win_idx = [i for i, rg in enumerate(true_regime[W - 1:]) if rg == "crisis"]
mean_dc_calm = d_to_calm[calm_win_idx].mean()
mean_dc_crisis = d_to_crisis[crisis_win_idx].mean()
mean_crisis_to_calm = d_to_calm[crisis_win_idx].mean()   # 危机窗口 → 平静参考（应偏大）
mean_calm_to_crisis = d_to_crisis[calm_win_idx].mean()   # 平静窗口 → 危机参考（应偏大）
print(f"  能量距离两样本分类正确率 = {acc*100:.1f}%")
print(f"  平静窗口→平静参考 = {mean_dc_calm:.5f}  危机窗口→危机参考 = {mean_dc_crisis:.5f}  (同类，小)")
print(f"  平静窗口→危机参考 = {mean_calm_to_crisis:.5f}  危机窗口→平静参考 = {mean_crisis_to_calm:.5f}  (异类，大)")


# ============ 4. HSIC 也被噪声稀释（与 dCor 同样的诚实边界） ============
noise_levels = np.linspace(0.02, 0.60, 12)
dc_vs_noise = []; hsic_vs_noise = []
for sd in noise_levels:
    y = x_sq ** 2 + rng.normal(0, sd, N)
    dc_vs_noise.append(dcor(x_sq, y))
    hsic_vs_noise.append(hsic(x_sq, y))
dc_vs_noise = np.array(dc_vs_noise)
hsic_vs_noise = np.array(hsic_vs_noise)


# ============ 5. 图像 ============
outdir = "public/images/dcov-nonlinear-dependence"
os.makedirs(outdir, exist_ok=True)

# 图1 cover：Pearson vs dCor vs HSIC 在 4 类关系上
fig, ax = plt.subplots(figsize=(11, 5.4))
names = [r[0] for r in rows]
r_vals = [r[1] for r in rows]
dc_vals = [r[2] for r in rows]
h_vals = [r[3] for r in rows]
xpos = np.arange(len(names))
w = 0.27
ax.bar(xpos - w, r_vals, width=w, color="#7f7f7f", label="Pearson r（只能线性）")
ax.bar(xpos, dc_vals, width=w, color="#1a9850", label="距离协方差 dCor")
ax.bar(xpos + w, h_vals, width=w, color="#2166ac", label="核 HSIC（RKHS 视角）")
for i, (rv, dv, hv) in enumerate(zip(r_vals, dc_vals, h_vals)):
    ax.text(i - w, rv + 0.02, f"{rv:+.2f}", ha="center", fontsize=8, color="#444")
    ax.text(i, dv + 0.02, f"{dv:.2f}", ha="center", fontsize=8, color="#1a6b32")
    ax.text(i + w, hv + 0.02, f"{hv:.2f}", ha="center", fontsize=8, color="#15375f")
ax.set_xticks(xpos); ax.set_xticklabels(names, fontsize=10)
ax.set_ylim(-0.1, 1.15); ax.set_ylabel("依赖量取值")
ax.set_title("同一回事两种写法：距离(dCor)与核(HSIC)在二次依赖上双双看穿 Pearson 的盲区",
             fontsize=12.6, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9); ax.grid(alpha=0.25, axis="y")
fig.tight_layout(); fig.savefig(f"{outdir}/cover.png", dpi=130); plt.close(fig)

# 图2：HSIC 交互核 K H L H（二次关系）
fig, ax = plt.subplots(figsize=(6.2, 5.2))
im = ax.imshow(KHLH, cmap="RdBu_r", aspect="auto")
ax.set_title("HSIC 的交互核 K·H·L·H：\n依赖被写进核矩阵的非对角结构（二次关系）",
             fontsize=11.5, fontweight="bold", color="#1f3a5f")
ax.axis("off"); fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)
fig.tight_layout(); fig.savefig(f"{outdir}/hsic_kernel.png", dpi=110); plt.close(fig)

# 图3：能量距离两样本检验（平静/危机 regime 识别）
fig, ax = plt.subplots(figsize=(11.5, 5.2))
ts = np.arange(len(d_to_calm))
ax.plot(ts, d_to_calm, color="#4393c3", lw=1.8, label="窗口 → 平静参考 的能量距离")
ax.plot(ts, d_to_crisis, color="#d73027", lw=1.8, label="窗口 → 危机参考 的能量距离")
# 着色真实 regime 背景
for start, end, rg in [(0, 150, "calm"), (150, 300, "crisis"),
                       (300, 450, "calm"), (450, 600, "crisis")]:
    s0 = max(0, start - W + 1); e0 = end - W + 1
    ax.axvspan(s0, e0, color=("#4393c3" if rg == "calm" else "#d73027"),
               alpha=0.08)
ax.set_xlabel("时间（窗口起点）"); ax.set_ylabel("能量距离")
ax.set_title(f"能量距离当「分布指纹」：危机窗口离危机参考更近、离平静参考更远（分类正确率 {acc*100:.0f}%）",
             fontsize=12.2, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9, loc="upper right"); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{outdir}/energy_two_sample.png", dpi=130); plt.close(fig)

# 图4：噪声稀释（dCor 与 HSIC 都被稀释）
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.plot(noise_levels, dc_vs_noise, "o-", color="#1a9850", lw=1.8, label="dCor（距离视角）")
ax.plot(noise_levels, hsic_vs_noise, "s-", color="#2166ac", lw=1.8, label="HSIC（核视角）")
ax.set_xlabel("噪声标准差 σ（加到 Y=X² 上）")
ax.set_ylabel("依赖量")
ax.set_title("诚实边界：两种写法都量「强度」而非「有无」——噪声一上来就把依赖稀释掉",
             fontsize=12.4, fontweight="bold", color="#1f3a5f")
ax.legend(fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(f"{outdir}/noise_dilution.png", dpi=130); plt.close(fig)

# ============ 6. stats ============
stats = {
    "N": N,
    "relations": {name: {"pearson": round(r, 4), "dcor": round(dc, 4), "hsic": round(h, 4)}
                  for name, r, dc, h in rows},
    "two_sample_accuracy": round(float(acc), 4),
    "energy_calm_to_calm_ref": round(float(mean_dc_calm), 5),
    "energy_crisis_to_crisis_ref": round(float(mean_dc_crisis), 5),
    "energy_calm_to_crisis_ref": round(float(mean_calm_to_crisis), 5),
    "energy_crisis_to_calm_ref": round(float(mean_crisis_to_calm), 5),
    "dcor_vs_noise": {f"{sd:.2f}": round(float(v), 4) for sd, v in zip(noise_levels, dc_vs_noise)},
    "hsic_vs_noise": {f"{sd:.2f}": round(float(v), 4) for sd, v in zip(noise_levels, hsic_vs_noise)},
}
with open("public/images/dcov-nonlinear-dependence/stats.json", "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)

print("\nDCov images written. two-sample accuracy =", round(acc * 100, 1), "%")
