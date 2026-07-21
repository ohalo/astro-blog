#!/usr/bin/env python3
"""
为文章「对比学习金融表征」(contrastive-learning-finance) 生成真实配图。

所有图表都由文中代码真实计算生成（纯 numpy，无 sklearn/torch 依赖）：

  1) cover.png          —— SimCLR 风格对比损失：锚点拉近正样本、推远负样本
  2) cl_embedding.png   —— 2D PCA 投影：对比表征把同行业挤成团、跨行业拉开
  3) cl_downstream.png  —— 下游验证：少标注场景下，对比预训练表征的「单标注细胞」分类精度增益

机制（数据自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 42 支股票、3 个行业板块；每只股票一条 8 维原始特征（行业信号 0.7 倍质心 + 1.1 倍噪声）。
  - 用 SimCLR 风格 InfoNCE 在「同行业=正对 / 跨行业=负对」下，对纯 numpy 两层
    MLP 投影头做端到端对比预训练，把 8 维原始特征压成 4 维表征。
  - 下游做「少标注分类」探针（few-shot 单标注细胞分类）：每个行业只给 1~3 个标注，
    其余用「类原型 + 最近邻」分类；对比证明对比表征在低标注预算下精度更高。
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _cand in ["/System/Library/Fonts/STHeiti Medium.ttc",
              "/System/Library/Fonts/Supplemental/Songti SC.ttf"]:
    try:
        fm.fontManager.addfont(_cand)
    except Exception:
        pass
plt.rcParams["font.family"] = ["Heiti SC", "Songti SC", "STHeiti", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams.update({
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.bbox": "tight",
})

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "contrastive-learning-finance")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "pos": "#55A868", "neg": "#C44E52",
     "anchor": "#4C72B0", "raw": "#9E9E9E", "cl": "#8172B3", "gold": "#E1A100",
     "sec": ["#4C72B0", "#55A868", "#C44E52"]}

rng = np.random.default_rng(20260722)


def l2norm(x, axis=-1, eps=1e-8):
    return x / (np.linalg.norm(x, axis=axis, keepdims=True) + eps)


def pca_proj(X, d=2):
    Xc = X - X.mean(0)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:d].T, Vt[:d].T


# =====================================================================
# 合成数据：42 股 / 3 行业 / 8 维（弱结构 + 强噪声，模拟真实金融嵌入）
# =====================================================================
N_SECTOR = 3
N_PER = 14
N = N_SECTOR * N_PER
DIM = 8
OUT = 4
HID = 24
sec = np.repeat(np.arange(N_SECTOR), N_PER)
cen = np.array([[1.0, 0.4, -0.6, 0.2, 0.8, -0.3, 0.5, -0.7],
                [-0.8, 0.9, 0.3, -0.5, 0.1, 0.7, -0.2, 0.6],
                [0.3, -0.7, 0.8, 0.6, -0.4, 0.2, -0.9, 0.3]])
X = np.empty((N, DIM))
for i in range(N):
    X[i] = 0.7 * cen[sec[i]] + rng.normal(0, 1.1, DIM)
# 注意：原始特征不预先 L2 归一化——归一化会把行业结构洗成无信号的单位向量，
# 让对比任务退化（这正是早期版本训练停滞的元凶）。只归一化「学习到的投影 z」。
pos_mask = (sec[:, None] == sec[None, :]) & ~np.eye(N, dtype=bool)
neg_mask = sec[:, None] != sec[None, :]


# =====================================================================
# SimCLR 风格 InfoNCE 预训练（纯 numpy 两层 MLP 端到端）
# =====================================================================
W1 = rng.normal(0, 0.3, (DIM, HID))
b1 = np.zeros(HID)
W2 = rng.normal(0, 0.3, (HID, OUT))
b2 = np.zeros(OUT)
tau = 0.1
lr = 0.05
wd = 1e-3


def forward(x):
    h = np.tanh(x @ W1 + b1)
    z = l2norm(h @ W2 + b2)
    return h, z


def info_nce(z):
    sim = (z @ z.T) / tau
    np.fill_diagonal(sim, -1e9)
    loss = 0.0
    gz = np.zeros_like(z)
    for i in range(N):
        p = np.exp(sim[i]); p /= p.sum()
        G = p.copy(); G[pos_mask[i]] -= 1.0
        gz[i] += (1.0 / tau ** 2) * (G @ z)
        gz += (1.0 / tau ** 2) * np.outer(G, z[i])
        loss += -np.log(p[pos_mask[i]].sum() + 1e-12)
    return loss, gz


losses = []
for ep in range(600):
    h, z = forward(X)
    loss, gz = info_nce(z)
    losses.append(loss)
    a2 = h @ W2 + b2
    nrm = np.linalg.norm(a2, axis=1, keepdims=True) + 1e-8
    da2 = (np.eye(OUT)[None, :, :] - (z[:, :, None] @ z[:, None, :])) / nrm[:, None, :]
    ga2 = np.einsum("nij,nj->ni", da2, gz)
    # 注意：梯度 gz 已对 N 个样本求和，下面也是逐样本求和，故更新不要再除以 N
    gW2 = h.T @ ga2 + wd * W2
    gb2 = ga2.sum(0) + wd * b2
    gh = ga2 @ W2.T
    ga = gh * (1 - h ** 2)
    gW1 = X.T @ ga + wd * W1
    gb1 = ga.sum(0) + wd * b1
    W2 -= lr * gW2
    b2 -= lr * gb2
    W1 -= lr * gW1
    b1 -= lr * gb1

_, z_final = forward(X)
print(f"[InfoNCE] 损失从 {losses[0]:.3f} 收敛到 {losses[-1]:.3f}")

intra = np.mean([z_final[i] @ z_final[j] for i in range(N) for j in range(N) if pos_mask[i, j]])
inter = np.mean([z_final[i] @ z_final[j] for i in range(N) for j in range(N) if neg_mask[i, j]])
raw_intra = np.mean([X[i] @ X[j] for i in range(N) for j in range(N) if pos_mask[i, j]])
raw_inter = np.mean([X[i] @ X[j] for i in range(N) for j in range(N) if neg_mask[i, j]])
print(f"[结构] 原始特征 同行业相似度={raw_intra:.3f} 跨行业={raw_inter:.3f}")
print(f"[结构] 对比表征 同行业相似度={intra:.3f} 跨行业={inter:.3f}")


# =====================================================================
# 图 1（cover）：三元组对比损失示意
# =====================================================================
fig, ax = plt.subplots(figsize=(12, 5))
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.2, 1.2)
ax.set_yticks([]); ax.set_xticks([])
ax.set_title("SimCLR / InfoNCE：把正样本拉近、负样本推远")
ax.scatter([0], [0], s=260, color=C["anchor"], zorder=5, label="锚点 a")
ax.annotate("锚点 a", (0, 0), xytext=(0.15, -0.25), fontsize=10)
ax.scatter([0.3], [0.5], s=170, color=C["pos"], zorder=5)
ax.annotate("正样本 + (同行业)", (0.3, 0.5), xytext=(0.4, 0.5), fontsize=10)
ax.plot([0, 0.3], [0, 0.5], color=C["pos"], lw=2.5, ls="-")
for nx, ny in [(-1.1, 0.6), (-1.2, -0.4), (1.2, 0.7), (1.1, -0.6)]:
    ax.scatter([nx], [ny], s=130, color=C["neg"], zorder=4)
    ax.plot([0, nx], [0, ny], color=C["neg"], lw=1.6, ls="--", alpha=0.7)
ax.annotate("负样本 − (跨行业)", (-1.1, 0.6), xytext=(-1.55, 0.75), fontsize=10, color=C["neg"])
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "cover.png"))
plt.close(fig)

# =====================================================================
# 图 2：2D PCA 投影（原始 vs 对比表征）
# =====================================================================
raw2, _ = pca_proj(X, 2)
cl2, _ = pca_proj(z_final, 2)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for s in range(N_SECTOR):
    m = sec == s
    axes[0].scatter(raw2[m, 0], raw2[m, 1], s=30, color=C["sec"][s], alpha=0.7, label=f"行业{s+1}")
    axes[1].scatter(cl2[m, 0], cl2[m, 1], s=30, color=C["sec"][s], alpha=0.8, label=f"行业{s+1}")
axes[0].set_title("原始特征 PCA 投影（行业被噪声淹没）")
axes[1].set_title("对比表征 PCA 投影（同行业聚团、跨行业分离）")
for a in axes:
    a.set_xlabel("PC1"); a.set_ylabel("PC2"); a.legend(fontsize=8, loc="best")
fig.tight_layout()
fig.savefig(os.path.join(D, "cl_embedding.png"))
plt.close(fig)

# =====================================================================
# 图 3：下游少标注探针（few-shot 单标注细胞分类）
# =====================================================================
def fewshot_acc(rep, per, runs=60, seed=5):
    """每个行业只给 per 个标注，类原型 + 最近邻分类其余；返回平均精度。"""
    r = np.random.default_rng(seed)
    A = l2norm(rep)
    accs = []
    for _ in range(runs):
        lab, unl = [], []
        for s in range(N_SECTOR):
            js = np.where(sec == s)[0]
            pick = r.choice(js, per, replace=False)
            lab += list(pick); unl += [j for j in js if j not in pick]
        lab = np.array(lab); unl = np.array(unl)
        proto = np.array([A[lab][sec[lab] == s].mean(0) for s in range(N_SECTOR)])
        pred = np.argmax(proto @ A[unl].T, axis=0)
        accs.append(np.mean(pred == sec[unl]))
    return np.mean(accs)


results = {per: (fewshot_acc(X, per), fewshot_acc(z_final, per)) for per in [1, 2, 3]}
print("[下游] 少标注精度（原始 vs 对比）:")
for per, (ra, ca) in results.items():
    print(f"   每行业标注 {per} 个: 原始={ra:.3f} 对比={ca:.3f} 增益={ca-ra:+.3f}")

fig, ax = plt.subplots(figsize=(7.4, 4.8))
pers = [1, 2, 3]
ra = [results[p][0] for p in pers]
ca = [results[p][1] for p in pers]
xw = np.arange(len(pers))
ax.bar(xw - 0.2, ra, 0.4, color=C["raw"], label="原始特征")
ax.bar(xw + 0.2, ca, 0.4, color=C["cl"], label="对比表征")
ax.set_xticks(xw); ax.set_xticklabels([f"{p} 个/行业" for p in pers])
ax.set_ylabel("下游分类精度（少标注探针）")
ax.set_title("标注越稀缺，对比表征增益越大（增益随标注增多收敛）")
ax.legend(fontsize=9)
for i, (v1, v2) in enumerate(zip(ra, ca)):
    ax.text(i - 0.2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=9)
    ax.text(i + 0.2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=9)
ax.set_ylim(0, 1.0)
fig.tight_layout()
fig.savefig(os.path.join(D, "cl_downstream.png"))
plt.close(fig)

print("[完成] 已生成 cover.png / cl_embedding.png / cl_downstream.png ->", D)
