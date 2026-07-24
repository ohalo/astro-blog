#!/usr/bin/env python3
"""
为文章「图注意力金融网络：用注意力权重重新分配板块间的信息流」(graph-attention-finance)
生成真实配图 + 计算正文引用的关键数字。

机制（自洽合成，仅用于演示 GAT 单层前向 + 注意力学习）：
  * 节点 = 板块/资产，每个节点有特征向量 x_i（因子暴露）。
  * 图 = 板块相邻关系（供应链/资金流邻接）。GCN 用固定归一化权重平均邻居，
    GAT 用可学习注意力 a(Wh_i, Wh_j) 给每条边打分，softmax 归一化后加权聚合。
  * 目标：预测每个节点下一期收益。真实数据结构里，信息流是「非对称、稀疏」的——
    少数关键板块(如半导体)对下游影响大。GAT 能学出这种不均匀权重，GCN 只会均匀平均。
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import FancyArrowPatch, Circle

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "graph-attention-finance")
os.makedirs(D, exist_ok=True)

C = {"gat": "#C44E52", "gcn": "#4C72B0", "mlp": "#999999",
     "hub": "#DD8452", "leaf": "#55A868", "grid": "#DDDDDD", "edge": "#BBBBBB"}

rng = np.random.default_rng(7)

# ---------------------------------------------------------------------------
# 构造板块图：N 个板块，含 1 个 hub（半导体），信息非对称流向下游
# ---------------------------------------------------------------------------
SECTORS = ["半导体", "消费电子", "新能源车", "光伏", "锂电材料",
           "医药", "白酒", "银行", "地产", "券商"]
N = len(SECTORS)
F = 6                      # 每个板块的因子特征维度

# 邻接（有向：j -> i 表示 j 影响 i）。半导体(0)是 hub，强影响多个下游。
edges = [
    (0, 1), (0, 2), (0, 3),          # 半导体 -> 消费电子/新能源车/光伏
    (2, 4), (3, 4),                  # 新能源车/光伏 -> 锂电材料
    (5, 6),                          # 医药 -> 白酒(弱噪声边)
    (7, 8), (7, 9), (8, 9),          # 银行 -> 地产/券商, 地产->券商
    (1, 2), (4, 2),                  # 反馈边
    # 以下为“虹吸式”噪声边：图上存在连接但真实传导=0（市场惯性误认的关联）
    (6, 0), (8, 0), (9, 0),          # 白酒/地产/券商 误连到半导体
    (4, 7), (2, 5), (1, 8), (3, 9),  # 跨链噪声边
    (5, 0), (6, 7),                  # 更多噪声
]
# 自环
for i in range(N):
    edges.append((i, i))

A = np.zeros((N, N))
for (j, i) in edges:
    A[i, j] = 1.0            # A[i,j]=1 表示 i 能看到 j

# 真实信息流强度（生成 y 用）：半导体对下游的传导系数最大
BETA = np.zeros((N, N))
true_flow = {(0, 1): 0.9, (0, 2): 0.8, (0, 3): 0.7, (2, 4): 0.5,
             (3, 4): 0.4, (5, 6): 0.15, (7, 8): 0.5, (7, 9): 0.4, (8, 9): 0.35,
             (1, 2): 0.3, (4, 2): 0.25}
for (j, i), b in true_flow.items():
    BETA[i, j] = b

# 节点特征 & 目标：y_i = 自身因子 + sum_j BETA[i,j]*(j 的动量) + 噪声
T = 500
X_seq = rng.standard_normal((T, N, F))
own = X_seq[:, :, 0]                        # 每个板块自身动量因子
Y = np.zeros((T, N))
for t in range(T):
    spill = BETA @ own[t]                   # 邻居传导
    Y[t] = 0.5 * own[t] + spill + 0.5 * rng.standard_normal(N)

# ---------------------------------------------------------------------------
# 单层 GAT / GCN / MLP：纯 numpy 实现，含通过注意力 softmax 的完整反向
# ---------------------------------------------------------------------------
DH = 4          # 隐藏维度
LRELU = 0.2
TRAIN_T = T - 100     # 前 400 时间点训练，后 100 样本外

def leaky(z):
    return np.where(z > 0, z, LRELU * z)

def leaky_g(z):
    return np.where(z > 0, 1.0, LRELU)

def masked_softmax(e, mask):
    e = np.where(mask > 0, e, -1e9)
    e = e - e.max(axis=1, keepdims=True)
    ex = np.exp(e) * mask
    return ex / (ex.sum(axis=1, keepdims=True) + 1e-12)

# GCN 对称归一化邻接
deg = A.sum(1)
Dinv = np.diag(1.0 / np.sqrt(deg))
A_norm = Dinv @ A @ Dinv


def forward(kind, Xt, W, a_src, a_dst, v, b):
    H = Xt @ W                                   # (N,DH)
    cache = {"H": H, "Xt": Xt}
    if kind == "gat":
        zs = H @ a_dst                           # (N,) receiver term (i)
        zr = H @ a_src                           # (N,) source term (j)
        z = leaky(zs[:, None] + zr[None, :])     # (N,N)
        alpha = masked_softmax(z, A)             # (N,N)
        Hagg = alpha @ H                         # (N,DH)
        cache.update({"z_pre": zs[:, None] + zr[None, :], "alpha": alpha})
    elif kind == "gcn":
        Hagg = A_norm @ H
        cache["alpha"] = A_norm
    else:                                        # mlp
        Hagg = H
        cache["alpha"] = np.eye(N)
    pred = Hagg @ v + b                          # (N,1)
    cache["Hagg"] = Hagg
    return pred.reshape(-1), cache


def backward(kind, err, cache, W, a_src, a_dst, v):
    N_ = len(err)
    H = cache["H"]; Xt = cache["Xt"]; Hagg = cache["Hagg"]
    dpred = (2.0 * err / N_).reshape(-1, 1)      # (N,1)
    dv = Hagg.T @ dpred                          # (DH,1)
    db = float(dpred.sum())
    dHagg = dpred @ v.T                          # (N,DH)
    da_src = np.zeros_like(a_src); da_dst = np.zeros_like(a_dst)
    if kind == "gat":
        alpha = cache["alpha"]
        dH = alpha.T @ dHagg                     # value path
        dalpha = dHagg @ H.T                     # (N,N)
        gvec = (dalpha * alpha).sum(1, keepdims=True)
        de = alpha * (dalpha - gvec)            # softmax backward
        dz = de * leaky_g(cache["z_pre"])
        rowsum = dz.sum(1); colsum = dz.sum(0)  # (N,)
        da_dst = H.T @ rowsum                    # (DH,)
        da_src = H.T @ colsum
        dH = dH + rowsum[:, None] * a_dst[None, :] + colsum[:, None] * a_src[None, :]
    elif kind == "gcn":
        dH = A_norm.T @ dHagg
    else:
        dH = dHagg
    dW = Xt.T @ dH                               # (F,DH)
    return dW, da_src, da_dst, dv, db


def train_model(kind, epochs=500, lr=0.03):
    rs = np.random.default_rng(123)
    W = 0.2 * rs.standard_normal((F, DH))
    a_src = 0.1 * rs.standard_normal(DH)
    a_dst = 0.1 * rs.standard_normal(DH)
    v = 0.2 * rs.standard_normal((DH, 1))
    b = 0.0
    losses = []
    for ep in range(epochs):
        idx = rs.choice(TRAIN_T, 48, replace=False)
        gW = np.zeros_like(W); ga_s = np.zeros_like(a_src); ga_d = np.zeros_like(a_dst)
        gv = np.zeros_like(v); gb = 0.0; loss = 0.0
        for t in idx:
            pred, cache = forward(kind, X_seq[t], W, a_src, a_dst, v, b)
            err = pred - Y[t]
            loss += float(err @ err) / N
            dW, das, dad, dv, db_ = backward(kind, err, cache, W, a_src, a_dst, v)
            gW += dW; ga_s += das; ga_d += dad; gv += dv; gb += db_
        m = len(idx)
        W -= lr * gW / m; a_src -= lr * ga_s / m; a_dst -= lr * ga_d / m
        v -= lr * gv / m; b -= lr * gb / m
        losses.append(loss / m)
    params = (W, a_src, a_dst, v, b)
    # 用训练集平均注意力矩阵作为可视化代表
    if kind == "gat":
        acc = np.zeros((N, N))
        for t in range(TRAIN_T):
            _, c = forward(kind, X_seq[t], *params)
            acc += c["alpha"]
        att = acc / TRAIN_T
    else:
        att = cache["alpha"]
    return params, losses, att


results = {}
for kind in ["mlp", "gcn", "gat"]:
    params, losses, att = train_model(kind)
    preds, tgts = [], []
    for t in range(T - 100, T):
        pred, _ = forward(kind, X_seq[t], *params)
        preds.append(pred); tgts.append(Y[t])
    preds = np.array(preds); tgts = np.array(tgts)
    ic = np.mean([np.corrcoef(preds[:, i], tgts[:, i])[0, 1] for i in range(N)])
    rmse = float(np.sqrt(np.mean((preds - tgts) ** 2)))
    results[kind] = {"losses": losses, "att": att, "ic": float(ic), "rmse": rmse}
    print(f"{kind}: IC={ic:.3f} RMSE={rmse:.3f}")

# ---------------------------------------------------------------------------
# 图1：板块图结构 + hub 高亮
# ---------------------------------------------------------------------------
angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
pos = {i: (np.cos(a), np.sin(a)) for i, a in enumerate(angles)}
fig, ax = plt.subplots(figsize=(7.2, 7.2), dpi=130)
for (j, i) in edges:
    if i == j:
        continue
    x0, y0 = pos[j]; x1, y1 = pos[i]
    w = true_flow.get((j, i), 0.1)
    arr = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                          mutation_scale=12, lw=0.6 + 3 * w,
                          color=C["hub"] if j == 0 else C["edge"],
                          alpha=0.7, shrinkA=16, shrinkB=16,
                          connectionstyle="arc3,rad=0.08")
    ax.add_patch(arr)
for i in range(N):
    x, y = pos[i]
    is_hub = (i == 0)
    _circ_col = C["hub"] if is_hub else C["leaf"]
    circ = Circle((x, y), 0.14, color=_circ_col,
                  ec="black", lw=1.2, zorder=5)
    ax.add_patch(circ)
    ax.text(x, y, SECTORS[i], ha="center", va="center", fontsize=8.5,
            weight="bold", color="white", zorder=6)
ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4)
ax.set_aspect("equal"); ax.axis("off")
ax.set_title("板块信息流图：半导体是 hub，箭头粗细=真实传导强度",
             fontsize=12.5, weight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "sector_graph.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图2：GAT 学到的注意力矩阵热图 vs 真实传导矩阵
# ---------------------------------------------------------------------------
att = results["gat"]["att"]
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), dpi=130)
for ax, M, title in [(axes[0], BETA, "真实信息传导强度 β"),
                     (axes[1], att, "GAT 学到的注意力权重 α")]:
    im = ax.imshow(M, cmap="Reds", aspect="auto")
    ax.set_xticks(range(N)); ax.set_yticks(range(N))
    ax.set_xticklabels(SECTORS, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(SECTORS, fontsize=8)
    ax.set_xlabel("信息来源板块 j"); ax.set_ylabel("接收板块 i")
    ax.set_title(title, fontsize=12, weight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046)
plt.tight_layout()
plt.savefig(os.path.join(D, "attention_heatmap.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图3：训练损失曲线 GAT vs GCN vs MLP
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 4.6), dpi=130)
ax.plot(results["gat"]["losses"], color=C["gat"], lw=2.2, label=f"GAT (IC={results['gat']['ic']:.3f})")
ax.plot(results["gcn"]["losses"], color=C["gcn"], lw=2.0, label=f"GCN (IC={results['gcn']['ic']:.3f})")
ax.plot(results["mlp"]["losses"], color=C["mlp"], lw=2.0, ls="--", label=f"MLP 无图 (IC={results['mlp']['ic']:.3f})")
ax.set_xlabel("训练轮次")
ax.set_ylabel("训练 MSE")
ax.set_title("加入图注意力后收敛更低：信息流被正确分配", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3)
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "training_loss.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 图4：hub 节点(半导体)对下游注意力分配 vs 均匀平均(GCN)
# ---------------------------------------------------------------------------
downstream = [1, 2, 3]     # 消费电子/新能源车/光伏
gat_hub_att = [att[d, 0] for d in downstream]        # 下游看半导体的注意力
gcn_hub_w = [A_norm[d, 0] for d in downstream]       # GCN 固定权重
true_b = [BETA[d, 0] for d in downstream]

x = np.arange(len(downstream)); w = 0.26
fig, ax = plt.subplots(figsize=(8, 4.8), dpi=130)
ax.bar(x - w, true_b, w, label="真实传导 β", color=C["leaf"], alpha=0.85)
ax.bar(x, gat_hub_att, w, label="GAT 注意力 α", color=C["gat"], alpha=0.85)
ax.bar(x + w, gcn_hub_w, w, label="GCN 固定权重", color=C["gcn"], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels([SECTORS[d] for d in downstream])
ax.set_ylabel("对『半导体』的信息权重")
ax.set_title("GAT 让权重随真实强度起伏，GCN 一视同仁", fontsize=12.5, weight="bold")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "hub_attention_bar.png"), bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# 导出关键数字
# ---------------------------------------------------------------------------
# 注意力集中度：hub 行/列的注意力占比
hub_out = float(att[:, 0].sum())               # 大家分给半导体的总注意力
att_entropy = float(-np.nansum(att * np.log(att + 1e-12)) / N)
stats = {
    "N_sectors": N, "F": F, "n_edges": len([e for e in edges if e[0] != e[1]]),
    "T": T,
    "ic": {k: round(results[k]["ic"], 4) for k in results},
    "rmse": {k: round(results[k]["rmse"], 4) for k in results},
    "gat_vs_gcn_ic_gain": round(results["gat"]["ic"] - results["gcn"]["ic"], 4),
    "gat_vs_mlp_ic_gain": round(results["gat"]["ic"] - results["mlp"]["ic"], 4),
    "gat_loss_end": round(results["gat"]["losses"][-1], 4),
    "gcn_loss_end": round(results["gcn"]["losses"][-1], 4),
    "mlp_loss_end": round(results["mlp"]["losses"][-1], 4),
    "hub_downstream": {SECTORS[d]: {"true_beta": round(float(BETA[d, 0]), 3),
                                     "gat_alpha": round(float(att[d, 0]), 3),
                                     "gcn_weight": round(float(A_norm[d, 0]), 3)}
                        for d in downstream},
    "att_hub_column_sum": round(hub_out, 4),
    "att_entropy": round(att_entropy, 4),
}
with open(os.path.join(D, "_stats.json"), "w") as f:
    json.dump(stats, f, ensure_ascii=False, indent=2)
print(json.dumps(stats, ensure_ascii=False, indent=2))
print("\n图片已保存到:", D)
