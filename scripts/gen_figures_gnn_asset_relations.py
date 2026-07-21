# -*- coding: utf-8 -*-
"""图神经网络 GNN 资产关系建模 —— 生成配图 + 打印统计(自洽合成, 纯 numpy)"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti SC", "PingFang SC", "STHeiti", "SimHei", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 110
plt.rcParams["savefig.dpi"] = 110

OUT = "public/images/gnn-asset-relations"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(20260722)

# ---------------- 合成资产关系图 ----------------
# N 只股票, 3 个行业(0,1,2). 同行业股票的收益由共同的行业因子驱动 -> 图上应连边.
N = 60
G = 3
sector = np.repeat(np.arange(G), N // G)          # 每股所属行业
T = 500                                            # 交易日

# 行业因子 + 个股特质
sector_factor = rng.normal(0, 1.0, size=(G, T)) * 0.012
idio = rng.normal(0, 1.0, size=(N, T)) * 0.010
beta = 0.6 + 0.5 * rng.random(N)                   # 各股对行业因子的暴露
ret = beta[:, None] * sector_factor[sector] + idio  # 日收益 (N,T)

# 邻接矩阵: 用滚动相关阈值化(真实里常用行业/供应链/相关图)
C = np.corrcoef(ret)
np.fill_diagonal(C, 0.0)
thr = 0.30
A = (C > thr).astype(float)                         # 无权邻接
deg = A.sum(1)
print(f"节点 {N} | 边(单向) {int(A.sum())} | 平均度 {deg.mean():.1f}")

# 图内/图外相关对比(边多来自同行业)
same = (sector[:, None] == sector[None, :])
np.fill_diagonal(same, False)
edge_same_ratio = A[same].sum() / max(A.sum(), 1)
print(f"边落在同行业内的比例: {edge_same_ratio:.2%} (随机基线≈{ (same.sum()/(N*N-N)):.2%} )")

# ---------------- 极简 GCN 前向(一层图卷积) 做行业分类 ----------------
# 归一化邻接 D^-1/2 (A+I) D^-1/2
Ah = A + np.eye(N)
d = Ah.sum(1)
Dinv = np.diag(1.0 / np.sqrt(d))
Ahat = Dinv @ Ah @ Dinv

# 节点特征: 每股最近 20 日收益的 8 个滚动统计(均值/波动/偏度近似等) 简化为随机投影特征
feat_raw = np.stack([
    ret[:, -20:].mean(1), ret[:, -20:].std(1),
    ret[:, -60:].mean(1), ret[:, -60:].std(1),
    beta, (ret > 0).mean(1),
    np.abs(ret).mean(1), C.mean(1)
], axis=1)
feat = (feat_raw - feat_raw.mean(0)) / (feat_raw.std(0) + 1e-9)
F = feat.shape[1]

# 训练一层 GCN: H = softmax(Ahat @ X @ W), 交叉熵, 半监督(每行业 4 个标注)
W = rng.normal(0, 0.3, size=(F, G))
labels = sector
train_mask = np.zeros(N, bool)
for g in range(G):
    idx = np.where(sector == g)[0]
    train_mask[rng.choice(idx, 4, replace=False)] = True

def softmax(Z):
    Z = Z - Z.max(1, keepdims=True)
    e = np.exp(Z)
    return e / e.sum(1, keepdims=True)

XW_input = Ahat @ feat
lr = 0.5
loss_hist, acc_hist = [], []
for it in range(200):
    logits = XW_input @ W
    P = softmax(logits)
    Y = np.eye(G)[labels]
    # 只在训练节点上算梯度
    grad_logits = (P - Y) * train_mask[:, None] / train_mask.sum()
    gW = XW_input.T @ grad_logits
    W -= lr * gW
    loss = -np.log(P[train_mask, labels[train_mask]] + 1e-9).mean()
    pred = P.argmax(1)
    test_acc = (pred[~train_mask] == labels[~train_mask]).mean()
    loss_hist.append(loss); acc_hist.append(test_acc)

# 对照: 不用图, 只用节点特征做同样的线性 softmax
W2 = rng.normal(0, 0.3, size=(F, G))
loss2_hist, acc2_hist = [], []
for it in range(200):
    logits = feat @ W2
    P = softmax(logits)
    Y = np.eye(G)[labels]
    grad_logits = (P - Y) * train_mask[:, None] / train_mask.sum()
    gW = feat.T @ grad_logits
    W2 -= lr * gW
    loss2 = -np.log(P[train_mask, labels[train_mask]] + 1e-9).mean()
    pred = P.argmax(1)
    acc2 = (pred[~train_mask] == labels[~train_mask]).mean()
    loss2_hist.append(loss2); acc2_hist.append(acc2)

print(f"GCN(用图) 测试集行业分类准确率: {acc_hist[-1]:.3f}")
print(f"MLP(不用图) 测试集行业分类准确率: {acc2_hist[-1]:.3f}")

# ================= 图1 cover: 资产关系图可视化 =================
fig, ax = plt.subplots(figsize=(9, 6.2))
# 环形布局, 按行业分块着色
ang = np.linspace(0, 2*np.pi, N, endpoint=False)
# 让同行业聚在一起
order = np.argsort(sector)
pos = np.zeros((N, 2))
for rank, i in enumerate(order):
    a = 2*np.pi * rank / N
    pos[i] = [np.cos(a), np.sin(a)]
colors = ["#e74c3c", "#3498db", "#2ecc71"]
# 边
for i in range(N):
    for j in range(i+1, N):
        if A[i, j] > 0:
            c = "#95a5a6" if sector[i] == sector[j] else "#f39c12"
            ax.plot([pos[i,0], pos[j,0]], [pos[i,1], pos[j,1]], color=c,
                    lw=0.5, alpha=0.35, zorder=1)
for g in range(G):
    m = sector == g
    ax.scatter(pos[m,0], pos[m,1], s=90, c=colors[g], edgecolors="white",
               linewidths=1.2, zorder=3, label=f"行业 {g+1}")
ax.set_title("资产关系图：同行业节点由相关阈值连边（灰=同行业边，橙=跨行业边）", fontsize=12)
ax.legend(loc="upper right", framealpha=0.9)
ax.set_aspect("equal"); ax.axis("off")
plt.tight_layout()
plt.savefig(f"{OUT}/cover.png", bbox_inches="tight")
plt.close()

# ================= 图2: GCN vs MLP 收敛曲线 =================
fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
ax[0].plot(loss_hist, label="GCN(含图卷积)", color="#e74c3c", lw=2)
ax[0].plot(loss2_hist, label="MLP(仅节点特征)", color="#3498db", lw=2, ls="--")
ax[0].set_title("训练交叉熵损失"); ax[0].set_xlabel("迭代"); ax[0].set_ylabel("loss")
ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(acc_hist, label="GCN(含图卷积)", color="#e74c3c", lw=2)
ax[1].plot(acc2_hist, label="MLP(仅节点特征)", color="#3498db", lw=2, ls="--")
ax[1].axhline(1/G, color="gray", ls=":", label="随机基线 1/3")
ax[1].set_title(f"测试集行业分类准确率（终值 GCN={acc_hist[-1]:.2f} vs MLP={acc2_hist[-1]:.2f}）")
ax[1].set_xlabel("迭代"); ax[1].set_ylabel("accuracy")
ax[1].legend(); ax[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/gcn_vs_mlp.png", bbox_inches="tight")
plt.close()

# ================= 图3: 邻接矩阵热力图(块对角结构) =================
fig, ax = plt.subplots(figsize=(6.6, 5.6))
A_ord = A[np.ix_(order, order)]
im = ax.imshow(A_ord, cmap="Greys", aspect="auto")
# 行业分界线
b = N // G
for k in range(1, G):
    ax.axhline(k*b-0.5, color="#e74c3c", lw=1.2)
    ax.axvline(k*b-0.5, color="#e74c3c", lw=1.2)
ax.set_title(f"邻接矩阵（按行业排序后呈块对角）\n同行业边占比 {edge_same_ratio:.0%}")
ax.set_xlabel("节点(按行业排序)"); ax.set_ylabel("节点(按行业排序)")
plt.colorbar(im, ax=ax, fraction=0.046, label="连边(0/1)")
plt.tight_layout()
plt.savefig(f"{OUT}/adjacency_block.png", bbox_inches="tight")
plt.close()

print("图已生成:", os.listdir(OUT))
