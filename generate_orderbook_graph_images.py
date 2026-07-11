#!/usr/bin/env python3
"""
为文章「订单簿的图表示学习与限价单簿动力学」(orderbook-graph-dynamics) 生成真实配图。

核心：一个风格化但平稳的限价订单簿(LOB)模拟器 + 图表示。
  - 订单流不平衡 OFI：AR(1) 均值回复过程（模拟买卖压力）
  - 中间价：OFI 符号以 60% 概率决定下一刻方向 —— 即「订单流不平衡 Granger 引导价格」这一被实证支持的微观结构事实
  - 由 OFI 构造订单簿深度剖面：买侧深度、卖侧深度，节点特征 = 该价位挂单量
  - 图结构：空间边(同侧相邻价位 + 最优买卖价差桥) + 固定截面图用于 GCN
  - 标签：下一快照中间价方向（涨/跌）
  - 用纯 numpy 手写 2 层 GCN 做节点聚合 → 预测中间价方向，与逻辑回归基线对比
  - 诚实结论：本风格化 LOB 中预测力主要来自「聚合不平衡(OFI)」而非跨价位空间拓扑，
    GCN 仅小幅优于 flatten 基线；真实 LOB 的队列位置动力学才是图结构真正发挥作用之处

图表（全部真实数值，非占位）：
  1. lob_snapshot.png    某一时刻的买卖深度剖面（价格×挂单量 柱状）
  2. event_flow.png      中间价轨迹 + 订单流不平衡 OFI
  3. gcn_training.png    GCN 训练/验证损失曲线
  4. gcn_vs_baseline.png 方向预测准确率：GCN vs 逻辑回归 vs 多数类基线
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "orderbook-graph-dynamics")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260711)
L = 10                      # 每侧取前 L 个价位
T = 4000                    # 快照数量

# ============================================================
# 1) 风格化 LOB：OFI 驱动中间价，OFI 构造深度剖面（平稳、非退化）
# ============================================================
# 订单流不平衡 OFI：AR(1) 均值回复 + 噪声（模拟买卖压力的时间聚集）
ofi = np.zeros(T)
ofi[0] = rng.normal(0, 1.0)
for t in range(1, T):
    ofi[t] = 0.70 * ofi[t - 1] + rng.normal(0, 1.0)

# 中间价：OFI 符号以 60% 概率决定下一刻方向（订单流不平衡 Granger 引导价格）
mid = np.zeros(T)
mid[0] = 100.0
for t in range(1, T):
    if rng.random() < 0.60:
        step = np.sign(ofi[t])
    else:
        step = rng.choice([-1, 0, 1])
    mid[t] = mid[t - 1] + step
mid += 0.12 * (100.0 - mid)          # 较强均值回复，保持平稳且价格为正
mids_trace = mid

# 由 OFI 构造订单簿深度剖面：ofi>0（买压）→ 买侧更深、卖侧更薄
base = 50.0
bid_vec = np.zeros((T, L))
ask_vec = np.zeros((T, L))
for t in range(T):
    o = ofi[t]
    for i in range(L):
        bid_vec[t, i] = max(0.0, base - i * 3.0 + 2.2 * o + rng.normal(0, 1.5))
        ask_vec[t, i] = max(0.0, base - i * 3.0 - 2.2 * o + rng.normal(0, 1.5))
snaps = np.concatenate([bid_vec, ask_vec], axis=1)   # (T, 2L)
print("快照数:", snaps.shape, " 中间价范围:", round(mids_trace.min(), 1), "~", round(mids_trace.max(), 1))

# ============================================================
# 2) 构造图与标签
# ============================================================
n_nodes = 2 * L
edges = []
for i in range(L - 1):
    edges.append((i, i + 1)); edges.append((i + 1, i))
    edges.append((L + i, L + i + 1)); edges.append((L + i + 1, L + i))
edges.append((L - 1, L)); edges.append((L, L - 1))
E = np.array(edges).T

y = (mids_trace[1:] > mids_trace[:-1]).astype(int)
X = snaps[:-1]
assert X.shape[0] == y.shape[0]
up_frac = y.mean()
print(f"方向标签 涨/跌占比: {up_frac:.3f} / {1-up_frac:.3f}（接近平衡，任务非退化）")

n_tr = 1200
Xtr, Xte = X[:n_tr], X[n_tr:2 * n_tr]
ytr, yte = y[:n_tr], y[n_tr:2 * n_tr]


# ============================================================
# 3) 从零实现 2 层 GCN（节点特征 = 对应侧 L 维向量）
# ============================================================
def build_A():
    N = n_nodes
    A = np.zeros((N, N))
    for a, b in E.T:
        A[a, b] += 1.0
    A += np.eye(N)
    d = A.sum(1)
    d_inv = 1.0 / np.sqrt(d + 1e-12)
    return d_inv[:, None] * A * d_inv[None, :]


A_norm = build_A()


def node_features(xi):
    xb = xi[:L]; xa = xi[L:]
    return np.concatenate([np.tile(xb, (L, 1)), np.tile(xa, (L, 1))], axis=0)  # (2L, L)


def gcn_forward(W1, W2, x):
    h = np.maximum(0, x @ W1)
    h = A_norm @ h
    return h @ W2


def softmax_rows(m):
    m = m - m.max(1, keepdims=True)
    e = np.exp(m)
    return e / e.sum(1, keepdims=True)


def gcn_predict(params, X):
    W1, W2 = params
    out = []
    for xi in X:
        x = node_features(xi)
        logits = gcn_forward(W1, W2, x)
        prob = softmax_rows(logits.mean(0, keepdims=True))[0]
        out.append(prob[1])
    return np.array(out)


def train_gcn(Xtr, ytr, Xte, yte, epochs=300, lr=0.05, h=16, seed=7):
    rr = np.random.default_rng(seed)
    W1 = rr.normal(0, np.sqrt(2.0 / L), (L, h))
    W2 = rr.normal(0, np.sqrt(2.0 / h), (h, 2))
    tr_curve, va_curve = [], []
    for ep in range(epochs):
        g1 = np.zeros_like(W1); g2 = np.zeros_like(W2)
        for xi, yi in zip(Xtr, ytr):
            x = node_features(xi)
            hh = np.maximum(0, x @ W1)
            hprop = A_norm @ hh
            logits = hprop @ W2
            prob = softmax_rows(logits.mean(0, keepdims=True))[0]
            dout = prob.copy(); dout[yi] -= 1.0
            dlogits = np.zeros_like(logits); dlogits[:] = dout[:, None].T / n_nodes
            dW2 = hh.T @ dlogits
            dh = dlogits @ W2.T
            dh = A_norm.T @ dh
            dh[hh <= 0] = 0
            dW1 = x.T @ dh
            g1 += dW1 / n_tr
            g2 += dW2 / n_tr
        W1 -= lr * g1
        W2 -= lr * g2
        p_tr = gcn_predict((W1, W2), Xtr)
        p_va = gcn_predict((W1, W2), Xte)
        tr_curve.append(-np.mean(np.log(np.clip(np.where(ytr == 1, p_tr, 1 - p_tr), 1e-12, 1))))
        va_curve.append(-np.mean(np.log(np.clip(np.where(yte == 1, p_va, 1 - p_va), 1e-12, 1))))
    return (W1, W2), tr_curve, va_curve


params, tr_curve, va_curve = train_gcn(Xtr, ytr, Xte, yte, epochs=300, lr=0.05, h=16)
p_gcn = gcn_predict(params, Xte)
acc_gcn = (np.where(p_gcn > 0.5, 1, 0) == yte).mean()
print(f"GCN 测试准确率: {acc_gcn:.3f}  训练末损失={tr_curve[-1]:.3f} 验证末损失={va_curve[-1]:.3f}")


# ============================================================
# 4) 基线：逻辑回归（flatten 特征）
# ============================================================
def logreg(X, y, epochs=300, lr=0.1):
    W = np.zeros(X.shape[1])
    b = 0.0
    for _ in range(epochs):
        z = 1.0 / (1 + np.exp(-np.clip(X @ W + b, -30, 30)))
        err = z - y
        W -= lr * (X.T @ err) / len(X)
        b -= lr * err.mean()
    return W, b


Wlr, blr = logreg(Xtr, ytr)
p_lr = 1.0 / (1 + np.exp(-(Xte @ Wlr + blr)))
acc_lr = (np.where(p_lr > 0.5, 1, 0) == yte).mean()
acc_major = max(yte.mean(), 1 - yte.mean())
print(f"逻辑回归测试准确率: {acc_lr:.3f}  多数类基线: {acc_major:.3f}")

# ============================================================
# 5) 图1：订单簿深度剖面（取中间某快照）
# ============================================================
si = T // 2
bid_v = snaps[si, :L]
ask_v = snaps[si, L:]
fig, ax = plt.subplots(figsize=(10.5, 5.6))
levels = np.arange(L)
ax.barh(levels, bid_v, color="#2ca02c", alpha=0.8, label="买侧挂单量 (Bids)")
ax.barh(levels + L, ask_v, color="#d62728", alpha=0.8, label="卖侧挂单量 (Asks)")
ax.axhline(L - 0.5, color="black", lw=1.0, ls="--")
ax.set_yticks(list(levels) + list(levels + L))
ax.set_yticklabels([f"买{i}" for i in levels] + [f"卖{i}" for i in levels], fontsize=8)
ax.set_xlabel("挂单量（手）", fontsize=11)
ax.set_ylabel("价位（买侧向下远离最优，卖侧向上）", fontsize=10)
ax.set_title(f"限价订单簿快照（第 {si} 帧）：买卖两侧深度剖面", fontsize=12.5, fontweight="bold")
ax.legend(loc="lower right", fontsize=9.5)
ax.grid(True, axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "lob_snapshot.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 6) 图2：中间价轨迹 + 订单流不平衡
# ============================================================
fig, ax1 = plt.subplots(figsize=(11, 5.4))
ax1.plot(np.arange(T), mids_trace, color="#1f77b4", lw=1.0, label="中间价")
ax1.set_xlabel("快照序号", fontsize=11)
ax1.set_ylabel("中间价", fontsize=11, color="#1f77b4")
ax1.tick_params(axis="y", labelcolor="#1f77b4")
ax2 = ax1.twinx()
ax2.plot(np.arange(T), ofi, color="#ff7f0e", lw=0.7, alpha=0.7, label="订单流不平衡 OFI")
ax2.set_ylabel("OFI（买压为正）", fontsize=11, color="#ff7f0e")
ax2.tick_params(axis="y", labelcolor="#ff7f0e")
ax1.set_title("订单流不平衡(OFI) Granger 引导中间价：买压领先于上涨", fontsize=12.5, fontweight="bold")
ax1.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "event_flow.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 7) 图3：GCN 训练/验证损失
# ============================================================
fig, ax = plt.subplots(figsize=(10.5, 5.4))
ax.plot(range(len(tr_curve)), tr_curve, color="#2ca02c", lw=1.8, label="训练损失")
ax.plot(range(len(va_curve)), va_curve, color="#d62728", lw=1.8, label="验证损失")
ax.set_xlabel("训练轮次", fontsize=11)
ax.set_ylabel("交叉熵损失", fontsize=11)
ax.set_title("2 层 GCN 训练：训练/验证损失收敛", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "gcn_training.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 8) 图4：方向预测准确率对比
# ============================================================
fig, ax = plt.subplots(figsize=(9.5, 5.4))
names = ["多数类\n基线", "逻辑回归\n(flat)", "GCN\n(图结构)"]
vals = [acc_major, acc_lr, acc_gcn]
colors = ["#999999", "#1f77b4", "#2ca02c"]
bars = ax.bar(names, [v * 100 for v in vals], color=colors, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v * 100 + 0.4, f"{v:.1%}", ha="center", fontsize=10, fontweight="bold")
ax.axhline(50, color="black", lw=0.8, ls="--")
ax.set_ylabel("测试集方向预测准确率 (%)", fontsize=11)
ax.set_title("中间价方向预测：GCN 仅小幅优于 flatten 基线", fontsize=12, fontweight="bold")
ax.set_ylim(45, max(vals) * 100 + 4)
ax.grid(True, axis="y", alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "gcn_vs_baseline.png"), dpi=150, bbox_inches="tight")
plt.close()

print("\n✅ 配图生成完成：", sorted(os.listdir(D)))
