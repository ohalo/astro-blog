#!/usr/bin/env python3
"""
为文章「联邦学习：在隐私约束下联合训练跨机构量化模型」(federated-learning-quant)
生成真实配图。

核心逻辑：
  - 构造 K 家机构（客户端），各自持有不同市场体制下的股票涨跌标签数据（非独立同分布 / non-IID）；
  - 用 numpy 实现 FedAvg：各家本地 SGD 若干轮 → 按样本量加权平均参数 → 重复；
  - 与「数据池化集中训练」「纯本地不共享」对比；并扫描 non-IID 程度与差分隐私噪声；
  - 全部可复现，非占位图。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "federated-learning-quant")
os.makedirs(D, exist_ok=True)

d = 8            # 特征维度
K = 6            # 机构数
n_per = 400      # 每家样本数
label_noise = 0.12   # 机构内部标注噪声（让本地最优偏离全局最优）

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

def bce(w, b, X, y):
    p = sigmoid(X @ w + b)
    return -np.mean(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12))

def predict(w, b, X):
    return (sigmoid(X @ w + b) > 0.5).astype(int)

def acc(w, b, X, y):
    return np.mean(predict(w, b, X) == y)

def make_client(rng, w_global, drift, nc, mean_shift):
    """一家机构的数据：本地概念 = 全局概念 + drift 扰动（non-IID）。
    偏置与特征分布也随机构变化，模拟不同市场体制。"""
    u = rng.standard_normal(d); u /= np.linalg.norm(u) + 1e-9
    w_c = w_global + drift * u
    b_c = rng.uniform(-1.5, 1.5) * (0.5 + drift)   # drift 越大，偏置越散
    X = rng.standard_normal((nc, d)) + mean_shift * (1 + drift)
    p = sigmoid(X @ w_c + b_c)
    y = (p > 0.5).astype(int)
    flip = rng.random(nc) < label_noise
    y[flip] = 1 - y[flip]
    return X, y, w_c, b_c

def local_train(w, b, X, y, epochs, lr):
    w = w.copy(); b = b
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        err = p - y
        gw = X.T @ err / len(y)
        gb = err.mean()
        w -= lr * gw
        b -= lr * gb
    return w, b

def fedavg(global_w, global_b, clients, rounds, local_epochs, lr):
    w, b = global_w.copy(), global_b
    ws, bs = [], []
    for _ in range(rounds):
        new_w = np.zeros(d); new_b = 0.0; tot = 0
        for (X, y, _, _) in clients:
            wc, bc = local_train(w, b, X, y, local_epochs, lr)
            m = len(y); tot += m
            new_w += m * wc; new_b += m * bc
        w = new_w / tot; b = new_b / tot
        ws.append(w.copy()); bs.append(b)
    return ws, bs

# ---------- 数据：全局概念 + 各机构 drift（non-IID） ----------
rng = np.random.default_rng(2024)
w_global = rng.standard_normal(d)
X_test = rng.standard_normal((800, d))
y_test = (sigmoid(X_test @ w_global + 0.1) > 0.5).astype(int)

def build_clients(drift):
    clients = []
    for c in range(K):
        ms = rng.standard_normal(d) * 0.6
        X, y, _, _ = make_client(rng, w_global, drift, n_per, ms)
        clients.append((X, y, None, None))
    return clients

# ---------- 图 1：收敛曲线（FedAvg vs 集中式 vs 纯本地） ----------
drift0 = 1.0
clients = build_clients(drift0)
w0 = np.zeros(d); b0 = 0.0

# 集中式（池化）
Xall = np.vstack([c[0] for c in clients]); yall = np.concatenate([c[1] for c in clients])
wc, bc = w0.copy(), b0
cent_acc = []
for t in range(40):
    wc, bc = local_train(wc, bc, Xall, yall, 1, 0.1)
    cent_acc.append(acc(wc, bc, X_test, y_test))
cent_final = cent_acc[-1]

# FedAvg（每轮本地 2 epoch，共 20 轮）
fed_ws, fed_bs = fedavg(w0, b0, clients, 20, 2, 0.1)
fed_acc = [acc(w, b, X_test, y_test) for w, b in zip(fed_ws, fed_bs)]

# 纯本地（不共享，各家各自训练后取平均预测）
local_preds = []
for (X, y, _, _) in clients:
    wl, bl = local_train(w0.copy(), b0, X, y, 40, 0.1)
    local_preds.append(predict(wl, bl, X_test))
local_acc = np.mean(np.array(local_preds) == y_test[None, :]) if False else \
            np.mean((np.mean(local_preds, 0) > 0.5) == y_test)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, len(cent_acc) + 1), cent_acc, "-", color="#2ca02c", lw=2.2, label=f"集中式(数据池化) 终={cent_final:.3f}")
xs = list(range(1, len(fed_acc) + 1))
ax.plot(xs, fed_acc, "-o", color="#1f77b4", lw=2, ms=5, label=f"FedAvg(联邦平均) 终={fed_acc[-1]:.3f}")
ax.axhline(local_acc, ls="--", color="#d62728", lw=1.8, label=f"纯本地(不共享) 终={local_acc:.3f}")
ax.set_xlabel("通信轮次 / 训练轮次", fontsize=12)
ax.set_ylabel("全局测试准确率", fontsize=12)
ax.set_title("FedAvg 收敛贴近集中式，却无需任何机构交出原始数据", fontsize=12.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_fedavg.png"), dpi=130); plt.close(fig)

# ---------- 图 2：non-IID 程度对精度的影响 ----------
drifts = [0.5, 1.0, 1.5, 2.5, 4.0]
fed_curve, cent_curve = [], []
for dr in drifts:
    cl = build_clients(dr)
    ws, bs = fedavg(w0, b0, cl, 20, 2, 0.1)
    fed_curve.append(acc(ws[-1], bs[-1], X_test, y_test))
    # 集中式参考
    Xa = np.vstack([c[0] for c in cl]); ya = np.concatenate([c[1] for c in cl])
    wcc, bcc = w0.copy(), b0
    for _ in range(40):
        wcc, bcc = local_train(wcc, bcc, Xa, ya, 1, 0.1)
    cent_curve.append(acc(wcc, bcc, X_test, y_test))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(drifts, cent_curve, "-s", color="#2ca02c", lw=2, label="集中式(上界)")
ax.plot(drifts, fed_curve, "-o", color="#1f77b4", lw=2, label="FedAvg")
ax.fill_between(drifts, fed_curve, cent_curve, color="#1f77b4", alpha=0.08, label="FedAvg 与集中式差距")
ax.set_xlabel("机构间数据异质度 (drift，越大越 non-IID)", fontsize=11.5)
ax.set_ylabel("全局测试准确率", fontsize=12)
ax.set_title("non-IID 越严重，FedAvg 越落后于集中式", fontsize=13)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_noniid.png"), dpi=130); plt.close(fig)

# ---------- 图 3：差分隐私(DP)的隐私-效用权衡 ----------
clients = build_clients(1.0)
ws, bs = fedavg(w0, b0, clients, 20, 2, 0.1)
w_clean = ws[-1]; b_clean = bs[-1]
clean_acc = acc(w_clean, b_clean, X_test, y_test)

sigmas = np.linspace(0.0, 0.6, 13)
dp_acc = []
for s in sigmas:
    # 对聚合后的参数加高斯噪声（差分隐私的简化：后处理加噪）
    noise = rng.standard_normal(d) * s
    wn = w_clean + noise
    dp_acc.append(acc(wn, b_clean, X_test, y_test))
dp_acc = np.array(dp_acc)
# 粗略 epsilon 估算：epsilon ≈ (Δ·√(2ln(1.25/δ)))/σ
delta = 1e-5
eps = (np.sqrt(2 * np.log(1.25 / delta))) / (sigmas + 1e-9)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(sigmas, dp_acc, "-o", color="#1f77b4", lw=2, ms=5, label="加噪后测试准确率")
ax.axhline(clean_acc, ls="--", color="#2ca02c", lw=1.6, label=f"无噪声基线 ({clean_acc:.3f})")
ax.set_xlabel("差分隐私噪声乘子 σ (越大越隐私)", fontsize=11.5)
ax.set_ylabel("全局测试准确率", fontsize=12)
ax2 = ax.twinx()
ax2.plot(sigmas, eps, "-", color="#d62728", lw=1.4, alpha=0.7, label="对应 ε (越小越隐私)")
ax2.set_ylabel("近似 ε (差分隐私预算)", fontsize=11, color="#d62728")
ax2.tick_params(axis="y", labelcolor="#d62728")
ax.set_title("差分隐私的隐私-效用权衡：噪声越大，精度越低", fontsize=12.5)
ax.legend(fontsize=8.5, loc="upper right"); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "fig_dp.png"), dpi=130); plt.close(fig)

# 保存数值
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    f.write(f"K={K} d={d} n_per={n_per}\n")
    f.write(f"central_final={cent_final:.4f}\n")
    f.write(f"fedavg_final={fed_acc[-1]:.4f}\n")
    f.write(f"local_final={local_acc:.4f}\n")
    f.write("drifts," + ",".join(map(str, drifts)) + "\n")
    f.write("fed," + ",".join(f"{x:.4f}" for x in fed_curve) + "\n")
    f.write("cent," + ",".join(f"{x:.4f}" for x in cent_curve) + "\n")
    f.write(f"clean_acc={clean_acc:.4f}\n")
    f.write("dp_acc," + ",".join(f"{x:.4f}" for x in dp_acc) + "\n")

print("✅ federated-learning 配图生成完成")
print(f"central={cent_final:.3f} fedavg={fed_acc[-1]:.3f} local={local_acc:.3f}")
