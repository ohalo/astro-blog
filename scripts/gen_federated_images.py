#!/usr/bin/env python3
"""为文章「联邦学习因子共享：在不交换原始数据的前提下联合训练选股模型」
(federated-learning-factor) 生成真实配图（matplotlib，non-placeholder）。

全部数字来自真实 numpy 计算，可复现（seed=20260828）。

设计：
  - d=8 维选股特征，K=6 家机构（客户端），各自持有 non-IID 数据：
    每家机构的概念权重 w_c = w_global + drift * u，u 为单位随机方向（概念漂移）。
  - 三种训练范式对照：集中池化(pooled) / 纯本地不共享(local-only) / FedAvg。
  - 扫描 non-IID 程度(drift) 与差分隐私噪声(σ)。
  - 头部数字与 non-IID 扫描在 drift=0.6 处使用同一批客户端，保证叙述一致。
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
D = os.path.join(BASE, "federated-learning-factor")
os.makedirs(D, exist_ok=True)

C_POOL = "#1f4e79"   # 集中池化
C_FED = "#27ae60"    # FedAvg
C_LOC = "#c0392b"    # 本地不共享
C_DP = "#8e44ad"     # 差分隐私
GRID = "#e6e6e6"

d = 8
K = 6
n_per = 500
w_global = np.array([0.45, -0.30, 0.55, -0.20, 0.35, -0.40, 0.25, -0.15])
w_global = w_global / np.linalg.norm(w_global)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def acc(w, b, X, y):
    return np.mean((sigmoid(X @ w + b) > 0.5).astype(int) == y)


def make_client(rng, drift, label_noise):
    u = rng.standard_normal(d); u /= np.linalg.norm(u) + 1e-9
    w_c = w_global + drift * u
    b_c = rng.uniform(-1.0, 1.0)
    X = rng.standard_normal((n_per, d))
    p = sigmoid(X @ w_c + b_c)
    y = (p > 0.5).astype(int)
    flip = rng.random(n_per) < label_noise
    y[flip] = 1 - y[flip]
    return X, y, w_c, b_c


def make_clients(drift, base_seed):
    r = np.random.default_rng(base_seed)
    return [make_client(r, drift, 0.10) for _ in range(K)]


def local_train(w, b, X, y, epochs, lr):
    w = w.copy()
    for _ in range(epochs):
        p = sigmoid(X @ w + b)
        g = p - y
        gw = X.T @ g / max(len(y), 1)
        gb = np.mean(g)
        w -= lr * gw
        b -= lr * gb
    return w, b


def fedavg(clients, R, epochs, lr, noise_sigma=0.0, clip=1.0, dp_scale=0.08):
    w = np.zeros(d); b = 0.0
    ns = np.array([len(y) for _, y, _, _ in clients], float)
    rng = np.random.default_rng(777)
    for _ in range(R):
        ws, bs = [], []
        for X, y, _, _ in clients:
            wk, bk = local_train(w, b, X, y, epochs, lr)
            nrm = np.linalg.norm(wk)
            if nrm > clip:
                wk = wk * (clip / nrm)
            ws.append(wk); bs.append(bk)
        w = np.sum(ns[:, None] * np.array(ws), axis=0) / ns.sum()
        b = np.sum(ns * np.array(bs)) / ns.sum()
        if noise_sigma > 0:
            w = w + rng.normal(0, noise_sigma * dp_scale, d)
            b = b + rng.normal(0, noise_sigma * dp_scale)
    return w, b


# ---------- 头部客户端（drift=0.6, seed=1294）----------
clients_main = make_clients(0.6, 1294)
rng_t = np.random.default_rng(20260828)
X_test = rng_t.standard_normal((2000, d))
y_test = (sigmoid(X_test @ w_global) > 0.5).astype(int)

# 集中池化
X_all = np.vstack([c[0] for c in clients_main])
y_all = np.concatenate([c[1] for c in clients_main])
wp, bp = local_train(np.zeros(d), 0.0, X_all, y_all, 60, 0.05)
acc_pool = acc(wp, bp, X_test, y_test)

# 纯本地
loc_accs = [acc(*local_train(np.zeros(d), 0.0, c[0], c[1], 60, 0.05), X_test, y_test) for c in clients_main]
acc_loc = float(np.mean(loc_accs))

# 收敛曲线（与头部客户端一致）
rounds = list(range(1, 21))
fed_curve = []
for R in rounds:
    wf, bf = fedavg(clients_main, R, 5, 0.05)
    fed_curve.append(acc(wf, bf, X_test, y_test))
acc_fed_final = fed_curve[-1]

print(f"[FL] HEADLINE acc_pool={acc_pool:.4f} acc_loc={acc_loc:.4f} acc_fed_final={acc_fed_final:.4f}")

# ===== 图1: 收敛曲线 =====
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(rounds, [acc_pool] * len(rounds), color=C_POOL, lw=2.0, ls="--", label=f"集中池化 (pooled) = {acc_pool:.3f}")
ax.plot(rounds, fed_curve, color=C_FED, lw=2.0, marker="o", ms=4, label=f"FedAvg = {acc_fed_final:.3f}")
ax.axhline(acc_loc, color=C_LOC, lw=2.0, ls=":", label=f"纯本地不共享 = {acc_loc:.3f}")
ax.set_title("通信轮次 vs 全局测试准确率：FedAvg 不交换原始数据逼近池化上限", fontsize=13, fontweight="bold")
ax.set_xlabel("通信轮次 (communication rounds)"); ax.set_ylabel("全局测试准确率")
ax.set_ylim(0.5, 1.02); ax.legend(loc="lower right", fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{D}/fedavg_convergence.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图2: non-IID 扫描（drift=0.6 用 clients_main，保证与头部一致）=====
drifts = [0.0, 0.3, 0.6, 1.0, 1.5]
pool_row, fed_row, loc_row = [], [], []
for k, dr in enumerate(drifts):
    if abs(dr - 0.6) < 1e-9:
        cls = clients_main
    else:
        cls = make_clients(dr, 1234 + int(round(dr * 100)))
    Xa = np.vstack([c[0] for c in cls]); ya = np.concatenate([c[1] for c in cls])
    wpp, bpp = local_train(np.zeros(d), 0.0, Xa, ya, 60, 0.05)
    pool_row.append(acc(wpp, bpp, X_test, y_test))
    wf, bf = fedavg(cls, 15, 5, 0.05)
    fed_row.append(acc(wf, bf, X_test, y_test))
    la = np.mean([acc(*local_train(np.zeros(d), 0.0, c[0], c[1], 60, 0.05), X_test, y_test) for c in cls])
    loc_row.append(la)
print(f"[FL] noniid drifts={drifts}")
print(f"[FL]   pooled={[round(x,3) for x in pool_row]}")
print(f"[FL]   fedavg={[round(x,3) for x in fed_row]}")
print(f"[FL]   local ={[round(x,3) for x in loc_row]}")

fig, ax = plt.subplots(figsize=(11, 5.6))
x = np.arange(len(drifts)); wbar = 0.26
ax.bar(x - wbar, pool_row, wbar, color=C_POOL, label="集中池化")
ax.bar(x, fed_row, wbar, color=C_FED, label="FedAvg")
ax.bar(x + wbar, loc_row, wbar, color=C_LOC, label="纯本地不共享")
ax.set_xticks(x); ax.set_xticklabels([f"{dr:.1f}" for dr in drifts])
ax.set_xlabel("non-IID 程度 (概念漂移 drift)"); ax.set_ylabel("全局测试准确率")
ax.set_title("non-IID 越严重，FedAvg 越贴近池化、越甩开纯本地", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, color=GRID, axis="y")
fig.tight_layout(); fig.savefig(f"{D}/noniid_sweep.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图3: 差分隐私噪声权衡 =====
sigmas = [0.0, 0.05, 0.1, 0.2, 0.4]
dp_row = []
for s in sigmas:
    wf, bf = fedavg(clients_main, 15, 5, 0.05, noise_sigma=s, clip=1.0, dp_scale=0.08)
    dp_row.append(acc(wf, bf, X_test, y_test))
print(f"[FL] dp sigmas={sigmas} -> acc={[round(x,3) for x in dp_row]}")
fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(sigmas, dp_row, color=C_DP, lw=2.2, marker="s", ms=6, label="FedAvg + DP 噪声")
ax.axhline(acc_fed_final, color=C_FED, ls="--", lw=1.6, label=f"无噪声 FedAvg = {acc_fed_final:.3f}")
ax.set_xlabel("差分隐私噪声尺度 σ (越大越隐私，越大越损精度)"); ax.set_ylabel("全局测试准确率")
ax.set_title("隐私-效用权衡：中等噪声代价可控，过噪才崩", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{D}/dp_tradeoff.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图4: 概念漂移热图（各机构权重 vs 全局）=====
Wc = np.array([c[2] for c in clients_main])  # (K, d)
heat = np.vstack([w_global, Wc])              # (K+1, d)
fig, ax = plt.subplots(figsize=(11, 5.0))
im = ax.imshow(heat, aspect="auto", cmap="RdBu_r", vmin=-1.2, vmax=1.2)
ax.set_yticks(range(K + 1)); ax.set_yticklabels(["全局 w*"] + [f"机构 {i+1}" for i in range(K)])
ax.set_xlabel("因子维度 (特征序号)")
ax.set_title("各机构概念权重 vs 全局权重：non-IID 来自机构间权重方向的系统性偏离", fontsize=13, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="权重分量")
fig.tight_layout(); fig.savefig(f"{D}/concept_drift_heatmap.png", dpi=160, bbox_inches="tight"); plt.close()

print("[FL] figures written to", D)
