#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成两篇文章的实验数据与配图：
1. Set Transformer 金融表征：诱导点注意力聚合多资产 (set-transformer-financial)
2. 分位数 RNN 风险预测：分位损失给出收益区间 (qrnn-quantile-forecast)
纯 numpy 实现，所有图与指标同一份代码产出。
"""
import numpy as np
import time as _time
import os
from statistics import NormalDist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for f in ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/halo/workspace/astro-blog/public/images"
DIR_ST = os.path.join(ROOT, "set-transformer-financial")
DIR_QR = os.path.join(ROOT, "qrnn-quantile-forecast")
os.makedirs(DIR_ST, exist_ok=True)
os.makedirs(DIR_QR, exist_ok=True)

rng = np.random.default_rng(42)

# =========================================================
# 通用：Adam
# =========================================================
class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.p = params; self.lr = lr; self.b1 = b1; self.b2 = b2; self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0
    def step(self, grads):
        self.t += 1
        for k in self.p:
            g = grads[k]
            self.m[k] = self.b1 * self.m[k] + (1 - self.b1) * g
            self.v[k] = self.b2 * self.v[k] + (1 - self.b2) * g * g
            mh = self.m[k] / (1 - self.b1 ** self.t)
            vh = self.v[k] / (1 - self.b2 ** self.t)
            self.p[k] -= self.lr * mh / (np.sqrt(vh) + self.eps)

def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)

def r2(y, yh):
    return 1 - np.sum((y - yh) ** 2) / np.sum((y - y.mean()) ** 2)

# =========================================================
# PART 1: Set Transformer —— 组合拥挤度任务
# =========================================================
print("=" * 60)
print("PART 1: Set Transformer / 诱导点注意力")
print("=" * 60)

K_SET, D_IN, H = 8, 4, 32

def make_set_data(n, seed):
    r = np.random.default_rng(seed)
    X = r.standard_normal((n, K_SET, D_IN))
    a = np.array([0.6, -0.4, 0.3, 0.5])
    lin = (X @ a).mean(axis=1)                      # 逐资产线性部分
    # 拥挤度：所有资产对的 RBF 相似度均值（真·成对交互）
    d2 = ((X[:, :, None, :] - X[:, None, :, :]) ** 2).sum(-1)  # (n,K,K)
    iu = np.triu_indices(K_SET, 1)
    crowd = np.exp(-d2[:, iu[0], iu[1]] / 2.0).mean(axis=1)
    y = 0.5 * lin + 3.0 * crowd
    y = y + 0.05 * r.standard_normal(n)
    return X.astype(np.float64), y

Xtr, ytr = make_set_data(3000, 1)
Xte, yte = make_set_data(1000, 2)
ym, ys = ytr.mean(), ytr.std()
ytr_n, yte_n = (ytr - ym) / ys, (yte - ym) / ys

# ---------- Set Transformer (SAB + PMA, 单头简化) ----------
def st_init(h=H, d=D_IN, seed=7):
    r = np.random.default_rng(seed)
    def xa(shape, fan):
        return r.standard_normal(shape) * np.sqrt(2.0 / fan)
    return {
        "We": xa((d, h), d), "be": np.zeros(h),
        "Wq": xa((h, h), h), "Wk": xa((h, h), h), "Wv": xa((h, h), h),
        "W1": xa((h, h), h), "b1": np.zeros(h),
        "ws": xa((h,), h), "w2": xa((h,), h), "b2": np.zeros(1),
    }

def st_forward(p, X):
    h = p["We"].shape[1]; s = np.sqrt(h)
    E1 = X @ p["We"] + p["be"]; E = np.maximum(E1, 0)
    Q = E @ p["Wq"]; Kk = E @ p["Wk"]; V = E @ p["Wv"]
    S = np.einsum("bik,bjk->bij", Q, Kk) / s
    A = softmax(S)
    U = np.einsum("bij,bjh->bih", A, V)
    F1 = U @ p["W1"] + p["b1"]; F = np.maximum(F1, 0)
    sc = F @ p["ws"]; al = softmax(sc)
    z = np.einsum("bk,bkh->bh", al, F)
    y = z @ p["w2"] + p["b2"][0]
    cache = (X, E, Q, Kk, V, A, U, F, al, z, s)
    return y, cache

def st_backward(p, dy, cache):
    X, E, Q, Kk, V, A, U, F, al, z, s = cache
    g = {}
    g["w2"] = z.T @ dy; g["b2"] = np.array([dy.sum()])
    dz = dy[:, None] * p["w2"][None, :]
    dal = np.einsum("bh,bkh->bk", dz, F)
    dF = al[:, :, None] * dz[:, None, :]
    dsc = al * (dal - (al * dal).sum(-1, keepdims=True))
    g["ws"] = np.einsum("bkh,bk->h", F, dsc)
    dF += dsc[:, :, None] * p["ws"][None, None, :]
    dF1 = dF * (F > 0)
    g["W1"] = np.einsum("bki,bkj->ij", U, dF1); g["b1"] = dF1.sum((0, 1))
    dU = dF1 @ p["W1"].T
    dA = np.einsum("bih,bjh->bij", dU, V)
    dV = np.einsum("bij,bih->bjh", A, dU)
    dS = A * (dA - (A * dA).sum(-1, keepdims=True))
    dQ = np.einsum("bij,bjk->bik", dS, Kk) / s
    dKk = np.einsum("bij,bik->bjk", dS, Q) / s
    g["Wq"] = np.einsum("bki,bkj->ij", E, dQ)
    g["Wk"] = np.einsum("bki,bkj->ij", E, dKk)
    g["Wv"] = np.einsum("bki,bkj->ij", E, dV)
    dE = dQ @ p["Wq"].T + dKk @ p["Wk"].T + dV @ p["Wv"].T
    dE1 = dE * (E > 0)
    g["We"] = np.einsum("bkd,bkh->dh", X, dE1); g["be"] = dE1.sum((0, 1))
    return g

# ---------- 梯度校验 ----------
def grad_check():
    r = np.random.default_rng(0)
    Xs = r.standard_normal((2, 3, D_IN)); ts = r.standard_normal(2)
    p = st_init(h=6, seed=3)
    yh, c = st_forward(p, Xs)
    dy = 2 * (yh - ts) / len(ts)
    g = st_backward(p, dy, c)
    errs = {}
    for k in p:
        gn = np.zeros_like(p[k]); eps = 1e-6
        it = np.nditer(p[k], flags=["multi_index"])
        while not it.finished:
            idx = it.multi_index
            old = p[k][idx]
            p[k][idx] = old + eps; y1, _ = st_forward(p, Xs); l1 = np.mean((y1 - ts) ** 2)
            p[k][idx] = old - eps; y2, _ = st_forward(p, Xs); l2 = np.mean((y2 - ts) ** 2)
            p[k][idx] = old
            gn[idx] = (l1 - l2) / (2 * eps)
            it.iternext()
        denom = max(np.abs(g[k]).max(), np.abs(gn).max(), 1e-12)
        errs[k] = np.abs(g[k] - gn).max() / denom
    return errs

errs = grad_check()
print("Set Transformer 梯度校验（相对误差）:")
for k, v in errs.items():
    print(f"  {k}: {v:.2e}")
GC_MAX = max(errs.values())

# ---------- DeepSets 对照 ----------
def ds_init(h=H, d=D_IN, seed=9):
    r = np.random.default_rng(seed)
    def xa(shape, fan): return r.standard_normal(shape) * np.sqrt(2.0 / fan)
    return {"We": xa((d, h), d), "be": np.zeros(h),
            "W1": xa((h, h), h), "b1": np.zeros(h),
            "w2": xa((h,), h), "b2": np.zeros(1)}

def ds_forward(p, X):
    E1 = X @ p["We"] + p["be"]; E = np.maximum(E1, 0)
    z = E.mean(axis=1)
    G1 = z @ p["W1"] + p["b1"]; G = np.maximum(G1, 0)
    y = G @ p["w2"] + p["b2"][0]
    return y, (X, E, z, G)

def ds_backward(p, dy, cache):
    X, E, z, G = cache
    g = {}
    g["w2"] = G.T @ dy; g["b2"] = np.array([dy.sum()])
    dG = dy[:, None] * p["w2"][None, :]
    dG1 = dG * (G > 0)
    g["W1"] = z.T @ dG1; g["b1"] = dG1.sum(0)
    dz = dG1 @ p["W1"].T
    dE = np.repeat(dz[:, None, :], X.shape[1], axis=1) / X.shape[1]
    dE1 = dE * (E > 0)
    g["We"] = np.einsum("bkd,bkh->dh", X, dE1); g["be"] = dE1.sum((0, 1))
    return g

def train_model(init_fn, fwd, bwd, Xtr, ytr, Xte, yte, epochs=100, bs=128, lr=1e-3, seed=7):
    p = init_fn(seed=seed)
    opt = Adam(p, lr=lr)
    n = len(ytr); hist = []
    r = np.random.default_rng(seed)
    for ep in range(epochs):
        idx = r.permutation(n)
        for s0 in range(0, n, bs):
            b = idx[s0:s0 + bs]
            yh, c = fwd(p, Xtr[b])
            dy = 2 * (yh - ytr[b]) / len(b)
            opt.step(bwd(p, dy, c))
        if (ep + 1) % 5 == 0 or ep == 0:
            yh_te, _ = fwd(p, Xte)
            hist.append((ep + 1, r2(yte, yh_te)))
    yh_tr, _ = fwd(p, Xtr); yh_te, _ = fwd(p, Xte)
    return p, r2(ytr, yh_tr), r2(yte, yh_te), hist

t0 = _time.time()
p_st, r2tr_st, r2te_st, hist_st = train_model(st_init, st_forward, st_backward,
                                              Xtr, ytr_n, Xte, yte_n, epochs=100)
t_st = _time.time() - t0
t0 = _time.time()
p_ds, r2tr_ds, r2te_ds, hist_ds = train_model(ds_init, ds_forward, ds_backward,
                                              Xtr, ytr_n, Xte, yte_n, epochs=100)
t_ds = _time.time() - t0

# ---------- 手工聚合统计 + Ridge ----------
def pooled_stats(X):
    return np.concatenate([X.mean(1), X.std(1), X.max(1), X.min(1)], axis=1)

Ptr, Pte = pooled_stats(Xtr), pooled_stats(Xte)
Ptr1 = np.column_stack([Ptr, np.ones(len(Ptr))])
Pte1 = np.column_stack([Pte, np.ones(len(Pte))])
w_r = np.linalg.solve(Ptr1.T @ Ptr1 + 1e-3 * np.eye(Ptr1.shape[1]), Ptr1.T @ ytr_n)
r2te_ridge = r2(yte_n, Pte1 @ w_r)

print(f"\n[主实验 K=8 拥挤度任务] 测试集 R²:")
print(f"  SetTransformer(SAB+PMA): {r2te_st:.3f} (train {r2tr_st:.3f}, {t_st:.0f}s)")
print(f"  DeepSets-mean:           {r2te_ds:.3f} (train {r2tr_ds:.3f}, {t_ds:.0f}s)")
print(f"  统计聚合+Ridge:           {r2te_ridge:.3f}")

# ---------- 置换不变性 ----------
perm = rng.permutation(K_SET)
y1, _ = st_forward(p_st, Xte); y2, _ = st_forward(p_st, Xte[:, perm, :])
pi_st = np.abs(y1 - y2).max()
y1d, _ = ds_forward(p_ds, Xte); y2d, _ = ds_forward(p_ds, Xte[:, perm, :])
pi_ds = np.abs(y1d - y2d).max()
print(f"置换不变性偏移: SetTransformer {pi_st:.2e}, DeepSets {pi_ds:.2e}")

# ---------- 小样本翻车实验 ----------
Xtr_s, ytr_s = Xtr[:300], ytr_n[:300]
_, r2tr_st_s, r2te_st_s, _ = train_model(st_init, st_forward, st_backward,
                                         Xtr_s, ytr_s, Xte, yte_n, epochs=200, bs=64)
_, r2tr_ds_s, r2te_ds_s, _ = train_model(ds_init, ds_forward, ds_backward,
                                         Xtr_s, ytr_s, Xte, yte_n, epochs=200, bs=64)
print(f"\n[小样本 n=300] SetTransformer: train {r2tr_st_s:.3f} / test {r2te_st_s:.3f}")
print(f"[小样本 n=300] DeepSets:       train {r2tr_ds_s:.3f} / test {r2te_ds_s:.3f}")

# ---------- ISAB 前向计时：O(K²) vs O(K·m) ----------
def timing_isab():
    h, m, B = 32, 16, 8
    r = np.random.default_rng(0)
    Ks = [64, 128, 256, 512, 1024, 2048]
    t_sab, t_isab = [], []
    Wq = r.standard_normal((h, h)) * 0.1; Wk = Wq.copy(); Wv = Wq.copy()
    I = r.standard_normal((m, h)) * 0.1
    for Kn in Ks:
        E = r.standard_normal((B, Kn, h))
        # SAB: K×K
        t0 = _time.time()
        for _ in range(3):
            Q = E @ Wq; Kx = E @ Wk; V = E @ Wv
            A = softmax(np.einsum("bik,bjk->bij", Q, Kx) / np.sqrt(h))
            _ = np.einsum("bij,bjh->bih", A, V)
        t_sab.append((_time.time() - t0) / 3)
        # ISAB: m×K + K×m
        t0 = _time.time()
        for _ in range(3):
            Kx = E @ Wk; V = E @ Wv
            A1 = softmax(np.einsum("mk,bjk->bmj", I @ Wq, Kx) / np.sqrt(h))
            H1 = np.einsum("bmj,bjh->bmh", A1, V)
            Q2 = E @ Wq; K1 = H1 @ Wk; V1 = H1 @ Wv
            A2 = softmax(np.einsum("bik,bmk->bim", Q2, K1) / np.sqrt(h))
            _ = np.einsum("bim,bmh->bih", A2, V1)
        t_isab.append((_time.time() - t0) / 3)
    return Ks, t_sab, t_isab

Ks_t, t_sab, t_isab = timing_isab()
print("\nISAB 计时 (m=16):")
for Kn, a, b in zip(Ks_t, t_sab, t_isab):
    print(f"  K={Kn}: SAB {a*1000:.1f}ms, ISAB {b*1000:.1f}ms, 加速 {a/b:.1f}x")

# ---------- 注意力可视化样本 ----------
# 找一个含有"高度相似资产对"的样本
d2_te = ((Xte[:, :, None, :] - Xte[:, None, :, :]) ** 2).sum(-1)
iu = np.triu_indices(K_SET, 1)
min_d = d2_te[:, iu[0], iu[1]].min(axis=1)
sample_i = int(np.argmin(min_d))
Xs = Xte[sample_i:sample_i + 1]
_, cache_s = st_forward(p_st, Xs)
A_vis = cache_s[5][0]          # (K,K) 注意力
al_vis = cache_s[8][0]         # (K,) PMA 池化权重
sim_vis = np.exp(-d2_te[sample_i] / 2)

# =========== 图 1: cover ===========
fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
im0 = axes[0].imshow(sim_vis, cmap="viridis")
axes[0].set_title("真实成对相似度 exp(-‖xᵢ-xⱼ‖²/2)", fontsize=11)
axes[0].set_xlabel("资产 j"); axes[0].set_ylabel("资产 i")
plt.colorbar(im0, ax=axes[0], fraction=0.046)
im1 = axes[1].imshow(A_vis, cmap="magma")
axes[1].set_title("SAB 学到的注意力矩阵 A", fontsize=11)
axes[1].set_xlabel("被关注资产 j"); axes[1].set_ylabel("查询资产 i")
plt.colorbar(im1, ax=axes[1], fraction=0.046)
axes[2].bar(range(K_SET), al_vis, color="#4C72B0")
axes[2].set_title("PMA 池化权重（注意力代替平均）", fontsize=11)
axes[2].set_xlabel("资产 i"); axes[2].set_ylabel("池化权重 αᵢ")
axes[2].axhline(1 / K_SET, color="crimson", ls="--", lw=1, label="均值池化=1/8")
axes[2].legend()
fig.suptitle("Set Transformer：注意力显式建模资产间成对交互，PMA 用注意力做池化", fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(DIR_ST, "cover.png"), dpi=110, bbox_inches="tight")
plt.close()

# =========== 图 2: 性能对比 ===========
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
names = ["SetTransformer\n(SAB+PMA)", "DeepSets\n(mean-pool)", "统计聚合\n+Ridge"]
vals = [r2te_st, r2te_ds, r2te_ridge]
colors = ["#4C72B0", "#DD8452", "#55A868"]
bars = axes[0].bar(names, vals, color=colors)
for b, v in zip(bars, vals):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}", ha="center", fontsize=11)
axes[0].set_ylabel("测试集 R²")
axes[0].set_title("组合拥挤度任务（K=8, n=3000）", fontsize=12)
axes[0].set_ylim(0, 1.05)
e_st, r_st = zip(*hist_st); e_ds, r_ds = zip(*hist_ds)
axes[1].plot(e_st, r_st, "o-", color="#4C72B0", label="SetTransformer")
axes[1].plot(e_ds, r_ds, "s-", color="#DD8452", label="DeepSets-mean")
axes[1].set_xlabel("epoch"); axes[1].set_ylabel("测试集 R²")
axes[1].set_title("学习曲线", fontsize=12)
axes[1].legend(); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_ST, "perf.png"), dpi=110, bbox_inches="tight")
plt.close()

# =========== 图 3: ISAB 计时 + 小样本 ===========
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
axes[0].loglog(Ks_t, [x * 1000 for x in t_sab], "o-", color="#C44E52", label="SAB 全注意力 O(K²)")
axes[0].loglog(Ks_t, [x * 1000 for x in t_isab], "s-", color="#4C72B0", label="ISAB 诱导点 O(K·m), m=16")
axes[0].set_xlabel("集合大小 K（资产数）"); axes[0].set_ylabel("前向耗时 (ms)")
axes[0].set_title(f"ISAB 加速：K=2048 时 {t_sab[-1]/t_isab[-1]:.1f}x", fontsize=12)
axes[0].legend(); axes[0].grid(alpha=0.3, which="both")
x = np.arange(2); w = 0.35
axes[1].bar(x - w / 2, [r2te_st, r2te_st_s], w, color="#4C72B0", label="SetTransformer")
axes[1].bar(x + w / 2, [r2te_ds, r2te_ds_s], w, color="#DD8452", label="DeepSets-mean")
for i, (a, b) in enumerate([(r2te_st, r2te_ds), (r2te_st_s, r2te_ds_s)]):
    axes[1].text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=10)
    axes[1].text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=10)
axes[1].set_xticks(x); axes[1].set_xticklabels(["n=3000", "n=300（小样本）"])
axes[1].set_ylabel("测试集 R²")
axes[1].set_title("小样本翻车：注意力参数多，先过拟合", fontsize=12)
axes[1].legend()
plt.tight_layout()
plt.savefig(os.path.join(DIR_ST, "scaling.png"), dpi=110, bbox_inches="tight")
plt.close()
print("Part 1 figures saved.")

# =========================================================
# PART 2: QRNN —— 分位数 RNN 收益区间预测
# =========================================================
print("\n" + "=" * 60)
print("PART 2: QRNN / 分位损失")
print("=" * 60)

# ---------- GARCH(1,1)+t(5) 合成收益 ----------
N2 = 4000
omega, alpha_g, beta_g = 0.02, 0.09, 0.89
nu = 5.0
r2g = np.random.default_rng(123)
# 标准化 t(5)
t_raw = r2g.standard_t(nu, N2)
t_std = t_raw / np.sqrt(nu / (nu - 2))
sig2 = np.zeros(N2); sig2[0] = omega / (1 - alpha_g - beta_g)
ret = np.zeros(N2)
for i in range(1, N2):
    sig2[i] = omega + alpha_g * ret[i - 1] ** 2 + beta_g * sig2[i - 1]
    ret[i] = np.sqrt(sig2[i]) * t_std[i]
sigma_true = np.sqrt(sig2)

TAUS = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
N_TR = 3000

# 特征: [r_t, |r_t|, r_t^2]，训练集标准化
feats = np.column_stack([ret, np.abs(ret), ret ** 2])
fm, fs = feats[:N_TR].mean(0), feats[:N_TR].std(0)
Xf = (feats - fm) / fs
# 目标: r_{t+1}
target = ret.copy()

# ---------- Elman RNN + pinball, 截断 BPTT ----------
HID = 16
def qrnn_init(seed=5):
    r = np.random.default_rng(seed)
    return {"Wx": r.standard_normal((3, HID)) * 0.3,
            "Wh": r.standard_normal((HID, HID)) * 0.3 / np.sqrt(HID),
            "bh": np.zeros(HID),
            "Wo": r.standard_normal((HID, len(TAUS))) * 0.1,
            "bo": np.zeros(len(TAUS))}

def pinball(y, q, taus):
    u = y[:, None] - q
    return np.mean(np.maximum(taus * u, (taus - 1) * u))

def qrnn_run(p, X, h0=None):
    T = len(X); Hs = np.zeros((T, HID)); h = np.zeros(HID) if h0 is None else h0
    for i in range(T):
        h = np.tanh(X[i] @ p["Wx"] + h @ p["Wh"] + p["bh"])
        Hs[i] = h
    Q = Hs @ p["Wo"] + p["bo"]
    return Q, Hs, h

def qrnn_train(p, X, y, epochs=60, chunk=64, lr=3e-3):
    opt = Adam(p, lr=lr)
    T = len(X) - 1  # 预测 y[t+1]
    for ep in range(epochs):
        h = np.zeros(HID)
        for s0 in range(0, T, chunk):
            e0 = min(s0 + chunk, T)
            Xc = X[s0:e0]; yc = y[s0 + 1:e0 + 1]
            L = e0 - s0
            # forward
            Hs = np.zeros((L, HID)); hh = h.copy()
            for i in range(L):
                hh = np.tanh(Xc[i] @ p["Wx"] + hh @ p["Wh"] + p["bh"])
                Hs[i] = hh
            Q = Hs @ p["Wo"] + p["bo"]
            u = yc[:, None] - Q
            dQ = (np.where(u < 0, 1.0, 0.0) - TAUS[None, :]) / (L * len(TAUS))
            g = {k: np.zeros_like(v) for k, v in p.items()}
            g["Wo"] = Hs.T @ dQ; g["bo"] = dQ.sum(0)
            dh_next = np.zeros(HID)
            for i in range(L - 1, -1, -1):
                dh = dQ[i] @ p["Wo"].T + dh_next
                da = dh * (1 - Hs[i] ** 2)
                hprev = Hs[i - 1] if i > 0 else h
                g["Wx"] += np.outer(Xc[i], da)
                g["Wh"] += np.outer(hprev, da)
                g["bh"] += da
                dh_next = da @ p["Wh"].T
            # 梯度裁剪
            gn = np.sqrt(sum((gv ** 2).sum() for gv in g.values()))
            if gn > 5.0:
                for k in g: g[k] *= 5.0 / gn
            opt.step(g)
            h = hh  # 状态跨 chunk 传递，梯度截断
    return p

p_q = qrnn_init()
t0 = _time.time()
p_q = qrnn_train(p_q, Xf[:N_TR], target[:N_TR], epochs=60)
print(f"QRNN 训练完成 {_time.time()-t0:.0f}s")

# 全序列前向（状态热启动）
Q_all, _, _ = qrnn_run(p_q, Xf[:-1])
Q_test_raw = Q_all[N_TR - 1:]          # 预测 target[N_TR..N2-1]
y_test = target[N_TR:]
sig_test = sigma_true[N_TR:]

# ---------- 分位数交叉统计与重排 ----------
crossings = (np.diff(Q_test_raw, axis=1) < 0).any(axis=1)
cross_rate = crossings.mean()
Q_test = np.sort(Q_test_raw, axis=1)
print(f"联合训练 QRNN 分位数交叉率（重排前）: {cross_rate*100:.1f}%")

# ---------- 对照：每个分位数独立训练一个 QRNN → 经典交叉翻车 ----------
def qrnn_train_single(tau, seed, epochs=60):
    ps = qrnn_init(seed=seed)
    ps["Wo"] = ps["Wo"][:, :1].copy(); ps["bo"] = np.zeros(1)
    opt = Adam(ps, lr=3e-3)
    X, y = Xf[:N_TR], target[:N_TR]
    T = len(X) - 1; chunk = 64; tau_a = np.array([tau])
    for ep in range(epochs):
        h = np.zeros(HID)
        for s0 in range(0, T, chunk):
            e0 = min(s0 + chunk, T)
            Xc = X[s0:e0]; yc = y[s0 + 1:e0 + 1]; L = e0 - s0
            Hs = np.zeros((L, HID)); hh = h.copy()
            for i in range(L):
                hh = np.tanh(Xc[i] @ ps["Wx"] + hh @ ps["Wh"] + ps["bh"])
                Hs[i] = hh
            Q = Hs @ ps["Wo"] + ps["bo"]
            u = yc[:, None] - Q
            dQ = (np.where(u < 0, 1.0, 0.0) - tau_a[None, :]) / L
            g = {k: np.zeros_like(v) for k, v in ps.items()}
            g["Wo"] = Hs.T @ dQ; g["bo"] = dQ.sum(0)
            dh_next = np.zeros(HID)
            for i in range(L - 1, -1, -1):
                dh = dQ[i] @ ps["Wo"].T + dh_next
                da = dh * (1 - Hs[i] ** 2)
                hprev = Hs[i - 1] if i > 0 else h
                g["Wx"] += np.outer(Xc[i], da); g["Wh"] += np.outer(hprev, da); g["bh"] += da
                dh_next = da @ ps["Wh"].T
            gn = np.sqrt(sum((gv ** 2).sum() for gv in g.values()))
            if gn > 5.0:
                for k in g: g[k] *= 5.0 / gn
            opt.step(g)
            h = hh
    Qs, _, _ = qrnn_run(ps, Xf[:-1])
    return Qs[N_TR - 1:, 0]

Q_indep = np.column_stack([qrnn_train_single(t, seed=50 + j) for j, t in enumerate(TAUS)])
cross_indep = (np.diff(Q_indep, axis=1) < 0).any(axis=1).mean()
print(f"独立训练 5 个 QRNN 分位数交叉率: {cross_indep*100:.1f}%")

# ---------- 基线 ----------
nd = NormalDist()
z_taus = np.array([nd.inv_cdf(t) for t in TAUS])
# 1) EWMA 高斯 (RiskMetrics)
lam = 0.94
s2 = np.zeros(N2); s2[0] = ret[:50].var()
for i in range(1, N2):
    s2[i] = lam * s2[i - 1] + (1 - lam) * ret[i - 1] ** 2
Q_ewma = np.sqrt(s2[N_TR:, None]) * z_taus[None, :]
# 2) 滚动历史分位 (250)
Q_hist = np.zeros((N2 - N_TR, len(TAUS)))
for i in range(N_TR, N2):
    Q_hist[i - N_TR] = np.quantile(ret[i - 250:i], TAUS)
# 3) 线性分位回归（同特征，pinball 梯度下降）
p_lin = {"W": np.zeros((3, len(TAUS))), "b": np.zeros(len(TAUS))}
opt_l = Adam(p_lin, lr=5e-3)
Xl, yl = Xf[:N_TR - 1], target[1:N_TR]
for ep in range(300):
    Ql = Xl @ p_lin["W"] + p_lin["b"]
    u = yl[:, None] - Ql
    dQ = (np.where(u < 0, 1.0, 0.0) - TAUS[None, :]) / (len(yl) * len(TAUS))
    opt_l.step({"W": Xl.T @ dQ, "b": dQ.sum(0)})
Q_lin = np.sort(Xf[N_TR - 1:-1] @ p_lin["W"] + p_lin["b"], axis=1)
# 4) Oracle: 真实 σ_t × t(5) 标准化分位（下界）
t_big = r2g.standard_t(nu, 2_000_000) / np.sqrt(nu / (nu - 2))
tq = np.quantile(t_big, TAUS)
Q_oracle = sig_test[:, None] * tq[None, :]

models_q = {"QRNN（重排后）": Q_test, "EWMA-高斯": Q_ewma,
            "滚动历史250": Q_hist, "线性QR": Q_lin, "Oracle(真σ+t分位)": Q_oracle}
print("\n[测试集 1000 步] pinball / 90%区间覆盖率 / 区间宽度-真σ相关:")
res_q = {}
for nm, Qm in models_q.items():
    pb = pinball(y_test, Qm, TAUS)
    cov90 = ((y_test >= Qm[:, 0]) & (y_test <= Qm[:, -1])).mean()
    width = Qm[:, -1] - Qm[:, 0]
    wc = np.corrcoef(width, sig_test)[0, 1]
    res_q[nm] = (pb, cov90, wc)
    print(f"  {nm}: pinball={pb:.4f}, cov90={cov90:.3f}, width-σ corr={wc:.3f}")

# 逐分位校准
calib = {}
for nm, Qm in models_q.items():
    calib[nm] = [(y_test <= Qm[:, j]).mean() for j in range(len(TAUS))]

# 按波动三分位的覆盖
terc = np.quantile(sig_test, [1 / 3, 2 / 3])
regime = np.digitize(sig_test, terc)  # 0 低波 1 中波 2 高波
cov_regime = {}
for nm in ["QRNN（重排后）", "EWMA-高斯", "滚动历史250"]:
    Qm = models_q[nm]
    cov_regime[nm] = [((y_test >= Qm[:, 0]) & (y_test <= Qm[:, -1]))[regime == g].mean()
                      for g in range(3)]
print("\n按波动 regime 的 90% 覆盖率 (低/中/高):")
for nm, v in cov_regime.items():
    print(f"  {nm}: {v[0]:.3f} / {v[1]:.3f} / {v[2]:.3f}")

# =========== 图 1: cover 扇形图 ===========
seg = slice(0, 400)
xs = np.arange(400)
fig, ax = plt.subplots(figsize=(14, 5.5))
ax.fill_between(xs, Q_test[seg, 0], Q_test[seg, 4], alpha=0.25, color="#4C72B0",
                label="QRNN 5%–95% 区间")
ax.fill_between(xs, Q_test[seg, 1], Q_test[seg, 3], alpha=0.35, color="#4C72B0",
                label="QRNN 25%–75% 区间")
ax.plot(xs, Q_test[seg, 2], color="#4C72B0", lw=1, label="QRNN 中位数")
ax.plot(xs, y_test[seg], "k.", ms=3, alpha=0.6, label="实际收益")
out = (y_test[seg] < Q_test[seg, 0]) | (y_test[seg] > Q_test[seg, 4])
ax.plot(xs[out], y_test[seg][out], "rx", ms=7, label=f"区间外 ({out.mean()*100:.0f}%)")
ax.set_xlabel("测试集时间步"); ax.set_ylabel("收益率 (%)")
ax.set_title("QRNN 分位数区间预测：区间宽度随波动聚集自动收放（GARCH-t 合成收益，样本外）", fontsize=13)
ax.legend(loc="upper right", ncol=2); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_QR, "cover.png"), dpi=110, bbox_inches="tight")
plt.close()

# =========== 图 2: 校准 + pinball ===========
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
mkers = {"QRNN（重排后）": ("o-", "#4C72B0"), "EWMA-高斯": ("s-", "#DD8452"),
         "滚动历史250": ("^-", "#55A868"), "线性QR": ("d-", "#8172B3"),
         "Oracle(真σ+t分位)": ("*-", "#937860")}
for nm, (mk, c) in mkers.items():
    axes[0].plot(TAUS, calib[nm], mk, color=c, label=nm, ms=6)
axes[0].plot([0, 1], [0, 1], "k--", lw=1, label="理想校准")
axes[0].set_xlabel("名义分位数 τ"); axes[0].set_ylabel("经验命中率 P(y ≤ q̂_τ)")
axes[0].set_title("分位数校准图（越贴对角线越好）", fontsize=12)
axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
names_q = list(res_q.keys())
pbs = [res_q[n][0] for n in names_q]
cols = [mkers[n][1] for n in names_q]
bars = axes[1].bar(range(len(names_q)), pbs, color=cols)
for b, v in zip(bars, pbs):
    axes[1].text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}", ha="center", fontsize=10)
axes[1].set_xticks(range(len(names_q)))
axes[1].set_xticklabels([n.replace("（", "\n（") for n in names_q], fontsize=9)
axes[1].set_ylabel("平均 pinball 损失（越低越好）")
axes[1].set_title("样本外 pinball 损失", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(DIR_QR, "calibration.png"), dpi=110, bbox_inches="tight")
plt.close()

# =========== 图 3: 翻车与 regime ===========
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5))
# 交叉对比：联合训练 vs 独立训练
bars = axes[0].bar(["联合训练\n(共享隐状态+5输出头)", "独立训练\n(每个 τ 一个模型)"],
                   [cross_rate * 100, cross_indep * 100], color=["#55A868", "#C44E52"], width=0.5)
for b, v in zip(bars, [cross_rate * 100, cross_indep * 100]):
    axes[0].text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=12)
axes[0].set_ylabel("分位数交叉率 (%)")
axes[0].set_title("翻车 1：独立训练每个分位数 → 大量交叉（q̂₀.₀₅ > q̂₀.₂₅）", fontsize=11)
ci = np.where((np.diff(Q_indep, axis=1) < 0).any(axis=1))[0]
if len(ci) > 0:
    ex = ci[0]
    txt = " / ".join(f"τ={t:.2f}: {v:.2f}" for t, v in zip(TAUS, Q_indep[ex]))
    axes[0].text(0.5, 0.55, f"独立训练交叉示例 t={ex}:\n{txt}", transform=axes[0].transAxes,
                 ha="center", fontsize=9, bbox=dict(fc="lightyellow", ec="gray"))
x = np.arange(3); w = 0.25
for k, (nm, v) in enumerate(cov_regime.items()):
    axes[1].bar(x + (k - 1) * w, v, w, color=mkers[nm][1], label=nm)
axes[1].axhline(0.90, color="k", ls="--", lw=1, label="目标 90%")
axes[1].set_xticks(x); axes[1].set_xticklabels(["低波动", "中波动", "高波动"])
axes[1].set_ylabel("90% 区间经验覆盖率")
axes[1].set_ylim(0.5, 1.02)
axes[1].set_title("翻车 2：高波动 regime 下覆盖率普遍掉队", fontsize=11)
axes[1].legend(fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(DIR_QR, "pitfalls.png"), dpi=110, bbox_inches="tight")
plt.close()
print("Part 2 figures saved.")

# =========================================================
# 汇总
# =========================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"[ST] grad-check max rel err: {GC_MAX:.2e}")
print(f"[ST] R2 test: ST={r2te_st:.3f} DS={r2te_ds:.3f} Ridge={r2te_ridge:.3f}")
print(f"[ST] perm shift: ST={pi_st:.2e} DS={pi_ds:.2e}")
print(f"[ST] small-n: ST train={r2tr_st_s:.3f} test={r2te_st_s:.3f} | DS train={r2tr_ds_s:.3f} test={r2te_ds_s:.3f}")
print(f"[ST] ISAB speedup @K=2048: {t_sab[-1]/t_isab[-1]:.1f}x")
print(f"[QR] cross rate joint: {cross_rate*100:.1f}%, indep: {cross_indep*100:.1f}%")
for nm, (pb, cov, wc) in res_q.items():
    print(f"[QR] {nm}: pinball={pb:.4f} cov90={cov:.3f} widthcorr={wc:.3f}")
