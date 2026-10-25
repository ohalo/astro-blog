#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-25（第五弹）两篇量化文章的核心数值 + 配图（固定 seed 可复现）。
  A. temporal-conv-causal-inference  时序卷积因果推断：用膨胀卷积预测滞后传导
     - 分布式滞后模型(DLM)视角：不是预测 Y 本身，而是恢复 X→Y 的滞后核（IRF）
     - numpy 从零实现因果膨胀卷积 TCN（d=1/4/16 三层 + ReLU + 1x1 头，Adam 训练）
     - E1 线性 DGP：核恢复对比 TCN-IRF / Almon / Ridge / Oracle-support
     - E2 非线性状态依赖核：lag-8 系数随波动状态翻号 → 线性模型失效，TCN-IRF 分状态恢复
     - E3 AR(1) 噪声：Almon 朴素 t 值虚高 → Newey-West 修正；TCN-IRF block-bootstrap CI
  B. cross-asset-macro-momentum-revisit  跨资产宏观动量再检验
     - 20 年四资产合成面板（8 个 regime；无前视：信号 at t 赚 R[t+1]）
     - 12-1 月时序动量 + 腿级波动目标 + 等风险聚合；对照 buy&hold / 趋势过滤 / 1 月反转叠加
所有图表均为真实计算图，数值固定随机种子可复现。
"""
import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"

BASE = "/Users/halo/workspace/astro-blog/public/images"
RESULTS = {}
C_RED = "#c0392b"; C_BLUE = "#2471a3"; C_GREEN = "#1e8449"; C_ORANGE = "#ca6f1e"
C_PURPLE = "#7d3c98"; C_GRAY = "#7f8c8d"


# ============================================================
# 文章 A
# ============================================================
def causal_conv(X, W, b, dilation):
    """X:(B,Cin,T) W:(Co,Cin,k) b:(Co,) 左填充因果膨胀卷积"""
    B, Cin, T = X.shape
    Co, _, k = W.shape
    pad = (k - 1) * dilation
    Xp = np.zeros((B, Cin, T + pad))
    Xp[:, :, pad:] = X
    out = np.zeros((B, Co, T))
    for j in range(k):
        out += np.einsum("bct,oc->bot", Xp[:, :, pad - j * dilation: pad - j * dilation + T], W[:, :, j])
    out += b[None, :, None]
    return out


def causal_conv_backward(dY, X, W, dilation):
    """返回 dW, db, dX"""
    B, Cin, T = X.shape
    Co, _, k = W.shape
    pad = (k - 1) * dilation
    Xp = np.zeros((B, Cin, T + pad))
    Xp[:, :, pad:] = X
    dXp = np.zeros_like(Xp)
    dW = np.zeros_like(W); db = dY.sum(axis=(0, 2))
    for j in range(k):
        sl = Xp[:, :, pad - j * dilation: pad - j * dilation + T]
        dW[:, :, j] = np.einsum("bct,bot->oc", sl, dY)
        dXp[:, :, pad - j * dilation: pad - j * dilation + T] += np.einsum("bot,oc->bct", dY, W[:, :, j])
    return dW, db, dXp[:, :, pad:]


class TinyTCN:
    """三层因果膨胀卷积(d=1/4/16) + ReLU + 1x1 头读末位。in_ch=2 时第二通道为外部状态。"""

    def __init__(self, rng, C=8, in_ch=1):
        self.C = C; self.in_ch = in_ch; self.dils = [1, 4, 16]
        self.W1 = rng.normal(0, 0.5, (C, in_ch, 3)); self.b1 = np.zeros(C)
        self.W2 = rng.normal(0, 0.5 / np.sqrt(C), (C, C, 3)); self.b2 = np.zeros(C)
        self.W3 = rng.normal(0, 0.5 / np.sqrt(C), (C, C, 3)); self.b3 = np.zeros(C)
        self.Wh = rng.normal(0, 0.5 / np.sqrt(C), (1, C, 1)); self.bh = np.zeros(1)

    def params(self):
        return [self.W1, self.b1, self.W2, self.b2, self.W3, self.b3, self.Wh, self.bh]

    def forward(self, X, cache=False):
        z1 = causal_conv(X, self.W1, self.b1, self.dils[0]); a1 = np.maximum(z1, 0)
        z2 = causal_conv(a1, self.W2, self.b2, self.dils[1]); a2 = np.maximum(z2, 0)
        z3 = causal_conv(a2, self.W3, self.b3, self.dils[2]); a3 = np.maximum(z3, 0)
        h = causal_conv(a3, self.Wh, self.bh, 1)[:, 0, -1]
        if cache:
            return h, (X, z1, a1, z2, a2, z3, a3)
        return h

    def backward(self, cache, dh):
        X, z1, a1, z2, a2, z3, a3 = cache
        da3 = np.zeros_like(a3)
        da3[:, :, -1] = dh[:, None] * self.Wh[0, :, 0][None, :]
        dWh = np.zeros_like(self.Wh); dbh = np.array([dh.sum()])
        dWh[0, :, 0] = (dh[:, None] * a3[:, :, -1]).sum(axis=0)
        dz3 = da3 * (z3 > 0)
        dW3, db3, da2 = causal_conv_backward(dz3, a2, self.W3, self.dils[2])
        dz2 = da2 * (z2 > 0)
        dW2, db2, da1 = causal_conv_backward(dz2, a1, self.W2, self.dils[1])
        dz1 = da1 * (z1 > 0)
        dW1, db1, dX = causal_conv_backward(dz1, X, self.W1, self.dils[0])
        return [dW1, db1, dW2, db2, dW3, db3, dWh, dbh], dX


def train_tcn(X, Y, rng, epochs=80, bs=256, lr=3e-3, ckpts=(), Xte=None, Yte=None, in_ch=1):
    net = TinyTCN(rng, in_ch=in_ch)
    m = [np.zeros_like(p) for p in net.params()]
    v = [np.zeros_like(p) for p in net.params()]
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    n = len(X); step = 0; losses = []
    for ep in range(epochs):
        idx = rng.permutation(n)
        for s in range(0, n, bs):
            bb = idx[s:s + bs]
            h, cache = net.forward(X[bb], cache=True)
            dh = 2.0 * (h - Y[bb]) / len(bb)
            grads, _ = net.backward(cache, dh)
            step += 1
            for i, (p, g) in enumerate(zip(net.params(), grads)):
                m[i] = b1a * m[i] + (1 - b1a) * g
                v[i] = b2a * v[i] + (1 - b2a) * g * g
                p -= lr * (m[i] / (1 - b1a ** step)) / (np.sqrt(v[i] / (1 - b2a ** step)) + eps)
        if Xte is not None and (ep + 1) in ckpts:
            losses.append((ep + 1, float(np.mean((Yte - net.forward(Xte)) ** 2))))
    return net, losses


def corr(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def gen_article_a():
    slug = "temporal-conv-causal-inference"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261025)
    L = 48           # 输入窗口（感受野 43，覆盖）
    LAG_MAX = 31
    T = 10000

    # ---------- 真实滞后核（DGP 用原始尺度 0.15×，图里归一化比较形状） ----------
    lags = np.arange(LAG_MAX + 1)
    w_main = 3.0 * (lags / 4.0) * np.exp(-lags / 4.0)
    w_main[lags > 12] = 0.0
    w_spill = np.zeros(LAG_MAX + 1)
    msk = (lags >= 20) & (lags <= 28)
    w_spill[msk] = 0.35 * np.exp(-((lags[msk] - 22) ** 2) / 8.0)
    w_true = 0.15 * (w_main + w_spill)   # 原始尺度，信号 std ≈ 0.44
    w_true_n = w_true / np.abs(w_true).sum()

    # ---------- X: AR(1) ----------
    Xall = np.zeros(T + 2 * L)
    for t in range(1, len(Xall)):
        Xall[t] = 0.5 * Xall[t - 1] + rng.normal(0, 1.0)
    Xs = (Xall - Xall.mean()) / Xall.std()
    absx = np.abs(Xs)
    roll = np.convolve(absx, np.ones(20) / 20, mode="full")[:len(absx)]
    s_state = np.where(roll > np.median(roll), 1.0, -1.0)

    def build_y(noise_ar=0.0, gamma=0.0, seed=1):
        r = np.random.default_rng(seed)
        e = r.normal(0, 0.25, len(Xs))
        eta = np.zeros(len(Xs))
        for t in range(1, len(eta)):
            eta[t] = noise_ar * eta[t - 1] + e[t]
        Y = np.zeros(len(Xs))
        for t in range(L, len(Y)):
            k = min(t, LAG_MAX)
            Y[t] = float(np.dot(w_true[:k + 1], Xs[t - k:t + 1][::-1]))
            if gamma > 0:
                Y[t] += gamma * s_state[t - 1] * Xs[t - 8]
            Y[t] += eta[t]
        return Y

    # 窗口：X[k:k+L] 预测 Y[k+L-1]（末位即 t）
    def windows(x, y, L):
        n = len(x) - L
        X = np.stack([x[k:k + L] for k in range(n)])[:, None, :]
        Y = y[L - 1:L - 1 + n]
        return X, Y

    def lagmat(Xb):
        """窗口取最后 LAG_MAX+1 个值并按滞后升序排成 (n, 32)"""
        return Xb[:, 0, -LAG_MAX - 1:][:, ::-1]

    n_tr = 8000

    # ================= E1 线性 DGP =================
    Y1 = build_y(noise_ar=0.0, gamma=0.0, seed=11)
    Xw, Yw = windows(Xs, Y1, L)
    Xtr, Ytr, Xte, Yte = Xw[:n_tr], Yw[:n_tr], Xw[n_tr:], Yw[n_tr:]

    t0 = time.time()
    net1, losses1 = train_tcn(Xtr, Ytr, rng, epochs=80, bs=256, lr=3e-3,
                              ckpts=(8, 20, 40, 60, 80), Xte=Xte, Yte=Yte)
    print(f"[A] E1 TCN trained {time.time()-t0:.0f}s, losses={losses1}")

    Ztr, Zte = lagmat(Xtr), lagmat(Xte)

    # Almon（degree 3）
    P = np.vander(lags, 4, increasing=True)
    Za, ya = Ztr @ P, Ytr
    beta_a, *_ = np.linalg.lstsq(Za, ya, rcond=None)
    w_almon = P @ beta_a
    resid = ya - Za @ beta_a
    sig2 = resid @ resid / (len(ya) - 4)
    XtXi = np.linalg.pinv(Za.T @ Za)
    se_naive = np.sqrt(np.diag(sig2 * XtXi))
    S = Za.T @ Za
    for l in range(1, 9):
        G = Za[l:].T @ Za[:len(Za) - l]
        S += (1 - l / 9.0) * (G + G.T)
    se_nw = np.sqrt(np.diag(sig2 * (XtXi @ S @ XtXi)))
    t_naive, t_nw = beta_a / se_naive, beta_a / se_nw

    # Ridge（验证集选 alpha）
    Zv, yv = Ztr[-2000:], Ytr[-2000:]
    best = (None, None, 1e18)
    for alpha in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]:
        b = np.linalg.solve(Ztr.T @ Ztr + alpha * np.eye(32), Ztr.T @ Ytr)
        mse = float(np.mean((Zv @ b - yv) ** 2))
        if mse < best[2]:
            best = (alpha, b, mse)
    alpha_r, b_r, mse_r = best
    w_ridge = b_r

    # Oracle support
    support = (lags <= 12) | msk
    Zo = Ztr[:, support]
    b_o = np.linalg.solve(Zo.T @ Zo + 0.1 * np.eye(Zo.shape[1]), Zo.T @ Ytr)
    w_oracle = np.zeros(32); w_oracle[support] = b_o

    # TCN-IRF
    def tcn_irf(net, lag_list, in_ch=1):
        X0 = np.zeros((1, in_ch, L))
        f0 = net.forward(X0)[0]
        out = np.zeros(len(lag_list))
        for i, lag in enumerate(lag_list):
            Xi = X0.copy(); Xi[0, 0, L - 1 - lag] = 1.0
            out[i] = net.forward(Xi)[0] - f0
        return out

    irf_tcn = tcn_irf(net1, lags)
    nz = lambda a: a / np.abs(a).sum()
    r2 = lambda p, y=Yte: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    p_tcn = net1.forward(Xte)
    p_ridge, p_almon, p_or = Zte @ b_r, (Zte @ P) @ beta_a, Zte[:, support] @ b_o

    e1 = {
        "corr_tcn": corr(nz(irf_tcn), w_true_n), "corr_almon": corr(nz(w_almon), w_true_n),
        "corr_ridge": corr(nz(w_ridge), w_true_n), "corr_oracle": corr(nz(w_oracle), w_true_n),
        "rmse_tcn": float(np.sqrt(np.mean((nz(irf_tcn) - w_true_n) ** 2))),
        "rmse_almon": float(np.sqrt(np.mean((nz(w_almon) - w_true_n) ** 2))),
        "rmse_ridge": float(np.sqrt(np.mean((nz(w_ridge) - w_true_n) ** 2))),
        "r2_tcn": r2(p_tcn), "r2_ridge": r2(p_ridge), "r2_almon": r2(p_almon), "r2_oracle": r2(p_or),
        "alpha_ridge": alpha_r, "mse_ridge_val": mse_r,
        "almon_peak_lag": int(np.argmax(w_almon)), "tcn_peak_lag": int(np.argmax(irf_tcn)),
        "true_peak_lag": int(np.argmax(w_true)),
        "almon_spill_max": float(w_almon[20:29].max()), "tcn_spill_max": float(irf_tcn[20:29].max()),
        "true_spill_max": float(w_true[20:29].max()),
    }

    # ================= E2 非线性状态依赖核 =================
    gam = 0.35
    Y2 = build_y(noise_ar=0.0, gamma=gam, seed=12)
    Xw2, Yw2 = windows(Xs, Y2, L)
    n2 = len(Xw2)
    Sw = np.stack([s_state[k:k + L] for k in range(n2)])  # 与 X 窗口对齐的状态历史
    Xtr2 = np.concatenate([Xw2[:n_tr], Sw[:n_tr][:, None, :]], axis=1)
    Xte2 = np.concatenate([Xw2[n_tr:], Sw[n_tr:][:, None, :]], axis=1)
    Ytr2, Yte2 = Yw2[:n_tr], Yw2[n_tr:]
    net2, _ = train_tcn(Xtr2, Ytr2, rng, epochs=80, bs=256, lr=3e-3, in_ch=2)
    Ztr2, Zte2 = lagmat(Xw2[:n_tr]), lagmat(Xw2[n_tr:])
    b_r2 = np.linalg.solve(Ztr2.T @ Ztr2 + 0.3 * np.eye(32), Ztr2.T @ Ytr2)
    w_lin2 = b_r2

    def tcn_irf2(net, lag_list, state_val):
        X0 = np.zeros((1, 2, L)); X0[0, 1, :] = state_val
        f0 = net.forward(X0)[0]
        out = np.zeros(len(lag_list))
        for i, lag in enumerate(lag_list):
            Xi = X0.copy(); Xi[0, 0, L - 1 - lag] = 1.0
            out[i] = net.forward(Xi)[0] - f0
        return out

    irf_hi, irf_lo = tcn_irf2(net2, lags, 1.0), tcn_irf2(net2, lags, -1.0)
    irf_hi_n, irf_lo_n = nz(irf_hi), nz(irf_lo)
    w_hi = w_true + gam * (lags == 8); w_lo = w_true - gam * (lags == 8)
    w_hi_n, w_lo_n = nz(w_hi), nz(w_lo)
    p2_tcn = net2.forward(Xte2)
    p2_lin = Zte2 @ b_r2
    r2b = lambda p, y: 1 - np.sum((y - p) ** 2) / np.sum((y - y.mean()) ** 2)
    e2 = {
        "corr_hi": corr(irf_hi_n, w_hi_n), "corr_lo": corr(irf_lo_n, w_lo_n),
        "corr_lin_hi": corr(nz(w_lin2), w_hi_n), "corr_lin_lo": corr(nz(w_lin2), w_lo_n),
        "r2_tcn": r2b(p2_tcn, Yte2), "r2_lin": r2b(p2_lin, Yte2),
        "lag8_hi": float(irf_hi[8]), "lag8_lo": float(irf_lo[8]),
        "lag8_true_hi": float(w_hi[8]), "lag8_true_lo": float(w_lo[8]),
        "lag8_lin": float(w_lin2[8]),
    }

    # ================= E3 AR(1) 噪声推断 =================
    Y3 = build_y(noise_ar=0.7, gamma=0.0, seed=13)
    Xw3, Yw3 = windows(Xs, Y3, L)
    Xtr3, Ytr3 = Xw3[:n_tr], Yw3[:n_tr]
    Za3 = lagmat(Xtr3) @ P
    ya3 = Ytr3
    beta3, *_ = np.linalg.lstsq(Za3, ya3, rcond=None)
    resid3 = ya3 - Za3 @ beta3
    sig23 = resid3 @ resid3 / (len(ya3) - 4)
    XtXi3 = np.linalg.pinv(Za3.T @ Za3)
    se_n3 = np.sqrt(np.diag(sig23 * XtXi3))
    S3 = Za3.T @ Za3
    for l in range(1, 9):
        G = Za3[l:].T @ Za3[:len(Za3) - l]
        S3 += (1 - l / 9.0) * (G + G.T)
    se_nw3 = np.sqrt(np.diag(sig23 * (XtXi3 @ S3 @ XtXi3)))
    t_naive3, t_nw3 = beta3 / se_n3, beta3 / se_nw3
    w_almon3 = nz(P @ beta3)
    se_ratio = float(np.abs(t_naive3).max() / np.abs(t_nw3).max())

    # TCN block-bootstrap IRF CI
    B_boot = 6
    boot = []
    for i in range(B_boot):
        rb = np.random.default_rng(1000 + i)
        blk = 40; nb = n_tr // blk
        starts = rb.integers(0, n_tr - blk, nb)
        idx = np.concatenate([np.arange(s, s + blk) for s in starts])
        nb_net, _ = train_tcn(Xtr3[idx], Ytr3[idx], rb, epochs=50, bs=256, lr=3e-3)
        boot.append(nz(tcn_irf(nb_net, lags)))
        print(f"[A] bootstrap {i+1}/{B_boot} done")
    boot = np.array(boot)
    ci_lo, ci_hi = np.percentile(boot, 5, axis=0), np.percentile(boot, 95, axis=0)
    cover = float(np.mean((w_true_n >= ci_lo) & (w_true_n <= ci_hi)))
    e3 = {
        "spurious_naive": int(np.sum(np.abs(t_naive3) > 1.96)),
        "spurious_nw": int(np.sum(np.abs(t_nw3) > 1.96)),
        "true_sig": int(np.sum(np.abs(w_true) > 0.01)),
        "t_naive_max": float(np.abs(t_naive3).max()), "t_nw_max": float(np.abs(t_nw3).max()),
        "se_ratio": se_ratio,
        "cover": cover, "corr_almon_ar": corr(w_almon3, w_true_n),
        "rmse_almon_ar": float(np.sqrt(np.mean((w_almon3 - w_true_n) ** 2))),
    }
    RESULTS["A"] = {"E1": e1, "E2": e2, "E3": e3}

    # ================= 图 1：DGP =================
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 8.2))
    seg = slice(600, 760)
    axes[0].plot(np.arange(seg.start, seg.stop), Xs[seg], color=C_BLUE, lw=1.2)
    axes[0].set_title("驱动过程 X：AR(1)，ρ=0.5", fontsize=11); axes[0].set_ylabel("X")
    axes[1].plot(lags, w_true_n, color=C_RED, lw=1.8)
    axes[1].stem(lags, w_true_n, basefmt=" ", markerfmt="o", linefmt=C_GRAY)
    axes[1].set_title("真实滞后核 w（主峰 lag 4，长尾溢出 lag 20-28，已归一化）", fontsize=11)
    axes[1].set_ylabel("w"); axes[1].set_xlabel("滞后阶数 k")
    axes[2].plot(np.arange(seg.start, seg.stop), Y1[seg], color=C_GREEN, lw=1.2)
    axes[2].set_title("响应 Y = Σ w_k·X(t−k) + ε：冲击经滞后核整形后传导", fontsize=11)
    axes[2].set_ylabel("Y"); axes[2].set_xlabel("t")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lag_kernel_dgp.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 2：架构 + 训练 =================
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    layers = ["输入\n(1ch)", "conv d=1\nk=3", "conv d=4\nk=3", "conv d=16\nk=3", "1×1 头\n读末位"]
    rf = [1, 3, 11, 43, 43]
    bars = ax.bar(range(5), rf, color=[C_GRAY, C_BLUE, C_BLUE, C_BLUE, C_GREEN], alpha=0.85)
    for b, v in zip(bars, rf):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.2, str(v), ha="center", fontsize=10)
    ax.set_xticks(range(5)); ax.set_xticklabels(layers, fontsize=9)
    ax.set_ylabel("感受野（步）"); ax.set_ylim(0, 52)
    ax.set_title("膨胀堆叠：感受野 1 → 43 步，全程因果", fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ex = [l[0] for l in losses1]; ey = [l[1] for l in losses1]
    ax.plot(ex, ey, "o-", color=C_RED, lw=1.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("测试 MSE")
    ax.set_title(f"训练进程（Adam，lr=3e-3）：MSE {ey[0]:.3f} → {ey[-1]:.3f}", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "tcn_architecture.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 3：核恢复 =================
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    ax.plot(lags, w_true_n, "k-", lw=2.4, label="真实核 w")
    ax.plot(lags, nz(irf_tcn), "o-", color=C_RED, lw=1.6, ms=4, label=f"TCN-IRF（corr {e1['corr_tcn']:.3f}）")
    ax.plot(lags, nz(w_almon), "s--", color=C_BLUE, lw=1.4, ms=4, label=f"Almon 多项式滞后（{e1['corr_almon']:.3f}）")
    ax.plot(lags, nz(w_ridge), "^--", color=C_ORANGE, lw=1.4, ms=4, label=f"Ridge 原始滞后（{e1['corr_ridge']:.3f}）")
    ax.plot(lags, nz(w_oracle), "d:", color=C_GREEN, lw=1.4, ms=4, label=f"Oracle support Ridge（{e1['corr_oracle']:.3f}）")
    ax.set_xlabel("滞后阶数 k"); ax.set_ylabel("核权重（归一化）")
    ax.set_title("分布式滞后核恢复：形状都能抓，精度看方法", fontsize=11)
    ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "kernel_recovery.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 4：预测对比 =================
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.4))
    seg2 = slice(300, 420)
    t_ax = np.arange(seg2.start, seg2.stop)
    axes[0].plot(t_ax, Yte[seg2], "k-", lw=1.6, label="实际 Y")
    axes[0].plot(t_ax, p_tcn[seg2], color=C_RED, lw=1.3, label=f"TCN（R²={e1['r2_tcn']:.3f}）")
    axes[0].plot(t_ax, p_ridge[seg2], color=C_ORANGE, lw=1.1, ls="--", label=f"Ridge（{e1['r2_ridge']:.3f}）")
    axes[0].plot(t_ax, p_or[seg2], color=C_GREEN, lw=1.1, ls=":", label=f"Oracle（{e1['r2_oracle']:.3f}）")
    axes[0].set_title("测试段预测：120 个样本外窗口", fontsize=11)
    axes[0].legend(fontsize=8.5, ncol=2); axes[0].grid(alpha=0.3)
    names = ["Oracle\nsupport", "Ridge\n原始滞后", "TCN", "Almon\n多项式"]
    vals = [e1["r2_oracle"], e1["r2_ridge"], e1["r2_tcn"], e1["r2_almon"]]
    cols = [C_GREEN, C_ORANGE, C_RED, C_BLUE]
    bars = axes[1].bar(names, vals, color=cols, alpha=0.85)
    for b, v in zip(bars, vals):
        axes[1].text(b.get_x() + b.get_width() / 2, v + 0.004, f"{v:.3f}", ha="center", fontsize=10)
    axes[1].set_ylabel("样本外 R²")
    axes[1].set_title("预测精度：知道滞后支撑的 Oracle 领跑", fontsize=11)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "forecast_vs_actual.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 5：非线性 + 推断 =================
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    ax.plot(lags, w_hi_n, "k-", lw=2.2, label="真实核·高波状态")
    ax.plot(lags, w_lo_n, color=C_GRAY, lw=2.2, label="真实核·低波状态")
    ax.plot(lags, irf_hi_n, "o--", color=C_RED, lw=1.4, ms=3.5, label=f"TCN-IRF·高波（{e2['corr_hi']:.3f}）")
    ax.plot(lags, irf_lo_n, "s--", color=C_BLUE, lw=1.4, ms=3.5, label=f"TCN-IRF·低波（{e2['corr_lo']:.3f}）")
    ax.plot(lags, nz(w_lin2), "^:", color=C_ORANGE, lw=1.3, ms=3.5, label=f"线性 Ridge·合并（高波 {e2['corr_lin_hi']:.3f}）")
    ax.axvline(8, color=C_PURPLE, lw=1, alpha=0.6)
    ax.text(8.4, ax.get_ylim()[1] * 0.8, "lag-8 系数\n随状态翻号", fontsize=8.5, color=C_PURPLE)
    ax.set_xlabel("滞后阶数 k"); ax.set_ylabel("核权重（归一化）")
    ax.set_title("E2 状态依赖核：TCN 分状态恢复，线性模型只给一个平均核", fontsize=11)
    ax.legend(fontsize=8, ncol=2); ax.grid(alpha=0.3)
    ax = axes[1]
    ax.fill_between(lags, ci_lo, ci_hi, color=C_RED, alpha=0.15, label="TCN-IRF 90% bootstrap CI")
    ax.plot(lags, boot.mean(axis=0), "o-", color=C_RED, lw=1.4, ms=3.5, label=f"TCN-IRF 均值（覆盖率 {e3['cover']:.2f}）")
    ax.plot(lags, w_true_n, "k-", lw=2.2, label="真实核")
    ax.plot(lags, w_almon3, "s--", color=C_BLUE, lw=1.3, ms=3.5, label=f"Almon·AR 噪声（corr {e3['corr_almon_ar']:.3f}）")
    ax.set_xlabel("滞后阶数 k"); ax.set_ylabel("核权重（归一化）")
    ax.set_title(f"E3 AR(1) 噪声：Almon 朴素 SE 低估 {se_ratio:.1f}×；TCN bootstrap 覆盖率 {e3['cover']:.2f}", fontsize=11)
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "nonlinear_and_inference.png"), bbox_inches="tight"); plt.close(fig)

    for nm, arr in [("w_true", w_true_n), ("irf", nz(irf_tcn)), ("almon", nz(w_almon)),
                    ("ridge", nz(w_ridge)), ("oracle", nz(w_oracle)), ("irf_hi", irf_hi_n),
                    ("irf_lo", irf_lo_n), ("w_hi", w_hi_n), ("w_lo", w_lo_n),
                    ("boot_mean", boot.mean(axis=0)), ("almon_ar", w_almon3),
                    ("ci_lo", ci_lo), ("ci_hi", ci_hi), ("lin2", nz(w_lin2))]:
        np.save(f"/tmp/oct25e_a_{nm}.npy", arr)
    print("[A] done")


# ============================================================
# 文章 B
# ============================================================
def gen_article_b():
    slug = "cross-asset-macro-momentum-revisit"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20261026)

    regimes = [
        ("2003-2007 大缓和牛市", 0, 1800, [0.00085, 0.00022, 0.00065, 0.00015], 1.0, False),
        ("2008 次贷危机", 1800, 2150, [-0.00180, 0.00035, -0.00130, 0.00012], 2.4, True),
        ("2009-2012 复苏", 2150, 3150, [0.00060, -0.00002, 0.00045, -0.00005], 1.3, False),
        ("2013-2015 震荡衰减", 3150, 3900, [0.00002, 0.00004, -0.00007, 0.00003], 1.0, False),
        ("2016-2019 平稳趋势", 3900, 4900, [0.00050, 0.00004, 0.00006, 0.00001], 0.9, False),
        ("2020 新冠 V 型", 4900, 5080, [0.00055, 0.00015, 0.00042, -0.00035], 3.0, True),
        ("2021-2022 通胀与趋势反转", 5080, 5580, [-0.00022, -0.00025, 0.00036, 0.00022], 1.5, False),
        ("2023-2026 AI 行情", 5580, 7500, [0.00060, -0.00002, 0.00009, -0.00003], 1.1, False),
    ]
    T = 7500
    names = ["股票", "债券", "商品", "外汇"]
    cols = [C_RED, C_BLUE, C_ORANGE, C_GREEN]
    R = np.zeros((T, 4))
    sig_base = np.array([0.010, 0.0035, 0.011, 0.006])
    C_crisis = np.array([
        [1.00, -0.50, 0.85, -0.60],
        [-0.50, 1.00, -0.25, 0.45],
        [0.85, -0.25, 1.00, -0.50],
        [-0.60, 0.45, -0.50, 1.00]])
    C_calm = np.array([
        [1.00, 0.15, 0.35, -0.10],
        [0.15, 1.00, 0.05, 0.20],
        [0.35, 0.05, 1.00, 0.10],
        [-0.10, 0.20, 0.10, 1.00]])
    for (nm, a, b, mu, sc, crisis) in regimes:
        n = b - a
        C = C_crisis if crisis else C_calm
        ev = np.linalg.eigvalsh(C)
        if ev.min() < 0:
            C = C + np.eye(4) * (abs(ev.min()) + 1e-6)
        L = np.linalg.cholesky(C)
        Z = rng.normal(0, 1, (n, 4)) @ L.T
        if not crisis:
            for j in range(4):  # 趋势持续性
                r_ar = np.zeros(n)
                for i in range(1, n):
                    r_ar[i] = 0.20 * r_ar[i - 1] + Z[i, j]
                Z[:, j] = r_ar
        v = np.zeros(n)
        if "新冠" in nm:
            v[:40] = 1.5; v[40:80] = -1.5; v[80:] = 0.0
        R[a:b] = Z * sig_base * sc + np.array(mu) * (1 + 1.5 * v[:, None])

    # ---------- 12-1 月时序动量 + 腿级波动目标（经典 TSMOM 构造） ----------
    LOOK, SKIP = 250, 21
    cum = np.cumsum(R, axis=0)
    mom = np.zeros((T, 4))
    for t in range(LOOK, T):
        mom[t] = cum[t - SKIP] - cum[t - LOOK]
    m1 = np.zeros((T, 4))
    for t in range(LOOK, T):
        m1[t] = cum[t] - cum[t - 21]
    # 每腿 60 日年化波动
    sig = np.zeros((T, 4))
    for t in range(LOOK, T):
        for j in range(4):
            sig[t, j] = R[max(0, t - 60):t, j].std() * np.sqrt(252)
    # 信号 = 动量 / (年化波动 × 窗口尺度因子)，两种速度的信号归一到同尺度（无漂移时 sd=1）
    def make_ts(look, skip):
        n_days = look - skip
        norm = np.sqrt(n_days * 1.5 / 252.0)  # AR(0.2) 下窗口标准差（年化单位）
        m = np.zeros((T, 4))
        for t in range(look, T):
            m[t] = cum[t - skip] - cum[t - look]
        return np.where(sig > 1e-9, m / (np.where(sig > 1e-9, sig, 1.0) * norm), 0.0)

    ts = make_ts(250, 21)    # 12-1 月慢信号
    ts1 = make_ts(21, 0)     # 1 月动量（反转叠加用）
    ts_f = make_ts(63, 21)   # 3-1 月快信号
    POS_K, POS_CAP = 0.2, 0.75

    def run(sig_full, cost_bps=5.0):
        pos = np.zeros((T, 4))
        for t in range(LOOK, T):
            pos[t] = np.clip(sig_full[t] * POS_K, -POS_CAP, POS_CAP)
        pret = np.zeros(T)
        pret[1:] = (pos[:-1] * R[1:]).sum(axis=1)
        turn = np.abs(np.diff(pos, axis=0)).sum(axis=1)
        pret[1:] -= turn * cost_bps / 1e4
        return pret, pos, np.ones(T)

    def stats(r):
        ann = r.mean() * 252; vol = r.std() * np.sqrt(252)
        nav = np.cumprod(1 + r)
        dd = nav / np.maximum.accumulate(nav) - 1
        return {"ann": float(ann), "vol": float(vol), "sharpe": float(ann / vol) if vol > 0 else 0,
                "total": float(nav[-1] - 1), "mdd": float(dd.min())}

    pret_mom, pos_mom, sc_mom = run(ts)
    pret_filt, pos_filt, _ = run(np.where(np.abs(ts) > 0.75, ts, 0.0))
    # 反转叠加：signal - 0.5 × 1 月动量/波动
    pret_rev, pos_rev, _ = run(ts - 0.5 * ts1)
    # 快信号 3-1 月（LOOK=63, SKIP=21，已按窗口尺度归一化）
    pret_fast, pos_fast, _ = run(ts_f)
    r_bh = R.mean(axis=1)

    s_mom, s_filt, s_rev = stats(pret_mom[LOOK:]), stats(pret_filt[LOOK:]), stats(pret_rev[LOOK:])
    s_fast = stats(pret_fast[LOOK:])
    s_fast_gross = stats(run(ts_f, cost_bps=0.0)[0][LOOK:])
    s_bh = stats(r_bh[LOOK:])

    def run_single(j):
        pos = np.zeros(T)
        for t in range(LOOK, T):
            pos[t] = np.clip(ts[t, j] * POS_K, -POS_CAP, POS_CAP)
        pr = np.zeros(T)
        pr[1:] = pos[:-1] * R[1:, j]
        pr[1:] -= np.abs(np.diff(pos)) * 5.0 / 1e4
        return pr
    s_single = [stats(run_single(j)[LOOK:]) for j in range(4)]

    sub_defs = [
        ("2003-2007 牛市", 0, 1800), ("2008 危机", 1800, 2150), ("2009-2012 复苏", 2150, 3150),
        ("2013-2015 震荡", 3150, 3900), ("2016-2019 趋势", 3900, 4900), ("2020 新冠", 4900, 5080),
        ("2021-2022 反转", 5080, 5580), ("2023-2026 AI", 5580, 7500),
    ]
    subs = []
    for nm, a, b in sub_defs:
        sm, sb = stats(pret_mom[a:b]), stats(r_bh[a:b])
        sf = stats(pret_fast[a:b])
        subs.append({"name": nm, "mom_sharpe": sm["sharpe"], "bh_sharpe": sb["sharpe"],
                     "mom_ret": sm["total"], "bh_ret": sb["total"], "mom_mdd": sm["mdd"],
                     "fast_sharpe": sf["sharpe"], "fast_ret": sf["total"]})

    cost_rows = []
    for cbps in [0, 5, 20, 50]:
        st = stats(run(ts, cost_bps=cbps)[0][LOOK:])
        cost_rows.append({"bps": cbps, "sharpe": st["sharpe"], "ann": st["ann"], "total": st["total"]})
    cost_rows_fast = []
    for cbps in [0, 5, 20, 50]:
        st = stats(run(ts_f, cost_bps=cbps)[0][LOOK:])
        cost_rows_fast.append({"bps": cbps, "sharpe": st["sharpe"], "ann": st["ann"], "total": st["total"]})

    corr_rows = []
    for nm, a, b, *_ in regimes:
        sub = R[a + 60:b]
        Cm = np.corrcoef(sub.T)
        corr_rows.append({"name": nm, "eq_com": float(Cm[0, 2]), "eq_bd": float(Cm[0, 1]),
                          "mean_abs": float(np.abs(Cm[np.triu_indices(4, 1)]).mean())})

    win = 252 * 3
    roll_t, roll_s = [], []
    for t in range(LOOK + win, T, 21):
        roll_t.append(t); roll_s.append(stats(pret_mom[t - win:t])["sharpe"])
    roll_t, roll_s = np.array(roll_t), np.array(roll_s)

    gross = np.abs(pos_mom).sum(axis=1)
    turn_daily = np.abs(np.diff(pos_mom, axis=0)).sum(axis=1)
    turn_fast = np.abs(np.diff(pos_fast, axis=0)).sum(axis=1)
    n_beat = int(np.sum([s["mom_sharpe"] > s["bh_sharpe"] for s in subs]))
    calm_abs = np.mean([c["mean_abs"] for c in corr_rows if "危机" not in c["name"] and "新冠" not in c["name"]])
    crisis_abs = np.mean([c["mean_abs"] for c in corr_rows if "危机" in c["name"] or "新冠" in c["name"]])

    RESULTS["B"] = {
        "mom": s_mom, "filt": s_filt, "rev": s_rev, "fast": s_fast, "fast_gross": s_fast_gross, "bh": s_bh,
        "single": [{"name": names[j], **s_single[j]} for j in range(4)],
        "subs": subs, "costs": cost_rows, "costs_fast": cost_rows_fast, "corr": corr_rows,
        "lev_mean": float(gross[LOOK:].mean()), "turn_ann": float(turn_daily[LOOK:].mean() * 252),
        "turn_fast_ann": float(turn_fast[LOOK:].mean() * 252),
        "worst_roll": float(roll_s.min()), "worst_roll_t": int(roll_t[int(np.argmin(roll_s))]),
        "best_roll": float(roll_s.max()), "n_beat": n_beat,
        "calm_abs_corr": float(calm_abs), "crisis_abs_corr": float(crisis_abs),
    }

    # ================= 图 1：资产路径 =================
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.8))
    for j in range(4):
        ax = axes[j // 2, j % 2]
        cumj = np.cumprod(1 + R[:, j])
        ax.plot(cumj, color=cols[j], lw=1.2)
        for (nm, a, b, mu, sc, crisis) in regimes:
            ax.axvspan(a, b, color="#f5b7b1" if crisis else "#eaf2f8", alpha=0.45, lw=0)
        ax.set_title(f"{names[j]}（年化 {R[:, j].mean()*252*100:.1f}%，波动 {R[:, j].std()*np.sqrt(252)*100:.1f}%）", fontsize=10.5)
        ax.set_ylabel("净值")
        if j == 0:
            ax.text(0.02, 0.92, "红色=危机区", transform=ax.transAxes, fontsize=8.5, color=C_RED)
        ax.grid(alpha=0.3)
    fig.suptitle("合成四资产 20 年日频面板（7500 个交易日，8 个 regime）", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "asset_paths.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 2：信号构建 =================
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=True)
    seg = slice(4400, 4730)  # 2020 新冠区间
    t_ax = np.arange(seg.start, seg.stop)
    for j in range(4):
        axes[0].plot(t_ax, ts[seg, j], lw=1.3, color=cols[j], label=names[j])
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].axvline(4550, color=C_RED, lw=1.2, ls="--")
    axes[0].text(4553, axes[0].get_ylim()[1] * 0.6, "新冠 shock", fontsize=8.5, color=C_RED)
    axes[0].set_ylabel("归一化 TSMOM 信号")
    axes[0].set_title("TSMOM 信号：每腿 12-1 月动量 ÷（60 日年化波动 × 窗口尺度），两种速度同尺度可比", fontsize=11)
    axes[0].legend(fontsize=8.5, ncol=4); axes[0].grid(alpha=0.3)
    for j in range(4):
        axes[1].plot(t_ax, pos_mom[seg, j], lw=1.3, color=cols[j], label=names[j])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_ylabel("腿仓位（风险单位）"); axes[1].set_xlabel("交易日")
    axes[1].set_title("腿仓位 = clip(0.2×信号, ±0.75)：弱信号自动缩仓，强信号触顶", fontsize=11)
    axes[1].legend(fontsize=8.5, ncol=4); axes[1].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "signal_construction.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 3：组合表现 =================
    fig, ax = plt.subplots(figsize=(11, 5.2))
    cum_bh = (np.cumprod(1 + r_bh) - 1) * 100
    cum_mom = (np.cumprod(1 + pret_mom) - 1) * 100
    cum_filt = (np.cumprod(1 + pret_filt) - 1) * 100
    cum_rev = (np.cumprod(1 + pret_rev) - 1) * 100
    cum_fast = (np.cumprod(1 + pret_fast) - 1) * 100
    ax.plot(cum_bh, color=C_GRAY, lw=1.3, label=f"等权 buy&hold（Sharpe {s_bh['sharpe']:.2f}，MDD {s_bh['mdd']*100:.1f}%）")
    ax.plot(cum_mom, color=C_RED, lw=1.6, label=f"宏观动量 12-1（Sharpe {s_mom['sharpe']:.2f}，MDD {s_mom['mdd']*100:.1f}%）")
    ax.plot(cum_fast, color=C_PURPLE, lw=1.4, ls="-.", label=f"宏观动量 3-1 快信号（{s_fast['sharpe']:.2f}，毛 {s_fast_gross['sharpe']:.2f}）")
    ax.plot(cum_filt, color=C_BLUE, lw=1.2, ls="--", label=f"12-1+趋势强度过滤（{s_filt['sharpe']:.2f}）")
    ax.plot(cum_rev, color=C_GREEN, lw=1.2, ls=":", label=f"12-1+1 月反转叠加（{s_rev['sharpe']:.2f}）")
    for (nm, a, b, mu, sc, crisis) in regimes:
        ax.axvspan(a, b, color="#f5b7b1" if crisis else "#eaf2f8", alpha=0.4, lw=0)
    ax.set_ylabel("累计收益（%）")
    ax.set_title("跨资产宏观动量 vs buy&hold：20 年样本（成本 5bps）", fontsize=12)
    ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "performance.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 4：子区间 + 滚动 =================
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    xpos = np.arange(len(subs)); wd = 0.38
    ax.bar(xpos - wd / 2, [s["mom_sharpe"] for s in subs], wd, color=C_RED, alpha=0.85, label="宏观动量")
    ax.bar(xpos + wd / 2, [s["bh_sharpe"] for s in subs], wd, color=C_GRAY, alpha=0.85, label="buy&hold")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([s["name"].replace(" ", "\n") for s in subs], fontsize=7.5)
    ax.set_ylabel("子区间 Sharpe")
    ax.set_title(f"子区间分解：动量 {n_beat}/8 个区间跑赢 buy&hold", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    ax = axes[1]
    ax.plot(roll_t, roll_s, color=C_RED, lw=1.5)
    ax.axhline(0, color="k", lw=0.8)
    ax.fill_between(roll_t, 0, roll_s, where=roll_s < 0, color=C_GREEN, alpha=0.25)
    ax.fill_between(roll_t, 0, roll_s, where=roll_s >= 0, color=C_RED, alpha=0.18)
    ax.set_xlabel("窗口结束日"); ax.set_ylabel("滚动 36M Sharpe")
    ax.set_title(f"滚动 36 个月 Sharpe：最差 {roll_s.min():.2f}（约第 {int(roll_t[int(np.argmin(roll_s))])} 日）", fontsize=11)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "regime_decomposition.png"), bbox_inches="tight"); plt.close(fig)

    # ================= 图 5：成本 + 相关性 =================
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))
    ax = axes[0]
    bps = [c["bps"] for c in cost_rows]
    sh = [c["sharpe"] for c in cost_rows]
    shf = [c["sharpe"] for c in cost_rows_fast]
    ax.plot(bps, sh, "o-", color=C_RED, lw=1.8, label="12-1 慢信号")
    ax.plot(bps, shf, "s-", color=C_PURPLE, lw=1.8, label="3-1 快信号")
    for c in cost_rows:
        ax.annotate(f"{c['sharpe']:.2f}", (c["bps"], c["sharpe"]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)
    for c in cost_rows_fast:
        ax.annotate(f"{c['sharpe']:.2f}", (c["bps"], c["sharpe"]), textcoords="offset points",
                    xytext=(0, -14), ha="center", fontsize=9)
    ax.axhline(0, color="k", lw=0.8, ls="--")
    ax.set_xlabel("单边成本（bps）"); ax.set_ylabel("Sharpe")
    ax.set_title("成本敏感性：快信号毛 Sharpe 高，但成本斜率也陡", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    ax = axes[1]
    xpos = np.arange(len(corr_rows))
    ax.bar(xpos - 0.2, [c["eq_com"] for c in corr_rows], 0.4, color=C_RED, alpha=0.85, label="股票-商品")
    ax.bar(xpos + 0.2, [c["eq_bd"] for c in corr_rows], 0.4, color=C_BLUE, alpha=0.85, label="股票-债券")
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(xpos); ax.set_xticklabels([c["name"].replace(" ", "\n") for c in corr_rows], fontsize=7.5)
    ax.set_ylabel("相关系数")
    ax.set_title("腿相关性分 regime：危机区股票-商品抱团到 0.85+", fontsize=11)
    ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "costs_correlation.png"), bbox_inches="tight"); plt.close(fig)

    np.save("/tmp/oct25e_b_cum_bh.npy", cum_bh)
    np.save("/tmp/oct25e_b_cum_mom.npy", cum_mom)
    print("[B] done")


if __name__ == "__main__":
    t0 = time.time()
    gen_article_a()
    gen_article_b()
    print(f"total {time.time()-t0:.0f}s")
    print(json.dumps(RESULTS, ensure_ascii=False, indent=1, default=float))
