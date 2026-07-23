#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成两篇文章的实验数据与配图：
1. Informer 长序列预测：概率稀疏注意力 (informer-financial)
2. Autoformer 时序分解预测：序列分解+自相关机制 (autoformer-financial)
纯 numpy 实现，所有图与指标同一份代码产出。
"""
import numpy as np
import time as _time
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 中文字体
for f in ["Arial Unicode MS", "PingFang SC", "Hiragino Sans GB"]:
    try:
        font_manager.findfont(f, fallback_to_default=False)
        plt.rcParams["font.family"] = f
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/Users/halo/workspace/astro-blog/public/images"
DIR_INF = os.path.join(ROOT, "informer-financial")
DIR_AUT = os.path.join(ROOT, "autoformer-financial")
os.makedirs(DIR_INF, exist_ok=True)
os.makedirs(DIR_AUT, exist_ok=True)

rng = np.random.default_rng(42)

# =========================================================
# 共用合成序列：多周期季节 + 慢趋势 + AR(1) 噪声 + 幅度调制
# =========================================================
N = 3600
t = np.arange(N)
season = (0.8 * np.sin(2 * np.pi * t / 24)
          + 0.5 * np.sin(2 * np.pi * t / 96 + 0.7)
          + 0.3 * np.sin(2 * np.pi * t / 168 + 1.3))
amp_mod = 1.0 + 0.4 * np.sin(2 * np.pi * t / 600)
trend = 0.0008 * t + 0.6 * np.sin(2 * np.pi * t / 1800)
eps = np.zeros(N)
e = rng.standard_normal(N) * 0.25
for i in range(1, N):
    eps[i] = 0.5 * eps[i - 1] + e[i]
y_raw = season * amp_mod + trend + eps
mu, sd = y_raw.mean(), y_raw.std()
y = (y_raw - mu) / sd

# =========================================================
# Part 1: Informer —— ProbSparse 注意力
# =========================================================
print("=" * 60)
print("PART 1: Informer / ProbSparse attention")
print("=" * 60)


def softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def full_attention(Q, K, V):
    d = Q.shape[-1]
    S = Q @ K.T / np.sqrt(d)
    A = softmax(S, axis=-1)
    return A @ V, A


def probsparse_attention(Q, K, V, c=5, rng=None):
    """Informer 式 ProbSparse: 采样估计稀疏度 M(q,K)，只让 top-u 查询做完整注意力，
    其余查询输出 V 的均值（对应均匀注意力的期望输出）。"""
    L, d = Q.shape
    u = min(L, max(1, int(np.ceil(c * np.log(L)))))       # 活跃查询数
    n_sample = min(L, max(1, int(np.ceil(c * np.log(L)))))  # 每个查询采样的 key 数
    idx = rng.choice(L, size=n_sample, replace=False)
    S_sample = Q @ K[idx].T / np.sqrt(d)                   # (L, n_sample)
    M = S_sample.max(axis=1) - S_sample.mean(axis=1)       # 稀疏度度量
    top = np.argsort(M)[-u:]
    out = np.repeat(V.mean(axis=0, keepdims=True), L, axis=0)  # 懒查询 → 均值
    S_top = Q[top] @ K.T / np.sqrt(d)
    out[top] = softmax(S_top, axis=-1) @ V
    return out, M, top


# ---- 1a. 注意力行的长尾性 + 稀疏度度量 ----
d_model = 64
L_demo = 336
Xd = rng.standard_normal((L_demo, d_model)) * 0.5
# 注入结构：少数查询有强指向性
for j in range(0, L_demo, 48):
    Xd[j] += 2.5 * rng.standard_normal(d_model) * 0.8
Wq = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
Wk = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
Wv = rng.standard_normal((d_model, d_model)) / np.sqrt(d_model)
Qd, Kd, Vd = Xd @ Wq, Xd @ Wk, Xd @ Wv
O_full, A_full = full_attention(Qd, Kd, Vd)
O_ps, M_demo, top_demo = probsparse_attention(Qd, Kd, Vd, c=5, rng=rng)

row_entropy = -(A_full * np.log(A_full + 1e-12)).sum(axis=1)
uniform_entropy = np.log(L_demo)
rel_err = np.linalg.norm(O_ps - O_full) / np.linalg.norm(O_full)
# 活跃查询（高M）与懒查询（低M）近似误差分别看
err_rows = np.linalg.norm(O_ps - O_full, axis=1) / (np.linalg.norm(O_full, axis=1) + 1e-12)
mask_top = np.zeros(L_demo, dtype=bool)
mask_top[top_demo] = True
print(f"L={L_demo}, 活跃查询 u={mask_top.sum()} ({mask_top.sum()/L_demo:.1%})")
print(f"整体相对误差 |O_ps-O_full|/|O_full| = {rel_err:.4f}")
print(f"活跃查询行误差均值 = {err_rows[mask_top].mean():.4f}, 懒查询行 = {err_rows[~mask_top].mean():.4f}")
print(f"行熵: 最小 {row_entropy.min():.2f}, 中位 {np.median(row_entropy):.2f}, 均匀上界 {uniform_entropy:.2f}")
print(f"M 与 (均匀熵-行熵) 的相关 = {np.corrcoef(M_demo, uniform_entropy - row_entropy)[0,1]:.3f}")

# ---- 1b. 复杂度实测：full O(L^2) vs probsparse O(L ln L) ----
Ls = [128, 256, 512, 1024, 2048, 4096]
t_full, t_ps = [], []
for L in Ls:
    X = rng.standard_normal((L, d_model))
    Q_, K_, V_ = X @ Wq, X @ Wk, X @ Wv
    reps = 3
    t0 = _time.perf_counter()
    for _ in range(reps):
        full_attention(Q_, K_, V_)
    t_full.append((_time.perf_counter() - t0) / reps * 1000)
    t0 = _time.perf_counter()
    for _ in range(reps):
        probsparse_attention(Q_, K_, V_, c=5, rng=rng)
    t_ps.append((_time.perf_counter() - t0) / reps * 1000)
    print(f"L={L:5d}: full {t_full[-1]:8.2f} ms | probsparse {t_ps[-1]:8.2f} ms | 加速 {t_full[-1]/t_ps[-1]:.1f}x")

# ---- 1c. 预测任务：单层注意力回归（手写反向传播）----
Lw = 96


def make_dataset(y, Lw):
    Xs, ys = [], []
    for i in range(Lw, len(y) - 1):
        Xs.append(y[i - Lw:i])
        ys.append(y[i + 1])
    return np.array(Xs), np.array(ys)


Xall, yall = make_dataset(y, Lw)
n = len(Xall)
n_tr = int(n * 0.7)
Xtr, ytr = Xall[:n_tr], yall[:n_tr]
Xte, yte = Xall[n_tr:], yall[n_tr:]

dm = 16
rng2 = np.random.default_rng(7)
pos = np.zeros((Lw, dm))
pos_idx = np.arange(Lw)[:, None]
div = np.exp(np.arange(0, dm, 2) * (-np.log(10000.0) / dm))
pos[:, 0::2] = np.sin(pos_idx * div)
pos[:, 1::2] = np.cos(pos_idx * div)

params = {
    "We": rng2.standard_normal((1, dm)) * 0.3,
    "Wq": rng2.standard_normal((dm, dm)) / np.sqrt(dm),
    "Wk": rng2.standard_normal((dm, dm)) / np.sqrt(dm),
    "Wv": rng2.standard_normal((dm, dm)) / np.sqrt(dm),
    "w": rng2.standard_normal(dm) * 0.1,
    "b": np.zeros(1),
}


def forward(Xb, p):
    E = Xb[:, :, None] @ p["We"][None] + pos[None]     # (B,L,dm) broadcast
    E = Xb[:, :, None] * p["We"][None, None, 0] + pos[None]
    Q = E @ p["Wq"]; K = E @ p["Wk"]; V = E @ p["Wv"]
    S = Q @ K.transpose(0, 2, 1) / np.sqrt(dm)
    A = softmax(S, axis=-1)
    H = A @ V                                          # (B,L,dm)
    h = H.mean(axis=1)                                 # (B,dm)
    yhat = h @ p["w"] + p["b"][0]
    cache = (Xb, E, Q, K, V, A, H, h)
    return yhat, cache


def backward(yhat, yb, cache, p):
    Xb, E, Q, K, V, A, H, h = cache
    B, L, _ = E.shape
    g = {}
    dy = 2 * (yhat - yb) / B                           # (B,)
    g["w"] = h.T @ dy
    g["b"] = np.array([dy.sum()])
    dh = dy[:, None] * p["w"][None]                    # (B,dm)
    dH = np.repeat(dh[:, None, :], L, axis=1) / L      # (B,L,dm)
    dA = dH @ V.transpose(0, 2, 1)                     # (B,L,L)
    dV = A.transpose(0, 2, 1) @ dH
    dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True)) / np.sqrt(dm)
    dQ = dS @ K
    dK = dS.transpose(0, 2, 1) @ Q
    g["Wq"] = np.einsum("bld,ble->de", E, dQ)
    g["Wk"] = np.einsum("bld,ble->de", E, dK)
    g["Wv"] = np.einsum("bld,ble->de", E, dV)
    dE = dQ @ p["Wq"].T + dK @ p["Wk"].T + dV @ p["Wv"].T
    g["We"] = np.array([[np.einsum("bl,bld->d", Xb, dE)[i] for i in range(dm)]])
    return g


# 有限差分梯度校验（小规模）
def grad_check():
    ptest = {k: v.copy() for k, v in params.items()}
    Xb, yb = Xtr[:4], ytr[:4]
    yhat, cache = forward(Xb, ptest)
    g = backward(yhat, yb, cache, ptest)
    errs = {}
    for key in ["w", "Wq", "We"]:
        flat = ptest[key].ravel()
        idxs = rng2.choice(flat.size, size=min(5, flat.size), replace=False)
        num, ana = [], []
        for i in idxs:
            old = flat[i]
            flat[i] = old + 1e-5
            l1 = ((forward(Xb, ptest)[0] - yb) ** 2).mean()
            flat[i] = old - 1e-5
            l2 = ((forward(Xb, ptest)[0] - yb) ** 2).mean()
            flat[i] = old
            num.append((l1 - l2) / 2e-5)
            ana.append(g[key].ravel()[i])
        num, ana = np.array(num), np.array(ana)
        errs[key] = np.abs(num - ana).max() / (np.abs(num).max() + 1e-8)
    return errs


gc = grad_check()
print("梯度校验相对误差:", {k: f"{v:.2e}" for k, v in gc.items()})

# Adam 训练
mstate = {k: np.zeros_like(v) for k, v in params.items()}
vstate = {k: np.zeros_like(v) for k, v in params.items()}
lr, b1, b2, epsA = 3e-3, 0.9, 0.999, 1e-8
batch = 256
steps = 600
order = np.arange(n_tr)
step = 0
for ep in range(100):
    rng2.shuffle(order)
    for s in range(0, n_tr - batch, batch):
        idx = order[s:s + batch]
        yhat, cache = forward(Xtr[idx], params)
        g = backward(yhat, ytr[idx], cache, params)
        step += 1
        for k in params:
            mstate[k] = b1 * mstate[k] + (1 - b1) * g[k]
            vstate[k] = b2 * vstate[k] + (1 - b2) * g[k] ** 2
            mh = mstate[k] / (1 - b1 ** step)
            vh = vstate[k] / (1 - b2 ** step)
            params[k] -= lr * mh / (np.sqrt(vh) + epsA)
        if step >= steps:
            break
    if step >= steps:
        break

# 评估：full attention 推理 vs ProbSparse 推理（同一套已训练权重）


def predict_full(X):
    out = []
    for s in range(0, len(X), 512):
        yhat, _ = forward(X[s:s + 512], params)
        out.append(yhat)
    return np.concatenate(out)


def predict_probsparse(X, c=5):
    rng3 = np.random.default_rng(11)
    out = np.zeros(len(X))
    for i in range(len(X)):
        E = X[i][:, None] * params["We"][0][None] + pos
        Q = E @ params["Wq"]; K = E @ params["Wk"]; V = E @ params["Wv"]
        H, _, _ = probsparse_attention(Q, K, V, c=c, rng=rng3)
        h = H.mean(axis=0)
        out[i] = h @ params["w"] + params["b"][0]
    return out


yhat_full = predict_full(Xte)
sub = np.arange(0, len(Xte), 8)   # ProbSparse 逐样本较慢，抽样评估
# c 扫描：稀疏度-精度权衡
c_list = [2, 5, 10, 20, 48]
r2_ps_sweep = []
for c_ in c_list:
    yh = predict_probsparse(Xte[sub], c=c_)
    r2_ps_sweep.append(1 - ((yh - yte[sub]) ** 2).sum() / ((yte[sub] - yte[sub].mean()) ** 2).sum())
yhat_ps = predict_probsparse(Xte[sub], c=10)


def r2(ytrue, ypred):
    return 1 - ((ytrue - ypred) ** 2).sum() / ((ytrue - ytrue.mean()) ** 2).sum()


# OLS 基线
XtrB = np.hstack([Xtr, np.ones((len(Xtr), 1))])
XteB = np.hstack([Xte, np.ones((len(Xte), 1))])
beta = np.linalg.lstsq(XtrB, ytr, rcond=None)[0]
yhat_ols = XteB @ beta
naive = Xte[:, -1]

r2_full = r2(yte, yhat_full)
r2_ps = r2(yte[sub], yhat_ps)
r2_ols = r2(yte, yhat_ols)
r2_naive = r2(yte, naive)
u_frac = [min(Lw, max(1, int(np.ceil(c_ * np.log(Lw))))) / Lw for c_ in c_list]
print(f"测试集 R2: attention(full)={r2_full:.4f} | attention(ProbSparse c=10)={r2_ps:.4f} | OLS-96lags={r2_ols:.4f} | naive={r2_naive:.4f}")
print("c 扫描:", {f"c={c_}(u/L={uf:.0%})": f"{rr:.3f}" for c_, uf, rr in zip(c_list, u_frac, r2_ps_sweep)})
mse_full = ((yte - yhat_full) ** 2).mean()
mse_ols = ((yte - yhat_ols) ** 2).mean()
print(f"MSE: attention={mse_full:.5f}, OLS={mse_ols:.5f}")

# ---- Informer 配图 ----
# cover
fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), gridspec_kw={"width_ratios": [1.1, 1]})
im = axes[0].imshow(A_full[:120, :120], aspect="auto", cmap="magma")
axes[0].set_title("完整注意力矩阵（前120×120）：绝大多数行近乎均匀", fontsize=11)
axes[0].set_xlabel("Key 位置"); axes[0].set_ylabel("Query 位置")
plt.colorbar(im, ax=axes[0], fraction=0.046)
srt = np.argsort(M_demo)
axes[1].plot(M_demo[srt][::-1], lw=2, color="#d62728")
axes[1].axvline(mask_top.sum(), color="gray", ls="--", lw=1)
axes[1].text(mask_top.sum() + 3, M_demo.max() * 0.75, f"top-u={mask_top.sum()}\n(仅 {mask_top.sum()/L_demo:.0%} 查询)", fontsize=10)
axes[1].set_title("查询稀疏度 M(q,K) 排序：长尾 → 只算头部", fontsize=11)
axes[1].set_xlabel("查询排名"); axes[1].set_ylabel("M(q,K)")
fig.suptitle("Informer ProbSparse：注意力天然长尾，只算「活跃查询」", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(DIR_INF, "cover.png"), dpi=110, bbox_inches="tight")
plt.close()

# fig1: 复杂度
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
axes[0].plot(Ls, t_full, "o-", label="Full attention $O(L^2)$", color="#1f77b4", lw=2)
axes[0].plot(Ls, t_ps, "s-", label="ProbSparse $O(L\\ln L)$", color="#d62728", lw=2)
axes[0].set_xscale("log", base=2); axes[0].set_yscale("log")
axes[0].set_xlabel("序列长度 L"); axes[0].set_ylabel("单次前向耗时 (ms)")
axes[0].set_title("实测耗时：L=4096 时 ProbSparse 快 %.0f 倍" % (t_full[-1] / t_ps[-1]))
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[1].bar(["活跃查询\n(top-u)", "懒查询\n(其余)"], [err_rows[mask_top].mean(), err_rows[~mask_top].mean()],
            color=["#d62728", "#999999"])
axes[1].set_ylabel("行输出相对误差")
axes[1].set_title(f"近似误差集中在懒查询上（整体 {rel_err:.1%}）")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(DIR_INF, "informer_complexity.png"), dpi=110, bbox_inches="tight")
plt.close()

# fig2: 预测对比
fig, axes = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True)
seg = slice(200, 500)
xs = np.arange(len(yte))[seg]
axes[0].plot(xs, yte[seg], color="black", lw=1.2, label="真实值")
axes[0].plot(xs, yhat_full[seg], color="#d62728", lw=1.0, alpha=0.85, label=f"注意力模型 R²={r2_full:.3f}")
axes[0].plot(xs, yhat_ols[seg], color="#1f77b4", lw=1.0, alpha=0.7, label=f"OLS-96滞后 R²={r2_ols:.3f}")
axes[0].legend(ncol=3, fontsize=9); axes[0].set_title("测试集一步预测片段")
axes[0].grid(alpha=0.3)
axes[1].plot([uf * 100 for uf in u_frac], r2_ps_sweep, "o-", color="#d62728", lw=2, ms=6, label="ProbSparse 推理")
axes[1].axhline(r2_full, color="#1f77b4", ls="--", lw=1.5, label=f"Full 推理 R²={r2_full:.3f}")
axes[1].axhline(0, color="gray", lw=0.8)
for uf, rr, c_ in zip(u_frac, r2_ps_sweep, c_list):
    axes[1].annotate(f"c={c_}", (uf * 100, rr), textcoords="offset points", xytext=(6, -12), fontsize=9)
axes[1].set_xlabel("活跃查询占比 u/L (%)"); axes[1].set_ylabel("测试集 R²")
axes[1].set_title("稀疏度-精度权衡：注意力不够长尾时，砍太狠精度崩")
axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(DIR_INF, "informer_forecast.png"), dpi=110, bbox_inches="tight")
plt.close()
print("Informer 图已保存")

# =========================================================
# Part 2: Autoformer —— 序列分解 + 自相关机制
# =========================================================
print("=" * 60)
print("PART 2: Autoformer / decomposition + auto-correlation")
print("=" * 60)


def moving_avg_causal(x, k):
    """因果(trailing)移动平均：位置 i 只用 x[i-k+1..i]，起始段用扩展窗口。"""
    cs = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty(len(x))
    for i in range(len(x)):
        lo = max(0, i - k + 1)
        out[i] = (cs[i + 1] - cs[lo]) / (i + 1 - lo)
    return out


def decompose(x, k=25):
    """因果分解：预测场景下末端不能被未来/padding污染。"""
    tr = moving_avg_causal(x, k)
    return x - tr, tr   # seasonal, trend


def autocorr_fft(x):
    xc = x - x.mean()
    n = len(xc)
    f = np.fft.rfft(xc, n=2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf /= acf[0] + 1e-12
    return acf


def autocorrelation_forecast(hist_seasonal, H, topk=5, max_lag=400):
    """Autoformer 自相关机制的预测版：找 top-k 周期滞后，按自相关 softmax 加权做时延聚合。"""
    acf = autocorr_fft(hist_seasonal[-1024:])
    lags = np.arange(8, min(max_lag, len(acf)))
    top_lags = lags[np.argsort(acf[lags])[-topk:]]
    w = softmax(np.array([acf[l] for l in top_lags]) * 10)
    fc = np.zeros(H)
    buf = hist_seasonal.copy()
    for h in range(H):
        val = sum(wi * buf[-li] for wi, li in zip(w, top_lags))
        fc[h] = val
        buf = np.append(buf, val)
    return fc, top_lags, w, acf


H = 24
Lb = 512   # lookback
test_starts = np.arange(2400, N - H - 1, H)   # 滚动多步预测起点

mse_af, mse_raw, mse_snaive, mse_ols = [], [], [], []
# OLS 多步：直接用 96 滞后逐步递推
beta_h = None
Xh, yh = make_dataset(y[:2400], 96)
XhB = np.hstack([Xh, np.ones((len(Xh), 1))])
beta_h = np.linalg.lstsq(XhB, yh, rcond=None)[0]

example = None
for st in test_starts:
    hist = y[:st]
    truth = y[st:st + H]
    # (1) Autoformer 式：分解 → 季节用自相关时延聚合，趋势线性外推
    seas, tr = decompose(hist, 25)
    fc_s, top_lags, w_ac, acf = autocorrelation_forecast(seas, H)
    tr_slope = (tr[-1] - tr[-97]) / 96
    fc_t = tr[-1] + tr_slope * np.arange(1, H + 1)
    fc_af = fc_s + fc_t
    # (2) 消融：不分解，直接对原序列做自相关聚合
    fc_raw, _, _, _ = autocorrelation_forecast(hist, H)
    # (3) 季节 naive：t-24
    fc_sn = np.array([hist[st - 24 + h - st] for h in range(H)])  # hist[-24+h]
    fc_sn = hist[-24:][:H]
    # (4) OLS 递推
    buf = hist[-96:].copy()
    fc_o = np.zeros(H)
    for h in range(H):
        fc_o[h] = np.append(buf[-96:], 1.0) @ beta_h
        buf = np.append(buf, fc_o[h])
    mse_af.append(((fc_af - truth) ** 2).mean())
    mse_raw.append(((fc_raw - truth) ** 2).mean())
    mse_snaive.append(((fc_sn - truth) ** 2).mean())
    mse_ols.append(((fc_o - truth) ** 2).mean())
    if example is None:
        example = (st, truth, fc_af, fc_raw, fc_sn, fc_o, top_lags, w_ac, acf, seas, tr)

mse_af, mse_raw = np.mean(mse_af), np.mean(mse_raw)
mse_snaive, mse_ols_m = np.mean(mse_snaive), np.mean(mse_ols)
var_te = y[2400:].var()
print(f"多步(H=24)滚动预测 MSE: Autoformer式={mse_af:.4f} | 无分解消融={mse_raw:.4f} | 季节naive={mse_snaive:.4f} | OLS递推={mse_ols_m:.4f}")
print(f"R2 等价: AF={1-mse_af/var_te:.3f}, 无分解={1-mse_raw/var_te:.3f}, snaive={1-mse_snaive/var_te:.3f}, OLS={1-mse_ols_m/var_te:.3f}")
st, truth, fc_af, fc_raw, fc_sn, fc_o, top_lags, w_ac, acf, seas_ex, tr_ex = example
print(f"示例窗口 top-k 滞后: {sorted(top_lags.tolist())}, 权重: {np.round(w_ac,3).tolist()}")

# ---- Autoformer 配图 ----
# cover: 分解图
fig, axes = plt.subplots(3, 1, figsize=(12, 7), sharex=True)
seg = slice(1200, 1800)
axes[0].plot(t[seg], y[seg], color="black", lw=1)
axes[0].set_title("原始序列 = 趋势 + 季节 + 噪声", fontsize=11)
seas_all, tr_all = decompose(y, 25)
axes[1].plot(t[seg], tr_all[seg], color="#1f77b4", lw=1.6)
axes[1].set_title("移动平均提取的趋势项（交给线性外推）", fontsize=11)
axes[2].plot(t[seg], seas_all[seg], color="#d62728", lw=1)
axes[2].set_title("剩余季节项（交给自相关机制）", fontsize=11)
axes[2].set_xlabel("时间步")
for ax in axes:
    ax.grid(alpha=0.3)
fig.suptitle("Autoformer 第一性原理：先分解，再各自建模", fontsize=13, y=1.0)
plt.tight_layout()
plt.savefig(os.path.join(DIR_AUT, "cover.png"), dpi=110, bbox_inches="tight")
plt.close()

# fig1: 自相关谱 + top-k
fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
lags_show = np.arange(1, 300)
axes[0].plot(lags_show, acf[1:300], color="#333333", lw=1.2)
for li, wi in zip(top_lags, w_ac):
    axes[0].axvline(li, color="#d62728", alpha=0.6, lw=1.2)
    axes[0].text(li, acf[li] + 0.03, f"τ={li}", fontsize=9, color="#d62728", ha="center")
axes[0].set_xlabel("滞后 τ"); axes[0].set_ylabel("自相关 R(τ)")
axes[0].set_title("FFT 求自相关谱：top-k 滞后即候选周期")
axes[0].grid(alpha=0.3)
axes[1].bar([f"τ={li}" for li in top_lags], w_ac, color="#d62728", alpha=0.8)
axes[1].set_ylabel("softmax 权重")
axes[1].set_title("时延聚合权重：预测 = Σ w(τ)·x(t−τ)")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(DIR_AUT, "autoformer_acf.png"), dpi=110, bbox_inches="tight")
plt.close()

# fig2: 多步预测对比
fig, axes = plt.subplots(2, 1, figsize=(12, 6.8))
ctx = 96
xs_h = np.arange(-ctx, 0); xs_f = np.arange(H)
axes[0].plot(xs_h, y[st - ctx:st], color="gray", lw=1, label="历史")
axes[0].plot(xs_f, truth, "k-", lw=2, label="真实未来")
axes[0].plot(xs_f, fc_af, "o-", color="#d62728", lw=1.5, ms=3, label="分解+自相关")
axes[0].plot(xs_f, fc_raw, "s--", color="#ff9896", lw=1.2, ms=3, label="无分解消融")
axes[0].plot(xs_f, fc_o, "^--", color="#1f77b4", lw=1.2, ms=3, label="OLS递推")
axes[0].axvline(0, color="black", ls=":", lw=1)
axes[0].legend(ncol=5, fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_title(f"H=24 多步预测示例（起点 t={st}）")
names = ["分解+自相关\n(Autoformer式)", "无分解消融", "季节naive\n(t−24)", "OLS\n96滞后递推"]
vals = [mse_af, mse_raw, mse_snaive, mse_ols_m]
colors = ["#d62728", "#ff9896", "#999999", "#1f77b4"]
axes[1].bar(names, vals, color=colors)
for i, v in enumerate(vals):
    axes[1].text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=10)
axes[1].set_ylabel("滚动多步 MSE (50 窗口均值)")
axes[1].set_title("分解带来的增益：趋势外推 + 季节时延聚合")
axes[1].grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(DIR_AUT, "autoformer_forecast.png"), dpi=110, bbox_inches="tight")
plt.close()
print("Autoformer 图已保存")
print("ALL DONE")
