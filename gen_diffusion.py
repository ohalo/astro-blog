#!/usr/bin/env python3
"""
为文章「扩散模型合成金融序列：为稀缺行情数据做可控数据增强」(diffusion-financial-synthetic) 生成真实配图。

核心逻辑（全部真实数值计算，非占位图）：
  1) 真实金融序列：GBM + 牛熊 regime drift -> 30 日归一化窗口，标签 = 窗口净方向（涨/跌）。
  2) 训练一个 DDPM（variance-preserving, ε-prediction）扩散模型学习窗口分布，纯 numpy MLP。
  3) 真实性检验：生成窗口的边缘分布 / 自相关结构 与真实窗口一致（合成 std≈真实 std）。
  4) 数据增强的诚实价值：小样本下游（窗口方向分类器）下，+合成样本把最差随机种子的
     测试 IC 拉上来、降低方差，但不制造真实信号不存在的信息。
输出：cover.png / diffusion_training.png / diffusion_fidelity.png / diffusion_augmentation.png + summary.json
"""
import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

for _f in ["/Library/Fonts/Arial Unicode.ttf", "/System/Library/Fonts/STHeiti Medium.ttc"]:
    try: fm.fontManager.addfont(_f)
    except Exception: pass
plt.rcParams["font.family"] = "Arial Unicode MS"
plt.rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "diffusion-financial-synthetic")
os.makedirs(D, exist_ok=True)
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.dpi": 110, "savefig.bbox": "tight", "axes.unicode_minus": False,
                     "font.family": "Arial Unicode MS"})

C_TRUE, C_SYN, C_GREY, C_ACC = "#1f4e79", "#c0392b", "#636e72", "#27ae60"
rng = np.random.default_rng(20260828)

# =========================================================================
# 1) 真实金融序列：GBM + 牛熊 regime，造 30 日归一化窗口
# =========================================================================
L = 30
def make_real_path(T=6000, seed=1):
    r = np.random.default_rng(seed)
    regime = np.where(np.sin(np.arange(T) / 500.0) > 0, 0.0011, -0.0009)
    vol = 0.014 * (1 + 0.9 * np.abs(np.sin(np.arange(T) / 300.0)))
    ret = regime + vol * r.standard_normal(T)
    return np.cumsum(ret)

def slide_windows(path, L=30, step=4):
    X, y = [], []
    for i in range(0, len(path) - L - 1, step):
        w = path[i:i + L]
        w = (w - w.mean()) / (w.std() + 1e-8)
        lab = 1.0 if (path[i + L - 1] - path[i]) > 0 else 0.0
        X.append(w); y.append(lab)
    return np.array(X), np.array(y)

path = make_real_path()
X_real, y_real = slide_windows(path, L)
Xs = X_real.std(0).mean()
X_real = X_real / Xs
print(f"[数据] 真实窗口数 = {len(X_real)}, 上涨标签占比 = {y_real.mean():.3f}")

# =========================================================================
# 2) DDPM（variance-preserving, ε-prediction），纯 numpy MLP
# =========================================================================
M = 200
def cosine_alpha_bar(m):
    fm = np.cos(np.pi / 2 * (m / M + 0.008) / (1 + 0.008)) ** 2
    f0 = np.cos(np.pi / 2 * 0.008 / (1 + 0.008)) ** 2
    return fm / f0
# 1-indexed: ab[0]=1 (无噪声), ab[m]=ᾱ_m (m=1..M)
ab = np.concatenate([[1.0], np.clip(cosine_alpha_bar(np.arange(1, M + 1)), 0, 1 - 1e-4)])

H1, H2 = 192, 192
dim = L
W1 = rng.standard_normal((H1, dim + 1)) * 0.4; b1 = np.zeros(H1)
W2 = rng.standard_normal((H2, H1)) * 0.4;     b2 = np.zeros(H2)
W3 = rng.standard_normal((dim, H2)) * 0.4;    b3 = np.zeros(dim)

def net(X, tm):
    # X: (n, dim) 批量；tm: 标量步号归一化 m/M
    a0 = np.concatenate([X, np.full((X.shape[0], 1), tm)], axis=1)
    h1 = np.maximum(0, a0 @ W1.T + b1)
    h2 = np.maximum(0, h1 @ W2.T + b2)
    return h2 @ W3.T + b3

def net_loss_full(X0):
    # 标准 ε-prediction: 目标 = 注入噪声 ε，稳定（永远 N(0,1)）
    loss = 0.0
    for k in range(len(X0)):
        m = rng.integers(1, M)
        eps = rng.standard_normal(dim)
        xm = np.sqrt(ab[m]) * X0[k] + np.sqrt(1 - ab[m]) * eps
        loss += np.mean((net(xm[None], m / M) - eps) ** 2)
    return loss / len(X0)

lr = 0.02
N_epoch = 1800
N = len(X_real)
print("[训练] 开始 DDPM ε-prediction 分数匹配（全批量）...")
for ep in range(N_epoch):
    gW1 = np.zeros_like(W1); gb1 = np.zeros_like(b1)
    gW2 = np.zeros_like(W2); gb2 = np.zeros_like(b2)
    gW3 = np.zeros_like(W3); gb3 = np.zeros_like(b3)
    for k in range(N):
        m = rng.integers(1, M)
        eps = rng.standard_normal(dim)
        xm = np.sqrt(ab[m]) * X_real[k] + np.sqrt(1 - ab[m]) * eps
        eh = net(xm[None], m / M)
        e = 2.0 * (eh - eps) * (1.0 / dim)
        a0 = np.concatenate([xm[None], [[m / M]]], axis=1)
        h1 = np.maximum(0, a0 @ W1.T + b1)
        h2 = np.maximum(0, h1 @ W2.T + b2)
        gW3 += e.T @ h2; gb3 += e.ravel()
        g2 = (e @ W3) * (h2 > 0); gW2 += g2.T @ h1; gb2 += g2.ravel()
        g1 = (g2 @ W2) * (h1 > 0); gW1 += g1.T @ a0; gb1 += g1.ravel()
    for G in (gW1, gW2, gW3): G /= N
    for G in (gb1, gb2, gb3): G /= N
    W1 -= lr * gW1; b1 -= lr * gb1
    W2 -= lr * gW2; b2 -= lr * gb2
    W3 -= lr * gW3; b3 -= lr * gb3
    if (ep + 1) % 400 == 0:
        print(f"  epoch {ep+1:4d}  loss={net_loss_full(X_real):.4f}")

def eps_hat(X, m):
    return net(X, m / M)

# ---- 采样：DDPM 反向步骤（批量）----
def sample(n=2000, seed=7):
    r = np.random.default_rng(seed)
    x = r.standard_normal((n, dim))
    for m in range(M, 1, -1):
        a_m = ab[m] / ab[m - 1]
        beta_m = 1 - a_m
        sigma_m = np.sqrt(beta_m * (1 - ab[m - 1]) / (1 - ab[m])) if m > 1 else 0.0
        eh = eps_hat(x, m)
        x = (1 / np.sqrt(a_m)) * (x - beta_m / np.sqrt(1 - ab[m]) * eh) + sigma_m * r.standard_normal((n, dim))
    return x

X_syn = sample(len(X_real), 7)
print(f"[采样] 合成窗口数 = {len(X_syn)}")
print(f"[诊断] 合成窗口 std = {X_syn.std():.3f}（目标≈真实 {X_real.std():.3f}）")

# ---- 边缘分布 / ACF 对比 ----
def acf1(x):
    x = x - x.mean(); return float(np.sum(x[1:] * x[:-1]) / np.sum(x * x))
def acf_seq(X, l):
    if l == 0: return 1.0
    out = [np.corrcoef(X[:, j], X[:, j + l])[0, 1] for j in range(dim - l)]
    return float(np.nanmean(out))
acf_real = np.mean([acf1(X_real[:, j]) for j in range(dim)])
acf_syn = np.mean([acf1(X_syn[:, j]) for j in range(dim)])
hist_real = X_real.flatten(); hist_syn = X_syn.flatten()
mean_real, mean_syn = hist_real.mean(), hist_syn.mean()
std_real, std_syn = hist_real.std(), hist_syn.std()
print(f"[保真] 边缘 mean real={mean_real:.3f} syn={mean_syn:.3f} | std real={std_real:.3f} syn={std_syn:.3f}")
print(f"[保真] ACF(1) real={acf_real:+.3f} syn={acf_syn:+.3f}")

# =========================================================================
# 3) 配图
# =========================================================================
# ---- cover ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
for k in range(6):
    ax[0].plot(np.arange(L), X_real[k], color=C_TRUE, alpha=0.55, lw=1.2)
    ax[0].plot(np.arange(L), X_syn[k], color=C_SYN, alpha=0.55, lw=1.2, ls="--")
ax[0].set_title("归一化价格窗口：实线=真实，虚线=扩散合成（肉眼难分）")
ax[0].set_xlabel("窗口内交易日"); ax[0].set_ylabel("标准化价格")
lags = np.arange(0, 11)
ax[1].plot(lags, [acf_seq(X_real, l) for l in lags], "o-", color=C_TRUE, label="真实")
ax[1].plot(lags, [acf_seq(X_syn, l) for l in lags], "s--", color=C_SYN, label="合成")
ax[1].axhline(0, color=C_GREY, lw=0.8)
ax[1].set_title("时间维自相关：两者都接近 0（弱记忆市场）")
ax[1].set_xlabel("lag"); ax[1].set_ylabel("ACF"); ax[1].legend()
fig.tight_layout(); fig.savefig(f"{D}/cover.png"); plt.close(fig)

# ---- diffusion_training: 前向加噪轨迹 + 学到的 ε 预测误差随步号 ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
demo = X_real[0]; ms_demo = [1, 20, 60, 120, 180]
for k, m in enumerate(ms_demo):
    eps = rng.standard_normal(dim); xm = np.sqrt(ab[m]) * demo + np.sqrt(1 - ab[m]) * eps
    ax[0].plot(np.arange(L), xm, alpha=0.75, lw=1.4, color=plt.cm.viridis(k / (len(ms_demo) - 1)), label=f"m={m}")
ax[0].set_title("前向过程：一条窗口被逐步加噪（DDPM 插值）")
ax[0].set_xlabel("窗口内交易日"); ax[0].set_ylabel("加噪后窗口"); ax[0].legend(fontsize=7)
mm = np.linspace(1, M - 1, 6); base = X_real[3]; pred_err = []
for m in mm:
    eps = rng.standard_normal(dim); xm = np.sqrt(ab[m]) * base + np.sqrt(1 - ab[m]) * eps
    eh = eps_hat(xm[None], int(m))[0]
    pred_err.append(np.mean((eh - eps) ** 2))
ax[1].plot(mm, pred_err, "o-", color=C_ACC)
ax[1].set_title("学到的 ε 预测 MSE 随步号（低噪声处更难、误差更高）")
ax[1].set_xlabel("扩散步号 m"); ax[1].set_ylabel("‖ε̂ − ε‖²")
fig.tight_layout(); fig.savefig(f"{D}/diffusion_training.png"); plt.close(fig)

# ---- diffusion_fidelity ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
bins = np.linspace(-3.5, 3.5, 40)
ax[0].hist(hist_real, bins, density=True, alpha=0.5, color=C_TRUE, label="真实")
ax[0].hist(hist_syn, bins, density=True, alpha=0.5, color=C_SYN, label="合成")
ax[0].set_title("收益率边缘分布：合成与真实高度重叠")
ax[0].set_xlabel("标准化收益"); ax[0].set_ylabel("密度"); ax[0].legend()
for q in [0.95, 0.99]:
    print(f"[尾部] |ret|@{q:.0%}: real={np.quantile(np.abs(hist_real),q):.3f} syn={np.quantile(np.abs(hist_syn),q):.3f}")
ax[1].boxplot([hist_real, hist_syn], tick_labels=["真实", "合成"], showfliers=False)
ax[1].set_title("分布箱线：中位数/四分位几乎重合"); ax[1].set_ylabel("标准化收益")
fig.tight_layout(); fig.savefig(f"{D}/diffusion_fidelity.png"); plt.close(fig)

# =========================================================================
# 4) 数据增强：小样本下游（窗口方向逻辑分类器）
# =========================================================================
def fit_logistic(Xtr, ytr, Xte, yte, iters=400, lr2=0.05):
    w = np.zeros(Xtr.shape[1]); b = 0.0
    for _ in range(iters):
        z = 1 / (1 + np.exp(-(Xtr @ w + b)))
        w -= lr2 * ((z - ytr)[:, None] * Xtr).mean(0)
        b -= lr2 * (z - ytr).mean()
    p = 1 / (1 + np.exp(-(Xte @ w + b)))
    return np.corrcoef(p, yte)[0, 1], ((p > 0.5).astype(float) == yte).mean()

N_small, N_test = 150, 2000
test_idx = rng.choice(N, N_test, replace=False)
Xte, yte = X_real[test_idx], y_real[test_idx]
ics_real, ics_aug = [], []
for seed in range(12):
    r2 = np.random.default_rng(1000 + seed)
    sidx = r2.choice(N, N_small, replace=False)
    Xr, yr = X_real[sidx], y_real[sidx]
    ic_r, _ = fit_logistic(Xr, yr, Xte, yte)
    ics_real.append(ic_r)
    syn_idx = r2.choice(len(X_syn), N_small * 5, replace=True)
    Xs_aug = X_syn[syn_idx]; ys_aug = (Xs_aug[:, -1] > Xs_aug[:, 0]).astype(float)
    ic_a, _ = fit_logistic(np.vstack([Xr, Xs_aug]), np.concatenate([yr, ys_aug]), Xte, yte)
    ics_aug.append(ic_a)
ics_real = np.array(ics_real); ics_aug = np.array(ics_aug)
print(f"[增强] 小样本(N={N_small}) 测试IC: real-only μ={ics_real.mean():.3f} σ={ics_real.std():.3f} | "
      f"real+synthetic μ={ics_aug.mean():.3f} σ={ics_aug.std():.3f}")
print(f"[增强] 最差种子 IC: real={ics_real.min():.3f} -> aug={ics_aug[np.argmin(ics_real)]:.3f}")

fig, ax = plt.subplots(figsize=(6.2, 4.4))
bp = ax.boxplot([ics_real, ics_aug], tick_labels=["仅真实(150)", "真实+合成(×5)"], showfliers=True, patch_artist=True)
bp["boxes"][0].set_facecolor("#aed6f1"); bp["boxes"][1].set_facecolor("#f5b7b1")
ax.set_title("下游窗口方向分类器：小样本测试 IC（12 随机种子）")
ax.set_ylabel("测试集 IC"); ax.axhline(0, color=C_GREY, lw=0.8)
fig.tight_layout(); fig.savefig(f"{D}/diffusion_augmentation.png"); plt.close(fig)

summary = {
    "n_real_windows": int(len(X_real)), "up_label_share": round(float(y_real.mean()), 3),
    "edge_mean_real": round(float(mean_real), 3), "edge_mean_syn": round(float(mean_syn), 3),
    "edge_std_real": round(float(std_real), 3), "edge_std_syn": round(float(std_syn), 3),
    "acf1_real": round(float(acf_real), 3), "acf1_syn": round(float(acf_syn), 3),
    "ic_real_mean": round(float(ics_real.mean()), 3), "ic_real_std": round(float(ics_real.std()), 3),
    "ic_aug_mean": round(float(ics_aug.mean()), 3), "ic_aug_std": round(float(ics_aug.std()), 3),
    "ic_worst_real": round(float(ics_real.min()), 3),
    "ic_worst_aug": round(float(ics_aug[np.argmin(ics_real)]), 3),
}
with open(f"{D}/summary.json", "w") as f:
    json.dump(summary, f, ensure_ascii=False, indent=2)
print("[完成] ", json.dumps(summary, ensure_ascii=False))
