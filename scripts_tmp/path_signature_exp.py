# -*- coding: utf-8 -*-
"""路径签名实验：迭代积分特征 vs 传统统计特征"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(42)
OUT = "/Users/halo/workspace/astro-blog/public/images/path-signature-rough"
os.makedirs(OUT, exist_ok=True)

# ---------- 签名计算（2维路径，截断到 level 3） ----------
def signature_level3(path):
    """path: (T, d) 分段线性路径，返回截断签名 (level1 + level2 + level3)
    level1: d 项, level2: d^2 项, level3: d^3 项"""
    dx = np.diff(path, axis=0)  # (T-1, d)
    T1, d = dx.shape
    # S1
    S1 = dx.sum(axis=0)
    # 逐段递推 Chen 恒等式
    # 维护 S1_run (d,), S2_run (d,d), S3_run (d,d,d)
    S1r = np.zeros(d)
    S2r = np.zeros((d, d))
    S3r = np.zeros((d, d, d))
    for t in range(T1):
        v = dx[t]
        # 单段签名: s1=v, s2=v⊗v/2, s3=v⊗v⊗v/6
        s2 = np.outer(v, v) / 2.0
        s3 = np.einsum("i,j,k->ijk", v, v, v) / 6.0
        # Chen: S3_new = S3 + S2⊗s1 + S1⊗s2 + s3
        S3r = S3r + np.einsum("ij,k->ijk", S2r, v) + np.einsum("i,jk->ijk", S1r, s2) + s3
        S2r = S2r + np.outer(S1r, v) + s2
        S1r = S1r + v
    return np.concatenate([S1r, S2r.ravel(), S3r.ravel()])

def make_path(x):
    """时间增广: (t/T, 累计收益)"""
    T = len(x)
    tt = np.arange(T + 1) / T
    px = np.concatenate([[0.0], np.cumsum(x)])
    return np.column_stack([tt, px])

# ---------- 模拟数据：窗口内路径形状决定下期收益 ----------
def simulate(n, T=40, seed=0):
    r = np.random.default_rng(seed)
    X, y, kinds = [], [], []
    for i in range(n):
        kind = r.integers(0, 3)
        if kind == 0:  # 趋势
            drift = r.choice([-1, 1]) * r.uniform(0.05, 0.15)
            x = drift + r.normal(0, 0.8, T) * 0.1
            fwd = 0.6 * drift  # 趋势延续
        elif kind == 1:  # 均值回复（先冲高后回落的往返路径）
            amp = r.choice([-1, 1]) * r.uniform(0.8, 1.5)
            shape = np.sin(np.linspace(0, np.pi, T)) * amp / T * 4
            x = shape + r.normal(0, 0.08, T)
            fwd = -0.25 * amp / 10  # 往返后反向
        else:  # 纯噪声
            x = r.normal(0, 0.1, T)
            fwd = 0.0
        fwd += r.normal(0, 0.02)
        X.append(x)
        y.append(fwd)
        kinds.append(kind)
    return np.array(X), np.array(y), np.array(kinds)

Xtr, ytr, ktr = simulate(1500, seed=1)
Xte, yte, kte = simulate(600, seed=2)

# 签名特征
def sig_feats(X):
    return np.array([signature_level3(make_path(x)) for x in X])

def base_feats(X):
    """传统统计特征：均值/波动/动量/偏度/末段动量/最大回撤"""
    f = []
    for x in X:
        c = np.cumsum(x)
        mdd = np.max(np.maximum.accumulate(c) - c)
        f.append([x.mean(), x.std(), c[-1], ((x - x.mean()) ** 3).mean() / (x.std() ** 3 + 1e-12),
                  x[-10:].sum(), mdd])
    return np.array(f)

Ftr_s, Fte_s = sig_feats(Xtr), sig_feats(Xte)
Ftr_b, Fte_b = base_feats(Xtr), base_feats(Xte)

def ridge_ic(Ftr, ytr, Fte, yte, lam=1.0):
    mu, sd = Ftr.mean(0), Ftr.std(0) + 1e-12
    A = (Ftr - mu) / sd
    B = (Fte - mu) / sd
    w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ ytr)
    pred = B @ w
    return np.corrcoef(pred, yte)[0, 1], pred, w

ic_sig, pred_sig, w_sig = ridge_ic(Ftr_s, ytr, Fte_s, yte, lam=5.0)
ic_base, pred_base, _ = ridge_ic(Ftr_b, ytr, Fte_b, yte, lam=5.0)

# 分 level 消融
d = 2
lv1 = slice(0, d)
lv12 = slice(0, d + d * d)
ic_lv1, _, _ = ridge_ic(Ftr_s[:, lv1], ytr, Fte_s[:, lv1], yte, lam=5.0)
ic_lv12, _, _ = ridge_ic(Ftr_s[:, lv12], ytr, Fte_s[:, lv12], yte, lam=5.0)
ic_lv123 = ic_sig

print(f"IC 签名(level1-3): {ic_sig:.3f}")
print(f"IC 签名(level1):   {ic_lv1:.3f}")
print(f"IC 签名(level1-2): {ic_lv12:.3f}")
print(f"IC 传统特征:       {ic_base:.3f}")

# Levy 面积对比：趋势 vs 均值回复
def levy_area(path):
    dx = np.diff(path, axis=0)
    A = 0.0
    x = path[:-1]
    # A = 1/2 ∫ (x1 dx2 - x2 dx1)
    for t in range(len(dx)):
        A += 0.5 * (x[t, 0] * dx[t, 1] - x[t, 1] * dx[t, 0])
    return A

def norm_path(x):
    p = make_path(x)
    p2 = p.copy()
    sd = p[:, 1].std() + 1e-12
    p2[:, 1] = p[:, 1] / sd
    return p2

la_trend = [levy_area(norm_path(Xte[i])) for i in range(len(Xte)) if kte[i] == 0]
la_mr = [levy_area(norm_path(Xte[i])) for i in range(len(Xte)) if kte[i] == 1]
la_noise = [levy_area(norm_path(Xte[i])) for i in range(len(Xte)) if kte[i] == 2]
print(f"Levy 面积均值: 趋势={np.mean(la_trend):.4f} 回复={np.mean(la_mr):.4f} 噪声={np.mean(la_noise):.4f}")
print(f"Levy 面积|均值|: 趋势={np.mean(np.abs(la_trend)):.4f} 回复={np.mean(np.abs(la_mr)):.4f} 噪声={np.mean(np.abs(la_noise)):.4f}")

# ---------- 图1：三类路径示例 ----------
fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
titles = ["趋势路径", "均值回复路径（往返）", "纯噪声路径"]
for k, ax in enumerate(axes):
    cnt = 0
    for i in range(len(Xte)):
        if kte[i] == k:
            ax.plot(np.cumsum(Xte[i]), alpha=0.7, lw=1.2)
            cnt += 1
            if cnt >= 6:
                break
    ax.set_title(titles[k])
    ax.set_xlabel("时间步")
    ax.grid(alpha=0.3)
axes[0].set_ylabel("累计收益")
plt.tight_layout()
plt.savefig(f"{OUT}/path_types.png", dpi=130)
plt.close()

# ---------- 图2：Levy 面积分布 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.hist(la_trend, bins=40, alpha=0.6, label=f"趋势 (|A|均值 {np.mean(np.abs(la_trend)):.3f})", density=True)
ax.hist(la_mr, bins=40, alpha=0.6, label=f"均值回复 (|A|均值 {np.mean(np.abs(la_mr)):.3f})", density=True)
ax.hist(la_noise, bins=40, alpha=0.6, label=f"噪声 (|A|均值 {np.mean(np.abs(la_noise)):.3f})", density=True)
ax.set_xlabel("Lévy 面积（时间-价格路径的二阶反对称项）")
ax.set_ylabel("密度")
ax.set_title("Lévy 面积分布：噪声路径可与结构化路径分开，但趋势与均值回复重叠——印证 level-2 增量有限")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/levy_area_dist.png", dpi=130)
plt.close()

# ---------- 图3：IC 对比 ----------
fig, ax = plt.subplots(figsize=(8, 4.5))
names = ["传统统计特征\n(6维)", "签名 level-1\n(2维)", "签名 level-1~2\n(6维)", "签名 level-1~3\n(14维)"]
vals = [ic_base, ic_lv1, ic_lv12, ic_lv123]
colors = ["#888888", "#8ecae6", "#219ebc", "#023047"]
bars = ax.bar(names, vals, color=colors)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=11)
ax.set_ylabel("样本外 IC")
ax.set_title("样本外预测 IC：签名特征逐级加入高阶项后的提升")
ax.grid(alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/ic_comparison.png", dpi=130)
plt.close()

# ---------- 图4：维度灾难（level vs 特征数 vs 增量IC） ----------
fig, ax1 = plt.subplots(figsize=(8, 4.5))
levels = [1, 2, 3, 4, 5]
dims = [2 ** 1, 2 ** 1 + 2 ** 2, 14, 14 + 16, 14 + 16 + 32]
dims = [sum(2 ** k for k in range(1, L + 1)) for L in levels]
ax1.plot(levels, dims, "o-", color="#d62828", lw=2, label="特征维度 (d=2)")
dims5 = [sum(5 ** k for k in range(1, L + 1)) for L in levels]
ax1.plot(levels, dims5, "s--", color="#f77f00", lw=2, label="特征维度 (d=5)")
ax1.set_yscale("log")
ax1.set_xlabel("签名截断 level")
ax1.set_ylabel("特征维度（对数轴）")
ax1.set_xticks(levels)
ax1.set_title("签名维度随截断阶数指数爆炸：d 维路径 level-L 共 Σ d^k 项")
ax1.legend()
ax1.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/dim_explosion.png", dpi=130)
plt.close()

print("charts saved:", os.listdir(OUT))
