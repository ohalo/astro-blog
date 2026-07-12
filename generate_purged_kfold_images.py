#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「Purged K-Fold 交叉验证：给回测一个诚实的样本外」
(purged-kfold-cv) 生成真实配图与真实统计数字。

核心主题：金融 ML 的标签通常"向前看"(例如 Triple Barrier 标签用未来 h 根 K 线的
收益判定止盈/止损)，于是相邻观测的标签高度重叠 —— 普通 KFold(shuffle=True) 会把
一个测试点的"近邻(时间相邻、特征近重复)"留在训练集里，模型靠"抄邻居标签"刷出虚假高分。
Purged K-Fold 把任何标签窗口与测试窗口重叠的训练样本 purge 掉(+ embargo 缓冲)，
逼模型只能学真正样本外的结构。

所有图表与数字均由文中逻辑真实计算生成：
  1) pkf_overlap_diagram.png —— 普通 KFold(打乱) vs Purged K-Fold 的时间线示意图
  2) pkf_cv_gap.png         —— 三种评估口径下的 AUC / 准确率：shuffled 虚高、purged 诚实
  3) pkf_leakage_vs_h.png   —— 泄漏溢价(shuffled - purged)随标签前瞻窗口 h 增大而放大

数据机制(全 numpy 合成，固定种子，用于演示，非真实行情)：
  - 收益 r_t ~ i.i.d. N(0,1) 白噪声；价格 p_t = cumsum(r_t)
  - 特征 X_t = 最近 lag=20 根收益组成的窗口向量
      => 时间相邻的 X_t 与 X_{t+1} 共享 19/20 个分量 => 特征空间里的"近重复"
  - 标签 y_t = 1 若 未来 h 根收益之和 > 0
      => r_t 独立 => 标签与过去特征无关(真实 AUC≈0.5)
      => 但 y_t 与 y_{t+1} 共享 h-1 根收益 => 标签高度相关 => 打乱 CV 被"抄作业"虚高
模型：KNeighborsClassifier(n_neighbors=5) —— 擅长利用"近重复"邻居，放大泄漏信号。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score

# ---------- 字体 / 配色 ----------
rcParams = matplotlib.rcParams
rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "purged-kfold-cv")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "up": "#55A868", "dn": "#C44E52", "grid": "#E2E2E2",
     "shuf": "#C44E52", "purg": "#4C72B0", "hold": "#55A868", "emb": "#DD8452",
     "warn": "#C44E52", "ok": "#55A868", "ink": "#2b2b2b"}

# =====================================================================
# 1) 合成数据：平滑持久的潜在过程作特征 + 独立的“前瞻噪声”作标签
# =====================================================================
# 关键设计(为什么这样能暴露泄漏)：
#   - 特征 X_t 取平滑持久过程 s_t 的滞后窗口：s_t = φ·s_{t-1} + σ_s·η_t，φ≈0.995
#       => 时间相邻的 X_t 与 X_{t+1} 几乎完全重合(共享 19/20 分量、且值极近)
#       => 在特征空间里，X_t 的“近重复”唯一地就是它的时间邻居(路径平滑、不会巧合重访)
#   - 标签 y_t 用一段“完全独立”的前瞻噪声 ζ 判定：y_t=1 若 Σ_{j=1..h} ζ_{t+j} > 0
#       => ζ 与 s 相互独立 => X_t 对 y_t 真实 AUC≈0.5(零预测力，这是真相)
#       => 但 y_t 与 y_{t+1} 共享 h-1 个 ζ => 标签高度相关(重叠)
#   => 普通打乱 CV 下，kNN 找到时间邻居、抄到相关标签 => 虚假高分
#   => Purged K-Fold 把重叠邻域 purge 掉 => 只能落到 0.5 的真相
def build_data(N=4000, lag=20, h=10, phi=0.995, seed=20260712):
    rng = np.random.default_rng(seed)
    eta = rng.standard_normal(N)            # 驱动 s_t 的噪声
    zeta = rng.standard_normal(N + h)       # 独立的“前瞻噪声”，决定标签
    s = np.zeros(N)
    s[0] = eta[0]
    for t in range(1, N):
        s[t] = phi * s[t - 1] + 0.05 * eta[t]
    # 特征：每点取过去 lag 个 s 值作窗口
    X = np.zeros((N, lag))
    for t in range(N):
        lo = max(0, t - lag + 1)
        seq = s[lo:t + 1]
        X[t, -len(seq):] = seq
    # 标签：未来 h 个独立噪声之和的符号(向前看 h 步，与特征无关)
    y = np.zeros(N, dtype=int)
    for t in range(N):
        if t + h < N:
            y[t] = 1 if zeta[t + 1:t + 1 + h].sum() > 0 else 0
    mask = np.arange(N) + h < N
    return X[mask], y[mask], np.arange(N)[mask]

# =====================================================================
# 2) Purged K-Fold 切分器(含 embargo)
#    测试折 = 时间区间 [a, b]，其标签窗口覆盖 [a, b+h]
#    purge 训练点 t 当 [t, t+h] 与 [a, b+h] 重叠  => t ∈ [a-h, b+h]
#    embargo：再剔除 (b, b+embargo_frac*N] 的训练点，避免边界污染
# =====================================================================
class PurgedKFold:
    def __init__(self, n_splits=5, h=10, embargo_frac=0.0):
        self.n_splits = n_splits
        self.h = h
        self.embargo_frac = embargo_frac

    def split(self, n):
        edges = np.linspace(0, n, self.n_splits + 1).astype(int)
        folds = []
        for i in range(self.n_splits):
            a, b = edges[i], edges[i + 1] - 1
            test = np.arange(a, b + 1)
            # 候选训练 = 全部，先 purge 重叠
            train = np.arange(n)
            # purge: t 的标签窗口 [t, t+h] 与测试标签窗口 [a, b+h] 重叠
            purge_lo = a - self.h
            purge_hi = b + self.h
            keep = (train < purge_lo) | (train > purge_hi)
            # embargo: 测试结束后的缓冲段也剔除
            if self.embargo_frac > 0:
                emb_end = int(b + self.embargo_frac * n)
                keep &= train > emb_end
            train = train[keep]
            folds.append((train, test))
        return folds

# =====================================================================
# 3) 三种评估口径：AUC(更接近 0.5 = 越诚实)
# =====================================================================
def cv_score(X, y, h, n_splits=5, purged=True, embargo_frac=0.0):
    n = len(y)
    aucs = []
    if purged:
        for tr, te in PurgedKFold(n_splits=n_splits, h=h,
                                  embargo_frac=embargo_frac).split(n):
            if len(tr) < 50 or len(te) < 5:
                continue
            m = KNeighborsClassifier(n_neighbors=5).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    else:
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        for tr, te in kf.split(X):
            m = KNeighborsClassifier(n_neighbors=5).fit(X[tr], y[tr])
            p = m.predict_proba(X[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
    return float(np.mean(aucs)), float(np.std(aucs))

# =====================================================================
# 4) 真实时间外样本：前 70% 训练，后 30% 测试(两个"最终模型"都这样训)
# =====================================================================
def true_holdout(X, y):
    n = len(y)
    cut = int(n * 0.7)
    m = KNeighborsClassifier(n_neighbors=5).fit(X[:cut], y[:cut])
    p = m.predict_proba(X[cut:])[:, 1]
    return float(roc_auc_score(y[cut:], p))

# =====================================================================
# 主计算
# =====================================================================
N, LAG, H = 4000, 20, 10
X, y, tidx = build_data(N=N, lag=LAG, h=H, seed=20260712)
print(f"[数据] N={len(y)}  特征维度={X.shape[1]}  标签正类占比={y.mean():.3f} (≈0.5 即无方向性真实信号)")

auc_shuf, sd_shuf = cv_score(X, y, H, purged=False)
auc_purg, sd_purg = cv_score(X, y, H, purged=True, embargo_frac=0.0)
auc_emb,  sd_emb  = cv_score(X, y, H, purged=True, embargo_frac=0.01)
auc_hold = true_holdout(X, y)
print(f"[普通 KFold 打乱]      CV AUC = {auc_shuf:.3f} ± {sd_shuf:.3f}   <- 虚高(泄漏)")
print(f"[Purged K-Fold]        CV AUC = {auc_purg:.3f} ± {sd_purg:.3f}   <- 诚实")
print(f"[Purged + embargo]     CV AUC = {auc_emb:.3f}  ± {sd_emb:.3f}")
print(f"[真实时间外(后30%)]     AUC = {auc_hold:.3f}   <- 两模型最终都落在这里")
print(f"[泄漏溢价 shuffled-purged] = {auc_shuf - auc_purg:.3f}")

# =====================================================================
# 图1：时间线示意图(概念图) —— 普通 KFold 打乱 vs Purged K-Fold
# =====================================================================
fig, axes = plt.subplots(2, 1, figsize=(9.2, 5.4), sharex=True)
n_t = 60
xs = np.arange(n_t)
# 上：普通 KFold 打乱 —— 相邻时间被随机分到 train/test，泄漏
rng = np.random.default_rng(7)
shuf_lbl = rng.integers(0, 2, n_t)  # 0=train 1=test(示意)
ax = axes[0]
colors = [C["shuf"] if v else "#BBD3C9" for v in shuf_lbl]
ax.bar(xs, np.ones(n_t), color=colors, width=1.0)
ax.set_yticks([])
ax.set_ylabel("普通 KFold\n(打乱)", fontsize=10)
ax.set_title("时间相邻的观测被随机拆进训练/测试 → 近重复样本互相泄露", fontsize=11)
# 标一对相邻的 train/test 泄露点
for i in range(n_t - 1):
    if shuf_lbl[i] != shuf_lbl[i + 1]:
        ax.annotate("", xy=(i + 1, 0.5), xytext=(i, 0.5),
                    arrowprops=dict(arrowstyle="<->", color=C["warn"], lw=1.3))
        ax.text((i + 0.5), 1.02, "泄漏", color=C["warn"], fontsize=9, ha="center")
        break
# 下：Purged K-Fold —— 时间有序 + 测试折 + purge 区(阴影)
ax = axes[1]
folds = [(0, 11), (12, 23), (24, 35), (36, 47), (48, 59)]
for (a, b) in folds:
    ax.bar(xs[a:b + 1], np.ones(b - a + 1), color="#BBD3C9", width=1.0)
ax.bar(xs[36:48], np.ones(12), color=C["purg"], width=1.0)  # 当前测试折
# purge 区：测试折 [36,47] 标签窗口 [36,47+h]，训练被 purge 的邻域 [36-h,47+h]
pa, pb = 36 - H, 47 + H
ax.axvspan(pa, 36, color=C["warn"], alpha=0.18)
ax.axvspan(48, pb, color=C["warn"], alpha=0.18)
ax.set_yticks([])
ax.set_ylabel("Purged K-Fold\n(有序)", fontsize=10)
ax.set_xlabel("时间 →", fontsize=10)
ax.set_title("测试折(蓝)之外的标签重叠邻域(红)被 purge，模型无从抄作业", fontsize=11)
ax.text(36 + 6, 1.02, "测试折", color=C["purg"], fontsize=9, ha="center")
fig.tight_layout()
plt.savefig(os.path.join(D, "pkf_overlap_diagram.png"), dpi=130)
plt.close()

# =====================================================================
# 图2：三种评估口径 AUC 对比
# =====================================================================
fig, ax = plt.subplots(figsize=(8.2, 4.4))
labels = ["普通 KFold\n(打乱)", "Purged\nK-Fold", "Purged+\nembargo", "真实时间外\n(后30%)"]
vals = [auc_shuf, auc_purg, auc_emb, auc_hold]
cols = [C["shuf"], C["purg"], C["emb"], C["hold"]]
bars = ax.bar(labels, vals, color=cols, width=0.6)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.3f}",
            ha="center", fontsize=10, fontweight="bold")
ax.axhline(0.5, color=C["ink"], ls="--", lw=1)
ax.text(3.45, 0.505, "无信号基线 0.5", fontsize=8, color=C["ink"])
ax.set_ylim(0.45, max(vals) + 0.06)
ax.set_ylabel("交叉验证 AUC")
ax.set_title("打乱 CV 把 AUC 刷到 {:.2f}，Purged K-Fold 把它打回 {:.2f} 的真相".format(
    auc_shuf, auc_purg))
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
plt.savefig(os.path.join(D, "pkf_cv_gap.png"), dpi=130)
plt.close()

# =====================================================================
# 图3：泄漏溢价随标签前瞻窗口 h 的变化
# =====================================================================
hs = [2, 4, 6, 8, 10, 12, 15, 20]
premium, shuf_aucs, purg_aucs = [], [], []
for hh in hs:
    Xh, yh, _ = build_data(N=4000, lag=20, h=hh, seed=20260712)
    a_s, _ = cv_score(Xh, yh, hh, purged=False)
    a_p, _ = cv_score(Xh, yh, hh, purged=True)
    shuf_aucs.append(a_s); purg_aucs.append(a_p)
    premium.append(a_s - a_p)
    print(f"  h={hh:2d}   shuffled={a_s:.3f}  purged={a_p:.3f}  溢价={a_s-a_p:+.3f}")

fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.plot(hs, shuf_aucs, "o-", color=C["shuf"], label="普通 KFold(打乱)")
ax.plot(hs, purg_aucs, "s-", color=C["purg"], label="Purged K-Fold")
ax.plot(hs, premium, "^-", color=C["emb"], label="泄漏溢价 (差)")
ax.axhline(0.5, color=C["ink"], ls="--", lw=1)
ax.set_xlabel("标签前瞻窗口 h(根 K 线)")
ax.set_ylabel("AUC / 溢价")
ax.set_title("h 越大，标签重叠越多 → 泄漏溢价被放大(最高 {:.2f})".format(max(premium)))
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
plt.savefig(os.path.join(D, "pkf_leakage_vs_h.png"), dpi=130)
plt.close()

print("\nDONE ->", D)
for f in sorted(os.listdir(D)):
    print("  ", f, os.path.getsize(os.path.join(D, f)), "bytes")
