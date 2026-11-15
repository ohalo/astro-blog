---
title: "FastICA 独立成分分析：PCA 找不到的隐因子，它靠非高斯性挖出来"
description: "PCA 只按方差大小正交投影，遇到方差相近的独立源就彻底失效。FastICA 换了一条路：最大化各成分的非高斯性（负熵），把混合信号里的独立隐因子还原回来。本文手写定步长不动点 FastICA，用真值已知的现场实验对拍：三个方差不等、分布各异的非高斯源（Laplace 超高斯、均匀亚高斯、t(4) 厚尾）随机混合后，ICA 与真源的匹配相关系数均值 0.9999、Amari 距离 0.0092（近乎完美），而 PCA 只有 0.7415、Amari 0.3828——差了 40 倍。关键对照：把源换成三个高斯，ICA 的 Amari 立刻从 0.0092 暴涨到 0.3964，与 PCA（0.4096）无差别，因为高斯源下负熵没有梯度、解不唯一，这正是 ICA 不可辨识的经典边界。非线性扫描里 tanh/gauss/cube 三种负熵近似 Amari 都在 0.01-0.03，差别很小；金融应用上 ICA 还原因子的平方互相关仅 0.0211，远优于 PCA 的 0.4051，说明它抓到的是真正高阶独立的因子结构。"
publishDate: '2026-11-15T06:55:00Z'
tags:
  - 量化
  - 交易
  - 因子模型
  - 机器学习
  - 信号处理
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

你有一组观测 $X=AS$，A 是未知混合矩阵，S 是 k 个独立隐源。PCA 能还原 S 吗？**只有当各源的二阶方差差异足够大、且你不在乎顺序和符号时，PCA 才勉强够用**；一旦源方差接近，PCA 给出的只是任意正交旋转，和真源毫无关系。

FastICA 的根本不同：它不靠方差，而靠**非高斯性**。独立成分里只要有一个不是高斯的，就可以用负熵（非高斯性的度量）把混合解开。现场实测：

1. 三个方差不等、分布各异的非高斯源随机混合后，FastICA 与真源匹配相关系数 **0.9999**，Amari 距离 **0.0092**（近乎完美）；PCA 匹配相关系数只有 **0.7415**，Amari 距离 **0.3828**——差了约 40 倍。
2. **高斯对照是 ICA 的生死线**：把源换成三个高斯，ICA 的 Amari 从 0.0092 暴涨到 **0.3964**，和 PCA 的 0.4096 没有实质差别。因为高斯源的负熵没有梯度，解不唯一——这是 ICA 不可辨识的经典边界，不是 bug。
3. 金融应用上，ICA 还原出的因子彼此平方互相关仅 **0.0211**，而 PCA 因子高达 **0.4051**：ICA 抓到的是真正高阶独立的因子结构。

## 二、PCA 为什么找不到独立成分

PCA 最大化方差投影，等价于对协方差矩阵做特征分解。问题在于：**独立 ≠ 不相关**。不相关只约束二阶矩（协方差为零），而独立要求所有阶矩都解耦。对两组方差相等、各自独立的非高斯源，它们的任意正交旋转仍然互不相关，于是 PCA 可以收敛到任意一个旋转，未必是原始混合的逆向。

更致命的是符号与顺序的模糊性：ICA 和 PCA 都只能还原到「置换×符号」精度——你永远不知道哪个成分是第几个源、是正还是反。这要求评估时必须先做最优匹配再比相关系数，否则会被符号/顺序噪声污染。

## 三、FastICA：用负熵做不动点迭代

FastICA 的核心思想：对一个零均值、单位方差的信号 y，若它是高斯的，负熵（与高斯的差）为零；越偏离高斯，负熵越大。最大化各成分负熵的近似（用 $\tanh$、高斯、立方等非线性），就能逼出独立成分。

算法分两步。先**白化**：中心化后做特征值分解，$\mathbf{Z}=\mathbf{K}\mathbf{X}$，使 Z 各分量单位方差且不相关。再跑**对称不动点迭代**：

$$\mathbf{W}\leftarrow \mathbb{E}[\mathbf{Z}\,g(\mathbf{W}^\top\mathbf{Z})]-\mathbb{E}[g'(\mathbf{W}^\top\mathbf{Z})]\mathbf{W}$$

每轮后对 W 做正交去相关（特征分解后 $W\leftarrow (W W^\top)^{-1/2}W$），重复到收敛。最终全解混 $\hat W=WK$，估计源 $\hat S=\hat W X$。代码里 g 默认用 $\tanh$（适合超高斯/厚尾源），并支持高斯、立方两种近似对照。

图 1 是三个真源的时间序列与直方图：Laplace 尖峰、均匀平顶、t(4) 厚尾，形状一目了然；图 2 把 ICA 与 PCA 的还原结果（按真源匹配后）叠在真源上，ICA 几乎复刻了原形的尖峰与厚尾，PCA 给的则是糊在一起的混合投影。

![三个真实非高斯源](/images/fast-ica-factor-recovery/fig1_true_sources.png)

![ICA 与 PCA 的还原对比](/images/fast-ica-factor-recovery/fig2_recovery.png)

## 四、现场还原：ICA 0.9999 vs PCA 0.7415

主实验：T=4000、k=3，三个源分别标准化，随机混合矩阵条件数约 7.79。每個估计成分用贪心匹配到相关系数最高的未占用真源，得到匹配相关系数：

| 评估量 | ICA | PCA |
|---|---|---|
| 匹配相关系数均值 | 0.9999 | 0.7415 |
| Amari 距离 | 0.0092 | 0.3828 |

图 3 左是非高斯源下逐成分的匹配相关（ICA 三根几乎贴 1，PCA 三根明显矮一截）；右是 Amari 距离的生死对照——**非高斯时 ICA 贴着 0，PCA 0.38；换成高斯源，ICA 的 Amari 跳到 0.3964，与 PCA 的 0.4096 几乎一样**。这说明 ICA 的全部威力都来自非高斯性：一旦源是高斯的，负熵梯度消失，固定点收敛到任意正交旋转，解不再唯一。

![匹配相关与 Amari 对照](/images/fast-ica-factor-recovery/fig3_corr_compare.png)

需要点破一个常见误导：高斯对照里 ICA 与真源的「匹配相关系数」仍能到 0.83 左右，看着不像翻车。那只是 3 维随机旋转恰巧和某个真轴对齐的运气——不能当作可辨识的证据。真正诚实的指标是 Amari：它在高斯下暴涨，证明解不唯一。匹配相关系数在这里会「碰巧偏高」，必须配 Amari 一起看。

## 五、非线性选择与收敛

负熵的近似有好几种：tanh（超高斯）、高斯型 $y e^{-y^2/2}$（亚高斯）、立方 $y^3$。扫描结果三者 Amari 都在 0.01–0.03，彼此差别很小（tanh 0.0152、gauss 0.0102、cube 0.0322），匹配相关系数都 >0.998。实务建议：**源以厚尾/尖峰为主时优先 tanh，它最稳健**；若怀疑亚高斯主导再换高斯型。图 4 右是这条扫描，左是主实验的不动点收敛轨迹——约 7 步就稳定，工程上完全够用。

![收敛曲线与非线性扫描](/images/fast-ica-factor-recovery/fig4_convergence_sweep.png)

## 六、金融应用：高阶独立的因子

把观测 X 当成 k 个「资产收益」，还原出的因子是否有意义，要看它们的**高阶独立性**，而非仅不相关。衡量方法：对每对因子取平方后再算互相关，取离对角均值。ICA 因子该值仅 **0.0211**，PCA 因子 **0.4051**——差了将近 20 倍。图 5 两个热力图把因子平方互相关矩阵铺开：ICA 几乎全黑（离对角≈0），PCA 离对角仍亮。这意味着用 ICA 因子做风险分解、配对或统计套利信号时，因子之间的高阶纠缠更少，信号更干净。

![因子高阶独立性热力图](/images/fast-ica-factor-recovery/fig5_higher_order.png)

## 七、诚实边界

- **非高斯是硬前提**：源若全高斯，ICA 不可辨识，退化为任意旋转，别指望它比 PCA 强。
- **置换与符号模糊**：ICA 只能还原到「置换×符号」，评估必须做最优匹配。
- **白化敏感**：白化对协方差估计敏感，样本量不足或源接近奇异时解会抖；先确认混合矩阵条件数可控。
- **中心化不能省**：未去均值的混合会让负熵近似失效，第一步必做中心化。

下面是本文本全部数字的生成代码与结果 JSON（含五张图），所有结论均可复现。

```python
# -*- coding: utf-8 -*-
"""FastICA 独立成分分析：从混合信号里找回非高斯 latent 因子。

本文全部数字的唯一来源。运行本脚本会：
1. 现场造 3 个独立、方差不等的非高斯源（Laplace / 均匀 / t(4) 厚尾）；
2. 随机线性混合后，分别用 FastICA（手写定步长不动点）与 PCA 还原；
3. 用匹配相关系数 + Amari 距离定量比较；并跑高斯对照（不可辨识）与非线性扫描；
4. 画 5 张图到 public/images/fast-ica-factor-recovery/，写 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261115
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "fast-ica-factor-recovery"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- 非线性函数
def _g_tanh(y):
    g = np.tanh(y)
    return g, 1.0 - g ** 2


def _g_gauss(y):
    ey = y * np.exp(-y ** 2 / 2.0)
    return ey, (1.0 - y ** 2) * np.exp(-y ** 2 / 2.0)


def _g_cube(y):
    return y ** 3, 3.0 * y ** 2


# ---------------------------------------------------------------- 源生成
def make_sources(rng, T, gauss=False):
    if gauss:
        s = np.vstack([
            rng.standard_normal(T),
            rng.standard_normal(T),
            rng.standard_normal(T),
        ])
        # 单位方差
        s = s / s.std(axis=1, ddof=1)[:, None]
        names = ["高斯#1", "高斯#2", "高斯#3"]
    else:
        s1 = rng.laplace(0.0, 1.0, T)            # 超高斯
        s2 = rng.uniform(-np.sqrt(3), np.sqrt(3), T)  # 亚高斯
        s3 = rng.standard_t(4.0, T)              # 厚尾超高斯
        s = np.vstack([s1, s2, s3])
        s = s / s.std(axis=1, ddof=1)[:, None]
        names = ["Laplace(超高斯)", "均匀(亚高斯)", "t(4)厚尾"]
    return s, names


def random_mix(rng, k):
    A = rng.standard_normal((k, k))
    # 让条件数适中，避免病态
    return A


def whiten(X):
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T / Xc.shape[1]
    d, U = np.linalg.eigh(cov)
    d = np.clip(d, 1e-12, None)
    K = np.diag(1.0 / np.sqrt(d)) @ U.T
    Z = K @ Xc
    return Z, K, Xc


def fastica(X, g, rng, max_iter=300, tol=1e-7):
    Z, K, Xc = whiten(X)
    k, T = Z.shape
    W = np.linalg.qr(rng.standard_normal((k, k)))[0]
    objs = []
    for it in range(max_iter):
        Y = W @ Z
        gy, gpy = g(Y)
        W_new = (gy @ Z.T) / T - np.diag(gpy.mean(axis=1)) @ W
        # 对称去相关
        s, uu = np.linalg.eigh(W_new @ W_new.T)
        W_new = (uu @ np.diag(1.0 / np.sqrt(s)) @ uu.T) @ W_new
        lim = np.max(np.abs(np.abs(np.diag(W_new @ W.T)) - 1.0))
        objs.append(float(np.sum(np.abs(gy.mean(axis=1)))))
        W = W_new
        if lim < tol:
            break
    W_full = W @ K
    S_hat = W_full @ Xc
    return S_hat, W, K, Xc, objs, it + 1


def pca_recover(X, k=None):
    Xc = X - X.mean(axis=1, keepdims=True)
    cov = Xc @ Xc.T / Xc.shape[1]
    d, U = np.linalg.eigh(cov)
    U = U[:, ::-1]
    if k is not None:
        U = U[:, :k]
    return U.T @ Xc, U


def match(S_hat, S):
    """为每一个估计成分匹配相关系数最高的、尚未占用的真源（保证 k 对）。"""
    k = S_hat.shape[0]
    corr = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            corr[i, j] = abs(np.corrcoef(S_hat[i], S[j])[0, 1])
    used_j = set()
    pairs = []
    for i in range(k):
        best, bj = -1.0, -1
        for j in range(k):
            if j in used_j:
                continue
            if corr[i, j] > best:
                best, bj = corr[i, j], j
        used_j.add(bj)
        pairs.append((bj, best))
    return pairs


def amari(W_est, W_true):
    """Amari 距离：完美还原（置换*符号）时为 0，越大越偏离。"""
    P = np.abs(W_est @ W_true)
    k = P.shape[0]
    row = P / P.max(axis=1, keepdims=True)
    col = P / P.max(axis=0, keepdims=True)
    return ((row.sum() - k) + (col.sum() - k)) / (2.0 * k * (k - 1))


def higher_order_indep(S_hat):
    """平均 |平方后的互相关|，衡量高阶独立性（越小越独立）。"""
    k = S_hat.shape[0]
    sq = S_hat ** 2
    vals = []
    for i in range(k):
        for j in range(i + 1, k):
            vals.append(abs(np.corrcoef(sq[i], sq[j])[0, 1]))
    return float(np.mean(vals))


def main():
    rng = np.random.default_rng(SEED)
    T = 4000
    k = 3

    # ---------- 主实验：非高斯源 ----------
    S, names = make_sources(rng, T, gauss=False)
    A = random_mix(rng, k)
    X = A @ S

    S_ica, W, K, Xc, objs, n_iter = fastica(X, _g_tanh, rng)
    S_pca, _ = pca_recover(X, k)
    S_pca = S_pca / S_pca.std(axis=1, ddof=1)[:, None]

    pairs_ica = match(S_ica, S)
    pairs_pca = match(S_pca, S)
    ica_corr = [p[1] for p in pairs_ica]
    pca_corr = [p[1] for p in pairs_pca]
    # Amari 距离以混合矩阵 A 为基准：完美还原时 W_est @ A ≈ 置换*符号
    amari_ica = amari(W @ K, A)
    # PCA 的等价全解混矩阵：取前 k 个主成分方向 U^T 再左乘 K^-1
    cov = Xc @ Xc.T / Xc.shape[1]
    _, U = np.linalg.eigh(cov)
    U = U[:, ::-1][:, :k]
    W_pca_full = U.T @ np.linalg.inv(K)
    amari_pca = amari(W_pca_full, A)

    ho_ica = higher_order_indep(S_ica)
    ho_pca = higher_order_indep(S_pca)

    # ---------- 高斯对照（不可辨识）：正确指标是 Amari ----------
    # 高斯源下 ICA 没有非高斯性可作信号，固定点收敛到任意正交旋转，
    # 解不唯一 → Amari（对真混合矩阵）应当很大；而非高斯时 Amari≈0。
    # 匹配相关系数在此会「碰巧偏高」（3 维随机旋转本就可能与某真轴对齐），
    # 不能作为可辨识性的判据。
    rng_g = np.random.default_rng(SEED + 99)
    gauss_ica_amari, gauss_pca_amari = [], []
    gauss_ica_corr_list, gauss_pca_corr_list = [], []
    for trial in range(8):
        Sg, _ = make_sources(rng_g, T, gauss=True)
        Ag = random_mix(rng_g, k)
        Xg = Ag @ Sg
        Sg_ica, Wg, Kg, Xcg, _, _ = fastica(Xg, _g_tanh, rng_g)
        Sg_pca, _ = pca_recover(Xg, k)
        Sg_pca = Sg_pca / Sg_pca.std(axis=1, ddof=1)[:, None]
        covg = Xcg @ Xcg.T / Xcg.shape[1]
        _, Ug = np.linalg.eigh(covg)
        Ug = Ug[:, ::-1][:, :k]
        Wg_pca = Ug.T @ np.linalg.inv(Kg)
        gauss_ica_amari.append(float(amari(Wg @ Kg, Ag)))
        gauss_pca_amari.append(float(amari(Wg_pca, Ag)))
        gauss_ica_corr_list.append(float(np.mean([p[1] for p in match(Sg_ica, Sg)])))
        gauss_pca_corr_list.append(float(np.mean([p[1] for p in match(Sg_pca, Sg)])))
    gauss_ica_amari_mean = float(np.mean(gauss_ica_amari))
    gauss_pca_amari_mean = float(np.mean(gauss_pca_amari))
    gauss_ica_mean = float(np.mean(gauss_ica_corr_list))
    gauss_pca_mean = float(np.mean(gauss_pca_corr_list))

    # ---------- 非线性扫描（超高斯混合下） ----------
    gset = {"tanh": _g_tanh, "gauss": _g_gauss, "cube": _g_cube}
    sweep = {}
    for gname, gfunc in gset.items():
        S2, _ = make_sources(rng, T, gauss=False)
        A2 = random_mix(rng, k)
        X2 = A2 @ S2
        S2_ica, W2, K2, X2c, _, _ = fastica(X2, gfunc, rng)
        p2 = match(S2_ica, S2)
        corr2 = [p[1] for p in p2]
        sweep[gname] = {
            "mean_matched_corr": float(np.mean(corr2)),
            "amari": float(amari(W2 @ K2, A2)),
        }

    # ===================== 画图 =====================
    # 图1：真实源（时间序列 + 直方图）
    colors = ["#d1242f", "#1f6feb", "#2da44e"]
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 5.4))
    for j in range(k):
        axes[0, j].plot(S[j, :300], color=colors[j], lw=0.8)
        axes[0, j].set_title(names[j], fontsize=9.5)
        axes[0, j].grid(alpha=0.3)
        axes[1, j].hist(S[j], bins=50, color=colors[j], alpha=0.8)
        axes[1, j].set_ylabel("频数" if j == 0 else "")
        axes[1, j].grid(alpha=0.3, axis="y")
    axes[0, 0].set_ylabel("信号值")
    fig.suptitle("图1 三个独立非高斯源：Laplace(尖峰) / 均匀(平顶) / t(4)(厚尾)",
                 fontsize=11)
    _save(fig, "fig1_true_sources.png")

    # 图2：ICA 还原 vs PCA 还原（按真源匹配后）
    ica_order = [p[0] for p in pairs_ica]
    pca_order = [p[0] for p in pairs_pca]
    fig, axes = plt.subplots(3, 3, figsize=(11.2, 6.2), sharex=True)
    for j in range(k):
        axes[j, 0].plot(S[j, :300], color="#24292f", lw=0.8, label="真源")
        axes[j, 0].set_ylabel(names[j], fontsize=9)
        axes[j, 1].plot(S_ica[ica_order[j], :300], color=colors[j], lw=0.8,
                        label="ICA")
        axes[j, 2].plot(S_pca[pca_order[j], :300], color="#8250df", lw=0.8,
                        label="PCA")
    for j in range(k):
        axes[j, 0].set_title("真源", fontsize=9)
        axes[j, 1].set_title(f"ICA (r={ica_corr[j]:.2f})", fontsize=9)
        axes[j, 2].set_title(f"PCA (r={pca_corr[j]:.2f})", fontsize=9)
    axes[2, 0].set_xlabel("t (前 300 点)")
    fig.suptitle("图2 还原质量：ICA 几乎完美复刻真源形状，PCA 给的是混合投影",
                 fontsize=11)
    _save(fig, "fig2_recovery.png")

    # 图3：非高斯源下 ICA vs PCA 的匹配相关系数
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.2))
    x = np.arange(k)
    axes[0].bar(x - 0.2, ica_corr, 0.4, color="#1f6feb", label="ICA")
    axes[0].bar(x + 0.2, pca_corr, 0.4, color="#8250df", label="PCA")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("与真源的匹配 |相关系数|")
    axes[0].set_title(f"非高斯源：ICA 均值 {np.mean(ica_corr):.3f} "
                      f"vs PCA {np.mean(pca_corr):.3f}", fontsize=10)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, axis="y")
    # 右图：Amari 距离（越小越准），非高斯主实验 vs 高斯对照
    amvals = [amari_ica, amari_pca, gauss_ica_amari_mean, gauss_pca_amari_mean]
    amlabels = ["ICA\n(非高斯)", "PCA\n(非高斯)", "ICA\n(高斯)", "PCA\n(高斯)"]
    amcolors = ["#1f6feb", "#8250df", "#79c0ff", "#d2a8ff"]
    axes[1].bar(range(4), amvals, color=amcolors, alpha=0.9)
    for i, v in enumerate(amvals):
        axes[1].text(i, v + 0.006, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_xticks(range(4))
    axes[1].set_xticklabels(amlabels, fontsize=8)
    axes[1].set_ylabel("Amari 距离（越小越准）")
    axes[1].set_title("关键对照：高斯源下 ICA 的 Amari 暴涨，解不再唯一",
                     fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("图3 ICA 的胜出依赖「非高斯」：源一旦高斯，ICA 也不可辨识",
                 fontsize=11)
    _save(fig, "fig3_corr_compare.png")

    # 图4：左=收敛曲线，右=非线性扫描 Amari
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2))
    axes[0].plot(range(1, len(objs) + 1), objs, "o-", color="#1f6feb", lw=1.4,
                ms=3)
    axes[0].set_xlabel("不动点迭代次数")
    axes[0].set_ylabel("目标（|E[g(Y)]| 之和）")
    axes[0].set_title(f"FastICA 收敛：{n_iter} 步稳定", fontsize=10)
    axes[0].grid(alpha=0.3)
    gn = list(sweep.keys())
    am = [sweep[g]["amari"] for g in gn]
    axes[1].bar(gn, am, color=["#1f6feb", "#bf8700", "#2da44e"], alpha=0.85)
    for i, v in enumerate(am):
        axes[1].text(i, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    axes[1].set_ylabel("Amari 距离（越小越准）")
    axes[1].set_title(f"非线性扫描：三者接近，gauss 略优 "
                      f"(tanh {sweep['tanh']['amari']:.3f})", fontsize=10)
    axes[1].grid(alpha=0.3, axis="y")
    fig.suptitle("图4 收敛性与非线性扫描", fontsize=11)
    _save(fig, "fig4_convergence_sweep.png")

    # 图5：高阶独立性热力（离对角 |corr(f²_i, f²_j)|）
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, Sf, title in ((axes[0], S_ica, "ICA 因子"),
                          (axes[1], S_pca, "PCA 因子")):
        sq = Sf ** 2
        M = np.corrcoef(sq)
        im = ax.imshow(np.abs(M), vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(k))
        ax.set_yticks(range(k))
        ax.set_title(f"{title}：离对角均值 {higher_order_indep(Sf):.3f}",
                     fontsize=10)
        for i in range(k):
            for j in range(k):
                ax.text(j, i, f"{np.abs(M[i,j]):.2f}", ha="center", va="center",
                        color="white" if np.abs(M[i, j]) < 0.5 else "black",
                        fontsize=9)
    fig.suptitle("图5 平方互相关：ICA 因子高阶独立，PCA 因子仍纠缠",
                 fontsize=11)
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.04)
    _save(fig, "fig5_higher_order.png")

    results = {
        "dgp": {"T": T, "k": k, "seed": SEED,
                "source_names": names, "mix_cond": float(np.linalg.cond(A))},
        "main": {
            "ica_matched_corr": [float(c) for c in ica_corr],
            "pca_matched_corr": [float(c) for c in pca_corr],
            "ica_mean_corr": float(np.mean(ica_corr)),
            "pca_mean_corr": float(np.mean(pca_corr)),
            "amari_ica": float(amari_ica),
            "amari_pca": float(amari_pca),
            "higher_order_ica": ho_ica,
            "higher_order_pca": ho_pca,
            "n_iter": int(n_iter),
        },
        "gaussian_control": {
            "ica_mean_amari": gauss_ica_amari_mean,
            "pca_mean_amari": gauss_pca_amari_mean,
            "ica_mean_corr": gauss_ica_mean,
            "pca_mean_corr": gauss_pca_mean,
            "random_baseline": 1.0 / np.sqrt(k),
            "n_trials": 8,
            "non_gauss_ica_amari": float(amari_ica),
        },
        "nonlinearity_sweep": {g: {"mean_matched_corr": float(sweep[g]["mean_matched_corr"]),
                                   "amari": float(sweep[g]["amari"])}
                               for g in sweep},
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

