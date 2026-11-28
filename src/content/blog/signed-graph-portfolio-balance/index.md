---
title: "符号网络结构平衡与多空对冲：把负相关写成负边，为什么最平衡子图能拆出干净的多空腿、又在负边稀疏时退化"
description: "40 资产、20 对 20 真分组的合成实验：样本相关矩阵取最强的 25% 边，真实组内相关 0.8394、组间 -0.8386，符号拉普拉斯最小特征值约 0、无不平衡三角；随机翻转 25% 边后最小特征值升到 0.8728、43.38% 三角不平衡，分群准确率从 100% 掉到 67.5%，主导因子暴露从 -3.6 缩到 -1.344。只留 5% 边时图裂成 15 个连通块，准确率 62.5%；样本从 60 增至 120 后准确率稳定在 100%。特质波动升到 20 时组间相关只剩 -0.0061，准确率跌到 60%。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T13:06:00Z'
tags:
  - 量化
  - 网络分析
  - 风险管理
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

普通相关性网络把「关系强」写成正权边，于是正负相关被压成同一种边，符号信息在建模第一步就丢了。但多头腿和空头腿的区别恰恰是符号：同一组资产应该正相关，对冲组之间应该负相关。把相关系数直接写成带符号的边，就能用图论里的结构平衡来判断这份多空结构是不是干净。

我用 40 个资产、20 对 20 的真分组做合成实验：主导因子在两组上载荷分别是 +1.8 与 -1.8，另有两组弱因子和特质噪声。固定种子 `20261128` 的实测结果：

- 500 天样本估计出的相关矩阵里，真组内平均相关 **0.8394**、组间 **-0.8386**；取最强 25% 的边共 117 条，还原分组准确率 **100%**；
- 对应的符号拉普拉斯最小特征值约 **0**，非平衡三角形占比 **0**，说明这张图在符号意义上是完全平衡的；
- 随机翻转 25% 条边的符号后，最小特征值升到 **0.8728**，**43.38%** 的三角形变成非平衡，分群准确率掉到 **67.5%**；
- 用分群构造组内对冲、组间留方向的多空组合，正确分组的主导因子暴露是 **-3.6**，翻转符号后只剩 **-1.344**；组合波动从 **3.5735** 降到 **1.3839**，看似风险变小，其实是因子暴露被破坏；
- 只保留 5% 的边时，图裂成 **15 个连通块**，虽然每个块内部仍平衡、最小特征值仍近似 0，分群准确率却只有 **62.5%**；
- 样本量从 60 增到 120，准确率从 **97.5%** 升到 **100%** 后稳定；特质波动升到 20 时组间相关只剩 **-0.0061**，准确率跌到 **60%**。

![500 天样本估计的带符号相关矩阵](/images/signed-graph-portfolio-balance/fig1_signed_correlation.png)

结论很直接：符号网络不是用来把波动率变小，而是用来确认多空腿是否真的站在相反的风险因子上。符号错了，波动率往往也会变小，但那是因为方向性暴露被对冲掉了。

## 二、什么是结构平衡

社会网络里的结构平衡说的很朴素：朋友的朋友还是朋友，敌人的朋友是敌人。把「朋友」记为正边、「敌人」记为负边，一张图完全平衡当且仅当它可以把节点劈成两堆，堆内全是正边、堆间全是负边。金融上这两堆就是天然的多头组和空头组。

代数判据来自符号拉普拉斯。设 $A_{ij}\in\{-1,0,1\}$ 是带符号邻接矩阵，$D_{ii}=\sum_j|A_{ij}|$，则

$$
L=D-A.
$$

对于连通图，$L$ 的最小特征值 $\lambda_{\min}(L)=0$ 当且仅当图完全平衡；非平衡程度越高，它离 0 越远。另一个直观指标是非平衡三角形比例：三个节点之间三条边的符号乘积为负，就形成一个「敌人的敌人还是敌人」的悖论。本次实验中正确符号图的两项指标分别是 0 与 0，翻转 25% 边后变成 0.8728 与 43.38%。

三角形比例可以用矩阵迹快速统计：$A^3$ 的对角和把每个三角形按六个方向数了一遍，平衡三角形贡献正号、非平衡三角形贡献负号，因此用 $(\,1-\operatorname{tr}(A^3)/(6T)\,)/2$ 就能换算出非平衡占比，其中 $T$ 是三角形总数。这个量比特征值更容易解释：43.38% 意味着每两个多资产三角关系里就有一个符号逻辑冲突，此时按 Fiedler 向量切出来的组别必然要牺牲一部分资产。

为什么不用普通的相关性网络？因为最小生成树或普通拉普拉斯只看边的绝对值，负相关会被当成「弱一点的正相关」。在本次数据里，正是那 59 条负边把两组资产区分开；把它们全部改成正边，图仍然连通、谱仍然平滑，却再也读不出多空结构。

![正确符号与翻转符号下的符号拉普拉斯谱](/images/signed-graph-portfolio-balance/fig2_laplacian_spectrum.png)

## 三、从 Fiedler 向量到多空权重

最小特征值对应的特征向量叫 Fiedler 向量。完全平衡时，它的分量符号正好给出两堆的归属；接近平衡时，它给出「最接近平衡」的切分。把符号取出来，再让每组内部等权、两组合起来净敞口为零，就得到一组可交易的多空腿。

![恢复出的分组与对应多空权重](/images/signed-graph-portfolio-balance/fig3_partition_weights.png)

这里必须澄清一个容易误解的点：组内等权是在对冲特质噪声，组间一多一空是在**保留**主导因子的方向暴露，不是把因子风险也对冲掉。本次实验里正确分组的组合波动 3.5735，反而比只持有多头组的 1.7919 更高，因为它同时吃进两组相反载荷叠加出来的因子方向。它的价值在于方向纯度：正确分组的主导因子暴露是 -3.6，符号翻转后只剩 -1.344。波动率下降在这里不是好消息，而是分组失效的信号。

真实部署时，可以把这条暴露当成监控指标：每周用滚动窗口重估符号图，记录最小特征值、非平衡三角形比例、连通块数和组合的因子暴露。只要前三项同时恶化，或者因子暴露相对设计值缩水超过三分之一，就应该降低这组多空腿的仓位，而不是等回测曲线走坏再反应。相比收益指标，这些结构指标更早也更稳定。

合成数据的另一个用途是设定阈值。我们已知真分组，因此可以问：在多大的边噪声、多短的历史窗口下，Fiedler 切分仍然可靠？本次实验给出的答案是 120 天以上、25% 边比例、特质波动不超过主导因子贡献的十倍。换到真实资产池时，这些阈值要用同样的真值实验重新标定，不能直接照搬。

## 四、稀疏边会让平衡判据失灵

![保留边比例对分群准确率和组合波动的影响](/images/signed-graph-portfolio-balance/fig4_edge_sparsity.png)

我只保留最强的一部分边，从 5% 一路加到 60%。结果里有一个反直觉现象：所有比例下最小特征值都近似 0，看起来「完全平衡」，但分群准确率从 62.5% 一路爬到 100%。原因藏在连通块数量里：5% 边时图裂成 15 块，每块各自平衡，自然也能给出零特征值；只是这些局部平衡拼不出全局两分组。10% 时还有 4 块，准确率 95%；到 25% 时才成为单一连通块，准确率 100%。

所以工程上不能只看 $\lambda_{\min}$，必须同时检查连通性。一个被切成许多小岛的图，平衡性检验会给出漂亮的零，却完全没有交易意义。

## 五、样本长度与特质噪声的边界

![样本长度与特质噪声对符号恢复的影响](/images/signed-graph-portfolio-balance/fig5_sample_noise.png)

样本量从 60 增到 120，准确率从 97.5% 升到 100%，之后一直稳定。这说明在本实验的信噪比下，120 天日频数据就足以把主导因子的符号方向估准；继续拉长窗口主要改善权重精度，而不是改变分组。

真正危险的是特质噪声。特质波动从 0.7 升到 20，组间平均相关从 -0.8394 被稀释到 -0.0061。有趣的是，准确率直到特质波动 8 仍然保持 100%：因为符号只需要正负方向正确，不需要幅度大。到 12 时首次掉到 97.5%，到 20 时才崩到 60%，组合波动也随之跌到 0.7951——又一次「波动变小、结构变坏」。实务中若发现某个多空组合的波动异常低，先别夸风控，检查它是不是已经不再持有预设的风险方向。

## 六、完整实现

下面的脚本只依赖 numpy 与 matplotlib，运行后生成五张图与 `results.json`，所有随机性由固定种子控制。真实数据落地时，建议把显著性检验加到选边环节：只有统计上显著为正或为负的相关才写成边，否则噪声会制造大量虚假的非平衡三角形。

```python
# -*- coding: utf-8 -*-
"""Signed graph balance and long-short hedge experiment (deterministic)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "signed-graph-portfolio-balance")
os.makedirs(OUT, exist_ok=True)

N = 40
HALF = N // 2
TRUE_SIGN = np.r_[np.ones(HALF), -np.ones(HALF)]
FACTOR_VAR = np.array([1.0, 0.3, 0.3])
IDIO = 0.7


def loadings(rng, idio=IDIO):
    b0 = TRUE_SIGN * 1.8
    b1 = rng.normal(0.0, 0.5, N)
    b2 = rng.normal(0.0, 0.5, N)
    return np.column_stack((b0, b1, b2))


def make_returns(seed, n, idio=IDIO, B=None):
    rng = np.random.default_rng(seed)
    if B is None:
        B = loadings(rng, idio)
    f = rng.normal(size=(n, 3)) * np.sqrt(FACTOR_VAR)
    e = rng.normal(0.0, idio, size=(n, N))
    return f @ B.T + e, B


def corr_matrix(X):
    Z = X - X.mean(axis=0)
    C = Z.T @ Z
    d = np.sqrt(np.diag(C))
    return C / np.outer(d, d)


def upper_pairs():
    return [(i, j) for i in range(N) for j in range(i + 1, N)]


PAIRS = upper_pairs()


def signed_graph(C, edge_fraction=0.25, sign_flip=0.0, rng_seed=SEED):
    order = np.argsort(-np.abs(np.array([C[i, j] for i, j in PAIRS])))
    keep = set(order[: int(round(edge_fraction * len(PAIRS)))])
    A = np.zeros((N, N))
    rng = np.random.default_rng(rng_seed)
    for idx, (i, j) in enumerate(PAIRS):
        if idx not in keep:
            continue
        s = 1.0 if C[i, j] >= 0 else -1.0
        if sign_flip > 0 and rng.random() < sign_flip:
            s = -s
        A[i, j] = A[j, i] = s
    return A


def connected_components(A):
    seen = set()
    count = 0
    for i in range(N):
        if i in seen:
            continue
        count += 1
        stack = [i]
        seen.add(i)
        while stack:
            u = stack.pop()
            for v in np.where(np.abs(A[u]) > 0)[0]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
    return count


def signed_laplacian(A):
    return np.diag(np.sum(np.abs(A), axis=1)) - A


def frustration(A):
    binary = (np.abs(A) > 0).astype(float)
    tri = np.trace(binary @ binary @ binary) / 6.0
    if tri == 0:
        return 0.0
    prod = np.trace(A @ A @ A) / (6.0 * tri)
    return float(np.clip((1.0 - prod) / 2.0, 0.0, 1.0))


def fiedler_sign(A):
    L = signed_laplacian(A)
    vals, vecs = np.linalg.eigh(L)
    v = vecs[:, 0]
    s = np.sign(v)
    if s @ TRUE_SIGN < 0:
        s = -s
    return s, float(vals[0]), vals


def weights_from_sign(s):
    w = np.zeros(N)
    w[s > 0] = -1.0 / max(1, np.sum(s > 0))
    w[s < 0] = 1.0 / max(1, np.sum(s < 0))
    return w


def portfolio_vol(R, w):
    return float(np.std(R @ w, ddof=1))


def sign_accuracy(s):
    return float(np.mean(s == TRUE_SIGN))


# Baseline: population graph, sample graph, and a graph with 25% signs flipped.
B = loadings(np.random.default_rng(SEED))
train, _ = make_returns(SEED, 500, B=B)
test, _ = make_returns(SEED + 1, 1000, B=B)
B_pop = loadings(np.random.default_rng(SEED))
cov_pop = B_pop @ np.diag(FACTOR_VAR) @ B_pop.T + np.eye(N) * IDIO ** 2
C_pop = cov_pop / np.outer(np.sqrt(np.diag(cov_pop)), np.sqrt(np.diag(cov_pop)))
C_train = corr_matrix(train)

A_pop = signed_graph(C_pop, 0.25)
A_train = signed_graph(C_train, 0.25)
A_flip = signed_graph(C_train, 0.25, sign_flip=0.25, rng_seed=SEED + 2)

s_pop, lam_pop, eig_pop = fiedler_sign(A_pop)
s_train, lam_train, eig_train = fiedler_sign(A_train)
s_flip, lam_flip, eig_flip = fiedler_sign(A_flip)

w_true = weights_from_sign(TRUE_SIGN)
w_train = weights_from_sign(s_train)
w_flip = weights_from_sign(s_flip)
rng_rand = np.random.default_rng(SEED + 3)
s_rand = TRUE_SIGN * rng_rand.choice([-1.0, 1.0], size=N)
w_rand = weights_from_sign(s_rand)

vol_true = portfolio_vol(test, w_true)
vol_train = portfolio_vol(test, w_train)
vol_flip = portfolio_vol(test, w_flip)
vol_rand = portfolio_vol(test, w_rand)
vol_equal = portfolio_vol(test, weights_from_sign(np.where(TRUE_SIGN > 0, 1.0, 0.0)))

cross_corr = float(np.mean(C_pop[np.ix_(TRUE_SIGN > 0, TRUE_SIGN < 0)]))
within_corr = float(np.mean(C_pop[np.ix_(TRUE_SIGN > 0, TRUE_SIGN > 0)][np.triu_indices(HALF, 1)]))

# Fig 1: sample signed correlation heatmap.
fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(C_train, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xlabel("Asset index")
ax.set_ylabel("Asset index")
ax.set_title("Sample signed correlation, n=500")
fig.colorbar(im, ax=ax, label="Correlation")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_signed_correlation.png"), dpi=150)
plt.close(fig)

# Fig 2: signed Laplacian spectra.
fig, ax = plt.subplots(figsize=(8, 4.8))
idx = np.arange(1, N + 1)
ax.plot(idx, eig_pop, "o-", color="#1a7f4b", lw=1.6, label="Population signs")
ax.plot(idx, eig_train, "s-", color="#1764ab", lw=1.6, label="Sample signs")
ax.plot(idx, eig_flip, "^-", color="#c23b22", lw=1.6, label="25% signs flipped")
ax.set_xlabel("Eigenvalue rank")
ax.set_ylabel("Signed Laplacian eigenvalue")
ax.set_title("Balance shows up in the smallest eigenvalue")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_laplacian_spectrum.png"), dpi=150)
plt.close(fig)

# Fig 3: recovered partition and resulting long-short weights.
fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
axes[0].plot(TRUE_SIGN, color="#333333", lw=1.6, label="True camp")
axes[0].plot(s_train, color="#1764ab", lw=1.2, alpha=0.85, label="Fiedler camp")
axes[0].set_xlabel("Asset index")
axes[0].set_ylabel("Camp sign")
axes[0].set_title("Recovered signed partition")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].bar(np.arange(N), w_train, color=np.where(w_train > 0, "#c23b22", "#1764ab"))
axes[1].set_xlabel("Asset index")
axes[1].set_ylabel("Portfolio weight")
axes[1].set_title("Dollar-neutral long-short legs")
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_partition_weights.png"), dpi=150)
plt.close(fig)

# Fig 4: edge sparsity and sign noise.
edge_fractions = [0.05, 0.10, 0.15, 0.25, 0.40, 0.60]
sparse_lam, sparse_acc, sparse_vol, sparse_comp = [], [], [], []
for frac in edge_fractions:
    A = signed_graph(C_train, frac)
    s, lam, _ = fiedler_sign(A)
    sparse_lam.append(lam)
    sparse_acc.append(sign_accuracy(s))
    sparse_vol.append(portfolio_vol(test, weights_from_sign(s)))
    sparse_comp.append(connected_components(A))

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(edge_fractions, sparse_vol, "o-", color="#c23b22", lw=2, label="Test vol")
ax2 = ax.twinx()
ax2.plot(edge_fractions, sparse_lam, "s--", color="#1764ab", lw=1.7, label="Smallest Laplacian eigenvalue")
ax2.plot(edge_fractions, sparse_acc, "^--", color="#1a7f4b", lw=1.7, label="Camp accuracy")
ax.set_xlabel("Fraction of asset pairs kept as edges")
ax.set_ylabel("Daily portfolio volatility")
ax2.set_ylabel("Eigenvalue / accuracy")
ax.set_title("Too few signed edges lose the partition")
ax.legend(loc="upper right")
ax2.legend(loc="lower right")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_edge_sparsity.png"), dpi=150)
plt.close(fig)

# Fig 5: sample length and idiosyncratic noise.
sample_sizes = [60, 120, 250, 500, 1000, 2000]
size_acc, size_vol, size_lam = [], [], []
for n in sample_sizes:
    Xn, _ = make_returns(SEED + 10, n, B=B)
    An = signed_graph(corr_matrix(Xn), 0.25)
    sn, ln, _ = fiedler_sign(An)
    size_acc.append(sign_accuracy(sn))
    size_vol.append(portfolio_vol(test, weights_from_sign(sn)))
    size_lam.append(ln)

idio_levels = [0.7, 2.5, 5.0, 8.0, 12.0, 20.0]
idio_acc,idio_vol, idio_corr = [], [], []
for level in idio_levels:
    Xi, _ = make_returns(SEED + 20, 500, idio=level, B=B)
    Cx = corr_matrix(Xi)
    Ai = signed_graph(Cx, 0.25)
    si, _, _ = fiedler_sign(Ai)
    idio_acc.append(sign_accuracy(si))
    idio_vol.append(portfolio_vol(test, weights_from_sign(si)))
    idio_corr.append(float(np.mean(Cx[np.ix_(TRUE_SIGN > 0, TRUE_SIGN < 0)])))

fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.6))
axes[0].plot(sample_sizes, size_acc, "o-", color="#1a7f4b", lw=2, label="Camp accuracy")
axes[0].plot(sample_sizes, size_vol, "s--", color="#c23b22", lw=1.7, label="Test vol")
axes[0].set_xlabel("Training sample length")
axes[0].set_ylabel("Accuracy / volatility")
axes[0].set_title("Correlation sign needs enough history")
axes[0].legend()
axes[0].grid(alpha=0.3)
axes[1].plot(idio_levels, idio_corr, "o-", color="#7b3fa1", lw=2, label="Cross-camp correlation")
axes[1].plot(idio_levels, idio_acc, "s--", color="#1a7f4b", lw=1.7, label="Camp accuracy")
axes[1].plot(idio_levels, idio_vol, "^--", color="#c23b22", lw=1.7, label="Test vol")
axes[1].set_xlabel("Idiosyncratic volatility")
axes[1].set_ylabel("Correlation / accuracy / volatility")
axes[1].set_title("Idiosyncratic noise erases signed edges")
axes[1].legend()
axes[1].grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_sample_noise.png"), dpi=150)
plt.close(fig)

results = {
    "assets": N,
    "true_cross_camp_correlation": cross_corr,
    "true_within_camp_correlation": within_corr,
    "edge_fraction": 0.25,
    "population_lambda_min": lam_pop,
    "sample_lambda_min": lam_train,
    "flipped_lambda_min": lam_flip,
    "population_frustration": frustration(A_pop),
    "sample_frustration": frustration(A_train),
    "flipped_frustration": frustration(A_flip),
    "sample_camp_accuracy": sign_accuracy(s_train),
    "flipped_camp_accuracy": sign_accuracy(s_flip),
    "vol_long_camp": vol_equal,
    "vol_true_partition": vol_true,
    "vol_sample_partition": vol_train,
    "vol_flipped_partition": vol_flip,
    "vol_random_partition": vol_rand,
    "factor_exposure_true": float(w_true @ B[:, 0]),
    "factor_exposure_sample": float(w_train @ B[:, 0]),
    "factor_exposure_flipped": float(w_flip @ B[:, 0]),
    "factor_exposure_random": float(w_rand @ B[:, 0]),
    "factor_purity_true": abs(float(w_true @ B[:, 0])) / vol_true,
    "factor_purity_sample": abs(float(w_train @ B[:, 0])) / vol_train,
    "factor_purity_flipped": abs(float(w_flip @ B[:, 0])) / vol_flip,
    "vol_reduction_pct": 100.0 * (1.0 - vol_train / vol_equal),
    "vol_gap_vs_oracle_pct": 100.0 * (vol_train / vol_true - 1.0),
    "edge_fractions": edge_fractions,
    "sparse_lambdas": [float(x) for x in sparse_lam],
    "sparse_accuracies": [float(x) for x in sparse_acc],
    "sparse_vols": [float(x) for x in sparse_vol],
    "sparse_components": sparse_comp,
    "sample_sizes": sample_sizes,
    "size_accuracies": [float(x) for x in size_acc],
    "size_vols": [float(x) for x in size_vol],
    "idio_levels": idio_levels,
    "idio_cross_correlations": idio_corr,
    "idio_accuracies": [float(x) for x in idio_acc],
    "idio_vols": [float(x) for x in idio_vol],
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    pass
```

```

把符号网络当作风控仪表盘，比把它当作选股信号更稳妥：它回答的是「我的多空腿是否还站在相反方向上」，而不是「明天哪只股票会涨」。
