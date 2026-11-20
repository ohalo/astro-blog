---
title: "回归显著性在杠杆点面前有多脆：OLS 标准误只剩真实值的 46%，HC0 也救不回来"
description: "异方差稳健标准误几乎是量化研究的标配，但 HC0 到底稳健到什么程度？本文在真值已知、带一个高杠杆点的横截面回归上把 OLS、HC0–HC3 与 Rademacher wild bootstrap 正面对拍。杠杆点从 1 拉到 5 个标准差：OLS 的 t 检验虚警率从 0.168 崩到 0.397，HC0 从 0.105 爬到 0.201，HC3 稳在 0.072–0.091。标准误的准确性更触目：杠杆 5 处 OLS 的平均标准误只有蒙特卡洛真实标准差的 0.461 倍、HC0 0.645 倍、HC3 0.916 倍——严重低估是虚警的根源。反直觉的是样本量：杠杆固定时把 n 从 30 加到 400，OLS 虚警率只从 0.211 勉强降到 0.159，几乎不收敛，因为单个杠杆点的hat 值不随 n 稀释。n=30 加杠杆 4 的极端场景下 wild bootstrap 虚警率 0.055，比 HC3 的 0.098 更贴近 5%。结论：稳健标准误买的不是免死金牌，杠杆点 + 异方差才是真正的杀手组合。"
publishDate: '2026-11-20T05:20:00Z'
tags:
  - 量化
  - 交易
  - 计量经济学
  - 统计推断
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

因子回归里跑一句 `.fit()` 拿到 t 值、看到星号就宣布因子显著，是量化研究里最流畅也最危险的动作。异方差稳健标准误（heteroskedasticity-robust SE）几乎是标配，于是很多人以为加个 `cov_type='HC0'` 就万事大吉。本文在真值已知的数据上把这件事钉死：**异方差 + 高杠杆点，才是让 t 检验彻底失效的组合，而 HC0 远没有传说中万能。**

四条结论，一条比一条反直觉：

1. **杠杆点能把虚警率顶到 40%。** 数据生成过程里 b1 的真值严格为 0，名义 5% 的检验本该只有 5% 的概率误报。可当那个远离主体的点从 1 个标准差拉到 5 个标准差，OLS 的虚警率从 0.168 一路崩到 0.397——**四次里误报近两次**。HC0 从 0.105 爬到 0.201，也远不是「稳健」二字能概括的。只有 HC3 稳在 0.072–0.091，最接近名义水平。
2. **根源是标准误被系统性低估，而且低估得极有规律。** 杠杆 5 处，OLS 的平均标准误只有蒙特卡洛真实标准差的 0.461 倍，HC0 是 0.645 倍，HC1 0.662 倍，HC2 0.760 倍，HC3 0.916 倍。标准误小一半，t 值就大近一倍，虚警当然满天飞。**t 检验的崩塌不是随机的，是标准误低估的直接、可量化的后果。**
3. **加样本量救不了杠杆点。** 直觉上「样本不够大所以不稳」，那就加大样本。可把杠杆点固定在 3 个标准差、n 从 30 加到 400，OLS 虚警率只从 0.211 勉强降到 0.159，几乎纹丝不动；HC0 从 0.144 收敛到 0.051，HC3 从 0.092 收敛到 0.051。**为什么 OLS 不收敛？因为单个杠杆点的 hat 值 h_ii 不随 n 稀释**，它一个人占了设计矩阵里一小块固定的信息份额，样本再多也摊不掉。
4. **小样本极端场景下，wild bootstrap 比 HC3 更准。** n=30、杠杆 4 的场景下，OLS 虚警率 0.287、HC0 0.182、HC3 0.098，而 Rademacher wild bootstrap 是 0.055——几乎完美命中 5%。它不靠渐近近似，靠重抽样直接构造统计量的零分布，代价是计算量大了近四百倍（本文每档用 399 次重抽样）。

![图1：异方差——残差的离散程度随 |x| 增大，最右侧的点同时是高杠杆点](/images/heteroskedasticity-robust-se-wild-bootstrap/fig1_hetero.png)

## 二、杠杆点为什么特别毒

先把设定讲清楚。回归是 y=b0+b1·x+ε，真值 b0=b1=0，噪声 ε 的标准差随 |x| 线性放大：σ_i=0.5+0.5|x_i|。这就是最典型的异方差——波动随自变量增大，个股收益对市场收益的回归、规模因子对市值的回归，都是这个形状。再往数据里塞一个「杠杆点」：最后一个观测的 x 取值被拉到 1 到 5 个标准差之外。杠杆点指的是在自变量方向上远离主体的观测，它离得越远，对拟合直线的杠杆作用越大。

杠杆值 h_ii 衡量第 i 个点对自己拟合值的影响程度，对角元素越大，这个点越能单独撬动斜率。普通 OLS 标准误的公式是 σ̂²·(X′X)⁻¹，它假设所有观测的噪声方差相同，用一个统一的 σ̂² 去乘。异方差一进来，这个假设就垮了：杠杆点方差又大、影响又强，它的残差被强行按统一方差缩放，结果就是标准误算错。

HC0 到 HC3 是一系列越来越激进的修正，核心区别在残差怎么除：

$$
\widehat{V}_{HC_l}=(X'X)^{-1}\Big(\sum_i \frac{\hat\varepsilon_i^2}{1-h_{ii}}\Big)(X'X)^{-1}
$$

其中 HC0 的分母是 1，HC1 做自由度校正，HC2 除以 (1−h_ii)，HC3 除以 (1−h_ii)²。**分母里的 h_ii 就是杠杆的影子**：点的杠杆越高，1−h_ii 越小，除完之后的修正幅度越大。HC3 修正最猛，这正是它在杠杆场景下最接近真值的原因——它把高杠杆点的残差按平方放大，抵消了杠杆对标准误的压缩。

## 三、虚警率与标准误的两条曲线

把杠杆点从 1 拉到 5，每档跑 2000 次重复，统计 b1 的 t 检验有多少次越过 1.96。结果见下图：OLS 从 0.168 崩到 0.397，HC0 从 0.105 爬到 0.201，HC1 略好于 HC0，HC2 再好一档，HC3 全程压在 0.072–0.091。**杠杆越高，各方法的差距拉得越开**——杠杆 1 时 HC3（0.077）和 OLS（0.168）差 0.09，杠杆 5 时差 0.31。这说明稳健标准误的价值恰恰在最危险的场景里最大，而那正是最容易被忽略的场景。

![图2：杠杆越高虚警越失控，OLS 与 HC0 崩得最快，HC3 最接近 5%](/images/heteroskedasticity-robust-se-wild-bootstrap/fig2_size_leverage.png)

虚警的根子藏在标准误里。用 800 次重复算出 b1 估计量的蒙特卡洛标准差当「真实标准误」，再把每种方法报告的平均标准误除以它，比值 1.0 才算准确。杠杆 5 处：OLS 0.461、HC0 0.645、HC1 0.662、HC2 0.760、HC3 0.916。**OLS 把标准误砍掉了一半还多**，t 值被放大到两倍，虚警率飙到 0.397 完全在预料之中。HC3 的 0.916 仍有轻微低估，但已经把误差从 54% 压到 8%。

![图3：杠杆升高时 OLS/HC0 的标准误严重低估，HC3 始终贴近真实值](/images/heteroskedasticity-robust-se-wild-bootstrap/fig3_se_ratio.png)

这里有个容易被忽略的细节：杠杆 1 到 2 时 HC3 的比值是 1.015、1.004，**略微高估**；杠杆拉到 5 才变成 0.916 的低估。也就是说 HC3 在低杠杆处偏保守、高杠杆处偏激进，它的「稳健」是相对的，不是处处完美。这也解释了为什么高杠杆处 HC3 的虚警率是 0.091 而非精确的 0.05——修到位了，但没修过头。

## 四、加样本量救不了，wild bootstrap 能

「样本量不够大」是最常见的辩解。本文把杠杆点固定在 3 个标准差，n 从 30 加到 400：OLS 虚警率 0.211→0.190→0.181→0.175→0.159，**几乎不动**；HC0 0.144→0.083→0.077→0.065→0.051，HC3 0.092→0.056→0.065→0.062→0.051，都稳稳收敛到 5%。

为什么 OLS 不动？杠杆值 h_ii 是帽子矩阵的对角元素，单个高杠杆点的 h_ii 由它离主体多远决定，和样本里有多少个普通点几乎无关。n 增大只增加普通观测，那个杠杆点的 h_ii 依然很大，它对标准误的污染就一直在。**「大样本能治异方差」是只对了一半的说法：它治得了普通异方差，治不了杠杆点。**

![图4：n=30、杠杆 4 时 wild bootstrap 虚警率 0.055，优于 HC3 的 0.098](/images/heteroskedasticity-robust-se-wild-bootstrap/fig4_wild.png)

当样本既小又有杠杆时，渐近近似本身就可疑，这时 wild bootstrap 是个更靠谱的选择。它的做法不是套公式，而是重抽样：用受限模型（b1=0）的残差乘以随机的 Rademacher 权重（±1 各半）生成新的因变量，重新估计 b1 和它的稳健标准误，重复 399 次构造出 t 统计量的经验零分布，再看真实的 t 值在这个分布里排第几。n=30、杠杆 4 的极端场景下，它的虚警率是 0.055，比 HC3 的 0.098 更贴近 5%——因为它不依赖「n 足够大时 t 分布近似成立」这个前提，代价是每档要多跑近四百次回归。

![图5：杠杆固定时 OLS 虚警率到 n=400 仍有 0.159，HC0/HC3 收敛到 5%](/images/heteroskedasticity-robust-se-wild-bootstrap/fig5_nscan.png)

## 五、实践建议与边界

给量化研究的操作顺序是：先画自变量与残差的散点图，看有没有方差随 x 放大的形状，再看有没有孤立在远端的杠杆点。如果两者都有，直接用 HC3，小样本（n<50）再加 wild bootstrap 交叉验证；HC0 只适合没有杠杆点的温和异方差，把它当万能选项是在赌数据里没有异常点。看到一个带星号的因子系数，先问一句「这个显著性在换用 HC3 后还在吗」，往往比再多加十个控制变量更有用。

边界同样要讲清。本文只检验了单个回归元、单个杠杆点、线性方差函数 σ=0.5+0.5|x| 的情形；真实数据里可能有多个杠杆点相互掩蔽，方差函数可能是非线性甚至未知的，高维控制变量下的杠杆结构更复杂。wild bootstrap 的有效性依赖残差至少满足对称性之外的条件（本文用 Rademacher 权重配受限残差），在强非对称误差下需要用 Mammen 权重或其他变体。本文的蒙特卡洛用固定 seed 保证可复现，所有数字都能运行下方代码逐位重现，但「固定 seed 下的一组数」不等于「所有数据生成过程下的定理」——把结论外推到你的数据前，先确认数据形状确实落在本文覆盖的范围里。

## 六、可运行代码

下面这段代码就是生成本文全部数字与五张图的完整实现：数据生成、OLS 与 HC0–HC3 标准误的手算、wild bootstrap 重抽样、虚警率与标准误准确性模拟、样本量扫描都在里面，直接运行即可复现，统计结果写入 `results.json` 且与图同目录。核心是帽子矩阵与 meat 矩阵的向量化计算，以及用固定 seed 保证每次运行数字完全一致。

```python
# -*- coding: utf-8 -*-
"""Heteroskedasticity-robust standard errors HC0-HC3 vs wild bootstrap: leverage breaks HC0.

Single source of truth for all numbers and figures in the article.
DGP: y_i = b0 + b1*x_i + eps_i, eps_i ~ N(0, sigma_i^2), sigma_i grows with |x_i|.
A high-leverage point (one x far out) with high variance is the killer combination.
We compare OLS classical SE, HC0, HC1, HC2, HC3 and Rademacher wild bootstrap
on size of t-test for b1, and on SE accuracy vs the Monte-Carlo truth.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
REPS = 2000
B_WILD = 399
Z95 = stats.norm.ppf(0.975)

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "public", "images", "heteroskedasticity-robust-se-wild-bootstrap")
os.makedirs(_OUT, exist_ok=True)


def design(n, lever, rng):
    """x with one high-leverage point at `lever` (in sd units); hetero sigma=|x|-scaled."""
    x = rng.standard_normal(n)
    x[-1] = lever
    sigma = 0.5 + 0.5 * np.abs(x)
    X = np.column_stack([np.ones(n), x])
    return x, sigma, X


def fit_se(x, sigma, X, rng):
    """Return b1 and SEs from OLS / HC0-HC3 for one dataset with b1=0 true."""
    n = len(x)
    eps = rng.normal(0, sigma)
    y = eps  # b0=b1=0 under H0
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    se_ols = np.sqrt(XtX_inv[1, 1] * (resid @ resid) / (n - 2))
    h = np.einsum("ij,jk,ik->i", X, XtX_inv, X)
    meat0 = (X * resid[:, None]).T @ (X * resid[:, None])
    v0 = XtX_inv @ meat0 @ XtX_inv
    se_hc0 = np.sqrt(v0[1, 1])
    se_hc1 = se_hc0 * np.sqrt(n / (n - 2))
    meat2 = (X * (resid / np.sqrt(1 - h))[:, None]).T @ (X * (resid / np.sqrt(1 - h))[:, None])
    se_hc2 = np.sqrt((XtX_inv @ meat2 @ XtX_inv)[1, 1])
    meat3 = (X * (resid / (1 - h))[:, None]).T @ (X * (resid / (1 - h))[:, None])
    se_hc3 = np.sqrt((XtX_inv @ meat3 @ XtX_inv)[1, 1])
    return beta[1], dict(ols=se_ols, hc0=se_hc0, hc1=se_hc1, hc2=se_hc2, hc3=se_hc3), \
        X, XtX_inv, resid, h


def wild_pvalue(y, X, XtX_inv, resid, B, rng):
    """Rademacher wild bootstrap, restricted residuals, two-sided p for b1."""
    n = len(X)
    Xr = X[:, [0]]  # restricted design (b1=0)
    beta_r = np.linalg.lstsq(Xr, y, rcond=None)[0]
    rres = y - Xr @ beta_r
    t_obs = None
    # observed using HC0-ish baseline stat on full model
    beta_full = XtX_inv @ (X.T @ y)
    rf = y - X @ beta_full
    meat = (X * rf[:, None]).T @ (X * rf[:, None])
    se = np.sqrt((XtX_inv @ meat @ XtX_inv)[1, 1])
    if se < 1e-12:
        return 1.0
    t_obs = abs(beta_full[1] / se)
    count = 0
    for _ in range(B):
        w = rng.choice([-1.0, 1.0], size=n)
        yb = Xr @ beta_r + (rres * w)
        bb = XtX_inv @ (X.T @ yb)
        rb = yb - X @ bb
        meatb = (X * rb[:, None]).T @ (X * rb[:, None])
        seb = np.sqrt((XtX_inv @ meatb @ XtX_inv)[1, 1])
        if seb > 1e-12 and abs(bb[1] / seb) >= t_obs:
            count += 1
    return (count + 1) / (B + 1)


def size_experiment(lever, n, reps, wild=False, seed=SEED):
    rng = np.random.default_rng(seed)
    rej = {k: 0 for k in ["ols", "hc0", "hc1", "hc2", "hc3"]}
    if wild:
        rej["wild"] = 0
    for _ in range(reps):
        x, sigma, X = design(n, lever, rng)
        b1, ses, X, XtX_inv, resid, h = fit_se(x, sigma, X, rng)
        y = X @ np.array([0.0, b1]) + resid  # reconstruct y from fitted pieces
        for k, se in ses.items():
            rej[k] += abs(b1 / se) > Z95
        if wild:
            p = wild_pvalue(y, X, XtX_inv, resid, B_WILD, rng)
            rej["wild"] += p < 0.05
    return {k: v / reps for k, v in rej.items()}


def se_accuracy(lever, n, reps, seed=SEED + 1):
    """Compare mean estimated SE vs Monte-Carlo SD of b1 (the truth)."""
    rng = np.random.default_rng(seed)
    ses_sum = {k: 0.0 for k in ["ols", "hc0", "hc1", "hc2", "hc3"]}
    bs = []
    for _ in range(reps):
        x, sigma, X = design(n, lever, rng)
        b1, ses, _, _, _, _ = fit_se(x, sigma, X, rng)
        bs.append(b1)
        for k in ses_sum:
            ses_sum[k] += ses[k]
    true_sd = float(np.std(bs))
    return {"true_sd": true_sd,
            "mean_se": {k: v / reps for k, v in ses_sum.items()}}


def plot1_rows():
    rng = np.random.default_rng(SEED + 2)
    n = 60
    x, sigma, X = design(n, 3.0, rng)
    eps = rng.normal(0, sigma)
    y = eps
    fig, ax = plt.subplots(figsize=(8, 4.6))
    sc = ax.scatter(x, y, c=sigma, cmap="viridis", s=35, edgecolor="k", linewidth=.3)
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("x (last point = high leverage)")
    ax.set_ylabel("residual")
    ax.set_title("Fig1: heteroskedasticity - spread grows with |x|; far point has high leverage")
    fig.colorbar(sc, ax=ax, label="sigma_i")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_hetero.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_hetero.png"


def plot2_size(size_res):
    keys = ["ols", "hc0", "hc1", "hc2", "hc3"]
    labels = ["OLS", "HC0", "HC1", "HC2", "HC3"]
    levers = [float(x) for x in size_res["lever"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, lab in zip(keys, labels):
        ax.plot(levers, size_res[k], "o-", lw=2, label=lab)
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xlabel("leverage of the far point (sd units)")
    ax.set_ylabel("size of t-test for b1")
    ax.set_title("Fig2: high leverage inflates OLS and HC0 size; HC3 stays closest to 5%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_size_leverage.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_size_leverage.png"


def plot3_se(acc):
    keys = ["ols", "hc0", "hc1", "hc2", "hc3"]
    labels = ["OLS", "HC0", "HC1", "HC2", "HC3"]
    levers = [float(x) for x in acc["lever"]]
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    for k, lab in zip(keys, labels):
        ratios = [acc["rows"][i]["mean_se"][k] / acc["rows"][i]["true_sd"]
                  for i in range(len(levers))]
        ax.plot(levers, ratios, "o-", lw=2, label=lab)
    ax.axhline(1.0, color="k", ls="--", lw=1, label="true SE")
    ax.set_xlabel("leverage of the far point")
    ax.set_ylabel("mean estimated SE / true MC SD")
    ax.set_title("Fig3: OLS/HC0 underestimate SE as leverage rises; HC3 stays near the truth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_se_ratio.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_se_ratio.png"


def plot4_wild(wild_res):
    methods = ["ols", "hc0", "hc3", "wild"]
    labels = ["OLS", "HC0", "HC3", "Wild boot"]
    xs = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    vals = [wild_res["size"][m] for m in methods]
    bars = ax.bar(xs, vals, color=["#c0392b", "#e67e22", "#27ae60", "#2980b9"])
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    for xi, v in zip(xs, vals):
        ax.text(xi, v + .005, f"{v:.3f}", ha="center", fontsize=9)
    ax.set_xticks(xs); ax.set_xticklabels(labels)
    ax.set_ylabel("size")
    ax.set_title("Fig4: wild bootstrap (0.055) beats HC3 (0.098) size with n=30 + leverage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_wild.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_wild.png"


def plot5_nscan(nscan):
    ns = [int(x) for x in nscan["n"]]
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for k, lab, c in [("ols", "OLS", "#c0392b"), ("hc0", "HC0", "#e67e22"),
                      ("hc3", "HC3", "#27ae60")]:
        ax.plot(ns, nscan[k], "o-", lw=2, color=c, label=lab)
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xlabel("sample size n (leverage fixed)")
    ax.set_ylabel("size")
    ax.set_title("Fig5: OLS leverage size bias persists at n=400; HC0/HC3 converge to 5%")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_nscan.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_nscan.png"


def main():
    # Size vs leverage
    levers = [1.0, 2.0, 3.0, 4.0, 5.0]
    size_rows = []
    for lv in levers:
        size_rows.append(size_experiment(lv, 40, REPS))
    size_res = {"lever": levers}
    for k in ["ols", "hc0", "hc1", "hc2", "hc3"]:
        size_res[k] = [r[k] for r in size_rows]

    # SE accuracy vs leverage
    acc_rows = []
    for lv in levers:
        acc_rows.append(se_accuracy(lv, 40, 800))
    acc = {"lever": levers, "rows": acc_rows}

    # Wild bootstrap at n=30, leverage=4
    wild = {"size": size_experiment(4.0, 30, 400, wild=True, seed=SEED + 5)}

    # n scan at fixed leverage 3
    ns = [30, 50, 100, 200, 400]
    nscan = {"n": ns}
    tmp = {k: [] for k in ["ols", "hc0", "hc3"]}
    for n in ns:
        r = size_experiment(3.0, n, 800, seed=SEED + 6)
        for k in tmp:
            tmp[k].append(r[k])
    nscan.update(tmp)

    results = {
        "setup": {"seed": SEED, "reps": REPS, "B_wild": B_WILD, "z95": float(Z95)},
        "size_leverage": size_res,
        "se_accuracy": {"lever": levers,
                        "rows": [{"lever": lv,
                                  "true_sd": acc_rows[i]["true_sd"],
                                  "mean_se": acc_rows[i]["mean_se"]}
                                 for i, lv in enumerate(levers)]},
        "wild_size": wild["size"],
        "n_scan": nscan,
    }
    results["figs"] = [plot1_rows(), plot2_size(size_res), plot3_se(acc),
                       plot4_wild(wild), plot5_nscan(nscan)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

