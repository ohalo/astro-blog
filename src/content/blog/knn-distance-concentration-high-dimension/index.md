---
title: "kNN 找相似日在高维里为什么会失灵：最近邻距离差从 0.49 掉到 0.012，白化只救回两成"
description: "真值已知的 1100 点实验拆开高维距离集中。每维标准正态下，最近与最远距离的相对差从 p=2 的 3.512 收到 p=1000 的 0.174；最近邻标签相关系数从 0.740 掉到 0.04 至 0.12 区间，1-NN 测试 MSE 冲到 2.11，比只预测均值的 1.215 还差。十维以上堆技术指标换来的不是信息而是噪声：10-NN 全特征 MSE 从 0.494 升到 1.528，只用三个真信号维度是 0.440。固定总噪声能量可把 p=1000 的 MSE 从 1.293 拉回 0.220；协方差白化只从 1.241 改到 0.990，仍远落后三维 kNN 的 0.168。附完整代码与五张真实计算图。"
publishDate: '2026-11-26T01:20:00Z'
tags:
  - 量化
  - 高维统计
  - 机器学习
  - 距离度量
  - Python
language: zh
difficulty: advanced
image: "/images/knn-distance-concentration-high-dimension/fig01_concentration.png"
---

## 一、结论先行：维度越多，邻居越不像邻居

用 K 近邻找历史相似日是直观做法：今天的技术指标摊成一个向量，去历史库里找最近的几天，用它们的平均收益做预测。问题在于，指标从 3 个加到 100 个时，欧氏距离会越来越分不出谁近谁远——所有点对的距离挤到同一个数附近，「最近邻」这个名字开始失去意义。

1100 个标准正态点的实验给出五组数字：

- 每维距离的相对离散度（最大减最小再除以均值）从 $p=2$ 的 **3.512** 收到 $p=1000$ 的 **0.174**。理论近似 $\sqrt{2\log n/p}$ 给出 0.118，同量级但偏低，说明这只是一个量级参照不是精确公式；到 1000 维也没跌破 0.1。
- 最近邻与第二近邻的距离差（除以第二近距离）从 **0.491** 压到 **0.012**，邻居之间几乎无法区分。
- 最近邻标签相关系数从 0.740 掉到 0.038 至 0.118 的噪声区间，1-NN 的测试 MSE 从 0.624 冲到 **2.112**，而只预测训练均值的基线是 **1.215**——高维 1-NN 不只是退化成均值，而是比均值更差。
- 10-NN 用全部特征的 MSE 从 $p=2$ 的 0.494 爬到 $p=1000$ 的 1.528；只用三个真信号维度是 0.440，且完全不随无关维度增加。这里还有个反转：三个信号维度上做 OLS 只有 0.786，**输给**三维 kNN，因为真函数含平方项而 OLS 是线性设定。别把「线性模型更稳」当成普适结论。
- 救法有效但有边界。把每个噪声维度的方差按 $c/p$ 缩小、保持总噪声能量固定后，$p=1000$ 的 5-NN MSE 是 0.220（$c=1$）和 0.300（$c=16$），而未缩放的每维方差 1 是 1.293。协方差白化把带相关噪声的 63 维原始 kNN 从 1.241 改到 0.990，改善 20.2%，仍远落后三维 kNN 的 0.168。

![距离集中](/images/knn-distance-concentration-high-dimension/fig01_concentration.png)

## 二、距离集中把「近」这个概念抹平

设 $X$ 是 $p$ 维独立标准正态向量，随机点到原点的距离平方是 $p$ 个独立卡方量之和，均值为 $p$、标准差为 $\sqrt{2p}$。于是相对波动是 $\sqrt{2/p}$，随 $p^{-1/2}$ 收缩。把样本数 $n$ 的极值考虑进来，最大与最小距离的相对差大约是 $\sqrt{2\log n/p}$。

图上实测曲线与理论形态一致但整体偏高：1100 个点、$p=1000$ 时实测 0.174 对理论 0.118。这不奇怪——理论式来自正态尾部的渐近近似，有限样本与极值波动都会把它顶高。真正要记住的是量级：$p$ 从 2 走到 1000，相对差从 3.5 掉到 0.17，排序意义上的「近」还剩一点，但绝对意义上的距离几乎人人相等。

更要命的是，$p$ 增大时每个新特征都带来一份独立的噪声能量。如果信号只藏在头三个维度，后面 997 维的距离贡献会像雪崩一样把前三位的差异淹没。这和[随机矩阵协方差去噪](/blog/random-matrix-covariance-denoising/)是同一个病的两副面孔：样本协方差在高维下偏离真值，成对距离在高维下失去分辨力。

## 三、最近邻还认得标签吗

第二个实验把标签设成 $y=X_1+\varepsilon$（$\varepsilon$ 标准差 0.6），只有第一维带信息，其余全是噪声维。训练 700 点、测试 300 点，九档 $p$ 共用同一批随机数。

![最近邻信号](/images/knn-distance-concentration-high-dimension/fig02_nn_signal.png)

$p=2$ 时最近邻预测与真标签相关 0.740、MSE 0.624；$p=500$ 时相关系数只剩 0.038，MSE 2.326；$p=1000$ 时相关 0.118，MSE 2.112。曲线不是严格单调——0.038 到 0.118 的回升是抽样噪声，不该解读成「维度再高一点反而变好」。距离差曲线则干净地从 0.491 单调降到 0.012，说明邻居确实变得无法区分。

这张图最值得盯的是那条虚线：预测训练均值的 MSE 是 1.215。$p$ 超过 50 之后，1-NN 已经稳定地比这个懒惰基线更差。如果你在回测里看到高维 KNN 策略表现不错，先问三件事：是不是用了样本内邻居、是不是特征里混了未来信息、是不是只报告了选出来那一版。这里的测试集是干净的样本外。

## 四、堆特征到底换来什么

第三个实验把真函数设为 $y=0.8X_1+0.5X_2^2-0.3X_3+\varepsilon$，噪声标准差 0.5，比较三种 kNN 与一个 OLS 基线。

![kNN 预测误差](/images/knn-distance-concentration-high-dimension/fig03_knn_mse.png)

1-NN 全特征 MSE 从 $p=2$ 的 0.667 涨到 $p=1000$ 的 2.897；把邻居数提到 10 只能压到 1.528，仍是三维 kNN 0.440 的 3.5 倍。三维 kNN 那条曲线是水平的，因为无关维度根本没进距离计算——这不是什么技巧，就是特征选择。

OLS 那条水平线在 0.786，比三维 kNN 差 78.6%。原因是真函数里 $X_2^2$ 这一项在线性设定下完全没被捕捉，而局部平均天然能吃到这种非线性。所以「降维之后用简单模型」不是默认最优；如果关系非线性，低维非参数方法仍然可能赢。真正要避免的是把 100 个指标一股脑丢进距离函数，而不是避免所有复杂度。

## 五、救法一：让每个噪声维度小到可以忽略

既然高维灾难来自无关维度贡献的总噪声能量，一个直接想法是让噪声方差随维度缩小：$\sigma_p^2=c/p$，于是总能量固定在 $c$。第四个实验用三个真信号维度加噪声维，5-NN 的 MSE 曲线如下。

![噪声尺度缩放](/images/knn-distance-concentration-high-dimension/fig04_noise_scaling.png)

每维方差固定为 1 时，MSE 从 $p=3$ 的 0.219 单调爬到 $p=1000$ 的 1.293；总能量固定为 1 时，1000 维仍是 0.220，几乎回到只有信号时的水平；总能量放宽到 16，也只到 0.300。换句话说，高维本身不是诅咒，**每个维度的噪声能量不变**才是。

这对特征工程有直接含义：与其往距离里塞 100 个等权重的标准化指标，不如先估计每个指标的有效信息量，再按信噪比给它配权。但要知道 $c/p$ 这个缩放是理想化的——它假设你知道哪些维度是信号、知道噪声方差、且噪声维彼此独立。现实里这三样都要估，估错的代价不比高维本身小。

## 六、救法二：白化，以及白化救不回来的部分

第五个实验把 60 个噪声维造成五因子加个体噪声的相关结构，最大特征值占总方差 30.3%。原始 63 维欧氏距离会被这几个高方差因子方向主导。用训练集协方差做特征分解、按特征值平方根缩放后重算 kNN：

![白化对照](/images/knn-distance-concentration-high-dimension/fig05_whitening.png)

原始 kNN 的 MSE 是 1.241，白化后 0.990，改善 20.2%。这是真实收益，但不是奇迹：只用三个信号维度的 kNN 是 0.168，OLS 是 0.120，白化后的 63 维仍然差 5 到 8 倍。原因很直白——白化只能让各方向方差可比，不能告诉你哪个方向带标签信息；它把噪声拉平，却没有把噪声删掉。

还有一个反直觉风险写在图上：如果真信号恰好落在白化后方差被放大的低方差方向，白化会让情况更糟。这里信号维是独立标准正态、方差相同，白化才没有帮倒忙。真实截面因子里，低波动、低相关的指标往往正是研究者想用的信号，用样本协方差盲白化前最好先看看[主成分选因子](/blog/pca-statistical-arbitrage/)那类做法怎么处理方向解释。

## 七、完整代码

以下脚本自包含生成全部数据、实验、五张图与 `results.json`，固定随机种子 20261126，可直接执行。

```python
# -*- coding: utf-8 -*-
"""Distance concentration and kNN failure in high dimensions, with 5 real figures.
Self-contained: all data, experiments, figures and results.json from one script.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261126)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "knn-distance-concentration-high-dimension")
os.makedirs(OUT, exist_ok=True)

PMAX = 1024
PS = np.array([2, 5, 10, 20, 50, 100, 200, 500, 1000])


def sqdist(A, B):
    return (np.sum(A * A, axis=1)[:, None] + np.sum(B * B, axis=1)[None, :]
            - 2.0 * A @ B.T)


def knn_pred(Xtr, ytr, Xte, k):
    d = sqdist(Xte, Xtr)
    idx = np.argpartition(d, k - 1, axis=1)[:, :k]
    return ytr[idx].mean(axis=1)


def mse(pred, y):
    return float(np.mean((pred - y) ** 2))


def main():
    res = {}
    rng = RNG
    n = 1100
    Z = rng.normal(0, 1, size=(n, PMAX))

    # exp1: concentration ratio of per-coordinate RMS distance from origin
    ratios, theories = [], []
    for p in PS:
        d = np.sqrt(np.mean(Z[:, :p] ** 2, axis=1))
        ratios.append(float((d.max() - d.min()) / d.mean()))
        theories.append(float(np.sqrt(2.0 * np.log(n) / p)))
    res["exp1_ps"] = [int(p) for p in PS]
    res["exp1_ratio"] = ratios
    res["exp1_theory"] = theories
    res["exp1_ratio_p2"] = ratios[0]
    res["exp1_ratio_p1000"] = ratios[-1]
    res["exp1_theory_p1000"] = theories[-1]
    res["exp1_first_below_010"] = int(PS[int(np.argmax(np.array(ratios) < 0.1))]) if any(np.array(ratios) < 0.1) else -1

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(PS, ratios, "o-", color="#c0392b", label="measured (max-min)/mean")
    a1.plot(PS, theories, "s--", color="#2c7fb8", label="sqrt(2 log n / p)")
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("relative spread")
    a1.set_title("Distance spread collapses as p grows")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig01_concentration.png"), dpi=130); plt.close()

    # exp2: is the nearest neighbour actually informative about y?
    ntr, nte = 700, 300
    y_all = Z[:, 0] + rng.normal(0, 0.6, n)
    Xtr, Xte = Z[:ntr], Z[ntr:ntr + nte]
    ytr, yte = y_all[:ntr], y_all[ntr:ntr + nte]
    cors, nn_mse, gaps = [], [], []
    for p in PS:
        d = sqdist(Xte[:, :p], Xtr[:, :p])
        nn = np.argmin(d, axis=1)
        part = np.partition(d, 1, axis=1)[:, :2]
        gaps.append(float(np.mean((part[:, 1] - part[:, 0]) / np.maximum(part[:, 1], 1e-12))))
        pred = ytr[nn]
        cors.append(float(np.corrcoef(pred, yte)[0, 1]))
        nn_mse.append(mse(pred, yte))
    baseline = mse(np.full_like(yte, ytr.mean()), yte)
    res["exp2_ps"] = [int(p) for p in PS]
    res["exp2_corr"] = cors
    res["exp2_nn_mse"] = nn_mse
    res["exp2_dn_gap"] = gaps
    res["exp2_baseline_mse"] = baseline
    res["exp2_corr_p2"] = cors[0]
    res["exp2_corr_p1000"] = cors[-1]
    res["exp2_gap_p2"] = gaps[0]
    res["exp2_gap_p1000"] = gaps[-1]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(PS, cors, "o-", color="#c0392b")
    a1.axhline(0, color="#555", lw=0.8)
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("corr(y_test, y_NN)")
    a1.set_title("Nearest-neighbour label correlation dies")
    a2.plot(PS, nn_mse, "o-", color="#c0392b", label="1-NN")
    a2.axhline(baseline, color="#2c7fb8", ls="--", label="predict training mean")
    a2.set_xscale("log"); a2.set_xlabel("dimension p"); a2.set_ylabel("test MSE")
    a2.set_title("1-NN degenerates to the mean predictor")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_nn_signal.png"), dpi=130); plt.close()

    # exp3: kNN regression error vs p, compared to known-signal methods
    y2 = 0.8 * Z[:, 0] + 0.5 * Z[:, 1] ** 2 - 0.3 * Z[:, 2] + rng.normal(0, 0.5, n)
    a, b = y2[:ntr], y2[ntr:ntr + nte]
    knn1, knn10, knn_oracle = [], [], []
    for p in PS:
        knn1.append(mse(knn_pred(Xtr[:, :p], a, Xte[:, :p], 1), b))
        knn10.append(mse(knn_pred(Xtr[:, :p], a, Xte[:, :p], 10), b))
        knn_oracle.append(mse(knn_pred(Xtr[:, :3], a, Xte[:, :3], 10), b))
    D = np.column_stack([np.ones(ntr), Xtr[:, :3]])
    coef = np.linalg.lstsq(D, a, rcond=None)[0]
    ols = mse(np.column_stack([np.ones(nte), Xte[:, :3]]) @ coef, b)
    res["exp3_ps"] = [int(p) for p in PS]
    res["exp3_knn1"] = knn1
    res["exp3_knn10"] = knn10
    res["exp3_knn3d"] = knn_oracle
    res["exp3_ols3d"] = ols
    res["exp3_knn1_p2"] = knn1[0]
    res["exp3_knn10_p1000"] = knn10[-1]
    res["exp3_knn3d_mse"] = float(knn_oracle[0])

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(PS, knn1, "o-", color="#c0392b", label="1-NN, all p features")
    a1.plot(PS, knn10, "s-", color="#e67e22", label="10-NN, all p features")
    a1.plot(PS, knn_oracle, "^--", color="#2c7fb8", label="10-NN, 3 signal features")
    a1.axhline(ols, color="#27ae60", ls=":", label="OLS on 3 signal features")
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("test MSE")
    a1.set_title("Raw-space kNN loses to feature selection")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig03_knn_mse.png"), dpi=130); plt.close()

    # exp4: shrinking noise variance per dimension keeps total noise energy fixed
    ps4 = np.array([3, 10, 30, 100, 300, 1000])
    curves = {}
    noise = rng.normal(0, 1, size=(n, int(ps4.max())))
    signal3 = Z[:, :3]
    y4 = 0.8 * signal3[:, 0] + 0.5 * signal3[:, 1] ** 2 - 0.3 * signal3[:, 2]
    y4 += rng.normal(0, 0.3, n)
    a4, b4 = y4[:ntr], y4[ntr:ntr + nte]
    for tag, c in [("fixed_sigma", 1.0), ("energy_1", 1.0), ("energy_4", 4.0), ("energy_16", 16.0)]:
        vals = []
        for p in ps4:
            scale = c / p if tag != "fixed_sigma" else 1.0
            full = np.column_stack([signal3, noise[:, :max(0, int(p) - 3)] * np.sqrt(scale)])[:, :int(p)]
            vals.append(mse(knn_pred(full[:ntr], a4, full[ntr:ntr + nte], 5), b4))
        curves[tag] = vals
    res["exp4_ps"] = [int(p) for p in ps4]
    res["exp4_curves"] = {k: [float(v) for v in arr] for k, arr in curves.items()}
    res["exp4_fixed_p1000"] = curves["fixed_sigma"][-1]
    res["exp4_energy1_p1000"] = curves["energy_1"][-1]
    res["exp4_energy16_p1000"] = curves["energy_16"][-1]

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    styles = [("fixed_sigma", "#c0392b", "noise var 1 per dim"),
              ("energy_1", "#2c7fb8", "total noise energy 1"),
              ("energy_4", "#e67e22", "total noise energy 4"),
              ("energy_16", "#8e44ad", "total noise energy 16")]
    for k, col, lab in styles:
        a1.plot(ps4, curves[k], "o-", color=col, label=lab)
    a1.set_xscale("log"); a1.set_xlabel("dimension p"); a1.set_ylabel("5-NN test MSE")
    a1.set_title("Fixing total noise energy rescues kNN")
    a1.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig04_noise_scaling.png"), dpi=130); plt.close()

    # exp5: correlated noise directions, raw vs covariance-whitened kNN
    kfac = 5
    F = rng.normal(0, 1, size=(n, kfac))
    L = rng.normal(0, 1, size=(60, kfac))
    corr_noise = F @ L.T + rng.normal(0, 0.35, size=(n, 60))
    sig = Z[:, :3]
    X5raw = np.column_stack([sig, corr_noise])
    y5 = 0.9 * sig[:, 0] + 0.6 * sig[:, 1] - 0.4 * sig[:, 2] + rng.normal(0, 0.4, n)
    a5, b5 = y5[:ntr], y5[ntr:ntr + nte]
    mu = X5raw[:ntr].mean(axis=0)
    C = np.cov((X5raw[:ntr] - mu).T)
    evals, evecs = np.linalg.eigh(C)
    evals = np.maximum(evals, 1e-6)
    W = (evecs / np.sqrt(evals)) @ evecs.T
    X5w = (X5raw - mu) @ W.T
    raw_mse = mse(knn_pred(X5raw[:ntr], a5, X5raw[ntr:ntr + nte], 10), b5)
    wht_mse = mse(knn_pred(X5w[:ntr], a5, X5w[ntr:ntr + nte], 10), b5)
    ora_mse = mse(knn_pred(sig[:ntr], a5, sig[ntr:ntr + nte], 10), b5)
    D5 = np.column_stack([np.ones(ntr), sig[:ntr]])
    c5 = np.linalg.lstsq(D5, a5, rcond=None)[0]
    ols5 = mse(np.column_stack([np.ones(nte), sig[ntr:ntr + nte]]) @ c5, b5)
    res["exp5_raw_mse"] = raw_mse
    res["exp5_whitened_mse"] = wht_mse
    res["exp5_oracle3d_mse"] = ora_mse
    res["exp5_ols3d_mse"] = ols5
    res["exp5_top_eig_share"] = float(evals.max() / evals.sum())

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    labels = ["raw 63d", "whitened", "signal 3d", "OLS 3d"]
    vals = [raw_mse, wht_mse, ora_mse, ols5]
    bars = a1.bar(labels, vals, color=["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"])
    for bar, v in zip(bars, vals):
        a1.text(bar.get_x() + bar.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    a1.set_ylabel("test MSE"); a1.set_title("Whitening vs raw Euclidean kNN")
    a2.plot(np.arange(1, len(evals) + 1), evals / evals.sum(), color="#8e44ad", lw=1.8)
    a2.set_xlabel("eigenvalue rank"); a2.set_ylabel("share of total variance")
    a2.set_title(f"top eigenvalue carries {res['exp5_top_eig_share']:.1%}")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_whitening.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 八、什么时候可以放心用 kNN

三条经验：第一，先画最近邻距离的分布，如果第 $k$ 近与第 $k+1$ 近几乎相等，说明名次本身不可信，再谈加权毫无意义；第二，把「只用信号维」当成一个必须跑的对照，它给你的不是最优结果，而是判断特征工程有没有价值的标尺；第三，任何基于全样本协方差的预处理都必须只在训练集上拟合，测试集只用变换——本文的白化矩阵严格来自前 700 点。

最后说清楚适用范围：这里的标签、噪声结构、维度都是已知的合成设定，目的是把距离集中的机制量化和可复现，不是声称某个真实市场策略的收益。1100 个样本、九档维度的结论是定性的；换成真实日频数据，序列相关、非平稳和交易成本会叠加进来，那是另一篇文章要处理的事。

## 九、参考阅读

同源问题在站内还有另外两处：协方差估计的[随机矩阵去噪](/blog/random-matrix-covariance-denoising/)处理「矩阵里的高维噪声」，因子个数的[信息准则选择](/blog/bai-ng-factor-number-selection/)处理「该留几个方向」。三篇合起来刚好是从距离、协方差到因子数的完整链条。
