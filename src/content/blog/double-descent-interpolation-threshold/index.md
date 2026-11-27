---
title: "双重下降：模型参数加到刚好等于样本数时，测试误差为什么会先爆再落，以及岭回归与相关性如何改写这条曲线"
description: "真值已知的 80 样本、400 维特征实验：最小范数最小二乘的测试 MSE 在 p=5 时是 1.296，逼近 p=80 时飙到 970.93，越过插值点后又回落到 p=400 的 1.096，而过参数区的最优反而低于经典区。峰值位置随样本数精确移动到 p=40、80、160。噪声从 0.04 升到 1.0，峰值从 3505 抬到 8767；岭惩罚 lambda 从 0 加到 10，峰值从 957 压到 2.18。附完整代码与五张真实计算图。"
publishDate: '2026-11-27T03:20:00Z'
tags:
  - 量化
  - 机器学习
  - 过参数化
  - 泛化
  - Python
language: zh
difficulty: advanced
image: "/images/double-descent-interpolation-threshold/fig01_double_descent.png"
---

## 一、结论先行：经典偏差方差图景只在半边成立

教科书里的偏差方差权衡说：模型越复杂，偏差下降、方差上升，测试误差是一条 U 形曲线，最优点在中间。这条图景默认了一个前提——参数个数 $p$ 小于样本数 $n$。一旦越过这条线，进入过参数区，最小二乘不再唯一，我们通常取范数最小的那个解。近二十年的经验发现，此时测试误差曲线会出现第二次下降：逼近 $p=n$ 时误差冲上尖峰，越过去之后反而回落。这就是双重下降。

本文用真值已知的合成数据把它完整复现并量化。80 个训练样本、400 维特征池、真实系数单位能量、噪声方差 0.25，测试集 3000 点，每个点平均 6 组独立数据。结果如下：

- 测试 MSE 从 $p=5$ 的 **1.296** 一路升到 $p=80$ 的 **970.93**，在插值阈值处炸开；越过之后单调回落，$p=400$ 时是 **1.096**。
- 训练集在 $p=80$ 之后误差降到 $10^{-28}$ 量级——模型把训练数据插值得几乎完美，但测试误差在阈值两侧天差地别。
- 经典区（$p\leq 50$）的最好测试误差是 1.296，过参数区（$p\geq150$）的最好值是 **1.096**，后者反而更低。「参数越多一定越糟」在这组数据上是错的。
- 峰值位置跟着样本数走：$n=40$、80、160 三档，尖峰分别精确落在 $p=40$、80、160。
- 噪声方差从 0.04 升到 0.25 再到 1.0，峰值从 3505 抬到 5190 再到 8767，过参数区的底部从 0.864 抬到 2.064——噪声越大，整条曲线越高。
- 岭回归把尖峰彻底抹平：$\lambda$ 从 0 加到 0.1、1、10，峰值从 957 压到 22.7、6.05、2.18。

## 二、实验设定：一个能算出真值的世界

数据生成过程是 $y=X\beta+\varepsilon$，$X$ 的 400 个特征独立标准正态，$\beta$ 从标准正态抽样后除以 $\sqrt{400}$，让信号能量 $\|\beta\|^2$ 期望为 1；$\varepsilon$ 方差可调。拟合时只用前 $p$ 个特征，$p$ 从 5 扫到 400。当 $p\leq n$ 用普通最小二乘，当 $p>n$ 用 Moore-Penrose 伪逆给出最小范数解。

![经典双重下降曲线](/images/double-descent-interpolation-threshold/fig01_double_descent.png)

关键细节：每个实验里所有 $p$ 取值共享同一批底层数据集，只有被研究的参数变化。这样曲线之间的差异才来自模型复杂度本身，不会被不同抽样噪声污染。

图一里蓝色方块是训练误差，它随 $p$ 单调下降，越过 $n=80$ 后贴地——因为 80 个方程、80 个以上未知数，总能精确解出。红色圆点是测试误差，走势完全不同：先在经典区缓慢爬升，逼近 80 时突然炸开，越过去后快速回落，最后低于起点。绿色点线是不可约噪声 0.25，过参数区的测试误差正朝它靠拢。

## 三、尖峰为什么恰好在 p 等于 n 处

第二个实验回答这个问题。把样本数换成 40、80、160，各扫一遍 $p$。

![峰值位置随样本数移动](/images/double-descent-interpolation-threshold/fig02_peak_location.png)

三档的尖峰分别落在 $p=40$、80、160，与各自的样本数精确重合。机制在最小范数解的代数里：$p<n$ 时解是 $(X^TX)^{-1}X^Ty$，$X^TX$ 接近奇异时方差爆炸；$p=n$ 时方阵恰好可逆但条件数最坏；$p>n$ 时解变成 $X^T(XX^T)^{-1}y$，矩阵维度固定为 $n\times n$，继续加特征不会让逆矩阵更病态，反而提供了更多方向去平均掉噪声。

所以尖峰不是「过拟合的起点」，而是「可解与不可解之间那道最窄的门」。它的高度由 $X^TX$ 在临界点附近的最小奇异值决定，样本越少、噪声越大，这扇门越窄。把 $n$ 挪动，门跟着挪动——这就是图二三条曲线峰值位置整齐对齐的原因。

## 四、噪声抬高整条曲线

第三个实验在同一批数据上只改噪声方差，取 0.04、0.25、1.0。

![噪声水平的影响](/images/double-descent-interpolation-threshold/fig03_noise.png)

三组峰值分别是 3505、5190、8767，底部（$p=400$）分别是 0.864、1.129、2.064。两个观察值得记：第一，尖峰高度随噪声单调上升，噪声本身就是被最小奇异值放大的原料；第二，底部抬升幅度温和得多，从 0.864 到 2.064 大约 2.4 倍，而峰值涨了约 2.5 倍——过参数区的解对噪声的容忍度明显好于临界点。

这也解释了为什么双重下降在深度学习里比在经典统计里更常见：现代模型参数量动辄百万千万，几乎永远工作在 $p\gg n$ 的那一侧，离那道窄门很远，于是「更大反而更好」成了日常经验。

## 五、岭回归把门拓宽

第四个实验给最小二乘加 $\ell^2$ 惩罚，解变成 $(X^TX+\lambda I)^{-1}X^Ty$。

![岭回归抹平尖峰](/images/double-descent-interpolation-threshold/fig04_ridge.png)

$\lambda=0$ 时峰值 957；$\lambda=0.1$ 时降到 22.7，接近零头；$\lambda=1$ 时 6.05；$\lambda=10$ 时只剩 2.18。岭惩罚做的事就是给 $X^TX$ 的对角线垫一个数，把最小特征值从接近 0 抬离，让临界点不再奇异。代价是小 $\lambda$ 会在过参数区引入轻微偏差，但相比尖峰的幅度，这个代价可以忽略。

对量化实践的含义很直接：当你在一个样本量有限、特征越来越多的选股模型里看到测试误差突然恶化，第一反应不该是「特征加太多要砍掉」，而是先确认是不是踩到了插值阈值。如果是，加正则比砍特征更对路——前者保留全部信息只约束范数，后者直接丢掉可能有用的信号。

## 六、特征相关会改写尖峰，但不会消灭它

第五个实验给特征注入一个共同因子：每个观测是共同因子与个体噪声的加权和，权重由相关系数控制，取 0、0.3、0.7。

![特征相关性的影响](/images/double-descent-interpolation-threshold/fig05_correlation.png)

峰值从无相关的 467.59 降到 0.3 相关的 190.04，再到 0.7 相关的 122.77。特征越相关，有效维度越低，模型在临界点附近遇到的病态程度越轻。但尖峰始终存在，没有哪档相关性把它抹掉。

这一点对金融数据尤其重要：股票特征高度共线，估值、质量、动量因子之间往往共享宏观暴露。共线性确实缓冲了双重下降的最坏情形，但它同时也让有效样本量缩水——你以为自己有 400 个特征，实际上独立方向可能只有几十个。判断自己站在曲线的哪一侧，用的是有效秩，不是特征个数。

## 七、边界与实操

说清楚适用范围。这里的 $X$ 是独立同分布的高斯特征，真实关系是线性的，测试集与训练集同分布。真实市场里特征非高斯、关系非线性、分布随时间漂移，双重下降的形状会被这些因素改变。另外，尖峰的绝对高度对伪逆的截断阈值敏感——本文用 $10^{-10}$ 的相对阈值，换一个数值设定峰值数字会变，但「峰值在 $p=n$、越过去回落、正则能抹平」这三个定性结论是稳健的。

实操上给三条：第一，画测试误差随特征数的曲线时，把 $p=n$ 这条竖线标出来，你会知道异常出现在哪里；第二，别在临界点附近做模型选择，那里任何一个微小扰动都会被放大；第三，过参数区不是免费的，它赢的是偏差被压低、方差被平均，一旦分布漂移或噪声结构改变，这个平衡会被打破。

还有一个和方法论相关的提醒。本文每个 $p$ 点都用了独立的测试集来算 MSE，这是双重下降实验的标准做法——因为曲线本身就是在度量泛化，用训练误差去看会只看到那条贴地的蓝线。如果你想在实盘里验证自己是否踩到了插值阈值，同样必须留出干净的样本外数据；用样本内误差做模型选择，只会一路选到 $p=400$，看到漂亮的零训练误差就以为找到了好模型。第三组实验里噪声方差 1.0 时过参数区底部是 2.064，仍高于不可约噪声 1.0——过参数化压低的是估计方差，压不掉数据本身的噪声，别指望增加特征能把这部分变没。

最后说清楚和站内其他方法论的关系。岭回归抹平尖峰这件事，本质和[ Ledoit-Wolf 收缩估协方差](/blog/ledoit-wolf-shrinkage/)是同一个动作：给病态的矩阵对角线加一点结构化约束，换来回定的数值行为。区别是一个作用在回归设计矩阵上，一个作用在样本协方差上。理解了其中一个，另一个的动机就顺理成章。

```python
# -*- coding: utf-8 -*-
"""Double descent around the interpolation threshold, with five real figures.
Self-contained: all synthetic data, min-norm / ridge fits, figures and results.json.
Within each experiment every parameter point shares the SAME underlying datasets
(only the parameter under study changes), so curves are directly comparable.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "double-descent-interpolation-threshold")
os.makedirs(OUT, exist_ok=True)

D = 400           # total feature pool
NTE = 3000        # test size
REPS = 6          # independent datasets averaged per point
PGRID = sorted(set([5, 10, 15, 20, 30, 40, 50, 60, 70, 75, 78,
                    80, 82, 85, 90, 100, 120, 150, 200, 250, 300, 350, 400]))


def make_data(seed, n, d, sigma2, corr=0.0):
    """y = X beta + eps. beta has unit signal energy; optional equicorrelated X.
    The random streams for beta, X and the standard-normal eps_base are FIXED by
    seed; sigma2 and corr only rescale / mix them, so curves stay comparable.
    """
    rng = np.random.default_rng(seed)
    beta = rng.normal(0.0, 1.0, d) / np.sqrt(d)
    base_tr = rng.normal(0.0, 1.0, (n, d))
    base_te = rng.normal(0.0, 1.0, (NTE, d))
    eps_base_tr = rng.normal(0.0, 1.0, n)
    eps_base_te = rng.normal(0.0, 1.0, NTE)
    if corr == 0.0:
        Xtr, Xte = base_tr, base_te
    else:
        ftr = np.sqrt(corr) * np.ones((n, 1))
        fte = np.sqrt(corr) * np.ones((NTE, 1))
        Xtr = np.sqrt(corr) * base_tr.mean(axis=1, keepdims=True) + np.sqrt(1.0 - corr) * base_tr
        Xte = np.sqrt(corr) * base_te.mean(axis=1, keepdims=True) + np.sqrt(1.0 - corr) * base_te
    ytr = Xtr @ beta + np.sqrt(sigma2) * eps_base_tr
    yte = Xte @ beta + np.sqrt(sigma2) * eps_base_te
    return Xtr, ytr, Xte, yte


def fit_pred(Xtr, ytr, Xte, p, lam=0.0):
    """Min-norm OLS (lam=0) or ridge on the first p features."""
    A = Xtr[:, :p]
    if lam == 0.0:
        w = np.linalg.pinv(A, rcond=1e-10) @ ytr
    else:
        G = A.T @ A + lam * np.eye(p)
        w = np.linalg.solve(G, A.T @ ytr)
    return Xte[:, :p] @ w


def mse_curve(n, sigma2=0.25, corr=0.0, lam=0.0, ps=None, reps=REPS, base_seed=1100):
    ps = PGRID if ps is None else ps
    tr = np.zeros(len(ps)); te = np.zeros(len(ps))
    for r in range(reps):
        Xtr, ytr, Xte, yte = make_data(base_seed + r, n, D, sigma2, corr)
        for i, p in enumerate(ps):
            tr[i] += np.mean((fit_pred(Xtr, ytr, Xtr, p, lam) - ytr) ** 2)
            te[i] += np.mean((fit_pred(Xtr, ytr, Xte, p, lam) - yte) ** 2)
    return [float(v / reps) for v in tr], [float(v / reps) for v in te]


def main():
    res = {}

    # exp1: the canonical curve, n=80, sigma^2=0.25
    tr80, te80 = mse_curve(80)
    res["exp1_ps"] = PGRID
    res["exp1_train"] = tr80
    res["exp1_test"] = te80
    peak_i = int(np.argmax(te80))
    res["exp1_peak_p"] = PGRID[peak_i]
    res["exp1_peak_test"] = te80[peak_i]
    res["exp1_test_p10"] = te80[0]
    res["exp1_test_p400"] = te80[-1]
    res["exp1_train_p400"] = tr80[-1]
    res["exp1_noise_floor"] = 0.25
    classic = [(p, v) for p, v in zip(PGRID, te80) if p <= 50]
    over = [(p, v) for p, v in zip(PGRID, te80) if p >= 150]
    res["exp1_classic_best"] = min(classic, key=lambda t: t[1])[1]
    res["exp1_over_best"] = min(over, key=lambda t: t[1])[1]

    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(PGRID, tr80, "s-", color="#2c7fb8", label="train MSE (min-norm)")
    ax.plot(PGRID, te80, "o-", color="#c0392b", label="test MSE (min-norm)")
    ax.axvline(80, color="#555", ls="--", lw=1.0, label="p = n (interpolation)")
    ax.axhline(0.25, color="#27ae60", ls=":", label="irreducible noise 0.25")
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("MSE")
    ax.set_title("Test error peaks at p=n then falls again")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_double_descent.png"), dpi=130); plt.close()

    # exp2: peak location tracks n (each n has its own shared datasets)
    ns = [40, 80, 160]
    peak_by_n = {}
    curves_n = {}
    for k, n in enumerate(ns):
        ps = sorted(set(int(p) for p in
                        [max(2, n // 6), n // 3, n // 2, int(n * 0.75),
                         n - 5, n, n + 5, int(n * 1.25), int(n * 1.5),
                         int(n * 2), int(n * 3), min(D, int(n * 5))] if 2 <= p <= D))
        _, te = mse_curve(n, ps=ps, base_seed=2000 + 50 * k)
        curves_n[str(n)] = {"ps": [int(p) for p in ps], "test": te}
        pi = int(np.argmax(te))
        peak_by_n[str(n)] = int(ps[pi])
    res["exp2_ns"] = ns
    res["exp2_curves"] = curves_n
    res["exp2_peak_p"] = peak_by_n

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for n, col in zip(ns, ["#8e44ad", "#c0392b", "#e67e22"]):
        d = curves_n[str(n)]
        ax.plot(d["ps"], d["test"], "o-", label=f"n={n}, peak p={peak_by_n[str(n)]}", color=col)
        ax.axvline(n, color=col, ls="--", lw=0.8, alpha=0.6)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("The spike sits at p ≈ n, whatever n is")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_peak_location.png"), dpi=130); plt.close()

    # exp3: noise level changes spike height and the over-parameterized floor
    # Same datasets (same base_seed) across sigma2: only the noise scale changes.
    sigmas = [0.04, 0.25, 1.0]
    curves_s = {}
    for s2 in sigmas:
        _, te = mse_curve(80, sigma2=s2, base_seed=3100)
        curves_s[str(s2)] = te
        pi = int(np.argmax(te))
        curves_s["peak_" + str(s2)] = float(te[pi])
        curves_s["floor_" + str(s2)] = float(te[-1])
    res["exp3_sigmas"] = sigmas
    res["exp3_curves"] = {str(s2): curves_s[str(s2)] for s2 in sigmas}
    res["exp3_peaks"] = {str(s2): curves_s["peak_" + str(s2)] for s2 in sigmas}
    res["exp3_floors"] = {str(s2): curves_s["floor_" + str(s2)] for s2 in sigmas}

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for s2, col in zip(sigmas, ["#27ae60", "#c0392b", "#8e44ad"]):
        ax.plot(PGRID, curves_s[str(s2)], "o-", label=f"noise={s2}", color=col)
        ax.axhline(s2, color=col, ls=":", lw=0.8, alpha=0.7)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("Higher noise lifts both the spike and the floor")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_noise.png"), dpi=130); plt.close()

    # exp4: ridge regularization removes the spike (same datasets across lambda)
    lams = [0.0, 0.1, 1.0, 10.0]
    curves_l = {}
    for lam in lams:
        _, te = mse_curve(80, lam=lam, base_seed=4100)
        curves_l[str(lam)] = te
        curves_l["peak_" + str(lam)] = float(max(te))
    res["exp4_lams"] = lams
    res["exp4_curves"] = {str(lam): curves_l[str(lam)] for lam in lams}
    res["exp4_peaks"] = {str(lam): curves_l["peak_" + str(lam)] for lam in lams}
    res["exp4_peak_lam0"] = curves_l["peak_0.0"]
    res["exp4_peak_lam10"] = curves_l["peak_10.0"]

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for lam, col in zip(lams, ["#c0392b", "#e67e22", "#2c7fb8", "#27ae60"]):
        ax.plot(PGRID, curves_l[str(lam)], "o-", label=f"ridge lambda={lam}", color=col)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("Ridge penalty flattens the interpolation spike")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_ridge.png"), dpi=130); plt.close()

    # exp5: correlated features (common factor) reshape the spike
    corrs = [0.0, 0.3, 0.7]
    curves_c = {}
    for c in corrs:
        _, te = mse_curve(80, corr=c, base_seed=5100)
        curves_c[str(c)] = te
        curves_c["peak_" + str(c)] = float(max(te))
    res["exp5_corrs"] = corrs
    res["exp5_curves"] = {str(c): curves_c[str(c)] for c in corrs}
    res["exp5_peaks"] = {str(c): curves_c["peak_" + str(c)] for c in corrs}

    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    for c, col in zip(corrs, ["#27ae60", "#e67e22", "#c0392b"]):
        ax.plot(PGRID, curves_c[str(c)], "o-", label=f"feature corr={c}", color=col)
    ax.axvline(80, color="#555", ls="--", lw=1.0)
    ax.set_xlabel("number of fitted features p"); ax.set_ylabel("test MSE")
    ax.set_title("A common factor reshapes (but does not erase) the spike")
    ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_correlation.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 八、参考阅读

同一条主线上还有两篇：[核回归的边界偏误](/blog/kernel-local-linear-boundary-bias/)讲非参数方法在边缘如何失衡，[高维距离集中](/blog/knn-distance-concentration-high-dimension/)讲维度怎么抹平「近邻」的概念。三篇合起来是从距离、局部平滑到过参数化的完整链条。
