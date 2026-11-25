---
title: "核回归在样本边界为什么会歪：局部线性把边缘偏误砍掉 8.6 倍，以及它内部的方差一分没少"
description: "用真值已知曲线把 Nadaraya-Watson 与局部线性回归拆开对拍。240 次重复下边界偏差平方 0.023109 对 0.002694，边缘 MSE 0.023556 对 0.003564；内部两种方法的 MSE 只差 0.2%、方差几乎逐位相同。带宽扫描显示 NW 的边缘误差随 h 从 0.02 的 0.00249 单调爬到 0.32 的 0.04783，局部线性在 h=0.04 最优。AR(1) 相关误差把 LOOCV 选带推向 0.02、真实 MSE 从 0.0012 抬到 0.0067。边缘 Alpha 实例中 top 10% 选中的真信号均值 0.002587 对 0.003457。附完整代码与五张真实计算图。"
publishDate: '2026-11-26T01:10:00Z'
tags:
  - 量化
  - 非参数回归
  - 统计估计
  - 交易信号
  - Python
language: zh
difficulty: advanced
image: "/images/kernel-local-linear-boundary-bias/fig01_curves.png"
---

## 一、结论先行：边缘样本最缺，偏偏最容易被平滑器拉偏

给一支股票的动量分数估计期望收益，或者给日内某个时点估一条非线性趋势，常用的核平滑做法是把附近样本加权平均。问题出在样本边界：分数接近 0 或 1 的地方，一侧根本没有数据，权重分布天然不对称，平均出来的曲线会被硬拽向内部。

真值已知的 300 点实验（$m(x)=0.3+1.2x+0.35\sin(4\pi x)$，噪声标准差 0.15）给出四组硬结论：

- 单次样本、带宽 0.08 时，左端点 $x=0$ 的真值是 0.3000，Nadaraya-Watson（下称 NW）报出 **0.5920**（偏 +0.2920），局部线性（下称 LL）报出 **0.4247**（偏 +0.1247）。右端点真值 1.5000，NW 报 1.2452，LL 报 1.4211。
- 240 次重复后，边缘区间（$x\le 0.08$ 或 $x\ge 0.92$）的偏差平方均值 NW 是 **0.023109**、LL 是 **0.002694**，差 **8.6 倍**；边缘 MSE 从 0.023556 降到 0.003564。
- 但 LL 不是免费午餐的直译：内部区间（$0.2\le x\le 0.8$）两者 MSE 只差 **0.2%**（0.008614 对 0.008597），逐点方差甚至是 **0.000252446 对 0.000252494**，几乎逐位相同。均匀设计下 LL 的主场只有边界。
- 带宽扫描出现一个反直觉结果：NW 的边缘误差在 $h=0.02$ 时只有 0.00249，到 $h=0.32$ 涨到 0.04783；LL 的最低点在 $h=0.04$（0.00156）。所谓「平滑一点更稳」在边界上恰好相反。

最后落到一个交易场景：真期望收益只在分数 0.9 附近有尖峰时，NW 把边缘均值低估 0.002654，LL 只低估 0.001457；按各自拟合值选出的 top 10% 组合，真正拿到的 Alpha 均值是 0.002587 对 0.003457，两个名单只有 65.6% 重合。

![边界曲线对比](/images/kernel-local-linear-boundary-bias/fig01_curves.png)

## 二、一阶项能不能被消掉，决定边界命运

NW 的估计量写成

$$
\hat m_{NW}(x)=\frac{\sum_i K_h(x-X_i)Y_i}{\sum_i K_h(x-X_i)},\qquad K_h(u)=h^{-1}K(u/h)
$$

在内部，权重关于 $x$ 近似对称，一阶项 $\sum_i K_h(x-X_i)(X_i-x)$ 接近零，偏差从二阶项开始，量级 $O(h^2)$。到了 $x=0$，一侧权重被截断，一阶矩不再为零，偏差变成 $O(h m'(0))$。直觉上它等于把端点处的值替换成半个窗口内的平均值——函数越陡，错得越离谱。

LL 的解法是在每个查询点做一次局部加权直线拟合，取截距作为函数值：

$$
(\hat\beta_0,\hat\beta_1)=\arg\min_{\beta_0,\beta_1}\sum_i K_h(x-X_i)\bigl(Y_i-\beta_0-\beta_1(X_i-x)\bigr)^2,\qquad \hat m_{LL}(x)=\hat\beta_0
$$

因为回归里显式放了一次项，对称性不再由核函数独自承担，权重自带的那一阶矩被最小二乘正规方程抵消，边界偏差回到 $O(h^2)$。代价是局部要估计两个参数，方差上升；但在均匀设计、内点窗口对称的场合，这个方差涨幅可以小到测不出来，这正是实验里内部方差几乎相等的原因。

需要提醒的是偏差方向不固定。第一组实验函数单调递增，左边界内部值更高，NW 于是**高估**；第五组的边缘 Alpha 尖峰高于内部，NW 变成**低估**。NW 做的是向窗口内均值收缩，方向取决于内部均值相对端点在哪一侧，不能背「边界总是压低」这种口诀。

## 三、把偏差和方差拆开看

单次样本只能看到一个结果，不能证明方法好坏。固定 300 个等距设计点，只重抽噪声，跑 240 次后在每个网格点分别累加估计值与平方，就能把 MSE 拆成偏差平方和方差两条曲线。

![偏差与方差分解](/images/kernel-local-linear-boundary-bias/fig02_bias_variance.png)

边缘偏差平方 0.023109 对 0.002694 已经说明 LL 的优势来自结构而不是运气。但也别过度解读全网格平均：全网格 MSE 是 0.011571 对 0.008164，改善 29.4%，远小于边缘的 6.6 倍——因为内部区域占了大头，而那里两种方法几乎一样。一个只在边缘占优的方法，能带来多大整体收益，完全取决于你的样本分布有多少落在边缘。这也是[滚动窗口偏差-方差漂移](/blog/rolling-window-bias-variance-drift/)那篇强调过的口径问题：先问误差集中在哪儿，再谈方法排名。

另一个容易踩的坑是**设计点随机还是固定**。这里用等距设计把设计密度固定住，只让噪声变化，所以能干净地观察边界权重截断效应。真实截面数据往往在特征两端更稀疏，此时 NW 的问题会比图上更严重：稀疏意味着一个端点附近可能只有两三个样本，权重几乎全压在离得最近的那一个上，方差也会跳起来。

## 四、带宽扫描：NW 想要更窄，LL 有自己的 U 形

八档带宽、每档 70 次重复，把边缘与内部 MSE 分开画：

![带宽扫描](/images/kernel-local-linear-boundary-bias/fig03_bandwidth.png)

NW 的边缘误差从 $h=0.02$ 的 0.00249 单调爬到 $h=0.32$ 的 0.04783（中间 $h=0.24$ 为 0.03435，$h=0.16$ 为 0.03218），没有回头。原因就是带宽越大，端点权重伸进内部的距离越长，一阶项越大。LL 的边缘曲线是先降后升：$h=0.04$ 时最低 0.00156，$h=0.08$ 为 0.00376，$h=0.32$ 又回到 0.01412——大带宽虽然方差小，但二阶偏差和局部线性假设的失效开始占上风。

内部曲线则是两种方法一模一样的单调上升：从 $h=0.02$ 的 0.00116 涨到 $h=0.32$ 的 0.08761（LL 为 0.06220）。所以「边界用窄带、内部用宽带」在数学上说得通，但实操里你通常只有一条曲线一个带宽。折中方案是把特征端点附近单独处理：要么换 LL，要么显式做边界反射/局部外推，要么干脆承认端点不可靠、不要在那儿给交易信号。

## 五、交叉验证在相关误差下会把带宽压得更小

LOOCV 是选带宽的常用手段。把误差改成两种：独立高斯，以及 $\phi=0.75$ 的 AR(1)（同样标准差），每种跑 60 次，看 CV 选出的带宽和真实网格 MSE。

![LOOCV 带宽](/images/kernel-local-linear-boundary-bias/fig04_loocv.png)

独立误差下，LL 的 CV 带宽中位数是 0.03、NW 是 0.02；换成 AR(1) 后两者中位数都压到 0.02。真实 MSE 从 0.001151（LL）涨到 0.006651，NW 从 0.001412 涨到 0.006772，约 5 倍。这跟[滚动窗口](/blog/rolling-window-bias-variance-drift/)里的漂移不是一回事：这里真函数没变，是误差的序列相关让「留一条出来」的高估偏差变了形，CV 误以为更窄的窗口能追上相邻点的相关结构。日内序列、按时间排序的截面都该警惕这一点，检验残差自相关比继续调网格有用。

顺带说明，本实验的 CV 是在同一个设计上做的，误差重抽独立；如果你把自举重抽也用在相关误差上，需要[区块自助](/blog/block-bootstrap-coverage-block-length/)那套做法，否则区间宽度同样会失真。

## 六、边缘 Alpha：名单重合只有三分之二

最后做一个更贴近选股的版本。900 个分数来自 Beta(2,2) 分布（两端天然稀疏），真期望收益为 $0.001(s-0.5)+0.006\exp(-((s-0.9)/0.07)^2)$，噪声标准差 0.012，带宽固定 0.07。

![边缘 Alpha 拟合](/images/kernel-local-linear-boundary-bias/fig05_edge_alpha.png)

在 $s\ge 0.84$ 的边缘区，真均值 0.004379，NW 拟出 0.001726、LL 拟出 0.002923，偏差分别是 −0.002654 与 −0.001457。按拟合值取 top 10%（90 只），NW 名单里股票的真实 Alpha 均值 0.002587，LL 名单 0.003457，高出 33.6%；两个名单交集占 65.6%。也就是说，即便整体曲线看起来差不多，选出来的组合已经是两个东西。这跟[SURE 收缩](/blog/stein-unbiased-risk-estimation/)的教训一致：评价估计量要看你实际消费的那个泛函，而不是全样本平均误差。

要泼的冷水也在图里：LL 边缘均值仍低于真值 33.2%，没有解决稀疏区方差问题；单次 900 样本、噪声 0.012 时，顶端尖峰只有 0.006，信号噪声比并不宽裕。带宽换成 0.04 或 0.12，名次还可能变。

还有一个常被忽略的判断：边缘偏误是否重要，取决于你拿拟合值做什么。如果只关心整条曲线的形状或内部区间的趋势，边缘 6.6 倍的 MSE 优势对总体指标贡献有限；如果策略只在分数最高的十分之一建仓，那么边缘那几个点的误差就是全部收益来源。换句话说，方法评估的权重应该跟着决策规则走，而不是跟着样本量走。这也是为什么本文刻意把「全网格 MSE」和「top 10% 真 Alpha」并列：前者 LL 只改善不到三成，后者改善超过三成，两个数字不矛盾，回答的是不同问题。若你的选股规则本来就不碰两端，那么为了边缘偏误换方法的收益接近零，反而要承担局部拟合在稀疏区的额外波动。

## 七、完整代码

下面脚本自包含：数据生成、四种实验、五张图与 `results.json` 全部由它一次跑出，可直接执行。

```python
# -*- coding: utf-8 -*-
"""Nadaraya-Watson boundary bias vs local linear regression, with MC checks.
Self-contained: data generation + experiments + 5 figures + results.json.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261126)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "kernel-local-linear-boundary-bias")
os.makedirs(OUT, exist_ok=True)

N = 300
SIGMA = 0.15
GRID = np.linspace(0.0, 1.0, 101)


def m_true(x):
    return 0.3 + 1.2 * x + 0.35 * np.sin(4.0 * np.pi * x)


def fit_nw(xq, xt, yt, h):
    d = (xq[:, None] - xt[None, :]) / h
    w = np.exp(-0.5 * d * d)
    return (w @ yt) / np.maximum(w.sum(axis=1), 1e-12)


def fit_ll(xq, xt, yt, h):
    z = xt[None, :] - xq[:, None]
    w = np.exp(-0.5 * (z / h) ** 2)
    s0 = w.sum(axis=1)
    s1 = (w * z).sum(axis=1)
    s2 = (w * z * z).sum(axis=1)
    t0 = w @ yt
    t1 = (w * z) @ yt
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    return (t0 * s2 - t1 * s1) / det


def loo_nw(xt, yt, h):
    d = (xt[:, None] - xt[None, :]) / h
    w = np.exp(-0.5 * d * d)
    np.fill_diagonal(w, 0.0)
    return (w @ yt) / w.sum(axis=1), w.sum(axis=1)


def loo_ll(xt, yt, h):
    z = xt[None, :] - xt[:, None]
    w = np.exp(-0.5 * (z / h) ** 2)
    np.fill_diagonal(w, 0.0)
    s0 = w.sum(axis=1); s1 = (w * z).sum(axis=1); s2 = (w * z * z).sum(axis=1)
    t0 = w @ yt; t1 = (w * z) @ yt
    det = s0 * s2 - s1 * s1
    det = np.where(np.abs(det) < 1e-14, 1e-14, det)
    return (t0 * s2 - t1 * s1) / det, det


def main():
    res = {}
    rng = RNG
    xt = (np.arange(N) + 0.5) / N
    mt = m_true(xt)
    mg = m_true(GRID)
    h0 = 0.08

    # exp1: single sample curves, boundary zoom
    y1 = mt + rng.normal(0, SIGMA, N)
    nw1 = fit_nw(GRID, xt, y1, h0)
    ll1 = fit_ll(GRID, xt, y1, h0)
    res["exp1_h"] = h0
    res["exp1_m0"] = float(m_true(np.array([0.0]))[0])
    res["exp1_m1"] = float(m_true(np.array([1.0]))[0])
    res["exp1_nw_at_0"] = float(nw1[0])
    res["exp1_ll_at_0"] = float(ll1[0])
    res["exp1_nw_at_1"] = float(nw1[-1])
    res["exp1_ll_at_1"] = float(ll1[-1])
    res["exp1_nw_bias_0"] = float(nw1[0] - mg[0])
    res["exp1_ll_bias_0"] = float(ll1[0] - mg[0])
    res["exp1_nw_bias_1"] = float(nw1[-1] - mg[-1])
    res["exp1_ll_bias_1"] = float(ll1[-1] - mg[-1])
    res["exp1_mse_nw"] = float(np.mean((nw1 - mg) ** 2))
    res["exp1_mse_ll"] = float(np.mean((ll1 - mg) ** 2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(GRID, mg, color="#222", lw=2, label="true m(x)")
    a1.plot(GRID, nw1, color="#c0392b", lw=1.8, label="Nadaraya-Watson")
    a1.plot(GRID, ll1, color="#2c7fb8", lw=1.8, label="local linear")
    a1.scatter(xt[::6], y1[::6], s=8, color="#999", alpha=0.5)
    a1.set_xlabel("x"); a1.set_ylabel("m(x)")
    a1.set_title("h=0.08: whole domain")
    a1.legend(fontsize=8)
    mask = GRID <= 0.12
    a2.plot(GRID[mask], mg[mask], color="#222", lw=2, label="true")
    a2.plot(GRID[mask], nw1[mask], color="#c0392b", lw=2, label="NW")
    a2.plot(GRID[mask], ll1[mask], color="#2c7fb8", lw=2, label="LL")
    a2.axvline(0, color="#555", ls=":")
    a2.set_xlabel("x"); a2.set_ylabel("m(x)")
    a2.set_title(f"left edge: NW bias {res['exp1_nw_bias_0']:+.3f}, LL {res['exp1_ll_bias_0']:+.3f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_curves.png"), dpi=130); plt.close()

    # exp2: MC pointwise bias^2 / variance at h=0.08
    reps = 240
    edge = (GRID <= 0.08) | (GRID >= 0.92)
    inner = (GRID >= 0.2) & (GRID <= 0.8)
    acc_nw = np.zeros(len(GRID)); acc_ll = np.zeros(len(GRID))
    sq_nw = np.zeros(len(GRID)); sq_ll = np.zeros(len(GRID))
    for _ in range(reps):
        yy = mt + rng.normal(0, SIGMA, N)
        a = fit_nw(GRID, xt, yy, h0)
        b = fit_ll(GRID, xt, yy, h0)
        acc_nw += a; acc_ll += b
        sq_nw += a * a; sq_ll += b * b
    acc_nw /= reps; acc_ll /= reps
    sq_nw /= reps; sq_ll /= reps
    var_nw = np.maximum(sq_nw - acc_nw ** 2, 0.0)
    var_ll = np.maximum(sq_ll - acc_ll ** 2, 0.0)
    bias_nw = acc_nw - mg; bias_ll = acc_ll - mg
    mse_nw = bias_nw ** 2 + var_nw
    mse_ll = bias_ll ** 2 + var_ll
    for nm, arr in [("nw", mse_nw), ("ll", mse_ll),
                    ("nw_edge", mse_nw[edge]), ("ll_edge", mse_ll[edge]),
                    ("nw_inner", mse_nw[inner]), ("ll_inner", mse_ll[inner])]:
        res["exp2_mse_" + nm] = float(np.mean(arr))
    res["exp2_bias2_nw_edge"] = float(np.mean(bias_nw[edge] ** 2))
    res["exp2_bias2_ll_edge"] = float(np.mean(bias_ll[edge] ** 2))
    res["exp2_var_nw_inner"] = float(np.mean(var_nw[inner]))
    res["exp2_var_ll_inner"] = float(np.mean(var_ll[inner]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(GRID, bias_nw ** 2, color="#c0392b", lw=1.8, label="NW bias^2")
    a1.plot(GRID, bias_ll ** 2, color="#2c7fb8", lw=1.8, label="LL bias^2")
    a1.axvspan(0, 0.08, color="#eee"); a1.axvspan(0.92, 1, color="#eee")
    a1.set_xlabel("x"); a1.set_ylabel("squared bias")
    a1.set_title(f"edge bias^2: NW {res['exp2_bias2_nw_edge']:.5f} vs LL {res['exp2_bias2_ll_edge']:.5f}")
    a1.legend(fontsize=8)
    a2.plot(GRID, var_nw, color="#c0392b", lw=1.8, ls="--", label="NW variance")
    a2.plot(GRID, var_ll, color="#2c7fb8", lw=1.8, ls="--", label="LL variance")
    a2.set_xlabel("x"); a2.set_ylabel("pointwise variance")
    a2.set_title(f"interior variance: NW {res['exp2_var_nw_inner']:.5f} vs LL {res['exp2_var_ll_inner']:.5f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_bias_variance.png"), dpi=130); plt.close()

    # exp3: bandwidth scan, edge vs interior MSE
    hs = np.array([0.02, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24, 0.32])
    scan = {k: [] for k in ["nw_edge", "ll_edge", "nw_inner", "ll_inner"]}
    for h in hs:
        se_nw = se_ll = si_nw = si_ll = 0.0
        for _ in range(70):
            yy = mt + rng.normal(0, SIGMA, N)
            a = fit_nw(GRID, xt, yy, h); b = fit_ll(GRID, xt, yy, h)
            se_nw += np.mean((a[edge] - mg[edge]) ** 2)
            se_ll += np.mean((b[edge] - mg[edge]) ** 2)
            si_nw += np.mean((a[inner] - mg[inner]) ** 2)
            si_ll += np.mean((b[inner] - mg[inner]) ** 2)
        scan["nw_edge"].append(se_nw / 70); scan["ll_edge"].append(se_ll / 70)
        scan["nw_inner"].append(si_nw / 70); scan["ll_inner"].append(si_ll / 70)
    res["exp3_hs"] = [float(x) for x in hs]
    res["exp3_scan"] = {k: [float(v) for v in arr] for k, arr in scan.items()}
    res["exp3_best_nw_edge_h"] = float(hs[int(np.argmin(scan["nw_edge"]))])
    res["exp3_best_ll_edge_h"] = float(hs[int(np.argmin(scan["ll_edge"]))])
    res["exp3_nw_edge_at_h016"] = float(scan["nw_edge"][4])
    res["exp3_ll_edge_at_h016"] = float(scan["ll_edge"][4])
    res["exp3_ll_inner_at_h002"] = float(scan["ll_inner"][0])
    res["exp3_ll_inner_best"] = float(np.min(scan["ll_inner"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.plot(hs, scan["nw_edge"], "o-", color="#c0392b", label="NW edge")
    a1.plot(hs, scan["ll_edge"], "s-", color="#2c7fb8", label="LL edge")
    a1.set_xscale("log"); a1.set_xlabel("bandwidth h"); a1.set_ylabel("MSE at edges")
    a1.set_title("NW edge error grows with h; LL has a U shape")
    a1.legend(fontsize=8)
    a2.plot(hs, scan["nw_inner"], "o-", color="#c0392b", label="NW interior")
    a2.plot(hs, scan["ll_inner"], "s-", color="#2c7fb8", label="LL interior")
    a2.set_xscale("log"); a2.set_xlabel("bandwidth h"); a2.set_ylabel("MSE in interior")
    a2.set_title("Interior: LL pays variance at small h")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_bandwidth.png"), dpi=130); plt.close()

    # exp4: LOOCV bandwidth under iid vs AR(1) errors
    hcv = np.array([0.02, 0.03, 0.04, 0.06, 0.08, 0.12, 0.16, 0.24])
    cvres = {}
    for tag, phi in [("iid", 0.0), ("ar1", 0.75)]:
        picks = {"nw": [], "ll": []}
        realized = {"nw": [], "ll": []}
        for _ in range(60):
            eps = rng.normal(0, SIGMA, N)
            if phi > 0:
                for j in range(1, N):
                    eps[j] = phi * eps[j - 1] + np.sqrt(1 - phi ** 2) * eps[j]
            yy = mt + eps
            scores = {"nw": np.zeros(len(hcv)), "ll": np.zeros(len(hcv))}
            for i, h in enumerate(hcv):
                pn, sn = loo_nw(xt, yy, h)
                pl, _ = loo_ll(xt, yy, h)
                scores["nw"][i] = np.mean((pn - yy) ** 2)
                scores["ll"][i] = np.mean((pl - yy) ** 2)
            for k in picks:
                hsel = float(hcv[int(np.argmin(scores[k]))])
                picks[k].append(hsel)
                est = fit_nw(GRID, xt, yy, hsel) if k == "nw" else fit_ll(GRID, xt, yy, hsel)
                realized[k].append(float(np.mean((est - mg) ** 2)))
        cvres[tag] = {"picks": picks, "realized": realized}
    res["exp4_h_grid"] = [float(x) for x in hcv]
    for tag in cvres:
        for k in ["nw", "ll"]:
            res[f"exp4_{tag}_{k}_h_med"] = float(np.median(cvres[tag]["picks"][k]))
            res[f"exp4_{tag}_{k}_mse"] = float(np.mean(cvres[tag]["realized"][k]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for tag, col in [("iid", "#2c7fb8"), ("ar1", "#c0392b")]:
        a1.hist(cvres[tag]["picks"]["ll"], bins=hcv, alpha=0.55, label=f"LL {tag}", color=col)
    a1.set_xlabel("LOOCV selected h"); a1.set_ylabel("count")
    a1.set_title("LOOCV bandwidth: independent vs AR(1) errors")
    a1.legend(fontsize=8)
    labels = ["iid NW", "iid LL", "AR1 NW", "AR1 LL"]
    vals = [res["exp4_iid_nw_mse"], res["exp4_iid_ll_mse"],
            res["exp4_ar1_nw_mse"], res["exp4_ar1_ll_mse"]]
    a2.bar(labels, vals, color=["#e0a3a3", "#8fbfe0", "#c0392b", "#2c7fb8"])
    for i, v in enumerate(vals):
        a2.text(i, v, f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("true MSE on grid")
    a2.set_title("CV-chosen h, evaluated against truth")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_loocv.png"), dpi=130); plt.close()

    # exp5: edge alpha in a score-to-return curve
    n2 = 900
    xs = rng.beta(2.0, 2.0, n2)
    def edge_alpha(s):
        return 0.001 * (s - 0.5) + 0.006 * np.exp(-((s - 0.9) / 0.07) ** 2)
    ys = edge_alpha(xs) + rng.normal(0, 0.012, n2)
    xq = np.linspace(0, 1, 201)
    hsel = 0.07
    nwq = fit_nw(xq, xs, ys, hsel)
    llq = fit_ll(xq, xs, ys, hsel)
    truth_q = edge_alpha(xq)
    edge_mask = xq >= 0.84
    res["exp5_h"] = hsel
    res["exp5_true_edge_mean"] = float(np.mean(truth_q[edge_mask]))
    res["exp5_nw_edge_mean"] = float(np.mean(nwq[edge_mask]))
    res["exp5_ll_edge_mean"] = float(np.mean(llq[edge_mask]))
    top = int(0.1 * n2)
    # evaluate the estimated top set against true alpha at those score points
    qnw = np.interp(xs, xq, nwq); qll = np.interp(xs, xq, llq)
    topn = np.argsort(qnw)[-top:]; topl = np.argsort(qll)[-top:]
    res["exp5_nw_top_true_alpha"] = float(np.mean(edge_alpha(xs[topn])))
    res["exp5_ll_top_true_alpha"] = float(np.mean(edge_alpha(xs[topl])))
    res["exp5_top_overlap"] = float(len(set(topn) & set(topl)) / top)
    res["exp5_nw_edge_bias"] = float(np.mean(nwq[edge_mask] - truth_q[edge_mask]))
    res["exp5_ll_edge_bias"] = float(np.mean(llq[edge_mask] - truth_q[edge_mask]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    a1.scatter(xs, ys, s=5, color="#bbb", alpha=0.4)
    a1.plot(xq, truth_q, color="#222", lw=2, label="true alpha")
    a1.plot(xq, nwq, color="#c0392b", lw=1.8, label="NW fit")
    a1.plot(xq, llq, color="#2c7fb8", lw=1.8, label="LL fit")
    a1.axvspan(0.84, 1.0, color="#f5e6e6")
    a1.set_xlabel("score s"); a1.set_ylabel("expected return")
    a1.set_title("Edge alpha sits where data are sparse")
    a1.legend(fontsize=8)
    a2.bar(["NW", "LL"], [res["exp5_nw_top_true_alpha"], res["exp5_ll_top_true_alpha"]],
           color=["#c0392b", "#2c7fb8"])
    a2.axhline(res["exp5_true_edge_mean"], color="#222", ls="--", lw=1.5, label="truth edge mean")
    for i, v in enumerate([res["exp5_nw_top_true_alpha"], res["exp5_ll_top_true_alpha"]]):
        a2.text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=9)
    a2.set_ylabel("true alpha of selected top 10%")
    a2.set_title(f"top-set overlap {res['exp5_top_overlap']:.2f}")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_edge_alpha.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 八、实操边界

结论只在你确实需要端点值时成立。如果特征本身是排序分位，实务上更稳的做法是把两端各留 5% 不交易，或者把分数映射到有界区间后再做局部拟合——后者只是把边界挪走，并不消灭偏误，别当成万能药。设计密度高度不均时，局部线性的方差代价会显著超过这里的等距实验，必要时改用局部二次或对设计密度做变带宽。最后，本文所有数字来自同一个固定随机种子 20261126 的模拟，不是实盘收益；把它们当成方法行为的量级参照，而不是策略预期收益的估计。
