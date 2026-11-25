---
title: "Whittle 似然估长记忆：只吃周期图快约 6.3 倍，为什么理论 95% 区间只覆盖 78.8%，AR(1) 又会伪装成 d=0.49"
description: "长记忆参数 d 的频域估计：Whittle 似然绕开 n×n 协方差矩阵，只用周期图。真值实验里 n=256 单次拟合 0.207ms 对 exact MLE 的 1.343ms（快约 6.3 倍），九档 n×d 组合的 RMSE 与 exact 同量级；但理论渐近标准差 0.0345 低于实测 0.0543，名义 95% 置信区间实际只覆盖 78.8%。数据含 AR(1) 时把它当纯长记忆会把 d=0.25 估成 0.4899，加入 AR 项后回到 0.2781。附五张真实计算图与可运行代码。"
publishDate: '2026-11-25T11:20:00Z'
tags:
  - 量化
  - 时间序列
  - 频域分析
  - 长记忆
  - Python
language: zh
difficulty: advanced
image: "/images/whittle-likelihood-long-memory/fig01_acf_spectrum.png"
---

## 一、结论先行：快是真快，置信区间是假准

估一个 ARFIMA 型长记忆参数 $d$，最直白的办法是写出 $n\times n$ 的协方差矩阵，反复做 Cholesky 分解求 exact 高斯似然。Whittle 似然把这件事搬到频域：用周期图 $I(\lambda)$ 和谱密度 $f_\theta(\lambda)$ 构造目标函数 $\sum_j\{\log f_\theta(\lambda_j)+I(\lambda_j)/f_\theta(\lambda_j)\}$，计算量从矩阵分解降到一次 FFT。

真值已知、共同随机数的实验给出四个结论：

- 速度：$n=256$ 时单次拟合 **0.21ms**，exact MLE 是 **1.33ms**，快 **6.3 倍**；样本越长差距越大。
- 点估计：九档 $n\times d$ 组合下，Whittle 的偏差绝对值最大 **0.0170**，RMSE 与 exact MLE 同量级（平均比值 1.17，小样本略吃亏）。渐近理论并没有在点上骗人。
- 区间估计骗人：理论标准差 $\sqrt{3/(n\pi^2)}=0.0345$，160 次重复里实测 **0.0543**，名义 95% 的区间实际只覆盖 **78.8%**。
- 短记忆误设最危险：真值 $d=0.25$ 的数据叠了 $\varphi=0.85$ 的 AR(1) 后，纯长记忆 Whittle 把 $d$ 顶到 **0.4899**（边界附近），加上 AR 项重估拉回 **0.2781**。

一句话：Whittle 适合快速筛参数，不适合直接拿渐近标准误做统计推断。

## 二、频域目标函数与它的理论承诺

对 ARFIMA(0,d,0)，谱密度是
$$

f_d(\lambda)=\sigma^2\,|1-e^{-i\lambda}|^{-2d}=\sigma^2(2\sin(\lambda/2))^{-2d}.

$$
在低频处 $f_d(\lambda)\sim \lambda^{-2d}$，所以 $d$ 其实就是 log-log 图上周期图的负斜率除以二。Whittle 目标函数把这个直觉变成了可优化的似然代理，省掉协方差矩阵，也省掉对 $\sigma^2$ 的显式建模——它作为尺度参数被目标函数自动消去。![样本自相关与理论谱](/images/whittle-likelihood-long-memory/fig01_acf_spectrum.png)

图 1 里 $n=512$、真值 $d=0.35$ 的单条路径很能说明长记忆估计的难处：理论 lag-1 自相关 **0.5385**，样本只有 **0.4478**；lag-20 理论 **0.2214**，样本掉到 **0.0135**。一条路径上肉眼几乎看不出慢衰减，周期图更是剧烈抖动。Whittle 目标函数吃的就是这些抖动的平均形状，不是某几个 lag。

理论还给出 $d$ 的渐近方差 $3/(n\pi^2)$。本文的 exact MLE 作为对拍基准，使用同样的平稳协方差 $\gamma(k)=\sigma^2\frac{\Gamma(1-2d)}{\Gamma(1-d)\Gamma(d)}\frac{\Gamma(k+d)}{\Gamma(k+1-d)}$，不做去均值，避免口径不一致造成假偏差。这里有个容易混淆的细节：$k=0$ 时 $\gamma(0)=\sigma^2\Gamma(1-2d)/\Gamma(1-d)^2$ 是**过程方差**，不是新息方差 $\sigma^2$；把两者混为一谈会让协方差矩阵丧失正定性。

### 周期图能估谱，但不能直接当谱用

Whittle 目标函数里每个 Fourier 频率贡献一项 $-\log f(\lambda_j)-I(\lambda_j)/f(\lambda_j)$。它成立的关键是：在 Fourier 频率 $\lambda_j=2\pi j/n$ 上，周期图的期望近似等于真谱，$E[I(\lambda_j)]\approx f(\lambda_j)$，不同频率渐近独立。但周期图在每一个频率上的方差不会随 $n$ 下降——$I(\lambda_j)$ 近似服从指数分布，标准差与均值同阶。也就是说，把 $n$ 加大只会给你更多个频率点，不会让某一点的谱估计更准。图 1 中周期图的毛刺正是这个性质的体现。

不过 Whittle 的聪明之处在于它把估计问题拆成两步：先靠 FFT 把数据压缩成 $n/2$ 个近似独立的频率观测，再在这些观测上做参数似然。信息没有凭空增加，收敛速度仍由低频区间的谱斜率决定。这也是为什么长记忆比短记忆难估：$d$ 的信息几乎全集中在 $\lambda\to0$ 的少数几个频率上，而实践里为了去掉样本均值，恰好要丢掉 $j=0$，$j=1$ 又最容易受低频泄露污染。样本量翻倍时，$d$ 的估计误差只按 $n^{-1/2}$ 缩，常数还被长记忆放大。

## 三、点估计：和 exact MLE 的正面对拍

同一条 $n=384$、$d=0.30$ 的路径上，exact 似然给出 **0.3028**，Whittle 给出 **0.2738**，差 **0.0289**。![两种目标函数在同一路径上的轮廓](/images/whittle-likelihood-long-memory/fig02_likelihood_curves.png)

九档蒙特卡洛（$n=128/256/512$，$d=0.10/0.30/0.40$，每档 40–70 次重复，各自独立种子）把这条观察推广开：exact MLE 基本无偏（最坏的偏差是 $n=512,d=0.30$ 时的 **−0.0215**），Whittle 偏差最大 **0.0170**；Whittle 的平均 RMSE 是 exact 的 **1.17 倍**，主要输在 $n=128$ 的短样本（$d=0.10$ 时 0.0841 对 0.0605），到 $n=512$ 两者已经非常接近（$d=0.40$ 时 0.0321 对 0.0279）。![九档样本量与真实 d 的 RMSE](/images/whittle-likelihood-long-memory/fig03_mc_rmse.png)

一个必须交代的实现坑：ARFIMA 协方差的 gamma-ratio 公式在 $d<0$ 时可能不再正定，而 Cholesky 分解的 `cho_factor` 不主动校验正定性，会给出一个看似很低的目标值，优化器就跑到负 $d$ 边界，制造出“exact MLE 大样本负偏 0.096”的假象。本文把优化下界锁在 0，并对非正定返回大值，假偏差随即消失。数值实验里，先疑实现，再疑理论。

## 四、两盆冷水：渐近标准误与短记忆误设

渐近方差 $3/(n\pi^2)$ 只在样本足够大时成立。$n=256$、$d=0.30$、160 次重复下，Whittle 估计的标准差实测 **0.0543**，理论只有 **0.0345**；用 $\hat d\pm1.96\sigma_{asym}$ 构造宽度 **0.1351** 的区间，实际覆盖 **78.8%**，离 95% 差了 16 个百分点。![覆盖度与速度对拍](/images/whittle-likelihood-long-memory/fig05_coverage_speed.png)

原因有两层：一是有限样本下周期图低频泄露，去均值又吃掉最低频；二是长记忆本身的收敛速度慢，$n=256$ 对 $d=0.3$ 还远远不够。渐近标准差的推导来自 Fisher 信息 $I(d)=\frac{1}{\pi}\int_0^\pi[\partial\log f_d(\lambda)/\partial d]^2d\lambda$，代入 $\partial\log f/\partial d=-2\log(2\sin(\lambda/2))$ 和积分 $\int_0^\pi\log^2(2\sin(\lambda/2))d\lambda=\pi^3/12$，得到每观测信息 $\pi^2/3$，于是 $Var(\hat d)\approx3/(n\pi^2)$。这个积分对频带做了均匀加权，隐含假设所有频率都携带等量的 $d$ 信息；实际上低频权重远高于高频，而低频恰恰是有限样本里被泄露和去均值破坏最严重的部分。渐近方差自然被低估。所以在报告 $d$ 的置信区间时，要么用自助法重标定，要么老老实实写“点估计”，别拿渐近公式吓人。

更危险的是模型误设。构造一个真 $d=0.25$ 的长记忆新息，再让它通过 AR(1) 滤波器（$\varphi\in\{0,0.3,0.6,0.85\}$），数据就成了 ARFIMA(1,d,0)。如果仍用纯 ARFIMA(0,d,0) 的 Whittle 谱去拟合，短记忆的低频峰会被当成更强的长记忆：$\varphi=0.3$ 时 $\hat d$ 已经 **0.4637**，$\varphi=0.85$ 时顶到 **0.4899**，偏了 **0.2399**。把 $\varphi$ 一并放进谱密度重估，$\varphi=0.85$ 时 $\hat d$ 回到 **0.2781**（偏 **0.0281**），估计出的 $\hat\varphi$ 均值 **0.8236**，也接近真值。![短记忆伪装成长记忆](/images/whittle-likelihood-long-memory/fig04_short_memory_bias.png)

注意这里的 exact MLE 用的是同一个误设谱，所以它也被推到 0.49 附近——这不是 Whittle 独有的毛病，是谱设定错了。真正的防线是研究流程：估长记忆之前先检验残差里有没有短记忆结构，把 ARMA 项和 $d$ 一起估。

## 五、用在波动率与交易信号上

长记忆估计在实盘里常见于两类问题：已实现波动率的持续性参数，以及订单流不平衡、成交间隔等微观结构序列的衰减速度。建议流程是：先用 Whittle 快速扫一遍 $d$，同时把 AR(1) 项放进去做对照；两个设定差超过 0.05 就别信纯长记忆；再用 exact MLE 在最终样本上复核点估计；置信区间用块自助法重算，不要用 $3/(n\pi^2)$。

也要清楚它不承诺什么：$d>0.5$ 的非平稳区域本文没有覆盖，结构突变、季节性、日内周期都会把谱的低频抬高，制造出假的长记忆。一个均值水平在样本中途跳变两格的序列，即使每一段内部完全没有记忆，合并后的周期图低频也会膨胀，纯 Whittle 估计能把 $d$ 顶到 0.4 以上——这不是长记忆，是没建模的截距。实操上至少做两道检查：把样本切成前后两半分别估 $d$，若差异超过 0.1，先怀疑突变而不是对着全样本调参；再把拟合残差拿去做短记忆检验，残差还有显著的一阶自相关就回到上一节的联合估计。季节性则要在谱上剔除对应频率的尖峰，否则它同样会被当成慢衰减的一部分。Whittle 省的是矩阵分解的时间，不是识别假设的功夫。

```python
# -*- coding: utf-8 -*-
"""Whittle 似然估长记忆 d：与 exact Gaussian MLE 对拍、短记忆误设、覆盖度与计时。
自包含：数据生成 + 实验 + 画图 + 结果 JSON，可直接 exec 复跑。"""
import os
import json
import time
import numpy as np
from scipy.special import gammaln
from scipy.linalg import toeplitz, cholesky, cho_factor, cho_solve
from scipy.optimize import minimize_scalar, minimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261125)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "whittle-likelihood-long-memory")
os.makedirs(OUT, exist_ok=True)


def acvf(k, d, sigma2=1.0):
    k = np.asarray(k)
    c = np.exp(gammaln(1 - 2 * d) - gammaln(1 - d) - gammaln(d))
    # k=0 时 gamma(0)=sigma^2*Gamma(1-2d)/Gamma(1-d)^2，是过程方差（非新息方差），公式对 k>=0 自洽
    return sigma2 * c * np.exp(gammaln(k + d) - gammaln(k + 1 - d))


def cov_matrix(n, d):
    k = np.arange(n)
    return toeplitz(acvf(k, d))


def gen_arfima(n, d, rng):
    L = cholesky(cov_matrix(n, d), lower=True, check_finite=False)
    return L @ rng.standard_normal(n)


def exact_nll(d, y):
    if d <= 0:  # d<0 的 gamma-ratio 协方差非正定，cho_factor 不校验会给出假的低目标值
        return 1e12
    n = len(y)
    S = cov_matrix(n, d)
    from numpy.linalg import LinAlgError
    try:
        c, low = cho_factor(S, lower=True, check_finite=False)
    except LinAlgError:
        return 1e12
    if np.any(np.diag(c) <= 0):
        return 1e12
    q = cho_solve((c, low), y, check_finite=False)
    return float(n * np.log(2 * np.pi) + 2 * np.sum(np.log(np.diag(c))) + y @ q)


def exact_mle(y):
    r = minimize_scalar(exact_nll, args=(y,), bounds=(1e-6, 0.49), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def periodogram(y):
    n = len(y)
    z = np.fft.rfft(y - y.mean()) / np.sqrt(n)
    j = np.arange(1, n // 2)
    lam = 2 * np.pi * j / n
    return lam, (np.abs(z[j]) ** 2), j


def whittle_nll(d, y, phi=0.0):
    lam, I, _ = periodogram(y)
    logf = -2 * d * np.log(2 * np.sin(lam / 2)) - np.log(np.abs(1 - phi * np.exp(-1j * lam)) ** 2)
    return float(np.sum(logf + I * np.exp(-logf)))


def whittle_mle(y):
    r = minimize_scalar(whittle_nll, args=(y,), bounds=(-0.44, 0.49), method="bounded",
                        options={"xatol": 1e-4})
    return float(r.x)


def whittle_mle_ar(y):
    def obj(p):
        return whittle_nll(p[0], y, phi=p[1])
    r = minimize(obj, x0=[0.1, 0.0], bounds=[(-0.44, 0.49), (-0.90, 0.90)], method="L-BFGS-B")
    return float(r.x[0]), float(r.x[1])


def main():
    rng = RNG
    res = {}

    # --- exp1: 样本 ACF / 周期图 vs 理论谱 ---
    n1 = 512
    d1 = 0.35
    y1 = gen_arfima(n1, d1, rng)
    ks = np.arange(1, 41)
    acf_theory = acvf(ks, d1) / acvf(0, d1)
    x1 = y1 - y1.mean()
    acf_emp = np.array([np.dot(x1[:-k], x1[k:]) / np.dot(x1, x1) for k in ks])
    lam, I, _ = periodogram(y1)
    sf = (2 * np.sin(lam / 2)) ** (-2 * d1)
    res["exp1_acf_lag1_theory"] = float(acf_theory[0])
    res["exp1_acf_lag1_emp"] = float(acf_emp[0])
    res["exp1_acf_lag20_theory"] = float(acf_theory[19])
    res["exp1_acf_lag20_emp"] = float(acf_emp[19])
    res["exp1_periodogram_first8"] = [float(v) for v in I[:8]]

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(ks, acf_theory, color="#2c7fb8", lw=2, label="theory ACF")
    a1.plot(ks, acf_emp, color="#c0392b", lw=1.5, label="sample ACF")
    a1.set_xlabel("lag"); a1.set_title(f"ARFIMA(0,d=0.35), n=512"); a1.legend(fontsize=8)
    a2.loglog(lam, I, color="#95a5a6", lw=0.8, label="periodogram")
    a2.loglog(lam, sf, color="#c0392b", lw=2, label="theory f(lambda)")
    a2.set_xlabel("Fourier frequency"); a2.set_ylabel("spectral density")
    a2.set_title("Periodogram is noisy but unbiased in log slope")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig01_acf_spectrum.png"), dpi=130); plt.close()

    # --- exp2: 同一路径的 exact / Whittle 目标函数与最优点 ---
    n2 = 384
    d2 = 0.30
    y2 = gen_arfima(n2, d2, rng)
    gd = np.linspace(-0.15, 0.49, 81)
    ecurve = np.array([exact_nll(float(d), y2) for d in gd])
    wcurve = np.array([whittle_nll(float(d), y2) for d in gd])
    de = exact_mle(y2)
    dw = whittle_mle(y2)
    res["exp2_d_true"] = d2
    res["exp2_d_exact"] = de
    res["exp2_d_whittle"] = dw
    res["exp2_gap"] = abs(de - dw)
    res["exp2_nll_exact_at_true"] = float(exact_nll(d2, y2))
    res["exp2_nll_exact_at_mle"] = float(exact_nll(de, y2))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.plot(gd, ecurve - ecurve.min(), color="#2c7fb8", lw=2, label="exact -log L")
    a1.axvline(de, color="#2c7fb8", ls="--", label=f"exact d={de:.3f}")
    a1.set_xlabel("d"); a1.set_title("Exact Gaussian likelihood"); a1.legend(fontsize=8)
    a2.plot(gd, wcurve - wcurve.min(), color="#c0392b", lw=2, label="Whittle objective")
    a2.axvline(dw, color="#c0392b", ls="--", label=f"Whittle d={dw:.3f}")
    a2.axvline(d2, color="#333", ls=":", label=f"true d={d2}")
    a2.set_xlabel("d"); a2.set_title("Whittle objective uses only the periodogram")
    a2.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig02_likelihood_curves.png"), dpi=130); plt.close()

    # --- exp3: 蒙特卡洛 bias / RMSE，三档 n × 三档 d；共同随机数 ---
    ns = [128, 256, 512]
    ds = [0.10, 0.30, 0.40]
    reps_map = {128: 70, 256: 60, 512: 40}
    vals = {}
    for n in ns:
        for d in ds:
            ee, ww = [], []
            rr = np.random.default_rng(7000 + n + int(d * 100))
            for _ in range(reps_map[n]):
                yy = gen_arfima(n, d, rr)
                ee.append(exact_mle(yy))
                ww.append(whittle_mle(yy))
            ee = np.array(ee); ww = np.array(ww)
            key = f"n{n}_d{int(d*100)}"
            vals[key] = {
                "exact_bias": float(np.mean(ee) - d),
                "exact_rmse": float(np.sqrt(np.mean((ee - d) ** 2))),
                "whittle_bias": float(np.mean(ww) - d),
                "whittle_rmse": float(np.sqrt(np.mean((ww - d) ** 2))),
                "whittle_sd": float(np.std(ww)),
                "theory_sd": float(np.sqrt(3 / (n * np.pi ** 2))),
                "reps": reps_map[n],
            }
    res["exp3"] = vals
    res["exp3_worst_whittle_bias"] = max(abs(v["whittle_bias"]) for v in vals.values())
    res["exp3_mean_rmse_ratio"] = float(np.mean([v["whittle_rmse"] / v["exact_rmse"] for v in vals.values()]))

    x = np.arange(len(ds)); w = 0.16
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.2), sharey=False)
    for ax, n in zip(axes, ns):
        for i, key in enumerate(["exact_rmse", "whittle_rmse"]):
            col = "#2c7fb8" if i == 0 else "#c0392b"
            ax.bar(x + (i - 0.5) * w, [vals[f"n{n}_d{int(d*100)}"][key] for d in ds], w,
                   color=col, label="exact MLE" if i == 0 else "Whittle")
        ax.set_xticks(x); ax.set_xticklabels([str(d) for d in ds])
        ax.set_xlabel("true d"); ax.set_title(f"n={n}, RMSE"); ax.legend(fontsize=8)
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig03_mc_rmse.png"), dpi=130); plt.close()

    # --- exp4: 短记忆误设：ARFIMA(0,d,0) 新息过 AR(1) 滤波器 ---
    n4 = 512
    phi_true = [0.0, 0.3, 0.6, 0.85]
    d4 = 0.25
    mis = []
    for ph in phi_true:
        e_hat, w_hat, c_hat, c_phi = [], [], [], []
        rr = np.random.default_rng(9100 + int(ph * 100))
        for _ in range(45):
            eta = gen_arfima(n4, d4, rr)
            u = np.zeros(n4)
            for t in range(1, n4):
                u[t] = ph * u[t - 1] + eta[t]
            yy = u  # 真均值为 0；exact 似然与平稳协方差必须同源，不做去均值
            e_hat.append(exact_mle(yy))
            w_hat.append(whittle_mle(yy))
            cd, cp = whittle_mle_ar(yy)
            c_hat.append(cd); c_phi.append(cp)
        mis.append({
            "phi": ph,
            "exact_mean": float(np.mean(e_hat)),
            "whittle_mis_mean": float(np.mean(w_hat)),
            "whittle_ar_mean": float(np.mean(c_hat)),
            "phi_hat_mean": float(np.mean(c_phi)),
        })
    res["exp4_misspec"] = mis
    res["exp4_bias_at_0.85_mis"] = mis[-1]["whittle_mis_mean"] - d4
    res["exp4_bias_at_0.85_fixed"] = mis[-1]["whittle_ar_mean"] - d4

    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    phs = [m["phi"] for m in mis]
    ax.plot(phs, [m["exact_mean"] for m in mis], color="#2c7fb8", marker="o", label="exact ARFIMA(0,d,0)")
    ax.plot(phs, [m["whittle_mis_mean"] for m in mis], color="#c0392b", marker="s", label="Whittle misspecified")
    ax.plot(phs, [m["whittle_ar_mean"] for m in mis], color="#27ae60", marker="^", label="Whittle with AR(1)")
    ax.axhline(d4, color="#555", ls=":", label=f"true d={d4}")
    ax.set_xlabel("AR(1) phi in data"); ax.set_ylabel("mean d_hat")
    ax.set_title("Short memory looks like long memory if you ignore AR(1)")
    ax.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig04_short_memory_bias.png"), dpi=130); plt.close()

    # --- exp5: 覆盖度 + 计时 ---
    n5 = 256
    d5 = 0.30
    rr = np.random.default_rng(555)
    ds_hat = []
    for _ in range(160):
        yy = gen_arfima(n5, d5, rr)
        ds_hat.append(whittle_mle(yy))
    ds_hat = np.array(ds_hat)
    theory_sd = np.sqrt(3 / (n5 * np.pi ** 2))
    lo, hi = d5 - 1.96 * theory_sd, d5 + 1.96 * theory_sd
    cov = float(np.mean((ds_hat >= lo) & (ds_hat <= hi)))
    res["exp5_coverage"] = cov
    res["exp5_nominal"] = 0.95
    res["exp5_theory_sd"] = float(theory_sd)
    res["exp5_emp_sd"] = float(np.std(ds_hat))
    res["exp5_ci_width"] = float(hi - lo)

    bench = gen_arfima(n5, 0.35, rr)
    t0 = time.perf_counter()
    for _ in range(12):
        exact_mle(bench)
    t_exact = (time.perf_counter() - t0) / 12
    t0 = time.perf_counter()
    for _ in range(12):
        whittle_mle(bench)
    t_whittle = (time.perf_counter() - t0) / 12
    res["t_exact_ms"] = t_exact * 1000
    res["t_whittle_ms"] = t_whittle * 1000
    res["speedup"] = t_exact / t_whittle

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.hist(ds_hat, bins=24, color="#95a5a6", edgecolor="white")
    a1.axvspan(lo, hi, color="#2c7fb8", alpha=0.18, label="theory 95% CI")
    a1.axvline(d5, color="#c0392b", lw=2, label=f"true d={d5}")
    a1.set_xlabel("Whittle d_hat"); a1.set_title(f"coverage {cov:.3f} (nominal 0.95)")
    a1.legend(fontsize=8)
    a2.bar(["exact MLE", "Whittle"], [t_exact * 1000, t_whittle * 1000],
           color=["#2c7fb8", "#c0392b"])
    a2.set_ylabel("ms per fit (n=256)")
    a2.set_title(f"Whittle is {t_exact/t_whittle:.1f}x faster")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_coverage_speed.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 六、复现与口径

脚本固定种子 20261125，五张图与 `results.json` 同源：计时字段随机器负载抖动，以 `results.json` 当次记录为准、`exp5_coverage=0.7875`、`exp4_bias_at_0.85_mis=0.2399`、`exp4_bias_at_0.85_fixed=0.0281` 均可逐项核对。蒙特卡洛各档使用独立种子以保证共同随机数可复现；计时是对同一条路径各做 12 次拟合取平均，只做相对比较，不承诺绝对性能。
