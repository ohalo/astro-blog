---
title: "SURE 无偏风险估计：不知道真 Alpha 也能给收缩阈值定档，为什么软阈值一分不差、硬阈值却在 Stein 跳变处漏掉 0.84"
description: "给因子暴露和 Alpha 估计选收缩阈值，通常只能靠交叉验证。SURE 用 Stein 引理把真风险的无偏估计直接算出来：400 维 24 个真信号、400 次重复下软阈值偏差仅 0.0064；把同一自由度公式硬套到不连续的硬阈值上，偏差 -0.8374，恰被跳变边界项 +0.8323 抵消。σ 低估 40% 让实际风险从 0.2056 涨到 0.4410，高估 60% 只涨到 0.2242；t(5) 新息下 Stein 恒等式破功，偏差 -0.1733。五张真实计算图与可运行代码。"
publishDate: '2026-11-25T11:10:00Z'
tags:
  - 量化
  - 统计估计
  - 收缩
  - 模型选择
  - Python
language: zh
difficulty: advanced
image: "/images/stein-unbiased-risk-estimation/fig01_sure_vs_risk.png"
---

## 一、结论先行：调参可以不需要真值

给 400 个资产的预期收益或因子暴露做阈值收缩，阈值 $t$ 取多大？没有真值，交叉验证之外的另一条路是 **SURE（Stein Unbiased Risk Estimate）**：只用手里这份带噪观测 $y_i=\theta_i+\varepsilon_i$ 和已知的噪声方差 $\sigma^2$，就能算出任意估计规则 $\hat\theta=f(y)$ 的真实均方风险的**无偏估计**。它不需要知道 $\theta$，也不需要重抽样。

真值已知的 400 维模拟（24 个非零坐标，噪声方差 1）给出四个硬结论：

- 软阈值套用 SURE 是准的：400 次重复里平均 SURE 与真实风险只差 **0.0064**，单次样本上 SURE 选点带来的风险 0.2488， oracle 是 0.2411，只差 3.2%。
- 同一自由度公式硬套到**硬阈值**上是错的：平均偏差 **−0.8374** 每坐标，和 Stein 跳变边界项的理论值 **+0.8323** 量级吻合，加上边界修正后偏差缩到 **−0.0051**。
- 六档稀疏度扫描里，软阈值 SURE 相对 oracle 的平均风险差只有 **0.0226**，而把硬阈值误用公式选出的阈值平均多亏 **0.60** 左右；六种稀疏度下它**六次全输**给软阈值 SURE。
- SURE 不等于免调参：把噪声尺度估错，代价不对称。$\hat\sigma/\sigma=0.6$ 时实际风险 **0.4410**（正确时 0.2056，翻了一倍多），而 1.6 时只有 **0.2242**。噪声换成同方差 t(5) 后 Stein 恒等式不成立，偏差 **−0.1733**。

第一篇讲贝叶斯收缩的旧文从先验分布出发；这里走的是频率路线：不问先验，只问这条估计规则的风险能不能被数据自己报出来。

## 二、Stein 引理给出的那一项

若 $y=\theta+\varepsilon$，$\varepsilon\sim N(0,\sigma^2 I)$，且 $g(y)$ 对 $y$ 弱可微，Stein 引理说
$$

E[(y-\theta)^\top g(y)]=\sigma^2 E[\nabla\cdot g(y)].

$$
把它代进均方风险展开式，就得到 SURE：
$$

\widehat R(g)=\|y-g(y)\|^2-n\sigma^2+2\sigma^2\,\nabla\cdot g(y), \quad E[\widehat R]=E\|g(y)-\theta\|^2.

$$
三项的直觉：第一项是用观测代替真值算残差，第二项扣掉观测自带的 $n\sigma^2$ 噪声，第三项把估计规则对数据的敏感度（自由度）加回来。对**软阈值** $g_i=\operatorname{sign}(y_i)(|y_i|-t)_+$，$\partial g_i/\partial y_i=\mathbf 1\{|y_i|>t\}$ 几乎处处成立，于是
$$

\operatorname{SURE}_{soft}(t)=\sum_i\min(y_i^2,t^2)+2\sigma^2\#\{|y_i|>t\}-n\sigma^2.

$$
这个式子可以直接对 $t$ 扫网格取最小，一次数据扫完，不重抽、不拆训练验证集。

## 三、硬阈值：自由度公式看着一样，实际漏了跳变

**这是本文最重要的坑。** 硬阈值 $g_i=y_i\mathbf 1\{|y_i|>t\}$ 在 $|y_i|=t$ 处有跳跃，不是连续可微函数。很多人直接照抄软阈值的自由度 $\#\{|y_i|>t\}$，得到
$$

\operatorname{SURE}_{hard}^{naive}(t)=\sum_{|y_i|\le t} y_i^2+2\sigma^2\#\{|y_i|>t\}-n\sigma^2.

$$
在 400 维、24 个幅度 1.6–3.0 的真信号上重复 400 次，$t=1.4$ 时软阈值的 SURE 偏差是 **+0.0064**（蒙特卡洛噪声量级），而硬阈值误用公式的偏差是 **−0.8374**：它系统性地把真实风险报低了七成以上。![软阈值 SURE 与真风险对齐，硬阈值误用公式整体下移](/images/stein-unbiased-risk-estimation/fig01_sure_vs_risk.png)

这不是随机数问题，是漏项。跳跃处 Stein 分部积分多出边界贡献；在真值已知时，这一项可以精确写成 $2\sigma^2\sum_i \frac{t}{\sigma}[\varphi((t-\theta_i)/\sigma)+\varphi((t+\theta_i)/\sigma)]$。400 次重复平均后它等于 **+0.8323**，加到误用公式上，偏差从 −0.8374 收到 **−0.0051**。![400 次重复的偏差审计](/images/stein-unbiased-risk-estimation/fig02_unbiased_check.png)

结论有两条：其一，软阈值之所以在稀疏恢复文献里被反复推荐，不只是因为估计连续，也因为它的 SURE 是可用的；其二，硬阈值不是不能调参，而是不能拿软阈值的自由度糊弄过去——要么用交叉验证这类不依赖求导的规则，要么显式处理跳变边界。

## 四、和交叉验证正面对拍

实验在六档非零坐标数（0、4、12、24、60、400）上重复：每档生成一份新数据，同一个 $\theta$ 下分别计算软阈值 SURE 选点、5 折 plug-in 交叉验证选点、真风险 oracle 选点、不收缩，以及硬阈值误用公式选点的实际风险。![六档稀疏度下的真实风险对比](/images/stein-unbiased-risk-estimation/fig03_sparsity.png)

- 软阈值 SURE 相对 oracle 的平均风险差 **0.0226**；五档稀疏信号里最差的是 60 个非零坐标时（0.4086 对 0.3612），完全空白信号时 SURE 也只多亏 0.0295。
- 5 折 plug-in 交叉验证平均差 **0.1759**，在 24 个真信号时风险 0.3162，对 SURE 的 0.2317 和 oracle 的 0.2194，差距明显。
- 硬阈值误用公式在六档里**每一档都输给软阈值 SURE**，最夸张的全空白信号场景风险 0.6624，而软阈值 SURE 只有 0.0302。

必须讲清口径：这里的交叉验证把 $\theta_i^2$ 用 $y_i^2-\sigma^2$ 代入做 plug-in 风险，属于已知 $\sigma$ 下的近似，不是对所有 CV 方案的判决。它输给 SURE 的真正原因是 SURE 直接利用了噪声分布，而 plug-in 近似多扛了一层 $y_i^2$ 的抽样噪声。具体看，软阈值把观测变成 $c(|y_i|)y_i$，真实风险在坐标 $i$ 上是 $(1-c)^2\theta_i^2+c^2\sigma^2$；plug-in 版本把 $\theta_i^2$ 换成 $y_i^2-\sigma^2$，对单个坐标是无偏的，但 $y_i^2$ 的方差在 $n=400$ 里被累加了 400 次，选点时的抖动足以把阈值压到保守区间。SURE 的第三项 $2\sigma^2\#\{|y_i|>t\}$ 恰好用 Stein 引理把这层抖动解析地抵消掉，不需要重抽样。

那么什么时候该回到交叉验证？三种情况：噪声方差未知或显著异方差，此时 SURE 里的 $\sigma^2$ 本身是错设源；估计规则不连续或含离散选择，Stein 引理的边界项不可忽略；数据结构有依赖（比如按交易日分块的重叠样本），此时独立坐标假设失效，分块交叉验证反而更贴近真实抽样机制。本文的对照是在最有利于 SURE 的环境下做的：坐标独立、方差已知且真实、估计规则连续。结论不能外推到这三条之外。

## 五、两盆冷水：噪声尺度与分布假设

SURE 公式里的 $\sigma^2$ 必须真是噪声方差。实盘里它往往来自日内反转估计、残差估计或做市商报价，误差很常见。在 24 个真信号、$n=400$ 的同一份样本上固定真风险曲线，只改代进 SURE 的 $\hat\sigma^2$，结果是不对称的：![σ 错设下 SURE 选点与真实风险](/images/stein-unbiased-risk-estimation/fig04_sigma_mismatch.png)

- $\hat\sigma/\sigma=0.6$：选出的阈值塌到 **0.44**，实际风险 **0.4410**，是正确尺度下 **0.2056** 的 2.1 倍。低估噪声让 SURE 以为观测很干净，于是几乎不收缩，把噪声全留下。
- $\hat\sigma/\sigma=1.6$：选出的阈值涨到 **2.32**，风险 **0.2242**，只多了 9%。过度收缩确实伤信号，但在本设定下比低估温和得多。

第二盆冷水是分布。Stein 引理要求高斯新息；把噪声换成同方差的 t(5)（自由度为 5 的 t 分布除以 $\sqrt{5/3}$ 使方差归一），60 次重复平均下软阈值 SURE 的真实偏差是 **−0.1733** 每坐标，高斯对照是 **+0.0106**。此时 SURE 曲线仍然光滑、仍然有最小值，但那个最小值不再瞄准真实风险。金融收益的厚尾是常态，所以把 SURE 用在原始日收益上之前，先做波动率标准化或换稳健分布假设。![非高斯下引理破功与标量收缩对照](/images/stein-unbiased-risk-estimation/fig05_limits_and_shrink.png)

同图右侧给了一个更稳的用法：当估计规则只有一个标量收缩因子 $c$，$\hat\theta_i=c y_i$ 时，SURE 是 $c$ 的二次函数，最优点有闭式解。这份数据上 SURE 选 $c=0.18$，oracle 是 0.24，两者风险 0.2528 对 0.2489，而不收缩要 1.3767。规则越简单，SURE 越可靠；规则越复杂（跳变、数据依赖的离散选择），越要回头检查 Stein 引理的前提。

## 六、怎么用在策略研究里

一套可落地的流程是：先对资产收益做波动率标准化，让 $\varepsilon_i$ 尽量接近正态；再选连续可微或至少连续的收缩规则（软阈值、标量收缩）；用滑动窗口估计 $\sigma^2$，并刻意把窗口估出来的 $\hat\sigma$ 上下扰动 20% 看阈值是否大幅漂移；最后把 SURE 选出的阈值和 5 折交叉验证的结果并列看，两者接近才动手。

不要用它做三件事：给硬阈值规则直接套自由度公式；在厚尾原始收益上假装噪声是高斯；把 SURE 的最小值当成策略样本外收益的承诺——它估计的是估计误差，不是夏普比率。

```python
# -*- coding: utf-8 -*-
"""SURE 无偏风险估计：Soft 阈值 Stein 引理成立、Hard 阈值边界项破功。
自包含：数据生成 + 实验 + 画图 + 结果 JSON 全部在本文件，可直接 exec 复跑。"""
import os
import json
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RNG = np.random.default_rng(20261125)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "stein-unbiased-risk-estimation")
os.makedirs(OUT, exist_ok=True)


def soft(y, t):
    return np.sign(y) * np.maximum(np.abs(y) - t, 0.0)


def hard(y, t):
    return np.where(np.abs(y) > t, y, 0.0)


def sure_soft(y, t, sigma2):
    """连续软阈值：Stein 引理适用，df = #{|y|>t}。"""
    return float(np.sum(np.minimum(y * y, t * t)) + 2 * sigma2 * np.sum(np.abs(y) > t) - len(y) * sigma2)


def sure_hard_naive(y, t, sigma2):
    """把软阈值自由度公式硬套在不连续硬阈值上（文章中的反例）。"""
    keep = np.abs(y) > t
    return float(np.sum(y[~keep] ** 2) + 2 * sigma2 * np.sum(keep) - len(y) * sigma2)


def boundary_term(theta, t, sigma2):
    """硬阈值跳跃的 Stein 边界修正（oracle 版）：2*sigma^2*E[deltag/dy] 的跳变贡献。"""
    s = np.sqrt(sigma2)
    return float(2 * sigma2 * np.sum(t / s * (norm.pdf((t - theta) / s) + norm.pdf((t + theta) / s))))


def risk(est, theta):
    return float(np.mean((est - theta) ** 2))


def make_theta(n, s, rng):
    th = np.zeros(n)
    if s:
        ii = rng.choice(n, s, replace=False)
        th[ii] = rng.choice([-1, 1], s) * rng.uniform(1.6, 3.0, s)
    return th


def cv_pick_soft(y, folds=5, seeds=(11, 29, 47, 83, 101)):
    """5 折 plug-in 风险选软阈值：E[(y_te-theta)^2]=(1-c)^2*theta^2 + sigma^2*c^2，
    其中保留时 c=1-t/|y|，置零时 c=0；代入 theta^2=y^2-sigma^2（边际无偏）。"""
    ts = np.linspace(0, 3.0, 151)
    err = np.zeros_like(ts)
    for seed in seeds:
        perm = np.random.default_rng(seed).permutation(len(y))
        for k in range(folds):
            te = perm[k::folds]
            for j, t in enumerate(ts):
                c = np.maximum(1 - t / np.abs(y[te]), 0.0)
                err[j] += np.sum((1 - c) ** 2 * (y[te] ** 2 - 1.0) + c ** 2)
    return float(ts[np.argmin(err)])


def main():
    rng = RNG
    n = 400
    res = {}
    tg = np.linspace(0, 3.0, 301)

    # --- exp1: 单次样本上四条曲线（软 SURE / 软真风险 / 硬误用 SURE / 硬真风险） ---
    th = make_theta(n, 24, rng)
    y = th + rng.normal(0, 1.0, n)
    s_curve = np.array([sure_soft(y, t, 1.0) for t in tg]) / n
    r_soft = np.array([risk(soft(y, t), th) for t in tg])
    h_curve = np.array([sure_hard_naive(y, t, 1.0) for t in tg]) / n
    r_hard = np.array([risk(hard(y, t), th) for t in tg])
    t_sure = float(tg[np.argmin(s_curve)])
    t_cv = cv_pick_soft(y)
    t_or = float(tg[np.argmin(r_soft)])
    res["exp1_t_sure_soft"] = t_sure
    res["exp1_t_cv_soft"] = t_cv
    res["exp1_t_oracle_soft"] = t_or
    res["exp1_risk_sure_pick"] = float(r_soft[np.argmin(s_curve)])
    res["exp1_risk_cv_pick"] = float(r_soft[np.argmin(np.abs(tg - t_cv))])
    res["exp1_risk_oracle"] = float(np.min(r_soft))
    res["exp1_risk_no_shrink"] = float(r_soft[0])
    res["exp1_sure_minus_risk_mean"] = float(np.mean(s_curve - r_soft))
    res["exp1_hardnaive_minus_risk_mean"] = float(np.mean(h_curve - r_hard))

    plt.figure(figsize=(8.4, 5))
    plt.plot(tg, s_curve, color="#c0392b", lw=2.2, label="SURE-soft (valid)")
    plt.plot(tg, r_soft, color="#2c7fb8", lw=2, ls="--", label="true risk soft")
    plt.plot(tg, h_curve, color="#e67e22", lw=1.8, label="naive SURE on hard (invalid)")
    plt.plot(tg, r_hard, color="#27ae60", lw=1.8, ls="--", label="true risk hard")
    plt.axvline(t_sure, color="#c0392b", ls=":")
    plt.axvline(t_or, color="#2c7fb8", ls=":")
    plt.xlabel("threshold t"); plt.ylabel("risk per coordinate")
    plt.title("Soft SURE tracks risk; hard-threshold df formula fails")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig01_sure_vs_risk.png"), dpi=130); plt.close()

    # --- exp2: 500 次重复验证无偏性 + 跳变边界项理论对拍 ---
    t0 = 1.4
    acc = {"ss": 0.0, "rs": 0.0, "hn": 0.0, "rh": 0.0, "corr": 0.0}
    reps = 400
    for _ in range(reps):
        th2 = make_theta(n, 24, rng)
        y2 = th2 + rng.normal(0, 1.0, n)
        acc["ss"] += sure_soft(y2, t0, 1.0) / n
        acc["rs"] += risk(soft(y2, t0), th2)
        acc["hn"] += sure_hard_naive(y2, t0, 1.0) / n
        acc["rh"] += risk(hard(y2, t0), th2)
        acc["corr"] += boundary_term(th2, t0, 1.0) / n
    for k in acc:
        acc[k] /= reps
    res["exp2_t"] = t0
    res["exp2_reps"] = reps
    res["exp2_soft_sure_mean"] = acc["ss"]
    res["exp2_soft_risk_mean"] = acc["rs"]
    res["exp2_soft_bias"] = acc["ss"] - acc["rs"]
    res["exp2_hard_naive_mean"] = acc["hn"]
    res["exp2_hard_risk_mean"] = acc["rh"]
    res["exp2_hard_bias"] = acc["hn"] - acc["rh"]
    res["exp2_boundary_theory"] = acc["corr"]
    res["exp2_hard_bias_plus_boundary"] = acc["hn"] + acc["corr"] - acc["rh"]

    labels = ["SURE-soft", "risk-soft", "naive SURE-hard", "risk-hard", "boundary term"]
    vals = [acc["ss"], acc["rs"], acc["hn"], acc["rh"], acc["corr"]]
    cols = ["#c0392b", "#2c7fb8", "#e67e22", "#27ae60", "#8e44ad"]
    plt.figure(figsize=(8.4, 4.8))
    bars = plt.bar(labels, vals, color=cols)
    plt.axhline(0, color="#555", lw=0.8)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                 va="bottom" if v >= 0 else "top", fontsize=9)
    plt.ylabel("mean risk per coordinate")
    plt.title(f"{reps} repeats: soft bias = {acc['ss']-acc['rs']:.4f}, hard bias = {acc['hn']-acc['rh']:.4f}")
    plt.xticks(fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig02_unbiased_check.png"), dpi=130); plt.close()

    # --- exp3: 稀疏度扫描：soft SURE / plug-in CV / oracle / 不收缩 ---
    sparsities = [0, 4, 12, 24, 60, 400]
    names = ["sure_soft", "cv_soft", "oracle_soft", "no_shrink", "naive_sure_hard"]
    rk = {k: [] for k in names}
    for s in sparsities:
        th3 = make_theta(n, s, rng)
        y3 = th3 + rng.normal(0, 1.0, n)
        sg = np.linspace(0, 3.0, 151)
        sc = np.array([sure_soft(y3, t, 1.0) for t in sg])
        rc = np.array([risk(soft(y3, t), th3) for t in sg])
        kh = np.array([sure_hard_naive(y3, t, 1.0) for t in sg])
        rh = np.array([risk(hard(y3, t), th3) for t in sg])
        rk["sure_soft"].append(float(rc[np.argmin(sc)]))
        rk["oracle_soft"].append(float(np.min(rc)))
        rk["no_shrink"].append(float(rc[0]))
        rk["naive_sure_hard"].append(float(rh[np.argmin(kh)]))
        tc = cv_pick_soft(y3)
        rk["cv_soft"].append(float(risk(soft(y3, tc), th3)))
    res["exp3_sparsities"] = sparsities
    res["exp3_risks"] = rk
    res["exp3_sure_vs_oracle_gap"] = float(np.mean(np.array(rk["sure_soft"]) - np.array(rk["oracle_soft"])))
    res["exp3_cv_vs_oracle_gap"] = float(np.mean(np.array(rk["cv_soft"]) - np.array(rk["oracle_soft"])))
    res["exp3_hard_vs_soft_bad_cases"] = int(sum(1 for a, b in zip(rk["naive_sure_hard"], rk["sure_soft"]) if a > b))

    x = np.arange(len(sparsities)); w = 0.16
    plt.figure(figsize=(9.4, 5))
    for i, (k, col) in enumerate(zip(names, ["#c0392b", "#e67e22", "#2c7fb8", "#7f8c8d", "#8e44ad"])):
        plt.bar(x + (i - 2) * w, rk[k], w, label=k, color=col)
    plt.xticks(x, [str(s) for s in sparsities]); plt.xlabel("nonzero coordinates (n=400)")
    plt.ylabel("risk E||th^-th||^2/n"); plt.title("Soft SURE across sparsity; naive hard formula hurts")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "fig03_sparsity.png"), dpi=130); plt.close()

    # --- exp4: sigma 错设 ---
    ratios = np.round(np.linspace(0.6, 1.6, 21), 3)
    th4 = make_theta(n, 24, rng)
    y4 = th4 + rng.normal(0, 1.0, n)
    sg = np.linspace(0, 3.0, 151)
    rc = np.array([risk(soft(y4, t), th4) for t in sg])
    picked, rr = [], []
    for q in ratios:
        sc = np.array([sure_soft(y4, t, q * q) for t in sg])
        k = int(np.argmin(sc)); picked.append(float(sg[k])); rr.append(float(rc[k]))
    res["exp4_ratios"] = [float(q) for q in ratios]
    res["exp4_picked_t"] = picked
    res["exp4_risk"] = rr
    res["exp4_risk_0.6"] = rr[0]
    res["exp4_risk_1.0"] = rr[np.argmin(np.abs(ratios - 1.0))]
    res["exp4_risk_1.6"] = rr[-1]
    res["exp4_t_0.6"] = picked[0]
    res["exp4_t_1.6"] = picked[-1]

    fig, a1 = plt.subplots(figsize=(8.4, 4.8))
    a1.plot(ratios, picked, color="#c0392b", lw=2, label="picked t")
    a1.set_xlabel("sigma_hat / sigma_true"); a1.set_ylabel("picked t", color="#c0392b")
    a2 = a1.twinx()
    a2.plot(ratios, rr, color="#8e44ad", lw=2, label="realized risk")
    a2.set_ylabel("realized risk", color="#8e44ad")
    plt.title("Wrong noise scale shifts SURE choice")
    h1, l1 = a1.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a1.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig04_sigma_mismatch.png"), dpi=130); plt.close()

    # --- exp5: 非高斯新息下 Stein 引理破功 + 标量收缩 c 的 SURE ---
    biases = {}
    for dist in ["gauss", "t5"]:
        bb = []
        for _ in range(60):
            th5 = make_theta(n, 24, rng)
            eps = rng.normal(0, 1, n) if dist == "gauss" else rng.standard_t(5, n) / np.sqrt(5 / 3)
            y5 = th5 + eps
            sc = np.array([sure_soft(y5, t, 1.0) for t in sg])
            rc = np.array([risk(soft(y5, t), th5) for t in sg])
            bb.append(np.mean(sc / n - rc))
        biases[dist] = float(np.mean(bb))
    res["exp5_bias_gauss"] = biases["gauss"]
    res["exp5_bias_t5"] = biases["t5"]

    th6 = make_theta(n, 24, rng)
    y6 = th6 + rng.normal(0, 1.0, n)
    cs = np.linspace(0, 1.2, 121)
    sc6 = np.array([float(np.sum((1 - c) ** 2 * y6 * y6) + 2 * c * n - n) / n for c in cs])
    rc6 = np.array([risk(c * y6, th6) for c in cs])
    c_sure = float(cs[np.argmin(sc6)])
    res["exp5_c_sure"] = c_sure
    res["exp5_c_oracle"] = float(cs[np.argmin(rc6)])
    res["exp5_risk_sure_c"] = float(rc6[np.argmin(sc6)])
    res["exp5_risk_oracle_c"] = float(np.min(rc6))
    res["exp5_risk_no_shrink"] = float(rc6[-1])

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3))
    a1.bar(["Gaussian", "t(5) scaled"], [biases["gauss"], biases["t5"]],
           color=["#2c7fb8", "#c0392b"])
    a1.axhline(0, color="#555", lw=0.8)
    a1.set_ylabel("mean SURE - true risk")
    a1.set_title("Stein identity needs Gaussian noise")
    a2.plot(cs, sc6, color="#c0392b", lw=2, label="SURE(c)")
    a2.plot(cs, rc6, color="#2c7fb8", lw=2, ls="--", label="true risk(c)")
    a2.axvline(c_sure, color="#c0392b", ls=":")
    a2.set_xlabel("shrink factor c"); a2.legend(fontsize=8)
    a2.set_title("Scalar c: SURE picks near oracle")
    plt.tight_layout(); plt.savefig(os.path.join(OUT, "fig05_limits_and_shrink.png"), dpi=130); plt.close()

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
    print(json.dumps(res, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```


## 七、复现与口径

上面的脚本自包含：固定种子 20261125，生成全部数据、直接计算真实风险与 SURE 曲线、输出五张 PNG 与 `results.json`。所有正文数字都来自同一次运行；`t_sure_soft=1.77`、`risk_sure_pick=0.2488`、`exp2_soft_bias=0.0064`、`exp2_hard_bias=-0.8374`、`exp2_boundary_theory=0.8323`、`exp4_risk_0.6=0.4410`、`exp5_bias_t5=-0.1733` 均可在结果文件中逐项核对。图中真实风险使用已知真值计算，只用于验证与画图，从不参与阈值选择。
