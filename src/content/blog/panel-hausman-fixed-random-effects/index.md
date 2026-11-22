---
title: "固定效应还是随机效应？Hausman 检验在因子面板上的真实表现，以及它最容易被误读的三个地方"
description: "面板数据该用固定效应（FE）还是随机效应（RE）？教科书说 H0 成立时两者一致、RE 更有效，H0 不成立时只有 FE 一致。一个真值已知的蒙特卡洛实验（平衡面板 200 个体 ×10 期，600 次重复）给出更细的答案：个体效应与解释变量的相关性 delta 从 0 升到 0.8，RE 的 beta 偏误从 -0.0003 线性涨到 +0.0403、MSE 从 0.00053 翻到 0.00215，FE 偏误始终钉在 -0.0002。Hausman 检验在 delta=0 时虚警率 5.3%、统计量均值 0.987（贴 chi2(1) 理论均值 1），delta=0.1 时功效就冲到 84.8%，delta=0.2 时 99.8%。三个反直觉发现：其一，当解释变量全是时变的，RE 的效率优势只有约 1%，Var(FE)/Var(RE) 从 T=3 到 40 都在 1.00 到 1.04 之间；其二，静态面板里异方差与序列相关几乎不动 Hausman 的虚警率（iid 5.3%、异方差 5.0%、序列相关 5.2%），方差误设在 FE-RE 差值里大体抵消了；其三，小样本 T=5 的 chi2 近似就已经很准（虚警 5.0%）。结论：Hausman 真正要防的是内生性本身，不是方差设定。"
publishDate: '2026-11-22T01:20:00Z'
tags:
  - 量化
  - 计量经济学
  - 面板数据
  - 因子模型
  - Python
language: zh
difficulty: advanced
image: "/images/panel-hausman-fixed-random-effects/fig1_bias_mse.png"
---

## 一、结论先行

做多因子选股，回归里几乎总有一个绕不开的选择：个股的「个体差异」——那些不随时间变化的公司特质——到底该当成固定效应还是随机效应？选错了，轻则损失效率，重则整条风险溢价都被内生性污染。Hausman 检验是教科书给出的裁判，但它在实战里的表现，和教科书给人的印象并不完全一样。

本文用一个真值已知的面板实验把这件事量化。数据生成过程写死：200 个个体、每个 10 期，被解释变量由 beta=0.8 的真系数驱动。内生性通过一个共同的个体驱动因子注入——它同时影响个体效应 alpha 和解释变量 x，强度由 delta 控制。delta=0 时 alpha 与 x 无关，是随机效应的理想世界；delta>0 时两者相关，随机效应开始有偏。所有数字来自同一段脚本，附在文末。

先把结论摆出来。第一，delta 从 0 升到 0.8，RE 的 beta 偏误从 -0.0003 线性涨到 +0.0403，MSE 翻了四倍，而 FE 始终无偏。第二，Hausman 检验在原假设下虚警率 5.3%、统计量均值 0.987，贴着卡方分布的理论均值 1；delta 只要到 0.1，功效就冲到 84.8%。第三，也是最值得警惕的：当解释变量全是时变的，RE 的效率优势只有约 1%，教科书那句「RE 更有效」在这个设定下几乎不成立。第四，静态面板里异方差和序列相关几乎不动 Hausman 的虚警率——方差误设在 FE 与 RE 的差值里大体抵消了。第五，小样本 T=5 的卡方近似就已经很准。一句话：Hausman 真正要防的是内生性本身，不是方差设定。

## 二、两个估计量，一个裁判

面板模型的写法是 $y_{it} = \beta x_{it} + \alpha_i + \varepsilon_{it}$。个体效应 $\alpha_i$ 是关键：它就是那些不随时间变化、又会影响 y 的公司特质。

固定效应（FE）的做法是对每个个体做组内去均值，把 $\alpha_i$ 直接消掉，再用去均值后的 x 和 y 回归。代价是损失了所有时不变变量的信息。随机效应（RE）则假设 $\alpha_i$ 与 x 不相关，用广义最小二乘（GLS）做一个部分的 quasi-demeaning，参数 $\theta = 1 - \sqrt{\sigma_\varepsilon^2/(T\sigma_\alpha^2 + \sigma_\varepsilon^2)}$。它保留了组间变异，所以在 H0 成立时更高效；但如果 $\alpha_i$ 其实和 x 相关，RE 就会把这部分相关错误地算进 beta，产生遗漏变量偏误。

Hausman 检验的逻辑很优雅：如果 H0 成立，FE 与 RE 都一致，但 FE 效率低，所以两者的差应该只反映抽样噪声；如果 H0 不成立，RE 有偏而 FE 无偏，两者的差就会系统性偏离零。统计量写成

$$
H = \frac{(\hat\beta_{FE} - \hat\beta_{RE})^2}{\widehat{\mathrm{Var}}(\hat\beta_{FE}) - \widehat{\mathrm{Var}}(\hat\beta_{RE})}
$$

在原假设下服从自由度 1 的卡方分布。分母必须用两个估计量方差的差——因为 $\hat\beta_{FE}$ 与 $\hat\beta_{RE}$ 高度相关，直接算差值方差会重复计算协方差项，而 Hausman 的妙处正在于这个差恰好给出了正确的差值方差。

## 三、内生性一上来，RE 就崩

先看最核心的对比。让 delta 从 0 扫到 0.8，观察两个估计量的偏误和均方误差。

![图1：FE 与 RE 的偏误和 MSE 随内生性强度 delta 的变化](/images/panel-hausman-fixed-random-effects/fig1_bias_mse.png)

delta=0 时，FE 偏误 -0.0002、RE 偏误 -0.0003，两者都贴着真值，MSE 都是 0.00053——这是 H0 成立时应有的样子。delta 只要到 0.1，RE 偏误就涨到 +0.0050，FE 仍是 -0.0002；delta=0.4 时 RE 偏误 +0.0206，MSE 从 0.00053 膨胀到 0.00095；delta=0.8 时 RE 偏误 +0.0403，MSE 达到 0.00215，是 FE 的四倍。FE 的偏误从头到尾钉在 -0.0002，MSE 恒定在 0.00053，完全不受内生性影响。

这正是 FE 的价值：它把个体效应整个消掉，因此无论 alpha 和 x 怎么相关，都不污染 beta。代价是它也消掉了所有时不变的解释变量——如果你想估计「公司注册地」「是否国企」这类不随时间变化的变量效应，FE 无能为力。这是选 FE 之前必须接受的取舍。

## 四、被夸大的效率优势

教科书推荐 RE 的核心理由是效率：H0 成立时 RE 比 FE 方差更小。本文把这个优势放到显微镜下，让 T 从 3 变到 40，看 Var(FE)/Var(RE) 的比值。

![图2：原假设下 FE 与 RE 的方差比，RE 的效率优势只有约 1%](/images/panel-hausman-fixed-random-effects/fig2_efficiency.png)

结果出人意料。T=3 时比值 1.039，T=5 时 1.007，T=10 时 1.011，T=20 和 40 时都是 1.00 出头。换句话说，在这个解释变量全是时变特征的设定下，RE 的方差只比 FE 小约 1%，几乎可以忽略。

原因值得讲清楚。RE 的效率增益来自利用组间变异——那些个体层面的、不随时间变化的信息。但当解释变量本身每个时期都在变（本文里 $x_{it} = \delta z_i + w_{it}$，$w_{it}$ 是每期独立的噪声），组间变异里能提供的额外信息本就很少，RE 能榨出来的效率自然有限。只有在你确实有一批时不变的解释变量、且误差结构接近随机效应假设时，RE 的效率优势才会明显。

这对实践是个重要提醒：如果因子全是时变的（估值、动量、波动率这类大多数技术面因子都是），那么「为了效率而选 RE」的理由基本不成立。既然 RE 省不了多少方差，不如直接用 FE，把内生性的隐患一并消掉。

## 五、Hausman 的虚警与功效

裁判本身准不准？先看原假设下的虚警率，再看备择假设下的功效。

![图3：原假设下不同 N、T 组合的 Hausman 虚警率](/images/panel-hausman-fixed-random-effects/fig3_size.png)

原假设下，八组 N、T 配置的虚警率全部落在 4.5% 到 5.7% 之间：N=50、T=5 时 5.0%，N=200、T=10 时 5.3%，N=500、T=20 时 4.9%。统计量均值在 0.959 到 1.037 之间，全都贴着卡方分布的理论均值 1。特别值得注意的是 T=5——很多人担心个体数少、时间短时卡方近似会失真，但这里即便 T=5，虚警率也守在 5.0%。自由度 1 的卡方近似在静态面板下比想象中稳健。

![图4：Hausman 检验的功效随内生性强度和样本量的变化](/images/panel-hausman-fixed-random-effects/fig4_power.png)

功效这边同样符合预期。delta=0 时虚警率 4.4% 到 4.5%（就是 size）；delta=0.05 时，N=100 的功效 18.0%、N=300 的 46.5%；delta=0.1 时冲到 55.0% 与 96.3%；delta=0.2 时 N=100 已达 97.9%，N=300 满格 100%。样本量的作用非常明显——同样的内生性强度，个体数从 100 增到 300，delta=0.1 的功效从 55.0% 跳到 96.3%。这说明 Hausman 检验检出轻微内生性需要足够的截面样本，小样本下「检验不显著」不等于「没有内生性」。

## 六、方差设定误设：一个被过度警告的风险

计量教材常提醒：异方差和序列相关会让 Hausman 检验失真，要用聚类稳健标准误。这句话在动态面板里是对的，但在静态面板里，情况没那么严重。本文把误差项分别改成异方差（方差随 |x| 变化）和 AR(1) 序列相关，看虚警率怎么变。

![图5：异方差与序列相关下，朴素 Hausman 的虚警率几乎不动](/images/panel-hausman-fixed-random-effects/fig5_robust.png)

结果显示：iid 场景虚警率 5.3%，异方差场景 5.0%，序列相关场景 5.2%，三者的方差校准 Wald 统计量也都在 5.3% 到 5.4%。朴素卡方 Hausman 面对方差误设几乎毫发无损。

原因在分母的结构。Hausman 统计量用的是 FE 与 RE 估计量方差的差，而这两个方差都正比于同一个误差方差 $\sigma_\varepsilon^2$。当误差项变成异方差或序列相关，这个共同的膨胀因子在分子和分母里同时出现，做差时大体抵消了。所以真正让 Hausman 失去意义的不是方差设定，而是内生的个体效应本身——那正是它要检测的东西。

当然，这个结论有边界：一旦模型引入滞后因变量变成动态面板，FE 估计量本身会产生 Nickell 偏误，Hausman 的前提就不再成立，那时需要差分 GMM 或系统 GMM，本文的结论不适用。

## 七、给量化研究者的三条建议

第一，先问你的因子是不是时变的。如果全是时变因子（大多数技术面因子都是），RE 的效率优势微乎其微，直接用 FE 更省心，还能免疫个体效应与因子的内生性。只有当你的模型确实包含重要的时不变变量、且有把握它们外生时，RE 才值得考虑。第二，Hausman 检验不显著不代表安全——功效分析显示，样本不够大时它连 delta=0.1 的内生性都可能漏检。把它当成一个筛查工具，而不是通行证。第三，别被「异方差要用稳健 Hausman」的教条吓住：静态面板里朴素检验对方差误设相当耐用，真正该花精力的是检查个体效应与解释变量的相关结构。

## 八、完整代码与复现

上面全部数字与五张图来自下面这段自包含脚本。它用固定种子 20261122，通过共同个体驱动因子注入内生性，依次计算 FE 与 RE 估计量、偏误与 MSE、效率比、虚警率、功效，以及方差误设下的稳健性。FE 用组内去均值，RE 用 GLS 的 quasi-demeaning，方差分量从组内残差与组间变异分别估计。正文每个数字都能在脚本输出中逐项核对。

```python
# -*- coding: utf-8 -*-
"""Fixed effects vs random effects: the Hausman test, honestly simulated.

Single source of truth for every number and figure in the article.

Balanced panel, N individuals x T periods:
    y_it = beta*x_it + alpha_i + eps_it
Endogeneity is generated by a shared individual driver z_i:
    alpha_i = gamma*z_i + a_i
    x_it    = delta*z_i + w_it
delta = 0  -> alpha_i independent of x  (H0, RE efficient & consistent)
delta > 0  -> alpha_i correlated with x (H1, RE biased, FE consistent)

Estimators:
    FE : within (demean by individual) OLS -> beta_FE
    RE : GLS quasi-demeaning with theta = 1 - sqrt(s2e/(T*s2a+s2e))
         -> beta_RE
Under H0 both are consistent; RE is more efficient (smaller variance).
Hausman:  H = (b_FE - b_RE)^2 / (s2e*(1/Sww - 1/Stt))  ~ chi2(1)
where Sww = sum (x - xbar)^2, Stt = sum (x - theta*xbar)^2,
and s2e is estimated from FE residuals (consistent under both H0 and H1).

Key honest findings the article documents:
  * delta=0: FE and RE both unbiased, RE variance smaller; Hausman size ~5%.
  * delta>0: RE bias grows linearly, FE stays unbiased; Hausman power rises.
  * small T: chi2(1) approximation is already accurate even at T=5 here.
  * heteroskedasticity / serial correlation: in a STATIC panel the naive
    chi2 Hausman size barely moves (the variance misspecification largely
    cancels in the FE-RE difference); the real danger is endogeneity itself,
    which is what the test is designed to detect.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats, special

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261122
BETA = 0.8
CONF = 0.95
CHI95 = float(stats.chi2.ppf(CONF, 1))     # 3.841

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "panel-hausman-fixed-random-effects")
os.makedirs(_OUT, exist_ok=True)


# ---------------------------------------------------------------- DGP & estimators
def draw_panel(rng, n, t, delta, gamma=1.0, hetero=False, serial=False,
               s2e=1.0, s2a=1.0, s2w=1.0):
    z = rng.standard_normal(n)                       # shared individual driver
    a = rng.standard_normal(n) * np.sqrt(s2a)
    alpha = gamma * z + a
    w = rng.standard_normal((n, t)) * np.sqrt(s2w)
    x = delta * z[:, None] + w
    eps = rng.standard_normal((n, t)) * np.sqrt(s2e)
    if hetero:
        # variance scales with |x| -> RE/FE variance estimates get distorted
        scale = 0.5 + 0.6 * np.abs(x)
        eps = eps * scale / np.sqrt(np.mean((0.5 + 0.6 * np.abs(w)) ** 2))
    if serial:
        # AR(1) within-figure innovation, rho=0.6
        rho = 0.6
        e0 = rng.standard_normal(n)
        for k in range(t):
            e0 = rho * e0 + np.sqrt(1 - rho ** 2) * rng.standard_normal(n)
            eps[:, k] = e0 * np.sqrt(s2e)
    y = BETA * x + alpha[:, None] + eps
    return x, y, alpha


def _theta(s2e, s2a, t):
    denom = t * s2a + s2e
    return 1.0 - np.sqrt(s2e / denom) if denom > 0 else 0.0


def estimate(x, y, t):
    """Return beta_FE, beta_RE, Hausman H, and variance components."""
    n = x.shape[0]
    xb = x.mean(axis=1, keepdims=True)
    yb = y.mean(axis=1, keepdims=True)
    xt = x - xb
    yt = y - yb
    sww = float(np.sum(xt ** 2))
    b_fe = float(np.sum(xt * yt) / sww)
    resid_fe = yt - b_fe * xt
    s2e = float(np.sum(resid_fe ** 2) / (n * (t - 1)))

    # variance of the individual effect from between variation
    between = yb[:, 0] - b_fe * xb[:, 0]
    s2a = max(0.0, float(np.var(between, ddof=1) - s2e / t))
    th = _theta(s2e, s2a, t)
    xq = x - th * xb
    yq = y - th * yb
    stt = float(np.sum(xq ** 2))
    b_re = float(np.sum(xq * yq) / stt)

    vdiff = s2e * (1.0 / sww - 1.0 / stt)
    h = (b_fe - b_re) ** 2 / vdiff if vdiff > 0 else np.nan
    return {"b_fe": b_fe, "b_re": b_re, "h": h, "s2e": s2e, "s2a": s2a,
            "theta": th, "vdiff": vdiff, "sww": sww, "stt": stt}


def simulate(n, t, delta, reps, seed, **kw):
    rng = np.random.default_rng(seed)
    b_fe = np.empty(reps); b_re = np.empty(reps)
    hs = np.empty(reps); ok = np.empty(reps, dtype=bool)
    for r in range(reps):
        x, y, _ = draw_panel(rng, n, t, delta, **kw)
        e = estimate(x, y, t)
        b_fe[r] = e["b_fe"]; b_re[r] = e["b_re"]
        hs[r] = e["h"]; ok[r] = np.isfinite(e["h"])
    return {
        "b_fe": b_fe, "b_re": b_re, "h": hs, "ok": ok,
        "bias_fe": float(np.mean(b_fe) - BETA),
        "bias_re": float(np.mean(b_re) - BETA),
        "sd_fe": float(np.std(b_fe, ddof=1)),
        "sd_re": float(np.std(b_re, ddof=1)),
        "mse_fe": float(np.mean((b_fe - BETA) ** 2)),
        "mse_re": float(np.mean((b_re - BETA) ** 2)),
        "size": float(np.mean(hs[ok] > CHI95)) if ok.any() else np.nan,
        "h_mean": float(np.nanmean(hs)) if ok.any() else np.nan,
        "valid_frac": float(np.mean(ok)),
    }


# ---------------------------------------------------------------- experiments
def exp_bias_mse():
    reps = 600
    rows = []
    for delta in [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]:
        s = simulate(200, 10, delta, reps, SEED + 1)
        rows.append({
            "delta": delta,
            "bias_fe": s["bias_fe"], "bias_re": s["bias_re"],
            "sd_fe": s["sd_fe"], "sd_re": s["sd_re"],
            "mse_fe": s["mse_fe"], "mse_re": s["mse_re"],
            "size": s["size"], "h_mean": s["h_mean"],
        })
    return {"reps": reps, "n": 200, "t": 10, "rows": rows}


def exp_efficiency():
    """Under H0 (delta=0), compare FE and RE sampling variance across T."""
    reps = 800
    rows = []
    for t in [3, 5, 10, 20, 40]:
        s = simulate(200, t, 0.0, reps, SEED + 2)
        rows.append({"t": t, "sd_fe": s["sd_fe"], "sd_re": s["sd_re"],
                     "var_ratio": (s["sd_fe"] / s["sd_re"]) ** 2,
                     "mse_fe": s["mse_fe"], "mse_re": s["mse_re"]})
    return {"reps": reps, "n": 200, "rows": rows}


def exp_size():
    """Hausman size under H0 for various N, T."""
    reps = 1500
    rows = []
    for (n, t) in [(50, 5), (50, 10), (50, 20), (200, 5), (200, 10),
                   (200, 20), (500, 10), (500, 20)]:
        s = simulate(n, t, 0.0, reps, SEED + 3)
        rows.append({"n": n, "t": t, "size": s["size"],
                     "h_mean": s["h_mean"], "sd_fe": s["sd_fe"],
                     "sd_re": s["sd_re"], "valid_frac": s["valid_frac"]})
    return {"reps": reps, "target": CONF, "chi95": CHI95, "rows": rows}


def exp_power():
    """Hausman power as endogeneity delta grows."""
    reps = 800
    rows = []
    for delta in [0.0, 0.05, 0.1, 0.2, 0.4, 0.6]:
        for (n, t) in [(100, 10), (300, 10)]:
            s = simulate(n, t, delta, reps, SEED + 4)
            rows.append({"delta": delta, "n": n, "t": t,
                         "power": s["size"], "bias_re": s["bias_re"],
                         "bias_fe": s["bias_fe"]})
    return {"reps": reps, "rows": rows}


def exp_robust_size():
    """Naive Hausman under heteroskedasticity / serial correlation.

    Naive variance difference uses a single pooled s2e. For a robust check we
    build the Wald statistic on d = b_FE - b_RE using the Monte-Carlo sampling
    variance of d under H0 (a benchmark that a cluster-robust estimator would
    target). This exposes how badly the naive chi2 size distorts.
    """
    reps = 1200
    n, t = 200, 10
    out = {}
    for scenario, kw in [("iid", {}), ("hetero", {"hetero": True}),
                         ("serial", {"serial": True})]:
        rng = np.random.default_rng(SEED + 5)
        ds = np.empty(reps); hs = np.empty(reps); ok = np.empty(reps, dtype=bool)
        for r in range(reps):
            x, y, _ = draw_panel(rng, n, t, 0.0, **kw)
            e = estimate(x, y, t)
            ds[r] = e["b_fe"] - e["b_re"]
            hs[r] = e["h"]; ok[r] = np.isfinite(e["h"])
        sd_d = float(np.std(ds, ddof=1))
        wald = (ds / sd_d) ** 2
        out[scenario] = {
            "naive_size": float(np.mean(hs[ok] > CHI95)),
            "wald_sd_d": sd_d,
            "wald_size": float(np.mean(wald > CHI95)),
            "h_mean": float(np.nanmean(hs)),
        }
    return {"reps": reps, "n": n, "t": t, "scenarios": out}


# ---------------------------------------------------------------- plots
def plot1(r):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    xs = [q["delta"] for q in r["rows"]]
    ax[0].plot(xs, [q["bias_fe"] for q in r["rows"]], "o-", color="#2980b9", label="FE")
    ax[0].plot(xs, [q["bias_re"] for q in r["rows"]], "s-", color="#c0392b", label="RE")
    ax[0].axhline(0, color="grey", lw=0.6); ax[0].set_xlabel("endogeneity delta")
    ax[0].set_ylabel("bias in beta"); ax[0].set_title("RE bias grows, FE stays unbiased")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(xs, [q["mse_fe"] for q in r["rows"]], "o-", color="#2980b9", label="FE MSE")
    ax[1].plot(xs, [q["mse_re"] for q in r["rows"]], "s-", color="#c0392b", label="RE MSE")
    ax[1].set_xlabel("endogeneity delta"); ax[1].set_ylabel("MSE of beta")
    ax[1].set_title("RE wins at delta=0, loses as endogeneity rises")
    ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig1_bias_mse.png"), dpi=120)
    plt.close(fig); return "fig1_bias_mse.png"


def plot2(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot([rr["t"] for rr in r["rows"]], [rr["var_ratio"] for rr in r["rows"]],
            "o-", color="#2980b9", label="Var(FE)/Var(RE)")
    ax.axhline(1, color="grey", ls="--", lw=1, label="equal efficiency")
    ax.set_xlabel("time periods T"); ax.set_ylabel("Var(FE) / Var(RE)")
    ax.set_title("With only time-varying x, RE efficiency gain is tiny (~1%)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig2_efficiency.png"), dpi=120)
    plt.close(fig); return "fig2_efficiency.png"


def plot3(r):
    fig, ax = plt.subplots(figsize=(9, 5))
    tvals = sorted({q["t"] for q in r["rows"]})
    for t in tvals:
        pts = [q for q in r["rows"] if q["t"] == t]
        ax.plot([q["n"] for q in pts], [q["size"] for q in pts], "o-", label=f"T={t}")
    ax.axhline(CONF, color="grey", ls="--", lw=1, label="5% target")
    ax.set_xlabel("individuals N"); ax.set_ylabel("Hausman rejection rate under H0")
    ax.set_title("Hausman size: small T over-rejects")
    ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 0.35)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig3_size.png"), dpi=120)
    plt.close(fig); return "fig3_size.png"


def plot4(r):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (n, t), c in [((100, 10), "#2980b9"), ((300, 10), "#c0392b")]:
        pts = [q for q in r["rows"] if q["n"] == n and q["t"] == t]
        ax.plot([q["delta"] for q in pts], [q["power"] for q in pts], "o-",
                color=c, label=f"N={n},T={t}")
    ax.axhline(CONF, color="grey", ls="--", lw=1, label="5% size")
    ax.set_xlabel("endogeneity delta"); ax.set_ylabel("Hausman rejection rate")
    ax.set_title("Hausman power rises with endogeneity and sample size")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig4_power.png"), dpi=120)
    plt.close(fig); return "fig4_power.png"


def plot5(r):
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(r["scenarios"].keys())
    naive = [r["scenarios"][k]["naive_size"] for k in names]
    wald = [r["scenarios"][k]["wald_size"] for k in names]
    x = np.arange(len(names)); w = 0.35
    ax.bar(x - w / 2, naive, w, label="naive chi2 Hausman", color="#c0392b")
    ax.bar(x + w / 2, wald, w, label="variance-calibrated Wald", color="#16a085")
    ax.axhline(CONF, color="grey", ls="--", lw=1, label="5% target")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylim(0, 0.2)
    ax.set_ylabel("rejection rate under H0")
    ax.set_title("Static panel: variance misspecification barely moves Hausman size")
    ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(_OUT, "fig5_robust.png"), dpi=120)
    plt.close(fig); return "fig5_robust.png"


def main():
    e1 = exp_bias_mse()
    e2 = exp_efficiency()
    e3 = exp_size()
    e4 = exp_power()
    e5 = exp_robust_size()
    results = {
        "setup": {"seed": SEED, "beta": BETA, "conf": CONF, "chi95": CHI95},
        "bias_mse": e1, "efficiency": e2, "size": e3,
        "power": e4, "robust": e5,
    }
    results["figs"] = [plot1(e1), plot2(e2), plot3(e3), plot4(e4), plot5(e5)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 九、小结

固定效应与随机效应之争，本质是一场效率与一致性的交易。Hausman 检验帮我们判断这场交易值不值得做，但它真正的价值在于揭示内生性，而非挑剔方差设定。当解释变量全是时变的，RE 的效率优势小到可以忽略，FE 几乎是免费的一致性保险。看懂了这一点，面板回归的模型选择就不再是玄学，而是一道有明确答案的计算题。
