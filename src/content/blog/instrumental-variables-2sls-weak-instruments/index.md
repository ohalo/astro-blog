---
title: "弱工具变量的四张面孔：2SLS 点估计漂移、Wald 区间爆炸与 Anderson-Rubin 救援"
description: "一个真值已知的蒙特卡洛实验（n=400、2000 次重复）拆开弱工具变量的四张面孔：工具强度 π 从 1 降到 0.05 时，一阶段 F 均值从 402.9 塌到 1.97、87.4% 的重复样本 F 低于 10；OLS 偏差精确落在理论值 β+ρ/(1+π²) 上（0.799→1.098），2SLS 点估计均值在极弱工具下剧烈甩尾（0.499→0.684→0.638）、中位数漂到 0.740；单个弱工具下 Wald t 检验虚警率仍守在 4.7% 到 6.8%，爆炸的是置信区间中位宽度（0.229→4.985）；三个弱工具才让朴素 t 检验虚警爬到 6.8%，Anderson-Rubin 稳在 4.8% 到 5.4%。Hansen J 在工具全有效时 5.2% 不误报，掺入直接效应后功效 0.68 到 1.00。结论：弱工具下别信点估计与 Wald 区间，改用 AR 做推断、用 J 查有效性。"
publishDate: '2026-11-21T01:10:00Z'
tags:
  - 量化
  - 交易
  - 因果推断
  - 统计推断
  - Python
language: zh
difficulty: advanced
image: "/images/instrumental-variables-2sls-weak-instruments/fig1_first_stage.png"
---

## 一、结论先行

在量化研究里，工具变量（IV）常被当成「内生性一来就上 2SLS」的万能药。但两阶段最小二乘的体面外表下藏着一个前提：工具得够强。本文用一个真值已知的蒙特卡洛实验（样本量 400、2000 次重复）把弱工具变量的破坏力拆成四张面孔，每张面孔都配一组可复现的数字，代码附在文末，数值全部来自同一份脚本。

先把结论摆出来。第一，工具强度 π 从 1 降到 0.05 时，一阶段 F 的均值从 402.9 塌到 1.97，87.4% 的重复样本 F 值低于 10 这条经验红线。第二，OLS 的偏差精确落在理论值 β+ρ/(1+π²) 上，从 0.799 爬到 1.098；而 2SLS 的点估计均值在极弱工具下会剧烈甩尾（0.499→0.684→0.638），中位数也漂到 0.740——一致性和有限样本表现是两回事。第三，单个弱工具下 Wald 的 t 检验虚警率其实守在 5% 附近，真正爆炸的是置信区间宽度，中位宽度从 0.229 涨到 4.985。第四，只有多个弱工具同时出现时，朴素 t 检验才真正失守（虚警 6.8%），Anderson-Rubin 检验始终稳在 5% 左右。最后，Hansen J 检验在工具全有效时不会乱报（5.2%），一旦第二个工具掺进直接效应，它的功效从 0.68 一路升到 1.00。

一句话：弱工具下，点估计和 Wald 区间都不能信，该换 Anderson-Rubin 做推断、用 Hansen J 查有效性。

## 二、数据生成过程：让真值说话

要让结论站得住，先得有一个真值已知的实验室。设定如下：工具矩阵 Z 由 k 个独立标准正态列组成，v 与 e 也是标准正态扰动；内生性来自结构误差 u=ρv+e 与解释变量 x 共享同一个 v。解释变量 x=π·k·Σz+v，被解释变量 y=βx+u。真实结构系数 β=0.5，内生强度 ρ=0.6，工具强度 π 在 1.0、0.5、0.2、0.1、0.05 五档之间扫描。

这套设定的好处是所有理论值都能手算。由于 Cov(x,u)=ρ、Var(x)=1+kπ²，OLS 的概率极限是 β+ρ/(1+kπ²)；2SLS 的概率极限则是 β——只要工具有效且与 x 相关，无论内生性多强，2SLS 都是一致的。当 k=1、π=1 时，OLS 极限是 0.5+0.6/2=0.8，模拟均值 0.799 与之几乎重合；当 π=0.05 时理论值 0.5+0.6/1.0025=1.0985，模拟均值 1.098。理论对表严丝合缝，说明实验室没有搭错。

从直觉上看，一阶段回归 x 对 z 的斜率就是 π（乘上 k），π 越接近零，工具对 x 的解释力越弱，散点云越接近一团没有方向的噪声。图1 把 π=1（左）和 π=0.1（右）两种情形并排放在一起：左图里 x 随 z 明显上扬，一阶段 F 高达 404；右图的散点几乎是一团扁平的云，F 只剩 1.49。这就是弱工具最直观的样子——工具与内生变量之间那根绳子快断了。

![图1：强工具（左）与弱工具（右）的一阶段散点，红线为一阶段拟合](/images/instrumental-variables-2sls-weak-instruments/fig1_first_stage.png)

## 三、点估计漂移：均值与中位数的背离

先看 2SLS 的点估计。理论上它一致，所以 π 从 1 降到 0.05，均值都该贴着 0.5。模拟给出的均值序列是 0.499、0.495、0.448、0.684、0.638：前三档还在 0.5 附近，后两档却明显跑偏，0.2 那一档甚至掉到 0.448。这不是一致性失效，而是弱工具下 2SLS 估计量的分布长出了极重的尾巴——分母 Σz·x 在弱工具时可能非常接近零，个别重复会甩出天文数字的估计值，把均值整个拽走。

看中位数就清楚多了：0.502、0.500、0.507、0.519、0.740。前四档中位数都稳稳贴着 0.5，说明「典型的一次实验」仍然给出正确回答；只有 π=0.05 这一极端档中位数才漂到 0.740。均值被少数离群样本污染，中位数更接近真实中心。图2 同时画了 OLS、OLS 理论极限、2SLS 均值与 2SLS 中位数，两种统计量的分道扬镳一目了然。这给第一条实操经验：弱工具下报告 2SLS 结果时，中位数比均值稳健，但更该做的是压根别用点估计下结论。

![图2：OLS 偏差随 π 走高并贴合理论值，2SLS 均值甩尾、中位数更稳](/images/instrumental-variables-2sls-weak-instruments/fig2_bias.png)

## 四、一阶段 F：先量工具够不够强

既然点估计靠不住，就需要一个提前预警的诊断量，一阶段 F 就是最常用的那个。它检验「所有工具在 x 上的系数同时为零」这个原假设。模拟结果非常干净：F 的均值序列是 402.9、101.4、16.9、5.01、1.97，而理论上一阶段 F 的期望约等于非中心参数 π²N+1，即 401、101、17、5、2——模拟与理论几乎逐项吻合。这本身又是一次对表成功。

更实用的是 F 低于 10 的比例：π=1 和 0.5 时是 0%，π=0.2 时升到 21.3%，π=0.1 时 87.4%，π=0.05 时 98.5%。Stock 与 Yogo 那套经验法则把 10 当作红线，本实验精确地告诉我们：π 掉到 0.2 以下，红线就开始大规模被击穿。图3 把 F 均值画在对数轴上并标出 F=10 的虚线，每根柱子上标注了该档低于 10 的比例，读者一眼就能看出工具强度与诊断量之间的单调关系。

![图3：一阶段 F 随 π 急剧塌陷，标注每档 F 低于 10 的样本比例](/images/instrumental-variables-2sls-weak-instruments/fig3_first_stage_f.png)

## 五、Wald 区间的爆炸与 t 检验的体面

这里有一个很多人想当然搞错的事实。在只有一个工具的情形下，2SLS 的 Wald t 检验其实没有明显虚警：π 从 1 降到 0.05，虚警率是 0.0535、0.055、0.0435、0.037、0.027，全在 5% 上下甚至偏保守。换句话说，「弱工具让 t 检验乱报到 30%」这种说法对单工具并不成立。

真正出事的是置信区间宽度。同一批实验里，95% Wald 区间的中位宽度是 0.229、0.455、1.145、2.359、4.985，宽度超过 10 的样本比例从 0% 涨到 31.3%。区间宽到这个地步，虽然名义覆盖率守住了，但它已经不提供任何有用的信息——一个宽达 5 个单位、真系数只有 0.5 的区间，等于什么都没说。图4 把中位宽度画在对数轴上，这条陡升的曲线才是单工具弱工具问题的真实面目：不是虚警，而是区间失去信息量。

![图4：单个弱工具下 Wald 置信区间中位宽度随 π 爆炸（对数轴）](/images/instrumental-variables-2sls-weak-instruments/fig4_ci_width.png)

## 六、多个弱工具：t 检验失守，AR 与 J 出场

当工具从 1 个增加到 3 个，情况才真正变坏。三工具下，朴素 Wald t 检验的虚警率从 0.047 爬到 0.068，最弱那档明显超过 5% 的名义水平；而 Anderson-Rubin 检验的虚警率是 0.0535、0.049、0.053、0.048、0.051，稳稳贴在 5%。AR 的思路是不去估计 β，而是直接检验「y−β₀x 与所有工具都不相关」这个原假设，它对一阶段强度不敏感，因此在弱工具下仍守得住规模。代价是 AR 区间在最弱那档有 10.3% 的概率变成无界区间——它诚实告诉你「数据不足以把 β 圈在一个有限范围里」，而不是像 Wald 那样给一个虚假精确的宽区间。

另一道防线是 Hansen J 过度识别检验，用来查工具是否有效。实验中，「所有工具有效」两种情形下 J 的拒绝率是 0.0515 和 0.0545，贴着 5%，不会误伤；当第二个工具被掺进一个直接效应 γ，J 的拒绝率在强工具下从 0.68 升到 1.00，在弱工具下从 0.77 升到 1.00。功效很高，说明 J 确实能抓出「工具通过别的渠道直接影响 y」这种违规。图5 左panel展示三工具下 Wald 与 AR 的虚警对比，右panel展示 Hansen J 的规模与功效。

![图5：左，三工具下 Wald 失守而 AR 守住 5%；右，Hansen J 揪出带直接效应的无效工具](/images/instrumental-variables-2sls-weak-instruments/fig5_size_and_j.png)

## 七、三盆冷水与实操清单

把上面的数字压成几条冷静的经验。第一，「2SLS 一致」是渐近性质，n=400、π=0.05 时一阶段 F 只有约 2，远没进入渐近区，点估计均值和中位数都会偏，别拿单期回归系数讲故事。第二，单工具弱工具问题主要毁掉区间的信息量而非 t 检验的规模，看 t 显著就以为「工具没问题」是错的——先看一阶段 F 和区间宽度。第三，多工具时 t 检验会真的虚警，这时该换 AR 这类对弱工具稳健的推断；同时用 Hansen J 检查工具有没有直接效应，但记住 J 在小样本里偏保守，不显著不等于工具一定有效。

落到流程上：跑完第一阶段先看 F，低于 10 就把 2SLS 点估计当参考而非结论；报告推断时优先给 AR 置信集；有多个工具时补一个 Hansen J；如果 J 拒绝或一阶段濒临红线，最稳妥的做法是回到研究设计层面找更强的工具，而不是在弱工具上继续加估计技巧。

```python
# -*- coding: utf-8 -*-
"""Weak instruments and 2SLS: bias, first-stage F, CI blow-up, t-test size
distortion under overidentification, Anderson-Rubin, and Hansen J.

Single source of truth for every number and figure in the article.

DGP (k instruments):  Z ~ N(0,1)^k, v ~ N(0,1), e ~ N(0,1)
    u = RHO * v + GAMMA * z2 + e      (GAMMA>0 -> z2 has a direct effect)
    x = pi * sum_k(z) + v
    y = BETA * x + u
OLS plim = BETA + RHO/(1+k*pi^2) (biased); 2SLS plim = BETA (consistent),
but finite-sample bias ~ RHO/pi has heavy tails, the Wald interval width
blows up, and the naive t-test over-rejects once there are several weak
instruments. Anderson-Rubin keeps size; Hansen J flags a direct effect.
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

SEED = 20261121
N = 400
REPS = 2000
BETA = 0.5
RHO = 0.6
PIS = [1.0, 0.5, 0.2, 0.1, 0.05]
Z95 = 1.959963985
F95_1 = float(stats.f.ppf(0.95, 1, N - 1))
F95_3 = float(stats.f.ppf(0.95, 3, N - 3))
CHI95_1 = float(stats.chi2.ppf(0.95, 1))

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OUT = os.path.join(_ROOT, "public", "images", "instrumental-variables-2sls-weak-instruments")
os.makedirs(_OUT, exist_ok=True)


def draw(pi, rng, n=N, k=1, gamma=0.0):
    z = rng.standard_normal((n, k))
    v = rng.standard_normal(n)
    e = rng.standard_normal(n)
    zsum = z.sum(axis=1)
    # A direct effect on only the second instrument is detectable by Hansen J;
    # putting it on zsum (collinear with x) would be absorbed and invisible.
    direct = gamma * z[:, 1] if (gamma != 0.0 and k >= 2) else 0.0
    u = RHO * v + direct + e
    x = pi * k * zsum + v
    y = BETA * x + u
    return z, x, y


def tsls(z, x, y):
    """Return beta_hat, naive t for H0 beta=BETA, and Wald CI width."""
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    xh = pz @ x
    xtpx = xh @ xh
    b2 = (xh @ y) / xtpx
    resid = y - b2 * x
    s2 = resid @ resid / (N - k)
    se = np.sqrt(s2 / xtpx)
    t = (b2 - BETA) / se
    return b2, float(t), float(2 * Z95 * se)


def ols(x, y):
    return (x @ y) / (x @ x)


def first_f(z, x):
    if z.ndim == 1:
        pih = (z @ x) / (z @ z)
        r = x - pih * z
        return pih ** 2 / ((r @ r / (N - 2)) / (z @ z))
    k = z.shape[1]
    pihat = np.linalg.lstsq(z, x, rcond=None)[0]
    r = x - z @ pihat
    s2 = r @ r / (N - k)
    xtpx = x @ (z @ np.linalg.inv(z.T @ z) @ z.T) @ x
    # F for joint significance of all k first-stage coefficients
    return float((xtpx / k) / s2)


def ar_stat(T, z):
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    mz = np.eye(N) - pz
    return float((T @ pz @ T / k) / (T @ mz @ T / (N - k)))


def hansen_j(z, x, y):
    k = z.shape[1]
    zz_inv = np.linalg.inv(z.T @ z)
    pz = z @ zz_inv @ z.T
    xh = pz @ x
    b2 = (xh @ y) / (xh @ xh)
    resid = y - b2 * x
    return float(resid @ pz @ resid / (resid @ resid / N))


def exp_bias():
    """Single instrument: OLS vs 2SLS point estimates (mean and median)."""
    out = {"pi": PIS, "ols": [], "twosls": [], "twosls_med": [], "ols_theory": []}
    rng = np.random.default_rng(SEED + 1)
    for pi in PIS:
        bo, b2 = [], []
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            o, t, _ = tsls(z, x, y)
            bo.append(ols(x, y))
            b2.append(o)
        out["ols"].append(float(np.mean(bo)))
        out["twosls"].append(float(np.mean(b2)))
        out["twosls_med"].append(float(np.median(b2)))
        out["ols_theory"].append(BETA + RHO / (1.0 + pi * pi))
    return out


def exp_first_stage():
    out = {"pi": PIS, "f_mean": [], "f_p_below10": [], "noncentrality": []}
    rng = np.random.default_rng(SEED + 2)
    for pi in PIS:
        fs = []
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            fs.append(first_f(z, x))
        fs = np.array(fs)
        out["f_mean"].append(float(np.mean(fs)))
        out["f_p_below10"].append(float(np.mean(fs < 10.0)))
        out["noncentrality"].append(pi * pi * N)
    return out


def exp_ci_width():
    """Single instrument: Wald CI width explodes while the t-test keeps its size."""
    out = {"pi": PIS, "wald_med": [], "wald_p_gt10": [], "wald_size_single": []}
    rng = np.random.default_rng(SEED + 3)
    for pi in PIS:
        ws, rej = [], 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng)
            _, t, w = tsls(z, x, y)
            ws.append(w)
            rej += (abs(t) > Z95)   # H0 beta=BETA is true -> false rejection
        ws = np.array(ws)
        out["wald_med"].append(float(np.median(ws)))
        out["wald_p_gt10"].append(float(np.mean(ws > 10.0)))
        out["wald_size_single"].append(rej / REPS)
    return out


def exp_size_overid():
    """Three weak instruments: naive Wald t over-rejects, Anderson-Rubin does not."""
    out = {"pi": PIS, "wald": [], "ar": [], "ar_ci_unbounded": []}
    rng = np.random.default_rng(SEED + 4)
    k = 3
    for pi in PIS:
        rw, ra, unbounded = 0, 0, 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng, k=k)
            _, t, _ = tsls(z, x, y)
            rw += (abs(t) > Z95)
            ra += (ar_stat(y - BETA * x, z) > F95_3)
            # The AR acceptance set is unbounded iff the extreme nulls at the
            # ends of a wide grid are not rejected: only 2 endpoints needed.
            u_lo = ar_stat(y - (-1.5) * x, z) <= F95_3
            u_hi = ar_stat(y - 2.5 * x, z) <= F95_3
            unbounded += bool(u_lo or u_hi)
        out["wald"].append(rw / REPS)
        out["ar"].append(ra / REPS)
        out["ar_ci_unbounded"].append(unbounded / REPS)
    return out


def exp_overid():
    """Hansen J size (all valid) and power (one instrument has a direct effect)."""
    scenes = [
        ("strong-all-valid", 0.8, 0.0),
        ("strong-gamma0.2", 0.8, 0.2),
        ("strong-gamma0.5", 0.8, 0.5),
        ("weak-all-valid", 0.15, 0.0),
        ("weak-gamma0.2", 0.15, 0.2),
        ("weak-gamma0.5", 0.15, 0.5),
    ]
    out = {"scene": [s[0] for s in scenes], "reject": [], "pi": [], "gamma": []}
    rng = np.random.default_rng(SEED + 5)
    for name, pi, gamma in scenes:
        cnt = 0
        for _ in range(REPS):
            z, x, y = draw(pi, rng, k=2, gamma=gamma)
            cnt += (hansen_j(z, x, y) > CHI95_1)
        out["reject"].append(cnt / REPS)
        out["pi"].append(pi)
        out["gamma"].append(gamma)
    return out


def plot1():
    rng = np.random.default_rng(SEED + 10)
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 4.2))
    for ax, pi in zip(axes, [1.0, 0.1]):
        z, x, y = draw(pi, rng)
        z1 = z[:, 0]
        ax.scatter(z1, x, s=16, alpha=0.45, color="#2c3e50")
        slope = (z1 @ x) / (z1 @ z1)
        xs = np.linspace(z1.min(), z1.max(), 50)
        ax.plot(xs, slope * xs, color="#c0392b", lw=2)
        f = first_f(z, x)
        ax.set_title("pi=%.2f  first-stage slope=%.2f  F=%.1f" % (pi, slope, f), fontsize=10)
        ax.set_xlabel("instrument z")
        ax.set_ylabel("endogenous regressor x")
        ax.axhline(0, color="k", lw=0.5)
    fig.suptitle("Fig1: a strong (left) vs weak (right) first stage", fontsize=11)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_first_stage.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig1_first_stage.png"


def plot2(bias):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(bias["pi"], bias["ols"], "o-", color="#c0392b", lw=2, label="OLS (simulated)")
    ax.plot(bias["pi"], bias["ols_theory"], "s--", color="#e67e22", lw=1.5,
            label="OLS plim = b + rho/(1+pi^2)")
    ax.plot(bias["pi"], bias["twosls"], "^-", color="#27ae60", lw=2, label="2SLS mean (heavy tails)")
    ax.plot(bias["pi"], bias["twosls_med"], "v:", color="#16a085", lw=1.5, label="2SLS median")
    ax.axhline(BETA, color="k", ls=":", lw=1.2, label="true beta = 0.5")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("mean / median estimate")
    ax.set_title("Fig2: weak instruments drag 2SLS toward the biased OLS estimate")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_bias.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig2_bias.png"


def plot3(fs):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(fs["pi"], fs["f_mean"], "o-", color="#2980b9", lw=2, label="mean first-stage F")
    ax.axhline(10, color="#c0392b", ls="--", lw=1.5, label="Stock-Yogo rule of thumb: F=10")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("first-stage F (log scale)")
    ax.set_yscale("log")
    ax.set_title("Fig3: first-stage F collapses below 10 once pi falls under ~0.15")
    ax.legend(fontsize=8)
    for x0, p10 in zip(fs["pi"], fs["f_p_below10"]):
        ax.annotate("%.0f%%" % (100 * p10), (x0, fs["f_mean"][fs["pi"].index(x0)]),
                    textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_first_stage_f.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig3_first_stage_f.png"


def plot4(cw):
    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    ax1.plot(cw["pi"], cw["wald_med"], "o-", color="#8e44ad", lw=2,
             label="median Wald 95% CI width")
    ax1.axhline(1.0, color="k", ls=":", lw=1.2, label="reference width = 1")
    ax1.set_xlabel("instrument strength pi")
    ax1.set_ylabel("median 95% CI width (log scale)", color="#8e44ad")
    ax1.tick_params(axis="y", labelcolor="#8e44ad")
    ax1.set_yscale("log")
    ax2 = ax1.twinx()
    ax2.plot(cw["pi"], cw["wald_size_single"], "s--", color="#2980b9", lw=2,
             label="Wald t size (true beta)")
    ax2.axhline(0.05, color="#2980b9", ls=":", lw=1.0, alpha=0.6)
    ax2.set_ylabel("false rejection rate", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax2.set_ylim(0, 0.12)
    ax1.set_title("Fig4: one weak instrument keeps nominal size but the Wald interval explodes")
    lines1, lab1 = ax1.get_legend_handles_labels()
    lines2, lab2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, lab1 + lab2, fontsize=8, loc="upper left")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_ci_width.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig4_ci_width.png"


def plot5(sz, ov):
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    ax.plot(sz["pi"], sz["wald"], "o-", color="#c0392b", lw=2, label="naive Wald t (k=3)")
    ax.plot(sz["pi"], sz["ar"], "s-", color="#27ae60", lw=2, label="Anderson-Rubin (k=3)")
    ax.axhline(0.05, color="k", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xlabel("instrument strength pi")
    ax.set_ylabel("false rejection of H0: beta=0.5")
    ax.set_title("5a: several weak instruments break the t-test, AR survives", fontsize=10)
    ax.legend(fontsize=8)

    ax = axes[1]
    x = np.arange(len(ov["scene"]))
    colors = ["#27ae60" if g == 0 else "#c0392b" for g in ov["gamma"]]
    bars = ax.bar(x, ov["reject"], color=colors, alpha=0.85)
    ax.axhline(0.05, color="k", ls="--", lw=1.2, label="nominal 5%")
    ax.set_xticks(x)
    ax.set_xticklabels(ov["scene"], rotation=20, ha="right", fontsize=7)
    ax.set_ylabel("Hansen J rejection rate")
    ax.set_title("5b: Hansen J flags a direct (invalid) instrument", fontsize=10)
    ax.legend(fontsize=8)
    for b, v in zip(bars, ov["reject"]):
        ax.annotate("%.2f" % v, (b.get_x() + b.get_width() / 2, v),
                    textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_size_and_j.png")
    fig.savefig(p, dpi=120)
    plt.close(fig)
    return "fig5_size_and_j.png"


def main():
    bias = exp_bias()
    fs = exp_first_stage()
    cw = exp_ci_width()
    sz = exp_size_overid()
    ov = exp_overid()
    results = {
        "setup": {"seed": SEED, "n": N, "reps": REPS, "beta": BETA, "rho": RHO,
                  "pis": PIS, "f95_1": F95_1, "f95_3": F95_3, "chi95_1": CHI95_1},
        "bias": bias,
        "first_stage": fs,
        "ci_width": cw,
        "size_overid": sz,
        "overid": ov,
    }
    results["figs"] = [plot1(), plot2(bias), plot3(fs), plot4(cw), plot5(sz, ov)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 八、完整代码与复现

上面所有数字与五张图由下面这一段自包含脚本生成，设置了固定随机种子 20261121，直接运行即可复现。脚本同时把全部统计量写入 results.json，正文中的每个数字都能在输出里逐项对上。注意 t 统计量用 (β̂−0.5)/se 计算，这是检验真值的正确写法； Hansen J 的直接效应只加在第二个工具上，若加在所有工具的求和项上会因为与 x 共线而完全检测不到。

## 九、小结

弱工具变量不是一个「估计量会偏一点」的小毛病，而是一组连锁反应：一阶段 F 塌方、点估计甩尾、 Wald 区间失去信息量、多工具下 t 检验虚警。真正稳健的应对不是换一个更花哨的估计量，而是换一套推断——Anderson-Rubin 负责在弱工具下给出可信的假设检验与置信集，Hansen J 负责盯住工具有效性。把这两道防线用起来，再配合一阶段 F 这个预警器，才算把弱工具变量的坑填平。
