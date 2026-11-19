---
title: "局部投影脉冲响应：VAR 递推设定对了省 6 倍样本，错一个 MA 项置信区间直接归零"
description: "想知道一次冲击对未来收益的影响路径，多数人第一反应是估 VAR 再递推脉冲响应。本文在真值已知的蒙特卡洛上把 VAR 递推与 Jordà 局部投影正面拆开：AR(1) 设定正确时，VAR 在 h=5 的 RMSE 只有 0.0106，是 LP 的 0.0622 的六分之一，置信区间窄到 0.0414 对 0.2456（省约 6 倍样本）；但真过程换成 ARMA(1,1)、模型漏掉 MA 项后，VAR 在 h=1 的响应从真值 1.10 低估到 0.7527（偏差 -0.3474），95% 置信区间覆盖率直接崩到 0.0000，等于用漂亮的窄区间系统性地讲错故事；LP 在同样误设下偏差仅 -0.0038、覆盖率仍有 0.9483。样本量扫描给出最反直觉的交叉：设定正确时 VAR/LP 的 RMSE 比稳定在 0.14，误设时却从 n=100 的 0.55 单调升到 n=1600 的 2.11——数据越多，VAR 只是越自信地错下去。附完整可运行代码与五张真实计算图。"
publishDate: '2026-11-19T01:10:00Z'
tags:
  - 量化
  - 交易
  - 计量经济学
  - 脉冲响应
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

评估一次政策冲击、一笔异常订单流、一次流动性枯竭对未来收益的影响路径，是量化研究里最高频的动作之一。两条主流路线：**先估一个向量自回归 VAR，再从系数矩阵递推脉冲响应函数（IRF）**；或者按 Jordà（2005）的**局部投影（Local Projection, LP）**，对每个 horizon 单独跑一次回归。

教科书偏爱前者，因为它有效率；实务里越来越多人转向后者，因为它稳健。到底谁更好？本文用真值已知的模拟把这件事钉死，结论有两条，而且互相打架：

1. **设定正确时 VAR 赢得干净**。AR(1) 真过程上，VAR 递推在 $h=5$ 的 RMSE 是 0.0106，LP 是 0.0622——VAR 只用六分之一左右的估计误差；95% 区间宽度 0.0414 对 0.2456，相当于同样精度下 VAR 省了约 6 倍样本。此时 LP 的多参数逐点估计就是纯粹的浪费。
2. **一旦动态设定错了，VAR 的优势瞬间反转为负债**。真过程是 ARMA(1,1)、而模型只估到 AR(1) 时，VAR 在 $h=1$ 把响应从真值 1.10 估成 0.7527（偏差 $-0.3474$），95% 区间覆盖率**塌到 0.0000**——区间又窄又自信，却几乎从不包含真值。LP 在同样条件下偏差 $-0.0038$、覆盖率 0.9483，几乎没受影响。

真正扎心的是样本量扫描：设定正确时 VAR/LP 的 RMSE 比一直在 0.14 附近，误设时却随样本量从 0.55 单调爬到 2.11。**样本越多，错误的 VAR 不是被数据纠正，而是越自信地把偏差钉死。**所以这不是"哪个方法更好"，而是"你敢不敢赌动态设定是对的"。

## 二、两套估计为什么会有分歧

先说清楚被估的东西。设结构冲击 $\varepsilon_t$ 可观测（比如政策利率意外、FOMC 措辞冲击），它对 $y_{t+h}$ 的真脉冲响应定义为：控制冲击发生前的状态后，$\varepsilon_t$ 变动一单位带来的 $y_{t+h}$ 变化。

**VAR 递推**的思路是把动态压缩成一套参数。用一个单方程代理：

$$y_t=a\,y_{t-1}+c\,\varepsilon_t+u_t,\qquad \widehat{\mathrm{IRF}}(h)=\hat c\,\hat a^{\,h}$$

只估一次 $(a,c)$，所有 horizon 的响应都由 $\hat a^h$ 这条几何衰减曲线生成。参数少、样本利用率高，这是它效率优势的来源；代价是**整条响应曲线被锁死在"单一指数衰减"这个形状上**。

**局部投影**不假设形状。对每个 $h$ 单独回归：

$$y_{t+h}=\alpha_h+\beta_h\varepsilon_t+\gamma_h y_{t-1}+\delta_h\varepsilon_{t-1}+e_{t,h}$$

$\hat\beta_h$ 就是第 $h$ 期的响应。$h=0$ 到 $H$ 各估一次，每次控制冲击前的状态（$y_{t-1},\varepsilon_{t-1}$）。它不强迫响应按照几何形式衰减，代价是每个 horizon 多估参数、且有效样本随 $h$ 缩短。

分歧点就在这：**真过程若恰是几何衰减，VAR 的形状假设免费送你效率；真过程一旦偏离（比如带移动平均项），VAR 的假设就变成枷锁，而 LP 因为它逐点自由，天然绕开。**

![图1：AR(1) 设定正确，两套估计都贴着真 IRF](/images/local-projection-impulse-response/fig1_irf_correct.png)

## 三、实验设计：把真值握在手里

要判断"谁更准"，必须先有标准答案。本文用两组真值已知的数据生成过程，参数固定为 $\phi=0.6$、$b=1.0$、$d=0.5$、噪声标准差 $\sigma_\eta=0.5$，每组 1200 次独立重复、$n=500$、horizon 到 12。

**A 组（设定正确）**：$y_t=\phi y_{t-1}+b\varepsilon_t+\eta_t$。这是纯 AR(1)，VAR 代理里的 $\hat a$ 正好对应 $\phi$，真响应 $\mathrm{IRF}(h)=b\phi^h$。VAR 的形状假设完全正确。

**B 组（误设）**：$y_t=\phi y_{t-1}+b\varepsilon_t+d\varepsilon_{t-1}+\eta_t$。多了个 MA(1) 项，真响应在 $h\ge1$ 变成 $\phi^{h-1}(\phi b+d)$。但研究者的模型仍只估 $y_t=a y_{t-1}+c\varepsilon_t$，漏掉了 $\varepsilon_{t-1}$——这是实务中最常见的动态误设之一。LP 的控制集里保留了 $\varepsilon_{t-1}$，所以不受影响。

一个关键的实现细节：必须在 $y$ 方程里放独立于冲击的噪声 $\eta_t$。如果让 $\varepsilon$ 直接等于 $y$ 的新息，VAR 回归 $y_t$ 对 $\varepsilon_t$ 会完全拟合，RMSE 归零、置信区间退化，得到的是平凡且误导的结论。加上 $\eta_t$ 后，估计才有真实的抽样误差。

## 四、设定正确：VAR 的效率优势有多大

![图2：设定正确时 VAR 的 RMSE 全面更低，全 horizon 均值仅 LP 的 0.144 倍](/images/local-projection-impulse-response/fig2_rmse_correct.png)

在 A 组上，$h=0$ 两者几乎打平（RMSE 都是 0.0223），因为第 0 期响应就是当期系数，两种估计都只估一个斜率。差距从 $h=1$ 开始拉开：VAR 的 RMSE 是 0.0211，LP 是 0.0521；到 $h=5$ 变成 0.0106 对 0.0622，VAR 约为 LP 的六分之一；到 $h=12$ 是 0.0007 对 0.0629。全 horizon 平均下来 VAR 的 RMSE 只有 LP 的 0.144 倍。

区间宽度讲同一个故事：$h=5$ 处 VAR 的 95% 区间宽 0.0414，LP 是 0.2456，差约 6 倍。换句话说，要达到 VAR 同样的精度，LP 大约要多付 6 倍的样本——这就是"逐点自由"的价签。两套估计的覆盖率都贴在名义 95% 附近（VAR 0.928–0.959，LP 0.948–0.958），说明在小样本里两者都基本诚实，只是 LP 松、VAR 紧。

## 五、误设：窄区间变成最危险的谎言

![图3：ARMA(1,1) 误设下 VAR 把响应估歪，LP 仍贴合真值](/images/local-projection-impulse-response/fig3_irf_misspec.png)

![图4：VAR(1) 漏掉 MA(1) 后响应被系统性估歪；LP 偏差近零](/images/local-projection-impulse-response/fig4_bias_misspec.png)

B 组是全文的核心。真响应在 $h=1$ 是 1.10，VAR 递推给出 0.7527，偏差 $-0.3474$——它把 $d\varepsilon_{t-1}$ 的贡献错误地塞进了衰减率 $\hat a$，于是起点就被压低。到 $h=5$，真值 0.1426 被高估成 0.2419（偏差 $+0.0994$），偏差换了符号，因为被污染的 $\hat a$ 在更长 horizon 上把曲线抬了起来。LP 这边，$h=1$ 偏差只有 $-0.0038$，$h=5$ 是 $-0.0090$，量级差了近两个数量级。

最严重的不是偏差本身，而是**置信区间彻底失信**。VAR 在 $h=1、5、12$ 的覆盖率分别是 0.0000、0.0008、0.0000——它给出的 95% 区间几乎从不覆盖真值。这不是"区间太窄"能搪塞的：区间窄且系统偏离，正是最危险的组合，因为它让你对自己的错误深信不疑。LP 同期覆盖率是 0.9483、0.9325、0.9583，误设几乎没有伤到它。

## 六、样本量交叉：数据越多，错的越自信

![图5：效率与稳健性交叉——设定越对 VAR 越省，误设越大 LP 越稳](/images/local-projection-impulse-response/fig5_efficiency_tradeoff.png)

把样本量从 100 扫到 1600，两条曲线走出完全相反的方向。设定正确那条（绿色）几乎水平，VAR/LP 的 RMSE 比稳定在 0.14，说明 VAR 的效率优势不随样本量消失，是结构性的。误设那条（紫色）从 $n=100$ 的 0.55 一路升到 $n=1600$ 的 2.11，中间在 $n\approx400$ 附近穿过 1——过了这条线，LP 反超，而且样本越大反超越多。

这个交叉的法理很清楚：误设带来的偏差不随 $n$ 衰减，而 VAR 的标准误按 $1/\sqrt n$ 收缩。样本越大，参数估得越精确，只是精确地估向一个错误的目标。LP 因为不把曲线锁死在几何形状上，偏差项小得多，样本增大时它稳步收敛到真值。**所以"多拿点数据就好了"对设定错误是无效的，甚至会让情况更糟——你只是更精确地错了。**

## 七、实战取舍与边界

落到实盘，我的建议是按可信度分层。若你有强理论或高频识别（如 FOMC 分钟级意外）能确认冲击传播近似几何衰减，VAR 递推更划算，尤其样本只有几百条时。若动态形式存疑——收益带尖峰、有成交量放大、有状态切换——宁可付 LP 的样本代价，买那份稳健。

三条必须写明的边界。第一，本文用单方程代理，多元 VAR 的 Cholesky 识别还叠加排序敏感性， LP 同样要选控制集，两者都不是自动正确的。第二，LP 的逐点回归在长 horizon 上有效样本锐减，尾部估计方差天然更大，本文 $h=12$ 的 LP 区间确实比 VAR 宽一个量级。第三，覆盖率结论依赖 1200 次模拟的抽样精度，名义 95% 附近的零点几个百分点波动不该被过度解读；真正稳健的信号是 B 组覆盖率**从 0.95 塌到 0.00**这种数量级的崩塌。

代码把所有实验、五张图与结果 JSON 放在一起，改参数即可复算。

```python
# -*- coding: utf-8 -*-
"""局部投影脉冲响应：VAR 递推 vs Jordà 逐点回归，设定正确时的效率与误设下的稳健性。

本文全部数字的唯一来源。运行本脚本会：
1. 在真值已知的 AR(1)（VAR 设定正确）与 ARMA(1,1)（VAR 漏掉 MA 项）上做蒙特卡洛；
   结构冲击 eps_t 可观测（如政策意外），另有独立噪声 eta_t，避免完全拟合的平凡情形；
2. 对每个 horizon h 用局部投影 LP 直接回归 y_{t+h} on eps_t + 冲击前控制集估计 IRF；
3. 用单方程 VAR 代理 y_t=a·y_{t-1}+c·eps_t 递推 IRF(h)=c·a^h，对比偏差/RMSE/覆盖率；
4. 扫描样本量 n，展示正确设定下两者接近、误设下 LP 反超的交叉；
5. 画 5 张图到 public/images/local-projection-impulse-response/，
   统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261119
R = 1200          # 蒙特卡洛重复数
N = 500           # 主实验样本量
H = 12            # 最大 horizon
PHI, B, D, SIG_ETA = 0.6, 1.0, 0.5, 0.5
Z975 = 1.959963984540054

_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "local-projection-impulse-response",
)
os.makedirs(_OUT, exist_ok=True)


def true_irf(h, ma=False):
    """eps_t 对 y_{t+h} 的真 IRF，控制冲击前状态。
    AR(1):  b·phi^h；ARMA(1,1): h=0 为 b，h>=1 为 phi^(h-1)·(phi·b+d)。
    """
    if not ma:
        return B * PHI ** h
    if h == 0:
        return B
    return PHI ** (h - 1) * (PHI * B + D)


def simulate(rng, n, ma=False, reps=R):
    """生成 eps(n,reps) 结构冲击、eta 独立噪声、y(n,reps) 路径。"""
    eps = rng.standard_normal((n, reps))
    eta = rng.standard_normal((n, reps)) * SIG_ETA
    y = np.empty((n, reps))
    prev = np.zeros(reps)
    for t in range(n):
        prev = PHI * prev + B * eps[t] + (D * eps[t - 1] if (ma and t >= 1) else 0.0) + eta[t]
        y[t] = prev
    return eps, eta, y


def lp_irf(eps, y, h):
    """y_{t+h}=a+beta·eps_t+gamma·y_{t-1}+delta·eps_{t-1}，beta 即 LP 的 IRF。
    eps/y 均为 (n, reps)，每个 rep 的控制集不同，批量解 (reps,4,4) 正规方程。"""
    n = len(y)
    # t = 1 .. n-1-h，保证 y_{t+h} 有效且 t-1 >= 0
    yy = y[1 + h:n]         # (T, reps) y_{t+h}
    e0 = eps[1:n - h]       # eps_t
    ym1 = y[0:n - 1 - h]     # y_{t-1}
    em1 = eps[0:n - 1 - h]   # eps_{t-1}
    T = yy.shape[0]
    # X: (T, 4, reps)
    Rn = eps.shape[1]
    X = np.stack([np.ones((T, Rn)), e0, ym1, em1], axis=1)  # (T,4,reps)
    G = np.einsum("tir,tjr->rij", X, X)          # (reps,4,4) = X'X
    Xty = np.einsum("tir,tr->ri", X, yy)         # (reps,4)
    coef = np.linalg.solve(G, Xty[:, :, None])[:, :, 0]   # (reps,4)
    beta = coef[:, 1]
    resid = yy - np.einsum("tir,ri->tr", X, coef)
    s2 = (resid * resid).sum(0) / (T - 4)
    inv_diag = np.linalg.inv(G)[:, 1, 1]
    se = np.sqrt(np.maximum(s2 * inv_diag, 0.0))
    return beta, se


def var_irf(eps, y, h):
    """VAR 代理 y_t=a·y_{t-1}+c·eps_t，IRF(h)=c·a^h；SE 用 delta method。"""
    ylag, ynow, elag = y[:-1], y[1:], eps[1:]
    s11 = (ylag * ylag).sum(0)
    s12 = (ylag * elag).sum(0)
    s22 = (elag * elag).sum(0)
    t1 = (ylag * ynow).sum(0)
    t2 = (elag * ynow).sum(0)
    det = s11 * s22 - s12 * s12
    a = (t1 * s22 - t2 * s12) / det
    c = (t2 * s11 - t1 * s12) / det
    irf = c * a ** h
    resid = ynow - a * ylag - c * elag
    s2 = (resid * resid).sum(0) / (len(ynow) - 2)
    da = c * h * a ** (h - 1) if h >= 1 else 0.0
    dc = a ** h
    var = s2 / det * (s22 * da * da - 2 * s12 * da * dc + s11 * dc * dc)
    return irf, np.sqrt(np.maximum(var, 0.0))


def run_dgp(ma, n=N, reps=R):
    """在一个 DGP 上跑全 horizon，返回各方法的偏差/RMSE/覆盖率与 CI 宽度。"""
    rng = np.random.default_rng(SEED + (11 if ma else 0))
    eps, eta, y = simulate(rng, n, ma=ma, reps=reps)
    out = {"irf_true": [], "lp_mean": [], "lp_bias": [], "lp_rmse": [],
           "lp_cov": [], "lp_ciw": [], "var_mean": [], "var_bias": [],
           "var_rmse": [], "var_cov": [], "var_ciw": []}
    for h in range(H + 1):
        tv = true_irf(h, ma=ma)
        bl, sl = lp_irf(eps, y, h)
        bv, sv = var_irf(eps, y, h)
        out["irf_true"].append(tv)
        out["lp_mean"].append(float(bl.mean()))
        out["lp_bias"].append(float(bl.mean() - tv))
        out["lp_rmse"].append(float(np.sqrt(np.mean((bl - tv) ** 2))))
        out["lp_cov"].append(float(np.mean(np.abs(bl - tv) <= Z975 * sl)))
        out["lp_ciw"].append(float(np.mean(2 * Z975 * sl)))
        out["var_mean"].append(float(bv.mean()))
        out["var_bias"].append(float(bv.mean() - tv))
        out["var_rmse"].append(float(np.sqrt(np.mean((bv - tv) ** 2))))
        out["var_cov"].append(float(np.mean(np.abs(bv - tv) <= Z975 * sv)))
        out["var_ciw"].append(float(np.mean(2 * Z975 * sv)))
    return out


def n_scan():
    """样本量扫描：h=1..H 平均 RMSE 的 VAR/LP 比率，两条 DGP 各一条曲线。"""
    rows = []
    for n in [100, 200, 400, 800, 1600]:
        rec = {"n": n}
        for ma, key in [(False, "correct"), (True, "misspec")]:
            a = run_dgp(ma, n=n, reps=400)
            m_lp = float(np.mean(a["lp_rmse"][1:]))
            m_var = float(np.mean(a["var_rmse"][1:]))
            rec[f"ratio_{key}"] = m_var / m_lp
            rec[f"lp_rmse_{key}"] = m_lp
            rec[f"var_rmse_{key}"] = m_var
        rows.append(rec)
    return rows


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot1(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(hs, a["irf_true"], "k-", lw=2.5, label="真 IRF  b·phi^h")
    ax.plot(hs, a["lp_mean"], "o-", color="#c0392b", lw=1.8, ms=5, label="LP 逐点回归均值")
    ax.plot(hs, a["var_mean"], "s--", color="#2980b9", lw=1.8, ms=5, label="VAR 递推均值")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("脉冲响应")
    ax.set_title("图1：AR(1) 设定正确，LP 与 VAR 都贴近真 IRF（n=500, 1200 次）")
    ax.legend(fontsize=9)
    return _save(fig, "fig1_irf_correct.png")


def plot2(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.plot(hs, a["var_rmse"], "s-", color="#2980b9", lw=2, label="VAR 递推 RMSE")
    ax.plot(hs, a["lp_rmse"], "o-", color="#c0392b", lw=2, label="LP 逐点 RMSE")
    ratio = float(np.mean(a["var_rmse"][1:])) / float(np.mean(a["lp_rmse"][1:]))
    ax.set_yscale("log")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("IRF 估计 RMSE（对数轴）")
    ax.set_title(f"图2：设定正确时 VAR 的 RMSE 更低，全 horizon 均值仅 LP 的 {ratio:.3f} 倍")
    ax.legend(fontsize=9)
    return _save(fig, "fig2_rmse_correct.png")


def plot3(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(hs, a["irf_true"], "k-", lw=2.5, label="真 IRF  phi^(h-1)(phi·b+d)")
    ax.plot(hs, a["lp_mean"], "o-", color="#c0392b", lw=1.8, ms=5, label="LP 逐点回归均值")
    ax.plot(hs, a["var_mean"], "s--", color="#2980b9", lw=1.8, ms=5, label="VAR(1) 递推均值（漏 MA 项）")
    ax.set_xlabel("horizon h")
    ax.set_ylabel("脉冲响应")
    ax.set_title("图3：ARMA(1,1) 误设下 VAR 把响应估歪，LP 仍贴合真值")
    ax.legend(fontsize=9)
    return _save(fig, "fig3_irf_misspec.png")


def plot4(a):
    hs = list(range(H + 1))
    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    ax.bar(np.array(hs) - 0.18, a["var_bias"], 0.36, color="#2980b9", label="VAR 偏差")
    ax.bar(np.array(hs) + 0.18, a["lp_bias"], 0.36, color="#c0392b", label="LP 偏差")
    ax.axhline(0, color="k", lw=1)
    ax.set_xlabel("horizon h")
    ax.set_ylabel("IRF 估计偏差（估计均值 − 真值）")
    ax.set_title("图4：VAR(1) 漏掉 MA(1) 后响应被系统性估歪；LP 偏差近零")
    ax.legend(fontsize=9)
    return _save(fig, "fig4_bias_misspec.png")


def plot5(rows):
    ns = [r["n"] for r in rows]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(ns, [r["ratio_correct"] for r in rows], "o-", color="#27ae60", lw=2,
            label="AR(1) 设定正确：VAR/LP RMSE 比")
    ax.plot(ns, [r["ratio_misspec"] for r in rows], "s-", color="#8e44ad", lw=2,
            label="ARMA(1,1) 误设：VAR/LP RMSE 比")
    ax.axhline(1.0, color="k", ls="--", lw=1.2)
    ax.text(ns[0], 1.03, "1 = 两者持平", fontsize=8)
    ax.set_xscale("log", base=2)
    ax.set_xticks(ns)
    ax.set_xticklabels(ns)
    ax.set_xlabel("样本量 n")
    ax.set_ylabel("VAR 平均 RMSE ÷ LP 平均 RMSE")
    ax.set_title("图5：效率与稳健性交叉——设定越对 VAR 越省，误设越大 LP 越稳")
    ax.legend(fontsize=9)
    return _save(fig, "fig5_efficiency_tradeoff.png")


def main():
    a = run_dgp(False)
    b = run_dgp(True)
    scan = n_scan()
    results = {
        "setup": {"seed": SEED, "reps": R, "n": N, "H": H, "phi": PHI,
                  "b": B, "d": D, "sig_eta": SIG_ETA, "z975": Z975},
        "dgpA_correct": a,
        "dgpB_misspec": b,
        "n_scan": scan,
    }
    results["figs"] = [
        plot1(a), plot2(a), plot3(b), plot4(b), plot5(scan),
    ]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
```

