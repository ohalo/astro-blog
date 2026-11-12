---
title: "滚动窗口不是越长越好：参数漂移下最优 107 天，而且与噪声大小无关"
description: "AR(1) 系数按三角波漂移（φ 在 0.20 至 0.70、|g|=0.00125、T=20000），滚动 OLS 做一步预测：窗口 10 天超额 MSE 0.119999，100 天降到 0.014175，400 天回升到 0.042544，抛物线细化的最优 w*=107.36，闭式理论 98.45。漂移周期从 400 扫到 3200 天，w* 对 P 的 log-log 斜率 0.675792（理论 2/3，R2=0.993400）；创新标准差从 1 翻到 2，超额损失精确变 4 倍而 w* 一动不动。突变区制下滚动 w=100 得 0.015844、RLS 最优 0.017676、调好的 Kalman 0.015659，但 q 设错 1000 倍就恶化到 0.081030 甚至 0.266800。预序在线选窗口 K=50 遗憾 +1.74%，K=1000 仍 +0.40%。三盆冷水：窗口超过一个漂移周期后 U 形会二次下行、跟踪误差不等于预测误差、事后最优恰好等于固定 w*。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-12T09:31:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 参数估计
  - Python
language: zh
difficulty: advanced
---

## 一、把「用多少历史」变成一个能算的量

滚动窗口几乎是所有在线估计的默认配置：估 beta、估均值回复速度、估协方差，都要先拍一个 60 天或 250 天。这个数字通常来自习惯，很少有人问它对应的损失是多少。问题其实可以写得很具体：参数在慢慢漂移，窗口太短则估计噪声大，太长则把已经过期的参数平均进来，两者之间必然有一个最小值。

我用一个真值全程已知的过程把它算出来。$x_t=\phi_t x_{t-1}+\varepsilon_t$，$\varepsilon_t\sim N(0,1)$，系数按三角波 $\phi_t=0.45+0.25\,\mathrm{tri}(t/800)$ 在 0.20 与 0.70 之间往返，斜率绝对值 $|g|=4A/P=0.00125$；路径长 20000 天，前 500 天烧掉。每个时点用最近 $w$ 对 $(x_{i-1},x_i)$ 做滚动最小二乘得 $\hat\phi_t$，预测 $x_{t+1}=\hat\phi_t x_t+\epsilon$，损失减去不可约的 $\varepsilon_{t+1}^2$ 后叫超额 MSE。所有对照共用同一串创新随机数。

窗口网格刻意截在 $w\le P/2=400$，因为下面的局部线性推导只在窗口远小于一个漂移周期时成立。实测是一条干净的 U：10 天 0.119999，20 天 0.05766，60 天 0.01892，100 天 0.014175 见底，130 天 0.01472，220 天 0.02292，400 天回升到 0.042544。在 $\log w$ 上对最低点附近做抛物线细化，最优 $w^\*=107.36$。

![漂移系数与三种窗口的跟踪轨迹，以及超额 MSE 的 U 形曲线](/images/rolling-window-bias-variance-drift/fig1_drift_and_u_shape.png)

推导只需两行。窗口内的 $\hat\phi$ 近似是那段区间 $\phi$ 的平均，而目标是 $\phi_{t+1}$，于是偏差约 $-g(w+1)/2$；估计方差约 $\sigma^2/(w\,\mathbb E[x^2])$，乘上预测误差里的 $x_t^2$ 后正好是 $\sigma^2/w$。合起来

$$\text{excess MSE}(w)\approx \mathbb E[x^2]\,\frac{g^2(w+1)^2}{4}+\frac{\sigma^2}{w},\qquad w^\*\approx\Big(\frac{2\sigma^2}{\mathbb E[x^2]g^2}\Big)^{1/3}.$$

实测 $\mathbb E[x^2]=1.341290$（按局部平稳方差解析算是 1.327254），代入得闭式 $w^\*=98.45$，把 $(w+1)$ 项保留做数值最小化是 98。实测 107.36 比理论高约 9%，来源是有限样本下 $\hat\phi$ 的方差比一阶近似略大、以及 $\mathbb E[x^2]$ 本身随 $\phi_t$ 变化。这个量级的误差不算推翻理论，但也不该把 98 当成可以照抄的精确配置。

把公式在三个窗口上拆开看更直观（图 5 左）：$w=20$ 时方差项 $\sigma^2/w=0.05$，偏差项只有 0.000231，损失几乎全部来自估计噪声；$w=107$ 附近偏差项 0.006111 与方差项 0.009346 处在同一量级，这正是极小的定义；$w=300$ 时偏差项涨到 0.047470，方差项缩到 0.003333，损失几乎全由滞后造成。这三行是解析分解而非从实测曲线反推：图 1 右图里实测点与理论虚线在短窗口端几乎重合，到长窗口端才逐渐分开，分开的部分正是 $\mathbb E[x^2]$ 随 $\phi_t$ 变化这一阶近似没覆盖的项。

## 二、$w^\*\propto P^{2/3}$，以及一个反直觉的噪声不变性

公式最有用的不是给出某个数，而是给出尺度律：$g=1/P$，所以 $w^\*\propto P^{2/3}$。把周期从 400 扫到 3200 天，每个周期都复用同一串创新，实测最优窗口依次是 69.36、85.26、107.36、133.85、165.30、239.73、268.41，理论值 62.05、81.18、98.45、128.81、155.88、205.31、248.35。对 $\log P$ 回归 $\log w^\*$ 得到斜率 0.675792，理论 2/3≈0.6667，R²=0.993400。偏差在长周期端变大（P=2400 时 239.73 对 205.31，高 16.7%），原因是 20000 天只装得下八个多周期，长窗口那一端的有效样本变少——这是样本设计的限制，不是公式失效。

更反直觉的是噪声。把创新标准差从 1 翻到 2、其余完全不变，超额损失曲线整体精确放大 4 倍（最低点 0.014175 → 0.056701），而 $w^\*$ 一动不动，仍是 107.356537。这不是巧合：$\mathbb E[x^2]$ 与 $\sigma^2$ 同比例放大，偏差项与方差项一起放大，两条曲线的极小点严格重合。所以「行情噪声大就该把窗口拉长」是错的；真正决定最优窗口的是**漂移速度**，不是波动幅度。想让窗口变长，只能让参数变慢。

![最优窗口对漂移周期的 2/3 尺度律，以及噪声翻倍后不动的极小点](/images/rolling-window-bias-variance-drift/fig2_scaling_noise_invariance.png)

## 三、突变区制：自适应方法也有隐藏旋钮

同步漂移是温和情形。换成硬跳变：4000 天里 $\phi$ 依次是 0.30、0.75、0.25、0.65，在第 1000、2000、3000 天突变。这时把三种估计器放在同一条路径、同一个损失口径下对拍：滚动窗口（$w=25\ldots800$）、带遗忘因子的递推最小二乘 RLS（$\lambda=0.95\ldots0.999$）、以及把 $\phi_t$ 当随机游走状态的 Kalman 滤波（观测噪声 $R=1$ 已知，过程噪声 $q$ 待定）。

滚动窗口的最优是 $w=100$，超额 MSE 0.015844；$w=25$ 恶化到 0.042884，$w=800$ 到 0.072597。RLS 的最优 $\lambda=0.99$ 只有 0.017676，**反而输给滚动窗口**。Kalman 的最优 $q=10^{-4}$ 拿到 0.015659，只比滚动窗口好 1.2%。

关键在旋钮的代价：$q$ 从 $10^{-4}$ 缩到 $10^{-7}$，损失变成 0.081030（差 5.2 倍）；放大到 0.1 则变成 0.266800（差 17 倍）。RLS 的 $\lambda$ 也一样呈 U 形，0.95 是 0.032127、0.999 是 0.066944。「自适应」不是免费的午餐，它只是把「选窗口」换成了「选 $q$ 或 $\lambda$」，而且设错的代价远大于设对能拿到的那 1.2% 收益。

还有个容易混淆的指标：突变后头 100 天里 $|\hat\phi-\phi|$ 的平均跟踪误差，滚动 $w=100$ 是 0.216945，比调好的 Kalman 0.230464 还低，RLS 是 0.252085，$q=10^{-7}$ 的 Kalman 是 0.297303。跟踪误差小不等于预测损失小——短窗口抖得厉害但反应快，两种损失衡量的不是同一个东西。

![突变区制下的参数轨迹，以及滚动、RLS、Kalman 三条调参曲线](/images/rolling-window-bias-variance-drift/fig3_breaks_adaptive.png)

## 四、在线挑窗口，也要付选择的代价

既然最优窗口依赖漂移速度，直觉是每天用过去一段时间的表现挑一个 $w$。这就是预序（prequential）选择：在时点 $t$ 只用 $t-1$ 之前已实现的损失，取过去 $K$ 天平均损失最小的 $w$，再用它预测 $x_{t+1}$；全程不看未来。路径换成 6000 天、周期仍是 800，窗口网格同样截在 400。

固定取 $w=100$（网格里最接近 $w^\*=107.36$ 的一档）的平均损失是 1.061661。用整条路径的全部损失事后挑出的最优窗口**也是 100**，乐观偏差为零——这说明在这条路径上最优值本来就是稳定的，所以后面所有遗憾都纯粹来自在线选择的噪声，而不是来自「最优窗口在变」。

$K=50$ 时平均损失 1.080179，相对各自评估区间内固定 $w$ 的遗憾 +0.018518（+1.74%）；16.8% 的交易日挑到 $w<50$，32.7% 挑到 $w>200$，合起来近一半落在极端档，选中位数 130。$K=250$ 遗憾降到 +0.50%，极端档降到 7.9% 与 12.1%；$K=1000$ 遗憾 +0.40%，$w<50$ 归零、$w>200$ 只剩 2.0%。验证窗口越长，选择噪声越小，但遗憾没有归零。

一个必须写清的口径：三档 $K$ 的评估起点不同（$K$ 越大起点越晚），所以 1.080179、1.066970、1.069137 这三个平均损失**不能直接横比**，能横比的是各自区间内算出的遗憾百分比 1.74%、0.50%、0.40%。把不同评估区间的均值排成一列说「K=250 最好」是错的。

![预序在线选窗口的遗憾与所选窗口的分布](/images/rolling-window-bias-variance-drift/fig4_prequential_selection.png)

## 五、边界、失效模式与实操

最要紧的一条边界是：U 形只在 $w\ll P$ 时成立。把网格放宽到超过一个漂移周期，实测超额 MSE 在 $w=550$ 见到 0.04779 之后掉头向下，750 天 0.03448、1000 天 0.02418。这不是白捡的更优窗口，而是窗口长到把整轮漂移平均掉之后，偏差不再随 $w$ 增长、只剩方差下降；此时估计出来的 $\phi$ 是跨区制的平均值，对任何一个当下区制都不适用。所以本文主实验把网格截在 $P/2$，看到二次下行时先检查窗口是否已经覆盖了完整周期。

其余限制也不该省：三角波漂移与 $\sigma^2$ 已知是理想化设定，实盘里漂移速度得自己估，而估出来的 $g$ 直接进 $w^\*$ 的三次根；Kalman 的 $R=1$ 是白送的真值，$R$ 也要估时 $q$ 的最优会一起漂；预序选择假设损失函数就是一步预测 MSE，若真实目标是协方差估计、组合权重或分位数损失，偏差项的形式会变，这个 107 天不能照搬；所有对照都在单变量 AR(1) 上做，多元情形下参数维度上升会同时改变方差项。

可操作的结论只有三条：先用能接受的漂移尺度估 $g$，按 $(2\sigma^2/(\mathbb E[x^2]g^2))^{1/3}$ 给一个起始量级，而不是照抄某个「业界常用」的 60 或 250；把 $w^\*$ 当区间而不是点，因为实测与闭式解本来就差 9%，长周期端能差 16.7%；在线调整窗口时预留足够长的验证长度，并把选择噪声写进成本预算。最后，别把本文的 U 形当成万能形状——它成立的前提是「参数连续缓慢漂移且窗口远小于漂移周期」，遇到跳变区制，该比较的是自适应方法连同它们的调参成本。

![偏差方差两项的量级分解，以及突变后各方法的跟踪误差](/images/rolling-window-bias-variance-drift/fig5_decomposition_tracking.png)

下面这段代码是本文全部数值与五张图的来源，自包含可直接运行，输出与站点同目录的 results.json 逐项一致。

```python
"""Optimal rolling window under parameter drift: bias-variance and a 2/3 scaling law.

This file IS the runnable code block embedded in the article
`rolling-window-bias-variance-drift`. Run it directly:

    python3 article_core_rolling_window_20261112pm.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default cwd).
Every number quoted in the article comes from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 2026111202

PHI0, AMP = 0.45, 0.25       # phi_t = PHI0 + AMP * triangle(t / P)
W_GRID = [10, 15, 20, 30, 40, 60, 80, 100, 130, 170, 220, 300, 400, 550, 750, 1000]
P_GRID = [400, 600, 800, 1200, 1600, 2400, 3200]
T_DRIFT, BURN = 20000, 500


def w_for(P):
    """Local-linear bias theory only holds while w is well inside one period."""
    return [w for w in W_GRID if w <= P / 2]


def triangle_phi(T, P, phi0=PHI0, amp=AMP):
    t = np.arange(T, dtype=float)
    phase = (t / P) % 1.0
    tri = 1.0 - 4.0 * np.abs(phase - 0.5)     # [-1, 1], |slope| = 4/P
    return phi0 + amp * tri


def simulate(eps, phi):
    """AR(1) with a time-varying coefficient, x[t] = phi[t]*x[t-1] + eps[t]."""
    T = len(phi)
    x = np.empty(T)
    x[0] = eps[0] / np.sqrt(max(1 - phi[0] ** 2, 1e-6))
    for t in range(1, T):
        x[t] = phi[t] * x[t - 1] + eps[t]
    return x


def rolling_phi(x, w):
    """phi_hat_t from the last w pairs (x[i-1], x[i]), i = t-w+1 .. t."""
    A = np.zeros(len(x)); B = np.zeros(len(x))
    A[1:] = np.cumsum(x[:-1] * x[1:])
    B[1:] = np.cumsum(x[:-1] ** 2)
    t = np.arange(max(w, BURN), len(x) - 1)
    num = A[t] - A[t - w]
    den = B[t] - B[t - w]
    return t, num / np.maximum(den, 1e-12)


def excess_curve(x, e, phi, w_grid=W_GRID):
    """mean one-step forecast MSE minus the irreducible innovation variance."""
    out = np.zeros(len(w_grid))
    for i, w in enumerate(w_grid):
        t, ph = rolling_phi(x, w)
        err = x[t + 1] - ph * x[t]
        out[i] = float(np.mean(err ** 2) - np.mean(e[t + 1] ** 2))
    return out


def refined_argmin(w_grid, curve):
    """Parabolic interpolation in log(w) around the grid minimum."""
    j = int(np.argmin(curve))
    lo, hi = max(0, j - 2), min(len(curve), j + 3)
    lx = np.log(np.asarray(w_grid[lo:hi], dtype=float))
    c = np.polyfit(lx, curve[lo:hi], 2)
    vertex = -c[1] / (2 * c[0]) if c[0] > 0 else lx[int(np.argmin(curve[lo:hi]))]
    return float(np.exp(vertex)), float(curve[j])


# ---------------- experiments ----------------
def run_all():
    S, F = {}, {}

    # --- Exp 1: triangular drift, U-shaped excess MSE, theory decomposition ---
    rng = np.random.default_rng(SEED + 1)
    e_common = rng.standard_normal(T_DRIFT)          # common random numbers
    P0 = 800
    W1 = w_for(P0)                            # w <= P/2 keeps the local-linear theory valid
    phi0 = triangle_phi(T_DRIFT, P0)
    x0 = simulate(e_common, phi0)
    curve0 = excess_curve(x0, e_common, phi0, W1)
    w_emp0, min0 = refined_argmin(W1, curve0)
    g0 = 4 * AMP / P0
    ex2 = float(np.mean(x0[BURN:] ** 2))            # E[x^2] of the drifting AR(1)
    e2_analytic = float(np.mean(1.0 / (1.0 - phi0[BURN:] ** 2)))
    w_theory_cf = float((2.0 / (ex2 * g0 ** 2)) ** (1.0 / 3.0))
    wgrid_fine = np.arange(5, 3001)
    theory_curve = ex2 * (g0 * (wgrid_fine + 1) / 2) ** 2 + 1.0 / wgrid_fine
    w_theory_full = float(wgrid_fine[int(np.argmin(theory_curve))])
    S["exp1"] = {
        "T": T_DRIFT, "period": P0, "slope_g": g0,
        "phi_min": float(phi0.min()), "phi_max": float(phi0.max()),
        "E_x2_empirical": ex2, "E_x2_analytic_local": e2_analytic,
        "w_grid": W1, "max_w_relative_to_period": W1[-1] / P0,
        "excess_mse_grid": [float(v) for v in curve0],
        "w_star_empirical": w_emp0, "excess_mse_at_empirical": min0,
        "w_star_theory_closed_form": w_theory_cf,
        "w_star_theory_full_min": w_theory_full,
        "excess_mse_w10": float(curve0[0]), "excess_mse_w_max": float(curve0[-1]),
        "theory_curve_at_empirical": float(np.interp(w_emp0, wgrid_fine, theory_curve)),
    }
    # decomposition at three windows, all measured from the same path
    S["exp1"]["decomposition"] = []
    for w in [20, int(round(w_emp0)), 300]:
        tb, phb = rolling_phi(x0, w)
        # local true coefficient implicit in the window average
        bias_proxy = float(np.sqrt(max(np.mean((x0[tb + 1] - phb * x0[tb]) ** 2) -
                                       np.mean(e_common[tb + 1] ** 2), 0)))
        S["exp1"]["decomposition"].append({
            "w": int(w),
            "bias2_term": ex2 * (g0 * (w + 1) / 2) ** 2,
            "variance_term": 1.0 / w,
            "measured_excess": bias_proxy ** 2,
        })
    # beyond one drift period the window averages a whole cycle: the U turns down
    extra_w = [w for w in W_GRID if w > P0 / 2]
    extra_c = excess_curve(x0, e_common, phi0, extra_w)
    S["exp1"]["beyond_period_w"] = extra_w
    S["exp1"]["beyond_period_excess_mse"] = [float(v) for v in extra_c]
    F["exp1"] = dict(phi=phi0, x=x0, curve=curve0, theory_curve=theory_curve,
                     wgrid_fine=wgrid_fine, w_emp=w_emp0, w_grid=W1,
                     extra_w=extra_w, extra_c=extra_c)

    # tracks of the rolling estimator for three windows (figure 1)
    F["tracks"] = {}
    for w in [30, 100, 400]:
        t, ph = rolling_phi(x0, w)
        F["tracks"][w] = (t, ph)

    # --- Exp 2: scaling law w* ~ P^(2/3) and invariance to innovation noise ---
    rows = []
    curves = {}
    for P in P_GRID:
        ph = triangle_phi(T_DRIFT, P)
        xp = simulate(e_common, ph)                  # same innovations for every P
        wp = w_for(P)
        c = excess_curve(xp, e_common, ph, wp)
        wst, mn = refined_argmin(wp, c)
        g = 4 * AMP / P
        ex2p = float(np.mean(xp[BURN:] ** 2))
        rows.append({"P": P, "g": g, "w_star": wst, "excess_min": mn,
                     "w_theory": float((2.0 / (ex2p * g ** 2)) ** (1.0 / 3.0)),
                     "E_x2": ex2p})
        curves[P] = c
    lgP = np.log([r["P"] for r in rows])
    sl, ic = np.polyfit(lgP, np.log([r["w_star"] for r in rows]), 1)
    pred = sl * lgP + ic
    r2 = 1 - float(np.sum((np.log([r["w_star"] for r in rows]) - pred) ** 2) /
                   np.sum((np.log([r["w_star"] for r in rows]) - pred.mean()) ** 2))
    S["exp2"] = {
        "rows": rows, "loglog_slope": float(sl), "loglog_r2": r2,
        "theoretical_slope": 2.0 / 3.0,
    }
    F["exp2"] = dict(rows=rows, curves=curves, slope=float(sl), r2=r2)
    # noise scale: sigma = 1 vs 2 at the reference period
    x2 = simulate(2.0 * e_common, phi0)
    c2 = excess_curve(x2, 2.0 * e_common, phi0, W1)
    w2, m2 = refined_argmin(W1, c2)
    S["exp2"]["noise_invariance"] = {
        "sigma1_w_star": w_emp0, "sigma2_w_star": w2,
        "ratio": w2 / w_emp0,
        "sigma1_excess_min": min0, "sigma2_excess_min": m2,
        "excess_ratio": m2 / min0,
        "sigma1_curve": [float(v) for v in curve0],
        "sigma2_curve": [float(v) for v in c2],
    }
    F["exp2"]["curve2"] = c2
    F["exp2"]["w2"] = w2

    # --- Exp 3: abrupt breaks, adaptive estimators and their hidden tuning ---
    T3 = 4000
    rng3 = np.random.default_rng(SEED + 3)
    e3 = rng3.standard_normal(T3)
    phi3 = np.concatenate([np.full(1000, 0.30), np.full(1000, 0.75),
                           np.full(1000, 0.25), np.full(1000, 0.65)])
    x3 = simulate(e3, phi3)
    start = 400

    def loss_of(ph_hat):
        t = np.arange(start, T3 - 1)
        err = x3[t + 1] - ph_hat[t] * x3[t]
        return float(np.mean(err ** 2) - np.mean(e3[t + 1] ** 2))

    roll_w = [25, 50, 100, 200, 400, 800]
    roll_loss = {}
    for w in roll_w:
        t, ph = rolling_phi(x3, w)
        arr = np.full(T3, np.nan)
        arr[t] = ph
        # before the first full window, use the TRUE coefficient (oracle) so the
        # excess loss there is exactly zero and no method is penalised unfairly
        roll_loss[w] = loss_of(np.where(np.isnan(arr), phi3, arr))
    # RLS with forgetting factor
    def rls(lam):
        out = np.zeros(T3)
        m, Pv = 0.5, 1.0
        for t in range(1, T3):
            h = x3[t - 1]
            den = lam + Pv * h * h
            K = Pv * h / den
            m = m + K * (x3[t] - h * m)
            Pv = (Pv - K * Pv * h) / lam
            out[t] = m
        return out
    lam_grid = [0.95, 0.97, 0.99, 0.995, 0.999]
    rls_loss = {l: loss_of(rls(l)) for l in lam_grid}
    # Kalman random-walk coefficient, R = 1 known
    def kalman(q):
        out = np.zeros(T3)
        m, Pv = 0.5, 1.0
        for t in range(1, T3):
            Ppred = Pv + q
            h = x3[t - 1]
            Sden = h * h * Ppred + 1.0
            K = Ppred * h / Sden
            m = m + K * (x3[t] - h * m)
            Pv = (1 - K * h) * Ppred
            out[t] = m
        return out
    q_grid = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    kal_loss = {q: loss_of(kalman(q)) for q in q_grid}
    best_w = min(roll_loss, key=roll_loss.get)
    best_l = min(rls_loss, key=rls_loss.get)
    best_q = min(kal_loss, key=kal_loss.get)
    S["exp3"] = {
        "T": T3, "break_days": [1000, 2000, 3000],
        "phi_segments": [0.30, 0.75, 0.25, 0.65],
        "rolling_w": roll_w, "rolling_loss": {str(k): float(v) for k, v in roll_loss.items()},
        "rls_lambda": lam_grid, "rls_loss": {str(k): float(v) for k, v in rls_loss.items()},
        "kalman_q": q_grid, "kalman_loss": {str(k): float(v) for k, v in kal_loss.items()},
        "best_rolling_w": best_w, "best_rolling_loss": float(roll_loss[best_w]),
        "best_rls_lambda": best_l, "best_rls_loss": float(rls_loss[best_l]),
        "best_kalman_q": best_q, "best_kalman_loss": float(kal_loss[best_q]),
        "kalman_q_too_small_loss": float(kal_loss[1e-7]),
        "kalman_q_too_large_loss": float(kal_loss[1e-1]),
        "oracle_excess": 0.0,
    }
    # post-break tracking error in the first 100 days after each break
    def tracking(ph_hat):
        vals = []
        for b in [1000, 2000, 3000]:
            ii = np.arange(b, b + 100)
            vals.append(float(np.mean(np.abs(ph_hat[ii] - phi3[ii]))))
        return float(np.mean(vals))
    kb = kalman(best_q); rb = rls(best_l)
    tw, tp = rolling_phi(x3, best_w)
    arr_w = np.full(T3, np.nan); arr_w[tw] = tp
    S["exp3"]["tracking_error_first100"] = {
        "best_kalman": tracking(kb), "best_rls": tracking(rb),
        "best_rolling": tracking(np.where(np.isnan(arr_w), phi3, arr_w)),
        "kalman_q_1e-7": tracking(kalman(1e-7)),
    }
    F["exp3"] = dict(phi=phi3, roll_loss=roll_loss, rls_loss=rls_loss,
                     kal_loss=kal_loss, kal_best=kb, roll_best_w=best_w)

    # --- Exp 4: prequential window selection is noisy and leaks optimism ---
    T4 = 6000
    rng4 = np.random.default_rng(SEED + 4)
    e4 = rng4.standard_normal(T4)
    phi4 = triangle_phi(T4, P0)
    x4 = simulate(e4, phi4)
    wl = w_for(P0)
    L = np.full((T4, len(wl)), np.nan)
    for j, w in enumerate(wl):
        t, ph = rolling_phi(x4, w)
        L[t, j] = (x4[t + 1] - ph * x4[t]) ** 2
    valid = np.arange(1000, T4 - 2)
    in_sample_mean = np.array([np.nanmean(L[valid, j]) for j in range(len(wl))])
    w_cheat = int(wl[int(np.argmin(in_sample_mean))])
    fixed_w = int(min(wl, key=lambda w: abs(np.log(w / w_emp0))))

    cs = np.nancumsum(np.nan_to_num(L, nan=0.0), axis=0)
    cnt = np.cumsum(~np.isnan(L), axis=0)
    jf, jc = wl.index(fixed_w), wl.index(w_cheat)

    def prequential(K):
        # decision at t uses losses realised up to t-1; the resulting forecast
        # is for x[t+1], whose error is exactly L[t, j]. No future information.
        ev = np.arange(max(BURN + K, 1000), T4 - 2)
        picks, losses = [], []
        for t in ev:
            past = cs[t - 1] - cs[t - 1 - K]
            n = cnt[t - 1] - cnt[t - 1 - K]
            scores = np.divide(past, n, out=np.full(len(wl), np.inf), where=n > 0)
            j = int(np.argmin(scores))
            picks.append(wl[j])
            losses.append(L[t, j])
        fixed_same = float(np.nanmean(L[ev, jf]))
        cheat_same = float(np.nanmean(L[ev, jc]))
        return float(np.mean(losses)), np.asarray(picks), fixed_same, cheat_same

    Ks = [50, 250, 1000]
    pre = {}
    for K in Ks:
        m, pk, fixed_same, cheat_same = prequential(K)
        pre[K] = {
            "mean_loss": m,
            "fixed_loss_same_range": fixed_same,
            "cheating_loss_same_range": cheat_same,
            "regret_vs_fixed": m - fixed_same,
            "regret_vs_cheating": m - cheat_same,
            "frac_w_lt50": float(np.mean(pk < 50)),
            "frac_w_gt200": float(np.mean(pk > 200)),
            "median_w": float(np.median(pk)),
        }
    ref = pre[250]
    S["exp4"] = {
        "T": T4, "K_grid": Ks, "prequential": {str(k): pre[k] for k in Ks},
        "fixed_w": fixed_w, "fixed_loss": ref["fixed_loss_same_range"],
        "w_full_sample_cheating": w_cheat,
        "full_sample_cheating_loss": ref["cheating_loss_same_range"],
        "optimism_gap": ref["fixed_loss_same_range"] - ref["cheating_loss_same_range"],
        "hindsight_equals_fixed": bool(w_cheat == fixed_w),
        "w_star_from_exp1": w_emp0,
    }
    for k in Ks:
        pre[k]["regret_pct_vs_fixed"] = 100.0 * pre[k]["regret_vs_fixed"] / pre[k]["fixed_loss_same_range"]
        pre[k]["regret_pct_vs_cheating"] = 100.0 * pre[k]["regret_vs_cheating"] / pre[k]["cheating_loss_same_range"]
    F["exp4"] = dict(pre=pre, Ks=Ks, L=L, valid=valid, wl=wl, fixed_w=fixed_w)
    return S, F


# ---------------- figures ----------------
def make_figures(F):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                         "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: drifting coefficient and three rolling estimates
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    seg = slice(0, 2400)
    ax[0].plot(F["exp1"]["phi"][seg], color="#111827", lw=1.8, label="true phi_t")
    for w, col in [(30, "#d62728"), (100, "#2563eb"), (400, "#16a34a")]:
        t, ph = F["tracks"][w]
        m = t < 2400
        ax[0].plot(t[m], ph[m], lw=1.0, color=col, alpha=0.85, label=f"rolling w={w}")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("AR(1) coefficient")
    ax[0].set_title("Short windows are noisy, long windows lag")
    ax[0].legend(frameon=False, fontsize=7.5, ncol=2)
    c = F["exp1"]["curve"]; th = F["exp1"]["theory_curve"]; wf = F["exp1"]["wgrid_fine"]
    ax[1].plot(F["exp1"]["w_grid"], c, "o-", color="#2563eb", lw=1.5, ms=4,
               label="measured excess MSE")
    ax[1].plot(wf, th, "--", color="#7c3aed", lw=1.4, label="bias^2 + variance theory")
    ax[1].axvline(F["exp1"]["w_emp"], color="#d62728", ls=":", lw=1.4,
                  label=f"empirical w* = {F['exp1']['w_emp']:.0f}")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("window length w"); ax[1].set_ylabel("excess one-step MSE")
    ax[1].set_title("The U shape is a bias-variance trade-off")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_drift_and_u_shape.png", dpi=130); plt.close(fig)

    # fig 2: scaling law + noise invariance
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    rows = F["exp2"]["rows"]
    Ps = [r["P"] for r in rows]
    ws = [r["w_star"] for r in rows]
    wt = [r["w_theory"] for r in rows]
    ax[0].loglog(Ps, ws, "o-", color="#2563eb", lw=1.5, label="measured w*")
    ax[0].loglog(Ps, wt, "s--", color="#7c3aed", lw=1.3, label="closed-form theory")
    ax[0].set_xlabel("drift period P (slower drift to the right)")
    ax[0].set_ylabel("optimal window w*")
    ax[0].set_title("w* ~ P^(2/3), measured slope %.3f (R2 %.3f)" %
                    (F["exp2"]["slope"], F["exp2"]["r2"]))
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(F["exp1"]["w_grid"], F["exp1"]["curve"],
               "o-", color="#2563eb", ms=4, lw=1.4, label="sigma = 1")
    ax[1].plot(F["exp1"]["w_grid"], F["exp2"]["curve2"],
               "s-", color="#d62728", ms=4, lw=1.4, label="sigma = 2")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("window length w"); ax[1].set_ylabel("excess MSE")
    ax[1].set_title("Doubling the noise does NOT move the optimum")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig2_scaling_noise_invariance.png", dpi=130); plt.close(fig)

    # fig 3: abrupt breaks and the hidden tuning of adaptive methods
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    seg = slice(700, 1400)
    ax[0].plot(np.arange(700, 1400), F["exp3"]["phi"][seg], color="#111827", lw=2, label="true phi")
    ax[0].plot(F["exp3"]["kal_best"][seg], color="#16a34a", lw=1.1, label="tuned Kalman")
    ax[0].set_xlabel("day"); ax[0].set_ylabel("phi")
    ax[0].set_title("Abrupt break at day 1000")
    ax[0].legend(frameon=False, fontsize=7.5)
    rl = F["exp3"]["roll_loss"]
    ax[1].plot(list(rl), [rl[k] for k in rl], "o-", color="#2563eb", lw=1.4, ms=4)
    ax[1].set_xscale("log"); ax[1].set_xlabel("rolling w")
    ax[1].set_ylabel("excess MSE"); ax[1].set_title("Rolling: a U shape")
    rls = F["exp3"]["rls_loss"]
    ax[2].plot(list(rls), [rls[k] for k in rls], "s-", color="#f59e0b", lw=1.4, ms=4)
    ax[2].set_xscale("log"); ax[2].set_xlabel("RLS forgetting lambda")
    ax[2].set_title("RLS: also a U shape")
    kl = F["exp3"]["kal_loss"]
    ax[3].plot(list(kl), [kl[k] for k in kl], "o-", color="#16a34a", lw=1.4, ms=4)
    ax[3].set_xscale("log"); ax[3].set_xlabel("Kalman process noise q")
    ax[3].set_title("Kalman: q must be tuned")
    fig.tight_layout(); fig.savefig(OUT / "fig3_breaks_adaptive.png", dpi=130); plt.close(fig)

    # fig 4: prequential selection
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    pre, Ks = F["exp4"]["pre"], F["exp4"]["Ks"]
    vals = [pre[K]["mean_loss"] for K in Ks]
    ax[0].bar([str(K) for K in Ks], vals, color="#60a5fa")
    ax[0].axhline(F["exp4"]["fixed_loss"], color="#2563eb", ls="--", lw=1.3,
                  label="fixed w* = 100 (also the hindsight optimum)")
    ax[0].set_xlabel("prequential validation length K"); ax[0].set_ylabel("mean forecast loss")
    ax[0].set_ylim(min(vals + [F["exp4"]["fixed_loss"]]) * 0.985,
                   max(vals + [F["exp4"]["fixed_loss"]]) * 1.005)
    ax[0].set_title("Selecting w online costs regret")
    ax[0].legend(frameon=False, fontsize=7.5)
    frac_lo = [pre[K]["frac_w_lt50"] for K in Ks]
    frac_hi = [pre[K]["frac_w_gt200"] for K in Ks]
    xx = np.arange(len(Ks))
    ax[1].bar(xx - 0.2, frac_lo, 0.4, color="#d62728", label="picks w < 50")
    ax[1].bar(xx + 0.2, frac_hi, 0.4, color="#f59e0b", label="picks w > 200")
    ax[1].set_xticks(xx); ax[1].set_xticklabels([str(K) for K in Ks])
    ax[1].set_xlabel("prequential validation length K"); ax[1].set_ylabel("fraction of days")
    ax[1].set_title("Short validation windows pick extreme w")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_prequential_selection.png", dpi=130); plt.close(fig)

    # fig 5: theory decomposition bars + tracking error after breaks
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ws = F["exp1"].get("decomposition", [])
    if ws:
        labels = [f"w={d['w']}" for d in ws]
        b2 = [d["bias2_term"] for d in ws]
        vv = [d["variance_term"] for d in ws]
        xx = np.arange(len(ws))
        ax[0].bar(xx - 0.2, b2, 0.4, color="#d62728", label="bias^2 term")
        ax[0].bar(xx + 0.2, vv, 0.4, color="#2563eb", label="variance term")
        ax[0].set_yscale("log"); ax[0].set_xticks(xx); ax[0].set_xticklabels(labels)
        ax[0].set_ylabel("excess MSE contribution")
        ax[0].set_title("Where the U shape comes from")
        ax[0].legend(frameon=False, fontsize=8)
    te = F["exp3"]["tracking"]
    names = list(te.keys()); vv = [te[k] for k in names]
    ax[1].bar(names, vv, color=["#16a34a", "#60a5fa", "#93c5fd", "#d62728"])
    for i, v in enumerate(vv): ax[1].text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylabel("mean |phi_hat - phi| in first 100 days")
    ax[1].set_ylim(0, max(vv) * 1.25); ax[1].tick_params(axis="x", labelsize=7.5)
    ax[1].set_title("Mis-specified process noise adapts slowly")
    fig.tight_layout(); fig.savefig(OUT / "fig5_decomposition_tracking.png", dpi=130); plt.close(fig)


def _ser(o):
    if isinstance(o, dict):
        return {k: _ser(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_ser(v) for v in o]
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return _ser(o.tolist())
    return o


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S, F = run_all()
    F["exp1"]["decomposition"] = S["exp1"]["decomposition"]
    F["exp3"]["tracking"] = S["exp3"]["tracking_error_first100"]
    F["exp4"]["fixed_loss"] = S["exp4"]["fixed_loss"]
    F["exp4"]["full_sample_cheating_loss"] = S["exp4"]["full_sample_cheating_loss"]
    make_figures(F)
    (OUT / "results.json").write_text(
        json.dumps(_ser(S), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("w* empirical / theory =", S["exp1"]["w_star_empirical"],
          S["exp1"]["w_star_theory_closed_form"])
    print("log-log slope =", S["exp2"]["loglog_slope"], "R2 =", S["exp2"]["loglog_r2"])
    print("sigma1 / sigma2 w* =", S["exp2"]["noise_invariance"]["sigma1_w_star"],
          S["exp2"]["noise_invariance"]["sigma2_w_star"])
    print("best rolling / RLS / Kalman =", S["exp3"]["best_rolling_loss"],
          S["exp3"]["best_rls_loss"], S["exp3"]["best_kalman_loss"])
    print("prequential =", {k: v["mean_loss"] for k, v in S["exp4"]["prequential"].items()})
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```
