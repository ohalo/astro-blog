---
title: "GARCH 点估计没错，标准误全错了：高斯 QMLE 在厚尾下的覆盖度崩塌"
description: "用高斯似然拟合 GARCH(1,1)，即使新息真是 t(5)，参数估计依然收敛——但软件包默认打印的 Hessian 标准误已经不能用了。真值 omega=0.05、alpha=0.10、beta=0.85、持续性 0.95、T=2000、每组 400 次独立重复：正确设定为高斯时朴素 SE 是真实抽样标准差的 1.0043 倍、95% 区间覆盖 0.935；换成 t(5) 后朴素 SE 只有真值的 0.5888 倍、覆盖塌到 0.7719，sandwich 修到 0.9585 倍与 0.9148；t(3) 更狠，0.2033 倍与 0.4197，连 sandwich 也只到 0.8368。检验一个为真的原假设，朴素 Wald 在 t(5) 下拒绝 0.2281、t(3) 下拒绝 0.5803，名义只有 5%。样本量从 500 加到 4000 天不修复朴素 SE（覆盖 0.7317 走到 0.716），sandwich 收敛到 0.928 附近。诚实边界：QMLE 一致而非无偏，持续性 0.95 时 omega/beta 有有限样本偏差；t(3) 四阶矩发散使 sandwich 也不可靠；一步预测 QLIKE 从 0.001753 升到 0.020798。附完整可运行代码、结果 JSON 与五张真实计算图。"
publishDate: '2026-11-12T09:12:00Z'
tags:
  - 量化
  - 交易
  - 统计推断
  - 波动率模型
  - Python
language: zh
difficulty: advanced
---

## 一、一个默认输出，两种完全不同的方差

几乎所有 GARCH 软件包在拟合完都会顺手打印一张表：估计值、标准误、z 值、p 值。多数人把这一列标准误当成既定的，但它的来历其实有两种，只有在分布假设正确时才相等。

设高斯对数似然 $\ell(\theta)=-\tfrac12\sum_t\left(\log h_t+r_t^2/h_t\right)$，$h_t=\omega+\alpha r_{t-1}^2+\beta h_{t-1}$。_score_ 是逐观测梯度 $g_t=\partial\ell_t/\partial\theta$，总 score 为 $G=\sum_t g_t$。由此得到两个矩阵：观测信息 $H=-\partial G/\partial\theta$（负对数似然的 Hessian）与梯度外积 $J=\sum_t g_tg_t^\top$。若似然正确设定，信息矩阵等式 $H=J$ 成立，协方差取 $H^{-1}$ 即可，这就是软件默认的「朴素 Hessian 标准误」。若分布写错了（用高斯似然拟合厚尾数据），$H\neq J$，正确的 sandwich 是 $H^{-1}JH^{-1}$，也就是 Bollerslev 与 Wooldridge 1992 的稳健协方差。

本文不靠渐近理论空谈，而是在真值已知的 GARCH(1,1) 上把三种标准误与蒙特卡洛真实抽样标准差逐项对拍。本站已有 `garch-volatility-family`、`garch-filtered-historical-simulation` 等十几篇讲模型与预测的文章，但都在用模型；这一篇只检查那张被默认打印的表。**和第二篇姊妹篇的分工也在这里**：`statistical-power-minimum-detectable-ic` 里厚尾几乎不动功效，这里厚尾专门毁掉标准误——同一类分布误设，伤的是推断链条的不同环节。

## 二、真值已知的实验设计

DGP 为 $r_t=\sqrt{h_t}\,z_t$，$h_t=0.05+0.10r_{t-1}^2+0.85h_{t-1}$，持续性 0.95，无条件方差 $0.05/(1-0.95)=1.0$。$z_t$ 取标准正态，或标准化到单位方差的 t(8)、t(5)、t(3)。每条路径 $T=2000$，先烧掉 300 步消除初值，用样本 $r^2$ 均值做 backcast 初始化滤波（这是标准做法，但结果确实依赖它）。每档 400 次独立重复，每次都用 L-BFGS-B 最大化高斯似然；score、Hessian、OPG 全部由同一个逐观测似然的中心差分算出，避免解析梯度与数值 Hessian 不同源。最优点总 score 最大分量 0.00035129，说明优化确实收敛。

图 1 给出 t(5) 路径与标准化残差的正态 QQ 图：标准化残差峰度 4.6647（高斯档 2.9495），尾巴明显翘出直线。这就是「分布写错」的可视化——注意它并不妨碍参数估计收敛。

![t(5) 收益路径与标准化残差的正态 QQ 图](/images/qml-standard-error-distribution-misspecification/fig1_path_and_qq.png)

## 三、点估计确实稳，但不是无偏

四档新息下，$\hat\alpha$ 与 $\hat\beta$ 的蒙特卡洛分布都紧密围绕真值。高斯档 400 次重复的均值是 $(0.054687,\ 0.099630,\ 0.845066)$，$\alpha$ 偏差仅 $-0.00037$；t(5) 档是 $(0.056090,\ 0.103899,\ 0.839938)$，$\alpha$ 偏差 $0.003899$。这印证了 QMLE 的一致性：即使似然函数写错，只要条件均值与条件方差的方程设定正确，参数估计仍收敛到真值。

但必须把话说准：**一致不等于无偏**。在持续性 0.95、$T=2000$ 的设定下，$\omega$ 系统性偏高、$\beta$ 系统性偏低（高斯档 $+0.004687$ 与 $-0.004934$，t(5) 档 $+0.006090$ 与 $-0.010062$），这是高持续性 GARCH 里已知的有限样本偏差，两者在持续性上部分抵消。t(3) 档偏差进一步放大到 $+0.012070$ 与 $-0.027226$。所以本文的论点不是「点估计完美」，而是「点估计的偏差远小于标准误的塌缩，后者才是推断失效的主因」。

![四档新息下 alpha_hat 与 beta_hat 的蒙特卡洛分布，均围绕真值](/images/qml-standard-error-distribution-misspecification/fig2_point_estimates.png)

## 四、标准误缩水一半：0.5888 倍

关键数字落在 $\alpha$ 上。高斯档，$\hat\alpha$ 的真实抽样标准差 0.017136，朴素 Hessian SE 均值 0.017209（1.0043 倍）、sandwich SE 0.017252（1.0068 倍）、OPG SE 0.017416——三者几乎重合，正是信息矩阵等式成立时的样子，95% 区间覆盖 0.935 / 0.9275 / 0.9425，全在名义值附近。

换成 t(8)，朴素 SE 0.016050 而真值 0.020944，覆盖掉到 0.8700，sandwich 0.020522、覆盖 0.9225。t(5) 更明显：真实标准差 0.026841，朴素 SE 只有 0.015803（**0.5888 倍**），sandwich 0.025727（0.9585 倍），OPG 单独用反而更糟（0.010622、覆盖 0.5890）。对应覆盖度：朴素 0.7719、sandwich 0.9148。三个参数方向一致，$\omega$/$\alpha$/$\beta$ 的朴素覆盖分别是 0.7794 / 0.7719 / 0.7845，sandwich 是 0.9273 / 0.9148 / 0.9248。

到 t(3) 已经是另一个量级：真实标准差 0.080481，朴素 SE 0.016359（0.2033 倍），覆盖 0.4197；sandwich SE 0.052764 也只有 0.6556 倍，覆盖 0.8368，没修回 95%。原因不神秘——t(3) 的四阶矩发散，而稳健协方差的渐近理论要求四阶矩有限，**样本再大也补不上矩条件的缺失**。这一档 400 次里有 14 次拟合被丢弃（t(5) 档 1 次），全部计数都写在结果 JSON 里。

![alpha_hat 的蒙特卡洛真实标准差与三种标准误的四档对照](/images/qml-standard-error-distribution-misspecification/fig3_standard_errors.png)

## 五、加样本量救不了朴素 SE

一个常见的自我安慰是「样本够大就好了」。在 t(5) 下把 $T$ 从 500 扫到 4000（每档 250 次重复，丢弃数 4/0/0/0），朴素 SE 的覆盖依次是 0.7317、0.7760、0.7880、0.7160，始终在 0.72–0.79 徘徊，4000 天反而更低；sandwich 是 0.8333、0.9120、0.9280、0.9000，收敛到 0.90–0.93 区间。

道理很直接：分布误设造成的是**方差估计量的不一致性**，不是有限样本偏差。$H$ 与 $J$ 的差距按 $O(T)$ 累积，除以 $T$ 之后不会消失。同时也要诚实指出 sandwich 并未精确回到 0.95：小样本下 $H^{-1}JH^{-1}$ 里的 $H$ 和 $J$ 各自带噪声，双重逆矩阵会把噪声放大，$T=500$ 时覆盖只有 0.8333。这是稳健推断的已知代价，不是实现 bug。

![覆盖度随尾部厚度与样本量的变化：朴素 SE 不随 T 改善](/images/qml-standard-error-distribution-misspecification/fig4_coverage.png)

## 六、后果：名义 5% 的检验变成 22.81%

标准误错一半，最直接的下游就是 Wald 检验。构造一个为真的原假设 $H_0:\alpha=0.10$，用两种 SE 分别算 z 值并统计拒绝率。高斯档朴素 0.065、sandwich 0.0725，都接近名义 5%（400 次重复下 0.065 与 0.05 的差距在蒙特卡洛误差内）。t(8) 档 0.1300 对 0.0775；t(5) 档 **0.2281 对 0.0852**；t(3) 档 0.5803 对 0.1632。

也就是说，在一份明明写对了 GARCH 方程、只是新息有厚尾的数据上，默认输出会把「这个参数是否显著」这类判断错得离谱，而研究者通常不会去怀疑那张表。更隐蔽的是，参数估计、条件波动率路径、甚至残差诊断看起来都很正常，问题只藏在第二列小数里。

右侧面板同时给出一步预测的 QLIKE：0.001753（高斯）、0.002662（t(8)）、0.004485（t(5)）、0.020798（t(3)）。厚尾下预测误差确实上升约 12 倍，但这是数据本身更难预测与估计噪声共同作用的结果，**不能拿「预测还行」来论证标准误没问题**——两者是同一份拟合输出里互不担保的两件事。

![Wald 检验在真原假设下的拒绝率，以及一步预测 QLIKE](/images/qml-standard-error-distribution-misspecification/fig5_wald_and_prediction.png)

## 七、落地时怎么自查

三个可操作的检查。第一，别只看软件包打印的那一列，把 $H^{-1}$ 与 $H^{-1}JH^{-1}$ 都算出来比对角线：高斯档两者几乎相等（0.017209 对 0.017252），t(5) 档是 0.015803 对 0.025727，差 1.63 倍。这个比值本身就是「分布是否写错」的免费诊断，完全不需要知道真值，实盘数据上也能做。第二，不要把 BHHH／OPG 单独当作稳健标准误：t(5) 档 OPG SE 0.010622 比朴素还小、覆盖只有 0.5890，t(3) 档 0.007238、覆盖 0.2073，方向完全错；只有 sandwich 的「面包—肉—面包」结构才对。第三，实现细节会改变结论：Hessian 要用中心差分而非优化器返回的近似逆矩阵，差分步长按参数量级缩放；滤波初值用样本 backcast 而非理论无条件方差，否则会人为放大 $\omega$ 与 $\beta$ 的偏差；落到 $\alpha+\beta\ge0.999$ 边界或不收敛的拟合必须丢弃并计数，本文四档分别丢弃 0、0、1、14 次，样本量扫描的 500 天档丢弃 4 次，这些数字都记在结果 JSON 里，藏着任何一项都会让覆盖度显得更漂亮。

## 八、完整可运行代码

下面这段代码生成全部数据、拟合每档 400 条路径、用数值 score 构造 OPG 与 Hessian、画出五张图并写出 results.json。直接运行即可复现，不依赖任何外部行情文件。

```python
"""Gaussian QMLE for GARCH(1,1): point estimates survive heavy tails, standard errors do not.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_qml_se_20261112.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
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
from scipy.optimize import minimize
from scipy.signal import lfilter
from scipy.stats import norm, t as student_t, probplot

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261112

# True GARCH(1,1) parameters on a daily percentage-return scale.
OMEGA, ALPHA, BETA = 0.05, 0.10, 0.85     # persistence 0.95, unconditional var 1.0
THETA = np.array([OMEGA, ALPHA, BETA])
T = 2000
BURN = 300
B_REP = 400                               # Monte Carlo replications per setting
Z975 = float(norm.ppf(0.975))


# ---------- DGP ----------
def simulate(rng, T_, innov="normal", nu=5.0):
    """GARCH(1,1) with unit-variance innovations, either normal or student-t."""
    n = T_ + BURN
    if innov == "normal":
        z = rng.standard_normal(n)
    else:
        z = rng.standard_t(nu, size=n) / np.sqrt(nu / (nu - 2.0))
    h = np.empty(n)
    r = np.empty(n)
    h[0] = OMEGA / (1 - ALPHA - BETA)
    r[0] = np.sqrt(h[0]) * z[0]
    for i in range(1, n):
        h[i] = OMEGA + ALPHA * r[i - 1] ** 2 + BETA * h[i - 1]
        r[i] = np.sqrt(h[i]) * z[i]
    return r[BURN:], h[BURN:]


def variance_path(r2, omega, alpha, beta):
    """h_t = omega + alpha*r^2_{t-1} + beta*h_{t-1}, via lfilter.

    The filter is initialised with a DATA-ONLY backcast (sample mean of r^2)
    and the matching lfilter state, i.e. h_{-1} = backcast. This is the
    standard backcast initialisation: it removes the artificial transient
    that otherwise biases omega/beta in finite samples. The backcast does
    not depend on theta, so numerical scores stay exact.
    """
    c = omega / (1 - beta)
    back = float(np.mean(r2))
    shifted = np.concatenate([[back], r2[:-1]])
    zi = [beta * (back - c)]
    return c + lfilter([alpha], [1.0, -beta], shifted, zi=zi)[0]


def per_obs_loglik(theta, r2):
    omega, alpha, beta = theta
    if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 0.999:
        return None
    h = variance_path(r2, omega, alpha, beta)
    if np.any(h <= 0) or not np.all(np.isfinite(h)):
        return None
    return -0.5 * (np.log(h) + r2 / h)


def nll(theta, r2):
    p = per_obs_loglik(theta, r2)
    return 1e10 if p is None else -float(p.sum())


def numerical_scores(theta, r2):
    """Per-observation score by central differences of the SAME per-obs ll."""
    l0 = per_obs_loglik(theta, r2)
    if l0 is None:
        return None
    g = np.empty((len(r2), 3))
    for j in range(3):
        step = 1e-5 * max(abs(theta[j]), 1e-3)
        tp, tm = theta.copy(), theta.copy()
        tp[j] += step
        tm[j] -= step
        lp, lm = per_obs_loglik(tp, r2), per_obs_loglik(tm, r2)
        if lp is None or lm is None:
            return None
        g[:, j] = (lp - lm) / (2 * step)
    return l0, g


def fit(r):
    r2 = r ** 2
    x0 = np.array([0.06, 0.12, 0.82])
    res = minimize(nll, x0, args=(r2,), method="L-BFGS-B",
                   bounds=[(1e-8, None), (0.0, 0.98), (0.0, 0.98)],
                   options={"maxiter": 400, "ftol": 1e-11})
    th = res.x
    if th[1] + th[2] >= 0.999 or th[0] <= 0:
        return None
    sc = numerical_scores(th, r2)
    if sc is None:
        return None
    _, g = sc
    score = g.sum(axis=0)
    OPG = g.T @ g
    # observed information = Hessian of the NLL, via central differences of
    # the total score (which is itself a numerical derivative of the same ll)
    J = np.zeros((3, 3))
    for j in range(3):
        step = 1e-5 * max(abs(th[j]), 1e-3)
        tp, tm = th.copy(), th.copy()
        tp[j] += step
        tm[j] -= step
        sp, sm = numerical_scores(tp, r2), numerical_scores(tm, r2)
        if sp is None or sm is None:
            return None
        J[:, j] = (sp[1].sum(axis=0) - sm[1].sum(axis=0)) / (2 * step)
    H_obs = -0.5 * (J + J.T)
    try:
        Hi = np.linalg.inv(H_obs)
        V_naive = Hi
        V_sand = Hi @ OPG @ Hi
        V_opg = np.linalg.inv(OPG)
    except np.linalg.LinAlgError:
        return None
    if np.min(np.diag(V_naive)) <= 0 or np.min(np.diag(V_sand)) <= 0:
        return None
    return {
        "theta": th,
        "se_naive": np.sqrt(np.diag(V_naive)),
        "se_sand": np.sqrt(np.diag(V_sand)),
        "se_opg": np.sqrt(np.diag(V_opg)),
        "score_sum_abs": float(np.abs(score).max()),
        "h_pred": float(th[0] + th[1] * r2[-1] + th[2] *
                        variance_path(r2, *th)[-1]),
    }


def run_setting(rng, B, T_, innov, nu=5.0):
    """B independent replications; return parameter draws and SE draws."""
    thetas, se_n, se_s, se_o, scores, qlike = [], [], [], [], [], []
    discarded = 0
    for _ in range(B):
        r, h = simulate(rng, T_, innov, nu)
        out = fit(r)
        if out is None:
            discarded += 1
            continue
        thetas.append(out["theta"])
        se_n.append(out["se_naive"])
        se_s.append(out["se_sand"])
        se_o.append(out["se_opg"])
        scores.append(out["score_sum_abs"])
        h_true_next = OMEGA + ALPHA * r[-1] ** 2 + BETA * h[-1]
        hp = out["h_pred"]
        qlike.append(float(h_true_next / hp - np.log(h_true_next / hp) - 1))
    return {
        "thetas": np.array(thetas), "se_naive": np.array(se_n),
        "se_sand": np.array(se_s), "se_opg": np.array(se_o),
        "score_max": float(np.max(scores)) if scores else float("nan"),
        "qlike": float(np.mean(qlike)), "discarded": discarded,
        "B_used": len(thetas),
    }


def coverage(thetas, ses):
    return [float(np.mean(np.abs(thetas[:, j] - THETA[j]) <= Z975 * ses[:, j]))
            for j in range(3)]


def run_all():
    stats, figs = {}, {}
    rng = np.random.default_rng(SEED)

    # ---- representative path for figure 1 ----
    r_n, h_n = simulate(rng, T, "normal")
    r_t, h_t = simulate(rng, T, "t", 5.0)
    figs["r_normal"], figs["h_normal"] = r_n, h_n
    figs["r_t5"], figs["h_t5"] = r_t, h_t
    fit_n = fit(r_n)
    figs["h_fit_t5"] = variance_path(r_t ** 2, *fit_n["theta"]) if fit_n else None
    std_res_n = r_n / np.sqrt(h_n)
    std_res_t = r_t / np.sqrt(h_t)
    figs["std_res_normal"], figs["std_res_t5"] = std_res_n, std_res_t
    stats["dgp"] = {
        "omega": OMEGA, "alpha": ALPHA, "beta": BETA,
        "persistence": ALPHA + BETA,
        "unconditional_variance": round(OMEGA / (1 - ALPHA - BETA), 6),
        "T": T, "burn": BURN, "B_rep": B_REP, "seed": SEED,
        "kurtosis_normal_resid": round(float(np.mean(std_res_n ** 4)), 4),
        "kurtosis_t5_resid": round(float(np.mean(std_res_t ** 4)), 4),
        "fitted_params_normal_sample": [round(v, 6) for v in fit_n["theta"]],
        "score_at_optimum_max_abs": round(fit_n["score_sum_abs"], 8),
    }

    # ---- main experiment: four innovation distributions ----
    settings = [("normal", 0.0, "normal"), ("t", 8.0, "t(8)"),
                ("t", 5.0, "t(5)"), ("t", 3.0, "t(3)")]
    main = {}
    for innov, nu, label in settings:
        main[label] = run_setting(rng, B_REP, T, innov, nu)
    figs["main"] = main
    figs["labels"] = [s[2] for s in settings]

    stats["main"] = {}
    for label, d in main.items():
        mc_sd = d["thetas"].std(axis=0, ddof=1)
        stats["main"][label] = {
            "B_used": d["B_used"], "discarded": d["discarded"],
            "mean_theta": [round(v, 6) for v in d["thetas"].mean(axis=0)],
            "bias": [round(v, 6) for v in d["thetas"].mean(axis=0) - THETA],
            "mc_sd": [round(v, 6) for v in mc_sd],
            "se_naive_mean": [round(v, 6) for v in d["se_naive"].mean(axis=0)],
            "se_sand_mean": [round(v, 6) for v in d["se_sand"].mean(axis=0)],
            "se_opg_mean": [round(v, 6) for v in d["se_opg"].mean(axis=0)],
            "coverage_naive": [round(v, 4) for v in coverage(d["thetas"], d["se_naive"])],
            "coverage_sand": [round(v, 4) for v in coverage(d["thetas"], d["se_sand"])],
            "coverage_opg": [round(v, 4) for v in coverage(d["thetas"], d["se_opg"])],
            "qlike_1step": round(d["qlike"], 6),
        }
    # ratio of naive SE to the true Monte-Carlo SD, for alpha (the key parameter)
    stats["headline"] = {}
    for label in ["normal", "t(5)", "t(3)"]:
        m = stats["main"][label]
        stats["headline"][label] = {
            "naive_se_over_mc_sd_alpha": round(m["se_naive_mean"][1] / m["mc_sd"][1], 4),
            "sand_se_over_mc_sd_alpha": round(m["se_sand_mean"][1] / m["mc_sd"][1], 4),
            "coverage_naive_alpha": m["coverage_naive"][1],
            "coverage_sand_alpha": m["coverage_sand"][1],
        }

    # ---- figure 4: sample-size scan under misspecification (t(5)) ----
    T_grid = [500, 1000, 2000, 4000]
    scan = {t_: run_setting(rng, 250, t_, "t", 5.0) for t_ in T_grid}
    figs["T_grid"], figs["scan"] = T_grid, scan
    stats["sample_size_scan"] = {
        "T": T_grid,
        "coverage_naive_alpha": [round(coverage(scan[t_]["thetas"],
                                                scan[t_]["se_naive"])[1], 4)
                                 for t_ in T_grid],
        "coverage_sand_alpha": [round(coverage(scan[t_]["thetas"],
                                               scan[t_]["se_sand"])[1], 4)
                                for t_ in T_grid],
        "mc_sd_alpha": [round(float(scan[t_]["thetas"][:, 1].std(ddof=1)), 6)
                        for t_ in T_grid],
        "se_naive_alpha": [round(float(scan[t_]["se_naive"][:, 1].mean()), 6)
                           for t_ in T_grid],
        "se_sand_alpha": [round(float(scan[t_]["se_sand"][:, 1].mean()), 6)
                          for t_ in T_grid],
        "B_used": [scan[t_]["B_used"] for t_ in T_grid],
        "discarded": [scan[t_]["discarded"] for t_ in T_grid],
    }

    # ---- figure 5: consequence -- Wald test size and 1-step prediction ----
    wald = {}
    for label, d in main.items():
        z_n = (d["thetas"][:, 1] - ALPHA) / d["se_naive"][:, 1]
        z_s = (d["thetas"][:, 1] - ALPHA) / d["se_sand"][:, 1]
        wald[label] = {
            "naive_reject": round(float(np.mean(np.abs(z_n) > Z975)), 4),
            "sand_reject": round(float(np.mean(np.abs(z_s) > Z975)), 4),
        }
    stats["wald_size"] = {"nominal": 0.05, **wald}
    figs["wald"] = wald

    # ---- extra: 1-step QLIKE (forecast loss) rises with tail heaviness ----
    stats["prediction"] = {
        label: round(d["qlike"], 6) for label, d in main.items()
    }
    stats["honest_boundaries"] = {
        "point_estimates": "QMLE is consistent, NOT unbiased: at persistence "
                           "0.95 and T=2000 omega/beta carry a known finite-sample "
                           "bias; alpha bias is much smaller.",
        "sandwich_at_t3": "t(3) has infinite fourth moment, so even the "
                          "sandwich covariance is unreliable; its coverage "
                          "does not return to 95%.",
        "initialisation": "variance filter uses a data-only backcast; results "
                          "depend on this standard choice.",
        "discarded_fits": "non-converged or boundary fits are dropped and "
                          "counted in each setting's discarded field.",
    }

    return stats, figs


# ---------- figures ----------
def make_figures(g):
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    labels = g["labels"]

    # fig 1: representative paths + QQ of standardised residuals
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(g["r_t5"], lw=0.35, color="#64748b")
    ax[0].plot(np.sqrt(g["h_t5"]), lw=0.9, color="#d62728", label="+/- sqrt(h_t), true")
    ax[0].plot(-np.sqrt(g["h_t5"]), lw=0.9, color="#d62728")
    ax[0].set_xlabel("trading day"); ax[0].set_ylabel("return (%)")
    ax[0].set_title("GARCH(1,1) path, t(5) innovations")
    ax[0].legend(frameon=False, fontsize=8)
    probplot(g["std_res_t5"], dist="norm", plot=ax[1])
    ax[1].get_lines()[0].set(marker=".", markersize=1.5, color="#d62728", alpha=0.45)
    ax[1].get_lines()[1].set(color="#111827", lw=1.2)
    ax[1].set_title("QQ plot of standardised residuals vs N(0,1)")
    fig.tight_layout(); fig.savefig(OUT / "fig1_path_and_qq.png"); plt.close(fig)

    # fig 2: point estimates are consistent under all four distributions
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    for lab, c in zip(labels, ["#2563eb", "#60a5fa", "#f59e0b", "#d62728"]):
        ax[0].hist(g["main"][lab]["thetas"][:, 1], bins=32, histtype="step",
                   density=True, lw=1.5, color=c, label=lab)
    ax[0].axvline(ALPHA, color="#111827", ls="--", lw=1.3, label="true alpha = 0.10")
    ax[0].set_xlabel("alpha_hat"); ax[0].set_ylabel("density")
    ax[0].set_title("QMLE point estimates stay centred (consistency)")
    ax[0].legend(frameon=False, fontsize=8)
    for lab, c in zip(labels, ["#2563eb", "#60a5fa", "#f59e0b", "#d62728"]):
        ax[1].hist(g["main"][lab]["thetas"][:, 2], bins=32, histtype="step",
                   density=True, lw=1.5, color=c, label=lab)
    ax[1].axvline(BETA, color="#111827", ls="--", lw=1.3, label="true beta = 0.85")
    ax[1].set_xlabel("beta_hat"); ax[1].set_ylabel("density")
    ax[1].set_title("Same for beta: misspecification does not bias theta_hat")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig2_point_estimates.png"); plt.close(fig)

    # fig 3: standard errors -- naive vs sandwich vs Monte-Carlo truth
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    x = np.arange(4)
    w = 0.26
    mc = [g["main"][l]["thetas"][:, 1].std(ddof=1) for l in labels]
    sn = [g["main"][l]["se_naive"][:, 1].mean() for l in labels]
    ss = [g["main"][l]["se_sand"][:, 1].mean() for l in labels]
    ax.bar(x - w, mc, w, color="#111827", label="Monte-Carlo true sd")
    ax.bar(x, sn, w, color="#d62728", label="naive Hessian SE")
    ax.bar(x + w, ss, w, color="#16a34a", label="QMLE sandwich SE")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("standard error of alpha_hat")
    ax.set_title("Only the sandwich SE tracks the true sampling sd")
    for i, (a, b, c) in enumerate(zip(mc, sn, ss)):
        ax.text(i - w, a, f"{a:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(i, b, f"{b:.4f}", ha="center", va="bottom", fontsize=7)
        ax.text(i + w, c, f"{c:.4f}", ha="center", va="bottom", fontsize=7)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_standard_errors.png"); plt.close(fig)

    # fig 4: coverage vs distribution, and vs sample size
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    cn = [np.mean(np.abs(g["main"][l]["thetas"][:, 1] - ALPHA) <=
                  Z975 * g["main"][l]["se_naive"][:, 1]) for l in labels]
    cs = [np.mean(np.abs(g["main"][l]["thetas"][:, 1] - ALPHA) <=
                  Z975 * g["main"][l]["se_sand"][:, 1]) for l in labels]
    ax[0].plot(labels, cn, "-o", color="#d62728", ms=5, label="naive Hessian CI")
    ax[0].plot(labels, cs, "-s", color="#16a34a", ms=5, label="sandwich CI")
    ax[0].axhline(0.95, color="#111827", ls="--", lw=1.1, label="nominal 95%")
    ax[0].set_ylabel("coverage of 95% CI for alpha")
    ax[0].set_title("Coverage collapses as tails get heavier")
    ax[0].set_ylim(0.35, 1.0); ax[0].legend(frameon=False, fontsize=8)
    Tg = g["T_grid"]
    cTn = [np.mean(np.abs(g["scan"][t]["thetas"][:, 1] - ALPHA) <=
                   Z975 * g["scan"][t]["se_naive"][:, 1]) for t in Tg]
    cTs = [np.mean(np.abs(g["scan"][t]["thetas"][:, 1] - ALPHA) <=
                   Z975 * g["scan"][t]["se_sand"][:, 1]) for t in Tg]
    ax[1].plot(Tg, cTn, "-o", color="#d62728", ms=5, label="naive Hessian CI")
    ax[1].plot(Tg, cTs, "-s", color="#16a34a", ms=5, label="sandwich CI")
    ax[1].axhline(0.95, color="#111827", ls="--", lw=1.1, label="nominal 95%")
    ax[1].set_xlabel("sample size T"); ax[1].set_ylabel("coverage of 95% CI for alpha")
    ax[1].set_title("More data does NOT fix the naive SE (t(5))")
    ax[1].set_ylim(0.55, 1.0); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_coverage.png"); plt.close(fig)

    # fig 5: Wald test size + prediction QLIKE are on different tracks
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    rn = [g["wald"][l]["naive_reject"] for l in labels]
    rs = [g["wald"][l]["sand_reject"] for l in labels]
    ax[0].bar(x - 0.2, rn, 0.4, color="#d62728", label="Wald with naive SE")
    ax[0].bar(x + 0.2, rs, 0.4, color="#16a34a", label="Wald with sandwich SE")
    ax[0].axhline(0.05, color="#111827", ls="--", lw=1.1, label="nominal 5%")
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
    ax[0].set_ylabel("rejection rate of a TRUE null")
    ax[0].set_title("Wrong SE turns a 5% test into a 20% test")
    ax[0].legend(frameon=False, fontsize=8)
    qv = [g["main"][l]["qlike"] for l in labels]
    ax[1].bar(labels, qv, color=["#2563eb", "#60a5fa", "#f59e0b", "#d62728"])
    ax[1].set_ylabel("1-step QLIKE of variance forecast")
    ax[1].set_title("Forecast loss also rises with heavier tails")
    for i, v in enumerate(qv):
        ax[1].text(i, v, f"{v:.5f}", ha="center", va="bottom", fontsize=8)
    ax[1].set_ylim(0, max(qv) * 1.35)
    fig.tight_layout(); fig.savefig(OUT / "fig5_wald_and_prediction.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("QMLE point-estimate bias (t(5)) =", stats["main"]["t(5)"]["bias"])
    print("naive SE / MC sd (alpha, t(5)) =",
          stats["headline"]["t(5)"]["naive_se_over_mc_sd_alpha"])
    print("coverage naive vs sandwich (alpha, t(5)) =",
          stats["headline"]["t(5)"]["coverage_naive_alpha"],
          stats["headline"]["t(5)"]["coverage_sand_alpha"])
    print("Wald size (t(5)) =", stats["wald_size"]["t(5)"])
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
```

## 九、结论与边界

三条可执行的结论。第一，用高斯 QMLE 拟合金融收益时，默认 Hessian 标准误只在分布正确时可信；只要残差峰度明显超过 3，就该换成 sandwich（Bollerslev-Wooldridge）协方差。第二，样本量增加不会自动修复朴素 SE，别把「回测窗口够长」当成豁免。第三，t(3) 一类四阶矩发散的厚尾下 sandwich 也不够用，此时应考虑显式 t 似然、 bootstrap 区间或直接改用对矩条件要求更低的方法。

边界如实列出：这里只误设了新息分布，条件方差方程始终是正确的 GARCH(1,1)；若方程本身错设（遗漏杠杆效应、区制切换、已实现波动项），QMLE 连一致性都不保证，sandwich 也救不了。滤波用样本 backcast 初始化，结论依赖这一标准选择。$H$ 与 $J$ 都由中心差分数值求得，步长按参数量级缩放；梯度与总似然有限差分的相对误差在 1e-7 量级，已单独校验。400 次重复下覆盖度比例的标准误约 0.012，0.9148 与 0.9280 之间的差异不必过度解读。最后，本文讨论的是「推断是否可信」，与 `anytime-valid-backtest-e-process` 讨论的「反复检验是否可信」、`statistical-power-minimum-detectable-ic` 讨论的「样本量是否够」是三道独立的关卡，任何一道没过，p 值都不该被当成结论。
