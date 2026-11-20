---
title: "两个预测模型谁更准之外：嵌套检验问的是好模型有没有把差模型的信息全吃掉"
description: "Diebold-Mariano 只回答谁更准，encompassing 检验问的是另一件事——较强的模型是否已经吸收了较弱模型的全部增量信息。本文用真值已知的蒙特卡洛把这件事钉死：H0 下辅助回归斜率均值 -0.0113（理论 0），H1 下斜率 0.7266 精确落在理论值 0.4/(0.4+0.15)=0.7273 上。序列相关是隐形杀手：误差的 MA(1) 系数从 0 拉到 0.9，naive OLS-t 的虚警率从 0.046 爬到 0.100，Newey-West HAC 却稳在 0.061–0.081。功效随增量信号 beta 从 0.10 升到 0.62 由 0.197 涨到 1.000，噪声标准差翻倍则把 beta=0.34 处的功效从 0.890 压到 0.512。最反直觉的一条：把弱模型的估计偏差从 0 拉到 1.5，斜率从 1.000 掉到 0.211，但检验功效几乎不动（0.969→0.967）——统计显著和经济价值根本是两回事，斜率与它的标准误同比例缩小，t 统计量纹丝不动。"
publishDate: '2026-11-20T05:10:00Z'
tags:
  - 量化
  - 交易
  - 预测检验
  - 统计推断
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

比较两个预测模型，大多数人只会问一句：谁的均方误差更小？Diebold-Mariano 检验把这个「谁更准」变成了统计问题，但它回答不了一个更要紧的问题——**较准的那个模型，是不是已经把较准模型之外的信息全吸收了？** 换句话说，如果把弱模型的预测当成一条额外线索加进强模型，还能不能把误差压下去？这就是嵌套检验（encompassing test）要回答的事，也是本文唯一的主角。

本文用真值已知的蒙特卡洛把这件事钉死，四条结论里有两条是坏消息：

1. **嵌套检验的斜率能被真值精确验证。** H0（强模型已吸收全部信息）下，辅助回归斜率均值 -0.0113、理论值为 0；H1（弱模型握有强模型没有的信号）下，斜率均值 0.7266，而理论值 β/(β+b)=0.4/0.55=0.7273，两者吻合到小数点后三位。这个检验不是黑箱，它的每个数字都能对上公式。
2. **序列相关会让 naive 检验悄悄虚警。** 当预测误差带 MA(1) 结构、系数从 0 拉到 0.9，naive OLS-t 的虚警率从 0.046 一路爬到 0.100——名义 5% 的检验，实际十次里误报一次。Newey-West HAC 修正把它稳在 0.061–0.081。做多步预测、滚动预测时误差天然带序列相关，这一条几乎人人会踩。
3. **功效实打实来自增量信号，但被噪声狠狠惩罚。** 弱模型独有的信号强度 β 从 0.10 升到 0.62，HAC 检验功效从 0.197 涨到 1.000；可只要把噪声标准差从 1 翻到 2，β=0.34 处的功效就从 0.890 塌到 0.512。信号是真的也未必测得出，测不出的原因是噪声，不是信号不存在。
4. **统计显著不等于经济价值，这是最隐蔽的一条。** 把弱模型的估计偏差 b 从 0 拉到 1.5，斜率从 1.000 掉到 0.211——弱模型握有的可用信息被削掉近八成——可检验功效几乎不动（0.969→0.967）。原因是斜率和它的标准误同比例缩小，t 统计量纹丝不动。你能信心满满地宣布「它含信息」，却说不清这点信息值多少钱。

![图1：强模型忽略信号 z，弱模型用它；弱模型的预测因此能跟上真值里的信号成分](/images/forecast-encompassing-test/fig1_paths.png)

## 二、嵌套检验到底在检验什么

先说清楚设定。假设目标变量 y 里有一个信号成分 z，强模型 A 已经用了它，弱模型 B 用了 z 的一个带偏差的估计。A 的误差记作 e_A，B 的预测减去 A 的预测记作 x=f_B−f_A。所谓「A encompass B（A 吃掉 B）」，意思是 B 的预测里没有任何 A 还没用上的东西。

把 A 的误差对 x 做一次无截距回归：

$$
e_{A,t}=\delta\,x_t+u_t
$$

原假设是 δ=0：B 相对 A 的偏离，解释不了 A 的误差，说明 B 没有增量信息。备择假设 δ≠0：B 的偏离里藏着 A 没抓住的东西，这时候正确的做法不是二选一，而是把两个模型组合起来。这个框架来自 Davidson-MacKinnon 一类的嵌套检验，和 Diebold-Mariano 是互补关系：DM 比较损失大小，嵌套检验比较信息含量。

为什么真值能算出来？在本文的数据生成过程里 y=βz+η+u，A 完全不用 z，B 用 (β+b)z。于是 A 的误差 e_A=βz+η+u，而 x=f_B−f_A=(β+b)z。代入回归斜率的公式：

$$
\delta=\frac{\mathrm{Cov}(e_A,x)}{\mathrm{Var}(x)}=\frac{\beta(\beta+b)\mathrm{Var}(z)}{(\beta+b)^2\mathrm{Var}(z)}=\frac{\beta}{\beta+b}
$$

这个式子是全文的锚：H0 对应 β=0（弱模型握有的信号本身为零），δ=0；H1 对应 β>0，δ 为正。第四节会用数据把这个公式验证到小数点后三位。

## 三、序列相关：名义 5% 实际 10%

问题出在误差项 u 上。多步预测和滚动预测里，误差几乎从不是独立的——今天没预测到的部分，明天往往还没预测到。本文给噪声加了 MA(1) 结构，系数 ψ 从 0 调到 0.9。

先在原假设下量虚警率：跑 1500 次重复，看有多少次 t 统计量的绝对值越过 1.96 这条 5% 的线。结果很直白：ψ=0 时 naive OLS-t 的虚警率 0.046，中规中矩；ψ=0.3 升到 0.071，ψ=0.5 到 0.077，ψ=0.7 到 0.089，ψ=0.9 冲到 0.100。**名义 5% 的检验，在强序列相关下虚警率翻了一倍。** 如果你用它在 10 个策略里挑「显著优于基准」的那一个，纯靠噪声就能挑出假阳性。

![图2：序列相关把 naive 检验的虚警率从 0.046 顶到 0.100，HAC 修正稳在 0.061–0.081](/images/forecast-encompassing-test/fig2_size.png)

Newey-West HAC 修正的表现是另一种画风：同样五档 ψ 下，虚警率依次是 0.064、0.061、0.075、0.081、0.069。它在低序列相关处略偏保守（ψ=0 时 0.064 比 naive 的 0.046 高一点），但关键是**不随 ψ 单调爬升**——这是一个受控的检验该有的样子。代价是 HAC 要选滞后阶数，本文固定用 8 阶；阶数选得太小修不干净，太大会损失功效，这个权衡在有限样本里没有免费午餐。

## 四、功效、噪声与那条反直觉曲线

把 β 从 0.10 扫到 0.62，HAC 检验的功效从 0.197 单调爬到 1.000：β=0.26 时 0.726，β=0.34 时 0.890，β=0.42 时 0.981。这条 S 型曲线符合直觉——增量信号越强，越容易检出。但把噪声标准差从 1 翻到 2，同一批 β 上的功效变成 0.110、0.217、0.339、0.512……β=0.34 处直接从 0.890 掉到 0.512，几乎腰斩。**检测能力是信号与噪声的比值问题，不是信号的绝对值问题。**

![图3：功效随增量信号上升；噪声翻倍把 β=0.34 处的功效从 0.890 压到 0.512](/images/forecast-encompassing-test/fig3_power.png)

斜率估计的分布给出了最干净的验证。H0 下 1500 次重复的斜率均值 -0.0113、标准差 0.4443，分布以零为中心；H1（β=0.4、b=0.15）下斜率均值 0.7266、标准差 0.1933，而理论值 β/(β+b)=0.7273。**0.7266 对 0.7273，差 0.0007。** 检验统计量不是凭空算出来的，它在忠实地估计一个可以手算的量。

![图4：H0 下斜率分布在 0 附近，H1 下精确落在理论值 0.7273 上](/images/forecast-encompassing-test/fig4_slope_dist.png)

现在说最反直觉的发现。固定 β=0.4，把弱模型的估计偏差 b 从 0 拉到 1.5：按公式斜率会从 1.000 掉到 0.4/1.9=0.211，弱模型握有的可用信息被削掉近八成。可 HAC 检验的功效在 b=0 时是 0.969，b=1.5 时是 0.967，**几乎纹丝不动**。

这不是 bug，是 t 统计量的结构决定的。斜率 δ=β/(β+b) 下降的同时，回归元 x=(β+b)z 的方差按 (β+b)² 下降，斜率的标准误也跟着同比例缩小，于是 δ/se(δ) 保持稳定。你能非常有把握地宣布「B 含增量信息」，但这份信息的经济价值——用斜率衡量——已经从 1.000 缩到 0.211。**统计显著性和经济价值是两个维度，嵌套检验只保证前者。**

![图5：弱模型越不可靠，斜率越小但功效不降——显著不等于有价值](/images/forecast-encompassing-test/fig5_corr_power.png)

## 五、怎么用，以及边界在哪

实操上，嵌套检验的正确位置是在 DM 检验之后：先用 DM 确认谁的平均损失更小，再用嵌套检验问较强的那个是否已经吸收了较弱的信息。如果答案是「没有」，两个模型就该组合而不是二选一；如果答案是「已经吸收」，那弱模型可以直接丢掉，省下维护成本。本文第四节的最后一条提醒：即使检验显著，也要看斜率本身有多大，斜率接近零的「显著」在组合里几乎不带来收益。

边界必须说清。本文的推导假设误差是 MA(1)、信号是 AR(1)、样本量固定 240、回归无截距，这些是为真值可控而做的简化。真实预测误差的序列相关结构远比 MA(1) 复杂，HAC 的滞后阶数需要按样本量自适应选取（常见经验法则如 ⌊4(n/100)^(2/9)⌋）；预测若经过参数估计或滚动重训，误差里还会混入估计不确定性，斜率的渐近分布需要相应修正，本文的固定系数设定没有覆盖这一层。此外，本文只检验了单向嵌套（A 是否吃掉 B），两个模型互不嵌套时应当双向各做一次，两个方向都拒绝恰恰是组合预测最强的理由。这些情形都不在本文实验里，不能把「240 期、MA(1) 误差上的结论」直接外推成「任何嵌套比较都照此办理」。

## 六、可运行代码

下面这段代码就是生成本文全部数字与五张图的完整实现：数据生成、辅助回归、naive 与 HAC 两种 t 统计量、虚警率与功效模拟、斜率分布以及信号强度扫描都在里面，直接运行即可复现，统计结果写入 `results.json` 且与图同目录。核心是用固定 seed 的蒙特卡洛保证每次运行的数字完全一致。

```python
# -*- coding: utf-8 -*-
"""Forecast encompassing test: does the good model absorb all information in the weak one?

This script is the single source of truth for every number and figure in the article.
Setup: y = beta*z + eta + u, where z is an AR(1) signal and u is an MA(1) noise.
Model A (restricted) may or may not use z; Model B uses an estimate of z.
Encompassing H0: regressing A's error e1 on (fB - fA) has slope 0.
We compare naive OLS-t (ignores serial correlation) with Newey-West HAC-t.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261120
N = 240
REPS = 1500
PHI_Z = 0.5       # AR(1) coefficient of the signal z (serial correlation in forecast diff)
PSI_U = 0.5       # MA(1) coefficient of noise u (serial correlation in error)
L_HAC = 8         # Newey-West lags
Z95 = 1.959963985

_OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "public", "images", "forecast-encompassing-test")
os.makedirs(_OUT, exist_ok=True)


def gen(beta, a1, b_bias, phi_z, psi_u, sigma_u, n, rng):
    """Return e1, x=(fB-fA). a1: A's loading on z; B uses beta+b_bias."""
    e_z = rng.standard_normal(n)
    z = np.zeros(n)
    z[0] = e_z[0]
    for t in range(1, n):
        z[t] = phi_z * z[t - 1] + np.sqrt(1 - phi_z ** 2) * e_z[t]
    xi = rng.standard_normal(n)
    u = sigma_u * (xi + psi_u * np.r_[0.0, xi[:-1]])
    eta = rng.standard_normal(n)
    y = beta * z + eta + u
    f_a = a1 * z
    f_b = (beta + b_bias) * z
    e1 = y - f_a
    x = f_b - f_a
    return e1, x


def ols_hac(e1, x):
    """Slope of e1 on x (no intercept) with naive and Newey-West HAC t stats."""
    n = len(x)
    xx = float(x @ x)
    if xx < 1e-12:
        return np.nan, np.nan, np.nan
    b = float(x @ e1) / xx
    resid = e1 - b * x
    s2 = float(resid @ resid) / (n - 1)
    se_naive = np.sqrt(s2 / xx)
    # Newey-West with Bartlett kernel
    g0 = float((resid * x) @ (resid * x))
    var = g0
    for lag in range(1, L_HAC + 1):
        w = 1.0 - lag / (L_HAC + 1.0)
        gl = float(np.sum((resid[:n - lag] * x[:n - lag]) * (resid[lag:] * x[lag:])))
        var += 2.0 * w * gl
    se_hac = np.sqrt(max(var, 1e-12)) / xx
    return b, b / se_naive, b / se_hac


def reject_rate(beta, a1, b_bias, phi_z, psi_u, sigma_u, reps, rng, kind):
    cnt = 0
    for _ in range(reps):
        e1, x = gen(beta, a1, b_bias, phi_z, psi_u, sigma_u, N, rng)
        _, tn, th = ols_hac(e1, x)
        t = tn if kind == "naive" else th
        cnt += (not np.isnan(t)) and (abs(t) > Z95)
    return cnt / reps


def exp_size_scan():
    """H0 (a1=beta): slope=0. Vary serial correlation; naive size should inflate."""
    out = {"psi": [0.0, 0.3, 0.5, 0.7, 0.9], "naive": [], "hac": []}
    rng = np.random.default_rng(SEED + 1)
    for psi in out["psi"]:
        out["naive"].append(reject_rate(0.4, 0.4, 0.25, PHI_Z, psi, 1.0, REPS, rng, "naive"))
        out["hac"].append(reject_rate(0.4, 0.4, 0.25, PHI_Z, psi, 1.0, REPS, rng, "hac"))
    return out


def exp_power_scan():
    """H1 (a1=0): B has incremental info beta; power vs beta at two noise levels."""
    out = {"beta": [0.10, 0.18, 0.26, 0.34, 0.42, 0.50, 0.62],
           "hac_s1": [], "naive_s1": [], "hac_s2": []}
    rng = np.random.default_rng(SEED + 2)
    for beta in out["beta"]:
        out["hac_s1"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 1.0, REPS, rng, "hac"))
        out["naive_s1"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 1.0, REPS, rng, "naive"))
        out["hac_s2"].append(reject_rate(beta, 0.0, 0.15, PHI_Z, PSI_U, 2.0, REPS, rng, "hac"))
    return out


def exp_slope_dist():
    """Distribution of slope estimate under H0 and H1."""
    rng = np.random.default_rng(SEED + 3)
    h0, h1 = [], []
    for _ in range(REPS):
        e1, x = gen(0.4, 0.4, 0.25, PHI_Z, PSI_U, 1.0, N, rng)
        b, _, _ = ols_hac(e1, x)
        h0.append(b)
        e1b, xb = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
        bb, _, _ = ols_hac(e1b, xb)
        h1.append(bb)
    return {"h0_mean": float(np.mean(h0)), "h0_sd": float(np.std(h0)),
            "h1_mean": float(np.mean(h1)), "h1_sd": float(np.std(h1)),
            "h1_theory": 0.4 / (0.4 + 0.15), "h0": h0, "h1": h1}


def exp_corr_power():
    """As B's estimate gets noisier (larger b_bias), the encompassing slope falls as
    beta/(beta+b) while HAC power stays flat (slope and its SE shrink in the same
    proportion). The economically meaningful share of incremental information shrinks
    even though the test still rejects: statistical significance != economic value."""
    out = {"b_bias": [0.0, 0.2, 0.4, 0.7, 1.0, 1.5], "power": [], "slope": []}
    rng = np.random.default_rng(SEED + 4)
    for bb in out["b_bias"]:
        out["power"].append(reject_rate(0.4, 0.0, bb, PHI_Z, PSI_U, 1.0, REPS, rng, "hac"))
        out["slope"].append(0.4 / (0.4 + bb))
    return out


def plot1():
    rng = np.random.default_rng(SEED + 5)
    e1, x = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
    z = np.zeros(N); ez = rng.standard_normal(N); z[0] = ez[0]
    for t in range(1, N):
        z[t] = PHI_Z * z[t - 1] + np.sqrt(1 - PHI_Z ** 2) * ez[t]
    y = e1  # a1=0 -> e1 = y
    fb = 0.55 * z
    fig, axes = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
    axes[0].plot(y, lw=1, color="#2c3e50", label="y (target)")
    axes[0].plot(fb, lw=1, color="#c0392b", label="f_B (uses z)")
    axes[0].axhline(0, color="k", lw=.6)
    axes[0].set_title("Fig1: A ignores z, B uses it; B's forecast tracks the signal")
    axes[0].legend(fontsize=8, loc="upper right")
    e_b = e1 - fb  # a1=0 so e1=y; f_b=(beta+b_bias) z = 0.55 z
    axes[1].plot(e1, lw=1, color="#2980b9", label="e_A = y - 0")
    axes[1].plot(e_b, lw=1, color="#c0392b", label="e_B = y - f_B")
    axes[1].axhline(0, color="k", lw=.6)
    axes[1].set_title("Fig1: forecast errors - B removes the z component from A's error")
    axes[1].legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig1_paths.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig1_paths.png"


def plot2(sz):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    x = np.arange(len(sz["psi"])); w = .38
    ax.bar(x - w / 2, sz["naive"], w, label="naive OLS-t", color="#c0392b")
    ax.bar(x + w / 2, sz["hac"], w, label="HAC-t (Newey-West)", color="#27ae60")
    ax.axhline(0.05, color="k", ls="--", lw=1, label="5% nominal")
    ax.set_xticks(x); ax.set_xticklabels([str(v) for v in sz["psi"]])
    ax.set_xlabel("MA(1) coefficient of error psi")
    ax.set_ylabel("size (false rejection)")
    ax.set_title("Fig2: serial correlation inflates naive size; HAC restores it")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig2_size.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig2_size.png"


def plot3(pw):
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.plot(pw["beta"], pw["hac_s1"], "o-", color="#27ae60", lw=2, label="HAC, sigma=1")
    ax.plot(pw["beta"], pw["naive_s1"], "s--", color="#c0392b", lw=1.5, label="naive, sigma=1")
    ax.plot(pw["beta"], pw["hac_s2"], "^:", color="#8e44ad", lw=2, label="HAC, sigma=2")
    ax.axhline(0.05, color="k", ls="--", lw=1)
    ax.set_xlabel("B's incremental signal beta")
    ax.set_ylabel("power")
    ax.set_title("Fig3: power rises with incremental info; noise halves detection")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig3_power.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig3_power.png"


def plot4(dist):
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(dist["h0"], bins=45, density=True, alpha=.6, color="#2980b9", label="H0 slope")
    ax.hist(dist["h1"], bins=45, density=True, alpha=.6, color="#c0392b", label="H1 slope")
    ax.axvline(0, color="#2980b9", ls="--", lw=1.5)
    ax.axvline(dist["h1_theory"], color="#c0392b", ls="--", lw=1.5,
               label="theory beta/(beta+b)")
    ax.set_xlabel("encompassing slope estimate")
    ax.set_ylabel("density")
    ax.set_title("Fig4: slope centers on 0 under H0, on beta/(beta+b) under H1")
    ax.legend(fontsize=8)
    fig.tight_layout()
    p = os.path.join(_OUT, "fig4_slope_dist.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig4_slope_dist.png"


def plot5(cp):
    fig, ax1 = plt.subplots(figsize=(8.4, 4.6))
    ax1.plot(cp["b_bias"], cp["power"], "o-", color="#c0392b", lw=2, label="HAC power")
    ax1.set_xlabel("B's estimation bias b (B drifts from A)")
    ax1.set_ylabel("power", color="#c0392b")
    ax1.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax1.twinx()
    ax2.plot(cp["b_bias"], cp["slope"], "s--", color="#2980b9", lw=2,
             label="theory slope beta/(beta+b)")
    ax2.set_ylabel("encompassing slope", color="#2980b9")
    ax2.tick_params(axis="y", labelcolor="#2980b9")
    ax1.set_title("Fig5: a noisier B shrinks the slope but not the test power - significance != value")
    fig.tight_layout()
    p = os.path.join(_OUT, "fig5_corr_power.png")
    fig.savefig(p, dpi=120); plt.close(fig); return "fig5_corr_power.png"


def main():
    rng = np.random.default_rng(SEED)
    e1, x = gen(0.4, 0.0, 0.15, PHI_Z, PSI_U, 1.0, N, rng)
    _, tn, th = ols_hac(e1, x)
    sz = exp_size_scan()
    pw = exp_power_scan()
    dist = exp_slope_dist()
    cp = exp_corr_power()
    results = {
        "setup": {"seed": SEED, "n": N, "reps": REPS, "phi_z": PHI_Z,
                  "psi_u": PSI_U, "L_hac": L_HAC, "z95": Z95},
        "example_t": {"naive": float(tn), "hac": float(th)},
        "size": sz, "power": pw,
        "slope_dist": {k: v for k, v in dist.items() if k not in ("h0", "h1")},
        "corr_power": cp,
    }
    results["figs"] = [plot1(), plot2(sz), plot3(pw), plot4(dist), plot5(cp)]
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

