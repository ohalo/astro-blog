---
title: "阶梯重置期权 Cliquet：把路径依赖拆成一段段远期起点，为什么重置频率决定波动率敞口、又在跳跃下被平波动率报价系统性低估"
description: "12.8 万条路径与泊松跳跃解析解对拍：年度一次重置的阶梯看涨价 0.005293，月度重置升到 0.035174，日度重置 0.056243；按总方差折成平波动率 23.32% 的基准分别只报 0.003570、0.017179、0.038435。年度重置时上下限命中率 39.93% 与 32.78%，日度则降到 0.041% 与 0.134%。跳跃强度升到 3 时日度错价扩大到 0.066336，跳均值反号后错价翻成 -0.116036。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T15:40:00Z'
tags:
  - 量化
  - 期权定价
  - 路径依赖
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

阶梯重置期权（cliquet）的收益不是看终点在哪，而是看一年里每一段returns被敲上上下限之后能攒下多少。这个看似只在产品说明书里出现的结构，其实是理解路径依赖定价的一把好尺子：重置日越密，投资者买的就越不是「年底的涨跌」，而是「反复兑现小幅上涨、同时把大跌截断」的权利。

在一组可解析对拍的跳跃扩散参数下（S0=100、无风险利率 4%、期限一年、扩散波动率 20%、每年 0.8 次均值 -6% 的跳跃），实测结果是：

- 年度一次重置的折现价是 **0.005293**，季度（12 次）是 **0.035174**，日度（252 次）是 **0.056243**；
- 用同样的总波动率折成一个平波动率 **23.32%** 的 Black-Scholes 基准，对应只有 **0.003570**、**0.017179**、**0.038435**，季度重置低估了 **51.2%**；
- 冲击来自上下限被反复触发：年度重置的封顶命中率 **39.93%**、触底命中率 **32.78%**，日度重置分别只有 **0.041%** 和 **0.134%**；
- 跳跃强度从 0.8 提到 3.0，日度错价从 **0.017809** 扩大到 **0.066336**；把跳跃均值翻成正 6%，错价反号到 **-0.116036**。

关键结论是：重置频率改变的不是波动率数字，而是**分布形状被消费的方式**。把整条路径的方差折成一个常数波动率，能匹配二阶矩，却匹配不了每个重置区间里跳跃把价格推穿上下限的概率。频率越高，这种形状差异被兑现的次数越多。

![跳跃扩散路径与阶梯收益](/images/cliquet-reset-frequency-path-dependence/fig1_paths_clipped.png)

## 二、把路径依赖写成可加的区间收益

设一年内取 n 个等长重置区间，第 i 段的简单收益为 Rᵢ=S(tᵢ)/S(tᵢ₋₁)-1，每段收益被限制在 -f 与 +c 之间，总收益是这些截断收益的加总，再按无风险利率折现。本文取 c=f=8%。写成公式：

$$
\text{payoff}=e^{-rT}\sum_{i=1}^{n}\min\big(\max(R_i,-f),c\big).
$$

这个结构有两个容易被忽略的性质。第一，如果完全没有上下限，各段简单收益的期望之和恒等于 e^{rT}-1，与路径怎么走、波动率怎么变都无关；因此阶梯期权对波动率的敏感，**全部来自截断**：没有上下限就没有路径依赖可谈。第二，每段收益的期望只依赖该段起点的价格相对变化，因此在扩散模型里可以用一个远期起点正态分布直接算；可一旦加入跳跃，每段收益变成「正态分布被泊松计数混合」的有限混合，闭式解依然可得，但和常数波动率给出的数字不是一回事。

为了把对照做干净，基准不是随手取 20% 的波动率，而是把跳跃带来的方差率也算进去：σ_flat²=σ²+λ(μJ²+σJ²)。本例算得 **0.2332**。也就是说，基准已经「知道」跳跃让总方差变大了，只是假装收益仍然服从对数正态。这样得到的任何价差，都只能归因于分布形状，不能归因于方差没对齐。

## 三、有限混合解析解与模拟对拍

对长度为 h=T/n 的一段，设跳跃次数为 k，泊松概率为 P(k)=e^{-λh}(λh)^k/k!。在给定 k 的条件下，对数收益服从均值 mₖ=(r-λκ-σ²/2)h+kμJ、方差 sₖ²=σ²h+kσJ² 的正态分布，其中 κ=e^{μJ+σJ²/2}-1。令 a=ln(1-f)、b=ln(1+c)，则单段的截断收益期望是：

$$
E[g]= -f\Phi(\tfrac{a-m_k}{s_k}) + e^{m_k+s_k^2/2}\Big[\Phi(\tfrac{b-m_k-s_k^2}{s_k})-\Phi(\tfrac{a-m_k-s_k^2}{s_k})\Big] - \Delta\Phi + c\big(1-\Phi(\tfrac{b-m_k}{s_k})\big).
$$

对所有 k 加权求和，再乘 n 与折现因子，就是精确价格。同一批 12.8 万条日频路径被复用到所有重置频率上：先生成 252 天的跳跃扩散对数收益，再按每段的天数求和，因此不同频率之间的比较没有被抽样噪声污染。

对拍结果很干净：六种频率下蒙特卡洛价格与解析价的差距都在一个标准误之内，最大的 z 值只有 **-0.98**（4 次重置时）。日度重置的模拟价 **0.055882** 对解析价 **0.056243**，标准误 **0.000579**。这说明后面看到的频率效应不是模拟路径不够多造成的抖动。

![价格随重置频率上升](/images/cliquet-reset-frequency-path-dependence/fig2_price_vs_frequency.png)

## 四、频率把形状差异兑现了多少

价格曲线在低频区近乎线性上升：从 1 次到 12 次重置，精确价从 0.005293 涨到 0.035174，翻了 **6.6 倍**；从 12 次到 252 次只再涨 **0.021069**，曲线明显变平。平波动率基准在同样区间也从 0.003570 涨到 0.017179 再到 0.038435，但同一频率下的缺口始终存在，最大出现在 63 次重置附近，为 **0.018898**。

为什么缺口在中高频最大？看命中率就清楚了。年度区间长达一年，8% 的上下限只相当于不到半个标准差，封顶和触底分别有 **39.93%** 与 **32.78%** 的概率，截断几乎吞掉了整段收益；此时分布是尖峰还是厚尾，对期望的影响被压扁。日度区间的单段标准差只有约 1.3%，8% 的上下限远在尾部之外，单段命中率降到 **0.041%** 与 **0.134%**，但一年要兑现 252 次，跳跃造成的尾部质量仍然被一次次切下来变成确定收益。换句话说，频率不是放大了单次跳跃，而是增加了**从分布尾部提取价值的次数**。

![上下限命中率与频率](/images/cliquet-reset-frequency-path-dependence/fig3_hit_rates.png)

图中虚线是解析概率，Marker 是模拟频率，两者在六种频率上都贴合。这里也解释了另一个实务细节：报价台上把「月度重置的阶梯」说成「一年 12 份短期期权」并不算错，但前提是每份短期期权的隐含波动率必须按短期限的尾部形状单独定，而不是把年度平波动率直接套用。

## 五、跳跃敏感性、偏斜反转与分布形状

固定扩散波动率 20%，只调跳跃强度，季度（12 次）重置的错价从 λ=0.4 的 **0.000425** 涨到 λ=3.0 的 **0.003159**；日度重置则从 **0.008916** 涨到 **0.066336**，放大了约 **7.4 倍**。同一张图里把跳跃均值从 -6% 翻成 +6%，错价全部转负：λ=0.8 时季度 **-0.001484**、日度 **-0.031168**，λ=3.0 时日度达到 **-0.116036**。

这个反号很重要。它说明「平波动率低估阶梯期权」并不是普遍真理，而是负偏跳跃下的特例。正偏跳跃把尾部搬到上方，封顶先被触发，截断反而削掉了本该拿到的上涨；此时按总方差折出来的平波动率会**高估**价格。任何只记住「跳跃让看涨价更贵」的报价规则，在偏斜反转时都会站错边。

![跳跃强度与错价方向](/images/cliquet-reset-frequency-path-dependence/fig4_jump_sensitivity.png)

从整条收益分布看，日度重置的分布明显右移且变得更窄：12 次重置时总收益的均值约 0.035，252 次时约 0.056，同时极左尾被每天截断的动作削薄。年化波动率并没有消失，只是被转换成了更稳定的小额累积。这也是结构化产品中「每日观察敲出」与「到期一次性观察」定价差异巨大的根源：两者可以有相同的名义波动率，却对应完全不同的收益分布。

![总收益分布对比](/images/cliquet-reset-frequency-path-dependence/fig5_payoff_distribution.png)

## 六、边界与实践建议

这套解析对拍依赖两个假设，用之前必须检查。第一，跳跃幅度服从正态分布且各段独立同分布；真实市场的跳跃有聚集性，危机期间 λ 与 μJ 会同时恶化，此时应把参数改成随状态切换，解析式要按状态再展开一层。第二，本文的上下限固定在简单收益上；不少产品把敲出线写成相对初始价或移动均线，那样各段收益不再独立同分布，只能用数值方法逐段递推。

还有一个执行层面的坑：重置日遇到节假日会顺延，实际每段的 h 并不相等。把 252 个交易日硬切成 63 段时每段恰好 4 天，切成 52 段就会出现 4 天与 5 天混杂，若仍按 T/52 计算解析价，蒙特卡洛与解析价会系统性错开。本文特意选了能整除 252 的重置数，就是为了让对拍不受日历噪声干扰。

实操上建议把重置频率当作独立的风险因子来报：同一张期权报价单上同时列出年度、季度、日度的理论价与平波动率基准价差，再单独列出上下限命中率。前者衡量形状风险的价格，后者衡量结构被触发的频次；只看其中一个，都不足以判断产品是卖贵了还是买贵了。

## 七、完整可运行代码

下面脚本固定种子，一次生成全部数值、五张图与 results.json，可独立运行。

```python
"""Cliquet option: reset frequency, capped/floored returns and jump risk."""

def main():
    import json
    import math
    import os

    import numpy as np
    from scipy.stats import norm, poisson

    SEED = 20261128
    OUT = "/Users/halo/workspace/astro-blog/public/images/cliquet-reset-frequency-path-dependence"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    S0 = 100.0
    R = 0.04
    T = 1.0
    SIG = 0.20
    LAM = 0.8
    MUJ = -0.06
    SDJ = 0.12
    CAP = 0.08
    FLOOR = 0.08
    STEPS = 252
    PATHS = 120000
    CHUNK = 20000
    RESETS = [1, 2, 4, 12, 63, 252]

    kappa = math.exp(MUJ + 0.5 * SDJ ** 2) - 1.0
    mu_log = R - LAM * kappa - 0.5 * SIG ** 2
    dt = T / STEPS
    a_log = math.log(1.0 - FLOOR)
    b_log = math.log(1.0 + CAP)

    def analytic(lam, muj, sdj, sig, horizon):
        """Closed-form per-period expectation by Poisson mixing over jump counts."""
        kap = math.exp(muj + 0.5 * sdj ** 2) - 1.0
        drift = R - lam * kap - 0.5 * sig ** 2
        per = 0.0
        pc = 0.0
        pf = 0.0
        for k in range(16):
            w = poisson.pmf(k, lam * horizon)
            mk = drift * horizon + k * muj
            sk = math.sqrt(max(sig ** 2 * horizon + k * sdj ** 2, 1e-18))
            za = (a_log - mk) / sk
            zb = (b_log - mk) / sk
            Fa, Fb = norm.cdf(za), norm.cdf(zb)
            Da = norm.cdf(za - sk)
            Db = norm.cdf(zb - sk)
            growth = math.exp(mk + 0.5 * sk * sk)
            per += w * (-FLOOR * Fa + growth * (Db - Da) - (Fb - Fa) + CAP * (1.0 - Fb))
            pc += w * (1.0 - Fb)
            pf += w * Fa
        return per, pc, pf

    # Flat-vol benchmark: match total variance rate of the jump-diffusion log return.
    sig_flat = math.sqrt(SIG ** 2 + LAM * (MUJ ** 2 + SDJ ** 2))
    true_daily, true_cap_p, true_floor_p = analytic(LAM, MUJ, SDJ, SIG, dt)
    flat_daily, _, _ = analytic(0.0, 0.0, 1.0, sig_flat, dt)

    # One shared panel of daily log returns, reused for every reset frequency.
    log_paths = np.empty((PATHS, STEPS), dtype=np.float64)
    for start in range(0, PATHS, CHUNK):
        end = min(start + CHUNK, PATHS)
        n = end - start
        z = rng.standard_normal((n, STEPS))
        counts = rng.poisson(LAM * dt, size=(n, STEPS))
        jumps = MUJ * counts + SDJ * np.sqrt(counts) * rng.standard_normal((n, STEPS))
        log_paths[start:end] = mu_log * dt + SIG * math.sqrt(dt) * z + jumps

    def mc_payoff(n):
        step = STEPS // n
        per_log = log_paths[:, : n * step].reshape(PATHS, n, step).sum(axis=2)
        gross = np.exp(per_log) - 1.0
        clipped = np.clip(gross, -FLOOR, CAP)
        payoff = clipped.sum(axis=1) * math.exp(-R * T)
        se = payoff.std(ddof=1) / math.sqrt(PATHS)
        return payoff.mean(), se, float((gross > CAP).mean()), float((gross < -FLOOR).mean()), payoff

    mc = {}
    for n in RESETS:
        mean, se, cp, fp, p = mc_payoff(n)
        mc[n] = {"price": float(mean), "se": float(se), "cap_hit": cp, "floor_hit": fp}
        if n == 12:
            payoff_12 = p
        if n == 252:
            payoff_252 = p

    prices = {
        "jd_analytic": {},
        "flat_analytic": {},
        "mc": {str(n): mc[n]["price"] for n in RESETS},
    }
    hit_exact = {}
    for n in RESETS:
        h = T / n
        per, cp, fp = analytic(LAM, MUJ, SDJ, SIG, h)
        flat_p, _, _ = analytic(0.0, 0.0, 1.0, sig_flat, h)
        prices["jd_analytic"][str(n)] = float(math.exp(-R * T) * n * per)
        prices["flat_analytic"][str(n)] = float(math.exp(-R * T) * n * flat_p)
        hit_exact[str(n)] = {"cap": cp, "floor": fp}
    gap = {str(n): prices["jd_analytic"][str(n)] - prices["flat_analytic"][str(n)] for n in RESETS}

    # Jump sensitivity: vary intensity with either negative or positive jump mean.
    sens = {}
    for sign, muj in (("negative", MUJ), ("positive", -MUJ)):
        rows = {}
        for lam in [0.0, 0.4, 0.8, 1.6, 3.0]:
            sig_flat_l = math.sqrt(SIG ** 2 + lam * (muj ** 2 + SDJ ** 2))
            rows[str(lam)] = {}
            for n in (12, 252):
                pp, _, _ = analytic(lam, muj, SDJ, SIG, T / n)
                ff, _, _ = analytic(0.0, 0.0, 1.0, sig_flat_l, T / n)
                rows[str(lam)][f"n{n}"] = float(math.exp(-R * T) * n * (pp - ff))
        sens[sign] = rows

    results = {
        "params": {"S0": S0, "r": R, "T": T, "sigma": SIG, "lambda": LAM, "muJ": MUJ,
                   "sdJ": SDJ, "cap": CAP, "floor": FLOOR, "paths": PATHS, "seed": SEED,
                   "sigma_flat": float(sig_flat), "kappa": float(kappa)},
        "per_period": {"jd_daily": float(true_daily), "flat_daily": float(flat_daily),
                       "cap_prob_daily": float(true_cap_p), "floor_prob_daily": float(true_floor_p),
                       "hit_by_n": hit_exact},
        "prices": prices,
        "gap": gap,
        "mc": {str(n): mc[n] for n in RESETS},
        "sensitivity": sens,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: representative paths and the clipped increments they generate.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    idxs = [7, 41, 88]
    step12 = STEPS // 12
    for j, idx in enumerate(idxs):
        s = S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_paths[idx])]))
        axes[0].plot(np.arange(STEPS + 1), s, lw=1.2, label=f"path {j + 1}")
        per_log = log_paths[idx, :12 * step12].reshape(12, step12).sum(axis=1)
        per_ret = np.exp(per_log) - 1.0
        axes[1].vlines(np.arange(1, 13) * step12, 0, np.clip(per_ret, -FLOOR, CAP),
                       alpha=0.75, label=f"path {j + 1}")
    axes[0].set_ylabel("underlying price")
    axes[0].set_title("Jump-diffusion paths with 12 reset dates")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(CAP, color="red", ls="--", lw=0.8)
    axes[1].axhline(-FLOOR, color="green", ls="--", lw=0.8)
    axes[1].set_ylabel("capped increment")
    axes[1].set_xlabel("trading day")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_paths_clipped.png"), dpi=150)
    plt.close(fig)

    # fig2: price vs reset frequency.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ns = np.array(RESETS, dtype=float)
    jd = np.array([prices["jd_analytic"][str(n)] for n in RESETS])
    fl = np.array([prices["flat_analytic"][str(n)] for n in RESETS])
    mp = np.array([mc[n]["price"] for n in RESETS])
    ms = np.array([mc[n]["se"] for n in RESETS])
    ax.plot(ns, jd, "o-", color="#4c78a8", label="jump-diffusion exact")
    ax.plot(ns, fl, "s--", color="#e45756", label="flat-vol benchmark")
    ax.errorbar(ns, mp, yerr=2 * ms, fmt="^", color="black", capsize=3, label="Monte Carlo +/-2SE")
    ax.set_xscale("log")
    ax.set_xlabel("number of resets per year")
    ax.set_ylabel("discounted cliquet price")
    ax.set_title("Reset frequency changes the price of path dependence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_vs_frequency.png"), dpi=150)
    plt.close(fig)

    # fig3: hit probabilities and expected annual counts.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax = axes[0]
    ax.plot(ns, [mc[n]["cap_hit"] for n in RESETS], "o", color="#e45756", label="MC cap hit")
    ax.plot(ns, [mc[n]["floor_hit"] for n in RESETS], "s", color="#2ca02c", label="MC floor hit")
    ax.plot(ns, [hit_exact[str(n)]["cap"] for n in RESETS], "--", color="#e45756", lw=1.2)
    ax.plot(ns, [hit_exact[str(n)]["floor"] for n in RESETS], "--", color="#2ca02c", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("resets per year")
    ax.set_ylabel("per-period hit probability")
    ax.set_title("Monte Carlo (markers) vs exact (dashed)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(ns, [12 * mc[n]["cap_hit"] for n in RESETS], "o-", color="#e45756", label="cap events/year")
    ax.plot(ns, [12 * mc[n]["floor_hit"] for n in RESETS], "s-", color="#2ca02c", label="floor events/year")
    ax.set_xscale("log")
    ax.set_xlabel("resets per year")
    ax.set_ylabel("expected events per year (12-period scale)")
    ax.set_title("Frequent resets multiply barrier touches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_hit_rates.png"), dpi=150)
    plt.close(fig)

    # fig4: mispricing against jump intensity, with both skew signs.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    lams = [0.0, 0.4, 0.8, 1.6, 3.0]
    for sign, color in (("negative", "#4c78a8"), ("positive", "#e45756")):
        for key, ls in (("n12", "-"), ("n252", "--")):
            ax.plot(lams, [sens[sign][str(l)][key] for l in lams], color=color, ls=ls,
                    marker="o", label=f"{sign} skew, {key}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("jump intensity lambda")
    ax.set_ylabel("exact price minus flat-vol price")
    ax.set_title("A flat-vol quote can err in either direction")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_jump_sensitivity.png"), dpi=150)
    plt.close(fig)

    # fig5: realised payoff distributions.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.hist(payoff_12, bins=70, density=True, alpha=0.55, color="#4c78a8", label="12 resets")
    ax.hist(payoff_252, bins=70, density=True, alpha=0.45, color="#e45756", label="252 resets")
    ax.axvline(prices["jd_analytic"]["12"], color="#4c78a8", ls="--", lw=1.4)
    ax.axvline(prices["jd_analytic"]["252"], color="#e45756", ls="--", lw=1.4)
    ax.set_xlabel("discounted total payoff")
    ax.set_ylabel("density")
    ax.set_title("Reset schedule reshapes the whole payoff distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_payoff_distribution.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```


## 八、结语

阶梯重置期权最值得带走的一点，是它把「波动率是一个数」这句话拆穿了。只要收益被截断、被观察、被分段兑现，分布的形状就变成了可以直接计价的东西。重置频率越高，越多的尾部质量被加工成确定收益；平波动率模型因为只保留二阶矩，必然把这部分价值记错账，且错的方向取决于偏斜的符号。

这也给日常风控提了个醒：当两个策略的夏普与年化波动看起来一样时，先看它们的收益是被一次性观察还是逐日兑现。路径依赖不会写在波动率数字里，它写在每一次重置的那一刻。

