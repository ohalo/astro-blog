---
title: "Bachelier 正态波动率定价：负利率与负油价下为什么正态模型比 Black-Scholes 更自然，平价自检、微笑形状与移位对数正态的边界"
description: "S0=100、r=4%、T=1 下把正态波动率 σN=FσL 与 lognormal σL=20% 做平价匹配：ATM 价 7.97885 对 7.96557，K=140 时 0.34361 对 0.70505，最大价差 0.45573。用对数正态价格反解正态隐含波动率，70 到 130 执行价上从 17.155 升到 23.273；反向反解的对数正态隐含波动率从 0.2429 降到 0.1788。F=1、σN=1.6 的近零远期上允许负执行价，四种移位对数正态同时校准 ATM 后，K=0 的价差仍达 0.12364。30 万路径蒙特卡洛六个价格 z 值绝对值均小于 1.6。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T16:55:00Z'
tags:
  - 量化
  - 期权定价
  - 波动率
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

2020 年原油期货出现负结算价之后，很多原本默认「标的价格永远为正」的定价代码直接失效。对数正态分布的定义域把负价格挡在门外，而 Bachelier 在 1900 年给出的正态模型从一开始就把标的价格写成「远期加减一个正态扰动」，天然允许负值。它并不是粗糙的近似，而是利率、商品与信用利差这类可能穿越零的标的的标准工具。

本文用一组可精确对拍的数字说明三件事。第一，正态与对数正态在 ATM 附近可以几乎重合，但在两档尾部立刻分叉：S0=100、r=4%、T=1、σL=20%，取 σN=FσL 后两模型 ATM 价是 **7.97885** 与 **7.96557**，而 K=140 时变成 **0.34361** 对 **0.70505**，最大绝对价差 **0.45573**。第二，交叉反解隐含波动率会造出方向相反的微笑：用对数正态价格反解正态波动率，执行价从 70 到 130 时隐含值从 **17.155** 单调升到 **23.273**；反向从正态价格反解对数正态波动率，则从 **0.2429** 单调降到 **0.1788**。第三，移位对数正态看起来能救负价格，但四个移位参数全部校准到同一个 ATM 价 **0.61328** 之后，在 K=0 上仍然给出 **0.12364** 的价格离散。

![同一远期与 ATM 波动率下的终端密度](/images/bachelier-normal-volatility-pricing/fig1_terminal_density.png)

## 二、模型设定与平价自检

记贴现因子 D=e^{-rT}，远期 F=S0e^{rT}。Bachelier 假设远期价格服从 F+σN√T Z，其中 Z 是标准正态；于是看涨价格为：

$$
C_B=D\big[(F-K)\Phi(d)+\sigma_N\sqrt{T}\varphi(d)\big],\quad d=\frac{F-K}{\sigma_N\sqrt{T}}.
$$

看跌价把 d 换成 -d 并把 F-K 换成 K-F 即可。对数正态模型则是 S_T=S0exp[(r-σL²/2)T+σL√T Z]，看涨价用标准的 d₁、d₂ 表达式。两个模型必须同时通过同一条无套利平价：

$$
C-P=D(F-K).
$$

在本文的六个执行价上，Bachelier 的平价最大残差是 **7.1e-15**，Black-Scholes 是 **1.4e-14**，都在浮点误差量级。这一步看似 trivial，却是最容易抓出实现错误的自检：一旦把贴现因子漏掉，或者把 F 误写成 S0，残差会立刻放大到几分钱。

把 σN 设成 FσL 是业界常见的 ATM 换算。它的来源是两者 ATM 价格的一阶展开都等于 S0σL√(T/2π)，但只是渐近相等，不是恒等。所以后文所有比较都不是「同一个模型换了参数写法」，而是两个在 ATM 匹配、其余位置允许分叉的模型。

平价自检虽然能通过，却抓不到另一类更隐蔽的实现错误：把换算里的远期误写成即期。若把 σN 设成 S0σL=20 而不是 FσL=20.81622，同一执行价 F 上的正态价会从 **7.97885** 掉到 **7.66599**，凭空少了 **0.31286**，占 ATM 价格的 **3.92%**。这类错误在平价检验里完全隐形，因为 call 与 put 的差仍然正确，只是两边同时被压低。实践中建议在价格函数之外单独维护一份「ATM 定标」测试：给定 σL，用两个模型在 K=F 上互标，误差超过千分之一就拒绝上线。

## 三、ATM 匹配后价格在哪里分叉

执行价图显示出一个很干净的结构：在 F=104.08 附近两条曲线几乎贴在一起，向左走（深度实值）差距温和扩大，向右走（深度虚值）差距急剧拉开。K=80 时两价是 **24.36190** 与 **23.90616**，差 **0.45573**；K=120 时是 **2.55720** 与 **2.99995**，差 **0.44275**；K=140 时相对差距已经超过一倍。

![两条 ATM 匹配的价格曲线](/images/bachelier-normal-volatility-pricing/fig2_price_curves.png)

原因藏在终端密度里。对数正态的右尾更厚，因此高执行价的虚值看涨更贵；它的左尾被零价截断，因此深度实值看涨相对便宜。正态分布的尾部是对称的，但以 F 为中心可有负值，所以它在两端都给出非零质量。若交易的是执行价远离远期的数字期权或价外看涨，用 ATM 波动率换算过去报价，误差不是几个基点，而可能是价格本身的百分之几十。

往实值方向看，K=60 的两价是 **42.47584** 与 **42.36605**，差距只有 **0.10979**，远小于虚值端。这不是因为左尾更接近，而是因为实值看涨的价格主要由内在价值 F-K 主导，尾部差异只占其中一小块；一旦把期权改成数字看涨，内在价值这项常数消失，尾部差异就会重新变成全部价格。K=140 的普通看涨差 **0.36145**，换算成同一档数字看涨时差异会更加醒目。因此我们在检查模型风险时，不能只看普通看涨是否接近，还要看 payoff 对尾部质量是否敏感：数字期权、价差组合与区间累计结构都属于这一类。

## 四、交叉反解出的微笑方向相反

实务里更常见的场景不是比较两个模型，而是拿一种报价惯例波动率去标另一种模型的价格。我们把 Black-Scholes 在 σL=20% 生成的价格拿去反解 Bachelier 波动率，再把 Bachelier 在 σN=FσL 生成的价格拿去反解对数正态波动率。

$$
\sigma_{N,\text{imp}}(K):\; C_B(K,\sigma)=C_{BS}(K,0.20).
$$

结果是两条方向相反、但同样单调的微笑。正态隐含值在 K=70 是 **17.155**，到 K=130 升至 **23.273**，全宽 **6.1185** 个波动率点；对数正态隐含值在 K=70 是 **0.2429**，到 K=130 降至 **0.1788**，全宽 **0.0641**。也就是说，若一个交易台用正态波动率报价、另一个用对数正态波动率报价，即便两边对同一批成交价都做了隐含波动率校准，他们看到的微笑斜率符号也会相反。把「波动率上升」直接解释成行情变热，在这两种坐标下含义完全不同。

![互为镜像的两条隐含波动率微笑](/images/bachelier-normal-volatility-pricing/fig3_cross_smiles.png)

## 五、近零远期：负执行价与移位的代价

令远期 F=1、正态波动率 σN=1.6，这时终端价格有相当概率落在零以下，执行价 K=-0.4 与 K=0 都是合法报价。Bachelier 公式直接给出看涨价 **1.50670** 与 **1.20970**。对数正态模型对 K≤0 连对数项都写不出来，常见补救是移位：把价格整体加上 a，用 F+a 与 K+a 做对数正态定价。

$$
C_{\text{shift}}(K,a)=D\big[(F+a)\Phi(d_1)-(K+a)\Phi(d_2)\big].
$$

我们取 a=0.6、1、2、5，并对每个 a 单独校准 σshift，使 K=F 处价格精确等于正态价 **0.61328**。四个模型在 ATM 上完全一致，但离开ATM后立刻分家：K=-0.4 的价格从 **1.34958** 到 **1.45836**，极差 **0.10878**；K=0 从 **1.18446** 到 **1.30810**，极差 **0.12364**。而且即使取最大的移位 5，K=-0.4 的价格 **1.45836** 仍明显低于正态模型的 **1.50670**。结论是：移位参数本身就是一个未被市场报价识别的自由度，校准一个执行价并不能约束其他执行价，负价格区域尤其如此。

![近零远期上四种移位的分叉](/images/bachelier-normal-volatility-pricing/fig4_negative_forward.png)

## 六、σN=FσL 什么时候不够用

把 σL 从 5% 扫到 80%，每个档位都取 K=F，比较换算后的正态价与对数正态价。σL=5% 时相对误差只有 **0.0104%**，20% 时 **0.1667%**，40% 时 **0.6671%**，到 80% 已经升到 **2.6735%**。误差大致按波动率平方增长：这是一阶换算的固有代价，高波动商品或危机中的股指期权不能忽略。

![换算误差与蒙特卡洛对拍](/images/bachelier-normal-volatility-pricing/fig5_conversion_and_mc.png)

最后用 30 万条固定种子的路径对两个闭式解做对拍。正态模型在 K=80、104.08、120 的蒙特卡洛价分别是 **24.31110**、**7.95515**、**2.54932**，对应解析价 **24.36190**、**7.97885**、**2.55720**，六个价格（含对数正态）的标准化误差绝对值最大只有 **1.553**，全部落在两个标准误以内。代码里的随机数种子、路径数与输出字段都是固定的，复跑会逐字得到同一份结果。

## 七、可交易的含义

第一，报价坐标的选择会改变你对微笑方向的判断，跨交易台比较波动率前必须先统一模型口径。第二，执行价离远期越远，ATM 换算越不可靠，尤其虚值看涨：本文 K=140 的相对误差超过一倍，足以让一个基于对数正态外推的报价完全失去意义。第三，遇到负价格或近零远期，移位对数正态的移位参数要当成风险因子来管理：至少用多个执行价联合校准，并报告价格对移位的敏感性，而不是只报一个「校准好了」的波动率。

对冲上也要换尺子。正态模型的 vega 是对 σN 求导，对数正态模型的 vega 是对 σL 求导，两者在 ATM 附近相差约一个远期因子 F；若风险系统把两个台的 vega 直接相加，数值上看起来对冲干净，实际上隐含波动率一移动，两个模型的损益就会朝不同方向走。更稳妥的做法是把所有头寸统一映射到一个报价坐标，再计算组合敏感度；本文的交叉反解微笑正说明，这个映射本身会随执行价变化，不能用一个常数了事。

完整实验脚本如下，包含五个图的生成与 results.json 输出。

```python
"""Bachelier normal-vol pricing: parity, smile inversion, negative strikes and shift sensitivity."""

def main():
    import json
    import math
    import os

    import numpy as np
    from scipy.optimize import brentq
    from scipy.stats import norm

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "/Users/halo/workspace/astro-blog/public/images/bachelier-normal-volatility-pricing"
    os.makedirs(OUT, exist_ok=True)

    S0 = 100.0
    R = 0.04
    T = 1.0
    DF = math.exp(-R * T)
    F = S0 * math.exp(R * T)
    SIG_L = 0.20
    SIG_N = F * SIG_L
    STRIKES = [60.0, 80.0, 100.0, F, 120.0, 140.0]

    def bach_call(k, sn):
        d = (F - k) / (sn * math.sqrt(T))
        return DF * ((F - k) * norm.cdf(d) + sn * math.sqrt(T) * norm.pdf(d))

    def bach_put(k, sn):
        d = (F - k) / (sn * math.sqrt(T))
        return DF * ((k - F) * norm.cdf(-d) + sn * math.sqrt(T) * norm.pdf(d))

    def bs_call(k, sl):
        if k <= 0:
            return float("nan")
        d1 = (math.log(S0 / k) + (R + 0.5 * sl * sl) * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return S0 * norm.cdf(d1) - DF * k * norm.cdf(d2)

    def bs_put(k, sl):
        if k <= 0:
            return float("nan")
        d1 = (math.log(S0 / k) + (R + 0.5 * sl * sl) * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return DF * k * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    def shift_call(k, sl, a):
        return bs_call(k + a, sl)

    # 1) exact parity for both models.
    parity = {
        "bach_max_abs": max(abs(bach_call(k, SIG_N) - bach_put(k, SIG_N) - DF * (F - k)) for k in STRIKES),
        "bs_max_abs": max(abs(bs_call(k, SIG_L) - bs_put(k, SIG_L) - DF * (F - k)) for k in STRIKES),
    }

    # 2) ATM-matched price curves.
    prices = {
        "strike": STRIKES,
        "bachelier": [bach_call(k, SIG_N) for k in STRIKES],
        "lognormal": [bs_call(k, SIG_L) for k in STRIKES],
    }
    abs_diff = [abs(a - b) for a, b in zip(prices["bachelier"], prices["lognormal"])]
    kgrid = np.linspace(60.0, 140.0, 161)

    # 3) cross-model smile inversion: each model prices, the other model inverts.
    k_smile = [70.0, 80.0, 90.0, 100.0, F, 110.0, 120.0, 130.0]
    normal_iv = [brentq(lambda s: bach_call(k, s) - bs_call(k, SIG_L), 0.01, 400.0) for k in k_smile]
    lognormal_iv = [brentq(lambda s: bs_call(k, s) - bach_call(k, SIG_N), 1e-4, 4.0) for k in k_smile]
    smile = {
        "strike": k_smile,
        "normal_iv_from_lognormal": normal_iv,
        "lognormal_iv_from_normal": lognormal_iv,
        "normal_iv_spread": max(normal_iv) - min(normal_iv),
        "lognormal_iv_spread": max(lognormal_iv) - min(lognormal_iv),
    }

    # 4) near-zero forward: negative strikes are legal in Bachelier, not in lognormal.
    F2 = 1.0
    SN2 = 1.6
    DF2 = DF
    neg_strikes = [-0.4, -0.2, 0.0, 0.5, F2, 2.0]
    shifts = [0.6, 1.0, 2.0, 5.0]

    def bach_call_f(k, fwd, sn):
        d = (fwd - k) / (sn * math.sqrt(T))
        return DF2 * ((fwd - k) * norm.cdf(d) + sn * math.sqrt(T) * norm.pdf(d))

    def shifted_price(k, a, sl):
        d1 = (math.log((F2 + a) / (k + a)) + 0.5 * sl * sl * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return DF2 * ((F2 + a) * norm.cdf(d1) - (k + a) * norm.cdf(d2))

    # Calibrate every shifted model to the same at-the-forward price.
    atm_normal = bach_call_f(F2, F2, SN2)
    shift_sl = {a: brentq(lambda s, aa=a: shifted_price(F2, aa, s) - atm_normal, 1e-4, 10.0)
                for a in shifts}
    shift_grid = {str(a): [shifted_price(k, a, shift_sl[a]) for k in neg_strikes] for a in shifts}
    shift_atm_err = {str(a): abs(shifted_price(F2, a, shift_sl[a]) - atm_normal) for a in shifts}
    neg = {
        "strikes": neg_strikes,
        "bachelier": [bach_call_f(k, F2, SN2) for k in neg_strikes],
        "shifted": shift_grid,
        "shift_atm_error": shift_atm_err,
        "spread_k0": max(v[1] for v in shift_grid.values()) - min(v[1] for v in shift_grid.values()),
        "spread_kneg": max(v[0] for v in shift_grid.values()) - min(v[0] for v in shift_grid.values()),
        "atm_normal": atm_normal,
    }

    # 5) ATM normal/lognormal conversion error grows with volatility.
    vols = [0.05, 0.10, 0.20, 0.40, 0.80]
    conv = []
    for sl in vols:
        p_bs = bs_call(F, sl)
        p_ba = bach_call(F, F * sl)
        conv.append({"sigma_log": sl, "bs": p_bs, "bachelier": p_ba, "rel_error": (p_ba - p_bs) / p_bs})

    # 6) Monte Carlo cross-check of both closed forms (fixed seed).
    rng = np.random.default_rng(20261128)
    paths = 300000
    zn = rng.standard_normal(paths)
    st_normal = F + SIG_N * math.sqrt(T) * zn
    st_log = F * np.exp(-0.5 * SIG_L * SIG_L * T + SIG_L * math.sqrt(T) * zn)
    mc = {}
    for k in (80.0, F, 120.0):
        pn = np.maximum(st_normal - k, 0.0)
        pl = np.maximum(st_log - k, 0.0)
        mc[str(k)] = {
            "normal": DF * float(pn.mean()),
            "normal_se": DF * float(pn.std(ddof=1)) / math.sqrt(paths),
            "log": DF * float(pl.mean()),
            "log_se": DF * float(pl.std(ddof=1)) / math.sqrt(paths),
        }

    results = {
        "params": {"S0": S0, "r": R, "T": T, "DF": DF, "F": F, "sigma_log": SIG_L,
                   "sigma_normal": SIG_N, "paths": paths},
        "parity": parity,
        "prices": prices,
        "max_abs_price_diff": max(abs_diff),
        "smile": smile,
        "negative_forward": neg,
        "atm_conversion": conv,
        "monte_carlo": mc,
    }

    # fig1: terminal density, normal vs lognormal at matched ATM volatility.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.linspace(-20, 180, 700)
    dens_n = norm.pdf(x, loc=F, scale=SIG_N * math.sqrt(T))
    dens_l = np.zeros_like(x)
    pos = x > 0
    dens_l[pos] = np.exp(norm.logpdf(np.log(x[pos]), math.log(F) - 0.5 * SIG_L ** 2 * T,
                                     SIG_L * math.sqrt(T))) / x[pos]
    ax.plot(x, dens_n, color="#4c78a8", lw=2, label="Bachelier normal")
    ax.plot(x, dens_l, color="#e45756", lw=2, label="GBM lognormal")
    ax.axvline(120, color="black", ls="--", lw=1, label="K = 120")
    ax.set_xlabel("terminal forward price")
    ax.set_ylabel("density")
    ax.set_title("Same forward and ATM volatility, different tails")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_terminal_density.png"), dpi=150)
    plt.close(fig)

    # fig2: price curves and parity residuals.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(kgrid, [bach_call(float(k), SIG_N) for k in kgrid], color="#4c78a8", label="Bachelier")
    axes[0].plot(kgrid, [bs_call(float(k), SIG_L) for k in kgrid], color="#e45756", label="Black-Scholes")
    axes[0].axvline(F, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("strike K")
    axes[0].set_ylabel("discounted call price")
    axes[0].set_title("ATM-matched, not strike-matched")
    axes[0].legend(fontsize=9)
    axes[1].bar(range(len(STRIKES)), abs_diff, color="#72b7b2")
    axes[1].set_xticks(range(len(STRIKES)), [str(int(k)) for k in STRIKES], rotation=30)
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("absolute price gap")
    axes[1].set_title("Largest gap away from the forward")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_curves.png"), dpi=150)
    plt.close(fig)

    # fig3: implied vol smiles seen by the other model.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(k_smile, normal_iv, "o-", color="#4c78a8")
    axes[0].axhline(SIG_N, color="gray", ls="--", lw=1, label="true normal vol")
    axes[0].set_xlabel("strike K")
    axes[0].set_ylabel("implied normal vol")
    axes[0].set_title("Lognormal prices -> normal smile")
    axes[0].legend(fontsize=9)
    axes[1].plot(k_smile, lognormal_iv, "o-", color="#e45756")
    axes[1].axhline(SIG_L, color="gray", ls="--", lw=1, label="true log vol")
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("implied lognormal vol")
    axes[1].set_title("Normal prices -> lognormal smile")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_cross_smiles.png"), dpi=150)
    plt.close(fig)

    # fig4: near-zero forward and shifted-lognormal instability.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(neg_strikes, neg["bachelier"], "ko-", lw=2, label="Bachelier (K may be <= 0)")
    for a in shifts:
        ax.plot(neg_strikes, shift_grid[str(a)], "o-", lw=1.4, label=f"shifted lognormal, a={a}")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("strike K")
    ax.set_ylabel("discounted call price")
    ax.set_title("All shifts match ATM, then disagree between strikes")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_negative_forward.png"), dpi=150)
    plt.close(fig)

    # fig5: ATM conversion error and Monte Carlo validation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(vols, [c["rel_error"] for c in conv], "o-", color="#e45756")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("lognormal volatility")
    axes[0].set_ylabel("relative ATM pricing error")
    axes[0].set_title("sigma_N = F sigma_L is only a first-order rule")
    ks = sorted(mc, key=float)
    exact_n = [bach_call(float(k), SIG_N) for k in ks]
    exact_l = [bs_call(float(k), SIG_L) for k in ks]
    axes[1].errorbar(ks, [mc[k]["normal"] for k in ks], yerr=[2 * mc[k]["normal_se"] for k in ks],
                     fmt="o", capsize=4, color="#4c78a8", label="normal MC")
    axes[1].plot(ks, exact_n, "s--", color="#4c78a8", label="normal exact")
    axes[1].errorbar(ks, [mc[k]["log"] for k in ks], yerr=[2 * mc[k]["log_se"] for k in ks],
                     fmt="o", capsize=4, color="#e45756", label="log MC")
    axes[1].plot(ks, exact_l, "s--", color="#e45756", label="log exact")
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("discounted call price")
    axes[1].set_title("300k-path MC vs closed forms (2 SE bars)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_conversion_and_mc.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

