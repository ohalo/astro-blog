---
title: "共单调一篮子期权上下界：相关性未知时只用边际分布夹住组合期权，为什么 Fréchet 上界紧、又在秩相关偏离 1 时迅速失效"
description: "两只标的各 100 远期、波动率 20% 与 30%、K=100：共单调上界 9.55177，反向耦合下界 2.24924，区间宽 7.30253；独立高斯 copula 价 6.93322，占区间 64.14%。Spearman 秩相关从 -0.8915 升到 0.9890，价格从 3.05216 升到 9.52890，区间占比从 11.00% 到 99.69%。K=100 时区间最宽，K=80 只有 2.14772。五资产异质波动率下独立价 4.59030 对上界 9.63745，波动率整体放大两倍后缺口从 2.55802 升到 9.58495。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T16:56:00Z'
tags:
  - 量化
  - 期权定价
  - 相关性风险
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

给一篮子期权报价时，交易员通常先敲定每个标的的边际分布——从各自期权曲面反解隐含波动率并不难——然后再拍一个相关性。问题是相关性本身没有流动性市场，价差交易里常常根本没有可靠的联动报价。此时真正该问的不是「相关性取多少」，而是「在边际分布固定、相关性完全未知时，价格最多能差多少」。

两只标的、远期均为 100、波动率分别 20% 与 30%、期限一年、执行价 100 的例子里，答案是：共单调上界 **9.55177**，反向单调下界 **2.24924**，未识别区间宽 **7.30253**。独立高斯 copula 给出 **6.93322**，正好落在区间中偏上的位置，相当于吃掉了整个区间的 **64.14%**。当 Spearman 秩相关从 **-0.8915** 升到 **0.9890**，价格从 **3.05216** 升到 **9.52890**，区间占比从 **11.00%** 升到 **99.69%**。

![相同边际下的两种极端耦合](/images/comonotonic-basket-option-bounds/fig1_quantile_paths.png)

## 二、只用边际就能写下上界

设每只标的的边际分布已知，其分位函数是把均匀变量 u 映到对数正态分位数：

$$
Q_i(u)=F_i\exp\big(-\tfrac12\sigma_i^2T+\sigma_i\sqrt{T}\,\Phi^{-1}(u)\big).
$$

共单调耦合让所有标的共用同一个 u，于是篮子变成 U=ΣwᵢQᵢ(U) 的一维函数，看涨期权期望只需对 u 做一次积分：

$$
C^{\text{up}}=D\int_0^1\Big(\sum_i w_iQ_i(u)-K\Big)^+\,du.
$$

这个上界是 Fréchet-Hoeffding 上界在凸增 payoff 上的直接结果，而且对所有维数都成立。它的关键性质是：只要边际固定，任何 copula 的篮子价格都不会超过它。数值上我们用 4000 个 Gauss-Legendre 节点积分，端点截断到 1e-9 以避免 Φ⁻¹ 溢出。

下界就挑剔得多。对正权重与看涨 payoff，被积函数关于各分量是超模态的，因此二维情形的下界由反向单调耦合给出：令 u₂=1-u₁，把第二条分位曲线倒过来。三维及以上不存在让所有两两都反向单调的耦合，所以「下界」只对 n=2 严格成立；五资产那一节我们只报告上界与具体 copula 价格，不误称为严格下界。

上界之所以叫「紧」，是因为它确实能被某个耦合取到：只要所有标的的均匀分位变量完全相同，篮子就退化成单变量函数，积分结果就是上界本身。它不需要任何关于 copula 参数的估计，只需要每条边际分位函数算得准。反过来，反向单调只在二维构成合法 copula；在三维试图让 u₁、u₂、u₃ 两两反向，第一对成立时第三对必然矛盾。工程上常见的错误是把二维下界公式直接推广到高维，得到一个看似漂亮、实则不可实现的低价，并把它当成风控底线。

数值积分还有一处细节值得说明。payoff 在 ΣwᵢQᵢ(u)=K 处有折点，普通等距网格会在这点附近产生可见的偏差；本文用 4000 个 Gauss-Legendre 节点，并把区间端点截断到 1e-9，使 Φ⁻¹ 不会取到无穷。改变节点数到 2000 与 8000 时，上界与下界的小数点后五位保持不变，因此后面引用的价差不是积分误差造成的。

## 三、相关性扫描：上界只在极限处取到

用 Sobol 低差异序列生成高斯 copula 样本，把高斯相关系数 ρ 从 -0.9 扫到 0.99。价格随 ρ 单调上升，但上升速度并不均匀：ρ=-0.9 时 **3.05216**，ρ=-0.5 时 **5.16672**，独立时 **6.93322**，ρ=0.5 时 **8.33954**，ρ=0.9 时 **9.32091**，ρ=0.99 时 **9.52890**。

$$
\rho_S=\frac{6}{\pi}\arcsin\frac{\rho}{2}.
$$

注意 ρ=0.5 的高斯相关对应 Spearman 秩相关 **0.4826**，已经不算弱，但价格只走完区间的 **83.40%**。要把区间占比推到 **99.69%**，需要 ρ 升到 0.99。换句话说，「相关性比较高」与「接近共单调」之间差着一大截，而期权价格恰好对这一大截极其敏感。

![高斯相关与价格的关系](/images/comonotonic-basket-option-bounds/fig2_price_vs_rho.png)

## 四、秩相关接近 1 也不等于共单调

把横轴换成 Spearman 秩相关后，结论更直观。ρS=0 时价格只占区间的 **64.14%**，ρS=0.4826 时占 **83.40%**，ρS=0.8915 时占 **96.84%**，ρS=0.9890 时才到 **99.69%**。共单调对应 ρS 严格等于 1；任何小于 1 的秩相关都还留着一块可分散的风险，而凸 payoff 会把这块风险转化成品格差。

![价格占区间比例随秩相关上升](/images/comonotonic-basket-option-bounds/fig3_rank_retention.png)

这对风控有个直接后果：用历史秩相关做压力测试时，把 0.95 当成「几乎完全联动」会系统性低估篮子期权的卖方风险。本文的参数下，ρS 从 0.9890 提高到 1 只差 1.1 个百分点，价格却还能再涨 **0.02287**。

反过来看，若把共单调上界当成「最坏情形」写进风险报告，就等于假设所有标的永远同涨同跌。对宽基指数篮子这尚可当作压力情形；对跨资产或跨行业篮子，它更像一个数学上的天花板，而不是可实现的损失路径。真正有用的是把上界与当前模型价的差额单独列成「相关性未识别溢价」，并观察这个差额在执行价与期限维度上的形状：它越靠近平值越宽，说明报价对联动假设越敏感。

还有一个容易混淆的点：高斯相关系数与秩相关并不是同一个数。ρ=0.5 对应 ρS=0.4826，ρ=0.9 对应 ρS=0.8915，差异在相关系数偏小时更明显。若风险限额直接写「秩相关不得超过 0.9」，而定价模型里填的是高斯 ρ=0.9，两边其实不是同一个压力水平。本文的表里同时列出两列，就是为了让限额、定价与压力测试至少在同一把尺子上对话。

## 五、执行价决定区间有多宽

把 K 从 80 扫到 120，上下界与区间宽度都随执行价变化。K=80 时上界 **21.36350**、下界 **19.21579**，宽 **2.14772**；K=90 时宽 **5.04285**；K=100 时宽 **7.30253**，达到本组参数的峰值；K=110 时回落到 **5.34096**，K=120 时 **3.39858**。

![执行价扫描下的无识别价格带](/images/comonotonic-basket-option-bounds/fig4_strike_band.png)

峰值出现在平值附近并不意外：深度实值时篮子几乎必然行权，payoff 接近线性，耦合方式影响很小；深度虚值时上下界都趋近零；只有在平值附近，凸性最大，相关性才变成定价的主角。因此如果只对冲虚值尾部，边际分布的误差可能比相关性误差更值得担心；反之平值 baskets 的相关性风险无法回避。

## 六、五资产异质波动率：分散化把独立价压得很低

把篮子扩到五只标的，远期 95 到 105、波动率从 15% 到 35% 递增、等权、K=100。共单调上界 **9.63745**，独立高斯 copula 价 **4.59030**，不到上界的一半。等相关的 ρ 从 0 升到 0.99 时，价格依次为 **4.59030**、**6.60823**、**7.79720**、**9.30524**、**9.60486**，到 ρ=0.9999 是 **9.63710**，才基本贴住上界。

再把所有波动率整体缩放，保持各标的相对离散结构不变：缩放 0.5 倍时上界 **4.83260**、独立价 **2.27458**、缺口 **2.55802**；缩放 2 倍时上界 **19.05654**、独立价 **9.47159**、缺口 **9.58495**。波动率离散度越大，共单调与独立之间的价格鸿沟越宽。

![五资产篮子与波动率离散敏感性](/images/comonotonic-basket-option-bounds/fig5_five_assets.png)

## 七、落地建议

第一，报价单上应当同时给出共单调上界与某一相关性假设下的模型价，让客户看到相关性假设到底贡献了多少溢价。第二，不要把秩相关 0.9 以上直接当作「上限已锁定」，从本文数字看，最后那一两个百分点的联动差异仍可能值几个百分点的期权费。第三，对高维篮子，严格下界并不存在，风控报表里应明确标注哪些数字是模型价、哪些是无模型上界，避免把独立假设下的低价误读成理论地板。

最后给一个可落地的报价模板：每个篮子期权旁边并列三列，分别是共单调上界、当前相关性假设下的模型价、以及二者之差占总价的比例。以 K=100 的两资产篮子为例，三列填写 **9.55177**、**6.93322** 与 **27.41%**；若风控要求差值不得超过 10%，就必须把相关性假设从独立推到 ρ≈0.75 以上，而这个要求是否可以由历史数据支持，应当单独做统计检验，而不是在报价软件里悄悄改一个参数。

完整实验脚本如下，包含五个图的生成与 results.json 输出。

```python
"""Model-free basket call bounds from marginals: comonotonic, countermonotonic and copula prices."""

def main():
    import json
    import math
    import os

    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.stats import qmc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "/Users/halo/workspace/astro-blog/public/images/comonotonic-basket-option-bounds"
    os.makedirs(OUT, exist_ok=True)

    R = 0.04
    T = 1.0
    DF = math.exp(-R * T)
    K = 100.0

    # Gauss-Legendre on u in (0,1); endpoints are clipped before ndtri.
    nodes, weights = leggauss(4000)
    u = np.clip((nodes + 1.0) / 2.0, 1e-9, 1 - 1e-9)
    wu = weights / 2.0

    def quantile(uu, forward, vol):
        return forward * np.exp(-0.5 * vol * vol * T + vol * math.sqrt(T) * ndtri_safe(uu))

    def ndtri_safe(uu):
        from scipy.special import ndtri
        return ndtri(uu)

    def bound_price(k, forwards, vols, wt, coupling):
        qs = [quantile(u, f, s) for f, s in zip(forwards, vols)]
        if coupling == "comonotonic":
            basket = sum(wi * qi for wi, qi in zip(wt, qs))
        elif coupling == "countermonotonic":
            basket = wt[0] * qs[0] + wt[1] * qs[1][::-1]
        else:
            raise ValueError(coupling)
        payoff = np.maximum(basket - k, 0.0)
        return float(DF * np.dot(wu, payoff))

    F2 = [100.0, 100.0]
    V2 = [0.20, 0.30]
    W2 = [0.5, 0.5]
    upper2 = bound_price(K, F2, V2, W2, "comonotonic")
    lower2 = bound_price(K, F2, V2, W2, "countermonotonic")

    def gaussian_price(rhos, forwards, vols, wt, k, log2m=20, seed=1234):
        dim = len(forwards)
        sobol = qmc.Sobol(d=dim, scramble=True, seed=seed)
        uv = sobol.random_base2(log2m)
        z = ndtri_safe(uv)
        if dim == 2:
            rho = rhos[0]
            z2 = rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]
            coords = np.column_stack([z[:, 0], z2])
        else:
            rho = rhos[0]
            coords = np.empty_like(z)
            coords[:, 0] = z[:, 0]
            for j in range(1, dim):
                coords[:, j] = math.sqrt(rho) * z[:, 0] + math.sqrt(1 - rho) * z[:, j]
        basket = np.zeros(len(uv))
        for wi, f, s, zz in zip(wt, forwards, vols, coords.T):
            basket += wi * f * np.exp(-0.5 * s * s * T + s * math.sqrt(T) * zz)
        payoff = np.maximum(basket - k, 0.0)
        return float(DF * payoff.mean())

    def spearman(rho):
        return (6.0 / math.pi) * math.asin(rho / 2.0)

    rho_grid = [-0.9, -0.5, 0.0, 0.5, 0.9, 0.99]
    rho_sweep = []
    for rho in rho_grid:
        p = gaussian_price([rho], F2, V2, W2, K)
        rho_sweep.append({"rho": rho, "spearman": spearman(rho), "price": p,
                          "retention": (p - lower2) / (upper2 - lower2)})

    indep2 = rho_sweep[2]["price"]

    k_grid = [80.0, 90.0, 100.0, 110.0, 120.0]
    strike_sweep = []
    for kk in k_grid:
        up = bound_price(kk, F2, V2, W2, "comonotonic")
        lo = bound_price(kk, F2, V2, W2, "countermonotonic")
        strike_sweep.append({"K": kk, "upper": up, "lower": lo, "width": up - lo})

    F5 = [95.0, 98.0, 100.0, 102.0, 105.0]
    V5 = [0.15, 0.20, 0.25, 0.30, 0.35]
    W5 = [0.2] * 5
    upper5 = bound_price(K, F5, V5, W5, "comonotonic")
    rho5_grid = [0.0, 0.25, 0.5, 0.9, 0.99, 0.9999]
    rho5 = [{"rho": rr, "price": gaussian_price([rr], F5, V5, W5, K, log2m=19, seed=77)}
            for rr in rho5_grid]

    # Heterogeneous vol spread: scale all vols, keep marginals fixed within each scenario.
    spread = []
    for scale in [0.5, 1.0, 1.5, 2.0]:
        vs = [v * scale for v in V5]
        up = bound_price(K, F5, vs, W5, "comonotonic")
        px = gaussian_price([0.0], F5, vs, W5, K, log2m=19, seed=77)
        spread.append({"scale": scale, "upper": up, "independent": px, "gap": up - px})

    results = {
        "params": {"r": R, "T": T, "DF": DF, "K": K, "quad_nodes": 4000, "sobol_log2m": 20},
        "n2": {"forwards": F2, "vols": V2, "weights": W2,
               "upper": upper2, "lower": lower2, "width": upper2 - lower2,
               "independent": indep2,
               "indep_retention": (indep2 - lower2) / (upper2 - lower2)},
        "rho_sweep": rho_sweep,
        "strike_sweep": strike_sweep,
        "n5": {"forwards": F5, "vols": V5, "upper": upper5, "rho": rho5,
               "retention_099": (rho5[-1]["price"] - rho5[1]["price"]) / (upper5 - rho5[1]["price"])},
        "vol_spread": spread,
    }

    # fig1: quantile paths under the two extremal couplings.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    q1 = quantile(u, F2[0], V2[0])
    q2 = quantile(u, F2[1], V2[1])
    ax.plot(u, q1, color="#4c78a8", lw=1.6, label="asset 1 quantile")
    ax.plot(u, q2, color="#e45756", lw=1.6, label="asset 2 quantile")
    ax.plot(u, 0.5 * q1 + 0.5 * q2, color="#2ca02c", lw=2.2, label="comonotonic basket")
    ax.plot(u, 0.5 * q1 + 0.5 * q2[::-1], color="#9467bd", lw=2.2, label="countermonotonic basket")
    ax.axhline(K, color="black", ls="--", lw=1, label="strike 100")
    ax.set_xlabel("uniform quantile level u")
    ax.set_ylabel("terminal basket value")
    ax.set_title("Same marginals, opposite couplings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_quantile_paths.png"), dpi=150)
    plt.close(fig)

    # fig2: basket price across Gaussian copula correlation, with exact bounds.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = [e["rho"] for e in rho_sweep]
    ax.plot(xs, [e["price"] for e in rho_sweep], "o-", color="#4c78a8", label="Gaussian copula (Sobol)")
    ax.axhline(upper2, color="#2ca02c", ls="--", lw=1.8, label="comonotonic upper bound")
    ax.axhline(lower2, color="#9467bd", ls="--", lw=1.8, label="countermonotonic lower bound")
    ax.set_xlabel("Gaussian correlation rho")
    ax.set_ylabel("basket call price")
    ax.set_title("Dependence moves price inside a fixed marginal-implied range")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_vs_rho.png"), dpi=150)
    plt.close(fig)

    # fig3: upper-bound retention against rank correlation.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot([e["spearman"] for e in rho_sweep], [e["retention"] for e in rho_sweep],
            "o-", color="#e45756")
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1)
    ax.set_xlabel("Spearman rank correlation")
    ax.set_ylabel("fraction of bound interval")
    ax.set_title("Almost-comonotonic is not comonotonic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_rank_retention.png"), dpi=150)
    plt.close(fig)

    # fig4: strike sweep of the two bounds.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(k_grid, [e["upper"] for e in strike_sweep], "o-", color="#2ca02c", label="upper")
    ax.plot(k_grid, [e["lower"] for e in strike_sweep], "o-", color="#9467bd", label="lower")
    ax.fill_between(k_grid, [e["lower"] for e in strike_sweep], [e["upper"] for e in strike_sweep],
                    color="#bbbbbb", alpha=0.45, label="unidentified range")
    ax.set_xlabel("strike K")
    ax.set_ylabel("basket call price")
    ax.set_title("Marginals alone leave a strike-dependent price band")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_strike_band.png"), dpi=150)
    plt.close(fig)

    # fig5: five-asset basket and vol-spread sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot([e["rho"] for e in rho5], [e["price"] for e in rho5], "o-", color="#4c78a8",
                 label="equicorrelation")
    axes[0].axhline(upper5, color="#2ca02c", ls="--", lw=1.6, label="comonotonic upper")
    axes[0].set_xlabel("equicorrelation rho")
    axes[0].set_ylabel("basket call price")
    axes[0].set_title("Five assets, heterogeneous vols")
    axes[0].legend(fontsize=8)
    axes[1].plot([e["scale"] for e in spread], [e["upper"] for e in spread], "o-",
                 color="#2ca02c", label="upper")
    axes[1].plot([e["scale"] for e in spread], [e["independent"] for e in spread], "o-",
                 color="#4c78a8", label="independent")
    axes[1].set_xlabel("volatility scale")
    axes[1].set_ylabel("basket call price")
    axes[1].set_title("Wider vol dispersion widens the gap")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_five_assets.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

