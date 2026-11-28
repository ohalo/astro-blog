---
title: "鞅最优传输无模型期权定价：只用看涨价差与线性规划夹出上下界，为什么额外矩约束能收窄区间、又在稀疏报价下失效"
description: "141 点终端价格网格、9 档看涨报价的实测：只知道现价 100 与鞅条件时，120 数字看涨的无模型区间是 0 到 0.75；加上 60 至 140 的 9 档看涨报价后收窄到 0.1303 至 0.2190，宽度下降 88.18%，真实价 0.1718 落在区间内。若只剩 120、130、140 三档高执行价，宽度回到 0.5828；再补一条二阶矩报价，收窄 59.68% 至 0.2350。报价噪声 0.08 时 20 次里 2 次无解，区间看似变窄其实是可行集被挤空。附完整可运行代码与五张真实计算图。"
publishDate: '2026-11-28T13:06:00Z'
tags:
  - 量化
  - 衍生品
  - 最优化
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

给一个期权定价，通常要先选模型：Black-Scholes、Heston、局部波动率，每个模型都替你假设了一条价格路径的分布。但市场上真正无争议的信息往往只有几条：标的价格、几档看涨期权报价、以及「折现后的标的价是鞅」这条无套利底线。能不能绕开模型，直接问这些已知条件允许价格落在什么区间？

可以，而且答案是一个线性规划。我把终端价离散到 40 至 180 的 141 个格点，用一个双成分对数正态混合分布生成真实风险中性测度，再把 60 至 140 共 9 档看涨报价喂给规划器。固定种子 `20261128` 的实测结果：

- 只知道现价 100 与鞅条件时，执行价 120 的数字看涨期权价格区间是 **0 到 0.75**，宽度 0.75，几乎没有任何定价信息；
- 加上 9 档看涨报价后，区间收紧到 **0.1303 至 0.2190**，宽度 0.0887，下降 **88.18%**；真实价格 **0.1718** 落在区间内部；
- 如果市场只剩 120、130、140 三档高执行价报价，宽度回到 **0.5828**；再补一条二阶矩（等价于方差互换报价）后收窄到 **0.2350**，改善 **59.68%**；
- 但报价并非越多越单调：把报价档位从 3 档均匀分布加到 9 档，宽度从 0.3483 降到 0.0887；加到 11 档时反而升到 0.1089，因为这是重新布点而不是嵌套追加；
- 给报价加入标准差 0.08 的噪声后，20 次重采样里有 **2 次线性规划无解**。此时区间有时反而更窄，那是可行集被噪声挤空，不是信息变多。

![9 档看涨报价与生成它们的风险中性分布](/images/martingale-transport-option-bounds/fig1_call_quotes.png)

这些数字说明，无模型定价真正卖的不是「一个价格」，而是「在报价能约束住的那部分尾部里，价格动不了」。区间宽度就是报价信息的货币化度量。

实验中的真实分布是两个对数正态成分的加权混合，再用指数倾斜把均值精确拉回 100。这样构造的好处是：真实价格可以独立算出来，用它检验上下界是否真的把真值夹住，而不是只比较两种模型谁更靠近谁。9 档报价中，60 执行价的看涨值 40.0924，140 执行价只剩 0.8743，接近虚值端的信息本来就稀薄；数字看涨的执行价 120 正好落在报价较密的区间，因此收窄效果明显。若把目标换成执行价 170 的极端虚值数字期权，即使有这 9 档报价，区间也会重新张开。

## 二、把无套利写成线性规划

设终端价 $S_T$ 取值于有限格点 $s_1,\dots,s_n$，未知数是每个格点的风险中性概率 $p_i\ge 0$。无套利要求三件事：概率和为 1、折现标的价是鞅、看涨报价被精确复制。

$$
\sum_i p_i=1,\qquad \sum_i p_i s_i=S_0,\qquad \sum_i p_i(s_i-K_j)^+=C(K_j),\ \ j=1,\dots,m.
$$

在这些等式约束下，任意只在终端兑现的衍生品 payoff $f(s_i)$ 的期望都是 $p^\top f$。于是它的最大、最小可能价格就是两个线性规划：

$$
\overline V=\max_{p}\ p^\top f,\qquad \underline V=\min_{p}\ p^\top f,
$$

约束集完全相同。这就是离散版鞅最优传输：在所有把现价输运到终值、且满足期权报价的联合分布里，找一个把 payoff 最大化和最小化的极端测度。它不需要假设波动率是常数，也不需要假设路径连续，只要终端分布落在约束里就行。

一个容易被忽略的细节是：报价必须自身无套利。若看涨价对执行价不再凸，或者低于内在价值，约束集直接为空，求解器会报不可行。这时正确的反应是清理报价，而不是放宽鞅条件。

为什么这个界是「最优」的？因为线性规划的可行域正好是所有与报价相容的离散概率向量，目标函数又对概率线性，最优解必然落在多面体顶点上。所谓顶点，就是只由少数格点支撑的极端分布。任何模型给出的单一价格，只要它的隐含分布满足同一组报价，就必然落在这个区间里；反过来，区间外的报价无论用哪个模型包装，都与无套利约束冲突。这一点让区间比任何单点估计更难争论。代价则是：约束之外的尾部形态完全自由，区间只能告诉你「已知报价支持什么」，不能告诉你真实世界一定如何。

## 三、9 档报价到底锁住了多少

数字看涨在 $S_T\ge 120$ 时付 1，否则付 0。它是最能体现尾部不确定性的合约之一：payoff 不连续，无法靠局部的蝶式报价精确复制。图中三层带宽对应三种信息集。

![不同信息集下的数字看涨上下界](/images/martingale-transport-option-bounds/fig2_digital_bounds.png)

只用鞅条件时，理论上可以把全部概率放在 40 和 180 两点，调权重让均值等于 100，数字看涨概率可以从 0 一路取到 0.75。加上 9 档看涨报价后，两端的极端测度必须同时 reproduce 从 40.09 到 0.874 的整条报价曲线，活动空间骤减。上界 0.2190、下界 0.1303，真实价 0.1718 到两端的距离分别是 0.0472 和 0.0415，大致居中。

这里有个实践含义：如果一个做市商对同一标的的 9 档报价都可信，那么 120 数字看涨报出 0.30 就已经明显越界；报 0.18 则落在无套利区间内，模型差异无法再用来判定谁对。区间越窄，模型假设越不重要。

## 四、极端测度到底长什么样

![达到上下界的两个极端风险中性分布](/images/martingale-transport-option-bounds/fig3_extremal_measures.png)

达到上界的测度把质量尽量堆到执行价右侧、同时在左侧保留足以压低看涨报价的质量；达到下界的测度则反过来，把质量搬到执行价之下。两条极端曲线不是在真实分布上加个波动率参数，而是形状完全不同：上界分布更偏向远端右尾，下界分布则在 120 以下堆积。真实分布介于两者之间，这正是不确定性区间的几何直观。

交易上可以把这条区间当成一个压力区间：卖出数字看涨时，不能用「我认为真实分布接近对数正态」来掩盖上界 0.2190 的可能；反过来，报价方如果给出 0.22 以上，买方至少知道市场报价已经在无套利意义下偏贵，不需要争辩模型。

## 五、报价稀疏时，区间不是线性变窄

![报价档位数对上下界宽度的影响](/images/martingale-transport-option-bounds/fig4_quote_sparsity.png)

我把报价档位从 3 档逐步加到 17 档，每次都在 60 至 140 上重新等距布点。3 档时宽度 0.3483，9 档时 0.0887，17 档时 0.0438，总体趋势是信息越多越窄。但 11 档时出现 0.1089，比 9 档更宽：因为新布点把原本落在 120 附近的报价挪走了，这并非「追加信息反而有害」，而是不同报价集的覆盖位置不同。实务中比较两份报价表时，必须先对齐执行价，再谈谁的信息更多。

二阶矩的作用是另一回事。只用 120、130、140 三档高执行价看涨时，左尾几乎不受约束，宽度 0.5828；加上一条 $\mathbb E[S_T^2]=10506.75$ 的矩报价后，宽度降到 0.2350，降幅 59.68%。但如果已经有 60 至 140 的 9 档看涨报价，再补二阶矩几乎不再改善：看涨曲线的凸性与鞅条件已经把这二阶信息锁死。所以「加一个方差互换报价」是救急手段，不是万能增强。

## 六、报价噪声会把区间挤空

![报价噪声对区间宽度、中心偏移与可行性的影响](/images/martingale-transport-option-bounds/fig5_quote_noise.png)

真实报价有买卖价差和模型插值误差。我对 9 档报价加入正态噪声，每个噪声水平跑 20 次。噪声为 0 时宽度 0.0887；噪声升到 0.04 时平均宽度 0.0921、区间中心平均偏离真实价 0.0034；噪声到 0.08 时 20 次里有 2 次约束集为空，其余平均宽度反而降到 0.0763。这个反直觉结果必须警惕：可行性检查失败时，不能用「区间更窄」当作更精确。工程上应当在求解前先做报价凸性清洗，并在求解后记录求解器状态，而不是只记上下界数值。

## 七、完整实现

下面脚本只依赖 numpy、scipy 与 matplotlib，运行后会在 `public/images/martingale-transport-option-bounds/` 下生成五张图与 `results.json`。所有随机性都由固定种子控制，可逐位复跑。

```python
# -*- coding: utf-8 -*-
"""Martingale optimal transport bounds from sparse call quotes (deterministic experiment)."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq, linprog
from scipy.stats import lognorm

S0 = 100.0
SEED = 20261128
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "martingale-transport-option-bounds")
os.makedirs(OUT, exist_ok=True)

grid = np.arange(40.0, 181.0, 1.0)
one = np.ones_like(grid)


def true_distribution():
    w = np.zeros_like(grid)
    for weight, mean, sigma in ((0.62, 100.0, 0.18), (0.38, 128.0, 0.34)):
        scale = mean * np.exp(-0.5 * sigma * sigma)
        w += weight * lognorm.pdf(grid, sigma, scale=scale)
    w /= w.sum()

    def mean_gap(eta):
        tilted = w * np.exp(eta * grid)
        tilted /= tilted.sum()
        return tilted @ grid - S0

    eta = brentq(mean_gap, -0.5, 0.5, xtol=1e-14)
    w = w * np.exp(eta * grid)
    w /= w.sum()
    return w


p_true = true_distribution()
mom1 = float(p_true @ grid)
mom2 = float(p_true @ (grid ** 2))


def call_price(strike):
    return float(p_true @ np.maximum(grid - strike, 0.0))


def digital_price(strike):
    return float(p_true @ (grid >= strike))


quote_strikes = np.arange(60.0, 141.0, 10.0)
quotes = [call_price(k) for k in quote_strikes]


def bounds(payoff, call_list=None, use_moment=False):
    rows = [one.tolist(), grid.tolist()]
    rhs = [1.0, S0]
    if call_list:
        for strike, price in call_list:
            rows.append(np.maximum(grid - strike, 0.0).tolist())
            rhs.append(price)
    if use_moment:
        rows.append((grid ** 2).tolist())
        rhs.append(mom2)
    A = np.array(rows)
    b = np.array(rhs)
    up = linprog(-payoff, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    lo = linprog(payoff, A_eq=A, b_eq=b, bounds=(0, None), method="highs")
    if up.status != 0 or lo.status != 0:
        return None, None, None, None
    return float(-up.fun), float(lo.fun), up.x, lo.x


K = 120.0
payoff = (grid >= K).astype(float)
q = [(float(k), float(c)) for k, c in zip(quote_strikes, quotes)]
q3 = [(k, c) for k, c in q if k >= 120.0]

d_up, d_lo, _, _ = bounds(payoff)
m_up, m_lo, _, _ = bounds(payoff, q, use_moment=False)
b3_up, b3_lo, _, _ = bounds(payoff, q3, use_moment=False)
t_up, t_lo, p_up, p_lo = bounds(payoff, q3, use_moment=True)

# Fig 1: quote curve and the risk-neutral distribution used to generate it.
fig, ax = plt.subplots(figsize=(8, 4.8))
ks = np.arange(50.0, 151.0, 1.0)
ax.plot(ks, [call_price(k) for k in ks], color="#1764ab", lw=2, label="True call curve")
ax.scatter(quote_strikes, quotes, s=58, color="#c23b22", zorder=3, label="Observed quotes")
ax.set_xlabel("Strike")
ax.set_ylabel("Call price")
ax.set_title("Sparse call quotes sampled from a two-component distribution")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig1_call_quotes.png"), dpi=150)
plt.close(fig)

# Fig 2: digital bounds across strikes under three information sets.
ks2 = np.arange(60.0, 151.0, 5.0)
mart_up, mart_lo, quote_up, quote_lo, mom_up_s, mom_lo_s = [], [], [], [], [], []
for strike in ks2:
    f = (grid >= strike).astype(float)
    a, b, _, _ = bounds(f)
    c, d, _, _ = bounds(f, q, use_moment=False)
    e, g, _, _ = bounds(f, [(k, c) for k, c in q if k >= 120.0], use_moment=True)
    mart_up.append(a)
    mart_lo.append(b)
    quote_up.append(c)
    quote_lo.append(d)
    mom_up_s.append(e)
    mom_lo_s.append(g)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.fill_between(ks2, mart_lo, mart_up, color="#d9e6f2", label="Martingale only")
ax.fill_between(ks2, quote_lo, quote_up, color="#f6c9a8", label="Call quotes")
ax.plot(ks2, mom_lo_s, color="#1a7f4b", lw=1.8, label="3 high quotes + second moment")
ax.plot(ks2, mom_up_s, color="#1a7f4b", lw=1.8)
ax.plot(ks2, [digital_price(k) for k in ks2], "k--", lw=1.7, label="True digital price")
ax.set_xlabel("Digital strike")
ax.set_ylabel("Price bound")
ax.set_title("Model-free digital bounds narrow as option information grows")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig2_digital_bounds.png"), dpi=150)
plt.close(fig)

# Fig 3: extremal distributions that attain the K=120 bounds.
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(grid, p_true, color="#333333", lw=2, label="True implied distribution")
ax.plot(grid, p_up, color="#c23b22", lw=1.5, label="Upper-bound distribution")
ax.plot(grid, p_lo, color="#1764ab", lw=1.5, label="Lower-bound distribution")
ax.axvline(K, color="#777777", ls="--", lw=1)
ax.set_xlabel("Terminal underlying price")
ax.set_ylabel("Probability mass")
ax.set_title("Extremal martingale measures for the digital payoff")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig3_extremal_measures.png"), dpi=150)
plt.close(fig)

# Fig 4: how many quote strikes are needed.
quote_counts = list(range(3, 18, 2))
sparse_widths = []
sparse_lows = []
sparse_highs = []
for count in quote_counts:
    sub_strikes = np.linspace(60.0, 140.0, count)
    sub_quotes = [(float(k), call_price(k)) for k in sub_strikes]
    hi, low, _, _ = bounds(payoff, sub_quotes, use_moment=False)
    sparse_highs.append(hi)
    sparse_lows.append(low)
    sparse_widths.append(hi - low)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(quote_counts, sparse_widths, "o-", color="#7b3fa1", lw=2, label="Bound width")
ax.plot(quote_counts, sparse_highs, "s--", color="#c23b22", lw=1.3, label="Upper bound")
ax.plot(quote_counts, sparse_lows, "^--", color="#1764ab", lw=1.3, label="Lower bound")
ax.axhline(digital_price(K), color="#333333", ls=":", lw=1.5, label="True digital price")
ax.set_xlabel("Number of quoted strikes")
ax.set_ylabel("Price / width")
ax.set_title("Sparse quote sensitivity for the K=120 digital")
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig4_quote_sparsity.png"), dpi=150)
plt.close(fig)

# Fig 5: quote noise can break feasibility or move the whole interval.
rng = np.random.default_rng(SEED)
noise_levels = [0.0, 0.005, 0.01, 0.02, 0.04, 0.08]
noise_widths = []
noise_failures = []
noise_centers = []
for level in noise_levels:
    widths, centers, fails = [], [], 0
    for _ in range(20):
        noisy = [(float(k), float(c + rng.normal(0.0, level))) for k, c in zip(quote_strikes, quotes)]
        hi, low, _, _ = bounds(payoff, noisy, use_moment=False)
        if hi is None:
            fails += 1
            continue
        widths.append(hi - low)
        centers.append(0.5 * (hi + low))
    noise_widths.append(float(np.mean(widths)) if widths else 0.0)
    noise_centers.append(float(np.mean(np.abs(np.array(centers) - digital_price(K)))) if centers else 0.0)
    noise_failures.append(fails)

fig, ax = plt.subplots(figsize=(8, 4.8))
ax.plot(noise_levels, noise_widths, "o-", color="#1a7f4b", lw=2, label="Mean bound width")
ax.plot(noise_levels, noise_centers, "s--", color="#c23b22", lw=1.7, label="Mean center error")
ax2 = ax.twinx()
ax2.bar(noise_levels, noise_failures, width=0.003, color="#7b3fa1", alpha=0.4, label="Infeasible runs")
ax2.set_ylabel("Infeasible count out of 20")
ax.set_xlabel("Quote noise standard deviation")
ax.set_ylabel("Price units")
ax.set_title("Noisy quotes widen and displace the no-arbitrage interval")
ax.legend(loc="upper left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "fig5_quote_noise.png"), dpi=150)
plt.close(fig)

results = {
    "grid_points": len(grid),
    "S0": S0,
    "true_mean": mom1,
    "true_second_moment": mom2,
    "quote_strikes": [float(k) for k in quote_strikes],
    "quote_prices": [float(c) for c in quotes],
    "digital_strike": K,
    "true_digital_price": digital_price(K),
    "martingale_lower": d_lo,
    "martingale_upper": d_up,
    "martingale_width": d_up - d_lo,
    "quotes_lower": m_lo,
    "quotes_upper": m_up,
    "quotes_width": m_up - m_lo,
    "width_reduction_pct": 100.0 * (1.0 - (m_up - m_lo) / (d_up - d_lo)),
    "three_high_quotes_lower": b3_lo,
    "three_high_quotes_upper": b3_up,
    "three_high_quotes_width": b3_up - b3_lo,
    "moment_lower": t_lo,
    "moment_upper": t_up,
    "moment_width": t_up - t_lo,
    "extra_moment_reduction_pct": 100.0 * (1.0 - (t_up - t_lo) / (b3_up - b3_lo)),
    "quote_counts": quote_counts,
    "sparse_widths": [float(x) for x in sparse_widths],
    "sparse_lows": [float(x) for x in sparse_lows],
    "sparse_highs": [float(x) for x in sparse_highs],
    "noise_levels": noise_levels,
    "noise_mean_widths": noise_widths,
    "noise_mean_center_errors": noise_centers,
    "noise_infeasible_counts": noise_failures,
    "deep_itm_call_at_60": call_price(60.0),
    "deep_otm_call_at_140": call_price(140.0),
}
with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as fh:
    json.dump(results, fh, ensure_ascii=False, indent=2, sort_keys=True)
print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    pass
```

```

需要把方法搬到真实市场时，最重要的是三条：报价本身要先过无套利清洗；终端网格要覆盖报价隐含的最大与最小可达价；求解器状态要进入生产监控。只要这三条做到，线性规划给出的区间就是一个不依赖模型的价格护栏。
