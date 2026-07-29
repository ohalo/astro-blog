---
title: "方差互换复制：用一篮子期权静态复制已实现方差"
description: "方差互换让你直接交易『未来已实现方差』，而它的公允价可以完全不依赖任何波动率模型——只需要一整条行权价上的期权报价，按 1/K² 加权买成一篮子，再配上动态 delta 对冲。本文从对数合约恒等式出发推导复制逻辑，用 Python 验证三件事：1/K² 权重组合对 log payoff 的逼近精度、行权价截断如何系统性低估公允方差（VIX 也逃不掉）、以及跳跃如何撕开复制的裂缝让方差互换多头意外受益。附完整蒙特卡洛验证代码与四类实盘陷阱（中高阶）。"
publishDate: '2026-07-29'
tags:
  - 量化交易
  - 方差互换
  - 波动率交易
  - 期权复制
  - 对数合约
  - VIX
  - 衍生品定价
  - Python
language: Chinese
difficulty: advanced
---

想直接做多或做空"未来的波动"，最干净的工具不是期权，而是**方差互换（variance swap）**：到期时，多头收到"已实现方差 − 事先约定的行权方差"乘以名义金额。没有 delta、没有 gamma 路径依赖的纠缠，收益直接就是方差本身。

结论先放这：**方差互换的公允行权价 K_var 不需要任何波动率模型，它等于一条按 1/K² 加权的 OTM 期权组合的价格。** 这是衍生品定价里少有的"model-free"结果之一——你不需要假设 Black-Scholes、Heston 或任何动态，只需要市场上有一整条行权价的期权报价。VIX 的官方计算公式就是这个逻辑的直接应用。但 model-free 不等于没有陷阱：**行权价截断会系统性低估公允价，跳跃会撕开复制的裂缝**。本文用 Python 把这三层都验证一遍。

## 一、为什么是 1/K²：对数合约的魔法

一切从一个恒等式开始。对任意足够光滑的价格路径（暂时不考虑跳跃），Itô 引理给出：

$$
d(\ln S_t) = \frac{dS_t}{S_t} - \frac{1}{2}\sigma_t^2 \, dt
$$

移项并在 $[0, T]$ 上积分：

$$
\int_0^T \sigma_t^2 \, dt = 2\left(\int_0^T \frac{dS_t}{S_t} - \ln\frac{S_T}{S_0}\right)
$$

左边就是我们想要的**已实现方差**（乘 $1/T$ 年化）。右边两项分别是：

1. $\int_0^T dS_t/S_t$：**动态持有 $1/S_t$ 股标的**的累计收益——这是一个可以每天调仓实现的自融资策略；
2. $-\ln(S_T/S_0)$：**做空一个对数合约**——到期支付 $\ln(S_T/S_0)$ 的奇异衍生品。

第一项能自己交易出来，问题只剩：市场上没有对数合约这种产品，怎么办？

答案是 Carr-Madan 静态复制定理：**任何到期 payoff $f(S_T)$ 都可以拆成现金 + 远期 + 一条行权价上的欧式期权组合**：

$$
f(S_T) = f(F) + f'(F)(S_T - F) + \int_0^F f''(K)(K-S_T)^+ dK + \int_F^\infty f''(K)(S_T-K)^+ dK
$$

对 $f(S) = \ln S$，二阶导是 $f''(K) = -1/K^2$。于是**做空对数合约 = 买入所有行权价的 OTM 期权，每个的权重是 $dK/K^2$**——低行权价的 put 权重大，高行权价的 call 权重小。最终公允行权方差为：

$$
K_{var} = \frac{2 e^{rT}}{T}\left[\int_0^F \frac{P(K)}{K^2}dK + \int_F^\infty \frac{C(K)}{K^2}dK\right]
$$

其中 $F$ 是远期价，$P(K)$、$C(K)$ 是 OTM put 和 call 的现价。**这条公式里没有任何波动率参数**——所有关于未来波动的信息都已经藏在整条期权报价里了。

![1/K²权重的期权组合](/images/variance-swap-replication/varswap-strike-weights.png)

为什么低行权价权重更大？直觉是：方差对"大跌"和"大涨"一视同仁，但同样 1% 的对数收益，价格越低的区域每 1 美元价格变动对应的对数变动越大，所以需要更多的低行权价期权来"补齐"下方的方差敏感度。

## 二、用 Python 验证复制精度

先验证静态复制的核心：离散行权价网格的期权组合，能多好地逼近对数合约的 payoff？

```python
import numpy as np
from scipy.stats import norm

S0, r, T = 100.0, 0.02, 0.25
F = S0 * np.exp(r * T)          # 远期价

def bs_price(S, K, T, r, sigma, cp):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    if cp == 'c':
        return S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
    return K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)

def portfolio_payoff(ST, strikes, dK):
    """1/K² 加权 OTM 期权组合的到期 payoff"""
    payoff = np.zeros_like(ST)
    for K in strikes:
        w = 2.0/T * dK / K**2
        if K < F:
            payoff += w * np.maximum(K - ST, 0)   # OTM put
        else:
            payoff += w * np.maximum(ST - K, 0)   # OTM call
    return payoff

# 目标 payoff：2/T * ((ST-F)/F - ln(ST/F))
ST = np.linspace(50, 160, 400)
target = 2.0/T * ((ST - F)/F - np.log(ST/F))
```

三种行权价间距（ΔK = 20 / 10 / 2.5）的逼近效果：

![对数合约的期权组合逼近](/images/variance-swap-replication/varswap-log-contract-approx.png)

ΔK=2.5 时组合 payoff 和目标曲线肉眼已无法区分。**离散化误差是二阶小量**——这也是为什么交易所上市期权的行权价网格（通常 2.5 或 5 美元间距）在实务中够用。

## 三、截断误差：VIX 也逃不掉的系统性低估

理论公式要求行权价从 0 积到无穷，但真实市场的报价只覆盖有限区间。**截断的后果不是噪声，而是单向的低估**——你漏掉的都是正贡献。

用 σ=25% 的 Black-Scholes 世界做定价实验，看不同截断范围下期权组合给出的 K_var：

```python
def kvar_strip(kmin, kmax, dK, sigma):
    """有限行权价条带定出的方差互换行权价"""
    ks = np.arange(kmin, kmax + 1e-9, dK)
    total = 0.0
    for K in ks:
        cp = 'p' if K < F else 'c'
        total += dK / K**2 * bs_price(S0, K, T, r, sigma, cp)
    return 2.0/T * np.exp(r*T) * total

for kmin, kmax in [(80,125), (70,140), (60,150), (50,170), (30,220), (10,300)]:
    kv = kvar_strip(kmin, kmax, 1.0, 0.25)
    print(f"[{kmin},{kmax}]: Kvar = {kv:.5f}  (真值 0.06250)")
```

![截断误差](/images/variance-swap-replication/varswap-truncation-error.png)

行权价范围覆盖 [10, 300]（约 ±4 个标准差以上）时，K_var = 0.06252，与理论真值 0.06250 只差 0.03%。但只用 [80, 125]（大约 ±1.5σ）时低估超过 5%。**在高波动环境下这个误差会急剧放大**——因为分布变宽，更多方差质量落在截断区间之外。

这不是纸上谈兵：VIX 的官方口径就是这条公式的离散版本，CBOE 的截断规则是"从 ATM 向两边扫描，连续两个行权价零买价就停止"。2008 年和 2020 年 3 月的极端行情里，深度 OTM put 的报价断档直接影响了 VIX 的计算精度——**指数本身在最需要它的时刻反而最不可靠**。

## 四、跳跃：复制裂缝里的意外之财

上面的推导有个隐藏假设：价格路径连续。一旦有跳跃，Itô 引理多出一截，对数合约复制的就不再是真实的已实现方差。

用蒙特卡洛对比两个世界：纯 GBM，和叠加 Merton 跳跃（跳跃强度 λ=4/年，平均跳幅 −3%）：

```python
rng = np.random.default_rng(42)
n_paths, n_steps = 60000, 63
dt = T / n_steps

# 连续部分
Z = rng.standard_normal((n_paths, n_steps))
logret = (r - 0.5*0.25**2)*dt + 0.25*np.sqrt(dt)*Z

# 跳跃部分（Merton）
lam, mu_j, sig_j = 4.0, -0.03, 0.04
NJ = rng.poisson(lam*dt, (n_paths, n_steps))
J = NJ*mu_j + np.sqrt(NJ)*sig_j*rng.standard_normal((n_paths, n_steps))

rv_nojump = (logret**2).sum(axis=1) / T
rv_jump   = ((logret + J)**2).sum(axis=1) / T

kvar = kvar_strip(10, 300, 0.5, 0.25)
pnl_nojump = 10000 * (rv_nojump - kvar)   # variance notional = 1万
pnl_jump   = 10000 * (rv_jump   - kvar)
```

![跳跃下的P&L分布](/images/variance-swap-replication/varswap-pnl-jumps.png)

无跳跃世界里，P&L 分布紧紧围绕零对称——复制是公平的。加入跳跃后，已实现方差的均值从 0.0625 抬到 0.0725，**方差互换多头平均白赚约 16%**（在这个仅按连续世界定价的 K_var 下）。原因：已实现方差按 $(\Delta \ln S)^2$ 计算，把跳跃全额收进来；而对数合约复制在跳跃下的误差是三阶项 $\frac{2}{3}(\Delta \ln S)^3$ 的累积——对下跳（负收益）这一项让复制组合**少付**，多头反而受益。

这就是为什么 2008 年之后，单票方差互换市场几乎死掉了：个股一个 −40% 的跳跃（破产、被收购）能让方差互换空头（通常是做市商）瞬间损失名义本金的几十倍。市场的修复方案是**给方差互换加 cap**（通常 2.5 倍行权方差封顶），或者干脆转向 payoff 更温和的**波动率互换**和 VIX 期货。

## 五、实盘陷阱清单

**陷阱一：用 ATM 隐含波动率平方近似 K_var。** K_var 是整条微笑按 1/K² 的加权平均，在有 skew 的市场里它系统性高于 ATM 隐波的平方。SPX 的典型 skew 下，K_var 对应的"公允波动率"比 ATM 隐波高 1-3 个波动率点。用 ATM 隐波报价方差互换，等于送钱。

**陷阱二：忽略离散采样与连续积分的差异。** 合约条款按日收盘价算 $\sum (\ln S_{i+1}/S_i)^2$，理论用连续积分。日频采样下两者的差异均值为零但有方差，短期限合约（1 个月以内）该噪声不可忽略；且合约通常规定"分母用固定的预期交易日数"，节假日安排会引入细微偏差。

**陷阱三：股息与借券成本污染远期价。** 复制公式里 $F$ 必须是真实远期（含股息、借券成本）。个股方差互换里用错远期，put/call 的 OTM 分界点就错了，权重分配跟着错。

**陷阱四：把 VIX² 直接当作 30 天方差互换的可交易价格。** VIX 是两个到期日的插值 + 截断 + 离散化的产物，与场外方差互换的实际报价存在基差（通常 VIX² 略低，因为截断漏掉了尾部）。做 VIX 与方差互换的基差套利，首先要把这些口径差异全部对齐。

## 六、总结

方差互换复制是"model-free 定价"最优雅的案例：**一个恒等式（Itô）+ 一个分解定理（Carr-Madan），把『未来的方差』变成了『今天的一篮子期权价格』**。三条实证结论：

1. **1/K² 加权的静态组合 + 动态 delta 对冲即可完全复制方差**——ΔK=2.5 的行权价网格下，K_var 定价误差 0.03%；
2. **截断是单向低估**：行权价只覆盖 ±1.5σ 时低估超过 5%，且高波动时误差放大——VIX 在危机中的可靠性折损正来自于此；
3. **跳跃打破复制**：−3% 均值的下跳让方差互换多头平均多赚 16%，这也是单票方差互换加 cap、市场转向波动率互换的根本原因。

理解这条复制链，你就同时理解了 VIX 的构造、方差风险溢价的度量口径、以及 2008 年方差互换市场崩溃的技术根源——三件事其实是同一件事。
