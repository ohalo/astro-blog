---
title: "期权 Gamma Scalping：用动态对冲把 Theta 变成 Alpha"
description: "买入跨式组合等于做多 Gamma、做空 Theta。本文用 Black-Scholes 真实定价 + GBM 路径模拟，展示如何靠每日 delta 对冲把「波动」低买高卖成现金，并点明盈利的真正前提是已实现波动高于隐含波动，以及交易成本、跳空、钉住风险三大真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 期权
  - Gamma Scalping
  - 动态对冲
  - 波动率
  - Python
language: Chinese
difficulty: advanced
---

一句话结论：**Gamma Scalping 不是「预测涨跌」的策略，而是「赌已实现波动 > 隐含波动」的对冲游戏。** 你买入一个跨式组合（straddle，1 张看涨 + 1 张看跌），等价于做多 Gamma、做空 Theta——每天把 Delta 对冲回中性，靠在标的便宜时买、贵时卖，把价格波动一点点变成现金。我们的蒙特卡洛（4000 条路径，1 个月 ATM 跨式，隐含波动 20%）显示：**当已实现波动 30% 时，对冲组合均值盈利 +2.76 元（占标的 2.76%）、胜率 99.2%；当已实现波动只有 12% 时，均值亏损 −2.19、胜率 0%**——权利金（5.50 元）就是那条生死线。

## 一、Gamma Scalping 到底在「 Scalp 」什么

先拆开一个跨式组合的希腊字母。买入 ATM 跨式 = 同时持有一张看涨和一张看跌，行权价都等于现价 $K=S_0$。它的 Delta 近似为 0（看涨 Delta≈+0.5，看跌 Delta≈−0.5），所以一开始就是 Delta 中性的；但它有**很大的正 Gamma** 和**不小的负 Theta**。

Gamma 的物理意义是「Delta 对价格的敏感度」。价格每动一步，你的 Delta 就偏一点，于是你有机会：价格跌了、Delta 变负，你**买回**些标的把 Delta 补回 0；价格涨了、Delta 变正，你**卖出**些标的把 Delta 压回 0。一买一卖之间，你总是在「相对低的地方买、相对高的地方卖」——这就是 scalping（薅波动的羊毛）。

数学上，一次再平衡带来的 Gamma P&L 近似为：

$$\Delta\Pi \approx \frac{1}{2}\Gamma\,(\Delta S)^2 - \Theta\,\Delta t$$

左边第一项是你薅到的波动收益（与价格位移平方成正比，所以**涨跌都赚**），第二项是你每天付出的时间价值损耗（Theta 为负，持续流血）。策略能不能活，就看 $(\Delta S)^2$ 累积起来够不够付 Theta。而 $(\Delta S)^2$ 的期望，正比于**已实现方差** $\sigma_{\text{real}}^2$；Theta 则按**隐含波动** $\sigma_{\text{imp}}$ 定价。于是结论极其干净：

> **只有当 $\sigma_{\text{real}} > \sigma_{\text{imp}}$ 时，Gamma Scalping 才长期为正。** 它赚的不是方向，是「实际波动比期权卖你时更疯」的那部分溢价。

## 二、从零写一个 Black-Scholes 定价器

下面是一段不依赖任何期权库的 BS 定价器（用标准正态 CDF 的误差函数实现），后面整篇模拟都靠它：

```python
import math
import numpy as np

def _ndf(x):                       # 标准正态累积分布
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

def bs_call(S, K, T, r, sigma):
    if T <= 0:
        return max(S - K, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return S * _ndf(d1) - K * math.exp(-r * T) * _ndf(d2)

def bs_put(S, K, T, r, sigma):
    if T <= 0:
        return max(K - S, 0.0)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return K * math.exp(-r * T) * _ndf(-d2) - S * _ndf(-d1)

def straddle_delta(S, K, T, r, sigma):
    # 看涨 Delta + 看跌 Delta；ATM 附近约为 0
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    return _ndf(d1) + (_ndf(d1) - 1.0)
```

用 $S_0=K=100$、$T=30/252$（一个月）、无风险利率 0、隐含波动 20% 算一下权利金：`bs_call + bs_put ≈ 5.50` 元，相当于标的的 5.5%。这笔钱就是你入场先垫付、靠 scalping 慢慢赚回来的成本。

## 三、动态对冲模拟：把波动薅成现金

核心循环很简单——每天（或更频繁）重新计算跨式的 Delta，把持仓标的调整到 $-\Delta_{\text{straddle}}$ 股，让组合回到 Delta 中性。现金流来自每次调仓的买卖价差（这里先假设零交易成本），到期时把标的平仓、跨式按内在价值结算：

```python
def simulate_path(seed, sigma_real, rebal_every=1,
                  S0=100.0, K=100.0, T=30/252, r=0.0, sigma_imp=0.20):
    rng = np.random.default_rng(seed)
    N = 30
    dt = T / N
    t = np.linspace(0, T, N + 1)
    Z = rng.normal(size=N)
    S = np.zeros(N + 1); S[0] = S0
    for i in range(1, N + 1):          # 用真实波动 sigma_real 生成 GBM 路径
        S[i] = S[i-1] * math.exp((r - 0.5*sigma_real**2)*dt
                                 + sigma_real*math.sqrt(dt)*Z[i-1])
    premium = bs_call(S0, K, T, r, sigma_imp) + bs_put(S0, K, T, r, sigma_imp)
    cash = -premium                    # 买入跨式先付权利金
    shares = 0.0
    for i in range(1, N + 1):
        if i % rebal_every == 0:       # 每 rebal_every 个交易日对冲一次
            Ti = max(T - t[i], 1e-6)
            dS = straddle_delta(S[i], K, Ti, r, sigma_imp)
            target = -dS                # 持 -Delta 股保持中性
            trade = target - shares
            cash += -trade * S[i]       # 买卖标的的现金流
            shares = target
    cash += shares * S[N] + abs(S[N] - K)   # 到期：平标的 + 跨式内在价值
    return S, cash
```

注意一个工程细节：**对冲时用的 Delta 是用隐含波动 $\sigma_{\text{imp}}$ 算的**，而不是真实波动。这正是 scalping 利润的源头——你用一个「错误的」波动假设去对冲一个按真实波动运动的标的，误差在 $\sigma_{\text{real}}\neq\sigma_{\text{imp}}$ 时直接变现成 P&L。

![一条 28% 已实现波动路径上的动态对冲：标的走势与对冲组合累计盈亏](/images/gamma-scalping-options/gamma_path_pnl.png)

上图是一条典型路径：标的在区间里上上下下，对冲组合从 −5.5（权利金）起步，靠反复低买高卖，到期累计转正。关键不是某一天赚多少，而是**只要波动够大、来回够多，锯齿状的 scalping 现金流会系统性地把 Theta 填平并反超**。

## 四、盈亏分布：已实现波动才是真正的裁判

把上面单条路径扩展到 4000 条，分别用「高已实现波动 30%」和「低已实现波动 12%」两档驱动标的，看对冲组合的到期 P&L 分布：

![盈亏分布：已实现波动高于隐含波动时策略才真正赚钱](/images/gamma-scalping-options/gamma_pnl_dist.png)

结果非常刺眼也非常重要：

- **已实现 30%（高于隐含 20%）**：均值 **+2.76**，胜率 **99.2%**——波动真的比期权卖你时更疯，scalping 大赚；
- **已实现 12%（低于隐含 20%）**：均值 **−2.19**，胜率 **0%**——市场太安静，Theta 每天流血却薅不到羊毛，几乎必亏。

这就是 Gamma Scalping 的全部真相：**它不是「波动越大越好」的无脑策略，而是「实际波动 > 你付出的隐含波动」的价差交易。** 你卖空的是波动率风险溢价（VRP）的反面——你在赌 VRP 为负，即实际波动会超过市场定价。这也解释了为什么专业交易员做 scalping 前，一定会先看 VIX / IV 相对历史 RV 的位置：便宜的 Gamma 才值得薅。

进一步，可以把「打平所需的已实现波动」显式算出来。把一天内 scalping 的 Gamma 收益 $\tfrac12\Gamma\mathbb{E}[(\Delta S)^2]=\tfrac12\Gamma\sigma_{\text{real}}^2 S^2\Delta t$ 与 Theta 损耗 $\Theta\Delta t$ 放在同一格里持平，得到近似打平条件 $\sigma_{\text{real}}^2 \approx -2\Theta/(\Gamma S^2)$。代入我们的参数（ATM、剩余 1 个月、隐含 20%），反解出的打平已实现波动落在 19%~20% 附近——几乎等于你买入时付的隐含波动。换句话说：**你买的不是「波动」，而是「比 20% 更疯的波动」；任何低于这条线的行情都在给你放血。** 这也是 scalping 者盯着 RV/IV 价差入场、而非盯着波动率绝对值的原因。

## 五、对冲频率：再平衡越勤，Gamma 抓得越满

连续对冲（每秒）理论上能把 Gamma P&L 精确兑现；现实中只能离散再平衡。我们用已实现 28% 一档，比较每 1 / 2 / 3 / 5 / 10 / 21 个交易日对冲一次的期望 P&L：

![对冲频率：再平衡越频繁，Gamma 抓取越充分](/images/gamma-scalping-options/gamma_freq_pnl.png)

曲线清楚地告诉我们：**再平衡越频繁，期望 P&L 越高**——因为离散对冲的误差（每次错过的一段 $\Delta S$）在频繁调仓下被压到最小，Gamma 的凸性收益被最大程度兑现。但真实世界有交易成本：每多一次调仓就多付一次双边手续费和买卖价差。所以实务上的最优频率，是「更频繁带来的 Gamma 增益」恰好覆盖「更频繁带来的交易成本」那一点——通常日内 scalping 在分钟级、跨式 scalp 在日频到小时频之间找平衡点。

## 六、三个必须直说的真实陷阱

1. **交易成本会吃掉一切。** 上面的盈利都没算手续费、买卖价差、冲击成本。Gamma 越高、对冲越勤，成本斜率越陡。实操里 scalping 的净利润往往比「零成本」曲线低一大截，甚至转负。务必在回测里把每次调仓的单边成本算进去。

2. **跳空（Gap）与钉住风险（Pin risk）。** GBM 假设连续路径，但财报、非农、加息这类事件会让标的隔夜跳空，你的 Delta 在跳空那一格完全失效，Gamma 凸性收益变成跳空损失。临近到期、标的价格正好卡在行权价附近时（pin risk），结算不确定性会让对冲头寸暴露方向性风险。

3. **Vega 与波动率曲面的二阶风险。** 跨式同时是负 Theta、正 Vega。如果 scalping 期间 IV 整体下移（vol crush，常见于事件兑现后），Vega 亏损会和 Theta 一起咬你。Gamma Scalping 赚的是**路径上的波动**，不是 IV 水平；别把它和「做多波动率」混为一谈。

最后一句收口：Gamma Scalping 是把「波动」这件商品低买高卖的对冲手艺，它的 alpha 不来自预测，而来自**你以隐含波动买入、以已实现波动结算**的那道价差。把它当成 VRP 的反向下注来风控，而不是当成「波动大就赚」的印钞机——后者在实盘里会被成本和跳空教育得很惨。

> 本文所有图表均由 Python（NumPy + Matplotlib）基于 Black-Scholes 定价与 GBM 蒙特卡洛模拟生成，数据为合成但结构贴近真实期权，仅用于方法演示，不构成任何交易建议。
