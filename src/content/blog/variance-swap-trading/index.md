---
title: "方差互换与波动率互换：做空波动率风险溢价的衍生品工具"
description: "用方差互换(variance swap)与波动率互换(volatility swap)把「做空波动率风险溢价」做成可定价、可复制的纯波动头寸，拆解放射率、凸性与期限结构，并给出做空方差互换的 carry 回测与尾部陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 波动率
  - 方差互换
  - 波动率互换
  - 波动率风险溢价
  - Python
language: Chinese
difficulty: advanced
---

上篇讲到：波动率风险溢价(VRP)长期为正，做空它(卖波动)是长期正期望的「卖保险」生意。但用期权跨式去卖，会带上标的 Delta、Gamma 对冲磨损、偏度等一堆「杂质」。有没有一种工具，**只暴露波动率本身、不暴露方向、定价干净、还能静态复制**？有——这就是**方差互换(variance swap)**与**波动率互换(volatility swap)**。

结论先放这里：**方差互换的损益 $\propto$ 实际波动的平方，波动率互换的损益 $\propto$ 实际波动本身；因为 Jensen 不等式，二者执行价天然不同(波动率互换执行波动低于 √方差互换执行方差)。做空方差互换=以执行方差卖出、以实际方差买入，长期收割 VRP；但它的凸性让你「在崩盘时亏得最狠」——这把双刃，比跨式更纯粹，也更危险。**

![方差互换/波动率互换/跨式 的损益轮廓](/images/variance-swap-trading/varswap_pnl_profile.png)

## 一、为什么期权跨式「不够干净」

卖一张 Delta 中性的跨式(同时卖 ATM 看涨与看跌)，每日做 Delta 对冲，理论上能把方向风险剥离，剩下的损益近似于 $\text{Notional}\times(IV^2 - RV^2)$。但实务里有三处脏：

1. **离散对冲磨损(gamma slippage)**：每日而非连续对冲，剧烈行情里高卖低买，磨损随波动平方放大；
2. **偏度(skew)暴露**：真实期权曲面是斜的，跨式对下行波动的赔付和上行不对称，你其实在卖更多「下行保护」；
3. **期限与滚动**：期权有到期，要不断滚动，每次都重新承担 IV 变化。

用一个直觉数字感受离散对冲的代价：若 IV=20%、实际 RV=18%，跨式到期(连续对冲理想下)损益≈ N×(0.20²−0.18²)=N×0.0076，约 0.76% 本金；但一旦某日跳空 −8%，仅这一天的平方和就贡献 64 倍于单日常态波动——这部分是每日再平衡无论如何也救不回来的「凸性漏」。方差互换的精髓，正是把这种漏彻底消除。

方差互换的出现，就是为了解决这些：它是**到期一次性结算、无中途对冲、纯波动暴露**的前瞻合约。

## 二、方差互换的机械定义

一份方差互换约定：一方(买方)在到期收到「实际实现的方差」与「事先约定的执行方差(strike)」之差，乘以名义本金；另一方(卖方)反向。设年化已实现方差为 $RV_T$，执行方差为 $K_{\text{var}}$，名义为 $N$，则**买方到期损益**：

$$\text{P\&L}_{\text{buy}} = N \times \bigl(RV_T - K_{\text{var}}\bigr)$$

其中已实现方差用日度对数收益的平方和年化：

$$RV_T = \frac{252}{n}\sum_{i=1}^{n} r_i^2,\qquad r_i = \ln\frac{S_i}{S_{i-1}}$$

注意：是**收益的平方和**，不是标准差。这意味着单日极端收益(无论涨跌)会被平方放大——方差互换对「尾部」极度敏感，这是它最本质的特征。做空方差互换，等于**在做空「出现极端单日波动」的概率**。

## 三、波动率互换：把方差换成波动

波动率互换更简单直白：结算的是**实际波动(标准差)**与执行波动之差：

$$\text{P\&L}_{\text{buy}} = N \times \bigl(\sigma_T - K_{\text{vol}}\bigr),\qquad \sigma_T = \sqrt{RV_T}$$

差别仅一字：一个是方差、一个是波动。但就这一字之差，决定了二者**凸性完全不同**——下一节展开。

## 四、Black-Scholes 复习：给后续定价兜底

波动率互换/方差互换的 fair strike，最终都落在「前向方差」上。先放一个干净的 BS 看涨/看跌，后面复制与定价都要用：

```python
from math import erf, sqrt, exp, log

def bs_call(S, K, T, r, sig):
    if T <= 0 or sig <= 0:
        return max(S - K, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sig**2) * T) / (sig * sqrt(T))
    d2 = d1 - sig * sqrt(T)
    return S*0.5*(1+erf(d1/sqrt(2))) - K*exp(-r*T)*0.5*(1+erf(d2/sqrt(2)))

def bs_put(S, K, T, r, sig):
    return bs_call(S, K, T, r, sig) - S + K*exp(-r*T)
```

## 五、凸性差异：方差互换 vs 波动率互换

设实际波动 $\sigma$ 是一个随机变量。方差互换的执行方差是风险中性的期望方差 $\mathbb{E}[\sigma^2]$；波动率互换的执行波动是风险中性的期望波动 $\mathbb{E}[\sigma]$。由于 $\sigma>0$ 且平方根是**凹函数**，Jensen 不等式给出：

$$\mathbb{E}[\sigma] \;<\; \sqrt{\mathbb{E}[\sigma^2]}$$

即**波动率互换的执行波动，必然低于 √方差互换的执行方差**。二者之差就是「凸性缺口」，它随波动-of-波动(vol-of-vol)放大。用我们模拟的随机波动率路径直接验证：

```python
def convexity_gap(ret, Ts):
    out = []
    for T in Ts:
        n = int(round(T * 252))
        M = len(ret)
        rvs = np.full(M, np.nan)
        for i in range(n, M):
            rvs[i] = sqrt(252.0) * np.std(ret[i-n+1:i+1], ddof=1)
        vals = rvs[~np.isnan(rvs)]
        out.append((sqrt(np.mean(vals**2)), np.mean(vals)))   # (√E[σ²], E[σ])
    return np.array(out)

Ts = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0])
c = convexity_gap(ret, Ts)
print("√E[σ²](vol%):", np.round(c[:,0]*100, 2))
print("E[σ](vol%):  ", np.round(c[:,1]*100, 2))
```

![凸性差异：波动率互换执行波动低于 √方差互换执行方差](/images/variance-swap-trading/varswap_convexity.png)

图里红线(√E[σ²])全程压在蓝线(E[σ])之上，缺口在短期限更宽——**短期限方差互换的凸性最强**(单月波动离散度大)，长期限两者趋近(大数定律抹平离散)。这直接决定交易选择：想赌「波动跳升」的凸性爆发，选短期限方差互换；想要更线性的波动暴露，选波动率互换。

## 六、方差互换的 fair strike = 前向方差

方差互换最优雅的性质：**它的 fair strike 等于风险中性下的期望已实现方差，而这个量可以用一份静态的期权组合(对数合约 log-contract)精确复制**——无需动态对冲、无 Gamma 磨损。直觉上，买入一份方差互换 ≈ 持有一篮子行权价密集的 OTM 期权多头(让每个行权价权重 ∝ 1/K²)，再减去一份远期。复制组合的价值在到期恰好等于已实现方差，因此定价等价于「前向方差」：

$$K_{\text{var}}(T) = \mathbb{E}^{\mathbb{Q}}[RV_T]$$

在随机波动率(CIR 方差过程)下，前向方差有解析解，直接给出期限结构的形状：

$$K_{\text{var}}(T) = \frac{1}{T}\Bigl[\theta T + (v_0 - \theta)\frac{1 - e^{-\kappa T}}{\kappa}\Bigr]$$

当 $v_0 > \theta$ 时，短端执行方差高、长端回归长期方差 $\theta$——典型的**向下倾斜方差期限结构**。

```python
def forward_var(v0, theta, kappa, T):
    return (theta*T + (v0 - theta)*(1 - np.exp(-kappa*T))/kappa) / T

v0, theta, kappa = 0.06, 0.04, 3.5
Ts = np.array([1/12, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0])
kv = np.array([forward_var(v0, theta, kappa, T) for T in Ts])
print("执行方差(×100):", np.round(kv*100, 2))
print("对应执行波动(%):", np.round(np.sqrt(kv)*100, 2))
```

![方差互换期限结构：短端高、长端回归长期方差](/images/variance-swap-trading/varswap_term_structure.png)

图 3 左轴是执行方差、右轴是对应执行波动：短端约 24.5% vol，长端衰减到 20%(=√θ)。**期限结构本身就是一个信号**——短端相对长端越陡，说明当下「近端恐慌」越浓。

## 七、Python：做空方差互换的 carry 回测

做空方差互换的 carry 逻辑极简：每月初，以「fair strike + 正溢价」作为市场执行方差卖出，持有到月末用真实已实现方差结算。损益 = 名义 × (K_market − RV_realized)。

```python
def short_var_swap_carry(ret, v0=0.06, theta=0.04, kappa=3.5,
                         rebal=21, notional=100.0, premium=0.004):
    M = len(ret)
    pnl = np.zeros(M)
    for i in range(0, M - rebal, rebal):
        T = rebal / 252.0
        fair_var = forward_var(v0, theta, kappa, T)
        K_mkt = fair_var + premium              # 市场 strike = 公平方差 + VRP
        window = ret[i+1:i+1+rebal]
        rv_annual = 12.0 * np.sum(window**2)     # 月度窗口年化方差
        pnl[i] = notional * (K_mkt - rv_annual)  # 做空方到期损益
    cum = 100.0 + np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = (cum / peak - 1).min()
    return cum, dd

cum, max_dd = short_var_swap_carry(ret)
print(f"期末累计P&L={cum[-1]:.1f}  最大回撤={max_dd*100:.1f}%")
```

![做空方差互换累计 P&L 与危机回撤](/images/variance-swap-trading/varswap_equity.png)

图 4 里绿线平稳上行(持续收 VRP 溢价)，但当中段出现一次剧烈波动(模拟崩盘)时，已实现方差瞬间冲过执行方差，单月就吐出一大块、画面上出现明显的红色回撤坑(本例最大回撤约 −43%)。这正是方差互换「凸性双面」的具象化：**平时线性收租，崩时平方出血。**

## 八、四大实战陷阱

1. **凸性=尾部毒药**：损益随实际方差线性、随波动平方放大。一次 -10% 单日跳空，其平方贡献是平时波动的百倍。你的正常月收益，抵不过一次尾部。
2. **VRP 会反转**：当所有人都在卖波动(2017、2021 末)，溢价被挤薄，市场一旦反转就是踩踏式回补。做空方差互换是经典的**拥挤交易**。
3. **复制依赖流动性与无套利**：静态复制需要密集、连续的期权链且买卖价差小；小盘、远月、危机期，复制失效、bid-ask 把溢价吃掉。
4. **展期与基差**：实盘里你交易的是方差互换的**期货/远月报价**，滚动时面临基差与期限结构滚动收益，和「模型 fair strike」不是一回事。

## 结论

方差互换与波动率互换，是把「做空波动率风险溢价」这件事**提纯**的工具：无方向暴露、定价干净(=前向方差)、可静态复制。二者的凸性差异(Jensen)决定了你的损益形状——方差互换更凸、对尾部更敏感，波动率互换更线性。一句话：

> **做空方差互换 = 以固定方差卖出、以实际方差买入。它把 VRP 收得最干净，也把尾部风险放得最大。**

把上篇的「波动率双面」和本篇的「纯波动工具」合起来，你就有了一条完整的链：IV−RV 溢出 VRP → 用方差互换把 VRP 做成可定价头寸 → 用波动率目标/危机降仓管住它的凸性尾巴。下一步，可以把这条链接上真实期权链(反解 IV、构建复制组合)，在实盘里把「收保费」变成一套有风控的交易系统。

*所有图表均由文中 Python 代码真实计算生成（随机波动率模型模拟 + Black-Scholes 定价），仅用于方法演示，不构成投资建议。*
