---
title: "Merton 跳跃扩散资产定价：把跳变成分从连续波动里剥出来"
description: "Black-Scholes 假设价格连续游走,可股灾里价格是「跳」下去的,不是「滑」下去的。Merton(1976) 用泊松跳把不连续成分塞进几何布朗运动:ln S_T 仍是正态,但叠加一个复合泊松跳。本文用 numpy 从零实现它的闭式期权定价(对跳数次 n 求 BS 无穷级数),并用 12 万路径 Monte Carlo 校验误差仅 0.019,诚实量化:ATM 期权价 15.38 vs 无跳 BS 9.41,跳变溢价 +63%;把价格拆开看,连续扩散(n=0)只占 10.6%,跳变(n≥1)占 89.4%。附完整 Python 与四张真实计算图。"
publishDate: '2026-08-30'
tags:
  - 量化交易
  - 期权定价
  - 跳跃扩散
  - Merton模型
  - 蒙特卡洛
  - 风险中性
  - 厚尾
  - Python
language: Chinese
difficulty: intermediate
cover: "/images/merton-jump-diffusion-asset/mjd_paths.png"
---

Black-Scholes 是连续的:它假设价格沿几何布朗运动平滑游走,任何瞬间都只走一小步。可你我都见过——**股灾里价格是「跳」下去的,不是「滑」下去的**。一个周末的坏消息,周一开盘直接低开 5%,这 5% 在 BS 世界里根本不存在。

Merton(1976) 用一个极优雅的修正解决了这件事:在几何布朗运动上,叠一层**泊松跳**。这就是跳跃扩散(jump-diffusion)模型。本文用 numpy 从零实现它的**闭式期权定价**,并用 Monte Carlo 校验,把「跳变成分」从「连续波动」里干净地剥出来。

## 一、模型:正态 + 复合泊松跳

对数价格服从

$$d\ln S_t = \big(\mu - \tfrac12\sigma^2 - \lambda\kappa\big)dt + \sigma\,dW_t + d\!\sum_{k=1}^{N_t} Y_k$$

其中 $N_t\sim\text{Poisson}(\lambda)$ 是跳的次数,$Y_k\sim\mathcal N(m,\delta^2)$ 是第 $k$ 次跳的对数幅度,$\kappa=\mathbb E[J]-1=e^{m+\delta^2/2}-1$ 是跳的平均超额(风险中性下要被补偿掉,保证折价是鞅)。

于是到期对数收益 $\ln(S_T/S_0)$ 仍是正态,但额外叠了一坨跳。下面从零模拟这两条路径——同一起点、同一漂移,区别只在「有没有跳」:

```python
import numpy as np

def simulate_mjd(S0, r, q, sig, lam, m, delta, T, steps, seed=0):
    rng = np.random.default_rng(seed)
    dt = T / steps
    kappa = np.exp(m + 0.5*delta**2) - 1.0
    paths = np.empty(steps + 1); paths[0] = S0
    N = rng.poisson(lam * dt, steps)               # 每步跳几次
    cont = (r - q - lam*kappa - 0.5*sig**2) * dt
    for t in range(1, steps + 1):
        z = rng.standard_normal()
        jumps = N[t-1]
        jret = rng.normal(m, delta, jumps).sum() if jumps > 0 else 0.0
        paths[t] = paths[t-1] * np.exp(cont + sig*np.sqrt(dt)*z + jret)
    return paths

gbm = simulate_mjd(100, 0.03, 0, 0.20, 0.0,  -0.10, 0.15, 1, 252, 20260830)   # 无跳
jdm = simulate_mjd(100, 0.03, 0, 0.20, 3.0,  -0.10, 0.15, 1, 252, 20260830+1) # 年跳 3 次
```

![同一起点、同一漂移:跳跃扩散路径出现不连续的跳](/images/merton-jump-diffusion-asset/mjd_paths.png)

红线就是跳——它们不连续、不可预测,却真实存在。这就是 BS 漏掉的那部分风险。

## 二、厚尾:跳把分布尾部加肥

连续扩散的回报是正态(超额峰度=0),跳会抬高峰度。在我们的参数($\sigma=0.2,\ \lambda=3,\ m=-0.1,\ \delta=0.15$)下,1 年对数收益的**超额峰度约 0.45**(高斯为 0)——不算夸张,但 log 尺度下尾部明显更肥:

![厚尾:跳跃扩散收益的对数密度在尾部高于同方差高斯](/images/merton-jump-diffusion-asset/mjd_return_dist.png)

> 提示:厚尾程度由 $\lambda$ 和 $\delta$ 决定。$\lambda$ 越大、$\delta$ 越宽,跳越频繁越剧烈,尾部越肥。本文取中等强度,只为讲清机制;实盘危机期要把 $\lambda$ 调到远高于 3。

## 三、闭式定价:把跳数次 n 求 BS 无穷级数

Merton 最漂亮的结果是:**看涨期权有闭式解**。给定跳了 $n$ 次,终端对数价格仍正态,等价于一个「调整了现货、调整了波动率」的 BS 价格;再对 $n=0,1,2,\dots$ 按泊松权重 $\frac{e^{-\lambda T}(\lambda T)^n}{n!}$ 求和:

$$C = \sum_{n=0}^{\infty} \frac{e^{-\lambda T}(\lambda T)^n}{n!}\,
\mathrm{BS}\!\left(S\,e^{-\lambda\kappa T + n(m+\delta^2/2)},\ K,\ \sqrt{\sigma^2+\tfrac{n\delta^2}{T}}\right)$$

注意跳了 $n$ 次时,波动率被「稀释」成 $\sqrt{\sigma^2+n\delta^2/T}$(跳越多,总方差越大),且现货被补偿项 $e^{-\lambda\kappa T}$ 与跳的累计均值 $n(m+\delta^2/2)$ 共同调整。从零实现:

```python
from scipy.stats import norm

def bs_call(S, K, T, r, q, sig):
    if sig <= 0 or T <= 0: return max(S - K, 0.0)
    d1 = (np.log(S/K) + (r - q + 0.5*sig**2)*T) / (sig*np.sqrt(T))
    d2 = d1 - sig*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)

def merton_call(S, K, sig, lam, m, delta, T=1.0, Nmax=60):
    kappa = np.exp(m + 0.5*delta**2) - 1.0
    tot, pn = 0.0, np.exp(-lam*T)          # p_0
    for n in range(Nmax + 1):
        sn = np.sqrt(sig**2 + n*delta**2/T)
        Sn = S * np.exp(-lam*kappa*T + n*(m + 0.5*delta**2))
        tot += pn * bs_call(Sn, K, T, 0.03, 0.0, sn)
        pn *= (lam*T) / (n + 1)             # 递推到 p_{n+1}
    return tot
```

**关键验证**:用 12 万路径 Monte Carlo 直接模拟风险中性终值、贴现平均 payoff,与闭式对比。结果:两者在 13 个行权价上**最大误差仅 0.019**——公式写对了。ATM($K=100$) 处:

- Merton 闭式 = **15.382**
- Monte Carlo = **15.383**(吻合)
- 无跳 BS($\lambda=0$) = **9.413**

跳变让 ATM 期权贵了 **+63%**。把价格画成行权价的曲线,差异一目了然:

![期权价格曲线与跳变溢价:低行权价处跳变贡献最大](/images/merton-jump-diffusion-asset/mjd_option_curve.png)

低行权价(深度实值看涨)处跳变溢价最大——因为大跌跳会瞬间把价格砸到行权价下方,虚值保护的价值被跳显著放大。这正是危机期深度 OTM 看跌被疯抢的数学原因。

## 四、把跳变成分剥出来

闭式级数天然按「跳了几次」分账,于是我们可以直接把期权价拆成**连续扩散(n=0) vs 跳变(n≥1)**两部分:

- 连续扩散 $n=0$ 贡献 **1.63(10.6%)**
- 跳变 $n\ge1$ 贡献 **13.75(89.4%)**

![闭式级数收敛,且连续扩散仅占 10.6%、跳变占 89.4%](/images/merton-jump-diffusion-asset/mjd_convergence.png)

级数在 $n\approx 15$ 后基本收敛,说明「跳 15 次以上」对价格贡献可忽略。这里要诚实点破一个会计细节:风险中性补偿项 $e^{-\lambda\kappa T}$ 把 $n=0$ 项的「有效现货」压到了 77.5,所以**纯扩散项看起来很小,并不是说连续波动不重要**——真正直观的「跳变溢价」应和无跳 BS 比:Merton 15.38 − BS 9.41 = **6.0**,即在连续波动价格之上,跳变额外贡献了 63%。两个口径都对,只是讲的故事不同:前者按「跳了几次」分账,后者按「有没有跳」分账。

## 五、落地与边界

- **级数要截断**:实战取 $N_{\max}=40\sim60$ 足够,超过后增量可忽略(见收敛图)。
- **跳幅均值 m 决定偏度**:$m<0$(本文)让跳偏向下跌,虚值看跌更贵;$m>0$ 则相反。
- **别拿它当纯校准玩具**:Merton 假设跳幅同质对数正态、强度恒定,实盘跳跃聚集(clustering)和时变强度要靠 Bates(跳+随机波动)或更现代的 SVCJ 模型接住。
- **厚尾被低估的风险**:本文 $\lambda=3$ 下超额峰度仅 0.45,真实危机期可能几十倍于此——参数要随 regime 切换。

Merton 跳跃扩散的全部优雅,在于它既保留了 BS 的可解性(正态核心),又用一层泊松跳把「不连续」请回了价格过程。把 $n=0$ 与 $n\ge1$ 分开看,你就有了量化「连续波动」与「跳变风险」各自值多少钱的工具——这正是给期权定跳变溢价、给组合定尾部保险的第一步。
