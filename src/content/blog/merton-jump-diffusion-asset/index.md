---
title: "Merton 跳跃扩散资产定价：把跳变成分从连续波动里剥出来"
description: "Black-Scholes 假设价格连续游走，可股灾里价格是「跳」下去的。本文 numpy 从零实现 Merton(1976) 跳跃扩散闭式期权定价(对跳数次 n 求 BS 无穷级数)，12 万路径 Monte Carlo 校验误差仅 0.019；ATM 期权价 15.38 vs 无跳 BS 9.41(跳变溢价 +63%)，并把价格拆开：连续扩散占 10.6%、跳变占 89.4%。附完整 Python 与四张真实计算图。"
publishDate: '2026-08-30'
tags:
  - 量化交易
  - 期权定价
  - Merton 跳跃扩散
  - 跳变风险
  - 蒙特卡洛
  - 厚尾
  - 波动率微笑
  - Python
language: Chinese
difficulty: advanced
---

Black-Scholes 最要命的假设是「价格连续游走」。可你我都见过：一只股票盘后暴雷，第二天直接低开 15%；一次闪崩，指数在几分钟内砸掉 5%。这些都不是布朗运动那条光滑曲线能描述的——**价格是「跳」下去的**。如果你用纯 BS 去给指数期权、单只暴雷股期权、甚至 crypto 期权定价，一定会系统性低估尾部风险，因为 BS 把跳变这一整块概率质量给抹平了。

Merton (1976) 的跳跃扩散模型就是在 BS 上「加跳」：在几何布朗运动里，定期按泊松过程插入对数正态的价格跳变。这个模型牛的地方在于——**它仍然有闭式解**，而且这个闭式解天然把期权价格拆成了「连续扩散」和「跳变」两部分。本文用 numpy 从零实现 Merton 闭式定价（对跳数次 $n$ 求 BS 无穷级数），用 12 万路径 Monte Carlo 校验误差仅 0.019，并回答一个最实际的问题：**ATM 期权里，跳变成分到底占多少？** 答案是——接近 90%。

![同一起点、同一漂移：跳跃扩散路径出现不连续的跳](/images/merton-jump-diffusion-asset/mjd_paths.png)

## 一、模型设定

Merton 假设对数价格由两部分叠加：

$$d\ln S_t = \big(\mu - \tfrac12\sigma^2 - \lambda\kappa\big)dt + \sigma\,dW_t + J_t$$

- $\sigma\,dW_t$：连续的布朗扩散（对应 BS 的波动率）
- $J_t$：跳变，发生频率是泊松强度 $\lambda$（每年平均 $\lambda$ 次），每次跳的大小 $\ln(1+J)\sim\mathcal N(m,\delta^2)$
- $\kappa = \mathbb E[J] = e^{m+\delta^2/2}-1$：价格跳的平均超额回报，漂移里减去 $\lambda\kappa$ 保证风险中性下期望漂移还是 $r$

设定一组有代表性的参数（指数期权尺度）：

```python
import numpy as np
from scipy.stats import norm

S0, T_, r, q = 100.0, 1.0, 0.03, 0.0
SIG, LAM, M, DELTA = 0.20, 3.0, -0.10, 0.15   # 扩散波动 / 年跳跃强度 / 跳size对数均值 / 对数std
KAPPA = np.exp(M + 0.5*DELTA**2) - 1.0         # E[J]-1
print(f"价格跳平均超额 κ = {KAPPA:.4f}")       # ≈ -0.0849（平均每次跳跌 8.5%）
```

这里 $\lambda=3$ 表示平均每年 3 次跳，$m=-0.10$ 表示跳偏向下、平均跌约 10%，$\delta=0.15$ 控制跳的离散度。注意 $\kappa\approx-0.085$：跳是净向下的，所以风险中性漂移要额外减掉 $\lambda\kappa$（即加回正项）来补偿。

## 二、闭式解：对跳数次 n 求 BS 无穷级数

Merton 的漂亮结论：在跳变存在下，看涨期权价格等于「把 Poisson 跳数次 $n=0,1,2,\dots$ 全部分支的 BS 价格加权平均」，权重是泊松概率 $p_n=e^{-\lambda T}(\lambda T)^n/n!$，而每个分支里标的被调整到 $S\cdot e^{-\lambda\kappa T+n(m+\delta^2/2)}$、波动被调到 $\sqrt{\sigma^2+n\delta^2/T}$。

```python
def bs_call(S, K, T_, r_, q_, sig):
    if sig <= 0 or T_ <= 0: return max(S-K, 0.0)
    d1 = (np.log(S/K) + (r_ - q_ + 0.5*sig**2)*T_) / (sig*np.sqrt(T_))
    d2 = d1 - sig*np.sqrt(T_)
    return S*np.exp(-q_*T_)*norm.cdf(d1) - K*np.exp(-r_*T_)*norm.cdf(d2)

def merton_call(S, K, sig, lam, m, delta, Nmax=60):
    tot = 0.0
    pn = np.exp(-lam*T_)                     # p_0 = e^{-λT}
    for n in range(Nmax + 1):
        sn = np.sqrt(sig**2 + n*delta**2/T_)                         # 分支波动
        Sn = S*np.exp(-lam*KAPPA*T_ + n*(m + 0.5*delta**2))          # 分支标的
        tot += pn * bs_call(Sn, K, T_, r, q, sn)
        pn *= (lam*T_) / (n + 1)            # 递推 p_{n+1} = p_n·(λT)/(n+1)
    return tot

price = merton_call(S0, 100, SIG, LAM, M, DELTA)
print(f"Merton ATM 闭式价 = {price:.3f}")    # ≈ 15.382
```

级数收敛极快（一般 10 项就够），`Nmax=60` 纯属保险。这就是 Merton 模型的工程优势：**定价是 O(级数项数) 的闭式循环，比 Monte Carlo 快几个数量级，且天然可微分**——做 Greeks、做校准都方便。

## 三、Monte Carlo 校验：误差 0.019

闭式解写得对不对？最朴素的验证是 Monte Carlo 直接模拟跳跃扩散路径，对比两者：

```python
def mc_merton_call(S, K, sig, lam, m, delta, M_=120000, seed=7):
    rng = np.random.default_rng(seed)
    Nr = rng.poisson(lam*T_, M_)              # 每条路径的跳次数
    Z  = rng.standard_normal(M_)
    total = int(Nr.sum())
    allj = rng.normal(m, delta, total)        # 所有跳的 size，拼成一维再切回去
    idx = np.cumsum(Nr); jsum = np.zeros(M_)
    if idx[0] > 0: jsum[0] = allj[:idx[0]].sum()
    for i in range(1, M_):
        if idx[i] > idx[i-1]: jsum[i] = allj[idx[i-1]:idx[i]].sum()
    drift = (r - q - lam*KAPPA - 0.5*sig**2)*T_
    ST = S*np.exp(drift + sig*np.sqrt(T_)*Z + jsum)
    return np.exp(-r*T_)*np.maximum(ST-K, 0).mean()

# 在 13 个行权价上对比闭式 vs MC
Ks = np.linspace(70, 130, 13)
cf = [merton_call(S0,k,SIG,LAM,M,DELTA) for k in Ks]
mc = [mc_merton_call(S0,k,SIG,LAM,M,DELTA) for k in Ks]
print(f"闭式 vs MC 最大误差 = {max(abs(a-b) for a,b in zip(cf,mc)):.4f}")   # 0.0187
print(f"ATM: Merton={cf[6]:.3f}  MC={mc[6]:.3f}  BS无跳={bs_call(S0,100,T_,r,q,SIG):.3f}")
# 最大误差 0.0187 ; ATM Merton 15.382 / MC 15.383 / BS 9.413
```

![1年对数收益分布：Merton 超额峰度远大于高斯(厚尾)](/images/merton-jump-diffusion-asset/mjd_return_dist.png)

误差 0.0187（相对 ATM 价约 0.12%），证明闭式公式和 MC 完全一致。而同一档 ATM 期权，**无跳 BS 只要 9.41，Merton 要 15.38**——差的那 6 块钱，就是「跳变溢价」。

## 四、期权曲线与跳变溢价

把整个行权价轴上的价格画出来，并减去同参数 BS 价，得到「跳变溢价」曲线：

```python
bs_only = [bs_call(S0, k, T_, r, q, SIG) for k in Ks]
jump_premium = np.array(cf) - np.array(bs_only)
```

![期权价格曲线与跳变溢价：低行权价处跳变贡献最大](/images/merton-jump-diffusion-asset/mjd_option_curve.png)

两个观察：(1) **跳变让整条曲线上移**，因为向下跳会推高看跌保护需求、间接抬高看涨（通过 put-call parity）；(2) **低行权价（深度虚值看跌 / 深度实值看涨）处跳变溢价最大**——这正是市场恐慌时深度 OTM 看跌期权被爆买、波动率微笑右翼翘起的结构性来源之一。Merton 模型能天然复现波动率微笑，而 BS 给的是一条平的直线。

## 五、把价格剥开：连续扩散 vs 跳变

Merton 闭式最被低估的价值——它能**直接分解**期权价里「连续扩散」和「跳变」各占多少。分支 $n=0$ 是没有跳的纯扩散（就是 BS 项），其余 $n\ge 1$ 全是跳变贡献：

```python
pn = np.exp(-LAM*T_); diff_part = 0.0; jump_part = 0.0
for n in range(61):
    sn = np.sqrt(SIG**2 + n*DELTA**2/T_)
    Sn = S0*np.exp(-LAM*KAPPA*T_ + n*(M + 0.5*DELTA**2))
    contrib = pn*bs_call(Sn, 100, T_, r, q, sn)
    if n == 0: diff_part = contrib
    else:      jump_part += contrib
    pn *= (LAM*T_)/(n+1)
final = diff_part + jump_part
print(f"连续扩散 n=0 占 {diff_part/final:.1%}；跳变 n>=1 占 {jump_part/final:.1%}")
# 连续扩散 10.6% / 跳变 89.4%
```

![闭式级数收敛，且 ATM 价中跳变成分占 89.4%](/images/merton-jump-diffusion-asset/mjd_convergence.png)

在这个参数下（每年 3 次跳、平均跌 8.5%），**ATM 期权价的 89.4% 来自跳变，连续扩散只贡献 10.6%**。这是反直觉但极重要的结论：很多人以为「期权价格主要是波动率的钱」，但在高跳变强度的资产上，期权费大部分是**为尾部跳变买的保险**。这直接解释了为什么纯 BS 隐含波动率会系统性偏低、为什么股灾前 OTM 看跌异常贵。

## 六、已知偏差与适用边界

- **跳变参数难估**：$\lambda, m, \delta$ 比 $\sigma$ 更难从有限样本里稳定估计，尤其 $\delta$ 对尾部极端值极度敏感。实盘建议用期权隐含校准而非历史估计。
- **对数正态跳的局限**：Merton 假设跳 size 对数正态，意味着跳也是「温和厚尾」。真实股灾的跳可能更肥（需用 CGMY / Kou 双指数跳）。
- **常数参数**：本文固定 $\sigma,\lambda$ 不随时间变。实盘里跳强度在危机期骤升，需用体制切换或随机波动率跳（Bates 模型）升级。
- **蒙特卡洛方差**：校验时 MC 误差 0.019 已够小，但若用更少路径（如 1 万），误差会涨到 0.05+，可能掩盖闭式实现的 bug——**校验路径数不能太省**。

## 七、小结

Merton 跳跃扩散用一根泊松跳，把 BS 抹平的尾部风险重新请了回来，而且**保留了闭式解**。本文从模型设定、无穷级数闭式定价、12 万路径 MC 校验（误差 0.019）、到把 ATM 期权价剥成「连续扩散 10.6% + 跳变 89.4%」完整跑通，并解释了为什么它能复现波动率微笑、为什么无跳 BS 会系统性低估期权费（本例 ATM 低估 63%）。**当你给「会跳」的资产定价时，跳变才是那张期权大部分价格的真实来源。**

附完整 Python 与四张真实计算图（路径对比 / 收益分布厚尾 / 期权曲线与跳变溢价 / 级数收敛与成分分解）。
