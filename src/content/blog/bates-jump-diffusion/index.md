---
title: "跳跃扩散与 Bates 模型：给波动率加跳，给期权定价加尾"
publishDate: '2026-07-11'
description: "Black-Scholes 假设价格连续、波动恒定，所以算不出崩盘时的肥尾。Merton(1976) 给扩散加了对数正态跳，Bates(1996) 再把随机波动(Heston)叠上来。本文用特征函数+FFT 给欧式期权定价，与蒙特卡洛对照，并展示跳如何把隐含波动率的微笑扭成左偏的 smirk。"
tags:
  - 量化交易
  - 期权定价
  - Bates模型
  - 跳跃扩散
  - 隐含波动率
  - 蒙特卡洛
language: Chinese
difficulty: advanced
---

如果你用 Black-Scholes 给指数期权定价，会发现一个诡异的现象：**深度虚值看跌（保护下行用的那种）永远被你算便宜了**。市场实际愿意为"暴跌保险"付的钱，比 BS 公式算出来的高一大截。这不是市场非理性，是 BS 的假设太干净——它假设价格连续运动、波动率是常数，于是价格的对数收益是完美的高斯，左右尾巴一样薄。

但真实市场里，价格会**跳**：财报、降息、黑天鹅，都是一夜之间跳一大截。跳是单边的、下行的更多（崩盘时跳得狠），所以下行尾巴比高斯肥得多，上行尾巴也肥一点。本文明天要回答：**怎么在期权定价里把"跳"建模进去，并且用可复现的代码验证它真的加出了肥尾。** 这也是为什么 1987 年股灾之后，整个波动率曲面建模都绕不开跳——BS 在那一天亏掉的，正是它假装不存在的尾部。

路线分三步：Merton 跳跃扩散（只加跳）→ Bates（跳 + 随机波动）→ 用特征函数做解析定价并和蒙特卡洛对拍。

## 一、Merton 跳跃扩散：给几何布朗运动加跳

Merton (1976) 的想法很朴素：在几何布朗运动（GBM）上，叠加一个**复合泊松过程**的跳。

$$dS_t = (r - q) S_t dt + \sigma S_t dW_t + S_{t-} (e^{J_t} - 1) dN_t$$

- $N_t$ 是强度为 $\lambda$ 的泊松过程，平均每年跳 $\lambda$ 次；
- 每次跳的大小 $\ln(\text{跳跃倍数}) = J_t \sim \mathcal{N}(\mu_J, \sigma_J^2)$，所以跳是对数正态的；
- $S_{t-}$ 表示跳发生前的价格。

关键点：为了让**贴现后的股价仍是鞅**（否则定价会出套利），漂移要扣掉跳的期望贡献。单次跳的平均倍数是 $m = \mathbb{E}[e^{J}-1] = e^{\mu_J + \sigma_J^2/2} - 1$，所以漂移里要减去 $\lambda m$。

跳给对数收益加了一堆"尖峰"：大多数时间走高斯扩散，偶尔来一记重击。这就把左右尾巴都撑肥了——但如果跳的均值 $\mu_J < 0$（下行跳更大），**左尾会比右尾更肥**。

在 Merton 模型里，对数收益其实是"高斯 + 泊松混合"：跳发生 $k$ 次的概率是 $\text{Poisson}(\lambda T)$，给定 $k$ 次跳时对数收益仍是正态。于是期权价格可写成无穷级数——这也解释了为什么跳让尾部分布变成"指数衰减的高斯混合"，左右尾同时变肥，下行更甚。

## 二、Bates 模型：再叠加随机波动率

光有跳还不够。波动率本身也在变：恐慌时 VIX 飙到 30、40，平静时 12、13。Heston 用 CIR 过程让方差 $v_t$ 自己演化，Bates (1996) 把 Heston 和 Merton 焊在一起：

$$dv_t = \kappa(\theta - v_t)dt + \sigma_v\sqrt{v_t}dW_t^v, \quad dW^S dW^v = \rho\,dt$$
$$d\ln S_t = (r - q - \tfrac{1}{2}v_t - \lambda m)dt + \sqrt{v_t}dW_t^S + J_t dN_t$$

五个"额外"参数需要管：$\kappa$（方差均值回归速度）、$\theta$（长期方差）、$\sigma_v$（方差波动率）、$\rho$（杠杆效应，通常取负：跌时波动升）、$\lambda,\mu_J,\sigma_J$（跳的频率/均值/离散度）。

这一套组合拳的威力：**随机波动负责日常的波动聚集与微笑曲率，跳负责尾部肥度与下行偏斜（smirk）**。这正是真实指数期权曲面的样子。

## 三、用特征函数做解析定价（不用蒙特卡洛也能算）

对欧式期权，Bates 有闭式特征函数（Gatheral 约定下的 Heston 扩散部分，再乘上独立的跳 MGF）：

$$\phi(u) = \exp\Big(i u \ln S_0 + i u(r - q - \lambda m)T + C(u,T) + D(u,T)v_0 + \lambda T\big(e^{i u\mu_J - \tfrac{1}{2}u^2\sigma_J^2} - 1\big)\Big)$$

其中 Heston 部分的 $C, D$ 是标准的（见下方代码）。有了 $\phi(u)=\mathbb{E}[e^{iu\ln S_T}]$，就可以用 **Carr-Madan FFT** 一次性把整条行权价网格上的看涨价算出来——比逐点蒙特卡洛快几个数量级，而且光滑无噪声。

FFT 定价的核心（已踩过坑，详见文末）：特征函数给出的是 $\ln S_T$ 的 CF，Carr-Madan 的网格要用 $dk = 2\pi/(N\eta)$、以 $0$ 为中心，K 取 $\exp(k)$，不是 $S_0 e^k$。

```python
import numpy as np
from scipy.stats import norm

S0, r, q, T = 100.0, 0.02, 0.01, 0.5
v0, kappa, theta, sigma_v, rho = 0.04, 2.0, 0.04, 0.3, -0.7
lam, muJ, sigJ = 0.5, -0.05, 0.10
m = np.exp(muJ + 0.5*sigJ**2) - 1.0

def bates_cf(u):
    iu = 1j*u
    d = np.sqrt((kappa - rho*sigma_v*iu)**2 + sigma_v**2*(iu + u**2))
    g = (kappa - rho*sigma_v*iu + d) / (kappa - rho*sigma_v*iu - d)
    eT = np.exp(d*T)
    Ccoef = (kappa*theta/sigma_v**2)*((kappa - rho*sigma_v*iu + d)*T - 2*np.log((1-g*eT)/(1-g)))
    Dcoef = (kappa - rho*sigma_v*iu + d)/sigma_v**2*(1-eT)/(1-g*eT)
    phi_heston = np.exp(iu*(np.log(S0)+(r-q)*T) + Ccoef + Dcoef*v0)
    phi_jump = np.exp(lam*T*(np.exp(iu*muJ - 0.5*u**2*sigJ**2) - 1))
    return np.exp(iu*(-lam*m)*T) * phi_heston * phi_jump   # 漂移项补偿保证鞅

def fft_call_prices(cf, alpha=1.5, N=2**15, eta=0.05):
    dk = 2*np.pi/(N*eta); b = np.pi/eta
    k = -b + np.arange(N)*dk; v = eta*np.arange(N)
    w = np.zeros(N)
    for j in range(N):
        w[j] = 1.0 if (j==0 or j==N-1) else (4.0 if j%2==1 else 2.0)
    w *= eta/3.0
    psi = np.exp(-r*T)*cf(v-(alpha+1)*1j)/((alpha+1j*v)*(alpha+1+1j*v))
    x = np.exp(-1j*v*b)*psi*w
    C = np.real(np.fft.fft(x))*np.exp(-alpha*k)/np.pi
    return np.exp(k), C
```

## 四、和蒙特卡洛对拍：先确认公式没写错

解析公式很容易写错（我就在 FFT 网格映射上栽过一次）。最扎实的做法是**用蒙特卡洛独立验证**。下面用全截断 Euler 模拟 Bates 路径，和 FFT 价格对照：

```python
def bs_call(S,K,T,r,q,vol):
    d1=(np.log(S/K)+(r-q+0.5*vol**2)*T)/(vol*np.sqrt(T)); d2=d1-vol*np.sqrt(T)
    return S*np.exp(-q*T)*norm.cdf(d1)-K*np.exp(-r*T)*norm.cdf(d2)

def implied_vol(price,K,lo=1e-4,hi=5.0):
    for _ in range(60):
        mid=0.5*(lo+hi); p=bs_call(100,K,T,r,q,mid)
        lo,hi = (lo,mid) if p>price else (mid,hi)
    return 0.5*(lo+hi)

# FFT 定价
K_grid, C_grid = fft_call_prices(bates_cf)
# 蒙特卡洛（20 万路径，向量化）
def sim_terminal(n_paths, steps, seed):
    rng=np.random.default_rng(seed); dt=T/steps; lnS=np.full(n_paths,np.log(S0)); v=np.full(n_paths,v0)
    Z1=rng.standard_normal((steps,n_paths)); Z2=rng.standard_normal((steps,n_paths))
    e=rho*Z1+np.sqrt(1-rho**2)*Z2
    Nj=rng.poisson(lam*dt,(steps,n_paths)); J=Nj*muJ+np.sqrt(np.maximum(Nj,0))*sigJ*rng.standard_normal((steps,n_paths))
    for t in range(steps):
        vs=np.maximum(v,0); v=v+kappa*(theta-vs)*dt+sigma_v*np.sqrt(vs)*np.sqrt(dt)*e[t]; v=np.maximum(v,0)
        lnS=lnS+(r-q-lam*m-0.5*vs)*dt+np.sqrt(vs)*np.sqrt(dt)*Z1[t]+J[t]
    return np.exp(lnS)

mc=sim_terminal(200_000,63,seed=7)
for KK in [90,100,110]:
    idx=np.argmin(np.abs(K_grid-KK)); fft_p=C_grid[idx]
    mc_p=np.mean(np.maximum(mc-KK,0))*np.exp(-r*T)
    print(f"K={KK}: FFT={fft_p:.4f}  MC={mc_p:.4f}")
print("E[S_T]=",mc.mean()," 目标 S0*exp((r-q)T)=",S0*np.exp((r-q)*T))
```

跑出来：K=100 时 FFT=6.0475、MC=6.0573（差 0.01，1‰ 量级）；且 $\mathbb{E}[S_T]=100.44\approx 100.50=S_0 e^{(r-q)T}$，鞅性质成立。**公式可信。**

## 五、跳把微笑扭成 smirk

把 FFT 价格反解成 BS 隐含波动率，对比三种模型：

- BS（常数 20% 波动）：IV 是一条水平线；
- Merton（只加跳、无随机波动）：对称的**微笑**（两侧都翘）；
- Bates（跳+随机波动）：明显**左偏的 smirk**——越低行权价 IV 越高。

实测数值（半年期、S0=100）：

| 行权价 K | BS(常数) | Bates IV |
|---|---|---|
| 90（深度虚值看跌保护） | 20.0% | **23.2%** |
| 100（平值） | 20.0% | 20.7% |
| 110（虚值看涨） | 20.0% | 18.9% |

左侧偏斜（IV@90 − IV@110）高达 **4.3 个 vol 点**。这就是"崩盘保险溢价"的数学来源：下行跳让深度虚值看跌变贵，BS 看不见跳，所以系统性低估它。

这 4.3 个 vol 点的左偏不是小数点游戏：在 S0=100、T=0.5 时，IV 每差 1 个 vol 点，平值附近期权价格就差约 1–2%；深度虚值看跌对 IV 更敏感，跳带来的溢价会被放大成实打实的贵。换句话说，BS 给你的是"没有崩盘的世界"的价格，Bates 才把崩盘的概率质量如实计价。

![三类价格过程：GBM 平滑、Merton 有跳、Bates 同时有随机波动与跳](/images/bates-jump-diffusion/bates_paths.png)

![到期价格分布：跳把左尾（暴跌）显著加肥，随机波动再额外加宽](/images/bates-jump-diffusion/bates_terminal_hist.png)

![隐含波动率形态：跳产生偏斜，随机波动与跳共同塑造真实左侧肥尾](/images/bates-jump-diffusion/bates_iv_smile.png)

![看涨期权价格曲线：深度虚值看跌保护（低 K）被跳显著抬价](/images/bates-jump-diffusion/bates_price_compare.png)

## 六、真实陷阱（别把 Bates 当万能钥匙）

**1. 校准是个病态逆问题。** 七个额外参数（$\kappa,\theta,\sigma_v,\rho,\lambda,\mu_J,\sigma_J$）高度纠缠：跳和随机波动都能制造"微笑曲率"，数据常常分不清到底是跳还是 SV 在起作用。不同初始值能校准出差别很大的参数组，却给出几乎一样的期权价格——你以为拟合好了，其实参数毫无经济含义。实战必须用多次随机重启 + 参数边界约束。

**2. 跳跃风险溢价（jump risk premium）。** 上面用的是**物理测度**的跳参数做演示。期权市场价隐含的是**风险中性测度**的跳——投资者为下行风险付费，所以风险中性的 $\mu_J$ 比物理测度更负、$\lambda$ 也可能不同。直接用历史跳参数给期权定价会系统性偏差，必须区分两个测度。

**3. Greeks 在跳处不连续。** 跳扩散下 Delta、Gamma 在跳发生时瞬时跳变，不是 BS 那种平滑曲线。用 BS 的 Greeks 做 Delta 对冲，遇到跳会突然亏一笔。要做跳的完整对冲，得用期权组合去对冲跳风险（jump-hedging），远不是一阶 Delta 能解决。

**4. 模型风险：跳本身也可能变。** 2008 和 2020 的跳，幅度、频率、相关性都和历史样本外不同。任何校准都基于"未来跳的统计≈过去"，而真正的尾部事件恰恰不在样本里。Bates 把尾部建模得更准，但**更准不等于能预测下一次黑天鹅**。

## 七、实践：什么时候该上跳模型

不是所有期权都得用 Bates。经验法则：

- **指数/大盘期权、恐慌指数相关产品**：跳的效应最强，BS 偏差最大，必须上跳模型或至少 stochastic volatility；
- **个股单票**：跳更多是个股特异事件（财报、停牌），且流动性差，跳模型校准噪声大，常常不如直接用市场隐含的 IV 曲面；
- **短期限权**：跳在极短期相对不重要（时间不够跳几次），BS 近似尚可；中长期限跳的累积效应才显著。

落地顺序建议：先 BS 看偏差 → 加 SV（Heston）修曲率 → 再上 Bates 修尾部，每一步都用 FFT 价格和市场报价对齐，别一上来就七个参数一起校准。

## 八、小结

- Merton 跳扩散 = GBM + 复合泊松对数正态跳，把左右尾巴都加肥；
- Bates = Merton + Heston 随机波动，跳负责尾部与左偏、SV 负责曲率与聚集；
- 用特征函数 + FFT 可解析、快速、光滑地给整条行权价网格定价，**但一定要用蒙特卡洛对拍验证**；
- 跳把隐含波动率的对称微笑扭成左偏 smirk，这正是深度虚值看跌"溢价"的根源；
- 落地时最该警惕的是**校准病态**与**跳的风险中性/物理测度混淆**。

> 附：本文全部图表与定价数值均由可运行的 Python 代码生成（特征函数 FFT 定价 + 蒙特卡洛校验），参数与结果见正文，可直接复现。
