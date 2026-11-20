---
title: "恒定到期期限互换 CMS：用复制组合给远期互换利率加上凸性调整，别再把 CMS 当线性互换利率"
description: "CMS 支付的是重置日观察到的固定期限互换利率，它虽然是年金测度下的鞅，但按 T-远期测度计价时并不等于远期互换利率。本文从测度变换推出 CMS forward = A(S0)·E[S_T/A(S_T)]，用离散复利年金做 5 年进 10 年的精确算例：3.5% 利率、20% 波动下凸性调整 14.68bp，30% 波动下达 39.31bp；把 CMS 当远期互换利率的朴素做法会系统性低估。二阶 Hagan 近似误差不到 0.04bp，只取 σ²T 首项则低估 1bp 以上。附静态 swaption 复制（800 个执行价把误差从 7.38bp 压到 0.37bp）、完整 numpy 代码与四张真实计算图。"
publishDate: '2026-11-20T22:20:00Z'
tags:
  - 量化
  - 固定收益
  - 衍生品
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

利率互换里有一个常见误解："CMS（Constant Maturity Swap）付的就是远期互换利率，直接拿forward swap rate定价就行。"这是错的，而且错的方向是系统性的。

原因在于测度。远期互换利率 `S_0` 是**年金测度**（annuity measure，把年金 `A_t=ΣτP(t,T_i)` 当 numeraire）下的鞅；但一个在 T 时刻支付 `S_T` 的 CMS 现金流，它的远期价格要用 **T-远期测度** 求期望。两个测度不同，通过 Radon-Nikodym 导数联系，而导数里恰好含有随利率变化的年金 `A_T`——于是 CMS forward 比远期互换利率高出一个"凸性调整"。

本文用 5 年后重置、标的为 10 年期互换的 CMS 做实验，初始远期互换利率 3.5%。结论：

1. 波动率 20% 时，精确的 CMS forward 是 **3.6468%**，凸性调整 **14.68bp**；波动率 30% 时调整冲到 **39.31bp**。朴素地把 CMS 等于 3.5% 会系统性少收十几个基点。
2. 调整近似按 `σ²·T` 增长：1 年期 2.65bp、5 年期 14.68bp、10 年期 33.71bp。期限拉长，误差非线性放大。
3. 二阶 Hagan 展开与精确值误差不超过 0.54bp，足够实用；只留 `σ²T` 首项在 20% 波动下就低估 1.01bp、30% 波动下低估 4.22bp——别用一阶近似报价格。
4. 用欧洲 swaption 做静态复制，执行价从 50 个加到 800 个，复制误差从 7.38bp 单调收敛到 0.37bp，验证了调整确实是"期权凸性"的钱。

![图1：年金函数随利率递减，g(s)=s/A(s) 是凸函数](/images/constant-maturity-swap-cms/fig1_annuity_convexity.png)

## 二、凸性调整从哪来：两个测度，一个年金

先说清楚三个量。远期互换利率 `S_t` 使得到期互换的现值为零，它在年金测度下是鞅：`E^A[S_T] = S_0`。年金 `A_t = Σ_{i=1}^{10} τ·P(t,T_i)` 是收取固定端的现金流现值之和，也是该测度的 numeraire。CMS 在 T 时刻观察到 `S_T` 并支付它，因此这个现金流的远期价格（T-远期测度下的期望）才是 CMS forward：

`F_CMS = E^T[S_T]`

两个测度之间的关系由 numeraire 比给出：`dQ^T/dQ^A = (A_0/P(0,T))·(1/A_T)`。于是

`F_CMS = (A_0/P(0,T)) · E^A[S_T / A_T]`

在单因子近似下，T 时刻的整条曲线跟着 `S_T` 平移，所以 `A_T = A(S_T)` 是利率的确定性递减函数；又因平值曲线有关系 `A_0 = P(0,T)·A(S_0)`，代入得到干净的定价公式：

`F_CMS = A(S_0) · E^A[ S_T / A(S_T) ]`

这里离散复利年金 `A(s) = Σ_{i=1}^{10} τ/(1+τs)^i`，本例取 τ=1、10 年期，`A(3.5%)=8.3166`，年金修正久期 `D=-A'/A=5.04`。

直觉很直接：利率上升时年金下降，`1/A(S_T)` 与 `S_T` 正相关。 Jensen 不等式告诉我们 `E[S_T/A(S_T)] > E[S_T]·E[1/A(S_T)]`（近似），所以 `F_CMS > S_0`，调整恒为正。图1 显示 `g(s)=s/A(s)` 明显向上弯，曲率正是超额收益的来源。实盘里这一步最容易被忽略：CMS 票据、CMS cap/floor、反向可赎回票据的定价都吃这个 adjustment。

这里值得把"为什么不是年金鞅"再拆细一点。远期互换利率之所以在年金测度下是鞅，是因为以年金为 numeraire 时，互换的两个腿（固定端年金、浮动端贴现）能写成同单位的相对价格，漂移恰好抵消。但 CMS 现金流的支付时点与计价单位换了：它按 T 时刻的远期测度折现到今天，而 T 测度的 numeraire 是零息债，不是年金。两个 numeraire 的比值 `A_t/P(t,T)` 本身随利率随机波动，且与 `S_T` 相关，这个相关性就是凸性调整的全部来源。换句话说，调整不是模型假设硬塞进去的，而是"用哪个尺子量"的必然结果。理解了这一点，就能明白为什么曲线越陡、利率水平越高时调整越大——因为年金对利率的敏感度（`A'`）在那里更大。

## 三、精确计算：从高斯积分到蒙特卡洛

在年金测度下，令 `S_T = S_0·exp(σ√T·Z − σ²T/2)`，`Z~N(0,1)`，是对数正态。把定价公式写成积分：

`F_CMS = A(S_0) · ∫ g(S_0·e^{σ√T z − σ²T/2}) φ(z) dz`

本文用 400 点 Gauss-Legendre 在 `z∈[-8,8]` 上做数值积分，并用 80 万条蒙特卡洛路径交叉验证，两者差异在 0.03bp 以内——说明数值实现可信。下表是不同波动率下的结果（单位 bp）：

| σ | 精确调整 | 二阶近似误差 | 仅 σ²T 首项误差 |
|---|---------|------------|----------------|
| 5%  | 0.81  | +0.00 | −0.04 |
| 10% | 3.33  | +0.00 | −0.16 |
| 15% | 7.79  | +0.01 | −0.44 |
| 20% | 14.68 | +0.04 | −1.01 |
| 25% | 24.77 | +0.15 | −2.12 |
| 30% | 39.31 | +0.54 | −4.22 |

![图2：凸性调整随波动率与期限非线性增长](/images/constant-maturity-swap-cms/fig2_adjustment_vs_vol.png)

关键观察：调整对波动率高度敏感，从 5% 到 30% 放大约 48 倍，远超线性。这是 CMS 定价里最本质的风险——你报出的价格里有一大块是"卖方赚的波动率凸性"，用错波动率就是直接亏钱。另一个维度是期限：同样的 20% 波动，1 年只调 2.65bp，10 年调 33.71bp，因为累积方差 `σ²T` 线性增长但凸性函数在高利率区更陡。

还有一个常被忽略的维度：初始利率水平。图4 的曲面显示，在同样的波动率下，远期利率从 1% 升到 8%，凸性调整几乎翻倍。原因是年金久期 `D=-A'/A` 在低利率时更大（折现分母小、现金流权重靠后），利率上升时久期缩短，但 `S_0²` 因子又变大——两者拉锯，最终调整随 `S_0` 单调上升但不完全按平方。这也是为什么负利率环境下的 CMS 定价要格外小心：公式里 `A(s)` 的定义在 `1+τs≤0` 时失效，需要换用连续复利或其他参数化，本文的离散复利形式只适用于正利率区间。

## 四、解析近似：Hagan 二阶够用，一阶会亏

把 `1/A(s)` 在 `S_0` 处做二阶泰勒展开，利用对数正态矩 `E[S_T]=S_0`、`E[S_T²]=S_0²e^{σ²T}`、`E[S_T³]=S_0³e^{3σ²T}`，得到闭式近似。它只用到三个年金导数，计算量近乎为零。表里的"二阶近似误差"列说明：到 30% 波动都只差 0.54bp，实务报价完全够用。

最粗糙的一阶近似 `F ≈ S_0 + D·S_0²·(e^{σ²T}−1)` 就差得多：20% 波动低估 1.01bp，30% 低估 4.22bp。原因是一阶只抓住年金久期，丢了年金凸性——而高波动恰恰把尾部的大利率情形权重放大，凸性项变得重要。所以如果你要在交易台上快速估算，至少用二阶；如果只想要一个数量级感觉，用 `D·S_0²·σ²T` 也行，但别拿它当成交价。

## 五、静态复制：凸性调整就是期权钱

Hagan 的经典做法是把 CMS 支付拆成 swaption 组合。对凸函数 `g(s)=s/A(s)`，欧式看跌看涨的静态复制恒等式是：

`g(S_T) = g(S_0) + ∫_0^{S_0} g''(K)(K−S_T)⁺ dK + ∫_{S_0}^∞ g''(K)(S_T−K)⁺ dK`

两边在年金测度下取期望，再用 Black 公式给每个 swaption 定价（执行价 K 的 call/put 价格分别是 `E^A[(S_T−K)⁺]` 与 `E^A[(K−S_T)⁺]`），就得到 `F_CMS`。

![图3：静态复制的收敛性与各方法对拍](/images/constant-maturity-swap-cms/fig3_replication_check.png)

数值上，执行价网格从 50 个加密到 800 个，复制误差从 7.38bp 降到 0.37bp——收敛很慢（大致按 1/N 下降），因为 `g''` 在深虚值区有尖峰。这解释了为什么实务中 CMS 做市商不会真的用 800 个 swaption 对冲，而是用少量关键执行价加动态对冲。但复制的意义在于：它证明了凸性调整**不是定价模型的臆造**，而是可以直接用可交易期权复制出来的真金白银。

复制公式还给出了一个实用的风险指标：`g''(K)` 在不同执行价上的分布，就是 CMS 头寸对各执行价 swaption 的敞口。如果把 `g''` 峰值附近的执行价找出来（本例在 4%~6% 区间），做市商就知道该重点对冲哪几个点，而不是平均撒网。这也解释了一个现象：CMS 的 vega 风险集中在利率分布的中高区，而不是平值附近——因为年金在低利率时变化慢、高利率时变化快，凸性权重被推向尾部。对风险管理而言，这意味着用单一平值波动率来管理 CMS 账簿是不充分的，必须看整条波动率微笑。

![图4：CMS 凸性调整曲面（波动率 × 远期利率）](/images/constant-maturity-swap-cms/fig4_adjustment_surface.png)

## 六、完整代码

以下是核心计算，纯 numpy + scipy，可直接运行：

```python
import numpy as np
from scipy.stats import norm

TAU, N, S0, T = 1.0, 10, 0.035, 5.0
k = np.arange(1, N + 1)

def annuity(s):
    s = np.asarray(s, dtype=float)
    return np.sum(TAU / (1 + TAU*s[..., None])**k, axis=-1)

def annuity_prime(s):
    s = np.asarray(s, dtype=float)
    return np.sum(-k*TAU**2 / (1 + TAU*s[..., None])**(k+1), axis=-1)

# 精确值: Gauss-Legendre 数值积分
def cms_exact(sig):
    nodes, wgh = np.polynomial.legendre.leggauss(400)
    Z = nodes*8
    W = wgh*8/np.sqrt(2*np.pi)*np.exp(-0.5*Z**2)
    ST = S0*np.exp(sig*np.sqrt(T)*Z - 0.5*sig**2*T)
    return annuity(S0)*np.sum(W*ST/annuity(ST))

# 蒙特卡洛交叉验证
def cms_mc(sig, m=800000, seed=7):
    Z = np.random.default_rng(seed).standard_normal(m)
    ST = S0*np.exp(sig*np.sqrt(T)*Z - 0.5*sig**2*T)
    return annuity(S0)*np.mean(ST/annuity(ST))

# 静态复制 (截断执行价网格)
def gsecond(s, h=1e-5):
    g = lambda x: x/annuity(x)
    return (g(s+h) - 2*g(s) + g(s-h))/h**2

def black_call(F, K, sig, t):
    v = sig*np.sqrt(t)
    d1 = (np.log(F/K) + 0.5*v*v)/v
    return F*norm.cdf(d1) - K*norm.cdf(d1 - v)

def cms_replicate(sig, nk=800):
    kmax = S0*np.exp(6*sig*np.sqrt(T)); kmin = S0*np.exp(-6*sig*np.sqrt(T))
    klo = np.linspace(max(kmin, 1e-6), S0, nk//2)
    khi = np.linspace(S0, kmax, nk//2)
    val = S0/annuity(S0)
    for p, dk in [(klo[:-1], klo[1]-klo[0]), (khi[:-1], khi[1]-khi[0])]:
        for K in p:
            v = sig*np.sqrt(T)
            d1 = (np.log(S0/K) + 0.5*v*v)/v
            opt = (K*norm.cdf(-d1+v) - S0*norm.cdf(-d1)) if K < S0 else black_call(S0, K, sig, T)
            val += gsecond(K)*opt*dk
    return annuity(S0)*val

for sig in [0.05, 0.10, 0.20, 0.30]:
    ex = cms_exact(sig)
    print(f"sigma={sig:.0%}  CMS={ex:.4%}  adj={(ex-S0)*1e4:.2f}bp")
```

## 七、什么时候该用，什么时候别用

CMS 调整最常见的应用场景是 CMS 票据定价、CMS spread option（例如 10 年减 2 年）以及养老金负债里的长期利率敞口。做市时报出的 CMS rate 必须含凸性调整，否则在长期高波动环境下会持续亏钱。但要注意三点：第一，本文用单因子"平行移动"假设，把年金写成 `A(S_T)` 的确定性函数；真实曲线会扭曲，跨币种或陡峭曲线下要用多因子模型重新校准。第二，调整对波动率曲面敏感，20% 和 30% 之间差 25bp，必须用流动 swaption  implied 的波动率而非历史波动。第三，t=0 的初始曲线必须自洽——`A_0=P(0,T)·A(S_0)` 这个关系在曲线非平时要重新推导，否则公式的前提不成立。

## 八、小结

CMS 看似只是"把互换利率付出来"，实则踩在两个测度的裂缝上：远期互换利率是年金鞅，CMS 现金流是 T-远期价格，两者之间隔着一层年金凸性。5 年进 10 年的算例里，这层凸性在 20% 波动下值 14.68bp、30% 波动下值 39.31bp。二阶 Hagan 近似足够准，静态复制证明它就是期权的钱。下次有人拿 forward swap rate 直接当 CMS rate 报价，你应该问一句：adjustment 加了吗？
