---
title: "Vasicek 利率模型：用 Ornstein-Uhlenbeck 过程给整条收益率曲线定价"
description: "1977 年 Vasicek 用一篇论文把「利率是怎么动的」写成了一个极简的随机微分方程：瞬时短利率服从 Ornstein-Uhlenbeck 过程 dr = a(b−r)dt + σdW。它均值回复、正态、可穿零，并且——最妙的是——债券价格和整条收益率曲线都有解析闭式解。本文从 OU 模拟路径讲起，用 Vasicek 闭式演示：①短端利率如何均值回复到长期中枢 b；②收益率曲线形状如何随当前短利率 r 相对中枢切换(低→上凸、高→反转)；③期限溢价(10Y−2Y)如何由回复速度 a 与波动 σ 决定；④央行收紧(抬升当前短端 r)时曲线如何整体抬升、短端首当其冲。附完整 Python 与六类真实陷阱(中阶)。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - Vasicek模型
  - 利率期限结构
  - Ornstein-Uhlenbeck
  - 收益率曲线
  - 债券定价
  - 利率衍生品
  - Python
language: Chinese
difficulty: intermediate
---

如果资产定价里有一篇「用最简单的方式说清一件大事」的论文，1977 年 Vasicek 的 *"An Equilibrium Characterization of the Term Structure"* 一定排得上号。它干了一件事：把「利率是怎么随机变动的」写成一个微分方程，然后**神奇地给出了债券价格和整条收益率曲线的解析解**。

这之前，人们要么用经验曲线拟合，要么假设利率恒定。Vasicek 第一次说：利率是随机的、有均值回复的、而且我们可以**精确算出来**任意期限债券该值多少钱。

## 一、核心设定：利率是个「被弹簧拉回中枢」的随机过程

瞬时短期利率 $r_t$ 服从 **Ornstein-Uhlenbeck(OU)过程**：

$$dr_t = a(b - r_t)\,dt + \sigma\,dW_t$$

三个参数各管一件事：

- $b$：**长期中枢**。利率不管怎么飘，最终被拉回这个水平(比如 6%)。
- $a$：**均值回复速度**。越大，拉回得越快。
- $\sigma$：**波动**。随机冲击的强度。

数学上这是「带阻尼的随机游走」:没有 $\sigma$ 时它是指数衰减回 $b$;有了 $\sigma$ 它就在 $b$ 附近高斯抖动。两个重要性质:

1. **正态分布**:任意时刻 $r_t$ 都服从正态分布,所以**可以变负**——这在 1977 年是个激进的设定(当时没人相信负利率),但 2010 年代欧日负利率让它变得异常现实。
2. **均值回复**:不像几何布朗运动(股价)会一路漂移,利率被 $a(b-r_t)$ 这股「弹簧力」拽着,不会无限上涨也不会归零。

我们先把 OU 过程模拟出来:

```python
import numpy as np
rng = np.random.default_rng(20260717)
a, b, sigma, r0 = 0.20, 0.06, 0.02, 0.03
dt = 1/252.0; steps = 252 * 8
paths = np.zeros((6, steps+1)); paths[:, 0] = r0
for i in range(6):
    r = r0
    for s in range(1, steps+1):
        r += a*(b - r)*dt + sigma*np.sqrt(dt)*rng.normal(0, 1)
        paths[i, s] = r
# 稳态: 均值 b=6%, 方差 sigma^2/(2a)
ou_std = sigma / np.sqrt(2*a)   # ≈ 3.16%
```

![Vasicek 短期利率：Ornstein-Uhlenbeck 模拟路径(均值回复至 b, 高斯可穿零)](/images/vasicek-rate-model/vasicek_ou_paths.png)

跑出来的 6 条路径清清楚楚:不管起点在哪,都**向红色虚线 b=6% 收敛**,在中枢上下高斯抖动;模拟区间里最低到过 **−1.5%**(穿零)、最高到过 **11.9%**。这正是 OU 过程「正态可负、均值回复」的写真。

## 二、最妙的部分:Vasicek 的解析债券定价公式

如果只是模拟路径,这篇论文不至于封神。真正厉害的是——**给定当前短利率 $r_t$,任意期限 $T$ 的债券价格有闭式解**。

零息债价格:

$$P(t,T) = A(t,T)\,e^{-B(t,T)\,r_t}$$

其中

$$B(t,T) = \frac{1 - e^{-a\tau}}{a}, \qquad \tau = T - t$$

$$A(t,T) = \exp\!\left[\frac{(B - \tau)(a^2 b - \sigma^2/2)}{a^2} - \frac{\sigma^2 B^2}{4a}\right]$$

即期收益率 $y(t,T) = -\frac{\ln P(t,T)}{\tau}$。注意一个关键结构:**收益率是 $r_t$ 的线性函数**($y = \text{常数项} - \frac{B(\tau)}{\tau}\,r_t$)。这意味着:

- 曲线的**水平**由 $a,b,\sigma$ 决定;
- 曲线的**斜率/形状**由当前 $r_t$ 相对中枢 $b$ 的位置决定。

```python
def vasicek_bond_price(r0, t, T, a, b, sigma):
    tau = T - t
    if tau <= 0: return 1.0
    B = (1 - np.exp(-a*tau)) / a
    A = np.exp((B - tau)*(a**2*b - sigma**2/2)/a**2 - sigma**2*B**2/(4*a))
    return A * np.exp(-B*r0)

def vasicek_spot(r0, t, T, a, b, sigma):
    tau = T - t
    if tau <= 0: return r0
    return -np.log(vasicek_bond_price(r0, t, T, a, b, sigma)) / tau
```

## 三、曲线形状随 $r_t$ 切换:低→上凸,高→反转

用同一个 $a,b,\sigma$,只改当前短利率 $r_t$,看收益率曲线怎么变:

```python
mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
y_low  = [vasicek_spot(0.02, 0, T, a, b, sigma) for T in mats]   # r_t 低
y_mid  = [vasicek_spot(b,    0, T, a, b, sigma) for T in mats]   # r_t = 中枢
y_high = [vasicek_spot(0.10, 0, T, a, b, sigma) for T in mats]   # r_t 高
```

![Vasicek 即期收益率曲线:随 r_t 相对中枢切换形状](/images/vasicek-rate-model/vasicek_curve_shapes.png)

三种情形对应真实市场里你会看到的三种曲线:

- **$r_t=2\%$ 低于中枢 6%**:曲线**上凸/陡峭化**——短端被压低,长端仍锚定在中枢附近,典型的「衰退/宽松预期」形状(2 年期 ≈2.7%,30 年期 ≈5.0%)。
- **$r_t=6\%=$ 中枢**:曲线接近**水平**(中心情形)。
- **$r_t=10\%$ 高于中枢**:曲线**反转/平坦化**——短端高高在上,长端知道利率终会回落,反而压得低(2 年期 ≈9.3%,30 年期 ≈6.3%),典型的「过热/政策紧缩」形状。

**一条公式,三种曲线形态,全部连续切换。**这就是 Vasicek 用「当前短利率 + 中枢」建模的威力:曲线不再是外生拟合,而是内生于状态变量 $r_t$。

## 四、期限溢价(10Y−2Y):谁决定的?

曲线交易里最常看的指标是期限溢价——10 年减 2 年收益率。在 Vasicek 里它也是闭式的,而且只取决于 $a$ 和 $\sigma$:

```python
tp = vasicek_spot(r0, 0, 10, a, b, sigma) - vasicek_spot(r0, 0, 2, a, b, sigma)   # 基准 ≈ 100.5 bp
```

我们扫一遍 $a \in [0.05, 0.6]$、$\sigma \in [0.5\%, 5\%]$ 的网格:

![Vasicek 期限溢价(10Y-2Y, bp):随 a 与 sigma 的变化](/images/vasicek-rate-model/vasicek_term_premium.png)

几个直觉:

- **$\sigma$ 越大,期限溢价越**正**(越高)**:波动越大,长端相对短端要求的补偿越多(风险溢价上升)。这与「波动率越高、期限溢价越厚」的实证一致。
- **$a$ 越大(回复越快),期限溢价越薄甚至转负**:利率很快回到中枢,长期不确定性被压缩,长端不再比短端贵多少。
- 网格里期限溢价从 **−226 bp 到 +132 bp** 大幅摆动,说明这个「曲线的斜率」完全是模型参数的产物,没有「天然该为正」的说法。

## 五、央行收紧:抬升当前短端,曲线整体抬升、短端首当其冲

把 Vasicek 用到政策分析上。假设央行收紧——体现为**当前短利率 $r_t$ 被政策利率抬升**(从 2% 到 6%):

```python
r_loose, r_tight = 0.02, 0.06
y_loose = [vasicek_spot(r_loose, 0, T, a, b, sigma) for T in mats]
y_tight = [vasicek_spot(r_tight, 0, T, a, b, sigma) for T in mats]

def rate_sens_to_r0(a, b, sigma, T):   # 解析 d(yield)/d(r0) = B(tau)/tau
    B = (1 - np.exp(-a*T)) / a
    return B / T
sens_short = rate_sens_to_r0(a, b, sigma, 0.25)   # 0.975
sens_long  = rate_sens_to_r0(a, b, sigma, 10)      # 0.432
```

![Vasicek 曲线对政策中枢的响应:收紧 r0 后整体抬升、短端首当其冲](/images/vasicek-rate-model/vasicek_policy_shock.png)

收紧后曲线**整体上移**,但**短端动得远比长端多**:短端(3 月)对 $r_0$ 的敏感度 **0.975**(几乎 1:1 跟随),长端(10 年)只有 **0.432**(只跟一半)。这正是现实里「加息时收益率曲线平坦化」的数学解释——短端被政策直接钉死,长端还看着远期中枢,于是曲线被压扁。

## 六、六类真实陷阱(实战必看)

1. **利率可负是特性也是缺陷**:Vasicek 的正态假设让利率能穿零,在 2010s 欧日负利率下反而比很多模型「更现实」;但它也意味着**模型会给出任意大的负利率概率**,极端情形下定价失真。这是后续 CIR、Hull-White 要修的点。
2. **$\sigma$ 恒定不现实**:真实波动是时变的(危机时利率波动暴涨)。Vasicek 的常数 $\sigma$ 在极端行情下系统性低估风险——要上 GARCH 或随机波动率。
3. **参数要校准不是拍脑袋**:$a,b,\sigma$ 得从真实国债曲线用最小二乘/极大似然校准,不能直接用本文演示值。校准出的 $a$ 通常很小(0.05~0.3),意味着实际回复非常慢。
4. **长期中枢 $b$ 也会漂移**:40 年降息周期里 $b$ 本身在变。固定 $b$ 的 Vasicek 在利率制度切换时会大幅错配——Hull-White(让 $b$ 变随机)正是为此而生。
5. **曲线拟合 ≠ 套利免费**:Vasicek 能完美定价初始那一条曲线(所谓「拟合初始期限结构」),但对冲比率 $\partial P/\partial r$、以及利率路径的风险,仍需 Monte Carlo 或有限差分数值验证,闭式只在 OU 假设下成立。
6. ** Vasicek 下没有真正的「凸性套利」**:它给的是风险中性定价下的均衡价,做市/相对价值交易时还要叠加信用利差、流动性溢价、交易成本和监管约束,纯模型价差不等于真能赚的钱。

## 七、小结:Vasicek 是利率建模的「Hello World」

Vasicek 不是最精确的利率模型(它没有非负约束、波动恒定、中枢固定),但它是**一切现代利率模型的起点和基准**:

- 想解决「利率可负」→ **CIR 模型**(平方根扩散,非负)。
- 想解决「中枢会漂移」→ **Hull-White 模型**(让 $b$ 随机,仍能解析)。
- 想给利率期权定价 → 在 Vasicek 之上加 **债券期权闭式**($A,B$ 直接进 Black 公式)。

对你做固收/宏观策略的意义:**先吃透 Vasicek 这一条 OU 过程 + 它的 $A,B$ 闭式**,你就拿到了理解整条收益率曲线、期限溢价、以及「加息时曲线怎么动」的钥匙。后面那些更复杂的模型,都是在它的骨架上补丁。

---

*代码与图表均由自包含 Python(numpy)真实计算,随机种子固定为 20260717,可完整复现。所有统计数字(OU 稳态、曲线形态、期限溢价、敏感度)来自文中脚本输出。模型参数 $a,b,\sigma$ 为演示设定,实盘需从真实国债曲线校准。*
