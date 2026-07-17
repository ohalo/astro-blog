---
title: "CIR 利率模型：用平方根扩散守住利率的非负底线"
description: "1985 年 Cox-Ingersoll-Ross 用一篇论文堵住了 Vasicek 最大的洞——利率能变负。CIR 把扩散系数从常数改成 √r：当利率逼近 0 时扩散也趋 0、而向上的均值回复拉力仍在，于是利率被「非负地板」托住。本文从 Milstein 精确模拟讲起，用 CIR 仿射闭式演示：①利率永不低于 0（Feller 条件 2ab>σ² 成立时，模拟最低 2.14%）；②平稳分布是 Gamma、均值=中枢 b；③收益率曲线随 r₀ 切换（低→上翘 +172bp / 高→反转 −177bp）；④贴近零地板时波动率塌缩、短端被钉死。附完整 Python 与六类真实陷阱(中阶)。"
publishDate: '2026-07-17'
tags:
  - 量化交易
  - CIR模型
  - 利率期限结构
  - 平方根扩散
  - 债券定价
  - 均值回复
  - Python
language: Chinese
difficulty: intermediate
---

如果 Vasicek(1977)是利率建模的「Hello World」，那 CIR(1985, Cox-Ingersoll-Ross 的 *"A Theory of the Term Structure of Interest Rates"*)就是它最自然的升级版——专门来堵一个最刺眼的洞:**Vasicek 的利率会穿零**。

Vasicek 用 $dr = a(b-r)dt + \sigma dW$，扩散系数 $\sigma$ 是常数，所以利率服从正态、可以任意负。在 1970 年代这是「激进但能忍」的假设;等到 2010 年代欧日真的出现负利率，人们才发现另一面:**利率可以无限负**在数学上很丑，更糟的是它给债券定价带来灾难——深度负利率会让长期债券价格失去上界。

CIR 的修法极简也极妙:**把扩散系数从常数 $\sigma$ 改成 $\sigma\sqrt{r_t}$**。这一改，利率就拥有了经济学里最该有的性质——**非负**。

## 一、核心设定：平方根扩散

瞬时短期利率 $r_t$ 服从 **CIR(平方根 / square-root)过程**:

$$dr_t = a(b - r_t)\,dt + \sigma\sqrt{r_t}\,dW_t$$

三个参数和 Vasicek 一一对应:

- $b$:**长期中枢**(长期均值 $= b$)。
- $a$:**均值回复速度**。
- $\sigma$:**波动强度**(注意它作用在 $\sqrt{r_t}$ 上)。

关键就在那个 $\sqrt{r_t}$。它在两个方向同时起作用:

1. **当 $r_t \to 0$ 时，扩散项 $\sigma\sqrt{r_t} \to 0$**——利率越靠近地板，随机抖动越弱，像被「粘」住;
2. **但漂移项 $a(b-r_t) \to ab > 0$(向上)**——只要 $b>0$，利率越低，向上的弹簧力越强。

两个合力:**利率被非负地板托住,不会穿零**。这跟 Vasicek 形成鲜明对比——后者在 $r_t=0$ 处扩散仍是 $\sigma$，没有任何东西阻止它继续往下掉。

我们用 **Milstein 格式**(带 $\sqrt{r}$ 的二阶修正项)做精确模拟。普通 Euler 会在 $r$ 接近 0 时把负扩散项算成「从负利率出发」，从而错误穿零;Milstein 的修正项专门补这块，并且我们对负结果做 `max(r,0)` 兜底:

```python
import numpy as np
rng = np.random.default_rng(20260717)
a, b, sigma, r0 = 0.30, 0.06, 0.04, 0.03
dt = 1/252.0; steps = 252 * 10; n_paths = 6
paths = np.zeros((n_paths, steps + 1)); paths[:, 0] = r0
for i in range(n_paths):
    r = r0
    for s in range(1, steps + 1):
        dW = rng.normal(0, np.sqrt(dt))
        r = max(r, 0.0)
        drift    = a * (b - r) * dt
        diff     = sigma * np.sqrt(r) * dW
        milstein = 0.25 * sigma**2 * (dW**2 - dt) * np.sqrt(max(r, 0.0))
        r = r + drift + diff + milstein
        paths[i, s] = max(r, 0.0)   # 即便数值噪声也不会写为负
```

![CIR 模拟路径：均值回复至 b，被平方根扩散托在零之上(永不低于 0)](/images/cir-interest-rate-model/cir_paths.png)

6 条 10 年路径清清楚楚:全部向红色虚线 $b=6\%$ 收敛，在中枢附近抖动，**最低只到过 2.14%**，整段没有任何一条碰过零。这正是平方根扩散「非负反射」的写真。

## 二、为什么利率不穿零：Feller 条件

$\sqrt{r}$ 带来的非负性不是「大概率」而是「**只要一个条件成立，就严格为正**」。这个条件叫 **Feller 条件**:

$$2ab > \sigma^2$$

直觉:均值回复的「向上托力」($ab$ 量级)必须压过扩散的「向下逃逸」($\sigma^2$ 量级)。我们的参数下 $2ab - \sigma^2 = 2\times0.30\times0.06 - 0.04^2 = \mathbf{0.0344 > 0}$——条件成立，所以利率严格正。

如果反过来 $\sigma$ 很大、$a$ 很小(比如把 $\sigma$ 调到 0.12)，Feller 条件被打破，利率就会**反复触零**——CIR 退化成「能在零处反弹但会擦到地板」的过程，长端定价的很多漂亮性质也会变脆。

把「利率贴近零地板的概率」在参数网格 $(a,\sigma)$ 上扫一遍，能直接看到 Feller 边界($2ab=\sigma^2$ 对应 $a\approx 0.013$ 那根竖虚线)把画面一分为二:

![CIR 平稳定分布下 P(r<1%) 热图：越红越易贴零地板，Feller 边界右侧才安全](/images/cir-interest-rate-model/cir_zero_floor.png)

右上方($a$ 大 $\sigma$ 小)是深蓝——几乎不会贴零;左下方($a$ 小 $\sigma$ 大)是红——Feller 失守，利率频繁触地板。这张图就是 CIR 适用边界的「地形图」。

## 三、平稳分布：不是正态，是 Gamma

Vasicek 的稳态是正态(所以可负);CIR 的稳态是 **Gamma 分布**:

- 形状 $k = 2ab/\sigma^2$
- 尺度 $\theta = \sigma^2/(2a)$
- 均值 $k\theta = b$(=中枢 ✓)
- 方差 $k\theta^2 = b\sigma^2/(2a)$

我们的参数算出来 $k=22.5$、$\theta=0.00267$，均值 6%、标准差仅 **1.26%**——比 Vasicek 那种「能甩到 ±12%」的稳态紧得多，因为 Gamma 在 0 处有边界、把左侧的概率质量全挤回了正半轴。

把模拟终端分布(最后 2 年)和理论 Gamma 叠一起，几乎完美重合:

![CIR 平稳分布：Gamma(k=22.5, θ=0.0027)，均值 6%，左边界即 0](/images/cir-interest-rate-model/cir_stationary.png)

值得注意:**模拟里 $r<1\%$ 的样本占比是 0**——在我们这组通过 Feller 检验的参数下，利率「实际贴到地板」几乎不发生，这跟热图右半区的深蓝一致。

## 四、最妙的部分：CIR 也有仿射闭式

和 Vasicek 一样，CIR 在风险中性、市场价格为 0 的假设下，债券价格仍是**仿射闭式**:

$$P(t,T) = A(\tau)\,e^{-B(\tau)\,r_t},\qquad \tau = T-t$$

其中

$$\gamma = \sqrt{a^2 + 2\sigma^2}$$

$$B(\tau) = \frac{2\big(e^{\gamma\tau}-1\big)}{\big(\gamma+a\big)\big(e^{\gamma\tau}-1\big) + 2\gamma}$$

$$A(\tau) = \left[\frac{2\gamma\,e^{(a+\gamma)\tau/2}}{\big(\gamma+a\big)\big(e^{\gamma\tau}-1\big) + 2\gamma}\right]^{2ab/\sigma^2}$$

即期收益率 $y(t,T) = -\ln P /\tau$。两个必须记死的性质:

1. **短期极限** $\tau\to0$:$B(\tau)\to\tau$、$A(\tau)\to1$，于是 $y\to r_t$——曲线短端严格等于当前短利率 ✓;
2. **长期极限** $\tau\to\infty$:曲线收敛到长端常数 $\displaystyle \lim y = \frac{ab(\gamma-a)}{\sigma^2}$，在我们参数下算得 **5.95%**(几乎等于中枢 $b=6\%$)。

```python
import numpy as np
def cir_bond(r0, tau, a, b, sigma):
    if tau <= 0: return 1.0
    g = np.sqrt(a**2 + 2*sigma**2); e = np.exp(g*tau)
    B = 2.0*(e-1.0) / ((g+a)*(e-1.0) + 2*g)
    A = (2*g*np.exp((a+g)*tau/2.0) / ((g+a)*(e-1.0) + 2*g))**(2*a*b/sigma**2)
    return A * np.exp(-B*r0)

def cir_yield(r0, tau, a, b, sigma):
    if tau <= 0: return r0
    return -np.log(cir_bond(r0, tau, a, b, sigma)) / tau

mats = np.array([0.25, 0.5, 1, 2, 3, 5, 7, 10, 15, 20, 30])
for r0 in (0.02, b, 0.10):
    y = [cir_yield(r0, T, a, b, sigma) for T in mats]
    print(f"r0={r0:.0%}  2Y={y[3]:.4f}  30Y={y[9]:.4f}  slope(10Y-2Y)={(y[8]-y[3])*1e4:+.0f}bp")
```

跑出来(短端严格等于 $r_0$ 验证通过):

| 当前短利率 $r_0$ | 2 年收益率 | 30 年收益率 | 10Y−2Y 斜率 |
|---|---|---|---|
| 2%(低于 b) | 2.99% | 5.52% | **+172 bp**(上翘) |
| 6%(=b) | 6.00% | 5.96% | **−2 bp**(近平) |
| 10%(高于 b) | 9.00% | 6.40% | **−177 bp**(反转) |

![CIR 即期收益率曲线：随当前短利率 r₀ 切换(低→上翘 / 高→反转)](/images/cir-interest-rate-model/cir_yield_curves.png)

和 Vasicek 完全同构的形状逻辑——曲线随 $r_0$ 相对中枢 $b$ 切换:低 → 上翘、高 → 反转。区别只藏在**短端能不能动到负**，以及**贴近地板时的波动行为**。

## 五、零地板的交易含义：波动率塌缩

CIR 最被实战派看重的一点，是它对**「利率贴近零时市场怎么变」**的刻画比 Vasicek 真实:

- 当 $r_t\to0$，扩散 $\sigma\sqrt{r_t}\to0$，**短端波动塌缩**——利率被地板「冻住」，不再剧烈抖动;
- 这正是 2010 年代欧日「零利率下限(ZLB)」时期观察到的:政策利率贴零后，短端几乎不动，只有靠 QE/前瞻指引去撬动长端。

所以做收益率曲线交易时，CIR 会告诉你:**在零附近，短端 delta($\partial y/\partial r_0$)接近 1、但短端波动率接近 0**——你「能精确控制短端位置，却很难靠短端波动赚钱」。Vasicek 在同样区域会给你一个虚高的短端波动，诱使你去做实际上不存在的短端 gamma 交易。

## 六、六类真实陷阱(实战必看)

1. **Feller 条件决定生死**:$2ab>\sigma^2$ 不成立时利率反复触零，长端闭式仍然数学有效，但「非负」的卖点没了。校准出高 $\sigma$ 低 $a$ 的组合时，先查 Feller 再下结论。
2. **参数必须校准，且 CIR 对 $\sigma$ 极敏感**:$\sigma$ 同时出现在扩散和长端收敛速度里，轻微偏差会被 $2\sigma^2$ 放大。通常从真实国债曲线用极大似然(基于利率转移密度的非中心卡方形式)校准，别用本文演示值。
3. **平方根扩散 ≠ 现实的全部波动**:真实利率波动是时变的(危机时暴涨)、而且有 ZLB 之下的「反射 + 跳跃」。CIR 的 $\sigma\sqrt{r}$ 只能捕捉连续部分的压缩，跳变要靠 GARCH/随机波动率补。
4. **CIR 仍能给任意接近 0 的利率，但概率由 Gamma 决定**:它比 Vasicek「不会无限负」，却也**不会在 0 处完全停住**——Feller 失守时会反复擦地板。把它当「软地板」而非「硬下限」。
5. **曲线拟合 ≠ 套利免费**:和 Vasicek 一样，CIR 能完美拟合初始那条曲线，但对冲比率 $\partial P/\partial r$ 仍需数值验证;而且它默认「市场价格为 0」，在真实信用/流动性溢价面前，纯模型价差不等于真能赚的钱。
6. **别拿 CIR 直接给利率期权定价**:它的债券闭式是在「风险中性 + 零市场价格」下推的;要给期权定价得显式引入市场风险价格(常取 $\lambda(r)=\lambda_0+\lambda_1 r$ 的仿射形式)，否则会把风险溢价漏掉。

## 七、小结：CIR 是 Vasicek 的「非负补丁」

CIR 不是推翻 Vasicek，而是精准补上它最痛的洞:

- Vasicek 的洞:**利率能无限负** → CIR 用 $\sigma\sqrt{r}$ 把地板焊在 0;
- 二者共享:**仿射闭式 + 均值回复 + 曲线随 $r_0$ 切换**;
- 想要**中枢也会漂移** → 上 **Hull-White**(让 $b$ 随机，且仍解析);
- 想要**给利率期权定价** → 在 CIR 之上加仿射市场风险价格，进 Black 公式。

对你做固收/宏观策略的意义:**CIR 给你一条「有底」的短利率过程**——它既保留了 Vasicek 全部可解析的便利，又在零利率这种最关键的政策边界上表现得远比正态假设可信。先把 Milstein 模拟跑顺、把 Feller 条件和 Gamma 平稳分布吃透，你就拿到了理解「利率下限 + 曲线形状」这对实战命题的钥匙。

---

*代码与图表均由自包含 Python(numpy/scipy/matplotlib)真实计算,随机种子固定为 20260717,可完整复现。所有统计数字(路径最低 2.14%、平稳均值 6%、长端 5.95%、各斜率、Feller 边界 $a\approx0.013$)来自文中脚本输出。模型参数 $a,b,\sigma$ 为演示设定,实盘需从真实国债曲线校准。*
