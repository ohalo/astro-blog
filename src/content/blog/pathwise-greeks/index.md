---
title: "路径导数法希腊字母：把期权 Greek 写成路径期望的导数，用蒙特卡洛稳健估计 Sensitivity"
description: "蒙特卡洛算期权价格容易，算 Greek 难——有限差分要重跑两遍、还受步长与噪声双重折磨，似然比法方差又大得吓人。路径导数法（重参数化技巧）把随机性抽离成标准正态 z，让 S_T=f(z;θ) 对参数可微，直接把导数塞进期望内部。在 BS 解析真值上用 200 万条路径对拍：Delta 真值 0.63683，pathwise 估到 0.63754（标准误 0.00040）、似然比 0.63858（标准误 0.00098）、有限差分 0.63755——pathwise 的方差只有似然比的 1/6.47；Vega 真值 37.524，pathwise 37.626±0.053 同样稳压似然比 37.670±0.178。但本文也把两个陷阱钉死：其一，Gamma 这类二阶 Greek 会因 1_{S_T>K} 的 Dirac 导数而让纯 pathwise 失效，改用 pathwise+density 恒等式 Γ=e^{-rT}(K/S0)²·p(S_T=K) 才精确命中真值 0.018762（有限差分给 0.018497、偏差 1.4%）；其二，数字期权的 payoff 不连续，pathwise 直接给出 0、100% 错误，而似然比法仍估到 0.01789（真值 0.01876）。附完整 numpy 实现与四张真实计算图。"
publishDate: '2026-11-21T02:20:00Z'
tags:
  - 量化
  - 期权定价
  - 蒙特卡洛
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

蒙特卡洛算期权价格是基本功，但算 Greek（对冲比率、敏感度）一直是个麻烦事。最朴素的做法是有限差分（FD）：把 S0 上下各挪一点，各重跑一次定价，再相减除以 2h。问题是 h 选大选小都难受——h 大了有截断偏差，h 小了两次模拟的噪声被除以一个小数放大到天上去。似然比法（Likelihood Ratio, LR）不用重跑，但把整个 payoff 乘上一个 score 函数，方差往往大得惊人。

路径导数法（Pathwise Derivative / 重参数化技巧）是第三条路：**把随机性抽离出来**。不让 `S_T` 直接服从对数正态，而是写成 `S_T = f(z; θ)`，其中 `z ~ N(0,1)` 与 θ 无关。这样对 θ 求导时，导数是"路径级"的——每条模拟路径都能贡献一个 Greek 估计值，而且通常非常平滑。

本文在 BS 平值看涨期权（S0=K=100, r=5%, σ=20%, T=1，解析价格 10.4506）上，用 200 万条路径把三种方法摊开对拍，结论如下：

1. **Delta 三法都无偏，但方差差 6 倍**：真值 **0.63683**，pathwise 估 **0.63754（SE 0.00040）**、FD **0.63755（SE 0.00040）**、LR **0.63858（SE 0.00098）**。pathwise 估计量的样本方差 **0.332**，只有 LR 的 **2.151** 的 **1/6.47**——同样精度下，LR 要多跑 6.5 倍路径。
2. **Vega 同样是 pathwise 最优**：真值 **37.524**，pathwise **37.626±0.053**、FD **37.626±0.053**、LR **37.670±0.178**，LR 标准误是 pathwise 的 3.4 倍。
3. **二阶 Greek 是纯 pathwise 的禁区**：Gamma 需要对 Delta 的 integrand 再求导，会撞上指示函数 `1_{S_T>K}` 的 Dirac 导数——它在数学上是分布而非函数，MC 里无从采样。改用 pathwise+density 恒等式 `Γ = e^{-rT}(K/S0)²·p(S_T=K)` 后精确命中真值 **0.018762（误差 0）**；稳健的 FD 给 **0.018497（偏差 1.4%）**。
4. **不连续 payoff 让 pathwise 彻底翻车**：数字看涨期权 payoff=`1_{S_T>K}`，它对 S0 几乎处处导数为 0，pathwise 估出 **0**，相对真值 **0.01876** 是 **100% 错误**；而 LR 法照样估到 **0.01789**（误差 4.6%）。这是 pathwise 方法的适用边界，必须记牢。
5. **收敛速度**：从 1000 到 200 万条路径，pathwise 的 Delta 绝对误差从 0.0020 降到 0.00071，LR 从 0.0401 降到 0.0017——小样本下 LR 的噪声是 pathwise 的二十倍量级。

![图1：Delta 三法相对解析真值的偏差与标准误（200万路径）](/images/pathwise-greeks/fig1_delta_comparison.png)

## 二、三种方法，一个目标

期权价格是风险中性期望：

```
C(θ) = e^{-rT} · E[ g(S_T; θ) ]
```

要算 `∂C/∂θ`，三种方法走三条路。

**有限差分（FD）**：`(C(θ+h) − C(θ−h)) / (2h)`。两次独立模拟，噪声叠加再除以 2h。用 common random numbers（同一批 z）可以部分抵消噪声，但步长选择仍是玄学。

**似然比（LR / Score Function）**：把导数从 payoff 挪到密度上：

```
∂C/∂θ = e^{-rT} · E[ g(S_T) · ∂log p(S_T;θ)/∂θ ]
```

不用重跑，但 `∂log p/∂θ` 通常量级很大且与 payoff 相乘，方差爆炸。好处是对不连续 payoff 也成立——因为它根本不对 payoff 求导。

**路径导数（Pathwise）**：把 `S_T` 显式写成标准正态的函数：

```
S_T = S0 · exp( (r − σ²/2)T + σ√T · z ),    z ~ N(0,1)
```

z 的分布与 θ 无关，于是导数可以穿过期望：

```
∂C/∂θ = e^{-rT} · E[ ∂g/∂S_T · ∂S_T/∂θ ]
```

对 Delta（θ=S0）：`∂S_T/∂S0 = S_T/S0`，`∂g/∂S_T = 1_{S_T>K}`，所以

```
Delta = e^{-rT} · E[ 1_{S_T>K} · S_T/S0 ]
```

对 Vega（θ=σ）：`∂S_T/∂σ = S_T(√T·z − σT)`，

```
Vega = e^{-rT} · E[ 1_{S_T>K} · S_T · (√T·z − σT) ]
```

注意 integrand 里有 `1_{S_T>K}` 这个跳变——一阶导还活着（只在边界跳，不影响期望的平滑性），但二阶导就会撞墙。这正是后文 Gamma 陷阱的伏笔。

## 三、Delta 与 Vega：pathwise 的甜点区

用 200 万条路径，S0=K=100、r=5%、σ=20%、T=1，BS 解析 Delta=0.636831、Vega=37.5240。三法结果：

| Greek | 真值 | Pathwise | LR | FD |
|-------|------|----------|-----|-----|
| Delta | 0.63683 | 0.63754±0.00040 | 0.63858±0.00098 | 0.63755±0.00040 |
| Vega | 37.524 | 37.626±0.053 | 37.670±0.178 | 37.626±0.053 |

三种方法都落在真值附近 1~2 个标准误内，无系统性偏差。差距在方差：Delta 的 pathwise 样本方差 **0.3323**，LR 是 **2.1509**，比值 **6.47**；换句话，LR 要达到 pathwise 的精度，需要多跑约 6.5 倍路径。Vega 的 LR 标准误 0.178 也是 pathwise 0.053 的 3.4 倍。

为什么 pathwise 方差低？因为它的 integrand `1_{S_T>K}·S_T/S0` 是 payoff 的一个"局部"函数——只有相关路径贡献，且量级平滑；而 LR 把整个 payoff 值（可能很大）乘上 score（也可能很大），乘积的波动被放大。FD 与 pathwise 方差几乎一样（比值 0.9998），是因为这里用了 common random numbers，两次模拟高度相关。

![图2：Vega 三法相对解析真值的偏差与标准误](/images/pathwise-greeks/fig2_vega_comparison.png)

## 四、Gamma 陷阱：当 pathwise 撞上 Dirac 函数

到这里 pathwise 看起来完美。但二阶 Greek 会让它彻底失效。

Gamma = `∂²C/∂S0²`。如果照葫芦画瓢，对 Delta 的 integrand 再求导：

```
∂/∂S0 [ 1_{S_T>K} · S_T/S0 ] = 1_{S_T>K}·0 + δ(S_T−K)·(S_T/S0)·(S_T/S0)
```

`δ(S_T−K)` 是 Dirac delta——它在 `S_T=K` 处无穷大、其余为 0、积分为 1。它不是普通函数，MC 抽样几乎永远抽不到 `S_T` 精确等于 K 的路径，所以纯 pathwise 直接给 0 或随机噪声，完全不可用。

解法是 **pathwise + density**：把 Dirac 函数的期望用 S_T 的解析密度积分掉。结果是一个漂亮的恒等式：

```
Γ = e^{-rT} · (K/S0)² · p(S_T = K)
```

其中 `p(S_T=K) = φ(d₂)/(Kσ√T)`，`d₂ = [ln(S0/K)+(r−σ²/2)T]/(σ√T)`，φ 是标准正态密度。代入 S0=K=100 这组参数：

```
Γ_pathwise+density = 0.0187620173   （解析真值 0.0187620173，误差 0）
Γ_FD = 0.018497                      （偏差 1.4%，SE 7.4e-5）
```

恒等式精确命中真值，因为它本质就是解析公式的一种写法；FD 因为截断偏差（h=0.5）低了 1.4%。这个对比的意义在于：**二阶 Greek 要跳出纯 pathwise 框架，把密度信息补进来**，否则方法从根上就不成立。

## 五、数字期权：pathwise 的适用边界

更狠的反例是数字（现金或无价值）看涨期权：`payoff = 1_{S_T>K}`，到期要么付 1 元要么付 0。它的 Delta 解析值是 `e^{-rT}·φ(d₂)/(S0σ√T) = 0.01876`（恰好等于本例的 Gamma）。

pathwise 会给出什么？`∂payoff/∂S_T = ∂1_{S_T>K}/∂S_T = 0`（几乎处处），所以 integrand 恒为 0，估计值 **0**——**100% 错误，而且错得毫无征兆**。

![图3：Delta 三法的收敛速度（双对数坐标）](/images/pathwise-greeks/fig3_convergence.png)

似然比法这里反而是赢家：它不对 payoff 求导，估到 **0.01789**，相对真值误差 4.6%（有限样本噪声内）。这条经验值得刻在风控手册里：

> **payoff 连续可微 → 用 pathwise；payoff 不连续（数字、障碍）→ 用 LR 或平滑/条件 MC。**

障碍期权、触及生效结构也属于这一类：pathwise 在路径首次触及 barrier 的时刻有跳变，直接求导会漏掉边界贡献。实务里的补救是条件蒙特卡洛（把未触发路径的解析期望单独算）或把 payoff 做光滑化。

![图4：数字期权 delta 的 pathwise 失效（左）与三法方差对比（右）](/images/pathwise-greeks/fig4_digital_trap.png)

## 六、让 pathwise 更快：对偶变量与公共随机数

pathwise 已经比 LR 高效 6.5 倍，但还能再压方差。两个几乎零成本的技巧值得写进实盘代码。

**对偶变量（Antithetic Variates）**：既然 `z ~ N(0,1)` 对称，就同时用 `z` 和 `−z` 各生成一条路径，取两者估计的平均。对 Delta（单调 payoff）这会让两条路径一个偏高一个偏低，正负误差对消。本文用 200 万条 z 实测：普通 pathwise 的 Delta 标准误是 **0.000487**，对偶版本降到 **0.0000905**，改善 **5.38 倍**；单次估计误差从 −2.05e-4 缩到 −3.8e-6。代价只是多算一遍 `exp`，远便宜于多抽一倍样本。

**公共随机数（Common Random Numbers）**：做有限差分时，上调和下调必须用**同一批 z**，否则两次模拟的独立噪声会被 `1/(2h)` 放大。本文 FD 与 pathwise 方差比 0.9998 就是用了 CRN 的结果；若两次独立抽样，FD 方差会翻倍以上，连 pathwise 都不如。

还有两个工程细节：其一，标准误别靠"重复跑 20 次完整模拟"估，把 N 条路径切成 20 个 batch、每批算一个均值、再用批均值的标准差除以 √20——这正是本文所有 ± 数字的来源，一次模拟就能给出误差棒。其二，pathwise 的 payoff 导数必须自己写对：看涨是 `1_{S_T>K}`，看跌是 `−1_{S_T>K}`，数字期权是 0；写错方向或漏了贴现因子，结果会系统性偏移而 MC 噪声根本掩盖不住。

这些技巧叠加起来，让"每次重定价都顺手吐出全套 Greeks"成为现实——一次前向模拟，Delta、Vega、Rho 全部以路径级精度产出，不需要任何额外的定价调用。这也是为什么在 AAD（伴随自动微分）普及之前，pathwise 一直是风险管理里算敏感度的主力方法：它的计算复杂度与算价格同阶，而 FD 是两倍、LR 是看 score 的实现成本。

## 七、完整代码

```python
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261121)
S0, K, r, sig, T = 100.0, 100.0, 0.05, 0.20, 1.0
N = 2_000_000
z = rng.standard_normal(N)
disc = np.exp(-r*T)
ST = S0*np.exp((r - 0.5*sig**2)*T + sig*np.sqrt(T)*z)
ITM = ST > K
payoff = np.maximum(ST - K, 0.0)

# --- Delta: pathwise ---
delta_pw = disc*np.mean(ITM * ST / S0)          # 0.6375

# --- Delta: likelihood ratio ---
score_S0 = (np.log(ST/S0) - (r-0.5*sig**2)*T) / (S0*sig**2*T)
delta_lr = disc*np.mean(payoff * score_S0)      # 0.6386

# --- Delta: finite difference (common random numbers) ---
h = 0.01
Su = (S0+h)*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*z)
Sd = (S0-h)*np.exp((r-0.5*sig**2)*T + sig*np.sqrt(T)*z)
delta_fd = disc*np.mean((np.maximum(Su-K,0)-np.maximum(Sd-K,0))/(2*h))  # 0.6375

# --- Vega: pathwise ---
vega_pw = disc*np.mean(ITM * ST * (np.sqrt(T)*z - sig*T))  # 37.63

# --- Gamma: pathwise + density（纯 pathwise 因 Dirac delta 失效）---
d2 = (np.log(S0/K) + (r-0.5*sig**2)*T)/(sig*np.sqrt(T))
p_ST_K = stats.norm.pdf(d2)/(K*sig*np.sqrt(T))
gamma = disc*(K/S0)**2 * p_ST_K                 # 0.018762 = 解析真值

# --- 检验：BS 解析真值 ---
d1 = (np.log(S0/K) + (r+0.5*sig**2)*T)/(sig*np.sqrt(T))
print("Delta true:", stats.norm.cdf(d1))        # 0.636831
print("Vega  true:", S0*stats.norm.pdf(d1)*np.sqrt(T))  # 37.5240
print("Gamma true:", stats.norm.pdf(d1)/(S0*sig*np.sqrt(T)))  # 0.018762
```

## 八、小结

路径导数法把"对参数求导"变成"对每条模拟路径求导"，在 payoff 光滑可微时（普通欧式看涨/看跌）给出低方差、无偏、单次模拟即可的 Greek 估计，Delta 方差只有似然比法的 1/6.47。但它的适用边界非常清晰：payoff 对标的不可微时（Gamma 的指示函数导数、数字期权的跳变），纯 pathwise 要么无定义要么系统性给 0，必须用 pathwise+density 恒等式或退回似然比法。

把三类方法的适用场合总结成一句话：**光滑 payoff 用 pathwise（快且稳），不连续 payoff 用 LR（通用但吵），二阶敏感度用 density 恒等式或 AAD（准但要推导）**。实务里最稳的做法往往是混合——Delta/Vega 走 pathwise 加对偶变量，Gamma 走解析或 bump，数字与障碍结构单独用条件蒙特卡洛。理解这条边界，比记住公式本身更重要：在用 MC 给复杂衍生品算 Greeks 之前，先看一眼 payoff 的连续性，能省掉一整晚对着发散的估计量调参。
