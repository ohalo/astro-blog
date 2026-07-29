---
title: "Vanna-Volga 定价：用三个希腊字母修正 BS 的微笑误差"
description: "「整条波动率微笑，只需要三个报价就能重建。」Vanna-Volga 是外汇期权柜台三十年的工作马：不引入任何随机波动率模型，直接用市场上流动性最好的三个工具（25Δ Put / ATM / 25Δ Call）的对冲成本修正 Black-Scholes 价格。逻辑是纯交易员式的——BS 假设波动率恒定，于是漏算了 vega（重挂）、vanna（∂Δ/∂σ）、volga（∂vega/∂σ）三个敞口的对冲费用；用三个锚点期权恰好搭出一个匹配这三个敞口的复制组合，组合的市场价与 BS 价之差就是微笑修正。Python 完整实现：解 3×3 线性方程组得对冲权重、修正价反推隐波，重建微笑在锚点覆盖区（±1.2σ）与参考微笑几乎重合；三个希腊字母各定微笑的一个几何自由度——vega 定水平、vanna 定偏斜（ATM 过零的奇函数）、volga 定曲率（两翼大 ATM 小），这正是它只需要三个报价的深层原因，与 SABR 的 α/ρ/ν 一一同构。误差分析给出诚实边界：锚点之间插值精确，两翼外推随期限拉长迅速发散，深度 OTM 与超长期限须换模型；它是插值器不是模型——没有动态、没有一致的联合分布，定价路径依赖产品会出错（高阶）。"
publishDate: '2026-07-29'
tags:
  - 量化交易
  - 期权定价
  - Vanna-Volga
  - 波动率微笑
  - 外汇期权
  - Python
language: Chinese
difficulty: advanced
---

## 一句话版本

Black-Scholes 假设波动率不动，于是漏算了三笔对冲费用——vega、vanna、volga 的再对冲成本；Vanna-Volga 方法用市场上流动性最好的三个期权把这三笔费用「买」出来加回去，三个报价就能重建整条微笑。

---

## 一、问题：BS 定价漏掉了什么

Black-Scholes 世界里波动率是常数 σ，对冲一个期权只需要 delta 对冲——因为价格的唯一风险源是标的 S。

真实世界里隐含波动率每天在动，而且和标的价格一起动。一个用 BS 框架做对冲的交易员实际暴露在三个额外风险上：

| 希腊字母 | 定义 | 直白含义 |
|---|---|---|
| **Vega** | ∂V/∂σ | 波动率水平变了，期权价值变多少 |
| **Vanna** | ∂²V/∂S∂σ | 标的动了，我的 vega 变多少（等价地：波动率动了，我的 delta 变多少） |
| **Volga** | ∂²V/∂σ²（vol-gamma） | 波动率动了，我的 vega 自己变多少——对波动率的凸性 |

BS 价格里没有这三项风险的补偿，因为模型假设 σ 不动、它们永远不会兑现。但市场知道 σ 会动——**所以市场价格 ≠ BS 价格，差额恰恰是这三个敞口的「保险费」**。这个差额的行权价形态，就是波动率微笑。

Vanna-Volga（VV）方法的洞察是交易员式的：与其建一个随机波动率模型去解释微笑，不如**直接去市场上把这三个敞口的对冲成本问出来**。

## 二、三个锚点：市场其实只报三个数

外汇期权市场的报价惯例天然配合这个思路。做市商屏幕上流动性最好的是三个组合：

- **ATM straddle** → 给出 ATM 隐波 σ_ATM（微笑的水平）
- **25Δ Risk Reversal**（买 25Δ call 卖 25Δ put）→ RR = σ_25C − σ_25P（微笑的偏斜）
- **25Δ Butterfly**（买两翼卖 ATM）→ BF = (σ_25C + σ_25P)/2 − σ_ATM（微笑的曲率）

三个报价换算出三个锚点的隐波：

$$\sigma_{25C} = \sigma_{ATM} + BF + \tfrac{RR}{2}, \qquad \sigma_{25P} = \sigma_{ATM} + BF - \tfrac{RR}{2}$$

**为什么恰好三个就够？** 因为微笑作为一条曲线，局部只有三个几何自由度：水平、斜率、曲率。而 vega、vanna、volga 的行权价形态恰好各控制一个：

![三个高阶希腊字母的行权价形态](/images/vanna-volga-pricing/vv-greeks-profiles.png)

- **Vega** 在 ATM 处最大、两翼对称衰减——它是「水平旋钮」；
- **Vanna** 在 ATM 处过零、两侧异号（奇函数形态）——它是「偏斜旋钮」，risk reversal 正是纯 vanna 工具；
- **Volga** 在 ATM 处最小、两翼大——它是「曲率旋钮」，butterfly 正是纯 volga 工具。

这组对应不是巧合：SABR 模型的 α（水平）、ρ（偏斜）、ν（曲率）与之一一同构。**VV 和 SABR 是同一件事的两种表述——前者用对冲成本说话，后者用随机过程说话。**

## 三、方法：一个复制论证

给任意行权价 K 的期权定价，VV 的步骤是一个干净的复制论证：

**第 1 步**：全部先用同一个「平坦」波动率 σ_ATM 计算 BS 价格和希腊字母。

**第 2 步**：用三个锚点期权（数量 x₁, x₂, x₃）搭一个组合，使它的 vega、vanna、volga 与目标期权完全相同——这是一个 3×3 线性方程组：

$$\begin{pmatrix} \mathcal{V}_1 & \mathcal{V}_2 & \mathcal{V}_3 \\ Va_1 & Va_2 & Va_3 \\ Vo_1 & Vo_2 & Vo_3 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix} = \begin{pmatrix} \mathcal{V}_K \\ Va_K \\ Vo_K \end{pmatrix}$$

**第 3 步**：这个组合在市场上的真实成本，比它的 BS 理论价贵（或便宜）多少？每个锚点的「市场价 − BS 价」是可观察的（用锚点各自的市场隐波 vs 平坦隐波各算一次 BS）。组合的总溢价：

$$\text{Adj} = \sum_{i=1}^{3} x_i \left[ V^{mkt}(K_i) - V^{BS}(K_i; \sigma_{ATM}) \right]$$

**第 4 步**：目标期权的 VV 价格 = BS 平坦价 + Adj。逻辑：既然组合对冲掉了目标期权全部的波动率相关敞口，那么目标期权相对 BS 的溢价就应该等于组合的溢价——否则存在（近似的）套利。

把 VV 价格反推回隐含波动率，对所有 K 重复，整条微笑就出来了。

## 四、Python 实现：60 行核心代码

```python
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq

def bs_price(S, K, T, r, sigma, cp=1):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return cp*(S*norm.cdf(cp*d1) - K*np.exp(-r*T)*norm.cdf(cp*d2))

def bs_vega(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    return S*norm.pdf(d1)*np.sqrt(T)

def bs_vanna(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return -norm.pdf(d1)*d2/sigma           # ∂²V/∂S∂σ

def bs_volga(S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    return bs_vega(S, K, T, r, sigma)*d1*d2/sigma   # ∂²V/∂σ²

# ── 市场输入：三个报价 ───────────────────────────────
S0, r, T = 100.0, 0.02, 0.5
sig_atm, rr25, bf25 = 0.20, -0.025, 0.008   # 股票型负偏斜
sig_25c = sig_atm + bf25 + rr25/2            # 19.55%
sig_25p = sig_atm + bf25 - rr25/2            # 22.05%
anchors_K   = np.array([92.03, 102.02, 111.94])   # 25ΔP / ATM / 25ΔC
anchors_sig = np.array([sig_25p, sig_atm, sig_25c])

# 锚点的希腊字母矩阵（全部用平坦 σ_ATM 计算）
G = np.array([[bs_vega(S0, Ki, T, r, sig_atm),
               bs_vanna(S0, Ki, T, r, sig_atm),
               bs_volga(S0, Ki, T, r, sig_atm)] for Ki in anchors_K])
# 锚点的市场溢价：市场隐波价 − 平坦价
cost = np.array([bs_price(S0, anchors_K[i], T, r, anchors_sig[i])
               - bs_price(S0, anchors_K[i], T, r, sig_atm) for i in range(3)])

def vv_implied_vol(K):
    g = np.array([bs_vega(S0, K, T, r, sig_atm),
                  bs_vanna(S0, K, T, r, sig_atm),
                  bs_volga(S0, K, T, r, sig_atm)])
    x = np.linalg.solve(G.T, g)              # 复制权重
    price = bs_price(S0, K, T, r, sig_atm) + x @ cost
    return brentq(lambda s: bs_price(S0, K, T, r, s) - price, 0.01, 1.5)
```

重建结果：

![Vanna-Volga：用三个报价重建整条微笑](/images/vanna-volga-pricing/vv-smile-reconstruction.png)

蓝色 VV 曲线精确穿过三个红色锚点（构造保证），且在锚点之间与灰色参考微笑几乎重合。**输入只有三个数：20%、-2.5%、0.8%**——这就是 VV 在外汇柜台成为标准工具的原因：报价屏上有什么，模型就吃什么，中间不需要任何校准迭代。

## 五、权重的交易员解读：每个报价单都是一张配方

VV 不只是插值公式——解出来的权重 x 有直接的对冲操作含义：

![任意行权价期权 = 三个锚点期权的组合](/images/vanna-volga-pricing/vv-hedge-weights.png)

- **深度 OTM put（K=85）**：权重 ≈ 1.41 份 25ΔP − 1.00 份 ATM + 0.43 份 25ΔC。直觉：它的 vanna/volga 形态最像左翼锚点，所以配方以 25ΔP 为主，做空 ATM 来中和多余的 vega。
- **OTM call（K=115）**：配方镜像地以 25ΔC 为主（1.27 份）。

卖出一张奇异行权价的期权后，交易员照着这张配方买入锚点组合，vega/vanna/volga 三个敞口即刻归零，剩下的只有 delta（一阶对冲掉）和高阶残余。**VV 价格「公允」的含义非常具体：它恰好等于你按市场价搭这个对冲组合的成本。** 定价与对冲在这个框架里是同一件事——这是它比「纯数学插值」（如 SVI 参数化）更受柜台信任的原因。

## 六、适用边界：插值器的荣耀与局限

对不同期限做误差分析（VV 隐波 − 参考微笑，横轴为标准化 moneyness）：

![Vanna-Volga 的适用边界](/images/vanna-volga-pricing/vv-error-analysis.png)

三条结论：

1. **锚点覆盖区（±1.2σ 左右，绿色区域）误差可忽略**——这里覆盖了绝大部分实际成交的行权价，也是 VV 三十年屹立不倒的地盘。
2. **两翼外推发散，且期限越长越糟**。T=2 年的深度 OTM 区域误差达到几十上百 bp。原因在结构上：三个基函数（vega/vanna/volga 形态）在远离锚点处的形状是「猜」的，没有任何市场信息约束它——三点定不出尾部。
3. **短期限（T=0.08）覆盖区窄但覆盖区内极准**——这符合它在外汇市场的主战场：1 周到 1 年的香草期权与一触即付（one-touch）等一代奇异期权。

更根本的局限要说透：**VV 是一个静态快照插值器，不是一个模型**。

- 它没有定义标的和波动率的联合动态——没有过程，就没有一致的微笑演化预测。今天用 VV 报价，明天微笑动了，重新插值即可；但**路径依赖产品（亚式、障碍中的复杂款、cliquet）的价值取决于微笑如何随时间演化**，VV 对此原则上沉默。柜台实务中对一触即付等简单奇异用「生存概率加权 VV 修正」的经验公式，那是打补丁，不是理论。
- 三个锚点意味着三个自由度：如果真实微笑在 25Δ 之外还有独立弯折（比如崩盘保护需求造成的深度 OTM put 局部凸起），VV 结构性无法表达。
- 无套利不保证：VV 微笑在极端参数下可能隐含负的风险中性密度（蝶式套利）。生产系统需要事后检查。

## 七、与 SABR 的分工：什么时候用哪个

| 维度 | Vanna-Volga | SABR |
|---|---|---|
| 输入 | 3 个市场报价，零校准 | 校准 3-4 个参数（最小二乘） |
| 哲学 | 对冲成本复制，无动态假设 | 随机过程，有微笑动态 |
| 微笑动态预测 | 无 | sticky-delta（微笑随远期平移） |
| 强项 | FX 香草与一代奇异、报价速度、交易员可解释 | 利率期权、微笑动态、delta 对冲一致性 |
| 弱项 | 尾部外推、路径依赖产品、无套利不保证 | 超长期限失真、负密度（需 shifted 版本） |

经验分工：**报价与插值用 VV，对冲比率与动态用 SABR / 随机波动率**。两者在锚点覆盖区给出的价格差异通常在 1-2 个 vol bp 之内——分歧出现的地方（深度 OTM、长期限、奇异结构），恰恰是「你需要一个真正的模型」的信号。

## 八、总结

| 命题 | 结论 |
|---|---|
| VV 在做什么 | 把 BS 漏掉的 vega/vanna/volga 对冲成本用三个市场工具「买」出来加回 BS 价 |
| 为什么三个报价就够 | 微笑局部只有水平/偏斜/曲率三个自由度，三个希腊字母恰好各管一个 |
| 核心计算 | 一个 3×3 线性方程组 + 一次隐波反解，无迭代校准，微秒级 |
| 权重的含义 | 直接是对冲配方：卖出目标期权后按权重买锚点，波动率敞口归零 |
| 适用区 | 锚点覆盖区（±1.2σ）内插值精确；FX 短中期香草与简单奇异 |
| 失效区 | 两翼外推（尾部无信息）、长期限、路径依赖产品（无动态）、极端参数下可能违反无套利 |

Vanna-Volga 是金融工程里「工程」二字的典范：不追求解释世界的随机过程，只回答一个交易员真正要付钱的问题——**把这个期权的波动率风险全部对冲掉，市场现在收我多少钱？** 三个报价，一个线性方程组，答案就在那里。
