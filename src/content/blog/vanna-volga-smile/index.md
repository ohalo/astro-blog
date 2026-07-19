---
title: "Vanna-Volga 校准：用蝴蝶/魏姬修正给微笑加半解析校正"
description: "Black-Scholes 给所有行权价同一个常数波动率，于是隐含波动率该是平的；但真实市场弯成微笑，外汇期权尤其典型。Vanna-Volga（VV 法，又称半解析微笑插值）不拟合整个曲面，只抓三个支柱期权（25-delta put / ATM / 25-delta call），用蝴蝶（凸度）和 Vega（波动率敏感度）把微笑「加」回 BS 价格。本文用自洽合成数据从零跑通三支柱 Lagrange 校正，证明它在支柱处精确还原市场 IV（残差=0），并诚实点出支柱外无套利失控、远端外推爆炸等六类真实陷阱（中阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 期权定价
  - 波动率微笑
  - Vanna-Volga
  - 外汇期权
  - 半解析插值
  - Python
language: Chinese
difficulty: intermediate
---

Black-Scholes（1973）给期权定价时，所有行权价 `K`、所有到期 `T` 都用**同一个常数波动率** `σ`。于是在 BS 世界里，把不同 `K` 的期权价格倒推出来的隐含波动率（IV）应该是一条**水平线**。可真实市场的 IV 是弯的——平值附近低、两端虚值高，叫「波动率微笑」；再加上左尾偏斜（smirk），就成了外汇期权里那张人人都见的歪嘴曲线。

问题是：定价一张不在标准网格上的期权（比如某个特殊行权价的 expiry），该用哪个 IV？**Vanna-Volga 法**（Castagna & Mercurio 2007，业界 FX 期权的标准微笑插值）给出一个极聪明的半解析答案：它**不拟合整张曲面**，只抓住市场报出的**三个支柱期权**——25-delta put 翼、ATM 远期、25-delta call 翼——然后用「蝴蝶（butterfly，凸度）修正 + Vega（波动率敏感度，魏姬）权重」把微笑从 BS 价格里**加回去**。

## 一、为什么需要 VV：BS 平了一根曲线，市场弯了

先看真实市场微笑长什么样。假设远期 `F = 100`，ATM 波动率 `σ_atm = 20%`，偏度（skew）`a = −0.15`、凸度（convexity/butterfly）`b = 0.30`，则市场隐含波动率为：

```
σ_mkt(K) = σ_atm + a·ln(K/F) + b·ln²(K/F)
```

左尾（低 `K`）被偏度压高、两翼被凸度抬升，是一张典型的 smirk + smile 叠加：

![市场真实波动率微笑：左尾偏斜(smirk) + 两翼抬升](/images/vanna-volga-smile/vv_smile.png)

如果傻傻地用 `σ_atm = 20%` 去给 `K=90` 的 put 定价，你会系统性低估它的价格——因为真实 put 翼 IV 是 21.9%，比平值高了近 2 个点。VV 的核心思想就是：**用 ATM vol 做基准 BS 价，再用三个支柱的市场价和基准价的「差额」插值出任意 `K` 的校正价。**

## 二、三支柱市场输入：风险反转、蝴蝶、平值

实战里市场不直接报三个 `K` 的 IV，而是报：

- **ATM 远期波动率** `σ_atm`：平值期权的 IV；
- **25-delta 风险反转（Risk Reversal, RR）**：`σ_25d_call − σ_25d_put`，度量偏度（左尾肥瘦）；
- **25-delta 蝴蝶（Butterfly, BF）**：`(σ_25d_call + σ_25d_put)/2 − σ_atm`，度量凸度（微笑弯曲）。

由这三个量就能反解出三个支柱行权价 `K₁`（put 翼）、`K₂ = F`（ATM）、`K₃`（call 翼）对应的市场 IV `σ_mkt(K_i)`，进而算出市场价 `V_mkt(K_i) = BS(σ_mkt(K_i); K_i)`。

## 三、半解析校正公式：把微笑「加」回去

**第一步**，用 ATM vol 算基准 BS 价：

```
V_BS(K) = BS(σ_atm; K)
```

**第二步**，算三个支柱处「市场价 − 基准价」的差额 `Δ_i = V_mkt(K_i) − V_BS(K_i)`。注意 ATM 支柱 `K₂=F` 处 `σ_mkt(K₂) = σ_atm`，所以 `Δ₂ = 0`——这正是 VV 法「不碰平值、只修两翼」的优雅之处。在我们的自洽合成里：

```
put  翼 K₁=90 : 市场价 14.2237  BS(atm) 13.5891  Δ₁ = +0.6346
ATM   K₂=100: 市场价 7.9656   BS(atm) 7.9656   Δ₂ = 0.0000
call 翼 K₃=110: 市场价 3.8648   BS(atm) 4.2920   Δ₃ = −0.4272
```

**第三步**，用**对数 moneyness** `x = ln(K/F)` 的 Lagrange 基函数做插值。对任意 `K`：

```
w_i(K) = Π_{j≠i} (x − x_j) / (x_i − x_j)        # 在支点 i 处取 1、其余取 0
V(K)   = V_BS(K) + w₁(K)·Δ₁ + w₂(K)·Δ₂ + w₃(K)·Δ₃
```

这就是 VV 的「半解析」本质：BS 部分解析、校正部分用三个已知差额做基函数加权。把 `Δ_i` 写成 `Vega(K_i)·(σ_mkt(K_i) − σ_atm)` 的形式，就得到业界常用的「Vega 加权」表述——**蝴蝶（凸度）修正体现在 Δ₁、Δ₃ 的翼差里，Vega（魏姬）则是把波动率差换算成价格差的敏感度**。

## 四、自洽验证：放进去的，收得回来

把上面算出的 `V(K)` 再倒推出隐含波动率 `IV_VV(K)`，和真实的 `σ_mkt(K)` 比：

![VV 校准一致性：在三个支柱处精确还原市场 IV](/images/vanna-volga-smile/vv_recovery.png)

三支柱处 `IV_VV` 与真实 IV **完全重合**（理论残差=0）；支柱之间靠基函数平滑插值。整段扫描的**最大 IV 残差仅 0.0207**——这个残差全部来自支柱之间的插值（不是拟合误差，因为支柱处是精确的），说明 VV 在「给定三支柱、问中间价」这个任务上自洽无误。

校正项 `ΔV(K) = V(K) − V_BS(K)` 可以拆成 put 翼贡献 `w₁·Δ₁` 和 call 翼贡献 `w₃·Δ₃`（ATM `Δ₂=0`），一半正一半负，恰好把两翼的微笑「补」进 BS 常数里：

![半解析校正项：把微笑从 BS 常数里补回去](/images/vanna-volga-smile/vv_correction.png)

三支柱 Lagrange 基函数本身满足「partition of unity」——在各自支点取 1、其余取 0，保证 `K→K_i` 时校正精确地退化为 `Δ_i`：

![三支柱 Lagrange 基函数：在支点处取 1、其余取 0](/images/vanna-volga-smile/vv_basis.png)

## 五、完整 Python 实现

下面是从零跑通上述校准的精简代码（与配图脚本一致，可直接运行）：

```python
import numpy as np
from scipy.stats import norm

S, r, q, T = 100.0, 0.0, 0.0, 1.0
F = S * np.exp((r - q) * T)          # 远期 = ATM
sigma_atm, a_skew, b_conv = 0.20, -0.15, 0.30

def sigma_mkt(K):                     # 真实市场微笑 (自洽合成)
    m = np.log(K / F)
    return sigma_atm + a_skew * m + b_conv * m**2

def bs_call(S, K, T, sigma):
    d1 = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

# 三支柱
K1, K2, K3 = 0.90 * F, F, 1.10 * F
C_mkt = {Ki: bs_call(S, Ki, T, sigma_mkt(Ki)) for Ki in (K1, K2, K3)}
V_BS  = {Ki: bs_call(S, Ki, T, sigma_atm)     for Ki in (K1, K2, K3)}

# Vanna-Volga 半解析校正 (对数moneyness 基函数)
def vv_price(K):
    x1, x2, x3 = np.log(K1 / K), np.log(K2 / K), np.log(K3 / K)
    w1 = x2 * x3 / ((x2 - x1) * (x3 - x1))
    w2 = x1 * x3 / ((x1 - x2) * (x3 - x2))
    w3 = x1 * x2 / ((x1 - x3) * (x2 - x3))
    d1 = C_mkt[K1] - V_BS[K1]
    d2 = C_mkt[K2] - V_BS[K2]          # ATM 处 = 0
    d3 = C_mkt[K3] - V_BS[K3]
    return bs_call(S, K, T, sigma_atm) + w1 * d1 + w2 * d2 + w3 * d3

# 反推 IV (二分)
def implied_vol(S, K, T, price, lo=1e-4, hi=5.0):
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        (lo, hi) = (mid, hi) if bs_call(S, K, T, mid) > price else (lo, mid)
    return 0.5 * (lo + hi)

Ks = np.linspace(80, 120, 161)
iv_vv   = [implied_vol(S, K, T, vv_price(K)) for K in Ks]
iv_true = [sigma_mkt(K) for K in Ks]
print("最大 IV 残差 =", max(abs(a - b) for a, b in zip(iv_vv, iv_true)))
```

## 六、六类真实陷阱

1. **支柱外无套利失控（最致命）**：VV 只在三个支点精确还原市场价，支点**之间**是 Lagrange 线性（双线性）插值，**不保证**相邻点满足无套利单调性。真实市场报价稀疏带噪，直接插值可能造出「短期 IV 高于长期、可无风险套利」的日历/蝶式套利曲面。实务上要先对输入支柱做套利清理再校准。
2. **远端外推爆炸**：基函数 `w_i(K)` 在 `K` 远离三支柱时**无界增长**——给一个极深度虚值期权定价，VV 价格会飞到荒谬值。VV 是**插值法不是外推法**，远端必须cutoff或切到边缘 vol。
3. **ATM 支柱假设**：VV 默认「ATM 处 `σ_mkt=σ_atm`、Δ₂=0」。如果你的 ATM 支柱 vol 本身被误报或含买卖价差，这个误差会**原样传播**到整条曲线（因为基准价 `V_BS` 用它算）。
4. **Vega 用 ATM vol 的 Vega，非局部**：Vega 加权版用 `Vega(K_i)` 把 vol 差转价格差，但这个 Vega 是按 `σ_atm` 算的，不是按真实局部 vol。对大幅偏离平值的翼，这是个**近似**而非精确——偏离越远近似越糙。
5. **只是静态插值、没有动力学**：VV 给你「此刻」的微笑一致价格，但**没有 forward vol、没有期限结构演化、不能给路径依赖/美式/二阶希腊**定价。它是微笑插值工具，不是定价模型。要动态须上 SABR / 随机波动率 / 局部波动率。
6. **对输入支柱高度敏感**：RR、BF、ATM 三个数一点点测量误差，会让整条曲线明显摆动。尤其是 25-delta 翼本身流动性差、报价宽，翼差的噪声会直接放大成两翼微笑的抖动。

## 七、和局部波动率 / SABR 的对照

- **VV** 是**半解析插值**，只消费三支柱、闭式、快，适合 FX 期权实时报价与风险，但无动力学、无外推、无套利不保；
- **局部波动率（Dupire）** 吃整张 `C(K,T)` 曲面、反演 `σ_loc(S,T)`，无套利但数值重、需整张曲面；
- **SABR** 用 `α/β/ρ/ν` 四参数给整条微笑一个**可校准、可外推、带动力学**的模型，适合长端/奇异期权。

一句话总结 VV 的定位：**它不追求「正确」，只追求「在三个已知点上和市场一致、中间平滑、闭式快」——这是交易台实时风控与报价的正确取舍。** 理解它，你就拿到了波动率微笑工程化的第一把快刀。

## 结语

Vanna-Volga 的精妙在于克制：它不试图拟合整个世界，只老老实实抓住三个支柱，用蝴蝶（凸度）和 Vega（敏感度）把微笑「加」回 BS。对 90% 的 vanilla 报价与实时希腊计算，这把快刀足够锋利。但也正因克制，它留下无套利失控、远端爆炸、无动力学三道边界——越过这些边界，就该请出 SABR 或局部波动率了。
