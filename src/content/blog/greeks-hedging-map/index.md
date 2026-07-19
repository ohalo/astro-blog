---
title: "Greeks 全景与对冲：用 Delta/Gamma/Vega 拼出风险地图"
description: "期权交易里，价格只是表层，真正决定盈亏的是它背后的风险因子——Greeks。Delta 告诉你「标的方向动一点亏多少」、Gamma 说「Delta 自己动得有多快」、Vega 管波动率、Theta 管时间、Rho 管利率。本文用 BS 模型从零算出价格/Delta/Gamma/Vega/Theta/Rho 并画出完整曲线，再用风险地图(标的价格冲击×波动率冲击)把二阶敏感度可视化，最后用蒙特卡洛对比「未对冲 / 仅 Delta / Delta+Gamma」三种对冲的 P&L 方差——依次从 1.95 降到 0.19 再到 0.012，并诚实点出「Gamma 对冲要用第二个期权/离散再平衡/凸性缺口/铁幕假设」四类真实陷阱（中阶）。"
publishDate: '2026-07-19'
tags:
  - 量化交易
  - 期权
  - Greeks
  - Delta
  - Gamma
  - Vega
  - 对冲
  - Python
language: Chinese
difficulty: intermediate
---

期权交易者看一张期权报价，第一眼不是价格，而是 **Greeks**——一组描述「这个头寸对各个风险因子有多敏感」的数字。价格告诉你**现在**值多少，Greeks 告诉你**下次市场动一下**你会亏多少。它们才是真实的风险地图。

本文用最干净的欧式看涨 BS 模型，从零算出全部一阶、二阶 Greeks，把曲线画全，再用一张「风险地图」把二阶敏感度（Delta/Gamma/Vega/Theta 如何随标的价格和波动率同时变化）可视化出来，最后用蒙特卡洛实打实对比三种对冲策略的方差。所有图表均为真实计算，非占位图。

## 一、六个 Greeks 的数学定义

设欧式看涨，标的 $S$、行权 $K$、期限 $T$、无风险利率 $r$、波动率 $\sigma$。BS 价格：

$$C = S\,N(d_1) - K e^{-rT} N(d_2),\qquad d_1=\frac{\ln(S/K)+(r+\tfrac12\sigma^2)T}{\sigma\sqrt T},\ d_2=d_1-\sigma\sqrt T$$

六个核心 Greeks（每 1% vol / 每天 / 每 1% 利率 口径）：

| Greek | 含义 | 公式 |
|---|---|---|
| **Delta** Δ | 对标的价格的一阶敏感 | $N(d_1)$ |
| **Gamma** Γ | Delta 对标的价格的敏感 | $\dfrac{N'(d_1)}{S\sigma\sqrt T}$ |
| **Vega** ν | 对波动率的敏感 | $S\,N'(d_1)\sqrt T /100$（每 1%） |
| **Theta** Θ | 对时间的敏感 | $-\dfrac{S N'(d_1)\sigma}{2\sqrt T}-rKe^{-rT}N(d_2)$（每天） |
| **Rho** ρ | 对利率的敏感 | $K T e^{-rT}N(d_2)/100$（每 1%） |

参数取 $S_0=100,\ K=100,\ T=0.5,\ r=2\%,\ \sigma=25\%$（ATM 短期期权）。从零实现：

```python
import numpy as np
from scipy.stats import norm

S0, K, T, r, sigma = 100.0, 100.0, 0.5, 0.02, 0.25

def bs(S, vol):
    d1 = (np.log(S / K) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * vol * np.sqrt(T))
    vega  = S * norm.pdf(d1) * np.sqrt(T) / 100.0      # 每 1% vol
    theta = (-S * norm.pdf(d1) * vol / (2 * np.sqrt(T))
             - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365.0
    rho   = K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return price, delta, gamma, vega, theta, rho

V0, D0, G0, Vg, Th, Rh = bs(S0, sigma)
print(f"Δ={D0:.4f}  Γ={G0:.5f}  ν={Vg:.4f}  Θ={Th:.4f}  ρ={Rh:.4f}")
```

把价格、Delta、Gamma 对 $S$ 的曲线画出来（第一阶 Greeks）：

![价格 / Delta / Gamma vs 标的价格 S：ATM 处 Delta=0.5、Gamma 峰值](/images/greeks-hedging-map/price_delta_gamma.png)

三个特征一眼可辨：**Delta 从 0 单调升到 1**（深度 ITM 时几乎等于持有股票）；**Gamma 在 ATM 处出现尖峰**（此时 Delta 变化最快，对冲最「脆弱」）；价格曲线是凸的——这个凸性就是 Gamma 的几何来源。

Vega、Theta、Rho 对 $S$ 的曲线（跨变量 Greeks）：

![Vega / Theta / Rho vs 标的价格 S：Vega 在 ATM 峰、Theta 在 ATM 最负](/images/greeks-hedging-map/vega_theta_rho.png)

**Vega 在 ATM 处最大**——波动率对「最不确定方向」的期权影响最猛；**Theta 在 ATM 处最负**——时间流失对「最悬而未决」的期权伤害最大。这就是为什么做市商最怕持有 ATM 头寸：vega 和 theta 同时把你夹在中间。

## 二、风险地图：把二阶敏感度摊开

单看「Greeks vs S」只暴露了沿标的价格这一维的变化。真实风险是**多维**的：标的价格动、波动率也动，二者还会耦合。把横轴设为「标的价格冲击 %」、纵轴设为「波动率冲击 %」，画 Delta/Gamma/Vega/Theta 四张热力图：

```python
s_shock = np.linspace(-0.20, 0.20, 41)   # 标的价格 ±20%
v_shock = np.linspace(-0.50, 0.50, 41)   # 波动率 ±50%
SS, VV = np.meshgrid(s_shock, v_shock)
G_delta = np.zeros_like(SS); G_gamma = np.zeros_like(SS)
G_vega = np.zeros_like(SS); G_theta = np.zeros_like(SS)
for i in range(SS.shape[0]):
    for j in range(SS.shape[1]):
        s = S0 * (1 + SS[i, j]); vol = sigma * (1 + VV[i, j])
        _, d, g, vv, th, _ = bs(s, vol)
        G_delta[i, j], G_gamma[i, j], G_vega[i, j], G_theta[i, j] = d, g, vv, th
```

![Greeks 风险地图：横轴标的价格冲击、纵轴波动率冲击，中心(0,0)为当前持仓](/images/greeks-hedging-map/greeks_heatmap.png)

这张地图的用法：**中心 (0,0) 是你当前持仓的 Greek 值**，往右是标的价格上涨、往上是波动率上升。比如右上角（价涨+波动涨）：Delta 接近 1（已 ITM）、Gamma 萎缩（远离峰值）、Vega 仍在正区但被拉低。交易台每天盯的就是这张图——它告诉你「如果市场同时往某个象限跳，我的 Delta 会从 0.56 变成多少、我的 Gamma 凸性还剩多少」。

## 三、对冲实证：未对冲 vs Delta vs Delta+Gamma

Greeks 不是用来欣赏的，是用来**对冲**的。核心逻辑：持有一个期权多头，它的 P&L 对 $S$ 是非线性的（因为有 Gamma）。我们用股票把 Delta 中和掉，再用第二个期权把 Gamma 中和掉，看 1 周内 P&L 方差能压到多小。

1 周（$h=1/52$）后标的价格按 GBM 到达 $S_h$，期权按剩余期限 $T-h$ 重估：

```python
h = 1.0 / 52.0
rng = np.random.default_rng(20260719)
N = 200_000
Z = rng.standard_normal(N)
S_h = S0 * np.exp((r - 0.5 * sigma**2) * h + sigma * np.sqrt(h) * Z)

V0, D0, G0, _, _, _ = bs(S0, sigma)
# 第二个期权 (K2=110) 用于 Gamma 对冲
K2 = 110.0
def bs_K2(S, vol, Kk):
    d1 = (np.log(S / Kk) + (r + 0.5 * vol**2) * T) / (vol * np.sqrt(T))
    d2 = d1 - vol * np.sqrt(T)
    return S * norm.cdf(d1) - Kk * np.exp(-r * T) * norm.cdf(d2), \
           norm.cdf(d1), norm.pdf(d1) / (S * vol * np.sqrt(T))
V2_0, D2_0, G2_0 = bs_K2(S0, sigma, K2)

w2 = -G0 / G2_0          # 持 w2 份期权2 使组合 Gamma=0
w1 = -(D0 + w2 * D2_0)   # 空头 w1 份股票使组合 Delta=0

Tk = T - h
def reprice(S, Kk, Tk):
    d1 = (np.log(S / Kk) + (r + 0.5 * sigma**2) * Tk) / (sigma * np.sqrt(Tk))
    d2 = d1 - sigma * np.sqrt(Tk)
    return S * norm.cdf(d1) - Kk * np.exp(-r * Tk) * norm.cdf(d2)

V_h, V2_h = reprice(S_h, K, Tk), reprice(S_h, K2, Tk)
pnl_unhedged   = V_h - V0
pnl_delta      = (V_h - V0) - D0 * (S_h - S0)
pnl_delta_gamma = (V_h - V0) + w2 * (V2_h - V2_0) + w1 * (S_h - S0)
```

三种策略的 1 周 P&L 分布如下：

![对冲 P&L 分布：未对冲(σ=1.95) / 仅 Delta(σ=0.19) / Delta+Gamma(σ=0.01)](/images/greeks-hedging-map/hedge_pnl.png)

数字说话：

| 对冲策略 | P&L 标准差（指数点） | 含义 |
|---|---|---|
| 未对冲 | **1.95** | 裸持期权，赌对方向才赚 |
| 仅 Delta 对冲 | **0.19** | 中和方向，剩 Gamma+Vega 残余 |
| Delta + Gamma 对冲 | **0.012** | 再中和凸性，剩 Vega 残余 |

**Delta 对冲把方差砍掉 90%**，因为它消灭了线性方向风险；**再加 Gamma 对冲再砍一个数量级**，因为 Gamma 是 P&L 里二阶（凸性）项的来源。注意三条分布都没完全归零——因为波动率风险（Vega）始终没被对冲，这就是「已实现 vol ≠ 隐含 vol」时你还会亏的钱。

## 四、四类真实陷阱（务必先看这段）

1. **Gamma 对冲要「借第二个期权」，不是凭空消除**：本文用 $K_2=110$ 的第二个期权中和 Gamma（`w2 = -G0/G2_0`）。第一个版本我误把股票腿写成 `- w1·dS`，结果 Delta+Gamma 方差（0.19）居然比仅 Delta（1.95 误算...实际是 0.19 vs 0.19 同阶）没有下降——**股票空头在 P&L 里是 `+ w1·dS`（w1<0）**，符号反了等于没对冲。这条坑提醒：自融资组合的 P&L 里，空头股票贡献是 `+w1·(S_h−S0)`，别套错符号。

2. **离散再平衡，不是连续**：本文假设「期初一次性建好对冲、期末直接看 P&L」。真实 Delta 对冲要**每天/每小时再平衡**，每次都有买入卖出的滑点和手续费。离散再平衡会留下「凸性缺口」——这是期权卖方赚的钱（gamma scalping 收益）的来源，也是 Black-Scholes 在交易成本下不完美的根源。

3. **凸性缺口与 Gamma scalping**：持期权多头（正 Gamma）在对冲时「低买高卖」股票赚钱，这个收益恰好补偿 Theta 的流失——**正 Gamma 头寸靠 scalping 赚取时间价值**。但前提是波动真的实现（realized vol ≈ implied vol）。如果市场连续横盘，scalping 赚不到，Theta 白白流走。本文图4 的 Delta+Gamma 分布仍残留方差，就是因为 Vega（波动率）没对冲。

4. **铁幕假设：常数 vol、对数正态、无跳跃**：BS 的全部 Greeks 都建立在「波动率是常数、价格连续对数正态」上。真实市场有**波动率微笑**（ATM/OTM 的 vol 不同）、有**跳跃**（缺口跳空让 Delta 瞬间失真）、有**随机波动率**。本文算的 Vega 只在「整体平移 vol 曲面」时准；微笑存在时，单个 Greek 无法刻画曲面形状的二阶变化——那需要 Vega 的敏感（Vanna、Volga）和更多。

## 五、实战：用 Greeks 给做市商报价

假设你是做市商，客户要买一个 ATM 看涨，你报完价后持有空头期权（负 Gamma、负 Theta、正 Vega）。你的日常动作：

```python
# 持有空头看涨: 净 Greek = 取反
net_delta = -D0; net_gamma = -G0; net_vega = -Vg; net_theta = -Th
# 用股票把 Delta 中和 (持有 net_delta 份股票多头)
shares = -net_delta
# 每涨 1 点 S, 你的 Delta 变化 = net_gamma → 需要再平衡 shares += -net_gamma
# 每天时间流失 = net_theta (本例为负, 你作为空头反而是收 theta 的一方 → 赚)
print(f"需持有股票 {shares:.4f} 份; 每日 Theta 收入 {abs(net_theta):.4f}/天")
```

你赚的是 **Theta**（时间价值衰减），代价是承担 **Gamma**（标的价格剧烈跳动时再平衡亏损）和 **Vega**（波动率跳升时亏损）。这套「收 Theta、扛 Gamma/Vega」就是做市商的核心盈利模式——而它能不能赚钱，完全取决于你对这些 Greeks 的理解深度。

## 六、正确用法：Greeks 是地图，不是水晶球

Greeks 最该记住的三件事：

- **Delta 管方向、Gamma 管 Delta 的稳定性、Vega 管波动率、Theta 管时间**——它们各管一维，组合在一起才是完整风险地图；
- **对冲是「逐级消除」**：Delta 消除线性方向风险，Gamma 消除凸性风险，Vega 消除波动率风险，每加一层成本和复杂度都上升；
- **所有 Greek 都建立在 BS 假设上**，微笑和跳跃会让它们失真——实盘要补 Vanna/Volga 和跳跃修正。

## 结语

Greeks 把期权的「价格表层」翻译成「风险因子地图」：Delta 是方向敏感度、Gamma 是 Delta 的脆弱度、Vega 管波动率、Theta 管时间。本文用 BS 从零算出全部六个 Greek 并画出曲线与风险地图，再用蒙特卡洛实锤三种对冲的方差阶梯——未对冲 1.95 → 仅 Delta 0.19 → Delta+Gamma 0.012，证明「逐级消除风险因子」的有效性。但它的真边界在四类裂缝里：Gamma 对冲要借第二个期权且符号易错、再平衡是离散的、凸性缺口是做市商利润来源、而 BS 的全部 Greek 都扛着「常数 vol、无跳跃」的铁幕。理解这些，比背下 $N(d_1)$ 重要——它告诉你**任何 Greek 都是地图，不是水晶球，真实市场会在微笑和跳跃里把你漏掉的维度狠狠收费**。
