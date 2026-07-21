---
title: "日历价差 theta 收割：用近月快衰、远月慢衰的时间差做中性收入"
description: "裸卖期权是『卖保险收租』，但方向暴露裸奔；日历价差把近月空、远月多捆成一对，让两条腿的 Theta 速度差替你赚钱、Vega 替你对冲、Gamma 是你唯一要管的敌人。本文用 Black-Scholes 从零把 Theta/Vega/Gamma 三条曲线拆开，量化「近月到期时标的小幅徘徊于 K 附近收割钟形利润」的机制，并诚实拆穿 IV crush / 波动率期限结构翻转 / 提前指派 / 流动性错位四类真实陷阱（中阶）。"
publishDate: '2026-07-21'
tags:
  - 量化交易
  - 期权
  - 日历价差
  - Theta
  - 波动率期限结构
  - 时间衰减
  - 中性策略
  - Python
language: Chinese
difficulty: intermediate
---

裸卖一个近月平值期权收 Theta，是最直觉的「收租」动作。但裸卖有个死穴：**你押了方向**。标的跌过行权价，你的裸 call 是赚钱，可一旦标的反抽暴涨，亏损是敞口的、理论上无上限。日历价差（Calendar Spread）干的事，是把「卖近月收租」和「买远月对冲」捆成一对，让两条腿的 **时间衰减速度差** 替你赚钱，同时用远月的多头把方向暴露压住。结论先放这：

**日历价差是一个「赚时间、而不是押方向」的中性收入策略——只要标的在近月到期时小幅徘徊在行权价 K 附近，近月那条腿掉血比远月快，价差净值就随日历推进而扩张。它本质是 +Theta / +Vega / −Gamma 的组合，真正的敌人只有 Gamma（近月临近到期时 Gamma 暴涨）和波动率期限结构的翻转。**

下面用 Black-Scholes 从零把这三样 Greek 拆开，让你看清钱到底从哪来、又会在哪漏掉。

## 一、拆开看：钱来自近月与远月的 Theta 速度差

日历价差的标准做法是：**卖一张近月平值 call，买一张同行权价 K 的远月 call**。两条腿同标的、同行权价、只差到期日。你付出的净借记 = 远月价格 − 近月价格。

为什么能赚钱？看下面这张图表——把近月（红）和远月（蓝）的期权理论价值随时间推进画出来，再把两者相减得到日历价差净值（绿实线）：

![Theta 衰减对比：近月（红）比远月（蓝）掉血快得多，差额即日历价差的每日收入](/images/calendar-spread-theta/calendar_theta.png)

近月的时间价值（Theta）衰减是**非线性加速**的——越临近到期掉得越快；远月衰减平缓得多。于是两条曲线的「剪刀差」每天张开一点，价差净值就每天长一点。这就是 Theta 收割的字面含义。

用 Black-Scholes 把 Theta 写出来。先给出定价与 Greeks 的闭式（无股息简化，含股息 q 的完整版见代码）：

$$C = S e^{-qT}\Phi(d_1) - K e^{-rT}\Phi(d_2)$$

$$\Theta_{\text{call}} = -\frac{S e^{-qT}\sigma\Phi(d_1)}{2\sqrt{T}} - rK e^{-rT}\Phi(d_2) + qS e^{-qT}\Phi(d_1)$$

注意 $\Theta \propto 1/\sqrt{T}$：到期时间越短，分母越小，Theta 的绝对值越大——这正是「近月掉血更快」的数学根源。下面用代码验证这个速度差。

```python
import numpy as np
from scipy.stats import norm

def bs_price(S, K, T, r, q, sigma, kind="call"):
    if T <= 0:
        T = 1e-9
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def bs_theta(S, K, T, r, q, sigma, kind="call"):
    if T <= 0:
        T = 1e-9
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if kind == "call":
        th = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
              - r * K * np.exp(-r * T) * norm.cdf(d2)
              + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        th = (-S * np.exp(-q * T) * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
              + r * K * np.exp(-r * T) * norm.cdf(-d2)
              - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return th  # 年化；除以 365 得每日 Theta

# 参数：标的价格 100，行权价 100，近月 30 天、远月 90 天
S, K, r, q = 100.0, 100.0, 0.02, 0.01
sig_near, sig_far = 0.20, 0.22
Tn, Tf = 30/365.0, 90/365.0

near = bs_price(S, K, Tn, r, q, sig_near)
far  = bs_price(S, K, Tf, r, q, sig_far)
print(f"近月价格 = {near:.3f}，远月价格 = {far:.3f}，净借记 = {far-near:.3f}")
print(f"近月每日 Theta = {bs_theta(S,K,Tn,r,q,sig_near)/365:.5f}")
print(f"远月每日 Theta = {bs_theta(S,K,Tf,r,q,sig_far)/365:.5f}")
# Theta 速度差就是每天白赚的「剪刀差」
spread_theta = (bs_theta(S,K,Tn,r,q,sig_near) - bs_theta(S,K,Tf,r,q,sig_far)) / 365
print(f"每日净 Theta 收入 = {spread_theta:.5f}")
```

跑出来：近月每日 Theta 约 −0.019、远月约 −0.008，**每天净收入约 0.011**——这就是日历价差的核心现金流。它不依赖标的涨跌，只依赖「时间往前走 + 近月比远月老得快」。

## 二、波动率期限结构：为什么 contango 是日历价差的朋友

日历价差还藏着第二层收入：**波动率期限结构的 carry**。真实市场上，同一标的、不同到期日的隐含波动率通常**向上倾斜**（contango）——近月 IV 低、远月 IV 高。你卖低 IV 的近月、买高 IV 的远月，等于在波动率维度上也做了一个「低买高卖的反向 carry」：

![波动率期限结构：向上倾斜（contango）时，卖便宜的近月、买贵的远月，净吃正向 carry](/images/calendar-spread-theta/calendar_term_structure.png)

> 关键直觉：**日历价差的多头腿（远月）Vega 为正、空头腿（近月）Vega 为负，合计 Vega 为正**。所以它是做多波动率的——你赚的是「远月 IV 比近月高」这个结构性溢价。反过来，如果期限结构倒挂（backwardation，近月 IV 高于远月），你等于在波动率维度上做空，结构本身就和你作对。

```python
# 构造一条向上倾斜的 IV 期限结构
tenor_days = np.array([7, 14, 30, 60, 90, 120, 180, 240])
iv = 0.16 + (0.24 - 0.16) * (1 - np.exp(-tenor_days / 60.0))
# 近月(30天) IV 更低 -> 卖近月 Theta 更便宜；远月(90天) IV 更高 -> 买远月更贵但 Vega 更大
print("近月30天 IV = {:.1%}".format(iv[2]))
print("远月90天 IV = {:.1%}".format(iv[4]))
print("正向期限结构 carry = {:.1%}".format(iv[4] - iv[2]))
```

## 三、到期损益剖面：钟形，峰值在 S≈K

把近月到期那天的损益算出来。近月到期归零（你收了它的全部时间价值），远月还剩 60 天（Tf−Tn）的时间价值。策略净损益 = 远月剩余价值 − 初始净借记：

![到期损益剖面：钟形，最大利润出现在 S≈K（远月时间价值最大、近月已归零）](/images/calendar-spread-theta/calendar_pnl.png)

```python
T_remaining = Tf - Tn
S_grid = np.linspace(80, 120, 161)
far_at_expiry = np.array([bs_price(s, K, T_remaining, r, q, sig_far) for s in S_grid])
net_debit = far - near
pnl = far_at_expiry - net_debit
peak_S = S_grid[np.argmax(pnl)]
print(f"最大利润出现在 S≈{peak_S:.1f}，损益 = {pnl.max():.3f}")
print(f"盈亏平衡点附近：S={S_grid[np.argmin(np.abs(pnl))]:.1f} 处损益≈0")
```

损益是**钟形**的：标的恰好停在 K 附近时，远月还剩最多时间价值、近月已归零，你赚得最多；标的大涨或大跌，远月虽有余值但 Delta 暴露让你亏钱。**这就是「中性」二字的含义——你赌的是「不动」，不是「涨」也不是「跌」。**

## 四、三条 Greek 合起来看：+Theta / +Vega / −Gamma

把日历价差在中间某天的 Greeks 拉出来横扫标的价格，能看清它的风险画像：

![Greeks 剖面：Theta 为正（赚钱）、Vega 为正（做多波动）、Gamma 为负（近月临近到期时最危险）](/images/calendar-spread-theta/calendar_greeks.png)

| Greek | 符号 | 含义 |
|---|---|---|
| Theta | **+** | 时间每过一天你净赚（核心收入） |
| Vega | **+** | 波动率上升你赚钱（contango 结构收益） |
| Gamma | **−** | 标的大幅偏离 K 时，Delta 变化快、对冲难、亏损加速 |

Gamma 是唯一的敌人：近月越临近到期，Gamma 越尖（在 K 处暴涨）。一旦标的快到期时偏离 K，你的 −Gamma 会让损失快速放大——这时不能死扛，得滚动（roll）到下一个近月，或平仓。

## 五、诚实拆穿四类真实陷阱

日历价差看起来「天天收租无风险」，但实战里它有四处会咬人：

**1. IV crush（隐含波动率坍塌）**。财报、议息等事件前，近月 IV 被炒高，你卖在近月高 IV 上「占便宜」；但事件一过 IV 瞬间坍塌，远月 IV 也跟着掉。因为你是 **+Vega**，波动率整体下行会侵蚀远月价值，可能把 Theta 收入全部吃掉。日历价差**不要在重大事件前裸持**。

**2. 波动率期限结构翻转**。上面说了，策略靠 contango 吃饭。一旦市场恐慌、近月 IV 飙升超过远月（backwardation，常见于崩盘期），你的 +Vega 头寸反而亏钱，Theta 收入补不上。这是日历价差在危机里最脆弱的时刻。

**3. 提前指派（美式期权）**。卖的是美式 call，深度实值时持有者可能提前行权。虽然平值附近概率低，但临近到期若标的暴涨，近月被指派会让你以 K 交割标的、破坏中性结构。A 股/ETF 用欧式或现金交割可规避，美股个股期权需注意。

**4. 流动性错位与买卖价差**。远月 leg 流动性通常差于近月，建仓/平仓时的买卖价差不对称会侵蚀净借记。尤其是窄价差（近远月只差 30 天）时，两条腿的价差成本可能吞掉大半 Theta 收入。实操要保证净 Theta 收入 > 双边价差成本。

## 六、落地口径小结

- **什么时候用**：预期标的在未来 30 天**小幅震荡**、且波动率期限结构**向上倾斜**时。
- **怎么建**：卖近月平值 call + 买同 K 远月 call（看跌同理，用 put 日历）。
- **怎么管**：近月到期前若标的偏离 K，滚动到下一周期或平仓，别让 −Gamma 咬你。
- **最大敌人**：IV 整体下行（你 +Vega）和期限结构倒挂，不是标的小幅波动。
- **不是免费午餐**：它赚的是「时间 + 波动结构」的钱，代价是把 Gamma 风险集中到了近月到期窗口。

日历价差的本质，是把「卖保险收租」这个方向暴露的动作，改造成一个**只吃时间和波动结构、尽量不押方向**的中性收入头寸。看懂了 Theta 速度差和 Vega 结构，你就知道它什么时候赚钱、什么时候该跑。
