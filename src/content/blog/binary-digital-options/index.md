---
title: "二元/数字期权定价与风险：用现金流复制敲开 exotic payoff"
description: "二元期权（Binary / Digital Option）的到期收益是一个开关：标的越过行权价就付固定现金（或一份资产），没越过就是零。payoff 从普通期权的连续斜坡变成不连续的阶跃，定价看似简单——现金-或-无看涨就是 N(d2) 的折现——但风险却极其凶险：临近到期时 Delta 和 Gamma 在行权价附近爆炸，做市商根本无法连续对冲。本文用纯 Python 从 Black-Scholes 出发，把数字期权拆成资产-或-无与现金-或-无两块积木，闭式定价并用蒙特卡洛校验（偏差 0.00003），画出阶跃 payoff、定价曲线、爆炸的 Greeks，再用看涨价差复制逼近真值，最后诚实拆穿 Pin Risk、复制成本、离散跳空与波动率微笑四类真实陷阱。"
publishDate: '2026-07-20'
tags:
  - 量化交易
  - 奇异期权
  - 二元期权
  - 期权定价
  - Python
language: Chinese
difficulty: advanced
---

普通欧式期权的到期收益是一条**斜坡**：标的越过行权价越多，赚得越多。二元期权（Binary Option，又叫 Digital Option 数字期权）把这条斜坡改成了一个**开关**——越过行权价就付一笔**固定**金额，差一分钱也是零。收益要么全有、要么全无，没有中间地带。

这个「不连续」的改动看似只是换了个 payoff 函数，定价甚至更简单，但它在风险管理上制造了一个噩梦：**到期前那一刻，如果标的正好卡在行权价附近，期权价值会在 0 和满额之间剧烈横跳，对冲所需的 Delta 趋于无穷大。** 这就是数字期权臭名昭著的 **Pin Risk（钉住风险）**。理解数字期权，本质上是理解「payoff 的不连续性如何把定价的容易和对冲的困难推向两个极端」。

![到期 payoff 对比：数字期权阶跃 vs 普通期权斜坡](/images/binary-digital-options/payoff_compare.png)

## 一、两种数字期权：现金-或-无 与 资产-或-无

数字期权有两种基本形态，它们是构造一切的积木：

- **现金-或-无看涨（Cash-or-Nothing Call）**：若 $S_T > K$，付固定现金 $Q$；否则付 0。
  $$\text{payoff} = Q \cdot \mathbf{1}_{\{S_T > K\}}$$
- **资产-或-无看涨（Asset-or-Nothing Call）**：若 $S_T > K$，付一份标的资产（价值 $S_T$）；否则付 0。
  $$\text{payoff} = S_T \cdot \mathbf{1}_{\{S_T > K\}}$$

这两块积木的妙处在于——**普通欧式看涨就是它们的差**：

$$\text{普通看涨} = \max(S_T - K, 0) = S_T\mathbf{1}_{\{S_T>K\}} - K\mathbf{1}_{\{S_T>K\}} = \text{资产-或-无} - K \times \text{现金-或-无}$$

这个恒等式是所有数字期权定价的骨架。反过来说，只要你会给普通期权定价，你就已经会给数字期权定价了。

## 二、Black-Scholes 闭式解

在 BS 框架下，风险中性测度里 $S_T > K$ 的概率正是 $N(d_2)$，于是**现金-或-无看涨**的价格就是这个概率的折现：

$$C_{\text{cash}} = Q \cdot e^{-rT} \cdot N(d_2), \qquad d_2 = \frac{\ln(S/K) + (r - \tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}$$

**资产-或-无看涨**则是：

$$C_{\text{asset}} = S \cdot N(d_1), \qquad d_1 = d_2 + \sigma\sqrt{T}$$

用 Python 实现这三块（含普通看涨的验证）：

```python
import numpy as np
from scipy.stats import norm

def d2(S, K, r, sigma, T):
    return (np.log(S/K) + (r - 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
def d1(S, K, r, sigma, T):
    return d2(S, K, r, sigma, T) + sigma*np.sqrt(T)

def cash_call(S, K, r, sigma, T, Q=1.0):        # 现金-或-无看涨
    return Q * np.exp(-r*T) * norm.cdf(d2(S, K, r, sigma, T))

def asset_call(S, K, r, sigma, T):              # 资产-或-无看涨
    return S * norm.cdf(d1(S, K, r, sigma, T))

def vanilla_call(S, K, r, sigma, T):            # 普通看涨 = 资产 - K*现金
    return asset_call(S, K, r, sigma, T) - K*cash_call(S, K, r, sigma, T, 1.0)

S0, K, r, sigma, T, Q = 100.0, 100.0, 0.03, 0.20, 1.0, 1.0
print("现金-或-无看涨 =", round(cash_call(S0, K, r, sigma, T, Q), 5))
```

平值（$S_0=K=100$）时，现金-或-无看涨价格约为 **0.50457**——直觉上很合理：越过行权价的风险中性概率略高于 0.5（因为漂移为正），再乘上一年期折现因子。

![现金-或-无看涨定价曲线：平滑的 N(d2)，K 附近最陡](/images/binary-digital-options/price_curve.png)

## 三、蒙特卡洛校验

数字期权的 MC 校验特别干净——只需数「有多少条路径终值越过 $K$」：

```python
np.random.seed(42)
N = 400_000
ST = S0 * np.exp((r - 0.5*sigma**2)*T + sigma*np.sqrt(T)*np.random.randn(N))
mc = np.exp(-r*T) * (ST > K).mean() * Q
cf = cash_call(S0, K, r, sigma, T, Q)
print(f"闭式={cf:.5f}  蒙特卡洛={mc:.5f}  偏差={abs(cf-mc):.5f}")
```

输出：

```
闭式=0.50457  蒙特卡洛=0.50460  偏差=0.00003
```

40 万条路径下偏差只有 **0.00003**，闭式解得到自洽验证。这也印证了现金-或-无看涨的本质就是「风险中性越价概率 × 折现」这一句话。

## 四、风险的凶险面：Greeks 在行权价爆炸

定价这么容易，为什么数字期权是做市商最头疼的品种之一？答案在 Greeks。

对 $S$ 求导，现金-或-无看涨的 Delta 是：

$$\Delta = \frac{\partial C_{\text{cash}}}{\partial S} = Q\,e^{-rT}\frac{n(d_2)}{S\sigma\sqrt{T}}$$

注意分母里有 $\sqrt{T}$。当 $T \to 0$（临近到期）且 $S \approx K$ 时，$d_2 \approx 0$，$n(d_2)$ 取最大值，而 $\sqrt{T} \to 0$——**Delta 趋于无穷大**。Gamma 更夸张，会在行权价两侧一正一负地剧烈甩动。

```python
h = 1e-3
def greek_delta(S):
    return (cash_call(S+h, K, r, sigma, T) - cash_call(S-h, K, r, sigma, T))/(2*h)
def greek_gamma(S):
    return (cash_call(S+h,K,r,sigma,T) - 2*cash_call(S,K,r,sigma,T)
            + cash_call(S-h,K,r,sigma,T))/(h**2)
```

![数字期权的 Delta/Gamma 在行权价爆炸（Pin Risk）](/images/binary-digital-options/greeks_explode.png)

图里 Delta（蓝）在 $K=100$ 附近形成尖峰，Gamma（红虚线）则在两侧甩出一正一负的尖刺。现实含义是：到期前如果标的钉在行权价附近，做市商为了 Delta 中性需要**在极短时间内买卖巨量标的**，交易成本和跳空风险会把理论对冲彻底击穿。这就是 **Pin Risk**——数字期权可以精确定价，却几乎无法被连续对冲。

## 五、用看涨价差复制：把不可对冲变成可近似对冲

既然纯数字期权无法安全对冲，实务里做市商通常**不卖真·数字期权，而是卖一个紧的看涨价差（call spread）去逼近它**：

$$\text{现金-或-无看涨} \approx \frac{1}{w}\Big[C_{\text{vanilla}}(K - \tfrac{w}{2}) - C_{\text{vanilla}}(K + \tfrac{w}{2})\Big]$$

其中 $w$ 是价差宽度。宽度 $w \to 0$ 时，这个组合在数学上收敛到真·数字期权（这正是「数字期权 = 看涨对 K 的负导数」的离散近似）。

```python
def cs_replication(S, K, width, r, sigma, T):
    lo, hi = K - width/2, K + width/2
    return (vanilla_call(S, lo, r, sigma, T)
            - vanilla_call(S, hi, r, sigma, T)) / width

for w in [20, 10, 4, 1]:
    approx = cs_replication(S0, K, w, r, sigma, T)
    print(f"价差宽度={w:2d} -> 复制价 {approx:.5f}  (真值 {cash_call(S0,K,r,sigma,T):.5f})")
```

![用看涨价差复制数字期权：越窄越逼近，但 Gamma 越危险](/images/binary-digital-options/spread_replication.png)

上图展示：价差越窄，复制曲线越贴近真·数字期权（红线），但代价是**越窄的价差 Gamma 越集中、越危险**——你只是把 Pin Risk 从「不可对冲」换成了「可对冲但成本高、且在两个行权价附近仍然凶险」。这就是做市商的权衡：卖一个稍宽的价差，赚取买卖价差，同时给自己留一条能真实对冲的路径。这也解释了为什么二元期权的报价通常隐含着比理论价更宽的买卖价差。

## A. 实现细节

- **定价字段**：仅用到期终值 $S_T$ 与行权价 $K$ 的比较（越价与否），是标准的欧式终点型 payoff，不涉及路径，用 BS 终端分布即可闭式求解。
- **两块积木**：所有价格都由「资产-或-无」与「现金-或-无」两块拼出，普通看涨 = 资产-或-无 − K×现金-或-无，这个恒等式贯穿全文。
- **执行/结算假设**：假设欧式行权、到期一次性结算固定现金 $Q=1$，无提前行权、无路径触碰条款（那属于障碍型 exotic，不在本文范围）。
- **Greeks 算法**：Delta / Gamma 用中心差分数值求导（步长 $h=10^{-3}$），与解析式趋势一致；临界区数值导数本身也会因不连续而放大，这恰好可视化了风险。
- **复制口径**：看涨价差复制用两只普通欧式看涨的差 / 宽度，未计入建仓的买卖价差与冲击成本。

## B. 已知偏差

- **Pin Risk 无法被完全消除**：本文用价差复制把不可对冲近似成可对冲，但临近到期在行权价附近，任何有限宽度的价差仍有巨大 Gamma。真实做市商还会叠加行权价错开、提前平仓、报价加宽等手段，本文未建模这些操作成本。
- **常数波动率假设**：BS 用单一 $\sigma$，但数字期权价格对**波动率微笑的斜率**极其敏感——现金-或-无看涨 ≈ 看涨对 K 的负导数，而市场看涨价格里含 skew，真实数字期权价必须用微笑斜率修正（所谓 skew-adjusted digital），本文的平价 $\sigma=0.2$ 未纳入 skew。
- **离散跳空风险**：连续 BS 假设标的连续变动，但真实标的在夜盘 / 事件时会跳空。数字期权在跳空穿越行权价时，对冲者完全来不及调整，实际损失远超模型。
- **无红利、无交易成本**：模型未含标的分红与对冲的交易成本，二者都会系统性影响真实报价。

## C. 结果解读

**定价极易、对冲极难，是数字期权的核心矛盾。** 现金-或-无看涨就是 $e^{-rT}N(d_2)$ 一行公式，蒙特卡洛偏差仅 0.00003 完全自洽——但这份「简单」是一种错觉。真正的成本藏在第四节：Delta 和 Gamma 在行权价附近爆炸，临近到期时对冲需要无穷大的换手，理论对冲在现实里根本执行不了。

**payoff 的不连续性是一切风险的根源。** 把普通期权的斜坡换成阶跃，数学上只是把 $\max(S-K,0)$ 换成 $\mathbf{1}_{\{S>K\}}$，但这一步让二阶导（Gamma）从有界变成在临界点发散。任何 payoff 里出现「硬阈值 / 全有全无」结构的衍生品，都会继承这种 Pin Risk——数字期权只是最纯粹的样本。

**价差复制揭示了做市商的真实做法。** 没有人真去裸卖数字期权。第五节说明业界用紧看涨价差逼近，宽度是「逼近精度」与「可对冲性」之间的权衡：价差越窄越像真·数字期权，但 Gamma 越集中越难对冲。理解这一点，就理解了为什么二元期权的市场报价总是隐含一个明显宽于理论价的买卖价差——那个价差不是暴利，而是做市商给 Pin Risk 上的保险。

**对波动率微笑的敏感性是实盘定价的关键。** 数字期权本质是「看涨价格对行权价的导数」，所以它对微笑的**斜率**而非单点波动率敏感。用平价 $\sigma$ 定价只是教学起点；任何要拿去交易的数字期权，都必须先从市场微笑里提取 skew，再对 $N(d_2)$ 做斜率修正——否则理论价会系统性地偏离市场，尤其在 skew 陡峭的品种上。
