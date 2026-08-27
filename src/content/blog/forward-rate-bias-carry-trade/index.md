---
title: "远期汇率偏差与 Carry Trade：UIP 失效的持久性"
description: "为什么高息货币在远期市场上'被预测贬值'，实际却往往升值或贬得不够？拆解无抛补利率平价（UIP）失效与 Carry Trade 收益来源，附 Python 完整模拟与风险校准。"
publishDate: '2026-08-27'
language: Chinese
tags:
  - 量化交易
  - 外汇
  - Carry Trade
  - 利率平价
categories: ["量化交易"]
slug: "forward-rate-bias-carry-trade"
image: "/images/forward-rate-bias-carry-trade/forward_bias_scatter.png"
---

外汇交易员常挂在嘴边的一句话是：**利差就是方向**。美元利率 5%，日元 0.5%，直觉上资金该从日元涌入美元，推高 USDJPY。可假如把这句话推向极端——「只要买高息货币、卖低息货币就能躺赚」——它就成了著名的 Carry Trade。

Carry Trade 有没有超额收益？教科书里的**无抛补利率平价（Uncovered Interest Rate Parity, UIP）**说：没有。远期汇率已经把这些利差「定价完毕」，高息货币未来的预期贬值恰好抵消利差。可过去四十年的实证数据反复打脸：UIP 不仅失效，而且失效得非常稳定。这篇文章从机制、实证、Python 模拟与风险管理四个维度，把远期汇率偏差拆清楚。

![远期汇率偏差：高息货币的实际贬值幅度小于 UIP 预测](/images/forward-rate-bias-carry-trade/forward_bias_scatter.png)

## 一、UIP 在说什么？一个简洁的零超额收益命题

UIP 的核心公式只有一行：

```
E[Δs] = i_d − i_f
```

- `E[Δs]`：本币计价下，外币的**预期**即期汇率变动（升值为正）。
- `i_d`、`i_f`：本币与外币的无风险利率。

如果美元利率 5%、日元 0.5%，以美元为本币、日元为外币，`i_d − i_f = 4.5%`，UIP 预测日元**相对美元**每年贬值 4.5%。反过来看，若一个美国投资者借入日元买入美元资产，他拿到 4.5% 的利差，但日元会贬值 4.5%——换成美元后正好抵消，超额收益为零。

这个命题在逻辑上很完美：如果存在确定性的正超额收益，套利资金会把它抹平。但它忽略了一个关键前提：**汇率风险被充分定价，且投资者风险中性**。现实中，这两个条件都不成立。

```python
import numpy as np

# 简例：USD 利率 5%，JPY 利率 0.5%
i_usd = 0.05
i_jpy = 0.005
uip_predicted_jpy_depreciation = i_usd - i_jpy
print(f"UIP 预测日元对美元年贬值: {uip_predicted_jpy_depreciation:.2%}")

# 若实际只贬值 2%，则美元投资者的多头 carry 净赚
actual_depreciation = 0.02
carry_excess = (i_usd - i_jpy) - actual_depreciation
print(f"实际贬值 2% 时，Carry 超额收益: {carry_excess:.2%}")
```

## 二、远期汇率偏差：数据到底怎么说？

Fama（1984）的经典回归把 UIP 可检验化：

```
Δs_{t+1} = α + β(f_t − s_t) + ε_{t+1}
```

- `f_t − s_t`：远期升贴水，等于利差（抛补利率平价 CIP 成立时）。
- UIP 要求 `β = 1`：远期升水 1%，下一年外币即期汇率升值 1%。

实证结果令人惊讶：绝大多数 G10 货币对的 `β` 显著**为负**。也就是说，远期市场「预测」高息货币会贬值，实际上高息货币却往往**升值**或贬得不够。这就是**远期汇率偏差（Forward Rate Bias）**，也是 Carry Trade 收益的来源。

```python
import numpy as np
import pandas as pd

# 模拟 10 个货币：横轴为利差，纵轴为实际即期变动
np.random.seed(42)
currencies = ['AUD', 'NZD', 'GBP', 'USD', 'EUR', 'JPY', 'CHF', 'SEK', 'NOK', 'CAD']
ir_diff = np.array([2.8, 2.5, 0.6, 0.0, -0.2, -1.2, -1.0, -0.5, -0.3, 0.1])  # %
# 模拟实际贬值：系数 -0.6，意味着 UIP 失效
spot_dep = -0.6 * ir_diff + np.random.normal(0, 0.8, len(ir_diff))

df = pd.DataFrame({'currency': currencies, 'ir_diff': ir_diff, 'spot_dep': spot_dep})
print(df.round(2))

# OLS 回归
beta, alpha = np.polyfit(ir_diff, spot_dep, 1)
print(f"\nFama 回归：α={alpha:.3f}, β={beta:.3f}  （UIP 要求 β≈1）")
```

这组模拟虽然简化，但抓住了主要 stylized fact：现实中 `β` 经常在 −1 到 0 之间，与 UIP 的 +1 相去甚远。

![Carry Trade 累积收益与 UIP 公平定价基准对比](/images/forward-rate-bias-carry-trade/carry_trade_cumreturns.png)

## 三、为什么 UIP 会失效？三个互补的解释

### 1. 风险溢价：你在赚的是「承担 Crash Risk 的保险费」

高息货币通常对应高增长、高风险、高杠杆的经济体。当全球风险偏好下降（Risk-Off），资金涌向低息避险货币，高息货币可能瞬间崩盘。Carry Trade 的正收益，可以被理解为**持续收取的保险费**，偶尔被一次性巨亏冲销。

学术上，这对应「外汇风险溢价」。远期汇率不预测未来即期，而是即期加风险溢价。高息货币「应该」贬值，但风险溢价让它升值或贬得少。

### 2. 非抛补套利者的资金约束与缓慢资本流动

即便存在偏离，套利者也受限于：
- **杠杆约束**：不能无限借钱。
- **止损与追加保证金**：短期波动会强制平仓。
- **机构授权**：很多资金不能裸空货币。

这意味着偏离可以长期存在，甚至数十年不衰减。

### 3. 央行干预与干预预期

高息国家央行有时通过升息或外汇干预阻止本币过度贬值，低息国家央行则可能乐见本币贬值。这种非对称干预会让「利差→贬值」的传导受阻。

## 四、用 Python 搭一个简化 Carry Trade 回测

下面演示一个经典的「分位组合」Carry Trade：每期按利差排序，做多前 30% 高息货币，做空后 30% 低息货币，等权重，月度再平衡。

```python
import numpy as np
import pandas as pd

np.random.seed(7)
n_currencies = 10
dates = pd.date_range('2019-01-01', periods=60, freq='ME')

# 模拟：每期利差（慢变 AR1）与下期即期收益（含 carry premium + crash risk）
ir_diff = np.zeros((len(dates), n_currencies))
ir_diff[0] = np.linspace(-2.5, 2.5, n_currencies)
for t in range(1, len(dates)):
    ir_diff[t] = 0.9 * ir_diff[t-1] + np.random.normal(0, 0.2, n_currencies)

# 下期收益：利差成分 + 风险溢价 + 噪声 + 偶发崩盘
returns = np.zeros_like(ir_diff)
for t in range(len(dates)-1):
    carry_component = ir_diff[t] / 12 / 100  # 月化
    risk_premium = 0.0015 * np.ones(n_currencies)  # 正向风险溢价
    noise = np.random.normal(0, 0.015, n_currencies)
    returns[t+1] = carry_component + risk_premium + noise

# 随机发生一次 global risk-off 崩盘
crash_t = np.random.choice(range(20, 50))
returns[crash_t] += np.random.choice([-0.12, 0.02], size=n_currencies,
                                      p=[0.5, 0.5])

# 分位组合：做多利差 top 3，做空利差 bottom 3
def carry_portfolio(rets, scores, q=3):
    long_idx = np.argsort(scores)[-q:]
    short_idx = np.argsort(scores)[:q]
    port_ret = rets[long_idx].mean() - rets[short_idx].mean()
    return port_ret

portfolio_rets = []
for t in range(len(dates)-1):
    r = carry_portfolio(returns[t+1], ir_diff[t], q=3)
    portfolio_rets.append(r)

portfolio_rets = np.array(portfolio_rets)
cum = np.cumprod(1 + portfolio_rets)
sharpe = portfolio_rets.mean() / portfolio_rets.std() * np.sqrt(12)
max_dd = (cum / np.maximum.accumulate(cum) - 1).min()

print(f"Carry Trade 组合：年化夏普 {sharpe:.2f}，最大回撤 {max_dd:.2%}")
print(f"月度收益均值 {portfolio_rets.mean():.2%}，标准差 {portfolio_rets.std():.2%}")
```

这个简化模型显示：Carry Trade 的收益分布是**正偏但肥左尾**的。大部分时间小赚，偶尔大亏。这也解释了为什么 2008 年 9 月 AUDJPY 单月跌幅可以超过 20%。

![Carry Trade 收益分布：中位数为正，但左侧存在厚尾崩溃](/images/forward-rate-bias-carry-trade/carry_return_distribution.png)

## 五、实务中的风险管理：不是不做，而是怎么做

Carry Trade 不是「买高息躺平」。实务中常见的风控手段包括：

1. **波动率缩放（Volatility Scaling）**：高波动时减仓。Carry Trade 崩溃往往伴随波动率飙升，动态降杠杆可以显著降低最大回撤。
2. **相关性约束**：避免同时重仓同一风险因子（如商品货币 AUD/CAD/NOK）。
3. **尾部对冲**：买入低delta的避险货币看涨期权，对冲极端风险。
4. **宏观过滤器**：只在风险偏好扩张期（如 VIX 低于中枢、信用利差收窄）建仓。

```python
# 波动率缩放示例
def vol_target_weights(returns_series, target_vol=0.05, lookback=12):
    """根据过去 lookback 期滚动标准差，调整仓位以匹配目标年化波动。"""
    rolling_std = pd.Series(returns_series).rolling(lookback).std().iloc[-1]
    if rolling_std == 0 or np.isnan(rolling_std):
        return 1.0
    return target_vol / (rolling_std * np.sqrt(12))

# 对 portfolio_rets 做波动率缩放
scaled_rets = []
for t in range(lookback := 12, len(portfolio_rets)):
    w = vol_target_weights(portfolio_rets[:t], target_vol=0.05, lookback=lookback)
    scaled_rets.append(w * portfolio_rets[t])
scaled_rets = np.array(scaled_rets)
print(f"原始夏普: {sharpe:.2f}  |  波动率缩放后夏普: "
      f"{scaled_rets.mean()/scaled_rets.std()*np.sqrt(12):.2f}")
```

## 六、结语

远期汇率偏差是外汇市场最持久的「异象」之一。UIP 的失效不是模型错了，而是它的假设——风险中性、无摩擦套利、风险被充分定价——在现实中不成立。

Carry Trade 赚的不是「免费午餐」，而是**承担崩溃风险的风险溢价**。它像卖保险：平时收保费，出事时赔付。真正的问题不是「有没有 alpha」，而是：

- 保费是否足以覆盖潜在的巨额赔付？
- 你的杠杆和止损机制是否会让你在赔付来临前先被清盘？
- 你的组合是否在单一风险事件上过度暴露？

理解了这些，Carry Trade 就不再是一个简单的「高息做多」信号，而是一个需要持续风险预算和动态调整的因子策略。

*本文代码均在 Python 3 环境下可直接运行，数据为模拟生成，仅用于教学演示，不构成任何投资建议。*
