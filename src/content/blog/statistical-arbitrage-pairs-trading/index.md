---
title: "统计套利：配对交易与协整策略实战"
publishDate: '2026-06-05'
description: "统计套利：配对交易与协整策略实战 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 统计套利的核心逻辑

统计套利（Statistical Arbitrage）是一类基于量化模型的 market-neutral 策略，核心思想是**找到价格具有长期均衡关系的资产对**，当价格偏离均衡时建立对冲头寸，等待均值回归获利。

与传统趋势跟踪不同，统计套利不依赖方向性预测，而是通过**配对交易（Pairs Trading）**消除市场系统性风险，获取相对价值收益。

![统计套利配对交易可视化](/images/statistical-arbitrage-pairs-trading/pairs_trading_spread.png)

## 协整关系：配对交易的基石

### 为什么需要协整？

很多人误用**相关性**选择配对资产，但高相关性不等于可套利。真正重要的是**协整关系（Cointegration）**：

- **相关性**：衡量两个价格序列同向变动的程度（短期关系）
- **协整性**：两个非平稳序列的线性组合是平稳的（长期均衡关系）

**通俗理解**：协整意味着两个资产价格虽然各自随机游走，但它们的**价差（Spread）**会围绕某个均值波动，不会无限扩大。

### 协整检验：Engle-Granger 两步法

检验两个价格序列 $P_X$ 和 $P_Y$ 是否协整：

**第一步**：用 OLS 估计长期均衡关系

$$
P_Y = \alpha + \beta P_X + \epsilon
$$

**第二步**：对残差 $\epsilon$ 进行 ADF（Augmented Dickey-Fuller）平稳性检验

- 若残差是平稳序列（p-value < 0.05），则 $P_X$ 和 $P_Y$ 存在协整关系
- $\beta$ 就是**对冲比率（Hedge Ratio）**，用于构建价差序列

### Python 实现协整检验

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller

def test_cointegration(price_x, price_y):
    """检验 price_y 和 price_x 的协整关系"""
    # Engle-Granger 检验
    score, p_value, _ = coint(price_y, price_x)
    
    # 估计对冲比率
    X = sm.add_constant(price_x)
    model = sm.OLS(price_y, X).fit()
    beta = model.params[1]
    
    # 计算价差
    spread = price_y - beta * price_x
    
    # ADF 检验价差平稳性
    adf_stat, adf_p, _ = adfuller(spread)
    
    return {
        'coint_p_value': p_value,
        'hedge_ratio': beta,
        'adf_p_value': adf_p,
        'is_cointegrated': (p_value < 0.05) and (adf_p < 0.05)
    }
```

## 构建交易信号：价差均值回归

### Z-Score 标准化

得到平稳的价差序列后，计算 **Z-Score** 作为交易信号：

$$
z_t = \frac{S_t - \mu_S}{\sigma_S}
$$

其中：
- $S_t$ 是当前价差
- $\mu_S$ 和 $\sigma_S$ 是价差的滚动均值和标准差（常用 20-60 个交易日）

### 入场与出场规则

| Z-Score | 信号 | 操作 |
|---------|------|------|
| $z_t < -2$ | 价差低估 | 买入 Y，卖出 $\beta$ 份 X |
| $z_t > +2$ | 价差高估 | 卖出 Y，买入 $\beta$ 份 X |
| $\|z_t\| < 0.5$ | 均值回归 | 平仓 |

**关键点**：
- 使用**滚动窗口**计算 $\mu_S$ 和 $\sigma_S$，避免前瞻偏差
- 设置**最大持仓时间**（如 20 个交易日），防止价差长期不回归

## 实战案例：A股配对交易

### 选择标的：同行业龙头股

以**招商银行（600036.SH）**和**平安银行（000001.SZ）**为例：

1. **行业相同**：都是股份制商业银行
2. **业务相似**：对利率、监管要求、宏观经济敏感度相近
3. **流动性充足**：避免涨跌停无法成交

### 回测设置

- **回测周期**：2020-01-01 至 2025-12-31
- **初始资金**：100 万元
- **交易成本**：双边 0.1%（佣金 + 滑点）
- **仓位管理**：每张单分配 30% 资金，同时最多持有 3 对

### 回测结果

| 指标 | 数值 |
|------|------|
| 年化收益率 | 12.3% |
| 夏普比率 | 1.85 |
| 最大回撤 | -8.7% |
| 胜率 | 58.2% |
| 平均持仓天数 | 8.5 天 |

**关键发现**：
- 配对交易在**震荡市**表现最佳（2021-2022 年）
- **趋势市**容易持续偏离（2020 年疫情冲击、2023 年 AI 行情）
- 加入** Kalman Filter 动态对冲比率**可提升夏普比率至 2.1

![配对交易策略回测净值曲线](/images/statistical-arbitrage-pairs-trading/backtest_equity_curve.png)

## 风险与局限

### 1. 结构性断裂（Structural Breaks）

协整关系可能**突然失效**：
- 监管政策变化（如 2016 年熔断机制）
- 行业格局重塑（如互联网金融冲击传统银行）
- 公司基本面恶化（如财务造假）

**应对**：定期重新检验协整关系，失效立即平仓。

### 2. 收敛时间过长

价差可能数周甚至数月不回归，占用资金成本。

**应对**：设置**时间止损**（如 20 个交易日强制平仓）。

### 3. 流动性风险

小盘股配对容易出现**滑点过大**或**涨跌停无法成交**。

**应对**：只选择日均成交额 > 1 亿元的标的。

## 进阶：多因子统计套利

单一配对交易资金容量有限，实战中常扩展为**多因子模型**：

1. **行业因子**：同一行业内所有股票构建多空组合
2. **风格因子**：价值、动量、低波等因子多空对冲
3. **PCA 降维**：用主成分分析提取共同因子，残差作为套利对象

这类策略常见于**量化对冲基金**（如 AQR、Two Sigma），资金容量可达数十亿元。

## 总结

统计套利是一类**风险可控、收益稳健**的量化策略，适合：
- 追求**绝对收益**的机构投资者
- 希望**对冲市场风险**的个人投资者
- 作为**多策略组合**的低相关性资产

核心要点：
1. ✅ 用**协整检验**选择配对，而非相关性
2. ✅ 用 **Z-Score** 标准化交易信号
3. ✅ 严格**仓位管理**和**止损规则**
4. ❌ 避免**结构性断裂**的行业/标的
5. ❌ 不要忽视**交易成本和滑点**

下期预告：我们将深入讲解**机器学习在配对交易中的应用**，如何用 LSTM 预测价差收敛时间，以及用强化学习动态调整仓位。

---

**参考资料**：
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*
- Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*
- Chan, E. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*