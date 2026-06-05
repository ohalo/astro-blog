---
title: "统计套利实战：配对交易与协整分析"
publishDate: '2026-06-05'
description: "统计套利实战：配对交易与协整分析 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 什么是配对交易？

配对交易（Pairs Trading）是一种**市场中性策略**，通过同时买入和做空两只高度相关的股票，从它们的价差回归中获利。

### 核心思想

1. **寻找配对**：找到历史价格走势高度相关的两只股票
2. **监控价差**：计算价差的Z-Score（标准化后的偏离度）
3. **交易信号**：
   - Z-Score < -2：买入股票A，做空股票B（价差偏小，预期扩大）
   - Z-Score > +2：做空股票A，买入股票B（价差偏大，预期收敛）
4. **平仓条件**：Z-Score回归到 ±0.5 以内

![配对交易价差可视化](/images/pairs-trading-cointegration/spread-visualization.jpg)

## 协整检验：数学基础

仅仅看相关性是不够的，我们必须进行**协整检验**（Cointegration Test）。

### 为什么需要协整？

两只股票的**价格序列可能都是非平稳的**（有趋势），但它们的**线性组合可能是平稳的**。这意味着长期来看，它们之间的价差会回归到均值。

### Augmented Dickey-Fuller (ADF) 检验

我们使用 ADF 检验来验证残差是否平稳：

```python
from statsmodels.tsa.stattools import adfuller
import numpy as np

def check_cointegration(stock_a, stock_b):
    # 1. 线性回归：stock_a = alpha + beta * stock_b + residual
    X = stock_b.values.reshape(-1, 1)
    y = stock_a.values
    beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0], y.mean() - stock_b.values.mean() * np.linalg.lstsq(X, y, rcond=None)[0]
    
    # 2. 计算残差
    residuals = y - (alpha + beta * stock_b.values)
    
    # 3. ADF检验
    result = adfuller(residuals)
    
    print(f"ADF Statistic: {result[0]:.4f}")
    print(f"p-value: {result[1]:.4f}")
    print(f"临界值: {result[4]}")
    
    # p-value < 0.05 说明残差平稳，存在协整关系
    return result[1] < 0.05
```

![协整关系示意图](/images/pairs-trading-cointegration/cointegration-diagram.jpg)

## 实战案例：招商银行 vs 平安银行

让我用一个A股案例演示完整的配对交易流程。

### Step 1: 数据获取与预处理

```python
import akshare as ak
import pandas as pd

# 获取招商银行(600036)和平安银行(000001)的日线数据
def get_stock_data(symbol, start="20240101", end="20251231"):
    df = ak.stock_zh_a_hist(symbol=symbol, period="daily", 
                              start_date=start, end_date=end)
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    return df['收盘']

stock_a = get_stock_data("600036")  # 招商银行
stock_b = get_stock_data("000001")  # 平安银行

# 对齐数据
data = pd.DataFrame({'招商银行': stock_a, '平安银行': stock_b}).dropna()
```

### Step 2: 计算对冲比例（Hedge Ratio）

```python
from sklearn.linear_model import LinearRegression

# 滚动窗口计算对冲比例（60个交易日）
window = 60
hedge_ratios = []

for i in range(window, len(data)):
    X = data['平安银行'].iloc[i-window:i].values.reshape(-1, 1)
    y = data['招商银行'].iloc[i-window:i].values
    model = LinearRegression().fit(X, y)
    hedge_ratios.append(model.coef_[0])

data = data.iloc[window:]
data['对冲比例'] = hedge_ratios

# 计算价差
data['价差'] = data['招商银行'] - data['对冲比例'] * data['平安银行']
```

### Step 3: 生成交易信号

```python
import numpy as np

# 计算价差的Z-Score（滚动窗口20日）
data['价差均值'] = data['价差'].rolling(window=20).mean()
data['价差std'] = data['价差'].rolling(window=20).std()
data['Z-Score'] = (data['价差'] - data['价差均值']) / data['价差std']

# 交易信号
data['持仓'] = 0
data.loc[data['Z-Score'] < -2, '持仓'] = 1   # 买入招商银行，做空平安银行
data.loc[data['Z-Score'] > 2, '持仓'] = -1   # 做空招商银行，买入平安银行
data.loc[abs(data['Z-Score']) < 0.5, '持仓'] = 0  # 平仓

# 持仓变化（避免重复交易）
data['持仓变化'] = data['持仓'].diff().fillna(0)
```

### Step 4: 回测 performance

```python
# 假设手续费万分之2.5，滑点0.1%
transaction_cost = 0.00025
slippage = 0.001

data['策略收益'] = 0

for i in range(1, len(data)):
    if data['持仓变化'].iloc[i] != 0:
        # 开仓/平仓成本
        cost = transaction_cost + slippage
        data.loc[data.index[i], '策略收益'] = -cost
    elif data['持仓'].iloc[i] != 0:
        # 持仓期间的价格变动
        price_change_a = data['招商银行'].iloc[i] / data['招商银行'].iloc[i-1] - 1
        price_change_b = data['平安银行'].iloc[i] / data['平安银行'].iloc[i-1] - 1
        
        # 根据持仓方向计算收益
        if data['持仓'].iloc[i] == 1:
            data.loc[data.index[i], '策略收益'] = price_change_a - price_change_b
        elif data['持仓'].iloc[i] == -1:
            data.loc[data.index[i], '策略收益'] = price_change_b - price_change_a

# 累计收益
data['累计收益'] = (1 + data['策略收益']).cumprod()

# 绩效指标
total_return = data['累计收益'].iloc[-1] - 1
sharpe_ratio = data['策略收益'].mean() / data['策略收益'].std() * np.sqrt(252)
max_drawdown = (data['累计收益'] / data['累计收益'].cummax() - 1).min()

print(f"总收益率: {total_return:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")
```

## 关键风险与注意事项

### 1. 结构性断裂（Structural Break）

配对关系可能因为公司基本面变化而失效：
- 并购重组
- 行业政策变化
- 管理层变动

**应对方法**：使用滚动窗口（60-90天）动态更新对冲比例，一旦协整检验失败立即停止交易。

### 2. 延迟均值回归

价差可能在极端位置停留很长时间，导致：
- 资金占用成本高
- 止损压力

**应对方法**：
- 设置时间止损（例如：持仓超过20天强制平仓）
- 使用期权对冲（买入跨式期权）

### 3. 交易成本侵蚀利润

配对交易通常交易频繁，交易成本至关重要。

**降低成本的方法**：
- 选择低佣金券商（万分之1.5以下）
- 使用限价单（Limit Order）减少滑点
- 只在Z-Score > 2.5 或 < -2.5 时交易（减少交易频次）

## 总结

配对交易是一种**低风险、市场中性**的策略，适合振荡市和横盘市。但要成功实施，必须：

1. ✅ 严格的协整检验（ADF p-value < 0.05）
2. ✅ 动态更新对冲比例（滚动回归）
3. ✅ 控制交易成本（选择低佣金券商）
4. ✅ 设置止损规则（时间止损 + 价差止损）
5. ✅ 分散投资（同时交易多个配对）

**下期预告**：我将介绍如何使用**机器学习（LSTM）**预测价差方向，进一步提升配对交易的成功率。

---

*如果你对完整的Python代码感兴趣，可以在评论区留言，我会分享Jupyter Notebook版本的回测框架。*
