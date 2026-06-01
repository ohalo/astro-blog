---
title: "量化交易的风险管理：止损、仓位控制与最大回撤"
publishDate: '2026-06-01'
description: "量化交易的风险管理：止损、仓位控制与最大回撤 - halo的技术博客"
tags:
  - 量化交易
language: Chinese
---

# 量化交易的风险管理：止损、仓位控制与最大回撤

很多人沉迷于策略的收益率，却忽视了风险管理的重要性。但真实的市场会教训每一个轻视风险的人：**活着比赚快钱更重要**。

本文从量化视角，系统讲解风险管理的三个核心问题：
1. 如何设置止损？
2. 如何分配仓位？
3. 如何控制最大回撤？

## 为什么风险管理比策略更重要？

**一个真实的故事：**

2015年股灾，某量化私募的阿尔法策略年化收益30%+，夏普比率2.5，看起来完美。但因为没有做好风控，在流动性危机中单日亏损15%，触发清盘线，被迫清盘。

**教训：**
- 高收益 ≠ 好策略
- 忽略尾部风险，迟早会爆仓
- 风控是量化交易的"安全带"

![风险管理金字塔](/images/2026-06-01-risk-management/risk-pyramid.jpg)

## 止损：活着才有未来

### 为什么需要止损？

**数学真相：**
- 亏损50%，需要涨100%才能回本
- 亏损90%，需要涨900%才能回本
- **控制单次亏损，比追求单次暴利重要100倍**

### 量化止损方法

#### 1. 固定比例止损

最简单的方法：单笔亏损达到账户资金的X%就止损。

```python
# 固定比例止损
MAX_LOSS_PCT = 0.02  # 单笔最大亏损2%

def check_stop_loss(position, current_price):
    entry_price = position.avg_cost
    loss_pct = (entry_price - current_price) / entry_price
    
    if loss_pct >= MAX_LOSS_PCT:
        return True  # 触发止损
    return False
```

**优点：** 简单直观
**缺点：** 没有考虑市场波动，可能被"噪音"止损

#### 2. 波动率止损（ATR）

根据市场波动率动态调整止损距离。波动大时止损宽，波动小时止损窄。

```python
import numpy as np

def calculate_atr(high, low, close, period=14):
    tr = np.maximum(high[1:] - low[1:],
            np.maximum(abs(high[1:] - close[:-1]),
                       abs(low[1:] - close[:-1])))
    atr = pd.Series(tr).rolling(period).mean().values
    return atr

# ATR止损：价格跌破入场价 - 2倍ATR时止损
stop_price = entry_price - 2 * atr[-1]

if current_price < stop_price:
    sell_all()
```

**优点：** 自适应市场波动
**缺点：** 需要计算ATR，参数敏感

#### 3. 时间止损

持仓超过N天仍未盈利，说明策略可能失效，平仓离场。

```python
MAX_HOLD_DAYS = 20

if (current_date - entry_date).days >= MAX_HOLD_DAYS:
    if unrealized_pnl < 0:
        sell_all()  # 亏损超期，止损
```

**逻辑：** 好的交易应该很快盈利，长期浮亏说明判断错误

#### 4. 追踪止损（Trailing Stop）

盈利后，不断上移止损线，锁定利润。

```python
highest_price = entry_price

for each day:
    highest_price = max(highest_price, current_price)
    stop_price = highest_price * 0.95  # 从最高价回撤5%止损
    
    if current_price < stop_price:
        sell_all()  # 触发追踪止损
```

**优点：** 让利润奔跑，同时保护盈利
**缺点：** 在震荡市中可能被反复止损

![止损方法对比](/images/2026-06-01-risk-management/stop-loss-methods.jpg)

## 仓位管理：不要把鸡蛋放在一个篮子里

### 为什么需要仓位管理？

**凯利公式的启示：**

假设一个策略胜率60%，盈亏比1:1，那么每次应该下注多少？

凯利公式：
$$f^* = \frac{p}{b} - \frac{1-p}{1}$$

其中：
- $f^*$：最优仓位比例
- $p$：胜率
- $b$：赔率（盈利/亏损）

对于上述例子：$f^* = 0.6 - 0.4 = 0.2$，即每次下注20%资金。

**但凯利公式有缺陷：**
- 假设胜率、赔率固定（现实中会变化）
- 全凯利（Full Kelly）波动太大，实战中用半凯利（Half Kelly）

### 量化仓位管理方法

#### 1. 固定分数法（Fixed Fractional）

每次风险固定比例的资金，例如每次风险2%。

```python
def calculate_position_size(account_equity, risk_pct, stop_loss_pct, price):
    """
    account_equity: 账户权益
    risk_pct: 风险比例（例如0.02 = 2%）
    stop_loss_pct: 止损比例（例如0.05 = 5%）
    price: 当前价格
    """
    risk_amount = account_equity * risk_pct
    position_size = risk_amount / (price * stop_loss_pct)
    return int(position_size)
```

**示例：**
- 账户10万，风险2% = 2000元
- 止损5%，价格10元
- 仓位 = 2000 / (10 * 0.05) = 4000股

#### 2. 波动率调整仓位（Volatility Targeting）

目标是让组合波动率达到目标值（例如年化15%）。

```python
TARGET_VOL = 0.15  # 目标波动率15%
current_vol = calculate_portfolio_vol(return_series)

if current_vol > 0:
    leverage = TARGET_VOL / current_vol
    adjust_position(leverage)
```

**逻辑：** 市场波动大时降低仓位，波动小时加仓

#### 3. 风险平价（Risk Parity）

让组合中每个资产贡献相同的风险。

```python
# 假设有股票、债券、商品三个资产
assets = ['stock', 'bond', 'commodity']
cov_matrix = calculate_covariance(returns)

# 风险平价权重
risk_budget = np.array([1/3, 1/3, 1/3])  # 等风险预算
weights = risk_parity_optimization(cov_matrix, risk_budget)
```

**优点：** 分散化效果好，回撤小
**缺点：** 在低利率环境下，债券贡献降低

#### 4. 马科维茨均值-方差优化

经典的现代投资组合理论（MPT）。

```python
import cvxpy as cp

def optimize_portfolio(returns, cov_matrix, risk_aversion=1.0):
    n = len(returns)
    weights = cp.Variable(n)
    
    expected_return = returns.values @ weights
    portfolio_variance = cp.quad_form(weights, cov_matrix.values)
    
    # 最大化夏普比率
    objective = cp.Maximize(expected_return - risk_aversion * portfolio_variance)
    constraints = [cp.sum(weights) == 1, weights >= 0]  # 全投资，不允许做空
    
    problem = cp.Problem(objective, constraints)
    problem.solve()
    
    return weights.value
```

## 最大回撤控制：量化交易的"生死线"

### 什么是最大回撤（Max Drawdown）？

**定义：**
$$MDD = \max_{t \in [0,T]} \left( \frac{\text{Peak}_t - \text{Trough}_t}{\text{Peak}_t} \right)$$

简单说：从最高点到最低点的最大跌幅。

**为什么重要？**
- 实盘心理承受极限（回撤30%，很多人会崩溃）
- 产品清盘线（私募通常0.8清盘，即回撤20%）
- 策略失效信号（超过历史最大回撤，可能过拟合）

### 控制最大回撤的方法

#### 1. 仓位与波动率挂钩

市场波动大时，自动降低仓位。

```python
def dynamic_position_sizing(vix_index, base_position=1.0):
    """
    VIX指数高时降低仓位
    """
    if vix_index < 15:  # 低波动
        return base_position
    elif vix_index < 25:  # 中等波动
        return base_position * 0.7
    else:  # 高波动
        return base_position * 0.4
```

#### 2. 回撤触发降仓

当回撤达到阈值时，降低仓位直到回撤恢复。

```python
MAX_DRAWDOWN = 0.15  # 最大回撤15%

def check_drawdown(current_equity, peak_equity):
    drawdown = (peak_equity - current_equity) / peak_equity
    
    if drawdown >= MAX_DRAWDOWN:
        reduce_position(0.5)  # 仓位减半
        send_alert("触发回撤警戒线！")
```

#### 3. 多策略分散

不要把所有资金放在一个策略上。

```python
strategies = {
    'trend_following': 0.3,  # 趋势跟踪30%
    'mean_reversion': 0.3,    # 均值回归30%
    'market_neutral': 0.4     # 市场中性40%
}

# 不同策略相关性低，组合回撤更小
```

#### 4. 使用止损 + 止盈

既控制亏损，也锁定利润。

```python
def take_profit_and_stop_loss(position, entry_price):
    current_price = get_current_price()
    
    profit_pct = (current_price - entry_price) / entry_price
    loss_pct = (entry_price - current_price) / entry_price
    
    # 盈利10%止盈，亏损5%止损
    if profit_pct >= 0.10:
        sell_all()
        print("止盈离场")
    elif loss_pct >= 0.05:
        sell_all()
        print("止损离场")
```

## 实战案例：完整风控系统

### 策略：多因子选股 + 市场风险对冲

**风控规则：**

1. **单笔止损**：个股亏损超过8%止损
2. **组合止损**：组合回撤超过15%，仓位降至50%
3. **仓位限制**：单只股票≤5%，单一行业≤30%
4. **波动率控制**：组合目标年化波动率18%
5. **黑名单**：ST股、退市股、流动性差的股票不碰

**代码框架：**

```python
class RiskManager:
    def __init__(self):
        self.max_position_pct = 0.05
        self.max_sector_pct = 0.30
        self.stop_loss_pct = 0.08
        self.max_drawdown = 0.15
    
    def check_risk(self, portfolio):
        # 1. 检查单笔止损
        for position in portfolio.positions:
            if position.unrealized_pnl_pct <= -self.stop_loss_pct:
                position.close()
        
        # 2. 检查组合回撤
        if portfolio.drawdown >= self.max_drawdown:
            portfolio.reduce_position(0.5)
        
        # 3. 检查仓位限制
        for position in portfolio.positions:
            if position.weight >= self.max_position_pct:
                position.reduce(self.max_position_pct)
        
        # 4. 检查行业暴露
        sector_weights = portfolio.sector_weights
        for sector, weight in sector_weights.items():
            if weight >= self.max_sector_pct:
                portfolio.reduce_sector(sector, self.max_sector_pct)
```

## 结语

风险管理不是"限制收益"，而是"保证生存"。在量化交易中：

1. **止损是纪律**：不执行止损的人，迟早被市场淘汰
2. **仓位是艺术**：太激进会爆仓，太保守跑不赢通胀
3. **回撤是试金石**：控制不了回撤，再高的收益也只是纸上富贵

**最后的话：**

市场永远比我们聪明。无论回测多完美，实盘总会有意外。唯有严格的风控，才能让我们在市场的风浪中活下去，并最终获利。

记住：**this time is never different**（这次不一样）。人性的贪婪和恐惧从未改变，风险管理的原则也永远不会过时。

---

*下一篇我们将讨论量化交易的数据获取：Tushare、AkShare、Baostock实战指南。*
