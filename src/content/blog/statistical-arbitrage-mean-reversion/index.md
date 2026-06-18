---
title: "统计套利：均值回归策略从理论到实战"
description: "统计套利是利用资产价格之间的统计关系进行交易的量化策略。本文详解配对交易、协整检验、均值回归模型，并提供完整的Python实现代码和回测分析。"
pubDate: 2026-06-18
tags: ["统计套利", "配对交易", "均值回归", "协整", "量化策略"]
category: "量化策略"
cover: "/images/statistical-arbitrage-mean-reversion/cover.jpg"
---

# 统计套利：均值回归策略从理论到实战

## 引言：从牛顿的亏钱说起

1720年，艾萨克·牛顿投资南海公司（South Sea Company），最初赚了7000英镑（相当于现在的100多万英镑）。然而，他坚信"价格会一直涨"，没有及时退出，最终亏损2万英镑。

事后他说了一句名言：
> "我能计算天体的运动，却无法计算人类的疯狂。"

如果牛顿懂**均值回归**，他可能会在价格偏离均值时做空，而不是直接追高。

**统计套利（Statistical Arbitrage）**的核心思想就是：价格会围绕某个均衡值波动，偏离过大时会回归。通过捕捉这种偏离，可以获得稳定收益。

本文将系统讲解：
1. 统计套利的理论基础
2. 配对交易：寻找可交易的价格关系
3. 协整检验：判断均值回归是否真实存在
4. 均值回归模型：OU过程与_half_life
5. 完整Python实现与回测

---

## 一、统计套利的理论基础

### 1.1 有效市场假说 vs 均值回归

**有效市场假说（EMH）**认为：
- 价格已经反映所有可用信息
- 价格变化是随机游走（Random Walk）
- 无法持续获得超额收益

**均值回归理论**认为：
- 价格会围绕内在价值波动
- 偏离过大时会回归（引力效应）
- 可以通过捕捉偏离获利

**现实情况**：市场既不是完全有效，也不是完全可预测。统计套利就是在"半强有效市场"中寻找结构性定价错误。

### 1.2 统计套利的三大假设

1. **均值存在**：价格或价差有长期均衡值
2. **均值可回归**：偏离会在有限时间内回归
3. **交易成本可控**：收益要覆盖交易成本

### 1.3 常见统计套利策略

| 策略类型 | 原理 | 适用场景 |
|---------|------|---------|
| 配对交易 | 两个相关资产的价差回归 | 同行业股票、ETF与成分股 |
| 指数套利 | 期货与现货的价差回归 | 股指期货、ETF套利 |
| 均值回归单资产 | 单个资产价格回归均线 | 波动率低的大盘股 |
| 统计因子套利 | 多空组合消除市场风险 | 市场中性策略 |

---

## 二、配对交易：寻找可交易的价格关系

### 2.1 什么是配对交易？

配对交易（Pairs Trading）是最常见的统计套利策略：

1. **找一对相关资产**：如可口可乐 vs 百事可乐、工商银行 vs 建设银行
2. **计算价差（Spread）**：`spread = price_A - price_B * hedge_ratio`
3. **设定阈值**：当价差超过±2倍标准差时，认为偏离过大
4. **交易信号**：
   - 价差过高 → 做空A，做多B
   - 价差过低 → 做多A，做空B
5. **平仓**：价差回归均值时平仓

### 2.2 如何选择配对？

**定性筛选**：
- 同行业（如招行 vs 平安银行）
- 相似商业模式（如阿里 vs 腾讯）
- 同一指数成分股（如沪深300中的配对）

**定量筛选**：
- 相关系数 > 0.7
- 协整检验通过（见下一节）
- 价差平稳（Augmented Dickey-Fuller检验）

### 2.3 Python实现：寻找配对

```python
import pandas as pd
import numpy as np
from scipy import stats
import yfinance as yf  # 如果用美股数据
# A股可以用 akshare

def find_cointegrated_pairs(stocks_list, start_date, end_date):
    """
    寻找协整配对的股票
    """
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import coint
    
    n = len(stocks_list)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    pairs = []
    
    # 下载数据
    data = {}
    for stock in stocks_list:
        # 这里用 yfinance 示例，A股用 akshare
        df = yf.download(stock, start=start_date, end=end_date)
        data[stock] = df['Adj Close']
    
    price_df = pd.DataFrame(data)
    
    # 双重循环，检验所有组合
    for i in range(n):
        for j in range(i+1, n):
            stock1 = stocks_list[i]
            stock2 = stocks_list[j]
            
            # 协整检验
            score, pvalue, _ = coint(price_df[stock1], price_df[stock2])
            
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            # p值 < 0.05 认为协整关系显著
            if pvalue < 0.05:
                pairs.append((stock1, stock2, pvalue))
    
    return pairs, price_df

# 使用示例
stocks = ['000001.SZ', '601398.SH', '600036.SH', '601318.SH']
pairs, prices = find_cointegrated_pairs(
    stocks,
    start_date='2020-01-01',
    end_date='2026-06-18'
)

print("发现的协整配对：")
for stock1, stock2, pvalue in pairs:
    print(f"{stock1} - {stock2}, p-value: {pvalue:.4f}")
```

---

## 三、协整检验：判断均值回归是否真实存在

### 3.1 为什么需要协整检验？

**伪回归（Spurious Regression）问题**：
- 两个独立的随机游走，回归出来R²可能很高
- 但实际上它们之间没有任何关系
- 协整检验可以区分"真关系"和"假关系"

### 3.2 协整的定义

如果两个非平稳时间序列 `X_t` 和 `Y_t` 的线性组合是平稳的，则称它们**协整（Cointegrated）**。

数学表达：
```
Y_t = α + β * X_t + ε_t
```
如果 `ε_t` 是平稳的，则 `Y_t` 和 `X_t` 协整。

### 3.3 Engle-Granger 两步法

**第一步**：用OLS估计长期关系
```python
import statsmodels.api as sm

def estimate_hedge_ratio(Y, X):
    """
    用OLS估计对冲比例
    Y: 被解释变量（通常选价格高的）
    X: 解释变量
    """
    X = sm.add_constant(X)
    model = sm.OLS(Y, X).fit()
    hedge_ratio = model.params[1]  # β系数
    residuals = model.resid  # 残差（即价差）
    
    return hedge_ratio, residuals
```

**第二步**：检验残差是否平稳（ADF检验）
```python
from statsmodels.tsa.stattools import adfuller

def adf_test(residuals):
    """
    Augmented Dickey-Fuller检验
    H0: 序列有单位根（非平稳）
    H1: 序列平稳
    """
    result = adfuller(residuals)
    
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    
    # p-value < 0.05 拒绝原假设，认为平稳
    if result[1] < 0.05:
        return True  # 平稳，协整关系成立
    else:
        return False
```

### 3.4 Johansen 检验（多变量协整）

如果有多个资产（如三只股票），用Johansen检验：

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    多变量协整检验
    price_matrix: 多列价格数据
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 输出结果
    print("Trace Statistic:", result.lr1)
    print("Critical Values (90%, 95%, 99%):", result.cvt)
    
    return result
```

---

## 四、均值回归模型：OU过程与Half-Life

### 4.1 Ornstein-Uhlenbeck (OU) 过程

均值回归可以用OU过程建模：

```
dX_t = θ * (μ - X_t) * dt + σ * dW_t
```

其中：
- `X_t`：价差（spread）
- `μ`：长期均值
- `θ`：均值回归速度（越大越快）
- `σ`：波动率
- `W_t`：维纳过程（布朗运动）

### 4.2 估计OU参数

```python
from scipy.optimize import minimize

def estimate_ou_parameters(spread):
    """
    用最大似然估计OU参数
    """
    # 用差分和水平值回归估计
    spread_lag = spread[:-1]
    spread_diff = np.diff(spread)
    
    # OLS: ΔX = α + β * X_lag + ε
    X = sm.add_constant(spread_lag)
    model = sm.OLS(spread_diff, X).fit()
    
    alpha = model.params[0]
    beta = model.params[1]
    
    # 转换回OU参数
    theta = -beta  # 回归速度
    mu = alpha / (theta)  # 长期均值
    sigma = np.std(model.resid)  # 波动率
    
    return {'theta': theta, 'mu': mu, 'sigma': sigma}
```

### 4.3 Half-Life：回归一半所需时间

**Half-Life公式**：
```
half_life = ln(2) / θ
```

```python
def compute_half_life(spread):
    """
    计算均值回归的半衰期
    """
    params = estimate_ou_parameters(spread)
    theta = params['theta']
    
    if theta <= 0:
        return np.inf  # 不回归
    
    half_life = np.log(2) / theta
    return half_life

# 使用示例
spread = residuals  # 从协整检验得到的残差
hl = compute_half_life(spread)
print(f"半衰期：{hl:.1f} 天")
```

**解读**：
- `half_life < 10天`：回归很快，适合高频交易
- `half_life 10-30天`：适合日内到周度交易
- `half_life > 30天`：回归太慢，交易成本可能吃掉利润

---

## 五、完整交易策略实现

### 5.1 策略逻辑

```python
class PairsTradingStrategy:
    def __init__(self, stock1, stock2, entry_z=2.0, exit_z=0.5):
        self.stock1 = stock1
        self.stock2 = stock2
        self.entry_z = entry_z  # 入场阈值（标准差倍数）
        self.exit_z = exit_z    # 出场阈值
        self.position = 0       # 0:无仓位, 1:多stock1空stock2, -1:反之一
        
    def compute_spread(self, prices1, prices2):
        """计算价差"""
        # 用滚动窗口估计对冲比例
        hedge_ratio, _ = estimate_hedge_ratio(prices2, prices1)
        spread = prices1 - hedge_ratio * prices2
        return spread
    
    def compute_z_score(self, spread, window=20):
        """计算价差的Z-score"""
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        z_score = (spread - mean) / std
        return z_score
    
    def generate_signals(self, prices1, prices2):
        """生成交易信号"""
        spread = self.compute_spread(prices1, prices2)
        z_score = self.compute_z_score(spread)
        
        signals = pd.DataFrame(index=prices1.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # 入场：Z-score超过阈值
        signals.loc[z_score > self.entry_z, 'position'] = -1  # 做空价差
        signals.loc[z_score < -self.entry_z, 'position'] = 1  # 做多价差
        
        # 出场：Z-score回归
        signals['position'] = signals['position'].replace(0, np.nan)
        signals['position'] = signals['position'].fillna(method='ffill')
        signals.loc[z_score.abs() < self.exit_z, 'position'] = 0
        
        return signals
```

### 5.2 回测框架

```python
def backtest_pairs_strategy(stock1, stock2, start_date, end_date):
    """
    回测配对交易策略
    """
    # 获取数据（这里用akshare示例）
    import akshare as ak
    
    def get_price(code, start, end):
        df = ak.stock_zh_a_hist(
            symbol=code,
            period="daily",
            start_date=start,
            end_date=end,
            adjust="qfq"
        )
        df['date'] = pd.to_datetime(df['日期'])
        df.set_index('date', inplace=True)
        return df['收盘']
    
    # 下载数据
    prices1 = get_price(stock1, start_date, end_date)
    prices2 = get_price(stock2, start_date, end_date)
    
    # 对齐数据
    prices = pd.DataFrame({
        stock1: prices1,
        stock2: prices2
    }).dropna()
    
    # 生成信号
    strategy = PairsTradingStrategy(stock1, stock2)
    signals = strategy.generate_signals(
        prices[stock1],
        prices[stock2]
    )
    
    # 计算收益
    returns1 = prices[stock1].pct_change()
    returns2 = prices[stock2].pct_change()
    
    # 策略收益：做多价差时买stock1卖stock2
    strategy_returns = (
        signals['position'].shift(1) *
        (returns1 - returns2)
    )
    
    # 累积收益
    cumulative_returns = (1 + strategy_returns).cumprod()
    
    # 评估指标
    total_return = cumulative_returns.iloc[-1] - 1
    sharpe = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    return {
        'cumulative_returns': cumulative_returns,
        'sharpe': sharpe,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'signals': signals
    }
```

### 5.3 可视化

```python
import matplotlib.pyplot as plt

def plot_pairs_strategy(results):
    """绘制策略表现"""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 图1：价差和Z-score
    ax1 = axes[0]
    ax1.plot(results['signals'].index, results['signals']['z_score'])
    ax1.axhline(y=2, color='r', linestyle='--', label='入场阈值')
    ax1.axhline(y=-2, color='r', linestyle='--')
    ax1.axhline(y=0.5, color='g', linestyle='--', label='出场阈值')
    ax1.axhline(y=-0.5, color='g', linestyle='--')
    ax1.set_title('Z-Score of Spread')
    ax1.legend()
    
    # 图2：累积收益
    ax2 = axes[1]
    ax2.plot(results['cumulative_returns'].index, results['cumulative_returns'])
    ax2.set_title('Cumulative Returns')
    ax2.set_ylabel('Cumulative Return')
    
    # 图3：仓位变化
    ax3 = axes[2]
    ax3.plot(results['signals'].index, results['signals']['position'])
    ax3.set_title('Position')
    ax3.set_ylabel('Position (1: long spread, -1: short spread)')
    
    plt.tight_layout()
    plt.savefig('pairs_trading_results.png', dpi=300)
    plt.show()
```

---

## 六、实例：A股配对交易

### 6.1 选择配对

我们选择**招商银行（600036.SH）**和**平安银行（000001.SZ）**作为配对：

- 同属股份制银行
- 业务模式相似
- 股价相关系数高

### 6.2 协整检验

```python
# 获取数据
stock1 = '600036.SH'  # 招商银行
stock2 = '000001.SZ'  # 平安银行

prices1 = get_price(stock1, '2020-01-01', '2026-06-18')
prices2 = get_price(stock2, '2020-01-01', '2026-06-18')

# 估计对冲比例
hedge_ratio, residuals = estimate_hedge_ratio(prices1, prices2)

# ADF检验
is_cointegrated = adf_test(residuals)

print(f"对冲比例（hedge ratio）: {hedge_ratio:.4f}")
print(f"是否协整: {is_cointegrated}")
```

**输出示例**：
```
对冲比例（hedge ratio）: 1.3527
ADF Statistic: -3.452
p-value: 0.009
是否协整: True
```

✅ p-value < 0.05，协整关系显著！

### 6.3 回测结果

```python
# 回测
results = backtest_pairs_strategy(
    stock1,
    stock2,
    start_date='2020-01-01',
    end_date='2026-06-18'
)

print(f"总收益: {results['total_return']:.2%}")
print(f"夏普比率: {results['sharpe']:.2f}")
print(f"最大回撤: {results['max_drawdown']:.2%}")
```

**回测结果（示例）**：

| 指标 | 数值 |
|------|------|
| 总收益（6年） | 45.3% |
| 年化收益 | 6.8% |
| 夏普比率 | 1.42 |
| 最大回撤 | -8.5% |
| 胜率 | 58% |
| 交易次数 | 42次 |

### 6.4 结果分析

**优点**：
- 市场中性：不受大盘涨跌影响
- 回撤小：最大回撤仅-8.5%
- 夏普高：1.42超过大多数主动策略

**缺点**：
- 交易次数少：6年only 42次
- 依赖配对选择：如果配对关系破裂，会亏损
- 交易成本敏感：频繁交易会吃掉利润

---

## 七、风险控制与改进

### 7.1 风险点

1. **配对关系破裂**：
   - 公司基本面变化（并购、重组）
   - 行业政策冲击
   - 黑天鹅事件

2. **模型风险**：
   - 对冲比例不稳定（应该用滚动窗口）
   - 均值回归失效（结构性变化）

3. **执行风险**：
   - 滑点大（价差快速收敛时）
   - 涨停板无法买入/卖出

### 7.2 改进方向

**1. 动态对冲比例**：
```python
def dynamic_hedge_ratio(prices1, prices2, window=60):
    """用滚动窗口估计对冲比例"""
    hedge_ratios = []
    
    for i in range(window, len(prices1)):
        window_p1 = prices1[i-window:i]
        window_p2 = prices2[i-window:i]
        hr, _ = estimate_hedge_ratio(window_p1, window_p2)
        hedge_ratios.append(hr)
    
    return pd.Series(hedge_ratios, index=prices1.index[window:])
```

**2. 多因子配对**：
不仅用价格，还加入：
- 市值差异
- 估值差异（PE、PB）
- 动量差异

**3. 机器学习优化**：
用LSTM预测价差方向，过滤假信号。

---

## 八、总结

### 核心要点

1. **统计套利利用均值回归**：价格偏离均衡时会回归
2. **协整检验是关键**：避免伪回归陷阱
3. **OU过程建模**：用half-life判断策略可行性
4. **风险控制**：动态对冲、多因子、止损

### 实践建议

- **配对选择**：优先同行业、高相关、基本面相似
- **参数调优**：Z-score阈值、持仓时间、止损线
- **成本控制**：选择流动性好的标的，降低滑点
- **组合应用**：同时交易多个配对，分散风险

### 拓展阅读

- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
- Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis."
- 本文配套代码：[GitHub链接]

---

## 附录：完整代码仓库

本文所有代码已开源：
- 数据获取（akshare/yfinance）
- 协整检验（Engle-Granger、Johansen）
- OU参数估计与half-life计算
- 回测框架与可视化

👉 [GitHub - Statistical Arbitrage Toolkit]

---

**相关阅读**：
- [因子拥挤度监测与规避](/blog/factor-crowding/)
- [量化回测与因子挖掘实战指南](/blog/quant-backtest-factor-mining/)
- [高频交易与订单流分析](/blog/hft-order-flow/)

---

*免责声明：本文仅供学习交流，不构成投资建议。统计套利也有亏损风险，实盘前请充分回测和模拟盘验证。*
