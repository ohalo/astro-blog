---
title: "配对交易与协整分析"
description: "深入理解配对交易策略的核心原理，掌握协整检验方法，学习如何构建稳健的市场中性套利策略。"
date: "2026-06-18"
tags: ["配对交易", "协整分析", "统计套利", "市场中性"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

配对交易（Pairs Trading）是最经典的市场中性策略之一，起源于摩根士丹利在1980年代的研究。该策略通过寻找具有长期均衡关系的股票对，在价格偏离时建立多空组合，等待均值回归获取收益。本文将系统介绍配对交易的理论基础、协整检验方法及实战应用。

## 配对交易的核心逻辑

配对交易基于一个简单而强大的思想：**相关性强且具备协整关系的两只股票，其价格比率在长期会维持稳定，短期偏离后终将回归**。

### 基本流程

1. **标的筛选**：寻找基本面相似、历史价格走势高度相关的股票对
2. **协整检验**：验证两只股票价格是否存在长期均衡关系
3. **信号生成**：当价格比偏离历史均值时触发交易信号
4. **风险管理**：设置止损、控制仓位、管理交易成本

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf

# 示例：分析两只股票的协整关系
def analyze_pair_cointegration(stock1, stock2, start_date, end_date):
    """
    分析股票对的协整关系
    
    Parameters:
    -----------
    stock1, stock2: str, 股票代码
    start_date, end_date: str, 日期范围
    
    Returns:
    --------
    results: dict, 协整检验结果
    """
    # 下载数据
    try:
        data1 = yf.download(stock1, start=start_date, end=end_date, progress=False)['Adj Close']
        data2 = yf.download(stock2, start=start_date, end=end_date, progress=False)['Adj Close']
        
        # 对齐数据
        df = pd.DataFrame({stock1: data1, stock2: data2}).dropna()
        
        if len(df) < 100:
            return {"error": "数据不足"}
        
        # 1. 计算相关性
        correlation = df[stock1].corr(df[stock2])
        
        # 2. 协整检验
        coint_stat, p_value, crit_values = coint(df[stock1], df[stock2])
        
        # 3. 计算对冲比例（OLS回归）
        model = OLS(df[stock1], df[stock2]).fit()
        hedge_ratio = model.params[0]
        
        # 4. 计算残差（价差）
        spread = df[stock1] - hedge_ratio * df[stock2]
        
        # 5. 单位根检验（ADF检验）
        adf_stat, adf_p_value, _, _, adf_crit_values, _ = adfuller(spread)
        
        # 6. 计算Z分数
        z_score = (spread - spread.mean()) / spread.std()
        
        results = {
            'correlation': correlation,
            'coint_stat': coint_stat,
            'coint_p_value': p_value,
            'crit_values': crit_values,
            'hedge_ratio': hedge_ratio,
            'adf_stat': adf_stat,
            'adf_p_value': adf_p_value,
            'adf_crit_values': adf_crit_values,
            'spread_mean': spread.mean(),
            'spread_std': spread.std(),
            'current_z_score': z_score.iloc[-1]
        }
        
        return results
        
    except Exception as e:
        return {"error": str(e)}

# 示例使用（使用模拟数据，因为yfinance可能需要网络）
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')

# 模拟两只协整股票的价格
n = len(dates)
stock1_price = 100 + np.cumsum(np.random.normal(0, 1, n))
stock2_price = 50 + 0.5 * (stock1_price - 100) + np.cumsum(np.random.normal(0, 0.5, n))

df = pd.DataFrame({'STOCK1': stock1_price, 'STOCK2': stock2_price}, index=dates)

# 协整检验
coint_stat, p_value, crit_values = coint(df['STOCK1'], df['STOCK2'])
print(f"协整检验结果：")
print(f"  检验统计量：{coint_stat:.4f}")
print(f"  P值：{p_value:.4f}")
print(f"  临界值（1%, 5%, 10%）：{crit_values}")
print(f"  是否协整（5%显著性水平）：{p_value < 0.05}")

# 计算对冲比例
model = OLS(df['STOCK1'], df['STOCK2']).fit()
hedge_ratio = model.params[0]
print(f"\n对冲比例（β）：{hedge_ratio:.4f}")

# 计算价差和Z分数
spread = df['STOCK1'] - hedge_ratio * df['STOCK2']
z_score = (spread - spread.mean()) / spread.std()

print(f"\n价差统计：")
print(f"  均值：{spread.mean():.4f}")
print(f"  标准差：{spread.std():.4f}")
print(f"  当前Z分数：{z_score.iloc[-1]:.4f}")
```

## 协整检验方法详解

协整（Cointegration）是配对交易的理论基石。协整关系意味着尽管两个时间序列各自是非平稳的（如随机游走），但它们的某个线性组合是平稳的。

### 1. Engle-Granger两步法

最常用的协整检验方法：

**第一步**：对两个序列进行OLS回归，得到残差序列
```
y_t = α + β * x_t + ε_t
```

**第二步**：对残差序列进行ADF检验，判断其是否平稳

```python
def engle_granger_test(y, x, significance_level=0.05):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y: pd.Series, 因变量
    x: pd.Series, 自变量
    significance_level: float, 显著性水平
    
    Returns:
    --------
    results: dict, 检验结果
    """
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tsa.stattools import adfuller
    
    # 第一步：OLS回归
    model = OLS(y, x).fit()
    residuals = model.resid
    
    # 第二步：ADF检验残差
    adf_stat, adf_p_value, _, _, crit_values, _ = adfuller(residuals)
    
    # 判断是否协整
    is_cointegrated = adf_p_value < significance_level
    
    results = {
        'hedge_ratio': model.params[0],
        'intercept': model.params[1] if len(model.params) > 1 else 0,
        'adf_statistic': adf_stat,
        'adf_p_value': adf_p_value,
        'critical_values': crit_values,
        'is_cointegrated': is_cointegrated,
        'residuals': residuals
    }
    
    return results

# 示例使用
np.random.seed(42)
n = 500
t = np.arange(n)

# 生成协整序列
random_walk = np.cumsum(np.random.normal(0, 1, n))
y = 0.5 * random_walk + np.random.normal(0, 0.5, n)
x = random_walk + np.random.normal(0, 0.5, n)

result = engle_granger_test(pd.Series(y), pd.Series(x))
print("Engle-Granger协整检验：")
print(f"  对冲比例：{result['hedge_ratio']:.4f}")
print(f"  ADF统计量：{result['adf_statistic']:.4f}")
print(f"  P值：{result['adf_p_value']:.4f}")
print(f"  是否协整：{result['is_cointegrated']}")
```

### 2. Johansen检验

适用于多变量协整关系检验，比Engle-Granger方法更稳健：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_cointegration_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    Parameters:
    -----------
    data: pd.DataFrame, 多变量时间序列
    det_order: int, 确定性项顺序（0: 无常数项, 1: 有常数项）
    k_ar_diff: int, 滞后阶数
    
    Returns:
    --------
    results: dict, 检验结果
    """
    # 进行Johansen检验
    joh_result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取结果
    trace_stat = joh_result.lr1  # 迹统计量
    max_stat = joh_result.lr2    # 最大特征值统计量
    trace_crit = joh_result.cvt  # 迹检验临界值
    max_crit = joh_result.cvm    # 最大特征值检验临界值
    
    results = {
        'trace_statistic': trace_stat,
        'max_eigenvalue_statistic': max_stat,
        'trace_critical_values': trace_crit,
        'max_eigenvalue_critical_values': max_crit,
        'num_cointegrating_vectors': (trace_stat > trace_crit[:, 1]).sum()  # 5%显著性水平
    }
    
    return results

# 示例使用
np.random.seed(42)
n = 500

# 生成三个协整序列
random_walk = np.cumsum(np.random.normal(0, 1, n))
series1 = random_walk + np.random.normal(0, 0.5, n)
series2 = 0.8 * random_walk + np.random.normal(0, 0.3, n)
series3 = -0.5 * random_walk + np.random.normal(0, 0.4, n)

data = pd.DataFrame({
    'Series1': series1,
    'Series2': series2,
    'Series3': series3
})

result = johansen_cointegration_test(data)
print("\nJohansen协整检验：")
print(f"  协整向量数量（5%显著性水平）：{result['num_cointegrating_vectors']}")
print(f"  迹统计量：{result['trace_statistic']}")
```

## 配对交易策略构建

### 1. 信号生成

基于Z分数的交易信号：

```python
def generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5):
    """
    基于Z分数生成交易信号
    
    Parameters:
    -----------
    z_score: pd.Series, 价差的Z分数
    entry_threshold: float, 入场阈值
    exit_threshold: float, 出场阈值
    
    Returns:
    --------
    signals: pd.DataFrame, 交易信号
    """
    signals = pd.DataFrame(index=z_score.index)
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 空仓, 1: 多配对, -1: 空配对
    
    # 生成信号
    for i in range(1, len(signals)):
        if signals['z_score'].iloc[i-1] > entry_threshold:
            # Z分数过高，做空配对（做空stock1，做多stock2）
            signals.loc[signals.index[i], 'position'] = -1
        elif signals['z_score'].iloc[i-1] < -entry_threshold:
            # Z分数过低，做多配对（做多stock1，做空stock2）
            signals.loc[signals.index[i], 'position'] = 1
        elif abs(signals['z_score'].iloc[i-1]) < exit_threshold:
            # Z分数回归，平仓
            signals.loc[signals.index[i], 'position'] = 0
        else:
            # 保持之前仓位
            signals.loc[signals.index[i], 'position'] = signals['position'].iloc[i-1]
    
    return signals

# 示例使用
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')

# 模拟Z分数（均值回归过程）
n = len(dates)
z_score = np.zeros(n)
z_score[0] = 0

for i in range(1, n):
    z_score[i] = 0.95 * z_score[i-1] + np.random.normal(0, 1)  # 均值回归

z_score = pd.Series(z_score, index=dates)
signals = generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5)

print("交易信号统计：")
print(f"  做多配对期间：{(signals['position'] == 1).sum()}天")
print(f"  做空配对期间：{(signals['position'] == -1).sum()}天")
print(f"  空仓期间：{(signals['position'] == 0).sum()}天")
```

### 2. 回测框架

完整的配对交易回测系统：

```python
class PairTradingBacktest:
    def __init__(self, stock1_prices, stock2_prices, hedge_ratio, 
                 initial_capital=100000, transaction_cost=0.001):
        """
        初始化配对交易回测
        
        Parameters:
        -----------
        stock1_prices, stock2_prices: pd.Series, 两只股票的价格序列
        hedge_ratio: float, 对冲比例
        initial_capital: float, 初始资金
        transaction_cost: float, 交易成本比例
        """
        self.prices = pd.DataFrame({
            'stock1': stock1_prices,
            'stock2': stock2_prices
        }).dropna()
        self.hedge_ratio = hedge_ratio
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 计算价差和Z分数
        self.prices['spread'] = self.prices['stock1'] - hedge_ratio * self.prices['stock2']
        self.prices['z_score'] = (self.prices['spread'] - self.prices['spread'].rolling(60).mean()) / self.prices['spread'].rolling(60).std()
    
    def run_backtest(self, entry_threshold=2.0, exit_threshold=0.5):
        """
        运行回测
        
        Returns:
        --------
        results: pd.DataFrame, 回测结果
        """
        results = self.prices.copy()
        results['position'] = 0
        results['stock1_shares'] = 0
        results['stock2_shares'] = 0
        results['capital'] = self.initial_capital
        results['portfolio_value'] = self.initial_capital
        
        current_position = 0
        capital = self.initial_capital
        
        for i in range(61, len(results)):  # 跳过前60天（用于计算滚动统计量）
            z_score = results['z_score'].iloc[i-1]
            
            # 信号生成
            if z_score > entry_threshold and current_position != -1:
                # 做空配对
                new_position = -1
            elif z_score < -entry_threshold and current_position != 1:
                # 做多配对
                new_position = 1
            elif abs(z_score) < exit_threshold and current_position != 0:
                # 平仓
                new_position = 0
            else:
                new_position = current_position
            
            # 执行交易
            if new_position != current_position:
                # 计算交易股数
                stock1_shares = int(capital / (results['stock1'].iloc[i] + self.hedge_ratio * results['stock2'].iloc[i]))
                stock2_shares = int(stock1_shares * self.hedge_ratio)
                
                # 计算交易成本
                if current_position != 0:
                    # 平旧仓位
                    cost = abs(current_position) * (stock1_shares * results['stock1'].iloc[i] * self.transaction_cost + 
                                                   stock2_shares * results['stock2'].iloc[i] * self.transaction_cost)
                    capital -= cost
                
                if new_position != 0:
                    # 建新仓位
                    cost = abs(new_position) * (stock1_shares * results['stock1'].iloc[i] * self.transaction_cost + 
                                               stock2_shares * results['stock2'].iloc[i] * self.transaction_cost)
                    capital -= cost
                
                current_position = new_position
                results.loc[results.index[i], 'stock1_shares'] = stock1_shares * new_position
                results.loc[results.index[i], 'stock2_shares'] = stock2_shares * (-new_position)  # 反向
            
            # 更新组合价值
            portfolio_value = capital
            if current_position != 0:
                portfolio_value += (results['stock1_shares'].iloc[i] * results['stock1'].iloc[i] + 
                                  results['stock2_shares'].iloc[i] * results['stock2'].iloc[i])
            
            results.loc[results.index[i], 'position'] = current_position
            results.loc[results.index[i], 'capital'] = capital
            results.loc[results.index[i], 'portfolio_value'] = portfolio_value
        
        return results
    
    def calculate_metrics(self, results):
        """
        计算策略表现指标
        """
        returns = results['portfolio_value'].pct_change()
        
        total_return = (results['portfolio_value'].iloc[-1] / self.initial_capital - 1) * 100
        annual_return = (1 + total_return/100) ** (252/len(results)) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = ((results['portfolio_value'] / results['portfolio_value'].cummax()) - 1).min() * 100
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return * 100,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': abs(max_drawdown),
            'num_trades': (results['position'] != results['position'].shift(1)).sum() // 2
        }
        
        return metrics

# 示例使用
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')

# 模拟两只协整股票
n = len(dates)
stock1 = 100 + np.cumsum(np.random.normal(0.0005, 0.02, n))
stock2 = 50 + 0.5 * (stock1 - 100) + np.cumsum(np.random.normal(0, 0.01, n))
stock1 = pd.Series(stock1, index=dates)
stock2 = pd.Series(stock2, index=dates)

# 运行回测
backtest = PairTradingBacktest(stock1, stock2, hedge_ratio=0.5)
results = backtest.run_backtest(entry_threshold=2.0, exit_threshold=0.5)
metrics = backtest.calculate_metrics(results)

print("\n配对交易回测结果：")
for k, v in metrics.items():
    if k in ['total_return', 'annual_return', 'max_drawdown']:
        print(f"  {k}: {v:.2f}%")
    elif k == 'sharpe_ratio':
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")
```

## 风险管理与实战要点

### 1. 止损策略

配对交易并非无风险，需要严格的风险管理：

- **时间止损**：持仓超过一定期限强制平仓
- **价格止损**：价差突破历史极值止损
- **波动率止损**：市场波动加剧时减小仓位

```python
def add_risk_management(results, max_holding_days=20, max_loss_pct=5.0):
    """
    添加风险管理规则
    
    Parameters:
    -----------
    results: pd.DataFrame, 回测结果
    max_holding_days: int, 最大持仓天数
    max_loss_pct: float, 最大亏损百分比
    
    Returns:
    --------
    results_with_rm: pd.DataFrame, 添加风险管理后的结果
    """
    results_with_rm = results.copy()
    results_with_rm['holding_days'] = 0
    results_with_rm['stopped_out'] = False
    
    current_position = 0
    entry_idx = None
    entry_portfolio_value = None
    
    for i in range(len(results_with_rm)):
        if results_with_rm['position'].iloc[i] != 0 and current_position == 0:
            # 新建仓位
            current_position = results_with_rm['position'].iloc[i]
            entry_idx = i
            entry_portfolio_value = results_with_rm['portfolio_value'].iloc[i]
        
        elif results_with_rm['position'].iloc[i] != 0 and current_position != 0:
            # 持有仓位
            holding_days = i - entry_idx
            current_return = (results_with_rm['portfolio_value'].iloc[i] / entry_portfolio_value - 1) * 100
            
            # 检查止损条件
            if holding_days > max_holding_days:
                # 时间止损
                results_with_rm.loc[results_with_rm.index[i], 'stopped_out'] = True
                results_with_rm.loc[results_with_rm.index[i], 'position'] = 0
                current_position = 0
            
            elif current_return < -max_loss_pct:
                # 亏损止损
                results_with_rm.loc[results_with_rm.index[i], 'stopped_out'] = True
                results_with_rm.loc[results_with_rm.index[i], 'position'] = 0
                current_position = 0
        
        elif results_with_rm['position'].iloc[i] == 0 and current_position != 0:
            # 平仓
            current_position = 0
    
    return results_with_rm

# 示例使用
results_with_rm = add_risk_management(results, max_holding_days=20, max_loss_pct=5.0)
num_stopped_out = results_with_rm['stopped_out'].sum()
print(f"\n风险管理触发次数：{num_stopped_out}")
```

### 2. 实战注意事项

- **标的筛选**：优先选择同行业、相似基本面的股票
- **流动性要求**：确保两只股票都有足够的流动性
- **市场状态**：在趋势性强的市场中，配对交易表现可能较差
- **成本敏感**：频繁交易会侵蚀收益，需优化执行算法

## 结论

配对交易是一种逻辑清晰、风险可控的量化策略。成功的关键在于：

1. **严格的统计检验**：确保标的确实存在协整关系
2. **合理的参数设置**：入场、出场阈值需要根据市场特征调整
3. **完善的风险管理**：设置多重止损机制
4. **持续的策略监控**：市场结构变化可能导致策略失效

对于希望进入量化交易领域的投资者，配对交易是一个理想的起点——它既有坚实的理论基础，又相对容易实现和监控。

---

**参考文献**：
1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley.
