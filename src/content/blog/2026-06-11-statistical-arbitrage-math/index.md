---
title: "统计套利的数学原理：从协整到均值回归的量化实现"
publishDate: '2026-06-11'
description: "统计套利的数学原理：从协整到均值回归的量化实现 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言：统计套利的核心逻辑

统计套利（Statistical Arbitrage）是量化交易的重要分支，其核心思想是：**利用相关资产的暂时性价格偏离，通过做多低估资产、做空高估资产，等待价格回归均衡来获取收益**。

本文将深入探讨统计套利的数学基础：从协整检验到均值回归建模，再到实际交易系统设计。

## 一、配对交易的理论基础

### 1.1 平稳性与协整

#### 平稳性检验（ADF检验）

时间序列平稳性检验是配对交易的第一步：

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    H0: 序列有单位根（非平稳）
    H1: 序列平稳
    """
    result = adfuller(series, autolag='AIC')
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print(f"{title} 是平稳的 (拒绝H0)")
        return True
    else:
        print(f"{title} 是非平稳的 (接受H0)")
        return False
```

#### 协整检验（Engle-Granger两步法）

当两个非平稳序列的线性组合是平稳的，则称它们协整：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS

def engle_granger_test(y, x, alpha=0.05):
    """
    Engle-Granger协整检验
    """
    # 第一步：OLS回归
    model = OLS(y, sm.add_constant(x)).fit()
    residual = model.resid
    
    # 第二步：对残差进行ADF检验
    adf_stat, p_value, *_ = adfuller(residual)
    
    # 临界值（MacKinnon近似）
    critical_values = {
        0.01: -3.96,
        0.05: -3.37,
        0.10: -3.07
    }
    
    is_cointegrated = adf_stat < critical_values[alpha]
    
    return {
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'hedge_ratio': model.params[1],
        'residual': residual,
        'is_cointegrated': is_cointegrated
    }
```

### 1.2 配对选择的量化指标

#### 距离法（Distance Approach）

```python
def calculate_ssd(spread, window=252):
    """
    计算标准化价差的距离（Sum of Squared Deviations）
    """
    normalized_spread = (spread - spread.mean()) / spread.std()
    ssd = (normalized_spread ** 2).sum()
    return ssd

def find_cointegrated_pairs(tickers, price_data):
    """
    寻找协整配对
    """
    n = len(tickers)
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = price_data[tickers[i]]
            S2 = price_data[tickers[j]]
            
            result = engle_granger_test(S1, S2)
            
            score_matrix[i, j] = result['adf_statistic']
            pvalue_matrix[i, j] = result['p_value']
            
            if result['p_value'] < 0.05:
                keys.append((tickers[i], tickers[j], 
                           result['hedge_ratio'], result['adf_statistic']))
    
    return keys, score_matrix, pvalue_matrix
```

## 二、均值回归建模

### 2.1 Ornstein-Uhlenbeck过程

配对交易的价差可以用OU过程建模：

```
dX_t = θ(μ - X_t)dt + σdW_t
```

其中：
- θ：均值回归速度
- μ：长期均值
- σ：波动率
- W_t：维纳过程

```python
from scipy.optimize import minimize

def ou_calibrate(spread, dt=1/252):
    """
    校准OU过程参数
    """
    X = spread.values
    dX = np.diff(X)
    X_lag = X[:-1]
    
    # OLS回归：dX = a + b*X_lag + error
    X_design = sm.add_constant(X_lag)
    model = OLS(dX, X_design).fit()
    
    a = model.params[0]
    b = model.params[1]
    
    # 转换为OU参数
    theta = -b / dt          # 均值回归速度
    mu = -a / b              # 长期均值
    sigma = np.std(model.resid) / np.sqrt(dt)  # 波动率
    
    # 半衰期
    half_life = np.log(2) / theta
    
    return {
        'theta': theta,
        'mu': mu,
        'sigma': sigma,
        'half_life': half_life,
        'residual_std': np.std(model.resid)
    }
```

### 2.2 交易信号生成

基于OU模型的z-score策略：

```python
def generate_trading_signals(spread, ou_params, entry_z=2.0, exit_z=0.5):
    """
    基于z-score的交易信号
    """
    mu = ou_params['mu']
    sigma = ou_params['sigma']
    
    # 计算z-score
    z_score = (spread - mu) / sigma
    
    # 生成信号
    signals = pd.Series(index=spread.index, dtype=int)
    signals[:] = 0  # 0: 无仓位
    
    position = 0  # 当前仓位：-1:空头, 0:无, 1:多头
    
    for i in range(len(z_score)):
        if position == 0:  # 无仓位
            if z_score.iloc[i] > entry_z:
                position = -1  # 做空价差
                signals.iloc[i] = -1
            elif z_score.iloc[i] < -entry_z:
                position = 1   # 做多价差
                signals.iloc[i] = 1
        else:  # 有仓位
            if abs(z_score.iloc[i]) < exit_z:
                position = 0   # 平仓
                signals.iloc[i] = 0
    
    return signals, z_score
```

## 三、风险管理与组合构建

### 3.1  Bollinger Band动态阈值

```python
def dynamic_bollinger_bands(spread, window=20, num_std=2):
    """
    动态布林带
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    upper_band = rolling_mean + num_std * rolling_std
    lower_band = rolling_mean - num_std * rolling_std
    
    return upper_band, lower_band, rolling_mean
```

### 3.2 组合层面的风险管理

```python
def portfolio_risk_management(positions, max_leverage=2.0, max_correlation=0.7):
    """
    组合层面风险管理
    """
    # 计算相关系数矩阵
    returns = positions.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # 检查相关性
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            if abs(corr_matrix.iloc[i, j]) > max_correlation:
                high_corr_pairs.append((
                    corr_matrix.columns[i],
                    corr_matrix.columns[j],
                    corr_matrix.iloc[i, j]
                ))
    
    # 计算组合杠杆
    total_exposure = positions.abs().sum(axis=1)
    leverage = total_exposure / positions.sum(axis=1).abs()
    
    # 去杠杆
    if leverage.max() > max_leverage:
        scale_factor = max_leverage / leverage.max()
        positions = positions * scale_factor
    
    return positions, high_corr_pairs
```

## 四、中国A股市场的实证研究

### 4.1 数据预处理

```python
def preprocess_a_stock_data(stock_codes, start_date, end_date):
    """
    A股数据预处理
    """
    import tushare as ts
    
    # 获取日线数据
    all_data = {}
    for code in stock_codes:
        df = ts.get_k_data(code, start=start_date, end=end_date)
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        all_data[code] = df['close']
    
    price_df = pd.DataFrame(all_data)
    
    # 处理停牌
    price_df = price_df.dropna(thresh=len(price_df)*0.9, axis=1)  # 删除停牌太多的股票
    price_df = price_df.fillna(method='ffill')  # 前向填充
    
    # 移除ST股票
    # ... (根据股票名称判断)
    
    return price_df
```

### 4.2 行业中性配对

```python
def industry_neutral_pairing(stock_data, industry_map):
    """
    行业中性配对交易
    """
    pairs = []
    
    # 按行业分组
    industry_groups = {}
    for stock, industry in industry_map.items():
        if industry not in industry_groups:
            industry_groups[industry] = []
        industry_groups[industry].append(stock)
    
    # 在每个行业内寻找配对
    for industry, stocks in industry_groups.items():
        if len(stocks) >= 2:
            industry_data = stock_data[stocks]
            industry_pairs, _, _ = find_cointegrated_pairs(stocks, industry_data)
            pairs.extend(industry_pairs)
    
    return pairs
```

## 五、回测框架与绩效评估

### 5.1 回测引擎

```python
class PairsTradingBacktest:
    def __init__(self, data, pairs, initial_capital=1000000):
        self.data = data
        self.pairs = pairs
        self.initial_capital = initial_capital
        self.portfolio_value = []
        
    def run_backtest(self, entry_z=2.0, exit_z=0.5, stop_loss_z=3.0):
        """
        执行回测
        """
        capital = self.initial_capital
        positions = {}  # {pair: (signal, entry_price, quantity)}
        
        for date in self.data.index:
            # 更新持仓市值
            portfolio_value_t = capital
            for pair, (signal, entry_spread, qty) in positions.items():
                stock1, stock2 = pair
                current_spread = self.data[stock1].loc[date] - self.data[stock2].loc[date]
                portfolio_value_t += qty * (current_spread - entry_spread)
            
            self.portfolio_value.append(portfolio_value_t)
            
            # 检查止损
            for pair in list(positions.keys()):
                stock1, stock2 = pair
                spread = self.data[stock1].loc[date] - self.data[stock2].loc[date]
                z_score = (spread - spread.mean()) / spread.std()
                
                if abs(z_score) > stop_loss_z:
                    # 止损平仓
                    del positions[pair]
                    capital = portfolio_value_t
            
            # 生成新信号
            for pair in self.pairs:
                if pair not in positions:
                    stock1, stock2 = pair
                    spread = self.data[stock1].loc[date] - self.data[stock2].loc[date]
                    ou_params = ou_calibrate(spread)
                    signals, z_score = generate_trading_signals(spread, ou_params, entry_z, exit_z)
                    
                    if signals.iloc[-1] != 0:
                        qty = capital * 0.1 / spread.iloc[-1]  # 10%资本分配
                        positions[pair] = (signals.iloc[-1], spread.iloc[-1], qty)
        
        return self.calculate_performance()
    
    def calculate_performance(self):
        """
        计算绩效指标
        """
        returns = pd.Series(self.portfolio_value).pct_change()
        
        total_return = (self.portfolio_value[-1] - self.initial_capital) / self.initial_capital
        annual_return = (1 + total_return) ** (252 / len(self.portfolio_value)) - 1
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        max_drawdown = (pd.Series(self.portfolio_value).cummax() - pd.Series(self.portfolio_value)).max() / pd.Series(self.portfolio_value).cummax().max()
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'portfolio_value': self.portfolio_value
        }
```

## 六、实战注意事项

### 6.1 交易成本考虑

```python
def transaction_cost_model(trade_value, commission_rate=0.0003, stamp_tax=0.001):
    """
    A股交易成本模型
    """
    commission = trade_value * commission_rate
    stamp_tax = trade_value * stamp_tax if trade_value > 0 else 0  # 卖出时收印花税
    slippage = trade_value * 0.001  # 假设0.1%滑点
    
    total_cost = commission + stamp_tax + slippage
    return total_cost
```

### 6.2 生存偏差与前瞻偏差

**常见陷阱**：
1. **生存偏差**：使用当前上市的股票回测历史数据，忽略了已退市的股票
2. **前瞻偏差**：使用包含未来信息的数据（如复权因子）
3. **过度优化**：对参数（entry_z, exit_z）进行过度优化

**解决方案**：
- 使用点阵回测（Point-in-Time）数据
- 样本外测试
-  walk-forward分析

## 七、总结与展望

统计套利是一个成熟的量化策略框架，其成功依赖于：

1. **严谨的数学基础**：协整检验、OU过程建模
2. **精细的风险管理**：动态阈值、组合层面风控
3. **执行力**：低延迟交易系统、交易成本控制
4. **持续改进**：市场结构变化时的模型更新

**未来方向**：
- 机器学习增强的配对选择
- 高频统计套利
- 跨市场统计套利（A股-港股套利）

---

*下期预告：另类数据在量化投资中的应用——卫星图像与社交媒体情绪分析*

> **完整代码已开源**：[GitHub链接](#)  
> **数据来源**：Tushare Pro API，回测框架使用Backtrader

![协整关系示意图](/images/2026-06-11-statistical-arbitrage-math/cointegration_diagram.jpg)

*图1：两个协整股票的价差均值回归特性*

![OU过程模拟](/images/2026-06-11-statistical-arbitrage-math/ou_process_simulation.jpg)

*图2：Ornstein-Uhlenbeck过程的均值回归模拟*
