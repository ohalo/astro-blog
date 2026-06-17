---
title: "统计套利：均值回归策略"
description: "深入探讨统计套利的核心原理与均值回归策略的实战应用，从协整检验到配对交易，构建系统化的市场中性策略。"
pubDate: 2026-06-17
tags: ["统计套利", "均值回归", "配对交易", "协整分析", "市场中性"]
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资的重要分支，通过统计方法识别资产价格的临时性偏离，利用均值回归特性获取稳定收益。本文将系统介绍统计套利的理论基础、方法论框架以及实战策略。

![价差分析](/images/statistical-arbitrage-mean-reversion/spread_analysis.png)

*图1：配对股票的价差序列及均值回归带*

## 统计套利的核心原理

### 均值回归的理论基础

均值回归是金融市场的重要特征。研究表明，大多数金融资产的价格序列都呈现出向长期均衡水平回归的趋势。这一现象的理论基础包括：

1. **基本面锚定**：资产价格最终由基本面价值决定
2. **套利机制**：价格偏离会吸引套利者，推动价格回归
3. **投资者行为偏差**：过度反应和反转效应
4. **流动性提供**：做市商在价格偏离时提供流动性

### 统计套利 vs 传统套利

| 维度 | 传统套利 | 统计套利 |
|------|---------|---------|
| 利润确定性 | 几乎确定 | 统计意义上显著 |
| 持有期 | 很短（秒级到天） | 中等（天到月） |
| 所需资本 | 较大 | 灵活 |
| 风险特征 | 极低 | 低到中等 |
| 策略容量 | 有限 | 较大 |

## 配对交易：统计套利的经典范式

### 配对选择的标准流程

配对交易的核心是找到具有长期协整关系的一对资产。标准流程如下：

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf

class PairsSelection:
    """
    配对选择框架
    """
    def __init__(self, universe, start_date, end_date):
        self.universe = universe
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = self.download_data()
        
    def download_data(self):
        """
        下载价格数据
        """
        data = {}
        for ticker in self.universe:
            try:
                stock = yf.Ticker(ticker)
                data[ticker] = stock.history(
                    start=self.start_date, 
                    end=self.end_date
                )['Close']
            except Exception as e:
                print(f"Error downloading {ticker}: {e}")
                continue
        
        return pd.DataFrame(data)
    
    def test_cointegration(self, price1, price2, significance=0.05):
        """
        协整检验
        """
        # Engle-Granger 两步法
        # 第一步：OLS回归
        X = sm.add_constant(price2)
        model = sm.OLS(price1, X).fit()
        residuals = model.resid
        
        # 第二步：ADF检验残差
        adf_result = adfuller(residuals, autolag='AIC')
        
        # 使用coint函数进行正式检验
        coint_stat, p_value, crit_values = coint(price1, price2)
        
        is_cointegrated = p_value < significance
        
        return {
            'is_cointegrated': is_cointegrated,
            'coint_stat': coint_stat,
            'p_value': p_value,
            'hedge_ratio': model.params[1],
            'intercept': model.params[0],
            'residuals': residuals
        }
    
    def calculate_half_life(self, residuals):
        """
        计算均值回归的半衰期
        """
        # 对残差进行AR(1)建模
        lagged_residuals = residuals.shift(1).dropna()
        current_residuals = residuals[1:]
        
        model = sm.OLS(current_residuals, sm.add_constant(lagged_residuals)).fit()
        rho = model.params[1]
        
        # 半衰期计算
        half_life = -np.log(2) / np.log(rho)
        
        return half_life
    
    def screen_pairs(self, max_half_life=60, min_half_life=5):
        """
        筛选有效配对
        """
        n = len(self.universe)
        valid_pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                ticker1 = self.universe[i]
                ticker2 = self.universe[j]
                
                price1 = self.price_data[ticker1].dropna()
                price2 = self.price_data[ticker2].dropna()
                
                # 对齐数据
                aligned = pd.concat([price1, price2], axis=1).dropna()
                
                if len(aligned) < 252:  # 至少需要一年数据
                    continue
                
                # 协整检验
                coint_result = self.test_cointegration(
                    aligned.iloc[:, 0], 
                    aligned.iloc[:, 1]
                )
                
                if coint_result['is_cointegrated']:
                    # 计算半衰期
                    half_life = self.calculate_half_life(
                        coint_result['residuals']
                    )
                    
                    # 筛选半衰期合理的配对
                    if min_half_life <= half_life <= max_half_life:
                        valid_pairs.append({
                            'pair': (ticker1, ticker2),
                            'half_life': half_life,
                            'hedge_ratio': coint_result['hedge_ratio'],
                            'p_value': coint_result['p_value']
                        })
        
        return valid_pairs
```

![协整检验](/images/statistical-arbitrage-mean-reversion/cointegration_test.png)

*图2：协整检验可视化 - 残差序列应为平稳序列*

### 交易信号的构建

基于协整关系的交易信号构建：

```python
class MeanReversionSignal:
    """
    均值回归信号生成器
    """
    def __init__(self, lookback=63, entry_zscore=2.0, exit_zscore=0.5):
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        
    def calculate_spread(self, price1, price2, hedge_ratio, intercept):
        """
        计算价差（残差）
        """
        spread = price1 - (hedge_ratio * price2 + intercept)
        return spread
    
    def calculate_zscore(self, spread):
        """
        计算价差的Z-score
        """
        mean = spread.rolling(window=self.lookback).mean()
        std = spread.rolling(window=self.lookback).std()
        
        zscore = (spread - mean) / std
        
        return zscore
    
    def generate_signals(self, price1, price2, hedge_ratio, intercept):
        """
        生成交易信号
        """
        spread = self.calculate_spread(price1, price2, hedge_ratio, intercept)
        zscore = self.calculate_zscore(spread)
        
        signals = pd.DataFrame(index=spread.index)
        signals['zscore'] = zscore
        signals['spread'] = spread
        
        # 入场信号
        signals['long_entry'] = zscore < -self.entry_zscore
        signals['short_entry'] = zscore > self.entry_zscore
        
        # 出场信号
        signals['exit_long'] = zscore >= -self.exit_zscore
        signals['exit_short'] = zscore <= self.exit_zscore
        
        # 持仓信号
        signals['long_position'] = 0
        signals['short_position'] = 0
        
        # 填充持仓
        for i in range(1, len(signals)):
            if signals['long_entry'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('long_position')] = 1
            elif signals['short_entry'].iloc[i]:
                signals.iloc[i, signals.columns.get_loc('short_position')] = -1
            elif signals['exit_long'].iloc[i] and signals['long_position'].iloc[i-1] == 1:
                signals.iloc[i, signals.columns.get_loc('long_position')] = 0
            elif signals['exit_short'].iloc[i] and signals['short_position'].iloc[i-1] == -1:
                signals.iloc[i, signals.columns.get_loc('short_position')] = 0
            else:
                signals.iloc[i, signals.columns.get_loc('long_position')] = \
                    signals['long_position'].iloc[i-1]
                signals.iloc[i, signals.columns.get_loc('short_position')] = \
                    signals['short_position'].iloc[i-1]
        
        return signals
```

## 实战案例：A股市场配对交易

### 数据准备与配对筛选

```python
def implement_pairs_trading_a_share():
    """
    A股市场配对交易实现
    """
    import akshare as ak
    
    # 选择同一行业的股票
    universe = ['600036.SH', '601398.SH', '601939.SH', '601288.SH',
                '600016.SH', '601166.SH']  # 银行股示例
    
    # 下载数据
    price_data = pd.DataFrame()
    
    for ticker in universe:
        try:
            data = ak.stock_zh_a_hist(
                symbol=ticker,
                period="daily",
                start_date="20200101",
                end_date="20250617"
            )
            price_data[ticker] = data['收盘']
        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    # 配对筛选
    selector = PairsSelection(universe, '2020-01-01', '2025-06-17')
    selector.price_data = price_data
    
    valid_pairs = selector.screen_pairs()
    
    print(f"找到 {len(valid_pairs)} 个有效配对")
    for pair_info in valid_pairs[:5]:
        print(f"配对: {pair_info['pair']}, 半衰期: {pair_info['half_life']:.2f} 天")
    
    return valid_pairs, price_data
```

### 回测框架

```python
class PairsTradingBacktest:
    """
    配对交易回测框架
    """
    def __init__(self, price1, price2, signals, initial_capital=1e6):
        self.price1 = price1
        self.price2 = price2
        self.signals = signals
        self.initial_capital = initial_capital
        
    def run_backtest(self, transaction_cost=0.001):
        """
        执行回测
        """
        portfolio_value = self.initial_capital
        portfolio = pd.DataFrame(index=self.signals.index)
        
        portfolio['capital'] = self.initial_capital
        portfolio['position_value'] = 0
        portfolio['cash'] = self.initial_capital
        
        # 持仓记录
        shares1 = 0
        shares2 = 0
        
        for i in range(1, len(self.signals)):
            date = self.signals.index[i]
            
            # 计算当前持仓价值
            if shares1 != 0:
                position_value = shares1 * self.price1.iloc[i] + \
                               shares2 * self.price2.iloc[i]
            else:
                position_value = 0
            
            # 交易信号执行
            if self.signals['long_entry'].iloc[i] and shares1 == 0:
                # 做多价差（做多股票1，做空股票2）
                capital_to_use = portfolio['cash'].iloc[i-1] * 0.5
                shares1 = int(capital_to_use / self.price1.iloc[i])
                shares2 = -int(capital_to_use / self.price2.iloc[i] * 
                              self.signals.get('hedge_ratio', 1))
                
                # 扣除交易成本
                cost = (abs(shares1) * self.price1.iloc[i] + 
                       abs(shares2) * self.price2.iloc[i]) * transaction_cost
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1] - cost
                
            elif self.signals['short_entry'].iloc[i] and shares1 == 0:
                # 做空价差（做空股票1，做多股票2）
                capital_to_use = portfolio['cash'].iloc[i-1] * 0.5
                shares1 = -int(capital_to_use / self.price1.iloc[i])
                shares2 = int(capital_to_use / self.price2.iloc[i] * 
                             self.signals.get('hedge_ratio', 1))
                
                # 扣除交易成本
                cost = (abs(shares1) * self.price1.iloc[i] + 
                       abs(shares2) * self.price2.iloc[i]) * transaction_cost
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1] - cost
                
            elif (self.signals['exit_long'].iloc[i] or 
                  self.signals['exit_short'].iloc[i]) and shares1 != 0:
                # 平仓
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1] + \
                                              position_value
                
                # 扣除交易成本
                cost = (abs(shares1) * self.price1.iloc[i] + 
                       abs(shares2) * self.price2.iloc[i]) * transaction_cost
                portfolio.loc[date, 'cash'] -= cost
                
                shares1 = 0
                shares2 = 0
            
            # 更新组合价值
            portfolio.loc[date, 'position_value'] = position_value
            portfolio.loc[date, 'capital'] = portfolio.loc[date, 'cash'] + \
                                            position_value
        
        return portfolio
    
    def calculate_performance(self, portfolio):
        """
        计算绩效指标
        """
        returns = portfolio['capital'].pct_change()
        
        metrics = {
            'total_return': (portfolio['capital'].iloc[-1] / 
                            self.initial_capital - 1) * 100,
            'annual_return': returns.mean() * 252 * 100,
            'volatility': returns.std() * np.sqrt(252) * 100,
            'sharpe_ratio': returns.mean() / returns.std() * np.sqrt(252),
            'max_drawdown': self.calculate_max_drawdown(portfolio['capital']),
            'win_rate': (returns > 0).sum() / len(returns)
        }
        
        return metrics
    
    def calculate_max_drawdown(self, capital_series):
        """
        计算最大回撤
        """
        cumulative_max = capital_series.cummax()
        drawdown = (capital_series - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min() * 100
        
        return max_drawdown
```

## 风险管理与策略优化

### 关键风险点

1. **协整关系破裂**：长期均衡关系可能失效
2. **价差持续扩大**：均值回归未如期发生
3. **流动性风险**：配对中某一资产流动性骤降
4. **行业冲击**：共同风险因子导致配对失效

### 风险控制措施

```python
def implement_risk_controls(portfolio, signals, max_holding_period=20,
                           stop_loss_zscore=3.0):
    """
    实施风险控制措施
    """
    risk_adjusted_signals = signals.copy()
    
    # 1. 最大持仓期限控制
    holding_period = 0
    for i in range(len(signals)):
        if (signals['long_position'].iloc[i] != 0 or 
            signals['short_position'].iloc[i] != 0):
            holding_period += 1
            if holding_period > max_holding_period:
                # 强制平仓
                risk_adjusted_signals.iloc[i, 
                    risk_adjusted_signals.columns.get_loc('exit_long')] = True
                risk_adjusted_signals.iloc[i, 
                    risk_adjusted_signals.columns.get_loc('exit_short')] = True
                holding_period = 0
        else:
            holding_period = 0
    
    # 2. 止损控制（基于Z-score）
    for i in range(len(signals)):
        if abs(signals['zscore'].iloc[i]) > stop_loss_zscore:
            # Z-score极端值，强制平仓
            risk_adjusted_signals.iloc[i, 
                risk_adjusted_signals.columns.get_loc('exit_long')] = True
            risk_adjusted_signals.iloc[i, 
                risk_adjusted_signals.columns.get_loc('exit_short')] = True
    
    # 3. 波动率调整仓位
    volatility = signals['spread'].rolling(window=20).std()
    max_volatility = volatility.quantile(0.95)
    
    position_scaling = np.where(volatility > max_volatility, 0.5, 1.0)
    risk_adjusted_signals['position_scaling'] = position_scaling
    
    return risk_adjusted_signals
```

## 绩效评估与实战建议

### 多维度评估体系

统计套利策略的评估应包括：

1. **收益指标**：年化收益、累计收益、风险调整收益
2. **风险指标**：最大回撤、波动率、VaR
3. **交易指标**：胜率、盈亏比、换手率
4. **市场中性验证**：Alpha、Beta、因子暴露

```python
def comprehensive_evaluation(portfolio_returns, benchmark_returns, 
                           factor_returns):
    """
    综合绩效评估
    """
    from sklearn.linear_model import LinearRegression
    
    # 基础指标
    basic_metrics = {
        'annual_return': portfolio_returns.mean() * 252 * 100,
        'volatility': portfolio_returns.std() * np.sqrt(252) * 100,
        'sharpe': portfolio_returns.mean() / portfolio_returns.std() * 
                  np.sqrt(252),
        'max_dd': calculate_max_drawdown(portfolio_returns)
    }
    
    # 市场中性检验
    model = LinearRegression()
    model.fit(factor_returns, portfolio_returns)
    
    neutrality_metrics = {
        'alpha': model.intercept_ * 252 * 100,
        'beta': model.coef_[0],
        'r_squared': model.score(factor_returns, portfolio_returns),
        'residual_risk': np.std(model.resid_) * np.sqrt(252) * 100
    }
    
    # 交易效率
    turnover = calculate_turnover(portfolio_returns)
    transaction_cost = calculate_transaction_cost(portfolio_returns)
    
    efficiency_metrics = {
        'turnover': turnover,
        'transaction_cost_bps': transaction_cost * 10000,
        'net_sharpe': (basic_metrics['sharpe'] - 
                       transaction_cost / basic_metrics['volatility'] * 252)
    }
    
    return {
        'basic': basic_metrics,
        'neutrality': neutrality_metrics,
        'efficiency': efficiency_metrics
    }
```

## 结论与展望

统计套利和均值回归策略为量化投资提供了获取稳定Alpha的重要途径。成功实施的关键在于：

1. **严谨的配对筛选**：协整检验、半衰期分析、行业逻辑验证
2. **精细的信号设计**：Z-score阈值、动态调参、机器学习增强
3. **完善的风险控制**：持仓期限、止损机制、波动率调整
4. **持续的策略迭代**：市场结构变化监测、参数自适应

未来发展方向：

- **高频统计套利**：利用分钟级、秒级数据捕捉短期偏离
- **跨资产类别**：股票-期货、跨市场、跨境配对
- **机器学习增强**：深度学习预测价差收敛概率
- **ESG整合**：将ESG因子纳入配对选择框架

---

*本文代码示例仅供参考，实际应用时需结合具体数据和市场环境进行调整。统计套利策略存在模型风险，实盘前务必充分回测和验证。*
