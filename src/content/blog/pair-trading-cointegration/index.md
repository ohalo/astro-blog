---
title: "配对交易与协整分析"
description: "从协整理论到实战策略，详细介绍配对交易的完整流程，包括股票对筛选、信号生成、风险管理和Python实现。"
publishDate: 2026-06-18
category: "量化策略"
tags: 
  - 配对交易
  - 协整分析
  - 统计套利
  - 市场中性
  - Python实战
featured: false
---

# 配对交易与协整分析

配对交易（Pairs Trading）是最经典的市场中性策略之一，通过对冲两只高度相关股票的头寸，捕捉价格背离后的收敛收益。本文从协整理论出发，提供完整的配对交易实战框架。

## 配对交易的核心逻辑

### 基本原理

配对交易基于一个简单假设：**具有长期均衡关系的两只股票，短期价格背离后会回归均值**。

策略流程：
1. 筛选具有协整关系的股票对
2. 计算价差（Spread）或Z-Score
3. 当价差偏离历史均值时开仓（做多低估标的，做空高估标的）
4. 价差的回归均值时平仓

### 为什么需要协整？

**伪回归（Spurious Regression）问题：**
- 两只股票的价差可能只是短期相关，没有长期均衡关系
- 用相关系数筛选股票对容易失效（例：2008年雷曼兄弟破产前，很多金融股高度相关）

**协整的优势：**
- 协整关系意味着存在长期均衡
- 即使短期背离，长期必然回归
- 提供统计上的严谨性

## 协整理论基础

### 定义

对于两个非平稳时间序列 $X_t$ 和 $Y_t$，如果存在线性组合 $Z_t = Y_t - \beta X_t$ 是平稳的，则称 $X_t$ 和 $Y_t$ 是协整的。

### Engle-Granger两步法

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

class CointegrationAnalyzer:
    """协整关系分析器"""
    
    def __init__(self, significance_level=0.05):
        """
        初始化
        
        参数:
            significance_level: 显著性水平（默认5%）
        """
        self.alpha = significance_level
        
    def engle_granger_test(self, y, x):
        """
        Engle-Granger两步法检验协整关系
        
        参数:
            y: 因变量（被解释变量）
            x: 自变量
        
        返回:
            result: Dict, 包含回归系数、残差、ADF检验结果
        """
        # 第一步：OLS回归
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        residuals = model.resid
        
        # 第二步：残差平稳性检验（ADF检验）
        adf_result = adfuller(residuals, autolag='AIC')
        
        result = {
            'beta': model.params[1],  # 对冲比率
            'alpha': model.params[0],  # 截距
            'residuals': residuals,
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'is_cointegrated': adf_result[1] < self.alpha
        }
        
        return result
    
    def johansen_test(self, data, det_order=0, k_ar_diff=1):
        """
        Johansen检验（适用于多变量）
        
        参数:
            data: DataFrame, 多只股票价格
            det_order: 确定性项（0: 无截距无趋势, 1: 有截距, 2: 有截距和趋势）
            k_ar_diff: 滞后阶数
        """
        from statsmodels.tsa.johansen import coint_johansen
        
        result = coint_johansen(data, det_order, k_ar_diff)
        
        # 输出结果
        print("特征值和特征向量：")
        print(result.eig)
        print("\n迹统计量（Trace Statistic）：")
        print(result.lr1)
        print("\n最大特征值统计量（Max Eigen Statistic）：")
        print(result.lr2)
        
        return result
    
    def calculate_half_life(self, spread):
        """
        计算价差的半衰期（回归速度）
        
        半衰期越短，均值回归越快
        """
        # 计算价差的一阶差分
        spread_lag = spread.shift(1)
        spread_diff = spread.diff().dropna()
        
        # OLS回归：spread_diff = alpha + beta * spread_lag + error
        X = sm.add_constant(spread_lag.dropna())
        model = OLS(spread_diff, X).fit()
        
        # beta的负数表示回归速度
        beta = model.params[1]
        half_life = -np.log(2) / beta
        
        return half_life
```

## 股票对筛选流程

### 1. 初步筛选

```python
def preliminary_screening(stock_list, start_date, end_date):
    """
    初步筛选潜在配对股票
    
    筛选条件：
    1. 同行业（GICS/申万行业分类）
    2. 相似市值（相差不超过2倍）
    3. 相似波动率
    4. 数据完整性（不超过5%缺失）
    """
    import tushare as ts  # 假设使用tushare获取A股数据
    
    candidates = []
    
    for i, stock1 in enumerate(stock_list):
        for stock2 in stock_list[i+1:]:
            # 1. 检查行业分类
            industry1 = get_industry(stock1)
            industry2 = get_industry(stock2)
            
            if industry1 != industry2:
                continue
            
            # 2. 获取价格和市值数据
            price1 = get_stock_data(stock1, start_date, end_date)
            price2 = get_stock_data(stock2, start_date, end_date)
            
            market_cap1 = get_market_cap(stock1, end_date)
            market_cap2 = get_market_cap(stock2, end_date)
            
            # 3. 市值筛选
            if max(market_cap1, market_cap2) / min(market_cap1, market_cap2) > 2:
                continue
            
            # 4. 波动率筛选
            vol1 = price1.pct_change().std() * np.sqrt(252)
            vol2 = price2.pct_change().std() * np.sqrt(252)
            
            if abs(vol1 - vol2) / min(vol1, vol2) > 0.5:
                continue
            
            # 5. 数据完整性
            if price1.isnull().sum() / len(price1) > 0.05:
                continue
            if price2.isnull().sum() / len(price2) > 0.05:
                continue
            
            candidates.append((stock1, stock2))
    
    return candidates

def get_industry(stock_code):
    """获取股票行业分类（示例）"""
    # 实际中可调用westock-data或tushare
    pass

def get_stock_data(stock_code, start, end):
    """获取股票价格数据"""
    pass

def get_market_cap(stock_code, date):
    """获取股票市值"""
    pass
```

### 2. 协整检验筛选

```python
def cointegration_screening(candidates, price_data, significance=0.05):
    """
    对候选股票对进行协整检验
    
    参数:
        candidates: 候选股票对列表
        price_data: Dict, {stock_code: price_series}
        significance: 显著性水平
    
    返回:
        cointegrated_pairs: 通过协整检验的股票对
    """
    analyzer = CointegrationAnalyzer(significance_level=significance)
    cointegrated_pairs = []
    
    for stock1, stock2 in candidates:
        # 获取价格数据
        price1 = price_data[stock1].dropna()
        price2 = price_data[stock2].dropna()
        
        # 对齐日期
        common_idx = price1.index.intersection(price2.index)
        price1 = price1.loc[common_idx]
        price2 = price2.loc[common_idx]
        
        # 协整检验
        result = analyzer.engle_granger_test(price1, price2)
        
        if result['is_cointegrated']:
            # 计算半衰期
            spread = price1 - result['beta'] * price2
            half_life = analyzer.calculate_half_life(spread)
            
            # 过滤：半衰期在5-60个交易日之间
            if 5 <= half_life <= 60:
                cointegrated_pairs.append({
                    'stock1': stock1,
                    'stock2': stock2,
                    'beta': result['beta'],
                    'half_life': half_life,
                    'p_value': result['p_value']
                })
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x['p_value'])
    
    return cointegrated_pairs
```

### 3. 稳定性检验

```python
def stability_test(stock1, stock2, price_data, window=252):
    """
    滚动窗口检验协整关系稳定性
    
    参数:
        window: 滚动窗口大小（交易日数）
    """
    analyzer = CointegrationAnalyzer()
    
    # 获取完整价格序列
    price1 = price_data[stock1].dropna()
    price2 = price_data[stock2].dropna()
    
    # 对齐日期
    common_idx = price1.index.intersection(price2.index)
    price1 = price1.loc[common_idx]
    price2 = price2.loc[common_idx]
    
    # 滚动检验
    results = []
    
    for start in range(0, len(price1) - window, 20):  # 每20天滚动一次
        end = start + window
        
        sub_price1 = price1.iloc[start:end]
        sub_price2 = price2.iloc[start:end]
        
        # 协整检验
        result = analyzer.engle_granger_test(sub_price1, sub_price2)
        
        results.append({
            'start_date': sub_price1.index[0],
            'end_date': sub_price1.index[-1],
            'p_value': result['p_value'],
            'beta': result['beta'],
            'is_cointegrated': result['is_cointegrated']
        })
    
    # 统计稳定性
    total_windows = len(results)
    cointegrated_windows = sum([r['is_cointegrated'] for r in results])
    stability_ratio = cointegrated_windows / total_windows
    
    print(f"协整关系稳定性：{stability_ratio:.2%}")
    print(f"Beta系数变异系数：{np.std([r['beta'] for r in results]) / np.mean([r['beta'] for r in results]):.4f}")
    
    return results, stability_ratio
```

## 交易信号生成

### 1. 基于Z-Score的信号

![Z-Score交易信号](/images/pair-trading-cointegration/z_score_signals.png)

*图2：配对交易价差Z-Score与交易信号示意图*

```python
class PairTradingSignal:
    """配对交易信号生成器"""
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, stop_loss_threshold=3.0):
        """
        初始化
        
        参数:
            entry_threshold: 入场阈值（Z-Score绝对值）
            exit_threshold: 出场阈值（Z-Score绝对值）
            stop_loss_threshold: 止损阈值
        """
        self.entry_z = entry_threshold
        self.exit_z = exit_threshold
        self.stop_z = stop_loss_threshold
        
    def calculate_spread(self, price1, price2, beta):
        """计算价差"""
        spread = price1 - beta * price2
        return spread
    
    def calculate_z_score(self, spread, window=60):
        """
        计算价差的Z-Score
        
        参数:
            window: 滚动窗口（用于计算均值和标准差）
        """
        mean = spread.rolling(window).mean()
        std = spread.rolling(window).std()
        
        z_score = (spread - mean) / std
        
        return z_score
    
    def generate_signals(self, price1, price2, beta):
        """
        生成交易信号
        
        返回:
            signals: DataFrame, 包含以下列：
                - z_score: Z-Score序列
                - position: 持仓方向（1: 做多价差, -1: 做空价差, 0: 空仓）
                - long_stock1: 是否做多stock1
                - short_stock1: 是否做空stock1
        """
        # 计算价差和Z-Score
        spread = self.calculate_spread(price1, price2, beta)
        z_score = self.calculate_z_score(spread)
        
        # 初始化信号
        signals = pd.DataFrame(index=price1.index)
        signals['z_score'] = z_score
        signals['position'] = 0
        
        # 生成信号
        current_position = 0
        
        for i in range(1, len(signals)):
            z = signals['z_score'].iloc[i]
            
            if current_position == 0:  # 当前空仓
                if z < -self.entry_z:  # 价差低估，做多价差（做多stock1，做空stock2）
                    current_position = 1
                elif z > self.entry_z:  # 价差高估，做空价差（做空stock1，做多stock2）
                    current_position = -1
                    
            elif current_position == 1:  # 当前做多价差
                if z >= -self.exit_z:  # 价差回归，平仓
                    current_position = 0
                elif z < -self.stop_z:  # 止损
                    current_position = 0
                    
            elif current_position == -1:  # 当前做空价差
                if z <= self.exit_z:  # 价差回归，平仓
                    current_position = 0
                elif z > self.stop_z:  # 止损
                    current_position = 0
            
            signals['position'].iloc[i] = current_position
        
        # 拆解持仓
        signals['long_stock1'] = (signals['position'] == 1)
        signals['short_stock1'] = (signals['position'] == -1)
        signals['long_stock2'] = (signals['position'] == -1)  # 做空价差时做多stock2
        signals['short_stock2'] = (signals['position'] == 1)  # 做多价差时做空stock2
        
        return signals
```

### 2. 基于Kalman Filter的动态对冲比率

```python
from pykalman import KalmanFilter

def dynamic_hedge_ratio_kalman(price1, price2):
    """
    使用Kalman Filter估计时变对冲比率
    
    Kalman Filter优势：
    1. 动态调整beta，适应市场结构变化
    2. 提供beta的不确定性估计
    """
    # 准备观测数据
    Y = price1.values.reshape(-1, 1)
    X = price2.values.reshape(-1, 1)
    
    # 初始化Kalman Filter
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=np.expand_dims(X, axis=1),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01
    )
    
    # 状态向量：[alpha, beta]
    state_means, state_covariances = kf.filter(Y)
    
    # 提取beta序列
    beta_dynamic = state_means[:, 1]
    
    return beta_dynamic
```

## 风险管理系统

### 1. 头寸规模管理

```python
class PositionSizer:
    """头寸规模管理器"""
    
    def __init__(self, total_capital, max_position_pct=0.05, max_leverage=2.0):
        """
        初始化
        
        参数:
            total_capital: 总资金
            max_position_pct: 单对最大仓位（占总资金比例）
            max_leverage: 最大杠杆
        """
        self.capital = total_capital
        self.max_pos_pct = max_position_pct
        self.max_lev = max_leverage
        
    def calculate_position_size(self, price1, price2, volatility, risk_budget=0.001):
        """
        基于风险预算的头寸规模
        
        参数:
            price1, price2: 两只股票当前价格
            volatility: 价差波动率（年化）
            risk_budget: 风险预算（单对最大日亏损占总资金比例）
        """
        # 价差
        spread = price1 - price2
        
        # 价差日均波动（绝对金额）
        daily_vol = volatility / np.sqrt(252) * spread
        
        # 基于风险预算的头寸
        max_loss = self.capital * risk_budget
        position_value = max_loss / daily_vol
        
        # 限制最大仓位
        max_position_value = self.capital * self.max_pos_pct
        position_value = min(position_value, max_position_value)
        
        # 计算股数
        shares1 = int(position_value / price1)
        shares2 = int(position_value / price2)
        
        return shares1, shares2
```

### 2. 止损与止盈

```python
def implement_stop_loss(signals, spread, entry_spread, max_loss_pct=0.05):
    """
    实施止损
    
    参数:
        signals: 信号DataFrame
        spread: 当前价差
        entry_spread: 入场时的价差
        max_loss_pct: 最大亏损比例
    """
    # 计算当前亏损
    current_loss = (spread - entry_spread) / entry_spread
    
    # 触发止损
    if abs(current_loss) > max_loss_pct:
        signals['position'] = 0  # 平仓
        print(f"触发止损！亏损：{current_loss:.2%}")
        
    return signals

def implement_profit_taking(signals, spread, entry_spread, profit_target=0.03):
    """
    实施止盈
    
    参数:
        profit_target: 止盈目标（价差回归到均值的距离）
    """
    # 计算当前盈利
    current_profit = (entry_spread - spread) / entry_spread
    
    # 触发止盈
    if current_profit > profit_target:
        signals['position'] = 0  # 平仓
        print(f"触发止盈！盈利：{current_profit:.2%}")
        
    return signals
```

### 3. 配对失效检测

```python
def detect_pair_breakdown(spread, window=60, z_threshold=3):
    """
    检测配对关系破裂
    
    迹象：
    1. 价差波动率突然放大
    2. Z-Score持续突破阈值
    3. 协整关系消失
    """
    # 1. 波动率突变检测
    vol = spread.rolling(window).std()
    vol_z_score = (vol - vol.rolling(window*2).mean()) / vol.rolling(window*2).std()
    
    if abs(vol_z_score.iloc[-1]) > 3:
        print("警告：价差波动率异常！")
        return True
    
    # 2. Z-Score持续极端值
    z_score = (spread - spread.rolling(window).mean()) / spread.rolling(window).std()
    
    if abs(z_score.iloc[-1]) > z_threshold:
        consecutive_extreme = (abs(z_score.iloc[-20:]) > z_threshold).sum()
        
        if consecutive_extreme > 10:  # 过去20天有10天以上极端值
            print("警告：Z-Score持续极端值，配对可能失效！")
            return True
    
    # 3. 滚动协整检验
    if len(spread) > window:
        sub_spread = spread.iloc[-window:]
        stock1_price = price1.iloc[-window:]
        stock2_price = price2.iloc[-window:]
        
        result = engle_granger_test(stock1_price, stock2_price)
        
        if not result['is_cointegrated']:
            print("警告：协整关系消失！")
            return True
    
    return False
```

## 完整策略回测

```python
class PairTradingBacktest:
    """配对交易回测框架"""
    
    def __init__(self, initial_capital=1e6, transaction_cost=0.001):
        """
        初始化
        
        参数:
            initial_capital: 初始资金
            transaction_cost: 交易成本（单边）
        """
        self.capital = initial_capital
        self.cost = transaction_cost
        self.positions = {}
        self.trades = []
        
    def run_backtest(self, stock1, stock2, price_data, beta, signal_generator):
        """
        运行回测
        
        参数:
            stock1, stock2: 股票代码
            price_data: 价格数据DataFrame
            beta: 对冲比率
            signal_generator: 信号生成器对象
        """
        # 获取价格
        price1 = price_data[stock1].dropna()
        price2 = price_data[stock2].dropna()
        
        # 对齐日期
        common_idx = price1.index.intersection(price2.index)
        price1 = price1.loc[common_idx]
        price2 = price2.loc[common_idx]
        
        # 生成信号
        signals = signal_generator.generate_signals(price1, price2, beta)
        
        # 初始化持仓
        holdings = {
            stock1: {'shares': 0, 'cost': 0},
            stock2: {'shares': 0, 'cost': 0}
        }
        
        portfolio_value = []
        current_position = 0
        
        for i, date in enumerate(signals.index):
            # 获取信号
            target_position = signals['position'].loc[date]
            
            # 仓位变化
            if target_position != current_position:
                # 平仓
                if current_position != 0:
                    # 平掉stock1
                    if holdings[stock1]['shares'] != 0:
                        # 计算盈亏
                        sell_value = holdings[stock1]['shares'] * price1.loc[date]
                        self.capital += sell_value
                        transaction_fee = abs(sell_value) * self.cost
                        self.capital -= transaction_fee
                        
                        self.trades.append({
                            'date': date,
                            'action': 'SELL',
                            'stock': stock1,
                            'shares': holdings[stock1]['shares'],
                            'price': price1.loc[date],
                            'value': sell_value,
                            'fee': transaction_fee
                        })
                        
                        holdings[stock1]['shares'] = 0
                    
                    # 平掉stock2
                    if holdings[stock2]['shares'] != 0:
                        sell_value = holdings[stock2]['shares'] * price2.loc[date]
                        self.capital += sell_value
                        transaction_fee = abs(sell_value) * self.cost
                        self.capital -= transaction_fee
                        
                        self.trades.append({
                            'date': date,
                            'action': 'SELL',
                            'stock': stock2,
                            'shares': holdings[stock2]['shares'],
                            'price': price2.loc[date],
                            'value': sell_value,
                            'fee': transaction_fee
                        })
                        
                        holdings[stock2]['shares'] = 0
                
                # 开仓
                if target_position == 1:  # 做多价差
                    # 做多stock1
                    shares1 = int(self.capital * 0.5 / price1.loc[date])
                    cost1 = shares1 * price1.loc[date]
                    
                    holdings[stock1]['shares'] = shares1
                    holdings[stock1]['cost'] = cost1
                    self.capital -= cost1
                    transaction_fee = cost1 * self.cost
                    self.capital -= transaction_fee
                    
                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'stock': stock1,
                        'shares': shares1,
                        'price': price1.loc[date],
                        'value': cost1,
                        'fee': transaction_fee
                    })
                    
                    # 做空stock2
                    shares2 = int(self.capital * 0.5 / price2.loc[date])
                    proceeds2 = shares2 * price2.loc[date]
                    
                    holdings[stock2]['shares'] = -shares2
                    holdings[stock2]['cost'] = proceeds2
                    self.capital += proceeds2
                    transaction_fee = proceeds2 * self.cost
                    self.capital -= transaction_fee
                    
                    self.trades.append({
                        'date': date,
                        'action': 'SHORT',
                        'stock': stock2,
                        'shares': shares2,
                        'price': price2.loc[date],
                        'value': proceeds2,
                        'fee': transaction_fee
                    })
                    
                elif target_position == -1:  # 做空价差
                    # 做空stock1
                    shares1 = int(self.capital * 0.5 / price1.loc[date])
                    proceeds1 = shares1 * price1.loc[date]
                    
                    holdings[stock1]['shares'] = -shares1
                    holdings[stock1]['cost'] = proceeds1
                    self.capital += proceeds1
                    transaction_fee = proceeds1 * self.cost
                    self.capital -= transaction_fee
                    
                    self.trades.append({
                        'date': date,
                        'action': 'SHORT',
                        'stock': stock1,
                        'shares': shares1,
                        'price': price1.loc[date],
                        'value': proceeds1,
                        'fee': transaction_fee
                    })
                    
                    # 做多stock2
                    shares2 = int(self.capital * 0.5 / price2.loc[date])
                    cost2 = shares2 * price2.loc[date]
                    
                    holdings[stock2]['shares'] = shares2
                    holdings[stock2]['cost'] = cost2
                    self.capital -= cost2
                    transaction_fee = cost2 * self.cost
                    self.capital -= transaction_fee
                    
                    self.trades.append({
                        'date': date,
                        'action': 'BUY',
                        'stock': stock2,
                        'shares': shares2,
                        'price': price2.loc[date],
                        'value': cost2,
                        'fee': transaction_fee
                    })
                
                current_position = target_position
            
            # 计算当日组合价值
            portfolio_val = self.capital
            
            # 加上持仓市值
            if holdings[stock1]['shares'] > 0:  # 多头
                portfolio_val += holdings[stock1]['shares'] * price1.loc[date]
            elif holdings[stock1]['shares'] < 0:  # 空头
                portfolio_val += holdings[stock1]['shares'] * price1.loc[date]  # 空头市值（负值）
            
            if holdings[stock2]['shares'] > 0:  # 多头
                portfolio_val += holdings[stock2]['shares'] * price2.loc[date]
            elif holdings[stock2]['shares'] < 0:  # 空头
                portfolio_val += holdings[stock2]['shares'] * price2.loc[date]
            
            portfolio_value.append(portfolio_val)
        
        # 转换为Series
        portfolio_value = pd.Series(portfolio_value, index=signals.index)
        
        return portfolio_value, self.trades
    
    def calculate_performance_metrics(self, portfolio_value):
        """计算策略表现指标"""
        # 收益率
        returns = portfolio_value.pct_change().dropna()
        
        # 累计收益
        cumulative_return = (portfolio_value.iloc[-1] / portfolio_value.iloc[0]) - 1
        
        # 年化收益
        annual_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
        
        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        win_rate = (returns > 0).sum() / len(returns)
        
        metrics = {
            '累计收益': cumulative_return,
            '年化收益': annual_return,
            '年化波动': annual_vol,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '胜率': win_rate,
            '交易次数': len(self.trades)
        }
        
        return metrics
```

## A股实战案例

### 案例：招商银行 vs 平安银行

```python
# 假设已获取数据
stock1 = '600036.SH'  # 招商银行
stock2 = '000001.SZ'  # 平安银行

# 1. 协整检验
analyzer = CointegrationAnalyzer()
price1 = get_stock_data(stock1, '2020-01-01', '2023-12-31')
price2 = get_stock_data(stock2, '2020-01-01', '2023-12-31')

result = analyzer.engle_granger_test(price1, price2)
print(f"协整检验结果：")
print(f"  Beta (对冲比率): {result['beta']:.4f}")
print(f"  ADF统计量: {result['adf_statistic']:.4f}")
print(f"  P值: {result['p_value']:.4f}")
print(f"  是否协整: {result['is_cointegrated']}")

# 2. 计算半衰期
spread = price1 - result['beta'] * price2
half_life = analyzer.calculate_half_life(spread)
print(f"  半衰期: {half_life:.1f} 个交易日")

# 3. 生成信号
signal_gen = PairTradingSignal(entry_threshold=2.0, exit_threshold=0.5)
signals = signal_gen.generate_signals(price1, price2, result['beta'])

# 4. 回测
backtest = PairTradingBacktest(initial_capital=1e6, transaction_cost=0.001)
portfolio_value, trades = backtest.run_backtest(
    stock1, stock2, 
    pd.DataFrame({stock1: price1, stock2: price2}), 
    result['beta'], 
    signal_gen
)

# 5. 输出指标
metrics = backtest.calculate_performance_metrics(portfolio_value)
print("\n策略表现：")
for key, value in metrics.items():
    if '率' in key or '收益' in key or '回撤' in key:
        print(f"  {key}: {value:.2%}")
    else:
        print(f"  {key}: {value:.4f}")

# 6. 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(15, 12))

# 子图1：价格走势
ax1 = axes[0]
ax1.plot(price1.index, price1.values, label=stock1, color='blue')
ax1.plot(price2.index, price2.values, label=stock2, color='red')
ax1.set_title('Price Trends')
ax1.legend()

# 子图2：价差和Z-Score
ax2 = axes[1]
ax2.plot(spread.index, spread.values, label='Spread', color='green')
ax2.axhline(spread.mean(), color='black', linestyle='--', label='Mean')
ax2.fill_between(spread.index, 
                  spread.mean() + 2*spread.std(), 
                  spread.mean() - 2*spread.std(), 
                  alpha=0.2, color='gray')
ax2.set_title('Spread')
ax2.legend()

# 子图3：组合净值
ax3 = axes[2]
ax3.plot(portfolio_value.index, portfolio_value.values, label='Portfolio Value', color='purple')
ax3.set_title('Portfolio Value')
ax3.legend()

plt.tight_layout()
plt.savefig('pair_trading_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 实战要点与注意事项

### 1. 数据质量

- **前复权调整**：必须使用前复权价格，避免除权除息导致价差突变
- **停牌处理**：停牌期间无法交易，需要向前填充或剔除
- **流动性筛选**：确保两只股票都有足够的成交量

### 2. 执行细节

- **买卖价差**：回测中使用中间价，实盘需要考虑买卖价差
- **滑点**：大单冲击市场，导致执行价格偏离信号价格
- **融券成本**：做空需要支付融券利息（A股约8-10%年化）

### 3. 模型风险

- **结构断裂**：并购、重组、行业政策变化可能导致协整关系永久破裂
- **过拟合**：在历史数据上优化参数容易导致过拟合
- **黑天鹅**：2008年、2020年3月等极端行情下，配对交易可能大幅亏损

## 结论

配对交易是一个理论严谨、逻辑清晰的市场中性策略。成功实施的关键在于：

1. **严格的统计检验**：确保协整关系真实存在且稳定
2. **动态风险管理**：实时监控配对关系是否破裂
3. **合理的交易成本建模**：避免因频繁交易侵蚀收益
4. **多对分散**：单对风险集中，建议同时交易10-20对

**未来方向：**
- 高频配对交易（利用分钟级数据）
- 跨市场配对（A股-H股价差）
- 机器学习优化信号（使用LSTM预测价差回归）

---

**免责声明**：本文仅供学术交流，不构成投资建议。配对交易虽然理论优美，但在实盘中面临诸多挑战，包括但不限于数据过拟合、交易成本、模型失效、融券限制等。读者在实盘应用前应进行充分的回测和风险评估，并遵守相关法律法规。

## 参考文献

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*.
4. Liu, J., & Timmermann, A. (2013). Optimal convergence trade strategies. *Review of Financial Studies*.
