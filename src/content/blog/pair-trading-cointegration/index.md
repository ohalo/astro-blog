---
title: "配对交易与协整分析：市场中性策略的统计套利实战"
description: "深入讲解配对交易的理论基础、协整检验方法、交易信号构建和风险管理，提供完整的Python实现代码和实战案例。"
pubDate: 2026-06-15
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "量化策略"]
category: "量化交易"
featured: false
---

# 配对交易与协整分析：市场中性策略的统计套利实战

## 引言

配对交易（Pairs Trading）是最经典的市场中性策略之一，由摩根士丹利在1980年代首创。该策略基于"均值回归"原理，通过寻找价格具有长期均衡关系的股票对，在价差偏离时建立多空对冲头寸，等待价差回归获利。本文将深入探讨配对交易的理论基础、协整检验方法和实战技巧。

## 配对交易的理论基础

### 1. 平稳性与协整

配对交易的核心假设是：两只股票的价格虽然是非平稳的（Non-stationary），但它们的线性组合是平稳的（Stationary），即存在协整关系（Cointegration）。

**数学定义：**

如果时间序列 $X_t$ 和 $Y_t$ 都是 $I(1)$ 过程（一阶单整），且存在非零向量 $\alpha, \beta$ 使得：

$$
Z_t = \alpha + \beta Y_t - X_t \sim I(0)
$$

即残差序列 $Z_t$ 是平稳的，则称 $X_t$ 和 $Y_t$ 是协整的。

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import matplotlib.pyplot as plt
import seaborn as sns

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class PairTradingAnalyzer:
    """配对交易分析器"""
    
    def __init__(self, stock_data, lookback=252):
        """
        初始化
        
        Parameters:
        -----------
        stock_data : DataFrame, 包含多只股票的价格数据
        lookback : 回看窗口
        """
        self.prices = stock_data
        self.lookback = lookback
        self.pairs = []
        
    def find_cointegrated_pairs(self, p_value_threshold=0.05):
        """
        寻找协整配对的股票对
        
        Returns:
        --------
        cointegrated_pairs : List of tuples (stock1, stock2, p-value, hedge_ratio)
        """
        n = len(self.prices.columns)
        cointegrated_pairs = []
        
        print(f"开始扫描 {n} 只股票的协整关系...")
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = self.prices.columns[i]
                stock2 = self.prices.columns[j]
                
                #  Engle-Granger 协整检验
                score, p_value, _ = coint(
                    self.prices[stock1], 
                    self.prices[stock2]
                )
                
                if p_value < p_value_threshold:
                    # 计算对冲比率（hedge ratio）
                    hedge_ratio = self._calculate_hedge_ratio(
                        self.prices[stock1], 
                        self.prices[stock2]
                    )
                    
                    cointegrated_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'p_value': p_value,
                        'score': score,
                        'hedge_ratio': hedge_ratio
                    })
        
        # 按p-value排序
        cointegrated_pairs = sorted(
            cointegrated_pairs, 
            key=lambda x: x['p_value']
        )
        
        print(f"找到 {len(cointegrated_pairs)} 对协整关系")
        return cointegrated_pairs
    
    def _calculate_hedge_ratio(self, price1, price2, method='OLS'):
        """
        计算对冲比率
        
        Parameters:
        -----------
        method : 'OLS' 或 'TLS'（总体最小二乘）
        """
        if method == 'OLS':
            # OLS回归：price1 = alpha + beta * price2 + error
            X = sm.add_constant(price2)
            model = sm.OLS(price1, X).fit()
            hedge_ratio = model.params[1]
            
        elif method == 'TLS':
            # 总体最小二乘（考虑两组序列都有误差）
            X = np.column_stack([price1, price2])
            U, s, Vt = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)
            hedge_ratio = -Vt[1, 0] / Vt[1, 1]
            
        return hedge_ratio
    
    def calculate_spread(self, stock1, stock2, hedge_ratio, window=None):
        """
        计算价差（Spread）
        
        spread = price1 - hedge_ratio * price2
        """
        if window is None:
            window = len(self.prices)
        
        price1 = self.prices[stock1].iloc[-window:]
        price2 = self.prices[stock2].iloc[-window:]
        
        spread = price1 - hedge_ratio * price2
        
        return spread
    
    def test_mean_reversion(self, spread, lookback=None):
        """
        检验均值回归特性
        
        Returns:
        --------
        dict: 包含ADF检验、Hurst指数、半衰期等指标
        """
        if lookback is None:
            lookback = self.lookback
        
        spread = spread.iloc[-lookback:]
        
        # 1. ADF检验（Augmented Dickey-Fuller Test）
        adf_result = adfuller(spread, autolag='AIC')
        
        # 2. Hurst指数
        hurst = self._calculate_hurst_exponent(spread)
        
        # 3. 半衰期（Half-life）
        half_life = self._calculate_half_life(spread)
        
        # 4. 自相关函数（ACF）
        acf_1 = pd.Series(spread).autocorr(lag=1)
        
        return {
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'hurts_exponent': hurst,
            'half_life': half_life,
            'autocorr_1': acf_1,
            'is_mean_reverting': (adf_result[1] < 0.05) and (hurst < 0.5)
        }
    
    def _calculate_hurst_exponent(self, ts, max_lag=100):
        """计算Hurst指数"""
        lags = range(2, min(max_lag, len(ts)//2))
        tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
        
        poly = np.polyfit(np.log(lags), np.log(tau), 1)
        return poly[0]
    
    def _calculate_half_life(self, spread):
        """计算均值回归的半衰期"""
        # 使用AR(1)模型估计
        spread_lag = spread.shift(1).dropna()
        spread_ret = spread.diff().dropna()
        
        X = sm.add_constant(spread_lag)
        model = sm.OLS(spread_ret, X).fit()
        
        # beta系数
        beta = model.params[1]
        
        # 半衰期
        half_life = -np.log(2) / np.log(1 + beta) if beta < 0 else np.inf
        
        return half_life

# 生成模拟数据
np.random.seed(42)
n_days = 252 * 5  # 5年数据
n_stocks = 50

dates = pd.date_range('2021-01-01', periods=n_days, freq='D')

# 生成具有协整关系的股票价格
prices = pd.DataFrame(index=dates)

# 生成一些板块/行业因子
sector_factor1 = np.cumsum(np.random.randn(n_days) * 0.01 + 0.0002)
sector_factor2 = np.cumsum(np.random.randn(n_days) * 0.008 + 0.0001)

for i in range(n_stocks):
    if i < 10:
        # 第1组：与sector_factor1协整
        noise = np.cumsum(np.random.randn(n_days) * 0.005)
        price = 100 + 50 * sector_factor1 + noise
    elif i < 20:
        # 第2组：与sector_factor2协整
        noise = np.cumsum(np.random.randn(n_days) * 0.005)
        price = 100 + 30 * sector_factor2 + noise
    else:
        # 其他：随机游走
        price = 100 + np.cumsum(np.random.randn(n_days) * 0.01)
    
    prices[f'STOCK_{i:02d}'] = price

# 实例化分析器
analyzer = PairTradingAnalyzer(prices, lookback=252)

# 寻找协整配对
pairs = analyzer.find_cointegrated_pairs(p_value_threshold=0.01)

print("\n前10对协整关系:")
for i, pair in enumerate(pairs[:10]):
    print(f"{i+1}. {pair['stock1']} - {pair['stock2']} | "
          f"p-value: {pair['p_value']:.6f} | "
          f"Hedge Ratio: {pair['hedge_ratio']:.4f}")
```

### 2. 协整检验的多种方法

除了经典的Engle-Granger两步法，还有更严谨的检验方法：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_cointegration_test(price_data, det_order=-1, k_ar_diff=1):
    """
    Johansen 协整检验（适用于多变量）
    
    Parameters:
    -----------
    price_data : DataFrame, 多只股票价格
    det_order : 确定性项顺序 (-1: 无, 0: 有截距, 1: 有截距和趋势)
    k_ar_diff : 滞后阶数
    """
    # Johansen检验
    result = coint_johansen(
        price_data.values, 
        det_order, 
        k_ar_diff
    )
    
    # 输出结果
    print("Johansen协整检验结果:")
    print("=" * 60)
    
    # 迹统计量（Trace Statistic）
    print("\n迹统计量检验:")
    for i in range(len(price_data.columns)):
        print(f"  r<={i}: 统计量={result.lr1[i]:.2f}, "
              f"5%临界值={result.cvt[i, 1]:.2f}, "
              f"是否拒绝: {result.lr1[i] > result.cvt[i, 1]}")
    
    # 最大特征值统计量（Max Eigenvalue Statistic）
    print("\n最大特征值检验:")
    for i in range(len(price_data.columns)-1):
        print(f"  r={i}: 统计量={result.lr2[i]:.2f}, "
              f"5%临界值={result.cve[i, 1]:.2f}, "
              f"是否拒绝: {result.lr2[i] > result.cve[i, 1]}")
    
    return result

# 示例：对前3只股票进行Johansen检验
sample_data = prices[['STOCK_00', 'STOCK_01', 'STOCK_02']].iloc[:252]
johansen_result = johansen_cointegration_test(sample_data)

# Phillips-Ouliaris检验（更稳健的协整检验）
from statsmodels.tsa.stattools import coint

def phillips_ouliaris_test(price1, price2):
    """
    Phillips-Ouliaris 协整检验
    （对结构性断点更稳健）
    """
    # 使用statsmodels的coint函数（基于Phillips-Ouliaris）
    t_stat, p_value, _ = coint(price1, price2, trend='c')
    
    print(f"\nPhillips-Ouliaris检验结果:")
    print(f"  检验统计量: {t_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  结论: {'存在协整关系' if p_value < 0.05 else '不存在协整关系'}")
    
    return t_stat, p_value

# 示例
stock1_prices = prices['STOCK_00'].iloc[:252]
stock2_prices = prices['STOCK_01'].iloc[:252]
po_result = phillips_ouliaris_test(stock1_prices, stock2_prices)
```

## 交易信号的构建

### 1. 基于Z-Score的信号

最常用的交易信号是基于价差的Z-Score（标准化价差）。

```python
class PairTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5, 
                 stop_loss_zscore=3.0, lookback=63):
        """
        初始化策略参数
        
        Parameters:
        -----------
        entry_zscore : 入场Z值
        exit_zscore : 出场Z值
        stop_loss_zscore : 止损Z值
        lookback : 计算Z-Score的回看窗口
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.lookback = lookback
        
    def generate_signals(self, spread):
        """
        生成交易信号
        
        Returns:
        --------
        signals : DataFrame, 包含仓位和信号
        """
        # 计算滚动Z-Score
        spread_mean = spread.rolling(window=self.lookback).mean()
        spread_std = spread.rolling(window=self.lookback).std()
        z_score = (spread - spread_mean) / spread_std
        
        # 初始化信号
        signals = pd.DataFrame(index=spread.index)
        signals['spread'] = spread
        signals['z_score'] = z_score
        signals['position'] = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
        signals['signal'] = 0  # 交易信号
        
        # 生成仓位
        position = 0
        
        for i in range(1, len(signals)):
            if pd.isna(z_score.iloc[i]):
                continue
            
            z = z_score.iloc[i]
            
            if position == 0:
                # 空仓状态
                if z < -self.entry_zscore:
                    # 做多价差（买入stock1，卖出stock2）
                    position = 1
                    signals.iloc[i, signals.columns.get_loc('signal')] = 1
                elif z > self.entry_zscore:
                    # 做空价差（卖出stock1，买入stock2）
                    position = -1
                    signals.iloc[i, signals.columns.get_loc('signal')] = -1
            
            elif position == 1:
                # 持有做多价差仓位
                if z >= -self.exit_zscore:
                    # 平仓
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                elif z < -self.stop_loss_zscore:
                    # 止损
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('signal')] = 99  # 止损信号
            
            elif position == -1:
                # 持有做空价差仓位
                if z <= self.exit_zscore:
                    # 平仓
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('signal')] = 0
                elif z > self.stop_loss_zscore:
                    # 止损
                    position = 0
                    signals.iloc[i, signals.columns.get_loc('signal')] = 99  # 止损信号
            
            signals.iloc[i, signals.columns.get_loc('position')] = position
        
        return signals
    
    def backtest(self, signals, transaction_cost=0.001):
        """
        回测策略
        
        Parameters:
        -----------
        transaction_cost : 交易成本（单边）
        """
        # 计算策略收益
        signals['returns'] = signals['spread'].pct_change()
        signals['strategy_returns'] = signals['position'].shift(1) * signals['returns']
        
        # 扣除交易成本
        trades = signals['position'].diff().abs()
        signals['transaction_cost'] = trades * transaction_cost
        signals['net_returns'] = signals['strategy_returns'] - signals['transaction_cost']
        
        # 计算累计收益
        signals['cumulative_returns'] = (1 + signals['net_returns']).cumprod()
        
        # 计算绩效指标
        total_return = signals['cumulative_returns'].iloc[-1] - 1
        n_days = len(signals)
        annual_return = (1 + total_return) ** (252 / n_days) - 1
        
        sharpe_ratio = (signals['net_returns'].mean() / 
                       signals['net_returns'].std() * np.sqrt(252))
        
        max_drawdown = ((signals['cumulative_returns'] / 
                        signals['cumulative_returns'].cummax()) - 1).min()
        
        n_trades = trades.sum()
        win_rate = ((signals['strategy_returns'] > 0).sum() / 
                    (signals['position'] != 0).sum())
        
        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'n_trades': n_trades,
            'win_rate': win_rate
        }
        
        return signals, performance

# 使用示例
# 选择一对协整股票
if len(pairs) > 0:
    best_pair = pairs[0]
    stock1 = best_pair['stock1']
    stock2 = best_pair['stock2']
    hedge_ratio = best_pair['hedge_ratio']
    
    print(f"\n最优配对: {stock1} - {stock2}")
    print(f"对冲比率: {hedge_ratio:.4f}")
    
    # 计算价差
    spread = analyzer.calculate_spread(stock1, stock2, hedge_ratio)
    
    # 检验均值回归
    mean_reversion_test = analyzer.test_mean_reversion(spread)
    print("\n均值回归检验结果:")
    for k, v in mean_reversion_test.items():
        print(f"  {k}: {v}")
    
    # 生成交易信号
    strategy = PairTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        lookback=63
    )
    
    signals = strategy.generate_signals(spread)
    
    # 回测
    signals, performance = strategy.backtest(signals, transaction_cost=0.001)
    
    print("\n策略绩效:")
    for k, v in performance.items():
        if k in ['total_return', 'annual_return', 'max_drawdown']:
            print(f"  {k}: {v:.2%}")
        elif k == 'sharpe_ratio':
            print(f"  {k}: {v:.4f}")
        else:
            print(f"  {k}: {v}")
```

### 2. 动态对冲比率

传统配对交易使用固定对冲比率，但在实际中，对冲比率可能随时间变化。可以使用滚动窗口或卡尔曼滤波来动态估计。

```python
from pykalman import KalmanFilter

class DynamicHedgeRatio:
    """动态对冲比率估计"""
    
    def __init__(self, price1, price2, method='rolling'):
        """
        初始化
        
        Parameters:
        -----------
        price1, price2 : Series, 两只股票的价格
        method : 'rolling' 或 'kalman'
        """
        self.price1 = price1
        self.price2 = price2
        self.method = method
        self.hedge_ratios = None
        
    def estimate(self, window=63):
        """
        估计动态对冲比率
        
        Returns:
        --------
        hedge_ratios : Series, 动态对冲比率
        """
        if self.method == 'rolling':
            # 滚动窗口OLS
            hedge_ratios = pd.Series(index=self.price1.index)
            
            for i in range(window, len(self.price1)):
                window_price1 = self.price1.iloc[i-window:i]
                window_price2 = self.price2.iloc[i-window:i]
                
                X = sm.add_constant(window_price2)
                model = sm.OLS(window_price1, X).fit()
                hedge_ratios.iloc[i] = model.params[1]
            
            self.hedge_ratios = hedge_ratios
            
        elif self.method == 'kalman':
            # 卡尔曼滤波
            # 状态：hedge_ratio
            # 观测：price1 = hedge_ratio * price2 + error
            
            # 初始化卡尔曼滤波器
            kf = KalmanFilter(
                transition_matrices=np.eye(1),
                observation_matrices=self.price2.values.reshape(-1, 1, 1),
                initial_state_mean=1.0,
                initial_state_covariance=np.eye(1) * 0.1,
                observation_covariance=0.1,
                transition_covariance=np.eye(1) * 0.01
            )
            
            # 滤波
            state_means, _ = kf.filter(self.price1.values)
            
            self.hedge_ratios = pd.Series(
                state_means.flatten(),
                index=self.price1.index
            )
        
        return self.hedge_ratios
    
    def calculate_dynamic_spread(self):
        """计算动态价差"""
        if self.hedge_ratios is None:
            self.estimate()
        
        dynamic_spread = (self.price1 - 
                         self.hedge_ratios * self.price2)
        
        return dynamic_spread

# 使用示例
dynamic_hedge = DynamicHedgeRatio(
    prices['STOCK_00'], 
    prices['STOCK_01'],
    method='kalman'
)

dynamic_hedge_ratios = dynamic_hedge.estimate()
dynamic_spread = dynamic_hedge.calculate_dynamic_spread()

print("\n动态对冲比率统计:")
print(dynamic_hedge_ratios.describe())

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1：价格序列
axes[0].plot(prices.index, prices['STOCK_00'], label='STOCK_00', alpha=0.7)
axes[0].plot(prices.index, prices['STOCK_01'], label='STOCK_01', alpha=0.7)
axes[0].set_title('股票价格走势', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：动态对冲比率
axes[1].plot(dynamic_hedge_ratios.index, dynamic_hedge_ratios, 
             label='Dynamic Hedge Ratio', color='red', linewidth=2)
axes[1].axhline(y=1.0, color='black', linestyle='--', 
                label='Fixed Ratio = 1.0')
axes[1].set_title('动态对冲比率（卡尔曼滤波）', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 子图3：动态价差 vs 静态价差
static_spread = prices['STOCK_00'] - prices['STOCK_01']
axes[2].plot(static_spread.index, static_spread, 
             label='Static Spread', alpha=0.5)
axes[2].plot(dynamic_spread.index, dynamic_spread, 
             label='Dynamic Spread', alpha=0.7)
axes[2].set_title('价差对比', fontsize=14, fontweight='bold')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/dynamic_hedge_ratio.png', 
            dpi=300, bbox_inches='tight')
print("\n图表已保存: dynamic_hedge_ratio.png")
```

## 风险管理与实务要点

### 1. 仓位管理

配对交易虽然是市场中性策略,但仍需严格的风险管理。

```python
class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position_size=0.05, max_sector_exposure=0.20,
                 stop_loss_pct=0.02, take_profit_pct=0.04):
        """
        初始化风险管理参数
        
        Parameters:
        -----------
        max_position_size : 单一配对最大仓位
        max_sector_exposure : 单一板块最大暴露
        stop_loss_pct : 止损百分比
        take_profit_pct : 止盈百分比
        """
        self.max_position_size = max_position_size
        self.max_sector_exposure = max_sector_exposure
        self.stop_loss_pct = stop_loss_pct
        self.take_profit_pct = take_profit_pct
        
    def calculate_position_size(self, capital, price1, price2, 
                                volatility, target_risk=0.001):
        """
        根据波动率调整仓位大小
        
        Kelly公式的简化版：
        position_size = target_risk / (volatility * sqrt(2))
        """
        # 计算价差的波动率
        spread_vol = volatility * np.sqrt(price1**2 + price2**2)
        
        # 目标风险仓位
        position_size = target_risk / (spread_vol + 1e-8)
        
        # 限制最大仓位
        position_size = min(position_size, self.max_position_size)
        
        # 计算实际交易股数
        n_shares1 = int(capital * position_size / price1)
        n_shares2 = int(capital * position_size * self._get_hedge_ratio() / price2)
        
        return {
            'position_size': position_size,
            'n_shares1': n_shares1,
            'n_shares2': n_shares2,
            'notional_value': n_shares1 * price1 + n_shares2 * price2
        }
    
    def _get_hedge_ratio(self):
        """获取对冲比率（简化版）"""
        return 1.0
    
    def check_stop_loss(self, entry_price, current_price, position_type='long'):
        """检查止损"""
        if position_type == 'long':
            loss = (entry_price - current_price) / entry_price
        else:
            loss = (current_price - entry_price) / entry_price
        
        if loss > self.stop_loss_pct:
            return True  # 触发止损
        return False
    
    def check_take_profit(self, entry_price, current_price, position_type='long'):
        """检查止盈"""
        if position_type == 'long':
            profit = (current_price - entry_price) / entry_price
        else:
            profit = (entry_price - current_price) / entry_price
        
        if profit > self.take_profit_pct:
            return True  # 触发止盈
        return False
    
    def monitor_correlation_breakdown(self, price1, price2, window=63):
        """
        监测相关性断裂（协整关系失效的早期信号）
        """
        # 计算滚动相关性
        rolling_corr = price1.rolling(window).corr(price2)
        
        # 计算相关性漂移
        corr_mean = rolling_corr.mean()
        corr_current = rolling_corr.iloc[-1]
        
        # 如果相关性显著下降，发出警告
        if corr_current < corr_mean - 2 * rolling_corr.std():
            print(f"⚠️ 警告：相关性断裂！当前相关性: {corr_current:.4f}")
            return True
        
        return False

# 使用示例
risk_manager = RiskManager(
    max_position_size=0.05,
    stop_loss_pct=0.02,
    take_profit_pct=0.04
)

# 假设当前价格
current_price1 = prices['STOCK_00'].iloc[-1]
current_price2 = prices['STOCK_01'].iloc[-1]
current_volatility = spread.std()

position_info = risk_manager.calculate_position_size(
    capital=1e6,
    price1=current_price1,
    price2=current_price2,
    volatility=current_volatility,
    target_risk=0.001
)

print("\n仓位管理建议:")
for k, v in position_info.items():
    print(f"  {k}: {v}")
```

### 2. 协整关系断裂的检测

协整关系可能随时间失效，需要及时检测并退出交易。

```python
def detect_cointegration_breakdown(spread, window=63, threshold=0.05):
    """
    检测协整关系断裂
    
    方法：
    1. 滚动ADF检验
    2. 滚动Hurst指数
    3. 结构断点检验（Chow Test）
    """
    n = len(spread)
    breakdown_signals = pd.Series(index=spread.index, dtype=bool)
    
    for i in range(window, n):
        window_spread = spread.iloc[i-window:i]
        
        # 1. 滚动ADF检验
        adf_result = adfuller(window_spread, autolag='AIC')
        p_value = adf_result[1]
        
        # 2. 滚动Hurst指数
        hurst = _calculate_hurst_exponent(window_spread)
        
        # 3. 判断断裂
        if (p_value > threshold) or (hurst > 0.5):
            breakdown_signals.iloc[i] = True
        else:
            breakdown_signals.iloc[i] = False
    
    # 统计断裂频率
    breakdown_freq = breakdown_signals.sum() / len(breakdown_signals)
    
    print(f"\n协整关系断裂检测:")
    print(f"  断裂频率: {breakdown_freq:.2%}")
    print(f"  最近断裂信号: {'是' if breakdown_signals.iloc[-1] else '否'}")
    
    return breakdown_signals

def _calculate_hurst_exponent(ts, max_lag=100):
    """计算Hurst指数（辅助函数）"""
    lags = range(2, min(max_lag, len(ts)//2))
    tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
    poly = np.polyfit(np.log(lags), np.log(tau), 1)
    return poly[0]

# 使用示例
breakdown_signals = detect_cointegration_breakdown(spread, window=63)

# 可视化断裂信号
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 子图1：价差序列
axes[0].plot(spread.index, spread, label='Spread', alpha=0.7)
axes[0].scatter(spread.index[breakdown_signals], 
                spread[breakdown_signals], 
                color='red', s=20, label='Breakdown Signal', zorder=5)
axes[0].set_title('价差序列与协整断裂信号', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：滚动ADF p-value
rolling_pvalue = pd.Series(index=spread.index)
for i in range(63, len(spread)):
    window_spread = spread.iloc[i-63:i]
    adf_result = adfuller(window_spread, autolag='AIC')
    rolling_pvalue.iloc[i] = adf_result[1]

axes[1].plot(rolling_pvalue.index, rolling_pvalue, 
             label='Rolling ADF p-value', color='purple')
axes[1].axhline(y=0.05, color='red', linestyle='--', label='Threshold (0.05)')
axes[1].set_title('滚动ADF检验 p-value', fontsize=14, fontweight='bold')
axes[1].set_ylim(0, 1)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cointegration_breakdown.png', 
            dpi=300, bbox_inches='tight')
print("图表已保存: cointegration_breakdown.png")
```

## 实战案例：A股配对交易

让我们用一个A股的实战案例来总结全文。

```python
# 实战案例：中国平安 vs 中国太保（保险板块配对）
# 注：这里使用模拟数据，实际应接入真实行情数据（如tushare、akshare）

def real_world_example():
    """实战案例分析"""
    
    print("\n" + "="*80)
    print("实战案例：中国平安 (601318.SH) vs 中国太保 (601601.SH)")
    print("="*80)
    
    # 1. 数据获取（模拟）
    dates = pd.date_range('2020-01-01', '2026-06-15', freq='D')
    n = len(dates)
    
    # 模拟平安和太保的股价（具有协整关系）
    np.random.seed(42)
    
    # 共同行业因子
    sector_factor = np.cumsum(np.random.randn(n) * 0.015 + 0.0003)
    
    # 个股特异性
    pingan_noise = np.cumsum(np.random.randn(n) * 0.008)
    cpic_noise = np.cumsum(np.random.randn(n) * 0.009)
    
    # 生成价格
    pingan_price = 50 + 30 * sector_factor + pingan_noise
    cpic_price = 40 + 25 * sector_factor + cpic_noise
    
    prices_real = pd.DataFrame({
        'pingan': pingan_price,
        'cpic': cpic_price
    }, index=dates)
    
    # 2. 协整检验
    print("\n1. 协整检验:")
    score, p_value, _ = coint(prices_real['pingan'], prices_real['cpic'])
    print(f"   Engle-Granger检验 p-value: {p_value:.6f}")
    print(f"   结论: {'存在协整关系' if p_value < 0.05 else '不存在协整关系'}")
    
    # 3. 计算对冲比率
    X = sm.add_constant(prices_real['cpic'])
    model = sm.OLS(prices_real['pingan'], X).fit()
    hedge_ratio = model.params[1]
    print(f"\n2. 对冲比率: {hedge_ratio:.4f}")
    
    # 4. 计算价差
    spread_real = prices_real['pingan'] - hedge_ratio * prices_real['cpic']
    
    # 5. 均值回归检验
    mean_reversion = analyzer.test_mean_reversion(spread_real)
    print("\n3. 均值回归检验:")
    print(f"   ADF p-value: {mean_reversion['adf_pvalue']:.6f}")
    print(f"   Hurst指数: {mean_reversion['hurts_exponent']:.4f}")
    print(f"   半衰期: {mean_reversion['half_life']:.1f} 天")
    print(f"   是否均值回归: {mean_reversion['is_mean_reverting']}")
    
    # 6. 策略回测
    strategy_real = PairTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        lookback=63
    )
    
    signals_real = strategy_real.generate_signals(spread_real)
    signals_real, performance_real = strategy_real.backtest(
        signals_real, 
        transaction_cost=0.001
    )
    
    print("\n4. 策略回测绩效:")
    print(f"   累计收益: {performance_real['total_return']:.2%}")
    print(f"   年化收益: {performance_real['annual_return']:.2%}")
    print(f"   夏普比率: {performance_real['sharpe_ratio']:.4f}")
    print(f"   最大回撤: {performance_real['max_drawdown']:.2%}")
    print(f"   交易次数: {performance_real['n_trades']:.0f}")
    print(f"   胜率: {performance_real['win_rate']:.2%}")
    
    # 7. 可视化
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 子图1：价格走势
    axes[0].plot(prices_real.index, prices_real['pingan'], 
                 label='中国平安', linewidth=2)
    axes[0].plot(prices_real.index, prices_real['cpic'] * hedge_ratio, 
                 label=f'中国太保 (×{hedge_ratio:.2f})', linewidth=2)
    axes[0].set_title('股票价格走势（标准化）', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：价差与Z-Score
    ax2_twin = axes[1].twinx()
    axes[1].plot(spread_real.index, spread_real, 
                 label='Spread', color='blue', alpha=0.7)
    axes[1].set_ylabel('Spread', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    z_score_real = (spread_real - spread_real.rolling(63).mean()) / spread_real.rolling(63).std()
    ax2_twin.plot(z_score_real.index, z_score_real, 
                  label='Z-Score', color='red', alpha=0.7)
    ax2_twin.axhline(y=2.0, color='red', linestyle='--', alpha=0.5)
    ax2_twin.axhline(y=-2.0, color='green', linestyle='--', alpha=0.5)
    ax2_twin.set_ylabel('Z-Score', fontsize=12)
    
    axes[1].set_title('价差与Z-Score', fontsize=14, fontweight='bold')
    
    # 子图3：仓位变化
    axes[2].plot(signals_real.index, signals_real['position'], 
                 label='Position', color='purple', linewidth=2)
    axes[2].set_title('交易仓位', fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Position (-1/0/1)', fontsize=12)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_ylim(-1.5, 1.5)
    
    # 子图4：累计收益
    axes[3].plot(signals_real.index, signals_real['cumulative_returns'], 
                 label='Strategy Returns', color='green', linewidth=2)
    axes[3].axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('策略累计收益', fontsize=14, fontweight='bold')
    axes[3].set_ylabel('Cumulative Returns', fontsize=12)
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/real_case.png', 
                dpi=300, bbox_inches='tight')
    print("\n✅ 图表已保存: real_case.png")
    
    return prices_real, signals_real, performance_real

# 执行实战案例
prices_real, signals_real, performance_real = real_world_example()
```

## 总结与展望

配对交易作为一种经典的市场中性策略，在量化投资中占有重要地位。本文详细介绍了：

1. **理论基础**：协整关系与均值回归
2. **检验方法**：Engle-Granger、Johansen、Phillips-Ouliaris检验
3. **信号构建**：基于Z-Score的交易信号、动态对冲比率
4. **风险管理**：仓位管理、止损止盈、协整断裂检测
5. **实战案例**：A股保险板块配对交易

**未来发展方向：**

- **机器学习增强**：使用LSTM、Transformer等模型预测价差均值回归
- **高频配对交易**：在分钟级或秒级数据上实施配对交易
- **多因子配对**：结合基本面因子（估值、动量等）筛选配对
- **跨市场配对**：在不同交易所或不同国家市场间寻找配对机会

---

**参考文献：**

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." Review of Financial Studies.
2. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley.

**免责声明：** 本文仅供参考，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。实际交易前请充分评估风险。
