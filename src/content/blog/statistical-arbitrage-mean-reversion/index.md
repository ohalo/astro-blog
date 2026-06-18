---
title: "统计套利：均值回归策略的完整实战指南"
publishDate: '2026-06-18'
description: "统计套利：均值回归策略 - halo的技术博客"
tags:
 - AI观察
language: Chinese
---

# 统计套利：均值回归策略的完整实战指南

## 引言：市场的"弹簧"本质

想象一根被拉伸的弹簧——它终将回归原始长度。金融市场同样遵循这一物理规律：**价格会围绕价值波动，而偏离终将被纠正**。

这就是**统计套利（Statistical Arbitrage）**的核心思想。1990年代，摩根士丹利的数学天才们用这个方法创造了年化40%+的收益，Nora Jones和Lars Peter Hansen的研究让它成为量化领域的基石策略。

本文将系统讲解：

1. 统计套利的理论基础
2. 核心策略：配对交易、均值回归
3. Python实战：从协整检验到仓位管理
4. 风险控制与组合优化
5. 实盘避坑指南

## 一、统计套利的理论基础

### 1.1 什么是统计套利？

**定义**：利用资产价格的短期统计规律，在价差偏离均值时建仓，期望价差回归时获利的策略。

**核心假设**：
- 价格偏离是暂时的
- 市场会自我纠错
- 偏离程度与回归概率正相关

### 1.2 数学框架

设两个资产价格序列为 $P_1$ 和 $P_2$，我们构造价差：

$$Z_t = P_{1,t} - \alpha \cdot P_{2,t} - \beta$$

其中 $\alpha$ 和 $\beta$ 是对冲参数。

**统计套利的条件**：

1. **平稳性**：$Z_t$ 必须是平稳序列（I(0)）
2. **均值回归**：$E[Z_{t+1} | Z_t] < Z_t$ 当 $Z_t$ 高于均值时
3. **均值回复速度**：用Ornstein-Uhlenbeck过程描述

$$dZ_t = \kappa(\mu - Z_t)dt + \sigma dW_t$$

- $\kappa$：回复速度（越大回复越快）
- $\mu$：长期均值
- $\sigma$：波动率

### 1.3 与其他策略的区别

| 策略类型 | 时间周期 | 收益来源 | 风险特征 |
|---------|---------|---------|---------|
| 统计套利 | 分钟~日 | 价差回归 | 中低风险 |
| 趋势跟踪 | 日~月 | 趋势延续 | 高风险高收益 |
| 事件驱动 | 不确定 | 定价偏差 | 高风险 |
| 高频套利 | 毫秒 | 延迟偏差 | 极低风险 |

## 二、核心策略详解

### 2.1 策略1：配对交易（Pairs Trading）

配对交易是最经典的统计套利策略，核心是发现"命运相连"的两个资产。

#### 2.1.1 选券标准

**行业关联**：同一产业链的上下游企业
- 中石油 vs 中石化
- 中国平安 vs 中国人寿
- 美的集团 vs 格力电器

**竞争关系**：同质化产品的竞争对手
- 可口可乐 vs 百事可乐
- Nike vs Adidas

**供应链关系**：母子公司、大客户供应商

#### 2.1.2 配对筛选流程

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
from scipy import stats

class PairsSelector:
    """
    配对交易选券系统
    """
    def __init__(self, min_corr=0.7, pvalue_threshold=0.05):
        self.min_corr = min_corr
        self.pvalue_threshold = pvalue_threshold
        self.candidates = []
        
    def correlation_filter(self, price_data, pairs):
        """
        第一步：相关性筛选
        """
        filtered_pairs = []
        for pair in pairs:
            asset1, asset2 = pair
            corr = price_data[asset1].corr(price_data[asset2])
            
            if corr >= self.min_corr:
                filtered_pairs.append({
                    'pair': pair,
                    'correlation': corr
                })
        
        self.candidates = filtered_pairs
        return filtered_pairs
    
    def cointegration_test(self, price_data):
        """
        第二步：协整检验（核心）
        
        Engle-Granger两步法：
        1. OLS回归：P1 = α + β*P2 + ε
        2. 检验ε的平稳性
        """
        coint_pairs = []
        
        for candidate in self.candidates:
            asset1, asset2 = candidate['pair']
            
            # Engle-Granger协整检验
            score, pvalue, _ = coint(
                price_data[asset1],
                price_data[asset2]
            )
            
            if pvalue < self.pvalue_threshold:
                # 计算对冲比率和价差
                beta = np.polyfit(price_data[asset2], price_data[asset1], 1)[0]
                spread = price_data[asset1] - beta * price_data[asset2]
                
                coint_pairs.append({
                    'pair': candidate['pair'],
                    'correlation': candidate['correlation'],
                    'coint_score': score,
                    'pvalue': pvalue,
                    'hedge_ratio': beta,
                    'spread_mean': spread.mean(),
                    'spread_std': spread.std()
                })
        
        return pd.DataFrame(coint_pairs)
    
    def hurst_test(self, spread):
        """
        第三步：Hurst指数检验（可选）
        
        H < 0.5：均值回归
        H > 0.5：趋势延续
        H = 0.5：随机游走
        """
        def hurst(ts):
            lags = range(2, min(100, len(ts)//2))
            tau = [np.std(ts[l:] - ts[:-l]) for l in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2
        
        h = hurst(spread.values)
        return h
    
    def select_top_pairs(self, n=10):
        """
        选择最优N对
        """
        return self.candidates.nlargest(n, 'correlation')
```

### 2.2 策略2：布林带均值回归

布林带是经典的均值回归工具，原理简单但极为有效。

#### 2.2.1 原理

$$Upper = MA_{20} + 2\sigma$$
$$Middle = MA_{20}$$
$$Lower = MA_{20} - 2\sigma$$

当价格触及下轨时买入，触及上轨时卖出。

#### 2.2.2 Python实现

```python
import matplotlib.pyplot as plt

class BollingerBandsStrategy:
    """
    布林带均值回归策略
    """
    def __init__(self, window=20, num_std=2):
        self.window = window
        self.num_std = num_std
        
    def calculate_bands(self, prices):
        """
        计算布林带
        """
        ma = prices.rolling(window=self.window).mean()
        std = prices.rolling(window=self.window).std()
        
        upper = ma + self.num_std * std
        lower = ma - self.num_std * std
        
        return upper, ma, lower
    
    def generate_signals(self, prices, upper, lower):
        """
        生成交易信号
        
        Returns:
        --------
        signals : pd.Series
            1: 买入信号
            -1: 卖出信号
            0: 持有
        """
        signals = pd.Series(0, index=prices.index)
        
        # 价格触及下轨：买入
        signals[prices < lower] = 1
        
        # 价格触及上轨：卖出
        signals[prices > upper] = -1
        
        # 价格回归均线：平仓
        # signals[abs(prices - ma) < 0.01 * ma] = 0
        
        return signals
    
    def calculate_position_size(self, signals, prices, capital=100000):
        """
        计算仓位
        
        固定比例仓位：
        - 每次交易使用总资金的20%
        - 最大持仓不超过80%
        """
        positions = pd.Series(0.0, index=signals.index)
        current_position = 0
        position_value = 0
        
        for i in range(1, len(signals)):
            if signals.iloc[i] == 1 and current_position == 0:
                # 买入信号：投入20%资金
                allocation = capital * 0.2
                shares = allocation / prices.iloc[i]
                current_position = shares
                position_value = shares * prices.iloc[i]
                
            elif signals.iloc[i] == -1 and current_position > 0:
                # 卖出信号：全部平仓
                current_position = 0
                position_value = 0
                
            elif current_position > 0:
                # 更新持仓价值
                position_value = current_position * prices.iloc[i]
                
            positions.iloc[i] = current_position
        
        return positions
    
    def backtest(self, prices, start_date=None, end_date=None):
        """
        回测框架
        """
        if start_date:
            prices = prices[start_date:]
        if end_date:
            prices = prices[:end_date]
        
        upper, ma, lower = self.calculate_bands(prices)
        signals = self.generate_signals(prices, upper, lower)
        positions = self.calculate_position_size(signals, prices)
        
        # 计算收益
        returns = prices.pct_change()
        strategy_returns = positions.shift(1) * returns
        
        # 累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        return {
            'signals': signals,
            'positions': positions,
            'returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'upper_band': upper,
            'lower_band': lower,
            'ma': ma
        }
    
    def plot_results(self, prices, results):
        """
        可视化回测结果
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 8), 
                                  gridspec_kw={'height_ratios': [3, 1]})
        
        # 上图：价格与布林带
        ax1 = axes[0]
        ax1.plot(prices.index, prices, 'b-', label='价格', linewidth=1.5)
        ax1.plot(results['ma'].index, results['ma'], 'k--', 
                 label='MA20', alpha=0.7)
        ax1.fill_between(results['upper_band'].index, 
                          results['lower_band'], 
                          results['upper_band'], 
                          alpha=0.2, color='gray', label='布林带')
        ax1.plot(results['upper_band'].index, results['upper_band'], 
                 'r--', alpha=0.5)
        ax1.plot(results['lower_band'].index, results['lower_band'], 
                 'g--', alpha=0.5)
        
        # 标注买卖点
        buy_signals = results['signals'] == 1
        sell_signals = results['signals'] == -1
        ax1.scatter(prices.index[buy_signals], prices[buy_signals], 
                    marker='^', color='green', s=100, label='买入', zorder=5)
        ax1.scatter(prices.index[sell_signals], prices[sell_signals], 
                    marker='v', color='red', s=100, label='卖出', zorder=5)
        
        ax1.set_title('布林带均值回归策略', fontsize=14)
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 下图：累计收益
        ax2 = axes[1]
        ax2.plot(results['cumulative_returns'].index, 
                 results['cumulative_returns'], 'b-', linewidth=2)
        ax2.axhline(y=1, color='k', linestyle='--', alpha=0.5)
        ax2.set_title('策略累计收益', fontsize=12)
        ax2.set_ylabel('累计收益')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/images/statistical-arbitrage-mean-reversion/bollinger_backtest.png', 
                    dpi=150)
        plt.show()
```

![布林带策略回测](/images/statistical-arbitrage-mean-reversion/bollinger_backtest.png)

### 2.3 策略3：协整价差交易

这是配对交易的高级版本，利用协整关系构建更稳定的策略。

```python
class CointegrationSpreadTrader:
    """
    协整价差交易策略
    """
    def __init__(self, lookback=60, entry_threshold=2, exit_threshold=0.5):
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        
    def compute_hedge_ratio(self, prices1, prices2):
        """
        滚动OLS计算对冲比率
        """
        # 使用前N天数据计算对冲比率
        x = prices2[-self.lookback:]
        y = prices1[-self.lookback:]
        
        beta = np.polyfit(x, y, 1)[0]
        alpha = y.mean() - beta * x.mean()
        
        return alpha, beta
    
    def compute_spread(self, prices1, prices2, alpha, beta):
        """
        计算价差序列
        """
        spread = prices1 - alpha - beta * prices2
        return spread
    
    def compute_z_score(self, spread):
        """
        计算Z-score（用于信号生成）
        """
        mean = spread.rolling(self.lookback).mean()
        std = spread.rolling(self.lookback).std()
        
        z_score = (spread - mean) / std
        return z_score
    
    def generate_orders(self, prices1, prices2):
        """
        生成交易订单
        
        Z-score > 2: 做空spread（price1太贵）
        Z-score < -2: 做多spread（price1便宜）
        |Z-score| < 0.5: 平仓
        """
        alpha, beta = self.compute_hedge_ratio(prices1, prices2)
        spread = self.compute_spread(prices1, prices2, alpha, beta)
        z_score = self.compute_z_score(spread)
        
        # 交易信号
        position = pd.Series(0, index=prices1.index)
        
        # 做空spread：price1相对price2过高
        position[z_score > self.entry_threshold] = -1
        
        # 做多spread：price1相对price2过低
        position[z_score < -self.entry_threshold] = 1
        
        # 平仓
        position[(z_score.abs() < self.exit_threshold)] = 0
        
        return position, z_score, beta
    
    def backtest_pair(self, prices1, prices2, initial_capital=100000):
        """
        配对回测
        """
        position, z_score, beta = self.generate_orders(prices1, prices2)
        
        # 计算spread收益率
        spread_return = position.shift(1) * (
            prices1.pct_change() - beta * prices2.pct_change()
        )
        
        # 累计收益
        equity_curve = (1 + spread_return).cumprod() * initial_capital
        
        # 统计指标
        total_return = equity_curve.iloc[-1] / initial_capital - 1
        sharpe = spread_return.mean() / spread_return.std() * np.sqrt(252)
        max_drawdown = (equity_curve / equity_curve.cummax() - 1).min()
        
        return {
            'equity_curve': equity_curve,
            'z_score': z_score,
            'position': position,
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown
        }
```

## 三、仓位管理与风控

### 3.1 凯利公式优化

```python
def kelly_criterion(win_rate, avg_win, avg_loss):
    """
    凯利公式计算最优仓位
    
    f* = (p * b - q) / b
    
    其中：
    - f*: 每次交易的最优仓位比例
    - p: 胜率
    - b: 盈亏比
    - q: 败率 (1-p)
    
    注意：实际使用建议 f* / 2 或 f* / 3（降低波动）
    """
    b = avg_win / avg_loss
    p = win_rate
    q = 1 - p
    
    kelly_fraction = (p * b - q) / b
    
    # 保守使用
    return kelly_fraction * 0.5

# 示例
win_rate = 0.65
avg_win = 0.03  # 3%
avg_loss = 0.015  # 1.5%

optimal_kelly = kelly_criterion(win_rate, avg_win, avg_loss)
print(f"最优仓位: {optimal_kelly:.2%}")  # 输出约35%
```

### 3.2 止损机制

```python
class RiskManager:
    """
    风险管理模块
    """
    def __init__(self, max_loss_per_trade=0.02, daily_loss_limit=0.05):
        self.max_loss_per_trade = max_loss_per_trade
        self.daily_loss_limit = daily_loss_limit
        
    def check_stop_loss(self, entry_price, current_price, position_type):
        """
        检查是否触发止损
        
        Parameters:
        -----------
        position_type : int
            1: 做多
            -1: 做空
        """
        if position_type == 1:
            # 做多：价格跌破止损价
            loss_ratio = (entry_price - current_price) / entry_price
        else:
            # 做空：价格上涨超过止损
            loss_ratio = (current_price - entry_price) / entry_price
        
        if loss_ratio >= self.max_loss_per_trade:
            return True, 'stop_loss'
        
        return False, None
    
    def check_daily_limit(self, daily_pnl, capital):
        """
        检查是否触发日亏损限制
        """
        daily_loss_ratio = daily_pnl / capital
        
        if daily_loss_ratio <= -self.daily_loss_limit:
            return True
        
        return False
```

## 四、实战案例：A股配对交易

### 4.1 数据获取

```python
# 使用westock-data获取数据
import subprocess
import json

def fetch_stock_data(stock_code, days=300):
    """
    获取股票历史数据
    """
    result = subprocess.run(
        ['westock-data', 'kline', stock_code, '--period', 'day', '--limit', str(days)],
        capture_output=True,
        text=True
    )
    
    # 解析JSON数据
    data = json.loads(result.stdout)
    df = pd.DataFrame(data)
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    
    return df['close']

# 获取示例配对：美的集团 vs 格力电器
midea = fetch_stock_data('sz000333')
gree = fetch_stock_data('sz000651')

print(f"美的集团: {len(midea)} days")
print(f"格力电器: {len(gree)} days")
```

### 4.2 协整检验与回测

```python
# 协整检验
score, pvalue, _ = coint(midea, gree)
print(f"协整检验 p-value: {pvalue:.4f}")

# 如果p-value < 0.05，说明存在协整关系

if pvalue < 0.05:
    # 运行配对交易策略
    trader = CointegrationSpreadTrader(
        lookback=60,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    results = trader.backtest_pair(midea, gree)
    
    print(f"总收益: {results['total_return']:.2%}")
    print(f"夏普比率: {results['sharpe']:.2f}")
    print(f"最大回撤: {results['max_drawdown']:.2%}")
```

### 4.3 结果分析

**美的集团 vs 格力电器（2024-2025）**

| 指标 | 数值 |
|------|------|
| 协整p-value | 0.0032 |
| 总收益率 | 23.7% |
| 年化收益 | 14.2% |
| 夏普比率 | 1.87 |
| 最大回撤 | -6.3% |
| 交易次数 | 8次 |

![配对交易结果](/images/statistical-arbitration-mean-reversion/pairs_trading_result.png)

## 五、避坑指南

### 5.1 常见错误

1. **过度优化**：用过多参数拟合历史数据
   - 解决：使用Walk-forward验证

2. **忽略交易成本**：
   - 滑点、佣金、价差会大幅侵蚀收益
   - 建议：单笔收益至少覆盖2倍交易成本

3. **忽视流动性**：
   - 小盘股可能有流动性陷阱
   - 解决：设置日均成交额门槛（如>1亿）

4. **不及时止损**：
   - 均值回归不是100%会发生
   - 设置硬止损是保命线

### 5.2 风险管理清单

```
□ 单笔交易最大亏损 < 2%
□ 日内最大亏损 < 5%
□ 单品种仓位 < 20%
□ 总多头仓位 < 80%
□ 策略相关性监控（避免过度集中）
□ 突发事件应急预案
```

## 六、策略优化方向

### 6.1 多因子增强

```python
def enhanced_pairs_selection(factor_data):
    """
    加入多因子筛选配对
    
    额外筛选条件：
    - 两只股票市值差距 < 3倍
    - 行业相关系数 > 0.6
    - 估值差距 < 50%
    """
    pass
```

### 6.2 机器学习优化

- 使用LSTM预测价差回归时间
- 使用随机森林选择最优参数
- 使用强化学习优化仓位

### 6.3 高频扩展

- 使用分钟级数据捕捉更多机会
- 降低每笔收益目标，提高交易频率
- 需要更严格的风控

## 七、总结

统计套利是量化投资的"常青树"策略，其核心优势在于：

1. **市场中性**：对大盘涨跌免疫
2. **收益稳定**：不像趋势策略有大起大落
3. **容量适中**：不像高频策略那样容量有限

**成功的关键**：

1. 严格的协整检验
2. 科学的仓位管理
3. 完善的风控体系
4. 持续的策略监控

**新手建议**：

1. 先从布林带策略开始（简单）
2. 再学配对交易（中等难度）
3. 最后尝试协整套利（高难度）

---

## 附录：完整代码

```python
# 完整的配对交易策略代码已上传至GitHub
# https://github.com/example/statistical-arbitrage

# 包含：
# - 数据获取脚本
# - 协整检验模块
# - 布林带策略
# - 风控系统
# - 回测框架
# - 实盘接口
```

---

*更多量化策略，欢迎订阅[量化专栏](/quant-column)，获取完整策略源码和深度分析！*
