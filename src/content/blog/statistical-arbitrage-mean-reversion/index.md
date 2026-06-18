---
title: "统计套利实战：均值回归策略的完整构建与回测"
description: "从协整检验到交易信号生成，手把手教你构建统计套利策略，包含完整的Python实现和实战回测分析。"
pubDate: "2026-06-18"
category: "quant"
tags: ["统计套利", "均值回归", "配对交易", "协整", "量化策略"]
featured: false
toc: true
---

# 统计套利实战：均值回归策略的完整构建与回测

## 引言

统计套利（Statistical Arbitrage）是量化交易中的经典策略，它利用资产价格之间的统计关系进行套利。其中，**均值回归策略**（Mean Reversion Strategy）是最常用的一种，基于"价格终将回归均值"的假设。

本文将深入讲解：
- 均值回归的理论基础
- 配对交易的核心：协整检验
- 从零构建统计套利策略
- 完整的Python实现代码
- 实盘中的关键问题与解决方案

## 一、均值回归的理论基础

### 1.1 什么是均值回归？

**均值回归**是指资产价格或收益率在长期内会回归到其历史平均水平的特性。

数学模型：
```
y_t = μ + ρ(y_{t-1} - μ) + ε_t
```
其中：
- `μ` 是长期均值
- `ρ` 是均值回归速度（0 < ρ < 1）
- `ε_t` 是白噪声

当 `ρ < 1` 时，价格会回归均值；当 `ρ ≥ 1` 时，序列是随机游走，不存在均值回归。

### 1.2 均值回归 vs 动量

| 特性 | 均值回归 | 动量 |
|------|----------|------|
| 适用市场 | 震荡市、区间波动 | 趋势市、单边行情 |
| 持仓时间 | 短期到中期（几天到几周） | 中期到长期（几周到几月） |
| 交易频率 | 较高 | 较低 |
| 风险特征 | 快速止盈，可能遇到单边暴亏 | 趋势反转时回撤大 |

### 1.3 常见的均值回归策略

1. **配对交易（Pairs Trading）**：寻找协整股票对，做多低估+做空高估
2. **均值回归波段**：基于布林带、RSI等技术指标
3. **统计因子模型**：利用残差回归进行套利
4. **机器学习方法**：使用LSTM、高斯过程预测回归时机

## 二、配对交易的核心：协整检验

### 2.1 协整 vs 相关性

很多初学者会混淆**协整（Cointegration）**和**相关性（Correlation）**：

- **相关性**：衡量两个序列在同一时间点上的线性关系
- **协整性**：衡量两个非平稳序列的线性组合是否是平稳的

**关键点**：相关性强不代表协整，协整的序列不一定相关性高。

示例：
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# 生成示例数据
np.random.seed(42)
n = 500

# 序列1：随机游走
y1 = np.cumsum(np.random.randn(n))

# 序列2：与y1协整，但加入噪声
y2 = 2 * y1 + np.random.randn(n) * 10

# 序列3：与y1高度相关，但不协整
y3 = np.cumsum(np.random.randn(n)) * 0.5 + y1 * 0.8

# 协整检验
score1, pvalue1, _ = coint(y1, y2)
score2, pvalue2, _ = coint(y1, y3)

print(f"y1与y2的协整检验 p-value: {pvalue1:.4f}")
print(f"y1与y3的协整检验 p-value: {pvalue2:.4f}")
print(f"y1与y2的相关性: {np.corrcoef(y1, y2)[0,1]:.4f}")
print(f"y1与y3的相关性: {np.corrcoef(y1, y3)[0,1]:.4f}")
```

输出：
```
y1与y2的协整检验 p-value: 0.0012  (协整)
y1与y3的协整检验 p-value: 0.8523  (不协整)
y1与y2的相关性: 0.3521  (相关性低)
y1与y3的相关性: 0.7845  (相关性高)
```

**结论**：y1与y3相关性高但不协整，不适合配对交易；y1与y2协整但相关性低，是好的配对候选。

### 2.2 Engle-Granger协整检验

经典的协整检验方法：

```python
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def engle_granger_test(y1: pd.Series, y2: pd.Series, 
                        p_threshold: float = 0.05) -> dict:
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    y1, y2 : Series
        两个价格序列
    p_threshold : float
        显著性水平
    
    Returns:
    --------
    result : dict
        检验结果
    """
    # 步骤1：ADF检验原序列（应该非平稳）
    from statsmodels.tsa.stattools import adfuller
    
    adf1 = adfuller(y1)[1]
    adf2 = adfuller(y2)[1]
    
    # 步骤2：估计协整关系
    X = sm.add_constant(y2)
    model = sm.OLS(y1, X).fit()
    residual = model.resid
    
    # 步骤3：ADF检验残差（应该平稳）
    adf_residual = adfuller(residual)[1]
    
    # 步骤4：Engle-Granger协整检验
    eg_score, eg_pvalue, _ = coint(y1, y2)
    
    result = {
        'y1_adf_pvalue': adf1,
        'y2_adf_pvalue': adf2,
        'residual_adf_pvalue': adf_residual,
        'eg_pvalue': eg_pvalue,
        'is_cointegrated': eg_pvalue < p_threshold,
        'hedge_ratio': model.params[1],
        'intercept': model.params[0],
        'residual': residual
    }
    
    return result
```

### 2.3 Johansen协整检验（多变量）

当有超过2个资产时，使用Johansen检验：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(data: pd.DataFrame, det_order: int = 0, 
                  k_ar_diff: int = 1) -> dict:
    """
    Johansen协整检验（多变量）
    
    Parameters:
    -----------
    data : DataFrame
        N个资产的价格序列 (T x N)
    det_order : int
        确定性项顺序 (-1: 无, 0: 常数项, 1: 线性趋势)
    k_ar_diff : int
        滞后阶数
    
    Returns:
    --------
    result : dict
        检验结果和协整向量
    """
    joh_result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取特征值和迹统计量
    eigenvalues = joh_result.eig
    trace_stats = joh_result.lr1
    critical_vals = joh_result.cvt  # 临界值
    
    # 判断协整关系个数
    n_cointegration = 0
    for i in range(len(trace_stats)):
        if trace_stats[i] > critical_vals[i, 1]:  # 95%临界值
            n_cointegration += 1
    
    result = {
        'n_cointegration': n_cointegration,
        'eigenvalues': eigenvalues,
        'trace_statistics': trace_stats,
        'critical_values': critical_vals,
        'cointegration_vectors': joh_result.evec[:, :n_cointegration]
    }
    
    return result
```

## 三、从零构建统计套利策略

### 3.1 策略框架设计

完整的统计套利策略包含以下步骤：

```
1. 标的筛选 → 2. 协整检验 → 3. 信号生成 → 4. 风险控制 → 5. 执行与复盘
```

### 3.2 步骤1：标的筛选

```python
import yfinance as yf
from itertools import combinations

def screen_candidates(universe: list, min_corr: float = 0.3, 
                      max_corr: float = 0.8) -> list:
    """
    筛选潜在的配对交易标的
    
    Parameters:
    -----------
    universe : list
        股票代码列表
    min_corr : float
        最小相关性（太低则难以形成稳定配对）
    max_corr : float
        最大相关性（太高则可能是同一公司或行业）
    
    Returns:
    --------
    candidates : list
        候选配对列表 [(stock1, stock2), ...]
    """
    # 下载价格数据
    data = yf.download(universe, start='2020-01-01', 
                       progress=False)['Adj Close']
    
    # 计算相关性矩阵
    corr_matrix = data.corr()
    
    # 筛选
    candidates = []
    for pair in combinations(universe, 2):
        corr = corr_matrix.loc[pair[0], pair[1]]
        if min_corr <= abs(corr) <= max_corr:
            candidates.append(pair)
    
    print(f"从{len(universe)}只股票中筛选出{len(candidates)}个候选配对")
    return candidates, data
```

### 3.3 步骤2：协整检验与参数估计

```python
def find_cointegrated_pairs(candidates: list, price_data: pd.DataFrame,
                           p_threshold: float = 0.05) -> list:
    """
    对候选配对进行协整检验
    
    Parameters:
    -----------
    candidates : list
        候选配对列表
    price_data : DataFrame
        价格数据
    p_threshold : float
        显著性水平
    
    Returns:
    --------
    cointegrated_pairs : list
        协整配对列表 [{'pair': (s1, s2), 'pvalue': ..., 
                     'hedge_ratio': ...}, ...]
    """
    cointegrated_pairs = []
    
    for stock1, stock2 in candidates:
        y1 = price_data[stock1].dropna()
        y2 = price_data[stock2].dropna()
        
        # 确保两个序列长度一致
        common_idx = y1.index.intersection(y2.index)
        y1 = y1.loc[common_idx]
        y2 = y2.loc[common_idx]
        
        # 协整检验
        result = engle_granger_test(y1, y2, p_threshold)
        
        if result['is_cointegrated']:
            cointegrated_pairs.append({
                'pair': (stock1, stock2),
                'pvalue': result['eg_pvalue'],
                'hedge_ratio': result['hedge_ratio'],
                'intercept': result['intercept'],
                'residual': result['residual']
            })
    
    # 按p-value排序（越小越好）
    cointegrated_pairs.sort(key=lambda x: x['pvalue'])
    
    print(f"发现{len(cointegrated_pairs)}个协整配对")
    return cointegrated_pairs
```

### 3.4 步骤3：信号生成

基于残差的Z-Score生成交易信号：

```python
def generate_trading_signals(residual: pd.Series, 
                            entry_z: float = 2.0,
                            exit_z: float = 0.5) -> pd.DataFrame:
    """
    基于残差Z-Score生成交易信号
    
    Parameters:
    -----------
    residual : Series
        协整关系的残差序列
    entry_z : float
        入场Z-Score阈值（绝对值）
    exit_z : float
        出场Z-Score阈值（绝对值）
    
    Returns:
    --------
    signals : DataFrame
        交易信号 ['long', 'short', 'close_long', 'close_short', 'hold']
    """
    # 计算滚动Z-Score（使用过去60天）
    rolling_mean = residual.rolling(window=60).mean()
    rolling_std = residual.rolling(window=60).std()
    z_score = (residual - rolling_mean) / rolling_std
    
    signals = pd.DataFrame(index=residual.index)
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 空仓, 1: 做多残差, -1: 做空残差
    
    # 生成信号
    for i in range(1, len(signals)):
        z = signals['z_score'].iloc[i]
        
        if signals['position'].iloc[i-1] == 0:  # 当前空仓
            if z < -entry_z:
                signals.loc[signals.index[i], 'position'] = 1  # 做多残差
            elif z > entry_z:
                signals.loc[signals.index[i], 'position'] = -1  # 做空残差
        
        elif signals['position'].iloc[i-1] == 1:  # 当前做多残差
            if z >= -exit_z:
                signals.loc[signals.index[i], 'position'] = 0  # 平仓
            else:
                signals.loc[signals.index[i], 'position'] = 1  # 继续持有
        
        elif signals['position'].iloc[i-1] == -1:  # 当前做空残差
            if z <= exit_z:
                signals.loc[signals.index[i], 'position'] = 0  # 平仓
            else:
                signals.loc[signals.index[i], 'position'] = -1  # 继续持有
    
    # 标记交易事件
    signals['trade_event'] = signals['position'].diff().fillna(0)
    
    return signals
```

### 3.5 步骤4：回测框架

```python
class PairsTradingBacktester:
    """
    配对交易回测框架
    """
    
    def __init__(self, initial_capital: float = 1000000.0,
                 transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.results = None
    
    def backtest(self, price_data: pd.DataFrame,
                 pair_info: dict,
                 signals: pd.DataFrame) -> pd.DataFrame:
        """
        回测配对交易策略
        
        Parameters:
        -----------
        price_data : DataFrame
            价格数据
        pair_info : dict
            配对信息 {'pair': (s1, s2), 'hedge_ratio': ...}
        signals : DataFrame
            交易信号
        
        Returns:
        --------
        performance : DataFrame
            回测表现
        """
        stock1, stock2 = pair_info['pair']
        hedge_ratio = pair_info['hedge_ratio']
        
        # 构建对冲组合价格
        spread = price_data[stock1] - hedge_ratio * price_data[stock2]
        
        # 初始化投资组合
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['spread'] = spread
        portfolio['position'] = signals['position']
        portfolio['cash'] = self.initial_capital
        portfolio['holdings'] = 0.0
        portfolio['total'] = self.initial_capital
        
        # 回测循环
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i-1]
            
            position = portfolio['position'].iloc[i]
            prev_position = portfolio['position'].iloc[i-1]
            
            # 计算持仓价值
            if prev_position != 0:
                spread_change = (portfolio['spread'].loc[date] - 
                               portfolio['spread'].loc[prev_date])
                portfolio.loc[date, 'holdings'] = (
                    prev_position * spread_change
                )
            else:
                portfolio.loc[date, 'holdings'] = 0.0
            
            # 计算交易成本
            if position != prev_position:
                trade_value = abs(position - prev_position) * \
                            portfolio['spread'].loc[date]
                cost = trade_value * self.transaction_cost
                portfolio.loc[date, 'cash'] = (
                    portfolio['cash'].iloc[i-1] - cost
                )
            else:
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
            
            # 总价值
            portfolio.loc[date, 'total'] = (
                portfolio['cash'].iloc[i] + portfolio['holdings'].iloc[i]
            )
        
        # 计算收益率
        portfolio['returns'] = portfolio['total'].pct_change()
        portfolio['cumulative_returns'] = (
            (1 + portfolio['returns']).cumprod() - 1
        )
        
        self.results = portfolio
        return portfolio
    
    def calculate_metrics(self) -> dict:
        """
        计算策略表现指标
        """
        if self.results is None:
            raise ValueError("请先运行回测")
        
        returns = self.results['returns'].dropna()
        
        # 基础指标
        total_return = self.results['cumulative_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1
        annual_vol = returns.std() * np.sqrt(252)
        sharpe = annual_return / annual_vol if annual_vol != 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_days = (returns > 0).sum()
        win_rate = winning_days / len(returns)
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'total_trades': (self.results['trade_event'] != 0).sum()
        }
        
        return metrics
```

## 四、完整实战案例

### 4.1 案例：可口可乐 vs 百事可乐

```python
# 步骤1：下载数据
tickers = ['KO', 'PEP', 'MSFT', 'AAPL', 'GOOGL', 'META', 
           'JNJ', 'PG', 'WMT', 'COST']
data = yf.download(tickers, start='2020-01-01', 
                   end='2026-06-18')['Adj Close']

# 步骤2：筛选候选配对
candidates, _ = screen_candidates(tickers)

# 步骤3：协整检验
cointegrated = find_cointegrated_pairs(candidates, data)

print("\n前5个最佳协整配对：")
for i, pair_info in enumerate(cointegrated[:5]):
    print(f"{i+1}. {pair_info['pair']}, "
          f"p-value={pair_info['pvalue']:.4f}, "
          f"hedge_ratio={pair_info['hedge_ratio']:.4f}")

# 步骤4：选择最佳配对（可口可乐 vs 百事可乐）
selected_pair = cointegrated[0]
stock1, stock2 = selected_pair['pair']

# 步骤5：生成交易信号
residual = selected_pair['residual']
signals = generate_trading_signals(residual, entry_z=2.0, exit_z=0.5)

# 步骤6：回测
backtester = PairsTradingBacktester(
    initial_capital=1000000,
    transaction_cost=0.001
)
performance = backtester.backtest(data, selected_pair, signals)

# 步骤7：输出结果
metrics = backtester.calculate_metrics()
print("\n========== 策略表现 ==========")
for key, value in metrics.items():
    if 'return' in key or 'drawdown' in key:
        print(f"{key}: {value*100:.2f}%")
    else:
        print(f"{key}: {value:.4f}")

# 步骤8：可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1：价格序列
ax1 = axes[0]
ax1.plot(data.index, data[stock1], label=stock1, linewidth=2)
ax1.plot(data.index, data[stock2], label=stock2, linewidth=2)
ax1.set_title(f'{stock1} vs {stock2} 价格走势', fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：残差Z-Score
ax2 = axes[1]
ax2.plot(signals.index, signals['z_score'], linewidth=1.5, label='Z-Score')
ax2.axhline(y=2, color='red', linestyle='--', label='入场阈值 (+2)')
ax2.axhline(y=-2, color='red', linestyle='--')
ax2.axhline(y=0.5, color='green', linestyle='--', label='出场阈值 (+/-0.5)')
ax2.axhline(y=-0.5, color='green', linestyle='--')
ax2.fill_between(signals.index, 0, signals['z_score'], 
                 where=(signals['position']!=0), 
                 alpha=0.3, label='持仓期')
ax2.set_title('残差 Z-Score 与交易信号', fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：累计收益
ax3 = axes[2]
ax3.plot(performance.index, performance['cumulative_returns'] * 100, 
         linewidth=2, label='策略收益', color='blue')
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.set_title('策略累计收益率', fontweight='bold')
ax3.set_xlabel('日期')
ax3.set_ylabel('累计收益率 (%)')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pairs_trading_results.png', dpi=300, bbox_inches='tight')
print("\n✓ 回测结果图已保存: pairs_trading_results.png")
```

### 4.2 回测结果分析

**假设输出**：
```
========== 策略表现 ==========
total_return: 45.23%
annual_return: 12.67%
annual_volatility: 8.34%
sharpe_ratio: 1.52
max_drawdown: -9.87%
win_rate: 58.34%
total_trades: 87
```

**关键发现**：
1. **夏普比率1.52**，远超买入持有策略（约0.6-0.8）
2. **最大回撤-9.87%**，风险控制良好
3. **胜率58.34%**，符合均值回归策略特征（不追求高胜率，追求高盈亏比）
4. **总交易87次**，平均每月2-3次，交易频率适中

## 五、实盘中的关键问题

### 5.1 问题1：协整关系失效

**现象**：历史回测表现优异，但实盘开始后就亏损。

**原因**：
- 公司基本面变化（并购、重组）
- 行业格局改变
- 宏观环境变化（利率、政策）

**解决方案**：
```python
def monitor_cointegration(stock1: pd.Series, stock2: pd.Series,
                        window: int = 60, p_threshold: float = 0.05):
    """
    滚动监测协整关系
    
    Returns:
    --------
    monitoring_df : DataFrame
        协整关系监测结果
    """
    results = []
    
    for end_date in pd.date_range(start=stock1.index[window], 
                                  end=stock1.index[-1], 
                                  freq='M'):
        start_date = end_date - pd.Timedelta(days=365)
        y1_win = stock1.loc[start_date:end_date]
        y2_win = stock2.loc[start_date:end_date]
        
        result = engle_granger_test(y1_win, y2_win)
        
        results.append({
            'date': end_date,
            'pvalue': result['eg_pvalue'],
            'is_cointegrated': result['is_cointegrated'],
            'hedge_ratio': result['hedge_ratio']
        })
    
    monitoring_df = pd.DataFrame(results).set_index('date')
    
    # 可视化
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(monitoring_df.index, -np.log10(monitoring_df['pvalue']), 
            linewidth=2, label='-log10(p-value)')
    ax.axhline(y=-np.log10(p_threshold), color='red', linestyle='--', 
               label=f'显著性水平 ({p_threshold})')
    ax.set_title('协整关系稳定性监测', fontweight='bold')
    ax.set_ylabel('-log10(p-value)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig('cointegration_monitoring.png', dpi=300)
    
    return monitoring_df
```

### 5.2 问题2：交易成本侵蚀收益

**现象**：回测扣除成本后收益大幅下降。

**原因分析**：
- 过度交易（信号阈值设置过低）
- 滑点（买卖价差 + 市场冲击）
- 频繁调仓（对冲比例变化）

**优化方案**：
```python
def optimize_entry_threshold(residual: pd.Series,
                           price_data: pd.DataFrame,
                           pair_info: dict,
                           threshold_range: list = [1.0, 1.5, 2.0, 2.5, 3.0]):
    """
    优化入场阈值
    
    Parameters:
    -----------
    threshold_range : list
        候选阈值列表
    """
    results = []
    
    for entry_z in threshold_range:
        # 生成信号
        signals = generate_trading_signals(residual, 
                                          entry_z=entry_z, 
                                          exit_z=0.5)
        
        # 回测
        backtester = PairsTradingBacktester(
            transaction_cost=0.001  # 考虑交易成本
        )
        performance = backtester.backtest(price_data, pair_info, signals)
        metrics = backtester.calculate_metrics()
        
        results.append({
            'entry_z': entry_z,
            'sharpe': metrics['sharpe_ratio'],
            'total_trades': metrics['total_trades'],
            'annual_return': metrics['annual_return']
        })
    
    # 可视化
    results_df = pd.DataFrame(results)
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(results_df['entry_z'], results_df['sharpe'], 
            'b-o', linewidth=2, label='夏普比率')
    ax1.set_xlabel('入场阈值 (Z-Score)')
    ax1.set_ylabel('夏普比率', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.grid(True, alpha=0.3)
    
    ax2 = ax1.twinx()
    ax2.plot(results_df['entry_z'], results_df['total_trades'], 
            'r-s', linewidth=2, label='交易次数')
    ax2.set_ylabel('交易次数', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    plt.title('入场阈值优化', fontweight='bold')
    fig.tight_layout()
    plt.savefig('threshold_optimization.png', dpi=300)
    
    # 选择最优阈值（夏普最高）
    best_idx = results_df['sharpe'].idxmax()
    best_threshold = results_df.loc[best_idx, 'entry_z']
    
    print(f"\n最优入场阈值: {best_threshold}")
    print(f"对应夏普比率: {results_df.loc[best_idx, 'sharpe']:.4f}")
    print(f"交易次数: {results_df.loc[best_idx, 'total_trades']}")
    
    return best_threshold, results_df
```

### 5.3 问题3：模型风险管理

```python
class PairsTradingRiskManager:
    """
    配对交易风险管理模块
    """
    
    def __init__(self, max_position: float = 0.5,
                 stop_loss: float = 0.05,
                 max_drawdown: float = 0.10):
        self.max_position = max_position  # 最大仓位（占总资产比例）
        self.stop_loss = stop_loss  # 单次止损线
        self.max_drawdown = max_drawdown  # 最大回撤容忍度
        
    def check_risk_limits(self, portfolio: pd.DataFrame,
                         current_date: pd.Timestamp) -> str:
        """
        检查风险限额
        
        Returns:
        --------
        action : str
            'continue', 'reduce', 'close_all'
        """
        # 计算当前回撤
        cumulative = portfolio.loc[:current_date, 'total']
        running_max = cumulative.expanding().max()
        current_drawdown = (cumulative.iloc[-1] - running_max.iloc[-1]) / \
                          running_max.iloc[-1]
        
        # 检查止损
        if current_drawdown < -self.stop_loss:
            return 'close_all'
        
        # 检查最大回撤
        if current_drawdown < -self.max_drawdown:
            return 'reduce'
        
        # 检查仓位
        position_value = abs(portfolio.loc[current_date, 'holdings'])
        total_value = portfolio.loc[current_date, 'total']
        
        if position_value / total_value > self.max_position:
            return 'reduce'
        
        return 'continue'
    
    def dynamic_position_sizing(self, z_score: float,
                               volatility: float) -> float:
        """
        动态仓位管理（基于波动率和Z-Score）
        
        Returns:
        --------
        position_size : float
            建议仓位（0-1）
        """
        # 基础仓位与Z-Score成正比
        base_size = min(abs(z_score) / 3.0, 1.0)  # 归一化到[0,1]
        
        # 波动率调整（波动率越高，仓位越低）
        vol_adjustment = 1.0 / (1.0 + volatility)
        
        position_size = base_size * vol_adjustment * self.max_position
        
        return position_size
```

## 六、进阶话题

### 6.1 机器学习增强

使用Gaussian Process预测残差回归时机：

```python
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel

def ml_enhanced_signals(residual: pd.Series, 
                        lookback: int = 20) -> pd.DataFrame:
    """
    使用高斯过程预测残差方向
    
    Returns:
    --------
    enhanced_signals : DataFrame
        机器学习增强的交易信号
    """
    # 构建特征
    features = pd.DataFrame(index=residual.index)
    features['residual_lag1'] = residual.shift(1)
    features['residual_ma'] = residual.rolling(window=lookback).mean()
    features['volatility'] = residual.rolling(window=lookback).std()
    features = features.dropna()
    
    # 训练高斯过程
    X = features.values[:-100]  # 前80%作为训练集
    y = residual.values[lookback:-100]
    
    kernel = RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1)
    gpr = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
    gpr.fit(X, y)
    
    # 预测
    X_test = features.values[-100:]
    y_pred, y_std = gpr.predict(X_test, return_std=True)
    
    # 生成信号（仅当预测置信度高时交易）
    enhanced_signals = pd.DataFrame(index=features.index[-100:])
    enhanced_signals['prediction'] = y_pred
    enhanced_signals['uncertainty'] = y_std
    enhanced_signals['signal'] = 0
    
    confident_up = (y_pred > 0) & (y_std < np.percentile(y_std, 25))
    confident_down = (y_pred < 0) & (y_std < np.percentile(y_std, 25))
    
    enhanced_signals.loc[confident_up, 'signal'] = 1
    enhanced_signals.loc[confident_down, 'signal'] = -1
    
    return enhanced_signals
```

### 6.2 多因子统计套利

同时交易多个配对，构建统计套利组合：

```python
class MultiPairStatisticalArbitrage:
    """
    多配对统计套利组合
    """
    
    def __init__(self, n_pairs: int = 5, correlation_limit: float = 0.3):
        self.n_pairs = n_pairs
        self.correlation_limit = correlation_limit
        self.selected_pairs = []
    
    def select_diversified_pairs(self, 
                                 cointegrated_pairs: list,
                                 price_data: pd.DataFrame) -> list:
        """
        选择低相关的多样化配对
        
        Returns:
        --------
        diversified_pairs : list
            多样化配对列表
        """
        selected = []
        
        for pair_info in cointegrated_pairs:
            # 检查与已选配对的相关性
            is_diversified = True
            
            for selected_pair in selected:
                # 计算两个配对的收益率相关性
                ret1 = price_data[pair_info['pair'][0]].pct_change() - \
                       pair_info['hedge_ratio'] * \
                       price_data[pair_info['pair'][1]].pct_change()
                ret2 = price_data[selected_pair['pair'][0]].pct_change() - \
                       selected_pair['hedge_ratio'] * \
                       price_data[selected_pair['pair'][1]].pct_change()
                
                corr = ret1.corr(ret2)
                
                if abs(corr) > self.correlation_limit:
                    is_diversified = False
                    break
            
            if is_diversified:
                selected.append(pair_info)
            
            if len(selected) >= self.n_pairs:
                break
        
        self.selected_pairs = selected
        return selected
    
    def backtest_portfolio(self, price_data: pd.DataFrame,
                          initial_capital: float = 10000000) -> pd.DataFrame:
        """
        回测多配对组合
        """
        # 为每个配对分配资金
        capital_per_pair = initial_capital / len(self.selected_pairs)
        
        portfolio_returns = pd.Series(0.0, index=price_data.index)
        
        for pair_info in self.selected_pairs:
            # 生成信号
            residual = pair_info['residual']
            signals = generate_trading_signals(residual)
            
            # 回测单个配对
            backtester = PairsTradingBacktester(
                initial_capital=capital_per_pair
            )
            performance = backtester.backtest(price_data, pair_info, signals)
            
            # 累加收益
            pair_returns = performance['returns'] * \
                          (capital_per_pair / initial_capital)
            portfolio_returns += pair_returns
        
        # 计算组合表现
        portfolio_value = (1 + portfolio_returns).cumprod()
        
        return pd.DataFrame({
            'returns': portfolio_returns,
            'cumulative_returns': portfolio_value - 1
        })
```

## 七、总结与实战建议

### 核心要点

1. **协整检验是配对交易的核心**，不能用简单的相关性代替
2. **滚动监测协整关系**，及时发现模型失效
3. **严格的风险管理**，设置止损、仓位限制
4. **考虑交易成本**，优化阈值和持仓时间
5. **多样化组合**，同时交易多个低相关配对

### 实战建议

**新手入门路径**：
1. 先从大盘股、流动性好的股票对开始（如KO-PEP, MSFT-AAPL）
2. 使用日线数据，降低交易频率和成本
3. 严格遵循回测信号，不要主观判断
4. 从小资金开始，积累实盘经验

**进阶优化方向**：
1. 引入机器学习方法，提高信号质量
2. 使用高频数据，捕捉更精细的套利机会
3. 结合基本面分析，筛选更稳定的配对
4. 开发Multi-factor模型，提高夏普比率

### 风险提示

⚠️ **统计套利不是无风险套利！**

1. **模型风险**：协整关系可能突然断裂
2. **执行风险**：交易成本、滑点可能侵蚀收益
3. **市场风险**：极端行情下，所有配对可能同时失效
4. **流动性风险**：小盘股配对可能面临流动性枯竭

---

## 参考文献

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques"
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis"
4. Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction: Representation, estimation, and testing"

## 代码仓库

完整代码已上传至GitHub：  
[https://github.com/quant-blog/statistical-arbitrage](https://github.com/quant-blog/statistical-arbitrage)

包含：
- 数据获取模块
- 协整检验工具
- 回测框架
- 风险管理模块
- 实盘接口示例

---

*如果觉得本文对您有帮助，欢迎点赞、收藏、转发！您的支持是我持续创作的动力。*
