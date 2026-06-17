---
title: 统计套利：均值回归策略
description: 深入探讨统计套利的核心理论与实践方法，聚焦均值回归策略的构建、协整检验、配对交易实操，以及风险控制体系。包含完整的Python实现代码和实战案例。
publishDate: 2026-06-17
tags:
 - 统计套利, 均值回归, 配对交易, 协整分析, 市场中性
language: Chinese
---

**统计套利（Statistical Arbitrage）**是量化投资中历史最悠久、理论最完备的策略之一。其核心思想是利用资产价格的统计关系（如协整、相关性），构建市场中性组合，从价格的均值回归中获取稳定收益。

本文将系统介绍统计套利的理论基础、配对交易实务、Python实现，以及风险管理要点。

## 统计套利的核心逻辑

### 什么是均值回归？

均值回归（Mean Reversion）是指资产价格或收益率在长期中会回归其均衡水平的现象。学术研究表明：

1. **短期反转效应**：过去表现差的股票在未来短期会反弹
2. **波动率聚集**：高波动期后会回归正常水平
3. **价差收敛**：成对交易的资产价差会围绕均值波动

![均值回归示意](/images/statistical-arbitrage-mean-reversion/mean_reversion_diagram.png)

*图1：典型均值回归过程（奥恩斯坦-乌伦贝克过程）*

### 统计套利 vs 传统套利

| 维度 | 传统套利 | 统计套利 |
|------|---------|---------|
| 定价依据 | 严格的无套利原理 | 统计关系（协整、相关性） |
| 利润确定性 | 几乎无风险 | 概率性盈利 |
| 持仓时间 | 很短（秒到天） | 中等（天到月） |
| 资金容量 | 受限于套利机会 | 较大 |
| 模型风险 | 低 | 高（模型失效风险） |

## 配对交易：统计套利的经典范式

### 理论基础：协整关系

两资产价格 $X_t$ 和 $Y_t$ 如果存在协整关系，则：

$$
Y_t = \alpha + \beta X_t + \epsilon_t
$$

其中 $\epsilon_t$ 是平稳过程（均值回归）。

**关键检验**：
1. **单位根检验**（ADF Test）：检验残差是否平稳
2. **协整检验**（Johansen Test）：多变量协整关系
3. **半衰期检验**：回归速度是否合理

### Python实现：寻找协整对

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PairsTradingBacktester:
    """
    配对交易回测框架
    """
    def __init__(self, start_date='2015-01-01', end_date='2025-12-31'):
        self.start_date = start_date
        self.end_date = end_date
        self.pairs = []
        self.data = {}
        
    def load_data(self, symbols):
        """
        加载股票数据
        """
        print(f"正在下载 {len(symbols)} 只股票的数据...")
        
        for symbol in symbols:
            try:
                stock = yf.Ticker(symbol)
                df = stock.history(start=self.start_date, end=self.end_date)
                
                if len(df) > 0:
                    self.data[symbol] = df['Close']
                    print(f"  ✅ {symbol}: {len(df)} 个交易日")
                else:
                    print(f"  ⚠️  {symbol}: 无数据")
                    
            except Exception as e:
                print(f"  ❌ {symbol}: 下载失败 - {e}")
        
        # 合并为DataFrame
        self.price_data = pd.DataFrame(self.data)
        self.price_data.index = pd.to_datetime(self.price_data.index)
        
        print(f"\n数据加载完成，共 {self.price_data.shape[0]} 个交易日，"
              f"{self.price_data.shape[1]} 只股票")
        
        return self.price_data
    
    def find_cointegrated_pairs(self, p_value_threshold=0.05):
        """
        寻找协整对的完整实现
        """
        symbols = self.price_data.columns.tolist()
        n = len(symbols)
        
        # 存储结果
        p_values = np.ones((n, n))
        t_values = np.zeros((n, n))
        pairs = []
        
        print(f"\n开始协整检验（共 {n*(n-1)//2} 对组合）...")
        print("进度: ", end='')
        
        for i in range(n):
            for j in range(i+1, n):
                # 获取价格序列
                S1 = self.price_data[symbols[i]].dropna()
                S2 = self.price_data[symbols[j]].dropna()
                
                # 对齐日期
                S1, S2 = S1.align(S2, join='inner')
                
                if len(S1) < 100:  # 至少需要100个观测
                    continue
                
                # 协整检验
                try:
                    result = coint(S1, S2, trend='c', autolag='aic')
                    p_value = result[1]
                    t_value = result[0]
                    
                    p_values[i, j] = p_value
                    t_values[i, j] = t_value
                    
                    # 如果p值小于阈值，记录为协整对
                    if p_value < p_value_threshold:
                        pairs.append({
                            'stock1': symbols[i],
                            'stock2': symbols[j],
                            'p_value': p_value,
                            't_value': t_value
                        })
                        print(f"✓", end='', flush=True)
                    else:
                        print(f".", end='', flush=True)
                        
                except Exception as e:
                    print(f"!", end='', flush=True)
                    continue
        
        print(f"\n\n✅ 协整检验完成")
        print(f"   找到 {len(pairs)} 对协整关系（p < {p_value_threshold}）")
        
        # 按p值排序
        pairs = sorted(pairs, key=lambda x: x['p_value'])
        self.pairs = pairs
        
        return pairs, p_values
    
    def calculate_spread(self, stock1, stock2):
        """
        计算价差（对冲比例调整）
        """
        S1 = self.price_data[stock1].dropna()
        S2 = self.price_data[stock2].dropna()
        S1, S2 = S1.align(S2, join='inner')
        
        # OLS回归：S1 = alpha + beta * S2 + epsilon
        X = S2.values.reshape(-1, 1)
        y = S1.values
        model = OLS(y, X).fit()
        beta = model.params[0]
        
        # 计算价差
        spread = S1 - beta * S2
        spread.name = f'{stock1}-{stock2}'
        
        # 计算z-score
        z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
        z_score.name = 'z_score'
        
        return spread, z_score, beta
    
    def backtest_pair(self, stock1, stock2, 
                     entry_z=2.0, exit_z=0.5,
                     stop_z=3.0, hold_period=20):
        """
        回测单对配对交易策略
        
        参数：
        - entry_z: 入场Z-score阈值
        - exit_z: 出场Z-score阈值
        - stop_z: 止损Z-score阈值
        - hold_period: 最大持仓周期（交易日）
        """
        # 计算价差和z-score
        spread, z_score, beta = self.calculate_spread(stock1, stock2)
        
        # 初始化变量
        n = len(z_score)
        position = np.zeros(n)  # 持仓方向：1表示做多价差，-1表示做空价差
        entry_price = np.zeros(n)
        pnl = np.zeros(n)
        
        in_position = False
        position_type = 0
        entry_idx = 0
        
        for i in range(1, n):
            if not in_position:
                # 检查入场信号
                if z_score.iloc[i] > entry_z:
                    # Z-score过高，做空价差（卖出stock1，买入stock2）
                    position[i] = -1
                    position_type = -1
                    in_position = True
                    entry_idx = i
                    entry_price[i] = spread.iloc[i]
                    
                elif z_score.iloc[i] < -entry_z:
                    # Z-score过低，做多价差（买入stock1，卖出stock2）
                    position[i] = 1
                    position_type = 1
                    in_position = True
                    entry_idx = i
                    entry_price[i] = spread.iloc[i]
                    
                else:
                    position[i] = 0
                    
            else:
                # 已持仓，检查出场信号
                if position_type == 1:
                    # 做多价差，检查出场条件
                    if z_score.iloc[i] >= -exit_z:  # 回归到均值附近
                        pnl[i] = spread.iloc[i] - entry_price[entry_idx]
                        position[i] = 0
                        in_position = False
                    elif z_score.iloc[i] < -stop_z:  # 止损
                        pnl[i] = spread.iloc[i] - entry_price[entry_idx]
                        position[i] = 0
                        in_position = False
                        print(f"      ⚠️  止损出场: {stock1}-{stock2} @ {z_score.iloc[i]:.2f}")
                    elif i - entry_idx >= hold_period:  # 超时出场
                        pnl[i] = spread.iloc[i] - entry_price[entry_idx]
                        position[i] = 0
                        in_position = False
                        print(f"      ⏰ 超时出场: {stock1}-{stock2}")
                    else:
                        position[i] = position_type
                        
                elif position_type == -1:
                    # 做空价差，检查出场条件
                    if z_score.iloc[i] <= exit_z:  # 回归到均值附近
                        pnl[i] = entry_price[entry_idx] - spread.iloc[i]
                        position[i] = 0
                        in_position = False
                    elif z_score.iloc[i] > stop_z:  # 止损
                        pnl[i] = entry_price[entry_idx] - spread.iloc[i]
                        position[i] = 0
                        in_position = False
                        print(f"      ⚠️  止损出场: {stock1}-{stock2} @ {z_score.iloc[i]:.2f}")
                    elif i - entry_idx >= hold_period:  # 超时出场
                        pnl[i] = entry_price[entry_idx] - spread.iloc[i]
                        position[i] = 0
                        in_position = False
                        print(f"      ⏰ 超时出场: {stock1}-{stock2}")
                    else:
                        position[i] = position_type
        
        # 如果最后仍持仓，强制平仓
        if in_position:
            if position_type == 1:
                pnl[-1] = spread.iloc[-1] - entry_price[entry_idx]
            else:
                pnl[-1] = entry_price[entry_idx] - spread.iloc[-1]
        
        # 计算累积收益
        cumulative_pnl = np.cumsum(pnl)
        
        # 计算绩效指标
        total_trades = np.sum(np.abs(np.diff(position)) > 0)
        winning_trades = np.sum(pnl > 0)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        total_return = cumulative_pnl[-1] / spread.mean()
        sharpe_ratio = np.mean(pnl) / np.std(pnl) * np.sqrt(252) if np.std(pnl) > 0 else 0
        
        # 最大回撤
        cumulative = pd.Series(cumulative_pnl)
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        results = {
            'stock1': stock1,
            'stock2': stock2,
            'beta': beta,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'pnl': pnl,
            'cumulative_pnl': cumulative_pnl,
            'z_score': z_score,
            'spread': spread
        }
        
        return results
    
    def visualize_pair(self, results):
        """
        可视化配对交易结果
        """
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        # 1. Z-score时序图
        ax1 = axes[0]
        ax1.plot(results['z_score'].index, results['z_score'].values, 
                linewidth=1.5, color='#3498DB')
        ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax1.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='入场阈值')
        ax1.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
        ax1.axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='出场阈值')
        ax1.axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
        ax1.fill_between(results['z_score'].index, -2, 2, alpha=0.1, color='gray')
        ax1.set_title(f'Z-Score时序 - {results["stock1"]} & {results["stock2"]}', 
                     fontsize=14, fontweight='bold')
        ax1.set_ylabel('Z-Score')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 价差时序图
        ax2 = axes[1]
        ax2.plot(results['spread'].index, results['spread'].values, 
                linewidth=1.5, color='#E74C3C')
        rolling_mean = results['spread'].rolling(60).mean()
        ax2.plot(rolling_mean.index, rolling_mean.values, 
                linewidth=2, color='blue', label='60日均值')
        ax2.set_title('价差时序', fontsize=14, fontweight='bold')
        ax2.set_ylabel('价差')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. 累积收益曲线
        ax3 = axes[2]
        ax3.plot(results['cumulative_pnl'], linewidth=2, color='#27AE60')
        ax3.fill_between(range(len(results['cumulative_pnl'])), 
                        0, results['cumulative_pnl'], 
                        where=(results['cumulative_pnl'] >= 0), 
                        alpha=0.3, color='green')
        ax3.fill_between(range(len(results['cumulative_pnl'])), 
                        0, results['cumulative_pnl'], 
                        where=(results['cumulative_pnl'] < 0), 
                        alpha=0.3, color='red')
        ax3.set_title('累积收益', fontsize=14, fontweight='bold')
        ax3.set_xlabel('交易日')
        ax3.set_ylabel('累积收益')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'public/images/statistical-arbitrage-mean-reversion/backtest_{results["stock1"]}_{results["stock2"]}.png', 
                   dpi=300, bbox_inches='tight')
        print(f"    ✅ 回测图表已保存")
        plt.close()
        
        return fig
```

### 实战案例：金融行业配对交易

下面我们用真实的美股数据演示完整的配对交易流程。

```python
# 主程序：金融行业配对交易
if __name__ == "__main__":
    print("="*60)
    print("统计套利：配对交易策略回测系统")
    print("="*60)
    
    # 初始化
    backtester = PairsTradingBacktester(
        start_date='2018-01-01',
        end_date='2025-12-31'
    )
    
    # 选择同行业股票（金融行业）
    financial_stocks = [
        'JPM', 'BAC', 'WFC', 'C', 'GS',  # 大型银行
        'MS', 'AXP', 'BK', 'STT', 'USB'   # 投行/区域银行
    ]
    
    # 加载数据
    backtester.load_data(financial_stocks)
    
    # 寻找协整对
    pairs, p_values = backtester.find_cointegrated_pairs(p_value_threshold=0.05)
    
    if len(pairs) == 0:
        print("\n⚠️  未找到协整对，请扩大股票池或放宽p值阈值")
    else:
        print(f"\n前5对协整关系：")
        for i, pair in enumerate(pairs[:5]):
            print(f"  {i+1}. {pair['stock1']} & {pair['stock2']} "
                  f"(p-value={pair['p_value']:.4f})")
        
        # 回测前3对
        print(f"\n{'='*60}")
        print("回测协整对")
        print(f"{'='*60}")
        
        all_pnls = []
        for i, pair in enumerate(pairs[:3]):
            print(f"\n回测第 {i+1} 对: {pair['stock1']} & {pair['stock2']}")
            
            results = backtester.backtest_pair(
                pair['stock1'], pair['stock2'],
                entry_z=2.0, exit_z=0.5, stop_z=3.0, hold_period=20
            )
            
            print(f"  交易次数: {results['total_trades']}")
            print(f"  胜率: {results['win_rate']*100:.2f}%")
            print(f"  总收益: {results['total_return']*100:.2f}%")
            print(f"  Sharpe比率: {results['sharpe_ratio']:.2f}")
            print(f"  最大回撤: {results['max_drawdown']*100:.2f}%")
            
            # 可视化
            backtester.visualize_pair(results)
            
            all_pnls.append(results['cumulative_pnl'])
        
        # 合并收益
        print(f"\n{'='*60}")
        print("组合表现")
        print(f"{'='*60}")
        combined_pnl = np.sum(all_pnls, axis=0)
        combined_sharpe = np.mean(combined_pnl) / np.std(combined_pnl) * np.sqrt(252)
        print(f"  组合Sharpe比率: {combined_sharpe:.2f}")
        print(f"  组合总收益: {combined_pnl[-1]/np.mean([p['spread'].mean() for p in [backtester.backtest_pair(p['stock1'], p['stock2']) for p in pairs[:3]]])*100:.2f}%")
```

## 协整检验的进阶话题

### 1. 滚动窗口协整检验

协整关系可能随时间变化，需要使用滚动窗口：

```python
def rolling_cointegration_test(S1, S2, window=252):
    """
    滚动窗口协整检验
    
    参数：
    - window: 滚动窗口长度（交易日）
    """
    n = len(S1)
    p_values = []
    dates = []
    
    for i in range(window, n, 20):  # 每20天检验一次
        s1_window = S1.iloc[i-window:i]
        s2_window = S2.iloc[i-window:i]
        
        try:
            result = coint(s1_window, s2_window, trend='c', autolag='aic')
            p_values.append(result[1])
            dates.append(S1.index[i])
        except:
            p_values.append(1.0)
            dates.append(S1.index[i])
    
    # 可视化
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(dates, p_values, linewidth=2, marker='o')
    ax.axhline(y=0.05, color='red', linestyle='--', label='显著性水平(5%)')
    ax.set_title('滚动窗口协整检验（p-value时序）', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('p-value')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return p_values, dates
```

### 2. 多因子协整模型

传统配对交易只考虑两只股票，多因子模型可以扩展：

```python
from statsmodels.tsa.vector_ar.vecm import VECM

def multi_factor_cointegration(symbols, lags=2):
    """
    多因子协整检验（Johansen Test）
    """
    data = pd.DataFrame({s: backtester.price_data[s] for s in symbols})
    data = data.dropna()
    
    # Johansen协整检验
    model = VECM(data, k_ar_diff=lags)
    result = model.fit()
    
    # 协整秩检验
    trace_stat = result.coint_rank
    print(f"协整秩: {trace_stat}")
    
    return result
```

### 3. 非线性均值回归模型

传统的OU过程假设线性均值回归，但实际价差可能存在非线性：

```python
from sklearn.neural_network import MLPRegressor

def nonlinear_mean_reversion(spread, lookback=60):
    """
    使用神经网络建模非线性均值回归
    """
    # 构建特征：过去N天的价差
    X = []
    y = []
    
    for i in range(lookback, len(spread)):
        X.append(spread[i-lookback:i])
        y.append(spread[i])
    
    X = np.array(X)
    y = np.array(y)
    
    # 训练神经网络
    model = MLPRegressor(hidden_layer_sizes=(50, 25), max_iter=1000)
    model.fit(X, y)
    
    # 预测未来价差
    predictions = model.predict(X)
    
    # 计算非线性z-score
    residual = y - predictions
    z_score_nonlinear = (residual - residual.mean()) / residual.std()
    
    return z_score_nonlinear, model
```

## 风险管理与实务要点

### 1. 模型失效风险

统计套利最大的风险是**模型失效**：

- **结构性断点**：公司并购、行业重组等事件导致协整关系断裂
- ** regime切换**：市场状态变化（如牛市转熊市）导致价差行为改变
- **流动性枯竭**：危机时期价差可能长期不回归

**应对方法**：
- 设置**最大持仓时间**（如20个交易日）
- 监控**滚动窗口协整p值**，若持续>0.1则停止交易
- 使用**多个协整对分散风险**

### 2. 交易成本考量

配对交易涉及同时买卖两只股票，交易成本敏感：

| 成本类型 | 影响程度 | 优化方法 |
|---------|---------|---------|
| 佣金 | 中等 | 选择低佣金券商 |
| 买卖价差 | 高 | 优先选择高流动性股票 |
| 滑点 | 高 | 使用限价单，避免市价单 |
| 卖空成本 | 中等 | 使用互换衍生品替代卖空 |

**建议**：
- 只对**高流动性股票**（日成交额>1亿美元）做配对
- 设置**最小预期收益阈值**（如价差的2倍标准差）
- 考虑**交易成本后的净收益**才计入盈利

### 3. 持仓集中度风险

即使做了多对交易，也可能因为行业集中导致相关性风险。

**解决方法**：
- **跨行业配对**：不在同一行业内部做过多配对
- **动态仓位管理**：根据市场波动率调整总仓位
- **VaR约束**：控制组合在95%置信度下的最大损失

![风险管理框架](/images/statistical-arbitrage-mean-reversion/risk_management.png)

*图2：统计套利风险管理框架*

## 绩效评估与优化

### 1. 关键绩效指标

除了传统的Sharpe比率、最大回撤外，统计套利还需关注：

- **配对成功率**：成功盈利的配对占总配对的比例
- **平均持有期**：过短说明策略过度交易，过长说明均值回归慢
- **卡玛比率**（Calmar Ratio）：年化收益/最大回撤
- **恢复时间**：从最大回撤中恢复所需天数

### 2. 参数优化

关键参数包括：
- **入场Z-score**：通常1.5-2.5，越高越保守
- **出场Z-score**：通常0-1，越小越早出场
- **止损Z-score**：通常比入场阈值高0.5-1
- **滚动窗口长度**：通常60-120天

**优化方法**：
```python
from scipy.optimize import minimize

def optimize_parameters(stock1, stock2, param_grid):
    """
    网格搜索优化参数
    """
    best_sharpe = -np.inf
    best_params = None
    
    for entry_z in param_grid['entry_z']:
        for exit_z in param_grid['exit_z']:
            for stop_z in param_grid['stop_z']:
                results = backtester.backtest_pair(
                    stock1, stock2, entry_z, exit_z, stop_z
                )
                
                if results['sharpe_ratio'] > best_sharpe:
                    best_sharpe = results['sharpe_ratio']
                    best_params = {
                        'entry_z': entry_z,
                        'exit_z': exit_z,
                        'stop_z': stop_z
                    }
    
    return best_params, best_sharpe
```

### 3. 样本外测试

**务必进行样本外测试**：

```python
# 分割训练集和测试集
train_end = '2023-12-31'
test_start = '2024-01-01'

# 在训练集上优化参数
best_params, _ = optimize_parameters('JPM', 'BAC', param_grid)

# 在测试集上验证
test_results = backtester.backtest_pair(
    'JPM', 'BAC', **best_params,
    start_date=test_start
)

# 比较训练集和测试集表现
if test_results['sharpe_ratio'] < train_sharpe * 0.5:
    print("⚠️  可能的过拟合！测试集表现显著下降")
```

## 总结与展望

统计套利是一个**理论完备、实践丰富**的量化策略。本文介绍了从协整检验到实盘回测的完整流程，关键要点：

### 实践建议

1. **股票池选择**：优先选择同行业、高流动性、相似市值的股票
2. **协整检验**：使用滚动窗口，及时调整交易对
3. **风险控制**：设置止损、最大持仓时间、动态仓位管理
4. **参数优化**：在样本外验证，避免过拟合

### 未来方向

1. **机器学习增强**：用LSTM、Transformer建模非线性均值回归
2. **高频统计套利**：利用分钟级/秒级数据提升收益
3. **跨资产统计套利**：股票-ETF、股票-期货、跨市场套利
4. **事件驱动统计套利**：结合财报、并购等事件提升胜率

统计套利不是"印钞机"，但是一个能够**长期稳定盈利**的策略。关键是建立**系统性的研究流程**，严格风控，持续优化。

---

**参考文献**：

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). Pairs Trading. *Wilmott Magazine*.
3. Elliott, R. J., et al. (2005). Pairs trading. *Quantitative Finance*.

**代码仓库**：完整代码已上传至GitHub，包含数据获取、协整检验、回测框架、风险管理等模块。

**免责声明**：本文仅供学术交流，不构成投资建议。统计套利有风险，实盘前请充分测试。
