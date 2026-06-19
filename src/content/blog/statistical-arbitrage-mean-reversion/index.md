---
title: "统计套利：均值回归策略的原理与实践"
date: 2024-06-19
description: "深入讲解统计套利的核心思想、均值回归策略的构建方法、协整检验的实战技巧，以及风险控制的关键要点。"
tags: ["统计套利", "均值回归", "配对交易", "协整分析", "量化策略"]
category: "量化策略"
featured: false
original: true
---

# 统计套利：均值回归策略的原理与实践

## 引言

统计套利（Statistical Arbitrage）是量化投资中最经典的策略之一，其核心思想是利用资产价格之间的统计关系，在价格偏离时进行套利交易。

与传统的无风险套利不同，统计套利建立在**概率优势**而非确定性之上。当价格关系暂时偏离时，我们下注它会**均值回归**（Mean Reversion）。

本文将系统讲解：

- 统计套利的理论基础
- 均值回归策略的构建流程
- 协整检验的实战方法
- 风险控制的关键环节
- Python 完整实现代码

---

## 一、统计套利的理论基础

### 1.1 什么是统计套利？

**统计套利**是指利用数学模型和统计方法，发现资产价格之间的临时性偏离，并通过多空组合获利的策略。

**核心特征**：

1. **基于统计关系**：价格关系由历史数据统计分析得出
2. **均值回归假设**：偏离是暂时的，价格会回归长期均衡
3. **市场中性和风险中性**：通常构建多空组合对冲系统性风险
4. **高频或中频交易**：持仓周期从几分钟到几个月不等

### 1.2 均值回归的理论依据

均值回归的背后有几个重要的理论支撑：

#### （1）群体心理学

市场参与者的过度反应（Overreaction）和反应不足（Underreaction）导致价格围绕均衡值波动。

#### （2）套利机制

当价格偏离时，套利者会进场交易，推动价格回归。

#### （3）均值回归的统计性质

许多金融时间序列具有**平稳性**（Stationarity）或**趋势平稳性**，这是均值回归的统计基础。

---

## 二、配对交易：统计套利的经典范式

### 2.1 配对交易的基本思想

**配对交易**（Pairs Trading）是统计套利最简单、最直观的形式：

1. 找到两个价格走势高度相关的资产（如可口可乐 vs 百事可乐）
2. 计算它们的价格差（Spread）或比值（Ratio）
3. 当价差偏离历史均值时，做多低估资产、做空高估资产
4. 等待价差收敛后平仓获利

### 2.2 配对选择的标准

选择合适的配对是策略成功的关键：

| 标准 | 说明 | 阈值建议 |
|------|------|----------|
| **相关性** | 价格变化的相关系数 | > 0.7 |
| **协整性** | 价格序列的协整检验 p 值 | < 0.05 |
| **行业一致性** | 属于同一行业/板块 | 必须 |
| **流动性** | 日均成交额 | > 1000万 |
| **相似性** | 市值、业务模式相似 | 定性判断 |

---

## 三、协整分析：配对交易的核心技术

### 3.1 什么是协整？

**协整**（Cointegration）是指多个非平稳时间序列的线性组合是平稳的。

**直观理解**：

- 两只股票的价格都是随机游走（非平稳）
- 但它们的价差（或线性组合）是平稳的
- 这意味着长期来看，两者之间存在稳定的关系

### 3.2 Engle-Granger 协整检验

经典的协整检验方法：

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
import yfinance as yf

def engle_granger_test(price1, price2):
    """
    Engle-Granger 协整检验
    
    参数：
    - price1: Series, 第一个资产的价格
    - price2: Series, 第二个资产的价格
    
    返回：
    - result: dict, 检验结果
    """
    # 步骤1：回归分析
    from sklearn.linear_model import LinearRegression
    
    X = price2.values.reshape(-1, 1)
    y = price1.values
    
    model = LinearRegression()
    model.fit(X, y)
    
    beta = model.coef_[0]
    alpha = model.intercept_
    
    # 步骤2：计算残差
    spread = price1 - (alpha + beta * price2)
    
    # 步骤3：ADF 检验残差是否平稳
    adf_result = adfuller(spread, autolag='AIC')
    
    # 步骤4：协整检验（直接方法）
    coint_result = coint(price1, price2)
    
    result = {
        'beta': beta,
        'alpha': alpha,
        'spread': spread,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'coint_statistic': coint_result[0],
        'coint_pvalue': coint_result[1],
        'is_cointegrated': coint_result[1] < 0.05
    }
    
    return result

# 示例使用
# stock1 = yf.download('KO', start='2020-01-01')['Adj Close']  # 可口可乐
# stock2 = yf.download('PEP', start='2020-01-01')['Adj Close']  # 百事可乐
# result = engle_granger_test(stock1, stock2)
# print(f"协整检验 p 值: {result['coint_pvalue']:.4f}")
# print(f"是否协整: {result['is_cointegrated']}")
```

### 3.3 Johansen 协整检验

当涉及多个资产时，Johansen 检验更合适：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验（适用于多资产）
    
    参数：
    - price_matrix: DataFrame, 多资产价格矩阵
    - det_order: int, 确定性项的顺序
                0: 无常数项，无趋势
                1: 有常数项，无趋势
                2: 有常数项，有趋势
    - k_ar_diff: int, 滞后阶数
    
    返回：
    - result: 检验结果
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 提取特征值和信息
    eigen_values = result.eig
    trace_statistic = result.lr1
    max_statistic = result.lr2
    
    # 临界值（5%显著性水平）
    critical_values_trace = result.cvt[:, 1]
    critical_values_max = result.cvm[:, 1]
    
    # 判断协整关系个数
    num_coint = np.sum(trace_statistic > critical_values_trace)
    
    return {
        'eigen_values': eigen_values,
        'trace_statistic': trace_statistic,
        'max_statistic': max_statistic,
        'num_cointegrating_relations': num_coint
    }
```

---

## 四、均值回归策略的构建

### 4.1 价差的标准化

为了量化"偏离"程度，需要对价差进行标准化：

```python
def calculate_zscore(spread, window=252):
    """
    计算价差的 Z-Score
    
    Z-Score = (当前价差 - 均值) / 标准差
    
    当 |Z-Score| > 2 时，认为出现显著偏离
    """
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std()
    
    zscore = (spread - mean) / std
    
    return zscore

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(14, 8))

# 子图1：价差序列
axes[0].plot(spread.index, spread.values, label='Spread', color='blue')
axes[0].axhline(y=spread.mean(), color='red', linestyle='--', label='Mean')
axes[0].set_title('Price Spread')
axes[0].legend()

# 子图2：Z-Score
zscore = calculate_zscore(spread)
axes[1].plot(zscore.index, zscore.values, label='Z-Score', color='green')
axes[1].axhline(y=2, color='red', linestyle='--', label='+2 SD')
axes[1].axhline(y=-2, color='red', linestyle='--', label='-2 SD')
axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
axes[1].set_title('Z-Score of Spread')
axes[1].legend()

plt.tight_layout()
plt.show()
```

### 4.2 交易信号的生成

基于 Z-Score 生成交易信号：

```python
def generate_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    生成配对交易的信号
    
    策略逻辑：
    - Z-Score < -2：做多价差（做多资产1，做空资产2）
    - Z-Score > +2：做空价差（做空资产1，做多资产2）
    - |Z-Score| < 0.5：平仓
    
    返回：
    - signal: Series, 交易信号
             1: 做多
             -1: 做空
             0: 平仓/不持仓
    """
    signal = pd.Series(0, index=zscore.index)
    
    # 入场信号
    signal[zscore < -entry_threshold] = 1   # 做多
    signal[zscore > entry_threshold] = -1    # 做空
    
    # 出场信号（需要记录持仓状态）
    position = 0  # 当前持仓
    for i in range(1, len(signal)):
        if position != 0 and abs(zscore.iloc[i]) < exit_threshold:
            signal.iloc[i] = 0  # 平仓
            position = 0
        elif position == 0:
            position = signal.iloc[i]
        else:
            signal.iloc[i] = position  # 维持持仓
    
    return signal

# 生成信号示例
zscore = calculate_zscore(spread)
signal = generate_trading_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)
```

### 4.3 仓位管理

合理的仓位管理至关重要：

```python
def calculate_position_size(signal, capital, price1, price2, max_leverage=2.0):
    """
    计算动态仓位大小
    
    风险控制：
    - 单个配对最大损失不超过总资本的 2%
    - 使用波动率调整仓位
    """
    # 计算价差的波动率
    spread_return = np.log(spread / spread.shift(1))
    volatility = spread_return.rolling(252).std() * np.sqrt(252)
    
    # 根据波动率调整仓位（波动率越高，仓位越低）
    vol_weight = 1 / (volatility + 1e-8)
    vol_weight = vol_weight / vol_weight.max()
    
    # 基础仓位
    base_position = capital * max_leverage / (price1 + price2)
    
    # 应用波动率权重
    position = base_position * vol_weight
    
    # 根据信号调整
    position = position * signal
    
    return position
```

---

## 五、策略回测与评估

### 5.1 回测框架

构建一个完整的回测系统：

```python
class PairsTradingBacktest:
    """配对交易回测框架"""
    
    def __init__(self, price1, price2, initial_capital=1000000):
        self.price1 = price1
        self.price2 = price2
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.positions = []
        self.trades = []
        
    def run_backtest(self, entry_threshold=2.0, exit_threshold=0.5):
        """运行回测"""
        # 计算价差和 Z-Score
        spread = calculate_spread(self.price1, self.price2)
        zscore = calculate_zscore(spread)
        
        # 生成信号
        signal = generate_trading_signals(zscore, entry_threshold, exit_threshold)
        
        # 回测循环
        position = 0  # 当前持仓
        entry_zscore = 0  # 入场时的 Z-Score
        
        for i in range(1, len(signal)):
            date = signal.index[i]
            
            if position == 0 and signal.iloc[i] != 0:
                # 开仓
                position = signal.iloc[i]
                entry_zscore = zscore.iloc[i]
                
                # 记录交易
                trade = {
                    'entry_date': date,
                    'entry_zscore': entry_zscore,
                    'direction': 'Long Spread' if position == 1 else 'Short Spread',
                    'price1': self.price1.iloc[i],
                    'price2': self.price2.iloc[i]
                }
                self.trades.append(trade)
                
            elif position != 0 and signal.iloc[i] == 0:
                # 平仓
                trade = self.trades[-1]
                trade['exit_date'] = date
                trade['exit_zscore'] = zscore.iloc[i]
                trade['exit_price1'] = self.price1.iloc[i]
                trade['exit_price2'] = self.price2.iloc[i]
                
                # 计算收益
                if trade['direction'] == 'Long Spread':
                    pnl = (self.price1.iloc[i] - trade['price1']) - \
                          (self.price2.iloc[i] - trade['price2'])
                else:
                    pnl = (trade['price1'] - self.price1.iloc[i]) - \
                          (trade['price2'] - self.price2.iloc[i])
                
                trade['pnl'] = pnl
                
                position = 0
                
            # 记录每日持仓
            self.positions.append({
                'date': date,
                'position': position,
                'zscore': zscore.iloc[i],
                'spread': spread.iloc[i]
            })
    
    def calculate_metrics(self):
        """计算策略评价指标"""
        # 转换为 DataFrame
        trades_df = pd.DataFrame(self.trades)
        
        if len(trades_df) == 0:
            return None
        
        # 计算累计收益
        trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
        
        # 计算评价指标
        total_trades = len(trades_df)
        winning_trades = len(trades_df[trades_df['pnl'] > 0])
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        avg_win = trades_df[trades_df['pnl'] > 0]['pnl'].mean() if winning_trades > 0 else 0
        avg_loss = trades_df[trades_df['pnl'] < 0]['pnl'].mean() if (total_trades - winning_trades) > 0 else 0
        
        profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
        
        metrics = {
            'Total Trades': total_trades,
            'Win Rate': win_rate,
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Total PnL': trades_df['pnl'].sum(),
            'Sharpe Ratio': self.calculate_sharpe_ratio()
        }
        
        return metrics
    
    def calculate_sharpe_ratio(self):
        """计算夏普比率"""
        positions_df = pd.DataFrame(self.positions)
        
        if len(positions_df) == 0:
            return 0
        
        # 计算每日收益
        positions_df['daily_return'] = positions_df['spread'].pct_change()
        
        # 年化夏普比率
        sharpe = positions_df['daily_return'].mean() / positions_df['daily_return'].std() * np.sqrt(252)
        
        return sharpe
```

### 5.2 策略评估指标

除了常规的夏普比率、最大回撤等指标，配对交易还需要关注：

```python
def evaluate_pairs_strategy(trades_df, spread_series):
    """配对交易专属评价指标"""
    
    metrics = {}
    
    # 1. 半衰期（Half-Life）
    # 衡量价差回归均值的速度
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools.tools import add_constant
    
    spread_lag = spread_series.shift(1).dropna()
    spread_diff = spread_series.diff().dropna()
    
    X = add_constant(spread_lag)
    model = OLS(spread_diff, X).fit()
    
    half_life = -np.log(2) / np.log(abs(model.params[1]))
    metrics['Half Life'] = half_life
    
    # 2. 收敛成功率
    # 交易在合理时间内收敛的概率
    successful_trades = trades_df[
        (trades_df['exit_zscore'] < 1.0) & (trades_df['exit_zscore'] > -1.0)
    ]
    convergence_rate = len(successful_trades) / len(trades_df)
    metrics['Convergence Rate'] = convergence_rate
    
    # 3. 平均持仓时间
    trades_df['holding_period'] = (trades_df['exit_date'] - trades_df['entry_date']).dt.days
    metrics['Average Holding Period (Days)'] = trades_df['holding_period'].mean()
    
    # 4. 回撤恢复时间
    # 计算最大回撤及其恢复时间
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.cummax()
    drawdown = (cumulative_pnl - running_max) / running_max
    
    max_dd = drawdown.min()
    metrics['Max Drawdown'] = max_dd
    
    return metrics
```

---

## 六、实战案例：A股配对交易

### 6.1 标的选取

选择A股市场中业务模式相似、市值接近的配对：

**案例配对**：贵州茅台 (600519.SH) vs 五粮液 (000858.SZ)

```python
import akshare as ak

# 获取历史数据
def get_stock_data(symbol, start_date, end_date):
    """使用 AkShare 获取A股数据"""
    df = ak.stock_zh_a_hist(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )
    
    df['date'] = pd.to_datetime(df['日期'])
    df.set_index('date', inplace=True)
    
    return df['收盘']

# 获取数据
moutai = get_stock_data('600519', '2020-01-01', '2024-06-01')
wuliangye = get_stock_data('000858', '2020-01-01', '2024-06-01')

# 协整检验
result = engle_granger_test(moutai, wuliangye)
print(f"协整检验 p 值: {result['coint_pvalue']:.4f}")
print(f"是否协整: {result['is_cointegrated']}")
```

### 6.2 策略表现

假设我们在2020-2024年期间执行该策略：

```python
# 回测
backtest = PairsTradingBacktest(moutai, wuliangye, initial_capital=1000000)
backtest.run_backtest(entry_threshold=2.0, exit_threshold=0.5)

# 评估结果
metrics = backtest.calculate_metrics()
for key, value in metrics.items():
    print(f"{key}: {value:.4f}")

# 可视化
plot_pairs_results(backtest)
```

**典型结果**（示意）：

```
Total Trades: 42
Win Rate: 64.3%
Average Win: 12560.5
Average Loss: -8920.3
Profit Factor: 1.41
Total PnL: 298500.0
Sharpe Ratio: 1.85
Half Life: 18.5 days
Convergence Rate: 78.6%
Average Holding Period (Days): 22.3
Max Drawdown: -8.2%
```

---

## 七、风险控制与实战要点

### 7.1 主要风险

统计套利虽然理论上市场中性，但实践中存在多种风险：

#### （1）模型风险

- **结构性断裂**：长期关系突然失效（如行业政策变化）
- ** regime 切换**：市场状态改变导致均值回归性质变化

**应对方法**：

```python
def monitor_model_stability(spread, window=252, significance_level=0.05):
    """
    监测模型的稳定性
    
    使用滚动窗口检验协整关系是否持续存在
    """
    stability_scores = []
    
    for i in range(window, len(spread)):
        sub_sample = spread[i-window:i]
        
        # ADF 检验
        adf_result = adfuller(sub_sample, autolag='AIC')
        
        # 记录 p 值
        stability_scores.append({
            'date': spread.index[i],
            'adf_pvalue': adf_result[1],
            'is_stable': adf_result[1] < significance_level
        })
    
    stability_df = pd.DataFrame(stability_scores)
    
    # 如果稳定性低于阈值，发出预警
    recent_stability = stability_df.tail(63)['is_stable'].mean()  # 最近3个月
    
    if recent_stability < 0.7:
        print("⚠️ 警告：模型稳定性下降，建议暂停交易或重新训练模型！")
    
    return stability_df
```

#### （2）执行风险

- **滑点**：实盘交易的滑点可能显著侵蚀收益
- **冲击成本**：大单交易会推动价格不利变动

**应对方法**：

```python
def estimate_trading_cost(signal, volume, market_volume, spread):
    """
    估算交易成本
    
    参数：
    - signal: 交易信号
    - volume: 计划交易量
    - market_volume: 市场成交量
    - spread: 买卖价差
    
    返回：
    - total_cost: 总交易成本
    """
    # 冲击成本（根据 Almgren-Chriss 模型）
    permanent_impact = 0.1 * (volume / market_volume)  # 永久性冲击
    temporary_impact = 0.5 * spread + 0.2 * (volume / market_volume)  # 临时性冲击
    
    # 总冲击成本
    impact_cost = (permanent_impact + temporary_impact) * volume
    
    # 手续费（假设万分之三）
    commission = volume * 0.0003
    
    total_cost = impact_cost + commission
    
    return total_cost
```

#### （3）尾部风险

- **价差不收敛**：极端情况下，价差可能持续扩大
- **黑天鹅事件**：如2020年3月新冠疫情导致流动性枯竭

**应对方法**：

```python
def implement_stop_loss(signal, zscore, max_loss_zscore=4.0, max_holding_days=60):
    """
    实施止损策略
    
    止损条件：
    1. Z-Score 超过极端值（如 ±4）
    2. 持仓时间超过上限
    
    参数：
    - signal: 交易信号
    - zscore: 当前 Z-Score
    - max_loss_zscore: 最大容忍的 Z-Score 绝对值
    - max_holding_days: 最大持仓天数
    
    返回：
    - adjusted_signal: 调整后的信号（加入止损）
    """
    adjusted_signal = signal.copy()
    
    position = 0
    entry_date = None
    
    for i in range(len(signal)):
        # 检查是否持仓
        if position != 0:
            holding_days = (signal.index[i] - entry_date).days
            
            # 止损条件1：Z-Score 极端值
            if abs(zscore.iloc[i]) > max_loss_zscore:
                adjusted_signal.iloc[i] = 0  # 平仓
                position = 0
                print(f"⚠️ 止损触发（极端 Z-Score）: {signal.index[i]}")
            
            # 止损条件2：持仓时间过长
            elif holding_days > max_holding_days:
                adjusted_signal.iloc[i] = 0  # 平仓
                position = 0
                print(f"⚠️ 止损触发（持仓超时）: {signal.index[i]}")
        
        # 更新持仓状态
        if adjusted_signal.iloc[i] != 0 and position == 0:
            position = adjusted_signal.iloc[i]
            entry_date = signal.index[i]
    
    return adjusted_signal
```

### 7.2 实战要点

1. **多样化配对**：不要将所有资金放在一个配对上
2. **动态调整参数**：根据市场状态调整入场/出场阈值
3. **实时监控**：建立自动化监控系统，及时发现异常
4. **资金管理**：严格控制单个配对的资金占比（建议 < 10%）

---

## 八、进阶话题

### 8.1 多资产统计套利

当扩展到多个资产时，可以使用**主成分分析**（PCA）或**因子模型**：

```python
from sklearn.decomposition import PCA

def pca_stat_arb(price_matrix, n_components=3):
    """
    基于PCA的统计套利
    
    思路：
    1. 对价格矩阵进行PCA分解
    2. 用前N个主成分拟合价格
    3. 残差即为套利空间
    """
    # 标准化
    returns = price_matrix.pct_change().dropna()
    standardized = (returns - returns.mean()) / returns.std()
    
    # PCA 分解
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(standardized)
    
    # 重构
    reconstructed = pca.inverse_transform(principal_components)
    
    # 残差（套利空间）
    residuals = standardized - reconstructed
    
    return residuals, pca.explained_variance_ratio_
```

### 8.2 机器学习增强

使用机器学习方法提升配对选择和风险预测：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def ml_enhanced_pairs_selection(features, pairs_labels):
    """
    使用随机森林预测配对是否成功
    
    特征包括：
    - 相关性
    - 协整 p 值
    - 半衰期
    - 波动率比
    - 行业相似度（编码）
    """
    X_train, X_test, y_train, y_test = train_test_split(
        features, pairs_labels, test_size=0.2, random_state=42
    )
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    
    # 预测
    predictions = rf.predict(X_test)
    probabilities = rf.predict_proba(X_test)[:, 1]
    
    # 评估
    accuracy = rf.score(X_test, y_test)
    print(f"模型准确率: {accuracy:.2%}")
    
    return rf, probabilities
```

---

## 九、总结

### 9.1 核心要点回顾

1. **统计套利的本质**是利用统计关系进行概率套利，不是无风险套利
2. **协整分析**是配对交易的核心技术，确保价格关系长期稳定
3. **风险管理**至关重要，包括模型风险、执行风险和尾部风险
4. **多样化**和**动态调整**是提升策略稳健性的关键

### 9.2 实践建议

- **从小资金开始**：先用小资金验证逻辑
- **充分回测**：使用样本外数据验证策略有效性
- **关注交易成本**：实盘成本可能显著高于回测
- **持续学习**：市场在变化，策略需要不断迭代

### 9.3 延伸阅读

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques"
3. Ganapathy, V. (2015). "Statistical Arbitrage and Pairs Trading"
4. Avellaneda, M., & Lee, J. H. (2010). "Statistical Arbitrage in the US Equities Market"

---

## 附录：完整代码框架

```python
# main.py - 统计套利策略完整框架

import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.stattools import coint
import matplotlib.pyplot as plt

class StatisticalArbitrage:
    """统计套利策略主类"""
    
    def __init__(self, symbols, start_date, end_date):
        self.symbols = symbols
        self.start_date = start_date
        self.end_date = end_date
        self.prices = self.download_data()
        
    def download_data(self):
        """下载数据"""
        data = yf.download(self.symbols, start=self.start_date, end=self.end_date)['Adj Close']
        return data
    
    def find_cointegrated_pairs(self, significance=0.05):
        """找出所有协整的配对"""
        n = len(self.symbols)
        pairs = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = self.prices[self.symbols[i]]
                stock2 = self.prices[self.symbols[j]]
                
                result = coint(stock1, stock2)
                
                if result[1] < significance:
                    pairs.append({
                        'pair': (self.symbols[i], self.symbols[j]),
                        'p_value': result[1],
                        'test_statistic': result[0]
                    })
        
        return pairs
    
    def backtest_pair(self, stock1, stock2):
        """回测单个配对"""
        # 计算价差
        spread = np.log(stock1) - np.log(stock2)
        
        # 计算 Z-Score
        zscore = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()
        
        # 生成信号
        signal = self.generate_signals(zscore)
        
        # 计算收益
        returns = self.calculate_returns(signal, stock1, stock2)
        
        return returns
    
    def generate_signals(self, zscore, entry=2.0, exit=0.5):
        """生成交易信号"""
        signal = pd.Series(0, index=zscore.index)
        
        position = 0
        for i in range(1, len(zscore)):
            if position == 0:
                if zscore.iloc[i] < -entry:
                    position = 1
                elif zscore.iloc[i] > entry:
                    position = -1
            elif position != 0:
                if abs(zscore.iloc[i]) < exit:
                    position = 0
            
            signal.iloc[i] = position
        
        return signal
    
    def calculate_returns(self, signal, stock1, stock2):
        """计算策略收益"""
        # 计算每日收益
        ret1 = stock1.pct_change()
        ret2 = stock2.pct_change()
        
        # 策略收益
        strategy_ret = signal.shift(1) * (ret1 - ret2)
        
        return strategy_ret
    
    def plot_results(self, returns):
        """可视化结果"""
        cumulative_returns = (1 + returns).cumprod()
        
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))
        
        # 累计收益曲线
        axes[0].plot(cumulative_returns.index, cumulative_returns.values)
        axes[0].set_title('Cumulative Returns')
        axes[0].set_ylabel('Cumulative Return')
        
        # 回撤曲线
        running_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - running_max) / running_max
        
        axes[1].fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        axes[1].set_title('Drawdown')
        axes[1].set_ylabel('Drawdown')
        
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 初始化
    symbols = ['KO', 'PEP', 'MSFT', 'AAPL', 'GOOGL', 'META']
    strategy = StatisticalArbitrage(symbols, '2020-01-01', '2024-06-01')
    
    # 找出协整配对
    pairs = strategy.find_cointegrated_pairs()
    print("协整配对：")
    for p in pairs:
        print(f"{p['pair']}: p-value = {p['p_value']:.4f}")
    
    # 回测第一个配对
    if len(pairs) > 0:
        stock1 = strategy.prices[pairs[0]['pair'][0]]
        stock2 = strategy.prices[pairs[0]['pair'][1]]
        
        returns = strategy.backtest_pair(stock1, stock2)
        
        # 评估结果
        sharpe = returns.mean() / returns.std() * np.sqrt(252)
        cumulative_ret = (1 + returns).cumprod().iloc[-1] - 1
        
        print(f"\n策略表现：")
        print(f"累计收益: {cumulative_ret:.2%}")
        print(f"夏普比率: {sharpe:.2f}")
        
        # 可视化
        strategy.plot_results(returns)
```

---

**免责声明**：本文仅供学习和研究使用，不构成任何投资建议。统计套利策略在实际操作中可能面临模型风险、执行风险等多种风险，请在充分理解策略原理和风险的基础上谨慎使用。
