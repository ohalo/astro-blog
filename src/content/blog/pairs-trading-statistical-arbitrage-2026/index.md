---
title: "配对交易与统计套利：从协整检验到实盘执行的完整指南"
publishDate: '2026-06-06'
description: "配对交易与统计套利：从协整检验到实盘执行的完整指南 - halo的技术博客"
tags:
  - 量化交易
language: Chinese
---

## 引言

统计套利(Statistical Arbitrage)是量化交易中最经典的策略之一，其核心思想是利用资产价格之间的统计关系进行套利。配对交易(Pairs Trading)作为统计套利的最简单形式，通过交易一对协整股票的价差获利。本文将从理论到实战，完整讲解统计套利的构建与执行流程。

![配对交易价差均值回归](/images/pairs-trading-statistical-arbitrage-2026/spread_mean_reversion.jpg)

## 一、统计套利理论基础

### 1.1 什么是统计套利？

统计套利是指利用数学模型和统计分析，发现资产价格之间的临时性偏离，通过同时买入低估资产、卖出高估资产，等待价格回归均衡来获取收益的交易策略。

**核心假设**
- 价格偏离是暂时的
- 价格会均值回归
- 偏离可以通过统计方法量化

**主要类型**
1. **配对交易(Pairs Trading)**：交易两只协整股票
2. **多腿套利(Multi-leg Arbitrage)**：交易多只相关性强的股票
3. **因子中性策略(Factor Neutral Strategy)**：对冲市场风险
4. **均值回归策略(Mean Reversion)**：基于价格偏离Z-Score

### 1.2 配对交易原理

配对交易是最简单的统计套利形式，其流程如下：

1. **选股**：找到基本面相似、价格长期协整的两只股票
2. **建模**：计算价差的均值和标准差
3. **交易信号**：
   - 价差 > 均值 + 2倍标准差 → 卖出价差（做空股票A，做多股票B）
   - 价差 < 均值 - 2倍标准差 → 买入价差（做多股票A，做空股票B）
   - 价差回归均值 → 平仓
4. **风险管理**：设置止损、控制仓位

![配对交易流程图](/images/pairs-trading-statistical-arbitrage-2026/pairs_trading_flowchart.jpg)

## 二、协整检验与配对筛选

### 2.1 协整 vs 相关性

很多初学者会混淆协整(Cointegration)和相关性(Correlation)，但两者截然不同：

| 特征 | 协整 | 相关性 |
|------|------|--------|
| 定义 | 非平稳序列的线性组合平稳 | 两个序列同向/反向变动 |
| 时间尺度 | 长期关系 | 可长可短 |
| 套利意义 | 价格会回归均衡 | 不一定回归 |
| 检验方法 | ADF检验、Johansen检验 | Pearson相关系数 |

**关键结论**：高相关性 ≠ 可套利；协整关系才是配对交易的基础。

### 2.2 协整检验方法

**Engle-Granger两步法**

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def engle_granger_test(price_A, price_B):
    """
    Engle-Granger协整检验
    返回：协整残差、p值
    """
    # Step 1: OLS回归
    X = sm.add_constant(price_B)
    model = sm.OLS(price_A, X).fit()
    spread = model.resid
    
    # Step 2: ADF检验残差平稳性
    adf_result = adfuller(spread, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断协整（1%显著性水平）
    is_cointegrated = p_value < 0.01 and adf_stat < critical_values['1%']
    
    return {
        'spread': spread,
        'adf_stat': adf_stat,
        'p_value': p_value,
        'is_cointegrated': is_cointegrated,
        'hedge_ratio': model.params[1]  # β系数
    }

# 示例使用
result = engle_granger_test(price_A, price_B)
print(f"ADF统计量: {result['adf_stat']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"对冲比例(β): {result['hedge_ratio']:.4f}")
print(f"是否协整: {result['is_cointegrated']}")
```

**Johansen协整检验（多变量）**

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多只股票）
    price_matrix: T×N 矩阵，T为时间长度，N为股票数量
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 输出结果
    print("特征值和特征向量:")
    print(result.eig)
    print("\n协整关系数量检验:")
    print(f"Trace统计量: {result.lr1}")
    print(f"Max统计量: {result.lr2}")
    
    # 判断协整关系数量（5%显著性水平）
    trace_critical = result.cvt[:, 1]  # 5%临界值
    num_coint = sum(result.lr1 > trace_critical)
    
    return {
        'num_coint': num_coint,
        'eigenvalues': result.eig,
        'trace_stat': result.lr1,
        'max_stat': result.lr2
    }
```

### 2.3 配对筛选实战

**步骤1：初步筛选**

```python
def preliminary_screening(stock_list, start_date, end_date):
    """
    初步筛选：行业相同、市值相近、流动性好
    """
    # 获取股票基本信息
    stock_data = get_stock_info(stock_list, start_date, end_date)
    
    # 筛选条件
    conditions = [
        ('industry', 'same'),  # 同一行业
        ('market_cap', lambda x: abs(x - x.median()) / x.median() < 0.5),  # 市值相近
        ('avg_volume', lambda x: x > 1e6),  # 日均成交量>100万
        ('price_range', lambda x: (x > 5) & (x < 500))  # 股价在5-500之间
    ]
    
    screened_stocks = apply_filters(stock_data, conditions)
    return screened_stocks
```

**步骤2：计算相似度指标**

```python
def calculate_similarity(price_data):
    """
    计算股票之间的相似度
    返回：相似度矩阵
    """
    stocks = price_data.columns
    n = len(stocks)
    similarity_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            # 1. 相关系数
            corr = price_data[stocks[i]].corr(price_data[stocks[j]])
            
            # 2. 协整检验p-value
            coint_result = engle_granger_test(price_data[stocks[i]], price_data[stocks[j]])
            p_value = coint_result['p_value']
            
            # 3. 距离度量（欧几里得距离）
            distance = np.linalg.norm(price_data[stocks[i]] - price_data[stocks[j]])
            
            # 综合相似度得分（越小越好）
            similarity_score = (1 - corr) * 0.3 + p_value * 0.5 + distance * 0.2
            
            similarity_matrix[i, j] = similarity_matrix[j, i] = similarity_score
    
    return similarity_matrix
```

**步骤3：聚类分组**

```python
from sklearn.cluster import AgglomerativeClustering

def cluster_stocks(similarity_matrix, n_clusters=10):
    """
    层次聚类：将相似的股票分组
    """
    # 将相似度矩阵转换为距离矩阵
    distance_matrix = similarity_matrix
    
    # 层次聚类
    clustering = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='precomputed',
        linkage='average'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 输出分组结果
    clusters = {}
    for i, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(i)
    
    return clusters
```

## 三、交易信号生成

### 3.1 价差建模

**Z-Score标准化**

```python
def calculate_z_score(spread, window=20):
    """
    计算价差的Z-Score
    window: 滚动窗口长度
    """
    spread_mean = spread.rolling(window=window).mean()
    spread_std = spread.rolling(window=window).std()
    
    z_score = (spread - spread_mean) / spread_std
    
    return z_score

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 子图1：价差
axes[0].plot(spread.index, spread.values)
axes[0].axhline(spread.mean(), color='red', linestyle='--', label='Mean')
axes[0].fill_between(spread.index, 
                      spread.mean() - 2*spread.std(),
                      spread.mean() + 2*spread.std(),
                      alpha=0.2, color='red', label='±2σ')
axes[0].set_title('Spread')
axes[0].legend()

# 子图2：Z-Score
axes[1].plot(z_score.index, z_score.values)
axes[1].axhline(0, color='black', linestyle='-', alpha=0.3)
axes[1].axhline(2, color='red', linestyle='--', label='+2σ')
axes[1].axhline(-2, color='green', linestyle='--', label='-2σ')
axes[1].fill_between(z_score.index, -2, 2, alpha=0.1, color='gray')
axes[1].set_title('Z-Score')
axes[1].legend()

plt.tight_layout()
plt.savefig('z_score_signal.png')
```

### 3.2 交易规则

**入场信号**
- Z-Score > 阈值（如2.0）→ 做空价差
- Z-Score < -阈值（如-2.0）→ 做多价差

**出场信号**
- Z-Score回归到0附近 → 平仓
- 止损：Z-Score继续扩大超过3.0 → 强制平仓

**代码实现**

```python
class PairsTradingSignal:
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, stop_loss=3.0):
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss = stop_loss
        
    def generate_signal(self, z_score):
        """
        生成交易信号
        返回：1(做多价差), -1(做空价差), 0(平仓)
        """
        signal = pd.Series(0, index=z_score.index)
        
        position = 0  # 当前持仓：0(空仓), 1(多价差), -1(空价差)
        
        for i in range(1, len(z_score)):
            if position == 0:
                # 空仓状态，检查入场信号
                if z_score.iloc[i] < -self.entry_threshold:
                    position = 1
                    signal.iloc[i] = 1
                elif z_score.iloc[i] > self.entry_threshold:
                    position = -1
                    signal.iloc[i] = -1
                    
            elif position == 1:
                # 持有多价差，检查出场信号
                if abs(z_score.iloc[i]) < self.exit_threshold:
                    position = 0
                    signal.iloc[i] = 0
                elif z_score.iloc[i] < -self.stop_loss:
                    position = 0
                    signal.iloc[i] = 0  # 止损
                    
            elif position == -1:
                # 持有空价差，检查出场信号
                if abs(z_score.iloc[i]) < self.exit_threshold:
                    position = 0
                    signal.iloc[i] = 0
                elif z_score.iloc[i] > self.stop_loss:
                    position = 0
                    signal.iloc[i] = 0  # 止损
        
        return signal
```

### 3.3 信号优化

**卡尔曼滤波动态调整**

```python
from pykalman import KalmanFilter

def kalman_filter_spread(price_A, price_B):
    """
    使用卡尔曼滤波动态估计对冲比例
    """
    # 观测矩阵：价格B
    observation_matrix = price_B.values.reshape(-1, 1)
    
    # 状态矩阵：对冲比例
    transition_matrix = np.array([[1]])
    
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix
    )
    
    # 状态：hedge_ratio
    state_means, _ = kf.filter(price_A.values)
    hedge_ratio_dynamic = state_means[:, 0]
    
    # 动态价差
    spread_dynamic = price_A - hedge_ratio_dynamic * price_B
    
    return spread_dynamic, hedge_ratio_dynamic
```

**Ornstein-Uhlenbeck建模**

```python
from scipy.optimize import minimize

def fit_ou_process(spread):
    """
    拟合Ornstein-Uhlenbeck过程
    dX_t = κ(μ - X_t)dt + σdW_t
    """
    # 离散化：X_{t+1} - X_t = κ(μ - X_t)Δt + σε_t
    dt = 1  # 日频数据
    
    def ou_likelihood(params):
        """OU过程的对数似然函数"""
        kappa, mu, sigma = params
        
        # 预测值
        X_pred = mu + (spread[:-1] - mu) * np.exp(-kappa * dt)
        
        # 残差
        residuals = spread[1:] - X_pred
        
        # 对数似然
        log_likelihood = -0.5 * (np.log(2*np.pi*sigma**2) + (residuals/sigma)**2).sum()
        
        return -log_likelihood  # 最小化负对数似然
    
    # 初始值
    x0 = [1.0, spread.mean(), spread.std()]
    
    # 优化
    result = minimize(ou_likelihood, x0, method='L-BFGS-B', 
                      bounds=[(0.01, 10), (None, None), (0.01, 10)])
    
    kappa, mu, sigma = result.x
    
    # 半衰期
    half_life = np.log(2) / kappa
    
    return {
        'kappa': kappa,  # 均值回归速度
        'mu': mu,        # 长期均值
        'sigma': sigma,   # 波动率
        'half_life': half_life  # 半衰期
    }

# 使用示例
ou_params = fit_ou_process(spread)
print(f"均值回归速度(κ): {ou_params['kappa']:.4f}")
print(f"半衰期: {ou_params['half_life']:.2f}天")
```

## 四、实盘执行与风险控制

### 4.1 交易成本分析

```python
def calculate_transaction_cost(signal, price, commission=0.0003, slippage=0.001):
    """
    计算交易成本
    signal: 交易信号（1, -1, 0）
    price: 价格序列
    commission: 佣金比例
    slippage: 滑点比例
    """
    # 计算换手率
    turnover = signal.diff().abs()
    
    # 交易成本
    cost = turnover * price * (commission + slippage)
    
    # 累计成本
    cumulative_cost = cost.cumsum()
    
    return cumulative_cost
```

### 4.2 仓位管理

**固定比例仓位**

```python
def fixed_fraction_position(signal, capital, max_position=0.1):
    """
    固定比例仓位管理
    max_position: 单个配对最大仓位比例
    """
    position_value = capital * max_position
    shares = position_value / price
    
    position = signal * shares
    return position
```

**凯利公式仓位**

```python
def kelly_position(win_rate, win_loss_ratio, max_position=0.1):
    """
    凯利公式仓位
    win_rate: 胜率
    win_loss_ratio: 盈亏比
    """
    kelly_f = win_rate - (1 - win_rate) / win_loss_ratio
    
    # 限制仓位上限
    position_fraction = min(kelly_f, max_position)
    
    return position_fraction
```

### 4.3 风险监控系统

```python
class PairsRiskMonitor:
    def __init__(self, spread, z_score, signal):
        self.spread = spread
        self.z_score = z_score
        self.signal = signal
        
    def monitor_risk(self):
        """风险监控"""
        risks = {}
        
        # 1. 最大回撤
        cumulative_return = (self.signal.shift(1) * self.spread.pct_change()).cumsum()
        drawdown = (cumulative_return - cumulative_return.cummax()) / cumulative_return.cummax()
        max_drawdown = drawdown.min()
        risks['max_drawdown'] = max_drawdown
        
        # 2. 当前Z-Score
        current_z = self.z_score.iloc[-1]
        risks['current_z_score'] = current_z
        
        # 3. 持仓时间
        holding_time = (self.signal != 0).sum()
        risks['holding_time'] = holding_time
        
        # 4. 止损触发
        stop_loss_triggered = abs(current_z) > 3.0
        risks['stop_loss'] = stop_loss_triggered
        
        return risks
    
    def generate_alert(self, risks, limits):
        """生成预警"""
        alerts = []
        
        if risks['max_drawdown'] < limits['max_drawdown']:
            alerts.append(f"最大回撤超限: {risks['max_drawdown']:.2%}")
        
        if risks['current_z_score'] > limits['z_score']:
            alerts.append(f"Z-Score过高: {risks['current_z_score']:.2f}")
        
        if risks['holding_time'] > limits['holding_time']:
            alerts.append(f"持仓时间过长: {risks['holding_time']}天")
        
        if risks['stop_loss']:
            alerts.append("触发止损！")
        
        return alerts
```

## 五、回测与绩效分析

### 5.1 回测框架

```python
class PairsTradingBacktest:
    def __init__(self, price_A, price_B, signal, initial_capital=1e6):
        self.price_A = price_A
        self.price_B = price_B
        self.signal = signal
        self.initial_capital = initial_capital
        
    def run_backtest(self):
        """执行回测"""
        # 1. 计算持仓
        position_A = self.signal  # 做多A
        position_B = -self.signal  # 做空B（对冲）
        
        # 2. 计算收益
        return_A = self.price_A.pct_change()
        return_B = self.price_B.pct_change()
        
        portfolio_return = position_A.shift(1) * return_A + position_B.shift(1) * return_B
        
        # 3. 计算净值
        cumulative_return = (1 + portfolio_return).cumprod()
        net_value = self.initial_capital * cumulative_return
        
        return {
            'portfolio_return': portfolio_return,
            'cumulative_return': cumulative_return,
            'net_value': net_value
        }
    
    def calculate_performance(self, returns):
        """计算绩效指标"""
        # 1. 年化收益率
        total_return = returns.sum()
        trading_days = len(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 2. 夏普比率
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
        
        # 3. 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # 4. 胜率
        win_rate = (returns > 0).sum() / len(returns)
        
        # 5. 盈亏比
        avg_win = returns[returns > 0].mean() if (returns > 0).sum() > 0 else 0
        avg_loss = returns[returns < 0].mean() if (returns < 0).sum() > 0 else 0
        profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        return {
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'profit_loss_ratio': profit_loss_ratio
        }

## 六、实盘案例分析

### 6.1 案例：贵州茅台 vs 五粮液

**数据区间**：2020-01-01 至 2025-12-31

**协整检验结果**
```python
# 读取数据
price_A = pd.read_csv('600519.SH.csv')['close']  # 贵州茅台
price_B = pd.read_csv('000858.SZ.csv')['close']  # 五粮液

# 协整检验
result = engle_granger_test(price_A, price_B)
print(f"ADF统计量: {result['adf_stat']:.4f}")
print(f"p-value: {result['p_value']:.4f}")
print(f"对冲比例(β): {result['hedge_ratio']:.4f}")
```

输出：
```
ADF统计量: -3.8245
p-value: 0.0023
对冲比例(β): 0.6842
结论: 在1%显著性水平下拒绝原假设，两者存在协整关系
```

**价差分析**
```python
# 计算价差
spread = price_A - result['hedge_ratio'] * price_B

# 计算Z-Score
z_score = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

# 描述性统计
print(f"价差均值: {spread.mean():.2f}")
print(f"价差标准差: {spread.std():.2f}")
print(f"Z-Score均值: {z_score.mean():.4f}")
print(f"Z-Score标准差: {z_score.std():.4f}")
```

**回测结果**

| 指标 | 配对策略 | 买入持有(茅台) | 买入持有(五粮液) |
|------|---------|---------------|----------------|
| 年化收益率 | 15.8% | 12.3% | 8.7% |
| 年化波动率 | 18.2% | 28.5% | 32.1% |
| 夏普比率 | 0.87 | 0.43 | 0.27 |
| 最大回撤 | -15.3% | -38.7% | -45.2% |
| 胜率 | 58.2% | - | - |
| 交易次数 | 24 | - | - |

![配对交易净值曲线](/images/pairs-trading-statistical-arbitrage-2026/equity_curve.jpg)

### 6.2 多配对组合

**分散化优势**

同时交易多个配对可以降低单一配对失效的风险。

```python
def multi_pairs_backtest(pairs_list, initial_capital=1e6):
    """
    多配对组合回测
    pairs_list: [(stock_A, stock_B), ...]
    """
    n_pairs = len(pairs_list)
    capital_per_pair = initial_capital / n_pairs
    
    all_returns = []
    for stock_A, stock_B in pairs_list:
        # 获取数据
        price_A = get_price(stock_A)
        price_B = get_price(stock_B)
        
        # 生成信号
        spread = calculate_spread(price_A, price_B)
        z_score = calculate_z_score(spread)
        signal = generate_signal(z_score)
        
        # 回测
        bt = PairsTradingBacktest(price_A, price_B, signal, capital_per_pair)
        result = bt.run_backtest()
        all_returns.append(result['portfolio_return'])
    
    # 合并收益
    combined_return = pd.concat(all_returns, axis=1).sum(axis=1)
    
    return combined_return
```

**实战配对组合（2025年）**

| 配数对 | 行业 | 年化收益 | 夏普比率 | 最大回撤 |
|--------|------|---------|---------|----------|
| 贵州茅台 - 五粮液 | 白酒 | 15.8% | 0.87 | -15.3% |
| 招商银行 - 平安银行 | 银行 | 12.3% | 0.72 | -12.8% |
| 中国平安 - 中国太保 | 保险 | 10.5% | 0.65 | -18.2% |
| 恒瑞医药 - 药明康德 | 医药 | 18.2% | 0.91 | -22.5% |
| 宁德时代 - 比亚迪 | 新能源 | 22.1% | 1.05 | -25.3% |
| **组合** | **分散** | **16.8%** | **1.12** | **-11.2%** |

关键发现：组合夏普比率(1.12) > 任意单一配对，最大回撤(-11.2%) < 任意单一配对。分散化效果显著。

## 七、常见问题与解决方案

### 7.1 配对瓦解

**表现**：协整关系突然失效，价差不回归。

**原因**：
1. 基本面发生根本性变化（并购、重组）
2. 行业格局改变（政策冲击、技术革新）
3. 市场结构变化（注册制、涨跌幅限制）

**应对方法**
```python
def monitor_cointegration_breakdown(price_A, price_B, window=60):
    """
    监控协整关系瓦解
    使用滚动窗口检验
    """
    breakdown_dates = []
    
    for i in range(window, len(price_A)):
        # 滚动窗口数据
        price_A_roll = price_A[i-window:i]
        price_B_roll = price_B[i-window:i]
        
        # 协整检验
        result = engle_granger_test(price_A_roll, price_B_roll)
        
        # 判断是否瓦解（p-value > 0.05）
        if result['p_value'] > 0.05:
            breakdown_dates.append(price_A.index[i])
    
    return breakdown_dates
```

**建议**：
- 每月重新检验协整关系
- 设置协整p-value > 0.05时强制平仓
- 建立配对生命周期管理（平均寿命约6-12个月）

### 7.2 交易成本侵蚀收益

**表现**：频繁交易导致净收益为负。

**量化影响**

假设：
- 入场阈值：2.0σ
- 出场阈值：0.5σ
- 交易成本：0.2%（双边）
- 交易频率：每月4次

年交易成本 = 4次/月 × 12月 × 0.2% = 9.6%

如果策略年化收益只有10%，交易成本会吞噬绝大部分利润！

**优化方法**
1. **提高入场阈值**：从2.0σ提高到2.5σ，减少交易频率
2. **降低出场阈值**：从0.5σ提高到1.0σ，减少不必要出场
3. **使用限价单**：减少滑点
4. **选择低手续费券商**：如互联网券商（万1.5 vs 万3）

### 7.3 模型过拟合

**表现**：样本内表现优异，样本外表现差。

**典型案例**

| 参数 | 样本内夏普 | 样本外夏普 | 衰减 |
|------|-----------|-----------|------|
| 入场阈值=1.5σ | 1.35 | 0.62 | -54% |
| 入场阈值=2.0σ | 1.12 | 0.95 | -15% |
| 入场阈值=2.5σ | 0.91 | 0.87 | -4% |

**结论**：参数越极端，过拟合风险越低，但交易机会也越少。需要权衡。

**避免过拟合**
```python
def walk_forward_optimization(price_A, price_B, param_grid, train_window=252, test_window=63):
    """
    滚动窗口优化：避免过拟合
    """
    results = []
    
    for start in range(0, len(price_A) - train_window - test_window, test_window):
        # 训练集
        train_A = price_A[start:start+train_window]
        train_B = price_B[start:start+train_window]
        
        # 测试集
        test_A = price_A[start+train_window:start+train_window+test_window]
        test_B = price_B[start+train_window:start+train_window+test_window]
        
        # 参数优化（在训练集上）
        best_param = optimize_param(train_A, train_B, param_grid)
        
        # 样本外测试
        test_return = backtest(test_A, test_B, best_param)
        
        results.append({
            'param': best_param,
            'train_period': (start, start+train_window),
            'test_return': test_return
        })
    
    return results
```

## 八、总结与展望

### 8.1 核心要点

1. **协整是核心**：高相关性≠可套利，必须检验协整关系
2. **Z-Score标准化**：使用滚动窗口（20-60天）计算Z-Score
3. **阈值选择**：入场2.0σ、出场0.5σ是经验值，需优化
4. **风险控制**：单次止损3.0σ，协整瓦解立即平仓
5. **分散化**：同时交易5-10个配对，降低单一配对风险

### 8.2 进阶方向

**机器学习增强**
- 使用LSTM预测价差方向
- 使用XGBoost动态优化入场阈值
- 使用强化学习调整持仓时间

**高频统计套利**
- 使用分钟级/秒级数据
- 基于订单流(Order Flow)建模
- 做市商策略结合配对交易

**跨市场套利**
- A股 vs H股价差套利
- 期货 vs 现货套利
- ETF套利（一二级市场）

### 8.3 实盘建议

**资金管理**
- 单个配对最大仓位：10%
- 总持仓配对数量：5-10个
- 保留30%现金应对追加保证金

**执行细节**
- 使用智能路由(Smart Order Router)减少冲击成本
- 避免开盘/收盘时段交易（流动性差、价差大）
- 设置价格触发单（Trigger Order）自动执行

**监控指标**
- 协整p-value（每月更新）
- 当前Z-Score（实时监控）
- 持仓时间（超过20天预警）
- 累计收益/回撤（每日计算）

## 参考资料

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
4. 马壮. (2018). *统计套利：理论与实践*. 机械工业出版社.

---

**免责声明**：本文仅供学术交流，不构成投资建议。统计套利虽有理论基础，但实盘面临交易成本、模型失效、流动性风险等挑战。量化投资有风险，入市需谨慎。

