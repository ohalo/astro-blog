---
title: "配对交易与协整分析：统计套利的核心技术"
publishDate: '2026-06-20'
description: "配对交易与协整分析：统计套利的核心技术 - 量化交易实战指南"
tags:
 - 量化交易
 - 统计套利
 - 协整分析
 - Python实战
language: Chinese
---

## 引言：从"两只股票的故事"说起

1980年代，摩根士丹利的一位交易员发现了一个有趣的现象：**可口可乐（KO）和百事可乐（PEP）的股价走势高度相关，但偶尔会出现偏离**。当两者的价差扩大到一定程度后，往往会回归到历史均值水平。

这不仅仅是巧合。可口可乐和百事可乐处于同一个行业、面临相同的宏观环境、拥有相似的商业模式。理论上，它们的估值差距不应该长期存在。

这就是**配对交易（Pairs Trading）**的核心思想：找到两个（或一组）价格走势具有长期均衡关系的股票，当它们的价差暂时偏离时，做多低估的、做空高估的，等待价差回归获利。

配对交易属于**统计套利（Statistical Arbitrage）**的一种，它的优势在于：
1. **市场中性**：多空对冲，不受大盘方向影响
2. **低风险**：依赖均值回归，而非趋势预测
3. **收益稳定**：在市场震荡期尤其有效

但配对交易也有一个核心难题：**如何找到真正具有长期均衡关系的股票对？**

答案就是本文的主角：**协整分析（Cointegration Analysis）**。

## 一、协整：比相关性更深层的关系

### 1.1 相关性 vs 协整性

很多初学者会混淆"相关性"和"协整性"，但它们是完全不同的概念。

**相关性（Correlation）**衡量的是两个序列**同期**的线性关系。即使两个序列完全不同步，只要它们的变化方向一致，就可能高度相关。

**协整性（Cointegration）**衡量的是两个（或多个）非平稳序列的**线性组合是否是平稳的**。换句话说，协整关系意味着两个序列之间存在**长期的均衡关系**，即使短期内会偏离，但长期会回归。

**举个例子**：

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# 生成两个协整序列
np.random.seed(42)
n = 1000
t = np.arange(n)

# 共同趋势
common_trend = 0.01 * t + np.cumsum(np.random.normal(0, 1, n))

# 两个序列围绕共同趋势波动
x = common_trend + np.random.normal(0, 0.5, n)
y = 1.5 * common_trend + np.random.normal(0, 0.5, n)

# 计算相关性
correlation = np.corrcoef(x, y)[0, 1]
print(f"相关性: {correlation:.4f}")

# 检验协整性
score, p_value, _ = coint(x, y)
print(f"协整检验 p-value: {p_value:.4f}")

# 可视化
fig, axes = plt.subplots(2, 1, figsize=(14, 10))

axes[0].plot(t, x, label='X', alpha=0.7)
axes[0].plot(t, y, label='Y', alpha=0.7)
axes[0].set_title(f'两个序列的价格走势（相关性={correlation:.4f}）', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 计算价差
spread = y - x
axes[1].plot(t, spread, label='价差 (Y - X)', color='green', alpha=0.7)
axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
axes[1].set_title(f'价差序列（协整检验 p-value={p_value:.4f}）', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/correlation-vs-cointegration.png', 
            dpi=300, bbox_inches='tight')
plt.show()
```

![相关性 vs 协整性示意图](/images/pair-trading-cointegration/correlation-vs-cointegration.png)

**关键洞察**：
- 两个序列可能高度相关，但不协整（比如同向但不同速的随机游走）
- 两个序列可能协整，但相关性不高（比如存在长期均衡关系，但短期波动不同步）
- **配对交易需要的是协整性，而非相关性**

### 1.2 协整的数学定义

严格来说，对于两个时间序列 \(X_t\) 和 \(Y_t\)：

如果：
1. \(X_t\) 和 \(Y_t\) 都是 **I(1)** 过程（即一阶差分后平稳）
2. 存在一个系数 \(\beta\)，使得 \(Y_t - \beta X_t = Z_t\) 是 **I(0)** 过程（即平稳）

那么 \(X_t\) 和 \(Y_t\) 是协整的，\(\beta\) 称为**协整系数**。

在实际应用中，我们通常用**Engle-Granger两步法**来检验协整：
1. 用OLS回归 \(Y_t = \alpha + \beta X_t + \epsilon_t\)
2. 对残差 \(\epsilon_t\) 进行**单位根检验**（如ADF检验）

如果残差是平稳的，则说明 \(X_t\) 和 \(Y_t\) 协整。

## 二、配对交易的完整流程

一个完整的配对交易系统包括以下步骤：

```
1. 股票池筛选 → 2. 候选对生成 → 3. 协整检验 → 4. 配对筛选 → 5. 交易信号生成 → 6. 风险控制 → 7. 执行与监控
```

让我们用Python逐步实现。

### Step 1: 股票池筛选

不是所有股票都适合做配对交易。筛选标准包括：
- 同行业（确保有共同的基本面驱动因素）
- 相似市值（避免流动性差异过大）
- 上市时间足够长（至少有2-3年历史数据）

```python
import yfinance as yf
import pandas as pd

# 示例：从同行业股票中筛选
# 这里以A股银行股为例
bank_stocks = [
    '601398.SS',  # 工商银行
    '601939.SS',  # 建设银行
    '601288.SS',  # 农业银行
    '601988.SS',  # 中国银行
    '600036.SS',  # 招商银行
    '601166.SS',  # 兴业银行
    '601328.SS',  # 交通银行
    '601998.SS',  # 中信银行
]

# 下载历史数据
start_date = '2020-01-01'
end_date = '2026-06-01'

data = yf.download(bank_stocks, start=start_date, end=end_date)['Adj Close']
data = data.dropna(axis=1)  # 去掉数据不完整的股票

print(f"成功下载 {data.shape[1]} 只股票的数据")
print(f"时间范围: {data.index[0]} 至 {data.index[-1]}")
print(f"数据形状: {data.shape}")
```

### Step 2: 候选对生成

对于有 \(N\) 只股票的股票池，理论上可以生成 \(C_N^2 = \frac{N(N-1)}{2}\) 个候选对。

但对于大股票池（如 \(N=500\)），全组合扫描计算量太大。可以用**预筛选**来减少候选对：
- 相关性筛选：只保留相关系数 > 0.5 的对
- 行业分类：只在同一二级行业内部配对

```python
from itertools import combinations

# 生成所有候选对
candidate_pairs = list(combinations(data.columns, 2))
print(f"候选对数量: {len(candidate_pairs)}")

# 预筛选：相关性筛选
correlation_matrix = data.corr()
filtered_pairs = []

for stock1, stock2 in candidate_pairs:
    corr = correlation_matrix.loc[stock1, stock2]
    if corr > 0.6:  # 相关性阈值
        filtered_pairs.append((stock1, stock2, corr))

print(f"相关性筛选后剩余: {len(filtered_pairs)} 对")
```

### Step 3: 协整检验

对每个候选对进行协整检验。这里使用**Engle-Granger检验**。

```python
from statsmodels.tsa.stattools import coint
import warnings
warnings.filterwarnings('ignore')

def test_cointegration(price1, price2, significance_level=0.05):
    """
    检验两个价格序列的协整性
    
    Returns:
    --------
    is_cointegrated : bool
        是否协整
    p_value : float
        协整检验的p-value
    hedge_ratio : float
        协整系数（对冲比率）
    """
    
    # Engle-Granger检验
    score, p_value, _ = coint(price1, price2)
    
    # 计算对冲比率（用OLS回归）
    from sklearn.linear_model import LinearRegression
    X = price1.values.reshape(-1, 1)
    y = price2.values
    model = LinearRegression()
    model.fit(X, y)
    hedge_ratio = model.coef_[0]
    
    is_cointegrated = p_value < significance_level
    
    return is_cointegrated, p_value, hedge_ratio

# 对所有候选对进行协整检验
cointegrated_pairs = []

for stock1, stock2, corr in filtered_pairs:
    price1 = data[stock1]
    price2 = data[stock2]
    
    is_cointegrated, p_value, hedge_ratio = test_cointegration(price1, price2)
    
    if is_cointegrated:
        cointegrated_pairs.append({
            'stock1': stock1,
            'stock2': stock2,
            'correlation': corr,
            'p_value': p_value,
            'hedge_ratio': hedge_ratio
        })

print(f"协整检验后剩余: {len(cointegrated_pairs)} 对")
```

### Step 4: 配对筛选

协整检验通过只是第一步。还需要评估配对的质量：

1. **价差的均值回归速度**：用**半衰期（Half-life）**衡量
2. **价差的波动性**：波动性太小，套利空间有限；太大，风险高
3. **历史表现**：模拟历史交易，计算夏普比率

```python
from statsmodels.tsa.ar_model import AutoReg

def calculate_half_life(spread):
    """
    计算价差的半衰期（均值回归速度）
    
    使用AR(1)模型：spread_t = alpha + beta * spread_{t-1} + epsilon_t
    半衰期 = -log(2) / log(beta)
    """
    
    model = AutoReg(spread, lags=1)
    results = model.fit()
    beta = results.params[1]
    
    if beta >= 1:
        return np.inf  # 不均值回归
    
    half_life = -np.log(2) / np.log(beta)
    
    return half_life

def evaluate_pair(price1, price2, hedge_ratio):
    """
    评估一个配对的质量
    """
    
    # 计算价差（对冲后）
    spread = price2 - hedge_ratio * price1
    
    # 标准化价差（方便比较）
    normalized_spread = (spread - spread.mean()) / spread.std()
    
    # 计算半衰期
    half_life = calculate_half_life(normalized_spread)
    
    # 计算价差的夏普比率（假设均值回归交易）
    # 简化：每天做多低估、做空高估
    z_score = normalized_spread
    signals = -np.sign(z_score)  # 负z-score做多，正z-score做空
    returns = signals.shift(1) * normalized_spread.diff()
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
    
    return {
        'half_life': half_life,
        'sharpe_ratio': sharpe_ratio,
        'spread_volatility': normalized_spread.std()
    }

# 评估所有协整对
for pair in cointegrated_pairs:
    price1 = data[pair['stock1']]
    price2 = data[pair['stock2']]
    hedge_ratio = pair['hedge_ratio']
    
    metrics = evaluate_pair(price1, price2, hedge_ratio)
    
    pair.update(metrics)

# 排序：按夏普比率降序
cointegrated_pairs.sort(key=lambda x: x['sharpe_ratio'], reverse=True)

print("\n=== Top 5 配对 ===")
for i, pair in enumerate(cointegrated_pairs[:5]):
    print(f"{i+1}. {pair['stock1']} - {pair['stock2']}")
    print(f"   相关性: {pair['correlation']:.4f}, p-value: {pair['p_value']:.4f}")
    print(f"   半衰期: {pair['half_life']:.1f} 天, 夏普比率: {pair['sharpe_ratio']:.4f}")
    print()
```

## 三、交易信号生成

找到优质配对后，下一步是生成具体的交易信号。主流方法有两种：

### 方法1：Z-Score阈值法

计算价差的Z-Score（标准化值），当 |Z-Score| 超过某个阈值（如1.5或2.0）时开仓，当 Z-Score 回归到0附近时平仓。

```python
def generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5):
    """
    基于Z-Score生成交易信号
    
    Returns:
    --------
    signals : DataFrame
        包含 'long' 和 'short' 两列，表示做多和做空信号
    """
    
    # 计算Z-Score
    z_score = (spread - spread.rolling(60).mean()) / spread.rolling(60).std()
    
    # 初始化信号
    signals = pd.DataFrame(index=spread.index, columns=['long', 'short'])
    signals[:] = 0
    
    # 生成信号
    position = 0  # 当前持仓：1表示做多价差，-1表示做空价差，0表示空仓
    
    for i in range(1, len(z_score)):
        if position == 0:
            # 空仓状态：检查是否开仓
            if z_score.iloc[i] < -entry_threshold:
                # 价差低估：做多价差（做多stock1，做空stock2）
                signals.iloc[i]['long'] = 1
                position = 1
            elif z_score.iloc[i] > entry_threshold:
                # 价差高估：做空价差（做空stock1，做多stock2）
                signals.iloc[i]['short'] = 1
                position = -1
        else:
            # 持仓状态：检查是否平仓
            if abs(z_score.iloc[i]) < exit_threshold:
                # 价差回归：平仓
                signals.iloc[i]['long'] = -1 if position == 1 else 0
                signals.iloc[i]['short'] = -1 if position == -1 else 0
                position = 0
    
    return signals, z_score

# 示例：为最优配对生成交易信号
best_pair = cointegrated_pairs[0]
price1 = data[best_pair['stock1']]
price2 = data[best_pair['stock2']]
hedge_ratio = best_pair['hedge_ratio']

spread = price2 - hedge_ratio * price1
signals, z_score = generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5)

# 可视化信号
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：价格走势
axes[0].plot(price1.index, price1 / price1.iloc[0], label=best_pair['stock1'], alpha=0.7)
axes[0].plot(price2.index, price2 / price2.iloc[0], label=best_pair['stock2'], alpha=0.7)
axes[0].set_title('标准化价格走势', fontsize=12)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 子图2：Z-Score
axes[1].plot(z_score.index, z_score, label='Z-Score', alpha=0.7)
axes[1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Entry Threshold')
axes[1].axhline(y=-2.0, color='red', linestyle='--', alpha=0.5)
axes[1].axhline(y=0.5, color='green', linestyle='--', alpha=0.5, label='Exit Threshold')
axes[1].axhline(y=-0.5, color='green', linestyle='--', alpha=0.5)
axes[1].set_title('价差的Z-Score', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 子图3：交易信号
axes[2].plot(z_score.index, z_score, label='Z-Score', alpha=0.3)
long_signals = signals[signals['long'] == 1]
short_signals = signals[signals['short'] == 1]
axes[2].scatter(long_signals.index, z_score[long_signals.index], 
                color='green', marker='^', s=100, label='Long Entry', alpha=0.7)
axes[2].scatter(short_signals.index, z_score[short_signals.index], 
                color='red', marker='v', s=100, label='Short Entry', alpha=0.7)
axes[2].set_title('交易信号', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/trading-signals.png', 
            dpi=300, bbox_inches='tight')
plt.show()
```

![配对交易信号生成示意图](/images/pair-trading-cointegration/trading-signals.png)

### 方法2：卡尔曼滤波法

Z-Score阈值法的缺点是**阈值固定**，无法适应价差波动性的变化。更高级的方法是使用**卡尔曼滤波（Kalman Filter）**来动态估计对冲比率和价差均值。

```python
from pykalman import KalmanFilter

def kalman_filter_pairs_trading(price1, price2):
    """
    使用卡尔曼滤波动态估计对冲比率
    """
    
    # 观测矩阵：price2是观测值，price1是状态的系数
    observations = price2.values
    X = price1.values.reshape(-1, 1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(1),
        observation_matrices=X.reshape(-1, 1, 1),
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=0.01
    )
    
    # 滤波
    state_means, state_covariances = kf.filter(observations)
    
    # 动态对冲比率
    dynamic_hedge_ratio = state_means.flatten()
    
    # 动态价差
    dynamic_spread = price2 - dynamic_hedge_ratio * price1
    
    return dynamic_spread, dynamic_hedge_ratio

# 应用卡尔曼滤波
dynamic_spread, dynamic_hedge_ratio = kalman_filter_pairs_trading(price1, price2)

# 可视化动态对冲比率
plt.figure(figsize=(14, 6))
plt.plot(price1.index, dynamic_hedge_ratio, label='动态对冲比率', linewidth=2)
plt.axhline(y=hedge_ratio, color='red', linestyle='--', label='静态对冲比率')
plt.xlabel('日期', fontsize=12)
plt.ylabel('对冲比率', fontsize=12)
plt.title('动态 vs 静态对冲比率', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/dynamic-hedge-ratio.png', 
            dpi=300, bbox_inches='tight')
plt.show()
```

![动态对冲比率示意图](/images/pair-trading-cointegration/dynamic-hedge-ratio.png)

## 四、风险控制：配对交易的致命弱点

配对交易看似低风险，但实际上有几个**致命弱点**：

### 风险1：协整关系破裂

协整关系是建立在历史数据上的统计规律，**不保证未来继续存在**。当两只股票的基本面发生根本性变化（如并购、行业变革），协整关系可能永久破裂。

**案例**：2008年金融危机期间，很多原本协整的银行股对突然失效——因为它们面对的不再是共同的行业因素，而是各自的流动性危机。

**应对策略**：
- **定期重新检验协整性**（如每季度）
- **设置止损**：当价差超过历史极值的X倍时（如3倍标准差），强制平仓
- **限制单对仓位**：不要把所有资金都押在一个 pair 上

```python
def check_cointegration_breakdown(spread, window=60, significance_level=0.05):
    """
    滚动检验协整关系是否破裂
    """
    
    breakdown_dates = []
    
    for i in range(window, len(spread)):
        window_data = spread[i-window:i]
        
        # ADF检验
        from statsmodels.tsa.stattools import adfuller
        result = adfuller(window_data)
        p_value = result[1]
        
        if p_value > significance_level:
            # 协整关系破裂
            breakdown_dates.append(spread.index[i])
    
    return breakdown_dates

# 检验最优配对的协整关系是否稳定
breakdown_dates = check_cointegration_breakdown(spread, window=60)
print(f"协整关系破裂的日期: {breakdown_dates[:5]}...")  # 只显示前5个
```

### 风险2：执行风险

理论上，配对交易是"同时做多和做空"。但实盘中，两笔订单的执行可能存在**时间差**，导致暴露在市场风险中。

**应对策略**：
- 使用**算法交易**（如VWAP、TWAP）来减少冲击成本
- 优先保证**主订单**（通常是流动性更好的那只股票）的执行
- 设置**容忍时间窗口**（如5分钟内完成两笔订单）

### 风险3：融资成本

配对交易需要**做空**，而做空涉及：
- 融券费用（尤其对于难以借到的股票）
- 卖空收益的利息损失

如果价差收敛的速度太慢，融资成本可能吃掉全部利润。

**应对策略**：
- 优先选择**融券成本低**的股票对
- 计算**盈亏平衡点**：价差需要收敛多少，才能覆盖融资成本
- 避免持有超过**半衰期 × 2**的时间

## 五、Python实战：完整回测框架

让我们将上述所有组件整合起来，构建一个完整的配对交易回测框架。

```python
class PairsTradingBacktester:
    """配对交易回测框架"""
    
    def __init__(self, data, pair, entry_threshold=2.0, exit_threshold=0.5, 
                 initial_capital=1000000, transaction_cost=0.001):
        """
        初始化回测框架
        
        Parameters:
        -----------
        data : DataFrame
            价格数据
        pair : dict
            配对信息（包含stock1, stock2, hedge_ratio）
        entry_threshold : float
            开仓阈值（Z-Score）
        exit_threshold : float
            平仓阈值（Z-Score）
        initial_capital : float
            初始资金
        transaction_cost : float
            交易成本（单边）
        """
        
        self.data = data
        self.stock1 = pair['stock1']
        self.stock2 = pair['stock2']
        self.hedge_ratio = pair['hedge_ratio']
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
        # 计算价差
        self.spread = data[self.stock2] - self.hedge_ratio * data[self.stock1]
        
        # 初始化投资组合
        self.portfolio = pd.DataFrame(index=data.index, columns=[
            'cash', 'stock1_shares', 'stock2_shares', 'stock1_value', 'stock2_value',
            'total_value', 'returns'
        ])
        self.portfolio['cash'] = initial_capital
        self.portfolio[['stock1_shares', 'stock2_shares']] = 0
        self.portfolio[['stock1_value', 'stock2_value']] = 0.0
        self.portfolio['total_value'] = initial_capital
        
        # 交易记录
        self.trades = []
        
    def run_backtest(self):
        """运行回测"""
        
        # 计算Z-Score
        z_score = (self.spread - self.spread.rolling(60).mean()) / self.spread.rolling(60).std()
        
        position = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
        
        for i in range(60, len(self.portfolio)):  # 前60天用来计算滚动统计量
            current_date = self.portfolio.index[i]
            prev_date = self.portfolio.index[i-1]
            
            # 当前价格
            price1 = self.data[self.stock1].iloc[i]
            price2 = self.data[self.stock2].iloc[i]
            
            # 前一天的Z-Score
            z = z_score.iloc[i-1]
            
            # 交易逻辑
            if position == 0:
                # 空仓：检查是否开仓
                if z < -self.entry_threshold:
                    # 做多价差：做多stock1，做空stock2
                    self._open_position(prev_date, current_date, price1, price2, 1)
                    position = 1
                elif z > self.entry_threshold:
                    # 做空价差：做空stock1，做多stock2
                    self._open_position(prev_date, current_date, price1, price2, -1)
                    position = -1
                    
            elif position == 1:
                # 做多价差：检查是否平仓
                if z >= -self.exit_threshold:
                    self._close_position(prev_date, current_date, price1, price2)
                    position = 0
                    
            elif position == -1:
                # 做空价差：检查是否平仓
                if z <= self.exit_threshold:
                    self._close_position(prev_date, current_date, price1, price2)
                    position = 0
            
            # 更新投资组合价值
            self._update_portfolio_value(current_date, price1, price2)
        
        # 计算回测指标
        self._calculate_metrics()
        
        return self.portfolio, self.trades
    
    def _open_position(self, prev_date, current_date, price1, price2, direction):
        """开仓"""
        
        # 计算可买股数（假设用一半现金买stock1，另一半做空stock2）
        available_cash = self.portfolio.loc[prev_date, 'cash']
        shares1 = int(available_cash / 2 / price1 / 100) * 100  # 按手数取整
        shares2 = int(shares1 * self.hedge_ratio / 100) * 100
        
        # 执行交易
        if direction == 1:
            # 做多stock1，做空stock2
            self.portfolio.loc[current_date, 'stock1_shares'] = shares1
            self.portfolio.loc[current_date, 'stock2_shares'] = -shares2
        else:
            # 做空stock1，做多stock2
            self.portfolio.loc[current_date, 'stock1_shares'] = -shares1
            self.portfolio.loc[current_date, 'stock2_shares'] = shares2
        
        # 记录交易
        self.trades.append({
            'date': current_date,
            'action': 'OPEN',
            'direction': direction,
            'price1': price1,
            'price2': price2,
            'shares1': shares1,
            'shares2': shares2
        })
    
    def _close_position(self, prev_date, current_date, price1, price2):
        """平仓"""
        
        # 平掉所有仓位
        self.portfolio.loc[current_date, 'stock1_shares'] = 0
        self.portfolio.loc[current_date, 'stock2_shares'] = 0
        
        # 记录交易
        self.trades.append({
            'date': current_date,
            'action': 'CLOSE',
            'price1': price1,
            'price2': price2
        })
    
    def _update_portfolio_value(self, date, price1, price2):
        """更新投资组合价值"""
        
        # 股票市值
        stock1_value = self.portfolio.loc[date, 'stock1_shares'] * price1
        stock2_value = self.portfolio.loc[date, 'stock2_shares'] * price2
        
        self.portfolio.loc[date, 'stock1_value'] = stock1_value
        self.portfolio.loc[date, 'stock2_value'] = stock2_value
        
        # 总市值
        total_value = self.portfolio.loc[date, 'cash'] + stock1_value + stock2_value
        self.portfolio.loc[date, 'total_value'] = total_value
        
        # 日收益率
        if date != self.portfolio.index[0]:
            prev_value = self.portfolio.loc[self.portfolio.index.get_loc(date)-1, 'total_value']
            daily_return = (total_value - prev_value) / prev_value
            self.portfolio.loc[date, 'returns'] = daily_return
    
    def _calculate_metrics(self):
        """计算回测指标"""
        
        returns = self.portfolio['returns'].dropna()
        
        # 总收益率
        total_return = (self.portfolio['total_value'].iloc[-1] - self.initial_capital) / self.initial_capital
        
        # 年化收益率
        trading_days = len(returns)
        annual_return = (1 + total_return) ** (252 / trading_days) - 1
        
        # 年化波动率
        annual_volatility = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe_ratio = annual_return / annual_volatility if annual_volatility > 0 else 0
        
        # 最大回撤
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        print("=== 回测结果 ===")
        print(f"总收益率: {total_return:.2%}")
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {annual_volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"交易次数: {len(self.trades)}")

# 运行回测
backtester = PairsTradingBacktester(data, best_pair)
portfolio, trades = backtester.run_backtest()
```

## 六、总结：配对交易的利与弊

### 优势

1. **市场中性**：多空对冲，不受大盘涨跌影响，适合震荡市
2. **低风险**：依赖均值回归，理论上风险有限
3. **收益稳定**：如果配对选择得当，可以带来稳定的alpha

### 劣势

1. **协整关系破裂风险**：这是配对交易的"黑天鹅"
2. **执行难度大**：需要同时管理多对交易，对系统要求高
3. **融资成本**：做空涉及融券成本，可能侵蚀利润
4. **容量限制**：太 popular 的配对，套利空间会被迅速抹平

### 实战建议

1. **分散投资**：同时交易10-20个配对，降低单对风险
2. **动态调整**：定期重新检验协整性，剔除失效的配对
3. **严格止损**：设置合理的止损线，不要幻想"价差总会回归"
4. **关注基本面**：协整是统计规律，但基本面的变化才是根本

配对交易不是"印钞机"，而是一种**需要持续维护和优化的策略**。那些能把配对交易做好的团队，往往拥有：
- 强大的数据采集和处理能力
- 高效的执行系统（减少滑点和冲击成本）
- 严谨的风险管理体系

对于个人投资者，我的建议是：**从小资金开始，只交易你最熟悉的行业，并且永远不要满仓**。

统计套利的本质，不是在市场中"战胜所有人"，而是在市场的无效性中，"捡起那些别人掉落的硬币"。

---
**相关阅读**：
- [统计套利：均值回归策略的深度解析与Python实战](/blog/statistical-arbitrage-mean-reversion/)
- [量化回测框架搭建实战：从零到第一笔虚拟交易](/blog/quant-backtesting-vectorbt-guide/)
- [波动率风险溢价捕捉：期权波动率交易的核心策略](/blog/volatility-risk-premium/)
