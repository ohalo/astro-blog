---
title: "配对交易与协整分析：统计套利的理论与实践"
description: "深入讲解配对交易的核心原理——协整关系，从理论到实战，教你如何构建稳健的统计套利策略。"
date: 2026-06-16
tags:
  - 量化交易
  - 统计套利
  - 配对交易
  - 协整分析
  - 难度：进阶
image: /images/pair-trading-cointegration/cover.jpg
---

# 配对交易与协整分析：统计套利的理论与实践

## 引言

配对交易（Pairs Trading）是统计套利中最经典的策略之一。其核心思想是通过识别两个具有长期均衡关系的资产，当价格偏离均衡时建立多空对冲头寸，等待价格回归时获利。

与传统的均值回归策略不同，配对交易依赖的是**协整关系（Cointegration）**，这是一种更严格的统计性质。本文将深入探讨：

- 协整关系的数学原理
- 配对选择的定量方法
- 交易信号的构建与优化
- Python实战完整流程

## 一、协整关系：配对交易的基石

### 1.1 平稳性与协整

在时间序列分析中，**平稳性（Stationarity）**是建模的关键前提。一个平稳序列的均值、方差和自协方差不随时间变化。

**单位根检验**：判断序列是否平稳的标准方法

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def adf_test(series, verbose=True):
    """
    Augmented Dickey-Fuller检验
    
    H0: 序列有单位根（非平稳）
    H1: 序列平稳
    
    Parameters:
    -----------
    series : pd.Series
        待检验的时间序列
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict : 检验结果
    """
    result = adfuller(series, autolag='AIC')
    
    output = {
        'ADF Statistic': result[0],
        'p-value': result[1],
        'Critical Values': result[4],
        'is_stationary': result[1] < 0.05
    }
    
    if verbose:
        print('ADF Statistic: {:.4f}'.format(result[0]))
        print('p-value: {:.4f}'.format(result[1]))
        print('Critical Values:')
        for key, value in result[4].items():
            print('\t{}: {:.4f}'.format(key, value))
        print('Stationary: {}'.format(output['is_stationary']))
    
    return output

# 示例使用
# price_a = pd.read_csv('stock_a.csv', index_col=0, parse_dates=True)['close']
# result = adf_test(price_a)
```

### 1.2 协整的定义与检验

两个（或多个）非平稳序列如果存在线性组合是平稳的，则称这些序列是**协整的**。

数学定义：
- 若 $X_t$ 和 $Y_t$ 都是I(1)过程（一阶单整）
- 存在系数 $\beta$，使得 $Z_t = Y_t - \beta X_t$ 是I(0)过程（平稳）
- 则称 $X_t$ 和 $Y_t$ 协整，β为协整系数

**Engle-Granger两步法**：

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import coint

def engle_granger_test(price_a, price_b, verbose=True):
    """
    Engle-Granger协整检验
    
    Parameters:
    -----------
    price_a, price_b : pd.Series
        两个价格序列
    verbose : bool
        是否打印详细信息
    
    Returns:
    --------
    dict : 检验结果
    """
    # 第一步：OLS回归
    model = OLS(price_b, price_a, missing='drop').fit()
    hedge_ratio = model.params[0]
    spread = price_b - hedge_ratio * price_a
    
    # 第二步：检验残差的平稳性
    adf_result = adfuller(spread, autolag='AIC')
    
    # 使用statsmodels内置的协整检验（更严格）
    coint_result = coint(price_a, price_b)
    
    output = {
        'hedge_ratio': hedge_ratio,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'coint_statistic': coint_result[0],
        'coint_pvalue': coint_result[1],
        'is_cointegrated': coint_result[1] < 0.05
    }
    
    if verbose:
        print('Hedge Ratio: {:.4f}'.format(hedge_ratio))
        print('ADF Statistic: {:.4f}'.format(adf_result[0]))
        print('ADF p-value: {:.4f}'.format(adf_result[1]))
        print('Cointegration p-value: {:.4f}'.format(coint_result[1]))
        print('Cointegrated: {}'.format(output['is_cointegrated']))
    
    return output

# 示例使用
# result = engle_granger_test(price_a, price_b)
```

### 1.3 Johansen检验：多变量协整

当处理多资产组合时，Johansen检验更合适。

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多变量）
    
    Parameters:
    -----------
    price_matrix : pd.DataFrame
        价格矩阵（每行是一个时间点，每列是一个资产）
    det_order : int
        确定性项的顺序（0: 无常数项，1: 有常数项，-1: 有常数项和趋势项）
    k_ar_diff : int
        VAR模型中的滞后阶数
    
    Returns:
    --------
    dict : 检验结果
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 提取特征值和迹统计量
    eigenvalues = result.eig
    trace_stat = result.lr1
    max_stat = result.lr2
    
    # 临界值（5%显著性水平）
    critical_values = result.cvt
    
    output = {
        'eigenvalues': eigenvalues,
        'trace_statistic': trace_stat,
        'max_statistic': max_stat,
        'critical_values': critical_values,
        'num_cointegrating_vectors': np.sum(trace_stat > critical_values[:, 1])
    }
    
    return output
```

## 二、配对选择：如何找到好的配对？

### 2.1 距离法（Distance Approach）

最简单的方法：计算价格序列的标准化距离。

```python
from scipy.spatial.distance import pdist, squareform

def distance_method(price_data, lookback=252):
    """
    距离法筛选配对
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据（股票×时间）
    lookback : int
        滚动窗口
    
    Returns:
    --------
    pd.DataFrame : 距离矩阵
    """
    # 标准化价格
    normalized_prices = price_data.apply(
        lambda x: x / x.iloc[0] * 100, axis=0
    )
    
    # 计算SSD（平方和距离）
    ssd_matrix = pd.DataFrame(
        index=price_data.columns,
        columns=price_data.columns
    )
    
    for i, stock_i in enumerate(price_data.columns):
        for j, stock_j in enumerate(price_data.columns):
            if i < j:
                # 计算SSD
                ssd = np.sum(
                    (normalized_prices[stock_i] - normalized_prices[stock_j]) ** 2
                )
                ssd_matrix.loc[stock_i, stock_j] = ssd
                ssd_matrix.loc[stock_j, stock_i] = ssd
    
    return ssd_matrix
```

### 2.2 相关性法

高相关性不一定意味着协整，但可以作为初筛。

```python
def correlation_screening(price_data, threshold=0.8):
    """
    相关性筛选
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据
    threshold : float
        相关性阈值
    
    Returns:
    --------
    list : 高相关性配对列表
    """
    # 计算收益率相关性
    returns = price_data.pct_change().dropna()
    corr_matrix = returns.corr()
    
    # 找出高相关性配对
    pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr = corr_matrix.iloc[i, j]
            if abs(corr) > threshold:
                pairs.append({
                    'stock_a': corr_matrix.columns[i],
                    'stock_b': corr_matrix.columns[j],
                    'correlation': corr
                })
    
    return pairs
```

### 2.3 协整评分法（推荐）

结合协整检验和基本面信息。

```python
def cointegration_screening(price_data, sectors=None):
    """
    协整评分法筛选配对
    
    Parameters:
    -----------
    price_data : pd.DataFrame
        价格数据
    sectors : dict
        股票所属行业（可选）
    
    Returns:
    --------
    pd.DataFrame : 配对评分表
    """
    pairs_scores = []
    
    stocks = price_data.columns
    
    for i in range(len(stocks)):
        for j in range(i+1, len(stocks)):
            stock_a = stocks[i]
            stock_b = stocks[j]
            
            # 协整检验
            try:
                coint_result = coint(
                    price_data[stock_a].dropna(),
                    price_data[stock_b].dropna()
                )
                
                # 计算评分
                pvalue = coint_result[1]
                
                # 基本面评分（如果提供行业信息）
                sector_score = 0
                if sectors is not None:
                    if sectors[stock_a] == sectors[stock_b]:
                        sector_score = 1  # 同行业加分
                
                # 综合评分（p-value越小越好）
                score = -np.log(pvalue) + sector_score
                
                pairs_scores.append({
                    'stock_a': stock_a,
                    'stock_b': stock_b,
                    'p_value': pvalue,
                    'sector_score': sector_score,
                    'total_score': score
                })
            
            except Exception as e:
                # 数据不足或其他错误，跳过
                continue
    
    # 转换为DataFrame并排序
    pairs_df = pd.DataFrame(pairs_scores)
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('total_score', ascending=False)
    
    return pairs_df
```

## 三、交易信号构建

### 3.1 简单Z-Score法

```python
def calculate_zscore(spread, window=20):
    """
    计算价差的Z-Score
    
    Parameters:
    -----------
    spread : pd.Series
        价差序列
    window : int
        滚动窗口
    
    Returns:
    --------
    pd.Series : Z-Score序列
    """
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    
    zscore = (spread - mean) / std
    
    return zscore

def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """
    生成交易信号
    
    Parameters:
    -----------
    zscore : pd.Series
        Z-Score序列
    entry_threshold : float
        入场阈值
    exit_threshold : float
        出场阈值
    
    Returns:
    --------
    pd.DataFrame : 交易信号（1: 多， -1: 空， 0: 平仓）
    """
    signals = pd.DataFrame(index=zscore.index)
    signals['zscore'] = zscore
    signals['position'] = 0
    
    # 入场信号
    signals.loc[signals['zscore'] > entry_threshold, 'position'] = -1  # 做空价差
    signals.loc[signals['zscore'] < -entry_threshold, 'position'] = 1  # 做多价差
    
    # 出场信号（向均值回归）
    signals['position'] = signals['position'].replace(0, np.nan)
    signals['position'] = signals['position'].fillna(method='ffill')
    
    # 当Z-Score回归到出场阈值时平仓
    exit_long = (signals['position'] == 1) & (signals['zscore'] > -exit_threshold)
    exit_short = (signals['position'] == -1) & (signals['zscore'] < exit_threshold)
    
    signals.loc[exit_long | exit_short, 'position'] = 0
    signals['position'] = signals['position'].fillna(0)
    
    return signals
```

### 3.2 卡尔曼滤波法（高级）

卡尔曼滤波可以动态调整对冲比例。

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(price_a, price_b):
    """
    使用卡尔曼滤波动态估计对冲比例
    
    Parameters:
    -----------
    price_a, price_b : pd.Series
        两个价格序列
    
    Returns:
    --------
    dict : 包含动态对冲比例和价差
    """
    # 准备观测矩阵
    X = price_a.values.reshape(-1, 1)
    Y = price_b.values
    
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
    state_means, state_covariances = kf.filter(Y)
    
    # 提取动态对冲比例
    dynamic_hedge_ratio = state_means.flatten()
    
    # 计算动态价差
    spread = price_b - dynamic_hedge_ratio * price_a
    
    return {
        'hedge_ratio': pd.Series(dynamic_hedge_ratio, index=price_a.index),
        'spread': spread
    }
```

### 3.3 机器学习优化（前沿）

使用机器学习模型预测价差的方向。

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def ml_signal_generation(spread, lookback=20, forecast_horizon=5):
    """
    使用随机森林生成交易信号
    
    Parameters:
    -----------
    spread : pd.Series
        价差序列
    lookback : int
        特征回溯期
    forecast_horizon : int
        预测周期
    
    Returns:
    --------
    pd.DataFrame : 包含预测信号的DataFrame
    """
    # 构建特征
    features = pd.DataFrame(index=spread.index)
    features['spread'] = spread
    features['mean_5'] = spread.rolling(window=5).mean()
    features['mean_20'] = spread.rolling(window=20).mean()
    features['std_20'] = spread.rolling(window=20).std()
    features['zscore'] = (spread - features['mean_20']) / features['std_20']
    features['momentum_5'] = spread.diff(5)
    features['volatility'] = spread.rolling(window=20).std()
    
    # 构建标签（未来价差方向）
    future_spread = spread.shift(-forecast_horizon)
    labels = np.sign(future_spread - spread)
    
    # 合并数据
    data = pd.concat([features, labels.rename('label')], axis=1).dropna()
    
    # 训练测试分割
    X = data[features.columns]
    y = data['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测
    predictions = model.predict(X_test)
    
    # 评估
    accuracy = model.score(X_test, y_test)
    print('Model Accuracy: {:.2%}'.format(accuracy))
    
    return predictions, model
```

## 四、风险管理与仓位控制

### 4.1 止损策略

```python
def add_stop_loss(signals, spread, max_loss=0.05, max_holding_period=20):
    """
    添加止损逻辑
    
    Parameters:
    -----------
    signals : pd.DataFrame
        交易信号
    spread : pd.Series
        价差序列
    max_loss : float
        最大亏损比例
    max_holding_period : int
        最大持有期（交易日）
    
    Returns:
    --------
    pd.DataFrame : 更新后的信号
    """
    signals = signals.copy()
    entry_price = None
    holding_days = 0
    
    for i, date in enumerate(signals.index):
        if signals.loc[date, 'position'] != 0:
            if entry_price is None:
                # 新入场
                entry_price = spread.loc[date]
                holding_days = 0
            else:
                # 持有中
                holding_days += 1
                current_price = spread.loc[date]
                
                # 计算亏损比例
                if signals.loc[date, 'position'] == 1:
                    loss = (entry_price - current_price) / entry_price
                else:  # position == -1
                    loss = (current_price - entry_price) / entry_price
                
                # 止损或止时
                if loss > max_loss or holding_days > max_holding_period:
                    signals.loc[date, 'position'] = 0
                    entry_price = None
                    holding_days = 0
        else:
            # 平仓状态
            entry_price = None
            holding_days = 0
    
    return signals
```

### 4.2 仓位管理

```python
def position_sizing(zscore, base_position=10000, max_position=50000):
    """
    根据Z-Score动态调整仓位
    
    Parameters:
    -----------
    zscore : pd.Series
        Z-Score序列
    base_position : float
        基础仓位金额
    max_position : float
        最大仓位金额
    
    Returns:
    --------
    pd.Series : 建议仓位序列
    """
    # Z-Score绝对值越大，仓位越高（但不超过上限）
    position_size = base_position * np.abs(zscore)
    position_size = position_size.clip(upper=max_position)
    
    return position_size
```

## 五、实战案例：A股配对交易

### 5.1 数据获取与预处理

```python
import tushare as ts

# 设置tushare pro token
ts.set_token('your_token_here')
pro = ts.pro_api()

def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票数据
    """
    df = pro.daily(
        ts_code=stock_code,
        start_date=start_date,
        end_date=end_date
    )
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df = df.set_index('trade_date').sort_index()
    
    return df['close']

# 示例：获取茅台和五粮液的数据
moutai = get_stock_data('600519.SH', '20200101', '20241231')
wuliangye = get_stock_data('000858.SZ', '20200101', '20241231')

# 对齐数据
prices = pd.concat(
    [moutai.rename('moutai'), wuliangye.rename('wuliangye')],
    axis=1
).dropna()
```

### 5.2 协整检验与信号生成

```python
# 协整检验
coint_result = engle_granger_test(
    prices['moutai'], 
    prices['wuliangye']
)

print('协整检验结果：')
print('对冲比例: {:.4f}'.format(coint_result['hedge_ratio']))
print('协整p-value: {:.4f}'.format(coint_result['coint_pvalue']))

# 计算价差
hedge_ratio = coint_result['hedge_ratio']
spread = prices['wuliangye'] - hedge_ratio * prices['moutai']

# 计算Z-Score
zscore = calculate_zscore(spread, window=20)

# 生成信号
signals = generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5)

# 添加止损
signals = add_stop_loss(signals, spread, max_loss=0.05, max_holding_period=20)
```

### 5.3 回测与绩效分析

```python
def backtest_pair_trading(prices, signals, hedge_ratio):
    """
    回测配对交易策略
    
    Parameters:
    -----------
    prices : pd.DataFrame
        价格数据
    signals : pd.DataFrame
        交易信号
    hedge_ratio : float
        对冲比例
    
    Returns:
    --------
    pd.DataFrame : 回测结果
    """
    # 计算策略收益
    stock_a_ret = prices.iloc[:, 0].pct_change()
    stock_b_ret = prices.iloc[:, 1].pct_change()
    
    # 策略收益 = 信号 * (股票B收益 - 对冲比例 * 股票A收益)
    strategy_ret = signals['position'].shift(1) * \
                   (stock_b_ret - hedge_ratio * stock_a_ret)
    
    # 累积收益
    cumulative_ret = (1 + strategy_ret).cumprod()
    
    # 计算绩效指标
    total_return = cumulative_ret.iloc[-1] - 1
    annual_return = strategy_ret.mean() * 252
    annual_vol = strategy_ret.std() * np.sqrt(252)
    sharpe = annual_return / annual_vol if annual_vol != 0 else 0
    max_dd = (cumulative_ret / cumulative_ret.cummax() - 1).min()
    
    performance = {
        'Total Return': '{:.2%}'.format(total_return),
        'Annual Return': '{:.2%}'.format(annual_return),
        'Annual Volatility': '{:.2%}'.format(annual_vol),
        'Sharpe Ratio': '{:.2f}'.format(sharpe),
        'Max Drawdown': '{:.2%}'.format(max_dd),
        'Win Rate': '{:.2%}'.format(
            (strategy_ret > 0).sum() / (strategy_ret != 0).sum()
        )
    }
    
    return strategy_ret, cumulative_ret, performance

# 执行回测
strategy_ret, cumulative_ret, performance = backtest_pair_trading(
    prices, signals, hedge_ratio
)

print('\n策略绩效：')
for key, value in performance.items():
    print('{}: {}'.format(key, value))
```

### 5.4 可视化

```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')

fig, axes = plt.subplots(4, 1, figsize=(15, 16))

# 1. 价格序列
ax1 = axes[0]
ax1.plot(prices.index, prices.iloc[:, 0], label=prices.columns[0])
ax1.plot(prices.index, prices.iloc[:, 1], label=prices.columns[1])
ax1.set_title('Price Series')
ax1.legend()

# 2. 价差与Z-Score
ax2 = axes[1]
ax2.plot(spread.index, spread, label='Spread', color='blue')
ax2.set_title('Spread')
ax2_twin = ax2.twinx()
ax2_twin.plot(zscore.index, zscore, label='Z-Score', color='red', alpha=0.6)
ax2_twin.axhline(y=2.0, color='black', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=-2.0, color='black', linestyle='--', alpha=0.5)
ax2_twin.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5)
ax2_twin.axhline(y=-0.5, color='gray', linestyle=':', alpha=0.5)

# 3. 交易信号
ax3 = axes[2]
ax3.plot(zscore.index, zscore, color='gray', alpha=0.5)
for i in range(len(signals)):
    if signals['position'].iloc[i] == 1:
        ax3.scatter(signals.index[i], zscore.iloc[i], 
                   color='green', marker='^', s=100, label='Long' if i==0 else '')
    elif signals['position'].iloc[i] == -1:
        ax3.scatter(signals.index[i], zscore.iloc[i], 
                   color='red', marker='v', s=100, label='Short' if i==0 else '')
ax3.set_title('Trading Signals')

# 4. 累积收益
ax4 = axes[3]
ax4.plot(cumulative_ret.index, cumulative_ret, label='Strategy', color='blue')
ax4.plot(cumulative_ret.index, 
         (1 + prices.iloc[:, 1].pct_change()).cumprod(), 
         label='Buy & Hold (Stock B)', color='gray', alpha=0.5)
ax4.set_title('Cumulative Returns')
ax4.legend()

plt.tight_layout()
plt.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
```

## 六、实战中的注意事项

### 6.1 数据质量问题

- **幸存者偏差**：确保使用的数据包含已退市的股票
- **前复权调整**：必须使用前复权价格，避免分红送股影响
- **停牌处理**：停牌期间的信号需要向前填充或跳过

### 6.2 交易成本

```python
def add_transaction_costs(strategy_ret, signals, commission=0.0003, 
                         slippage=0.001):
    """
    加入交易成本
    
    Parameters:
    -----------
    strategy_ret : pd.Series
        策略收益率
    signals : pd.DataFrame
        交易信号
    commission : float
        佣金比例
    slippage : float
        滑点比例
    
    Returns:
    --------
    pd.Series : 净收益
    """
    # 计算换手
    position_change = signals['position'].diff().abs()
    
    # 交易成本 = 佣金 + 滑点
    transaction_cost = position_change * (commission + slippage)
    
    # 净收益
    net_ret = strategy_ret - transaction_cost
    
    return net_ret
```

### 6.3 风险控制

**核心原则**：
1. 单一配对持仓不超过总资金的10%
2. 同时持有5-10个低相关性配对
3. 设置配对级别止损（5%）和组合级别止损（15%）
4. 定期重新检验协整关系（月度）

## 七、总结与展望

### 7.1 配对交易的优缺点

**优点**：
- 市场中性，降低系统性风险
- 收益来源明确（均值回归）
- 适合震荡市场

**缺点**：
- 趋势市场中表现差
- 协整关系可能断裂
- 交易成本敏感

### 7.2 改进方向

1. **多因子配对**：不只依赖价格，加入基本面、技术面因子
2. **机器学习**：用深度学习模型捕捉非线性关系
3. **高频数据**：利用分钟级或tick级数据提高频率
4. **跨市场配对**：不同交易所、不同国家的同类资产

### 7.3 实盘建议

- **先模拟后实盘**：至少3个月模拟盘验证
- **小资金起步**：初始资金不超过总资金的20%
- **持续优化**：每月回顾策略表现，调整参数
- **保持耐心**：配对交易是概率游戏，需要长期坚持

---

## 参考文献

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
2. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques." Wiley.
3. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." Econometrica.
4. Johansen, S. (1991). "Estimation and Hypothesis Testing of Cointegration Vectors in Gaussian Vector Autoregressive Models." Econometrica.
5. Elliott, G., Rothenberg, T. J., & Stock, J. H. (1996). "Efficient Tests for an Autoregressive Unit Root." Econometrica.

## 代码仓库

完整代码已上传至GitHub：  
[https://github.com/quantstrategy/pair-trading-cointegration](https://github.com/quantstrategy/pair-trading-cointegration)

## 附录：常用Python库

```python
# 数据处理
import pandas as pd
import numpy as np

# 统计分析
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.tsa.vector_ar.vecm import coint_johansen

# 机器学习
from sklearn.ensemble import RandomForestClassifier
from pykalman import KalmanFilter

# 可视化
import matplotlib.pyplot as plt
import seaborn as sns

# 金融数据
import tushare as ts  # A股数据
import yfinance as yf  # 美股数据
```

---

**免责声明**：本文仅供学术研究和教育目的，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。在实际应用中，请结合专业投资顾问的意见，并进行充分的风险评估。
