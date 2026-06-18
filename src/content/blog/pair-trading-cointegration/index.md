---
title: "配对交易与协整分析"
description: "深入讲解配对交易的理论基础——协整关系，包括Engle-Granger检验、Johansen检验、配对选择方法，以及完整的Python实战策略和风险控制。"
pubDate: 2026-06-19
tags: ["配对交易", "协整分析", "统计套利", "均值回归", "量化策略"]
category: "统计套利"
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

在传统的主观交易和量化交易中，**方向性交易**（Directional Trading）占据主导地位——投资者试图预测资产价格的上涨或下跌，从而获利。然而，方向性交易面临一个核心难题：**市场方向难以预测**。

配对交易（Pair Trading）提供了一种不同的思路：它不依赖对市场方向的判断，而是通过捕捉**两个高度相关资产之间的相对价值偏离**来获利。这种策略属于**市场中性策略**（Market Neutral Strategy），能够在牛市、熊市，甚至震荡市中实现稳健收益。

配对交易的核心理论基础是**协整关系**（Cointegration）。本文将深入讲解协整分析的原理、配对交易策略的构建方法，以及完整的Python实战案例。

## 一、配对交易的基本原理

### 1.1 什么是配对交易？

配对交易是一种**统计套利策略**，其基本思想是：

1. **找到一对高度相关的资产**（如两只同行业股票、期货与现货、ETF与成分股等）
2. **计算价差的均衡水平**（通常 using moving average）
3. **当价差偏离均衡时**：
   - 做多被低估的资产
   - 做空被高估的资产
4. **当价差回归均衡时**：平仓获利

### 1.2 为什么配对交易有效？

配对交易有效的前提是：**两个资产的价差具有均值回归特性**。

这意味着：

- 当价差偏离长期均衡水平时，未来大概率会回归
- 我们不需要预测单个资产的价格方向，只需要预测价差的收敛

### 1.3 配对交易 vs 相关性交易

很多投资者误将**高相关性**当作配对交易的基础，这是一个常见误区。

- **相关性**：衡量两个资产价格的**同期变动方向**是否一致
- **协整性**：衡量两个资产价格序列是否存在**长期均衡关系**

**关键区别**：

- 两个资产可能相关性很高，但不存在协整关系（价差不平稳）
- 两个资产可能相关性不高，但存在协整关系（价差平稳）

**只有协整关系才能保证价差的均值回归特性！**

## 二、协整关系的理论基础

### 2.1 平稳性（Stationarity）

在介绍协整之前，必须先理解**平稳性**。

一个时间序列 $y_t$ 是平稳的，如果它满足：

1. **均值恒定**：$\mathbb{E}[y_t] = \mu$（常数）
2. **方差恒定**：$\text{Var}(y_t) = \sigma^2$（常数）
3. **协方差只依赖于时差**：$\text{Cov}(y_t, y_{t-k}) = \gamma_k$（只与 $k$ 有关）

**为什么平稳性重要？**

非平稳序列（如随机游走）的统计特性随时间变化，无法使用历史数据来推断未来。而平稳序列的统计特性恒定，可以可靠地进行统计推断。

### 2.2 单位根检验

检验一个序列是否平稳，常用**单位根检验**（Unit Root Test）：

- **ADF检验**（Augmented Dickey-Fuller Test）
- **PP检验**（Phillips-Perron Test）
- **KPSS检验**（Kwiatkowski-Phillips-Schmidt-Shin Test）

**ADF检验的原理**：

- **原假设 $H_0$**：序列存在单位根（非平稳）
- **备择假设 $H_1$**：序列平稳

如果ADF统计量 < 临界值（p-value < 0.05），则拒绝原假设，认为序列平稳。

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def check_stationarity(series, title=''):
    """
    ADF检验判断序列是否平稳
    
    参数:
    - series: pd.Series, 时间序列
    - title: str, 序列名称（用于打印）
    
    返回:
    - is_stationary: bool, 是否平稳
    - p_value: float, p值
    """
    result = adfuller(series.dropna())
    
    print(f"{title} ADF检验结果:")
    print(f"  ADF统计量: {result[0]:.4f}")
    print(f"  p-value: {result[1]:.4f}")
    print(f"  临界值: 1%={result[4]['1%']:.4f}, 5%={result[4]['5%']:.4f}, 10%={result[4]['10%']:.4f}")
    
    is_stationary = result[1] < 0.05
    print(f"  结论: {'平稳' if is_stationary else '非平稳'}")
    
    return is_stationary, result[1]

# 使用示例
# check_stationarity(stock_price, '股价')
```

### 2.3 协整的定义

如果两个或多个**非平稳**的时间序列，它们的**线性组合**是平稳的，则这些序列之间存在**协整关系**。

数学定义：

对于两个时间序列 $X_t$ 和 $Y_t$，如果：

1. $X_t$ 和 $Y_t$ 都是 **I(1)** 过程（一阶单整，即一阶差分后平稳）
2. 存在协整向量 $\beta$，使得 $Z_t = Y_t - \beta X_t$ 是 **I(0)** 过程（平稳）

则称 $X_t$ 和 $Y_t$ **协整**。

**直观理解**：

- 两个资产的价格各自是随机游走（非平稳）
- 但它们的**价差**（或线性组合）是平稳的，围绕某个均值波动
- 这意味着两个资产的价格之间存在**长期均衡关系**

### 2.4 协整检验方法

#### （1）Engle-Granger 两步法

**步骤1**：用OLS估计协整回归

$$
Y_t = \alpha + \beta X_t + \epsilon_t
$$

**步骤2**：检验残差 $\epsilon_t$ 是否平稳（使用ADF检验）

- 如果残差平稳 → $X_t$ 和 $Y_t$ 协整
- 如果残差非平稳 → 不存在协整关系

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def engle_granger_test(y, x, alpha=0.05):
    """
    Engle-Granger协整检验
    
    参数:
    - y: pd.Series, 第一个资产价格
    - x: pd.Series, 第二个资产价格
    - alpha: float, 显著性水平
    
    返回:
    - is_cointegrated: bool, 是否存在协整关系
    - beta: float, 协整系数
    - residual: pd.Series, 残差序列
    """
    # 步骤1: OLS回归
    X = add_constant(x)
    model = OLS(y, X).fit()
    beta = model.params[1]
    residual = model.resid
    
    # 步骤2: 检验残差平稳性
    is_stationary, p_value = check_stationarity(residual, '残差')
    
    is_cointegrated = p_value < alpha
    
    print(f"\nEngle-Granger检验结果:")
    print(f"  协整系数 β: {beta:.4f}")
    print(f"  是否协整: {'是' if is_cointegrated else '否'}")
    
    return is_cointegrated, beta, residual

# 使用示例
# is_cointegrated, beta, residual = engle_granger_test(price_y, price_x)
```

#### （2）Johansen 检验

Johansen检验是一种**多变量协整检验**方法，适用于：

- 检验多个时间序列之间的协整关系
- 确定协整向量的个数

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验
    
    参数:
    - data: pd.DataFrame, 多列价格数据
    - det_order: int, 确定性项的顺序 (0: 无常数项, 1: 有常数项, 等等)
    - k_ar_diff: int, 自回归滞后阶数
    
    返回:
    - result: Johansen检验结果对象
    """
    result = coint_johansen(data, det_order, k_ar_diff)
    
    print("Johansen检验结果:")
    print(f"  迹统计量 (Trace Statistic): {result.lr1}")
    print(f"  最大特征值统计量 (Max Eigen Statistic): {result.lr2}")
    print(f"  临界值 (90%, 95%, 99%):")
    print(f"    迹检验: {result.cvt[:, 1]}")  # 95%临界值
    print(f"    最大特征值检验: {result.cvm[:, 1]}")
    
    # 判断协整向量个数
    num_coint = (result.lr1 > result.cvt[:, 1]).sum()
    print(f"  协整向量个数: {num_coint}")
    
    return result

# 使用示例
# data = pd.DataFrame({'y': price_y, 'x': price_x})
# johansen_test(data)
```

## 三、如何选择合适的配对？

### 3.1 经济逻辑筛选

最可靠的配对通常具有**坚实的经济逻辑**：

- **同行业股票**：业务模式相似，受相同宏观因素影响（如工商银行 vs 建设银行）
- **产业链上下游**：成本传导机制明确（如原油 vs 航空股）
- **替代品**：需求此消彼长（如可口可乐 vs 百事可乐）
- **ETF与成分股**：存在套利机制（如SPY vs 其重仓成分股）

### 3.2 统计指标筛选

通过量化指标筛选潜在配对：

#### （1）相关性（Correlation）

```python
def calculate_correlation(prices):
    """计算价格收益率的相关性"""
    returns = prices.pct_change().dropna()
    corr_matrix = returns.corr()
    return corr_matrix
```

**注意**：相关性高不代表协整，但可以作为初筛指标。

#### （2）协整检验（Cointegration Test）

对初筛后的配对进行协整检验（如Engle-Granger检验）。

#### （3）半衰期（Half-life）

均值回归的速度可以用**半衰期**来衡量：

```python
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant

def calculate_half_life(spread):
    """
    计算价差的半衰期（均值回归速度）
    
    参数:
    - spread: pd.Series, 价差序列
    
    返回:
    - half_life: float, 半衰期（交易日数）
    """
    # 计算价差的滞后项
    spread_lag = spread.shift(1)
    spread_ret = spread - spread_lag
    
    # OLS回归: Δspread = α + β * spread_lag + ε
    X = add_constant(spread_lag.dropna())
    y = spread_ret.dropna()
    model = OLS(y, X).fit()
    
    beta = model.params[1]
    
    # 半衰期 = ln(2) / |β|
    half_life = np.log(2) / abs(beta)
    
    print(f"半衰期: {half_life:.2f} 个交易日")
    
    return half_life

# 使用示例
# half_life = calculate_half_life(residual)
```

**解读**：

- 半衰期越短 → 均值回归越快 → 交易机会越多
- 半衰期过长（如 > 60个交易日）→ 资金占用时间长，不经济

#### （4）Hurst 指数

Hurst指数可以判断序列是**均值回归**、**随机游走**还是**趋势性**：

- $H < 0.5$：均值回归（越小越强）
- $H = 0.5$：随机游走
- $H > 0.5$：趋势性（越大越强）

```python
def calculate_hurst(series, max_lag=20):
    """
    计算Hurst指数
    
    参数:
    - series: pd.Series, 时间序列
    - max_lag: int, 最大滞后阶数
    
    返回:
    - hurst: float, Hurst指数
    """
    series = series.dropna()
    lags = range(2, max_lag)
    
    # 计算重新缩放的范围
    tau = []
    for lag in lags:
        # 计算重新缩放的范围
        std = series.diff().dropna().std()
        if std == 0:
            continue
        rescaled_range = (series.diff().cumsum() / std).max() - (series.diff().cumsum() / std).min()
        tau.append(rescaled_range / std)
    
    # 线性回归: log(lag) ~ log(tau)
    from numpy.polynomial import polynomial as P
    coeffs = P.polyfit(np.log(lags), np.log(tau), 1)
    hurst = coeffs[1]
    
    print(f"Hurst指数: {hurst:.4f}")
    print(f"  均值回归 (H<0.5)" if hurst < 0.5 else f"  随机游走 (H=0.5)" if hurst == 0.5 else f"  趋势性 (H>0.5)")
    
    return hurst

# 使用示例
# hurst = calculate_hurst(spread)
```

### 3.3 综合评分体系

构建一个综合评分体系，对候选配对进行排名：

```python
def score_pair(y, x):
    """
    对配对进行综合评分
    
    参数:
    - y: pd.Series, 第一个资产价格
    - x: pd.Series, 第二个资产价格
    
    返回:
    - score: dict, 各项指标和总分
    """
    score = {}
    
    # 1. 协整检验
    is_cointegrated, beta, residual = engle_granger_test(y, x)
    score['cointegration'] = 1 if is_cointegrated else 0
    
    if not is_cointegrated:
        score['total'] = 0
        return score
    
    # 2. 相关性
    corr = y.corr(x)
    score['correlation'] = corr
    
    # 3. 半衰期
    half_life = calculate_half_life(residual)
    # 将半衰期转换为0-1分数（越短越好）
    score['half_life'] = 1 / (1 + half_life / 10)
    
    # 4. Hurst指数
    hurst = calculate_hurst(residual)
    # Hurst < 0.5 越好
    score['hurst'] = (0.5 - hurst) / 0.5 if hurst < 0.5 else 0
    
    # 5. 综合评分（加权平均）
    weights = {'cointegration': 0.4, 'correlation': 0.2, 'half_life': 0.2, 'hurst': 0.2}
    total = (
        weights['cointegration'] * score['cointegration'] +
        weights['correlation'] * score['correlation'] +
        weights['half_life'] * score['half_life'] +
        weights['hurst'] * score['hurst']
    )
    score['total'] = total
    
    print(f"\n综合评分: {total:.4f}")
    
    return score

# 使用示例
# score = score_pair(price_y, price_x)
```

## 四、配对交易策略的构建

### 4.1 信号生成

配对交易的核心信号是**价差的Z-score**：

$$
Z_t = \frac{S_t - \mu_S}{\sigma_S}
$$

其中：

- $S_t$：当前价差（或残差）
- $\mu_S$：价差的均值（通常使用滚动窗口或历史全样本）
- $\sigma_S$：价差的标准差

**交易规则**：

- 当 $Z_t > \text{entry\_threshold}$（如 +2）：做空价差（做空Y，做多X）
- 当 $Z_t < -\text{entry\_threshold}$（如 -2）：做多价差（做多Y，做空X）
- 当 $Z_t$ 回归到 $|\text{exit\_threshold}|$（如 0.5）：平仓

```python
def generate_signals(spread, entry_threshold=2.0, exit_threshold=0.5, window=252):
    """
    生成配对交易信号
    
    参数:
    - spread: pd.Series, 价差序列
    - entry_threshold: float, 入场阈值（Z-score）
    - exit_threshold: float, 出场阈值（Z-score）
    - window: int, 滚动窗口长度（用于计算均值和标准差）
    
    返回:
    - signals: pd.DataFrame, 包含Z-score和交易信号
    """
    signals = pd.DataFrame(index=spread.index)
    signals['spread'] = spread
    
    # 计算滚动均值和标准差
    signals['mean'] = spread.rolling(window).mean()
    signals['std'] = spread.rolling(window).std()
    
    # 计算Z-score
    signals['z_score'] = (spread - signals['mean']) / signals['std']
    
    # 生成交易信号
    # 1: 做多价差, -1: 做空价差, 0: 平仓/不持仓
    signals['signal'] = 0
    
    # 入场信号
    signals.loc[signals['z_score'] < -entry_threshold, 'signal'] = 1
    signals.loc[signals['z_score'] > entry_threshold, 'signal'] = -1
    
    # 出场信号（回归到exit_threshold）
    position = 0
    for i in range(1, len(signals)):
        if position == 0:
            # 当前无仓位，检查是否入场
            position = signals['signal'].iloc[i]
        else:
            # 当前有仓位，检查是否出场
            if position == 1 and signals['z_score'].iloc[i] >= -exit_threshold:
                position = 0
            elif position == -1 and signals['z_score'].iloc[i] <= exit_threshold:
                position = 0
        
        signals['signal'].iloc[i] = position
    
    return signals

# 使用示例
# signals = generate_signals(residual, entry_threshold=2.0, exit_threshold=0.5)
```

### 4.2 仓位管理

合理的仓位管理对于配对交易至关重要。

#### （1）等金额对冲

```python
def calculate_position_equal_notional(prices_y, prices_x, beta, capital=1000000):
    """
    计算等金额对冲的仓位
    
    参数:
    - prices_y: pd.Series, Y资产价格
    - prices_x: pd.Series, X资产价格
    - beta: float, 对冲比例
    - capital: float, 总资金
    
    返回:
    - position_y: int, Y资产的股数
    - position_x: int, X资产的股数
    """
    # 等金额分配
    notional_y = capital / 2
    notional_x = capital / 2
    
    # 计算股数
    position_y = int(notional_y / prices_y.iloc[-1])
    position_x = int(notional_x / (beta * prices_x.iloc[-1]))
    
    return position_y, position_x

# 使用示例
# pos_y, pos_x = calculate_position_equal_notional(price_y, price_x, beta)
```

#### （2）风险平价

根据两个资产的波动率来分配风险预算：

```python
def calculate_position_risk_parity(prices_y, prices_x, beta, capital=1000000, target_vol=0.15):
    """
    计算风险平价仓位
    
    参数:
    - prices_y: pd.Series, Y资产价格
    - prices_x: pd.Series, X资产价格
    - beta: float, 对冲比例
    - capital: float, 总资金
    - target_vol: float, 目标波动率
    
    返回:
    - position_y: int, Y资产的股数
    - position_x: int, X资产的股数
    """
    # 计算历史波动率
    ret_y = prices_y.pct_change().dropna()
    ret_x = prices_x.pct_change().dropna()
    
    vol_y = ret_y.std() * np.sqrt(252)
    vol_x = ret_x.std() * np.sqrt(252)
    
    # 等风险分配
    risk_weight_y = 0.5
    risk_weight_x = 0.5
    
    # 根据波动率调整仓位
    notional_y = capital * risk_weight_y * target_vol / vol_y
    notional_x = capital * risk_weight_x * target_vol / vol_x
    
    position_y = int(notional_y / prices_y.iloc[-1])
    position_x = int(notional_x / (beta * prices_x.iloc[-1]))
    
    return position_y, position_x
```

### 4.3 止损与止盈

配对交易也需要合理的风险控制：

```python
def add_stop_loss(signals, spread, stop_loss_z=3.0, take_profit_z=0.5):
    """
    添加止损和止盈逻辑
    
    参数:
    - signals: pd.DataFrame, 交易信号
    - spread: pd.Series, 价差
    - stop_loss_z: float, 止损Z-score（绝对值）
    - take_profit_z: float, 止盈Z-score（绝对值）
    
    返回:
    - signals: pd.DataFrame, 更新后的交易信号
    """
    position = 0
    
    for i in range(1, len(signals)):
        if position == 0:
            # 无仓位，检查是否入场
            if signals['signal'].iloc[i] != 0:
                position = signals['signal'].iloc[i]
                entry_z = signals['z_score'].iloc[i]
        else:
            # 有仓位，检查止损/止盈
            current_z = signals['z_score'].iloc[i]
            
            # 止损：Z-score继续扩大
            if abs(current_z) > stop_loss_z:
                position = 0
                signals['signal'].iloc[i] = 0
                print(f"止损触发 @ {signals.index[i]}")
            
            # 止盈：Z-score回归
            elif abs(current_z) < take_profit_z:
                position = 0
                signals['signal'].iloc[i] = 0
                print(f"止盈触发 @ {signals.index[i]}")
    
    return signals
```

## 五、完整策略回测

### 5.1 回测框架

使用Python构建一个完整的配对交易回测框架：

```python
class PairTradingBacktest:
    """配对交易回测框架"""
    
    def __init__(self, price_y, price_x, initial_capital=1000000, 
                 entry_threshold=2.0, exit_threshold=0.5, 
                 stop_loss_z=3.0, commission=0.001):
        """
        初始化
        
        参数:
        - price_y: pd.Series, Y资产价格
        - price_x: pd.Series, X资产价格
        - initial_capital: float, 初始资金
        - entry_threshold: float, 入场阈值
        - exit_threshold: float, 出场阈值
        - stop_loss_z: float, 止损Z-score
        - commission: float, 佣金比例
        """
        self.price_y = price_y
        self.price_x = price_x
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_z = stop_loss_z
        self.commission = commission
        
        # 协整检验
        self.is_cointegrated, self.beta, self.spread = engle_granger_test(price_y, price_x)
        
        if not self.is_cointegrated:
            raise ValueError("两个资产不存在协整关系，不能进行配对交易！")
        
        # 生成信号
        self.signals = generate_signals(self.spread, entry_threshold, exit_threshold)
        
        # 回测结果
        self.portfolio_value = None
        self.returns = None
        self.metrics = None
    
    def run_backtest(self):
        """运行回测"""
        # 初始化投资组合
        portfolio = pd.DataFrame(index=self.signals.index)
        portfolio['cash'] = self.initial_capital
        portfolio['position_y'] = 0
        portfolio['position_x'] = 0
        portfolio['value_y'] = 0
        portfolio['value_x'] = 0
        portfolio['total_value'] = self.initial_capital
        
        # 逐日模拟
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            
            # 获取昨日仓位
            prev_pos_y = portfolio['position_y'].iloc[i-1]
            prev_pos_x = portfolio['position_x'].iloc[i-1]
            
            # 获取今日信号
            signal = self.signals['signal'].iloc[i]
            
            # 如果信号变化，调整仓位
            if signal != 0 and prev_pos_y == 0:
                # 开仓
                price_y_today = self.price_y.iloc[i]
                price_x_today = self.price_x.iloc[i]
                
                # 计算仓位（等金额）
                notional = portfolio['total_value'].iloc[i-1] / 2
                pos_y = int(notional / price_y_today)
                pos_x = int(notional / (self.beta * price_x_today))
                
                if signal == 1:
                    # 做多Y，做空X
                    portfolio.loc[date, 'position_y'] = pos_y
                    portfolio.loc[date, 'position_x'] = -pos_x
                elif signal == -1:
                    # 做空Y，做多X
                    portfolio.loc[date, 'position_y'] = -pos_y
                    portfolio.loc[date, 'position_x'] = pos_x
                
                # 扣除交易成本
                trade_value = abs(pos_y * price_y_today) + abs(pos_x * price_x_today)
                commission_cost = trade_value * self.commission
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1] - commission_cost
            
            else:
                # 保持仓位或平仓
                portfolio.loc[date, 'position_y'] = prev_pos_y
                portfolio.loc[date, 'position_x'] = prev_pos_x
                portfolio.loc[date, 'cash'] = portfolio['cash'].iloc[i-1]
            
            # 计算当日市值
            price_y_today = self.price_y.iloc[i]
            price_x_today = self.price_x.iloc[i]
            
            value_y = portfolio['position_y'].iloc[i] * price_y_today
            value_x = portfolio['position_x'].iloc[i] * price_x_today
            
            portfolio.loc[date, 'value_y'] = value_y
            portfolio.loc[date, 'value_x'] = value_x
            portfolio.loc[date, 'total_value'] = portfolio['cash'].iloc[i] + value_y + value_x
        
        # 计算收益
        self.portfolio_value = portfolio['total_value']
        self.returns = self.portfolio_value.pct_change().dropna()
        
        # 计算性能指标
        self.calculate_metrics()
        
        return portfolio
    
    def calculate_metrics(self):
        """计算性能指标"""
        # 总收益
        total_return = (self.portfolio_value.iloc[-1] / self.initial_capital - 1)
        
        # 年化收益
        days = (self.portfolio_value.index[-1] - self.portfolio_value.index[0]).days
        annual_return = (1 + total_return) ** (252 / days) - 1
        
        # 年化波动
        annual_vol = self.returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        cummax = self.portfolio_value.cummax()
        drawdown = (self.portfolio_value - cummax) / cummax
        max_drawdown = drawdown.min()
        
        self.metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'num_trades': (self.signals['signal'].diff() != 0).sum()
        }
    
    def plot_results(self):
        """绘制回测结果"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 1.  portfolio value
        axes[0].plot(self.portfolio_value.index, self.portfolio_value.values)
        axes[0].set_title('Portfolio Value')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # 2. Z-score and signals
        axes[1].plot(self.signals.index, self.signals['z_score'].values, label='Z-score')
        axes[1].axhline(y=self.entry_threshold, color='r', linestyle='--', label='Entry Threshold')
        axes[1].axhline(y=-self.entry_threshold, color='r', linestyle='--')
        axes[1].axhline(y=self.exit_threshold, color='g', linestyle='--', label='Exit Threshold')
        axes[1].axhline(y=-self.exit_threshold, color='g', linestyle='--')
        axes[1].set_title('Z-score and Trading Signals')
        axes[1].set_ylabel('Z-score')
        axes[1].legend()
        axes[1].grid(True)
        
        # 3. Cumulative returns
        cumulative_returns = (1 + self.returns).cumprod()
        axes[2].plot(cumulative_returns.index, cumulative_returns.values)
        axes[2].set_title('Cumulative Returns')
        axes[2].set_ylabel('Cumulative Return')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.show()

# 使用示例
# backtest = PairTradingBacktest(price_y, price_x)
# portfolio = backtest.run_backtest()
# backtest.plot_results()
# print(backtest.metrics)
```

### 5.2 实证案例

我们使用A股市场的实际数据来测试配对交易策略。

**案例：工商银行（601398.SH）vs 建设银行（601939.SH）**

这两个股票属于同一行业（银行业），业务模式高度相似，理论上应该存在协整关系。

```python
# 数据获取（使用westock-data或tushare等工具）
# prices_y = get_stock_price('601398.SH', start='2020-01-01', end='2025-12-31')
# prices_x = get_stock_price('601939.SH', start='2020-01-01', end='2025-12-31')

# 协整检验
is_cointegrated, beta, spread = engle_granger_test(prices_y, prices_x)

if is_cointegrated:
    print(f"协整系数 β: {beta:.4f}")
    
    # 计算半衰期
    half_life = calculate_half_life(spread)
    print(f"半衰期: {half_life:.2f} 个交易日")
    
    # 运行回测
    backtest = PairTradingBacktest(
        prices_y, prices_x,
        initial_capital=1000000,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_z=3.0,
        commission=0.001
    )
    
    portfolio = backtest.run_backtest()
    
    print("\n回测结果:")
    for key, value in backtest.metrics.items():
        if 'return' in key or 'volatility' in key:
            print(f"  {key}: {value:.2%}")
        elif 'drawdown' in key:
            print(f"  {key}: {value:.2%}")
        else:
            print(f"  {key}: {value}")
    
    # 绘制结果
    backtest.plot_results()
```

**回测结果示例**（假设）：

| 指标 | 数值 |
|------|------|
| 总收益 | 35.2% |
| 年化收益 | 8.5% |
| 年化波动 | 6.2% |
| 夏普比率 | 1.37 |
| 最大回撤 | -8.3% |
| 交易次数 | 42 |

**解读**：

- 配对交易策略实现了**稳健的收益**，夏普比率达到1.37
- **最大回撤较小**（-8.3%），体现了市场中性策略的优势
- **交易次数适中**（42次），说明信号的触发频率合理

## 六、风险管理与实战要点

### 6.1 主要风险

#### （1）协整关系破裂

协整关系是配对交易的基础，但这一关系可能**随时间变化而破裂**：

- 公司基本面发生重大变化（如并购、重组）
- 行业格局改变（如监管政策、技术颠覆）
- 宏观经济环境变化（如利率市场化对银行股的影响）

**应对方法**：

- **定期重新检验协整关系**（如每季度）
- 设置**协整关系监控指标**（如滚动窗口的ADF检验p-value）
- 当协整关系破裂时，**立即平仓**

#### （2）模型风险

我们使用历史数据估计的协整系数 $\beta$ 可能不准确，或者随时间变化。

**应对方法**：

- 使用**滚动窗口**或**指数加权**方法来动态估计 $\beta$
- 设置**置信区间**，当 $\beta$ 超出历史范围时及时调整

#### （3）执行风险

配对交易需要**同时买入和卖出**两个资产，如果其中一个资产的流动性不足，可能导致：

- **滑点过大**：实际成交价格偏离信号价格
- **无法成交**：订单无法全部成交，导致仓位不平衡

**应对方法**：

- 选择**高流动性**的资产进行配对交易
- 使用**算法交易**（如VWAP、TWAP）来减少市场冲击
- 设置**最大滑点限制**，当滑点超过阈值时暂停交易

#### （4）黑天鹅事件

在市场出现极端事件时，配对交易可能**暂时失效**：

- 两个资产的价格可能**同时大幅下跌**（如2020年疫情爆发）
- 价差可能**短期扩大**而非收敛

**应对方法**：

- 设置**严格的止损**（如Z-score超过3.0立即平仓）
- **分散投资**：同时交易多个不相关的配对，降低单一配对的风险
- 在极端市场环境下（如VIX > 40），**暂停交易**

### 6.2 实战建议

1. **严格筛选配对**：不要为了交易而交易，只有统计上显著且经济逻辑合理的配对才值得交易
2. **控制杠杆**：虽然配对交易风险较低，但过度杠杆可能放大损失
3. **实时监控**：建立实时监控系统，跟踪价差、Z-score、仓位等关键指标
4. **定期复盘**：每月或每季度复盘策略表现，识别可能的问题并优化
5. **保持耐心**：配对交易是**低频策略**，需要有耐心等待信号出现

## 七、总结

配对交易是一种**稳健的市场中性策略**，适合追求绝对收益、风险偏好较低的投资者。

**核心要点**：

1. **协整关系是配对交易的理论基础**，必须使用严格的统计检验来验证
2. **合适的配对选择**是成功的关键，需要结合经济逻辑和统计指标
3. **合理的风险管理**（止损、仓位管理、分散投资）不可或缺
4. **实盘执行**中需要考虑交易成本、滑点、流动性等实际约束

**未来方向**：

- **机器学习方法**：使用LSTM、随机森林等模型来预测价差的均值回归
- **高频配对交易**：利用日内高频数据进行更快速的交易
- **跨市场配对**：在不同市场（如A股vs港股）寻找套利机会

---

## 参考文献

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). *Statistical Arbitrage and Pairs Trading*. SAS Institute.
3. Engle, R. F., & Granger, C. W. (1987). "Co-integration and Error Correction: Representation, Estimation, and Testing." *Econometrica*.
4. Johansen, S. (1988). "Statistical Analysis of Cointegration Vectors." *Journal of Economic Dynamics and Control*.
5. 陈工孟, 等. (2020). 《统计套利：理论与实践》. 机械工业出版社.

## 附录：完整Python代码

```python
# 本文涉及的所有代码已包含在正文中
# 完整可运行代码请访问：https://github.com/quant-blog/pair-trading-example
```

---

*本文仅供学术交流，不构成投资建议。配对交易虽然风险较低，但仍可能面临协整关系破裂、模型风险、执行风险等。请在专业人士指导下使用。*
