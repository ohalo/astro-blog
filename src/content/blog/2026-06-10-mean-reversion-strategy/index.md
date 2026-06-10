---
title: "均值回归策略：统计套利的核心逻辑与实战"
publishDate: '2026-06-10'
description: "均值回归策略 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 均值回归的理论基础

均值回归（Mean Reversion）是量化交易中最经典的策略之一，其核心假设是：**资产价格会围绕某个均衡值波动，偏离均值后终将回归**。

这个概念可以追溯到19世纪的高尔顿（Francis Galton）提出的"回归到均值"现象。在金融市场中，均值回归策略广泛应用于：

- **配对交易**（Pairs Trading）
- **统计套利**（Statistical Arbitrage）
- **波动率交易**（Volatility Trading）
- **行业轮动**（Sector Rotation）

## 数学原理：Ornstein-Uhlenbeck过程

均值回归可以用**Ornstein-Uhlenbeck (OU) 过程**来建模：

```
dX_t = θ(μ - X_t)dt + σdW_t
```

其中：
- **θ (回归速度)**：价格回归均值的速度
- **μ (长期均值)**：均衡价格
- **σ (波动率)**：随机扰动的标准差
- **W_t**：维纳过程（布朗运动）

### 关键指标：半衰期（Half-Life）

半衰期为价格回归到均值一半所需的时间：

```
HL = ln(2) / θ
```

**实战意义**：
- HL < 10天：短期均值回归，适合高频交易
- HL 10-30天：中期策略，适合日内/隔夜
- HL > 30天：长期策略，适合持仓数周

## 构建均值回归策略的步骤

### 1. 选择标的

适合均值回归的资产特征：
- ✅ 有清晰的长期均值
- ✅ 波动率高但不过度（适合做波段）
- ✅ 流动性好（降低交易成本）
- ✅ 有经济逻辑支撑（不只是统计现象）

**经典标的**：
- 股指ETF（如沪深300ETF、标普500ETF）
- 商品期货（如原油、黄金）
- 外汇对（如EUR/USD）
- 配对股票（如可口可乐 vs 百事可乐）

### 2. 计算均值和标准差

常用方法：

```python
import pandas as pd
import numpy as np

# 方法1：简单移动平均
rolling_mean = price.rolling(window=20).mean()
rolling_std = price.rolling(window=20).std()

# 方法2：指数加权移动平均（更灵敏）
ewm_mean = price.ewm(span=20).mean()
ewm_std = price.ewm(span=20).std()

# 方法3：卡尔曼滤波（自适应）
from pykalman import KalmanFilter
kf = KalmanFilter(transition_matrices=[1], observation_matrices=[1])
state_means, _ = kf.filter(price.values)
```

### 3. 生成交易信号

**Z-Score策略**（最常用）：

```python
z_score = (price - rolling_mean) / rolling_std

# 交易规则
if z_score < -2:
    signal = 1  # 买入（价格低估）
elif z_score > 2:
    signal = -1  # 卖出（价格高估）
else:
    signal = 0  # 持有或空仓
```

**布林带策略**（Bollinger Bands）：

```python
upper_band = rolling_mean + 2 * rolling_std
lower_band = rolling_mean - 2 * rolling_std

if price < lower_band:
    signal = 1  # 买入
elif price > upper_band:
    signal = -1  # 卖出
```

### 4. 风险管理

均值回归策略的风险在于**价格可能不回归**（趋势行情）。

**防护措施**：

1. **止损线**：Z-Score突破±3时止损
2. **最大持仓时间**：超过N天未回归就平仓
3. **仓位管理**：根据Z-Score大小动态调整仓位
4. **相关性监控**：标的的相关性突然失效要警惕

```python
# 动态仓位管理
position_size = (1 / abs(z_score)) * base_position
position_size = min(position_size, max_position)  # 上限约束
```

## Python实战：配对交易示例

以**工商银行(601398) vs 建设银行(601939)**为例：

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint

# 1. 获取价格数据
gs = pd.read_csv('ICBC.csv', index_col=0, parse_dates=True)['Close']
ccb = pd.read_csv('CCB.csv', index_col=0, parse_dates=True)['Close']

# 2. 协整检验（确保长期均衡关系）
score, p_value, _ = coint(gs, ccb)
print(f"协整检验 p-value: {p_value:.4f}")  # <0.05 说明存在协整关系

# 3. 计算对冲比例（OLS回归）
from sklearn.linear_model import LinearRegression
X = ccb.values.reshape(-1, 1)
y = gs.values
model = LinearRegression().fit(X, y)
hedge_ratio = model.coef_[0]

# 4. 计算价差（Spread）
spread = gs - hedge_ratio * ccb

# 5. 计算Z-Score
z_score = (spread - spread.rolling(20).mean()) / spread.rolling(20).std()

# 6. 生成交易信号
signal = pd.Series(0, index=z_score.index)
signal[z_score < -2] = 1   # 买入工商银行，卖出建设银行
signal[z_score > 2] = -1   # 卖出工商银行，买入建设银行
signal[z_score.between(-0.5, 0.5)] = 0  # 平仓

# 7. 回测
returns = signal.shift(1) * (np.log(gs / gs.shift(1)) - hedge_ratio * np.log(ccb / ccb.shift(1)))
cumulative_returns = returns.cumsum()
```

## 均值回归 vs 动量策略

| 特征 | 均值回归 | 动量策略 |
|------|---------|---------|
| **市场状态** | 震荡市 | 趋势市 |
| **持仓时间** | 短-中期 | 中-长期 |
| **交易频率** | 高 | 低 |
| **胜率** | 高（60-70%） | 低（40-50%） |
| **盈亏比** | 低（0.5-1） | 高（2-3） |
| **最大风险** | 趋势行情 | 均值回归 |

**组合使用**：
- 用**ADF检验**（Augmented Dickey-Fuller）判断当前是趋势还是均值回归
- 趋势行情用动量，震荡行情用均值回归

## 实盘注意事项

### 1. 交易成本

均值回归策略交易频繁，交易成本影响大：

```python
# 考虑交易成本后的净收益
gross_return = 0.002  # 毛利0.2%
transaction_cost = 0.001  # 手续费0.1%
net_return = gross_return - transaction_cost  # 净利0.1%
```

**优化方法**：
- 提高入场阈值（Z-Score从±2提高到±2.5）
- 降低交易频率（用日线而非小时线）
- 选择低费率的券商

### 2. 滑点控制

均值回归策略常在极端价格成交（涨停/跌停），滑点大：

```python
# 限价单 + 智能路由
order = LimitOrder(price=current_price * 0.995)  # 限价单
order.set_smart_routing(True)  # 智能路由到最优交易所
```

### 3. 模型衰减

均值回归的**半衰期会变化**（市场结构改变）：

```python
# 滚动估计半衰期
half_life = []
for window in rolling_windows:
    theta = estimate_ou_parameters(price[window])['theta']
    half_life.append(np.log(2) / theta)

# 如果半衰期显著变长，说明均值回归失效
if half_life[-1] > 2 * half_life[0]:
    print("警告：均值回归速度显著下降！")
```

## 进阶：机器学习优化

传统均值回归的局限性：
- ❌ 固定阈值（Z-Score ±2）不适应市场变化
- ❌ 忽略其他信息（成交量、波动率、宏观因子）
- ❌ 无法捕捉非线性关系

**解决方案：用LSTM预测均值回归时机**

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 构建LSTM模型
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
    LSTM(50),
    Dense(1)
])

# 特征工程
features = pd.DataFrame({
    'z_score': z_score,
    'volatility': price.rolling(20).std(),
    'volume_ratio': volume / volume.rolling(20).mean(),
    'rsi': compute_rsi(price, 14)
})

# 标签：未来5天是否回归
labels = (price.shift(-5) - rolling_mean) / rolling_std < 0.5

# 训练模型
model.fit(features, labels, epochs=50, batch_size=32)
```

## 绩效评估

用以下指标评估均值回归策略：

```python
# 1. 夏普比率
sharpe = returns.mean() / returns.std() * np.sqrt(252)

# 2. 最大回撤
cumulative = (1 + returns).cumprod()
running_max = cumulative.cummax()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min()

# 3. 卡尔马比率（年化收益 / 最大回撤）
calmar = returns.mean() * 252 / abs(max_drawdown)

# 4. 胜率
win_rate = (returns > 0).sum() / len(returns)
```

## 总结

均值回归策略是量化交易的基石，核心要点：

1. **理论基础**：价格围绕均值波动，偏离后会回归
2. **数学工具**：OU过程、Z-Score、半衰期
3. **实战步骤**：选标的 → 算均值 → 生成信号 → 风险管理
4. **风险控制**：止损、最大持仓时间、动态仓位
5. **进阶方向**：机器学习优化、配对交易、统计套利

**适用场景**：
- ✅ 震荡市、无趋势行情
- ✅ 高波动率高流动性资产
- ✅ 有经济逻辑支撑的价差

**不适用场景**：
- ❌ 强趋势行情（容易止损）
- ❌ 黑天鹅事件（均值可能永久改变）
- ❌ 高交易成本的市场

下期预告：**LSTM神经网络在量化交易中的应用** — 用深度学习捕捉非线性模式！

---

*本文代码仅供参考，实盘交易需谨慎验证。均值回归策略在趋势行情中会发生连续止损，务必做好风险控制。*
