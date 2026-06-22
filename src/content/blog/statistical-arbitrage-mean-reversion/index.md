---
title: "统计套利：均值回归策略"
description: "深入探讨统计套利的理论基础、配对交易方法、协整检验以及均值回归策略的实战应用。包含完整的Python实现、风险管理和绩效评估框架。"
pubDate: 2026-06-23
tags: ["统计套利", "配对交易", "均值回归", "协整", "量化策略"]
cover: "/images/statistical-arbitrage-mean-reversion/cover.jpg"
---

# 统计套利：均值回归策略

## 引言

**统计套利（Statistical Arbitrage）**是一种基于量化模型的二级市场策略，通过统计方法挖掘资产价格之间的临时偏离，建立多空组合以获取稳定收益。

与传统的风险套利（如并购套利）不同，统计套利不依赖具体的公司事件，而是利用市场定价偏差和数据规律。其中，**均值回归（Mean Reversion）**是统计套利最核心的思想之一。

本文将系统介绍统计套利的理论基础、配对交易方法、协整检验技术，并提供完整的Python实现框架。

## 统计套利的核心思想

### 均值回归假设

统计套利基于一个简单而强大的假设：

> **资产价格或价格差会围绕某个均衡水平波动，短期偏离后终将回归均值。**

这一假设在多个市场和时间尺度上被验证：

- **配对股票**：同一行业的两只股票（如 Coca-Cola vs Pepsi），长期价格比相对稳定
- **ETF与成分股**：ETF价格与成分股加权平均价格应保持一致
- **股指期货与现货**：期货价格应与现货价格加上持有成本相等
- **不同期限的国债**：收益率曲线通常呈现平滑形态

![均值回归示意图](/images/statistical-arbitrage-mean-reversion/mean_reversion.png)

### 利润来源

统计套利的利润来自三个方面：

1. **定价偏差的修正**：市场短期非理性导致的价格偏离最终会修正
2. **均值回归的必然性**：数学上，极端值后出现反转的概率更高
3. **投资组合多样化**：同时持有多个不相关的套利组合，降低整体风险

## 配对交易：统计套利的经典方法

### 什么是配对交易？

**配对交易（Pairs Trading）**是统计套利最常见的一种形式：

1. **寻找配对**：找到两只价格走势高度相关的股票
2. **计算价差**：计算两只股票的价格差（或价格比）
3. **设定阈值**：当价差偏离均值超过N倍标准差时，认为出现套利机会
4. **建立头寸**：做多被低估的股票，做空被高估的股票
5. **平仓**：当价差回归均值时平仓，获取利润

### 配对选择的三个层次

#### 第一层：行业配对

最直观的配对方式——同一行业内业务模式相似的公司：

- 可口可乐（KO）vs 百事可乐（PEP）
- 摩根大通（JPM）vs 高盛（GS）
- 沃尔玛（WMT）vs 塔吉特（TGT）

**优点**：业务逻辑清晰，配对关系稳定
**缺点**：数量有限，难以大规模应用

#### 第二层：协整检验

使用统计方法——**协整检验（Cointegration Test）**——系统地筛选配对。

**协整的定义**：两个非平稳时间序列（如股价）的线性组合是平稳的，则称它们存在协整关系。

数学表达：
```
如果  y_t ~ I(1), x_t ~ I(1)
且存在系数 β 使得  z_t = y_t - β*x_t ~ I(0)
则 y_t 和 x_t 协整
```

**检验方法**：

1. **Engle-Granger两步法**：
   - 第一步：用OLS回归 y 对 x，得到残差
   - 第二步：对残差进行单位根检验（ADF检验）

2. **Johansen检验**：
   - 可以同时检验多个变量之间的协整关系
   - 适用于多资产配对

#### Python实现：协整配对筛选

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf

class PairsSelector:
    """配对交易筛选器"""
    
    def __init__(self, significance_level=0.05):
        """
        参数:
        - significance_level: float, 协整检验显著性水平
        """
        self.significance_level = significance_level
        self.pairs = []
        
    def engle_granger_test(self, y, x):
        """
        Engle-Granger两步法协整检验
        
        返回:
        - coint_score: float, 协整得分（负得越多，协整关系越强）
        - p_value: float, p值
        - is_cointegrated: bool, 是否存在协整关系
        """
        # 第一步：OLS回归
        model = OLS(y, x).fit()
        residuals = model.resid
        
        # 第二步：ADF检验残差
        adf_stat, p_value, _ = adfuller(residuals)
        
        is_cointegrated = p_value < self.significance_level
        
        return adf_stat, p_value, is_cointegrated, model.params[0]
    
    def screen_pairs(self, price_data, min_corr=0.5):
        """
        筛选配对
        
        参数:
        - price_data: DataFrame, 多只股票的价格数据
        - min_corr: float, 最小相关系数阈值
        """
        tickers = price_data.columns
        n = len(tickers)
        
        results = []
        
        for i in range(n):
            for j in range(i+1, n):
                stock1 = tickers[i]
                stock2 = tickers[j]
                
                # 计算相关系数
                corr = price_data[stock1].corr(price_data[stock2])
                
                if corr < min_corr:
                    continue
                
                # 协整检验
                score, p_value, is_coint, hedge_ratio = self.engle_granger_test(
                    price_data[stock1],
                    price_data[stock2]
                )
                
                if is_coint:
                    # 计算价差统计特征
                    spread = price_data[stock1] - hedge_ratio * price_data[stock2]
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    half_life = self.calculate_half_life(spread)
                    
                    results.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'correlation': corr,
                        'p_value': p_value,
                        'hedge_ratio': hedge_ratio,
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'half_life': half_life
                    })
        
        return pd.DataFrame(results).sort_values('p_value')
    
    def calculate_half_life(self, spread):
        """
        计算价差的半衰期
        
        半衰期越短，均值回归越快
        """
        spread_lag = spread.shift(1).dropna()
        spread_ret = spread.diff().dropna()
        
        model = OLS(spread_ret, spread_lag).fit()
        half_life = -np.log(2) / model.params[0]
        
        return half_life

# 使用示例
# 下载股票价格数据
tickers = ['KO', 'PEP', 'JPM', 'GS', 'WMT', 'TGT', 'XOM', 'CVX']
price_data = pd.DataFrame()

for ticker in tickers:
    data = yf.download(ticker, start='2018-01-01', end='2023-12-31', progress=False)
    price_data[ticker] = data['Adj Close']

# 筛选配对
selector = PairsSelector(significance_level=0.05)
pairs_df = selector.screen_pairs(price_data, min_corr=0.6)

print("找到的配对：")
print(pairs_df[['stock1', 'stock2', 'correlation', 'p_value', 'half_life']])
```

#### 第三层：机器学习配对

使用机器学习方法挖掘非线性配对关系：

- **聚类分析**：将股票按行业、风格、风险因子聚类，在簇内寻找配对
- **主成分分析（PCA）**：识别共同因子，寻找残差相关的配对
- **深度学习**：使用自编码器（Autoencoder）学习股票特征的低维表示，在表示空间中寻找邻近股票

```python
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class MLPairsSelector:
    """基于机器学习的配对筛选"""
    
    def __init__(self, n_clusters=10):
        self.n_clusters = n_clusters
        
    def cluster_stocks(self, features):
        """
        聚类分析
        
        参数:
        - features: DataFrame, 股票特征（市值、行业、估值、动量等）
        """
        # 标准化特征
        features_scaled = (features - features.mean()) / features.std()
        
        # KMeans聚类
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        # 在每个簇内筛选配对
        pairs = []
        for cluster_id in range(self.n_clusters):
            cluster_stocks = features.index[clusters == cluster_id]
            
            # 在簇内运行协整检验
            cluster_prices = price_data[cluster_stocks]
            selector = PairsSelector()
            cluster_pairs = selector.screen_pairs(cluster_prices)
            
            pairs.append(cluster_pairs)
        
        return pd.concat(pairs, ignore_index=True)
    
    def pca_pairs(self, returns_data, n_components=5):
        """
        基于PCA的配对筛选
        
        思路: 先去除共同因子影响，再寻找残差相关的股票
        """
        # PCA分解
        pca = PCA(n_components=n_components)
        factors = pca.fit_transform(returns_data)
        
        # 计算残差（实际收益 - 因子解释部分）
        explained_returns = pca.inverse_transform(factors)
        residuals = returns_data - explained_returns
        
        # 在残差中寻找相关性高的配对
        residual_corr = residuals.corr()
        
        pairs = []
        for i in range(len(residual_corr)):
            for j in range(i+1, len(residual_corr)):
                if residual_corr.iloc[i, j] > 0.3:  # 残差相关性阈值
                    pairs.append({
                        'stock1': residual_corr.index[i],
                        'stock2': residual_corr.columns[j],
                        'residual_corr': residual_corr.iloc[i, j]
                    })
        
        return pd.DataFrame(pairs).sort_values('residual_corr', ascending=False)
```

## 交易信号与风险管理

### 信号生成：Z-Score方法

最经典的配对交易信号基于**Z-Score（标准化价差）**：

```python
class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, entry_z=2.0, exit_z=0.5, stop_loss_z=3.0):
        """
        参数:
        - entry_z: float, 入场Z-Score阈值
        - exit_z: float, 出场Z-Score阈值
        - stop_loss_z: float, 止损Z-Score阈值
        """
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        
    def calculate_spread(self, price1, price2, hedge_ratio):
        """计算价差"""
        return price1 - hedge_ratio * price2
    
    def calculate_z_score(self, spread, window=60):
        """
        计算滚动Z-Score
        
        使用滚动窗口估计均值和标准差，避免前视偏差
        """
        spread_mean = spread.rolling(window).mean()
        spread_std = spread.rolling(window).std()
        
        z_score = (spread - spread_mean) / spread_std
        
        return z_score
    
    def generate_signals(self, price1, price2, hedge_ratio):
        """
        生成交易信号
        
        返回:
        - signals: DataFrame, 包含入场、出场、止损信号
        """
        spread = self.calculate_spread(price1, price2, hedge_ratio)
        z_score = self.calculate_z_score(spread)
        
        signals = pd.DataFrame(index=price1.index)
        signals['z_score'] = z_score
        signals['spread'] = spread
        
        # 初始化仓位
        signals['position'] = 0
        
        # 入场信号
        signals['long_entry'] = (z_score < -self.entry_z)  # 做多价差（买入stock1，卖出stock2）
        signals['short_entry'] = (z_score > self.entry_z)   # 做空价差
        
        # 出场信号
        signals['exit'] = (abs(z_score) < self.exit_z)
        
        # 止损信号
        signals['stop_loss'] = (abs(z_score) > self.stop_loss_z)
        
        # 生成仓位序列
        position = 0
        for t in range(1, len(signals)):
            if position == 0:
                # 空仓时，检查入场信号
                if signals['long_entry'].iloc[t]:
                    position = 1
                elif signals['short_entry'].iloc[t]:
                    position = -1
            else:
                # 有仓位时，检查出场或止损信号
                if signals['exit'].iloc[t] or signals['stop_loss'].iloc[t]:
                    position = 0
            
            signals['position'].iloc[t] = position
        
        return signals

# 使用示例
# 假设已找到配对：KO和PEP
ko_prices = price_data['KO']
pep_prices = price_data['PEP']
hedge_ratio = pairs_df[pairs_df['stock1']=='KO']['hedge_ratio'].values[0]

strategy = PairsTradingStrategy(entry_z=2.0, exit_z=0.5, stop_loss_z=3.0)
signals = strategy.generate_signals(ko_prices, pep_prices, hedge_ratio)

print(signals[['z_score', 'position']].tail(20))
```

### 风险管理

配对交易虽看似"市场中性"，但仍面临多种风险：

#### 1. 配对失效风险

**问题**：协整关系可能断裂（如行业格局变化、公司并购等）。

**应对**：
- 定期重新检验协整关系（如每季度）
- 设置配对失效止损（如价差突破历史极值的1.5倍）
- 分散投资多个不相关配对

```python
def monitor_cointegration(stock1_prices, stock2_prices, window=252):
    """
    滚动监测协整关系
    
    如果最近N天的p值持续高于阈值，说明配对失效
    """
    p_values = []
    
    for t in range(window, len(stock1_prices)):
        _, p_value, _, _ = selector.engle_granger_test(
            stock1_prices[t-window:t],
            stock2_prices[t-window:t]
        )
        p_values.append(p_value)
    
    # 如果最近20个交易日的p值都>0.1，发出警告
    recent_p_values = p_values[-20:]
    if all(p > 0.1 for p in recent_p_values):
        print("警告：配对协整关系可能已失效！")
        return False
    
    return True
```

#### 2. 模型风险

**问题**：Z-Score阈值、滚动窗口等参数依赖历史数据，可能不适用未来。

**应对**：
- 使用样本外数据优化参数
- 采用自适应阈值（如根据最近波动率动态调整）
- 组合多个信号（如Z-Score + Hurst指数 + 机器学习分类器）

```python
def adaptive_z_threshold(z_score, volatility_window=60):
    """
    自适应Z-Score阈值
    
    当市场波动率高时，放宽阈值；波动率低时，收紧阈值
    """
    vol = z_score.rolling(volatility_window).std()
    vol_percentile = vol.rolling(252).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
    
    # 根据波动率分位数动态调整阈值
    entry_z = 1.5 + vol_percentile * 1.0  # 1.5 ~ 2.5
    exit_z = 0.3 + vol_percentile * 0.4    # 0.3 ~ 0.7
    
    return entry_z, exit_z
```

#### 3. 执行风险

**问题**：做多和做空的执行时间差、卖空约束（如A股无法做空个股）。

**应对**：
- 使用ETF或期货代替个股做空
- 在美股市场，使用期权策略（如买入看跌期权+买入看涨期权）模拟配对交易
- 降低调仓频率，减少执行风险

## 多资产统计套利

### 统计套利组合

单一配对交易的资金容量有限，实际应用中通常同时交易数十甚至上百个配对。

**构建方法**：

1. **筛选顶层配对**：根据协整p值、半衰期、相关性等指标排序
2. **去相关性**：使用聚类或PCA，确保配对之间相关性低
3. **等权配置**：每个配对分配相同资金（或根据夏普比率加权）
4. **动态再平衡**：定期（如每月）重新筛选配对并调整权重

```python
class StatisticalArbitragePortfolio:
    """统计套利组合"""
    
    def __init__(self, n_pairs=20, rebalance_freq='M'):
        """
        参数:
        - n_pairs: int, 持有配对数量
        - rebalance_freq: str, 再平衡频率（'M'=月度，'W'=周度）
        """
        self.n_pairs = n_pairs
        self.rebalance_freq = rebalance_freq
        self.pairs = []
        self.weights = {}
        
    def select_pairs(self, price_data, date):
        """
        筛选顶层配对
        """
        # 使用过去2年数据筛选
        start_date = date - pd.Timedelta(days=730)
        historical_data = price_data.loc[start_date:date]
        
        selector = PairsSelector()
        pairs_df = selector.screen_pairs(historical_data, min_corr=0.5)
        
        # 选择前N个配对
        top_pairs = pairs_df.head(self.n_pairs)
        
        return top_pairs
    
    def calculate_pair_returns(self, pair, price_data):
        """
        计算单个配对的收益
        """
        stock1 = pair['stock1']
        stock2 = pair['stock2']
        hedge_ratio = pair['hedge_ratio']
        
        # 计算价差
        spread = price_data[stock1] - hedge_ratio * price_data[stock2]
        
        # 生成信号
        strategy = PairsTradingStrategy()
        signals = strategy.generate_signals(
            price_data[stock1],
            price_data[stock2],
            hedge_ratio
        )
        
        # 计算配对收益（假设每只股票投入50%资金）
        pair_returns = signals['position'].shift(1) * spread.pct_change()
        
        return pair_returns
    
    def backtest(self, price_data, start_date, end_date):
        """
        回测统计套利组合
        """
        dates = pd.date_range(start_date, end_date, freq=self.rebalance_freq)
        
        portfolio_returns = pd.Series(index=price_data.index, dtype=float)
        
        for i in range(len(dates)-1):
            # 重新筛选配对
            rebalance_date = dates[i]
            pairs = self.select_pairs(price_data, rebalance_date)
            
            # 计算下一个调仓周期内的收益
            next_date = dates[i+1]
            period_data = price_data.loc[rebalance_date:next_date]
            
            # 等权配置每个配对
            pair_returns = []
            for _, pair in pairs.iterrows():
                ret = self.calculate_pair_returns(pair, period_data)
                pair_returns.append(ret / self.n_pairs)
            
            # 汇总组合收益
            portfolio_returns.loc[rebalance_date:next_date] = pd.concat(pair_returns, axis=1).sum(axis=1)
        
        return portfolio_returns.dropna()

# 使用示例
portfolio = StatisticalArbitragePortfolio(n_pairs=20, rebalance_freq='M')
portfolio_returns = portfolio.backtest(price_data, '2020-01-01', '2023-12-31')

# 计算绩效指标
annual_return = portfolio_returns.mean() * 252
annual_vol = portfolio_returns.std() * np.sqrt(252)
sharpe = annual_return / annual_vol
max_drawdown = ((1 + portfolio_returns).cumprod() / (1 + portfolio_returns).cumprod().expanding().max() - 1).min()

print(f"年化收益: {annual_return:.2%}")
print(f"年化波动: {annual_vol:.2%}")
print(f"Sharpe比率: {sharpe:.2f}")
print(f"最大回撤: {max_drawdown:.2%}")
```

## 统计套利的局限性与替代方法

### 局限性

1. **市场结构变化**：金融危机、监管政策变化等可能导致大量配对同时失效
2. **套利竞争**：随着更多机构采用统计套利，定价偏差被迅速修正，利润空间压缩
3. **频率依赖**：日内高频统计套利需要低延迟基础设施，门槛高
4. **做空限制**：许多市场（如A股）缺乏方便的做空工具

### 替代方法

#### 1. 均值回归单一资产策略

不依赖配对，直接交易单一资产的均值回归：

- **RSI反转策略**：RSI < 30买入，RSI > 70卖出
- **布林带策略**：价格触及下轨买入，触及上轨卖出
- **乖离率策略**：价格偏离均线超过N倍标准差时反向操作

```python
def single_asset_mean_reversion(price, window=20, entry_std=2.0, exit_std=0.5):
    """
    单一资产均值回归策略
    
    使用布林带生成信号
    """
    # 计算移动平均和标准差
    ma = price.rolling(window).mean()
    std = price.rolling(window).std()
    
    # 计算布林带
    upper_band = ma + entry_std * std
    lower_band = ma - entry_std * std
    
    # 生成信号
    signals = pd.DataFrame(index=price.index)
    signals['price'] = price
    signals['ma'] = ma
    signals['position'] = 0
    
    # 入场：价格突破上下轨
    signals['long_entry'] = (price < lower_band)
    signals['short_entry'] = (price > upper_band)
    
    # 出场：价格回归均线
    signals['exit'] = (abs(price - ma) < exit_std * std)
    
    # 生成仓位序列（代码略，类似配对交易）
    
    return signals
```

#### 2. 统计套利与机器学习的结合

使用深度学习模型捕捉复杂的均值回归模式：

- **LSTM**：捕捉时间序列的长期依赖，预测价格回归时点
- **CNN**：识别价格形态（如头肩顶、双底等），这些形态往往伴随均值回归
- **强化学习**：将统计套利建模为马尔可夫决策过程（MDP），学习最优交易策略

```python
import tensorflow as tf
from tensorflow.keras import layers, models

def build_lstm_mean_reversion_model(input_shape):
    """
    构建LSTM均值回归预测模型
    """
    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=input_shape),
        layers.Dropout(0.2),
        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.2),
        layers.Dense(16, activation='relu'),
        layers.Dense(1, activation='tanh')  # 输出范围[-1, 1]，表示买入/卖出强度
    ])
    
    model.compile(optimizer='adam', loss='mse')
    
    return model

# 使用方式
# 1. 准备训练数据：过去60天价格 -> 未来5天收益方向
# 2. 训练LSTM模型
# 3. 使用模型预测信号，生成交易策略
```

## 实战案例：A股市场配对交易

### 数据准备

```python
# 使用Tushare或AKShare获取A股数据
import akshare as ak

# 获取沪深300成分股
hs300_stocks = ak.index_stock_cons_csindex(symbol="000300")

# 下载股票价格
def download_a_share_prices(stock_code, start_date='20200101', end_date='20231231'):
    """
    下载A股日线数据
    """
    df = ak.stock_zh_a_hist(
        symbol=stock_code,
        start_date=start_date,
        end_date=end_date,
        adjust="qfq"  # 前复权
    )
    
    df['日期'] = pd.to_datetime(df['日期'])
    df.set_index('日期', inplace=True)
    
    return df['收盘']

# 筛选配对（示例：银行股配对）
bank_stocks = ['601398', '601939', '601288', '601988']  # 四大行
bank_prices = pd.DataFrame()

for code in bank_stocks:
    bank_prices[code] = download_a_share_prices(code)

# 协整检验
selector = PairsSelector()
bank_pairs = selector.screen_pairs(bank_prices, min_corr=0.7)

print("银行股配对：")
print(bank_pairs)
```

### 注意事项

1. **A股做空限制**：无法直接做空个股，可以考虑：
   - 使用融券（如有标的）
   - 使用股指期货对冲市场风险
   - 仅做多，等待配对中的"低估"股票回归（半配对交易）

2. **涨停板限制**：A股有±10%涨跌停限制，可能导致价差无法及时修正

3. **T+1交易制度**：当天买入无法当天卖出，影响高频统计套利

## 结论

统计套利是一种相对稳健的量化策略，尤其适合市场中性、低风险偏好的投资者。

**关键要点**：

1. **均值回归是核心**：寻找价格偏离均衡的临时机会
2. **协整检验是基础**：系统化筛选配对，避免主观偏见
3. **风险管理至关重要**：配对失效、模型风险、执行风险都需要防范
4. **分散化是王道**：同时交易多个不相关配对，降低整体风险
5. **适应市场变化**：定期重新检验配对，动态调整策略参数

统计套利不是"印钞机"，但在严谨的研究、严格的风险控制和持续的策略迭代下，它可以成为量化投资组合中的重要组成部分。

---

**参考资料**：

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis"
2. Pole, A. (2007). "Statistical Arbitrage: Algorithmic Trading Insights and Techniques"
3. Montaña, J., et al. (2019). "Machine Learning-Based Statistical Arbitrage"
4. Do, B., et al. (2006). "Pairs Trading: A Cointegration Approach"

**免责声明**：本文仅供学术交流，不构成投资建议。统计套利涉及金融风险，请在专业人士指导下实践。
