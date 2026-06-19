---
title: "配对交易与协整分析"
description: "深入探讨配对交易的理论基础与实践方法，学习如何运用协整分析识别配对机会，构建市场中性策略。包含完整的Python实现代码和实战案例。"
pubDate: 2026-06-19
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "Python实战"]
coverImage: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

在传统股票投资中，投资者通常面临系统性风险（Market Risk）——无论选择哪只股票，都无法完全规避市场整体波动带来的风险。然而，**配对交易（Pairs Trading）**提供了一种市场中性（Market Neutral）的策略思路：通过同时买入一只股票并卖空另一只相关性较高的股票，构建对冲组合，从而在市场上涨或下跌时都能获取收益。

配对交易的核心思想是：**寻找两只价格具有长期均衡关系的股票，当它们的价格偏离这一均衡时，做多价格相对低估的股票，做空价格相对高估的股票，等待价格回归均衡后平仓获利**。

本文将深入探讨配对交易的理论基础——协整分析，介绍实战中的配对识别方法，并提供完整的Python实现代码。

## 配对交易的理论基础

### 1. 平稳性 vs 协整性

在介绍协整之前，我们需要先理解**平稳性（Stationarity）**的概念。

**平稳时间序列**是指其统计性质（如均值、方差、自协方差）不随时间变化的时间序列。对于平稳序列，我们可以使用传统的回归分析、ARIMA模型等方法进行建模和预测。

然而，金融价格序列（如股票价格）通常**不是平稳的**——它们往往具有趋势性，均值和方差会随时间变化。直接对两个非平稳的价格序列进行回归，会产生**伪回归（Spurious Regression）**问题：即使两个序列毫无关系，回归结果也可能显示显著的统计关系。

**协整（Cointegration）**解决了这一问题。协整的定义如下：

> 如果两个或多个非平稳时间序列的某种线性组合是平稳的，那么这些序列就是协整的。

数学表达：

设 $P_{1,t}$ 和 $P_{2,t}$ 是两个非平稳的价格序列（通常是一阶单整 $I(1)$）。如果存在系数 $\beta$，使得：

$$S_t = P_{1,t} - \beta P_{2,t}$$

是平稳序列（即 $S_t \sim I(0)$），那么 $P_{1,t}$ 和 $P_{2,t}$ 就是协整的。$S_t$ 称为**协整残差（Cointegrating Residual）**或**价差（Spread）**。

### 2. 协整的经济学含义

协整关系反映了两个价格序列之间的**长期均衡关系**。尽管短期内价格可能偏离这一均衡，但存在某种"引力"机制，使得价差最终会回归到均衡水平。

在股票市场中，协整关系通常存在于：

- **同行业公司**：如可口可乐和百事可乐、工商银行和建设银行
- **产业链上下游**：如原油价格和航空公司股价、铁矿石价格和钢铁股
- **替代品生产商**：如不同品牌的汽车制造商、不同航空公司

这些公司面临相似的经济环境、行业政策和市场冲击，因此它们的股价存在长期均衡关系。

### 3. 配对交易的利润来源

配对交易的利润来源于价差的**均值回归（Mean Reversion）**特性。

当价差 $S_t$ 偏离其长期均值 $\mu$ 时，我们预期它会在未来回归到均值附近。因此：

- 当 $S_t > \mu + k\sigma$（价差过高）时，我们认为 $P_{1,t}$ 相对高估，$P_{2,t}$ 相对低估，因此**做空 $P_1$，做多 $P_2$**
- 当 $S_t < \mu - k\sigma$（价差过低）时，我们认为 $P_{1,t}$ 相对低估，$P_{2,t}$ 相对高估，因此**做多 $P_1$，做空 $P_2$**

当价差回归到均值附近时，我们平仓获利。

## 协整检验方法

### 方法1：Engle-Granger两步法

Engle-Granger方法是检验协整关系最经典的 approach，分为两步：

**第一步：估计协整向量**

使用OLS回归估计长期均衡关系：

$$P_{1,t} = \alpha + \beta P_{2,t} + \epsilon_t$$

得到残差 $\hat{\epsilon}_t = P_{1,t} - \hat{\alpha} - \hat{\beta} P_{2,t}$

**第二步：检验残差的平稳性**

对残差 $\hat{\epsilon}_t$ 进行单位根检验（如ADF检验、PP检验）。如果残差是平稳的，则拒绝"无协整关系"的原假设，认为两个序列存在协整关系。

**Python实现：**

```python
import pandas as pd
import numpy as np
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

def engle_granger_test(price1, price2, plot_results=True):
    """
    Engle-Granger两步法协整检验
    
    参数:
    - price1: 第一个价格序列
    - price2: 第二个价格序列
    - plot_results: 是否绘制结果
    
    返回:
    - results: 包含回归结果、ADF检验结果等
    """
    # 第一步：OLS回归
    X = price2.values.reshape(-1, 1)
    X = np.hstack([np.ones_like(X), X])  # 添加截距项
    model = OLS(price1.values, X).fit()
    
    alpha = model.params[0]
    beta = model.params[1]
    residuals = model.resid
    
    # 第二步：ADF检验残差平稳性
    adf_result = adfuller(residuals, autolag='AIC')
    
    results = {
        'alpha': alpha,
        'beta': beta,
        'residuals': residuals,
        'adf_statistic': adf_result[0],
        'adf_pvalue': adf_result[1],
        'adf_critical_values': adf_result[4],
        'is_cointegrated': adf_result[1] < 0.05  # 5%显著性水平
    }
    
    # 可视化
    if plot_results:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 原始价格序列
        axes[0].plot(price1.index, price1.values, label='Price 1', linewidth=2)
        axes[0].plot(price2.index, price2.values, label='Price 2', linewidth=2)
        axes[0].set_title('Original Price Series')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 协整残差（价差）
        axes[1].plot(price1.index, residuals, color='purple', linewidth=1.5)
        axes[1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        axes[1].axhline(y=residuals.mean(), color='red', linestyle='--', 
                       label=f'Mean: {residuals.mean():.2f}')
        axes[1].fill_between(price1.index, 
                            residuals.mean() - 2*residuals.std(),
                            residuals.mean() + 2*residuals.std(),
                            alpha=0.2, color='gray', label='±2 STD')
        axes[1].set_title('Cointegrating Residuals (Spread)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # 价格比
        price_ratio = price1.values / price2.values
        axes[2].plot(price1.index, price_ratio, color='green', linewidth=1.5)
        axes[2].axhline(y=price_ratio.mean(), color='red', linestyle='--',
                       label=f'Mean Ratio: {price_ratio.mean():.2f}')
        axes[2].set_title('Price Ratio')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return results

# 使用示例
# results = engle_granger_test(stock1_prices, stock2_prices)
# print(f"Beta: {results['beta']:.4f}")
# print(f"ADF Statistic: {results['adf_statistic']:.4f}")
# print(f"ADF p-value: {results['adf_pvalue']:.4f}")
# print(f"Is Cointegrated: {results['is_cointegrated']}")
```

### 方法2：Johansen检验

Engle-Granger方法只适用于两个序列的协整检验。当需要检验**多个序列**之间是否存在协整关系时，需要使用**Johansen检验**。

Johansen检验基于向量自回归（VAR）模型，可以识别出多个协整向量。

**Python实现：**

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank
import warnings
warnings.filterwarnings('ignore')

def johansen_test(price_data, det_order=0, k_ar_diff=1):
    """
    Johansen协整检验（适用于多变量）
    
    参数:
    - price_data: DataFrame，每列为一个价格序列
    - det_order: 确定性项的顺序（0=无常数项，1=有常数项，等等）
    - k_ar_diff: VAR模型滞后阶数
    
    返回:
    - results: Johansen检验结果
    """
    # 选择协整秩（协整向量的个数）
    rank_selection = select_coint_rank(price_data, det_order, k_ar_diff)
    selected_rank = rank_selection.rank
    
    # 拟合VECM模型
    vecm_model = VECM(price_data, k_ar_diff=k_ar_diff, coint_rank=selected_rank, 
                      deterministic=det_order)
    vecm_results = vecm_model.fit()
    
    results = {
        'selected_rank': selected_rank,
        'coint_vectors': vecm_results.beta,  # 协整向量
        'loading_matrix': vecm_results.alpha,  # 加载矩阵
        'residuals': vecm_results.resid,
        'llf': vecm_results.llf  # 对数似然值
    }
    
    return results

# 使用示例（检验3只股票是否存在协整关系）
# stocks_data = pd.DataFrame({
#     'Stock1': stock1_prices,
#     'Stock2': stock2_prices,
#     'Stock3': stock3_prices
# })
# johansen_results = johansen_test(stocks_data)
# print(f"Number of cointegrating vectors: {johansen_results['selected_rank']}")
```

### 方法3：距离法（Distance Approach)

除了计量经济学方法，还可以使用**距离法**来识别配对机会。距离法的核心思想是：计算所有股票对之间的价格距离（如欧氏距离、马氏距离），距离最小的若干对即为潜在的配对。

**步骤：**

1. 标准化所有股票的价格序列（如计算Z-score）
2. 计算任意两只股票之间的累积距离（如sum of squared differences）
3. 选择距离最小的Top N对作为候选配对
4. 对候选配对进行协整检验，确认其统计显著性

**Python实现：**

```python
from itertools import combinations
from tqdm import tqdm

def find_pairs_distance(prices_df, top_n=50, min_days=252):
    """
    使用距离法寻找潜在的配对
    
    参数:
    - prices_df: DataFrame，每列为一只股票的价格序列
    - top_n: 返回的配对数量
    - min_days: 最小数据长度
    
    返回:
    - pairs: 按距离排序的配对列表
    """
    # 标准化价格
    normalized_prices = prices_df.apply(lambda x: (x - x.mean()) / x.std())
    
    pairs = []
    stock_names = normalized_prices.columns.tolist()
    
    # 计算所有配对的距離
    for stock1, stock2 in tqdm(combinations(stock_names, 2), 
                               total=len(list(combinations(stock_names, 2)))):
        # 计算欧氏距离
        distance = np.sqrt(((normalized_prices[stock1] - normalized_prices[stock2]) ** 2).sum())
        
        # 计算相关性
        correlation = normalized_prices[stock1].corr(normalized_prices[stock2])
        
        pairs.append({
            'stock1': stock1,
            'stock2': stock2,
            'distance': distance,
            'correlation': correlation
        })
    
    # 按距离排序
    pairs_df = pd.DataFrame(pairs)
    pairs_df = pairs_df.sort_values('distance').head(top_n)
    
    return pairs_df

# 使用示例
# potential_pairs = find_pairs_distance(stock_prices_df, top_n=100)
# for idx, row in potential_pairs.iterrows():
#     print(f"{row['stock1']} - {row['stock2']}: Distance={row['distance']:.2f}, Corr={row['correlation']:.2f}")
```

## 实战案例：构建配对交易策略

下面，我们构建一个完整的配对交易策略，包括配对识别、信号生成、回测等步骤。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.stattools import adfuller

class PairsTradingStrategy:
    """
    配对交易策略框架
    """
    def __init__(self, initial_capital=1000000, entry_threshold=2.0, exit_threshold=0.5):
        """
        初始化配对交易策略
        
        参数:
        - initial_capital: 初始资金
        - entry_threshold: 入场阈值（标准差的倍数）
        - exit_threshold: 出场阈值（标准差的倍数）
        """
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.positions = None
        self.portfolio_value = None
        
    def find_cointegrated_pairs(self, prices_df, significance_level=0.05):
        """
        寻找协整配对
        
        参数:
        - prices_df: DataFrame，每列为一只股票的价格
        - significance_level: 显著性水平
        
        返回:
        - cointegrated_pairs: 协整配对列表
        """
        cointegrated_pairs = []
        stock_names = prices_df.columns.tolist()
        
        for i in range(len(stock_names)):
            for j in range(i+1, len(stock_names)):
                stock1 = stock_names[i]
                stock2 = stock_names[j]
                
                # Engle-Granger检验
                X = prices_df[stock2].values.reshape(-1, 1)
                X = np.hstack([np.ones_like(X), X])
                model = OLS(prices_df[stock1].values, X).fit()
                residuals = model.resid
                
                # ADF检验
                adf_result = adfuller(residuals, autolag='AIC')
                
                if adf_result[1] < significance_level:
                    cointegrated_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'adf_statistic': adf_result[0],
                        'adf_pvalue': adf_result[1],
                        'hedge_ratio': model.params[1],
                        'intercept': model.params[0]
                    })
        
        return pd.DataFrame(cointegrated_pairs)
    
    def calculate_spread(self, price1, price2, hedge_ratio, intercept):
        """
        计算价差（协整残差）
        """
        spread = price1 - (intercept + hedge_ratio * price2)
        return spread
    
    def generate_signals(self, spread):
        """
        根据价差生成交易信号
        
        返回:
        - signals: 1（做多价差）、-1（做空价差）、0（平仓）
        """
        # 计算价差的Z-score
        spread_mean = spread.rolling(window=252).mean()
        spread_std = spread.rolling(window=252).std()
        z_score = (spread - spread_mean) / spread_std
        
        signals = pd.Series(index=spread.index, data=0)
        
        # 生成信号
        signals[z_score > self.entry_threshold] = -1  # 价差过高，做空
        signals[z_score < -self.entry_threshold] = 1   # 价差过低，做多
        
        # 平仓信号
        signals[(z_score > -self.exit_threshold) & (z_score < self.exit_threshold)] = 0
        
        # 填充信号（保持仓位直到平仓）
        signals = signals.replace(0, np.nan)
        signals = signals.fillna(method='ffill').fillna(0)
        
        return signals, z_score
    
    def backtest(self, price1, price2, hedge_ratio, intercept, 
                transaction_cost=0.001):
        """
        回测配对交易策略
        
        参数:
        - price1: 第一只股票价格
        - price2: 第二只股票价格
        - hedge_ratio: 对冲比率
        - intercept: 截距项
        - transaction_cost: 交易成本
        
        返回:
        - performance: 策略性能指标
        """
        # 计算价差
        spread = self.calculate_spread(price1, price2, hedge_ratio, intercept)
        
        # 生成信号
        signals, z_score = self.generate_signals(spread)
        
        # 计算收益
        price1_returns = price1.pct_change()
        price2_returns = price2.pct_change()
        
        # 组合收益（市场中性）
        # 当signal=1时：做多stock1，做空hedge_ratio单位的stock2
        # 当signal=-1时：做空stock1，做多hedge_ratio单位的stock2
        portfolio_returns = signals.shift(1) * price1_returns - \
                          (signals.shift(1) * hedge_ratio) * price2_returns
        
        # 计算交易成本
        trades = signals.diff().abs()
        portfolio_returns -= trades * transaction_cost
        
        # 计算累积净值
        cumulative_value = (1 + portfolio_returns).cumprod() * self.initial_capital
        
        # 计算性能指标
        performance = self._calculate_performance(portfolio_returns, cumulative_value)
        
        # 保存结果
        self.positions = signals
        self.portfolio_value = cumulative_value
        self.spread = spread
        self.z_score = z_score
        
        return performance, portfolio_returns
    
    def _calculate_performance(self, returns, cumulative_value):
        """
        计算策略性能指标
        """
        # 年化收益
        annual_return = returns.mean() * 252
        
        # 年化波动
        annual_vol = returns.std() * np.sqrt(252)
        
        # 夏普比率
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # 最大回撤
        peak = cumulative_value.expanding().max()
        drawdown = (cumulative_value - peak) / peak
        max_drawdown = drawdown.min()
        
        # 胜率
        winning_days = (returns > 0).sum()
        total_days = (returns != 0).sum()
        win_rate = winning_days / total_days if total_days > 0 else 0
        
        return {
            'Annual Return': f"{annual_return:.2%}",
            'Annual Volatility': f"{annual_vol:.2%}",
            'Sharpe Ratio': f"{sharpe:.2f}",
            'Max Drawdown': f"{max_drawdown:.2%}",
            'Win Rate': f"{win_rate:.2%}",
            'Total Return': f"{(cumulative_value.iloc[-1] / self.initial_capital - 1):.2%}"
        }
    
    def visualize_results(self, price1, price2, stock1_name='Stock1', stock2_name='Stock2'):
        """
        可视化回测结果
        """
        fig, axes = plt.subplots(4, 1, figsize=(15, 12))
        
        # 价格序列
        ax1 = axes[0]
        ax1.plot(price1.index, price1.values, label=stock1_name, linewidth=2)
        ax1.set_ylabel(stock1_name, color='blue')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        ax1_twin = ax1.twinx()
        ax1_twin.plot(price2.index, price2.values, label=stock2_name, 
                     linewidth=2, color='orange')
        ax1_twin.set_ylabel(stock2_name, color='orange')
        ax1_twin.legend(loc='upper right')
        
        # 价差和Z-score
        axes[1].plot(self.spread.index, self.spread.values, 
                    color='purple', linewidth=1.5, label='Spread')
        axes[1].axhline(y=self.spread.mean(), color='black', 
                       linestyle='--', label='Mean')
        axes[1].fill_between(self.spread.index, 
                            self.spread.mean() - 2*self.spread.std(),
                            self.spread.mean() + 2*self.spread.std(),
                            alpha=0.2, color='gray')
        axes[1].set_title('Spread (Cointegrating Residual)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(self.z_score.index, self.z_score.values, 
                    color='green', linewidth=1.5, label='Z-score')
        axes[2].axhline(y=self.entry_threshold, color='red', 
                       linestyle='--', label=f'Entry (+{self.entry_threshold})')
        axes[2].axhline(y=-self.entry_threshold, color='red', 
                       linestyle='--', label=f'Entry (-{self.entry_threshold})')
        axes[2].axhline(y=self.exit_threshold, color='blue', 
                       linestyle='--', label=f'Exit (+{self.exit_threshold})')
        axes[2].axhline(y=-self.exit_threshold, color='blue', 
                       linestyle='--', label=f'Exit (-{self.exit_threshold})')
        axes[2].fill_between(self.z_score.index, 
                            self.positions.values * 0.5, 
                            alpha=0.3, label='Position')
        axes[2].set_title('Z-score and Trading Signals')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # 累积净值
        axes[3].plot(self.portfolio_value.index, self.portfolio_value.values, 
                    color='darkgreen', linewidth=2, label='Pairs Trading')
        axes[3].axhline(y=self.initial_capital, color='black', 
                       linestyle='--', label='Initial Capital')
        axes[3].set_title('Cumulative Portfolio Value')
        axes[3].legend()
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return fig

# 完整回测示例
if __name__ == "__main__":
    # 生成模拟数据（两只协整的股票）
    np.random.seed(42)
    dates = pd.date_range('2018-01-01', '2025-12-31', freq='D')
    n_days = len(dates)
    
    # 共同的随机游走成分
    common_factor = np.cumsum(np.random.normal(0, 0.01, n_days))
    
    # 股票1和股票2的价格（具有协整关系）
    price1 = 100 + 10 * common_factor + np.cumsum(np.random.normal(0, 0.005, n_days))
    price2 = 50 + 5 * common_factor + np.cumsum(np.random.normal(0, 0.003, n_days))
    
    price1_series = pd.Series(price1, index=dates)
    price2_series = pd.Series(price2, index=dates)
    
    # 构建配对交易策略
    strategy = PairsTradingStrategy(
        initial_capital=1000000,
        entry_threshold=2.0,
        exit_threshold=0.5
    )
    
    # 回测
    performance, returns = strategy.backtest(
        price1_series, price2_series,
        hedge_ratio=2.0,  # 模拟的对冲比率
        intercept=0.0,
        transaction_cost=0.001
    )
    
    # 输出性能指標
    print("=== 配对交易策略性能 ===")
    for key, value in performance.items():
        print(f"{key}: {value}")
    
    # 可视化
    fig = strategy.visualize_results(price1_series, price2_series, 
                                   'Stock A', 'Stock B')
    plt.savefig('pairs_trading_backtest.png', dpi=300, bbox_inches='tight')
```

## 配对交易的关键挑战

### 1. 配对断裂（Pair Breakdown）

协整关系并非永久存在。当公司基本面发生重大变化（如并购、重组、行业政策变化等）时，原本存在的协整关系可能突然断裂，导致配对交易策略遭受重大损失。

**应对方法：**

- 定期重新检验协整关系（如每月或每季度）
- 设置止损机制（当价差突破历史极值时强制平仓）
- 分散投资多个配对，降低单一配对断裂的影响

### 2. 模型风险

协整检验本身存在模型风险：

- ADF检验的功效较低（容易犯第二类错误，即漏检真实的协整关系）
- 对冲比率（hedge ratio）可能时变，静态回归可能不准确
- 价差的均值和波动率可能非平稳（具有结构性断点）

**应对方法：**

- 使用滚动窗口或扩展窗口进行动态回归
- 结合多种协整检验方法（如Engle-Granger + Johansen）
- 使用贝叶斯方法估计时变参数

### 3. 执行风险

配对交易需要同时执行买入和卖空操作。在实战中，可能面临：

- **卖空限制**：某些股票可能无法卖空，或借券成本高昂
- **流动性风险**：两只股票的流动性可能差异较大，导致执行价格偏离预期
- **交易延迟**：买入和卖空指令可能无法同时成交，产生 legs risk

**应对方法：**

- 选择高流动性、可卖空的股票
- 使用算法交易（如VWAP、TWAP）降低冲击成本
- 设置容忍度（如允许部分成交，再逐步调整仓位）

### 4. 资金利用率低

配对交易通常是市场中性策略，收益来源于价差的均值回归，而非市场方向的判断。因此，其收益幅度通常较小，需要较高的资金利用率（杠杆）才能获得可观的绝对收益。

**应对方法：**

- 在风险可控的前提下，适度使用杠杆
- 同时交易多个配对，提高资金利用效率
- 结合其他策略（如动量、事件驱动等），构建多策略组合

## 结论

配对交易是一种经典的市场中性策略，通过协整分析识别具有长期均衡关系的股票对，捕捉价差的均值回归利润。尽管面临配对断裂、模型风险、执行风险等挑战，配对交易仍为量化投资提供了有价值的工具。

成功实施配对交易需要：

1. **扎实的计量经济学基础**：理解协整、平稳性等核心概念
2. **严谨的统计检验**：使用多种方法验证配对的有效性
3. **周密的风险管理**：设置止损、分散投资、定期重新检验
4. **高效的执行系统**：降低交易成本，管理legs risk

随着机器学习技术的发展，越来越多的研究开始探索**深度学习在配对交易中的应用**（如用神经网络预测价差方向、用强化学习优化调仓策略等）。这些新方法有望提升配对交易策略的性能和稳健性。

## 参考资料

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis."
2. Gatev, E., et al. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule."
3. Elliott, R. J., et al. (2005). "Pairs Trading."
4. Liu, J. (2019). "Machine Learning in Pairs Trading Strategies."

---

*本文提供的代码和策略仅用于教育目的，不构成投资建议。配对交易涉及卖空操作，可能面临无限损失风险。实际投资中请根据自身风险承受能力和投资目标谨慎决策。*
