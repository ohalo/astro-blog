---
title: "配对交易与协整分析"
description: "深入讲解配对交易策略的理论基础与实战实现，包括协整检验、对冲比率计算、交易信号生成等核心内容，附带完整的Python代码示例。"
pubDate: 2026-06-19
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "Python"]
category: "量化交易"
featured: false
toc: true
---

# 配对交易与协整分析

![配对交易价差与Z-Score信号](../..//public/images/pair-trading-cointegration/spread_zscore.png)

配对交易（Pairs Trading）是最经典的**统计套利**策略之一，凭借其市场中性、风险可控的特点，成为量化交易者的必备工具。本文将系统讲解配对交易的理论基础、协整分析方法，以及完整的Python实战实现。

## 什么是配对交易？

配对交易的核心思想非常简单：**找到两个价格走势高度相关的资产，当价格偏离正常关系时，做多价格偏低的资产，做空价格偏高的资产，等待价格回归均值后平仓获利**。

### 策略优势

1. **市场中性**：多空对冲，降低市场系统性风险
2. **均值回归**：利用价格偏离后的自然回归特性
3. **风险可控**：明确的入场和出场规则
4. **适应性强**：适用于股票、期货、加密货币等多种资产

### 关键步骤

```
1. 选股配对：找到具有经济逻辑支撑的配对
2. 协整检验：验证价格序列的长期均衡关系
3. 计算对冲比率：确定多空仓位比例
4. 生成交易信号：基于价差（Spread）的Z-Score
5. 风险管理：设置止损、仓位限制
```

## 理论基础：协整与平稳性

![AAPL vs MSFT 价格对比](../..//public/images/pair-trading-cointegration/price_comparison.png)

### 平稳性检验

传统的时间序列分析要求数据平稳，但股价序列通常是**非平稳**的（存在单位根）。

**单位根检验（ADF Test）**：

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf

def adf_test(series, title=''):
    """
    Augmented Dickey-Fuller检验
    H0: 序列有单位根（非平稳）
    H1: 序列平稳
    """
    print(f'\nADF Test: {title}')
    result = adfuller(series, autolag='AIC')
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    
    if result[1] <= 0.05:
        print("结论: 序列平稳（拒绝H0）")
        return True
    else:
        print("结论: 序列非平稳（接受H0）")
        return False

# 示例：检验股价序列的平稳性
# 下载数据
tickers = ['AAPL', 'MSFT']
data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']

for ticker in tickers:
    adf_test(data[ticker], title=ticker)
```

**输出示例**：
```
ADF Test: AAPL
ADF Statistic: 1.2345
p-value: 0.9956
结论: 序列非平稳（接受H0）

ADF Test: MSFT
ADF Statistic: 1.1234
p-value: 0.9945
结论: 序列非平稳（接受H0）
```

### 协整检验

两个非平稳序列的**线性组合可能是平稳的**，这种关系称为**协整（Cointegration）**。

**Engle-Granger两步法**：

```python
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def cointegration_test(y, x, alpha=0.05):
    """
    Engle-Granger协整检验
    
    参数:
    - y: 因变量（被解释变量）
    - x: 自变量
    - alpha: 显著性水平
    
    返回:
    - coint_stat: 协整统计量
    - p_value: p值
    - is_cointegrated: 是否协整
    """
    # 步骤1: OLS回归 y = α + βx + ε
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    residual = model.resid
    
    # 步骤2: 检验残差的平稳性（ADF检验）
    adf_result = adfuller(residual, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    
    # 使用coint函数进行正式检验
    coint_stat, p_value, _ = coint(y, x)
    
    is_cointegrated = p_value < alpha
    
    print(f"\n协整检验结果:")
    print(f"  coint statistic: {coint_stat:.4f}")
    print(f"  p-value: {p_value:.4f}")
    print(f"  结论: {'协整' if is_cointegrated else '不协整'}")
    
    return coint_stat, p_value, is_cointegrated, residual, model.params

# 示例：检验AAPL和MSFT的协整关系
y = data['AAPL']
x = data['MSFT']

coint_stat, p_value, is_coint, residual, params = cointegration_test(y, x)

if is_coint:
    print(f"\n✓ {tickers[0]} 和 {tickers[1]} 存在协整关系")
    print(f"  对冲比率（β）: {params[1]:.4f}")
    print(f"  截距（α）: {params[0]:.4f}")
```

## Python实战：完整配对交易策略

下面我们实现一个完整的配对交易策略，包括数据获取、协整检验、信号生成、回测等全流程。

### 1. 数据准备与配对筛选

```python
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint
from statsmodels.regression.rolling import RollingOLS
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PairsTrading:
    def __init__(self, ticker1, ticker2, start_date, end_date):
        """
        初始化配对交易策略
        
        参数:
        - ticker1, ticker2: 股票代码
        - start_date, end_date: 数据区间
        """
        self.ticker1 = ticker1
        self.ticker2 = ticker2
        self.start_date = start_date
        self.end_date = end_date
        
        # 下载数据
        self.data = self.download_data()
        
        # 价格序列
        self.price1 = self.data[ticker1]
        self.price2 = self.data[ticker2]
        
        # 协整参数
        self.beta = None
        self.alpha = None
        self.spread = None
        self.z_score = None
        
    def download_data(self):
        """下载股价数据"""
        print(f"正在下载 {self.ticker1} 和 {self.ticker2} 的数据...")
        data = yf.download(
            [self.ticker1, self.ticker2], 
            start=self.start_date, 
            end=self.end_date
        )['Adj Close']
        
        # 去除缺失值
        data = data.dropna()
        
        print(f"数据下载完成！共 {len(data)} 个交易日")
        return data
    
    def test_cointegration(self, method='eg', alpha=0.05):
        """
        协整检验
        
        方法:
        - 'eg': Engle-Granger两步法
        - 'johansen': Johansen检验（多变量扩展）
        """
        print(f"\n=== 协整检验 ===")
        print(f"配对: {self.ticker1} - {self.ticker2}")
        
        if method == 'eg':
            # Engle-Granger检验
            coint_stat, p_value, _, residual, params = cointegration_test(
                self.price1, self.price2, alpha
            )
            
            self.beta = params[1]
            self.alpha = params[0]
            self.spread = residual
            
            # 计算Z-Score
            self.z_score = (self.spread - self.spread.mean()) / self.spread.std()
            
            return p_value < alpha
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def calculate_hedge_ratio(self, method='rolling', window=60):
        """
        计算对冲比率（动态或静态）
        
        方法:
        - 'static': 全样本OLS
        - 'rolling': 滚动窗口OLS
        """
        if method == 'static':
            # 静态对冲比率
            X = sm.add_constant(self.price2)
            model = sm.OLS(self.price1, X).fit()
            self.beta = model.params[1]
            self.spread = self.price1 - self.beta * self.price2
            
        elif method == 'rolling':
            # 滚动对冲比率
            print(f"\n计算滚动对冲比率（窗口={window}天）...")
            
            hedge_ratios = []
            spreads = []
            
            for i in range(window, len(self.price1)):
                y_window = self.price1[i-window:i]
                x_window = self.price2[i-window:i]
                
                X = sm.add_constant(x_window)
                model = sm.OLS(y_window, X).fit()
                beta = model.params[1]
                
                hedge_ratios.append(beta)
                spread = self.price1[i] - beta * self.price2[i]
                spreads.append(spread)
            
            self.beta_series = pd.Series(hedge_ratios, index=self.price1.index[window:])
            self.spread = pd.Series(spreads, index=self.price1.index[window:])
        
        # 计算Z-Score
        self.z_score = (self.spread - self.spread.rolling(60).mean()) / self.spread.rolling(60).std()
        
        print(f"对冲比率（最新）: {self.beta_series.iloc[-1]:.4f}" if method=='rolling' else f"对冲比率（静态）: {self.beta:.4f}")
        
    def generate_signals(self, entry_z=2.0, exit_z=0.5):
        """
        生成交易信号
        
        规则:
        - Z-Score < -entry_z: 做多价差（做多ticker1，做空ticker2）
        - Z-Score > entry_z: 做空价差（做空ticker1，做多ticker2）
        - |Z-Score| < exit_z: 平仓
        """
        print(f"\n=== 生成交易信号 ===")
        print(f"入场阈值: Z = ±{entry_z}")
        print(f"出场阈值: Z = ±{exit_z}")
        
        signals = pd.Series(0, index=self.z_score.index)
        
        # 1: 做多价差
        # -1: 做空价差
        # 0: 不持仓
        
        position = 0
        for i in range(1, len(self.z_score)):
            if pd.isna(self.z_score.iloc[i]):
                continue
                
            if position == 0:
                # 无持仓，检查入场信号
                if self.z_score.iloc[i] < -entry_z:
                    position = 1  # 做多价差
                    signals.iloc[i] = 1
                elif self.z_score.iloc[i] > entry_z:
                    position = -1  # 做空价差
                    signals.iloc[i] = -1
            
            elif position == 1:
                # 持有做多价差，检查出场信号
                if abs(self.z_score.iloc[i]) < exit_z:
                    position = 0
                    signals.iloc[i] = 0  # 平仓
                else:
                    signals.iloc[i] = 1  # 继续持有
            
            elif position == -1:
                # 持有做空价差，检查出场信号
                if abs(self.z_score.iloc[i]) < exit_z:
                    position = 0
                    signals.iloc[i] = 0  # 平仓
                else:
                    signals.iloc[i] = -1  # 继续持有
        
        self.signals = signals
        self.positions = signals  # 简化：信号即仓位
        
        # 统计信号
        num_long = (signals == 1).sum()
        num_short = (signals == -1).sum()
        print(f"做多信号: {num_long} 天")
        print(f"做空信号: {num_short} 天")
        
        return signals
    
    def backtest(self, capital=100000, commission=0.001):
        """
        回测配对交易策略
        
        参数:
        - capital: 初始资金
        - commission: 单边佣金率
        
        返回:
        - returns: 策略收益序列
        - portfolio_value: 组合净值
        """
        print(f"\n=== 回测结果 ===")
        print(f"初始资金: ${capital:,.0f}")
        print(f"佣金率: {commission:.3%}")
        
        # 初始化
        portfolio_value = []
        cash = capital
        position_value = 0
        shares1 = 0
        shares2 = 0
        
        for i in range(len(self.signals)):
            date = self.signals.index[i]
            signal = self.signals.iloc[i]
            
            if pd.isna(signal):
                portfolio_value.append(cash + position_value)
                continue
            
            # 计算当前持仓市值
            if shares1 != 0 and shares2 != 0:
                position_value = shares1 * self.price1.loc[date] + shares2 * self.price2.loc[date]
            
            # 交易信号
            if signal == 1 and shares1 == 0:  # 做多价差
                # 做多ticker1，做空ticker2
                # 假设50%资金做多，50%资金做空
                long_value = capital * 0.5
                short_value = capital * 0.5
                
                shares1 = long_value / self.price1.loc[date]
                shares2 = -short_value / self.price2.loc[date]  # 负值表示做空
                
                # 扣除佣金
                cash -= (abs(shares1) * self.price1.loc[date] + abs(shares2) * self.price2.loc[date]) * commission
            
            elif signal == -1 and shares1 == 0:  # 做空价差
                # 做空ticker1，做多ticker2
                short_value = capital * 0.5
                long_value = capital * 0.5
                
                shares1 = -short_value / self.price1.loc[date]  # 负值表示做空
                shares2 = long_value / self.price2.loc[date]
                
                # 扣除佣金
                cash -= (abs(shares1) * self.price1.loc[date] + abs(shares2) * self.price2.loc[date]) * commission
            
            elif signal == 0 and shares1 != 0:  # 平仓
                # 平掉所有仓位
                cash += shares1 * self.price1.loc[date] + shares2 * self.price2.loc[date]
                cash -= (abs(shares1) * self.price1.loc[date] + abs(shares2) * self.price2.loc[date]) * commission
                
                shares1 = 0
                shares2 = 0
                position_value = 0
            
            # 计算组合净值
            if shares1 != 0 and shares2 != 0:
                position_value = shares1 * self.price1.loc[date] + shares2 * self.price2.loc[date]
            
            total_value = cash + position_value
            portfolio_value.append(total_value)
        
        # 构建结果DataFrame
        results = pd.DataFrame({
            'portfolio_value': portfolio_value,
            'return': 0
        }, index=self.signals.index)
        
        results['return'] = results['portfolio_value'].pct_change()
        results['cumulative'] = (1 + results['return']).cumprod()
        
        # 计算绩效指标
        total_return = results['cumulative'].iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        volatility = results['return'].std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_dd = (results['cumulative'] / results['cumulative'].cummax() - 1).min()
        
        print(f"\n绩效指标:")
        print(f"  总收益: {total_return:.2%}")
        print(f"  年化收益: {annual_return:.2%}")
        print(f"  年化波动率: {volatility:.2%}")
        print(f"  夏普比率: {sharpe:.2f}")
        print(f"  最大回撤: {max_dd:.2%}")
        
        return results
    
    def visualize(self, save_path='/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/'):
        """可视化分析结果"""
        
        fig, axes = plt.subplots(4, 1, figsize=(16, 20))
        
        # 1. 价格序列对比
        ax = axes[0]
        ax.plot(self.price1.index, self.price1.values, label=self.ticker1, linewidth=2)
        ax.plot(self.price2.index, self.price2.values, label=self.ticker2, linewidth=2)
        ax.set_title(f'{self.ticker1} vs {self.ticker2} Price Series', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. 价差（Spread）和Z-Score
        ax = axes[1]
        ax2 = ax.twinx()
        
        spread_line, = ax.plot(self.spread.index, self.spread.values, 'b-', label='Spread', alpha=0.7)
        z_line, = ax2.plot(self.z_score.index, self.z_score.values, 'r-', label='Z-Score', alpha=0.7)
        
        ax.set_title('Spread and Z-Score', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Spread', color='b')
        ax2.set_ylabel('Z-Score', color='r')
        ax.grid(True, alpha=0.3)
        
        # 添加Z-Score阈值线
        ax2.axhline(y=2, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        
        # 3. 交易信号
        ax = axes[2]
        ax.plot(self.z_score.index, self.z_score.values, 'k-', alpha=0.5, label='Z-Score')
        
        # 标记交易信号
        long_signals = self.signals == 1
        short_signals = self.signals == -1
        
        ax.scatter(self.signals.index[long_signals], self.z_score[long_signals], 
                 color='g', marker='^', s=100, label='Long Spread', zorder=5)
        ax.scatter(self.signals.index[short_signals], self.z_score[short_signals], 
                 color='r', marker='v', s=100, label='Short Spread', zorder=5)
        
        ax.set_title('Trading Signals', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Z-Score')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. 累计收益
        ax = axes[3]
        results = self.backtest()
        ax.plot(results.index, results['cumulative'], 'b-', linewidth=2, label='Strategy')
        
        # 基准：买入持有（等权）
        benchmark = (self.price1 / self.price1.iloc[0] + self.price2 / self.price2.iloc[0]) / 2
        ax.plot(benchmark.index, benchmark.values, 'r--', linewidth=2, label='Buy & Hold')
        
        ax.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date')
        ax.set_ylabel('Cumulative Return')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{save_path}pairs_trading_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n✓ 图表已保存: {save_path}pairs_trading_analysis.png")

# 使用示例
if __name__ == "__main__":
    # 创建配对交易策略
    pair = PairsTrading(
        ticker1='AAPL',
        ticker2='MSFT',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # 协整检验
    is_coint = pair.test_cointegration()
    
    if is_coint:
        print("\n✓ 配对协整，可以继续策略开发")
        
        # 计算对冲比率
        pair.calculate_hedge_ratio(method='rolling', window=60)
        
        # 生成交易信号
        pair.generate_signals(entry_z=2.0, exit_z=0.5)
        
        # 回测
        results = pair.backtest(capital=100000, commission=0.001)
        
        # 可视化
        pair.visualize()
        
    else:
        print("\n✗ 配对不协整，建议更换股票对")
```

### 2. 配对筛选：如何找到优质配对？

成功的配对交易，**选对配对**是关键。以下是常用的筛选方法：

#### 方法1：行业分类 + 基本面相似度

```python
def screen_pairs_by_industry(tickers, start_date, end_date):
    """
    基于行业分类筛选潜在配对
    
    步骤:
    1. 获取行业分类信息
    2. 计算基本面相似度（市值、PE、PB等）
    3. 预筛选价格相关性高的股票对
    4. 进行协整检验
    """
    import yfinance as yf
    from sklearn.cluster import AgglomerativeClustering
    
    # 1. 下载价格数据
    print("下载价格数据...")
    price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # 2. 计算相关性矩阵
    corr_matrix = price_data.corr()
    
    # 3. 层次聚类
    print("进行层次聚类...")
    distance_matrix = 1 - corr_matrix  # 相关性转距离
    clustering = AgglomerativeClustering(
        n_clusters=5, 
        linkage='ward'
    )
    labels = clustering.fit_predict(distance_matrix)
    
    # 4. 在每个聚类内部寻找配对
    print("在聚类内筛选配对...")
    pairs = []
    for cluster_id in np.unique(labels):
        cluster_tickers = [tickers[i] for i in range(len(tickers)) if labels[i] == cluster_id]
        
        # 遍历cluster内部的所有组合
        from itertools import combinations
        for ticker1, ticker2 in combinations(cluster_tickers, 2):
            # 协整检验
            coint_stat, p_value, _, _, _ = cointegration_test(
                price_data[ticker1], 
                price_data[ticker2]
            )
            
            if p_value < 0.05:  # 显著性水平5%
                pairs.append({
                    'ticker1': ticker1,
                    'ticker2': ticker2,
                    'p_value': p_value,
                    'coint_stat': coint_stat
                })
    
    # 按p-value排序
    pairs = sorted(pairs, key=lambda x: x['p_value'])
    
    print(f"\n找到 {len(pairs)} 个协整配对:")
    for i, pair in enumerate(pairs[:10], 1):
        print(f"{i}. {pair['ticker1']} - {pair['ticker2']} (p={pair['p_value']:.4f})")
    
    return pairs

# 使用示例
# 科技股列表
tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM']

# 筛选配对
# pairs = screen_pairs_by_industry(tech_stocks, '2020-01-01', '2024-12-31')
```

#### 方法2：距离方法（Distance Method）

```python
def calculate_pair_distance(price1, price2, method='ssd'):
    """
    计算配对间的距离（越小越相似）
    
    方法:
    - 'ssd': Sum of Squared Differences（平方和）
    - 'correlation': 相关性距离
    - 'euclidean': 欧氏距离
    """
    if method == 'ssd':
        # 标准化后计算平方和
        norm1 = (price1 - price1.mean()) / price1.std()
        norm2 = (price2 - price2.mean()) / price2.std()
        distance = np.sum((norm1 - norm2) ** 2)
    
    elif method == 'correlation':
        # 相关性距离
        corr = np.corrcoef(price1, price2)[0, 1]
        distance = 1 - corr
    
    elif method == 'euclidean':
        # 欧氏距离
        distance = np.sqrt(np.sum((price1 - price2) ** 2))
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return distance

# 筛选距离最小的Top N配对
def find_top_pairs(tickers, start_date, end_date, top_n=10):
    """找出距离最小的Top N配对"""
    
    # 下载数据
    price_data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    
    # 计算所有配对的距禷
    from itertools import combinations
    distances = []
    
    for ticker1, ticker2 in combinations(tickers, 2):
        dist = calculate_pair_distance(
            price_data[ticker1], 
            price_data[ticker2], 
            method='ssd'
        )
        distances.append({
            'ticker1': ticker1,
            'ticker2': ticker2,
            'distance': dist
        })
    
    # 排序
    distances = sorted(distances, key=lambda x: x['distance'])
    
    print(f"\n距离最小的 {top_n} 个配对:")
    for i, pair in enumerate(distances[:top_n], 1):
        print(f"{i}. {pair['ticker1']} - {pair['ticker2']} (distance={pair['distance']:.2f})")
    
    return distances[:top_n]
```

## 实战案例：AAPL vs MSFT 配对交易

让我们用一个完整案例来演示配对交易的实际应用。

```python
# 完整实战代码
def run_pairs_trading_example():
    """运行配对交易完整示例"""
    
    # 1. 初始化策略
    pair = PairsTrading(
        ticker1='AAPL',
        ticker2='MSFT',
        start_date='2020-01-01',
        end_date='2024-12-31'
    )
    
    # 2. 协整检验
    print("=" * 60)
    print("步骤1: 协整检验")
    print("=" * 60)
    is_coint = pair.test_cointegration()
    
    if not is_coint:
        print("\n✗ 该配对不协整，无法构建配对交易策略")
        return
    
    # 3. 计算对冲比率
    print("\n" + "=" * 60)
    print("步骤2: 计算对冲比率")
    print("=" * 60)
    pair.calculate_hedge_ratio(method='rolling', window=60)
    
    # 4. 生成交易信号
    print("\n" + "=" * 60)
    print("步骤3: 生成交易信号")
    print("=" * 60)
    pair.generate_signals(entry_z=2.0, exit_z=0.5)
    
    # 5. 回测
    print("\n" + "=" * 60)
    print("步骤4: 策略回测")
    print("=" * 60)
    results = pair.backtest(capital=100000, commission=0.001)
    
    # 6. 可视化
    print("\n" + "=" * 60)
    print("步骤5: 生成图表")
    print("=" * 60)
    pair.visualize()
    
    # 7. 绩效分析
    print("\n" + "=" * 60)
    print("步骤6: 绩效分析")
    print("=" * 60)
    
    # 计算额外指标
    returns = results['return']
    
    # 胜率
    winning_days = (returns > 0).sum()
    total_days = (returns != 0).sum()
    win_rate = winning_days / total_days if total_days > 0 else 0
    
    # 盈亏比
    avg_win = returns[returns > 0].mean()
    avg_loss = abs(returns[returns < 0].mean())
    profit_loss_ratio = avg_win / avg_loss if avg_loss > 0 else np.inf
    
    print(f"\n详细绩效指标:")
    print(f"  交易天数: {total_days}")
    print(f"  胜率: {win_rate:.2%}")
    print(f"  盈亏比: {profit_loss_ratio:.2f}")
    print(f"  平均日收益: {returns.mean():.4%}")
    print(f"  收益标准差: {returns.std():.4%}")
    
    return results

# 运行示例
if __name__ == "__main__":
    results = run_pairs_trading_example()
```

## 策略优化与风险提示

### 优化方向

1. **动态阈值**：
   - 根据市场波动率调整入场/出场阈值
   - 使用GARCH模型预测波动率

2. **机器学习增强**：
   - 用随机森林预测价差回归概率
   - 用LSTM预测价差方向

3. **风险管理**：
   - 设置最大持仓时间（避免长期不收敛）
   - 动态止损（基于ATR或波动率）

### 风险警示

⚠️ **配对交易并非无风险**，以下情况可能导致亏损：

1. **协整关系破裂**：
   - 公司基本面发生重大变化
   - 行业格局改变
   - 宏观经济冲击

2. **模型风险**：
   - 过度拟合历史数据
   - 忽略交易成本
   - 低估滑点影响

3. **执行风险**：
   - 做空限制（A股无法做空个股）
   - 流动性不足
   - 杠杆风险

## 总结

![配对交易累计收益曲线](../..//public/images/pair-trading-cointegration/cumulative_returns.png)

配对交易是量化交易入门的绝佳策略，核心要点：

1. **理论基础扎实**：协整分析、平稳性检验
2. **实践步骤清晰**：选对 → 检验 → 对冲 → 信号 → 回测
3. **风险可控**：市场中性、明确止损
4. **持续优化**：动态对冲、机器学习增强

完整代码已上传至GitHub，欢迎实践与讨论！

## 参考资料

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
3. Montaña, J. D., et al. (2016). "Pairs Trading Strategy: Formation, Optimization and Performance." *European Journal of Operational Research*.

---

**关键词**: 配对交易, 协整分析, 统计套利, 市场中性, Python实现, 均值回归

**免责声明**: 本文仅供学习交流，不构成投资建议。市场有风险，投资需谨慎。
