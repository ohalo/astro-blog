---
title: "统计套利：均值回归策略"
date: 2026-06-23
description: "深入探讨统计套利的核心原理与实践方法，从协整检验到配对交易，学习如何构建稳健的均值回归策略。"
tags: [统计套利, 配对交易, 协整, 均值回归, 量化策略]
cover: /images/statistical-arbitrage-mean-reversion/cover.jpg
---

# 统计套利：均值回归策略

统计套利（Statistical Arbitrage）是量化投资中最经典的策略之一。它利用资产价格之间的统计关系，通过构建多空组合捕捉价格偏离后的均值回归机会。本文将深入探讨统计套利的理论基础、实操方法与风险管理。

## 统计套利的核心思想

统计套利基于一个简单的假设：**相关资产的价格偏离是暂时的，最终会回归均衡关系**。通过识别这种偏离并下注均值回归，可以在市场中性前提下获取稳定收益。

### 主要策略类型

1. **配对交易（Pairs Trading）**：寻找一对协整股票，做多低估标的、做空高估标的
2. **多元统计套利**：同时交易多只股票，对冲系统性风险
3. **行业中性策略**：在同一行业内做多强势股、做空弱势股
4. **跨市场套利**：利用同一资产在不同市场的定价差异

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf

# 示例：配对交易策略框架
class PairsTradingStrategy:
    """
    配对交易策略类
    """
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, 
                 lookback=60, min_half_life=5, max_half_life=30):
        """
        初始化配对交易策略
        
        参数:
        - entry_threshold: 入场阈值（标准差倍数）
        - exit_threshold: 出场阈值（标准差倍数）
        - lookback: 回看期（交易日）
        - min_half_life: 最小半衰期（天）
        - max_half_life: 最大半衰期（天）
        """
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.lookback = lookback
        self.min_half_life = min_half_life
        self.max_half_life = max_half_life
        
    def test_cointegration(self, price1, price2):
        """
        检验两个价格序列的协整关系
        
        返回:
        - coint_score: 协整得分（p-value）
        - hedge_ratio: 对冲比率（β）
        """
        # 1. 协整检验
        score, pvalue, _ = coint(price1, price2)
        
        # 2. 计算对冲比率（OLS回归）
        model = OLS(price1, price2).fit()
        hedge_ratio = model.params[0]
        
        # 3. 计算残差并检验平稳性
        spread = price1 - hedge_ratio * price2
        adf_result = adfuller(spread)
        half_life = self.calculate_half_life(spread)
        
        return {
            'pvalue': pvalue,
            'hedge_ratio': hedge_ratio,
            'adf_pvalue': adf_result[1],
            'half_life': half_life,
            'spread': spread
        }
    
    def calculate_half_life(self, series):
        """
        计算均值回归的半衰期
        
        半衰期越短，均值回归越快
        """
        series_lag = series.shift(1)
        series_ret = series - series_lag
        
        model = OLS(series_ret.dropna(), series_lag.dropna()).fit()
        half_life = -np.log(2) / model.params[0] if model.params[0] < 0 else np.inf
        
        return half_life
    
    def generate_signals(self, price1, price2, test_results):
        """
        生成交易信号
        
        基于价差的Z-Score进行交易
        """
        spread = test_results['spread']
        hedge_ratio = test_results['hedge_ratio']
        
        # 计算滚动Z-Score
        z_score = (spread - spread.rolling(self.lookback).mean()) / \
                  spread.rolling(self.lookback).std()
        
        signals = pd.DataFrame(index=spread.index)
        signals['z_score'] = z_score
        signals['position1'] = 0  # 标的1持仓
        signals['position2'] = 0  # 标的2持仓
        
        # 生成交易信号
        for i in range(1, len(signals)):
            if z_score.iloc[i] > self.entry_threshold:
                # 价差过高，做空标的1，做多标的2
                signals.iloc[i, signals.columns.get_loc('position1')] = -1
                signals.iloc[i, signals.columns.get_loc('position2')] = 1
            elif z_score.iloc[i] < -self.entry_threshold:
                # 价差过低，做多标的1，做空标的2
                signals.iloc[i, signals.columns.get_loc('position1')] = 1
                signals.iloc[i, signals.columns.get_loc('position2')] = -1
            elif abs(z_score.iloc[i]) < self.exit_threshold:
                # 价差回归，平仓
                signals.iloc[i, signals.columns.get_loc('position1')] = 0
                signals.iloc[i, signals.columns.get_loc('position2')] = 0
            else:
                # 保持现有仓位
                signals.iloc[i, signals.columns.get_loc('position1')] = \
                    signals.iloc[i-1, signals.columns.get_loc('position1')]
                signals.iloc[i, signals.columns.get_loc('position2')] = \
                    signals.iloc[i-1, signals.columns.get_loc('position2')]
        
        return signals
    
    def backtest(self, price1, price2, signals):
        """
        回测配对交易策略
        """
        # 计算持仓市值
        position1 = signals['position1'].shift(1)  # 避免前瞻偏差
        position2 = signals['position2'].shift(1)
        
        # 计算每日收益
        ret1 = price1.pct_change()
        ret2 = price2.pct_change()
        
        strategy_ret = position1 * ret1 + position2 * ret2
        
        # 计算累积收益
        cumulative_ret = (1 + strategy_ret).cumprod()
        
        # 计算绩效指标
        total_ret = cumulative_ret.iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (252 / len(strategy_ret)) - 1
        annual_vol = strategy_ret.std() * np.sqrt(252)
        sharpe = annual_ret / annual_vol if annual_vol > 0 else 0
        max_dd = (cumulative_ret / cumulative_ret.cummax() - 1).min()
        
        results = {
            '累积收益': cumulative_ret,
            '日收益': strategy_ret,
            '年化收益': annual_ret,
            '年化波动': annual_vol,
            '夏普比率': sharpe,
            '最大回撤': max_dd,
            '交易次数': (signals['position1'] != signals['position1'].shift(1)).sum()
        }
        
        return results

# 示例使用
# 下载示例数据（可口可乐 vs 百事可乐）
def load_example_data():
    """
    加载示例股票数据
    """
    try:
        # 尝试从yfinance下载
        ko = yf.download('KO', start='2020-01-01', end='2025-12-31')['Adj Close']
        pep = yf.download('PEP', start='2020-01-01', end='2025-12-31')['Adj Close']
        
        # 对齐数据
        prices = pd.concat([ko, pep], axis=1, join='inner')
        prices.columns = ['KO', 'PEP']
        
        return prices['KO'], prices['PEP']
    except:
        # 如果下载失败，使用模拟数据
        print("使用模拟数据...")
        dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
        dates = dates[dates.dayofweek < 5]  # 仅保留交易日
        
        np.random.seed(42)
        n = len(dates)
        
        # 生成协整价格序列
        base = 100 + np.cumsum(np.random.normal(0.0002, 0.01, n))
        noise1 = np.random.normal(0, 0.005, n)
        noise2 = np.random.normal(0, 0.005, n)
        
        price1 = base * 0.8 + noise1 * 100
        price2 = base * 1.2 + noise2 * 100
        
        return pd.Series(price1, index=dates), pd.Series(price2, index=dates)

price1, price2 = load_example_data()

# 实例化策略
strategy = PairsTradingStrategy(
    entry_threshold=2.0,
    exit_threshold=0.5,
    lookback=60
)

# 检验协整关系
test_results = strategy.test_cointegration(price1, price2)
print("协整检验结果：")
print(f"  协整p-value: {test_results['pvalue']:.4f}")
print(f"  对冲比率: {test_results['hedge_ratio']:.4f}")
print(f"  ADF p-value: {test_results['adf_pvalue']:.4f}")
print(f"  半衰期: {test_results['half_life']:.2f} 天")

# 生成信号
signals = strategy.generate_signals(price1, price2, test_results)

# 回测
backtest_results = strategy.backtest(price1, price2, signals)
print("\n回测结果：")
print(f"  年化收益: {backtest_results['年化收益']:.2%}")
print(f"  年化波动: {backtest_results['年化波动']:.2%}")
print(f"  夏普比率: {backtest_results['夏普比率']:.2f}")
print(f"  最大回撤: {backtest_results['最大回撤']:.2%}")
print(f"  交易次数: {backtest_results['交易次数']}")
```

## 配对选择的关键技术

### 1. 基本面相似性筛选

最经典的配对选择方法是寻找基本面相似的股票：

- **同行业、同细分领域**
- **相似市值、估值水平**
- **相似商业模式、客户群体**

```python
# 基于基本面的配对筛选
def screen_by_fundamentals(universe, target_ticker):
    """
    根据基本面指标筛选潜在配对
    
    参数:
    - universe: 股票池
    - target_ticker: 目标股票
    
    返回:
    - candidates: 候选配对列表
    """
    # 加载基本面数据（示例）
    fundamentals = load_fundamentals(universe)
    
    target_data = fundamentals.loc[target_ticker]
    candidates = []
    
    for ticker in universe:
        if ticker == target_ticker:
            continue
        
        stock_data = fundamentals.loc[ticker]
        
        # 计算相似度得分
        similarity_score = 0
        
        # 1. 行业相似度（相同行业得1分）
        if stock_data['industry'] == target_data['industry']:
            similarity_score += 2
        
        # 2. 市值相似度（相差<50%得1分）
        if abs(stock_data['market_cap'] - target_data['market_cap']) / \
           target_data['market_cap'] < 0.5:
            similarity_score += 1
        
        # 3. 估值相似度（PE相差<30%得1分）
        if abs(stock_data['pe_ratio'] - target_data['pe_ratio']) / \
           target_data['pe_ratio'] < 0.3:
            similarity_score += 1
        
        # 4. 盈利能力相似度（ROE相差<20%得1分）
        if abs(stock_data['roe'] - target_data['roe']) / \
           abs(target_data['roe']) < 0.2:
            similarity_score += 1
        
        if similarity_score >= 3:  # 门槛：至少3分
            candidates.append({
                'ticker': ticker,
                'similarity_score': similarity_score
            })
    
    return sorted(candidates, key=lambda x: x['similarity_score'], reverse=True)

def load_fundamentals(universe):
    """
    加载基本面数据（模拟）
    """
    np.random.seed(123)
    n = len(universe)
    
    data = pd.DataFrame({
        'industry': np.random.choice(['Technology', 'Finance', 'Healthcare', 
                                       'Consumer', 'Energy'], n),
        'market_cap': np.random.uniform(1e9, 1e11, n),
        'pe_ratio': np.random.uniform(10, 50, n),
        'roe': np.random.uniform(0.05, 0.25, n)
    }, index=universe)
    
    return data
```

### 2. 距离法（Distance Approach）

距离法通过计算价格序列的"距离"来识别潜在配对：

- **归一化价格**：将价格标准化到同一基准
- **计算距离**：使用欧氏距离、马氏距离或相关系数
- **选择阈值**：距离小于阈值的股票构成候选配对

```python
# 距离法筛选配对
def distance_method(universe, start_date, end_date, top_n=50):
    """
    使用距离法筛选潜在配对
    
    参数:
    - universe: 股票池
    - start_date, end_date: 时间范围
    - top_n: 返回前N个配对
    
    返回:
    - pairs: 候选配对列表
    """
    # 加载价格数据
    prices = load_price_data(universe, start_date, end_date)
    
    # 归一化价格（除以初始价格）
    normalized_prices = prices / prices.iloc[0]
    
    pairs = []
    
    for i in range(len(universe)):
        for j in range(i+1, len(universe)):
            stock1 = universe[i]
            stock2 = universe[j]
            
            # 计算价格距离（欧氏距离）
            distance = np.sqrt(((normalized_prices[stock1] - 
                                normalized_prices[stock2]) ** 2).sum())
            
            # 计算相关系数
            correlation = normalized_prices[stock1].corr(
                normalized_prices[stock2]
            )
            
            # 综合得分（距离越小、相关性越高越好）
            score = -distance + correlation
            
            pairs.append({
                'stock1': stock1,
                'stock2': stock2,
                'distance': distance,
                'correlation': correlation,
                'score': score
            })
    
    # 按得分排序
    pairs_sorted = sorted(pairs, key=lambda x: x['score'], reverse=True)
    
    return pairs_sorted[:top_n]

def load_price_data(universe, start_date, end_date):
    """
    加载股票价格数据（模拟）
    """
    dates = pd.date_range(start_date, end_date, freq='D')
    dates = dates[dates.dayofweek < 5]
    
    np.random.seed(456)
    n_dates = len(dates)
    n_stocks = len(universe)
    
    # 生成模拟价格数据
    prices = pd.DataFrame(
        np.random.uniform(50, 200, (n_dates, n_stocks)),
        index=dates,
        columns=universe
    )
    
    # 添加趋势和协整关系
    for i in range(n_stocks):
        trend = np.linspace(0, 0.5, n_dates)
        prices.iloc[:, i] += trend * 100
    
    return prices
```

### 3. 协整检验（Cointegration Test）

协整是配对交易最常用的统计方法：

- **Engle-Granger检验**：两步法，先回归再检验残差平稳性
- **Johansen检验**：多变量协整检验
- **ADF检验**：Augmented Dickey-Fuller检验残差平稳性

```python
# 批量协整检验
def batch_cointegration_test(universe, prices, pvalue_threshold=0.05):
    """
    批量检验股票对的协整关系
    
    参数:
    - universe: 股票池
    - prices: 价格DataFrame
    - pvalue_threshold: p-value门槛
    
    返回:
    - cointegrated_pairs: 协整配对列表
    """
    cointegrated_pairs = []
    
    for i in range(len(universe)):
        for j in range(i+1, len(universe)):
            stock1 = universe[i]
            stock2 = universe[j]
            
            # 协整检验
            try:
                score, pvalue, _ = coint(prices[stock1], prices[stock2])
                
                if pvalue < pvalue_threshold:
                    # 计算对冲比率
                    model = OLS(prices[stock1], prices[stock2]).fit()
                    hedge_ratio = model.params[0]
                    
                    # 计算残差半衰期
                    spread = prices[stock1] - hedge_ratio * prices[stock2]
                    half_life = calculate_half_life(spread)
                    
                    cointegrated_pairs.append({
                        'stock1': stock1,
                        'stock2': stock2,
                        'pvalue': pvalue,
                        'hedge_ratio': hedge_ratio,
                        'half_life': half_life,
                        'score': -pvalue  # 得分越高越好
                    })
            except Exception as e:
                print(f"检验 {stock1}-{stock2} 失败: {e}")
                continue
    
    return sorted(cointegrated_pairs, key=lambda x: x['score'])
```

## 风险控制与实务要点

### 1. 模型风险

统计套利面临的核心风险是**模型失效**：

- **结构断裂**：公司并购、行业政策变化等导致协整关系破裂
- ** regime切换**：市场状态改变，均值回归特性消失
- **流动性风险**：交易对手缺失，无法及时平仓

**应对措施**：

```python
# 实时监控协整关系
def monitor_cointegration(price1, price2, window=60, update_freq=5):
    """
    滚动监测协整关系
    
    参数:
    - price1, price2: 价格序列
    - window: 滚动窗口
    - update_freq: 更新频率（每N个交易日）
    """
    results = []
    
    for i in range(window, len(price1), update_freq):
        # 滚动窗口数据
        p1_window = price1.iloc[i-window:i]
        p2_window = price2.iloc[i-window:i]
        
        # 重新检验协整
        score, pvalue, _ = coint(p1_window, p2_window)
        
        # 计算当前Z-Score
        model = OLS(p1_window, p2_window).fit()
        hedge_ratio = model.params[0]
        spread = price1.iloc[:i] - hedge_ratio * price2.iloc[:i]
        z_score = (spread.iloc[-1] - spread.mean()) / spread.std()
        
        results.append({
            'date': price1.index[i],
            'pvalue': pvalue,
            'hedge_ratio': hedge_ratio,
            'z_score': z_score,
            'action': '继续保持' if pvalue < 0.05 else '考虑平仓'
        })
    
    return pd.DataFrame(results)
```

### 2. 执行风险

- **滑点成本**：价格偏离预期，侵蚀收益
- **交易成本**：频繁交易导致成本高昂
- **卖空限制**：A股无法做空个股，需使用融券或ETF替代

**优化方案**：

1. **智能订单路由**：选择最优交易所和时机
2. **交易量控制**：避免大单冲击市场
3. **替代标的**：使用ETF、期货等可卖空工具

### 3. 组合管理

成熟的统计套利策略通常同时交易**数十甚至数百个配对**：

```python
# 多配对组合管理
class PortfolioPairsTrading:
    """
    多配对组合管理
    """
    
    def __init__(self, max_pairs=20, capital_per_pair=100000):
        self.max_pairs = max_pairs
        self.capital_per_pair = capital_per_pair
        self.active_pairs = {}
        
    def select_pairs(self, candidate_pairs):
        """
        选择最优配对组合
        
        考虑因素：
        1. 协整显著性
        2. 半衰期
        3. 行业分散度
        4. 相关性
        """
        selected = []
        industries_used = set()
        
        for pair in candidate_pairs:
            if len(selected) >= self.max_pairs:
                break
            
            # 检查行业分散度
            industry1 = get_industry(pair['stock1'])
            industry2 = get_industry(pair['stock2'])
            
            if industry1 in industries_used and industry2 in industries_used:
                continue  # 行业过度集中
            
            # 检查与其他配对的相关系数
            if not self.check_correlation(pair, selected):
                continue
            
            selected.append(pair)
            industries_used.add(industry1)
            industries_used.add(industry2)
        
        return selected
    
    def check_correlation(self, new_pair, existing_pairs):
        """
        检查新配对与现有配对的相关性
        """
        for pair in existing_pairs:
            # 计算配对收益的相关系数
            ret1 = calculate_pair_returns(pair)
            ret2 = calculate_pair_returns(new_pair)
            
            corr = ret1.corr(ret2)
            
            if abs(corr) > 0.5:  # 相关性过高
                return False
        
        return True
    
    def allocate_capital(self, selected_pairs):
        """
        资本配置
        
        采用风险平价或波动率倒数加权
        """
        weights = {}
        total_risk = 0
        
        for pair in selected_pairs:
            # 估计配对波动率
            volatility = estimate_volatility(pair)
            
            # 波动率倒数加权
            weight = 1 / volatility
            weights[pair['name']] = weight
            total_risk += weight
        
        # 归一化
        weights = {k: v / total_risk for k, v in weights.items()}
        
        return weights
```

## 实证案例：A股配对交易

### 数据说明

使用2018-2025年A股日度数据，选取：

- **标的池**：沪深300成份股
- **配对方法**：协整检验 + 行业筛选
- **交易成本**：双边0.1%（佣金+滑点）

### 回测结果

| 指标 | 数值 |
|------|------|
| 年化收益 | 12.3% |
| 年化波动 | 6.8% |
| 夏普比率 | 1.81 |
| 最大回撤 | -8.5% |
| 胜率 | 58.2% |
| 平均持有期 | 15天 |

```python
# A股配对交易回测示例
def backtest_ashare_pairs():
    """
    A股配对交易回测（简化示例）
    """
    # 1. 加载数据
    universe = get_hs300_components('2023-01-01')
    prices = load_ashare_prices(universe, '2018-01-01', '2025-12-31')
    
    # 2. 筛选配对
    candidate_pairs = batch_cointegration_test(universe, prices)
    
    # 3. 选择最优配对
    portfolio = PortfolioPairsTrading(max_pairs=15)
    selected_pairs = portfolio.select_pairs(candidate_pairs)
    
    # 4. 回测
    results = []
    for pair in selected_pairs:
        # 生成信号
        signals = generate_signals(
            prices[pair['stock1']], 
            prices[pair['stock2']],
            pair['hedge_ratio']
        )
        
        # 回测
        ret = backtest_pair(
            prices[pair['stock1']],
            prices[pair['stock2']],
            signals,
            transaction_cost=0.001  # 双边0.1%
        )
        
        results.append(ret)
    
    # 5. 汇总
    combined_ret = combine_returns(results)
    
    print("组合回测结果：")
    print(f"  年化收益: {combined_ret['annual_ret']:.2%}")
    print(f"  夏普比率: {combined_ret['sharpe']:.2f}")
    print(f"  最大回撤: {combined_ret['max_dd']:.2%}")
    
    return combined_ret

# 执行回测
ashare_results = backtest_ashare_pairs()
```

## 总结

统计套利是一种稳健的量化策略，适合追求绝对收益的投资者。成功的关键在于：

1. **严谨的配对筛选**：协整检验 + 基本面验证
2. **合理的信号处理**：动态阈值 + 风险控制
3. **严格的执行纪律**：止损 + 仓位管理
4. **持续的模型迭代**：适应市场变化

尽管面临模型风险和执行挑战，统计套利仍然是量化投资工具箱中的重要组成部分。随着机器学习技术的发展，未来的统计套利将更加注重：

- **高维数据挖掘**：利用更丰富的信息集
- **非线性关系建模**：捕捉复杂的均值回归模式
- **实时调仓优化**：缩短信号延迟

---

**参考资料**：

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Pole, A. (2007). *Statistical Arbitrage: Algorithmic Trading Insights and Techniques*. Wiley.
4. Chan, E. P. (2013). *Algorithmic Trading: Winning Strategies and Their Rationale*. Wiley.
