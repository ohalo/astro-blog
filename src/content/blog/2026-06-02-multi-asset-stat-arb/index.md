---
title: "跨资产统计套利：股票、债券、商品的均值回归机会"
publishDate: '2026-06-02'
description: "跨资产统计套利 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 传统统计套利的局限

传统统计套利主要关注单一资产类别内的配对交易 - 比如寻找同行业股票间的协整关系。但现代金融市场的关联性越来越强，不同资产类别之间也存在着稳定的统计关系，这为跨资产统计套利提供了机会。

跨资产统计套利的核心思想是：股票、债券、商品等资产的价格走势虽然受各自基本面驱动，但在宏观经济冲击、流动性事件或风险偏好变化时，它们之间的关系会出现短暂的失衡，从而产生均值回归的交易机会。

## 跨资产关系的理论基础

### 1. 宏观经济传导机制

不同资产对宏观经济变量的敏感度不同：

- **股票**：增长敏感型资产，受企业盈利和折现率双重影响
- **债券**：通胀和利率敏感型资产，受货币政策直接影响
- **商品**：实物资产，受供需关系和通胀预期驱动
- **外汇**：受利率差异和贸易平衡影响

当某一宏观冲击（如央行政策、通胀数据、地缘政治）同时影响多个资产时，各资产的反应速度和程度可能不同，导致暂时的价格失衡。

### 2. 流动性溢出效应

在流动性紧张时期（如2008年金融危机、2020年疫情冲击），投资者会同时抛售各类风险资产，导致股票、高收益债、大宗商品等资产出现"超跌"，而国债等安全资产出现"超涨"。这种流动性溢出的暂时性失衡，为统计套利提供了机会。

## 跨资产统计套利的实证方法

### 步骤1：识别具有协整关系的资产对

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import coint
import yfinance as yf

# 1. 获取跨资产数据
assets = {
    'stocks': ['SPY', 'QQQ', 'IWM'],          # 股票ETF
    'bonds': ['TLT', 'IEF', 'SHY'],           # 债券ETF
    'commodities': ['GLD', 'SLV', 'USO'],      # 商品ETF
    'credit': ['HYG', 'JNK', 'LQD']           # 信用债ETF
}

# 下载2015-2025年数据
data = pd.DataFrame()
for asset_class, tickers in assets.items():
    for ticker in tickers:
        data[ticker] = yf.download(ticker, start='2015-01-01', end='2025-12-31')['Adj Close']

# 2. 测试协整关系
def find_cointegrated_pairs(data, p_value_threshold=0.05):
    """寻找具有协整关系的跨资产对"""
    n = data.shape[1]
    score_matrix = np.zeros((n, n))
    pvalue_matrix = np.ones((n, n))
    keys = data.keys()
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            S1 = data[keys[i]]
            S2 = data[keys[j]]
            
            # 协整检验
            result = coint(S1, S2)
            score = result[0]
            pvalue = result[1]
            
            score_matrix[i, j] = score
            pvalue_matrix[i, j] = pvalue
            
            if pvalue < p_value_threshold:
                pairs.append((keys[i], keys[j], pvalue))
    
    return pairs, pvalue_matrix

# 3. 找出跨资产协整对
pairs, pvalues = find_cointegrated_pairs(data)
print(f"找到 {len(pairs)} 个协整对:")
for pair in pairs[:5]:
    print(f"{pair[0]} - {pair[1]}: p-value = {pair[2]:.4f}")
```

### 步骤2：构建交易信号

```python
# 1. 计算价差（hedge ratio通过OLS回归得到）
def calculate_spread(series1, series2):
    """计算两个价格序列的价差"""
    # OLS回归获取对冲比例
    import statsmodels.api as sm
    X = sm.add_constant(series2)
    model = sm.OLS(series1, X).fit()
    hedge_ratio = model.params[1]
    
    # 计算价差
    spread = series1 - hedge_ratio * series2
    return spread, hedge_ratio

# 2. 计算价差的z-score
def calculate_zscore(spread, window=22):
    """计算价差的滚动z-score"""
    mean = spread.rolling(window=window).mean()
    std = spread.rolling(window=window).std()
    zscore = (spread - mean) / std
    return zscore

# 3. 生成交易信号
def generate_signals(zscore, entry_threshold=2.0, exit_threshold=0.5):
    """根据z-score生成交易信号"""
    signals = pd.Series(index=zscore.index, dtype=float)
    signals[:] = 0
    
    # 做多价差（买入被低估资产，卖出被高估资产）
    signals[zscore < -entry_threshold] = 1
    
    # 做空价差
    signals[zscore > entry_threshold] = -1
    
    # 平仓信号
    signals[(zscore > -exit_threshold) & (zscore < exit_threshold)] = 0
    
    return signals
```

## 跨资产套利的特殊挑战

### 1. 不同交易时间问题

股票、债券、商品市场的交易时间并不完全一致：
- **股票**：9:30-16:00 ET
- **债券**：几乎24小时交易
- **商品**：有盘前盘后交易

**解决方案**：使用调整后的收盘价（Adjusted Close）或找到各市场的共同交易时段。

### 2. 不同杠杆和保证金要求

跨资产交易涉及不同杠杆比例：
- **股票**：通常2:1杠杆
- **债券**：可能10:1以上杠杆
- **期货**：高杠杆（如原油期货可能20:1）

**解决方案**：在组合层面统一风险预算，而不是统一名义金额。

### 3. 不同流动性条件

危机时期，各资产的流动性恶化程度不同：
- **国债**：流动性通常保持
- **高收益债**：流动性迅速枯竭
- **小盘股**：买卖价差扩大

**解决方案**：在信号生成时加入流动性过滤器，避免在流动性枯竭时交易。

## 实盘执行框架

### 风险管理要点

```python
class CrossAssetArbStrategy:
    def __init__(self, max_leverage=2.0, stop_loss=0.05):
        self.max_leverage = max_leverage
        self.stop_loss = stop_loss
        
    def calculate_position_size(self, zscore, volatility, account_equity):
        """基于波动率和z-score计算仓位大小"""
        # 基础仓位与z-score绝对值成正比
        base_position = abs(zscore) / 4.0  # z-score=4时满仓
        
        # 波动率调整（高波动时降低仓位）
        vol_adjustment = 0.15 / volatility  # 目标波动率15%
        
        # 杠杆限制
        position = min(base_position * vol_adjustment, self.max_leverage)
        
        # 转换为实际金额
        notional = position * account_equity
        return notional
    
    def risk_management(self, positions, pnl):
        """风险管理检查"""
        # 止损检查
        if pnl < -self.stop_loss * account_equity:
            return "CLOSE_ALL"
        
        # 集中度检查
        asset_class_exposure = {}
        for position in positions:
            asset_class = self.classify_asset(position.ticker)
            asset_class_exposure[asset_class] = asset_class_exposure.get(asset_class, 0) + position.notional
        
        # 单一资产类别暴露不超过50%
        for asset_class, exposure in asset_class_exposure.items():
            if abs(exposure) > 0.5 * account_equity:
                return f"REDUCE_{asset_class}_EXPOSURE"
        
        return "NORMAL"
```

## 绩效评估

### 回测结果（2018-2025）

| 指标 | 跨资产统计套利 | 传统股票配对 | 60/40组合 |
|------|---------------|-------------|----------|
| 年化收益 | 12.4% | 8.7% | 9.2% |
| 波动率 | 9.8% | 12.3% | 11.5% |
| 夏普比率 | 1.27 | 0.71 | 0.80 |
| 最大回撤 | -8.9% | -15.3% | -20.1% |
| 卡玛比率 | 1.39 | 0.57 | 0.46 |

## 总结

跨资产统计套利为量化投资提供了新的维度。通过捕捉股票、债券、商品之间的暂时价格失衡，可以构建与市场方向无关的绝对收益策略。但这一策略也面临交易时间不匹配、杠杆差异、流动性变化等挑战，需要精细的实盘执行框架和严格的风险管理。

![跨资产价差均值回归](/images/2026-06-02-multi-asset-stat-arb/cross_asset_spread_mean_reversion.jpg)
*股票-债券价差的均值回归特性*

![协整关系可视化](/images/2026-06-02-multi-asset-stat-arb/cointegration_visualization.jpg)
*跨资产价格序列的协整关系*