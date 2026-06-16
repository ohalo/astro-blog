---
title: "配对交易与协整分析：市场中性策略的理论与实践"
date: 2026-06-17
description: "深入讲解配对交易的核心原理——协整关系，以及如何用Python实现完整的配对交易策略，包括股票对选取、信号生成、风险管理和绩效评估。"
tags: ["配对交易", "协整分析", "市场中性", "统计套利", "量化策略"]
image: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：市场中性策略的理论与实践

## 引言

在传统量化策略中，我们往往面临一个难题：**如何在不预测市场方向的情况下获取稳定收益？**

配对交易（Pairs Trading）给出了一个优雅的答案。作为一种经典的市场中性策略，它不依赖市场涨跌，而是通过捕捉两个高度相关资产之间的暂时性偏离来获利。

本文将系统讲解：
1. 配对交易的核心原理与经济学逻辑
2. 协整关系的数学基础与检验方法
3. 股票对筛选的量化标准
4. 完整的Python实现（从数据获取到回测）
5. 实战中的陷阱与应对之道

## 一、配对交易的核心原理

### 1.1 什么是配对交易？

配对交易基于一个简单的观察：**某些资产的价格走势具有长期的均衡关系，短期内可能偏离，但终将回归。**

**策略逻辑：**
```
1. 找到两只价格走势高度相关的股票（如可口可乐 vs 百事可乐）
2. 计算它们的价差（Spread）或比率（Ratio）
3. 当价差偏离历史均值时：
   - 做空高估的 + 做多低估的
4. 当价差回归均值时平仓，赚取回归收益
```

**关键特性：**
- ✅ **市场中性**：多空对冲，不受大盘涨跌影响
- ✅ **均值回归**：利用价格的统计规律，而非方向预测
- ✅ **风险可控**：止损清晰（协整关系破裂）

### 1.2 为什么协整关系很重要？

很多初学者会混淆**相关性（Correlation）**和**协整性（Cointegration）**，但两者有本质区别：

| 维度 | 相关性 | 协整性 |
|------|--------|---------|
| 定义 | 短期联动程度 | 长期均衡关系 |
| 时间维度 | 任意时间段 | 必须长期 |
| 经济含义 | 可能纯属巧合 | 有共同的基本面驱动 |
| 交易含义 | 不适合配对交易 | 配对交易的基础 |

**经典反例：**
> 两只股票的日收益率相关系数达0.9，但它们的价格序列可能是随机游走，没有任何长期均衡关系。这样的"高相关"对无法进行配对交易。

**协整的经济学含义：**
如果股票A和B具有协整关系，意味着它们受到共同的因子驱动（如同行业、同产业链、替代关系等），短期偏离只是噪音，长期必然回归。

## 二、协整关系的数学基础

### 2.1 数学定义

对于两个非平稳时间序列 $\{X_t\}$ 和 $\{Y_t\}$，如果存在一个线性组合：

$$
Z_t = Y_t - \beta X_t
$$

使得 $Z_t$ 是**平稳序列**（Stationary），则称 $X_t$ 和 $Y_t$ 具有协整关系。

其中：
- $\beta$ 称为**协整系数**（Cointegrating Coefficient）
- $Z_t$ 称为**残差序列**或**价差序列**

### 2.2 平稳性的判定

一个时间序列是平稳的，当且仅当：
1. 均值恒定
2. 方差恒定
3. 自协方差只依赖于时滞

**常用检验方法：**
- **ADF检验（Augmented Dickey-Fuller Test）**：检验是否存在单位根
- **KPSS检验**：检验是否趋势平稳
- **PP检验（Phillips-Perron Test）**：对异方差稳健的ADF变体

### 2.3 Python实现：协整检验

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint, adfuller
import yfinance as yf
import matplotlib.pyplot as plt

class CointegrationAnalyzer:
    """协整关系分析器"""
    
    def __init__(self, stock_a, stock_b, start_date, end_date):
        """
        初始化
        
        Parameters:
        -----------
        stock_a, stock_b : str, 股票代码
        start_date, end_date : str, 日期范围
        """
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.start_date = start_date
        self.end_date = end_date
        self.price_a = None
        self.price_b = None
        self.spread = None
        self.hedge_ratio = None
        
    def load_data(self):
        """加载股票价格数据"""
        # 使用yfinance下载数据（实际中可用akshare/tushare）
        data_a = yf.download(self.stock_a, start=self.start_date, end=self.end_date, progress=False)
        data_b = yf.download(self.stock_b, start=self.start_date, end=self.end_date, progress=False)
        
        # 使用收盘价
        self.price_a = data_a['Adj Close']
        self.price_b = data_b['Adj Close']
        
        # 对齐日期
        df = pd.DataFrame({
            self.stock_a: self.price_a,
            self.stock_b: self.price_b
        }).dropna()
        
        self.price_a = df[self.stock_a]
        self.price_b = df[self.stock_b]
        
        print(f"✅ 数据加载成功：{len(df)} 个交易日")
        return df
    
    def calculate_hedge_ratio(self, method='OLS'):
        """计算对冲比率（协整系数）"""
        if method == 'OLS':
            # 方法1：普通最小二乘法
            X = sm.add_constant(self.price_a)
            model = sm.OLS(self.price_b, X).fit()
            self.hedge_ratio = model.params[self.stock_a]
            self.intercept = model.params['const']
            
        elif method == 'TLS':
            # 方法2：总体最小二乘法（考虑X和Y的误差）
            # 使用numpy的SVD实现
            X = self.price_a.values
            Y = self.price_b.values
            A = np.column_stack([X, -Y])
            U, S, Vt = np.linalg.svd(A, full_matrices=False)
            self.hedge_ratio = Vt[1, 0] / Vt[1, 1]
            self.intercept = 0
            
        print(f"对冲比率 (β): {self.hedge_ratio:.4f}")
        return self.hedge_ratio
    
    def calculate_spread(self):
        """计算价差序列"""
        if self.hedge_ratio is None:
            self.calculate_hedge_ratio()
        
        # Z_t = Y_t - β * X_t
        self.spread = self.price_b - self.hedge_ratio * self.price_a
        
        # 标准化价差（Z-score）
        self.spread_mean = self.spread.mean()
        self.spread_std = self.spread.std()
        self.spread_zscore = (self.spread - self.spread_mean) / self.spread_std
        
        return self.spread, self.spread_zscore
    
    def test_cointegration(self, significance_level=0.05):
        """
        协整检验（Engle-Granger两步法）
        
        Returns:
        --------
        is_cointegrated : bool, 是否协整
        p_value : float, 协整检验p值
        """
        # 步骤1：ADF检验确认两个序列都是非平稳的（I(1)）
        adf_result_a = adfuller(self.price_a, autolag='AIC')
        adf_result_b = adfuller(self.price_b, autolag='AIC')
        
        print(f"\n=== ADF检验（确认非平稳）===")
        print(f"{self.stock_a}: ADF统计量={adf_result_a[0]:.4f}, p值={adf_result_a[1]:.4f}")
        print(f"{self.stock_b}: ADF统计量={adf_result_b[0]:.4f}, p值={adf_result_b[1]:.4f}")
        
        # 如果p值 < 0.05，说明是平稳的，不适合协整检验
        if adf_result_a[1] < significance_level or adf_result_b[1] < significance_level:
            print("⚠️ 警告：至少一个序列已经是平稳的，不需要协整检验")
        
        # 步骤2：计算残差并检验平稳性
        if self.spread is None:
            self.calculate_spread()
        
        adf_result_spread = adfuller(self.spread, autolag='AIC')
        
        print(f"\n=== 残差序列的ADF检验 ===")
        print(f"ADF统计量: {adf_result_spread[0]:.4f}")
        print(f"p值: {adf_result_spread[1]:.4f}")
        print(f"1% 临界值: {adf_result_spread[4]['1%']:.4f}")
        print(f"5% 临界值: {adf_result_spread[4]['5%']:.4f}")
        print(f"10% 临界值: {adf_result_spread[4]['10%']:.4f}")
        
        # 判断协整性
        is_cointegrated = adf_result_spread[1] < significance_level
        
        if is_cointegrated:
            print(f"\n✅ {self.stock_a} 和 {self.stock_b} 存在协整关系 (p={adf_result_spread[1]:.4f})")
        else:
            print(f"\n❌ {self.stock_a} 和 {self.stock_b} 不存在协整关系 (p={adf_result_spread[1]:.4f})")
        
        return is_cointegrated, adf_result_spread[1]
    
    def visualize_pairs(self, figsize=(15, 10)):
        """可视化配对分析"""
        if self.spread is None:
            self.calculate_spread()
        
        fig, axes = plt.subplots(3, 1, figsize=figsize)
        fig.suptitle(f'{self.stock_a} vs {self.stock_b} - 配对交易分析', fontsize=16, fontweight='bold')
        
        # 图1：价格序列
        ax1 = axes[0]
        ax1.plot(self.price_a.index, self.price_a.values, label=self.stock_a, linewidth=2)
        ax1.plot(self.price_b.index, self.price_b.values, label=self.stock_b, linewidth=2)
        ax1.set_title('价格序列', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 图2：价差序列
        ax2 = axes[1]
        ax2.plot(self.spread.index, self.spread.values, color='blue', linewidth=2)
        ax2.axhline(y=self.spread_mean, color='red', linestyle='--', 
                     label=f'均值 ({self.spread_mean:.2f})')
        ax2.fill_between(self.spread.index, 
                         self.spread_mean - 2*self.spread_std,
                         self.spread_mean + 2*self.spread_std,
                         alpha=0.2, color='gray', label='±2σ')
        ax2.set_title('价差序列 (Spread)', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        
        # 图3：Z-score
        ax3 = axes[2]
        ax3.plot(self.spread_zscore.index, self.spread_zscore.values, 
                  color='purple', linewidth=2)
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax3.axhline(y=2, color='red', linestyle='--', linewidth=2, label='+2σ (做空信号)')
        ax3.axhline(y=-2, color='green', linestyle='--', linewidth=2, label='-2σ (做多信号)')
        ax3.fill_between(self.spread_zscore.index, -2, 2, alpha=0.2, color='gray')
        ax3.set_title('Z-Score (标准化价差)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Z-Score')
        ax3.legend(fontsize=11)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/analysis.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 配对分析图已保存")

# 使用示例
# analyzer = CointegrationAnalyzer('KO', 'PEP', '2020-01-01', '2026-06-17')
# df = analyzer.load_data()
# analyzer.calculate_hedge_ratio()
# analyzer.calculate_spread()
# is_coint, p_value = analyzer.test_cointegration()
# analyzer.visualize_pairs()
```

## 三、股票对筛选的量化标准

### 3.1 筛选流程

构建一个系统化的股票对筛选框架：

```python
import itertools
from tqdm import tqdm

class PairsSelector:
    """股票对筛选器"""
    
    def __init__(self, stock_universe, start_date, end_date):
        """
        初始化
        
        Parameters:
        -----------
        stock_universe : list, 股票代码列表
        start_date, end_date : str, 日期范围
        """
        self.stock_universe = stock_universe
        self.start_date = start_date
        self.end_date = end_date
        self.price_data = {}
        self.candidates = []
        
    def load_all_data(self):
        """批量加载所有股票的价格数据"""
        print(f"正在加载 {len(self.stock_universe)} 只股票的数据...")
        
        for ticker in tqdm(self.stock_universe):
            try:
                data = yf.download(ticker, start=self.start_date, 
                                   end=self.end_date, progress=False)
                if len(data) > 0:
                    self.price_data[ticker] = data['Adj Close']
            except Exception as e:
                print(f"⚠️ 加载 {ticker} 失败: {e}")
        
        print(f"✅ 成功加载 {len(self.price_data)} 只股票")
        
    def calculate_correlation(self, stock_a, stock_b, method='pearson'):
        """计算相关性（初筛）"""
        price_a = self.price_data.get(stock_a)
        price_b = self.price_data.get(stock_b)
        
        if price_a is None or price_b is None:
            return None
        
        # 对齐日期
        df = pd.DataFrame({
            stock_a: price_a,
            stock_b: price_b
        }).dropna()
        
        if len(df) < 252:  # 至少1年数据
            return None
        
        if method == 'pearson':
            corr = df[stock_a].corr(df[stock_b])
        elif method == 'spearman':
            corr = df[stock_a].corr(df[stock_b], method='spearman')
        
        return corr
    
    def test_cointegration_pair(self, stock_a, stock_b):
        """对单个股票对进行协整检验"""
        price_a = self.price_data.get(stock_a)
        price_b = self.price_data.get(stock_b)
        
        if price_a is None or price_b is None:
            return None, None
        
        # 对齐日期
        df = pd.DataFrame({
            stock_a: price_a,
            stock_b: price_b
        }).dropna()
        
        # 计算对冲比率
        X = sm.add_constant(df[stock_a])
        model = sm.OLS(df[stock_b], X).fit()
        hedge_ratio = model.params[stock_a]
        
        # 计算价差
        spread = df[stock_b] - hedge_ratio * df[stock_a]
        
        # ADF检验
        adf_result = adfuller(spread, autolag='AIC')
        p_value = adf_result[1]
        
        return p_value, hedge_ratio
    
    def screen_pairs(self, corr_threshold=0.7, p_threshold=0.05):
        """
        批量筛选股票对
        
        Parameters:
        -----------
        corr_threshold : float, 相关性阈值
        p_threshold : float, 协整检验p值阈值
        """
        self.load_all_data()
        
        print(f"\n开始筛选：相关性>{corr_threshold}, p值<{p_threshold}")
        print("="*60)
        
        # 生成所有组合
        all_pairs = list(itertools.combinations(self.price_data.keys(), 2))
        print(f"共 {len(all_pairs)} 个候选对\n")
        
        # 第一步：相关性初筛
        print("第一步：相关性初筛...")
        corr_passed = []
        
        for stock_a, stock_b in tqdm(all_pairs):
            corr = self.calculate_correlation(stock_a, stock_b)
            if corr is not None and corr > corr_threshold:
                corr_passed.append((stock_a, stock_b, corr))
        
        print(f"✅ 相关性筛选通过：{len(corr_passed)} 对\n")
        
        # 第二步：协整检验
        print("第二步：协整检验...")
        for stock_a, stock_b, corr in tqdm(corr_passed):
            p_value, hedge_ratio = self.test_cointegration_pair(stock_a, stock_b)
            
            if p_value is not None and p_value < p_threshold:
                self.candidates.append({
                    'stock_a': stock_a,
                    'stock_b': stock_b,
                    'correlation': corr,
                    'p_value': p_value,
                    'hedge_ratio': hedge_ratio
                })
        
        print(f"\n✅ 协整检验通过：{len(self.candidates)} 对")
        print("="*60)
        
        # 按p值排序
        self.candidates = sorted(self.candidates, key=lambda x: x['p_value'])
        
        return self.candidates
    
    def visualize_top_pairs(self, top_n=5):
        """可视化排名前N的股票对"""
        if len(self.candidates) == 0:
            print("⚠️ 没有候选对，请先运行 screen_pairs()")
            return
        
        print(f"\nTOP {top_n} 候选股票对：")
        print("="*60)
        for i, candidate in enumerate(self.candidates[:top_n]):
            print(f"{i+1}. {candidate['stock_a']} - {candidate['stock_b']}")
            print(f"   相关性: {candidate['correlation']:.4f}")
            print(f"   p值: {candidate['p_value']:.6f}")
            print(f"   对冲比率: {candidate['hedge_ratio']:.4f}")
            print()

# 使用示例
# selector = PairsSelector(['KO', 'PEP', 'MSFT', 'AAPL', 'GOOGL', 'META'], 
#                          '2020-01-01', '2026-06-17')
# candidates = selector.screen_pairs(corr_threshold=0.7, p_threshold=0.05)
# selector.visualize_top_pairs(top_n=5)
```

### 3.2 筛选标准总结

| 指标 | 阈值建议 | 说明 |
|------|----------|------|
| 相关性 | > 0.7 | 初筛，排除明显不相关的对 |
| 协整检验p值 | < 0.05 | 核心标准，必须协整 |
| 数据长度 | ≥ 252天 | 确保统计显著性 |
| 对冲比率稳定性 | 滚动窗口检验 | 避免时变对冲比率 |
| 价差均值回归速度 | 半衰期 < 30天 | 确保交易频率合理 |

## 四、完整的配对交易策略实现

### 4.1 策略逻辑

```python
class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, stock_a, stock_b, hedge_ratio, entry_z=2.0, exit_z=0.5, 
                 stop_loss_z=3.0, lookback=63):
        """
        初始化
        
        Parameters:
        -----------
        stock_a, stock_b : str, 股票代码
        hedge_ratio : float, 对冲比率
        entry_z : float, 入场Z-score阈值
        exit_z : float, 出场Z-score阈值
        stop_loss_z : float, 止损Z-score阈值
        lookback : int, 计算均值和标准差的滚动窗口
        """
        self.stock_a = stock_a
        self.stock_b = stock_b
        self.hedge_ratio = hedge_ratio
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.lookback = lookback
        
        self.price_a = None
        self.price_b = None
        self.spread = None
        self.spread_mean = None
        self.spread_std = None
        self.zscore = None
        
        self.position = 0  # 0: 无仓位, 1: 做多价差, -1: 做空价差
        self.entry_price_a = None
        self.entry_price_b = None
        self.trades = []
        
    def load_data(self, start_date, end_date):
        """加载数据"""
        data_a = yf.download(self.stock_a, start=start_date, end=end_date, progress=False)
        data_b = yf.download(self.stock_b, start=start_date, end=end_date, progress=False)
        
        self.price_a = data_a['Adj Close']
        self.price_b = data_b['Adj Close']
        
        # 对齐日期
        df = pd.DataFrame({
            self.stock_a: self.price_a,
            self.stock_b: self.price_b
        }).dropna()
        
        self.price_a = df[self.stock_a]
        self.price_b = df[self.stock_b]
        
        # 计算价差和Z-score
        self.spread = self.price_b - self.hedge_ratio * self.price_a
        
        # 滚动均值和标准差
        self.spread_mean = self.spread.rolling(window=self.lookback).mean()
        self.spread_std = self.spread.rolling(window=self.lookback).std()
        
        self.zscore = (self.spread - self.spread_mean) / self.spread_std
        
    def generate_signals(self):
        """生成交易信号"""
        signals = pd.DataFrame(index=self.zscore.index)
        signals['zscore'] = self.zscore
        signals['position'] = 0
        
        # 信号逻辑
        for i in range(1, len(signals)):
            z = signals['zscore'].iloc[i]
            z_prev = signals['zscore'].iloc[i-1]
            
            # 当前无仓位
            if self.position == 0:
                # Z-score < -entry_z：价差被低估，做多价差（做多B + 做空A）
                if z < -self.entry_z:
                    self.position = 1
                    signals['position'].iloc[i] = 1
                    
                # Z-score > entry_z：价差被高估，做空价差（做空B + 做多A）
                elif z > self.entry_z:
                    self.position = -1
                    signals['position'].iloc[i] = -1
                    
            # 当前持有仓位
            else:
                # 平仓信号：Z-score回归到exit_z以内
                if abs(z) < self.exit_z:
                    signals['position'].iloc[i] = 0  # 平仓
                    self.position = 0
                    
                # 止损信号：Z-score超过stop_loss_z
                elif abs(z) > self.stop_loss_z:
                    signals['position'].iloc[i] = 0  # 止损
                    self.position = 0
                    print(f"⚠️ 止损触发 @ {signals.index[i]}")
                    
                else:
                    # 保持仓位
                    signals['position'].iloc[i] = self.position
        
        return signals
    
    def backtest(self, commission=0.001):
        """
        回测
        
        Parameters:
        -----------
        commission : float, 手续费率（单边）
        """
        signals = self.generate_signals()
        
        # 计算收益
        returns_a = self.price_a.pct_change()
        returns_b = self.price_b.pct_change()
        
        # 策略收益
        strategy_returns = pd.Series(0, index=signals.index)
        
        # 持仓收益
        position_lagged = signals['position'].shift(1)  # 使用滞后仓位（避免未来函数）
        
        # 做多价差：做多B + 做空A
        # 做空价差：做空B + 做多A
        strategy_returns = position_lagged * (returns_b - self.hedge_ratio * returns_a)
        
        # 扣除手续费（每次换仓）
        turnover = position_lagged.diff().abs()
        strategy_returns -= turnover * commission * 2  # 双边手续费
        
        # 累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算绩效指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        sharpe_ratio = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        # 统计交易次数
        num_trades = (signals['position'] != signals['position'].shift(1)).sum() // 2
        
        results = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': num_trades,
            'strategy_returns': strategy_returns,
            'cumulative_returns': cumulative_returns,
            'signals': signals
        }
        
        return results
    
    def visualize_backtest(self, results, figsize=(15, 12)):
        """可视化回测结果"""
        fig, axes = plt.subplots(4, 1, figsize=figsize)
        fig.suptitle(f'{self.stock_a} - {self.stock_b} 配对交易回测', 
                     fontsize=16, fontweight='bold')
        
        # 图1：累计收益
        ax1 = axes[0]
        ax1.plot(results['cumulative_returns'].index, 
                  results['cumulative_returns'].values, 
                  linewidth=2, color='blue')
        ax1.fill_between(results['cumulative_returns'].index, 
                          1, results['cumulative_returns'].values,
                          alpha=0.3, color='blue')
        ax1.set_title('累计收益', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Cumulative Returns')
        ax1.grid(True, alpha=0.3)
        
        # 图2：Z-score + 交易信号
        ax2 = axes[1]
        ax2.plot(self.zscore.index, self.zscore.values, color='purple', linewidth=2)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax2.axhline(y=self.entry_z, color='red', linestyle='--', linewidth=2, label='入场阈值')
        ax2.axhline(y=-self.entry_z, color='green', linestyle='--', linewidth=2)
        ax2.axhline(y=self.exit_z, color='gray', linestyle=':', linewidth=1.5, label='出场阈值')
        ax2.axhline(y=-self.exit_z, color='gray', linestyle=':', linewidth=1.5)
        
        # 标注交易信号
        signals = results['signals']
        entry_signals = signals[(signals['position'] != 0) & 
                                (signals['position'].shift(1) == 0)]
        exit_signals = signals[(signals['position'] == 0) & 
                               (signals['position'].shift(1) != 0)]
        
        ax2.scatter(entry_signals.index, self.zscore.loc[entry_signals.index],
                    color='orange', s=100, marker='^', label='入场', zorder=5)
        ax2.scatter(exit_signals.index, self.zscore.loc[exit_signals.index],
                    color='cyan', s=100, marker='v', label='出场', zorder=5)
        
        ax2.set_title('Z-Score & 交易信号', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Z-Score')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 图3：价差序列
        ax3 = axes[2]
        ax3.plot(self.spread.index, self.spread.values, color='blue', linewidth=2)
        ax3.plot(self.spread_mean.index, self.spread_mean.values, 
                  color='red', linestyle='--', linewidth=2, label='滚动均值')
        ax3.fill_between(self.spread.index,
                         self.spread_mean - 2*self.spread_std,
                         self.spread_mean + 2*self.spread_std,
                         alpha=0.2, color='gray')
        ax3.set_title('价差序列', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Spread')
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # 图4：滚动夏普比率
        ax4 = axes[3]
        rolling_sharpe = strategy_returns.rolling(63).mean() / strategy_returns.rolling(63).std() * np.sqrt(252)
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values, color='green', linewidth=2)
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
        ax4.axhline(y=1, color='gray', linestyle='--', linewidth=1.5, label='Sharpe=1')
        ax4.set_title('滚动夏普比率 (63天)', fontsize=14, fontweight='bold')
        ax4.set_ylabel('Rolling Sharpe')
        ax4.set_xlabel('Date')
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest.png', 
                    dpi=300, bbox_inches='tight')
        plt.close()
        print("✅ 回测结果图已保存")

# 完整使用示例
# strategy = PairsTradingStrategy('KO', 'PEP', hedge_ratio=0.75, 
#                                  entry_z=2.0, exit_z=0.5, stop_loss_z=3.0)
# strategy.load_data('2020-01-01', '2026-06-17')
# results = strategy.backtest(commission=0.001)
# strategy.visualize_backtest(results)

# print(f"\n=== 回测结果 ===")
# print(f"总收益: {results['total_return']:.2%}")
# print(f"年化收益: {results['annual_return']:.2%}")
# print(f"夏普比率: {results['sharpe_ratio']:.2f}")
# print(f"最大回撤: {results['max_drawdown']:.2%}")
# print(f"交易次数: {results['num_trades']}")
```

## 五、实战中的陷阱与应对

### 5.1 常见陷阱

#### 陷阱1：**结构性断裂（Structural Breaks）**

**现象：** 协整关系突然失效（如行业政策变化、公司并购）

**应对：**
- 使用**滚动窗口**定期重新检验协整关系
- 设置**协整关系监控指标**（如滚动ADF检验p值）
- 当p值持续 > 0.1时，停止交易该对

#### 陷阱2：**幸存者偏差（Survivorship Bias）**

**现象：** 用当前上市的股票回测，忽略了已退市的股票

**应对：**
- 使用**point-in-time数据**（如Compustat数据库）
- 在回测中模拟退市事件

#### 陷阱3：**流动性风险**

**现象：** 理论收益很高，但实际交易时滑点巨大

**应对：**
- 筛选日均成交额 > 1000万的股票
- 在回测中加入**滑点模型**（如基于成交量的线性滑点）

#### 陷阱4：**过度优化（Overfitting）**

**现象：** 参数调优后回测完美，样本外崩溃

**应对：**
- 使用**Walk-Forward优化**（样本内优化 + 样本外验证）
- 设置**参数稳定性检验**（滚动窗口下参数是否稳定）

### 5.2 风险管理框架

```python
class RiskManager:
    """配对交易风险管理器"""
    
    def __init__(self, max_position=0.1, max_loss_per_trade=0.02, 
                 max_correlation=0.7, max_drawdown=0.15):
        """
        初始化
        
        Parameters:
        -----------
        max_position : float, 单个配对最大仓位（占总资金比例）
        max_loss_per_trade : float, 单笔交易最大损失
        max_correlation : float, 持仓配对间最大相关性
        max_drawdown : float, 最大回撤止损线
        """
        self.max_position = max_position
        self.max_loss_per_trade = max_loss_per_trade
        self.max_correlation = max_correlation
        self.max_drawdown = max_drawdown
        
        self.current_positions = {}
        self.equity_curve = []
        
    def check_position_limit(self, pair_name, new_position_value, total_equity):
        """检查仓位限制"""
        current_position_value = self.current_positions.get(pair_name, 0)
        total_position_value = sum(abs(v) for v in self.current_positions.values())
        
        # 检查单个配对仓位
        if abs(current_position_value + new_position_value) / total_equity > self.max_position:
            print(f"⚠️ 仓位超限：{pair_name}")
            return False
        
        # 检查总仓位
        if (total_position_value + abs(new_position_value)) / total_equity > 0.5:
            print(f"⚠️ 总仓位超限")
            return False
        
        return True
    
    def check_correlation(self, new_pair, existing_pairs, price_data):
        """检查新配对与现有持仓的相关性"""
        for existing_pair in existing_pairs:
            # 计算两个配对的收益相关性
            returns_new = self.calculate_pair_returns(new_pair, price_data)
            returns_existing = self.calculate_pair_returns(existing_pair, price_data)
            
            corr = returns_new.corr(returns_existing)
            
            if abs(corr) > self.max_correlation:
                print(f"⚠️ 相关性过高：{new_pair} vs {existing_pair} ({corr:.2f})")
                return False
        
        return True
    
    def check_drawdown(self, current_equity, peak_equity):
        """检查回撤"""
        drawdown = (current_equity - peak_equity) / peak_equity
        
        if drawdown < -self.max_drawdown:
            print(f"⚠️ 触发最大回撤止损：{drawdown:.2%}")
            return False
        
        return True
    
    def calculate_pair_returns(self, pair, price_data):
        """计算配对收益（简化版）"""
        stock_a, stock_b = pair
        price_a = price_data[stock_a]
        price_b = price_data[stock_b]
        
        # 假设对冲比率为1（实际应从协整分析得到）
        spread = price_b - price_a
        returns = spread.pct_change()
        
        return returns
```

## 六、总结与展望

### 6.1 核心要点回顾

1. **协整是配对交易的核心**：相关性 ≠ 协整性，必须用统计检验确认
2. **系统化管理**：使用量化工框架批量筛选、监控股票对
3. **风险管理优先**：仓位限制、相关性控制、回撤止损缺一不可
4. **避免过度优化**：Walk-Forward验证、参数稳定性检验

### 6.2 进阶方向

- **多因子配对**：不仅用价格协整，还加入基本面因子（如PE比、ROE差）
- **机器学习增强**：用LSTM预测价差均值回归时间
- **高频配对交易**：利用分钟级数据捕捉日内定价偏差

---

## 附录：完整代码仓库

完整代码已开源：[Pairs-Trading-Framework](https://github.com/example/pairs-trading)

包含：
- 数据获取模块（支持A股、美股）
- 协整检验模块（ADF、PP、KPSS）
- 股票对筛选模块（并行计算加速）
- 回测引擎（支持多配对组合）
- 风险管理系统

---

*如果本文对你有帮助，欢迎点赞、收藏、转发！也欢迎在评论区分享你的配对交易经验。*
