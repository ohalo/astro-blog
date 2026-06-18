---
title: "配对交易与协整分析：统计套利实战指南"
description: "深入讲解配对交易的核心原理——协整关系识别、配对选择、交易信号构建与风险控制，提供完整的Python实现框架与回测验证，帮助投资者掌握统计套利策略。"
pubDate: 2026-06-18
tags: ["配对交易", "协整分析", "统计套利", "市场中性", "量化策略"]
category: "量化策略"
cover: "/images/pair-trading-cointegration/cover.png"
---

# 配对交易与协整分析：统计套利实战指南

## 引言：市场中性策略的魅力

在传统股票投资中，投资者面临两个核心挑战：
1. **方向性风险**：市场大跌时，即使选股能力优秀也难以获利
2. **系统性风险**：黑天鹅事件（如2020年疫情）导致全市场暴跌

**配对交易**（Pair Trading）通过构建**市场中性组合**，同时做多一只股票、做空另一只高度相关的股票，消除市场方向性风险，仅捕获两只股票的**相对价格差异**。

本文将深入讲解：
- 协整关系的数学原理与检验方法
- 配对选择的量化标准
- 交易信号的构建与优化
- 完整Python实现与回测框架

---

## 一、配对交易的核心原理

### 1.1 基本概念

配对交易基于**均值回归**（Mean Reversion）假设：

> 如果两只股票的线性组合是**平稳的**（Stationary），则它们的价格差异最终会回归到长期均值。

**数学表达**：

对于两只股票 \(A\) 和 \(B\)，若存在系数 \(\beta\)，使得：

\[
Spread_t = P_{A,t} - \beta \times P_{B,t}
\]

是平稳序列（Stationary Series），则可以进行配对交易。

- 当 \(Spread_t\) 显著高于均值时：**做空A，做多B**（卖出高价，买入低价）
- 当 \(Spread_t\) 显著低于均值时：**做多A，做空B**（买入低价，卖出高价）

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf  # 实际中建议用westock-data

# 示例：配对交易数据加载
def load_pair_data(ticker1, ticker2, start='2020-01-01', end='2025-12-31'):
    """
    加载配对股票数据
    实际中应调用：westock-data kline {ticker} --period day
    """
    # 使用yfinance获取示例数据（仅演示）
    stock1 = yf.download(ticker1, start=start, end=end, progress=False)['Adj Close']
    stock2 = yf.download(ticker2, start=start, end=end, progress=False)['Adj Close']
    
    # 合并数据
    data = pd.DataFrame({
        ticker1: stock1,
        ticker2: stock2
    }).dropna()
    
    return data

# 测试：中国平安 vs 中国太保（同行业配对）
pair_data = load_pair_data('601318.SS', '601601.SS')
print("配对数据预览：")
print(pair_data.head())
print(f"\n数据期间：{pair_data.index[0].date()} 至 {pair_data.index[-1].date()}")
print(f"交易天数：{len(pair_data)}")
```

### 1.2 协整 vs 相关性

**常见误区**：高相关性 ≠ 可配对交易

| 指标 | 相关性（Correlation） | 协整（Cointegration） |
|------|---------------------|---------------------|
| 定义 | 价格变化的同步程度 | 长期均衡关系的存在性 |
| 平稳性要求 | 无 | 价差必须平稳 |
| 适用场景 | 短期动量策略 | 均值回归策略 |
| 检验方法 | Pearson/Spearman | Engle-Granger / Johansen |

**关键区别**：
- 两只股票可以**高度相关**但不协整（如两只科技股同涨同跌，但价差持续扩大）
- 两只股票可以**协整但相关性不高**（如一个行业龙头和一个跟随者）

```python
def compare_correlation_cointegration(data):
    """对比相关性与协整性"""
    
    # 1. 计算相关性
    corr = data.corr().iloc[0, 1]
    
    # 2. 协整检验（Engle-Granger）
    score, p_value, _ = coint(data.iloc[:, 0], data.iloc[:, 1])
    
    # 3. 平稳性检验（ADF on residuals）
    X = data.iloc[:, 1]
    Y = data.iloc[:, 0]
    beta = OLS(Y, X).fit().params[0]
    spread = Y - beta * X
    adf_stat, adf_p, _ = adfuller(spread)
    
    print("\n=== 相关性与协整性对比 ===")
    print(f"价格相关性：{corr:.4f}")
    print(f"协整检验 p-value：{p_value:.4f} {'✅ 协整' if p_value < 0.05 else '❌ 不协整'}")
    print(f"价差平稳性 p-value：{adf_p:.4f} {'✅ 平稳' if adf_p < 0.05 else '❌ 不平稳'}")
    
    return corr, p_value, adf_p

# 执行检验
corr, p_val, adf_p = compare_correlation_cointegration(pair_data)
```

---

## 二、协整关系的检验方法

### 2.1 Engle-Granger 两步法

**步骤1**：用OLS估计长期均衡关系

\[
P_{A,t} = \alpha + \beta \times P_{B,t} + \epsilon_t
\]

**步骤2**：检验残差 \(\epsilon_t\) 的平稳性（ADF检验）

```python
class CointegrationTester:
    """协整关系检验器"""
    
    def __init__(self, significance_level=0.05):
        self.alpha = significance_level
        
    def engle_granger_test(self, price1, price2):
        """
        Engle-Granger 协整检验
        返回：is_cointegrated, beta, spread
        """
        # 步骤1：OLS回归
        X = price2.values
        y = price1.values
        model = OLS(y, X).fit()
        beta = model.params[0]
        spread = y - beta * X
        
        # 步骤2：ADF检验残差
        adf_stat, p_value, crit_values = adfuller(spread)
        
        is_cointegrated = p_value < self.alpha
        
        return {
            'is_cointegrated': is_cointegrated,
            'beta': beta,
            'spread': spread,
            'adf_stat': adf_stat,
            'p_value': p_value,
            'critical_values': crit_values
        }
    
    def calculate_half_life(self, spread):
        """计算价差的半衰期（均值回归速度）"""
        # 构建自回归模型：spread[t] - spread[t-1] = alpha + beta * spread[t-1] + error
        spread_lag = spread[:-1]
        spread_diff = spread[1:] - spread_lag
        
        model = OLS(spread_diff, spread_lag).fit()
        beta = model.params[0]
        
        # 半衰期 = -ln(2) / ln(1 + beta)
        if beta < 0:
            half_life = -np.log(2) / np.log(1 + beta)
        else:
            half_life = np.inf  # 不均值回归
        
        return half_life

# 应用协整检验
tester = CointegrationTester()
result = tester.engle_granger_test(
    pair_data.iloc[:, 0],
    pair_data.iloc[:, 1]
)

print("\n=== Engle-Granger 协整检验结果 ===")
print(f"协整关系：{'是' if result['is_cointegrated'] else '否'}")
print(f"对冲比例 beta：{result['beta']:.4f}")
print(f"ADF统计量：{result['adf_stat']:.4f}")
print(f"p-value：{result['p_value']:.4f}")
print(f"临界值（5%）：{result['critical_values']['5%']:.4f}")

# 计算半衰期
half_life = tester.calculate_half_life(result['spread'])
print(f"\n价差半衰期：{half_life:.1f} 天")
```

### 2.2 配对选择标准

筛选可交易配对的多维标准：

| 标准 | 阈值 | 说明 |
|------|------|------|
| 协整p值 | < 0.05 | 统计显著性 |
| 对冲比例β | 0.5 ~ 2.0 | 避免过度杠杆 |
| 半衰期 | 10 ~ 60天 | 均值回归速度适中 |
| 价差波动率 | < 2% | 降低噪音交易 |
| 历史最大偏离 | < 3倍标准差 | 可控的回撤风险 |

```python
def screen_pair_universe(universe_tickers, start='2020-01-01', end='2025-12-31'):
    """
    批量筛选配对
    输入：universe_tickers - 股票列表
    输出：符合条件的配对列表
    """
    tester = CointegrationTester()
    valid_pairs = []
    
    for i in range(len(universe_tickers)):
        for j in range(i+1, len(universe_tickers)):
            ticker1 = universe_tickers[i]
            ticker2 = universe_tickers[j]
            
            # 加载数据（实际应调用westock-data）
            try:
                data = load_pair_data(ticker1, ticker2, start, end)
                
                # 协整检验
                result = tester.engle_granger_test(data.iloc[:, 0], data.iloc[:, 1])
                
                if result['is_cointegrated']:
                    # 计算额外指标
                    spread = result['spread']
                    half_life = tester.calculate_half_life(spread)
                    spread_vol = np.std(spread / data.iloc[:, 0].mean())
                    
                    # 筛选条件
                    if (0.5 <= result['beta'] <= 2.0 and
                        10 <= half_life <= 60 and
                        spread_vol < 0.02):
                        
                        valid_pairs.append({
                            'ticker1': ticker1,
                            'ticker2': ticker2,
                            'beta': result['beta'],
                            'p_value': result['p_value'],
                            'half_life': half_life,
                            'spread_vol': spread_vol
                        })
            except Exception as e:
                print(f"处理配对 {ticker1}-{ticker2} 时出错：{e}")
                continue
    
    return pd.DataFrame(valid_pairs)

# 示例：筛选金融行业配对（实际中应扩展股票池）
financial_stocks = ['601318.SS', '601601.SS', '601628.SS', '601336.SS']
pair_candidates = screen_pair_universe(financial_stocks)

print("\n=== 筛选出的有效配对 ===")
print(pair_candidates)
```

---

## 三、交易信号构建

### 3.1 Z-Score 信号

将价差标准化为**Z-Score**：

\[
Z_t = \frac{Spread_t - \mu_{Spread}}{\sigma_{Spread}}
\]

**交易规则**：
- **入场**：\(|Z_t| > threshold_{entry}\)（如2.0）
  - \(Z_t > +2\)：做空股票A，做多股票B
  - \(Z_t < -2\)：做多股票A，做空股票B
- **出场**：\(|Z_t| < threshold_{exit}\)（如0.5）或期限到达

```python
class PairTradingSignal:
    """配对交易信号生成器"""
    
    def __init__(self, entry_threshold=2.0, exit_threshold=0.5, lookback=60):
        self.entry_thresh = entry_threshold
        self.exit_thresh = exit_threshold
        self.lookback = lookback
        
    def calculate_z_score(self, spread, method='rolling'):
        """
        计算Z-Score
        method: 'rolling'（滚动窗口）或 'expanding'（全样本）
        """
        if method == 'rolling':
            mean = pd.Series(spread).rolling(self.lookback).mean()
            std = pd.Series(spread).rolling(self.lookback).std()
        elif method == 'expanding':
            mean = pd.Series(spread).expanding().mean()
            std = pd.Series(spread).expanding().std()
        else:
            raise ValueError("method must be 'rolling' or 'expanding'")
        
        z_score = (spread - mean) / std
        return z_score
    
    def generate_signals(self, spread):
        """
        生成交易信号
        返回：signal series (1=做多A做空B, -1=做空A做多B, 0=平仓)
        """
        z_score = self.calculate_z_score(spread, method='rolling')
        signals = pd.Series(0, index=spread.index)
        
        # 状态变量
        position = 0  # 当前持仓：1=多A空B, -1=空A多B, 0=空仓
        
        for i in range(1, len(z_score)):
            if pd.isna(z_score.iloc[i]):
                continue
                
            # 入场信号
            if position == 0:
                if z_score.iloc[i] < -self.entry_thresh:
                    position = 1  # 做多A，做空B
                    signals.iloc[i] = 1
                elif z_score.iloc[i] > self.entry_thresh:
                    position = -1  # 做空A，做多B
                    signals.iloc[i] = -1
            
            # 出场信号
            elif position == 1:
                if abs(z_score.iloc[i]) < self.exit_thresh:
                    position = 0
                    signals.iloc[i] = 0  # 平仓
                elif z_score.iloc[i] > self.entry_thresh:
                    # 反转信号（可选）
                    position = -1
                    signals.iloc[i] = -1
            
            elif position == -1:
                if abs(z_score.iloc[i]) < self.exit_thresh:
                    position = 0
                    signals.iloc[i] = 0  # 平仓
                elif z_score.iloc[i] < -self.entry_thresh:
                    # 反转信号（可选）
                    position = 1
                    signals.iloc[i] = 1
        
        return signals, z_score

# 测试信号生成
signal_gen = PairTradingSignal(entry_threshold=2.0, exit_threshold=0.5)
spread = result['spread']
signals, z_scores = signal_gen.generate_signals(spread)

print("\n=== 交易信号统计 ===")
print(f"总交易次数：{(signals != 0).sum()}")
print(f"做多A次数：{(signals == 1).sum()}")
print(f"做空A次数：{(signals == -1).sum()}")
```

### 3.2 信号可视化

```python
def visualize_signals(price1, price2, spread, z_scores, signals):
    """可视化配对交易信号"""
    
    fig, axes = plt.subplots(3, 1, figsize=(15, 12))
    
    # 图1：两只股票价格
    ax1 = axes[0]
    ax1.plot(price1.index, price1.values, label=price1.name, linewidth=2)
    ax1.plot(price2.index, price2.values, label=price2.name, linewidth=2)
    ax1.set_ylabel('价格', fontsize=12)
    ax1.legend()
    ax1.set_title('配对股票价格走势', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # 图2：价差与Z-Score
    ax2 = axes[1]
    ax2.plot(spread.index, spread, label='价差', linewidth=2, color='blue')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('价差', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # 次坐标轴：Z-Score
    ax2_twin = ax2.twinx()
    ax2_twin.plot(z_scores.index, z_scores, label='Z-Score', linewidth=1.5, color='red', alpha=0.7)
    ax2_twin.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax2_twin.axhline(y=-2, color='red', linestyle='--', alpha=0.5)
    ax2_twin.axhline(y=0, color='green', linestyle='--', alpha=0.5)
    ax2_twin.set_ylabel('Z-Score', fontsize=12)
    ax2_twin.legend(loc='upper right')
    
    # 图3：交易信号
    ax3 = axes[2]
    # 标记交易点
    entry_long = (signals == 1) & (signals.shift(1) == 0)
    entry_short = (signals == -1) & (signals.shift(1) == 0)
    exit_signal = (signals == 0) & (signals.shift(1) != 0)
    
    ax3.scatter(spread.index[entry_long], spread[entry_long], 
               marker='^', color='green', s=100, label='做多A（入场）', zorder=5)
    ax3.scatter(spread.index[entry_short], spread[entry_short], 
               marker='v', color='red', s=100, label='做空A（入场）', zorder=5)
    ax3.scatter(spread.index[exit_signal], spread[exit_signal], 
               marker='o', color='gray', s=80, label='平仓', zorder=5)
    
    ax3.plot(spread.index, spread, linewidth=1, alpha=0.5)
    ax3.set_ylabel('价差', fontsize=12)
    ax3.set_xlabel('日期', fontsize=12)
    ax3.set_title('交易信号标记', fontsize=14)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pair_trading_signals.png', dpi=150, bbox_inches='tight')
    print("\n✅ 信号可视化图已保存")

# 执行可视化
visualize_signals(
    pair_data.iloc[:, 0],
    pair_data.iloc[:, 1],
    spread,
    z_scores,
    signals
)
```

---

## 四、回测框架与绩效分析

### 4.1 回测引擎

构建完整的配对交易回测系统：

```python
class PairTradingBacktest:
    """配对交易回测引擎"""
    
    def __init__(self, price1, price2, beta, initial_capital=1000000):
        self.price1 = price1
        self.price2 = price2
        self.beta = beta
        self.initial_capital = initial_capital
        
    def run_backtest(self, signals, transaction_cost=0.001):
        """
        运行回测
        输入：signals - 交易信号Series
              transaction_cost - 交易成本（单边）
        输出：回测结果DataFrame
        """
        # 初始化组合状态
        portfolio = pd.DataFrame(index=signals.index)
        portfolio['position'] = signals  # 持仓方向
        portfolio['price1'] = self.price1.values
        portfolio['price2'] = self.price2.values
        
        # 计算持仓数量（等金额对冲）
        portfolio['shares1'] = 0
        portfolio['shares2'] = 0
        portfolio['cash'] = self.initial_capital
        portfolio['total_value'] = self.initial_capital
        
        current_position = 0
        for i in range(1, len(portfolio)):
            if portfolio['position'].iloc[i] != current_position:
                # 交易发生
                price1 = portfolio['price1'].iloc[i]
                price2 = portfolio['price2'].iloc[i]
                
                if current_position == 0:
                    # 开仓
                    notional = portfolio['total_value'].iloc[i-1] / 2
                    shares1 = int(notional / price1)
                    shares2 = int(notional * self.beta / price2)
                    
                    portfolio.loc[portfolio.index[i], 'shares1'] = shares1
                    portfolio.loc[portfolio.index[i], 'shares2'] = shares2
                    
                    # 扣除交易成本
                    cost = (shares1 * price1 + shares2 * price2) * transaction_cost
                    portfolio.loc[portfolio.index[i], 'cash'] = \
                        portfolio['cash'].iloc[i-1] - cost
                    
                elif portfolio['position'].iloc[i] == 0:
                    # 平仓
                    shares1 = portfolio['shares1'].iloc[i-1]
                    shares2 = portfolio['shares2'].iloc[i-1]
                    
                    # 计算平仓收益
                    pnl = (shares1 * price1 - shares1 * portfolio['price1'].iloc[i-1] +
                           shares2 * price2 - shares2 * portfolio['price2'].iloc[i-1])
                    
                    portfolio.loc[portfolio.index[i], 'shares1'] = 0
                    portfolio.loc[portfolio.index[i], 'shares2'] = 0
                    
                    # 扣除交易成本
                    cost = (abs(shares1) * price1 + abs(shares2) * price2) * transaction_cost
                    portfolio.loc[portfolio.index[i], 'cash'] = \
                        portfolio['cash'].iloc[i-1] + pnl - cost
                
                current_position = portfolio['position'].iloc[i]
            
            else:
                # 无交易，继承上一期状态
                portfolio.loc[portfolio.index[i], 'shares1'] = portfolio['shares1'].iloc[i-1]
                portfolio.loc[portfolio.index[i], 'shares2'] = portfolio['shares2'].iloc[i-1]
                portfolio.loc[portfolio.index[i], 'cash'] = portfolio['cash'].iloc[i-1]
            
            # 计算总市值
            portfolio.loc[portfolio.index[i], 'total_value'] = \
                portfolio['cash'].iloc[i] + \
                portfolio['shares1'].iloc[i] * portfolio['price1'].iloc[i] + \
                portfolio['shares2'].iloc[i] * portfolio['price2'].iloc[i]
        
        self.portfolio = portfolio
        return portfolio
    
    def calculate_performance(self):
        """计算绩效指标"""
        returns = self.portfolio['total_value'].pct_change()
        
        # 累计收益
        cumulative_ret = (self.portfolio['total_value'].iloc[-1] / self.initial_capital - 1)
        
        # 年化收益
        days = (self.portfolio.index[-1] - self.portfolio.index[0]).days
        ann_return = (1 + cumulative_ret) ** (252 / days) - 1
        
        # 夏普比率
        ann_vol = returns.std() * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0
        
        # 最大回撤
        cumulative = self.portfolio['total_value'] / self.initial_capital
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_dd = drawdown.min()
        
        # 胜率
        trade_returns = returns[returns != 0]
        win_rate = (trade_returns > 0).sum() / len(trade_returns) if len(trade_returns) > 0 else 0
        
        return {
            'Cumulative Return': cumulative_ret,
            'Annualized Return': ann_return,
            'Annualized Volatility': ann_vol,
            'Sharpe Ratio': sharpe,
            'Max Drawdown': max_dd,
            'Win Rate': win_rate,
            'Number of Trades': (self.portfolio['position'] != self.portfolio['position'].shift(1)).sum() // 2
        }

# 运行回测
backtest = PairTradingBacktest(
    pair_data.iloc[:, 0],
    pair_data.iloc[:, 1],
    beta=result['beta'],
    initial_capital=1000000
)

portfolio_result = backtest.run_backtest(signals, transaction_cost=0.001)
performance = backtest.calculate_performance()

print("\n=== 配对交易回测结果 ===")
for metric, value in performance.items():
    if 'Return' in metric or 'Drawdown' in metric or 'Volatility' in metric:
        print(f"{metric}: {value*100:.2f}%")
    elif metric == 'Number of Trades':
        print(f"{metric}: {value}")
    else:
        print(f"{metric}: {value:.2f}")
```

### 4.2 绩效改进方向

**潜在改进**：

1. **动态阈值**：根据市场波动率调整入场/出场阈值
2. **仓位管理**：基于Z-Score绝对值调整仓位大小
3. **多配对组合**：同时交易多个独立配对，分散风险
4. **止损机制**：当Z-Score突破3倍标准差时强制平仓

```python
# 示例：动态阈值调整
def dynamic_threshold(z_score, volatility_window=20):
    """
    根据市场波动率动态调整阈值
    高波动期提高阈值，低波动期降低阈值
    """
    recent_vol = abs(z_score).rolling(volatility_window).std()
    base_threshold = 2.0
    
    # 波动率调整因子
    vol_adjustment = recent_vol / recent_vol.mean()
    dynamic_entry = base_threshold * vol_adjustment
    dynamic_exit = 0.5 * vol_adjustment
    
    return dynamic_entry, dynamic_exit
```

---

## 五、实战注意事项

### 5.1 数据要求

**必须使用**：
- **复权价格**：确保分红、拆股不影响价差计算
- **分钟级数据**：精确捕捉入场/出场时点
- **幸存者偏差处理**：剔除已退市股票

**推荐数据源**：
```python
# 使用westock-data获取复权数据
# westock-data kline sh600519 --period day --adjust qfq --limit 1000
```

### 5.2 风险控制

| 风险类型 | 描述 | 应对措施 |
|---------|------|---------|
| 协整关系断裂 | 长期均衡关系失效 | 实时监控ADF p-value，>0.1时停止交易 |
| 流动性风险 | 无法及时平仓 | 选择日均成交额>1亿的标的 |
| 杠杆风险 | 过度杠杆导致爆仓 | 限制单配对敞口≤10%总资本 |
| 黑天鹅事件 | 极端市场导致价差失控 | 设置硬性止损（Z-Score>4） |

### 5.3 实盘部署建议

1. **模拟盘测试**：至少3个月模拟交易验证策略稳定性
2. **逐步建仓**：从1-2个配对开始，逐步扩展到5-10个
3. **实时监控**：开发Dashboard监控各配对的Z-Score和持仓状态
4. **定期复盘**：每月重新检验协整关系，剔除失效配对

---

## 六、总结与展望

配对交易是统计套利领域的经典策略：

✅ **核心优势**
- 市场中性，降低方向性风险
- 均值回归特性，胜率较高
- 策略逻辑清晰，易于风控

⚠️ **主要挑战**
- 协整关系可能突然断裂
- 交易成本高（双边交易）
- 需要精细的参数调优

🔮 **未来方向**
1. **机器学习增强**：用LSTM预测Z-Score回归时间
2. **高频配对交易**：利用分钟级数据捕捉短期定价偏差
3. **跨市场配对**：在A股与港股间寻找交叉上市的协整关系

---

## 参考文献

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. Review of Financial Studies.
3. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). *Pairs Trading*. Quantitative Finance.

---

**免责声明**：本文仅供学术交流，不构成投资建议。配对交易存在模型风险和交易成本侵蚀，实盘前需充分测试。

---

**版权声明**：本文章由量化策略专家生成，遵循CC BY-NC 4.0协议。欢迎转载学习，但需注明出处并不得用于商业用途。
