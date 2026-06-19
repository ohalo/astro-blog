---
title: "配对交易与协整分析"
date: 2026-06-20
description: "深入讲解配对交易策略的理论基础与实践方法，从协整检验到交易信号构建，包含完整的Python代码实现和实战案例分析。"
tags:
  - 配对交易
  - 协整分析
  - 统计套利
  - 均值回归
  - Python实战
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析

## 引言

在量化投资的世界里，**配对交易（Pairs Trading）**是一种经典的市场中性策略，它不依赖市场方向，而是通过捕捉两个高度相关资产之间的暂时性偏离来获利。这种策略的核心思想是：找到一对具有长期均衡关系的股票，当它们的价格出现短期偏离时，做多低估资产、做空高估资产，等待价格回归均衡后平仓获利。

配对交易的成功关键在于**协整关系（Cointegration）**的识别与验证。本文将系统介绍配对交易的理论基础、协整检验方法、交易信号构建，并通过Python代码展示从选股到实盘执行的完整流程。

## 配对交易的理论基础

### 1. 为什么需要协整？

传统的配对方法往往基于**相关性分析**——选择历史价格走势相似的股票进行交易。然而，高相关性并不保证价格偏离会回归，两个资产的价差可能随时间持续扩大。

**协整关系**则提供了更严格的统计基础：
- 两个非平稳时间序列（如股价）如果存在协整关系，则它们的线性组合是平稳的
- 这意味着虽然单个股价可能游走，但它们的相对关系（价差或比率）会在长期保持均衡
- 当短期偏离发生时，存在统计学上的"引力"将价格拉回均衡

### 2. 协整 vs 相关性

| 特征 | 相关性 | 协整 |
|------|--------|------|
| 定义 | 衡量线性依赖程度 | 描述长期均衡关系 |
| 平稳性要求 | 无 | 要求残差平稳 |
| 经济学意义 | 同步变动 | 均值回归机制 |
| 适用性 | 短期交易 | 长期套利 |

### 3. 配对交易的优势

1. **市场中性**：多空对冲，降低系统性风险
2. **均值回归**：基于统计学原理，具有可解释的盈利逻辑
3. **风险可控**：通过止损机制控制单次交易损失
4. **适应性强**：适用于股票、期货、ETF等多个市场

## 协整检验方法

### 方法一：Engle-Granger两步法

**步骤**：
1. 对两个价格序列进行OLS回归：$P_t^A = \alpha + \beta P_t^B + \epsilon_t$
2. 对残差 $\epsilon_t$ 进行ADF检验（Augmented Dickey-Fuller Test）
3. 如果残差是平稳的，则两个序列存在协整关系

**优点**：简单直观，易于实现
**缺点**：只能检验单一协整向量，且对哪个变量作为被解释变量敏感

### 方法二：Johansen检验

**原理**：基于向量自回归（VAR）模型，可以检验多个协整向量

**优点**：适用于多变量系统，更稳健
**缺点**：计算复杂，需要确定滞后阶数

### 方法三：Phillips-Ouliaris检验

**原理**：考虑回归残差的自相关结构，对Engle-Granger方法进行改进

**优点**：更稳健的统计量
**缺点**：计算量较大

## Python实战：构建配对交易系统

下面通过一个完整的案例，展示如何使用Python实现配对交易策略。

### 步骤1：数据获取与预处理

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 下载股票数据（示例：中国平安 vs 中国人寿）
tickers = ['601318.SS', '601628.SS']
start_date = '2020-01-01'
end_date = '2024-12-31'

# 使用yfinance下载数据
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

print("数据形状:", data.shape)
print("\n数据预览:")
print(data.head())
```

### 步骤2：协整检验

```python
def test_cointegration(price1, price2, model='engle-granger'):
    """
    协整检验函数
    
    参数：
    - price1, price2: 两个价格序列
    - model: 检验方法 ('engle-granger' 或 'johansen')
    
    返回：
    - result: 检验结果字典
    """
    
    if model == 'engle-granger':
        # Engle-Granger两步法
        # 第一步：OLS回归
        X = price2.values.reshape(-1, 1)
        y = price1.values
        model = OLS(y, X).fit()
        spread = model.resid
        
        # 第二步：ADF检验
        adf_result = adfuller(spread, autolag='AIC')
        
        result = {
            'method': 'Engle-Granger',
            'adf_statistic': adf_result[0],
            'p_value': adf_result[1],
            'critical_values': adf_result[4],
            'hedge_ratio': model.params[0],
            'spread': spread
        }
        
        # 判断是否存在协整关系（5%显著性水平）
        result['is_cointegrated'] = result['p_value'] < 0.05
        
    elif model == 'johansen':
        # Johansen检验（需要安装johansen库或使用statsmodels的coint_johansen）
        from statsmodels.tsa.vector_ar.vecm import coint_johansen
        
        data_matrix = np.column_stack([price1.values, price2.values])
        result_johansen = coint_johansen(data_matrix, 0, 1)
        
        result = {
            'method': 'Johansen',
            'trace_statistic': result_johansen.lr1,
            'critical_values': result_johansen.cvt,
            'eigen_statistic': result_johansen.lr2,
            'is_cointegrated': result_johansen.lr1[0] > result_johansen.cvt[0, 1]  # 5%临界值
        }
    
    return result

# 执行协整检验
prices1 = data[tickers[0]]
prices2 = data[tickers[1]]

coint_result = test_cointegration(prices1, prices2, model='engle-granger')

print("\n=== 协整检验结果 ===")
print(f"检验方法: {coint_result['method']}")
print(f"ADF统计量: {coint_result['adf_statistic']:.4f}")
print(f"p值: {coint_result['p_value']:.4f}")
print(f"对冲比率（hedge ratio）: {coint_result['hedge_ratio']:.4f}")
print(f"是否存在协整关系: {'是' if coint_result['is_cointegrated'] else '否'}")
```

### 步骤3：价差分析与可视化

```python
# 计算价差（spread）
spread = coint_result['spread']
hedge_ratio = coint_result['hedge_ratio']

# 计算价差的Z-score
z_score = (spread - spread.mean()) / spread.std()

# 绘制价格序列和价差
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：原始价格
ax1 = axes[0]
ax1.plot(prices1.index, prices1.values, label=tickers[0], linewidth=2)
ax1.plot(prices2.index, prices2.values, label=tickers[1], linewidth=2)
ax1.set_title('股票价格走势', fontsize=14, fontweight='bold')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：价差
ax2 = axes[1]
ax2.plot(spread.index, spread.values, color='blue', linewidth=2)
ax2.axhline(y=spread.mean(), color='red', linestyle='--', label='均值')
ax2.axhline(y=spread.mean() + 2*spread.std(), color='green', linestyle=':', label='+2σ')
ax2.axhline(y=spread.mean() - 2*spread.std(), color='green', linestyle=':', label='-2σ')
ax2.fill_between(spread.index, spread.mean() - 2*spread.std(), 
                 spread.mean() + 2*spread.std(), alpha=0.2, color='green')
ax2.set_title('价差（Spread）走势', fontsize=14, fontweight='bold')
ax2.set_ylabel('价差')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：Z-score
ax3 = axes[2]
ax3.plot(z_score.index, z_score.values, color='purple', linewidth=2)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax3.axhline(y=2, color='red', linestyle='--', label='入场信号（+2）')
ax3.axhline(y=-2, color='red', linestyle='--', label='入场信号（-2）')
ax3.axhline(y=0, color='green', linestyle=':', label='平仓信号（0）')
ax3.fill_between(z_score.index, -2, 2, alpha=0.2, color='gray')
ax3.set_title('价差Z-score', fontsize=14, fontweight='bold')
ax3.set_xlabel('日期')
ax3.set_ylabel('Z-score')
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/spread_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

### 步骤4：交易信号生成

```python
def generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.0):
    """
    生成交易信号
    
    参数：
    - z_score: 价差的Z-score序列
    - entry_threshold: 入场阈值（绝对值）
    - exit_threshold: 平仓阈值
    
    返回：
    - signals: 交易信号序列 (1: 做多价差, -1: 做空价差, 0: 不持仓)
    """
    
    signals = pd.Series(index=z_score.index, data=0)
    position = 0  # 当前持仓状态
    
    for i in range(1, len(z_score)):
        if position == 0:  # 空仓
            if z_score.iloc[i] < -entry_threshold:
                # Z-score过低，做多价差（做多股票1，做空股票2）
                signals.iloc[i] = 1
                position = 1
            elif z_score.iloc[i] > entry_threshold:
                # Z-score过高，做空价差（做空股票1，做多股票2）
                signals.iloc[i] = -1
                position = -1
        
        elif position == 1:  # 持多仓
            if abs(z_score.iloc[i]) < exit_threshold:
                # Z-score回归，平仓
                signals.iloc[i] = 0
                position = 0
        
        elif position == -1:  # 持空仓
            if abs(z_score.iloc[i]) < exit_threshold:
                # Z-score回归，平仓
                signals.iloc[i] = 0
                position = 0
    
    return signals

# 生成交易信号
trading_signals = generate_trading_signals(z_score, entry_threshold=2.0, exit_threshold=0.5)

print("\n=== 交易信号统计 ===")
print(f"总交易次数: {(trading_signals != 0).sum()}")
print(f"做多信号: {(trading_signals == 1).sum()}")
print(f"做空信号: {(trading_signals == -1).sum()}")
```

### 步骤5：回测配对交易策略

```python
def backtest_pair_trading(prices1, prices2, signals, hedge_ratio, initial_capital=1000000):
    """
    回测配对交易策略
    
    参数：
    - prices1, prices2: 两个股票的价格序列
    - signals: 交易信号序列
    - hedge_ratio: 对冲比率
    - initial_capital: 初始资金
    
    返回：
    - results: 回测结果DataFrame
    """
    
    # 初始化结果DataFrame
    results = pd.DataFrame(index=prices1.index)
    results['price1'] = prices1.values
    results['price2'] = prices2.values
    results['signal'] = signals.values
    results['spread'] = prices1.values - hedge_ratio * prices2.values
    
    # 计算持仓
    results['position1'] = 0  # 股票1持仓
    results['position2'] = 0  # 股票2持仓
    results['cash'] = initial_capital
    results['portfolio_value'] = initial_capital
    
    # 假设每只股票使用相同的资金（市场中性）
    shares_per_trade = 100  # 每次交易100股
    
    for i in range(1, len(results)):
        # 复制前一日的持仓和现金
        results.iloc[i, results.columns.get_loc('position1')] = results.iloc[i-1]['position1']
        results.iloc[i, results.columns.get_loc('position2')] = results.iloc[i-1]['position2']
        results.iloc[i, results.columns.get_loc('cash')] = results.iloc[i-1]['cash']
        
        # 如果有新的交易信号
        if results.iloc[i]['signal'] != 0 and results.iloc[i-1]['signal'] == 0:
            # 开仓
            if results.iloc[i]['signal'] == 1:  # 做多价差
                # 做多股票1，做空股票2
                results.iloc[i, results.columns.get_loc('position1')] = shares_per_trade
                results.iloc[i, results.columns.get_loc('position2')] = -int(shares_per_trade * hedge_ratio)
                # 扣除交易成本（假设0.1%）
                cost = (shares_per_trade * results.iloc[i]['price1'] + 
                       int(shares_per_trade * hedge_ratio) * results.iloc[i]['price2']) * 0.001
                results.iloc[i, results.columns.get_loc('cash')] -= cost
            
            elif results.iloc[i]['signal'] == -1:  # 做空价差
                # 做空股票1，做多股票2
                results.iloc[i, results.columns.get_loc('position1')] = -shares_per_trade
                results.iloc[i, results.columns.get_loc('position2')] = int(shares_per_trade * hedge_ratio)
                # 扣除交易成本
                cost = (shares_per_trade * results.iloc[i]['price1'] + 
                       int(shares_per_trade * hedge_ratio) * results.iloc[i]['price2']) * 0.001
                results.iloc[i, results.columns.get_loc('cash')] -= cost
        
        # 如果平仓
        elif results.iloc[i]['signal'] == 0 and results.iloc[i-1]['signal'] != 0:
            # 平掉所有持仓
            # 计算持仓市值
            position1_value = results.iloc[i]['position1'] * results.iloc[i]['price1']
            position2_value = results.iloc[i]['position2'] * results.iloc[i]['price2']
            
            # 平仓后的现金变化
            results.iloc[i, results.columns.get_loc('cash')] += position1_value + position2_value
            
            # 扣除交易成本
            cost = (abs(results.iloc[i]['position1']) * results.iloc[i]['price1'] + 
                   abs(results.iloc[i]['position2']) * results.iloc[i]['price2']) * 0.001
            results.iloc[i, results.columns.get_loc('cash')] -= cost
            
            # 清空持仓
            results.iloc[i, results.columns.get_loc('position1')] = 0
            results.iloc[i, results.columns.get_loc('position2')] = 0
        
        # 计算组合价值
        portfolio_value = (results.iloc[i]['position1'] * results.iloc[i]['price1'] +
                          results.iloc[i]['position2'] * results.iloc[i]['price2'] +
                          results.iloc[i]['cash'])
        results.iloc[i, results.columns.get_loc('portfolio_value')] = portfolio_value
    
    # 计算收益
    results['returns'] = results['portfolio_value'].pct_change()
    results['cumulative_returns'] = (1 + results['returns']).cumprod()
    
    return results

# 执行回测
backtest_results = backtest_pair_trading(
    prices1,
    prices2,
    trading_signals,
    hedge_ratio,
    initial_capital=1000000
)

print("\n=== 回测结果 ===")
total_return = (backtest_results['portfolio_value'].iloc[-1] / 1000000 - 1) * 100
annual_return = total_return / (len(backtest_results) / 252)
sharpe_ratio = (backtest_results['returns'].mean() / backtest_results['returns'].std()) * np.sqrt(252)
max_drawdown = ((backtest_results['portfolio_value'] / backtest_results['portfolio_value'].cummax()) - 1).min() * 100

print(f"总收益率: {total_return:.2f}%")
print(f"年化收益率: {annual_return:.2f}%")
print(f"夏普比率: {sharpe_ratio:.2f}")
print(f"最大回撤: {max_drawdown:.2f}%")
```

### 步骤6：结果可视化

```python
# 绘制回测结果
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 子图1：组合价值曲线
ax1 = axes[0, 0]
ax1.plot(backtest_results.index, backtest_results['portfolio_value'], 
         linewidth=2, color='blue')
ax1.set_title('组合价值曲线', fontsize=14, fontweight='bold')
ax1.set_xlabel('日期')
ax1.set_ylabel('组合价值')
ax1.grid(True, alpha=0.3)

# 子图2：累计收益
ax2 = axes[0, 1]
ax2.plot(backtest_results.index, backtest_results['cumulative_returns'], 
         linewidth=2, color='green')
ax2.set_title('累计收益', fontsize=14, fontweight='bold')
ax2.set_xlabel('日期')
ax2.set_ylabel('累计收益')
ax2.grid(True, alpha=0.3)

# 子图3：回撤曲线
ax3 = axes[1, 0]
cummax = backtest_results['portfolio_value'].cummax()
drawdown = (backtest_results['portfolio_value'] - cummax) / cummax * 100
ax3.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
ax3.plot(drawdown.index, drawdown.values, linewidth=1, color='darkred')
ax3.set_title('回撤曲线', fontsize=14, fontweight='bold')
ax3.set_xlabel('日期')
ax3.set_ylabel('回撤(%)')
ax3.grid(True, alpha=0.3)

# 子图4：交易信号与价差
ax4 = axes[1, 1]
ax4.plot(z_score.index, z_score.values, color='purple', linewidth=1, alpha=0.5)
# 标记交易信号
for i in range(len(trading_signals)):
    if trading_signals.iloc[i] == 1:
        ax4.scatter(z_score.index[i], z_score.iloc[i], color='green', marker='^', s=100)
    elif trading_signals.iloc[i] == -1:
        ax4.scatter(z_score.index[i], z_score.iloc[i], color='red', marker='v', s=100)
ax4.set_title('交易信号标记', fontsize=14, fontweight='bold')
ax4.set_xlabel('日期')
ax4.set_ylabel('Z-score')
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest_results.png', dpi=300, bbox_inches='tight')
plt.show()
```

## 配对交易的实战要点

### 1. 股票对筛选

成功的配对交易始于优质的股票对。筛选标准包括：

- **行业相关性**：同行业或产业链上下游企业
- **市值相似性**：避免流动性差异过大
- **历史协整检验**：通过统计检验验证长期关系
- **基本面分析**：确保业务逻辑支撑价格联动

### 2. 参数优化

关键参数需要通过历史数据优化：

- **入场阈值**：通常为1.5-2.5个标准差
- **平仓阈值**：0-1个标准差
- **止损阈值**：-2.5至-3个标准差
- **持有期限制**：防止长期不收敛

### 3. 风险控制

配对交易虽然市场中性，但仍需注意：

- **模型风险**：协整关系可能断裂
- **执行风险**：做空约束、流动性不足
- **集中度风险**：过度集中于某一行业
- **的黑天鹅事件**：金融危机等极端情况

### 4. 实务中的挑战

- **做空成本**：A股融券成本高，限制策略实施
- **交易滑点**：价差较小时，滑点影响显著
- **资金利用率**：保证金占用降低资金效率
- **税务影响**：频繁交易增加税务成本

## 进阶主题：机器学习在配对交易中的应用

### 1. 动态对冲比率

传统方法使用固定的对冲比率（hedge ratio），但实际上最优对冲比率可能随时间变化。可以使用：

- **卡尔曼滤波**：动态估计时变对冲比率
- **滚动回归**：使用滑动窗口更新回归系数
- **神经网络**：捕捉非线性对冲关系

### 2. 智能信号生成

使用机器学习模型优化交易信号：

- **SVM**：分类模型判断是否入场
- **LSTM**：预测价差未来走势
- **强化学习**：学习最优入场/出场策略

### 3. 多因子配对

将多个股票组合成配对，构建多因子配对交易系统：

- **主成分分析（PCA）**：降维提取共同因子
- **聚类分析**：自动发现潜在配对
- **图网络**：建模股票间的复杂关系

## 总结

配对交易是一种基于统计套利的量化策略，通过捕捉具有协整关系的股票对之间的暂时性偏离来获利。本文系统介绍了配对交易的理论基础、协整检验方法、Python实现步骤，以及实战中的关键要点。

成功的配对交易需要：
1. 严谨的统计检验确保协整关系存在
2. 合理的参数设置平衡收益与风险
3. 有效的风险控制应对模型失效
4. 持续的监控与优化适应市场变化

随着机器学习技术的发展，配对交易策略将更加智能和动态，为量化投资带来新的机遇。

---

**参考资料**：
1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis."
2. Ganapathy, V. (2004). "Statistical Arbitrage and Pairs Trading."
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis."
4. 相关学术期刊和实务报告

**免责声明**：本文仅供学术交流使用，不构成任何投资建议。配对交易策略存在风险，实盘应用需谨慎评估。
