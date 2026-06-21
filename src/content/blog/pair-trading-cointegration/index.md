---
title: "配对交易与协整分析"
description: "深入探讨配对交易策略的理论基础与实践方法，学习如何运用协整分析识别配对机会，构建均值回归交易策略。包含完整的Python实现与实证分析。"
pubDate: 2026-06-21
updateDate: 2026-06-21
heroImage: "/images/pair-trading-cointegration/hero.png"
heroAlt: "配对交易策略示意图"
tags: ["配对交易", "协整分析", "均值回归", "统计套利"]
---

# 配对交易与协整分析

## 引言

在量化投资领域，**配对交易（Pairs Trading）**是一种经典的市场中性策略，它通过寻找两个高度相关的资产，在价格偏离长期均衡关系时进行反向操作，从而在市场无论涨跌的情况下都能获取稳定收益。

配对交易的核心在于**协整关系（Cointegration）**的识别与利用。与简单的相关系数不同，协整关系揭示的是两个时间序列在长期内的均衡关系，这种关系为均值回归策略提供了坚实的理论基础。

本文将系统介绍配对交易的理论框架、协整检验方法、实战策略构建，并提供完整的Python代码实现。

## 理论基础

### 1. 平稳性与协整

在介绍协整之前，必须先理解**平稳性（Stationarity）**的概念。

一个时间序列 $\{Y_t\}$ 是平稳的，如果它满足：
1. 均值恒定：$\mathbb{E}[Y_t] = \mu$（常数）
2. 方差恒定：$\text{Var}(Y_t) = \sigma^2$（常数）
3. 自协方差只依赖于时滞：$\text{Cov}(Y_t, Y_{t-k}) = \gamma_k$（常数）

**为什么平稳性重要？**
- 非平稳序列的统计性质会随时间变化，导致传统的统计推断失效
- 回归分析要求残差平稳，否则会出现"伪回归"问题

### 2. 协整的定义

对于两个非平稳的时间序列 $\{X_t\}$ 和 $\{Y_t\}$（通常是I(1)过程，即一阶单整），如果存在一组系数 $\alpha$ 和 $\beta$，使得线性组合：

$$
Z_t = Y_t - (\alpha + \beta X_t)
$$

是平稳的（I(0)过程），则称 $X_t$ 和 $Y_t$ 是**协整的**。

直观理解：
- $X_t$ 和 $Y_t$ 各自会随机游走，但它们的某种线性组合是稳定的
- 这意味着两者之间存在长期的均衡关系
- 当价格偏离这个均衡时，会产生均值回归的力量

### 3. 配对交易的直觉

假设我们找到了两只协整的股票A和B：
- 长期来看，它们的价格比值（或价差）围绕某个均值波动
- 当价格比值过高时，卖出A、买入B
- 当价格比值过低时，买入A、卖出B
- 等待价格回归均值，平仓获利

**关键优势：**
- 市场中性：多空对冲，不受大盘方向影响
- 均值回归：利用统计学规律，而非方向性预测
- 风险可控：基于统计套利，理论上风险有限

## 协整检验方法

### 方法一：Engle-Granger两步法

这是最经典的协整检验方法，分为两步：

**第一步：** 用OLS估计协整向量
$$
Y_t = \alpha + \beta X_t + \epsilon_t
$$

**第二步：** 对残差 $\hat{\epsilon}_t$ 进行单位根检验（如ADF检验）

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

class EngleGrangerTest:
    """
    Engle-Granger两步法协整检验
    """
    
    def __init__(self, significance_level=0.05):
        """
        初始化
        
        Parameters
        ----------
        significance_level : float
            显著性水平（默认0.05）
        """
        self.significance_level = significance_level
        self.alpha = None
        self.beta = None
        self.residuals = None
        
    def fit(self, Y, X, verbose=True):
        """
        执行协整检验
        
        Parameters
        ----------
        Y : pd.Series
            因变量（通常是价格序列）
        X : pd.Series
            自变量（通常是价格序列）
        verbose : bool
            是否输出详细信息
            
        Returns
        -------
        is_cointegrated : bool
            是否存在协整关系
        p_value : float
            ADF检验的p值
        """
        # 第一步：OLS回归
        X_with_const = pd.concat([pd.Series(1, index=X.index, name='const'), X], axis=1)
        model = OLS(Y, X_with_const).fit()
        
        self.alpha = model.params['const']
        self.beta = model.params[X.name]
        self.residuals = model.resid
        
        if verbose:
            print("=" * 60)
            print("第一步：OLS回归结果")
            print("=" * 60)
            print(f"截距项 (alpha): {self.alpha:.4f}")
            print(f"斜率项 (beta): {self.beta:.4f}")
            print(f"R²: {model.rsquared:.4f}")
            print(f"调整R²: {model.rsquared_adj:.4f}")
        
        # 第二步：残差的单位根检验
        adf_result = adfuller(self.residuals, autolag='AIC')
        
        adf_statistic = adf_result[0]
        p_value = adf_result[1]
        critical_values = adf_result[4]
        
        if verbose:
            print("\n" + "=" * 60)
            print("第二步：ADF检验（残差平稳性检验）")
            print("=" * 60)
            print(f"ADF统计量: {adf_statistic:.4f}")
            print(f"p-value: {p_value:.4f}")
            print("临界值:")
            for key, value in critical_values.items():
                print(f"  {key}: {value:.4f}")
        
        # 判断是否协整
        is_cointegrated = p_value < self.significance_level
        
        if verbose:
            print("\n" + "=" * 60)
            print("结论")
            print("=" * 60)
            if is_cointegrated:
                print("✓ 存在协整关系（拒绝单位根假设）")
            else:
                print("✗ 不存在协整关系（无法拒绝单位根假设）")
        
        return is_cointegrated, p_value
    
    def plot_residuals(self, title="残差序列图"):
        """
        绘制残差序列图
        
        Parameters
        ----------
        title : str
            图表标题
        """
        if self.residuals is None:
            raise ValueError("请先执行 fit() 方法")
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 8))
        
        # 上图：残差时间序列
        axes[0].plot(self.residuals.index, self.residuals.values, 'b-', linewidth=1)
        axes[0].axhline(y=0, color='r', linestyle='--', alpha=0.5)
        axes[0].axhline(y=self.residuals.mean() + 2*self.residuals.std(), 
                        color='g', linestyle=':', alpha=0.5, label='±2σ')
        axes[0].axhline(y=self.residuals.mean() - 2*self.residuals.std(), 
                        color='g', linestyle=':', alpha=0.5)
        axes[0].set_title(title)
        axes[0].set_xlabel("日期")
        axes[0].set_ylabel("残差")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 下图：残差分布直方图
        axes[1].hist(self.residuals.values, bins=50, edgecolor='black', alpha=0.7)
        axes[1].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        axes[1].axvline(x=self.residuals.mean(), color='g', linestyle='-', 
                        alpha=0.5, label=f'均值: {self.residuals.mean():.4f}')
        axes[1].set_title("残差分布")
        axes[1].set_xlabel("残差值")
        axes[1].set_ylabel("频次")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
```

### 方法二：Johansen检验

Johansen检验是一种更强大的协整检验方法，它可以：
1. 检验多个时间序列之间的协整关系
2. 确定协整向量的数量
3. 提供更稳健的检验结果

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

class JohansenTest:
    """
    Johansen协整检验
    """
    
    def __init__(self, significance_level=0.05):
        """
        初始化
        
        Parameters
        ----------
        significance_level : float
            显著性水平
        """
        self.significance_level = significance_level
        
    def fit(self, data, det_order=0, k_ar_diff=1, verbose=True):
        """
        执行Johansen检验
        
        Parameters
        ----------
        data : pd.DataFrame
            多个时间序列组成的DataFrame
        det_order : int
            确定性项的顺序（0=无常数项，1=有常数项，-1=有线性趋势）
        k_ar_diff : int
            差分项的滞后阶数
        verbose : bool
            是否输出详细信息
            
        Returns
        -------
        coint_rank : int
            协整向量的数量
        trace_statistic : np.array
            Trace统计量
        max_eig_statistic : np.array
            Max-Eigen统计量
        """
        # 选择协整秩
        coint_rank = select_coint_rank(data, det_order, k_ar_diff)
        
        if verbose:
            print("=" * 60)
            print("Johansen协整检验")
            print("=" * 60)
            print(f"协整向量数量: {coint_rank}")
            print(f"数据维度: {data.shape}")
            print(f"检验期间: {data.index[0]} 至 {data.index[-1]}")
        
        # 估计VECM模型
        vecm_model = VECM(data, k_ar_diff=k_ar_diff, coint_rank=coint_rank, deterministic=det_order)
        vecm_result = vecm_model.fit()
        
        if verbose:
            print("\n" + "=" * 60)
            print("VECM模型结果")
            print("=" * 60)
            print(vecm_result.summary())
        
        return coint_rank, vecm_result
```

## 配对交易策略构建

### 1. 信号生成

基于协整关系的配对交易信号通常基于**价差（Spread）**或**Z-Score**：

```python
class PairsTradingStrategy:
    """
    配对交易策略
    """
    
    def __init__(self, entry_zscore=2.0, exit_zscore=0.5, 
                 stop_loss_zscore=3.0, lookback_window=60):
        """
        初始化
        
        Parameters
        ----------
        entry_zscore : float
            入场信号的Z-Score阈值
        exit_zscore : float
            出场信号的Z-Score阈值
        stop_loss_zscore : float
            止损Z-Score阈值
        lookback_window : int
            计算均值的滚动窗口
        """
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_loss_zscore = stop_loss_zscore
        self.lookback_window = lookback_window
        
    def calculate_spread(self, price_Y, price_X, beta):
        """
        计算价差
        
        Parameters
        ----------
        price_Y : pd.Series
            股票Y的价格
        price_X : pd.Series
            股票X的价格
        beta : float
            协整系数
            
        Returns
        -------
        spread : pd.Series
            价差序列
        """
        spread = price_Y - beta * price_X
        return spread
    
    def calculate_zscore(self, spread):
        """
        计算价差的Z-Score
        
        Parameters
        ----------
        spread : pd.Series
            价差序列
            
        Returns
        -------
        zscore : pd.Series
            Z-Score序列
        """
        zscore = pd.Series(index=spread.index, dtype=float)
        
        for i in range(self.lookback_window, len(spread)):
            window = spread.iloc[i-self.lookback_window:i]
            mean = window.mean()
            std = window.std()
            
            if std > 0:
                zscore.iloc[i] = (spread.iloc[i] - mean) / std
            else:
                zscore.iloc[i] = 0
        
        return zscore
    
    def generate_signals(self, zscore):
        """
        生成交易信号
        
        Parameters
        ----------
        zscore : pd.Series
            Z-Score序列
            
        Returns
        -------
        signals : pd.DataFrame
            交易信号（1=做多Y做空X，-1=做空Y做多X，0=无仓位）
        """
        signals = pd.DataFrame(
            index=zscore.index,
            columns=['position', 'action'],
            data=0
        )
        
        position = 0  # 当前仓位：1=多Y空X，-1=空Y多X，0=无
        
        for i in range(len(zscore)):
            if pd.isna(zscore.iloc[i]):
                continue
                
            z = zscore.iloc[i]
            
            # 入场信号
            if position == 0:
                if z > self.entry_zscore:
                    # Z-Score过高，做空Y做多X
                    position = -1
                    signals.iloc[i] = [position, 'open']
                elif z < -self.entry_zscore:
                    # Z-Score过低，做多Y做空X
                    position = 1
                    signals.iloc[i] = [position, 'open']
            
            # 出场信号
            elif position != 0:
                if abs(z) < self.exit_zscore:
                    # Z-Score回归，平仓
                    signals.iloc[i] = [0, 'close']
                    position = 0
                
                elif abs(z) > self.stop_loss_zscore:
                    # 触发止损
                    signals.iloc[i] = [0, 'stop_loss']
                    position = 0
            
            # 更新仓位
            if i < len(signals) - 1:
                signals.iloc[i+1, 0] = position
        
        return signals
```

### 2. 回测框架

有了信号之后，需要构建回测框架来评估策略表现：

```python
class PairsBacktester:
    """
    配对交易回测框架
    """
    
    def __init__(self, initial_capital=1000000, transaction_cost=0.001):
        """
        初始化
        
        Parameters
        ----------
        initial_capital : float
            初始资金
        transaction_cost : float
            交易费率（单边）
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        
    def backtest(self, price_Y, price_X, signals, beta, verbose=True):
        """
        执行回测
        
        Parameters
        ----------
        price_Y : pd.Series
            股票Y的价格
        price_X : pd.Series
            股票X的价格
        signals : pd.DataFrame
            交易信号
        beta : float
            协整系数
        verbose : bool
            是否输出详细信息
            
        Returns
        -------
        results : dict
            回测结果
        """
        # 初始化投资组合状态
        portfolio = pd.DataFrame(
            index=price_Y.index,
            columns=['cash', 'position_Y', 'position_X', 'value_Y', 
                     'value_X', 'total_value', 'returns', 'cumulative_returns']
        )
        
        portfolio['cash'] = self.initial_capital
        portfolio['position_Y'] = 0
        portfolio['position_X'] = 0
        portfolio['value_Y'] = 0.0
        portfolio['value_X'] = 0.0
        portfolio['total_value'] = self.initial_capital
        portfolio['returns'] = 0.0
        portfolio['cumulative_returns'] = 0.0
        
        # 交易记录
        trades = []
        
        for i in range(1, len(portfolio)):
            date = portfolio.index[i]
            prev_date = portfolio.index[i-1]
            
            # 复制上一期的仓位和现金
            portfolio.loc[date, 'position_Y'] = portfolio.loc[prev_date, 'position_Y']
            portfolio.loc[date, 'position_X'] = portfolio.loc[prev_date, 'position_X']
            portfolio.loc[date, 'cash'] = portfolio.loc[prev_date, 'cash']
            
            # 检查是否有交易信号
            if date in signals.index:
                signal_row = signals.loc[date]
                
                if signal_row['action'] == 'open':
                    # 开仓
                    position = signal_row['position']
                    
                    # 计算交易数量（等价值交易）
                    total_value = portfolio.loc[prev_date, 'total_value']
                    trade_value = total_value * 0.5  # 每只股票占50%
                    
                    if position == 1:  # 做多Y，做空X
                        n_Y = int(trade_value / price_Y.loc[date])
                        n_X = int(trade_value / price_X.loc[date])
                        
                        # 更新仓位
                        portfolio.loc[date, 'position_Y'] = n_Y
                        portfolio.loc[date, 'position_X'] = -n_X  # 做空
                        
                        # 更新现金（扣除交易成本）
                        cost = (n_Y * price_Y.loc[date] + n_X * price_X.loc[date]) * self.transaction_cost
                        portfolio.loc[date, 'cash'] -= cost
                        
                        trades.append({
                            'date': date,
                            'action': 'open_long_Y_short_X',
                            'n_Y': n_Y,
                            'n_X': n_X,
                            'price_Y': price_Y.loc[date],
                            'price_X': price_X.loc[date],
                            'cost': cost
                        })
                        
                    elif position == -1:  # 做空Y，做多X
                        n_Y = int(trade_value / price_Y.loc[date])
                        n_X = int(trade_value / price_X.loc[date])
                        
                        portfolio.loc[date, 'position_Y'] = -n_Y  # 做空
                        portfolio.loc[date, 'position_X'] = n_X
                        
                        cost = (n_Y * price_Y.loc[date] + n_X * price_X.loc[date]) * self.transaction_cost
                        portfolio.loc[date, 'cash'] -= cost
                        
                        trades.append({
                            'date': date,
                            'action': 'open_short_Y_long_X',
                            'n_Y': n_Y,
                            'n_X': n_X,
                            'price_Y': price_Y.loc[date],
                            'price_X': price_X.loc[date],
                            'cost': cost
                        })
                
                elif signal_row['action'] in ['close', 'stop_loss']:
                    # 平仓
                    n_Y = portfolio.loc[date, 'position_Y']
                    n_X = portfolio.loc[date, 'position_X']
                    
                    # 平仓收益
                    pnl_Y = n_Y * (price_Y.loc[date] - price_Y.loc[prev_date])
                    pnl_X = n_X * (price_X.loc[date] - price_X.loc[prev_date])
                    total_pnl = pnl_Y + pnl_X
                    
                    # 更新现金
                    portfolio.loc[date, 'cash'] += total_pnl
                    
                    # 清空仓位
                    portfolio.loc[date, 'position_Y'] = 0
                    portfolio.loc[date, 'position_X'] = 0
                    
                    # 交易成本
                    trade_value = abs(n_Y * price_Y.loc[date]) + abs(n_X * price_X.loc[date])
                    cost = trade_value * self.transaction_cost
                    portfolio.loc[date, 'cash'] -= cost
                    
                    trades.append({
                        'date': date,
                        'action': signal_row['action'],
                        'n_Y': n_Y,
                        'n_X': n_X,
                        'price_Y': price_Y.loc[date],
                        'price_X': price_X.loc[date],
                        'pnl': total_pnl,
                        'cost': cost
                    })
            
            # 计算当前组合价值
            value_Y = portfolio.loc[date, 'position_Y'] * price_Y.loc[date]
            value_X = portfolio.loc[date, 'position_X'] * price_X.loc[date]
            
            portfolio.loc[date, 'value_Y'] = value_Y
            portfolio.loc[date, 'value_X'] = value_X
            portfolio.loc[date, 'total_value'] = portfolio.loc[date, 'cash'] + value_Y + value_X
            
            # 计算收益
            portfolio.loc[date, 'returns'] = (
                portfolio.loc[date, 'total_value'] / portfolio.loc[prev_date, 'total_value'] - 1
            )
        
        # 计算累计收益
        portfolio['cumulative_returns'] = (1 + portfolio['returns']).cumprod() - 1
        
        # 计算绩效指标
        total_return = portfolio['cumulative_returns'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(portfolio)) - 1
        sharpe_ratio = np.sqrt(252) * portfolio['returns'].mean() / portfolio['returns'].std()
        max_drawdown = (portfolio['total_value'] / portfolio['total_value'].cummax() - 1).min()
        
        if verbose:
            print("=" * 60)
            print("回测结果")
            print("=" * 60)
            print(f"总收益率: {total_return * 100:.2f}%")
            print(f"年化收益率: {annual_return * 100:.2f}%")
            print(f"夏普比率: {sharpe_ratio:.4f}")
            print(f"最大回撤: {max_drawdown * 100:.2f}%")
            print(f"交易次数: {len(trades)}")
        
        results = {
            'portfolio': portfolio,
            'trades': trades,
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
        
        return results
```

## 实证分析：A股配对交易案例

让我们用一个实际案例来演示配对交易策略的构建与回测。

```python
# 生成模拟数据（实际中应导入真实价格数据）
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
n_days = len(dates)

# 生成协整的价格序列
beta = 1.5
alpha = 10.0

# 共同因子
common_factor = np.cumsum(np.random.normal(0, 0.01, n_days))

# 个股特异性因子
specific_Y = np.cumsum(np.random.normal(0, 0.005, n_days))
specific_X = np.cumsum(np.random.normal(0, 0.005, n_days))

# 构建价格序列
price_X = 100 + np.cumsum(np.random.normal(0.0005, 0.02, n_days))
price_Y = alpha + beta * price_X + np.random.normal(0, 2, n_days)

# 添加均值回归特性
for i in range(1, n_days):
    spread = price_Y[i] - (alpha + beta * price_X[i])
    price_Y[i] = price_Y[i] - 0.1 * spread + np.random.normal(0, 0.5)

price_Y = pd.Series(price_Y, index=dates)
price_X = pd.Series(price_X, index=dates)

# 1. 协整检验
eg_test = EngleGrangerTest(significance_level=0.05)
is_cointegrated, p_value = eg_test.fit(price_Y, price_X, verbose=True)

if is_cointegrated:
    # 2. 计算价差和Z-Score
    strategy = PairsTradingStrategy(
        entry_zscore=2.0,
        exit_zscore=0.5,
        stop_loss_zscore=3.0,
        lookback_window=20
    )
    
    spread = strategy.calculate_spread(price_Y, price_X, eg_test.beta)
    zscore = strategy.calculate_zscore(spread)
    
    # 3. 生成交易信号
    signals = strategy.generate_signals(zscore)
    
    # 4. 回测
    backtester = PairsBacktester(
        initial_capital=1000000,
        transaction_cost=0.001
    )
    
    results = backtester.backtest(
        price_Y, price_X, signals, eg_test.beta, verbose=True
    )
    
    # 5. 可视化结果
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    
    # 上图：价格序列
    ax1 = axes[0]
    ax1.plot(price_Y.index, price_Y.values, 'b-', label='股票Y', linewidth=1)
    ax1.plot(price_X.index, price_X.values, 'r-', label='股票X', linewidth=1)
    ax1.set_title("价格序列")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中图：Z-Score
    ax2 = axes[1]
    ax2.plot(zscore.index, zscore.values, 'g-', linewidth=1)
    ax2.axhline(y=strategy.entry_zscore, color='r', linestyle='--', 
                label=f'入场阈值 (±{strategy.entry_zscore})')
    ax2.axhline(y=-strategy.entry_zscore, color='r', linestyle='--')
    ax2.axhline(y=strategy.exit_zscore, color='orange', linestyle=':', 
                label=f'出场阈值 (±{strategy.exit_zscore})')
    ax2.axhline(y=-strategy.exit_zscore, color='orange', linestyle=':')
    ax2.fill_between(zscore.index, -strategy.entry_zscore, strategy.entry_zscore, 
                     alpha=0.1, color='gray')
    ax2.set_title("Z-Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 下图：累计收益
    ax3 = axes[2]
    portfolio = results['portfolio']
    ax3.plot(portfolio.index, portfolio['cumulative_returns'], 'b-', linewidth=2)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_title("策略累计收益")
    ax3.set_xlabel("日期")
    ax3.set_ylabel("累计收益率")
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
```

## 实战中的关键问题

### 1. 配对选择的挑战

寻找真正协整的股票对是配对交易成功的关键。常用方法包括：

- **行业匹配**：同一行业的股票更可能有协整关系
- **市值匹配**：市值相近的股票价格动态更相似
- **基本面相似**：业务模式、财务杠杆等相似的股票
- **机器学习筛选**：使用聚类算法自动发现潜在配对

### 2. 参数敏感性

配对交易策略对参数选择非常敏感：

- **Lookback Window**：计算均值和标准差的窗口长度
- **Entry/Exit Threshold**：入场和出场的Z-Score阈值
- **Stop-loss Level**：止损阈值

建议使用**滚动窗口交叉验证**来优化参数，避免过拟合。

### 3. 风险管理

配对交易虽然理论上是市场中性，但实践中仍存在多种风险：

- **模型风险**：协整关系可能断裂
- **执行风险**：两只股票可能无法同时成交
- **流动性风险**：其中一只股票流动性不足
- **黑天鹅风险**：极端市场条件下相关性崩溃

## 结论

配对交易是一种经典的统计套利策略，通过协整分析识别价格之间的长期均衡关系，并从中获利。本文系统介绍了协整检验的两种主要方法（Engle-Granger和Johansen），并提供了完整的Python实现代码。

然而，配对交易并非"印钞机"。它需要深厚的统计学功底、精细的策略设计、严格的风险管理。在实践中，建议从简单配对开始，逐步扩展到多因子模型、高频交易等复杂场景。

随着A股市场有效性的提升，简单的配对交易策略超额收益正在下降。未来的发展方向包括：
- 结合机器学习方法提升配对筛选能力
- 引入高频数据捕捉更精细的均值回归机会
- 扩展到多资产、多市场的全球配对交易

## 参考资料

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797-827.

2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. John Wiley & Sons.

3. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. John Wiley & Sons.

4. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.

---

**免责声明**：本文仅供学术交流使用，不构成投资建议。配对交易存在风险，历史表现不代表未来收益。在实际应用中，请结合自身风险承受能力和投资目标，谨慎决策。
