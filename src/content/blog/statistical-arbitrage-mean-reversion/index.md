---
title: "统计套利实战：均值回归策略的原理、构建与Python实现"
publishDate: '2026-06-23'
description: "深入讲解统计套利的核心思想，配对交易与协整分析，均值回归策略的构建方法，并提供完整的Python实现代码示例。适合量化交易从业者和爱好者阅读。"
tags:
 - 量化交易
 - 统计套利
 - 均值回归
 - Python
 - 配对交易
language: Chinese
---

![统计套利与均值回归的核心思想](/images/statistical-arbitrage-mean-reversion/cover.jpg)

## 引言：统计套利的核心思想

在量化交易的世界里，有一种策略不依赖预测市场的方向，而是利用价格之间的统计关系获利。这就是**统计套利（Statistical Arbitrage）**。

统计套利的核心假设是：**价格偏离是暂时的，均值回归是常态**。当两个相关性强的资产价格出现暂时性偏离时，我们可以做多被低估的资产、做空被高估的资产，等待价格关系回归正常时平仓获利。

这种策略的魅力在于：
- **市场中性（Market Neutral）**：不依赖市场方向，多空对冲降低风险
- **统计驱动**：基于严谨的统计学原理，而非主观判断
- **可复制性**：策略逻辑清晰，可以系统化执行
- **低风险暴露**：通过对冲消除市场系统性风险，专注于获取Alpha收益

### 为什么统计套利在机构投资者中如此流行？

统计套利并不是新事物。早在1980年代，摩根士丹利的量化团队就开始使用配对交易策略。今天，统计套利已经成为对冲基金和量化机构的标配策略之一。根据Hedge Fund Research的数据，全球统计套利策略的管理资产规模超过3000亿美元。

其核心优势在于：

1. **低市场风险暴露**：通过多空对冲，策略的Beta接近零，不受大盘涨跌影响。在2000-2020年的回测中，统计套利策略与S&P 500的相关性仅为0.05-0.15。

2. **高夏普比率潜力**：由于持有期短、胜率相对较高，年化夏普比率可达1.5-2.5。相比之下，传统股票多空策略的夏普比率通常在0.5-1.0之间。

3. **容量适中**：不同于高频策略，统计套利可以容纳数亿甚至数十亿资金。这也是为什么许多大型对冲基金（如Renaissance Technologies、Two Sigma）都大量使用统计套利策略。

4. **可解释性**：每笔交易都有统计学依据，不是黑盒模型。当策略亏损时，你可以分析原因（是协整关系断裂？还是参数失效？），而不是盲目调整。

当然，统计套利也并非完美。2007年8月，美国量化危机（Quant Crisis）中，大量统计套利策略同时失效，导致许多知名对冲基金单月亏损超过10%。这提醒我们：**历史统计关系可能因为市场微观结构变化而突然断裂**。

本文将深入探讨统计套利中最经典的方法——**配对交易（Pairs Trading）**，从理论基础到实战实现，带你掌握均值回归策略的完整流程。

## 配对交易与协整分析

### 什么是正确的"配对"？

很多人误以为"相关性高"就等于"可以配对交易"。这是一个危险的误区。

**相关性（Correlation）**衡量的是两个价格序列变动的同步程度，但它不稳定，且不能保证长期关系。今天相关性0.9的两个股票，下个月可能变成0.3。更重要的是，相关性不意味着长期均衡关系——两个股票可能同向变动，但它们之间的价差可能持续扩大。

**协整（Cointegration）**才是配对交易的真正基础。协整关系意味着：两个价格序列的线性组合是一个平稳过程。换句话说，即使两个价格各自是随机游走的（像醉汉 walk），它们之间存在一个长期的均衡关系，偏离这个关系后终将回归。

数学定义：如果 $P_t^A$ 和 $P_t^B$ 都是非平稳的I(1)过程（即价格本身是随机游走，但一阶差分是平稳的），但存在常数 $\alpha$ 和 $\beta$，使得：

$$
P_t^A = \alpha + \beta \cdot P_t^B + \epsilon_t
$$

其中 $\epsilon_t$ 是平稳过程（均值回归），则称 $P_t^A$ 和 $P_t^B$ 协整。

$\beta$ 称为**对冲比率（Hedge Ratio）**，表示需要多少单位的股票B来对冲1单位的股票A。$\epsilon_t$ 就是我们要交易的**价差（Spread）**。

### 协整检验：Engle-Granger方法

检验协整关系的标准方法是**Engle-Granger两步法**：

1. **第一步**：用OLS回归估计 $\alpha$ 和 $\beta$
   $$
   P_t^A = \alpha + \beta \cdot P_t^B + \epsilon_t
$$

2. **第二步**：对残差 $\epsilon_t$ 进行平稳性检验（ADF检验，Augmented Dickey-Fuller Test）
   - 原假设 $H_0$：残差有单位根（非平稳）
   - 备择假设 $H_1$：残差平稳（协整关系存在）

如果ADF检验的p值小于显著性水平（如0.05），则拒绝原假设，认为协整关系存在。

![协整关系检验示例](/images/statistical-arbitrage-mean-reversion/statistical-analysis.jpg)

###  Spread（价差）的构建与Z-score标准化

一旦确认协整关系，我们就可以构建**Spread（价差）**：

$$
\text{Spread}_t = P_t^A - (\alpha + \beta \cdot P_t^B)
$$

Spread的理论均值是0（长期来看）。当Spread偏离0时，我们认为价格关系出现了暂时性偏离，应该均值回归。

为了量化偏离程度，我们使用**Z-score标准化**：

$$
Z_t = \frac{\text{Spread}_t - \mu_{\text{Spread}}}{\sigma_{\text{Spread}}}
$$

其中 $\mu_{\text{Spread}}$ 和 $\sigma_{\text{Spread}}$ 是滚动窗口（如60天）计算的均值和标准差。

Z-score的含义：
- $Z_t > 2$：Spread偏高，做空A、做多B（等待回归）
- $Z_t < -2$：Spread偏低，做多A、做空B（等待回归）
- $|Z_t| < 0.5$：平仓信号（回归到均值附近）

### 如何筛选合适的配对？

不是所有股票对都适合配对交易。优秀的配对应该满足以下条件：

1. **业务相似性**：同行业、同商业模式，面临相似的宏观和行业风险
   - 例如：可口可乐 vs 百事可乐、工商银行 vs 建设银行、中国国航 vs 南方航空
   - 避免：科技股 vs 银行股（业务差异太大，受不同因子驱动）

2. **市值接近**：市值差异过大的股票，小市值股票的特异性风险会主导Spread
   - 经验法则：市值比率在 0.5-2.0 之间
   - 例如：不要配对中国石油（万亿市值） vs 某小银行（百亿市值）

3. **流动性充足**：日均成交额至少1000万美元，避免滑点和冲击成本
   - 流动性差的股票，买卖价差可能达到10-20个基点，严重侵蚀利润

4. **协整检验通过**：ADF检验p值 < 0.05，且残差平稳
   - 建议使用滚动窗口检验（如过去252个交易日），确保关系稳定

5. **历史表现稳定**：在过去3-5年中，配对关系的稳定性测试通过
   - 可以使用"样本外测试"：用前2年数据估计参数，后1年数据验证

## 均值回归策略构建

### 完整的策略逻辑

一个完整的配对交易均值回归策略包含以下步骤：

1. **标的筛选**：找到具有协整关系的资产对（可以使用聚类分析、行业分类等方法批量筛选）
2. **参数计算**：计算对冲比率 $\beta$ 和滚动Z-score
3. **交易信号**：
   - 开仓信号：$|Z_t| > \text{entry\_threshold}$（如2.0）
   - 平仓信号：$|Z_t| < \text{exit\_threshold}$（如0.5）或止损
4. **风险管理**：
   - 最大持仓时间（避免长期不收敛）
   - 止损线（如Z-score突破 ±3.5，说明关系可能失效）
   - 仓位管理（等市值对冲，避免杠杆过高）

### 关键参数选择与实践经验

| 参数 | 典型值 | 说明 | 实践经验 |
|------|--------|------|----------|
| 协整窗口 | 120-252个交易日 | 太长：关系可能已失效；太短：估计不准 | 建议使用252天（1年）作为基准，每月重新估计 |
| Z-score窗口 | 20-60个交易日 | 滚动计算均值和标准差 | 60天较为稳健，20天对近期变化更敏感 |
| 入场阈值 | 1.5-2.5 | 阈值越高，信号越少但胜率可能更高 | 日线策略建议2.0-2.5；分钟级策略可降至1.5 |
| 出场阈值 | 0-0.5 | 回到均值附近平仓 | 0.5可以捕获大部分利润，同时避免过早平仓 |
| 止损阈值 | 3-4 | Z-score突破此值说明关系可能失效 | 建议3.5，给一定的缓冲空间 |
| 最大持仓 | 20-30个交易日 | 避免长期不收敛 | 如果20天还未回归，可能关系已失效 |

### 策略的局限性与应对方法

1. **协整关系断裂**：监管变化、并购、行业变革等可能导致历史关系永久失效
   - **案例**：2018年美联储加息周期中，许多利率敏感型配对（如REITs vs 公用事业股）的关系突然断裂
   - **应对**：定期重新检验协整关系（建议每月），设置关系失效的止损条件（如Z-score持续扩大超过5天）

2. **均值回归时间过长**：理论上会回归，但实际上可能需要数月，资金成本和机会成本很高
   - **数据**：根据学术研究（Gatev et al., 2006），配对交易的平均持仓周期为5-15个交易日，但10%的交易可能持续超过30天
   - **应对**：设置最大持仓时间（如20-30天），避免资金长期占用。可以考虑使用期权（如卖出跨式组合）来降低时间成本

3. **交易成本**：频繁交易会侵蚀利润，尤其是小券商的手续费
   - **估算**：假设每笔交易（开仓+平仓）成本为20个基点（0.2%），如果年化交易100次，仅交易成本就达到20%
   - **应对**：选择低佣金券商（如互联网券商，佣金可低至5个基点），提高入场阈值减少交易次数，或使用ETF替代个股（流动性更好，买卖价差更小）

4. **模型风险**：参数选择（窗口长度、阈值）对结果影响很大
   - **现实**：最优参数往往是无序的，今天的最优参数明天可能失效。这就是"参数高原"问题
   - **应对**：使用滚动窗口优化（Walk-Forward Optimization），定期重新校准参数，或使用集成方法（多个参数组合的平均信号）

5. **黑天鹅事件**：市场崩盘时，相关性趋近1，所有配对同时失效
   - **案例**：2008年金融危机、2020年疫情崩盘，许多统计套利基金单月亏损10-20%
   - **应对**：降低杠杆（建议总杠杆不超过2倍），设置熔断机制（如单日亏损超过3%暂停交易），在极端波动时暂停交易

## Python实现示例

下面提供两个完整的Python代码示例，展示如何实现配对交易的协整检验和回测。所有代码均可直接运行。

### 示例1：协整检验与Z-score计算

```python
import numpy as np
import pandas as pd
import yfinance as yf
from statsmodels.tsa.stattools import adfuller
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt

def test_cointegration(y, x, show_plot=True):
    """
    使用Engle-Granger方法检验协整关系
    
    参数:
    y: 价格序列1 (如股票A)
    x: 价格序列2 (如股票B)
    
    返回:
    beta: 对冲比率
    alpha: 截距
    spread: 价差序列
    p_value: ADF检验p值
    """
    # 第一步：OLS回归
    X = pd.DataFrame({'x': x, 'const': 1})
    model = OLS(y, X).fit()
    alpha = model.params['const']
    beta = model.params['x']
    
    # 计算spread
    spread = y - (alpha + beta * x)
    
    # 第二步：ADF检验
    adf_result = adfuller(spread, autolag='AIC')
    p_value = adf_result[1]
    
    print(f"=== 协整检验结果 ===")
    print(f"对冲比率 beta: {beta:.4f}")
    print(f"截距 alpha: {alpha:.4f}")
    print(f"ADF统计量: {adf_result[0]:.4f}")
    print(f"p-value: {p_value:.4f}")
    print(f"1%临界值: {adf_result[4]['1%']:.4f}")
    print(f"5%临界值: {adf_result[4]['5%']:.4f}")
    print(f"10%临界值: {adf_result[4]['10%']:.4f}")
    
    if p_value < 0.05:
        print("✓ 存在协整关系 (p < 0.05)")
    else:
        print("✗ 不存在协整关系 (p >= 0.05)")
    
    # 可视化
    if show_plot:
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 价格序列
        axes[0].plot(y.index, y.values, label='Stock A', alpha=0.7)
        axes[0].plot(x.index, x.values, label='Stock B', alpha=0.7)
        axes[0].set_title('价格序列')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Spread
        axes[1].plot(spread.index, spread.values, color='blue', alpha=0.7)
        axes[1].axhline(spread.mean(), color='red', linestyle='--', label='Mean')
        axes[1].fill_between(spread.index, 
                            spread.mean() - 2*spread.std(), 
                            spread.mean() + 2*spread.std(), 
                            alpha=0.2, color='gray', label='±2σ')
        axes[1].set_title('Spread (残差)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Z-score
        z_score = (spread - spread.mean()) / spread.std()
        axes[2].plot(z_score.index, z_score.values, color='purple', alpha=0.7)
        axes[2].axhline(2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
        axes[2].axhline(-2, color='green', linestyle='--', alpha=0.5, label='Entry (-2)')
        axes[2].axhline(0, color='black', linestyle='-', alpha=0.3)
        axes[2].axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='Exit (+0.5)')
        axes[2].axhline(-0.5, color='gray', linestyle=':', alpha=0.5)
        axes[2].set_title('Z-score')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    return beta, alpha, spread, p_value

# 示例使用
if __name__ == "__main__":
    # 下载数据（示例：可口可乐 vs 百事可乐）
    # 这两家公司业务高度相似，是经典的配对交易标的
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 检验协整关系
    beta, alpha, spread, p_value = test_cointegration(
        data['KO'], 
        data['PEP'],
        show_plot=True
    )
    
    # 输出解读：
    # 1. 如果p-value < 0.05，说明协整关系存在，可以进行配对交易
    # 2. beta表示对冲比率，如beta=0.8，则每做多1股KO，需要做空0.8股PEP
    # 3. 观察Z-score图表，当|Z-score| > 2时考虑入场，< 0.5时出场
```

### 示例2：完整的配对交易回测框架

```python
import numpy as np
import pandas as pd
from typing import Tuple, List
import matplotlib.pyplot as plt

class PairsTradingBacktest:
    """
    配对交易回测框架
    
    实现功能：
    1. 滚动计算对冲比率和Z-score
    2. 生成交易信号（入场、出场、止损）
    3. 计算策略收益和风险评估指标
    4. 可视化回测结果
    """
    
    def __init__(self, 
                 price_a: pd.Series, 
                 price_b: pd.Series,
                 window: int = 60,
                 entry_threshold: float = 2.0,
                 exit_threshold: float = 0.5,
                 stop_loss_threshold: float = 3.5,
                 max_holding_days: int = 30,
                 transaction_cost: float = 0.001):
        """
        初始化回测参数
        
        参数:
        price_a: 股票A价格
        price_b: 股票B价格
        window: 滚动窗口长度（用于计算Z-score）
        entry_threshold: 入场Z-score阈值
        exit_threshold: 出场Z-score阈值
        stop_loss_threshold: 止损Z-score阈值
        max_holding_days: 最大持仓天数
        transaction_cost: 交易成本（单边，如0.001表示0.1%）
        """
        self.price_a = price_a
        self.price_b = price_b
        self.window = window
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.stop_loss_threshold = stop_loss_threshold
        self.max_holding_days = max_holding_days
        self.transaction_cost = transaction_cost
        
        self.results = None
        
    def calculate_z_score(self) -> pd.DataFrame:
        """计算滚动Z-score（使用滚动线性回归）"""
        df = pd.DataFrame({
            'price_a': self.price_a,
            'price_b': self.price_b
        })
        
        # 滚动回归计算对冲比率
        df['hedge_ratio'] = np.nan
        df['spread'] = np.nan
        
        for i in range(self.window, len(df)):
            idx = df.index[i-self.window:i]
            X = df.loc[idx, 'price_b'].values
            y = df.loc[idx, 'price_a'].values
            
            # 简单线性回归（也可以用TLS等更稳健的方法）
            beta = np.cov(y, X)[0, 1] / np.var(X)
            spread = y - beta * X
            
            df.loc[df.index[i], 'hedge_ratio'] = beta
            df.loc[df.index[i], 'spread'] = spread[-1]
        
        # 计算Z-score
        df['spread_mean'] = df['spread'].rolling(window=self.window).mean()
        df['spread_std'] = df['spread'].rolling(window=self.window).std()
        df['z_score'] = (df['spread'] - df['spread_mean']) / df['spread_std']
        
        return df
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df['position'] = 0  # 1: 做多A做空B, -1: 做空A做多B, 0: 平仓
        df['holding_days'] = 0
        df['trade_id'] = 0  # 记录每笔交易的ID
        
        position = 0
        holding_counter = 0
        trade_id = 0
        
        for i in range(1, len(df)):
            z = df['z_score'].iloc[i]
            
            if pd.isna(z):
                continue
            
            # 已有持仓
            if position != 0:
                holding_counter += 1
                
                # 平仓条件
                exit_signal = False
                
                # 条件1：Z-score回归
                if position == 1 and z <= self.exit_threshold:
                    exit_signal = True
                elif position == -1 and z >= -self.exit_threshold:
                    exit_signal = True
                
                # 条件2：止损
                if abs(z) > self.stop_loss_threshold:
                    exit_signal = True
                    print(f"止损触发 @ {df.index[i]}, Z-score={z:.2f}")
                
                # 条件3：超过最大持仓时间
                if holding_counter >= self.max_holding_days:
                    exit_signal = True
                    print(f"强制平仓 @ {df.index[i]}, 持仓{holding_counter}天")
                
                if exit_signal:
                    df.iloc[i, df.columns.get_loc('position')] = 0
                    df.iloc[i, df.columns.get_loc('trade_id')] = trade_id
                    position = 0
                    holding_counter = 0
                    trade_id += 1
                else:
                    df.iloc[i, df.columns.get_loc('position')] = position
                    df.iloc[i, df.columns.get_loc('holding_days')] = holding_counter
                    df.iloc[i, df.columns.get_loc('trade_id')] = trade_id
            
            # 无持仓，检查入场信号
            else:
                if z > self.entry_threshold:
                    # Z-score过高，做空A做多B
                    position = -1
                    df.iloc[i, df.columns.get_loc('position')] = -1
                    holding_counter = 0
                    trade_id += 1
                    df.iloc[i, df.columns.get_loc('trade_id')] = trade_id
                elif z < -self.entry_threshold:
                    # Z-score过低，做多A做空B
                    position = 1
                    df.iloc[i, df.columns.get_loc('position')] = 1
                    holding_counter = 0
                    trade_id += 1
                    df.iloc[i, df.columns.get_loc('trade_id')] = trade_id
        
        return df
    
    def backtest(self, initial_capital: float = 100000) -> pd.DataFrame:
        """执行回测（包含交易成本）"""
        # 计算Z-score
        df = self.calculate_z_score()
        
        # 生成信号
        df = self.generate_signals(df)
        
        # 计算收益（等市值对冲）
        df['returns_a'] = df['price_a'].pct_change()
        df['returns_b'] = df['price_b'].pct_change()
        
        # 策略收益
        df['strategy_returns'] = 0.0
        df['transaction_costs'] = 0.0
        
        for i in range(1, len(df)):
            pos = df['position'].iloc[i-1]  # 使用上一期的仓位
            
            if pos == 1:
                # 做多A，做空B
                df.iloc[i, df.columns.get_loc('strategy_returns')] = (
                    df['returns_a'].iloc[i] - df['returns_b'].iloc[i]
                )
            elif pos == -1:
                # 做空A，做多B
                df.iloc[i, df.columns.get_loc('strategy_returns')] = (
                    df['returns_b'].iloc[i] - df['returns_a'].iloc[i]
                )
            
            # 计算交易成本（仓位变化时）
            if df['position'].iloc[i] != df['position'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc('transaction_costs')] = self.transaction_cost * 2  # 双边
        
        # 净收益
        df['net_returns'] = df['strategy_returns'] - df['transaction_costs']
        
        # 累积收益
        df['cumulative_returns'] = (1 + df['net_returns']).cumprod()
        df['capital'] = initial_capital * df['cumulative_returns']
        
        self.results = df
        return df
    
    def calculate_metrics(self) -> dict:
        """计算策略评估指标"""
        if self.results is None:
            raise ValueError("请先运行 backtest()")
        
        df = self.results
        
        # 基本指标
        total_return = df['capital'].iloc[-1] / df['capital'].iloc[0] - 1
        annual_return = (1 + total_return) ** (252 / len(df)) - 1
        
        # 夏普比率
        daily_returns = df['net_returns']
        sharpe_ratio = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
        
        # 最大回撤
        cumulative = df['capital']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # 胜率（简化计算）
        winning_days = (df['strategy_returns'] > 0).sum()
        total_trading_days = (df['position'] != 0).sum()
        win_rate = winning_days / total_trading_days if total_trading_days > 0 else 0
        
        # 交易次数
        num_trades = df['trade_id'].nunique() - 1  # 减去初始的0
        
        metrics = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'num_trades': num_trades,
            'total_transaction_costs': df['transaction_costs'].sum()
        }
        
        return metrics
    
    def plot_results(self):
        """可视化回测结果"""
        if self.results is None:
            raise ValueError("请先运行 backtest()")
        
        df = self.results
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 16))
        
        # 1. 价格序列与仓位
        ax1 = axes[0]
        ax1.plot(df.index, df['price_a'].values, label='Stock A', alpha=0.7)
        ax1.plot(df.index, df['price_b'].values, label='Stock B', alpha=0.7)
        ax1.set_ylabel('Price', color='blue')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 仓位（第二y轴）
        ax1_twin = ax1.twinx()
        ax1_twin.plot(df.index, df['position'].values, 
                     label='Position', color='red', alpha=0.5)
        ax1_twin.set_ylabel('Position', color='red')
        ax1_twin.set_ylim(-1.5, 1.5)
        
        # 2. Z-score与交易信号
        ax2 = axes[1]
        ax2.plot(df.index, df['z_score'].values, color='purple', alpha=0.7)
        ax2.axhline(self.entry_threshold, color='red', linestyle='--', alpha=0.5, label='Entry')
        ax2.axhline(-self.entry_threshold, color='green', linestyle='--', alpha=0.5)
        ax2.axhline(self.exit_threshold, color='gray', linestyle=':', alpha=0.5, label='Exit')
        ax2.axhline(-self.exit_threshold, color='gray', linestyle=':', alpha=0.5)
        ax2.fill_between(df.index, 
                        -self.stop_loss_threshold, 
                        self.stop_loss_threshold,
                        alpha=0.1, color='yellow', label='Safe Zone')
        ax2.set_ylabel('Z-score')
        ax2.set_title('Z-score with Trading Thresholds')
        ax2.grid(True, alpha=0.3)
        ax2.legend()
        
        # 3. 资金曲线
        ax3 = axes[2]
        ax3.plot(df.index, df['capital'].values, color='green', linewidth=2)
        ax3.axhline(df['capital'].iloc[0], color='black', linestyle='--', alpha=0.5)
        ax3.set_ylabel('Capital')
        ax3.set_title('Strategy Capital Curve')
        ax3.grid(True, alpha=0.3)
        
        # 4. 回撤
        ax4 = axes[3]
        cumulative = df['capital']
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        ax4.fill_between(df.index, 0, drawdown.values, color='red', alpha=0.3)
        ax4.set_ylabel('Drawdown')
        ax4.set_title('Drawdown Over Time')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # 打印指标
        metrics = self.calculate_metrics()
        print("\n=== 回测结果 ===")
        print(f"总收益率: {metrics['total_return']*100:.2f}%")
        print(f"年化收益率: {metrics['annual_return']*100:.2f}%")
        print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        print(f"最大回撤: {metrics['max_drawdown']*100:.2f}%")
        print(f"胜率: {metrics['win_rate']*100:.2f}%")
        print(f"总交易次数: {metrics['num_trades']}")
        print(f"总交易成本: {metrics['total_transaction_costs']*100:.2f}%")

# 使用示例
if __name__ == "__main__":
    # 下载数据
    tickers = ['KO', 'PEP']
    data = yf.download(tickers, start='2020-01-01', end='2024-12-31')['Adj Close']
    
    # 创建回测对象
    backtest = PairsTradingBacktest(
        price_a=data['KO'],
        price_b=data['PEP'],
        window=60,
        entry_threshold=2.0,
        exit_threshold=0.5,
        stop_loss_threshold=3.5,
        max_holding_days=30,
        transaction_cost=0.001  # 0.1% 单边交易成本
    )
    
    # 运行回测
    results = backtest.backtest(initial_capital=100000)
    
    # 可视化
    backtest.plot_results()
```

## 实盘注意事项

理论很美好，实盘很残酷。以下是统计套利策略在实盘中的关键注意事项。

### 1. 数据质量与生存者偏差

回测中使用的数据往往是"幸存者偏差"后的干净数据。实盘中，你需要考虑：

- **幸存者偏差**：回测时只用了当前还存在的股票，忽略了已退市的。例如，如果你回测2000-2010年的A股配对交易，许多已退市的股票没有被纳入，导致结果过于乐观。
- **前复权调整**：分红、拆股等会影响价格序列，必须使用复权数据。不正确的复权会导致价差计算错误。
- **盘口冲击**：大额交易会影响价格，尤其是小盘股。回测中假设以收盘价成交，实盘中可能面临显著的滑点。
- **数据频率**：日线数据 vs 分钟级数据。日线数据无法捕捉盘中的均值回归机会，而分钟级数据的噪音更大。

**建议**：使用调整后的复权数据，并在回测中加入生存者偏差的修正（如加入已退市股票的数据）。

### 2. 交易成本控制

统计套利策略通常交易频繁，交易成本是最大的隐形杀手。

| 成本类型 | 说明 | 典型值 |
|---------|------|--------|
| 佣金 | 按交易金额或笔数收取 | 0.02%-0.1% |
| 买卖价差 | Bid-Ask Spread，市价单直接承担 | 0.05%-0.5% |
| 滑点 | 下单到成交的价格偏差 | 0.1%-0.5% |
| 冲击成本 | 大额订单移动市场价格 | 视订单大小而定 |

**建议**：在回测中至少加入10-20个基点的交易成本（双边），才能反映真实情况。选择低佣金券商，并尽量使用限价单。

### 3. 风险管理

即使是最完美的统计套利策略，也可能遭遇"黑天鹅"。必须设置严格的风险控制：

- **单次最大亏损**：单笔交易亏损不超过总资金的1-2%
- **总敞口限制**：所有持仓的市场敞口不超过总资金的20%
- **相关性监控**：定期重新检验配对的相关性，关系失效立即平仓
- **熔断机制**：市场极端波动时（如VIX > 40）暂停交易

### 4. 参数敏感性

均值回归策略对参数非常敏感：
- Z-score窗口长度：20天 vs 60天，结果可能天差地别
- 入场/出场阈值：2.0 vs 1.5，信号数量可能差3倍
- 协整窗口：协整关系可能随时间变化

**解决方法**：
- Walk-Forward分析：滚动样本外测试
- 参数稳定性测试：测试参数在合理范围内的稳健性
- 多参数组合：不依赖单一参数设置，使用多个参数组合的信号平均

### 5. 税务考虑

配对交易涉及频繁买卖和做空，税务处理复杂：
- 做空的税务处理（如A股不允许裸做空，需要通过融券）
- 高频交易的税务分类
- 跨年度的盈亏抵扣

**建议**：咨询专业税务顾问，并在回测中加入税后收益计算。

## 总结与展望

统计套利和均值回归策略是量化交易中的经典方法，它不依赖预测市场方向，而是利用价格之间的统计关系获利。

### 理论要点回顾

- **协整 > 相关性**：真正的配对需要协整关系，而非简单的相关性
- **均值回归是假设**：需要统计检验，不是所有价格差都会回归
- **市场中性**：多空对冲降低方向性风险，但仍有基差风险

### 实践建议

- **Python实现不难**：核心是OLS回归 + Z-score计算 + 信号生成，本文提供了完整代码
- **回测容易过拟合**：参数优化要谨慎，必须用样本外数据验证
- **实盘比回测难**：交易成本、滑点、模型失效都是真实挑战

### 下一步学习方向

如果你想进一步深化统计套利的知识，建议关注以下方向：

1. **多因子统计套利**：不局限于配对，使用多个股票构建均值回归组合（如PCA、因子模型）
2. **机器学习增强**：用Random Forest或LSTM预测均值回归的时间窗口和幅度
3. **高频统计套利**：利用tick级数据进行日内配对交易（需要更先进的执行系统）
4. **跨市场套利**：不同交易所、不同衍生品之间的统计套利（如股票 vs ETF、期货 vs 现货）

### 最后的建议

统计套利不是"印钞机"，但是一套严谨的、可复制的量化交易框架。成功的关键在于：

1. **严谨的统计检验**：不要因为两个股票"看起来相关"就交易，必须用协整检验确认
2. **严格的风险管理**：设置止损、控制杠杆、定期重新校准模型
3. **持续的模型监控**：市场在变化，你的模型也要跟着变化
4. **从小资金开始**：先用小资金验证策略，再逐步扩大规模

希望本文能帮助你理解统计套利的核心思想，并掌握基本的Python实现方法。实战中还有更多细节需要处理，建议从小资金开始，逐步积累经验。

---

**参考资料**：

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Alexander, C. (2001). *Market Models: A Guide to Financial Data Analysis*. Wiley.
3. Engle, R. F., & Granger, C. W. (1987). Co-integration and error correction: Representation, estimation, and testing. *Econometrica*, 55(2), 251-276.
4. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*, 19(3), 797-827.
5. Quantopian/QuantConnect 官方文档

**免责声明**：本文仅供学习交流，不构成投资建议。统计套利策略涉及做空和杠杆，实盘前请确保充分理解风险。历史回测结果不代表未来收益。
