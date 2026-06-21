---
title: "配对交易与协整分析"
date: 2026-06-21
description: "深入讲解配对交易的理论基础——协整关系，以及如何利用统计套利策略在震荡市中获取稳定收益。包含Python实战代码与回测框架。"
tags:
  - 配对交易
  - 协整分析
  - 统计套利
  - 市场中性
cover: "/images/pair-trading-cointegration/pair_prices.png"
---

# 配对交易与协整分析

## 引言

在传统趋势跟踪策略失效的**震荡市**中，如何获取稳定收益？**配对交易（Pairs Trading）**提供了一种优雅的解决方案：通过捕捉两个高度相关资产之间的**暂时性偏离**，进行**市场中性**的统计套利。

配对交易的核心思想是：
1. 找到两个**协整（Cointegrated）**的资产（如两只同行业股票）
2. 当它们的价格关系**暂时偏离**历史均衡时，做多低估资产、做空高估资产
3. 等待价格关系**均值回归**，平仓获利

本文将深入探讨：
1. 协整关系的理论基础
2. 如何识别协整对
3. Python实战：从数据获取到策略回测
4. 实战中的关键问题（手续费、滑点、持仓时间）
5. 配对交易的局限性与改进方向

---

## 一、协整关系：配对交易的理论基石

### 1.1 为什么需要协整？

**问题**：如果两个股票的价格序列都是**非平稳**的（如随机游走），它们的线性组合可能仍然是**平稳**的。这种关系就是**协整**。

**举例**：
- 中国石油（601857.SH）和中国石化（600028.SH）的股价都是随机游走（非平稳）
- 但它们的**价差**（或线性组合）可能围绕某个均值波动（平稳）
- 这就是协整关系：长期来看，两者价格存在**均衡关系**

### 1.2 协整 vs 相关性

**常见误区**：高相关性 = 好的配对

**事实**：
- **相关性**衡量的是**收益率**的同步性（短期关系）
- **协整**衡量的是**价格**的长期均衡关系（长期关系）

**反例**：
- 两只科技股可能短期相关性很高（同涨同跌），但长期价格趋势可能分化（不协整）
- 两只公用事业股可能相关性不高，但价格比长期稳定（协整）

**结论**：配对交易需要的是**协整关系**，而非简单的高相关性。

### 1.3 协整的数学定义

对于两个非平稳时间序列 \(X_t\) 和 \(Y_t\)，如果存在系数 \(\beta\) 使得：

\[
Z_t = Y_t - \beta X_t
\]

是**平稳序列**（即 \(Z_t\) 的均值和方差不随时间变化），则称 \(X_t\) 和 \(Y_t\) 是**协整**的。

**平稳性的检验标准**（ADF检验）：
- 原假设 \(H_0\)：序列有单位根（非平稳）
- 备择假设 \(H_1\)：序列平稳
- 如果 p-value < 0.05，拒绝原假设，认为序列平稳

---

## 二、如何识别协整对？

### 2.1 步骤1：初筛（基于基本面和行业分类）

**原则**：协整关系更可能出现在**同行业、同业务模式**的股票对。

**示例**：
- 银行股：工商银行 vs 建设银行
- 能源股：中国石油 vs 中国石化
- 零售股：永辉超市 vs 家家悦

**工具**：
```python
import tushare as ts

# 获取行业分类
industry_map = ts.get_industry_classified()
# 筛选同行业股票
bank_stocks = industry_map[industry_map['c_name'] == '银行']['code'].tolist()
```

### 2.2 步骤2：计算协整关系（Engle-Granger检验）

**两步法**：
1. 用OLS回归估计对冲比例 \(\beta\)：\(Y_t = \alpha + \beta X_t + \epsilon_t\)
2. 对残差 \(\epsilon_t\) 进行ADF检验

**Python实现**：

```python
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS

def test_cointegration(price_x, price_y, significance_level=0.05):
    """
    检验两个价格序列是否协整
    
    参数:
        price_x: Series, 第一个资产的价格
        price_y: Series, 第二个资产的价格
        significance_level: float, 显著性水平
    
    返回:
        is_cointegrated: bool, 是否协整
        hedge_ratio: float, 对冲比例（Y = alpha + beta * X）
        p_value: float, 协整检验的p-value
    """
    # 步骤1：OLS回归估计对冲比例
    model = OLS(price_y, sm.add_constant(price_x)).fit()
    hedge_ratio = model.params[1]  # beta
    residuals = model.resid  # 残差
    
    # 步骤2：ADF检验残差平稳性
    adf_result = adfuller(residuals, autolag='AIC')
    p_value = adf_result[1]
    
    # 判断是否协整
    is_cointegrated = p_value < significance_level
    
    return is_cointegrated, hedge_ratio, p_value

# 示例使用
# 假设我们有两只股票的价格数据
price_x = pd.Series(np.random.randn(1000).cumsum() + 100)  # 模拟价格
price_y = pd.Series(0.8 * price_x + np.random.randn(1000).cumsum() * 0.5 + 50)

is_coint, beta, pval = test_cointegration(price_x, price_y)
print(f"协整检验p-value: {pval:.4f}, 是否协整: {is_coint}")
```

### 2.3 步骤3：可视化验证

**方法**：绘制价差（或Z-Score）的时间序列图，观察是否均值回归。

```python
import matplotlib.pyplot as plt

def plot_spread(price_x, price_y, hedge_ratio, title='价差可视化'):
    """
    绘制价差及其Z-Score
    """
    # 计算价差
    spread = price_y - hedge_ratio * price_x
    
    # 计算Z-Score（标准化）
    spread_mean = spread.rolling(252).mean()  # 使用252天滚动窗口
    spread_std = spread.rolling(252).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 绘图
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    ax1.plot(spread, linewidth=1.5)
    ax1.axhline(y=spread_mean.iloc[-1], color='red', linestyle='--', label='历史均值')
    ax1.fill_between(range(len(spread)), 
                     spread_mean - 2*spread_std, 
                     spread_mean + 2*spread_std, 
                     alpha=0.2)
    ax1.set_title('价差（残差）', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(z_score, linewidth=1.5, color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='+2σ')
    ax2.axhline(y=-2, color='green', linestyle='--', alpha=0.5, label='-2σ')
    ax2.fill_between(range(len(z_score)), -2, 2, alpha=0.1, color='gray')
    ax2.set_title('价差的Z-Score（标准化）', fontsize=14)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('spread_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# 调用
plot_spread(price_x, price_y, hedge_ratio)
```

---

## 三、Python实战：构建配对交易策略

### 3.1 数据准备

假设我们使用A股数据（需要tushare或类似数据源）：

```python
import tushare as ts
import pandas as pd

# 设置token（需要提前注册tushare）
ts.set_token('your_token_here')
pro = ts.pro_api()

def get_stock_data(stock_code, start_date, end_date):
    """
    获取股票日线数据
    """
    df = pro.daily(ts_code=stock_code, 
                   start_date=start_date, 
                   end_date=end_date)
    df['trade_date'] = pd.to_datetime(df['trade_date'])
    df.set_index('trade_date', inplace=True)
    df = df.sort_index()
    return df['close']

# 获取两只银行股的数据
stock1 = get_stock_data('601398.SH', '20200101', '20251231')  # 工商银行
stock2 = get_stock_data('601939.SH', '20200101', '20251231')  # 建设银行

# 对齐数据
prices = pd.concat([stock1, stock2], axis=1, join='inner')
prices.columns = ['ICBC', 'CCB']
```

### 3.2 策略逻辑

**交易信号**（基于Z-Score）：
- 当 \(Z_t < -2\)：价差偏低，做多价差（买入股票A，卖出股票B）
- 当 \(Z_t > 2\)：价差偏高，做空价差（卖出股票A，买入股票B）
- 当 \(Z_t\) 回到 \([-0.5, 0.5]\)：平仓

**仓位管理**：
- 等权配置：每只股票投入总资金的50%
- 动态调仓：每天检查信号，只在信号变化时调仓

### 3.3 完整策略代码

```python
class PairsTradingStrategy:
    """
    配对交易策略类
    """
    def __init__(self, price_x, price_y, entry_z=2.0, exit_z=0.5, 
                 lookback=252, transaction_cost=0.001):
        """
        初始化策略
        
        参数:
            price_x: Series, 第一个资产的价格
            price_y: Series, 第二个资产的价格
            entry_z: float, 入场Z-Score阈值
            exit_z: float, 出场Z-Score阈值
            lookback: int, 计算均值的回溯期
            transaction_cost: float, 交易成本（单边）
        """
        self.price_x = price_x
        self.price_y = price_y
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.lookback = lookback
        self.transaction_cost = transaction_cost
        
        # 计算对冲比例（使用全样本OLS，实战中应滚动计算）
        self.hedge_ratio = self._calculate_hedge_ratio()
        
        # 计算价差和Z-Score
        self.spread = self.price_y - self.hedge_ratio * self.price_x
        self.z_score = self._calculate_z_score()
        
        # 初始化仓位和收益序列
        self.positions = pd.Series(0, index=price_x.index)
        self.returns = pd.Series(0.0, index=price_x.index)
    
    def _calculate_hedge_ratio(self):
        """计算对冲比例（OLS回归）"""
        X = sm.add_constant(self.price_x)
        model = OLS(self.price_y, X).fit()
        return model.params[1]
    
    def _calculate_z_score(self):
        """计算滚动Z-Score"""
        spread_mean = self.spread.rolling(self.lookback).mean()
        spread_std = self.spread.rolling(self.lookback).std()
        z_score = (self.spread - spread_mean) / spread_std
        return z_score
    
    def generate_signals(self):
        """生成交易信号"""
        signals = pd.Series(0, index=self.z_score.index)
        
        for i in range(1, len(self.z_score)):
            z_prev = self.z_score.iloc[i-1]
            z_curr = self.z_score.iloc[i]
            
            # 入场信号
            if z_prev > self.entry_z:  # 价差偏高，做空
                signals.iloc[i] = -1
            elif z_prev < -self.entry_z:  # 价差偏低，做多
                signals.iloc[i] = 1
            
            # 出场信号（如果已有仓位）
            elif abs(z_curr) < self.exit_z:
                signals.iloc[i] = 0  # 平仓
            else:
                signals.iloc[i] = signals.iloc[i-1]  # 保持仓位
        
        return signals
    
    def backtest(self):
        """回测策略"""
        signals = self.generate_signals()
        
        # 计算每日收益
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:  # 有仓位
                # 计算价差收益（做多Y、做空X）
                ret_x = self.price_x.iloc[i] / self.price_x.iloc[i-1] - 1
                ret_y = self.price_y.iloc[i] / self.price_y.iloc[i-1] - 1
                
                # 配对交易收益 = 信号 * (ret_y - hedge_ratio * ret_x)
                self.returns.iloc[i] = signals.iloc[i] * (ret_y - self.hedge_ratio * ret_x)
                
                # 减去交易成本（仅在调仓时）
                if signals.iloc[i] != signals.iloc[i-1]:
                    self.returns.iloc[i] -= 2 * self.transaction_cost  # 双边成本
        
        # 计算累计收益
        cumulative_returns = (1 + self.returns).cumprod()
        
        return cumulative_returns, self.returns
    
    def calculate_performance(self, cumulative_returns, strategy_returns):
        """计算策略表现指标"""
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(strategy_returns)) - 1
        volatility = strategy_returns.std() * np.sqrt(252)
        sharpe = annual_return / volatility if volatility > 0 else 0
        max_drawdown = ((cumulative_returns / cumulative_returns.cummax()) - 1).min()
        
        performance = {
            '累计收益': total_return,
            '年化收益': annual_return,
            '年化波动': volatility,
            '夏普比率': sharpe,
            '最大回撤': max_drawdown,
            '胜率': (strategy_returns > 0).sum() / len(strategy_returns)
        }
        
        return performance

# 使用示例
strategy = PairsTradingStrategy(prices['ICBC'], prices['CCB'], 
                                entry_z=2.0, exit_z=0.5, lookback=252)
cumulative_ret, daily_ret = strategy.backtest()
performance = strategy.calculate_performance(cumulative_ret, daily_ret)

print("=== 配对交易策略表现 ===")
for key, value in performance.items():
    print(f"{key}: {value:.4f}")
```

---

## 四、回测结果分析

### 4.1 性能表现（示例数据）

假设我们对2020-2025年工商银行vs建设银行的配对交易进行回测：

| 指标 | 配对交易策略 | 买入持有（等权） | 改善幅度 |
|------|-------------|-----------------|---------|
| 累计收益 | 35.2% | 18.7% | +16.5% |
| 年化收益 | 6.8% | 3.7% | +3.1% |
| 年化波动 | 8.2% | 22.5% | -14.3% |
| 夏普比率 | 0.83 | 0.16 | +418% |
| 最大回撤 | -9.3% | -28.6% | +19.3% |
| 胜率 | 58.7% | - | - |

**关键发现**：
1. **风险大幅降低**：年化波动从22.5%降至8.2%，最大回撤从-28.6%降至-9.3%
2. **收益稳定增强**：虽然年化收益绝对值不高（6.8%），但在震荡市中表现优异
3. **市场中性**：策略收益与大盘相关性低，适合市场不确定性高的环境

### 4.2 交易频率分析

```python
# 计算交易次数和持仓时间
def analyze_trading_activity(signals):
    """
    分析交易活动
    """
    # 计算调仓次数
    position_changes = (signals != signals.shift(1)).sum()
    
    # 计算平均持仓时间
    trades = []
    current_position = 0
    trade_start = None
    
    for i, signal in enumerate(signals):
        if signal != 0 and current_position == 0:  # 开仓
            trade_start = i
            current_position = signal
        elif signal == 0 and current_position != 0:  # 平仓
            trades.append(i - trade_start)
            current_position = 0
            trade_start = None
    
    avg_holding_period = np.mean(trades) if trades else 0
    
    return {
        '调仓次数': position_changes,
        '平均持仓天数': avg_holding_period,
        '交易次数': len(trades)
    }

trading_stats = analyze_trading_activity(strategy.generate_signals())
print("\n=== 交易活动分析 ===")
for key, value in trading_stats.items():
    print(f"{key}: {value}")
```

**典型结果**：
- 调仓次数：约20-30次/年（取决于波动率）
- 平均持仓时间：15-30天
- 交易次数：10-15次/年（每轮交易包括开仓和平仓）

---

## 五、实战中的关键问题

### 5.1 交易成本的影响

**问题**：配对交易频繁调仓，交易成本可能侵蚀大部分收益。

**应对方法**：
1. **优化阈值**：提高entry_z（如从2.0提高到2.5），减少交易次数
2. **使用低费率券商**：降低单边交易成本（如从0.1%降至0.05%）
3. **优化执行**：使用限价单而非市价单，减少滑点

**敏感性分析**：

```python
# 测试不同交易成本下的策略表现
costs = [0.0005, 0.001, 0.002, 0.003]
results = []

for cost in costs:
    strategy = PairsTradingStrategy(prices['ICBC'], prices['CCB'], 
                                    transaction_cost=cost)
    cum_ret, daily_ret = strategy.backtest()
    perf = strategy.calculate_performance(cum_ret, daily_ret)
    results.append(perf['年化收益'])

# 绘制敏感性分析图
plt.figure(figsize=(10, 6))
plt.plot([c*100 for c in costs], [r*100 for r in results], marker='o')
plt.xlabel('交易成本（%）')
plt.ylabel('年化收益（%）')
plt.title('交易成本对策略收益的影响')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('cost_sensitivity.png', dpi=300)
```

### 5.2 持仓时间的选择

**问题**：配对交易可能长期不收敛（价差持续偏离），导致资金占用时间过长。

**应对方法**：
1. **设置止损**：如果持仓超过N天（如60天）仍未收敛，强制平仓
2. **动态阈值**：根据持仓时间动态调整exit_z（如持仓30天后，exit_z从0.5放宽到1.0）
3. **分批建仓**：首次信号出现时建50%仓位，如果价差进一步扩大（如Z-Score从2.0升至2.5），再建剩余50%

### 5.3 协整关系的稳定性

**问题**：协整关系可能随时间破裂（如公司并购、行业格局变化）。

**应对方法**：
1. **滚动检验**：每季度重新检验协整关系，如果p-value>0.1，停止交易该配对
2. **多配对组合**：同时交易10-20个独立的配对，分散单一配对失效的风险
3. **基本面监控**：关注配对股票的重大事件（如财报、重组），及时调整策略

---

## 六、配对交易的改进方向

### 6.1 多因子配对交易

传统配对交易只使用**价格**信息，可以引入**基本面因子**优化选股：

**思路**：
1. 在同一行业内，根据**市值、估值、动量**等因子打分
2. 做多高分股票、做空低分股票，构建**因子中性**的配对组合

**代码示例**：

```python
def multi_factor_pairs(stock_universe, factors, weighting='equal'):
    """
    构建多因子配对组合
    
    参数:
        stock_universe: list, 股票代码列表
        factors: DataFrame, 各股票的因子暴露
        weighting: str, 权重方式（'equal'或'optimized'）
    
    返回:
        portfolio_weights: DataFrame, 组合权重
    """
    # 标准化因子
    normalized_factors = (factors - factors.mean()) / factors.std()
    
    # 计算综合得分
    scores = normalized_factors.sum(axis=1)
    
    # 分为多空两组
    threshold = scores.median()
    long_stocks = scores[scores > threshold].index
    short_stocks = scores[scores < threshold].index
    
    # 构建权重
    weights = pd.DataFrame(0, index=factors.index, columns=['weight'])
    if weighting == 'equal':
        weights.loc[long_stocks, 'weight'] = 1.0 / len(long_stocks)
        weights.loc[short_stocks, 'weight'] = -1.0 / len(short_stocks)
    
    return weights
```

### 6.2 机器学习优化信号

传统Z-Score阈值（如±2）是**固定**的，可以使用机器学习模型**动态预测**最优阈值。

**思路**：
1. 特征：波动率、成交量、市场状态、宏观变量
2. 标签：未来N天价差是否收敛（二分类）
3. 模型：随机森林、梯度提升树
4. 输出：动态入场/出场阈值

**代码示例**（简化版）：

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def ml_enhanced_pairs_trading(prices, features, lookahead=20):
    """
    使用机器学习优化配对交易
    
    参数:
        prices: DataFrame, 价格数据
        features: DataFrame, 特征数据
        lookahead: int, 预测未来N天的收敛概率
    """
    # 计算标签（未来N天价差是否收敛）
    spread = prices.iloc[:, 0] - prices.iloc[:, 1]
    z_score = (spread - spread.rolling(252).mean()) / spread.rolling(252).std()
    labels = (z_score.shift(-lookahead).abs() < 0.5).astype(int)  # 1 if converged, 0 otherwise
    
    # 训练模型
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # 预测收敛概率
    convergence_prob = model.predict_proba(features)[:, 1]
    
    # 根据概率动态调整阈值
    dynamic_threshold = 2.0 - convergence_prob  # 高概率时降低阈值（更激进）
    
    return model, dynamic_threshold
```

### 6.3 高频配对交易

将配对交易应用到**日内高频**数据（如分钟级或Tick级），捕捉更短期的定价偏差。

**挑战**：
- 数据质量要求高（需要清洁的Tick数据）
- 执行速度要求高（需要量化交易系统支持）
- 交易成本影响更大（需要极低的手续费）

---

## 七、总结与展望

### 7.1 核心要点

1. **协整是配对交易的基础**：必须严格检验协整关系，不能仅凭高相关性就贸然交易
2. **均值回归需要时间**：配对交易不是"快速获利"的策略，需要有耐心等待收敛
3. **交易成本是关键**：必须仔细估算交易成本（包括滑点、冲击成本），确保策略在扣除成本后仍有盈利
4. **风险管理不可少**：设置止损、控制单一配对仓位、定期检验协整关系

### 7.2 实践建议

**对于个人投资者**：
- 从**同行业大型股**开始（如银行股、保险股），它们的协整关系更稳定
- 使用**模拟盘**充分测试策略，至少观察3-6个月的表现
- 关注**税务影响**：频繁交易可能产生较高的资本利得税

**对于机构投资者**：
- 建立**系统化流程**：从配对筛选、信号生成、执行到风控，全流程自动化
- 投资于**数据质量**：清洁、低延迟的数据是配对交易成功的前提
- 优化**执行算法**：使用VWAP、TWAP等算法降低冲击成本

### 7.3 未来研究方向

1. **非对称配对交易**：允许对冲比例\(\beta\)随时间变化（时变协整）
2. **跨资产配对**：不仅在同一资产类别内配对，还可以跨资产（如股票vs可转债、股票vsETF）
3. **深度学习应用**：使用LSTM、Transformer等模型捕捉价差的 nonlinear patterns

---

## 参考文献

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs Trading: Performance of a Relative-Value Arbitrage Rule." *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Elliott, R. J., et al. (2005). "Pairs Trading." *Quantitative Finance*.
4. Huck, N. (2019). "Pairs Trading, Financial Turmoil, and the Brexit Referendum." *Journal of International Financial Markets*.
5. Do, B., & Faff, R. (2012). "Are Pairs Trading Profits Robust to Trading Costs?" *Journal of Financial Markets*.

---

## 附录：完整代码与数据

本文的完整Python代码（包括数据获取、协整检验、策略回测、性能分析）已上传至GitHub：
[GitHub链接]（待补充）

**数据来源**：
- 股票数据：Tushare Pro（需要注册获取token）
- 因子数据：CSMAR、Wind（学术研究可用）
- 宏观数据：FRED、Wind

---

**免责声明**：本文仅供参考，不构成投资建议。配对交易虽然理论上是市场中性策略，但仍面临模型风险、交易成本风险、协整关系破裂风险等。在实际投资前，请务必进行充分的风险评估和样本外测试。

**更新日期**：2026年6月21日
