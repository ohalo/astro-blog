---
title: "配对交易与协整分析：统计套利实战指南"
description: "从协整检验到交易信号生成，手把手教你构建配对交易策略。包含Python完整代码、风险管理和实盘注意事项。"
pubDate: 2026-06-15
tags: ["配对交易", "统计套利", "协整分析", "量化策略"]
cover: "/images/pair-trading-cointegration/cover.jpg"
---

# 配对交易与协整分析：统计套利实战指南

## 引言

配对交易（Pairs Trading）是统计套利中最经典的策略之一。它不依赖市场方向，而是通过捕捉两个高度相关资产之间的暂时性偏离来获利。从摩根士丹利的量化团队到文艺复兴科技，无数顶级量化机构都在使用这一策略。

本文将深入讲解：
- 协整理论与检验方法
- 配对选择的量化标准
- 交易信号生成与风控
- Python完整实战代码
- 实盘中的坑与解决方案

## 一、配对交易的理论基础

### 1.1 平稳性与协整

**平稳性（Stationarity）**是时间序列分析的核心概念。一个平稳序列的均值、方差和自协方差不随时间变化。

**伪回归问题**：
```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# 生成两个独立的随机游走
np.random.seed(42)
n = 1000
X = np.cumsum(np.random.randn(n))  # 随机游走1
Y = np.cumsum(np.random.randn(n))  # 随机游走2

# 回归分析（伪回归）
X_const = sm.add_constant(X)
model = sm.OLS(Y, X_const).fit()

print("=== 伪回归示例 ===")
print(f"R² = {model.rsquared:.4f}")
print(f"p-value = {model.pvalues[1]:.4f}")
print("结论：两个独立的随机游走，回归结果显示'显著相关'（错误！）")
```

**协整（Cointegration）**：
两个非平稳序列的线性组合是平稳的，则它们协整。

数学表达：
```
如果 Y_t 和 X_t 都是 I(1) 序列（一阶单整），
但存在参数 β 使得残差平稳：
  ε_t = Y_t - β·X_t ~ I(0)
则 Y_t 和 X_t 协整。
```

### 1.2 协整 vs 相关性

**关键区别**：
- **相关性**：衡量同期线性依赖，可能虚假（伪回归）
- **协整**：衡量长期均衡关系，有经济学意义

```python
def compare_correlation_cointegration(price1, price2):
    """
    对比相关性和协整性
    """
    # 相关性（可能误导）
    correlation = price1.corr(price2)
    
    # 协整检验（Engle-Granger)
    from statsmodels.tsa.stattools import coint
    coint_stat, p_value, _ = coint(price1, price2)
    
    print(f"相关系数: {correlation:.4f}")
    print(f"协整检验 p-value: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ 两序列协整（存在长期均衡关系）")
    else:
        print("✗ 两序列不协整（即使相关性高，也可能是伪回归）")
```

## 二、协整检验方法

### 2.1 Engle-Granger 两步法

```python
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import statsmodels.api as sm

def engle_granger_test(Y, X, significance=0.05):
    """
    Engle-Granger 协整检验
    
    步骤：
    1. 回归 Y_t = α + β·X_t + ε_t
    2. ADF检验残差 ε_t 的平稳性
    """
    # 步骤1：OLS回归
    X_const = sm.add_constant(X)
    model = OLS(Y, X_const).fit()
    residuals = model.resid
    
    # 步骤2：ADF检验残差
    adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(
        residuals, 
        maxlag=1, 
        regression='c'  # 残差均值为0
    )
    
    # 步骤3：协整检验（直接调用coint）
    coint_stat, coint_pvalue, trace_stats = coint(Y, X)
    
    # 结果解读
    is_cointegrated = coint_pvalue < significance
    
    result = {
        'hedge_ratio': model.params[1],
        'intercept': model.params[0],
        'residual_mean': residuals.mean(),
        'residual_std': residuals.std(),
        'adf_stat': adf_stat,
        'adf_pvalue': adf_pvalue,
        'coint_stat': coint_stat,
        'coint_pvalue': coint_pvalue,
        'is_cointegrated': is_cointegrated
    }
    
    return result

# 使用示例
# result = engle_granger_test(stock1_prices, stock2_prices)
# print(f"对冲比例: {result['hedge_ratio']:.4f}")
# print(f"协整 p-value: {result['coint_pvalue']:.4f}")
```

### 2.2 Johansen 检验（多变量协整）

当有超过2个资产时，使用Johansen检验：

```python
from statsmodels.tsa.johansen import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验（适用于多资产）
    
    参数：
    - price_matrix: DataFrame, 每列是一个资产的价格序列
    - det_order: 确定性项的阶数（0=无常数项，1=有常数项）
    - k_ar_diff: VAR模型滞后阶数
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 输出结果
    print("=== Johansen 协整检验结果 ===")
    print(f"特征值: {result.eig}")
    print(f"\n迹统计量 (Trace Statistic):")
    for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt[:, 1])):  # 5%临界值
        print(f"  r<={i}: 统计量={stat:.2f}, 5%临界值={crit:.2f}")
    
    # 判断协整关系个数
    num_coint = 0
    for i, (stat, crit) in enumerate(zip(result.lr1, result.cvt[:, 1])):
        if stat > crit:
            num_coint += 1
    
    print(f"\n结论: 存在 {num_coint} 个协整关系")
    
    return result, num_coint

# 使用示例
# prices = pd.DataFrame({
#     'Stock1': stock1,
#     'Stock2': stock2,
#     'Stock3': stock3
# })
# result, num_coint = johansen_test(prices)
```

### 2.3 滚动窗口协整检验

协整关系可能随时间变化，需要动态监测：

```python
def rolling_cointegration_test(Y, X, window=252, step=20):
    """
    滚动窗口协整检验
    
    参数：
    - window: 滚动窗口长度（默认252个交易日=1年）
    - step: 步长（默认每20个交易日检验一次）
    """
    results = []
    dates = []
    
    for i in range(window, len(Y), step):
        Y_window = Y[i-window:i]
        X_window = X[i-window:i]
        
        try:
            result = engle_granger_test(Y_window, X_window)
            results.append(result['is_cointegrated'])
            dates.append(Y.index[i])
        except:
            results.append(False)
            dates.append(Y.index[i])
    
    # 可视化
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, results, 'b-', linewidth=2, label='协整状态')
    ax.axhline(y=0.5, color='gray', linestyle='--', label='阈值')
    ax.set_ylabel('是否协整', fontsize=12)
    ax.set_xlabel('日期', fontsize=12)
    ax.set_title('滚动窗口协整检验（动态监测）', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rolling_cointegration.png', dpi=300, bbox_inches='tight')
    
    return pd.Series(results, index=dates)
```

## 三、配对选择策略

### 3.1 量化筛选标准

```python
def screen_pairs(universe_prices, method='distance', **kwargs):
    """
    批量筛选配对
    
    方法：
    - 'distance': 基于价格距离（最简单）
    - 'correlation': 基于相关性
    - 'cointegration': 基于协整检验（最严格）
    """
    n_assets = universe_prices.shape[1]
    pairs = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            stock1 = universe_prices.iloc[:, i]
            stock2 = universe_prices.iloc[:, j]
            
            if method == 'distance':
                # 方法1：价格距离（标准化后）
                normalized1 = (stock1 - stock1.mean()) / stock1.std()
                normalized2 = (stock2 - stock2.mean()) / stock2.std()
                distance = np.sqrt(((normalized1 - normalized2) ** 2).sum())
                
                if distance < kwargs.get('threshold', 10):
                    pairs.append({
                        'stock1': universe_prices.columns[i],
                        'stock2': universe_prices.columns[j],
                        'distance': distance
                    })
            
            elif method == 'correlation':
                # 方法2：相关性
                corr = stock1.corr(stock2)
                if corr > kwargs.get('threshold', 0.8):
                    pairs.append({
                        'stock1': universe_prices.columns[i],
                        'stock2': universe_prices.columns[j],
                        'correlation': corr
                    })
            
            elif method == 'cointegration':
                # 方法3：协整检验
                result = engle_granger_test(stock1, stock2)
                if result['is_cointegrated']:
                    pairs.append({
                        'stock1': universe_prices.columns[i],
                        'stock2': universe_prices.columns[j],
                        'pvalue': result['coint_pvalue'],
                        'hedge_ratio': result['hedge_ratio']
                    })
    
    # 按指标排序
    if len(pairs) > 0:
        if method == 'distance':
            pairs = sorted(pairs, key=lambda x: x['distance'])
        elif method == 'correlation':
            pairs = sorted(pairs, key=lambda x: x['correlation'], reverse=True)
        elif method == 'cointegration':
            pairs = sorted(pairs, key=lambda x: x['pvalue'])
    
    return pairs

# 使用示例
# pairs = screen_pairs(price_data, method='cointegration')
# print(f"找到 {len(pairs)} 个协整配对")
# for p in pairs[:5]:
#     print(f"{p['stock1']} - {p['stock2']}, p-value={p['pvalue']:.4f}")
```

### 3.2 基本面匹配

除了统计指标，还要考虑基本面：
- **同行业**：业务模式相似，受相同因素驱动
- **相似市值**：避免小盘股流动性问题
- **相似Beta**：市场暴露相近

```python
def filter_by_fundamentals(pairs, stock_info):
    """
    基于基本面信息过滤配对
    
    参数：
    - pairs: screen_pairs()的输出
    - stock_info: DataFrame, 包含行业、市值、Beta等信息
    """
    filtered = []
    
    for pair in pairs:
        s1 = pair['stock1']
        s2 = pair['stock2']
        
        info1 = stock_info.loc[s1]
        info2 = stock_info.loc[s2]
        
        # 条件1：同行业
        same_industry = (info1['industry'] == info2['industry'])
        
        # 条件2：市值相近（相差不超过2倍）
        market_cap_ratio = max(info1['market_cap'], info2['market_cap']) / \
                          min(info1['market_cap'], info2['market_cap'])
        similar_size = market_cap_ratio < 2
        
        # 条件3：Beta相近（相差不超过0.3）
        beta_diff = abs(info1['beta'] - info2['beta'])
        similar_beta = beta_diff < 0.3
        
        if same_industry and similar_size and similar_beta:
            pair['industry'] = info1['industry']
            pair['market_cap_ratio'] = market_cap_ratio
            pair['beta_diff'] = beta_diff
            filtered.append(pair)
    
    return filtered
```

## 四、交易信号与风险管理

### 4.1 信号生成

```python
class PairTradingStrategy:
    """
    配对交易策略类
    """
    def __init__(self, stock1, stock2, entry_z=2.0, exit_z=0.5, 
                 stop_loss_z=3.0, lookback=252):
        """
        初始化策略参数
        
        参数：
        - entry_z: 入场Z得分阈值（默认2.0）
        - exit_z: 出场Z得分阈值（默认0.5）
        - stop_loss_z: 止损Z得分阈值（默认3.0）
        - lookback: 计算均值的滚动窗口（默认252个交易日）
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.lookback = lookback
        
        self.hedge_ratio = None
        self.spread = None
        self.z_score = None
        self.position = 0  # 1=多配对, -1=空配对, 0=空仓
        
    def calculate_spread(self, prices1, prices2):
        """
        计算价差（或使用回归残差）
        """
        # 方法1：简单价差（需要标准化）
        # spread = prices1 - prices2
        
        # 方法2：回归残差（推荐）
        X = sm.add_constant(prices2)
        model = OLS(prices1, X).fit()
        self.hedge_ratio = model.params[1]
        spread = model.resid
        
        # 计算Z得分
        self.spread = spread.rolling(self.lookback).mean()
        self.spread_std = spread.rolling(self.lookback).std()
        self.z_score = (spread - self.spread) / self.spread_std
        
        return spread, self.z_score
    
    def generate_signals(self, z_score):
        """
        生成交易信号
        """
        signals = pd.Series(0, index=z_score.index)
        
        for i in range(1, len(z_score)):
            if self.position == 0:
                # 空仓时，检查入场信号
                if z_score.iloc[i] > self.entry_z:
                    # Z得分过高，做空配对（卖高买低）
                    self.position = -1
                    signals.iloc[i] = -1
                elif z_score.iloc[i] < -self.entry_z:
                    # Z得分过低，做多配对（买低卖高）
                    self.position = 1
                    signals.iloc[i] = 1
            
            elif self.position == 1:
                # 持多仓，检查出场或止损
                if abs(z_score.iloc[i]) < self.exit_z:
                    # Z得分回归，平仓
                    self.position = 0
                    signals.iloc[i] = 0
                elif z_score.iloc[i] < -self.stop_loss_z:
                    # 止损（价差继续扩大）
                    self.position = 0
                    signals.iloc[i] = 0
            
            elif self.position == -1:
                # 持空仓，检查出场或止损
                if abs(z_score.iloc[i]) < self.exit_z:
                    # Z得分回归，平仓
                    self.position = 0
                    signals.iloc[i] = 0
                elif z_score.iloc[i] > self.stop_loss_z:
                    # 止损
                    self.position = 0
                    signals.iloc[i] = 0
        
        return signals
    
    def backtest(self, prices1, prices2, commission=0.001):
        """
        回测策略
        
        参数：
        - commission: 单边交易手续费（默认0.1%）
        """
        # 计算价差和Z得分
        spread, z_score = self.calculate_spread(prices1, prices2)
        
        # 生成信号
        signals = self.generate_signals(z_score)
        
        # 计算收益
        returns1 = prices1.pct_change()
        returns2 = prices2.pct_change()
        
        strategy_returns = pd.Series(0, index=prices1.index)
        
        for i in range(1, len(signals)):
            if signals.iloc[i] != 0:
                # 有交易信号
                if signals.iloc[i] == 1:
                    # 做多配对：买stock1，卖stock2
                    ret = returns1.iloc[i] - self.hedge_ratio * returns2.iloc[i]
                elif signals.iloc[i] == -1:
                    # 做空配对：卖stock1，买stock2
                    ret = -returns1.iloc[i] + self.hedge_ratio * returns2.iloc[i]
                
                # 扣除手续费
                ret -= 2 * commission
                strategy_returns.iloc[i] = ret
        
        # 计算累计收益
        cumulative_returns = (1 + strategy_returns).cumprod()
        
        # 计算绩效指标
        total_return = cumulative_returns.iloc[-1] - 1
        annual_return = (1 + total_return) ** (252 / len(prices1)) - 1
        sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)
        max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
        
        performance = {
            'total_return': total_return,
            'annual_return': annual_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'num_trades': (signals != 0).sum()
        }
        
        return performance, cumulative_returns, signals
```

### 4.2 风险管理

```python
def risk_management(prices1, prices2, spread, z_score, position):
    """
    配对交易的风险管理
    """
    risks = {}
    
    # 风险1：协整关系破裂
    rolling_result = rolling_cointegration_test(prices1, prices2, window=126)
    if not rolling_result.iloc[-1]:
        risks['cointegration_break'] = "协整关系破裂，建议平仓"
        position = 0
    
    # 风险2：价差非平稳（结构性断裂）
    from statsmodels.stats.diagnostic import het_arch
    arch_test = het_arch(spread.dropna())[1]
    if arch_test < 0.05:
        risks['structural_break'] = "价差出现异方差，模型可能失效"
    
    # 风险3：流动性风险
    volume1 = prices1['volume'].mean()
    volume2 = prices2['volume'].mean()
    if min(volume1, volume2) < 100000:  # 日成交量低于10万
        risks['liquidity'] = "流动性不足，可能滑点较大"
    
    # 风险4：行业冲击
    # （需要外部数据，如行业指数、政策新闻等）
    
    return risks

# 仓位管理：Kelly公式
def kelly_position_sizing(win_rate, avg_win, avg_loss):
    """
    使用Kelly公式计算最优仓位
    
    f* = (p * b - q) / b
    其中：
    - p = 胜率
    - q = 败率 = 1 - p
    - b = 盈亏比 = avg_win / avg_loss
    """
    b = avg_win / avg_loss
    f_star = (win_rate * b - (1 - win_rate)) / b
    
    # 保守处理：使用Half-Kelly
    f_conservative = f_star / 2
    
    return max(0, min(f_conservative, 0.25))  # 限制单次最大仓位25%
```

## 五、Python完整实战案例

### 5.1 数据获取与预处理

```python
import yfinance as yf
import pandas as pd
import numpy as np

def get_pair_data(ticker1, ticker2, start_date, end_date):
    """
    获取配对交易数据
    """
    # 下载数据
    stock1 = yf.download(ticker1, start=start_date, end=end_date)
    stock2 = yf.download(ticker2, start=start_date, end=end_date)
    
    # 使用调整后的收盘价
    prices1 = stock1['Adj Close']
    prices2 = stock2['Adj Close']
    
    # 对齐日期
    prices = pd.DataFrame({
        ticker1: prices1,
        ticker2: prices2
    }).dropna()
    
    return prices[ticker1], prices[ticker2]

# 示例：可口可乐 vs 百事可乐
coke, pepsi = get_pair_data('KO', 'PEP', '2020-01-01', '2026-06-15')
```

### 5.2 策略回测与可视化

```python
# 初始化策略
strategy = PairTradingStrategy(
    stock1=coke,
    stock2=pepsi,
    entry_z=2.0,
    exit_z=0.5,
    stop_loss_z=3.0,
    lookback=252
)

# 回测
performance, cumulative_returns, signals = strategy.backtest(coke, pepsi)

print("=== 策略绩效 ===")
for key, value in performance.items():
    if key in ['total_return', 'annual_return', 'sharpe_ratio', 'max_drawdown']:
        print(f"{key}: {value:.4f}")

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：价格走势
ax1 = axes[0]
ax1.plot(coke.index, coke.values, 'b-', label='Coca-Cola (KO)', linewidth=2)
ax1.set_ylabel('KO Price', color='b', fontsize=11)
ax1.tick_params(axis='y', labelcolor='b')
ax1.legend(loc='upper left')

ax1_twin = ax1.twinx()
ax1_twin.plot(pepsi.index, pepsi.values, 'r-', label='Pepsi (PEP)', linewidth=2)
ax1_twin.set_ylabel('PEP Price', color='r', fontsize=11)
ax1_twin.tick_params(axis='y', labelcolor='r')
ax1_twin.legend(loc='upper right')

ax1.set_title('配对资产价格走势', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# 子图2：Z得分与交易信号
ax2 = axes[1]
ax2.plot(strategy.z_score.index, strategy.z_score.values, 'g-', 
         label='Z Score', linewidth=2)
ax2.axhline(y=strategy.entry_z, color='r', linestyle='--', 
            label=f'入场阈值 (±{strategy.entry_z})')
ax2.axhline(y=-strategy.entry_z, color='r', linestyle='--')
ax2.axhline(y=strategy.exit_z, color='orange', linestyle='--', 
            label=f'出场阈值 (±{strategy.exit_z})')
ax2.axhline(y=-strategy.exit_z, color='orange', linestyle='--')
ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

# 标记交易信号
for i in range(len(signals)):
    if signals.iloc[i] == 1:
        ax2.scatter(strategy.z_score.index[i], strategy.z_score.iloc[i], 
                   color='green', marker='^', s=100, label='做多' if i == signals.idxmax() else "")
    elif signals.iloc[i] == -1:
        ax2.scatter(strategy.z_score.index[i], strategy.z_score.iloc[i], 
                   color='red', marker='v', s=100, label='做空' if i == signals.idxmin() else "")

ax2.set_ylabel('Z Score', fontsize=11)
ax2.set_title('价差Z得分与交易信号', fontsize=13, fontweight='bold')
ax2.legend(loc='best', fontsize=9)
ax2.grid(True, alpha=0.3)

# 子图3：累计收益
ax3 = axes[2]
ax3.plot(cumulative_returns.index, cumulative_returns.values, 'b-', 
         linewidth=2.5, label='策略收益')
ax3.axhline(y=1, color='gray', linestyle='-', alpha=0.5)
ax3.fill_between(cumulative_returns.index, 1, cumulative_returns.values,
                 where=(cumulative_returns.values >= 1), alpha=0.3, color='green')
ax3.fill_between(cumulative_returns.index, 1, cumulative_returns.values,
                 where=(cumulative_returns.values < 1), alpha=0.3, color='red')
ax3.set_ylabel('累计收益', fontsize=11)
ax3.set_xlabel('日期', fontsize=11)
ax3.set_title('策略累计收益', fontsize=13, fontweight='bold')
ax3.legend(loc='best')
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pair_trading_backtest.png', dpi=300, bbox_inches='tight')
print("\n✅ 回测结果已保存: pair_trading_backtest.png")
```

## 六、实盘注意事项

### 6.1 常见问题与解决方案

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| **协整关系破裂** | 行业结构变化、并购、政策冲击 | 滚动检验 + 快速止损 |
| **价差不回归** | 模型误设、结构性断裂 | 使用多种均值回归模型对比 |
| **流动性不足** | 小盘股、冷门ETF | 设置最小成交量过滤 |
| **滑点过大** | 市价单、波动剧烈 | 使用限价单 + 分批建仓 |
| **过拟合** | 参数过度优化 | 样本外测试 +  Walk-Forward分析 |

### 6.2 执行建议

1. **多样化配对**：同时交易5-10个不相关配对
2. **动态监控**：每日检查协整关系和Z得分
3. **仓位管理**：单配对不超过总资金的10%
4. **交易成本**：考虑手续费、滑点、卖空成本
5. **税务优化**：了解持仓期限对税率的影响

### 6.3 进阶方向

- **机器学习增强**：使用LSTM预测价差回归时间
- **高频配对**：利用分钟级数据捕捉短期偏离
- **跨市场配对**：A股-H股、股票-ETF套利
- **多因子配对**：结合动量、价值因子筛选配对

## 七、总结

配对交易是一种经典但有效的统计套利策略。成功的关键在于：
1. **严格的协整检验**：避免伪回归陷阱
2. **动态风险管理**：协整关系可能随时破裂
3. **合理的参数选择**：避免过拟合
4. **纪律性执行**：不要因为短期亏损放弃策略

---

**免责声明**：本文仅供学术交流和策略研究，不构成投资建议。配对交易虽有理论基础，但仍面临市场风险、模型风险和执行风险。实盘前请充分回测和模拟交易。

## 参考文献

1. Vidyamurthy, G. (2004). "Pairs Trading: Quantitative Methods and Analysis." Wiley.
2. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). "Pairs trading: Performance of a relative-value arbitrage rule." Review of Financial Studies.
3. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." Wiley.

---

**标签**: #配对交易 #统计套利 #协整分析 #量化策略
