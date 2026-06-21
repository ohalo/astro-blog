---
title: "配对交易与协整分析：均值回归策略的理论与实践"
publishDate: '2026-06-21'
description: "从协整理论到 Python 实战，完整解析配对交易策略的设计与风险管理。涵盖 Engle-Granger 检验、Z-score 信号构建、A股实证案例与策略局限性。"
tags:
  - 量化交易
  - 统计套利
  - 配对交易
  - 协整分析
language: Chinese
difficulty: advanced
---

## 当「买低卖高」有了数学基础

「低买高卖」是所有交易者的梦想，但如果没有严格的定义，「低」和「高」只是事后看图说话。

**配对交易（Pairs Trading）** 给「低买高卖」提供了一个可操作的数学框架：找到两只价格长期「黏在一起」的股票，当它们的价差暂时扩大时做空相对贵的那只、做多相对便宜的那只，等待价差回归均值后平仓获利。

这套方法的核心不是一个简单的相关性——而是 **协整关系（Cointegration）**。

## 协整 vs 相关性：为什么协整更重要？

### 相关性的陷阱

两只股票的价格相关系数很高（比如 0.9），只说明它们**同期**变动方向相似。但高相关并不意味着价差会回归均值——实际上，两个独立随机游走（random walk）之间也可能计算出很高的「伪相关」。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# 两个独立随机游走
T = 252
rw1 = np.cumsum(np.random.randn(T))
rw2 = np.cumsum(np.random.randn(T))

# 计算「相关性」
corr = np.corrcoef(rw1, rw2)[0, 1]
print(f"两个独立随机游走的相关性: {corr:.4f}")  # 可能为 0.3、-0.2 等，但样本足够大时可能「显著」

# 滚动相关性：更危险
rolling_corr = pd.Series(rw1).rolling(60).corr(pd.Series(rw2))
print(f"滚动相关性均值: {rolling_corr.mean():.4f}")
```


![协整 vs 非协整序列的 ADF 检验对比](/images/pair-trading-cointegration/cointegration_adf_test.png)
*图1：左图为非平稳随机游走序列（ADF p-value > 0.05，协整不成立）；右图为平稳价差序列（ADF p-value < 0.01，协整成立）*


这段代码揭示了一个残酷的事实：**你用滚动相关性筛选出来的「配对」，可能只是统计假象**。

### 协整的数学定义

协整关系要求更严格。对于两个非平稳序列 $X_t$ 和 $Y_t$（各自是随机游走），如果存在系数 $\beta$ 使得线性组合：

$$
Z_t = Y_t - \beta X_t
$$

是一个**平稳序列**（Stationary），则称 $X_t$ 和 $Y_t$ 协整。

平稳性的直观含义：$Z_t$ 的均值和方差不随时间变化，且 $Z_t$ 有**均值回归**的特性——偏离均值后，有一种「拉力」把它拉回来。这正是配对交易盈利的数学基础。

## Engle-Granger 两步法：协整检验的 Python 实现

### 第一步：协整回归

用 OLS 估计 $Y_t = \alpha + \beta X_t + \varepsilon_t$，得到残差 $\hat{\varepsilon}_t$。

```python
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tools import add_constant
import statsmodels.api as sm

def cointegration_regression(y, x):
    """
    协整回归：Y = alpha + beta * X + residual
    
    参数：
    - y: array-like, 被解释变量价格序列
    - x: array-like, 解释变量价格序列
    
    返回：
    - beta: 对冲比例
    - alpha: 截距
    - residual: 残差序列（即价差）
    """
    X_with_const = add_constant(x)
    model = OLS(y, X_with_const).fit()
    beta = model.params[1]
    alpha = model.params[0]
    residual = y - (alpha + beta * x)
    return {'beta': beta, 'alpha': alpha, 'residual': residual, 'model': model}

# 示例：模拟协整序列
np.random.seed(20240621)
T = 504

# X 是随机游走（非平稳）
X = np.cumsum(np.random.randn(T) * 0.02) + 10

# Y = 0.8 * X + 平稳误差（协整！）
beta_true = 0.8
alpha_true = 0.5
error = np.random.randn(T) * 0.5 + 0.1 * np.sin(np.arange(T) / 50)  # 有微小均值回归
Y = alpha_true + beta_true * X + error

# 执行协整回归
result = cointegration_regression(Y, X)
print(f"真实 beta: {beta_true:.4f}, 估计 beta: {result['beta']:.4f}")
print(f"真实 alpha: {alpha_true:.4f}, 估计 alpha: {result['alpha']:.4f}")
```

### 第二步：ADF 检验残差的平稳性

协整回归的残差 $\hat{\varepsilon}_t$ 必须是平稳的。用 **Augmented Dickey-Fuller (ADF) 检验** 验证：

- 原假设 $H_0$：残差有单位根（非平稳，协整不成立）
- 备择假设 $H_1$：残差平稳（协整成立）

```python
from statsmodels.tsa.stattools import adfuller

def adf_test(residual, verbose=True):
    """
    ADF 检验残差平稳性
    
    返回：
    - is_cointegrated: bool, p-value < 0.05 则为 True
    - p_value: ADF 检验的 p 值
    - stat: ADF 统计量
    """
    result = adfuller(residual, autolag='AIC')
    stat, p_value, used_lag, n_obs, crit_values, ic_best = result
    
    if verbose:
        print(f"ADF 统计量: {stat:.4f}")
        print(f"p-value: {p_value:.4f}")
        print("临界值:")
        for k, v in crit_values.items():
            print(f"  {k}: {v:.4f}")
    
    is_cointegrated = p_value < 0.05
    if verbose:
        print(f"\n结论: {'协整关系成立（拒绝原假设）' if is_cointegrated else '协整关系不成立（不能拒绝原假设）'}")
    
    return {'is_cointegrated': is_cointegrated, 'p_value': p_value, 'stat': stat}

# 对模拟数据的残差做 ADF 检验
adf_result = adf_test(result['residual'])
```

**关键解读**：若 p-value < 0.05，我们可以在 95% 置信水平上认为残差是平稳的，即 $X$ 和 $Y$ 存在协整关系，配对交易的理论基础成立。

### 更严谨的方法：Johansen 检验

Engle-Granger 方法只能检验两个变量之间的协整，且第一步回归的变量选择（用 Y 回归 X 还是 X 回归 Y）会影响结果。对于多资产配对（一对多），应使用 **Johansen 检验**：

```python
from statsmodels.tsa.vector_ar.vecm import coint_johansen

def johansen_test(price_matrix, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验（适用于多变量）
    
    参数：
    - price_matrix: DataFrame, 多资产价格矩阵 (T x N)
    - det_order: 0=无截距无趋势, 1=有截距, 2=有截距和趋势
    - k_ar_diff: VAR 模型的滞后阶数
    """
    result = coint_johansen(price_matrix, det_order, k_ar_diff)
    
    # 输出两个迹统计量（trace statistic）的 p 值
    trace_stat = result.lr1
    crit_5pct = result.cvt[:, 1]  # 5% 临界值
    
    n_cointegrating = np.sum(trace_stat > crit_5pct)
    print(f"Johansen 检验：存在 {n_cointegrating} 个协整关系")
    return result
```

## 从协整到交易信号：Z-score 策略

### 构建交易信号

协整关系给了你一个平稳的价差序列 $Z_t$。交易信号的核心是把 $Z_t$ **标准化**：

$$
z_t = \frac{Z_t - \mu_Z}{\sigma_Z}
$$

其中 $\mu_Z$ 和 $\sigma_Z$ 用滚动窗口估计（如过去 20-60 个交易日）。

**交易规则**（经典 Z-score 策略）：

| Z-score | 操作 | 逻辑 |
|---|---|---|
| $z_t < -2$ | 买入组合（买 Y，卖 X） | 价差过低，预期回归 |
| $-2 < z_t < 2$ | 平仓或观望 | 价差在正常区间 |
| $z_t > 2$ | 卖出组合（卖 Y，买 X） | 价差过高，预期回归 |

```python
class PairsTradingStrategy:
    """
    配对交易策略引擎
    
    功能：
    1. 计算对冲比例（OLS 或 TLS）
    2. 构建 Z-score 信号
    3. 回测策略表现
    4. 计算交易成本影响
    """
    
    def __init__(self, lookback=20, entry_z=2.0, exit_z=0.5):
        self.lookback = lookback
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.spread = None
        self.z_score = None
        self.positions = None
        
    def fit_hedge_ratio(self, price_y, price_x, method='ols', half_life=None):
        """
        估计对冲比例
        
        参数：
        - method: 'ols'（普通最小二乘）或 'tls'（总最小二乘，考虑 X 的测量误差）
        - half_life: TLS 的指数加权半衰期
        """
        if method == 'ols':
            result = cointegration_regression(price_y, price_x)
            self.beta = result['beta']
            self.alpha = result['alpha']
            self.spread = price_y - (self.alpha + self.beta * price_x)
            
        elif method == 'tls':
            # Total Least Squares：用 SVD 解，同时考虑 X 和 Y 的误差
            # 对于金融数据，TLS 通常比 OLS 更稳健
            X_centered = price_x - price_x.mean()
            Y_centered = price_y - price_y.mean()
            data = np.column_stack([X_centered, Y_centered])
            
            U, S, Vt = np.linalg.svd(data, full_matrices=False)
            self.beta = -Vt[1, 0] / Vt[1, 1]  # TLS 解
            self.alpha = price_y.mean() - self.beta * price_x.mean()
            self.spread = price_y - (self.alpha + self.beta * price_x)
            
        return self.beta, self.alpha
    
    def compute_z_score(self, spread=None):
        """计算滚动 Z-score"""
        if spread is None:
            spread = self.spread
            
        spread_series = pd.Series(spread)
        rolling_mean = spread_series.rolling(self.lookback).mean()
        rolling_std = spread_series.rolling(self.lookback).std()
        
        self.z_score = (spread - rolling_mean) / rolling_std
        return self.z_score
    
    def generate_signals(self, z_score=None):
        """
        生成交易信号
        
        返回：
        - signals: Series, 1=做多价差, -1=做空价差, 0=平仓/观望
        """
        if z_score is None:
            z_score = self.z_score
            
        signals = pd.Series(0, index=np.arange(len(z_score)))
        
        # 入场信号
        signals[z_score < -self.entry_z] = 1   # 买入价差（买Y卖X）
        signals[z_score > self.entry_z] = -1    # 卖出价差（卖Y买X）
        
        # 平仓信号：Z-score 回归到 exit_z 以内
        for i in range(1, len(signals)):
            if signals[i-1] != 0 and abs(z_score[i]) < self.exit_z:
                signals[i] = 0  # 平仓
            elif signals[i-1] != 0:
                signals[i] = signals[i-1]  # 维持仓位
        
        self.positions = signals
        return signals
    
    def backtest(self, price_y, price_x, signals=None, txn_cost=0.001):
        """
        回测配对策略
        
        参数：
        - txn_cost: 单边交易成本（含佣金和滑点），如 0.001 = 0.1%
        
        返回：
        - results: DataFrame，含每日收益、累计净值等
        """
        if signals is None:
            signals = self.positions
            
        # 计算每日收益（假设等金额对冲）
        # 每日价差变化 × 持仓方向
        spread_ret = pd.Series(self.spread).pct_change()
        
        # 策略收益 = 持仓方向 × 价差收益 - 交易成本
        strategy_ret = signals.shift(1) * spread_ret
        
        # 计算交易成本（每次换仓时扣除）
        position_change = signals.diff().fillna(0).abs()
        txn_cost_series = position_change * txn_cost
        strategy_ret_net = strategy_ret - txn_cost_series
        
        # 累计净值
        cumulative = (1 + strategy_ret_net).cumprod()
        
        results = pd.DataFrame({
            'spread': self.spread,
            'z_score': self.z_score,
            'signal': signals,
            'daily_ret': strategy_ret_net,
            'cumulative': cumulative,
        })
        
        # 绩效指标
        total_ret = cumulative.iloc[-1] - 1
        ann_vol = strategy_ret_net.std() * np.sqrt(252)
        sharpe = (strategy_ret_net.mean() / strategy_ret_net.std()) * np.sqrt(252)
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        print(f"总收益: {total_ret:.2%}")
        print(f"年化波动率: {ann_vol:.2%}")
        print(f"Sharpe 比率: {sharpe:.2f}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"交易次数: {(position_change > 0).sum()}")
        
        return results
```

![配对交易 Z-score 信号图](/images/pair-trading-cointegration/pairs_trading_signals.png)
*图2：配对股票价差 Z-score 与交易信号示意图。当 Z-score 突破 ±2σ 时触发交易信号，回归 ±0.5σ 时平仓*

## A 股配对交易实证案例

### 案例：中国平安（601318.SH）vs 中国人寿（601628.SH）

这两只保险行业龙头，业务高度相似，是 A 股经典的配对交易候选。下面用 `westock-data` 获取真实数据并检验协整关系：

```python
# 注：以下代码需要 westock-data skill 支持
# 这里用模拟数据展示完整分析流程

np.random.seed(20240621)
T = 1260  # 5年数据

# 模拟平安和人寿的协整价格
X_pingan = 50 + np.cumsum(np.random.randn(T) * 0.015)
beta_ins = 0.92
spread_ins = np.random.randn(T) * 1.2
spread_ins = pd.Series(spread_ins).ewm(alpha=0.05).mean() + np.random.randn(T) * 0.3
Y_chinalife = 10 + beta_ins * X_pingan + spread_ins

# 协整检验
result = cointegration_regression(Y_chinalife, X_pingan)
adf = adf_test(result['residual'], verbose=False)

print(f"对冲比例 beta: {result['beta']:.4f}")
print(f"ADF p-value: {adf['p_value']:.4f}")
print(f"协整关系: {'成立' if adf['is_cointegrated'] else '不成立'}")

# 运行配对策略
strategy = PairsTradingStrategy(lookback=20, entry_z=2.0, exit_z=0.5)
strategy.fit_hedge_ratio(Y_chinalife, X_pingan, method='tls')
strategy.compute_z_score()
signals = strategy.generate_signals()
results = strategy.backtest(Y_chinalife, X_pingan, signals, txn_cost=0.0012)

# 可视化
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# 子图1：两只股票价格
axes[0].plot(X_pingan, label='中国平安', linewidth=1.5)
axes[0].plot(Y_chinalife, label='中国人寿（缩放后）', linewidth=1.5, alpha=0.7)
axes[0].set_title('配对股票价格走势')
axes[0].legend()

# 子图2：价差 Z-score 与信号
axes[1].plot(strategy.z_score, color='blue', linewidth=1.2, label='Z-score')
axes[1].axhline(y=2, color='red', linestyle='--', label='入场阈值(+2σ)')
axes[1].axhline(y=-2, color='green', linestyle='--', label='入场阈值(-2σ)')
axes[1].axhline(y=0, color='gray', linestyle='-', alpha=0.5)
buy_idx = np.where(signals == 1)[0]
sell_idx = np.where(signals == -1)[0]
axes[1].scatter(buy_idx, strategy.z_score[buy_idx], color='red', s=20, marker='^', label='买入信号')
axes[1].scatter(sell_idx, strategy.z_score[sell_idx], color='green', s=20, marker='v', label='卖出信号')
axes[1].set_title('Z-score 交易信号')
axes[1].legend()

# 子图3：策略累计净值
axes[2].plot(results['cumulative'], color='purple', linewidth=2, label='配对策略')
axes[2].axhline(y=1, color='gray', linestyle='--', alpha=0.5)
axes[2].set_title('策略累计净值')
axes[2].set_xlabel('交易日')
axes[2].legend()

plt.tight_layout()
plt.savefig('pairs_trading_result.png', dpi=150, bbox_inches='tight')
```

![配对交易策略累计净值](/images/pair-trading-cointegration/pairs_cumulative_return.png)
*图3：配对交易策略模拟累计净值（2年），展现均值回归策略的收益特征*

## 配对交易的风险管理

### 1. 协整关系断裂风险

协整不是永恒不变的。行业格局变化、公司并购、监管政策转向，都可能导致历史上的协整关系在未来失效。

**应对方法**：
- 用**滚动窗口**定期重新检验协整关系（如每季度）
- 当 ADF p-value 持续 > 0.10 时，停止交易该配对
- 设置**最大持仓时间**（如 20 个交易日），强制平仓

```python
def rolling_cointegration_test(price_y, price_x, window=252, step=63):
    """
    滚动协整检验：监控协整关系是否持续有效
    
    参数：
    - window: 滚动窗口大小（交易日）
    - step: 检验频率（每 step 个交易日检验一次）
    """
    results = []
    for start in range(0, len(price_y) - window, step):
        end = start + window
        y_win = price_y[start:end]
        x_win = price_x[start:end]
        
        beta, alpha = cointegration_regression(y_win, x_win)['beta'], cointegration_regression(y_win, x_win)['alpha']
        residual = y_win - (alpha + beta * x_win)
        p_val = adfuller(residual, autolag='AIC')[1]
        
        results.append({
            'start': start,
            'end': end,
            'beta': beta,
            'adf_pvalue': p_val,
            'is_cointegrated': p_val < 0.05
        })
    
    return pd.DataFrame(results)
```

### 2. 价差发散风险（Loss of Mean Reversion）

即使协整关系成立，价差也可能在短期内进一步发散（Z-score 从 2 走到 3 甚至 4）。这是配对交易最大的实盘风险——**「均值回归」不代表「马上回归」**。

**应对方法**：
- 设置**止损线**：当 Z-score 超过 ±3.5 时强制止损
- 用**仓位管理**而非全仓：首次信号用 50% 仓位，Z-score 继续极端时加仓（Martingale 需谨慎）
- 关注**基本面变化**：如果两只股票的基本面同时恶化（如行业性利空），价差可能永久性地台阶式偏移

### 3. 交易成本侵蚀

配对交易通常换手率较高（尤其是 Z-score 阈值设得较小时）。A 股双边交易成本约 0.12%-0.20%（佣金 + 印花税 + 滑点）。

**经验法则**：配对策略的**日均收益**必须 > 2 × 单边交易成本 / 平均持仓天数。对于 A 股，这意味着策略 Sharpe 必须 > 1.2 才有实盘价值。

## 配对交易的局限性与扩展

### 局限性

1. **可扩展性差**：找到 10 个有效配对已经很难，管理 1000 万以上资金时，配对机会稀缺
2. **模型风险**：协整检验的势（power）在样本量较小时很低，容易把噪声当信号
3. **尾部风险**：价差在极端市场条件下（如熔断、停牌）可能无法及时平仓

### 扩展方向

**多资产配对（Multivariate Cointegration）**：不止两只股票，而是一个「组合 vs 组合」的协整关系。用 Johansen 检验找到多个协整向量，构建市场中性的多空组合。

**机器学习增强**：用随机森林或 LSTM 预测 Z-score 的**回归速度**（半衰期），动态调整持仓时间，而非固定用 Z-score 穿越 0.5 作为平仓信号。

**高频配对**：将协整分析应用到分钟级或秒级数据，捕捉盘中价格的短暂偏离。需要极低的交易成本和稳定的数据接入。

## 总结

配对交易是量化投资中最优雅的策略之一——它不依赖「市场上涨」,不依赖「因子溢价」，只依赖一个简单而强大的数学事实：**协整价差的均值回归性**。

但要真正做好配对交易，你需要：

1. **严谨的协整检验**：ADF / Johansen 是门槛，不是可选项
2. **动态的参数更新**：协整关系会死，你要比它死得早知道
3. **务实的成本管理**：换手率决定生死，不要忽略交易成本
4. **足够大的配对池**：单一配对不足以支撑一个基金，你需要系统化的配对挖掘流程

配对交易不会让你一夜暴富，但它可以在市场大跌时依然赚钱——如果你做对了的话。

---

**延伸阅读**：
- Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). *Pairs Trading: Performance of a Relative-Value Arbitrage Rule*. Review of Financial Studies. （配对交易经典实证论文）
- Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley. （协整在配对交易中的系统应用）
- Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). *Pairs Trading*. Quantitative Finance. （状态空间模型下的配对交易框架）
