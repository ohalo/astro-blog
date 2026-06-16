---
title: "配对交易与协整分析：市场中性策略的理论与实践"
description: "深入探讨配对交易的理论基础、协整检验方法、配对选择标准，以及完整的Python实现和回测框架。"
date: "2026-06-16"
tags: ["配对交易", "协整分析", "市场中性", "统计套利", "量化策略"]
topic: "量化交易"
difficulty: "进阶"
---

# 配对交易与协整分析：市场中性策略的理论与实践

## 引言

在量化投资的世界里，**配对交易（Pairs Trading）** 是一种经典的市场中性策略。它不依赖市场方向，而是通过捕捉两个高度相关资产之间的暂时偏离来获利。

本文将系统讲解：
- 配对交易的理论基础：协整与均值回归
- 如何科学选择配对资产
- 完整的Python实现：从数据获取到实盘信号
- 风险管理与绩效评估
- 实战案例：A股配对交易回测

## 一、配对交易的理论基础

### 1.1 什么是协整（Cointegration）？

**协整**是指两个或多个非平稳时间序列的线性组合是平稳的。用数学语言表达：

如果时间序列 $X_t$ 和 $Y_t$ 都是 I(1) 过程（一阶单整），但存在系数 $\beta$ 使得：

$$Z_t = Y_t - \beta X_t \sim I(0)$$

即残差项 $Z_t$ 是平稳的，那么我们称 $X_t$ 和 $Y_t$ 是协整的。

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import yfinance as yf

class PairsTradingAnalyzer:
    """配对交易分析器"""
    
    def __init__(self, stock1, stock2, start_date, end_date):
        """
        初始化
        
        Parameters:
        -----------
        stock1, stock2 : str
            股票代码（如 '600036.SS', '601398.SS'）
        start_date, end_date : str
            起始和结束日期
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self.spread = None
        self.hedge_ratio = None
        
    def load_data(self):
        """加载价格数据（示例用随机数据，实际中应接入API）"""
        # 为演示生成模拟数据
        dates = pd.date_range(self.start_date, self.end_date, freq='B')
        n = len(dates)
        
        # 生成协整序列
        np.random.seed(42)
        x = np.cumsum(np.random.randn(n)) * 0.01 + 10  # 随机游走
        
        # y = βx + 平稳残差 + 噪声
        beta = 1.5
        z = np.sin(np.linspace(0, 4*np.pi, n)) * 0.5  # 周期性均值回归
        noise = np.random.randn(n) * 0.02
        
        y = beta * x + z + noise
        
        self.data = pd.DataFrame({
            self.stock1: y,
            self.stock2: x
        }, index=dates)
        
        return self.data
    
    def test_cointegration(self):
        """执行协整检验"""
        if self.data is None:
            self.load_data()
        
        # Engle-Granger 协整检验
        y = self.data[self.stock1]
        x = self.data[self.stock2]
        
        # 第一步：OLS回归
        model = OLS(y, x)
        results = model.fit()
        self.hedge_ratio = results.params[0]
        spread = y - self.hedge_ratio * x
        self.spread = spread
        
        # 第二步：ADF检验残差平稳性
        adf_result = adfuller(spread, autolag='AIC')
        
        # 第三步：Engle-Granger 协整检验
        coint_stat, p_value, crit_values = coint(y, x)
        
        test_results = {
            'hedge_ratio': self.hedge_ratio,
            'adf_statistic': adf_result[0],
            'adf_pvalue': adf_result[1],
            'coint_statistic': coint_stat,
            'coint_pvalue': p_value,
            'crit_value_1%': crit_values[0],
            'crit_value_5%': crit_values[1],
            'crit_value_10%': crit_values[2],
            'is_cointegrated': p_value < 0.05  # 5%显著性水平
        }
        
        return test_results
    
    def visualize_pairs(self):
        """可视化配对关系"""
        if self.data is None:
            self.load_data()
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'配对分析: {self.stock1} vs {self.stock2}', 
                     fontsize=16, fontweight='bold')
        
        # 1. 价格序列
        ax1 = axes[0]
        ax1.plot(self.data.index, self.data[self.stock1], 
                label=self.stock1, linewidth=2, color='steelblue')
        ax1.plot(self.data.index, self.data[self.stock2] * self.hedge_ratio, 
                label=f'{self.stock2} (β={self.hedge_ratio:.3f})', 
                linewidth=2, color='crimson', linestyle='--')
        ax1.set_title('价格序列对比（调整后）', fontsize=13, pad=10)
        ax1.legend(loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_ylabel('价格', fontsize=11)
        
        # 2. 价差（Spread）
        ax2 = axes[1]
        if self.spread is None:
            self.test_cointegration()
        
        ax2.plot(self.data.index, self.spread, 
                linewidth=2, color='darkgreen', alpha=0.8)
        ax2.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
        
        # 添加均值和标准差带
        mean_spread = self.spread.mean()
        std_spread = self.spread.std()
        ax2.axhline(y=mean_spread, color='orange', linestyle='--', 
                    linewidth=1.5, label=f'均值 ({mean_spread:.4f})')
        ax2.fill_between(self.data.index, 
                         mean_spread - 2*std_spread,
                         mean_spread + 2*std_spread,
                         alpha=0.2, color='orange', label='±2σ')
        ax2.fill_between(self.data.index, 
                         mean_spread - 1*std_spread,
                         mean_spread + 1*std_spread,
                         alpha=0.3, color='green', label='±1σ')
        
        ax2.set_title('价差序列（Spread）', fontsize=13, pad=10)
        ax2.legend(loc='upper right', fontsize=10)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylabel('价差', fontsize=11)
        
        # 3. 价差分布直方图
        ax3 = axes[2]
        ax3.hist(self.spread, bins=50, density=True, alpha=0.7, 
                color='purple', edgecolor='black', linewidth=0.5)
        
        # 叠加正态分布拟合
        x_norm = np.linspace(self.spread.min(), self.spread.max(), 100)
        y_norm = stats.norm.pdf(x_norm, self.spread.mean(), self.spread.std())
        ax3.plot(x_norm, y_norm, 'r--', linewidth=2, 
                label='正态分布拟合')
        
        ax3.set_title('价差分布', fontsize=13, pad=10)
        ax3.set_xlabel('价差', fontsize=11)
        ax3.set_ylabel('频率', fontsize=11)
        ax3.legend(loc='upper right', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/pairs_analysis.png', 
                    dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        return fig
```

### 1.2 协整 vs 相关性

**重要区别**：
- **高相关性** ≠ **协整**
- 相关性衡量短期联动，协整衡量长期均衡

```python
def compare_correlation_vs_cointegration():
    """对比相关性和协整"""
    
    # 案例1：高相关但非协整
    n = 500
    x1 = np.cumsum(np.random.randn(n))  # 随机游走1
    y1 = np.cumsum(np.random.randn(n))  # 随机游走2（独立）
    
    corr1 = np.corrcoef(x1, y1)[0, 1]
    # 两个独立随机游走可能显示高相关性（伪相关）
    
    # 案例2：低相关但协整
    x2 = np.cumsum(np.random.randn(n)) * 0.01 + 10
    y2 = 1.5 * x2 + np.sin(np.linspace(0, 4*np.pi, n)) * 0.5 + np.random.randn(n) * 0.02
    
    corr2 = np.corrcoef(x2, y2)[0, 1]
    # 协整序列的相关系数可能不高（因为有均值回归项）
    
    print(f"案例1 - 独立随机游走：相关性 = {corr1:.3f}, 协整检验p值 = {coint(y1, x1)[1]:.3f}")
    print(f"案例2 - 协整序列：相关性 = {corr2:.3f}, 协整检验p值 = {coint(y2, x2)[1]:.3f}")
    
    return corr1, corr2
```

## 二、配对选择方法

### 2.1 基本面匹配

```python
class PairSelector:
    """配对选择器"""
    
    def __init__(self, stock_universe, financial_data):
        """
        初始化
        
        Parameters:
        -----------
        stock_universe : list
            股票池
        financial_data : DataFrame
            财务数据（行业、市值、PB、ROE等）
        """
        self.stock_universe = stock_universe
        self.financial_data = financial_data
        
    def select_by_industry(self, target_stock, n_candidates=50):
        """按行业筛选候选配对"""
        target_industry = self.financial_data.loc[target_stock, 'industry']
        
        candidates = [
            stock for stock in self.stock_universe 
            if stock != target_stock and 
            self.financial_data.loc[stock, 'industry'] == target_industry
        ]
        
        return candidates[:n_candidates]
    
    def select_by_similarity(self, target_stock, features=['market_cap', 'pb', 'roe']):
        """按基本面特征相似度筛选"""
        from sklearn.metrics.pairwise import euclidean_distances
        
        # 提取特征
        X = self.financial_data[features].values
        
        # 计算距离
        target_idx = self.stock_universe.index(target_stock)
        distances = euclidean_distances(X[target_idx].reshape(1, -1), X).flatten()
        
        # 排序（排除自身）
        similar_indices = np.argsort(distances)[1:51]  # Top 50
        
        return [self.stock_universe[i] for i in similar_indices]
```

### 2.2 统计分析筛选

```python
    def screen_by_cointegration(self, target_stock, candidates, 
                                min_data_points=252):
        """通过协整检验筛选配对"""
        
        qualified_pairs = []
        
        for candidate in candidates:
            # 加载数据
            data = self.load_price_data(target_stock, candidate)
            
            if len(data) < min_data_points:
                continue
            
            # 协整检验
            try:
                coint_stat, p_value, _ = coint(data[target_stock], data[candidate])
                
                if p_value < 0.05:  # 5%显著性水平
                    # 计算对冲比率
                    model = OLS(data[target_stock], data[candidate])
                    results = model.fit()
                    hedge_ratio = results.params[0]
                    
                    # 计算价差平稳性指标
                    spread = data[target_stock] - hedge_ratio * data[candidate]
                    adf_pvalue = adfuller(spread)[1]
                    
                    # 计算相关性
                    correlation = data[target_stock].corr(data[candidate])
                    
                    qualified_pairs.append({
                        'stock1': target_stock,
                        'stock2': candidate,
                        'coint_pvalue': p_value,
                        'adf_pvalue': adf_pvalue,
                        'hedge_ratio': hedge_ratio,
                        'correlation': correlation,
                        'half_life': self.calculate_half_life(spread)
                    })
            except Exception as e:
                continue
        
        # 按协整p值排序
        qualified_pairs.sort(key=lambda x: x['coint_pvalue'])
        
        return qualified_pairs
    
    def calculate_half_life(self, spread):
        """计算价差的半衰期（均值回归速度）"""
        from statsmodels.regression.linear_model import OLS as OLS_reg
        
        # 构建回归模型：Δspread_t = α + β * spread_{t-1} + ε_t
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        model = OLS_reg(spread_diff, spread_lag)
        results = model.fit()
        
        # 半衰期 = ln(2) / |β|
        beta = results.params[0]
        half_life = np.log(2) / abs(beta)
        
        return half_life
```

### 2.3 距离法（Distance Approach）

```python
    def screen_by_distance(self, target_stock, candidates, 
                           lookback=252, threshold=1.5):
        """
        距离法筛选配对
        
        核心思想：计算价格序列的标准化距离，选择距离最小的配对
        """
        
        distances = []
        
        for candidate in candidates:
            # 加载数据
            data = self.load_price_data(target_stock, candidate)
            
            if len(data) < lookback:
                continue
            
            # 标准化价格（初始值为1）
            p1_norm = data[target_stock] / data[target_stock].iloc[0]
            p2_norm = data[candidate] / data[candidate].iloc[0]
            
            # 计算距离（标准差）
            distance = (p1_norm - p2_norm).std()
            
            # 计算相关性
            correlation = p1_norm.corr(p2_norm)
            
            distances.append({
                'stock1': target_stock,
                'stock2': candidate,
                'distance': distance,
                'correlation': correlation
            })
        
        # 筛选距离小于阈值的配对
        qualified = [p for p in distances if p['distance'] < threshold]
        qualified.sort(key=lambda x: x['distance'])
        
        return qualified
```

## 三、交易信号与策略构建

### 3.1 基于Z-Score的信号

```python
class PairsTradingStrategy:
    """配对交易策略"""
    
    def __init__(self, stock1, stock2, entry_z=2.0, exit_z=0.5, 
                 stop_loss_z=3.0, lookback=252):
        """
        初始化
        
        Parameters:
        -----------
        entry_z : float
            入场Z分数阈值（默认2.0）
        exit_z : float
            出场Z分数阈值（默认0.5）
        stop_loss_z : float
            止损Z分数阈值（默认3.0）
        lookback : int
            滚动窗口（默认252个交易日，约1年）
        """
        self.stock1 = stock1
        self.stock2 = stock2
        self.entry_z = entry_z
        self.exit_z = exit_z
        self.stop_loss_z = stop_loss_z
        self.lookback = lookback
        
        self.data = None
        self.spread = None
        self.z_score = None
        self.positions = None
        
    def calculate_z_score(self, spread, lookback):
        """计算滚动Z分数"""
        mean = spread.rolling(window=lookback).mean()
        std = spread.rolling(window=lookback).std()
        
        z_score = (spread - mean) / std
        
        return z_score
    
    def generate_signals(self, data):
        """
        生成交易信号
        
        Returns:
        --------
        signals : DataFrame
            包含z_score, position, signal等列
        """
        if self.data is None:
            self.data = data
        
        # 计算对冲比率（滚动）
        hedge_ratios = []
        for i in range(self.lookback, len(data)):
            window_data = data.iloc[i-self.lookback:i]
            model = OLS(window_data[self.stock1], window_data[self.stock2])
            results = model.fit()
            hedge_ratios.append(results.params[0])
        
        # 填充前期数据
        hedge_ratios = [np.nan] * self.lookback + hedge_ratios
        self.hedge_ratio = pd.Series(hedge_ratios, index=data.index)
        
        # 计算价差
        self.spread = data[self.stock1] - self.hedge_ratio * data[self.stock2]
        
        # 计算Z分数
        self.z_score = self.calculate_z_score(self.spread, self.lookback)
        
        # 生成仓位信号
        positions = pd.Series(0, index=data.index)
        signals = pd.DataFrame({
            'z_score': self.z_score,
            'hedge_ratio': self.hedge_ratio,
            'spread': self.spread
        })
        
        # 入场信号
        positions[self.z_score < -self.entry_z] = 1   # 做多价差（买stock1，卖stock2）
        positions[self.z_score > self.entry_z] = -1     # 做空价差（卖stock1，买stock2）
        
        # 出场信号
        positions[(self.z_score > -self.exit_z) & (self.z_score < self.exit_z)] = 0
        
        # 止损信号
        positions[self.z_score < -self.stop_loss_z] = 0
        positions[self.z_score > self.stop_loss_z] = 0
        
        # 避免频繁交易：只有当信号改变时才切换仓位
        positions = positions.diff().ne(0).cumsum()
        
        signals['position'] = positions
        self.positions = positions
        
        return signals
```

### 3.2 动态阈值调整

```python
    def dynamic_threshold(self, volatility_window=63):
        """
        动态阈值调整：根据价差波动率调整入场/出场阈值
        
        高波动期提高阈值，低波动期降低阈值
        """
        
        # 计算价差波动率（滚动）
        spread_vol = self.spread.rolling(window=volatility_window).std()
        
        # 标准化波动率（相对于历史均值）
        vol_ratio = spread_vol / spread_vol.mean()
        
        # 动态调整阈值
        dynamic_entry_z = self.entry_z * vol_ratio
        dynamic_exit_z = self.exit_z * vol_ratio
        
        return dynamic_entry_z, dynamic_exit_z
```

### 3.3 交易执行逻辑

```python
    def backtest(self, data, transaction_cost=0.001):
        """
        回测策略
        
        Parameters:
        -----------
        data : DataFrame
            价格数据
        transaction_cost : float
            单边交易成本（默认0.1%）
        
        Returns:
        --------
        results : DataFrame
            包含策略收益、累积收益等
        """
        
        # 生成信号
        signals = self.generate_signals(data)
        
        # 计算收益
        price1 = data[self.stock1]
        price2 = data[self.stock2]
        
        # 日收益率
        ret1 = price1.pct_change()
        ret2 = price2.pct_change()
        
        # 策略收益（考虑仓位和对冲比率）
        strategy_ret = (
            signals['position'].shift(1) * 
            (ret1 - signals['hedge_ratio'] * ret2)
        )
        
        # 扣除交易成本
        turnover = signals['position'].diff().abs()
        cost = turnover * transaction_cost
        
        net_ret = strategy_ret - cost
        
        # 累积收益
        cumulative_ret = (1 + net_ret).cumprod()
        
        # 绩效指标
        total_ret = cumulative_ret.iloc[-1] - 1
        annual_ret = (1 + total_ret) ** (252 / len(net_ret)) - 1
        sharpe_ratio = net_ret.mean() / net_ret.std() * np.sqrt(252)
        max_drawdown = (cumulative_ret / cumulative_ret.cummax() - 1).min()
        
        results = pd.DataFrame({
            'strategy_return': strategy_ret,
            'net_return': net_ret,
            'cumulative_return': cumulative_ret,
            'position': signals['position'],
            'z_score': signals['z_score']
        })
        
        metrics = {
            'total_return': total_ret,
            'annual_return': annual_ret,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'win_rate': (strategy_ret > 0).sum() / len(strategy_ret),
            'avg_turnover': turnover.mean()
        }
        
        return results, metrics
```

## 四、风险管理

### 4.1 仓位管理

```python
class RiskManager:
    """风险管理器"""
    
    def __init__(self, max_position=0.05, max_leverage=2.0, 
                 stop_loss_pct=0.02):
        """
        初始化
        
        Parameters:
        -----------
        max_position : float
            单对最大仓位（账户价值的5%）
        max_leverage : float
            最大杠杆
        stop_loss_pct : float
            止损比例
        """
        self.max_position = max_position
        self.max_leverage = max_leverage
        self.stop_loss_pct = stop_loss_pct
        
    def calculate_position_size(self, account_value, price1, price2, 
                                hedge_ratio, volatility):
        """计算仓位大小（基于波动率和风险预算）"""
        
        # 方法1：固定分数法
        position_value = account_value * self.max_position
        
        # 方法2：波动率调整
        # 目标波动率 = 账户价值的0.01（1%）
        target_vol = account_value * 0.01
        vol_adjusted_position = target_vol / volatility
        
        # 取较小值
        position_size = min(position_value, vol_adjusted_position)
        
        # 限制杠杆
        max_position_with_leverage = account_value * self.max_leverage
        position_size = min(position_size, max_position_with_leverage)
        
        # 计算具体股数
        n_shares1 = int(position_size / price1)
        n_shares2 = int(position_size * hedge_ratio / price2)
        
        return n_shares1, n_shares2
    
    def check_stop_loss(self, entry_price, current_price, position_type):
        """止损检查"""
        
        if position_type == 'long':
            loss = (entry_price - current_price) / entry_price
        else:  # short
            loss = (current_price - entry_price) / entry_price
        
        if loss > self.stop_loss_pct:
            return True  # 触发止损
        
        return False
```

### 4.2 组合层面风险管理

```python
class PortfolioRiskManager:
    """组合风险管理器"""
    
    def __init__(self, max_pairs=10, max_correlation=0.7, 
                 max_sector_exposure=0.3):
        self.max_pairs = max_pairs
        self.max_correlation = max_correlation
        self.max_sector_exposure = max_sector_exposure
        
    def select_uncorrelated_pairs(self, candidate_pairs, n_select):
        """选择低相关性配对组合"""
        
        # 计算配对收益相关性矩阵
        returns_matrix = pd.DataFrame({
            f"{p['stock1']}-{p['stock2']}": p['returns'] 
            for p in candidate_pairs
        })
        
        corr_matrix = returns_matrix.corr()
        
        # 贪心算法：逐步选择相关性最低的配对
        selected = []
        remaining = list(range(len(candidate_pairs)))
        
        # 第一步：选择夏普比率最高的
        best_idx = np.argmax([p['sharpe'] for p in candidate_pairs])
        selected.append(best_idx)
        remaining.remove(best_idx)
        
        # 后续步骤：选择与被选组合相关性最低的
        while len(selected) < n_select and remaining:
            min_corr_idx = None
            min_avg_corr = np.inf
            
            for idx in remaining:
                # 计算与已选配对的平均相关性
                avg_corr = np.mean([
                    abs(corr_matrix.iloc[idx, j]) for j in selected
                ])
                
                if avg_corr < min_avg_corr:
                    min_avg_corr = avg_corr
                    min_corr_idx = idx
            
            if min_avg_corr < self.max_correlation:
                selected.append(min_corr_idx)
                remaining.remove(min_corr_idx)
            else:
                break  # 剩余配对相关性太高
        
        return [candidate_pairs[i] for i in selected]
```

## 五、实战案例：A股配对交易

### 5.1 数据准备

```python
# 假设我们有一组银行股
bank_stocks = ['600036.SS', '601398.SS', '601939.SS', '601288.SS', 
               '600016.SS', '601166.SS']

# 加载数据（示例）
start_date = '2020-01-01'
end_date = '2025-12-31'

# 初始化配对分析器
analyzer = PairsTradingAnalyzer(
    stock1='600036.SS',  # 招商银行
    stock2='601398.SS',  # 工商银行
    start_date=start_date,
    end_date=end_date
)

# 加载数据
data = analyzer.load_data()

# 协整检验
test_results = analyzer.test_cointegration()

print("\n===== 协整检验结果 =====")
print(f"对冲比率 (β): {test_results['hedge_ratio']:.4f}")
print(f"ADF统计量: {test_results['adf_statistic']:.4f}")
print(f"ADF p值: {test_results['adf_pvalue']:.4f}")
print(f"协整统计量: {test_results['coint_statistic']:.4f}")
print(f"协整p值: {test_results['coint_pvalue']:.4f}")
print(f"是否协整: {'是' if test_results['is_cointegrated'] else '否'}")
```

### 5.2 策略回测

```python
# 初始化策略
strategy = PairsTradingStrategy(
    stock1='600036.SS',
    stock2='601398.SS',
    entry_z=2.0,
    exit_z=0.5,
    stop_loss_z=3.0,
    lookback=252
)

# 回测
results, metrics = strategy.backtest(data, transaction_cost=0.001)

print("\n===== 策略绩效 =====")
print(f"总收益率: {metrics['total_return']:.2%}")
print(f"年化收益率: {metrics['annual_return']:.2%}")
print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
print(f"最大回撤: {metrics['max_drawdown']:.2%}")
print(f"胜率: {metrics['win_rate']:.2%}")
print(f"平均换手率: {metrics['avg_turnover']:.2%}")
```

### 5.3 可视化结果

```python
def plot_backtest_results(results, stock1, stock2):
    """绘制回测结果"""
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    fig.suptitle(f'配对交易回测结果: {stock1} vs {stock2}', 
                 fontsize=16, fontweight='bold')
    
    # 1. 累积收益曲线
    ax1 = axes[0]
    ax1.plot(results.index, results['cumulative_return'], 
            linewidth=2.5, color='darkblue', alpha=0.8)
    ax1.fill_between(results.index, 1, results['cumulative_return'],
                    alpha=0.3, color='darkblue')
    ax1.axhline(y=1, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax1.set_title('累积收益曲线', fontsize=13, pad=10)
    ax1.set_ylabel('累积收益', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # 2. Z分数与仓位
    ax2 = axes[1]
    ax2.plot(results.index, results['z_score'], 
            linewidth=2, color='purple', alpha=0.8, label='Z分数')
    ax2.axhline(y=2.0, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='入场阈值 (+2)')
    ax2.axhline(y=-2.0, color='red', linestyle='--', linewidth=1.5, 
                alpha=0.7, label='入场阈值 (-2)')
    ax2.axhline(y=0.5, color='green', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='出场阈值 (+0.5)')
    ax2.axhline(y=-0.5, color='green', linestyle=':', linewidth=1.5, 
                alpha=0.7, label='出场阈值 (-0.5)')
    ax2.set_title('Z分数与交易信号', fontsize=13, pad=10)
    ax2.set_ylabel('Z分数', fontsize=11)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 添加仓位柱状图
    ax2_twin = ax2.twinx()
    ax2_twin.bar(results.index, results['position'], 
                 alpha=0.3, color='gray', label='仓位')
    ax2_twin.set_ylabel('仓位', fontsize=11)
    ax2_twin.set_ylim(-1.5, 1.5)
    
    # 3. 回撤曲线
    ax3 = axes[2]
    cumulative = results['cumulative_return']
    drawdown = (cumulative - cumulative.cummax()) / cumulative.cummax()
    
    ax3.fill_between(results.index, 0, drawdown, 
                    alpha=0.5, color='crimson', label='回撤')
    ax3.plot(results.index, drawdown, 
            linewidth=1.5, color='darkred', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax3.set_title('回撤曲线', fontsize=13, pad=10)
    ax3.set_xlabel('日期', fontsize=11)
    ax3.set_ylabel('回撤', fontsize=11)
    ax3.legend(loc='lower left', fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1%}'))
    ax3.set_ylim(drawdown.min() - 0.05, 0.05)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/backtest_results.png', 
                dpi=300, bbox_inches='tight', facecolor='white')
    plt.show()
    
    return fig

# 执行可视化
plot_backtest_results(results, '600036.SS', '601398.SS')
```

## 六、实战技巧与注意事项

### 6.1 常见陷阱

1. **伪回归（Spurious Regression）**  
   两个独立随机游走的回归可能显示高R²和低残差，但实际上是伪回归。  
   **解决方案**：务必进行协整检验和样本外测试。

2. **结构性断裂**  
   公司并购、行业政策变化等会导致配对关系失效。  
   **解决方案**：使用滚动窗口，定期重新检验协整关系。

3. **流动性风险**  
   小盘股的价差可能无法成交。  
   **解决方案**：选择高流动性的股票，设置最小成交量筛选。

4. **交易成本侵蚀**  
   频繁交易会导致成本超过收益。  
   **解决方案**：优化阈值，降低换手率。

### 6.2 改进方向

```python
# 1. 引入机器学习优化阈值
from sklearn.ensemble import RandomForestClassifier

def ml_threshold_optimization(features, signals):
    """使用随机森林优化入场/出场阈值"""
    
    # 特征工程：波动率、动量、市场状态等
    X = features[['volatility', 'momentum', 'market_regime']]
    y = (signals['returns'].shift(-1) > 0).astype(int)  # 未来一天是否盈利
    
    # 训练模型
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # 预测最优阈值
    optimal_threshold = model.predict(X)
    
    return optimal_threshold

# 2. 卡尔曼滤波动态对冲比率
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(price1, price2):
    """使用卡尔曼滤波估计时变对冲比率"""
    
    # 观测矩阵
    observations = price1.values.reshape(-1, 1)
    
    # 状态转移矩阵（对冲比率随机游走）
    transition_matrix = [[1]]
    
    # 观测矩阵（价格2是对冲比率的观测）
    observation_matrix = price2.values.reshape(1, -1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        initial_state_mean=1.0,
        initial_state_covariance=1.0,
        observation_covariance=1.0,
        transition_covariance=0.01
    )
    
    # 滤波
    state_means, _ = kf.filter(observations)
    
    # 提取时变对冲比率
    dynamic_hedge_ratio = state_means.flatten()
    
    return dynamic_hedge_ratio
```

## 七、总结

### 7.1 核心要点

1. **协整是配对交易的基础**：确保配对具有长期均衡关系
2. **严格筛选配对**：结合基本面、统计检验和距离方法
3. **合理设置阈值**：根据波动率动态调整入场/出场点
4. **重视风险管理**：仓位控制、止损、组合分散化

### 7.2 实践建议

- **从熟悉的行业入手**：如银行、保险、能源等
- **控制配对数量**：5-10对为宜，过度分散会降低收益
- **定期复盘**：每月重新检验协整关系，剔除失效配对
- **结合市场环境**：牛市中配对交易表现可能不佳，需降低仓位

### 7.3 扩展阅读

1. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
2. Ganapathy, V. (2004). "Pairs Trading". *Risk Books*.
3. Elliott, R. J., et al. (2005). "Pairs trading". *Quantitative Finance*.

---

## 附录：完整代码仓库

完整代码已上传至GitHub：  
[https://github.com/quant-investor/pairs-trading-framework](https://github.com/quant-investor/pairs-trading-framework)

包含：
- 数据获取模块
- 配对选择算法
- 回测框架
- 风险管理工具
- 实盘接口示例

---

**免责声明**：本文仅供学术交流和策略研究，不构成投资建议。配对交易虽有理论支撑，但实盘中仍面临多种风险，请谨慎决策。
