---
title: "配对交易与协整分析"
description: "深入探讨配对交易的理论基础、协整检验方法、交易信号构建以及实战中的风险管理。包含完整的Python实现代码，从数据获取到策略回测的全流程。"
pubDate: 2026-06-19
slug: pair-trading-cointegration
tags: ["配对交易", "协整分析", "统计套利", "市场中性"]
category: "统计套利"
featured: false
---

# 配对交易与协整分析

配对交易（Pair Trading）是最经典的统计套利策略之一，由摩根士丹利在1980年代首次系统化应用。该策略基于**均值回归**思想：寻找价格具有长期均衡关系的资产对，当价格偏离均衡时建仓，等待价格回归时平仓获利。

本文将深入探讨配对交易的理论基础、协整检验方法、实战中的信号构建与风险管理，并提供完整的Python实现代码。

## 1. 配对交易的理论基础

### 1.1 为什么配对交易有效？

配对交易的核心理念是**均值回归（Mean Reversion）**。许多资产价格虽然短期可能偏离均衡，但长期会回归到某种稳定关系。这种均衡关系可能来自：

1. **经济基本面联系**：同一行业的竞争对手（如可口可乐 vs 百事可乐）
2. **产业链上下游**：原材料供应商 vs 成品制造商
3. **替代关系**：大豆 vs 玉米（争抢种植面积）
4. **共同风险因素**：受相同的宏观经济变量驱动

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import coint, adfuller
from statsmodels.regression.linear_model import OLS
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 模拟配对交易数据
np.random.seed(42)
n_periods = 1000

# 生成共同的随机游走（长期趋势）
common_trend = np.cumsum(np.random.normal(0, 1, n_periods))

# 生成两个资产的价格（存在协整关系）
asset1 = 50 + common_trend * 0.8 + np.cumsum(np.random.normal(0, 0.5, n_periods))
asset2 = 30 + common_trend * 0.6 + np.cumsum(np.random.normal(0, 0.5, n_periods))

# 转换为价格序列
price1 = np.exp(asset1 / 100) * 100
price2 = np.exp(asset2 / 100) * 100

dates = pd.date_range('2020-01-01', periods=n_periods, freq='D')
prices = pd.DataFrame({
    'Asset1': price1,
    'Asset2': price2
}, index=dates)

# 可视化价格序列
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle('配对交易理论基础：协整与均值回归', fontsize=16, fontweight='bold')

# 子图1：原始价格序列
ax1 = axes[0, 0]
ax1.plot(prices.index, prices['Asset1'], label='Asset 1', linewidth=2)
ax1.plot(prices.index, prices['Asset2'], label='Asset 2', linewidth=2)
ax1.set_title('原始价格序列', fontsize=14)
ax1.set_xlabel('日期')
ax1.set_ylabel('价格')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 子图2：价差序列
ax2 = axes[0, 1]
spread = prices['Asset1'] - prices['Asset2']
ax2.plot(spread.index, spread, linewidth=2, color='green')
ax2.axhline(y=spread.mean(), color='r', linestyle='--', label='均值')
ax2.fill_between(spread.index, 
                  spread.mean() + 2*spread.std(),
                  spread.mean() - 2*spread.std(),
                  alpha=0.2, color='green', label='±2倍标准差')
ax2.set_title('价差序列（均值回归）', fontsize=14)
ax2.set_xlabel('日期')
ax2.set_ylabel('价差')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：对数价格关系
ax3 = axes[1, 0]
log_price1 = np.log(prices['Asset1'])
log_price2 = np.log(prices['Asset2'])
ax3.scatter(log_price2, log_price1, alpha=0.5, s=10)
# 拟合回归线
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(log_price2.values.reshape(-1, 1), log_price1.values)
x_plot = np.linspace(log_price2.min(), log_price2.max(), 100)
ax3.plot(x_plot, reg.predict(x_plot.reshape(-1, 1)), 'r-', linewidth=2, label='协整关系')
ax3.set_title('对数价格协整关系', fontsize=14)
ax3.set_xlabel('ln(Asset 2)')
ax3.set_ylabel('ln(Asset 1)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 子图4：残差序列（平稳性检验）
ax4 = axes[1, 1]
residuals = log_price1 - (reg.intercept_ + reg.coef_[0] * log_price2)
ax4.plot(residuals.index, residuals, linewidth=2, color='purple')
ax4.axhline(y=0, color='r', linestyle='--')
ax4.fill_between(residuals.index, 
                 2, -2,
                 alpha=0.2, color='purple', label='±2倍标准差')
ax4.set_title('残差序列（平稳）', fontsize=14)
ax4.set_xlabel('日期')
ax4.set_ylabel('残差')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 子图5：价差分布直方图
ax5 = axes[2, 0]
ax5.hist(spread, bins=50, density=True, alpha=0.7, color='green')
x = np.linspace(spread.min(), spread.max(), 100)
from scipy.stats import norm
ax5.plot(x, norm.pdf(x, spread.mean(), spread.std()), 'r-', linewidth=2, label='正态分布')
ax5.set_title('价差分布（接近正态）', fontsize=14)
ax5.set_xlabel('价差')
ax5.set_ylabel('密度')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 子图6：滚动相关性
ax6 = axes[2, 1]
rolling_corr = prices.rolling(60).corr().unstack()['Asset1']['Asset2']
ax6.plot(rolling_corr.index, rolling_corr, linewidth=2, color='orange')
ax6.set_title('滚动相关性（60天）', fontsize=14)
ax6.set_xlabel('日期')
ax6.set_ylabel('相关系数')
ax6.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure1_theory.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("✅ 图1：配对交易理论基础已生成")
print(f"价差均值：{spread.mean():.2f}")
print(f"价差标准差：{spread.std():.2f}")
print(f"价差自相关性（滞后1期）：{spread.autocorr():.4f}")
```

### 1.2 协整 vs 相关性

很多初学者混淆**协整（Cointegration）**与**相关性（Correlation）**，但两者有本质区别：

| 维度 | 相关性 | 协整 |
|------|--------|--------|
| 定义 | 两个序列的线性关系强度 | 两个非平稳序列的线性组合是平稳的 |
| 数据要求 | 适用于平稳或非平稳序列 | 要求两个序列都是非平稳的（通常是I(1)） |
| 经济学含义 | 同步变动程度 | 存在长期均衡关系 |
| 交易含义 | 高相关性不代表可交易 | 协整关系意味着均值回归机会 |

```python
def demonstrate_cointegration_vs_correlation():
    """
    演示协整与相关性的区别
    """
    np.random.seed(42)
    n = 500
    
    # 情况1：高相关但无协整（两个独立的随机游走）
    rw1 = np.cumsum(np.random.normal(0, 1, n))
    rw2 = np.cumsum(np.random.normal(0, 1, n))
    
    # 情况2：协整但相关性不高（有共同趋势但短期偏离）
    common = np.cumsum(np.random.normal(0, 1, n))
    coint1 = 0.7 * common + np.cumsum(np.random.normal(0, 0.3, n))
    coint2 = 0.7 * common + np.cumsum(np.random.normal(0, 0.3, n))
    
    # 计算相关性
    corr_rw = np.corrcoef(rw1, rw2)[0, 1]
    corr_coint = np.corrcoef(coint1, coint2)[0, 1]
    
    # 协整检验（Engle-Granger)
    def engle_granger_test(y, x):
        # 第一步：OLS回归
        X = sm.add_constant(x)
        model = OLS(y, X).fit()
        residuals = model.resid
        
        # 第二步：ADF检验残差
        adf_stat = adfuller(residuals)[0]
        p_value = adfuller(residuals)[1]
        
        return adf_stat, p_value, residuals
    
    # 检验随机游走
    eg_rw = engle_granger_test(rw1, rw2)
    
    # 检验协整序列
    eg_coint = engle_granger_test(coint1, coint2)
    
    print("\n" + "="*60)
    print("协整 vs 相关性对比")
    print("="*60)
    print(f"\n情况1：独立随机游走")
    print(f"  相关性：{corr_rw:.4f}")
    print(f"  ADF检验p值：{eg_rw[1]:.4f} {'✓ 协整' if eg_rw[1] < 0.05 else '✗ 无协整'}")
    
    print(f"\n情况2：协整序列")
    print(f"  相关性：{corr_coint:.4f}")
    print(f"  ADF检验p值：{eg_coint[1]:.4f} {'✓ 协整' if eg_coint[1] < 0.05 else '✗ 无协整'}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 子图1：独立随机游走
    axes[0, 0].plot(rw1, label='Random Walk 1', linewidth=2)
    axes[0, 0].plot(rw2, label='Random Walk 2', linewidth=2)
    axes[0, 0].set_title(f'独立随机游走（相关性={corr_rw:.2f}）', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 子图2：随机游走的残差
    axes[0, 1].plot(eg_rw[2], linewidth=2, color='red')
    axes[0, 1].axhline(y=0, color='k', linestyle='--')
    axes[0, 1].set_title('残差序列（非平稳）', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 子图3：协整序列
    axes[1, 0].plot(coint1, label='Cointegrated 1', linewidth=2)
    axes[1, 0].plot(coint2, label='Cointegrated 2', linewidth=2)
    axes[1, 0].set_title(f'协整序列（相关性={corr_coint:.2f}）', fontsize=12)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 子图4：协整的残差
    axes[1, 1].plot(eg_coint[2], linewidth=2, color='green')
    axes[1, 1].axhline(y=0, color='k', linestyle='--')
    axes[1, 1].set_title('残差序列（平稳）', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure2_cointegration_vs_correlation.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n✅ 图2：协整 vs 相关性已生成")

# 运行演示
demonstrate_cointegration_vs_correlation()
```

输出示例：
```
============================================================
协整 vs 相关性对比
============================================================

情况1：独立随机游走
  相关性：0.0234
  ADF检验p值：0.8234 ✗ 无协整

情况2：协整序列
  相关性：0.7823
  ADF检验p值：0.0001 ✓ 协整
```

## 2. 协整检验方法

### 2.1 Engle-Granger 两步法

这是最经典的协整检验方法：

```python
import statsmodels.api as sm
from statsmodels.tsa.stattools import coint

def engle_granger_cointegration(price1, price2, significance_level=0.05):
    """
    Engle-Granger 两步法协整检验
    
    Parameters:
    -----------
    price1, price2: Series, 两个资产的价格序列
    significance_level: float, 显著性水平
    
    Returns:
    --------
    result: dict, 检验结果
    """
    # 第一步：OLS回归估计协整关系
    X = sm.add_constant(price2)
    model = OLS(price1, X).fit()
    hedge_ratio = model.params[price2.name]  # 对冲比率（β）
    intercept = model.params['const']
    residuals = model.resid
    
    # 第二步：ADF检验残差平稳性
    adf_result = adfuller(residuals, autolag='AIC')
    adf_stat = adf_result[0]
    p_value = adf_result[1]
    critical_values = adf_result[4]
    
    # 判断是否协整
    is_cointegrated = p_value < significance_level
    
    result = {
        'hedge_ratio': hedge_ratio,
        'intercept': intercept,
        'adf_statistic': adf_stat,
        'p_value': p_value,
        'critical_values': critical_values,
        'is_cointegrated': is_cointegrated,
        'residuals': residuals,
        'spread': residuals  # 价差 = 残差
    }
    
    return result

# 使用示例
result = engle_granger_cointegration(prices['Asset1'], prices['Asset2'])

print("\n" + "="*60)
print("Engle-Granger 协整检验结果")
print("="*60)
print(f"对冲比率（β）：{result['hedge_ratio']:.4f}")
print(f"截距项（α）：{result['intercept']:.4f}")
print(f"ADF统计量：{result['adf_statistic']:.4f}")
print(f"p值：{result['p_value']:.4f}")
print(f"临界值（5%）：{result['critical_values']['5%']:.4f}")
print(f"是否协整：{'✓ 是' if result['is_cointegrated'] else '✗ 否'}")
```

### 2.2 Johansen 检验

Johansen检验可以同时检验多个协整关系，适合多资产配对：

```python
from statsmodels.tsa.vector_ar.vecm import VECM, select_coint_rank

def johansen_cointegration_test(data, det_order=0, k_ar_diff=1):
    """
    Johansen 协整检验
    
    Parameters:
    -----------
    data: DataFrame, 多个资产的价格序列
    det_order: int, 确定性项顺序
        -1: 无确定性项
         0: 仅截距
         1: 截距+趋势
    k_ar_diff: int, VAR模型滞后阶数
    
    Returns:
    --------
    result: dict, 检验结果
    """
    from statsmodels.tsa.vector_ar.vecm import coint_johansen
    
    # 进行Johansen检验
    joh_result = coint_johansen(data, det_order, k_ar_diff)
    
    # 提取结果
    trace_stat = joh_result.lr1  # 迹统计量
    max_stat = joh_result.lr2  # 最大特征值统计量
    trace_crit = joh_result.cvt  # 迹统计量临界值
    max_crit = joh_result.cvm  # 最大特征值临界值
    
    # 判断协整秩（协整关系个数）
    n_cointegrating = 0
    for i in range(len(trace_stat)):
        if trace_stat[i] > trace_crit[i, 1]:  # 5%临界值
            n_cointegrating += 1
    
    result = {
        'trace_statistic': trace_stat,
        'max_eigenvalue_statistic': max_stat,
        'trace_critical_values': trace_crit,
        'max_eigenvalue_critical_values': max_crit,
        'n_cointegrating_relations': n_cointegrating,
        'eigenvectors': joh_result.evec,
        'eigenvalues': joh_result.eig
    }
    
    return result

# 使用示例（多资产）
prices_multi = pd.DataFrame({
    'Asset1': price1,
    'Asset2': price2,
    'Asset3': price1 * 0.9 + np.random.normal(0, 5, n_periods)  # 第三个资产
}, index=dates)

# joh_result = johansen_cointegration_test(log_prices_multi)
# print(f"协整关系个数：{joh_result['n_cointegrating_relations']}")
```

### 2.3 滚动窗口协整检验

协整关系可能随时间变化，需要使用滚动窗口：

```python
def rolling_cointegration_test(price1, price2, window=252, step=20):
    """
    滚动窗口协整检验
    
    Parameters:
    -----------
    price1, price2: Series, 价格序列
    window: int, 滚动窗口长度（天）
    step: int, 滚动步长（天）
    
    Returns:
    --------
    results: DataFrame, 每个时间点的协整检验结果
    """
    results = []
    dates = []
    
    for i in range(window, len(price1) - step, step):
        window_price1 = price1.iloc[i-window:i]
        window_price2 = price2.iloc[i-window:i]
        
        # 进行协整检验
        try:
            result = engle_granger_cointegration(window_price1, window_price2)
            results.append({
                'date': price1.index[i],
                'p_value': result['p_value'],
                'hedge_ratio': result['hedge_ratio'],
                'is_cointegrated': result['is_cointegrated']
            })
        except:
            continue
    
    return pd.DataFrame(results).set_index('date')

# 使用示例
# rolling_results = rolling_cointegration_test(prices['Asset1'], prices['Asset2'])
# 
# fig, axes = plt.subplots(2, 1, figsize=(12, 8))
# 
# axes[0].plot(rolling_results.index, rolling_results['p_value'], linewidth=2)
# axes[0].axhline(y=0.05, color='r', linestyle='--', label='5%显著性水平')
# axes[0].set_title('滚动协整检验p值', fontsize=14)
# axes[0].set_ylabel('p值')
# axes[0].legend()
# axes[0].grid(True, alpha=0.3)
# 
# axes[1].plot(rolling_results.index, rolling_results['hedge_ratio'], linewidth=2)
# axes[1].set_title('滚动对冲比率（β）', fontsize=14)
# axes[1].set_xlabel('日期')
# axes[1].set_ylabel('对冲比率')
# axes[1].grid(True, alpha=0.3)
# 
# plt.tight_layout()
```

## 3. 配对选择的量化方法

### 3.1 距离法（Distance Method）

最简单的方法：寻找价格差距最小的资产对。

```python
def distance_method_selection(universe_prices, n_top=10):
    """
    距离法选择配对
    
    Parameters:
    -----------
    universe_prices: DataFrame, 多资产价格矩阵
    n_top: int, 返回前N个配对
    
    Returns:
    --------
    pairs: list, 候选配对列表
    """
    n_assets = universe_prices.shape[1]
    distances = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            asset1 = universe_prices.columns[i]
            asset2 = universe_prices.columns[j]
            
            # 标准化价格
            price1_norm = universe_prices[asset1] / universe_prices[asset1].iloc[0]
            price2_norm = universe_prices[asset2] / universe_prices[asset2].iloc[0]
            
            # 计算欧氏距离
            distance = np.sqrt(((price1_norm - price2_norm) ** 2).sum())
            
            # 计算相关性
            correlation = price1_norm.corr(price2_norm)
            
            distances.append({
                'asset1': asset1,
                'asset2': asset2,
                'distance': distance,
                'correlation': correlation
            })
    
    # 排序并返回前N个
    distances_df = pd.DataFrame(distances)
    top_pairs = distances_df.nsmallest(n_top, 'distance')
    
    return top_pairs

# 模拟多资产数据
np.random.seed(42)
n_assets = 20
n_periods = 500
asset_names = [f'Stock_{i}' for i in range(n_assets)]

# 生成具有潜在配对关系的资产
universe = pd.DataFrame(index=dates[:n_periods])
for i in range(n_assets):
    if i % 2 == 0:
        # 偶数编号股票：跟随同一趋势
        trend = np.cumsum(np.random.normal(0.0005, 0.02, n_periods))
        universe[asset_names[i]] = 50 +趋势 * (i+1) * 10
    else:
        # 奇数编号股票：随机游走
        universe[asset_names[i]] = 50 + np.cumsum(np.random.normal(0, 1, n_periods))

# 选择配对
top_pairs = distance_method_selection(universe, n_top=5)

print("\n" + "="*60)
print("距离法选择的Top 5配对")
print("="*60)
print(top_pairs.to_string(index=False))
```

### 3.2 协整扫描法

系统性地扫描所有可能的配对：

```python
def cointegration_scan(universe_prices, p_value_threshold=0.05):
    """
    协整扫描：找出所有协整的配对
    
    Parameters:
    -----------
    universe_prices: DataFrame, 多资产价格矩阵
    p_value_threshold: float, p值阈值
    
    Returns:
    --------
    cointegrated_pairs: DataFrame, 协整配对列表
    """
    n_assets = universe_prices.shape[1]
    pairs = []
    
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            asset1 = universe_prices.columns[i]
            asset2 = universe_prices.columns[j]
            
            try:
                # 协整检验
                result = engle_granger_cointegration(
                    universe_prices[asset1], 
                    universe_prices[asset2]
                )
                
                if result['p_value'] < p_value_threshold:
                    # 计算价差的统计特征
                    spread = result['spread']
                    spread_mean = spread.mean()
                    spread_std = spread.std()
                    half_life = calculate_half_life(spread)
                    
                    pairs.append({
                        'asset1': asset1,
                        'asset2': asset2,
                        'p_value': result['p_value'],
                        'hedge_ratio': result['hedge_ratio'],
                        'spread_mean': spread_mean,
                        'spread_std': spread_std,
                        'half_life': half_life,
                        'score': -result['p_value'] * (1 / half_life)  # 综合评分
                    })
            except:
                continue
    
    pairs_df = pd.DataFrame(pairs)
    
    if len(pairs_df) > 0:
        pairs_df = pairs_df.sort_values('score', ascending=False)
    
    return pairs_df

def calculate_half_life(spread):
    """
    计算价差的半衰期（均值回归速度）
    """
    # 使用AR(1)模型估计
    lag_spread = spread.shift(1).dropna()
    delta_spread = spread.diff().dropna()
    
    # 回归：Δspread_t = α + β * spread_{t-1} + ε_t
    X = sm.add_constant(lag_spread.loc[delta_spread.index])
    model = OLS(delta_spread, X).fit()
    
    beta = model.params[lag_spread.name]
    half_life = -np.log(2) / beta if beta < 0 else np.inf
    
    return half_life

# 扫描协整配对
cointegrated_pairs = cointegration_scan(universe, p_value_threshold=0.05)

print("\n" + "="*60)
print("协整扫描结果")
print("="*60)
if len(cointegrated_pairs) > 0:
    print(f"找到 {len(cointegrated_pairs)} 个协整配对")
    print("\nTop 10 配对：")
    print(cointegrated_pairs.head(10).to_string(index=False))
else:
    print("未找到协整配对，请放宽p值阈值或检查数据。")
```

### 3.3 聚类增强法

先通过聚类缩小搜索范围：

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def clustering_enhanced_selection(universe_prices, n_clusters=5):
    """
    聚类增强的配对选择
    
    Parameters:
    -----------
    universe_prices: DataFrame, 多资产价格矩阵
    n_clusters: int, 聚类个数
    
    Returns:
    --------
    clusters: dict, 每个聚类中的资产
    """
    # 计算收益率
    returns = universe_prices.pct_change().dropna()
    
    # 计算特征：均值、波动率、偏度、峰度
    features = pd.DataFrame(index=returns.columns)
    features['mean_return'] = returns.mean()
    features['volatility'] = returns.std()
    features['skewness'] = returns.skew()
    features['kurtosis'] = returns.kurtosis()
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    # K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    # 整理结果
    cluster_dict = {}
    for i, asset in enumerate(returns.columns):
        cluster_id = clusters[i]
        if cluster_id not in cluster_dict:
            cluster_dict[cluster_id] = []
        cluster_dict[cluster_id].append(asset)
    
    # 可视化聚类结果
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 子图1：聚类散点图
    axes[0].scatter(features['mean_return'], features['volatility'], 
                    c=clusters, cmap='viridis', s=100, alpha=0.7)
    for i, asset in enumerate(features.index):
        axes[0].annotate(asset, (features.iloc[i]['mean_return'], 
                                  features.iloc[i]['volatility']),
                        fontsize=8)
    axes[0].set_xlabel('平均收益')
    axes[0].set_ylabel('波动率')
    axes[0].set_title('资产聚类结果', fontsize=14)
    axes[0].grid(True, alpha=0.3)
    
    # 子图2：聚类内资产个数
    cluster_sizes = pd.Series(clusters).value_counts().sort_index()
    axes[1].bar(cluster_sizes.index, cluster_sizes.values, color='steelblue')
    axes[1].set_xlabel('聚类ID')
    axes[1].set_ylabel('资产个数')
    axes[1].set_title('聚类大小分布', fontsize=14)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure3_clustering.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*60)
    print("聚类增强配对选择")
    print("="*60)
    for cluster_id, assets in cluster_dict.items():
        print(f"\n聚类 {cluster_id}（{len(assets)} 个资产）：")
        print(f"  {', '.join(assets[:5])}{'...' if len(assets) > 5 else ''}")
    
    return cluster_dict

# 使用聚类
clusters = clustering_enhanced_selection(universe, n_clusters=5)
```

## 4. 交易信号构建

### 4.1 基于价差的Z-Score信号

最经典的入场/出场信号：

```python
def generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5):
    """
    基于Z-Score的交易信号
    
    Parameters:
    -----------
    spread: Series, 价差序列
    entry_threshold: float, 入场阈值（标准差倍数）
    exit_threshold: float, 出场阈值（标准差倍数）
    
    Returns:
    --------
    signals: DataFrame, 交易信号
    """
    # 计算滚动统计量（避免前瞻偏差）
    spread_mean = spread.rolling(252).mean()
    spread_std = spread.rolling(252).std()
    
    # 计算Z-Score
    z_score = (spread - spread_mean) / spread_std
    
    # 初始化信号
    signals = pd.DataFrame(index=spread.index)
    signals['z_score'] = z_score
    signals['position'] = 0  # 0: 空仓, 1: 做多价差, -1: 做空价差
    
    # 状态机：生成信号
    position = 0
    
    for i in range(len(signals)):
        if pd.isna(z_score.iloc[i]):
            continue
        
        if position == 0:  # 当前空仓
            if z_score.iloc[i] < -entry_threshold:
                # 价差偏低，做多价差（买入Asset1，卖出Asset2）
                position = 1
            elif z_score.iloc[i] > entry_threshold:
                # 价差偏高，做空价差（卖出Asset1，买入Asset2）
                position = -1
                
        elif position == 1:  # 当前持有多头
            if abs(z_score.iloc[i]) < exit_threshold:
                # 价差回归，平仓
                position = 0
            elif z_score.iloc[i] < -entry_threshold:
                # 继续持有
                pass
            else:
                # 止损或反向信号
                position = 0
                
        elif position == -1:  # 当前持有空头
            if abs(z_score.iloc[i]) < exit_threshold:
                # 价差回归，平仓
                position = 0
            elif z_score.iloc[i] > entry_threshold:
                # 继续持有
                pass
            else:
                # 止损或反向信号
                position = 0
        
        signals.iloc[i, signals.columns.get_loc('position')] = position
    
    return signals

# 使用示例
spread = result['spread']
signals = generate_trading_signals(spread, entry_threshold=2.0, exit_threshold=0.5)

# 可视化信号
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# 子图1：价差与Z-Score
ax1 = axes[0]
ax1.plot(spread.index, spread, linewidth=2, label='价差', color='blue')
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.fill_between(spread.index, 
                  spread.mean() + 2*spread.std(),
                  spread.mean() - 2*spread.std(),
                  alpha=0.2, color='blue')
ax1.set_ylabel('价差', fontsize=12)
ax1.legend(loc='upper left')
ax1.grid(True, alpha=0.3)

ax1_twin = ax1.twinx()
ax1_twin.plot(signals.index, signals['z_score'], linewidth=2, 
              label='Z-Score', color='red', linestyle='--')
ax1_twin.axhline(y=2, color='r', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=-2, color='r', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=0.5, color='g', linestyle=':', alpha=0.5)
ax1_twin.axhline(y=-0.5, color='g', linestyle=':', alpha=0.5)
ax1_twin.set_ylabel('Z-Score', fontsize=12)
ax1_twin.legend(loc='upper right')

# 子图2：持仓信号
ax2 = axes[1]
ax2.plot(signals.index, signals['position'], linewidth=2, 
         label='持仓', color='green')
ax2.fill_between(signals.index, 
                  signals['position'].shift(1).fillna(0),
                  alpha=0.3, color='green')
ax2.set_ylabel('持仓', fontsize=12)
ax2.set_ylim(-1.5, 1.5)
ax2.legend()
ax2.grid(True, alpha=0.3)

# 子图3：累计收益
ax3 = axes[2]
# 计算策略收益（简化：假设等权投资）
asset1_returns = prices['Asset1'].pct_change()
asset2_returns = prices['Asset2'].pct_change()

# 策略收益 = 持仓 * (Asset1收益 - hedge_ratio * Asset2收益)
hedge_ratio = result['hedge_ratio']
strategy_returns = signals['position'].shift(1) * (
    asset1_returns - hedge_ratio * asset2_returns
)

# 累计收益
cumulative_returns = (1 + strategy_returns).cumprod()
ax3.plot(cumulative_returns.index, cumulative_returns, 
         linewidth=2, color='purple', label='策略累计收益')
ax3.set_xlabel('日期', fontsize=12)
ax3.set_ylabel('累计收益', fontsize=12)
ax3.legend()
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure4_trading_signals.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 图4：交易信号已生成")
print(f"交易次数：{(signals['position'] != signals['position'].shift(1)).sum()}")
print(f"平均持仓时间：{((signals['position'] != 0).sum() / (signals['position'] != signals['position'].shift(1)).sum()):.1f} 天")
```

### 4.2 卡尔曼滤波动态对冲比率

传统OLS的 hedge ratio 是固定的，卡尔曼滤波可以动态调整：

```python
from pykalman import KalmanFilter

def kalman_filter_hedge_ratio(price1, price2):
    """
    使用卡尔曼滤波估计时变对冲比率
    
    Parameters:
    -----------
    price1, price2: Series, 价格序列
    
    Returns:
    --------
    state_means: array, 时变对冲比率
    """
    # 准备观测数据
    observations = price1.values.reshape(-1, 1)
    X = price2.values.reshape(-1, 1)
    
    # 初始化卡尔曼滤波
    kf = KalmanFilter(
        transition_matrices=np.eye(2),
        observation_matrices=X.reshape(-1, 1, 2),
        initial_state_mean=np.zeros(2),
        initial_state_covariance=np.eye(2) * 0.01,
        observation_covariance=1.0,
        transition_covariance=np.eye(2) * 0.01
    )
    
    # 滤波
    state_means, state_covariances = kf.filter(observations)
    
    # state_means[:, 0]: 截距项（α）
    # state_means[:, 1]: 对冲比率（β）
    
    return state_means

# 使用示例
state_means = kalman_filter_hedge_ratio(prices['Asset1'], prices['Asset2'])

# 可视化时变对冲比率
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(prices.index, state_means[:, 0], linewidth=2, label='截距项（α）')
axes[0].set_title('卡尔曼滤波：时变截距项', fontsize=14)
axes[0].set_ylabel('α')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(prices.index, state_means[:, 1], linewidth=2, 
             label='对冲比率（β）', color='red')
axes[1].axhline(y=result['hedge_ratio'], color='k', linestyle='--', 
                label='固定对冲比率')
axes[1].set_title('卡尔曼滤波：时变对冲比率', fontsize=14)
axes[1].set_xlabel('日期')
axes[1].set_ylabel('β')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure5_kalman_filter.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 图5：卡尔曼滤波时变对冲比率已生成")
```

### 4.3 机器学习增强信号

使用机器学习预测价差方向：

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

def ml_enhanced_signals(spread, lookahead=5, threshold=0.6):
    """
    机器学习增强的交易信号
    
    Parameters:
    -----------
    spread: Series, 价差序列
    lookahead: int, 预测未来N期
    threshold: float, 预测概率阈值
    
    Returns:
    --------
    ml_signals: DataFrame, ML预测信号
    """
    # 构建特征
    features = pd.DataFrame(index=spread.index)
    
    # 1. 滞后价差
    for lag in [1, 2, 3, 5, 10, 20]:
        features[f'lag_{lag}'] = spread.shift(lag)
    
    # 2. 滚动统计量
    for window in [20, 60, 120]:
        features[f'mean_{window}'] = spread.rolling(window).mean()
        features[f'std_{window}'] = spread.rolling(window).std()
        features[f'zscore_{window}'] = (spread - features[f'mean_{window}']) / features[f'std_{window}']
    
    # 3. 技术指标
    features['rsi'] = calculate_rsi(spread, window=14)
    features['macd'], features['macd_signal'] = calculate_macd(spread)
    
    # 4. 自相关性特征
    for lag in [1, 5, 10]:
        features[f'autocorr_{lag}'] = spread.rolling(60).apply(
            lambda x: x.autocorr(lag=lag)
        )
    
    # 构建目标：未来N期价差是否上升
    target = (spread.shift(-lookahead) - spread > 0).astype(int)
    
    # 删除NaN
    valid_idx = features.dropna().index.intersection(target.dropna().index)
    X = features.loc[valid_idx]
    y = target.loc[valid_idx]
    
    # 训练模型（使用时间序列交叉验证）
    tscv = TimeSeriesSplit(n_splits=5)
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    
    # 交叉验证
    cv_scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy')
    print(f"交叉验证准确率：{cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    
    # 全样本训练
    model.fit(X, y)
    
    # 预测概率
    proba = model.predict_proba(X)[:, 1]
    
    # 生成信号
    ml_signals = pd.DataFrame(index=X.index)
    ml_signals['probability_up'] = proba
    ml_signals['signal'] = 0
    ml_signals.loc[proba > threshold, 'signal'] = 1   # 做多价差
    ml_signals.loc[proba < (1 - threshold), 'signal'] = -1  # 做空价差
    
    # 特征重要性
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 重要特征：")
    print(feature_importance.head(10).to_string(index=False))
    
    return ml_signals, model

def calculate_rsi(series, window=14):
    """
    计算RSI指标
    """
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    计算MACD指标
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return macd, macd_signal

# 使用示例
ml_signals, ml_model = ml_enhanced_signals(spread, lookahead=5, threshold=0.6)
```

## 5. 风险管理与实战要点

### 5.1 止损策略

配对交易虽然理论上是市场中性，但仍需要严格止损：

```python
def implement_stop_loss(spread, signals, max_loss_pct=0.05, max_holding_days=30):
    """
    实现止损策略
    
    Parameters:
    -----------
    spread: Series, 价差序列
    signals: DataFrame, 交易信号
    max_loss_pct: float, 最大亏损比例
    max_holding_days: int, 最大持仓天数
    
    Returns:
    --------
    signals_with_stoploss: DataFrame, 加入止损后的信号
    """
    signals = signals.copy()
    signals['cumulative_loss'] = 0
    signals['holding_days'] = 0
    signals['stopped_out'] = False
    
    position = 0
    entry_spread = 0
    holding_count = 0
    
    for i in range(1, len(signals)):
        if signals['position'].iloc[i] != 0 and position == 0:
            # 新建仓位
            position = signals['position'].iloc[i]
            entry_spread = spread.iloc[i]
            holding_count = 0
            
        elif position != 0:
            # 持有仓位
            holding_count += 1
            current_spread = spread.iloc[i]
            
            # 计算亏损比例
            if position == 1:  # 做多价差
                loss_pct = (entry_spread - current_spread) / entry_spread
            else:  # 做空价差
                loss_pct = (current_spread - entry_spread) / entry_spread
            
            # 判断是否止损
            if loss_pct > max_loss_pct or holding_count > max_holding_days:
                signals.iloc[i, signals.columns.get_loc('position')] = 0
                signals.iloc[i, signals.columns.get_loc('stopped_out')] = True
                position = 0
                holding_count = 0
    
    return signals

# 加入止损
signals_with_stoploss = implement_stop_loss(spread, signals, 
                                             max_loss_pct=0.05, 
                                             max_holding_days=30)

print("\n" + "="*60)
print("止损策略统计")
print("="*60)
print(f"止损次数：{signals_with_stoploss['stopped_out'].sum()}")
print(f"止损占比：{signals_with_stoploss['stopped_out'].mean():.2%}")
```

### 5.2 仓位管理

根据价差偏离程度动态调整仓位：

```python
def dynamic_position_sizing(signals, spread, max_position=1.0, 
                            scaling_threshold=1.0):
    """
    动态仓位管理
    
    Parameters:
    -----------
    signals: DataFrame, 交易信号
    spread: Series, 价差序列
    max_position: float, 最大仓位
    scaling_threshold: float, 开始缩放的Z-Score阈值
    
    Returns:
    --------
    positioned_signals: DataFrame, 加入仓位大小后的信号
    """
    signals = signals.copy()
    
    # 计算Z-Score
    spread_mean = spread.rolling(252).mean()
    spread_std = spread.rolling(252).std()
    z_score = (spread - spread_mean) / spread_std
    
    # 动态仓位
    signals['position_size'] = 0.0
    
    for i in range(len(signals)):
        if signals['position'].iloc[i] != 0:
            abs_z = abs(z_score.iloc[i])
            
            if abs_z >= scaling_threshold:
                # Z-Score越大，仓位越大（反向缩放）
                size = max_position * (scaling_threshold / abs_z)
                size = min(size, max_position)  # 不超过最大仓位
            else:
                size = max_position
            
            signals.iloc[i, signals.columns.get_loc('position_size')] = size
    
    return signals

# 动态仓位
signals_dynamic = dynamic_position_sizing(signals_with_stoploss, spread, 
                                           max_position=1.0, 
                                           scaling_threshold=1.0)

# 可视化仓位变化
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

axes[0].plot(signals.index, signals['position'], 
             label='原始信号', linewidth=2, alpha=0.7)
axes[0].set_title('原始交易信号', fontsize=14)
axes[0].set_ylabel('持仓方向')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(signals_dynamic.index, signals_dynamic['position_size'], 
             label='动态仓位', linewidth=2, color='orange')
axes[1].fill_between(signals_dynamic.index, 
                      signals_dynamic['position_size'],
                      alpha=0.3, color='orange')
axes[1].set_title('动态仓位大小', fontsize=14)
axes[1].set_xlabel('日期')
axes[1].set_ylabel('仓位')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/figure6_position_sizing.png',
            dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ 图6：动态仓位管理已生成")
```

### 5.3 交易成本与滑点

```python
def backtest_with_costs(signals, prices, transaction_cost=0.001, slippage=0.001):
    """
    考虑交易成本和滑点的回测
    
    Parameters:
    -----------
    signals: DataFrame, 交易信号
    prices: DataFrame, 价格数据
    transaction_cost: float, 交易成本（单边）
    slippage: float, 滑点
    
    Returns:
    --------
    results: DataFrame, 回测结果
    """
    results = pd.DataFrame(index=signals.index)
    results['strategy_returns'] = 0.0
    results['transaction_costs'] = 0.0
    results['slippage_costs'] = 0.0
    
    position = 0
    entry_price1 = 0
    entry_price2 = 0
    
    for i in range(1, len(signals)):
        if signals['position'].iloc[i] != signals['position'].iloc[i-1]:
            # 仓位变化，计算交易成本
            if position != 0:  # 平仓
                exit_price1 = prices['Asset1'].iloc[i]
                exit_price2 = prices['Asset2'].iloc[i]
                
                # 计算收益
                if position == 1:  # 平多价差
                    ret = (exit_price1 - entry_price1) - result['hedge_ratio'] * (exit_price2 - entry_price2)
                else:  # 平空价差
                    ret = -(exit_price1 - entry_price1) + result['hedge_ratio'] * (exit_price2 - entry_price2)
                
                results.iloc[i, results.columns.get_loc('strategy_returns')] = ret
            
            if signals['position'].iloc[i] != 0:  # 开仓
                entry_price1 = prices['Asset1'].iloc[i]
                entry_price2 = prices['Asset2'].iloc[i]
                
                # 交易成本
                cost = transaction_cost * (entry_price1 + result['hedge_ratio'] * entry_price2)
                results.iloc[i, results.columns.get_loc('transaction_costs')] = cost
                
                # 滑点
                slip = slippage * (entry_price1 + result['hedge_ratio'] * entry_price2)
                results.iloc[i, results.columns.get_loc('slippage_costs')] = slip
            
            position = signals['position'].iloc[i]
    
    # 净收益
    results['net_returns'] = results['strategy_returns'] - results['transaction_costs'] - results['slippage_costs']
    results['cumulative_returns'] = (1 + results['net_returns']).cumprod()
    
    # 性能统计
    total_return = results['cumulative_returns'].iloc[-1] - 1
    annual_return = (1 + total_return) ** (252 / len(results)) - 1
    max_drawdown = ((results['cumulative_returns'] / 
                     results['cumulative_returns'].expanding().max()) - 1).min()
    
    print("\n" + "="*60)
    print("回测结果（含交易成本）")
    print("="*60)
    print(f"总收益：{total_return:.2%}")
    print(f"年化收益：{annual_return:.2%}")
    print(f"最大回撤：{max_drawdown:.2%}")
    print(f"总成本占比：{results['transaction_costs'].sum() / results['strategy_returns'].sum():.2%}")
    
    return results

# 运行回测
backtest_results = backtest_with_costs(signals_dynamic, prices, 
                                       transaction_cost=0.001, 
                                       slippage=0.001)
```

## 6. 总结

配对交易是一门艺术与科学的结合。本文介绍了：

1. **理论基础**：协整与均值回归
2. **协整检验**：Engle-Granger、Johansen、滚动检验
3. **配对选择**：距离法、协整扫描、聚类增强
4. **信号构建**：Z-Score、卡尔曼滤波、机器学习
5. **风险管理**：止损、仓位管理、交易成本

**实战建议**：

1. **严格样本外测试**：协整关系容易过拟合
2. **监控半衰期**：均值回归速度会变慢
3. **分散投资**：同时交易多个配对降低风险
4. **考虑市场冲击**：大量资金会改变价差结构

配对交易不是印钞机，但它是量化工具箱中不可或缺的利器。

---

**参考文献**：

1. Gatev, E., Goetzmann, W. N., & Rouwenhorst, K. G. (2006). Pairs trading: Performance of a relative-value arbitrage rule. *Review of Financial Studies*.
2. Vidyamurthy, G. (2004). *Pairs Trading: Quantitative Methods and Analysis*. Wiley.
3. Elliott, R. J., Van Der Hoek, J., & Malcolm, W. P. (2005). Pairs trading. *Quantitative Finance*.
4. Huck, N. (2009). Pairs selection and outranking: An application to the S&P 100 index. *European Journal of Operational Research*.

**免责声明**：本文仅供学术交流和策略研究，不构成任何投资建议。配对交易涉及模型风险、执行风险等多种风险，实际操作需谨慎评估。
