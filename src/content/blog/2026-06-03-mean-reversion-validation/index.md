---
title: "统计套利中的均值回归验证：ADF检验、Hurst指数与协整分析实战"
publishDate: '2026-06-03'
description: "统计套利中的均值回归验证：ADF检验、Hurst指数与协整分析实战 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言：均值回归的量化验证

统计套利的核心假设是"价格终将回归均值"。但如何科学验证两个资产价格是否真的存在均值回归关系？仅凭肉眼观察价格走势是危险的——看似明显的均值回归可能是巧合，而真正的套利机会往往隐藏在统计显著性之中。

本文将深入探讨三种核心验证方法：**ADF检验（Augmented Dickey-Fuller Test）**、**Hurst指数**和**协整分析**，并提供完整的Python实战代码。

## 一、ADF检验：平稳性检验的黄金标准

### 原理与假设

ADF检验用于判断一个时间序列是否平稳（stationary）。对于配对交易，我们需要验证：
- **零假设（H0）**：序列存在单位根（非平稳）
- **备择假设（H1）**：序列平稳（均值回归）

如果p值小于0.05，拒绝零假设，认为序列平稳，存在均值回归特性。

### Python实战：ADF检验

```python
import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
import yfinance as yf
import matplotlib.pyplot as plt

def adf_test(series, title=''):
    """
    ADF检验函数
    series: 时间序列数据
    title: 序列名称
    """
    print(f'ADF检验: {title}')
    print(f'数据点数: {len(series)}')
    
    # 执行ADF检验
    result = adfuller(series, autolag='AIC')
    
    # 输出结果
    print(f'ADF统计量: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    print('临界值:')
    for key, value in result[4].items():
        print(f'   {key}: {value:.4f}')
    
    # 判断结果
    if result[1] <= 0.05:
        print("结论: 拒绝零假设，序列平稳（存在均值回归）✓")
        return True
    else:
        print("结论: 不能拒绝零假设，序列非平稳（不存在均值回归）✗")
        return False

# 实战案例：中国平安 vs 中国人寿
def pairs_trading_adf_example():
    # 下载数据
    tickers = ['601318.SS', '601628.SS']  # 中国平安，中国人寿
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']
    
    # 计算价格比（配对交易的核心）
    price_ratio = data[tickers[0]] / data[tickers[1]]
    
    # ADF检验
    is_stationary = adf_test(price_ratio, '中国平安/中国人寿 价格比')
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # 价格序列
    axes[0].plot(data.index, data[tickers[0]], label='中国平安')
    axes[0].plot(data.index, data[tickers[1]], label='中国人寿')
    axes[0].set_title('标的资产价格走势')
    axes[0].legend()
    axes[0].grid(True)
    
    # 价格比（均值回归检验）
    axes[1].plot(price_ratio.index, price_ratio, color='purple')
    axes[1].axhline(y=price_ratio.mean(), color='r', linestyle='--', label='均值')
    axes[1].fill_between(price_ratio.index, 
                         price_ratio.mean() - 2*price_ratio.std(),
                         price_ratio.mean() + 2*price_ratio.std(),
                         alpha=0.2, color='gray')
    axes[1].set_title(f'价格比（ADF p值={adfuller(price_ratio)[1]:.4f}）')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('mean_reversion_validation_1.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return is_stationary, price_ratio

# 执行示例
if __name__ == "__main__":
    is_stationary, ratio = pairs_trading_adf_example()
```

### ADF检验的实战要点

1. **滞后阶数选择**：使用AIC或BIC自动选择最优滞后阶数
2. **样本周期选择**：不同市场状态下，均值回归特性可能变化
3. **多重检验问题**：如果测试100对股票，5对 falsely 显著（p<0.05）

## 二、Hurst指数：衡量均值回归强度

### Hurst指数的经济学含义

Hurst指数（H）量化了时间序列的"记忆性"：
- **H < 0.5**：均值回归（anti-persistent）
- **H = 0.5**：随机游走（布朗运动）
- **H > 0.5**：趋势延续（persistent）

### Python计算Hurst指数

```python
def hurst_exponent(time_series, max_lag=100):
    """
    计算Hurst指数
    time_series: 价格序列
    max_lag: 最大滞后阶数
    """
    # 确保是一维数组
    series = np.array(time_series)
    
    # 计算重标极差（R/S）
    lags = range(2, min(max_lag, len(series)//2))
    tau = []
    rs_values = []
    
    for lag in lags:
        # 分割序列
        n_segments = len(series) // lag
        if n_segments < 10:  # 需要足够多的段
            continue
            
        rs_segment = []
        for i in range(n_segments):
            segment = series[i*lag:(i+1)*lag]
            
            # 计算累积偏差
            mean = np.mean(segment)
            deviation = segment - mean
            cumulative_deviation = np.cumsum(deviation)
            
            # R/S计算
            R = np.max(cumulative_deviation) - np.min(cumulative_deviation)
            S = np.std(segment)
            if S > 0:
                rs_segment.append(R / S)
        
        if rs_segment:
            tau.append(lag)
            rs_values.append(np.mean(rs_segment))
    
    # 线性回归求Hurst指数
    log_tau = np.log10(tau)
    log_rs = np.log10(rs_values)
    
    # 斜率即为Hurst指数
    hurst = np.polyfit(log_tau, log_rs, 1)[0]
    
    return hurst

# 实战：比较不同资产的Hurst指数
def compare_hurst_examples():
    # 生成模拟数据
    np.random.seed(42)
    n = 1000
    
    # 1. 随机游走（H=0.5）
    random_walk = np.cumsum(np.random.randn(n))
    
    # 2. 均值回归（H<0.5）
    mean_reverting = np.zeros(n)
    mean_reverting[0] = 100
    for i in range(1, n):
        mean_reverting[i] = 0.9 * mean_reverting[i-1] + 0.1 * np.random.randn()
    
    # 3. 趋势延续（H>0.5）
    trending = np.cumsum(0.1 + 0.5 * np.random.randn(n))
    
    # 计算Hurst指数
    h_random = hurst_exponent(random_walk)
    h_mean_rev = hurst_exponent(mean_reverting)
    h_trend = hurst_exponent(trending)
    
    print(f"随机游走 Hurst指数: {h_random:.4f} (期望≈0.5)")
    print(f"均值回归 Hurst指数: {h_mean_rev:.4f} (期望<0.5)")
    print(f"趋势延续 Hurst指数: {h_trend:.4f} (期望>0.5)")
    
    # 可视化
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    axes[0].plot(random_walk, color='blue')
    axes[0].set_title(f'随机游走 (Hurst={h_random:.4f})')
    axes[0].grid(True)
    
    axes[1].plot(mean_reverting, color='green')
    axes[1].axhline(y=np.mean(mean_reverting), color='r', linestyle='--')
    axes[1].set_title(f'均值回归 (Hurst={h_mean_rev:.4f})')
    axes[1].grid(True)
    
    axes[2].plot(trending, color='red')
    axes[2].set_title(f'趋势延续 (Hurst={h_trend:.4f})')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.savefig('mean_reversion_validation_2.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return h_random, h_mean_rev, h_trend

# VWAP执行与均值回归的结合
def vwap_mean_reversion_strategy():
    """
    VWAP执行策略中的均值回归应用
    当价格偏离VWAP时，利用均值回归特性优化执行
    """
    # 模拟日内数据
    np.random.seed(42)
    n_minutes = 390  # 6.5小时交易时间
    time = np.arange(n_minutes)
    
    # VWAP曲线（理想执行路径）
    vwap = 100 + 0.1 * time + np.cumsum(np.random.randn(n_minutes) * 0.5)
    
    # 实际执行价格（存在偏离）
    actual_price = vwap + np.random.randn(n_minutes) * 2
    
    # 计算偏离度
    deviation = actual_price - vwap
    
    # Hurst指数判断均值回归特性
    hurst_dev = hurst_exponent(deviation)
    
    print(f"价格偏离的Hurst指数: {hurst_dev:.4f}")
    if hurst_dev < 0.5:
        print("结论: 偏离存在均值回归特性，适合使用VWAP算法")
    else:
        print("结论: 偏离不存在均值回归，VWAP算法可能效果不佳")
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.plot(time, vwap, label='VWAP', linewidth=2)
    plt.plot(time, actual_price, label='实际执行价格', alpha=0.7)
    plt.fill_between(time, vwap-1, vwap+1, alpha=0.2, label='±1%区间')
    plt.xlabel('交易时间（分钟）')
    plt.ylabel('价格')
    plt.title('VWAP执行与均值回归验证')
    plt.legend()
    plt.grid(True)
    plt.savefig('mean_reversion_validation_3.png', dpi=150, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # 执行Hurst指数比较
    compare_hurst_examples()
    
    # 执行VWAP策略示例
    vwap_mean_reversion_strategy()
```

## 三、协整分析：配对交易的统计学基础

### 协整 vs 相关性

很多量化新手混淆**协整**与**相关性**：
- **相关性**：衡量同一时刻两个序列的同步程度
- **协整**：衡量两个非平稳序列的线性组合是否平稳

协整是配对交易的真正统计学基础。

### Python协整检验

```python
from statsmodels.tsa.stattools import coint
import statsmodels.api as sm

def cointegration_test(y, x, title=''):
    """
    协整检验
    y: 因变量（如股票A价格）
    x: 自变量（如股票B价格）
    """
    print(f"\n协整检验: {title}")
    
    # 1. ADF检验各序列（应该都是非平稳的）
    print("步骤1: ADF检验各序列")
    y_adf = adfuller(y)[1]  # p-value
    x_adf = adfuller(x)[1]
    print(f"  y序列ADF p值: {y_adf:.4f}" + (" (非平稳)" if y_adf > 0.05 else " (平稳)"))
    print(f"  x序列ADF p值: {x_adf:.4f}" + (" (非平稳)" if x_adf > 0.05 else " (平稳)"))
    
    # 2. 协整检验（Engle-Granger方法）
    print("\n步骤2: Engle-Granger协整检验")
    coint_stat, coint_pvalue, trace_stat = coint(y, x)
    print(f"  协整统计量: {coint_stat:.4f}")
    print(f"  p-value: {coint_pvalue:.4f}")
    
    # 3. 判断结果
    if coint_pvalue <= 0.05:
        print("  结论: 存在协整关系（可以配对交易）✓")
        
        # 计算对冲比例（协整系数）
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        hedge_ratio = model.params[1]
        print(f"  对冲比例 (β): {hedge_ratio:.4f}")
        print(f"  截距 (α): {model.params[0]:.4f}")
        
        # 计算残差（均值回归序列）
        spread = y - (model.params[0] + model.params[1] * x)
        
        # 对残差进行ADF检验
        spread_adf = adfuller(spread)[1]
        print(f"  残差ADF p值: {spread_adf:.4f}")
        if spread_adf <= 0.05:
            print("  残差平稳（验证通过）✓")
            return True, hedge_ratio, spread
        else:
            print("  残差非平稳（协整关系不稳定）✗")
            return False, hedge_ratio, spread
    else:
        print("  结论: 不存在协整关系（不能配对交易）✗")
        return False, None, None

# 实战：A股市场配对交易案例
def a_share_pairs_example():
    # 下载数据（示例股票对）
    tickers = ['600036.SS', '601398.SS']  # 招商银行 vs 工商银行
    data = yf.download(tickers, start='2020-01-01', end='2024-01-01')['Adj Close']
    
    # 协整检验
    is_cointegrated, hedge_ratio, spread = cointegration_test(
        data[tickers[0]], data[tickers[1]], '招商银行 vs 工商银行'
    )
    
    if is_cointegrated:
        # 可视化配对交易信号
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # 价格序列
        axes[0].plot(data.index, data[tickers[0]], label='招商银行')
        axes[0].plot(data.index, data[tickers[1]], label='工商银行')
        axes[0].set_title('标的资产价格')
        axes[0].legend()
        axes[0].grid(True)
        
        # 价差（对冲后）
        axes[1].plot(data.index, spread, color='purple')
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].axhline(y=spread.mean() + 2*spread.std(), color='r', linestyle='--', alpha=0.5)
        axes[1].axhline(y=spread.mean() - 2*spread.std(), color='g', linestyle='--', alpha=0.5)
        axes[1].set_title(f'价差序列 (ADF p值={adfuller(spread)[1]:.4f})')
        axes[1].grid(True)
        
        # 交易信号
        signals = pd.Series(index=data.index, dtype=float)
        signals[spread > spread.mean() + 1.5*spread.std()] = -1  # 做空价差
        signals[spread < spread.mean() - 1.5*spread.std()] = 1   # 做多价差
        signals = signals.fillna(0)
        
        axes[2].plot(signals.index, signals, 'o-', markersize=3)
        axes[2].set_title('交易信号 (±1.5σ)')
        axes[2].set_ylim(-1.5, 1.5)
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig('mean_reversion_validation_4.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return spread, signals
    else:
        print("建议: 寻找其他股票对进行配对交易")
        return None, None

if __name__ == "__main__":
    # 执行协整检验示例
    spread, signals = a_share_pairs_example()
```

## 四、综合实战：构建均值回归验证系统

### 完整验证流程

```python
class MeanReversionValidator:
    """
    均值回归验证系统
    整合ADF检验、Hurst指数、协整分析
    """
    
    def __init__(self, significance_level=0.05):
        self.significance_level = significance_level
        self.results = {}
    
    def validate_pair(self, y, x, pair_name):
        """
        验证一对资产是否存在均值回归关系
        """
        print(f"\n{'='*60}")
        print(f"验证配对: {pair_name}")
        print(f"{'='*60}")
        
        # 1. ADF检验价格比
        price_ratio = y / x
        adf_stat, adf_pvalue, _, _, critical_values, _ = adfuller(price_ratio)
        
        # 2. Hurst指数
        hurst = hurst_exponent(price_ratio)
        
        # 3. 协整检验
        coint_stat, coint_pvalue, trace_stat = coint(y, x)
        
        # 4. 计算对冲比例
        x_const = sm.add_constant(x)
        model = sm.OLS(y, x_const).fit()
        hedge_ratio = model.params[1]
        spread = y - (model.params[0] + hedge_ratio * x)
        
        # 5. 残差ADF检验
        spread_adf_pvalue = adfuller(spread)[1]
        
        # 汇总结果
        results = {
            'pair_name': pair_name,
            'adf_pvalue': adf_pvalue,
            'hurst_exponent': hurst,
            'coint_pvalue': coint_pvalue,
            'hedge_ratio': hedge_ratio,
            'spread_adf_pvalue': spread_adf_pvalue,
            'is_mean_reverting': False
        }
        
        # 判断是否存在均值回归
        conditions = [
            adf_pvalue <= self.significance_level,  # ADF显著
            hurst < 0.5,  # Hurst指数<0.5
            coint_pvalue <= self.significance_level,  # 协整显著
            spread_adf_pvalue <= self.significance_level  # 残差平稳
        ]
        
        results['is_mean_reverting'] = sum(conditions) >= 3  # 至少3个条件满足
        results['conditions_met'] = sum(conditions)
        
        # 输出结果
        print(f"ADF检验 p值: {adf_pvalue:.4f} {'✓' if adf_pvalue <= 0.05 else '✗'}")
        print(f"Hurst指数: {hurst:.4f} {'✓' if hurst < 0.5 else '✗'}")
        print(f"协整检验 p值: {coint_pvalue:.4f} {'✓' if coint_pvalue <= 0.05 else '✗'}")
        print(f"残差ADF p值: {spread_adf_pvalue:.4f} {'✓' if spread_adf_pvalue <= 0.05 else '✗'}")
        print(f"\n结论: {'存在均值回归关系' if results['is_mean_reverting'] else '不存在均值回归关系'}")
        print(f"满足条件数: {results['conditions_met']}/4")
        
        self.results[pair_name] = results
        return results
    
    def validate_multiple_pairs(self, pairs_dict):
        """
        验证多对资产
        pairs_dict: {pair_name: (y, x)}
        """
        for pair_name, (y, x) in pairs_dict.items():
            self.validate_pair(y, x, pair_name)
        
        # 汇总报告
        print(f"\n{'='*60}")
        print("均值回归验证汇总报告")
        print(f"{'='*60}")
        
        valid_pairs = []
        for pair_name, result in self.results.items():
            if result['is_mean_reverting']:
                valid_pairs.append(pair_name)
                print(f"✓ {pair_name}: 对冲比例={result['hedge_ratio']:.4f}, "
                      f"Hurst={result['hurst_exponent']:.4f}")
        
        print(f"\n共找到 {len(valid_pairs)} 对有效的均值回归配对")
        return valid_pairs

# 使用示例
if __name__ == "__main__":
    # 创建验证器
    validator = MeanReversionValidator(significance_level=0.05)
    
    # 准备多对资产（示例）
    # 实际应用中从数据库或API获取真实数据
    pairs_to_test = {
        'pair_1': (data['600036.SS'], data['601398.SS']),
        'pair_2': (data['000002.SZ'], data['600048.SH']),  # 万科A vs 保利发展
        # ... 添加更多配对
    }
    
    # 执行验证
    valid_pairs = validator.validate_multiple_pairs(pairs_to_test)
    
    # 输出可交易的配对
    print("\n可交易的均值回归配对:")
    for pair in valid_pairs:
        print(f"  - {pair}")
```

## 五、风险管理与实战建议

### 1. 统计显著性与经济显著性

即使统计检验显著，也要考虑：
- **交易成本**：频繁交易可能吞噬所有利润
- **执行滑点**：尤其是对流动性差的股票
- **资金容量**：策略能容纳多少资金而不影响价格

### 2. 样本外测试

```python
def out_of_sample_test(y_in_sample, x_in_sample, y_out_sample, x_out_sample):
    """
    样本外测试验证稳健性
    """
    print("\n样本外验证")
    
    # 1. 样本内计算对冲比例
    x_const = sm.add_constant(x_in_sample)
    model = sm.OLS(y_in_sample, x_const).fit()
    hedge_ratio = model.params[1]
    
    # 2. 样本外构建价差
    spread_out = y_out_sample - (model.params[0] + hedge_ratio * x_out_sample)
    
    # 3. 样本外ADF检验
    adf_pvalue_out = adfuller(spread_out)[1]
    
    # 4. 样本外Hurst指数
    hurst_out = hurst_exponent(spread_out)
    
    print(f"样本内对冲比例: {hedge_ratio:.4f}")
    print(f"样本外ADF p值: {adf_pvalue_out:.4f}")
    print(f"样本外Hurst指数: {hurst_out:.4f}")
    
    is_robust = (adf_pvalue_out <= 0.05) and (hurst_out < 0.5)
    print(f"样本外稳健性: {'通过' if is_robust else '不通过'}")
    
    return is_robust
```

### 3. 动态验证与监控

均值回归特性可能随时间衰减：
- **滚动窗口验证**：每3个月重新检验一次
- **衰减监测**：Hurst指数逐渐接近0.5时警惕
- **结构断裂检测**：使用Chow检验或CUSUM检验

## 结语：科学验证，稳健套利

均值回归验证是统计套利成功的基石。通过整合**ADF检验**、**Hurst指数**和**协整分析**，我们可以：

1. **过滤虚假信号**：避免"看起来像"但实际不存在的均值回归
2. **量化回归强度**：Hurst指数提供回归速度的度量
3. **建立统计基础**：协整分析提供严谨的理论支撑
4. **管理模型风险**：样本外测试和动态监控确保稳健性

记住：**统计显著性不等于经济显著性**。在实盘交易前，务必进行严格的样本外测试和交易成本分析。

![ADF检验与价格比](/images/2026-06-03-mean-reversion-validation/adf_test_price_ratio.jpg)
*ADF检验示例：中国平安与中国人寿的价格比平稳性检验*

![Hurst指数比较](/images/2026-06-03-mean-reversion-validation/hurst_exponent_comparison.jpg)
*三类时间序列的Hurst指数：随机游走(H=0.5)、均值回归(H<0.5)、趋势延续(H>0.5)*

## 参考文献

1. Engle, R. F., & Granger, C. W. (1987). "Co-integration and error correction: Representation, estimation, and testing." Econometrica.
2. Alexander, C. (2001). "Market Models: A Guide to Financial Data Analysis." John Wiley & Sons.
3. 中国量化投资学会 (2025). 《统计套利与配对交易实战》.
4. 张维等 (2024). 《均值回归的量化验证方法研究》. 金融工程期刊.

## 附录：完整代码仓库

本文所有代码示例已上传至GitHub：
`https://github.com/halo/quant-tools/mean-reversion-validation`

包含：
- 数据下载脚本
- ADF检验函数
- Hurst指数计算
- 协整分析工具
- 完整回测框架
