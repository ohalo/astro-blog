#!/usr/bin/env python3
"""
为第二篇文章（配对交易与协整分析）生成配图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from scipy.stats import norm

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration', exist_ok=True)

# 生成模拟数据
np.random.seed(42)
n = 500

# 生成协整序列
beta_true = 1.5
alpha_true = 2.0

random_walk = np.cumsum(np.random.normal(0, 1, n))
cointegrated_spread = alpha_true + beta_true * random_walk + np.random.normal(0, 0.5, n)

price2 = random_walk + 100
price1 = cointegrated_spread

# 图1: 封面图 - 配对交易示意图
def generate_cover_image():
    """生成封面配图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建示意图：展示配对交易的概念
    dates = np.arange(n)
    
    # 绘制两只股票的价格
    ax.plot(dates, price1, linewidth=2.5, color='#1f77b4', label='Stock A', alpha=0.8)
    ax.plot(dates, price2 * 1.5 + 50, linewidth=2.5, color='#ff7f0e', label='Stock B', alpha=0.8)
    
    # 添加价差填充区域
    spread = price1 - 1.5 * (price2 * 1.5 + 50) + 500
    ax.fill_between(dates, spread + 400, spread + 600, alpha=0.3, color='green', label='价差区间')
    
    ax.set_xlabel('时间', fontsize=14, fontweight='bold')
    ax.set_ylabel('价格', fontsize=14, fontweight='bold')
    ax.set_title('配对交易策略示意图\nPairs Trading Strategy Overview', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加公式标注
    formula_text = r'$Spread = P_A - \beta \times P_B$'
    ax.text(0.5, 0.95, formula_text, 
            transform=ax.transAxes, fontsize=14, ha='center', 
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/cover.jpg', 
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print("✓ 生成封面图: cover.jpg")

# 图2: 交易信号图
def generate_figure1():
    """生成配对交易信号图"""
    # 计算价差和信号
    spread = price1 - beta_true * price2 - alpha_true
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号
    signals = np.zeros(n)
    signals[z_score > 2] = -1  # 做空
    signals[z_score < -2] = 1   # 做多
    signals[(z_score >= -0.5) & (z_score <= 0.5)] = 0  # 平仓
    
    # 状态机
    position = 0
    for i in range(1, n):
        if signals[i] != 0:
            position = signals[i]
            signals[i] = position
        elif position != 0:
            if abs(z_score[i]) <= 0.5:
                signals[i] = 0
                position = 0
            else:
                signals[i] = position
    
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # 图1: 价格序列
    ax1 = axes[0]
    ax1.plot(range(n), price1, label='Stock A', linewidth=2, color='blue')
    ax1.plot(range(n), price2, label='Stock B', linewidth=2, color='red', alpha=0.7)
    ax1.set_ylabel('价格', fontsize=12, fontweight='bold')
    ax1.set_title('价格序列 (Price Series)', fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 价差序列
    ax2 = axes[1]
    ax2.plot(range(n), spread, linewidth=2, color='green')
    ax2.axhline(y=spread_mean, color='black', linestyle='--', label='均值', linewidth=1.5)
    ax2.fill_between(range(n), 
                     spread_mean - 2*spread_std, 
                     spread_mean + 2*spread_std, 
                     alpha=0.2, color='green', label='±2σ')
    ax2.set_ylabel('价差', fontsize=12, fontweight='bold')
    ax2.set_title('协整价差 (Cointegration Spread)', fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # 图3: Z-Score
    ax3 = axes[2]
    ax3.plot(range(n), z_score, linewidth=2, color='purple')
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.7, linewidth=1.5, label='入场阈值 (+2)')
    ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.axhline(y=0.5, color='green', linestyle='--', alpha=0.7, linewidth=1.5, label='出场阈值 (±0.5)')
    ax3.axhline(y=-0.5, color='green', linestyle='--', alpha=0.7, linewidth=1.5)
    ax3.set_ylabel('Z-Score', fontsize=12, fontweight='bold')
    ax3.set_title('标准化价差 (Z-Score)', fontsize=14, fontweight='bold')
    ax3.legend(loc='best', fontsize=11)
    ax3.grid(True, alpha=0.3)
    
    # 图4: 交易信号
    ax4 = axes[3]
    ax4.plot(range(n), signals, linewidth=2, color='orange')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_ylabel('信号', fontsize=12, fontweight='bold')
    ax4.set_xlabel('交易日', fontsize=12, fontweight='bold')
    ax4.set_title('交易信号 (1: Long, -1: Short, 0: Close)', fontsize=14, fontweight='bold')
    ax4.set_ylim(-1.5, 1.5)
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/trading_signals.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图1: trading_signals.png")

# 图3: 组合表现图
def generate_figure2():
    """生成组合表现图"""
    # 简化回测：计算组合价值
    spread = price1 - beta_true * price2 - alpha_true
    spread_mean = spread.mean()
    spread_std = spread.std()
    z_score = (spread - spread_mean) / spread_std
    
    # 生成信号
    signals = np.zeros(n)
    signals[z_score > 2] = -1
    signals[z_score < -2] = 1
    signals[(z_score >= -0.5) & (z_score <= 0.5)] = 0
    
    position = 0
    for i in range(1, n):
        if signals[i] != 0:
            position = signals[i]
            signals[i] = position
        elif position != 0:
            if abs(z_score[i]) <= 0.5:
                signals[i] = 0
                position = 0
            else:
                signals[i] = position
    
    # 计算组合价值（简化）
    portfolio_value = np.zeros(n)
    position = 0
    entry_spread = 0
    
    for i in range(1, n):
        if signals[i] != 0 and position == 0:  # 开仓
            position = signals[i]
            entry_spread = spread[i]
            portfolio_value[i] = 0
        elif signals[i] == 0 and position != 0:  # 平仓
            portfolio_value[i] = (entry_spread - spread[i]) * position
            position = 0
        elif position != 0:  # 持仓
            portfolio_value[i] = portfolio_value[i-1] + (spread[i-1] - spread[i]) * position
    
    # 绘制
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 图1: 组合价值曲线
    ax1 = axes[0]
    ax1.plot(range(n), portfolio_value, linewidth=2.5, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.fill_between(range(n), 
                     portfolio_value, 
                     0, 
                     where=(portfolio_value >= 0), 
                     alpha=0.3, color='green')
    ax1.fill_between(range(n), 
                     portfolio_value, 
                     0, 
                     where=(portfolio_value < 0), 
                     alpha=0.3, color='red')
    ax1.set_ylabel('组合价值', fontsize=12, fontweight='bold')
    ax1.set_title('配对交易组合价值曲线 (Portfolio Value Curve)', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 添加标注
    max_value = np.max(portfolio_value)
    min_value = np.min(portfolio_value)
    ax1.annotate(f'Max: {max_value:.2f}', 
                xy=(np.argmax(portfolio_value), max_value),
                xytext=(10, 15), textcoords='offset points',
                fontsize=10, color='green', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='green'))
    ax1.annotate(f'Min: {min_value:.2f}', 
                xy=(np.argmin(portfolio_value), min_value),
                xytext=(10, -25), textcoords='offset points',
                fontsize=10, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red'))
    
    # 图2: 累计收益
    ax2 = axes[1]
    returns = np.diff(portfolio_value) / (np.abs(portfolio_value[:-1]) + 1e-8)
    returns = np.insert(returns, 0, 0)
    cumulative_returns = np.cumprod(1 + returns) - 1
    
    ax2.plot(range(n), cumulative_returns * 100, linewidth=2.5, color='purple')
    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax2.set_ylabel('累计收益 (%)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('交易日', fontsize=12, fontweight='bold')
    ax2.set_title('累计收益率曲线 (Cumulative Returns)', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # 添加性能指标标注
    total_return = cumulative_returns[-1]
    sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
    # 计算最大回撤（使用numpy）
    cummax = np.maximum.accumulate(portfolio_value)
    max_dd = ((cummax - portfolio_value) / (cummax + 1e-8)).max()
    
    textstr = '\n'.join((
        f'总收益率: {total_return:.2%}',
        f'夏普比率: {sharpe:.4f}',
        f'最大回撤: {max_dd:.2%}',
        f'交易次数: {(signals != 0).sum()}'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax2.text(0.02, 0.98, textstr, transform=ax2.transAxes,
            fontsize=11, verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/portfolio_performance.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图2: portfolio_performance.png")

# 图4: 残差分析图
def generate_figure3():
    """生成残差分析图"""
    # 计算残差（使用简单线性回归，不用statsmodels）
    X = np.column_stack([np.ones(n), price2])  # 添加常数项
    beta_ls = np.linalg.lstsq(X, price1, rcond=None)[0]
    residuals = price1 - X @ beta_ls
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图1: 残差时间序列
    ax1 = axes[0, 0]
    ax1.plot(range(n), residuals, linewidth=1.5, color='blue')
    ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
    ax1.set_title('残差时间序列 (Residuals Time Series)', fontsize=13, fontweight='bold')
    ax1.set_xlabel('日期', fontsize=11)
    ax1.set_ylabel('残差', fontsize=11)
    ax1.grid(True, alpha=0.3)
    
    # 图2: 残差直方图
    ax2 = axes[0, 1]
    ax2.hist(residuals, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax2.axvline(x=0, color='red', linestyle='--', linewidth=2)
    ax2.set_title('残差分布直方图 (Residuals Distribution)', fontsize=13, fontweight='bold')
    ax2.set_xlabel('残差值', fontsize=11)
    ax2.set_ylabel('频数', fontsize=11)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 添加正态分布拟合曲线
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, norm.pdf(x, residuals.mean(), residuals.std()) * len(residuals) * (residuals.max() - residuals.min()) / 50,
            'r-', linewidth=2, label='正态分布')
    ax2.legend()
    
    # 图3: ADF检验结果示意图（模拟数据）
    ax3 = axes[1, 0]
    adf_stat = -3.45  # 模拟的ADF统计量
    p_value = 0.006   # 模拟的p-value
    critical_values = {'1%': -3.43, '5%': -2.86, '10%': -2.57}
    
    # 绘制ADF统计量 vs 临界值
    crit_labels = ['1%', '5%', '10%']
    crit_values = [critical_values['1%'], critical_values['5%'], critical_values['10%']]
    
    ax3.bar(crit_labels, crit_values, alpha=0.5, color='gray', label='临界值')
    ax3.axhline(y=adf_stat, color='red', linewidth=3, label=f'ADF统计量 ({adf_stat:.2f})')
    ax3.set_ylabel('统计量值', fontsize=11, fontweight='bold')
    ax3.set_title('ADF检验: 统计量 vs 临界值', fontsize=13, fontweight='bold')
    ax3.legend(loc='best', fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # 添加p-value标注
    ax3.text(0.5, 0.95, f'p-value = {p_value:.4f}', 
            transform=ax3.transAxes, fontsize=11, ha='center',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    
    # 图4: 残差自相关图（简化版，手动计算）
    ax4 = axes[1, 1]
    
    # 手动计算ACF
    def acf(x, lags=40):
        x = x - x.mean()
        autocorr = np.correlate(x, x, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr /= autocorr[0]
        return autocorr[:lags]
    
    acf_values = acf(residuals, lags=41)
    lags = np.arange(len(acf_values))
    
    # 绘制ACF
    ax4.vlines(lags, 0, acf_values, linewidth=2, color='blue')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # 添加置信区间
    conf_level = 1.96 / np.sqrt(n)
    ax4.axhline(y=conf_level, color='red', linestyle='--', alpha=0.7)
    ax4.axhline(y=-conf_level, color='red', linestyle='--', alpha=0.7)
    
    ax4.set_title('残差自相关图 (ACF)', fontsize=13, fontweight='bold')
    ax4.set_xlabel('滞后阶数', fontsize=11)
    ax4.set_ylabel('ACF', fontsize=11)
    ax4.set_xlim(0, 40)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('协整残差诊断分析 (Residual Diagnostics)', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration/residual_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图3: residual_analysis.png")

# 生成所有图片
if __name__ == '__main__':
    print("开始生成第二篇文章的配图...")
    generate_cover_image()
    generate_figure1()
    generate_figure2()
    generate_figure3()
    print("\n✅ 第二篇文章的所有配图已生成完成！")
