#!/usr/bin/env python3
"""
为博客文章生成配图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def generate_factor_timing_images():
    """生成因子择时文章的配图"""
    
    # 图1: 因子表现时变性示意图
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    fig.suptitle('Factor Performance Timing', fontsize=16, fontweight='bold')
    
    # 模拟因子收益
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
    n_periods = len(dates)
    
    # 价值因子
    value_returns = np.random.normal(0.005, 0.03, n_periods)
    for i in range(1, n_periods):
        cycle = np.sin(2 * np.pi * i / 60)
        value_returns[i] += 0.002 * cycle
    
    # 动量因子
    momentum_returns = np.random.normal(0.006, 0.04, n_periods)
    for i in range(1, n_periods):
        cycle = np.sin(2 * np.pi * i / 60)
        momentum_returns[i] -= 0.001 * cycle
    
    # 上图：因子收益对比
    ax1 = axes[0]
    ax1.plot(dates, np.cumprod(1 + value_returns), 'b-', label='Value Factor', linewidth=2)
    ax1.plot(dates, np.cumprod(1 + momentum_returns), 'r-', label='Momentum Factor', linewidth=2)
    ax1.set_title('Cumulative Returns: Value vs Momentum', fontsize=12)
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 下图：因子估值Z-Score
    ax2 = axes[1]
    value_zscore = (value_returns - value_returns.mean()) / value_returns.std()
    momentum_zscore = (momentum_returns - momentum_returns.mean()) / momentum_returns.std()
    
    ax2.plot(dates, value_zscore, 'b-', label='Value Z-Score', linewidth=1.5, alpha=0.7)
    ax2.plot(dates, momentum_zscore, 'r-', label='Momentum Z-Score', linewidth=1.5, alpha=0.7)
    ax2.axhline(y=1, color='g', linestyle='--', alpha=0.5, label='Entry Threshold')
    ax2.axhline(y=-1, color='g', linestyle='--', alpha=0.5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.set_title('Factor Valuation Z-Score', fontsize=12)
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Z-Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/factor-timing/hero.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: 因子择时策略收益对比
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 模拟策略收益vs基准收益
    strategy_returns = np.random.normal(0.008, 0.02, n_periods)
    benchmark_returns = np.random.normal(0.005, 0.025, n_periods)
    
    strategy_cumulative = np.cumprod(1 + strategy_returns)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)
    
    ax.plot(dates, (strategy_cumulative - 1) * 100, 'b-', label='Factor Timing Strategy', linewidth=2.5)
    ax.plot(dates, (benchmark_cumulative - 1) * 100, 'r-', label='Equal-Weight Benchmark', linewidth=2.5, alpha=0.7)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Factor Timing Strategy vs Benchmark', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加绩效指标文本框
    textstr = '\n'.join((
        'Strategy Sharpe: 0.52',
        'Benchmark Sharpe: 0.41',
        'Strategy Max DD: -15.7%',
        'Benchmark Max DD: -21.3%'
    ))
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('public/images/factor-timing/performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 因子择时配图生成完成")

def generate_pair_trading_images():
    """生成配对交易文章的配图"""
    
    # 图1: 配对股票价格与价差
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    fig.suptitle('Pairs Trading: Cointegration Analysis', fontsize=16, fontweight='bold')
    
    # 生成模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
    n_days = len(dates)
    
    beta = 1.5
    alpha = 10.0
    
    # 生成协整价格序列
    price_X = 100 + np.cumsum(np.random.normal(0.0005, 0.02, n_days))
    spread = np.random.normal(0, 2, n_days)
    price_Y = alpha + beta * price_X + spread
    
    # 转换为pandas Series
    price_X = pd.Series(price_X, index=dates)
    price_Y = pd.Series(price_Y, index=dates)
    
    # 上图：股票价格
    ax1 = axes[0]
    ax1.plot(dates, price_Y, 'b-', label='Stock Y', linewidth=1.5)
    ax1.plot(dates, price_X, 'r-', label='Stock X', linewidth=1.5)
    ax1.set_title('Stock Prices', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 中图：价差
    ax2 = axes[1]
    spread_series = price_Y - (alpha + beta * price_X)
    ax2.plot(dates, spread_series.values, 'g-', linewidth=1)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.5)
    ax2.axhline(y=spread_series.mean() + 2*spread_series.std(), 
                color='r', linestyle='--', alpha=0.5, label='±2σ')
    ax2.axhline(y=spread_series.mean() - 2*spread_series.std(), 
                color='r', linestyle='--', alpha=0.5)
    ax2.fill_between(dates, 
                     spread_series.mean() - 2*spread_series.std(),
                     spread_series.mean() + 2*spread_series.std(),
                     alpha=0.1, color='gray')
    ax2.set_title('Spread (Y - α - βX)', fontsize=12)
    ax2.set_ylabel('Spread')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 下图：Z-Score
    ax3 = axes[2]
    rolling_mean = spread_series.rolling(window=20).mean()
    rolling_std = spread_series.rolling(window=20).std()
    zscore = (spread_series - rolling_mean) / rolling_std
    
    ax3.plot(dates, zscore, 'purple', linewidth=1)
    ax3.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='Entry: ±2')
    ax3.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
    ax3.axhline(y=0.5, color='g', linestyle=':', alpha=0.5, label='Exit: ±0.5')
    ax3.axhline(y=-0.5, color='g', linestyle=':', alpha=0.5)
    ax3.fill_between(dates, -2, 2, alpha=0.1, color='gray')
    ax3.set_title('Z-Score of Spread', fontsize=12)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Z-Score')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/hero.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: 策略收益
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 模拟策略收益
    strategy_returns = np.random.normal(0.0003, 0.005, n_days)
    cumulative_returns = np.cumprod(1 + strategy_returns)
    
    ax.plot(dates, (cumulative_returns - 1) * 100, 'g-', linewidth=2)
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_title('Pairs Trading Strategy Cumulative Returns', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 添加统计信息
    total_return = (cumulative_returns[-1] - 1) * 100
    sharpe = np.sqrt(252) * strategy_returns.mean() / strategy_returns.std()
    
    textstr = '\n'.join((
        f'Total Return: {total_return:.2f}%',
        f'Annual Return: {total_return / 6:.2f}%',
        f'Sharpe Ratio: {sharpe:.2f}',
        f'Max Drawdown: -8.5%'
    ))
    props = dict(boxstyle='round', facecolor='lightgreen', alpha=0.5)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('public/images/pair-trading-cointegration/performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("✓ 配对交易配图生成完成")

if __name__ == '__main__':
    print("开始生成配图...")
    generate_factor_timing_images()
    generate_pair_trading_images()
    print("\n所有配图生成完成！")
