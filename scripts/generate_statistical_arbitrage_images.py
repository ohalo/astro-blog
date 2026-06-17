#!/usr/bin/env python3
"""
生成统计套利文章的配图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 设置英文字体以避免中文警告
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_spread_analysis_chart():
    """生成价差分析图"""
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10))
    
    # 模拟时间轴
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='D')
    np.random.seed(42)
    
    # 模拟两只股票的价格（协整关系）
    n = len(dates)
    common_factor = np.cumsum(np.random.normal(0, 0.01, n))
    stock1 = 100 + common_factor + np.random.normal(0, 0.02, n)
    stock2 = 50 + 0.5 * common_factor + np.random.normal(0, 0.015, n)
    
    # 计算价差
    spread = stock1 - 2 * stock2  # 假设对冲比例为2
    spread_mean = spread.mean()
    spread_std = spread.std()
    
    # 子图1：两只股票价格
    ax1.plot(dates, stock1, label='Stock 1', linewidth=2, alpha=0.8)
    ax1.plot(dates, stock2 * 2, label='Stock 2 (x2)', linewidth=2, alpha=0.8)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Price Series of Pair', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：价差序列
    ax2.plot(dates, spread, linewidth=2, color='blue', alpha=0.8)
    ax2.axhline(y=spread_mean, color='red', linestyle='--', 
                label=f'Mean ({spread_mean:.2f})')
    ax2.axhline(y=spread_mean + 2*spread_std, color='green', 
                linestyle=':', label='+2 STD')
    ax2.axhline(y=spread_mean - 2*spread_std, color='green', 
                linestyle=':', label='-2 STD')
    ax2.fill_between(dates, spread_mean - 2*spread_std, 
                     spread_mean + 2*spread_std, alpha=0.2, color='gray')
    ax2.set_ylabel('Spread', fontsize=12)
    ax2.set_title('Spread with Mean Reversion Bands', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：Z-score
    zscore = (spread - spread_mean) / spread_std
    ax3.plot(dates, zscore, linewidth=2, color='purple', alpha=0.8)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax3.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
    ax3.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Entry (-2)')
    ax3.axhline(y=0.5, color='green', linestyle=':', alpha=0.5, label='Exit (+0.5)')
    ax3.axhline(y=-0.5, color='green', linestyle=':', alpha=0.5, label='Exit (-0.5)')
    ax3.fill_between(dates, -2, 2, alpha=0.1, color='yellow')
    ax3.set_ylabel('Z-score', fontsize=12)
    ax3.set_xlabel('Date', fontsize=12)
    ax3.set_title('Trading Signals (Z-score)', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/spread_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated spread analysis chart")
    plt.close()

def generate_pairs_performance_chart():
    """生成配对交易绩效图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模拟回测数据
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='D')
    np.random.seed(42)
    
    # 策略收益
    strategy_returns = np.random.normal(0.0005, 0.005, len(dates))
    strategy_cumulative = (1 + pd.Series(strategy_returns, index=dates)).cumprod()
    
    # 基准收益（市场）
    benchmark_returns = np.random.normal(0.0003, 0.008, len(dates))
    benchmark_cumulative = (1 + pd.Series(benchmark_returns, index=dates)).cumprod()
    
    # 绘制累计收益
    ax1.plot(dates, strategy_cumulative, label='Pairs Strategy', 
            linewidth=2, color='blue', alpha=0.8)
    ax1.plot(dates, benchmark_cumulative, label='Benchmark', 
            linewidth=2, color='gray', alpha=0.6, linestyle='--')
    ax1.set_title('Cumulative Returns', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return (x)', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 绘制回撤
    strategy_dd = 1 - strategy_cumulative / strategy_cumulative.cummax()
    benchmark_dd = 1 - benchmark_cumulative / benchmark_cumulative.cummax()
    
    ax2.fill_between(dates, -strategy_dd * 100, 0, 
                     alpha=0.3, color='blue', label='Pairs Strategy')
    ax2.fill_between(dates, -benchmark_dd * 100, 0, 
                     alpha=0.3, color='gray', label='Benchmark')
    ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/pairs_performance.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated pairs performance chart")
    plt.close()

def generate_cointegration_test_chart():
    """生成协整检验可视化图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    
    # 模拟数据
    np.random.seed(42)
    n = 500
    
    # 生成协整序列
    x = np.cumsum(np.random.normal(0, 1, n))
    y = 2 + 1.5 * x + np.random.normal(0, 0.5, n)
    
    # 子图1：原始价格序列
    ax1.plot(range(n), x, label='Independent Variable (X)', alpha=0.8)
    ax1.set_title('Independent Variable Series', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 子图2：因变量
    ax2.plot(range(n), y, label='Dependent Variable (Y)', alpha=0.8, color='orange')
    ax2.set_title('Dependent Variable Series', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Price')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 子图3：回归散点图
    ax3.scatter(x, y, alpha=0.5, s=10)
    model = np.polyfit(x, y, 1)
    ax3.plot(x, np.polyval(model, x), 'r-', linewidth=2, label=f'y = {model[0]:.2f}x + {model[1]:.2f}')
    ax3.set_title('Cointegration Regression', fontsize=12, fontweight='bold')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 子图4：残差（价差）
    residuals = y - np.polyval(model, x)
    ax4.plot(range(n), residuals, linewidth=2, color='green', alpha=0.8)
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax4.axhline(y=residuals.mean() + 2*residuals.std(), color='red', 
                linestyle='--', alpha=0.5)
    ax4.axhline(y=residuals.mean() - 2*residuals.std(), color='red', 
                linestyle='--', alpha=0.5)
    ax4.set_title('Residuals (Spread) - Should be Stationary', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Time')
    ax4.set_ylabel('Residual')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cointegration_test.png', 
                dpi=300, bbox_inches='tight')
    print("✓ Generated cointegration test chart")
    plt.close()

if __name__ == '__main__':
    print("Generating statistical arbitrage article images...")
    generate_spread_analysis_chart()
    generate_pairs_performance_chart()
    generate_cointegration_test_chart()
    print("✅ All images generated successfully!")
