#!/usr/bin/env python3
"""
为博客文章生成配图
"""

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

def generate_factor_timing_images():
    """为因子择时文章生成配图"""
    output_dir = "/Users/halo/workspace/astro-blog/public/images/factor-timing"
    os.makedirs(output_dir, exist_ok=True)
    
    # 图1: 不同因子在不同市场环境下的表现
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
    n = len(dates)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 模拟不同因子的累积收益
    factors = {
        '价值因子': np.cumsum(np.random.randn(n) * 0.02 + 0.008),
        '动量因子': np.cumsum(np.random.randn(n) * 0.025 + 0.006),
        '质量因子': np.cumsum(np.random.randn(n) * 0.015 + 0.01),
        '小盘因子': np.cumsum(np.random.randn(n) * 0.03 + 0.005),
    }
    
    for name, returns in factors.items():
        ax.plot(dates, returns * 100, label=name, linewidth=2)
    
    ax.set_title('Factor Performance Over Time (2020-2025)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/factor-performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: 因子择时 vs 静态因子配置
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 静态因子组合
    static = np.cumsum(np.random.randn(n) * 0.018 + 0.008)
    
    # 因子择时组合（在因子表现好时增加暴露）
    timing = np.ones(n) * 0.008
    for i in range(1, n):
        # 简单模拟：如果过去3个月因子表现好，增加暴露
        if i >= 3:
            recent_perf = np.mean([factors['价值因子'][i-j] - factors['价值因子'][i-j-1] for j in range(1, 4)])
            if recent_perf > 0:
                timing[i] += 0.004
        timing[i] = max(0.002, min(0.015, timing[i]))
    
    timing_returns = np.cumsum(timing + np.random.randn(n) * 0.015)
    
    ax.plot(dates, static * 100, label='Static Factor Exposure', linewidth=2, linestyle='--')
    ax.plot(dates, timing_returns * 100, label='Dynamic Factor Timing', linewidth=2)
    
    ax.set_title('Factor Timing vs Static Exposure', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/timing-vs-static.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 因子择时配图已生成: {output_dir}")
    return [
        f'{output_dir}/factor-performance.png',
        f'{output_dir}/timing-vs-static.png'
    ]

def generate_pair_trading_images():
    """为配对交易文章生成配图"""
    output_dir = "/Users/halo/workspace/astro-blog/public/images/pair-trading-cointegration"
    os.makedirs(output_dir, exist_ok=True)
    
    np.random.seed(42)
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='D')
    n = len(dates)
    
    # 生成协整的价格序列
    # 两个价格序列有共同的随机趋势
    common_trend = np.cumsum(np.random.randn(n) * 0.01)
    stock_a = 100 + np.cumsum(np.random.randn(n) * 0.005) + common_trend * 0.8
    stock_b = 80 + np.cumsum(np.random.randn(n) * 0.006) + common_trend * 0.75
    
    # 图1: 两只股票价格走势及价差
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax1.plot(dates, stock_a, label='Stock A', linewidth=2)
    ax1.plot(dates, stock_b, label='Stock B', linewidth=2)
    ax1.set_ylabel('Price', fontsize=12)
    ax1.set_title('Cointegrated Pair: Price Series', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 计算价差
    spread = stock_a - stock_b
    ax2.plot(dates, spread, color='darkorange', linewidth=1.5)
    ax2.axhline(y=spread.mean(), color='red', linestyle='--', label='Mean')
    ax2.axhline(y=spread.mean() + 2*spread.std(), color='green', linestyle=':', label='+2σ')
    ax2.axhline(y=spread.mean() - 2*spread.std(), color='green', linestyle=':', label='-2σ')
    ax2.set_ylabel('Spread (A - B)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_title('Spread Time Series', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/price-spread.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 图2: 配对交易信号
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 计算z-score
    z_score = (spread - spread.mean()) / spread.std()
    
    # 交易信号
    long_spread = z_score < -2  # 做多价差（买A卖B）
    short_spread = z_score > 2   # 做空价差（卖A买B）
    
    ax.plot(dates, z_score, label='Z-Score', linewidth=1.5)
    ax.scatter(dates[long_spread], z_score[long_spread], 
               color='green', s=30, label='Long Spread (Buy A, Sell B)', zorder=5)
    ax.scatter(dates[short_spread], z_score[short_spread], 
               color='red', s=30, label='Short Spread (Sell A, Buy B)', zorder=5)
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax.axhline(y=2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(y=-2, color='green', linestyle='--', alpha=0.5)
    
    ax.set_title('Pair Trading Signals (Z-Score)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Z-Score', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/trading-signals.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ 配对交易配图已生成: {output_dir}")
    return [
        f'{output_dir}/price-spread.png',
        f'{output_dir}/trading-signals.png'
    ]

if __name__ == "__main__":
    print("🎨 开始生成博客配图...")
    generate_factor_timing_images()
    generate_pair_trading_images()
    print("✅ 所有配图生成完成！")
