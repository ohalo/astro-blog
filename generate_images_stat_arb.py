#!/usr/bin/env python3
"""
为统计套利与均值回归文章生成配图
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion', exist_ok=True)

# 图1: 均值回归示意图
def generate_figure1():
    """图1: 均值回归过程示意图"""
    np.random.seed(42)
    n_periods = 250
    
    # 生成OU过程（均值回归）
    mu = 100  # 长期均值
    theta = 0.1  # 回归速度
    sigma = 5  # 波动率
    
    prices = [mu]
    for t in range(1, n_periods):
        drift = theta * (mu - prices[-1])
        shock = sigma * np.random.randn()
        new_price = prices[-1] + drift + shock
        prices.append(new_price)
    
    prices = pd.Series(prices)
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='B')
    prices.index = dates
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, prices, linewidth=2, color='#2c3e50', label='Price')
    ax.axhline(y=mu, color='red', linestyle='--', linewidth=2, label=f'Mean (μ={mu})')
    
    # 标注均值回归
    ax.fill_between(dates, mu - sigma*2, mu + sigma*2, alpha=0.2, color='green', label='±2σ Band')
    
    # 添加箭头表示回归
    for i in range(50, 200, 50):
        if prices.iloc[i] > mu:
            ax.annotate('', xy=(dates[i], mu), xytext=(dates[i], prices.iloc[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
        else:
            ax.annotate('', xy=(dates[i], mu), xytext=(dates[i], prices.iloc[i]),
                       arrowprops=dict(arrowstyle='->', color='blue', lw=2))
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.set_title('Mean Reversion Process (Ornstein-Uhlenbeck)', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/figure1.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: statistical-arbitrage-mean-reversion/figure1.png")

# 图2: 配对交易价差
def generate_figure2():
    """图2: 配对交易价差与Z-Score"""
    np.random.seed(42)
    n_periods = 250
    
    # 生成协整配对的价格
    base_price = 100 + np.cumsum(np.random.randn(n_periods) * 0.5)
    stock1 = base_price + np.random.randn(n_periods) * 2
    stock2 = base_price * 1.5 + np.random.randn(n_periods) * 2
    
    # 计算价差
    spread = stock1 - 0.67 * stock2
    
    # 计算Z-Score
    z_score = (spread - spread.mean()) / spread.std()
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='B')
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # 图2上：价差
    ax1 = axes[0]
    ax1.plot(dates, spread, linewidth=2, color='blue', label='Spread')
    ax1.axhline(y=spread.mean(), color='red', linestyle='--', label='Mean')
    ax1.fill_between(dates, 
                      spread.mean() - 2*spread.std(), 
                      spread.mean() + 2*spread.std(), 
                      alpha=0.2, color='gray', label='±2σ')
    ax1.set_ylabel('Spread', fontsize=12)
    ax1.set_title('Pair Trading: Spread and Z-Score', fontsize=15, fontweight='bold')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # 标注交易信号
    long_signals = z_score < -2
    short_signals = z_score > 2
    ax1.scatter(dates[long_signals], spread[long_signals], 
                marker='^', color='green', s=100, label='Long Spread', zorder=5)
    ax1.scatter(dates[short_signals], spread[short_signals], 
                marker='v', color='red', s=100, label='Short Spread', zorder=5)
    
    # 图2下：Z-Score
    ax2 = axes[1]
    ax2.plot(dates, z_score, linewidth=2, color='purple')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.axhline(y=2, color='red', linestyle='--', alpha=0.5, label='Entry (+2)')
    ax2.axhline(y=-2, color='red', linestyle='--', alpha=0.5, label='Entry (-2)')
    ax2.axhline(y=0, color='green', linestyle='--', alpha=0.5, label='Exit (0)')
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Z-Score', fontsize=12)
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/figure2.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: statistical-arbitrage-mean-reversion/figure2.png")

# 图3: 协整关系
def generate_figure3():
    """图3: 协整关系 vs 纯相关性"""
    np.random.seed(42)
    n_periods = 250
    
    # 协整序列
    coint_y = np.cumsum(np.random.randn(n_periods)) + 50
    coint_x = 1.5 * coint_y + np.random.randn(n_periods) * 10
    
    # 纯相关但非协整（两个独立的随机游走）
    random_walk1 = np.cumsum(np.random.randn(n_periods))
    random_walk2 = np.cumsum(np.random.randn(n_periods))
    
    # 计算滚动相关性
    coint_corr = pd.Series(coint_x).rolling(60).corr(pd.Series(coint_y))
    noncoint_corr = pd.Series(random_walk1).rolling(60).corr(pd.Series(random_walk2))
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='B')
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图3左上：协整序列
    axes[0, 0].plot(dates, coint_x, linewidth=2, label='X (Stock 1)', color='blue')
    axes[0, 0].plot(dates, coint_y, linewidth=2, label='Y (Stock 2)', color='red')
    axes[0, 0].set_title('Cointegrated Series', fontsize=13, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 图3右上：非协整序列
    axes[0, 1].plot(dates, random_walk1, linewidth=2, label='Random Walk 1', color='blue')
    axes[0, 1].plot(dates, random_walk2, linewidth=2, label='Random Walk 2', color='red')
    axes[0, 1].set_title('Non-Cointegrated (Independent Random Walks)', fontsize=13, fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 图3左下：协整的价差（平稳）
    spread_coint = pd.Series(coint_x) - 1.5 * pd.Series(coint_y)
    axes[1, 0].plot(dates, spread_coint, linewidth=2, color='green')
    axes[1, 0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 0].set_title('Spread (Stationary)', fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 图3右下：非协整的价差（非平稳）
    spread_noncoint = pd.Series(random_walk1) - pd.Series(random_walk2)
    axes[1, 1].plot(dates, spread_noncoint, linewidth=2, color='orange')
    axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
    axes[1, 1].set_title('Spread (Non-Stationary)', fontsize=13, fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/figure3.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: statistical-arbitrage-mean-reversion/figure3.png")

# 图4: 回测净值曲线
def generate_figure4():
    """图4: 配对交易策略 vs 买入持有"""
    np.random.seed(42)
    n_periods = 250
    
    # 生成策略收益（均值回归策略）
    strategy_returns = np.random.normal(0.0005, 0.01, n_periods)
    strategy_cumulative = np.cumprod(1 + strategy_returns) - 1
    
    # 生成基准收益（买入持有）
    benchmark_returns = np.random.normal(0.0003, 0.015, n_periods)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns) - 1
    
    dates = pd.date_range('2020-01-01', periods=n_periods, freq='B')
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, strategy_cumulative * 100, linewidth=2.5, 
            label='Pairs Trading Strategy', color='#2ecc71')
    ax.plot(dates, benchmark_cumulative * 100, linewidth=2.5, 
            label='Buy and Hold (Benchmark)', color='#e74c3c', linestyle='--')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return (%)', fontsize=12)
    ax.set_title('Pairs Trading Strategy vs Benchmark', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加最终收益标注
    final_strategy = strategy_cumulative[-1] * 100
    final_benchmark = benchmark_cumulative[-1] * 100
    ax.text(dates[-1], final_strategy, f'{final_strategy:.1f}%', 
            ha='right', va='bottom', fontsize=10, fontweight='bold', color='#2ecc71')
    ax.text(dates[-1], final_benchmark, f'{final_benchmark:.1f}%', 
            ha='right', va='bottom', fontsize=10, fontweight='bold', color='#e74c3c')
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/figure4.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: statistical-arbitrage-mean-reversion/figure4.png")

# 生成封面图
def generate_cover():
    """生成文章封面图"""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 创建均值回归的视觉化表示
    np.random.seed(42)
    x = np.linspace(0, 10, 1000)
    y = 50 + 10 * np.sin(x) + 5 * np.random.randn(1000)
    
    # 绘制价格序列
    ax.plot(x, y, linewidth=3, color='#3498db', label='Price')
    ax.axhline(y=50, color='#e74c3c', linestyle='--', linewidth=3, label='Mean')
    
    # 标注回归
    ax.fill_between(x, 50 - 10, 50 + 10, alpha=0.2, color='gray')
    
    # 添加箭头
    for i in [100, 300, 500, 700, 900]:
        if y[i] > 50:
            ax.annotate('', xy=(x[i], 50), xytext=(x[i], y[i]),
                       arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
        else:
            ax.annotate('', xy=(x[i], 50), xytext=(x[i], y[i]),
                       arrowprops=dict(arrowstyle='->', color='#2ecc71', lw=3))
    
    ax.set_xlabel('Time', fontsize=16)
    ax.set_ylabel('Price', fontsize=16)
    ax.set_title('Mean Reversion in Statistical Arbitrage', 
                 fontsize=22, fontweight='bold', pad=20)
    ax.legend(fontsize=14, loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/statistical-arbitrage-mean-reversion/cover.jpg', 
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print("✓ Cover generated: statistical-arbitrage-mean-reversion/cover.jpg")

if __name__ == '__main__':
    print("Generating images for Statistical Arbitrage article...")
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_cover()
    print("\n✅ All images generated successfully!")
