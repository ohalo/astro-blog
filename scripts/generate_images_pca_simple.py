#!/usr/bin/env python3
"""
生成PCA统计套利文章的配图（简化版 - 无需sklearn）
使用模拟数据展示概念和结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import os

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def generate_pca_variance_plot():
    """生成PCA解释方差图（使用模拟数据）"""
    print("生成PCA解释方差图...")
    
    # 模拟PCA解释方差（典型的特征值衰减模式）
    n_components = 20
    
    # 模拟前几个主成分解释大部分方差
    explained_variance = np.array([
        0.35,  # PC1: 市场因子
        0.12,  # PC2: 行业因子1
        0.08,  # PC3: 行业因子2
        0.06,  # PC4: 风格因子1
        0.05,  # PC5: 风格因子2
        0.04,  # PC6
        0.03,  # PC7
        0.025, # PC8
        0.020, # PC9
        0.015, # PC10
        0.012, # PC11
        0.010, # PC12
        0.008, # PC13
        0.007, # PC14
        0.006, # PC15
        0.005, # PC16
        0.004, # PC17
        0.003, # PC18
        0.002, # PC19
        0.001, # PC20
    ])
    
    cumulative_variance = np.cumsum(explained_variance)
    
    # 绘制图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 左图：单个主成分解释方差
    colors = ['#1f77b4'] * 3 + ['#ff7f0e'] * 7 + ['#2ca02c'] * 10
    bars = axes[0].bar(range(1, n_components + 1), explained_variance, 
                        alpha=0.7, color=colors, edgecolor='black', linewidth=1)
    
    axes[0].set_xlabel('主成分序号', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('解释方差比例', fontsize=12, fontweight='bold')
    axes[0].set_title('各主成分解释方差比例', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--', axis='y')
    axes[0].set_xticks(range(1, n_components + 1))
    axes[0].set_ylim([0, 0.4])
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars, explained_variance)):
        if i < 10:  # 只标注前10个
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{val:.1%}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 添加注释
    axes[0].text(3, 0.38, '前3个主成分\n解释55%方差', ha='center', fontsize=9,
                  bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.5))
    axes[0].axvline(x=3.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # 右图：累计解释方差
    axes[1].plot(range(1, n_components + 1), cumulative_variance, 
                 marker='o', color='darkred', linewidth=2.5, markersize=6, zorder=3)
    axes[1].fill_between(range(1, n_components + 1), 0, cumulative_variance, 
                          alpha=0.2, color='darkred', zorder=1)
    
    axes[1].axhline(y=0.8, color='gray', linestyle='--', linewidth=2, label='80%阈值', alpha=0.7)
    axes[1].axhline(y=0.9, color='orange', linestyle='--', linewidth=2, label='90%阈值', alpha=0.7)
    
    axes[1].set_xlabel('主成分序号', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('累计解释方差比例', fontsize=12, fontweight='bold')
    axes[1].set_title('累计解释方差', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower right', framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(range(1, n_components + 1))
    axes[1].set_ylim([0, 1])
    
    # 标注关键点
    for idx in [4, 9, 14, 19]:
        axes[1].annotate(f'{cumulative_variance[idx]:.1%}', 
                         xy=(idx+1, cumulative_variance[idx]),
                         xytext=(10, 10), textcoords='offset points',
                         fontsize=9, fontweight='bold',
                         bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                         arrowprops=dict(arrowstyle='->', lw=1.5))
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/pca_variance_explained.png', dpi=300, bbox_inches='tight')
    print("✓ PCA解释方差图已保存")
    plt.close()

def generate_trading_signals_plot():
    """生成交易信号示意图"""
    print("生成交易信号示意图...")
    
    # 生成模拟残差数据（均值回归过程）
    np.random.seed(42)
    n_days = 250
    
    # Ornstein-Uhlenbeck process (均值回归)
    residuals = np.zeros(n_days)
    residuals[0] = 0
    
    theta = 0.1  # 均值回归速度
    sigma = 0.5   # 波动率
    
    for i in range(1, n_days):
        dW = np.random.randn() * np.sqrt(1)  # 维纳过程增量
        residuals[i] = residuals[i-1] - theta * residuals[i-1] + sigma * dW
    
    # 标准化
    residuals = (residuals - residuals.mean()) / residuals.std()
    
    # 计算Z-Score并生成信号
    window = 20
    z_scores = np.zeros(n_days)
    signals = np.zeros(n_days)
    
    for i in range(window, n_days):
        mean = residuals[i-window:i].mean()
        std = residuals[i-window:i].std()
        z_scores[i] = (residuals[i] - mean) / std
        
        if z_scores[i] < -2.0:
            signals[i] = 1  # 买入信号
        elif z_scores[i] > 2.0:
            signals[i] = -1  # 卖出信号
    
    # 创建日期索引
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # 绘制图形
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True, facecolor='white')
    
    # 上图：残差序列
    axes[0].plot(dates, residuals, color='#1f77b4', linewidth=1.8, label='残差', zorder=3)
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1, alpha=0.5, zorder=2)
    
    # 填充正负区域
    axes[0].fill_between(dates, 0, residuals, where=(residuals >= 0), 
                          alpha=0.3, color='red', label='正值区域', zorder=1)
    axes[0].fill_between(dates, 0, residuals, where=(residuals < 0), 
                          alpha=0.3, color='green', label='负值区域', zorder=1)
    
    axes[0].set_ylabel('残差值', fontsize=12, fontweight='bold')
    axes[0].set_title('残差序列与均值回归特性', fontsize=14, fontweight='bold', pad=10)
    axes[0].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 下图：交易信号
    axes[1].plot(dates, residuals, color='gray', linewidth=1, alpha=0.6, label='残差', zorder=2)
    axes[1].axhline(y=2.0, color='darkred', linestyle='--', linewidth=2, label='卖出阈值 (+2σ)', alpha=0.8, zorder=3)
    axes[1].axhline(y=-2.0, color='darkgreen', linestyle='--', linewidth=2, label='买入阈值 (-2σ)', alpha=0.8, zorder=3)
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3, zorder=1)
    
    # 标记买入信号
    buy_indices = np.where(signals == 1)[0]
    if len(buy_indices) > 0:
        axes[1].scatter(dates[buy_indices], residuals[buy_indices], 
                        color='red', marker='^', s=200, 
                        label='买入信号', zorder=5, edgecolors='black', linewidth=2,
                        alpha=0.9)
    
    # 标记卖出信号
    sell_indices = np.where(signals == -1)[0]
    if len(sell_indices) > 0:
        axes[1].scatter(dates[sell_indices], residuals[sell_indices], 
                        color='green', marker='v', s=200, 
                        label='卖出信号', zorder=5, edgecolors='black', linewidth=2,
                        alpha=0.9)
    
    axes[1].set_xlabel('日期', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('残差值', fontsize=12, fontweight='bold')
    axes[1].set_title('基于Z-Score的交易信号生成', fontsize=14, fontweight='bold', pad=10)
    axes[1].legend(loc='upper right', fontsize=10, framealpha=0.9)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # 格式化x轴日期
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/trading_signals.png', dpi=300, bbox_inches='tight')
    print("✓ 交易信号图已保存")
    plt.close()

def generate_portfolio_performance_plot():
    """生成组合绩效曲线图"""
    print("生成组合绩效曲线图...")
    
    # 生成模拟的组合净值数据
    np.random.seed(42)
    n_days = 500
    
    # 模拟日收益率（年化12%，波动率10%）
    annual_return = 0.12
    annual_vol = 0.10
    
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    daily_returns = np.random.normal(daily_return, daily_vol, n_days)
    
    # 计算累计净值
    cumulative_return = np.cumprod(1 + daily_returns)
    
    # 计算基准（市场，年化8%）
    benchmark_annual_return = 0.08
    benchmark_daily_return = benchmark_annual_return / 252
    benchmark_daily_vol = annual_vol * 1.2 / np.sqrt(252)
    
    benchmark_daily_returns = np.random.normal(benchmark_daily_return, benchmark_daily_vol, n_days)
    benchmark_cumulative = np.cumprod(1 + benchmark_daily_returns)
    
    # 创建日期索引
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # 绘制图形
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # 绘制策略净值
    ax.plot(dates, cumulative_return, color='#1f77b4', linewidth=2.5, 
            label='PCA统计套利策略', zorder=4, alpha=0.9)
    
    # 绘制基准
    ax.plot(dates, benchmark_cumulative, color='gray', linewidth=2, linestyle='--', 
            label='市场基准 (S&P 500)', alpha=0.7, zorder=3)
    
    # 填充区域
    ax.fill_between(dates, 1, cumulative_return, where=(cumulative_return >= 1), 
                     alpha=0.2, color='green', zorder=1)
    ax.fill_between(dates, 1, cumulative_return, where=(cumulative_return < 1), 
                     alpha=0.2, color='red', zorder=1)
    
    # 标注关键指标
    total_return = (cumulative_return[-1] - 1) * 100
    annual_return_actual = ((cumulative_return[-1]) ** (252/n_days) - 1) * 100
    
    # 计算最大回撤（使用numpy的cummax等价操作）
    running_max = np.maximum.accumulate(cumulative_return)
    drawdown = (cumulative_return - running_max) / running_max
    max_drawdown = drawdown.min() * 100
    
    stats_text = f'总收益: {total_return:.1f}%\n年化收益: {annual_return_actual:.1f}%\n最大回撤: {max_drawdown:.1f}%'
    
    ax.text(0.02, 0.97, stats_text, 
             transform=ax.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.85, edgecolor='black'),
             fontweight='bold')
    
    # 标注开始和结束点
    ax.scatter([dates[0], dates[-1]], [cumulative_return[0], cumulative_return[-1]], 
               color='blue', s=100, zorder=5, edgecolors='black', linewidth=1.5)
    
    ax.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax.set_ylabel('累计净值', fontsize=12, fontweight='bold')
    ax.set_title('PCA统计套利策略净值曲线 (2022-2024)', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=11, framealpha=0.9, edgecolor='black')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 格式化y轴
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: f'{x:.2f}x'))
    
    # 添加网格线
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("✓ 组合绩效图已保存")
    plt.close()

def generate_factor_loadings_heatmap():
    """生成因子载荷热力图（模拟数据）"""
    print("生成因子载荷热力图...")
    
    # 生成模拟的因子载荷矩阵
    np.random.seed(42)
    n_stocks = 50
    n_factors = 10
    
    # 创建模拟的因子载荷
    loadings = np.random.randn(n_stocks, n_factors) * 0.2
    
    # 添加一些结构性模式（让热力图更有意义）
    # 因子1：影响前15只股票（大盘股）
    loadings[0:15, 0] = np.abs(loadings[0:15, 0]) + 0.4
    
    # 因子2：影响15-35只股票（中盘股）
    loadings[15:35, 1] = np.abs(loadings[15:35, 1]) + 0.3
    
    # 因子3：影响35-50只股票（小盘股）
    loadings[35:50, 2] = np.abs(loadings[35:50, 2]) + 0.3
    
    # 因子4-5：随机但较强的载荷
    loadings[:, 3] = loadings[:, 3] + 0.2
    loadings[:, 4] = loadings[:, 4] - 0.2
    
    # 创建股票和行业标签
    stock_names = [f'Stock_{i+1:02d}' for i in range(n_stocks)]
    factor_names = [f'PC{i+1}' for i in range(n_factors)]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 10), facecolor='white')
    
    # 使用imshow绘制热力图
    im = ax.imshow(loadings.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(range(n_stocks))
    ax.set_xticklabels(stock_names, rotation=90, fontsize=8)
    ax.set_yticks(range(n_factors))
    ax.set_yticklabels(factor_names, fontsize=11, fontweight='bold')
    
    ax.set_title('PCA因子载荷矩阵热力图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('股票', fontsize=12, fontweight='bold')
    ax.set_ylabel('主成分（因子）', fontsize=12, fontweight='bold')
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('因子载荷', fontsize=11, fontweight='bold')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, n_stocks, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_factors, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 因子载荷热力图已保存")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("开始生成PCA统计套利文章配图（简化版）...")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('public/images/pca-statistical-arbitrage', exist_ok=True)
    
    # 生成所有配图
    generate_pca_variance_plot()
    generate_trading_signals_plot()
    generate_portfolio_performance_plot()
    generate_factor_loadings_heatmap()
    
    print("="*60)
    print("✓ 所有配图生成完成！")
    print(f"  输出目录: public/images/pca-statistical-arbitrage/")
    print("  生成文件:")
    print("    - pca_variance_explained.png")
    print("    - trading_signals.png")
    print("    - portfolio_performance.png")
    print("    - factor_loadings_heatmap.png")
    print("="*60)
