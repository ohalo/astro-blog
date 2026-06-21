#!/usr/bin/env python3
"""
生成PCA统计套利文章的配图
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_pca_variance_plot():
    """生成PCA解释方差图"""
    print("生成PCA解释方差图...")
    
    # 使用示例数据
    np.random.seed(42)
    n_samples = 500
    n_features = 50
    
    # 生成模拟收益率数据
    X = np.random.randn(n_samples, n_features)
    
    # 添加一些结构性因子
    factor1 = np.random.randn(n_samples) * 2  # 主要因子
    factor2 = np.random.randn(n_samples) * 1.5  # 次要因子
    factor3 = np.random.randn(n_samples) * 1  # 第三因子
    
    X[:, 0:15] += factor1[:, np.newaxis] * 1.5  # 前15只股票受因子1影响
    X[:, 15:35] += factor2[:, np.newaxis] * 1.2  # 中间20只股票受因子2影响
    X[:, 35:50] += factor3[:, np.newaxis] * 0.8  # 后15只股票受因子3影响
    
    # 执行PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=20)
    pca.fit(X_scaled)
    
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
    
    # 绘制图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 单个主成分解释方差
    axes[0].bar(range(1, 21), explained_variance_ratio, alpha=0.7, color='steelblue')
    axes[0].set_xlabel('主成分序号', fontsize=12)
    axes[0].set_ylabel('解释方差比例', fontsize=12)
    axes[0].set_title('各主成分解释方差比例', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xticks(range(1, 21))
    
    # 在柱状图上添加数值标签
    for i, v in enumerate(explained_variance_ratio):
        if i < 10:  # 只标注前10个
            axes[0].text(i+1, v + 0.005, f'{v:.1%}', ha='center', va='bottom', fontsize=8)
    
    # 累计解释方差
    axes[1].plot(range(1, 21), cumulative_variance_ratio, marker='o', 
                 color='darkred', linewidth=2.5, markersize=6)
    axes[1].axhline(y=0.8, color='gray', linestyle='--', linewidth=1.5, label='80%阈值')
    axes[1].axhline(y=0.9, color='orange', linestyle='--', linewidth=1.5, label='90%阈值')
    axes[1].fill_between(range(1, 21), 0, cumulative_variance_ratio, alpha=0.2, color='darkred')
    axes[1].set_xlabel('主成分序号', fontsize=12)
    axes[1].set_ylabel('累计解释方差比例', fontsize=12)
    axes[1].set_title('累计解释方差', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xticks(range(1, 21))
    axes[1].set_ylim([0, 1])
    
    # 在曲线上添加关键点的数值
    for i in [4, 9, 14, 19]:
        axes[1].annotate(f'{cumulative_variance_ratio[i]:.1%}', 
                         xy=(i+1, cumulative_variance_ratio[i]),
                         xytext=(5, 5), textcoords='offset points',
                         fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/pca_variance_explained.png', dpi=300, bbox_inches='tight')
    print("✓ PCA解释方差图已保存：public/images/pca-statistical-arbitrage/pca_variance_explained.png")
    plt.close()

def generate_trading_signals_plot():
    """生成交易信号示意图"""
    print("生成交易信号示意图...")
    
    # 生成模拟残差数据
    np.random.seed(42)
    n_days = 250
    
    # 创建均值回归的残差序列
    residuals = np.zeros(n_days)
    residuals[0] = 0
    
    for i in range(1, n_days):
        # 均值回归过程： Ornstein-Uhlenbeck process
        mean_reversion = -0.1 * residuals[i-1]  # 均值回归项
        noise = np.random.randn() * 0.5  # 噪声
        residuals[i] = residuals[i-1] + mean_reversion + noise
    
    # 标准化
    residuals = (residuals - residuals.mean()) / residuals.std()
    
    # 生成交易信号（Z-Score超过±2）
    z_score = (residuals - residuals.mean()) / residuals.std()
    signals = np.zeros(n_days)
    signals[z_score < -2.0] = 1  # 买入信号
    signals[z_score > 2.0] = -1   # 卖出信号
    
    # 创建日期索引
    dates = pd.date_range(start='2023-01-01', periods=n_days, freq='B')
    
    # 绘制图形
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    
    # 上图：残差序列
    axes[0].plot(dates, residuals, color='steelblue', linewidth=1.5, label='残差')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
    axes[0].fill_between(dates, 0, residuals, where=(residuals > 0), 
                          alpha=0.2, color='red', label='正值区域')
    axes[0].fill_between(dates, 0, residuals, where=(residuals < 0), 
                          alpha=0.2, color='green', label='负值区域')
    axes[0].set_ylabel('残差值', fontsize=12)
    axes[0].set_title('残差序列与均值回归特性', fontsize=14, fontweight='bold')
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].grid(True, alpha=0.3, linestyle='--')
    
    # 下图：交易信号
    axes[1].plot(dates, residuals, color='gray', linewidth=0.8, alpha=0.5, label='残差')
    axes[1].axhline(y=2.0, color='red', linestyle='--', linewidth=1.2, label='卖出阈值 (+2σ)')
    axes[1].axhline(y=-2.0, color='green', linestyle='--', linewidth=1.2, label='买入阈值 (-2σ)')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
    
    # 标记买入信号
    buy_signals = np.where(signals == 1)[0]
    if len(buy_signals) > 0:
        axes[1].scatter(dates[buy_signals], residuals[buy_signals], 
                        color='red', marker='^', s=150, 
                        label='买入信号', zorder=5, edgecolors='black', linewidth=1.5)
    
    # 标记卖出信号
    sell_signals = np.where(signals == -1)[0]
    if len(sell_signals) > 0:
        axes[1].scatter(dates[sell_signals], residuals[sell_signals], 
                        color='green', marker='v', s=150, 
                        label='卖出信号', zorder=5, edgecolors='black', linewidth=1.5)
    
    axes[1].set_xlabel('日期', fontsize=12)
    axes[1].set_ylabel('残差值', fontsize=12)
    axes[1].set_title('基于Z-Score的交易信号生成', fontsize=14, fontweight='bold')
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].grid(True, alpha=0.3, linestyle='--')
    
    # 格式化x轴日期
    axes[1].xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    axes[1].xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=2))
    plt.setp(axes[1].xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/trading_signals.png', dpi=300, bbox_inches='tight')
    print("✓ 交易信号图已保存：public/images/pca-statistical-arbitrage/trading_signals.png")
    plt.close()

def generate_factor_loadings_heatmap():
    """生成因子载荷热力图"""
    print("生成因子载荷热力图...")
    
    # 生成模拟数据
    np.random.seed(42)
    n_stocks = 50
    n_factors = 10
    
    # 创建模拟的因子载荷矩阵
    loadings = np.random.randn(n_stocks, n_factors) * 0.3
    
    # 添加一些结构性模式
    loadings[0:15, 0] = np.abs(loadings[0:15, 0]) + 0.5  # 因子1对前15只股票影响大
    loadings[15:35, 1] = np.abs(loadings[15:35, 1]) + 0.4  # 因子2对中间股票影响大
    loadings[35:50, 2] = np.abs(loadings[35:50, 2]) + 0.3  # 因子3对后15只股票影响大
    
    # 创建股票名称
    stock_names = [f'Stock_{i+1:02d}' for i in range(n_stocks)]
    factor_names = [f'PC{i+1}' for i in range(n_factors)]
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(loadings, 
                xticklabels=factor_names, 
                yticklabels=stock_names,
                cmap='RdBu_r', 
                center=0, 
                annot=False, 
                fmt='.2f',
                cbar_kws={'label': '因子载荷'})
    
    ax.set_title('PCA因子载荷矩阵热力图', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('主成分（因子）', fontsize=12)
    ax.set_ylabel('股票', fontsize=12)
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/factor_loadings_heatmap.png', dpi=300, bbox_inches='tight')
    print("✓ 因子载荷热力图已保存：public/images/pca-statistical-arbitrage/factor_loadings_heatmap.png")
    plt.close()

def generate_portfolio_performance_plot():
    """生成组合绩效曲线图"""
    print("生成组合绩效曲线图...")
    
    # 生成模拟的组合净值数据
    np.random.seed(42)
    n_days = 500
    
    # 模拟日收益率（年化10%，波动率8%）
    daily_return = 0.10 / 252
    daily_vol = 0.08 / np.sqrt(252)
    
    returns = np.random.normal(daily_return, daily_vol, n_days)
    
    # 计算累计净值
    cumulative_return = np.cumprod(1 + returns)
    
    # 计算基准（市场，年化7%）
    benchmark_return = 0.07 / 252
    benchmark_returns = np.random.normal(benchmark_return, daily_vol * 1.2, n_days)
    benchmark_cumulative = np.cumprod(1 + benchmark_returns)
    
    # 创建日期索引
    dates = pd.date_range(start='2022-01-01', periods=n_days, freq='B')
    
    # 绘制图形
    fig, ax = plt.subplots(figsize=(14, 8))
    
    ax.plot(dates, cumulative_return, color='steelblue', linewidth=2.5, label='PCA统计套利策略', zorder=3)
    ax.plot(dates, benchmark_cumulative, color='gray', linewidth=2, linestyle='--', 
            label='市场基准', alpha=0.7, zorder=2)
    
    # 填充区域
    ax.fill_between(dates, 1, cumulative_return, where=(cumulative_return >= 1), 
                     alpha=0.2, color='green', zorder=1)
    ax.fill_between(dates, 1, cumulative_return, where=(cumulative_return < 1), 
                     alpha=0.2, color='red', zorder=1)
    
    # 标注关键指标
    total_return = (cumulative_return[-1] - 1) * 100
    annual_return = ((cumulative_return[-1]) ** (252/n_days) - 1) * 100
    
    ax.text(0.02, 0.98, f'总收益: {total_return:.1f}%\n年化收益: {annual_return:.1f}%', 
             transform=ax.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.set_xlabel('日期', fontsize=12)
    ax.set_ylabel('累计净值', fontsize=12)
    ax.set_title('PCA统计套利策略净值曲线', fontsize=16, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 格式化y轴
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: f'{x:.2f}x'))
    
    plt.tight_layout()
    plt.savefig('public/images/pca-statistical-arbitrage/portfolio_performance.png', dpi=300, bbox_inches='tight')
    print("✓ 组合绩效图已保存：public/images/pca-statistical-arbitrage/portfolio_performance.png")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("开始生成PCA统计套利文章配图...")
    print("="*60)
    
    # 创建输出目录
    import os
    os.makedirs('public/images/pca-statistical-arbitrage', exist_ok=True)
    
    # 生成所有配图
    generate_pca_variance_plot()
    generate_trading_signals_plot()
    generate_factor_loadings_heatmap()
    generate_portfolio_performance_plot()
    
    print("="*60)
    print("✓ 所有配图生成完成！")
    print("="*60)
