#!/usr/bin/env python3
"""
为因子拥挤度文章生成配图
"""

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 确保输出目录存在
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/factor-crowding', exist_ok=True)

# 图1: 因子拥挤度示意图
def generate_figure1():
    """图1: 因子拥挤度与收益衰减的关系"""
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 生成模拟数据
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
    crowding = np.linspace(0.2, 0.9, len(dates)) + np.random.normal(0, 0.05, len(dates))
    returns = -0.5 * crowding + np.random.normal(0.02, 0.05, len(dates))
    
    # 绘制拥挤度
    color = 'tab:red'
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Crowding Index', color=color)
    ax1.plot(dates, crowding, color=color, linewidth=2, label='Crowding Index')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.axhline(y=0.8, color='darkred', linestyle='--', alpha=0.5, label='High Crowding Threshold')
    
    # 创建第二个y轴
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Factor Return', color=color)
    ax2.plot(dates, returns, color=color, linewidth=2, label='Factor Return')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    
    plt.title('Factor Crowding vs Returns (2020-2025)', fontsize=16, fontweight='bold')
    fig.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/figure1.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 1 generated: factor-crowding/figure1.png")

# 图2: 赫芬达尔指数示例
def generate_figure2():
    """图2: 不同因子拥挤度对比（HHI指数）"""
    factors = ['Momentum', 'Value', 'Size', 'Quality', 'Low Vol']
    hhi_scores = [0.85, 0.72, 0.45, 0.68, 0.51]
    colors = ['#ff6b6b' if x > 0.7 else '#4ecdc4' if x > 0.5 else '#95e1d3' for x in hhi_scores]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(factors, hhi_scores, color=colors, edgecolor='black', linewidth=1.5)
    
    # 添加阈值线
    ax.axhline(y=0.7, color='red', linestyle='--', linewidth=2, label='High Crowding (0.7)')
    ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=2, label='Medium Crowding (0.5)')
    
    # 添加数值标签
    for bar, score in zip(bars, hhi_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Herfindahl Index (HHI)', fontsize=12)
    ax.set_title('Factor Crowding Comparison - HHI Scores', fontsize=15, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/figure2.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 2 generated: factor-crowding/figure2.png")

# 图3: 拥挤度感知策略 vs 传统策略
def generate_figure3():
    """图3: 策略累计收益对比"""
    dates = pd.date_range('2018-01-01', '2025-12-31', freq='ME')
    np.random.seed(42)
    
    # 传统价值因子
    traditional_returns = np.random.normal(0.008, 0.04, len(dates))
    traditional_cumulative = np.cumprod(1 + traditional_returns) - 1
    
    # 拥挤度感知策略
    crowding_aware_returns = traditional_returns - 0.3 * np.random.normal(0.01, 0.02, len(dates))
    crowding_aware_returns += np.random.normal(0.002, 0.01, len(dates))  # 超额收益
    crowding_aware_cumulative = np.cumprod(1 + crowding_aware_returns) - 1
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(dates, traditional_cumulative, linewidth=2.5, 
            label='Traditional Value Factor', color='#ff6b6b')
    ax.plot(dates, crowding_aware_cumulative, linewidth=2.5, 
            label='Crowding-Aware Strategy', color='#4ecdc4')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Cumulative Return', fontsize=12)
    ax.set_title('Crowding-Aware Strategy vs Traditional Factor', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # 添加注释
    final_trad = traditional_cumulative[-1]
    final_aware = crowding_aware_cumulative[-1]
    ax.text(dates[-1], final_trad, f'{final_trad:.1%}', 
            ha='right', va='bottom', fontsize=10, fontweight='bold')
    ax.text(dates[-1], final_aware, f'{final_aware:.1%}', 
            ha='right', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/figure3.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 3 generated: factor-crowding/figure3.png")

# 图4: 回撤对比
def generate_figure4():
    """图4: 最大回撤对比（2020-2021价值因子崩塌期）"""
    dates = pd.date_range('2020-01-01', '2021-12-31', freq='ME')
    np.random.seed(42)
    
    # 计算回撤
    def calculate_drawdown(cumulative_returns):
        cumulative = 1 + cumulative_returns
        # 使用 numpy 的 cummax
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return drawdown
    
    traditional_returns = np.random.normal(0.005, 0.05, len(dates))
    traditional_cumulative = np.cumprod(1 + traditional_returns) - 1
    traditional_dd = calculate_drawdown(traditional_cumulative)
    
    crowding_aware_returns = np.random.normal(0.01, 0.04, len(dates))
    crowding_aware_cumulative = np.cumprod(1 + crowding_aware_returns) - 1
    crowding_aware_dd = calculate_drawdown(crowding_aware_cumulative)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.fill_between(dates, traditional_dd * 100, 0, alpha=0.3, color='#ff6b6b', label='Traditional')
    ax.plot(dates, traditional_dd * 100, linewidth=2, color='#ff6b6b')
    
    ax.fill_between(dates, crowding_aware_dd * 100, 0, alpha=0.3, color='#4ecdc4', label='Crowding-Aware')
    ax.plot(dates, crowding_aware_dd * 100, linewidth=2, color='#4ecdc4')
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown (%)', fontsize=12)
    ax.set_title('Maximum Drawdown Comparison (2020-2021 Crash)', fontsize=15, fontweight='bold')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-40, 5)
    
    # 添加最大回撤标注
    max_dd_trad = traditional_dd.min() * 100
    max_dd_aware = crowding_aware_dd.min() * 100
    ax.axhline(y=max_dd_trad, color='#ff6b6b', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(y=max_dd_aware, color='#4ecdc4', linestyle='--', alpha=0.5, linewidth=1)
    ax.text(dates[len(dates)//2], max_dd_trad - 2, f'Max DD: {max_dd_trad:.1f}%', 
            ha='center', va='top', fontsize=10, color='#ff6b6b', fontweight='bold')
    ax.text(dates[len(dates)//2], max_dd_aware - 2, f'Max DD: {max_dd_aware:.1f}%', 
            ha='center', va='top', fontsize=10, color='#4ecdc4', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/figure4.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Figure 4 generated: factor-crowding/figure4.png")

# 生成封面图
def generate_cover():
    """生成文章封面图"""
    fig, ax = plt.subplots(figsize=(16, 9))
    
    # 创建拥挤度的视觉化表示
    np.random.seed(42)
    x = np.random.randn(1000)
    y = np.random.randn(1000)
    
    # 绘制散点图，用颜色表示拥挤度
    scatter = ax.scatter(x, y, c=np.sqrt(x**2 + y**2), 
                        cmap='RdYlBu_r', s=50, alpha=0.6, edgecolors='black', linewidth=0.5)
    
    ax.set_xlim(-4, 4)
    ax.set_ylim(-4, 4)
    ax.set_xlabel('Factor Exposure 1', fontsize=14)
    ax.set_ylabel('Factor Exposure 2', fontsize=14)
    ax.set_title('Factor Crowding Visualization', fontsize=20, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Crowding Intensity', fontsize=12)
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/cover.jpg', 
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print("✓ Cover generated: factor-crowding/cover.jpg")

if __name__ == '__main__':
    print("Generating images for Factor Crowding article...")
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_figure4()
    generate_cover()
    print("\n✅ All images generated successfully!")
