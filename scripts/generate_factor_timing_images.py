#!/usr/bin/env python3
"""
生成因子择时文章的配图
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.dates as mdates

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def generate_factor_regime_chart():
    """生成因子表现与市场状态的关系图"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # 模拟时间轴
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
    
    # 模拟市场状态
    np.random.seed(42)
    regimes = np.random.choice(['牛市', '震荡', '熊市'], size=len(dates), 
                              p=[0.3, 0.4, 0.3])
    
    # 不同状态下因子表现
    factor_returns = {
        '价值因子': {'牛市': 0.08, '震荡': 0.03, '熊市': -0.02},
        '动量因子': {'牛市': 0.10, '震荡': 0.01, '熊市': -0.05},
        '低波因子': {'牛市': 0.04, '震荡': 0.02, '熊市': 0.03},
        '质量因子': {'牛市': 0.06, '震荡': 0.04, '熊市': 0.02}
    }
    
    # 绘制第一个子图：市场状态
    colors = {'牛市': 'green', '震荡': 'gray', '熊市': 'red'}
    for i, regime in enumerate(regimes):
        ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color=colors[regime])
    
    ax1.set_ylabel('市场状态', fontsize=12)
    ax1.set_title('因子表现与市场状态关系', fontsize=14, fontweight='bold')
    ax1.set_xticks([])
    ax1.set_ylim(-0.5, 0.5)
    
    # 添加图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[r], alpha=0.3, label=r) 
                      for r in colors]
    ax1.legend(handles=legend_elements, loc='upper right')
    
    # 绘制第二个子图：因子收益
    x = np.arange(len(dates))
    width = 0.2
    
    for i, (factor, returns) in enumerate(factor_returns.items()):
        simulated_returns = np.array([
            returns[r] + np.random.normal(0, 0.02) for r in regimes
        ])
        ax2.plot(dates, simulated_returns, label=factor, linewidth=2, 
                alpha=0.7)
    
    ax2.set_ylabel('月收益率', fontsize=12)
    ax2.set_xlabel('时间', fontsize=12)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/factor_regime_analysis.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 生成因子状态分析图")
    plt.close()

def generate_timing_performance_chart():
    """生成因子择时策略绩效对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 模拟回测数据
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='ME')
    np.random.seed(42)
    
    # 静态因子投资（使用pandas Series以使用cummax）
    static_returns = pd.Series(np.random.normal(0.008, 0.04, len(dates)), index=dates)
    static_cumulative = (1 + static_returns).cumprod()
    
    # 动态因子择时
    timing_returns = static_returns.copy()
    timing_returns.iloc[::3] += 0.02  # 择时增强
    timing_returns.iloc[::5] -= 0.01   # 风险控制
    timing_cumulative = (1 + timing_returns).cumprod()
    
    # 绘制累计收益
    ax1.plot(dates, static_cumulative, label='静态因子投资', 
            linewidth=2, color='blue', alpha=0.6)
    ax1.plot(dates, timing_cumulative, label='动态因子择时', 
            linewidth=2, color='red', alpha=0.8)
    
    ax1.set_title('Cumulative Returns Comparison', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Return (x)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # 绘制回撤对比
    static_dd = 1 - static_cumulative / static_cumulative.cummax()
    timing_dd = 1 - timing_cumulative / timing_cumulative.cummax()
    
    # 转换为百分比
    static_dd = static_dd * 100
    timing_dd = timing_dd * 100
    
    ax2.fill_between(dates, -static_dd * 100, 0, 
                     alpha=0.3, color='blue', label='静态因子投资')
    ax2.fill_between(dates, -timing_dd * 100, 0, 
                     alpha=0.3, color='red', label='动态因子择时')
    
    ax2.set_title('Drawdown Comparison', fontsize=14, fontweight='bold')
    ax2.set_xlabel('时间', fontsize=12)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/timing_performance_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 生成择时绩效对比图")
    plt.close()

def generate_factor_exposure_chart():
    """生成动态因子暴露调整图"""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    dates = pd.date_range('2023-01-01', '2025-12-31', freq='ME')
    np.random.seed(42)
    
    # 模拟动态因子暴露
    base_exposure = 1.0
    timing_signals = np.random.uniform(0.3, 0.8, len(dates))
    adjusted_exposure = base_exposure + (timing_signals - 0.5) * 2
    
    # 绘制因子暴露
    ax.plot(dates, adjusted_exposure, linewidth=2, 
            color='purple', label='动态因子暴露', alpha=0.8)
    ax.axhline(y=1.0, color='black', linestyle='--', 
              label='基准暴露 (1.0)', alpha=0.5)
    
    # 标注择时区域
    ax.fill_between(dates, 0.5, 1.5, 
                    where=(adjusted_exposure > 1.2),
                    alpha=0.2, color='green', label='增持区域')
    ax.fill_between(dates, 0.5, 1.5, 
                    where=(adjusted_exposure < 0.8),
                    alpha=0.2, color='red', label='减持区域')
    
    ax.set_title('Dynamic Factor Exposure Adjustment', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Factor Exposure Multiple', fontsize=12)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.3, 1.7)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-timing/dynamic_factor_exposure.png', 
                dpi=300, bbox_inches='tight')
    print("✓ 生成动态因子暴露图")
    plt.close()

if __name__ == '__main__':
    print("开始生成因子择时文章配图...")
    generate_factor_regime_chart()
    generate_timing_performance_chart()
    generate_factor_exposure_chart()
    print("✅ 所有配图生成完成！")
