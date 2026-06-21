#!/usr/bin/env python3
"""
生成XGBoost与LightGBM文章配图（简化版 - 无需sklearn）
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

def generate_feature_importance_plot():
    """生成特征重要性对比图"""
    print("生成特征重要性图...")
    
    # 模拟特征重要性数据
    np.random.seed(42)
    n_features = 20
    
    # 创建模拟的特征名称（量化因子）
    feature_names = [
        'momentum_20', 'volatility_20', 'rsi_14', 'macd', 
        'pe_ratio', 'pb_ratio', 'roe', 'turn_over',
        'volume_change', 'momentum_5', 'momentum_60',
        'bollinger_position', 'ma_distance', 'volume_ratio',
        'profit_growth', 'revenue_growth', 'debt_ratio',
        'market_cap', 'day_of_week', 'month'
    ]
    
    # 模拟LightGBM特征重要性（前几个特征更重要）
    lgb_importance = np.zeros(n_features)
    lgb_importance[:5] = np.random.uniform(0.08, 0.15, 5)  # 前5个重要
    lgb_importance[5:10] = np.random.uniform(0.03, 0.07, 5)  # 中间10个
    lgb_importance[10:] = np.random.uniform(0.001, 0.03, 10)  # 后面不太重要
    lgb_importance = lgb_importance / lgb_importance.sum()  # 归一化
    
    # 模拟XGBoost特征重要性（类似但略有不同）
    xgb_importance = lgb_importance + np.random.uniform(-0.01, 0.01, n_features)
    xgb_importance = np.abs(xgb_importance) / xgb_importance.sum()
    
    # 按重要性排序
    indices = np.argsort(lgb_importance)[::-1]
    
    # 绘制图形
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 左图：LightGBM特征重要性
    colors_lgb = plt.cm.viridis(lgb_importance[indices] / lgb_importance[indices].max())
    bars1 = axes[0].barh(range(n_features), lgb_importance[indices][::-1], 
                           color=colors_lgb[::-1], edgecolor='black', linewidth=0.5)
    axes[0].set_yticks(range(n_features))
    axes[0].set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    axes[0].set_xlabel('重要性', fontsize=11, fontweight='bold')
    axes[0].set_title('LightGBM特征重要性', fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    axes[0].set_xlim([0, lgb_importance[indices].max() * 1.2])
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars1, lgb_importance[indices][::-1])):
        axes[0].text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                      f'{val:.3f}', va='center', fontsize=7)
    
    # 右图：XGBoost特征重要性
    colors_xgb = plt.cm.plasma(xgb_importance[indices] / xgb_importance[indices].max())
    bars2 = axes[1].barh(range(n_features), xgb_importance[indices][::-1], 
                           color=colors_xgb[::-1], edgecolor='black', linewidth=0.5)
    axes[1].set_yticks(range(n_features))
    axes[1].set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=9)
    axes[1].set_xlabel('重要性', fontsize=11, fontweight='bold')
    axes[1].set_title('XGBoost特征重要性', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].set_xlim([0, xgb_importance[indices].max() * 1.2])
    
    # 添加数值标签
    for i, (bar, val) in enumerate(zip(bars2, xgb_importance[indices][::-1])):
        axes[1].text(val + 0.002, bar.get_y() + bar.get_height()/2, 
                      f'{val:.3f}', va='center', fontsize=7)
    
    plt.tight_layout()
    plt.savefig('public/images/xgboost-lightgbm-stock-selection/feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ 特征重要性图已保存")
    plt.close()

def generate_model_comparison_plot():
    """生成模型性能对比图"""
    print("生成模型性能对比图...")
    
    # 模拟不同模型的性能数据
    models = ['Logistic\n回归', '随机森林', 'SVM', 'XGBoost', 'LightGBM']
    accuracy = [0.352, 0.401, 0.385, 0.418, 0.423]
    f1_score = [0.341, 0.395, 0.372, 0.408, 0.412]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    bars1 = ax.bar(x - width/2, accuracy, width, label='准确率 (Accuracy)', 
                    color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, f1_score, width, label='F1分数', 
                    color='darkorange', alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax.set_ylabel('分数', fontsize=12, fontweight='bold')
    ax.set_title('不同模型在量化选股任务上的性能对比', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=10)
    ax.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim([0, 0.5])
    
    # 添加数值标签
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 0.005,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # 标注最佳模型
    ax.axhline(y=max(accuracy), color='red', linestyle='--', linewidth=1.5, 
                alpha=0.6, label='最佳性能')
    
    plt.tight_layout()
    plt.savefig('public/images/xgboost-lightgbm-stock-selection/model_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ 模型性能对比图已保存")
    plt.close()

def generate_backtest_performance_plot():
    """生成策略回测净值曲线图"""
    print("生成策略回测净值曲线图...")
    
    # 生成模拟的净值数据
    np.random.seed(42)
    n_days = 1000
    
    # LightGBM策略（年化18%，波动率20%）
    strategy_return = 0.18 / 252
    strategy_vol = 0.20 / np.sqrt(252)
    strategy_daily_returns = np.random.normal(strategy_return, strategy_vol, n_days)
    strategy_nav = np.cumprod(1 + strategy_daily_returns)
    
    # 基准（沪深300，年化8%，波动率22%）
    benchmark_return = 0.08 / 252
    benchmark_vol = 0.22 / np.sqrt(252)
    benchmark_daily_returns = np.random.normal(benchmark_return, benchmark_vol, n_days)
    benchmark_nav = np.cumprod(1 + benchmark_daily_returns)
    
    # 创建日期索引
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='B')
    
    # 绘制图形
    fig, ax = plt.subplots(figsize=(14, 8), facecolor='white')
    
    # 绘制净值曲线
    ax.plot(dates, strategy_nav, color='#1f77b4', linewidth=2.5, 
            label='LightGBM选股策略', zorder=4, alpha=0.9)
    ax.plot(dates, benchmark_nav, color='gray', linewidth=2, linestyle='--', 
            label='基准 (沪深300)', alpha=0.7, zorder=3)
    
    # 填充区域
    ax.fill_between(dates, 1, strategy_nav, where=(strategy_nav >= 1), 
                     alpha=0.2, color='green', zorder=1)
    ax.fill_between(dates, 1, strategy_nav, where=(strategy_nav < 1), 
                     alpha=0.2, color='red', zorder=1)
    
    # 计算并标注关键指标
    strategy_total_return = (strategy_nav[-1] - 1) * 100
    benchmark_total_return = (benchmark_nav[-1] - 1) * 100
    excess_return = strategy_total_return - benchmark_total_return
    
    # 计算最大回撤
    strategy_dd = 1 - strategy_nav / np.maximum.accumulate(strategy_nav)
    max_dd = strategy_dd.max() * 100
    
    stats_text = (f'策略总收益: {strategy_total_return:.1f}%\n'
                  f'基准总收益: {benchmark_total_return:.1f}%\n'
                  f'超额收益: {excess_return:.1f}%\n'
                  f'最大回撤: {max_dd:.1f}%')
    
    ax.text(0.02, 0.97, stats_text, 
             transform=ax.transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.85, edgecolor='black'),
             fontweight='bold')
    
    # 标注关键时间点
    ax.scatter([dates[0], dates[-1]], [strategy_nav[0], strategy_nav[-1]], 
               color='blue', s=100, zorder=5, edgecolors='black', linewidth=1.5, label='策略起点/终点')
    ax.scatter([dates[0], dates[-1]], [benchmark_nav[0], benchmark_nav[-1]], 
               color='gray', s=100, zorder=5, edgecolors='black', linewidth=1.5, label='基准起点/终点')
    
    ax.set_xlabel('日期', fontsize=12, fontweight='bold')
    ax.set_ylabel('累计净值', fontsize=12, fontweight='bold')
    ax.set_title('LightGBM选股策略 vs 基准 - 回测净值曲线 (2020-2024)', 
                  fontsize=15, fontweight='bold', pad=15)
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9, edgecolor='black', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # 格式化y轴
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.FuncFormatter(lambda x, p: f'{x:.2f}x'))
    
    plt.tight_layout()
    plt.savefig('public/images/xgboost-lightgbm-stock-selection/backtest_performance.png', dpi=300, bbox_inches='tight')
    print("✓ 策略回测净值曲线图已保存")
    plt.close()

def generate_learning_curve_plot():
    """生成学习曲线图（展示训练集大小对性能的影响）"""
    print("生成学习曲线图...")
    
    # 模拟不同训练集大小下的模型性能
    train_sizes = np.array([100, 200, 500, 1000, 2000, 5000])
    
    # LightGBM：随数据量增加，性能提升
    lgb_train_score = np.array([0.38, 0.39, 0.40, 0.415, 0.422, 0.425])
    lgb_cv_score = np.array([0.35, 0.37, 0.39, 0.408, 0.415, 0.418])
    
    # XGBoost：类似但略低
    xgb_train_score = np.array([0.37, 0.38, 0.395, 0.410, 0.418, 0.420])
    xgb_cv_score = np.array([0.34, 0.36, 0.38, 0.402, 0.410, 0.412])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：LightGBM学习曲线
    axes[0].plot(train_sizes, lgb_train_score, 'o-', color='blue', linewidth=2, 
                  markersize=6, label='训练集分数')
    axes[0].plot(train_sizes, lgb_cv_score, 'o-', color='red', linewidth=2, 
                  markersize=6, label='交叉验证分数')
    axes[0].fill_between(train_sizes, 
                          lgb_cv_score - 0.01, 
                          lgb_cv_score + 0.01, 
                          alpha=0.2, color='red')
    axes[0].set_xlabel('训练样本数量', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('准确率', fontsize=11, fontweight='bold')
    axes[0].set_title('LightGBM学习曲线', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=10, loc='lower right')
    axes[0].grid(True, alpha=0.3, linestyle='--')
    axes[0].set_xscale('log')
    
    # 右图：XGBoost学习曲线
    axes[1].plot(train_sizes, xgb_train_score, 'o-', color='blue', linewidth=2, 
                  markersize=6, label='训练集分数')
    axes[1].plot(train_sizes, xgb_cv_score, 'o-', color='red', linewidth=2, 
                  markersize=6, label='交叉验证分数')
    axes[1].fill_between(train_sizes, 
                          xgb_cv_score - 0.01, 
                          xgb_cv_score + 0.01, 
                          alpha=0.2, color='red')
    axes[1].set_xlabel('训练样本数量', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('准确率', fontsize=11, fontweight='bold')
    axes[1].set_title('XGBoost学习曲线', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=10, loc='lower right')
    axes[1].grid(True, alpha=0.3, linestyle='--')
    axes[1].set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('public/images/xgboost-lightgbm-stock-selection/learning_curve.png', dpi=300, bbox_inches='tight')
    print("✓ 学习曲线图已保存")
    plt.close()

def generate_feature_correlation_heatmap():
    """生成特征相关性热力图"""
    print("生成特征相关性热力图...")
    
    # 模拟特征相关性矩阵
    np.random.seed(42)
    n_features = 10
    
    # 创建相关性矩阵（对称、对角为1）
    corr_matrix = np.eye(n_features)
    
    # 添加一些相关性结构
    for i in range(n_features):
        for j in range(i+1, n_features):
            if i < 3 and j < 3:  # 前3个特征高度相关（都是动量类）
                corr = np.random.uniform(0.6, 0.8)
            elif i >= 3 and j < 6:  # 动量类和价值类低相关
                corr = np.random.uniform(-0.2, 0.2)
            else:  # 其他中等相关
                corr = np.random.uniform(-0.4, 0.4)
            
            corr_matrix[i, j] = corr
            corr_matrix[j, i] = corr  # 对称
    
    # 特征名称
    feature_names = ['momentum_5', 'momentum_20', 'momentum_60', 
                     'pe_ratio', 'pb_ratio', 'roe',
                     'rsi_14', 'macd', 'volatility_20', 'volume_change']
    
    # 绘制热力图
    fig, ax = plt.subplots(figsize=(10, 8), facecolor='white')
    
    im = ax.imshow(corr_matrix, aspect='auto', cmap='RdBu_r', vmin=-1, vmax=1, interpolation='nearest')
    
    # 设置刻度
    ax.set_xticks(range(n_features))
    ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=9)
    ax.set_yticks(range(n_features))
    ax.set_yticklabels(feature_names, fontsize=9)
    
    ax.set_title('量化选股特征相关性矩阵', fontsize=14, fontweight='bold', pad=20)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label('相关系数', fontsize=11, fontweight='bold')
    
    # 在格子中添加数值
    for i in range(n_features):
        for j in range(n_features):
            text = ax.text(j, i, f'{corr_matrix[i, j]:.2f}',
                          ha='center', va='center', 
                          color='black' if abs(corr_matrix[i, j]) < 0.5 else 'white',
                          fontsize=8, fontweight='bold')
    
    # 添加网格线
    ax.set_xticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_features, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('public/images/xgboost-lightgbm-stock-selection/feature_correlation.png', dpi=300, bbox_inches='tight')
    print("✓ 特征相关性热力图已保存")
    plt.close()

if __name__ == "__main__":
    print("="*60)
    print("开始生成XGBoost/LightGBM文章配图（简化版）...")
    print("="*60)
    
    # 创建输出目录
    os.makedirs('public/images/xgboost-lightgbm-stock-selection', exist_ok=True)
    
    # 生成所有配图
    generate_feature_importance_plot()
    generate_model_comparison_plot()
    generate_backtest_performance_plot()
    generate_learning_curve_plot()
    generate_feature_correlation_heatmap()
    
    print("="*60)
    print("✓ 所有配图生成完成！")
    print(f"  输出目录: public/images/xgboost-lightgbm-stock-selection/")
    print("  生成文件:")
    print("    - feature_importance.png")
    print("    - model_comparison.png")
    print("    - backtest_performance.png")
    print("    - learning_curve.png")
    print("    - feature_correlation.png")
    print("="*60)
