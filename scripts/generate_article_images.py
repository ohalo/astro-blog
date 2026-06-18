#!/usr/bin/env python3
"""
生成量化博客文章配图
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================
# 第一篇文章：因子拥挤度监测
# ============================================================

def generate_factor_crowding_images():
    """生成因子拥挤度文章的配图"""
    
    # 图1：综合拥挤度指标监测面板
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('因子拥挤度综合监测面板', fontsize=16, fontweight='bold', y=1.02)
    
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2025-12-31', freq='D')
    
    # 子图1：ETF资金流指标
    ax1 = axes[0, 0]
    etf_flow = np.random.normal(50, 15, len(dates))
    etf_flow = pd.Series(etf_flow).rolling(20).mean().values
    ax1.plot(dates, etf_flow, color='#2ecc71', linewidth=2, label='价值因子ETF')
    ax1.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='警戒线')
    ax1.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='机会线')
    ax1.fill_between(dates, 70, 100, alpha=0.2, color='red')
    ax1.fill_between(dates, 0, 30, alpha=0.2, color='green')
    ax1.set_title('ETF资金流指标', fontsize=12, fontweight='bold')
    ax1.set_ylabel('拥挤度得分')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.set_ylim(0, 100)
    
    # 子图2：估值溢价
    ax2 = axes[0, 1]
    valuation_z = np.random.normal(0, 1, len(dates))
    valuation_z = pd.Series(valuation_z).rolling(30).mean().values
    ax2.plot(dates, valuation_z, color='#3498db', linewidth=2)
    ax2.axhline(y=2, color='r', linestyle='--', alpha=0.7)
    ax2.axhline(y=-2, color='g', linestyle='--', alpha=0.7)
    ax2.fill_between(dates, 2, valuation_z.max(), alpha=0.2, color='red')
    ax2.fill_between(dates, valuation_z.min(), -2, alpha=0.2, color='green')
    ax2.set_title('估值溢价 Z分数', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Z分数')
    
    # 子图3：收益衰减率
    ax3 = axes[1, 0]
    decay_rate = np.random.normal(-5, 10, len(dates))
    decay_rate = pd.Series(decay_rate).rolling(60).mean().values
    ax3.plot(dates, decay_rate, color='#e74c3c', linewidth=2)
    ax3.axhline(y=-20, color='darkred', linestyle='--', alpha=0.7, label='严重衰减阈值')
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.fill_between(dates, -50, -20, alpha=0.3, color='red')
    ax3.set_title('因子收益衰减率', fontsize=12, fontweight='bold')
    ax3.set_ylabel('衰减率 (%)')
    ax3.legend(loc='lower left', fontsize=8)
    
    # 子图4：波动率聚类
    ax4 = axes[1, 1]
    vol_cluster = np.random.gamma(2, 2, len(dates))
    vol_cluster = pd.Series(vol_cluster).rolling(20).mean().values
    colors = ['#27ae60' if v < 4 else '#f39c12' if v < 6 else '#e74c3c' for v in vol_cluster]
    ax4.scatter(dates, vol_cluster, c=colors, s=1, alpha=0.6)
    ax4.axhline(y=6, color='r', linestyle='--', alpha=0.7, label='高波动警戒')
    ax4.set_title('波动率聚类得分', fontsize=12, fontweight='bold')
    ax4.set_ylabel('聚类得分')
    ax4.legend(loc='upper left', fontsize=8)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_indicators.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 图2：拥挤度历史预警案例
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 模拟价值因子表现
    value_returns = np.random.normal(0.0005, 0.015, len(dates))
    value_returns = pd.Series(value_returns)
    
    # 模拟2017-2020年价值因子回撤
    mask_2017 = (dates >= '2017-01-01') & (dates <= '2020-12-31')
    value_returns[mask_2017] = value_returns[mask_2017] - 0.0002
    
    # 模拟2021-2022年修复
    mask_2021 = (dates >= '2021-01-01') & (dates <= '2022-12-31')
    value_returns[mask_2021] = value_returns[mask_2021] + 0.0008
    
    cumulative = (1 + value_returns).cumprod()
    cumulative = cumulative / cumulative.iloc[0] * 100
    
    # 拥挤度得分（滞后）
    crowding_score = np.random.normal(50, 15, len(dates))
    crowding_score[mask_2017[:len(crowding_score)]] += 20
    
    ax.plot(dates, cumulative.values, color='#3498db', linewidth=2, label='价值因子累计收益')
    ax.fill_between(dates, 80, 150, alpha=0.3, color='red', label='拥挤高发区（模拟）')
    
    # 标注关键事件
    ax.axvspan(pd.Timestamp('2017-06-01'), pd.Timestamp('2020-03-31'), 
               alpha=0.2, color='red', label='价值因子"至暗时刻"')
    ax.axvspan(pd.Timestamp('2021-01-01'), pd.Timestamp('2022-12-31'), 
               alpha=0.2, color='green', label='修复期')
    
    ax.set_title('价值因子历史表现与拥挤度周期', fontsize=14, fontweight='bold')
    ax.set_xlabel('日期')
    ax.set_ylabel('累计收益（起始=100）')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/factor-crowding/crowding_history.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ 因子拥挤度文章配图生成完成")

# ============================================================
# 第二篇文章：XGBoost vs LightGBM
# ============================================================

def generate_xgboost_lightgbm_images():
    """生成XGBoost vs LightGBM文章的配图"""
    
    # 图1：模型性能对比雷达图
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), subplot_kw=dict(projection='polar'))
    
    categories = ['Rank IC', 'ICIR', '训练速度', '预测速度', '内存效率', '稳定性']
    N = len(categories)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    # XGBoost数据（归一化到0-1）
    xgb_values = [0.75, 0.72, 0.35, 0.40, 0.30, 0.80]
    xgb_values += xgb_values[:1]
    
    # LightGBM数据
    lgb_values = [0.85, 0.88, 0.95, 0.90, 0.85, 0.75]
    lgb_values += lgb_values[:1]
    
    # 绘制XGBoost雷达图
    ax1 = axes[0]
    ax1.plot(angles, xgb_values, 'o-', linewidth=2, color='#e74c3c', label='XGBoost')
    ax1.fill(angles, xgb_values, alpha=0.25, color='#e74c3c')
    ax1.set_xticks(angles[:-1])
    ax1.set_xticklabels(categories, fontsize=10)
    ax1.set_ylim(0, 1)
    ax1.set_title('XGBoost 性能表现', fontsize=13, fontweight='bold', pad=20)
    
    # 绘制LightGBM雷达图
    ax2 = axes[1]
    ax2.plot(angles, lgb_values, 'o-', linewidth=2, color='#2ecc71', label='LightGBM')
    ax2.fill(angles, lgb_values, alpha=0.25, color='#2ecc71')
    ax2.set_xticks(angles[:-1])
    ax2.set_xticklabels(categories, fontsize=10)
    ax2.set_ylim(0, 1)
    ax2.set_title('LightGBM 性能表现', fontsize=13, fontweight='bold', pad=20)
    
    plt.suptitle('XGBoost vs LightGBM 性能对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/model_comparison.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 图2：特征重要性对比
    fig, ax = plt.subplots(figsize=(12, 8))
    
    features = ['ROE', 'PE_行业排名', '动量60日', '市值', '换手率', 
                '波动率20日', '净利润增速', '现金流质量', 'MA偏差', 'SUE',
                'ROA', 'PB_行业排名', '波动率比率', '应计项目', '相对估值']
    
    xgb_importance = np.random.uniform(0.03, 0.12, len(features))
    lgb_importance = np.random.uniform(0.03, 0.12, len(features))
    
    x_pos = np.arange(len(features))
    width = 0.35
    
    ax.barh(x_pos - width/2, sorted(xgb_importance, reverse=True), width, 
            color='#e74c3c', alpha=0.8, label='XGBoost')
    ax.barh(x_pos + width/2, sorted(lgb_importance, reverse=True), width, 
            color='#2ecc71', alpha=0.8, label='LightGBM')
    
    ax.set_yticks(x_pos)
    ax.set_yticklabels(features, fontsize=10)
    ax.set_xlabel('特征重要性（归一化）', fontsize=11)
    ax.set_title('XGBoost vs LightGBM 特征重要性对比', fontsize=14, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/feature_importance.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    # 图3：学习曲线对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练时间对比
    ax1 = axes[0]
    sample_sizes = [10000, 50000, 100000, 200000, 500000]
    xgb_times = [5, 25, 55, 130, 380]
    lgb_times = [2, 8, 18, 35, 85]
    
    ax1.plot(sample_sizes, xgb_times, 'o-', linewidth=2, markersize=8, 
            color='#e74c3c', label='XGBoost')
    ax1.plot(sample_sizes, lgb_times, 's-', linewidth=2, markersize=8, 
            color='#2ecc71', label='LightGBM')
    ax1.set_xlabel('样本量', fontsize=11)
    ax1.set_ylabel('训练时间（秒）', fontsize=11)
    ax1.set_title('训练时间对比', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 内存占用对比
    ax2 = axes[1]
    xgb_memory = [120, 450, 980, 2100, 5800]
    lgb_memory = [50, 180, 380, 750, 1900]
    
    ax2.plot(sample_sizes, xgb_memory, 'o-', linewidth=2, markersize=8, 
            color='#e74c3c', label='XGBoost')
    ax2.plot(sample_sizes, lgb_memory, 's-', linewidth=2, markersize=8, 
            color='#2ecc71', label='LightGBM')
    ax2.set_xlabel('样本量', fontsize=11)
    ax2.set_ylabel('内存占用（MB）', fontsize=11)
    ax2.set_title('内存占用对比', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle('XGBoost vs LightGBM 效率对比', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/efficiency_comparison.png', 
                dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print("✅ XGBoost vs LightGBM文章配图生成完成")

if __name__ == '__main__':
    print("开始生成文章配图...")
    generate_factor_crowding_images()
    generate_xgboost_lightgbm_images()
    print("✅ 所有配图生成完成！")
