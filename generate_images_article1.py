#!/usr/bin/env python3
"""
为第一篇文章（多因子模型风险分解）生成配图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
os.makedirs('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition', exist_ok=True)

# 生成模拟数据
def load_factor_data(start_date='2020-01-01', end_date='2025-12-31'):
    """加载因子数据"""
    dates = pd.date_range(start=start_date, end=end_date, freq='ME')
    np.random.seed(42)
    
    factor_data = pd.DataFrame({
        'date': dates,
        'MKT': np.random.normal(0.008, 0.04, len(dates)),
        'SMB': np.random.normal(0.002, 0.02, len(dates)),
        'HML': np.random.normal(0.003, 0.02, len(dates)),
        'MOM': np.random.normal(0.004, 0.03, len(dates)),
    })
    
    factor_data.set_index('date', inplace=True)
    return factor_data

factors = load_factor_data()

# 图1: 因子暴露时间序列
def generate_figure1():
    """生成因子暴露时间序列图"""
    # 模拟因子暴露数据
    np.random.seed(123)
    dates = pd.date_range(start='2020-01-01', end='2025-12-31', freq='ME')
    n_months = len(dates)
    
    # 生成随时间变化的因子暴露
    mkt_exposure = 1.2 + np.cumsum(np.random.normal(0, 0.02, n_months))
    smb_exposure = 0.5 + np.cumsum(np.random.normal(0, 0.01, n_months))
    hml_exposure = -0.3 + np.cumsum(np.random.normal(0, 0.015, n_months))
    mom_exposure = 0.8 + np.cumsum(np.random.normal(0, 0.02, n_months))
    
    exposure_df = pd.DataFrame({
        'MKT': mkt_exposure,
        'SMB': smb_exposure,
        'HML': hml_exposure,
        'MOM': mom_exposure,
    }, index=dates)
    
    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    factors = exposure_df.columns
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    for i, factor in enumerate(factors):
        ax = axes[i]
        ax.plot(exposure_df.index, exposure_df[factor], linewidth=2.5, color=colors[i])
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.3)
        
        # 添加移动平均线
        rolling_mean = exposure_df[factor].rolling(window=12).mean()
        ax.plot(rolling_mean.index, rolling_mean, linewidth=1.5, 
                color='gray', linestyle=':', alpha=0.7, label='12个月移动平均')
        
        ax.set_title(f'MKT因子暴露', fontsize=13, fontweight='bold', pad=10)
        ax.set_xlabel('日期', fontsize=11)
        ax.set_ylabel('暴露度', fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=9)
    
    # 调整子图标题
    axes[0].set_title('MKT因子暴露', fontsize=13, fontweight='bold', pad=10)
    axes[1].set_title('SMB因子暴露', fontsize=13, fontweight='bold', pad=10)
    axes[2].set_title('HML因子暴露', fontsize=13, fontweight='bold', pad=10)
    axes[3].set_title('MOM因子暴露', fontsize=13, fontweight='bold', pad=10)
    
    plt.suptitle('因子暴露时间序列分析（示例股票）', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/factor_exposure.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图1: factor_exposure.png")

# 图2: 风险贡献度饼图
def generate_figure2():
    """生成风险贡献度饼图"""
    # 模拟风险分解结果
    np.random.seed(456)
    
    # 因子风险贡献（方差）
    factor_contrib = {
        'MKT': 0.0156,
        'SMB': 0.0034,
        'HML': 0.0021,
        'MOM': 0.0089,
    }
    
    # 特异性风险
    specific_risk = 0.0123
    
    # 准备饼图数据
    labels = list(factor_contrib.keys()) + ['特异性风险']
    sizes = list(factor_contrib.values()) + [specific_risk]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#ff99cc']
    
    # 计算百分比
    total = sum(sizes)
    percentages = [s/total*100 for s in sizes]
    
    # 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%',
        startangle=90,
        colors=colors,
        textprops={'fontsize': 12, 'weight': 'bold'},
        pctdistance=0.85,
        wedgeprops=dict(linewidth=2, edgecolor='white')
    )
    
    # 美化百分比文字
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_weight('bold')
    
    ax.set_title('投资组合风险贡献度分解\n（总方差: {:.4f}）'.format(total), 
                 fontsize=15, fontweight='bold', pad=20)
    
    # 添加图例
    ax.legend(wedges, 
              ['{}: {:.2f}%'.format(l, p) for l, p in zip(labels, percentages)],
              title='风险来源',
              loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1),
              fontsize=10)
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/risk_contribution.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图2: risk_contribution.png")

# 图3: 因子协方差热力图
def generate_figure3():
    """生成因子协方差和相关系数热力图"""
    # 计算真实的因子协方差和相关系数
    cov_matrix = factors.cov()
    corr_matrix = factors.corr()
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 协方差矩阵热力图（使用matplotlib）
    cov_data = cov_matrix.values * 100  # 放大100倍
    im1 = axes[0].imshow(cov_data, cmap='RdYlBu_r', aspect='equal')
    
    # 添加数值标注
    for i in range(len(cov_matrix.index)):
        for j in range(len(cov_matrix.columns)):
            text = axes[0].text(j, i, f'{cov_data[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=10, weight='bold')
    
    axes[0].set_xticks(range(len(cov_matrix.columns)))
    axes[0].set_yticks(range(len(cov_matrix.index)))
    axes[0].set_xticklabels(cov_matrix.columns, fontsize=11)
    axes[0].set_yticklabels(cov_matrix.index, fontsize=11)
    axes[0].set_title('因子协方差矩阵 (×100)', fontsize=14, fontweight='bold', pad=15)
    
    # 添加颜色条
    plt.colorbar(im1, ax=axes[0], label='协方差 (×100)')
    
    # 相关系数矩阵热力图
    corr_data = corr_matrix.values
    im2 = axes[1].imshow(corr_data, cmap='RdYlBu_r', vmin=-1, vmax=1, aspect='equal')
    
    # 添加数值标注
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            text = axes[1].text(j, i, f'{corr_data[i, j]:.2f}',
                           ha='center', va='center', color='black', fontsize=10, weight='bold')
    
    axes[1].set_xticks(range(len(corr_matrix.columns)))
    axes[1].set_yticks(range(len(corr_matrix.index)))
    axes[1].set_xticklabels(corr_matrix.columns, fontsize=11)
    axes[1].set_yticklabels(corr_matrix.index, fontsize=11)
    axes[1].set_title('因子相关系数矩阵', fontsize=14, fontweight='bold', pad=15)
    
    # 添加颜色条
    plt.colorbar(im2, ax=axes[1], label='相关系数')
    
    plt.suptitle('因子协方差与相关系数分析', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/factor_cov_heatmap.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 生成图3: factor_cov_heatmap.png")

# 图4: 封面图 - 多因子模型示意图
def generate_cover_image():
    """生成封面配图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 创建示意图：展示多因子模型的结构
    factors = ['MKT', 'SMB', 'HML', 'MOM']
    weights = [0.35, 0.20, 0.25, 0.20]  # 因子权重
    
    # 绘制因子贡献柱状图
    x_pos = np.arange(len(factors))
    bars = ax.bar(x_pos, weights, color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#f9ca24'], 
                  edgecolor='black', linewidth=2, alpha=0.8)
    
    # 添加数值标签
    for i, (bar, weight) in enumerate(zip(bars, weights)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{weight:.0%}', ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_xlabel('风险因子', fontsize=14, fontweight='bold')
    ax.set_ylabel('因子权重', fontsize=14, fontweight='bold')
    ax.set_title('多因子模型风险分解示意图\nRisk Decomposition of Multi-Factor Model', 
                 fontsize=18, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(factors, fontsize=13)
    ax.set_ylim(0, max(weights) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # 添加解释性文字
    ax.text(0.5, 0.95, 'Portfolio Return = Σ(β_i × f_i) + α + ε', 
            transform=ax.transAxes, fontsize=12, ha='center', style='italic',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/halo/workspace/astro-blog/public/images/multi-factor-risk-decomposition/cover.jpg', 
                dpi=300, bbox_inches='tight', format='jpg')
    plt.close()
    print("✓ 生成封面图: cover.jpg")

# 生成所有图片
if __name__ == '__main__':
    print("开始生成第一篇文章的配图...")
    generate_figure1()
    generate_figure2()
    generate_figure3()
    generate_cover_image()
    print("\n✅ 第一篇文章的所有配图已生成完成！")
