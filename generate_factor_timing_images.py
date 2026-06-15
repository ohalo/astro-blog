#!/usr/bin/env python3
"""
生成因子择时文章的配图
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

# 创建输出目录
output_dir = '/Users/halo/workspace/astro-blog/public/images/factor-timing'
os.makedirs(output_dir, exist_ok=True)

# 生成图1：因子暴露随时间变化
np.random.seed(42)
dates = pd.date_range('2020-01-01', '2024-12-31', freq='ME')
n_months = len(dates)

# 模拟三个因子的暴露度
value_exposure = 0.5 + 0.3 * np.sin(2 * np.pi * np.arange(n_months) / 12) + np.random.normal(0, 0.1, n_months)
momentum_exposure = 0.3 + 0.2 * np.cos(2 * np.pi * np.arange(n_months) / 6) + np.random.normal(0, 0.08, n_months)
size_exposure = 0.4 + 0.15 * np.sin(2 * np.pi * np.arange(n_months) / 8 + np.pi/4) + np.random.normal(0, 0.05, n_months)

plt.figure(figsize=(12, 6))
plt.plot(dates, value_exposure, label='价值因子暴露', linewidth=2)
plt.plot(dates, momentum_exposure, label='动量因子暴露', linewidth=2)
plt.plot(dates, size_exposure, label='规模因子暴露', linewidth=2)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
plt.xlabel('时间', fontsize=12)
plt.ylabel('因子暴露', fontsize=12)
plt.title('因子暴露动态调整示例', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/factor_exposure_dynamic.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成图2：因子择时策略累计收益
np.random.seed(123)
n_days = 1000
dates_daily = pd.date_range('2022-01-01', periods=n_days, freq='D')

# 模拟收益
factor_timing_return = np.random.normal(0.0008, 0.008, n_days)  # 因子择时策略
buy_hold_return = np.random.normal(0.0005, 0.01, n_days)  # 买入持有策略
benchmark_return = np.random.normal(0.0003, 0.009, n_days)  # 基准

# 累计收益
factor_timing_cum = np.cumprod(1 + factor_timing_return) - 1
buy_hold_cum = np.cumprod(1 + buy_hold_return) - 1
benchmark_cum = np.cumprod(1 + benchmark_return) - 1

plt.figure(figsize=(12, 6))
plt.plot(dates_daily, factor_timing_cum * 100, label='因子择时策略', linewidth=2.5, color='#2E86AB')
plt.plot(dates_daily, buy_hold_cum * 100, label='买入持有策略', linewidth=2, color='#A23B72')
plt.plot(dates_daily, benchmark_cum * 100, label='基准指数', linewidth=2, color='#F18F01')
plt.xlabel('时间', fontsize=12)
plt.ylabel('累计收益 (%)', fontsize=12)
plt.title('因子择时策略 vs 买入持有策略', fontsize=14, fontweight='bold')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/factor_timing_performance.png', dpi=150, bbox_inches='tight')
plt.close()

# 生成图3：因子表现热力图
np.random.seed(456)
factors = ['价值', '动量', '规模', '质量', '低波', '成长']
time_periods = ['2022Q1', '2022Q2', '2022Q3', '2022Q4', '2023Q1', '2023Q2', '2023Q3', '2023Q4']

# 生成因子表现矩阵（模拟不同季度因子表现）
performance_matrix = np.random.uniform(-0.05, 0.08, (len(factors), len(time_periods)))

fig, ax = plt.subplots(figsize=(10, 6))
im = ax.imshow(performance_matrix, cmap='RdYlGn', aspect='auto', vmin=-0.05, vmax=0.08)

# 设置刻度
ax.set_xticks(np.arange(len(time_periods)))
ax.set_yticks(np.arange(len(factors)))
ax.set_xticklabels(time_periods)
ax.set_yticklabels(factors)

# 添加数值标注
for i in range(len(factors)):
    for j in range(len(time_periods)):
        text = ax.text(j, i, f'{performance_matrix[i, j]:.2%}',
                      ha='center', va='center', color='black', fontsize=9)

plt.title('不同季度因子表现热力图', fontsize=14, fontweight='bold')
plt.colorbar(im, ax=ax, label='季度收益')
plt.tight_layout()
plt.savefig(f'{output_dir}/factor_performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

print(f"✅ 已生成3张配图到 {output_dir}")
print("   - factor_exposure_dynamic.png")
print("   - factor_timing_performance.png")
print("   - factor_performance_heatmap.png")
