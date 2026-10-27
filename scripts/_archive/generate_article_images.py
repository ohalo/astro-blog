#!/usr/bin/env python3
"""
生成PCA和XGBoost文章所需的配图
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
os.makedirs('public/images/pca-statistical-arbitrage', exist_ok=True)
os.makedirs('public/images/xgboost-lightgbm-stock-selection', exist_ok=True)

# ==================== PCA文章配图 ====================

# 图1: PCA解释方差图
print("生成PCA解释方差图...")
fig, ax = plt.subplots(figsize=(10, 6))

components = range(1, 11)
explained_variance = np.array([0.35, 0.20, 0.15, 0.10, 0.05, 0.04, 0.03, 0.02, 0.01, 0.01])

ax.bar(components, explained_variance, color='steelblue', alpha=0.8, edgecolor='navy')
ax.plot(components, np.cumsum(explained_variance), 'ro-', linewidth=2, label='累积解释方差')
ax.set_xlabel('主成分 (Principal Component)', fontsize=12)
ax.set_ylabel('解释方差比 (Explained Variance Ratio)', fontsize=12)
ax.set_title('PCA主成分解释方差', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.0])

for i, v in enumerate(explained_variance):
    ax.text(i+1, v + 0.01, f'{v:.2%}', ha='center', fontsize=9)

plt.tight_layout()
plt.savefig('public/images/pca-statistical-arbitrage/pca_components.png', dpi=150, bbox_inches='tight')
plt.close()

# 图2: 因子收益曲线图
print("生成因子收益曲线图...")
np.random.seed(42)
dates = np.arange(100)
factor1_returns = np.cumprod(1 + np.random.normal(0.001, 0.02, 100))
factor2_returns = np.cumprod(1 + np.random.normal(0.0005, 0.015, 100))
factor3_returns = np.cumprod(1 + np.random.normal(0, 0.01, 100))

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(dates, factor1_returns, label='市场因子 (Market Factor)', linewidth=2, color='blue')
ax.plot(dates, factor2_returns, label='规模因子 (Size Factor)', linewidth=2, color='red')
ax.plot(dates, factor3_returns, label='价值因子 (Value Factor)', linewidth=2, color='green')
ax.set_xlabel('交易日 (Trading Days)', fontsize=12)
ax.set_ylabel('累积收益 (Cumulative Returns)', fontsize=12)
ax.set_title('因子收益曲线', fontsize=14, fontweight='bold')
ax.legend(loc='best')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('public/images/pca-statistical-arbitrage/factor_returns.png', dpi=150, bbox_inches='tight')
plt.close()

# ==================== XGBoost文章配图 ====================

# 图1: 特征重要性图
print("生成特征重要性图...")
np.random.seed(42)
features = ['动量_20D', '波动率_60D', '换手率', '市值', 'PE', 'PB', 'ROE', '营收增速', '现金流', '杠杆率']
importance_scores = np.random.uniform(0.05, 0.15, 10)
importance_scores = importance_scores / importance_scores.sum()

fig, ax = plt.subplots(figsize=(10, 6))
y_pos = np.arange(len(features))
ax.barh(y_pos, importance_scores, color='darkorange', alpha=0.8, edgecolor='darkred')
ax.set_yticks(y_pos)
ax.set_yticklabels(features, fontsize=10)
ax.set_xlabel('重要性得分 (Importance Score)', fontsize=12)
ax.set_title('XGBoost特征重要性', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

for i, v in enumerate(importance_scores):
    ax.text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('public/images/xgboost-lightgbm-stock-selection/feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()

# 图2: 模型性能对比图
print("生成模型性能对比图...")
models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost', 'LightGBM', 'Neural\nNetwork']
accuracy = [0.52, 0.58, 0.63, 0.64, 0.61]
precision = [0.51, 0.57, 0.62, 0.63, 0.60]
recall = [0.50, 0.56, 0.61, 0.62, 0.59]
f1_score = [0.505, 0.565, 0.615, 0.625, 0.595]

x = np.arange(len(models))
width = 0.2

fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(x - 1.5*width, accuracy, width, label='准确率 (Accuracy)', color='skyblue', alpha=0.8)
ax.bar(x - 0.5*width, precision, width, label='精确率 (Precision)', color='lightcoral', alpha=0.8)
ax.bar(x + 0.5*width, recall, width, label='召回率 (Recall)', color='lightgreen', alpha=0.8)
ax.bar(x + 1.5*width, f1_score, width, label='F1分数', color='gold', alpha=0.8)

ax.set_xlabel('模型 (Models)', fontsize=12)
ax.set_ylabel('得分 (Score)', fontsize=12)
ax.set_title('机器学习模型性能对比', fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(loc='best')
ax.grid(True, alpha=0.3, axis='y')
ax.set_ylim([0.45, 0.70])

plt.tight_layout()
plt.savefig('public/images/xgboost-lightgbm-stock-selection/model_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n所有配图生成完成！")
print("- PCA文章配图:")
print("  - public/images/pca-statistical-arbitrage/pca_components.png")
print("  - public/images/pca-statistical-arbitrage/factor_returns.png")
print("- XGBoost文章配图:")
print("  - public/images/xgboost-lightgbm-stock-selection/feature_importance.png")
print("  - public/images/xgboost-lightgbm-stock-selection/model_comparison.png")
