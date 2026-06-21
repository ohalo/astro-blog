#!/usr/bin/env python3
"""
为XGBoost与LightGBM量化选股文章生成配图（使用模拟数据，无需实际训练模型）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# 设置绘图风格
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.labelweight'] = 'bold'

# ========== 生成模拟数据 ==========
print("生成模拟量化选股数据...")

np.random.seed(42)
n_samples = 2000
n_features = 20

# 生成模拟因子数据
X, y = make_classification(
    n_samples=n_samples,
    n_features=n_features,
    n_informative=10,
    n_redundant=5,
    n_clusters_per_class=2,
    random_state=42
)

# 转换为DataFrame（模拟因子名称）
factor_names = [
    'PE_ratio', 'PB_ratio', 'ROE', 'Momentum_1M', 'Momentum_3M',
    'Momentum_6M', 'Volatility_20D', 'Volume_Change', 'RSI_14', 'MA_Ratio',
    'Revenue_Growth', 'Earnings_Growth', 'Debt_to_Equity', 'Current_Ratio',
    'Cash_Flow_Quality', 'Analyst_Upgrade', 'Northbound_Flow', 'Turnover_Rate',
    'Beta', 'Alpha'
]

X = pd.DataFrame(X, columns=factor_names)
y = pd.Series(y, name='Label')

# 划分训练集和测试集
X_train, X_test, y_train, y_test = X[:1600], X[1600:], y[:1600], y[1600:]

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 因子")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"正样本比例: {y_train.mean():.2%}")

# 模拟模型预测概率（不需要真实训练）
print("\n生成模拟预测结果...")
np.random.seed(42)
# 模拟XGBoost预测（AUC ≈ 0.75）
y_pred_proba_xgb = np.random.beta(a=3, b=1, size=len(X_test)) * 0.5 + 0.25
y_pred_proba_xgb[y_test==1] += 0.2
y_pred_proba_xgb = np.clip(y_pred_proba_xgb, 0, 1)

# 模拟LightGBM预测（AUC ≈ 0.77）
y_pred_proba_lgb = np.random.beta(a=3, b=1, size=len(X_test)) * 0.5 + 0.27
y_pred_proba_lgb[y_test==1] += 0.2
y_pred_proba_lgb = np.clip(y_pred_proba_lgb, 0, 1)

# 模拟预测标签
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)
y_pred_lgb = (y_pred_proba_lgb >= 0.5).astype(int)

# 计算模拟性能指标
from sklearn.metrics import accuracy_score, roc_auc_score
acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)
acc_lgb = accuracy_score(y_test, y_pred_lgb)
auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)

print(f"模拟XGBoost   - Accuracy: {acc_xgb:.4f}, AUC: {auc_xgb:.4f}")
print(f"模拟LightGBM  - Accuracy: {acc_lgb:.4f}, AUC: {auc_lgb:.4f}")

# ========== 1. 生成图1: 因子分布箱线图 ==========
print("\n生成图1: 因子分布箱线图...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(factor_names):
    data_pos = X_train[col].values[y_train==1] + np.random.normal(0, 0.1, sum(y_train==1))
    data_neg = X_train[col].values[y_train==0] + np.random.normal(0, 0.1, sum(y_train==0))
    
    bp = axes[i].boxplot([data_neg, data_pos], 
                         patch_artist=True,
                         boxprops=dict(facecolor='lightblue', alpha=0.7),
                         medianprops=dict(color='red', linewidth=2))
    axes[i].set_xticklabels(['Label=0', 'Label=1'])
    axes[i].set_title(f'{col}', fontsize=10, fontweight='bold')
    axes[i].set_ylabel('Value', fontsize=8)
    axes[i].tick_params(axis='both', labelsize=8)
    axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.suptitle('Factor Distributions by Label (Train Set)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/factor_distributions.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: factor_distributions.png")

# ========== 2. 生成图2: 特征重要性对比 ==========
print("生成图2: 特征重要性对比...")

# 模拟特征重要性
np.random.seed(42)
xgb_importance = np.random.exponential(scale=100, size=n_features)
lgb_importance = np.random.exponential(scale=110, size=n_features)

# 排序
xgb_rank = np.argsort(-xgb_importance)
lgb_rank = np.argsort(-lgb_importance)

# 创建DataFrame
importance_df = pd.DataFrame({
    'feature': factor_names,
    'xgb_importance': xgb_importance,
    'lgb_importance': lgb_importance
})

importance_sorted = importance_df.sort_values('xgb_importance', ascending=False)

# 可视化（Top 15）
top_n = 15
top_features = importance_sorted.head(top_n)

fig, axes = plt.subplots(1, 2, figsize=(16, 8))

# XGBoost
axes[0].barh(range(top_n), top_features['xgb_importance'], 
              color='steelblue', alpha=0.8, edgecolor='navy', linewidth=1.5)
axes[0].set_yticks(range(top_n))
axes[0].set_yticklabels(top_features['feature'])
axes[0].invert_yaxis()
axes[0].set_xlabel('Feature Importance (Gain)', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost Feature Importance (Top 15)', fontsize=13, fontweight='bold', pad=10)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# LightGBM
axes[1].barh(range(top_n), top_features['lgb_importance'], 
              color='green', alpha=0.8, edgecolor='darkgreen', linewidth=1.5)
axes[1].set_yticks(range(top_n))
axes[1].set_yticklabels(top_features['feature'])
axes[1].invert_yaxis()
axes[1].set_xlabel('Feature Importance (Gain)', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM Feature Importance (Top 15)', fontsize=13, fontweight='bold', pad=10)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/feature_importance_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: feature_importance_comparison.png")

# ========== 3. 生成图3: ROC曲线对比 ==========
print("生成图3: ROC曲线对比...")

fig, ax = plt.subplots(figsize=(10, 8))

# 计算ROC曲线（使用模拟预测）
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_proba_lgb)

# 绘制
ax.plot(fpr_xgb, tpr_xgb, linewidth=2.5, color='steelblue', 
        label=f'XGBoost (AUC = {auc_xgb:.3f})')
ax.plot(fpr_lgb, tpr_lgb, linewidth=2.5, color='green', 
        label=f'LightGBM (AUC = {auc_lgb:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label=f'Random (AUC = 0.500)')

ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve Comparison: XGBoost vs LightGBM', fontsize=14, fontweight='bold', pad=15)
ax.legend(fontsize=11, loc='lower right', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
ax.set_xlim([-0.01, 1.01])
ax.set_ylim([-0.01, 1.01])

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/roc_curve_comparison.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: roc_curve_comparison.png")

# ========== 4. 生成图4: 模型训练历史 ==========
print("生成图4: 模型训练历史...")

# 模拟训练历史
n_rounds = 500
epochs = np.arange(1, n_rounds + 1)

# XGBoost训练历史（模拟）
train_auc_xgb = 0.5 + 0.25 * (1 - np.exp(-epochs / 100)) + np.random.randn(n_rounds) * 0.01
test_auc_xgb = 0.5 + 0.23 * (1 - np.exp(-epochs / 100)) + np.random.randn(n_rounds) * 0.015
best_iter_xgb = np.argmax(test_auc_xgb)

# LightGBM训练历史（模拟）
train_auc_lgb = 0.5 + 0.27 * (1 - np.exp(-epochs / 80)) + np.random.randn(n_rounds) * 0.01
test_auc_lgb = 0.5 + 0.25 * (1 - np.exp(-epochs / 80)) + np.random.randn(n_rounds) * 0.015
best_iter_lgb = np.argmax(test_auc_lgb)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost
axes[0].plot(epochs, train_auc_xgb, linewidth=2, 
              color='steelblue', alpha=0.8, label='Train AUC')
axes[0].plot(epochs, test_auc_xgb, linewidth=2, 
              color='red', alpha=0.8, label='Test AUC')
axes[0].axvline(x=best_iter_xgb, color='green', 
                 linestyle='--', linewidth=2, label=f'Best Iter: {best_iter_xgb}')
axes[0].set_xlabel('Boosting Round', fontsize=11, fontweight='bold')
axes[0].set_ylabel('AUC', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost Training History', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[0].set_ylim([0.45, 0.85])

# LightGBM
axes[1].plot(epochs, train_auc_lgb, linewidth=2, 
              color='green', alpha=0.8, label='Train AUC')
axes[1].plot(epochs, test_auc_lgb, linewidth=2, 
              color='orange', alpha=0.8, label='Test AUC')
axes[1].axvline(x=best_iter_lgb, color='red', 
                 linestyle='--', linewidth=2, label=f'Best Iter: {best_iter_lgb}')
axes[1].set_xlabel('Boosting Round', fontsize=11, fontweight='bold')
axes[1].set_ylabel('AUC', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM Training History', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1].set_ylim([0.45, 0.85])

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/training_history.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: training_history.png")

# ========== 5. 生成图5: 预测概率分布 ==========
print("生成图5: 预测概率分布...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost预测概率分布
axes[0].hist(y_pred_proba_xgb[y_test==0], bins=30, alpha=0.6, 
              color='blue', edgecolor='black', linewidth=1.2, density=True, label='Actual: 0')
axes[0].hist(y_pred_proba_xgb[y_test==1], bins=30, alpha=0.6, 
              color='red', edgecolor='black', linewidth=1.2, density=True, label='Actual: 1')
axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 0.5')
axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost: Predicted Probability Distribution', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# LightGBM预测概率分布
axes[1].hist(y_pred_proba_lgb[y_test==0], bins=30, alpha=0.6, 
              color='blue', edgecolor='black', linewidth=1.2, density=True, label='Actual: 0')
axes[1].hist(y_pred_proba_lgb[y_test==1], bins=30, alpha=0.6, 
              color='red', edgecolor='black', linewidth=1.2, density=True, label='Actual: 1')
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 0.5')
axes[1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Density', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM: Predicted Probability Distribution', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/prediction_distribution.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: prediction_distribution.png")

# ========== 6. 生成图6: 因子相关性热力图 ==========
print("生成图6: 因子相关性热力图...")

# 计算相关性矩阵
corr_matrix = X_train.corr()

plt.figure(figsize=(14, 12))
sns.heatmap(corr_matrix, 
            annot=True, 
            fmt='.2f',
            cmap='RdBu_r',
            center=0,
            square=True,
            cbar_kws={'label': 'Correlation', 'shrink': 0.8},
            annot_kws={'size': 8, 'weight': 'bold'},
            linewidths=0.5,
            linecolor='gray')
plt.title('Factor Correlation Matrix (Train Set)', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/factor_correlation_heatmap.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: factor_correlation_heatmap.png")

# ========== 7. 生成图7: 策略回测结果 ==========
print("生成图7: 策略回测结果...")

# 模拟策略回测（基于AUC）
n_weeks = 52
strategy_returns = []
benchmark_returns = []

for week in range(n_weeks):
    # 策略收益（与AUC挂钩）
    weekly_return = np.random.normal(auc_lgb * 0.02, 0.025)  # 均值与AUC挂钩
    strategy_returns.append(weekly_return)
    
    # 基准收益（市场平均）
    benchmark_weekly_return = np.random.normal(0.002, 0.018)
    benchmark_returns.append(benchmark_weekly_return)

strategy_cumulative = np.cumprod(1 + np.array(strategy_returns))
benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns))

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 累积收益曲线
axes[0].plot(range(n_weeks), strategy_cumulative, linewidth=2.5, 
              color='darkblue', alpha=0.8, label=f'Strategy (Final: {strategy_cumulative[-1]:.2f}x)')
axes[0].plot(range(n_weeks), benchmark_cumulative, linewidth=2.5, 
              color='gray', alpha=0.8, label=f'Benchmark (Final: {benchmark_cumulative[-1]:.2f}x)')
axes[0].axhline(y=1, color='black', linestyle='--', alpha=0.5, linewidth=1.5)
axes[0].fill_between(range(n_weeks), 1, strategy_cumulative, 
                      where=(strategy_cumulative > 1), alpha=0.3, color='green')
axes[0].fill_between(range(n_weeks), 1, strategy_cumulative,
                      where=(strategy_cumulative < 1), alpha=0.3, color='red')
axes[0].set_xlabel('Week', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Cumulative Return (Multiple)', fontsize=11, fontweight='bold')
axes[0].set_title('Stock Selection Strategy Backtest', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=10, loc='upper left', framealpha=0.9)
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[0].set_xlim(0, n_weeks-1)

# 周度收益柱状图
colors = ['green' if r > 0 else 'red' for r in strategy_returns]
axes[1].bar(range(n_weeks), strategy_returns, color=colors, alpha=0.7, 
             edgecolor='black', linewidth=0.8, label='Strategy Weekly Return')
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
axes[1].plot(range(n_weeks), benchmark_returns, 'o-', linewidth=1.5, 
              color='gray', alpha=0.6, label='Benchmark Weekly Return')
axes[1].set_xlabel('Week', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Weekly Return', fontsize=11, fontweight='bold')
axes[1].set_title('Weekly Returns Comparison', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=10, loc='upper right', framealpha=0.9)
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)
axes[1].set_xlim(-0.5, n_weeks-0.5)

# 添加性能标注
total_return_strategy = (strategy_cumulative[-1] - 1) * 100
total_return_benchmark = (benchmark_cumulative[-1] - 1) * 100
sharpe_strategy = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(52)

textstr = f'Strategy:\n  Total Return: {total_return_strategy:.1f}%\n  Sharpe: {sharpe_strategy:.2f}\n\nBenchmark:\n  Total Return: {total_return_benchmark:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
axes[0].text(0.02, 0.98, textstr, transform=axes[0].transAxes, 
              fontsize=9, verticalalignment='top', bbox=props, fontweight='bold')

plt.tight_layout(pad=2.0, h_pad=3.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/backtest_results.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: backtest_results.png")

# ========== 总结 ==========
print("\n" + "="*70)
print("所有配图生成完成！")
print("="*70)
print(f"图片保存位置: /Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/")
print("\n生成的图片:")
print("  1. factor_distributions.png - 因子分布箱线图")
print("  2. feature_importance_comparison.png - 特征重要性对比")
print("  3. roc_curve_comparison.png - ROC曲线对比")
print("  4. training_history.png - 模型训练历史")
print("  5. prediction_distribution.png - 预测概率分布")
print("  6. factor_correlation_heatmap.png - 因子相关性热力图")
print("  7. backtest_results.png - 策略回测结果")
print("\n模拟模型性能（用于可视化）:")
print(f"  XGBoost   - Accuracy: {acc_xgb:.4f}, AUC: {auc_xgb:.4f}")
print(f"  LightGBM   - Accuracy: {acc_lgb:.4f}, AUC: {auc_lgb:.4f}")
print("="*70)
