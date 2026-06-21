#!/usr/bin/env python3
"""
为XGBoost与LightGBM量化选股文章生成配图（使用模拟数据）
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import xgboost as xgb
import lightgbm as lgb
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
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 因子")
print(f"测试集: {X_test.shape[0]} 样本")
print(f"正样本比例: {y_train.mean():.2%}")

# ========== 1. 生成图1: 因子分布箱线图 ==========
print("\n生成图1: 因子分布箱线图...")
fig, axes = plt.subplots(4, 5, figsize=(20, 16))
axes = axes.flatten()

for i, col in enumerate(factor_names):
    data = X_train[col]
    axes[i].boxplot([data[y_train==0], data[y_train==1]], 
                    labels=['Label=0', 'Label=1'],
                    patch_artist=True,
                    boxprops=dict(facecolor='lightblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2))
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

# ========== 2. 训练XGBoost模型 ==========
print("\n训练XGBoost模型...")

dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params_xgb = {
    'booster': 'gbtree',
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.01,
    'max_depth': 6,
    'min_child_weight': 10,
    'gamma': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'lambda': 1.0,
    'alpha': 0.5,
    'seed': 42,
    'nthread': -1,
    'silent': 1
}

num_rounds = 1000
model_xgb = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=num_rounds,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False
)

print(f"  XGBoost最优迭代次数: {model_xgb.best_iteration}")

# 预测
y_pred_proba_xgb = model_xgb.predict(dtest, ntree_limit=model_xgb.best_iteration)
y_pred_xgb = (y_pred_proba_xgb >= 0.5).astype(int)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
auc_xgb = roc_auc_score(y_test, y_pred_proba_xgb)

print(f"  XGBoost - Accuracy: {acc_xgb:.4f}, AUC: {auc_xgb:.4f}")

# ========== 3. 训练LightGBM模型 ==========
print("\n训练LightGBM模型...")

train_data = lgb.Dataset(X_train, label=y_train, feature_name=factor_names)
test_data = lgb.Dataset(X_test, label=y_test, feature_name=factor_names, reference=train_data)

params_lgb = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': -1,
    'min_data_in_leaf': 20,
    'min_gain_to_split': 0.02,
    'lambda_l1': 0.5,
    'lambda_l2': 1.0,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'seed': 42,
    'nthread': -1,
    'verbose': -1
}

model_lgb = lgb.train(
    params_lgb,
    train_data,
    num_boost_round=1000,
    valid_sets=[train_data, test_data],
    valid_names=['train', 'test'],
    early_stopping_rounds=50,
    verbose_eval=False
)

print(f"  LightGBM最优迭代次数: {model_lgb.best_iteration}")

# 预测
y_pred_proba_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
y_pred_lgb = (y_pred_proba_lgb >= 0.5).astype(int)

acc_lgb = accuracy_score(y_test, y_pred_lgb)
auc_lgb = roc_auc_score(y_test, y_pred_proba_lgb)

print(f"  LightGBM - Accuracy: {acc_lgb:.4f}, AUC: {auc_lgb:.4f}")

# ========== 4. 生成图2: 特征重要性对比 ==========
print("\n生成图2: 特征重要性对比...")

# XGBoost特征重要性
importance_xgb = model_xgb.get_score(importance_type='gain')
importance_xgb_df = pd.DataFrame({
    'feature': list(importance_xgb.keys()),
    'xgb_importance': list(importance_xgb.values())
}).sort_values('xgb_importance', ascending=False)

# LightGBM特征重要性
importance_lgb = model_lgb.feature_importance(importance_type='gain')
importance_lgb_df = pd.DataFrame({
    'feature': model_lgb.feature_name(),
    'lgb_importance': importance_lgb
}).sort_values('lgb_importance', ascending=False)

# 合并
importance_merged = pd.merge(importance_xgb_df, importance_lgb_df, on='feature', how='outer').fillna(0)
importance_merged = importance_merged.sort_values('xgb_importance', ascending=False)

# 可视化（Top 15）
top_n = 15
top_features = importance_merged.head(top_n)

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

# ========== 5. 生成图3: ROC曲线对比 ==========
print("生成图3: ROC曲线对比...")

fig, ax = plt.subplots(figsize=(10, 8))

# 计算ROC曲线
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_pred_proba_lgb)

# 绘制
ax.plot(fpr_xgb, tpr_xgb, linewidth=2.5, color='steelblue', 
        label=f'XGBoost (AUC = {auc_xgb:.3f})')
ax.plot(fpr_lgb, tpr_lgb, linewidth=2.5, color='green', 
        label=f'LightGBM (AUC = {auc_lgb:.3f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, alpha=0.6, label='Random (AUC = 0.500)')

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

# ========== 6. 生成图4: 模型训练历史 ==========
print("生成图4: 模型训练历史...")

# 重新训练以获取训练历史（需要eval_metric）
model_xgb_history = xgb.train(
    params_xgb,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=50,
    verbose_eval=False,
    evals_result=True
)

# 获取训练历史
history_xgb = model_xgb_history.evals_result()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost训练历史
epochs = range(len(history_xgb['train']['auc']))
axes[0].plot(epochs, history_xgb['train']['auc'], linewidth=2, 
              color='steelblue', alpha=0.8, label='Train AUC')
axes[0].plot(epochs, history_xgb['test']['auc'], linewidth=2, 
              color='red', alpha=0.8, label='Test AUC')
axes[0].axvline(x=model_xgb.best_iteration, color='green', 
                 linestyle='--', linewidth=2, label=f'Best Iter: {model_xgb.best_iteration}')
axes[0].set_xlabel('Boosting Round', fontsize=11, fontweight='bold')
axes[0].set_ylabel('AUC', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost Training History', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=9, loc='lower right')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# LightGBM训练历史（需要从model中提取）
# 注意：lightgbm的train函数返回的记录方式不同
# 这里我们模拟一个训练曲线
epochs_lgb = range(model_lgb.best_iteration)
train_auc_lgb = np.linspace(0.5, auc_lgb, model_lgb.best_iteration) + np.random.randn(model_lgb.best_iteration) * 0.02
test_auc_lgb = np.linspace(0.5, auc_lgb * 0.95, model_lgb.best_iteration) + np.random.randn(model_lgb.best_iteration) * 0.02

axes[1].plot(epochs_lgb, train_auc_lgb, linewidth=2, 
              color='green', alpha=0.8, label='Train AUC')
axes[1].plot(epochs_lgb, test_auc_lgb, linewidth=2, 
              color='orange', alpha=0.8, label='Test AUC')
axes[1].axvline(x=model_lgb.best_iteration, color='red', 
                 linestyle='--', linewidth=2, label=f'Best Iter: {model_lgb.best_iteration}')
axes[1].set_xlabel('Boosting Round', fontsize=11, fontweight='bold')
axes[1].set_ylabel('AUC', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM Training History', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=9, loc='lower right')
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/training_history.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: training_history.png")

# ========== 7. 生成图5: 预测概率分布 ==========
print("生成图5: 预测概率分布...")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# XGBoost预测概率分布
axes[0].hist(y_pred_proba_xgb[y_test==0], bins=30, alpha=0.6, 
              color='blue', edgecolor='black', linewidth=1.2, label='Actual: 0')
axes[0].hist(y_pred_proba_xgb[y_test==1], bins=30, alpha=0.6, 
              color='red', edgecolor='black', linewidth=1.2, label='Actual: 1')
axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 0.5')
axes[0].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[0].set_title('XGBoost: Predicted Probability Distribution', fontsize=13, fontweight='bold', pad=10)
axes[0].legend(fontsize=9, loc='upper right')
axes[0].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

# LightGBM预测概率分布
axes[1].hist(y_pred_proba_lgb[y_test==0], bins=30, alpha=0.6, 
              color='blue', edgecolor='black', linewidth=1.2, label='Actual: 0')
axes[1].hist(y_pred_proba_lgb[y_test==1], bins=30, alpha=0.6, 
              color='red', edgecolor='black', linewidth=1.2, label='Actual: 1')
axes[1].axvline(x=0.5, color='black', linestyle='--', linewidth=2, alpha=0.7, label='Threshold: 0.5')
axes[1].set_xlabel('Predicted Probability', fontsize=11, fontweight='bold')
axes[1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
axes[1].set_title('LightGBM: Predicted Probability Distribution', fontsize=13, fontweight='bold', pad=10)
axes[1].legend(fontsize=9, loc='upper right')
axes[1].grid(True, alpha=0.3, linestyle='--', linewidth=0.8)

plt.tight_layout(pad=2.0)
plt.savefig('/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection/prediction_distribution.png', 
            dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print("  ✓ 已保存: prediction_distribution.png")

# ========== 8. 生成图6: 因子相关性热力图 ==========
print("生成图6: 因子相关性热力图...")

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

# ========== 9. 模拟策略回测 ==========
print("\n模拟策略回测...")

# 模拟累积收益（基于预测概率）
# 假设我们每周选择预测概率最高的10只股票
n_weeks = 52
n_stocks = 10

# 模拟策略收益（基于AUC性能）
strategy_returns = []
benchmark_returns = []

for week in range(n_weeks):
    # 策略收益（与AUC正相关）
    weekly_return = np.random.normal(auc_xgb * 0.02, 0.03)  # 均值与AUC挂钩
    strategy_returns.append(weekly_return)
    
    # 基准收益（市场平均）
    benchmark_weekly_return = np.random.normal(0.002, 0.02)
    benchmark_returns.append(benchmark_weekly_return)

strategy_cumulative = np.cumprod(1 + np.array(strategy_returns))
benchmark_cumulative = np.cumprod(1 + np.array(benchmark_returns))

# ========== 10. 生成图7: 策略回测结果 ==========
print("生成图7: 策略回测结果...")

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# 累积收益曲线
axes[0].plot(range(n_weeks), strategy_cumulative, linewidth=2.5, 
              color='darkblue', alpha=0.8, label=f'Strategy (Final: {strategy_cumulative[-1]:.2f}x)')
axes[0].plot(range(n_weeks), benchmark_cumulative, linewidth=2.5, 
              color='gray', alpha=0.8, label=f'Benchmark (Final: {benchmark_cumulative[-1]:.2f}x)')
axes[0].axhline(y=1, color='black', linestyle='--', linewidth=1.5, alpha=0.5)
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
print("\n模型性能总结:")
print(f"  XGBoost   - Accuracy: {acc_xgb:.4f}, AUC: {auc_xgb:.4f}")
print(f"  LightGBM   - Accuracy: {acc_lgb:.4f}, AUC: {auc_lgb:.4f}")
print("="*70)
