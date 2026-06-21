"""
为XGBoost与LightGBM选股文章生成配图（使用合成数据）
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score
import xgboost as xgb
import lightgbm as lgb
from scipy import stats

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 创建图片保存目录
import os
save_dir = '/Users/halo/workspace/astro-blog/public/images/xgboost-lightgbm-stock-selection'
os.makedirs(save_dir, exist_ok=True)

# 生成合成数据
print("正在生成模拟数据...")

np.random.seed(42)
n_days = 1000
n_stocks = 50

# 生成日期索引
dates = pd.date_range('2022-01-01', periods=n_days, freq='D')

# 生成因子数据
def generate_factor_data(n_days, n_stocks, dates):
    all_features = []
    all_labels = []
    
    for i in range(n_stocks):
        # 生成特征
        features = pd.DataFrame(index=dates)
        
        # 1. 动量因子
        for period in [5, 10, 20]:
            features[f'return_{period}d'] = np.random.randn(n_days) * 0.02
        
        # 2. 波动率因子
        for period in [20, 60]:
            features[f'volatility_{period}d'] = np.abs(np.random.randn(n_days) * 0.01)
        
        # 3. 成交量因子
        features['volume_ratio'] = np.random.uniform(0.5, 2.0, n_days)
        
        # 4. 技术指标
        features['ma_ratio'] = np.random.uniform(0.8, 1.2, n_days)
        features['rsi_14'] = np.random.uniform(30, 70, n_days)
        
        # 5. 收益滞后特征
        for lag in [1, 2, 3, 5]:
            features[f'return_lag_{lag}'] = np.random.randn(n_days) * 0.015
        
        # 生成标签（未来5日收益）
        # 让一些特征与标签相关
        label = (
            features['return_5d'] * 0.3 +
            features['volatility_20d'] * (-0.2) +
            features['volume_ratio'] * 0.1 +
            np.random.randn(n_days) * 0.01
        )
        
        features['label'] = label
        features['stock_code'] = f'STOCK_{i:03d}.SS'
        
        all_features.append(features)
    
    # 合并数据
    data_df = pd.concat(all_features, ignore_index=False)
    data_df = data_df.dropna()
    
    return data_df

# 生成数据
print("正在提取特征和标签...")
data_df = generate_factor_data(n_days, n_stocks, dates)

feature_columns = [col for col in data_df.columns if col not in ['label', 'stock_code']]
X = data_df[feature_columns].values
y = data_df['label'].values

print(f"✓ 特征维度: {len(feature_columns)}")
print(f"✓ 样本数量: {data_df.shape[0]}")

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 存储结果
xgb_scores = []
lgb_scores = []
xgb_models = []
lgb_models = []

print("\n========== 模型训练与交叉验证 ==========")

for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    print(f"\nFold {fold + 1}/5")
    
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # XGBoost模型
    xgb_model = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_val)
    
    # 计算IC（信息系数）
    xgb_ic = np.corrcoef(xgb_pred, y_val)[0, 1]
    xgb_scores.append(xgb_ic)
    xgb_models.append(xgb_model)
    
    # LightGBM模型
    lgb_model = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    lgb_model.fit(X_train, y_train)
    lgb_pred = lgb_model.predict(X_val)
    
    lgb_ic = np.corrcoef(lgb_pred, y_val)[0, 1]
    lgb_scores.append(lgb_ic)
    lgb_models.append(lgb_model)
    
    print(f"  XGBoost IC: {xgb_ic:.4f}")
    print(f"  LightGBM IC: {lgb_ic:.4f}")

print(f"\n========== 交叉验证结果 ==========")
print(f"XGBoost 平均IC: {np.mean(xgb_scores):.4f} (+/- {np.std(xgb_scores):.4f})")
print(f"LightGBM 平均IC: {np.mean(lgb_scores):.4f} (+/- {np.std(lgb_scores):.4f})")

# 选择最佳模型
best_xgb_model = xgb_models[np.argmax(xgb_scores)]
best_lgb_model = lgb_models[np.argmax(lgb_scores)]

# 特征重要性
xgb_importance = best_xgb_model.feature_importances_
lgb_importance = best_lgb_model.feature_importances_

# 图1: 特征重要性对比
print("\n正在生成图1: 特征重要性对比...")
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# XGBoost特征重要性
xgb_imp_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': xgb_importance
}).sort_values('importance', ascending=True)

axes[0].barh(xgb_imp_df['feature'][-10:], xgb_imp_df['importance'][-10:], 
              color='blue', alpha=0.7)
axes[0].set_title('XGBoost Feature Importance (Top 10)', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Importance', fontsize=10)
axes[0].grid(True, alpha=0.3)

# LightGBM特征重要性
lgb_imp_df = pd.DataFrame({
    'feature': feature_columns,
    'importance': lgb_importance
}).sort_values('importance', ascending=True)

axes[1].barh(lgb_imp_df['feature'][-10:], lgb_imp_df['importance'][-10:], 
              color='green', alpha=0.7)
axes[1].set_title('LightGBM Feature Importance (Top 10)', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Importance', fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片1: feature_importance.png")
plt.close()

# 图2: 交叉验证IC对比
print("正在生成图2: 交叉验证IC对比...")
fig, ax = plt.subplots(figsize=(10, 6))

folds = range(1, 6)
xgb_scores_array = np.array(xgb_scores)
lgb_scores_array = np.array(lgb_scores)

ax.plot(folds, xgb_scores_array, 'bo-', linewidth=2, markersize=8, 
        label=f'XGBoost (Mean IC: {np.mean(xgb_scores):.3f})')
ax.plot(folds, lgb_scores_array, 'go-', linewidth=2, markersize=8, 
        label=f'LightGBM (Mean IC: {np.mean(lgb_scores):.3f})')

ax.set_xlabel('Fold', fontsize=12)
ax.set_ylabel('Information Coefficient (IC)', fontsize=12)
ax.set_title('Model Performance: Cross-Validation IC', fontsize=14, fontweight='bold')
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(fontsize=10, loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/cv_comparison.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片2: cv_comparison.png")
plt.close()

# 使用最后一次验证集进行分组分析
print("正在生成图3: 分组分析...")
X_train_full = X[:int(0.8 * len(X))]
y_train_full = y[:int(0.8 * len(y))]
X_test = X[int(0.8 * len(X)):]
y_test = y[int(0.8 * len(y)):]

# 训练最终模型
final_xgb = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1
)
final_xgb.fit(X_train_full, y_train_full)

final_lgb = lgb.LGBMRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)
final_lgb.fit(X_train_full, y_train_full)

# 预测
xgb_test_pred = final_xgb.predict(X_test)
lgb_test_pred = final_lgb.predict(X_test)

# 分组分析函数
def decile_analysis(predictions, actual_returns, n_groups=10):
    df = pd.DataFrame({
        'prediction': predictions,
        'actual': actual_returns
    })
    df['decile'] = pd.qcut(df['prediction'], q=n_groups, labels=False)
    decile_returns = df.groupby('decile')['actual'].mean()
    return decile_returns

# XGBoost分组分析
xgb_decile = decile_analysis(xgb_test_pred, y_test)
lgb_decile = decile_analysis(lgb_test_pred, y_test)

# 可视化分组分析
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(10), xgb_decile.values, color='blue', alpha=0.7, edgecolor='black')
axes[0].set_title('XGBoost: Decile Analysis', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Decile (0=Lowest, 9=Highest)', fontsize=10)
axes[0].set_ylabel('Average Actual Return', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].axhline(y=0, color='red', linestyle='--', linewidth=1)

axes[1].bar(range(10), lgb_decile.values, color='green', alpha=0.7, edgecolor='black')
axes[1].set_title('LightGBM: Decile Analysis', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Decile (0=Lowest, 9=Highest)', fontsize=10)
axes[1].set_ylabel('Average Actual Return', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].axhline(y=0, color='red', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig(f'{save_dir}/decile_analysis.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片3: decile_analysis.png")
plt.close()

# 图4: 预测vs实际散点图
print("正在生成图4: 预测vs实际散点图...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XGBoost
axes[0].scatter(xgb_test_pred, y_test, alpha=0.5, s=10)
axes[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
              'r--', linewidth=2)
axes[0].set_xlabel('Predicted Return', fontsize=10)
axes[0].set_ylabel('Actual Return', fontsize=10)
axes[0].set_title(f'XGBoost: Prediction vs Actual\nIC = {np.corrcoef(xgb_test_pred, y_test)[0, 1]:.3f}', 
                  fontsize=12, fontweight='bold')
axes[0].grid(True, alpha=0.3)

# LightGBM
axes[1].scatter(lgb_test_pred, y_test, alpha=0.5, s=10, color='green')
axes[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
              'r--', linewidth=2)
axes[1].set_xlabel('Predicted Return', fontsize=10)
axes[1].set_ylabel('Actual Return', fontsize=10)
axes[1].set_title(f'LightGBM: Prediction vs Actual\nIC = {np.corrcoef(lgb_test_pred, y_test)[0, 1]:.3f}', 
                  fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/prediction_vs_actual.png', dpi=300, bbox_inches='tight')
print("✓ 生成图片4: prediction_vs_actual.png")
plt.close()

# 图5: 封面图 - 梯度提升概念图
print("正在生成封面图...")
fig, ax = plt.subplots(figsize=(12, 8))

# 生成模拟数据
np.random.seed(42)
n_samples = 200
X_demo = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
y_demo = np.sin(X_demo).ravel() + np.random.normal(0, 0.1, n_samples)

# 模拟第1棵树、第2棵树、第k棵树的预测
y_pred_1 = 0.3 * np.sin(X_demo).ravel()
y_pred_2 = 0.5 * np.sin(X_demo).ravel()
y_pred_k = 0.9 * np.sin(X_demo).ravel()

# 绘制数据点
ax.scatter(X_demo, y_demo, s=20, alpha=0.6, color='gray', label='Data Points')

# 绘制每棵树的预测
ax.plot(X_demo, y_pred_1, 'r-', linewidth=2, alpha=0.6, label='Tree 1 Prediction')
ax.plot(X_demo, y_pred_2, 'g-', linewidth=2, alpha=0.6, label='Tree 1+2 Prediction')
ax.plot(X_demo, y_pred_k, 'b-', linewidth=3, label='Ensemble Prediction (K trees)')

# 绘制真实函数
ax.plot(X_demo, np.sin(X_demo).ravel(), 'k--', linewidth=2, label='True Function')

ax.set_xlabel('Feature', fontsize=14)
ax.set_ylabel('Target', fontsize=14)
ax.set_title('Gradient Boosting: Ensemble of Weak Learners', fontsize=16, fontweight='bold')
ax.legend(fontsize=10, loc='upper left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{save_dir}/cover.jpg', dpi=300, bbox_inches='tight')
print("✓ 生成封面图: cover.jpg")
plt.close()

print(f"\n✅ 所有配图已生成完成！")
print(f"图片保存位置: {save_dir}/")
print(f"共生成 5 张图片")
