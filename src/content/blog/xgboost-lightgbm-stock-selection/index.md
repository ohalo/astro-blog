---
title: "XGBoost与LightGBM在量化选股中的应用"
date: 2026-06-21
description: "深入探讨XGBoost和LightGBM在量化选股中的实战应用，从特征工程到模型融合，附带完整的Python实现代码和性能对比。"
tags:
  - 机器学习
  - XGBoost
  - LightGBM
  - 量化选股
  - Python实战
cover: /images/xgboost-lightgbm-stock-selection/cover.jpg
---

# XGBoost与LightGBM在量化选股中的应用

在量化投资领域，机器学习方法正逐渐成为因子挖掘和股票筛选的重要工具。其中，**XGBoost** (Extreme Gradient Boosting) 和 **LightGBM** (Light Gradient Boosting Machine) 作为梯度提升树（GBDT）的两大主流框架，因其在结构化数据上的卓越表现，被广泛应用于量化选股、因子合成、风险模型等场景。

本文将深入探讨这两个框架在量化选股中的实战应用，从数据准备、特征工程、模型训练到组合构建，附带完整的Python实现代码。

## 一、为什么选择梯度提升树？

### 1.1 传统方法的局限

传统的量化选股方法包括：
- **多因子打分**：对各因子Z-score后等权或IC加权
- **线性回归**：用因子预测下期收益，构建多空组合
- **逻辑回归**：预测涨跌方向，输出概率

这些方法存在局限：
- 假设因子与收益线性相关（现实往往非线性）
- 难以捕捉因子间的交互效应
- 对异常值敏感
- 特征工程依赖人工经验

### 1.2 梯度提升树的优势

**XGBoost** 和 **LightGBM** 属于集成学习中的Boosting家族，核心思想：
- 串行训练多棵决策树，每棵树拟合前一棵树的残差
- 通过梯度下降最小化损失函数
- 最终预测 = 所有树的预测加权求和

优势：
- **非线性建模**：自动学习因子与收益的非线性关系
- **特征交互**：自动捕捉因子间的交互效应（如"低PE + 高ROE"组合）
- **鲁棒性**：对缺失值、异常值不敏感
- **可解释性**：通过特征重要性、SHAP值解释模型

## 二、数据准备与特征工程

### 2.1 数据获取

量化选股的典型数据包括：
- **基本面因子**：PE、PB、ROE、营收增速等
- **技术面因子**：动量、波动率、换手率等
- **分析师因子**：一致预期、评级调整等
- **另类数据**：新闻情感、卫星图像等

以下代码展示如何用`akshare`库获取A股数据：

```python
import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# 获取A股列表
print("正在获取A股列表...")
stock_info = ak.stock_info_a_code_name()
print(f"共获取 {len(stock_info)} 只A股")

# 获取基本面数据
def get_fundamental_data(date):
    """获取给定日期的基本面数据"""
    try:
        # 估值因子
        pe_data = ak.stock_a_indicator_lg(symbol="沪深A股", date=date)
        pb_data = ak.stock_a_lg_indicator(symbol="沪深A股", date=date)
        
        # 财务指标
        fin_data = ak.stock_financial_abstract_ths(symbol="沪深A股", indicator="按报告期")
        
        return pe_data, pb_data, fin_data
    except Exception as e:
        print(f"获取数据失败: {e}")
        return None, None, None

# 示例：获取2024-12-31的数据
date_str = "20241231"
pe_data, pb_data, fin_data = get_fundamental_data(date_str)

print("\n=== 数据预览 ===")
if pe_data is not None:
    print("PE数据形状:", pe_data.shape)
    print(pe_data.head())
```

**注意**：`akshare`是第三方库，可能需要处理数据缺失、复权调整等问题。实盘中建议使用Wind、聚宽等专业数据接口。

### 2.2 特征工程

机器学习模型的效果很大程度上取决于特征质量。量化选股中常见的特征工程操作：

```python
def build_features(stock_data):
    """
    构建机器学习特征
    
    Parameters:
    -----------
    stock_data : DataFrame
        原始股票数据，包含价格、财务等字段
        
    Returns:
    --------
    features : DataFrame
        特征矩阵
    """
    features = pd.DataFrame(index=stock_data.index)
    
    # === 1. 估值因子 ===
    features['pe'] = stock_data['pe']
    features['pb'] = stock_data['pb']
    features['ps'] = stock_data['ps']
    
    # 取对数（处理右偏分布）
    features['log_pe'] = np.log(stock_data['pe'].clip(lower=1))
    features['log_pb'] = np.log(stock_data['pb'].clip(lower=0.1))
    
    # === 2. 盈利因子 ===
    features['roe'] = stock_data['roe']
    features['roa'] = stock_data['roa']
    features['gross_margin'] = stock_data['gross_margin']
    
    # 盈利稳定性（过去4个季度ROE的标准差）
    if 'roe_quarterly' in stock_data.columns:
        features['roe_stability'] = stock_data['roe_quarterly'].rolling(4).std()
    
    # === 3. 成长因子 ===
    features['revenue_growth'] = stock_data['revenue_growth']
    features['profit_growth'] = stock_data['profit_growth']
    
    # 成长加速（当期增速 - 上期增速）
    features['revenue_growth_acc'] = stock_data['revenue_growth'].diff()
    
    # === 4. 技术因子 ===
    # 动量（过去20个交易日收益率）
    features['momentum_20d'] = stock_data['close'].pct_change(20)
    
    # 波动率（过去20个交易日收益率标准差）
    features['volatility_20d'] = stock_data['return'].rolling(20).std()
    
    # 换手率
    features['turnover'] = stock_data['turnover']
    
    # === 5. 交叉因子 ===
    features['pe_x_roe'] = features['pe'] * features['roe']  # 估值 × 盈利
    features['momentum_x_vol'] = features['momentum_20d'] * features['volatility_20d']  # 动量 × 波动率
    
    # === 6. 行业哑变量 ===
    if 'industry' in stock_data.columns:
        dummies = pd.get_dummies(stock_data['industry'], prefix='ind')
        features = pd.concat([features, dummies], axis=1)
    
    # === 7. 标准化 ===
    numeric_cols = features.select_dtypes(include=[np.number]).columns
    features[numeric_cols] = features[numeric_cols].apply(
        lambda x: (x - x.mean()) / x.std() if x.std() != 0 else 0
    )
    
    # 处理缺失值
    features = features.fillna(0)
    
    return features

# 示例：构建特征
print("\n=== 特征工程示例 ===")
# 假设已有stock_data
# features = build_features(stock_data)
# print(f"特征维度: {features.shape}")
# print(f"特征列表: {features.columns.tolist()[:10]}...")
```

### 2.3 标签构建

机器学习需要标签（y）。量化选股中常见的标签构建方法：

```python
def build_labels(stock_data, method='return', horizon=20):
    """
    构建机器学习标签
    
    Parameters:
    -----------
    stock_data : DataFrame
        股票数据，需包含'close'列
    method : str
        标签方法：'return'（回归）、'direction'（分类）、'quantile'（分位数）
    horizon : int
        预测期限（交易日）
        
    Returns:
    --------
    labels : Series
        标签
    """
    if method == 'return':
        # 回归：预测下期收益率
        future_return = stock_data['close'].pct_change(horizon).shift(-horizon)
        labels = future_return
        
    elif method == 'direction':
        # 分类：预测涨跌方向
        future_return = stock_data['close'].pct_change(horizon).shift(-horizon)
        labels = (future_return > 0).astype(int)  # 1: 涨, 0: 跌
        
    elif method == 'quantile':
        # 分位数：将收益率分为5档（多分类）
        future_return = stock_data['close'].pct_change(horizon).shift(-horizon)
        labels = pd.qcut(future_return, q=5, labels=[0, 1, 2, 3, 4])
        
    return labels

# 示例
print("\n=== 标签构建示例 ===")
# labels_return = build_labels(stock_data, method='return', horizon=20)
# labels_direction = build_labels(stock_data, method='direction', horizon=20)
# print(f"回归标签范围: [{labels_return.min():.4f}, {labels_return.max():.4f}]")
# print(f"分类标签分布: {labels_direction.value_counts()}")
```

## 三、XGBoost模型训练与调优

### 3.1 基础模型训练

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt

# 假设已有特征和标签
# X = features.values
# y = labels_return.values

def train_xgboost(X, y, task='regression'):
    """
    训练XGBoost模型
    
    Parameters:
    -----------
    X : array-like
        特征矩阵
    y : array-like
        标签
    task : str
        任务类型：'regression' 或 'classification'
        
    Returns:
    --------
    model : xgboost.Booster
        训练好的模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 转换为DMatrix（XGBoost专用数据结构）
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 设置参数
    if task == 'regression':
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'rmse'
        }
    else:  # classification
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc'
        }
    
    # 训练模型
    evals = [(dtrain, 'train'), (dtest, 'test')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=evals,
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # 预测
    y_pred = model.predict(dtest)
    
    # 评估
    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\n测试集RMSE: {rmse:.6f}")
    else:
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        print(f"\n测试集AUC: {auc:.4f}")
        print(f"测试集准确率: {acc:.4f}")
    
    return model

# 示例
print("\n=== XGBoost训练示例 ===")
# model_xgb = train_xgboost(X, y, task='regression')
```

### 3.2 特征重要性分析

```python
def analyze_feature_importance(model, feature_names):
    """
    分析XGBoost特征重要性
    
    Parameters:
    -----------
    model : xgboost.Booster
        训练好的模型
    feature_names : list
        特征名称列表
    """
    # 获取特征重要性（三种方法）
    importance_gain = model.get_score(importance_type='gain')
    importance_weight = model.get_score(importance_type='weight')
    importance_cover = model.get_score(importance_type='cover')
    
    # 转换为DataFrame
    df_importance = pd.DataFrame({
        'feature': feature_names,
        'importance_gain': [importance_gain.get(f, 0) for f in feature_names],
        'importance_weight': [importance_weight.get(f, 0) for f in feature_names],
        'importance_cover': [importance_cover.get(f, 0) for f in feature_names]
    })
    
    # 按gain排序
    df_importance = df_importance.sort_values('importance_gain', ascending=False)
    
    print("\n=== Top 10 重要特征 (Gain) ===")
    print(df_importance.head(10).to_string(index=False))
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top 20特征（Gain）
    top_n = 20
    top_features = df_importance.head(top_n)
    axes[0].barh(range(top_n), top_features['importance_gain'][::-1])
    axes[0].set_yticks(range(top_n))
    axes[0].set_yticklabels(top_features['feature'][::-1])
    axes[0].set_xlabel('Importance (Gain)')
    axes[0].set_title(f'Top {top_n} 特征重要性 (Gain)')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # 特征重要性分布
    axes[1].hist(df_importance['importance_gain'], bins=50, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Importance (Gain)')
    axes[1].set_ylabel('特征数量')
    axes[1].set_title('特征重要性分布')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return df_importance

# 示例
# feature_importance = analyze_feature_importance(model_xgb, feature_names)
```

### 3.3 超参数调优

XGBoost有多个重要超参数：
- `max_depth`：树的最大深度
- `eta`：学习率
- `subsample`：样本采样比例
- `colsample_bytree`：特征采样比例
- `lambda`：L2正则化
- `alpha`：L1正则化

使用网格搜索或贝叶斯优化进行调优：

```python
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor

def tune_xgboost(X, y):
    """
    使用网格搜索调优XGBoost超参数
    """
    model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    param_grid = {
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.3],
        'n_estimators': [100, 200, 300],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    
    grid_search = GridSearchCV(
        model,
        param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X, y)
    
    print("\n=== 最佳参数 ===")
    print(grid_search.best_params_)
    print(f"最佳得分 (负MSE): {grid_search.best_score_:.6f}")
    
    return grid_search.best_estimator_

# 示例
# best_model = tune_xgboost(X, y)
```

**注意**：网格搜索计算量大，实盘中建议使用`RandomizedSearchCV`或贝叶斯优化（`optuna`库）。

## 四、LightGBM模型训练与对比

### 4.1 LightGBM基础训练

LightGBM是微软开源的梯度提升框架，相比XGBoost：
- **更快的训练速度**：使用直方图算法，减少特征分裂计算
- **更低的内存消耗**：离散化连续特征
- **更好的准确率**：支持类别特征、单边梯度采样等

```python
import lightgbm as lgb

def train_lightgbm(X, y, task='regression'):
    """
    训练LightGBM模型
    
    Parameters:
    -----------
    X : array-like
        特征矩阵
    y : array-like
        标签
    task : str
        任务类型：'regression' 或 'classification'
        
    Returns:
    --------
    model : lightgbm.Booster
        训练好的模型
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)
    
    # 设置参数
    if task == 'regression':
        params = {
            'objective': 'regression',
            'metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
    else:  # classification
        params = {
            'objective': 'binary',
            'metric': 'auc',
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'verbose': -1
        }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        early_stopping_rounds=10,
        verbose_eval=20
    )
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    if task == 'regression':
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\n测试集RMSE: {rmse:.6f}")
    else:
        auc = roc_auc_score(y_test, y_pred)
        acc = accuracy_score(y_test, (y_pred > 0.5).astype(int))
        print(f"\n测试集AUC: {auc:.4f}")
        print(f"测试集准确率: {acc:.4f}")
    
    return model

# 示例
print("\n=== LightGBM训练示例 ===")
# model_lgb = train_lightgbm(X, y, task='regression')
```

### 4.2 XGBoost vs LightGBM 性能对比

```python
def compare_models(X, y, task='regression', n_runs=5):
    """
    对比XGBoost和LightGBM的性能
    
    Parameters:
    -----------
    X, y : array-like
        特征和标签
    task : str
        任务类型
    n_runs : int
        重复运行次数（取平均）
    """
    import time
    
    results = {
        'XGBoost': {'time': [], 'metric': []},
        'LightGBM': {'time': [], 'metric': []}
    }
    
    for i in range(n_runs):
        print(f"\n=== 第 {i+1}/{n_runs} 次运行 ===")
        
        # XGBoost
        start_time = time.time()
        model_xgb = train_xgboost(X, y, task)
        xgb_time = time.time() - start_time
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42+i
        )
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        model_xgb = xgb.train(
            {'objective': 'reg:squarederror', 'eta': 0.1},
            dtrain,
            num_boost_round=100,
            verbose_eval=False
        )
        y_pred_xgb = model_xgb.predict(dtest)
        
        results['XGBoost']['time'].append(xgb_time)
        if task == 'regression':
            results['XGBoost']['metric'].append(np.sqrt(mean_squared_error(y_test, y_pred_xgb)))
        else:
            results['XGBoost']['metric'].append(roc_auc_score(y_test, y_pred_xgb))
        
        # LightGBM
        start_time = time.time()
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        model_lgb = lgb.train(
            {'objective': 'regression', 'learning_rate': 0.1, 'verbose': -1},
            train_data,
            num_boost_round=100,
            verbose_eval=False
        )
        y_pred_lgb = model_lgb.predict(X_test)
        
        lgb_time = time.time() - start_time
        results['LightGBM']['time'].append(lgb_time)
        if task == 'regression':
            results['LightGBM']['metric'].append(np.sqrt(mean_squared_error(y_test, y_pred_lgb)))
        else:
            results['LightGBM']['metric'].append(roc_auc_score(y_test, y_pred_lgb))
    
    # 汇总结果
    print("\n" + "="*60)
    print("性能对比汇总")
    print("="*60)
    
    for model_name in ['XGBoost', 'LightGBM']:
        avg_time = np.mean(results[model_name]['time'])
        std_time = np.std(results[model_name]['time'])
        avg_metric = np.mean(results[model_name]['metric'])
        std_metric = np.std(results[model_name]['metric'])
        
        print(f"\n{model_name}:")
        print(f"  平均训练时间: {avg_time:.3f}s (+/- {std_time:.3f}s)")
        if task == 'regression':
            print(f"  平均RMSE: {avg_metric:.6f} (+/- {std_metric:.6f})")
        else:
            print(f"  平均AUC: {avg_metric:.4f} (+/- {std_metric:.4f})")
    
    # 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练时间对比
    models = list(results.keys())
    avg_times = [np.mean(results[m]['time']) for m in models]
    std_times = [np.std(results[m]['time']) for m in models]
    
    axes[0].bar(models, avg_times, yerr=std_times, 
                capsize=10, color=['steelblue', 'darkorange'])
    axes[0].set_ylabel('训练时间 (秒)')
    axes[0].set_title('训练时间对比')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 性能指标对比
    if task == 'regression':
        avg_metrics = [np.mean(results[m]['metric']) for m in models]
        std_metrics = [np.std(results[m]['metric']) for m in models]
        metric_name = 'RMSE'
    else:
        avg_metrics = [np.mean(results[m]['metric']) for m in models]
        std_metrics = [np.std(results[m]['metric']) for m in models]
        metric_name = 'AUC'
    
    axes[1].bar(models, avg_metrics, yerr=std_metrics, 
                capsize=10, color=['steelblue', 'darkorange'])
    axes[1].set_ylabel(metric_name)
    axes[1].set_title(f'{metric_name}对比')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# 示例
# compare_models(X, y, task='regression', n_runs=5)
```

## 五、从预测到组合构建

模型训练完成后，如何将预测结果转化为实际投资组合？

### 5.1 基于预测得分的选股

```python
def build_portfolio_from_predictions(model, X, stock_codes, top_n=50):
    """
    根据模型预测得分构建投资组合
    
    Parameters:
    -----------
    model : 训练好的模型
    X : array-like
        特征矩阵（当前期）
    stock_codes : list
        股票代码列表
    top_n : int
        选择前N只股票
        
    Returns:
    --------
    portfolio : DataFrame
        投资组合（股票代码 + 预测得分）
    """
    # 预测
    if isinstance(model, xgb.Booster):
        dtest = xgb.DMatrix(X)
        predictions = model.predict(dtest)
    elif isinstance(model, lgb.Booster):
        predictions = model.predict(X)
    else:
        predictions = model.predict(X)
    
    # 构建DataFrame
    portfolio = pd.DataFrame({
        'stock_code': stock_codes,
        'predicted_return': predictions
    })
    
    # 按预测收益排序，选择Top N
    portfolio = portfolio.sort_values('predicted_return', ascending=False)
    portfolio = portfolio.head(top_n)
    
    # 等权分配
    portfolio['weight'] = 1.0 / top_n
    
    print(f"\n=== 投资组合（Top {top_n}）===")
    print(portfolio.head(10).to_string(index=False))
    print(f"\n...\n共 {len(portfolio)} 只股票")
    print(f"预测收益范围: [{portfolio['predicted_return'].min():.4f}, "
          f"{portfolio['predicted_return'].max():.4f}]")
    
    return portfolio

# 示例
# portfolio = build_portfolio_from_predictions(model_xgb, X_current, stock_codes, top_n=50)
```

### 5.2 风险模型整合

单纯按预测得分选股可能导致组合风险集中（如全部是科技股）。需加入风险约束：

```python
def build_portfolio_with_risk_constraints(model, X, stock_codes, stock_industries, top_n=50):
    """
    构建带风险约束的投资组合
    
    Parameters:
    -----------
    同 build_portfolio_from_predictions
    stock_industries : list
        股票所属行业列表
    """
    # 初步筛选
    portfolio = build_portfolio_from_predictions(model, X, stock_codes, top_n=100)
    portfolio['industry'] = stock_industries[:100]
    
    # 行业分散化：每个行业最多选N只
    max_per_industry = 10
    portfolio_diversified = []
    
    for industry in portfolio['industry'].unique():
        industry_stocks = portfolio[portfolio['industry'] == industry]
        portfolio_diversified.append(industry_stocks.head(max_per_industry))
    
    portfolio_diversified = pd.concat(portfolio_diversified, ignore_index=True)
    portfolio_diversified = portfolio_diversified.head(top_n)
    
    # 重新分配权重
    portfolio_diversified['weight'] = 1.0 / len(portfolio_diversified)
    
    print(f"\n=== 行业分布 ===")
    print(portfolio_diversified['industry'].value_counts().head(10))
    
    return portfolio_diversified

# 示例
# portfolio_risk = build_portfolio_with_risk_constraints(
#     model_xgb, X_current, stock_codes, stock_industries, top_n=50
# )
```

### 5.3 回测框架

```python
def backtest_portfolio(model, features_history, returns_history, 
                      stock_codes, initial_capital=1000000, top_n=50):
    """
    回测机器学习选股策略
    
    Parameters:
    -----------
    model : 训练好的模型
    features_history : DataFrame
        历史特征数据（多期）
    returns_history : DataFrame
        历史收益率数据（用于计算实际收益）
    stock_codes : list
        股票代码列表
    initial_capital : float
        初始资金
    top_n : int
        每期选择前N只股票
        
    Returns:
    --------
    performance : dict
        回测性能指标
    """
    dates = features_history.index.unique()
    portfolio_values = [initial_capital]
    
    for i, date in enumerate(dates[:-1]):
        # 当前期特征
        X_current = features_history.loc[date].values
        
        # 预测下期收益
        if isinstance(model, xgb.Booster):
            dcurrent = xgb.DMatrix(X_current)
            predictions = model.predict(dcurrent)
        else:
            predictions = model.predict(X_current)
        
        # 选择Top N股票
        portfolio = pd.DataFrame({
            'stock_code': stock_codes,
            'predicted_return': predictions
        })
        portfolio = portfolio.sort_values('predicted_return', ascending=False).head(top_n)
        selected_stocks = portfolio['stock_code'].tolist()
        
        # 计算下期实际收益
        next_date = dates[i+1]
        actual_returns = returns_history.loc[next_date, selected_stocks].mean()
        
        # 更新组合价值
        new_value = portfolio_values[-1] * (1 + actual_returns)
        portfolio_values.append(new_value)
        
        if (i+1) % 20 == 0:
            print(f"日期: {next_date}, 组合价值: {new_value:.2f}")
    
    # 计算性能指标
    portfolio_values = pd.Series(portfolio_values, index=dates)
    total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) - 1
    annual_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (252/len(dates)) - 1
    
    returns = portfolio_values.pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252)
    
    rolling_max = portfolio_values.expanding().max()
    drawdown = (portfolio_values - rolling_max) / rolling_max
    max_drawdown = drawdown.min()
    
    performance = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'portfolio_values': portfolio_values
    }
    
    return performance

# 示例
# performance = backtest_portfolio(model_xgb, features_history, returns_history, stock_codes)
# print(f"\n总收益: {performance['total_return']:.2%}")
# print(f"年化收益: {performance['annual_return']:.2%}")
# print(f"夏普比率: {performance['sharpe_ratio']:.2f}")
# print(f"最大回撤: {performance['max_drawdown']:.2%}")
```

## 六、实战建议与常见陷阱

### 6.1 数据泄露（Data Leakage）

**问题**：用未来数据预测过去（如用本期财务数据预测本期收益）

**解决**：
- 严格区分训练集和测试集的时间边界
- 使用滚动窗口交叉验证（Rolling Window CV）
- 标签要用**未来**收益率，特征要用**当前**或**过去**数据

```python
def rolling_window_cv(X, y, model, n_splits=5):
    """
    滚动窗口交叉验证（防止数据泄露）
    """
    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)
    
    cv_scores = []
    
    for i in range(n_splits):
        # 训练集：[:split_i]
        # 测试集：[split_i:split_i+fold_size]
        split_i = (i+1) * fold_size
        
        X_train = X[:split_i]
        y_train = y[:split_i]
        X_test = X[split_i:split_i+fold_size]
        y_test = y[split_i:split_i+fold_size]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = np.sqrt(mean_squared_error(y_test, y_pred))
        cv_scores.append(score)
        
        print(f"Fold {i+1}: RMSE = {score:.6f}")
    
    print(f"\n平均RMSE: {np.mean(cv_scores):.6f} (+/- {np.std(cv_scores):.6f})")
    return cv_scores
```

### 6.2 过拟合

**问题**：模型在训练集表现优异，测试集表现差

**解决**：
- 正则化（`lambda`、`alpha`）
- 早停（`early_stopping_rounds`）
- 特征选择（去除低重要性特征）
- 限制模型复杂度（`max_depth`、`min_child_weight`）

### 6.3 非平稳性

**问题**：股票市场分布随时间变化，模型老化

**解决**：
- 定期重新训练（如每月）
- 使用在线学习（Incremental Learning）
- 结合集成方法（Ensemble）

```python
def online_learning(X_new, y_new, model):
    """
    在线学习：用新数据更新模型
    """
    if isinstance(model, xgb.Booster):
        # XGBoost在线学习
        dtrain = xgb.DMatrix(X_new, label=y_new)
        model = xgb.train(
            model.attr,  # 继承原参数
            dtrain,
            num_boost_round=10,
            xgb_model=model  # 从原模型继续训练
        )
    elif isinstance(model, lgb.Booster):
        # LightGBM在线学习
        train_data = lgb.Dataset(X_new, label=y_new)
        model = lgb.train(
            model.params,
            train_data,
            num_boost_round=10,
            init_model=model  # 从原模型继续训练
        )
    
    return model
```

## 七、总结

本文详细介绍了XGBoost和LightGBM在量化选股中的应用，核心要点：

1. **梯度提升树适合量化选股**：非线性建模、自动特征交互、鲁棒性强
2. **特征工程是关键**：估值、盈利、成长、技术等多维度因子，需标准化和清洗
3. **防止数据泄露**：严格时间分割，使用滚动窗口验证
4. **组合构建需风险约束**：行业分散化、市值中性等
5. **持续监控与更新**：定期重训练，防止模型老化

**延伸阅读**：
- SHAP值解释模型预测
- 深度学习（LSTM、Transformer）在量化选股中的应用
- 强化学习用于动态调仓

---

**代码示例下载**：本文完整代码已上传至 [GitHub](https://github.com/example/quant-blog-code)

**免责声明**：本文仅供学习交流，不构成投资建议。机器学习模型有风险，实盘需谨慎。

