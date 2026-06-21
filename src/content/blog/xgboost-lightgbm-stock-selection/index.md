---
title: "XGBoost与LightGBM在量化选股中的应用"
description: "深入探讨XGBoost和LightGBM在量化选股中的实战应用，从特征工程到模型训练，从因子合成到组合优化，提供完整的Python代码示例"
publishDate: 2026-06-21
category: quant
tags:
  - 机器学习
  - XGBoost
  - LightGBM
  - 量化选股
  - 因子模型
  - Python实战
  - 梯度提升树
cover: /images/xgboost-lightgbm-stock-selection/cover.png
---

# XGBoost与LightGBM在量化选股中的应用

## 引言

在传统量化选股中，**多因子模型**（Multiple Factor Model）一直是主流方法。通过筛选有价值的因子（如价值、动量、质量等），构建线性模型预测股票收益。

然而，现实中因子与收益之间的关系往往是**非线性**的，且存在复杂的**因子交互效应**。这时候，**梯度提升树（Gradient Boosting Trees）** 算法就展现出强大优势。

**XGBoost**（Extreme Gradient Boosting）和**LightGBM**（Light Gradient Boosting Machine）作为两大主流梯度提升框架，在Kaggle竞赛和工业界都取得了巨大成功。

本文将深入探讨：
1. 梯度提升树的基本原理
2. XGBoost vs LightGBM：算法对比
3. 量化选股的特征工程（因子构建）
4. 模型训练与调参实战
5. 完整的Python量化选股系统
6. 性能评估与实盘部署

---

## 一、梯度提升树原理

### 1.1 从决策树到梯度提升

**决策树**通过递归分割特征空间，构建if-then规则。但单棵树容易过拟合，泛化能力差。

**Boosting思想**：
- 训练多棵弱分类器（浅层决策树）
- 每棵树修正前一棵树的错误
- 最终预测 = 所有树的加权求和

**梯度提升（Gradient Boosting）**：
- 第 $t$ 棵树拟合的是**负梯度**（残差的近似）
- 损失函数：$L(y, F(x)) = \frac{1}{2}(y - F(x))^2$（回归）
- 更新：$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x)$

### 1.2 XGBoost的目标函数

XGBoost在损失函数中加入了**正则化项**：

$$
\text{Obj} = \sum_{i=1}^n L(y_i, \hat{y}_i) + \sum_{k=1}^K \Omega(f_k)
$$

其中：
- $L(y_i, \hat{y}_i)$：损失函数（如均方误差）
- $\Omega(f_k) = \gamma T + \frac{1}{2}\lambda \|\omega\|^2$：正则化项
  - $T$：叶子节点数
  - $\omega$：叶子权重
  - $\gamma, \lambda$：超参数

**二阶泰勒展开**：
XGBoost利用损失函数的**一阶和二阶导数**进行优化，收敛更快。

### 1.3 LightGBM的改进

LightGBM在XGBoost基础上做了两大创新：

1. **Histogram-based算法**：
   - 将连续特征离散化为直方图（如255个bin）
   - 大幅减少计算复杂度：从 $O(\text{#data} \times \text{#features})$ 降到 $O(\text{#bins} \times \text{#features})$

2. **Leaf-wise生长策略**：
   - 传统：Level-wise（按层生长）
   - LightGBM：选择**损失下降最大**的叶子进行分裂
   - 优点：收敛更快；缺点：容易过拟合（需用 `max_depth` 限制）

---

## 二、量化选股的特征工程

### 2.1 因子体系构建

在量化选股中，**因子（Factor）** 是模型的输入特征。一个完整的因子体系通常包括：

| 因子类别 | 具体因子 | 计算方式 |
|---------|---------|---------|
| **价值因子** | 市盈率（PE） | 市值 / 净利润 |
|  | 市净率（PB） | 市值 / 净资产 |
|  | 市销率（PS） | 市值 / 营业收入 |
| **动量因子** | 过去N日收益率 | $r_{t-N:t}$ |
|  | 动量加速度 | $r_{t-5:t} - r_{t-20:t-5}$ |
| **质量因子** | ROE | 净利润 / 净资产 |
|  | 资产周转率 | 营业收入 / 总资产 |
|  | 财务杠杆 | 总资产 / 净资产 |
| **波动率因子** | 历史波动率 | $\sigma(r_{t-N:t})$ |
|  | 下行波动率 | $\sigma(r_{r<0})$ |
| **流动性因子** | 换手率 | 成交量 / 流通股本 |
|  | Amihud非流动性 | $|r_t| / \text{Volume}_t$ |

### 2.2 Python实现：因子计算

```python
import numpy as np
import pandas as pd
import tushare as ts  # 或使用akshare、yfinance

# ========== 1. 获取价格数据 ==========
def get_price_data(tickers, start_date, end_date):
    """
    获取股票ADJUSTED收盘价
    推荐数据源：Tushare Pro (A股), yfinance (美股), akshare (免费A股)
    """
    data = {}
    for ticker in tickers:
        # 示例：使用yfinance获取美股数据
        df = yf.download(ticker, start=start_date, end=end_date)
        data[ticker] = df['Adj Close']
    
    prices = pd.DataFrame(data)
    return prices

# ========== 2. 计算收益率因子 ==========
def calculate_momentum_factors(prices, periods=[5, 20, 60]):
    """计算动量因子"""
    returns = prices.pct_change()
    factors = pd.DataFrame(index=returns.index)
    
    for period in periods:
        factors[f'momentum_{period}d'] = returns.rolling(period).sum()
    
    # 动量加速度
    factors['momentum_accel'] = factors['momentum_5d'] - factors['momentum_20d']
    
    return factors

# ========== 3. 计算波动率因子 ==========
def calculate_volatility_factors(returns, windows=[20, 60]):
    """计算波动率因子"""
    factors = pd.DataFrame(index=returns.index)
    
    for window in windows:
        # 历史波动率
        factors[f'volatility_{window}d'] = returns.rolling(window).std() * np.sqrt(252)
        
        # 下行波动率 (只考虑负收益)
        downside_returns = returns.copy()
        downside_returns[returns > 0] = 0
        factors[f'downside_vol_{window}d'] = downside_returns.rolling(window).std() * np.sqrt(252)
    
    return factors

# ========== 4. 计算技术指标 ==========
def calculate_technical_factors(prices):
    """计算技术指标因子"""
    factors = pd.DataFrame(index=prices.index)
    
    # RSI (相对强弱指数)
    def calculate_rsi(prices, period=14):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    for ticker in prices.columns:
        factors[f'RSI_{ticker}'] = calculate_rsi(prices[ticker])
    
    # MACD
    def calculate_macd(prices, fast=12, slow=26, signal=9):
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal, adjust=False).mean()
        return macd, signal_line
    
    # 为简洁，这里只展示单个标的
    macd, signal_line = calculate_macd(prices.iloc[:, 0])
    factors['MACD'] = macd
    factors['MACD_signal'] = signal_line
    
    return factors

# ========== 5. 因子标准化 ==========
def standardize_factors(factors):
    """横截面标准化（每个交易日，对所有股票标准化）"""
    return factors.sub(factors.mean(axis=1), axis=0).div(factors.std(axis=1), axis=0)

# ========== 主程序 ==========
if __name__ == "__main__":
    # 参数设置
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'JPM', 'GS', 'BAC', 'WFC', 'C']
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    # 获取数据
    prices = get_price_data(tickers, start_date, end_date)
    returns = prices.pct_change()
    
    # 计算因子
    momentum_factors = calculate_momentum_factors(prices)
    volatility_factors = calculate_volatility_factors(returns)
    technical_factors = calculate_technical_factors(prices)
    
    # 合并所有因子
    all_factors = pd.concat([momentum_factors, volatility_factors, technical_factors], axis=1)
    
    # 标准化
    all_factors_standardized = standardize_factors(all_factors)
    
    print(f"因子维度: {all_factors_standardized.shape}")
    print(f"因子列表: {list(all_factors_standardized.columns)}")
```

---

## 三、XGBoost与LightGBM模型训练

### 3.1 数据准备与标签构建

在量化选股中，**标签（Label）** 的构建至关重要。常见做法：

1. **未来N日收益率**：
   $$y_t = r_{t+1:t+N}$$
   - 优点：直观，符合经济意义
   - 缺点：噪声大，预测难度大

2. **排名分位数**：
   $$y_t = \text{rank}(r_{t+1:t+N})$$
   - 优点：降低噪声，关注相对排名
   - 缺点：丢失绝对收益信息

3. **分类标签**（上涨/下跌/横盘）：
   $$y_t = \begin{cases}
   2 & \text{if } r_{t+1:t+N} > \alpha \\
   1 & \text{if } |r_{t+1:t+N}| \leq \alpha \\
   0 & \text{if } r_{t+1:t+N} < -\alpha
   \end{cases}$$

### 3.2 XGBoost模型训练实战

```python
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score

# ========== 1. 准备训练数据 ==========
def prepare_training_data(factors, prices, forward_days=5):
    """
    构建训练数据：因子 -> 未来N日收益率
    """
    # 计算未来N日收益率
    future_returns = prices.shift(-forward_days) / prices - 1
    
    # 转换为分类标签（上涨=1, 下跌=0）
    labels = (future_returns > 0).astype(int)
    
    # 合并因子和标签
    X = factors.dropna()
    y = labels.loc[X.index]
    
    return X, y

# ========== 2. 时间序列交叉验证 ==========
def time_series_cv(X, y, n_splits=5):
    """
    时间序列交叉验证（防止未来信息泄露）
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        yield X_train, X_test, y_train, y_test

# ========== 3. XGBoost模型训练 ==========
def train_xgboost(X_train, y_train, X_test, y_test):
    """
    训练XGBoost分类器
    """
    # 转换数据格式
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # 参数设置
    params = {
        'objective': 'binary:logistic',  # 二分类
        'max_depth': 6,                   # 最大深度
        'eta': 0.1,                      # 学习率
        'subsample': 0.8,                # 样本采样率
        'colsample_bytree': 0.8,         # 特征采样率
        'eval_metric': 'auc',             # 评估指标
        'random_state': 42
    }
    
    # 训练模型
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # 预测
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    return model

# ========== 4. 特征重要性分析 ==========
def plot_feature_importance(model, feature_names, top_n=20):
    """
    绘制特征重要性图
    """
    importance = model.get_score(importance_type='weight')
    importance_df = pd.DataFrame({
        'feature': list(importance.keys()),
        'importance': list(importance.values())
    }).sort_values('importance', ascending=False)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importance_df['importance'][:top_n], align='center')
    plt.yticks(range(top_n), importance_df['feature'][:top_n])
    plt.xlabel('Importance')
    plt.title('Top {} Feature Importance (XGBoost)'.format(top_n))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df

# ========== 主程序 ==========
if __name__ == "__main__":
    # 准备数据
    X, y = prepare_training_data(all_factors_standardized, prices, forward_days=5)
    
    # 时间序列交叉验证
    for fold, (X_train, X_test, y_train, y_test) in enumerate(time_series_cv(X, y)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}")
        print(f"{'='*50}")
        
        model = train_xgboost(X_train, y_train, X_test, y_test)
        
        if fold == 0:  # 只绘制第一次的特征重要性
            plot_feature_importance(model, X.columns)
```

### 3.3 LightGBM模型训练实战

```python
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

# ========== LightGBM模型训练 ==========
def train_lightgbm(X_train, y_train, X_test, y_test):
    """
    训练LightGBM分类器
    """
    # 创建数据集
    train_data = lgb.Dataset(X_train, label=y_train)
    test_data = lgb.Dataset(X_test, label=y_test)
    
    # 参数设置
    params = {
        'objective': 'binary',           # 二分类
        'metric': 'auc',                 # 评估指标
        'boosting_type': 'gbdt',         # 传统GBDT
        'num_leaves': 31,                # 叶子节点数
        'learning_rate': 0.1,            # 学习率
        'feature_fraction': 0.8,         # 特征采样率
        'bagging_fraction': 0.8,         # 样本采样率
        'bagging_freq': 5,               # 每5次迭代进行一次bagging
        'verbose': -1,                   # 不输出中间信息
        'random_state': 42
    }
    
    # 训练模型
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, test_data],
        valid_names=['train', 'test'],
        early_stopping_rounds=10,
        verbose_eval=10
    )
    
    # 预测
    y_pred_proba = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba > 0.5).astype(int)
    
    # 评估
    auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    
    return model

# ========== LightGBM特征重要性 ==========
def plot_lgb_feature_importance(model, feature_names, top_n=20):
    """
    绘制LightGBM特征重要性图
    """
    importance = model.feature_importance(importance_type='gain')
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)
    
    # 可视化
    plt.figure(figsize=(12, 6))
    plt.barh(range(top_n), importance_df['importance'][:top_n], align='center')
    plt.yticks(range(top_n), importance_df['feature'][:top_n])
    plt.xlabel('Importance (Gain)')
    plt.title('Top {} Feature Importance (LightGBM)'.format(top_n))
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('lightgbm_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return importance_df
```

---

## 四、模型融合与组合优化

### 4.1 XGBoost + LightGBM 模型融合

单一模型容易陷入局部最优，**模型融合**（Ensemble）可以提升泛化能力。

**常见融合方法**：
1. **简单平均**：$\hat{y} = \frac{1}{2}(\hat{y}_{xgb} + \hat{y}_{lgb})$
2. **加权平均**：$\hat{y} = \alpha \hat{y}_{xgb} + (1-\alpha) \hat{y}_{lgb}$
3. **Stacking**：用线性模型学习两个模型的权重

```python
# ========== 模型融合 ==========
def ensemble_predict(model_xgb, model_lgb, X_test, alpha=0.5):
    """
    XGBoost + LightGBM 简单加权平均
    """
    # XGBoost预测
    dtest = xgb.DMatrix(X_test)
    y_pred_xgb = model_xgb.predict(dtest)
    
    # LightGBM预测
    y_pred_lgb = model_lgb.predict(X_test, num_iteration=model_lgb.best_iteration)
    
    # 加权平均
    y_pred_ensemble = alpha * y_pred_xgb + (1 - alpha) * y_pred_lgb
    
    return y_pred_ensemble

# 评估融合模型
y_pred_ensemble = ensemble_predict(model_xgb, model_lgb, X_test, alpha=0.5)
y_pred_class = (y_pred_ensemble > 0.5).astype(int)

ensemble_accuracy = accuracy_score(y_test, y_pred_class)
print(f"Ensemble Accuracy: {ensemble_accuracy:.4f}")
```

### 4.2 从预测到组合构建

模型输出是**预测概率**（上涨概率），需要转换为**持仓权重**。

**常见方法**：
1. **Top-K多选股**：
   - 按预测概率排序，选择Top K只股票
   - 等权配置或按预测概率加权

2. **因子合成**：
   - 将预测概率视为一个新因子
   - 与其他传统因子（价值、动量等）合成综合因子
   - 按综合因子IC（信息系数）加权

3. **均值-方差优化**（Markowitz）：
   - 目标：$\max_w w^T \mu - \frac{\gamma}{2} w^T \Sigma w$
   - 约束：$w \geq 0$, $\sum w = 1$

```python
# ========== Top-K多选股策略 ==========
def top_k_strategy(predictions, tickers, k=10):
    """
    选择预测概率最高的K只股票
    """
    # 创建预测DataFrame
    pred_df = pd.DataFrame({'ticker': tickers, 'pred_prob': predictions})
    
    # 按预测概率排序
    pred_df_sorted = pred_df.sort_values('pred_prob', ascending=False)
    
    # 选择Top K
    top_k = pred_df_sorted.head(k)
    
    # 等权配置
    weights = pd.Series(1/k, index=top_k['ticker'])
    
    return weights

# ========== 回测框架 ==========
def backtest_strategy(weights_dict, returns, transaction_cost=0.001):
    """
    简单回测框架
    """
    portfolio_returns = []
    
    for date in weights_dict.keys():
        if date in returns.index:
            weights = weights_dict[date]
            daily_return = (weights * returns.loc[date]).sum()
            
            # 扣除交易成本
            turnover = weights.diff().abs().sum()
            cost = turnover * transaction_cost
            
            portfolio_returns.append(daily_return - cost)
    
    portfolio_returns = pd.Series(portfolio_returns, index=returns.index[:len(portfolio_returns)])
    
    # 计算累计收益
    cumulative_returns = (1 + portfolio_returns).cumprod()
    
    # 绩效指标
    sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
    max_drawdown = (cumulative_returns / cumulative_returns.cummax() - 1).min()
    
    return {
        'returns': portfolio_returns,
        'cumulative': cumulative_returns,
        'sharpe': sharpe,
        'max_drawdown': max_drawdown
    }
```

---

## 五、实战案例：完整的量化选股系统

### 5.1 系统架构

一个完整的量化选股系统包括：
1. **数据层**：价格、财务、另类数据
2. **因子层**：因子计算、标准化、去极值、中性化
3. **模型层**：XGBoost/LightGBM训练、预测
4. **组合层**：持仓权重优化
5. **回测层**：性能评估、风险控制

### 5.2 完整代码示例

```python
import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# ========== 完整量化选股系统 ==========
class QuantStockSelector:
    def __init__(self, tickers, start_date, end_date, forward_days=5):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.forward_days = forward_days
        self.factors = None
        self.model_xgb = None
        self.model_lgb = None
        self.prices = None
        self.returns = None
        
    def load_data(self):
        """加载价格数据"""
        print("正在下载价格数据...")
        data = {}
        for ticker in self.tickers:
            try:
                df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False)
                if not df.empty:
                    data[ticker] = df['Adj Close']
                else:
                    print(f"警告: {ticker} 无数据")
            except Exception as e:
                print(f"错误: 下载 {ticker} 失败 - {e}")
        
        if not data:
            raise ValueError("没有成功下载任何数据！")
        
        self.prices = pd.DataFrame(data)
        self.returns = self.prices.pct_change().dropna()
        print(f"✅ 数据加载完成: {self.prices.shape}")
        return self
    
    def calculate_all_factors(self):
        """计算所有因子"""
        print("正在计算因子...")
        
        # 1. 动量因子
        momentum_factors = pd.DataFrame(index=self.returns.index)
        for period in [5, 10, 20, 60]:
            momentum_factors[f'momentum_{period}d'] = self.returns.rolling(period).sum()
        
        # 2. 波动率因子
        volatility_factors = pd.DataFrame(index=self.returns.index)
        for window in [20, 60]:
            volatility_factors[f'vol_{window}d'] = self.returns.rolling(window).std() * np.sqrt(252)
        
        # 3. 成交量因子
        volume_factors = pd.DataFrame(index=self.returns.index)
        for ticker in self.tickers:
            if ticker in self.prices.columns:
                # 这里简化，实际需要获取成交量数据
                pass
        
        # 4. 技术指标
        technical_factors = pd.DataFrame(index=self.returns.index)
        for ticker in self.tickers:
            if ticker in self.prices.columns:
                prices_ticker = self.prices[ticker]
                
                # RSI
                delta = prices_ticker.diff()
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                technical_factors[f'RSI_{ticker}'] = rsi
                
                # 均线
                ma20 = prices_ticker.rolling(20).mean()
                technical_factors[f'MA20_ratio_{ticker}'] = prices_ticker / ma20 - 1
        
        # 合并所有因子
        self.factors = pd.concat([momentum_factors, volatility_factors, technical_factors], axis=1)
        
        # 横截面标准化
        self.factors = self.factors.sub(self.factors.mean(axis=1), axis=0).div(self.factors.std(axis=1), axis=0)
        
        print(f"✅ 因子计算完成: {self.factors.shape}")
        return self
    
    def prepare_labels(self):
        """准备训练标签（未来N日收益率）"""
        future_returns = self.prices.shift(-self.forward_days) / self.prices - 1
        labels = (future_returns > 0).astype(int)  # 二分类：上涨=1, 下跌=0
        
        # 对齐因子和标签
        self.factors_aligned = self.factors.dropna()
        self.labels_aligned = labels.loc[self.factors_aligned.index]
        
        print(f"✅ 标签准备完成: {self.labels_aligned.shape}")
        return self
    
    def train_models(self, test_size=0.2):
        """训练XGBoost和LightGBM模型"""
        print("正在训练模型...")
        
        # 时间序列分割（不用随机分割）
        split_idx = int(len(self.factors_aligned) * (1 - test_size))
        
        X_train = self.factors_aligned.iloc[:split_idx]
        X_test = self.factors_aligned.iloc[split_idx:]
        y_train = self.labels_aligned.iloc[:split_idx]
        y_test = self.labels_aligned.iloc[split_idx:]
        
        # 训练XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        params = {
            'objective': 'binary:logistic',
            'max_depth': 6,
            'eta': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'eval_metric': 'auc',
            'random_state': 42
        }
        
        self.model_xgb = xgb.train(params, dtrain, num_boost_round=100,
                                   evals=[(dtrain, 'train'), (dtest, 'test')],
                                   early_stopping_rounds=10, verbose_eval=False)
        
        # 训练LightGBM
        train_data = lgb.Dataset(X_train, label=y_train)
        test_data = lgb.Dataset(X_test, label=y_test)
        
        params_lgb = {
            'objective': 'binary',
            'metric': 'auc',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'verbose': -1,
            'random_state': 42
        }
        
        self.model_lgb = lgb.train(params_lgb, train_data, num_boost_round=100,
                                   valid_sets=[test_data], valid_names=['test'],
                                   early_stopping_rounds=10, verbose_eval=False)
        
        # 评估
        y_pred_xgb = self.model_xgb.predict(dtest) > 0.5
        y_pred_lgb = self.model_lgb.predict(X_test, num_iteration=self.model_lgb.best_iteration) > 0.5
        
        acc_xgb = accuracy_score(y_test, y_pred_xgb)
        acc_lgb = accuracy_score(y_test, y_pred_lgb)
        
        print(f"✅ XGBoost准确率: {acc_xgb:.4f}")
        print(f"✅ LightGBM准确率: {acc_lgb:.4f}")
        
        return self
    
    def predict_and_backtest(self):
        """预测并回测"""
        print("正在回测...")
        
        # 使用测试集进行回测
        split_idx = int(len(self.factors_aligned) * 0.8)
        X_test = self.factors_aligned.iloc[split_idx:]
        
        # 预测
        dtest = xgb.DMatrix(X_test)
        y_pred_xgb = self.model_xgb.predict(dtest)
        y_pred_lgb = self.model_lgb.predict(X_test, num_iteration=self.model_lgb.best_iteration)
        
        # 融合预测
        y_pred_ensemble = 0.5 * y_pred_xgb + 0.5 * y_pred_lgb
        
        # 构建组合：买入预测概率最高的前10只股票
        portfolio_returns = []
        for i in range(len(X_test)):
            date = X_test.index[i]
            pred_probs = y_pred_ensemble[i]
            
            # 选择Top 10
            top_10_idx = np.argsort(pred_probs)[-10:]
            selected_tickers = X_test.columns[top_10_idx]
            
            # 等权配置
            weights = np.ones(10) / 10
            
            # 计算组合收益
            if date in self.returns.index:
                daily_return = (weights * self.returns.loc[date, selected_tickers]).sum()
                portfolio_returns.append(daily_return)
        
        portfolio_returns = pd.Series(portfolio_returns, index=X_test.index[:len(portfolio_returns)])
        
        # 计算绩效
        cumulative = (1 + portfolio_returns).cumprod()
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        max_dd = (cumulative / cumulative.cummax() - 1).min()
        
        print(f"\n{'='*50}")
        print("回测结果")
        print(f"{'='*50}")
        print(f"夏普比率: {sharpe:.4f}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"累计收益: {cumulative.iloc[-1]:.2%}")
        
        return {
            'returns': portfolio_returns,
            'cumulative': cumulative,
            'sharpe': sharpe,
            'max_drawdown': max_dd
        }

# ========== 主程序 ==========
if __name__ == "__main__":
    # 参数设置
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 
               'TSLA', 'NVDA', 'JPM', 'GS', 'BAC']
    start_date = '2023-01-01'
    end_date = '2024-12-31'
    
    # 初始化系统
    qss = QuantStockSelector(tickers, start_date, end_date, forward_days=5)
    
    # 执行完整流程
    results = (qss
               .load_data()
               .calculate_all_factors()
               .prepare_labels()
               .train_models()
               .predict_and_backtest()
    )
    
    # 可视化
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # 累计收益曲线
    axes[0].plot(results['cumulative'], linewidth=2, label='Strategy')
    axes[0].set_title('Cumulative Returns (XGBoost + LightGBM)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Cumulative Returns', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    axes[0]..legend()
    
    # 回撤曲线
    drawdown = results['cumulative'] / results['cumulative'].cummax() - 1
    axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red', label='Drawdown')
    axes[1].plot(drawdown, color='darkred', linewidth=1)
    axes[1].set_title('Drawdown Curve', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Drawdown', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig('backtest_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 量化选股系统执行完成！")
