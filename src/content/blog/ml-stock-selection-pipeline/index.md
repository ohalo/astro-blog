---
title: "用 Python 从零搭建机器学习选股流水线：因子挖掘到信号生成"
publishDate: '2026-07-20'
description: "用 Python 从零搭建机器学习选股流水线：因子挖掘到信号生成 - halo的技术博客"
tags:
 - AI工具
 - 量化投资
language: Chinese
---

量化选股的核心逻辑并不复杂：**找到那些历史上能带来超额收益的规律，然后假设这些规律在未来依然有效**。但要把这个逻辑工程化落地，需要一套完整的数据处理、特征工程、模型训练和信号生成的流水线。本文用一个真实的 Python 项目，展示这套流水线从设计到实现的完整过程。

## 整体架构：五个模块

这套选股流水线分为五个核心模块：

1. **数据源模块**：整合股票价格、财务数据、因子库
2. **特征工程模块**：将原始数据转化为模型可学习的特征
3. **模型训练模块**：用历史数据训练预测模型
4. **信号生成模块**：对新数据输出选股信号
5. **回测验证模块**：在历史数据上验证策略有效性

整个流程用 Python 实现，主要依赖：`pandas`、`scikit-learn`、`LightGBM`、`Backtrader`。

## 数据源：打通 A 股数据闭环

A 股数据的获取有几条成熟路径：

```python
import akshare as ak
import pandas as pd

# 获取股票列表
stock_info = ak.stock_info_a_code_name()

# 获取日线行情（调整后价格）
daily_data = ak.stock_zh_a_hist(
    symbol="000001",
    period="daily",
    start_date="20180101",
    end_date="20260630",
    adjust="qfq"
)

# 获取财务数据
financial_data = ak.stock_financial_analysis_indicator(symbol="000001")
```

`akshare` 是 A 股数据获取最方便的库，支持大多数主流数据源。对于生产环境，建议将数据存储在本地数据库（PostgreSQL 或 MySQL），避免重复下载，同时保证数据一致性。

![机器学习与股票数据](/images/ml-stock-selection-pipeline/machine-learning-stocks.jpg)

## 特征工程：这是最重要的一步

**特征工程决定了模型的上限，模型只是逼近这个上限**。在选股场景下，特征可以分为三大类：

**第一类：价量特征**
- 收益率序列的统计量（均值、方差、偏度、峰度）
- 技术指标（MA、RSI、MACD、Bollinger Bands）
- 换手率、成交量异常
- 动量因子（不同周期的动量）

```python
def calc_price_features(df):
    """计算价量特征"""
    features = {}
    
    # 日收益率
    returns = df['close'].pct_change()
    
    # 短期动量
    features['momentum_5d'] = df['close'].pct_change(5)
    features['momentum_20d'] = df['close'].pct_change(20)
    
    # 波动率
    features['volatility_20d'] = returns.rolling(20).std()
    
    # 成交量异常
    features['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    
    # RSI
    delta = returns.rolling(14).mean()
    gain = delta.apply(lambda x: x if x > 0 else 0)
    loss = delta.apply(lambda x: -x if x < 0 else 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    features['rsi_14d'] = 100 - (100 / (1 + rs))
    
    return features
```

**第二类：财务特征**
- 市盈率（PE）、市净率（PB）、市销率（PS）
- 营收增速、净利润增速
- 资产负债率、流动比率
- 毛利率、净利率的变化趋势

**第三类：另类特征**
- 分析师一致预期数据（EPS 预期调整）
- 股东人数变化（筹码集中度代理指标）
- 龙虎榜数据（机构行为代理）

![股票图表分析](/images/ml-stock-selection-pipeline/stock-charts-analysis.jpg)

## 模型选择：LightGBM 是目前最优解

在选股场景下，LightGBM 是目前工业界最主流的选择，原因有三：

- **训练速度快**：可以在分钟内完成百万级样本的训练
- **支持稀疏特征**：A股有大量缺失数据，LightGBM 对此鲁棒
- **可解释性好**：特征重要性分析帮助理解模型逻辑

```python
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit

# 准备数据
X = feature_df.drop(['next_return', 'code', 'date'], axis=1)
y = feature_df['next_return']

# 时序交叉验证（避免未来数据泄露）
tscv = TimeSeriesSplit(n_splits=5)

params = {
    'objective': 'regression',
    'metric': 'mae',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_estimators': 500,
    'early_stopping_rounds': 50
}

# 训练
model = lgb.LGBMRegressor(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
)
```

**一个关键原则**：永远使用时序交叉验证（TimeSeriesSplit），而非随机切分。随机切分会让模型「看到未来数据」，导致虚高的回测表现。

## 信号生成：从预测到持仓

模型输出的是对未来收益的预测值，要转化为可执行的选股信号，还需要几步处理：

```python
def generate_signals(model, new_data):
    """生成选股信号"""
    # 1. 模型预测
    predictions = model.predict(new_data)
    
    # 2. 标准化处理（分行业/市值桶 neutralization）
    predictions = neutralize(predictions, new_data['industry'], new_data['mktcap'])
    
    # 3. 排序分组
    signals = pd.qcut(predictions, q=10, labels=False, duplicates='drop')
    
    # 4. 输出 top 20% 作为做多组合
    selected = signals[signals >= 8].index
    
    return selected
```

`neutralize`（中性化）是专业量化系统中非常重要的一步：去除行业偏好和市值偏好，确保选出的股票不是因为「偏好大盘股」或「偏好某个行业」，而是因为个体Alpha。

![Python机器学习流水线](/images/ml-stock-selection-pipeline/python-ml-pipeline.jpg)

## 回测：残酷的真相检验

回测不是终点，但它是检验策略有效性的第一道关卡。以下是几个最重要的回测指标：

- **年化收益率（Annualized Return）**：策略的盈利速度
- **夏普比率（Sharpe Ratio）**：单位风险的超额收益，越高越好
- **最大回撤（Max Drawdown）**：历史上从最高点到最低点的最大跌幅，这是最重要的风控指标
- **胜率（Win Rate）**和**盈亏比（Profit/Loss Ratio）**：交易层面的统计

```python
import backtrader as bt

class MLSelectionStrategy(bt.Strategy):
    def __init__(self):
        self.order = None
        
    def next(self):
        if self.order:
            return
            
        signals = self.datas[0].signals
        
        if signals > 0.8:  # 强信号
            self.order = self.buy()
        elif signals < 0.2:  # 弱信号
            self.order = self.sell()
```

## 实战中最容易踩的三个坑

**坑一：前视偏差（Look-ahead Bias）**

使用未来数据计算特征。比如在 T 时刻计算「未来 5 日涨幅」作为标签时不小心包含了 T+1 到 T+5 的数据。解决办法：严格确保特征计算和标签生成使用不重叠的时间窗口。

**坑二：幸存者偏差（Survivorship Bias）**

只用当前存活的公司做回测，忽略了历史上退市/破产的公司。真实市场中买到退市股是真实的损失，而幸存者偏差会让回测收益虚高。解决方案：使用包含退市公司的完整历史数据库。

**坑三：过拟合**

模型在训练集上表现很好，但样本外一塌糊涂。解决方法：减少特征数量、增加正则化、用更严格的交叉验证。记住：**简单的策略往往比复杂的策略更持久**。

## 结语

机器学习选股流水线是一条从数据到决策的完整链路，每一个环节都有大量细节值得深挖。本文展示的是一个最小可行版本的框架，实际生产环境还需要考虑：
- 实时数据接入与信号更新频率
- 交易成本（佣金、印花税、滑点）的精确建模
- 仓位管理与风险预算
- 多市场、多资产的扩展

但只要这条流水线跑通了，剩下的就是持续迭代和优化。量化投资的本质是**用系统化的方式管理人性的弱点**，而机器学习给这套系统装上了一个更强的大脑。
