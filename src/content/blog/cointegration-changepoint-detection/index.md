---
title: "协整关系变点检测与统计套利实战：捕捉配对交易的黄金窗口"
publishDate: '2026-06-05'
description: "协整关系变点检测与统计套利实战 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

## 引言：配对交易的痛点

传统配对交易依赖协整关系，但现实市场中，协整关系并非永恒不变。结构性断裂（Structural Breaks）随时可能发生，导致原本稳定的配对突然失效。

## 协整关系的脆弱性

### 为什么协整关系会断裂？

1. **公司基本面变化**：并购、重组、业务转型
2. **行业格局改变**：技术颠覆、政策冲击
3. **市场微观结构变化**：交易成本、流动性变化
4. **外部冲击**：金融危机、疫情等黑天鹅事件

### 传统方法的局限

- Engle-Granger检验：假设关系稳定
- Johansen检验：对变点不敏感
- 滚动窗口：滞后性强，容易错过变点

## 变点检测（Changepoint Detection）方法

### 1. CUSUM检验（累积和）

```python
# CUSUM变点检测（示例代码框架）
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import coint

# 假设有一对股票价格序列
stock_a = np.log(np.random.randn(1000).cumsum() + 100)
stock_b = np.log(np.random.randn(1000).cumsum() + 100)

# 计算残差
residuals = stock_a - stock_b

# CUSUM统计量
cusum = np.cumsum(residuals - residuals.mean())
threshold = 5 * np.std(residuals) * np.sqrt(len(residuals))

# 检测变点
changepoints = np.where(np.abs(cusum) > threshold)[0]
```

### 2. 贝叶斯变点检测

- 使用贝叶斯方法来估计变点位置
- 可以给出变点存在的概率
- 对噪声更鲁棒

### 3. 机器学习方法

- 使用LSTM检测非线性变点
- 随机森林识别结构断裂特征
- 图神经网络捕捉复杂依赖

## 实战框架：动态协整配对交易

### 步骤1：初始配对筛选

```python
# 初始配对筛选（示例代码框架）
import pandas as pd
from statsmodels.tsa.stattools import coint

def find_cointegrated_pairs(stocks_data):
    n = len(stocks_data.columns)
    pairs = []
    
    for i in range(n):
        for j in range(i+1, n):
            stock1 = stocks_data.iloc[:, i]
            stock2 = stocks_data.iloc[:, j]
            
            # Engle-Granger检验
            score, pvalue, _ = coint(stock1, stock2)
            
            if pvalue < 0.05:  # 协整关系显著
                pairs.append((stocks_data.columns[i], 
                             stocks_data.columns[j], 
                             pvalue))
    
    return pairs
```

### 步骤2：实时变点监控

- 每天收盘后重新检验协整关系
- 使用滚动窗口+CUSUM双重验证
- 设置变点预警阈值

### 步骤3：动态调整仓位

- 变点前：正常配对交易仓位
- 变点预警：降低仓位，准备撤离
- 变点确认：清仓，寻找新配对

### 步骤4：新配对发现

- 变点后重新扫描全市场
- 使用聚类算法发现新配对
- 快速验证并上线

## 案例分析：中石油与中石化的协整断裂

### 背景

2015年股灾前，中石油（601857.SH）与中石化（600028.SH）存在稳定协整关系。但2015年7月，协整关系突然断裂。

### 变点检测信号

1. **CUSUM信号**：2015年6月底突破阈值
2. **残差分布变化**：偏度从0.2变为1.5
3. **相关性下降**：相关系数从0.85降至0.4

### 交易策略调整

- **变点前**：持有配对交易仓位，年化收益12%
- **变点预警**：减仓50%，避免大幅回撤
- **变点后**：清仓并重新寻找配对

## 风险管理要点

1. **变点检测滞后**：任何方法都有滞后，需设置止损
2. **虚假变点**：市场噪音可能导致虚假信号，需多重验证
3. **流动性风险**：变点后配对股票可能流动性下降
4. **模型风险**：过度依赖历史数据，需结合基本面分析

## 结论

协整关系的变点检测是配对交易从理论走向实战的关键一环。只有建立动态的协整监控体系，才能在结构性变化发生时及时撤离，保护资本。

**记住：配对交易不是“设而不管”，而是需要持续监控的动态过程。**

![变点检测示意图](/images/cointegration-changepoint-detection/changepoint_diagram.jpg)

*CUSUM变点检测原理示意图*

![协整关系断裂](/images/cointegration-changepoint-detection/cointegration_break.jpg)

*协整关系结构性断裂示例图*
