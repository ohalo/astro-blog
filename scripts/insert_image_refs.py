#!/usr/bin/env python3
"""在文章适当位置插入图片引用"""

import re

# ===== Article 1: multi-factor-risk-decomposition =====
with open('/Users/halo/workspace/astro-blog/src/content/blog/multi-factor-risk-decomposition/index.md', 'r') as f:
    content1 = f.read()

# 在「Python 实现」代码块结束后插入第一张图
img1_insertion_point = content1.find('```python')
if img1_insertion_point != -1:
    # 找到第一个代码块结束位置
    code_end = content1.find('```', img1_insertion_point + 10)
    if code_end != -1:
        code_end = content1.find('\n', code_end) + 1
        images_block = """

![因子风险贡献条形图](/images/multi-factor-risk-decomposition/risk_contribution.png)
*图1：多因子模型各因子的风险贡献百分比（MKT因子占据最大风险敞口）*

![六大因子累计净值曲线](/images/multi-factor-risk-decomposition/factor_cumulative_returns.png)
*图2：模拟的六大因子累计净值曲线（252个交易日），可观察到不同因子的收益特征差异*

"""
        content1 = content1[:code_end] + images_block + content1[code_end:]

# 在「因子相关系数矩阵热图」段落附近插入第二张图（在 A 股实证部分前）
insert_before = "## A 股实证中的特殊考量"
if insert_before in content1:
    img_block2 = """![因子收益率相关系数矩阵](/images/multi-factor-risk-decomposition/factor_correlation_heatmap.png)
*图3：因子收益率相关系数矩阵热图（A股因子间相关性往往高于美股，需特别关注）*

"""
    content1 = content1.replace(insert_before, img_block2 + insert_before)

with open('/Users/halo/workspace/astro-blog/src/content/blog/multi-factor-risk-decomposition/index.md', 'w') as f:
    f.write(content1)
print("✅ Article 1 图片引用已插入")

# ===== Article 2: pair-trading-cointegration =====
with open('/Users/halo/workspace/astro-blog/src/content/blog/pair-trading-cointegration/index.md', 'r') as f:
    content2 = f.read()

# 在第一个 Python 代码块结束后插入协整检验图示
img_insert = content2.find('```python')
if img_insert != -1:
    code_end = content2.find('```', img_insert + 10)
    if code_end != -1:
        code_end = content2.find('\n', code_end) + 1
        images_block = """

![协整 vs 非协整序列的 ADF 检验对比](/images/pair-trading-cointegration/cointegration_adf_test.png)
*图1：左图为非平稳随机游走序列（ADF p-value > 0.05，协整不成立）；右图为平稳价差序列（ADF p-value < 0.01，协整成立）*

"""
        content2 = content2[:code_end] + images_block + content2[code_end:]

# 在「从协整到交易信号」章节结束后插入信号图
insert_before2 = "## A 股配对交易实证案例"
if insert_before2 in content2:
    img_block3 = """![配对交易 Z-score 信号图](/images/pair-trading-cointegration/pairs_trading_signals.png)
*图2：配对股票价差 Z-score 与交易信号示意图。当 Z-score 突破 ±2σ 时触发交易信号，回归 ±0.5σ 时平仓*

"""
    content2 = content2.replace(insert_before2, img_block3 + insert_before2)

# 在回测结果后插入累计收益图
insert_before3 = "## 配对交易的风险管理"
if insert_before3 in content2:
    img_block4 = """![配对交易策略累计净值](/images/pair-trading-cointegration/pairs_cumulative_return.png)
*图3：配对交易策略模拟累计净值（2年），展现均值回归策略的收益特征*

"""
    content2 = content2.replace(insert_before3, img_block4 + insert_before3)

with open('/Users/halo/workspace/astro-blog/src/content/blog/pair-trading-cointegration/index.md', 'w') as f:
    f.write(content2)
print("✅ Article 2 图片引用已插入")

print("\n🎉 两篇文章均已插入图片引用！")
