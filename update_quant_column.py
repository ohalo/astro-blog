#!/usr/bin/env python3
"""
更新quant-column.md文件，在最新文章顶部添加2篇新文章
"""
import re

# 读取文件
with open('/Users/halo/workspace/astro-blog/src/content/pages/quant-column.md', 'r', encoding='utf-8') as f:
    content = f.read()

# 新文章列表（要添加到顶部）
new_articles = """### 2026-06-19 新发布

- [2026-06-19 - 因子拥挤度监测与规避：量化投资中的隐形风险](/blog/factor-crowding/) - 🚀 🔴 深入探讨因子拥挤度的成因、多维度量化监测指标（资金流向/收益率离散度/换手率/自相关）、综合评分系统及规避策略（动态因子配置/分散执行/拥挤度约束），提供完整的Python实现代码（FactorCrowdingMonitor类/预警系统/动态权重调整）及2025年成长因子拥挤实战案例（高阶）
- [2026-06-19 - 统计套利：均值回归策略的深度解析与Python实战](/blog/statistical-arbitrage-mean-reversion/) - 🚀 🔴 从协整检验到配对交易，详解统计套利的核心原理、实战策略和风险控制，附完整Python代码示例。涵盖Engle-Granger检验、交易信号生成、多资产PCA降维、动态仓位管理（凯利公式）及完整策略回测框架（高阶）

"""

# 在"## 最新文章"后插入新文章
pattern = r'(## 最新文章\n\n)'
replacement = r'\1' + new_articles

new_content = re.sub(pattern, replacement, content, count=1)

# 写回文件
with open('/Users/halo/workspace/astro-blog/src/content/pages/quant-column.md', 'w', encoding='utf-8') as f:
    f.write(new_content)

print("✅ quant-column.md 更新完成！")
print("已添加2篇新文章到顶部")
