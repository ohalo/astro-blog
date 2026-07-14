# 任务产出：偏度偏好因子博客文章

## 目标
为量化博客生成文章「偏度偏好因子：散户为何为左尾买单、机构如何收割」(slug: skewness-preference-factor)，含完整 Python 代码 + 4 张真实配图。

## 交付物
1. **文章**：`src/content/blog/skewness-preference-factor/index.md`
   - frontmatter 严格参照 `vrp-timing-allocation/index.md`（title/description/publishDate/tags/language/difficulty）
   - 正文中文字数：2577 字（不含代码，达标 2000–3000）
   - 含完整可运行 Python：skew()、simulate_panel()、long_short()、perf()
   - 内容覆盖：经济学直觉(彩票偏好/预期偏度 iskew/MAX/Boyer-Mitton-Vorkink coskewness)、因子构造(形成期/持有期分离)、五分位组合、多空组合、5+ 类真实陷阱(前视/幸存者/微盘流动性/偏度估计噪声/散户泡沫挤兑 + 市值暴露/卖空约束)
2. **配图脚本**：`public/images/skewness-preference-factor/generate_images.py`
   - matplotlib Agg、中文字体 ["PingFang SC","Arial Unicode MS","SimHei","DejaVu Sans"]、axes.unicode_minus=False
3. **4 张真实 PNG**（均体现偏度偏好效应：高偏度未来收益更低）：
   - `skew_quintile_returns.png` — 五分位组合未来收益柱状图
   - `skew_distribution.png` — 截面偏度分布(右偏厚尾) + 偏度vs未来收益负斜率散点
   - `skew_long_short.png` — 多空组合净值曲线
   - `skew_drawdown.png` — 多空组合回撤曲线

## 关键数值（合成数据回测）
- 五分位未来收益：Q1 +0.39% → Q5 −0.28% （单调反转）
- 多空月均 +0.64%、年化 ~7.8%、夏普 1.7、最大回撤 −8.6%、t 值 ~14
- 散点斜率约 −0.11%/单位偏度

## 验收
- index.md 存在且字数达标 ✓
- public/images/skewness-preference-factor/ 下 4 张真实 PNG ✓
- 正文图片路径与实际文件名一一对应 ✓
