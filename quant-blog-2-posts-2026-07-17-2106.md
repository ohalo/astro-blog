# 量化博客自动发布任务 (2026-07-17)

## 目标
按 cron `quant-blog-2-posts-simplified` 流程自动生成 2 篇不重复的量化交易博客并发布到 Vercel。

## 主题选择 (select_blog_topics.py)
- 主题池已用尽,脚本自动兜底(校验 slug 全库不存在)
- 1. 日内成交量季节性：用 U 型曲线预测盘中流动性窗口 → `intraday-volume-seasonality`
- 2. 反演优化隐含风险：从组合的边际风险反推各大类资产的真实暴露 → `reverse-optimization-implied`

## 执行要点
- **配图**:各 4 张真实 matplotlib 图(固定种子 20260717),均由自包含 Python 真实计算
  - Article1 (`ivs_*`): U 型曲线 / 累计成交量S型 / 执行成本代理 / VWAP对比。关键数: U型两端/中段 3.84x、午盘成本比开盘高 2.93x、VWAP vs 均匀 IS 1.48bp vs 1.67bp(省11.4%)
  - Article2 (`rov_*`): 权重vs风险贡献 / 风险贡献条形 / 反演隐含收益 / ERC对比。关键数: 股票50%权重→84.1%风险、反演 γ=3.367、ERC 风险贡献精确 16.667%×6
- **踩坑修复**: ERC 原始固定点迭代 `w∝1/(Σw)` 在含负相关(国债-股票=-0.20)时循环不收敛(风险贡献差34%) → 改用 SLSQP 约束最小化,得到精确相等 RC
- **文章**: 2000-3000字, 含 Python 代码示例 + 六类真实陷阱 + 诚实声明, 严格对齐现有 frontmatter 格式 (title/description/publishDate/tags/language:Chinese/difficulty:intermediate)
- **quant-column.md**: 在「最新文章」顶部插入新日期分组 (2026-07-17 新发布 日内成交量季节性 / 反演优化隐含风险), 2 条链接+简介

## 提交说明
- `git add -A` 顺带纳入了之前 cron 遗留未提交的两套已完稿文章 (arfima-long-memory, vix-mean-reversion) — 均属应发布内容, 一并提交
- commit: `feat: add 2 quant articles 2026-07-17` → push 成功 (882b6d9..7bdc1b4)

## 部署验证
- 验证 curl: quant-column 200; 两篇新文章 200; 各 2 张配图 200
- Vercel 构建约 100s 后传播完成 (先 404 后 200)

## 结论
2 篇文章已成功生成、配图真实、更新专栏页、推送并验证部署通过。
