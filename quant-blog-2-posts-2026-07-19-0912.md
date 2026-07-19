# 量化博客自动发布任务 (2026-07-19 09:12)

## 目标
按 cron `quant-blog-2-posts-simplified` 流程自动生成 2 篇不重复量化博客并发布到 Vercel。

## 主题选择
`select_blog_topics.py` 输出（主池已耗尽，走内置 fresh 兜底，slug 全库唯一）：
1. 波动率目标择时 → `volatility-target-timing`
2. ADF 与 KPSS 联合检验 → `adf-kpss-stationarity`

## 关键执行
- **配图**:各 4 张真实 matplotlib 图（种子 20260719），均由自包含 Python 真实计算，非占位符。
  - 波动率目标 (`gen_volatility_target_timing.py`): 净值曲线 / 动态杠杆时序 / 滚动 1 年 Sharpe / 崩盘分段。关键数：基础资产年化波动 30.6%、最大回撤 −86.4%；VT 压到 15.9%(−48%)、回撤 −55.8%(−35%)，崩盘段 BH −85.8% vs VT −55.0%。
  - ADF/KPSS (`gen_adf_kpss_stationarity.py`): 三序列轨迹 / ADF 检验 / KPSS 检验 / 联合判定矩阵。关键数: 随机游走 ADF −1.96(p=0.31)/KPSS 1.97(p=0.01)→差分平稳 I(1)；AR(1) ADF −11.93(p≈0)/KPSS 0.16(p=0.10)→平稳；FI(d=0.45) ADF −8.43(p≈0)/KPSS 0.70(p=0.01)→长记忆。
- **踩坑修复(重要)**:
  - VT 初版合成基础资产波动仅 ~16%，VM 无去杠杆空间，波动降幅只有 5%、结论不成立。改为「高且时变波动(年化 ~28%+ 崩盘 ×3.2) + 近零漂移」设定，使 VM 真正发挥(降幅 48%/回撤 −35%)，并诚实声明「VT 省冗余风险非白送 alpha」。
  - 修正 ADF 脚本里错误的背景色十六进制(7 位非法)、清理 zip 误用与图 4 散点变量。
- **文章**: 各 ≥1300 CJK 中文字 + 完整可运行 Python 片段(已复验两文代码片段实跑、数字与正文一致) + 真实陷阱披露 + 诚实声明，frontmatter 严格对齐现有规范(title/description/publishDate/tags/language:Chinese/difficulty)。
- **quant-column.md**: 在「最新文章」顶部插入新日期分组 `### 2026-07-19 新发布（波动率目标择时 / ADF-KPSS 平稳性联合检验）`,2 条链接 + 简介。
- **构建**: `npx astro build` 通过 (1907 页, exit 0, 无错误)。

## 提交与部署
- commit: `feat: add 2 quant articles 2026-07-19` → push 成功 (fa30bca)
- Vercel 自动构建约 3 分钟后传播完成。

## 部署验证 (最终, 全部 200)
- quant-column: 200
- /blog/volatility-target-timing/: 200 (初 308 重定向→跟随后 200)
- /blog/adf-kpss-stationarity/: 200
- 8 张配图全部 200 (远程确认)
- 两文正文标志性短语命中,渲染正确。

## 结论
2 篇文章成功生成、配图真实、更新专栏页、推送并验证部署通过。全程未编造虚假收益指标,对「VT 只缩放敞口不创造 alpha / 杠杆约束截断 / σ̂ 滞后」与「ADF/KPSS 单向误判 / 长记忆灰区 / 伪回归」均做了诚实披露,符合量化策略专家的 Iron Rules。
