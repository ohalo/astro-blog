# 量化博客自动发布任务 (2026-07-19 02:04)

## 目标
按 cron `quant-blog-2-posts-simplified` 流程自动生成 2 篇不重复量化博客并发布到 Vercel。

## 关键前置问题 & 修复
- `select_blog_topics.py` 主题池**已全库耗尽（851 篇）**，兜底池也用尽，脚本 `assert len==2` 直接报错退出。
- 修复：在脚本 `fallback_pool` 之前插入 `fresh_pool`，新增 8 个经全库扫描确认**未发布**的全新 slug（Lasso / 共形预测 / 概念漂移 / 隐含相关性 / 贝叶斯模型平均 / 分数差分 / 卡尔曼跟踪 / GMM 资产定价）。脚本恢复输出 2 个主题。
- 选定主题（均与最近 10 篇不重复）：
  1. Lasso 与弹性网络变量选择 → `lasso-elasticnet-variable-selection`
  2. 共形预测在量化中的应用 → `conformal-prediction-quant`

## 执行要点
- **配图**:各 4 张真实 matplotlib 图(种子 20260719),均由自包含 Python 真实计算,非占位符
  - Lasso (`gen_lasso_elasticnet.py`): 系数路径 / ElasticNet OOS-MSE / 稀疏度阶梯 / OOS R² 对比。关键数: 全因子 OOS R²=−0.539、Ridge −0.207、Lasso +0.136、ElasticNet +0.134；真信号 15 个。
  - 共形 (`gen_conformal_prediction.py`): 区间覆盖 / 有限样本有效性 / 漂移下朴素 vs 共形 / 区间宽度 vs α。关键数: 图1 经验覆盖 89.0%(目标 90%)；图2 名义 0.80→0.798、0.90→0.899、0.95→0.950；图3 漂移测试 朴素 70.4% vs 共形 73.9%。
- **踩坑修复(重要)**:
  - 共形图2 初版经验覆盖严重偏低 (0.41 vs 0.80) —— 原因是 `split_conformal` 把每次迭代内的预测与**全局** `y_te`(来自图1) 对齐比较,维度/样本错位。改为显式传 `y_te` 参数后,覆盖恢复正常 (0.798/0.847/0.899/0.950)。已拒绝编造"接近名义"的幻象,先定位 bug 再重跑。
  - Lasso 图1 初版 R/S 在价格**水平**估计会糊成一坨 —— 已在本次实现中直接采用增量(收益)路径,干净分离真信号。
- **文章**: 各 ≥2000 字中文(CJK: Lasso 2061 / 共形 2052),含 Python 代码示例 + 真实陷阱披露 + 诚实声明,严格对齐现有 frontmatter (title/description/publishDate/tags/language:Chinese/difficulty)。
- **quant-column.md**: 在「最新文章」顶部插入新日期分组 `### 2026-07-19 新发布（Lasso 变量选择 / 共形预测）`,2 条链接 + 简介。
- **构建**: `npx astro build` 通过 (1859 页, exit 0, 无错误)。

## 提交与部署
- commit: `feat: add 2 quant articles 2026-07-19` → push 成功 (0b1ecd2)
- Vercel 自动构建约 3 分钟后传播完成 (先 308/404, 后 200)

## 部署验证 (最终)
- quant-column: 200
- /blog/lasso-elasticnet-variable-selection/: 308→200 (trailing slash 重定向)
- /blog/conformal-prediction-quant/: 308→200
- 8 张配图全部 200 (远程确认)
- 文章正文渲染确认(标志性短语命中)

## 结论
2 篇文章成功生成、配图真实、更新专栏页、推送并验证部署通过。全程未编造虚假收益指标,对"Lasso 只解决过拟合不保证盈利"与"共形保证依赖交换性/漂移需重校准"两处均做了诚实披露,符合量化策略专家的 Iron Rules。主题池已补充 8 个新 slug,后续自动运行可持续。
