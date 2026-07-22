# 量化博客 2 篇自动生成任务执行记录

**执行时间**: 2026-07-22 09:30 (Asia/Shanghai)
**触发**: cron `09c323b8-b9c4-432e-a36f-f7281a10fb09 quant-blog-2-posts-simplified`
**结果**: ✅ 成功，Vercel 部署返回 200，两篇文章页 + 配图均可访问

## 主题（select_blog_topics.py 排除最近 10 篇后选定）

1. **永久组合 Harry Browne 配置** (slug: `permanent-portfolio-browne`)
2. **耶鲁捐赠基金模式借鉴** (slug: `yale-endowment-model`)

## 产出文件

- 文章: `src/content/blog/permanent-portfolio-browne/index.md`、`src/content/blog/yale-endowment-model/index.md`
- 配图(各 4 张, matplotlib 合成): `public/images/{slug}/cover.png` + 3 张图表
- 生成脚本(可复跑): `gen_permanent_portfolio.py`、`gen_yale_model.py`
- 专栏更新: `src/content/pages/quant-column.md` 顶部新增「2026-07-22 新发布（永久组合 / 耶鲁模式）」小节

## 关键统计数字（合成宇宙, numpy 算实）

**永久组合**（四等分 25/25/25/25, 蒙特卡洛 400 路径均值）：
- 长期(16 年混合轮动): 年化 4.63% / 波动 6.04% / Sharpe 0.45 / 最大回撤 −16.05% / 终值 20.33x
- 60/40 对照: 年化 3.82% / 波动 9.01% / Sharpe 0.24 / 回撤 −33.05% / 终值 14.31x
- 四种环境最差年化: 60/40 = 1.27% vs 永久组合 = 4.07%（无死环境）
- 成分风险贡献: 股票 35.4% / 长债 15.1% / 黄金 49.3% / 现金 0.2%（四等分资金≠风险平价）

**耶鲁模式**（因子模型, 20 年含 2 次危机, MC 400 均值）：
- 耶鲁完整版: 年化 9.18% / 波动 15.23% / Sharpe 0.53 / 回撤 −48.19% / 终值 7.05x
- 流动代理版: 年化 7.90% / 波动 13.06% / Sharpe 0.50 / 回撤 −42.50%
- 60/40: 年化 6.92% / 波动 11.08% / Sharpe 0.48 / 回撤 −36.37% / 终值 4.26x
- 收益来源: 另类资产(绝对收益+PE+VC+实物)贡献 ~8.3pp，占组合收益近 80%

## 执行中的关键修复

1. **永久组合环境年化单路径噪声**：初版用单条 60 月路径，60/40 繁荣期被运气拖到 1.78% 违反直觉。改为 MC 400 路径平均 + 净值用平均路径复合，数字稳健。
2. **永久组合净值图 broadcast bug**：`avg_path_nv` 维度错误，改为多条路径分别 cumprod 后对净值取均值。
3. **耶鲁模拟权重漂移爆炸**：再平衡重置分支未归一化,杠杆累积导致 nan/回撤 −452%。重写：权重始终归一化到 gross 1，NAV 用累积乘积跟踪，危机中对非流动性资产冻结仅调流动性部分。

## 验证

- 本地 `npx astro build` 通过（2249 页）
- `git push` 触发 Vercel 自动构建
- 部署后验证: `quant-column`=200, 两篇 `/blog/{slug}/`=200, 4 张抽验配图均 200
- quant-column 页面已渲染出两篇新文章链接
