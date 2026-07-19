# 量化博客双文发布任务 (2026-07-20 04:04)

## 目标
自动生成 2 篇量化交易博客并发布到 Vercel (blog.halo26812.eu.org)。

## 主题选择
运行 `select_blog_topics.py`（主主题池已用尽，走内置兜底），排除最近 10 篇：
1. 碳价格因子：把气候风险写进可交易 alpha → slug: `carbon-price-factor`
2. 供应链集中度因子：用客户/供应商依赖量化基本面风险 → slug: `supply-chain-concentration-factor`

## 执行产物
- 配图脚本：`gen_carbon_price_factor.py`、`gen_supply_chain_concentration_factor.py`
- 配图：各 5 张 matplotlib PNG，存于 `public/images/{slug}/`（1170×546，真实图表非占位符）
- 文章：`src/content/blog/{slug}/index.md`（frontmatter 标准格式，含 Python 代码、2000-3000 字）

## 关键数值（自洽合成）
**碳价格因子**
- 碳因子年化 14.4%（多低强度/空高强度），纯多高强度组 −27.8%/yr，市场 9.1%
- 碳 beta = +2.74（对碳价变化率斜率，显著为正），市场载荷 ≈ −0.11（非市场 beta 伪装）
- 五分位年化单调：9.3% / 5.2% / 3.0% / −5.9% / −27.8%
- 碳价路径 15 → 77 $/吨，含政策冲击（120月 +12 跳升，48/175月回调）

**供应链集中度因子**
- 因子年化 9.5%，Sharpe 1.66
- 横截面 t(HHI) = −15.7（控制 MKT+size），负号=集中度越高收益越低
- alpha 主要来自危机期：危机 71.9% vs 平静 3.9%/yr
- 五分位单调下行；HHI 范围 0.25–0.78；冲击期高集中度组显著下沉、低集中度组免疫

## 陷阱披露（各 6 类）
- 碳：数据延迟/基准锚定/政策突变/碳价代理失真/做空成本/行业共线性
- 供应链：披露失真滞后/共同冲击混淆/幸存者偏差(最致命)/行业结构混淆/集中度≠风险/做空成本

## 验证
- 本地 `astro build` 成功：2010 pages, exit 0
- `git add -A` + commit `41dc31e` + push 到 origin/main 成功
- `quant-column.md` 「最新文章」顶部新增一节（碳价格因子 / 供应链集中度因子）
- 部署验证：
  - `/quant-column` → 200
  - `/blog/carbon-price-factor/` → 308(站点重定向) → `/blog/carbon-price-factor` → 200
  - `/blog/supply-chain-concentration-factor/` → 308 → 200
  - （Vercel 构建约 3-4 分钟，初查 404 为构建中，复查全部 200）

## 结论
任务完成：2 篇文章生成、配真实图表、更新专栏页、提交推送、部署验证均通过。
