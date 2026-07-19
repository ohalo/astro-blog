# 量化博客双文生成 + Vercel 部署（2026-07-20）

## 目标
按 cron `quant-blog-2-posts-simplified` 流程，自动生成 2 篇不重复的量化交易博客并发布到 Vercel。

## 主题（脚本选择，已排除最近 10 篇）
1. 通胀掉期与盈亏平衡通胀率：用互换价差拆出通胀预期 → slug: `inflation-swap-breakeven`
2. 利率互换 IRS 定价：用折现曲线做 bootstrap → slug: `interest-rate-swap-pricing`

## 执行要点
- **图片**：各 4 张真实 matplotlib 图表（非占位符），由 `gen_inflation_swap.py` / `gen_irs_pricing.py` 实跑生成，存 `public/images/{slug}/`。
- **数值自洽校验**：
  - IRS：用 bootstrap 出的折现曲线回算互换利率，精确还原市场报价（1y=2.80% / 5y=3.52% / 10y=3.78%），证明曲线内禀自洽；5y K*=3.52%，固定端现值=浮动端现值=15.97%。
  - ZCIS：5y BEI=2.12%，凸度调整约 2bps（恒正），由 BEI 反推远期通胀曲线（首段 2.22% 因输入曲线 1y 利差偏窄，已诚实披露）。
- **文章长度**：通胀掉期 正文CJK=2059 字；IRS 正文CJK=2213 字（均 ≥2000）。
- **frontmatter**：严格对齐现有规范（title/description/publishDate/language:Chinese/difficulty:intermediate/tags 含 Python）。
- **quant-column.md**：已在「最新文章」顶部新增 2026-07-20 通胀掉期/IRS 段落（含 2 篇链接）。
- **构建**：`astro build` 成功（1992 pages，exit 0）。
- **提交**：`feat: add 2 quant articles 2026-07-20`（commit 3ad99d6）→ push 成功。

## 部署验证（最终，全部 200）
- quant-column: 200（含新 slug）
- /blog/inflation-swap-breakeven/: 200（初 308 重定向→跟随后 200）
- /blog/interest-rate-swap-pricing/: 200
- 配图（inflation_curves.png / discount_curve.png 等）: 200

注：push 后 Vercel 自动构建约 3–5 分钟传播完成，期间会出现 308/404，属正常现象，已等待后复验通过。

## 诚实声明（两文均含「真实陷阱」段）
- 通胀掉期：BEI 含通胀风险溢价 + TIPS 流动性/richness 偏差；凸度调整双向性；CPI 指数滞后；跨境 FX 污染。
- IRS：单曲线 vs 双曲线（OIS 折现）；Day Count / 付息频率；插值方式决定曲线形状；Calendar/营业日惯例。
