# 量化博客自动发布任务 — 2026-07-16

## 目标
Cron `quant-blog-2-posts-simplified` 触发：自动选题 → 生成 2 篇量化文章（含真实配图）→ 更新 quant-column.md → git 提交推送 → 验证 Vercel 部署。

## 执行结果（全部成功）
- 主题选择脚本选出 2 个与近 10 篇不重复的主题：
  1. Nowcasting 混频实时预测：用高频指标盯住低频 GDP（slug: `nowcasting-midas-gdp`）
  2. 净权益发行异象：公司回购越多未来越该涨（slug: `composite-equity-issuance`）
- 两篇各 ~2000-2200 中文汉字、含完整 Python 代码、各 4 张 matplotlib 真图（非占位）。
- quant-column.md「最新文章」顶部已插入 2 篇新链接。
- `git add -A && commit && push` 成功（commit 2350d68）。
- Vercel 验证全部 200：quant-column 页、两篇文章页、8 张图片 URL。

## 关键数值（合成模型自洽）
- Nowcasting：序贯卡尔曼混频滤波，nowcast 对真实因子 RMSE 随信息累积收敛 0.644→0.291→0.200→0.129，最终与 GDP 相关 0.869。
- 净权益发行：十档收益严格单调递减（D0 回购最多 +1.27%/月 → D9 增发最多 +0.30%/月）；多回购空增发 L-S 组合年化 ~10.2%、t≈11.1、胜率 72.6%。
- 两篇文章均附六类真实陷阱（前视对齐、结构性断点、幸存者偏差、因子误设 / 持续性伪相关、做空成本、监管变迁等）。

## 产物文件
- 文章：`src/content/blog/nowcasting-midas-gdp/index.md`、`src/content/blog/composite-equity-issuance/index.md`
- 配图：`public/images/nowcasting-midas-gdp/*.png`（4 张）、`public/images/composite-equity-issuance/*.png`（4 张）
- 图生成脚本：`generate_nowcasting_midas_gdp_images.py`、`generate_composite_equity_issuance_images.py`
- 部署：https://blog.halo26812.eu.org/quant-column

## 备注
- 构建（npm run build）通过，1582 页。
- 首次推送后图片短暂 404，属 Vercel 静态文件传播延迟，约 1 分钟后全部 200，已复验。
