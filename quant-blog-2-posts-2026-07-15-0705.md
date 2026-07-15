# 量化博客自动发布（2026-07-15 07:05）

## 任务
cron `09c323b8` 触发：自动生成 2 篇量化交易博客并发布到 Vercel（astro-blog）。

## 主题（select_blog_topics.py，已排除最近 10 篇）
1. 预期短缺回测：用 ES 检验替代 VaR 的后验失序 → slug `expected-shortfall-backtesting`
2. 动态 Nelson-Siegel：用三因子状态空间跟踪收益率曲线形变 → slug `dynamic-nelson-siegel`

## 执行要点
- **配图**：各 4 张真实 matplotlib 图（非占位），CJK 字体配置同既有 LPPL/CoVaR 脚本。
  - ES 脚本初版有 bug（ES-z 零假设单位写错、Student-t 分位约定反了），重写为「正态世界 + 罕见灾难跳变」模型并数值校准：肥尾世界真实 VaR≈2.23%（绿，与正态一致）但真实 ES≈4.05%（远超正态 2.57%）。诊断：正态 z≈−0.53（绿）、肥尾 z≈1.62（抬升）；功效曲线 ES 0.37→0.86、VaR 红绿灯 0.04~0.09 装死。
  - DNS 脚本：三因子 AR(1) 状态空间 + 卡尔曼滤波/RTS 平滑，λ=2.5 固定载荷。诊断：卡尔曼还原整条 11 点曲线 RMSE 3.55bps（最大 3.91bps），注入「倒挂→修复」被斜率因子干净拆出。
- **文章**：结论先行 + 完整 Python 代码 + 六类真实陷阱 footer，符合既有风格（2000-3000 字）。
- **quant-column.md**：在「## 最新文章」下 2026-07-15 区块顶部插入 2 篇链接。
- **验证**：`npx astro build` 通过（1507 页，无错）；git commit + push 成功；等待 Vercel 部署传播后复核。

## 部署复核（最终，全部 200）
- `quant-column` → 200，且含两个新 slug
- `/blog/expected-shortfall-backtesting/` → 200
- `/blog/dynamic-nelson-siegel/` → 200
- 8 张配图 `/images/{slug}/*.png` → 全部 200

## 提交
`ebbde9d feat: add 2 quant articles 2026-07-15`（main → origin）
