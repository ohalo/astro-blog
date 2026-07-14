---
title: "任务产物：做市库存风险 Avellaneda-Stoikov 博客"
date: '2026-07-15'
subject: 量化博客文章 + 真实配图（market-making-inventory-avellaneda）
---

## 完成内容

### 文章
- 文件：`src/content/blog/market-making-inventory-avellaneda/index.md`
- 正文汉字数（不含代码）：**2918 字**（要求 2000–3000）
- frontmatter 字段严格对齐现有文章（vrp-timing-allocation / glosten-milgrom-mm），含 10 个标签
- 结构：结论先行 → 库存风险本质 → AS 模型推导（保留价 r=s−qγσ²(T−t)、最优价差 δ=γσ²(T−t)+(2/γ)ln(1+γ/k)、泊松成交 λ=A·exp(−k·δ)）→ 对称 vs 库存偏斜对比 → 蒙特卡洛结果表 → γ/k/A 参数作用 → 六类真实陷阱 → 完整可运行 Python → 收尾
- 正文嵌入 5 张配图相对路径，与实际文件名一一对应

### 配图（真实蒙特卡洛数据，非占位）
- 生成脚本：`src/content/blog/market-making-inventory-avellaneda/generate_images.py`（matplotlib Agg、rcParams 中文字体 ["PingFang SC","Arial Unicode MS","SimHei","DejaVu Sans"]、axes.unicode_minus=False）
- 输出目录：`public/images/market-making-inventory-avellaneda/`
- 文件清单（5 张 PNG，均 >37KB）：
  1. as_quotes_path.png —— 单条典型路径：中间价/保留价/买卖报价（库存偏斜可见）
  2. as_inventory_path.png —— 库存随时间演化（被拉回零附近）
  3. as_pnl_hist.png —— 终端 PnL 分布：AS vs 对称报价
  4. as_inventory_dist.png —— 终端库存分布对比（AS 显著收窄尾巴）
  5. as_gamma_sensitivity.png —— 风险厌恶 γ 敏感性（双轴）

### 关键真实数据（1000 条路径，S0=100, σ=0.5, T=1, N=600, A=140, k=1.5, γ=0.1）
- AS 库存偏斜：PnL mean 68.6 / std 6.82 / 库存|q| std 4.14 / max|q| 24
- 对称报价：   PnL mean 68.4 / std 7.62 / 库存|q| std 6.17 / max|q| 36
- 结论：AS 把库存波动砍约 1/3、极端持仓 36→24，平均 PnL 几乎不丢
- γ 敏感性：γ 0.02→0.80，|q| std 5.46→2.30，PnL mean 69.0→64.4

### 验收
- ✅ index.md 存在且 2918 字达标
- ✅ public/images/... 下 5 张真实 PNG 全部生成成功
- ✅ 正文 5 个图片路径与实际文件名一一对应
- ✅ 正文代码段独立运行，复现与图表一致的统计数字

### 踩坑记录
- 对称报价分支里 `ask_dist = spread/2.0` 是标量，导致 `rng.poisson` 只抽一次并广播到所有路径，库存恒等 → std≈0（虚假）。已改为 `np.full_like(S, spread/2.0)` 修复，修复后对称报价库存|q| std 回到 6.17（符合预期）。

### 图片文件名列表
as_quotes_path.png, as_inventory_path.png, as_pnl_hist.png, as_inventory_dist.png, as_gamma_sensitivity.png
