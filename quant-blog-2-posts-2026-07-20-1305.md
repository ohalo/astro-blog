# 量化博客 2 篇自动生成任务（2026-07-20 13:05）

## 执行结果：✅ 全部成功

### 主题（来自 select_blog_topics.py，已排除最近 10 篇）
1. 盈余预告惊喜与漂移 → slug: `earnings-guidance-drift`
2. 回购强度因子 → slug: `buyback-intensity-factor`

### 交付物
- **文章**：`src/content/blog/{slug}/index.md`，含标准 frontmatter、Python 代码示例、4 张真实计算配图引用
  - earnings: ~2277 CJK 正文（不含代码/前言），buyback: ~2084 CJK 正文，均超 2000 字下限
- **配图（真实 matplotlib 生成，非占位符）**：各 4 张
  - `public/images/earnings-guidance-drift/{gsi_distribution,gsi_ls_curve,gsi_decile,gsi_car_decay}.png`
  - `public/images/buyback-intensity-factor/{buyback_distribution,buyback_ls_curve,buyback_decile,buyback_beta_scatter}.png`
  - 配图脚本：`generate_earnings_guidance_drift_images.py` / `generate_buyback_intensity_images.py`
- **quant-column.md**：在「最新文章」顶部新增 2026-07-20 小节，含 2 篇链接+简介

### 内容要点（自洽合成数据，重点在方法可复现）
- **GSI 因子**：标准化盈余预告惊喜 = (真实盈利−一致预期)/截面标准差；月度再平衡长短因子，十分组严格单调，预告后 12 月 CAAR 衰减可视化「漂移」，因子 beta 高但截距正；拆穿公告前视/做空约束/幸存者偏差/覆盖偏差四类陷阱
- **回购强度因子**：年化回购收益率标准化；验证十分组单调、因子对市场 beta 近零且截距显著为正（独立低 beta alpha）；拆穿信号滞后/做空约束/幸存者偏差/杠杆回购四类陷阱

### 验证
- `npx astro build` 成功（2068 页，exit 0）
- git commit `dddafe1` + push 到 origin/main 成功
- 线上验证（curl）：
  - `/quant-column` → **200**
  - `/blog/earnings-guidance-drift/` → **200**（redirect 后）
  - `/blog/buyback-intensity-factor/` → **200**
  - 8 张配图均 → **200**
  - 线上 quant-column 已出现 2 篇新链接

### 备注
- 初版配图脚本系数过大导致累计净值溢出（约 1e9），已下调至 0.004 量级，终值约 6–8x（12 年），量级合理
- 早期 curl 部分 404 为 Vercel 部署延迟 + 旧文章错误文件名导致，非本任务引入；复测全部 200
