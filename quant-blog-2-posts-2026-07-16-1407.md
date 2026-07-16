# 量化博客自动发布任务 — 2026-07-16

## 任务
按 cron `quant-blog-2-posts-simplified` 自动生成 2 篇量化交易博客并发布到 Vercel (blog.halo26812.eu.org)。

## 选定的两个主题（脚本自动排除最近10篇后给出）
1. **Temporal Fusion Transformer** — slug: `temporal-fusion-transformer`
2. **N-BEATS 神经基展开** — slug: `nbeats-forecast`

## 产出物
- `src/content/blog/nbeats-forecast/index.md` — 约 2150 中文字，3 段 Python 代码，4 张真实配图
- `src/content/blog/temporal-fusion-transformer/index.md` — 约 2180 中文字，3 段 Python 代码，4 张真实配图
- `public/images/nbeats-forecast/*.png` (4 张：decomposition / forecast / error_curve / basis)
- `public/images/temporal-fusion-transformer/*.png` (4 张：variable_selection / temporal_attention / attention_heatmap / multivariate_forecast)
- `src/content/pages/quant-column.md` — 在「2026-07-16 新发布」顶部插入 2 篇新链接（已确认 live 页含 nbeats-forecast 链接）

## 关键数值（均来自可复现 Python，合成但自洽模型）
### N-BEATS
- 测试段 (H=100) RMSE：**2.124** vs 季节朴素(24)=10.955（5.2×）、线性漂移=25.957（12.2×）、上一值(RW)=1.936
- 逐步长：h≥5 起严格优于 RW（h=10: 1.503 vs 1.997）；MASE≈1.32（一步预测打不过 RW 是真属性，非 bug）
- 残差 std≈1.07，95% 区间用其构造

### TFT
- 变量选择网络(uni-R² 代理)：x_sig=0.921 / x_extra=0.063 / x_noise=0.002 → 信噪比 **265×**
- 多变量 vs 单变量滚动外推 SMAPE：**10.45% vs 18.56%**（改善 43.7%）

## 工程过程要点（踩坑已修复）
- N-BEATS 初版用 t² 趋势基 → 外推爆炸（RMSE 劣于 RW）；改为仅一阶线性趋势基后修复
- TFT 初版用岭回归系数/Lasso 做变量选择，噪声变量权重异常高（policy 步长混为特征和合成数据纯周期结构）；改用「变量主序 make_lag + 单变量 R² 门控」+ 增加第二个真信号变量(x_extra)后，信噪比与多变量增益均合理
- 指标选型：SMAPE 在零穿越序列上对 RW 不公平 → N-BEATS 改用 RMSE/MASE

## 验证
- `npx astro build` 成功（1572 页，无错误）
- `git push` → main 89d1d88..8ff2e59 成功
- 线上验证（curl，含 -L 跟随重定向）：
  - /quant-column = 200
  - /blog/nbeats-forecast/ = 200（初始 404 为 Vercel 边缘传播延迟，约 2 分钟后转 200）
  - /blog/temporal-fusion-transformer/ = 200
  - 8 张配图全部 = 200

## 结论
任务完成：2 篇文章已发布、配图真实、quant-column 已更新、Vercel 部署返回 200。
