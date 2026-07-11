# 每日博客发布 — 2026-07-11 22:05 (cron: quant-blog-2-posts-simplified)

## 发布文章（2 篇，均通过本地 `astro build` 校验）

### 1. Hurst 指数与分形市场假说
- **Slug**: `hurst-exponent-fractal`
- **URL**: https://blog.halo26812.eu.org/blog/hurst-exponent-fractal/
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `hurst_series.png` 三类合成价格序列
  - `hurst_rs_scaling.png` R/S 重标极差 log-log 拟合
  - `hurst_rolling.png` 机制切换滚动 Hurst
  - `hurst_histogram.png` 500 条随机游走 Hurst 分布（有限样本偏差）
- **核心结论（真实计算）**: 趋势型 H≈0.79、均值回复型 H≈0.35、随机游走 500 次均值 0.546（95%CI [0.435,0.669]）— 三类清晰分离。
- **关键方法修正**: Hurst 描述的是**增量/收益的持久性**，必须对 `np.diff(price)` 估计；直接对价格水平做 R/S 会得到 H≈1.0 的伪趋势（已踩坑并修正）。fGn 用 Davies–Harte 算法生成。

### 2. Fama-French 五因子模型 A 股实证
- **Slug**: `fama-french-five-factor-cn`
- **URL**: https://blog.halo26812.eu.org/blog/fama-french-five-factor-cn/
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `ff_cumulative.png` 五因子累计净值
  - `ff_factor_stats.png` 因子年化溢价与夏普
  - `ff_alpha_comparison.png` 六组合 CAPM/FF3/FF5 下 alpha 收敛
  - `ff_loading_heatmap.png` 六组合对五因子载荷热力图
- **核心结论（真实计算，受控模拟）**: 平均 |alpha| 随模型扩张收敛 CAPM 0.288% → FF3 0.155% → FF5 0.024%/月。小盘价值组合 CAPM alpha +0.44% 被 FF3 吸收到 −0.03%；大盘低盈利组合 CAPM −0.46%、FF3 仍 −0.29%（RMW 未覆盖）、FF5 吸收到 0.00%。演示「模型误设造成的伪 alpha」机制。
- **关键 bug 修正**: 因子已为超额收益（Mkt-RF 等），初版又减了一次 RF 导致每个 alpha 偏移 −0.22%/月；改为不重复减 RF 后结论正确。

## 工程要点
- 主题由 `select_blog_topics.py` 选择（已排除最近 10 篇，兜底主题池）。
- `quant-column.md` 已在「2026-07-11 新发布」顶部插入两篇新链接。
- 图片生成脚本 `generate_hurst_images.py` / `generate_fama_ff_images.py` 一并提交，保证图表可复现。
- 两篇文章 description 超 160 字符触发 Astro schema 报错，已缩短至 133/125 字符后构建通过。

## 验证结果
- `astro build`: 1120 页构建成功（EXIT 0）
- `git push`: 成功（main 3de7f42）
- 部署后 curl 校验：
  - `/quant-column` → HTTP 200
  - 两篇新博客 `/blog/{slug}/` → HTTP 200（标题正确）
  - 抽样配图 `/images/...png` → HTTP 200
  - quant-column 页面含两篇新链接 ✅
