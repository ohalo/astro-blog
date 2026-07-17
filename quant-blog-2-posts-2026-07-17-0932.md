# 量化博客自动发布（2 篇）— 2026-07-17 09:32

## 任务
按 cron 任务「quant-blog-2-posts-simplified」自动生成 2 篇量化交易博客并发布到 Vercel（astro-blog）。

## 主题（脚本自动选出，已排除最近 10 篇）
- `CIR 利率模型：用平方根扩散守住利率的非负底线` → slug **cir-interest-rate-model**
- `Hurst 指数与分形市场：用重标极差判断序列是随机游走还是有记忆` → slug **hurst-exponent-regime**

> 注意：仓库里已有一个 slug 为 `hurst-exponent-fractal` 的旧 Hurst 文章（2026-07-11 发布）。本次新主题经脚本给了**不同的 slug**（...-regime），因此无碰撞。

## 执行摘要
1. **取数/选题**：`select_blog_topics.py` 输出 2 个不重复主题。
2. **生成配图（真实计算，非占位）**：
   - `gen_cir_interest_rate_model.py` → 4 张图（路径 / 平稳 Gamma / 零地板热图 / 收益率曲线）。
   - `gen_hurst_exponent_regime.py` → 4 张图（三类序列 / R-S 拟合 / 滚动 Hurst / 偏差直方图）。
   - 全部由 numpy/scipy/matplotlib 真实计算，随机种子固定 `20260717`，指标写入各 `public/images/<slug>/_metrics.txt`。
3. **写文章**：两篇 `src/content/blog/<slug>/index.md`，含标准 frontmatter + Python 代码块 + 配图，字数约 2000–3000 字。
4. **更新 quant-column.md**：在「最新文章」顶部插入 2 篇新链接块（含 slug `cir-interest-rate-model` / `hurst-exponent-regime`）。
5. **Git 提交推送**：`feat: add 2 quant articles 2026-07-17` → 推送成功（`b86c140..13a1dd1 main -> main`）。
6. **验证部署**：
   - `/quant-column` → 200
   - `/blog/cir-interest-rate-model/` → 200
   - `/blog/hurst-exponent-regime/` → 200（Vercel 异步构建，约 90s 后由 404 转 200）
   - 两张示例配图 URL 均 200。

## 关键校验点（自检）
- **CIR 公式**：初版 `B(τ)` 漏了 leading factor 2，导致短端不反映 r₀；已修正为标准仿射闭式（`B=2(...) / (...)`）。修正后：y(2Y|r₀=6%)=6.00%（=r₀ 验证 ✓）、长端 5.95%≈b 6%（✓）、低 r₀ 斜率 +172bp / 高 r₀ −177bp（✓）、Feller 2ab−σ²=0.0344>0（✓）、路径最低 2.14% 不触零（✓）。
- **Hurst 滚动准确率初版 ~50%**：根因是窗口索引错位 + 合成路径拼接跳变。重构为「fGn 增量拼接后 cumsum」保证连续，并改用「长仓过滤器 + 无前视（信号[i] 赚 i→i+1）」。最终三类 H 干净分离（趋势 0.73 / 随机 0.44 / 回复 0.39），滚动三段中位数 0.68/0.29/0.62。
- **Hurst 过滤器诚实性**：不宣称凭空 alpha（Sharpe 0.67 vs 买入持有 0.75），而是**风险/状态过滤器**——回撤 −74.9%→−63.7%、仅 39.5% 仓位。文章据此如实表述，未夸大。
- **有限样本偏差**：500 条随机游走 R/S 估计中位数 0.539（非 0.5），文章明确此为经典坑，提醒用 bootstrap 置信带。

## 交付物
- `src/content/blog/cir-interest-rate-model/index.md`
- `src/content/blog/hurst-exponent-regime/index.md`
- `public/images/cir-interest-rate-model/*.png`（4 张）
- `public/images/hurst-exponent-regime/*.png`（4 张）
- `src/content/pages/quant-column.md`（已更新顶部链接）
- `gen_cir_interest_rate_model.py`、`gen_hurst_exponent_regime.py`（生成脚本，可复现）

## 状态：✅ 完成（两篇均发布、页面与配图均 200 可访问）
