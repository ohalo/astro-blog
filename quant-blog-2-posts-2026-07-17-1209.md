# 量化博客自动发布（2 篇）— 2026-07-17 12:09

## 任务
按 cron 任务「quant-blog-2-posts-simplified」自动生成 2 篇量化交易博客并发布到 Vercel（astro-blog）。

## 主题（脚本自动选出，已排除最近 10 篇）
- `Beta 中性组合：用市场对冲把方向风险剥离干净只留 alpha` → slug **beta-neutral-portfolio**
- `极值理论尾部依赖：用广义帕累托分布给极端联动定价` → slug **evt-tail-dependence**

> 脚本提示主主题池已用尽，使用内置兜底主题，已校验 slug 全库不存在（无碰撞）。

## 执行摘要
1. **取数/选题**：`select_blog_topics.py` 输出 2 个不重复主题。
2. **生成配图（真实计算，非占位）**：
   - `gen_beta_neutral_portfolio.py` → 4 张图（累计净值 / 暴露对比 / 对冲倍数敏感性 / 回撤）。
   - `gen_evt_tail_dependence.py` → 4 张图（GPD 拟合 / POT 均值超出 / 高斯 vs t 散点 / 尾部依赖曲线）。
   - 全部由 numpy/scipy/matplotlib 真实计算，随机种子固定 `20260717`，指标写入各 `public/images/<slug>/_metrics.txt`。
3. **写文章**：两篇 `src/content/blog/<slug>/index.md`，含标准 frontmatter + Python 代码块 + 配图，字数约 2000–3000 字。
4. **更新 quant-column.md**：在「最新文章」顶部插入 2 篇新链接块（含 slug `beta-neutral-portfolio` / `evt-tail-dependence`）。
5. **Git 提交推送**：`feat: add 2 quant articles 2026-07-17` → 推送成功（`aa4130d..c840049 main -> main`）。
6. **验证部署**：
   - `/quant-column` → 200（且服务端已含两个新 slug）
   - `/blog/beta-neutral-portfolio/` → 200（Vercel 异步构建，约 2 分钟后由 404 转 200）
   - `/blog/evt-tail-dependence/` → 200
   - 8 张配图 URL 均 200。

## 关键校验点（自检）
- **Beta 中性公式诚实性**：空头权重 = −组合 β（指数自身 β=1），中性化后总 β 精确归零。模拟口径：用前半段(252 日)OLS 估 β 避免前视，后半段实盘对冲。结果：组合 β=0.968、多头-市场相关 0.97 / 中性-市场相关 −0.07、回撤 −16.1%→−1.5%、Sharpe 1.65→2.21、净敞口 0.03、残差真实 β=−0.0166。对冲倍数图证明只有 k=1 时系统性暴露归零（不夸大）。
- **EVT 关键反直觉点**：同 ρ=0.6，高斯尾部依赖 λ→0（分散有效）、t(ν=4) λ≈0.31（分散崩溃），理论 t-λ=0.314 与实证 0.30 吻合。GPD 拟合 ξ=0.149>0 确认厚尾。如实指出高斯正态假设会让人危机里误判分散有效性——这正是 2008 的真相。
- **无前视**：Beta 文章用训练段估 β、测试段对冲；EVT 用独立阈值外样本做 GPD/POT/尾部依赖估计。
- **真实配图**：8 张图均由计算脚本生成，无占位符，_metrics.txt 留存可复现指标。

## 交付物
- `src/content/blog/beta-neutral-portfolio/index.md`
- `src/content/blog/evt-tail-dependence/index.md`
- `public/images/beta-neutral-portfolio/*.png`（4 张 + _metrics.txt）
- `public/images/evt-tail-dependence/*.png`（4 张 + _metrics.txt）
- `src/content/pages/quant-column.md`（已更新顶部链接）
- `gen_beta_neutral_portfolio.py`、`gen_evt_tail_dependence.py`（生成脚本，可复现）

## 状态：✅ 完成（两篇均发布、页面与配图均 200 可访问）
