# 量化博客双文自动生成 — 2026-07-12 08:08

## 任务
cron `quant-blog-2-posts-simplified`：生成 2 篇量化交易博客并发布到 Vercel。

## 主题选择（脚本失败，改为人工选）
`select_blog_topics.py` 因主题池耗尽只选出 1 个而 assert 失败。
仓库已有 **690 篇**文章，最近 10 篇涵盖 EVT/三重屏障/动量/宏观因子/MAX/PEAD/OU/CPPI/风格轮动 等。
人工选定 2 个**全新、不与近期及整库重复**的主题：

1. **`lasso-elasticnet-factor`** — LASSO 与弹性网络在多因子选股中的应用
2. **`overnight-intraday-anomaly`** — 隔夜效应与盘中效应（收益时间结构拆解）

## 产出（均含真实配图，非占位符）
每篇 4 张 matplotlib 图，数字由生成脚本真实计算后嵌入正文。

### 文1 LASSO/弹性网络（高维小样本演示）
- 数据：100 因子 / 10 真因子 / 块内高共线(ρ=0.55) / 弱信号(R²≈12%)；训练 3600 行、验证 2400、测试 6000。
- 关键结果：OLS 用满 86/100 系数、OOS IC 0.283；Lasso 用交叉验证精选 13 个、覆盖 9/10 真因子、OOS IC 0.299（+5.6%）；噪声因子 |coef| 中位数 Lasso=0.0000 vs OLS=0.0004。
- 样本外多空组合年化：Lasso 29.97% / OLS 27.40% / 真系数上限 28.97% / 随机 1.12%。
- 配图：`lasso_coef_path / lasso_coef_recovery / lasso_oos_ic / lasso_portfolio`.png

### 文2 隔夜/盘中效应（30 年合成日度）
- 数据：7560 交易日；隔夜 μ≈+0.00030/日（周一桥接周末 +0.00150）、盘中 μ≈−0.00003/日。
- 关键结果：隔夜-only 年化 15.84% / σ 8.02% / Sharpe 1.87 / 回撤 −9.98%；买入持有 15.62% / 12.55% / 1.22 / −20.45%；盘中-only 近零收益(−0.18%) 却回撤 −38.69%；**总收益 98.1% 来自隔夜段**。
- 配图：`overnight_cum / overnight_hist / overnight_strategy / overnight_weekday`.png

## 流程执行
1. 写 `generate_lasso_elasticnet_images.py` 与 `generate_overnight_intraday_images.py`（含真实统计输出）。
2. 运行两脚本生成 8 张图（脚本2 一次成功；脚本1 因 sklearn 1.9 移除 `n_alphas` 及 n≫p 导致正则化无优势，改为高维小样本设定 P=100/N=60/训练 3600 行后重跑，结果合理）。
3. 写两篇 `index.md`（frontmatter：title/publishDate/description/tags/language/difficulty=advanced，与全库一致）。
4. 更新 `src/content/pages/quant-column.md`，在「2026-07-12 新发布」顶部插入 2 条链接。
5. `git add -A && commit && push` → 成功（13 文件，main）。
6. 本地 `bun run build` 通过（1216 页，exit 0；Pagefind 警告为历史既有，无关）。
7. 验证线上：`/quant-column`、`/blog/lasso-elasticnet-factor`、`/blog/overnight-intraday-anomaly` 均返回 **200**，且正文/图片/链接均已生效。

## 部署状态
✅ 已发布。Vercel 初次轮询新文 404（旧部署在线、新构建进行中），约 115s 后新构建完成，三页全部 200。

## 备注 / 偏差
- 数据为 numpy 合成（与全库其它高阶文一致），用于演示机制；文内已加「合成≠真实」与 Sharpe  inflated 等真实陷阱段。
- 主题选择脚本需修复（池耗尽后的兜底逻辑应保证产出 2 个），本次为人工兜底。
