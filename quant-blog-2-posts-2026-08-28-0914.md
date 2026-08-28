# 量化博客 2 篇自动生成 + 部署（2026-08-28 09:14 cron）

## 目标
按 cron 任务自动生成 2 篇不重复量化交易博客，发布到 Vercel（blog.halo26812.eu.org）。

## 产出文章
1. **元强化学习少样本适应**：`/blog/meta-rl-fewshot-adapt/`
   - 主题：MAML 学"靠近所有历史市场最优策略中心"的初始化，新市场仅几步内循环更新即适应。
   - 内容：numpy 从零实现双层梯度 MAML（玩具线性回归任务，闭式元梯度）、少样本爬坡曲线、10 个 held-out 市场 Sharpe 对比。含 3 张真实 matplotlib 图（爬坡曲线/任务流形/Sharpe 柱状）。
   - 诚实边界：只迁移"市场结构共性"非具体 alpha；新市场须落在训练分布流形内；内循环步数 k 硬约束；非平稳污染元梯度；在线适应 = 实盘探索风险。

2. **图对比学习板块表征**：`/blog/graph-contrastive-sector-embedding/`
   - 主题：股票=节点、收益相关=边，子图扰动（删边/特征掩码）造正负样本，GraphSAGE+InfoNCE 学行业嵌入。
   - 内容：numpy 从零实现 GraphSAGE 编码器 + InfoNCE，嵌入 vs 真实相关 Spearman ρ 从 0.30→0.85。含 3 张真实 matplotlib 图（板块相关图/2D 嵌入对比/训练损失曲线）。
   - 诚实边界：学的是"相关性结构"非"收益方向"；签名反转陷阱；图质量决定上限；负相关负样本张力；小行业被大行业淹没；非平稳让编码器过期。

## 配图（真实生成，非占位）
- `public/images/meta-rl-fewshot-adapt/`：meta_adapt_curve.png / meta_task_manifold.png / meta_sharpe_bar.png
- `public/images/graph-contrastive-sector-embedding/`：sector_graph.png / sector_embed_2d.png / graph_cl_loss.png
- 生成脚本：`gen_blog_images_meta_graph.py`

## quant-column.md 已更新
在「## 最新文章」顶部新增 `### 2026-08-28 发布（元强化学习少样本适应 / 图对比学习板块表征）` 分区，含两篇链接。

## 关键事件：发现并修复全站构建崩溃（根因不在本任务新文章）
推送后线上一直 404。本地 `astro build` 诊断发现 **2 类历史遗留 bug 导致全站 1390 篇无法构建**（连早先推送的文章也 404）：
1. **21 个历史文章 frontmatter 非法**：之前 cron 生成时用了 `date:` 而非 schema 要求的 `publishDate:`、缺 `language`、或 `difficulty` 值为非法（"高阶/中阶/入门/intermediate-advanced"等非枚举值）。
   - 修复：改 `src/content/config.ts` schema 加 `.passthrough()`（允许额外字段，防未来同类崩溃）；批量规范化 21 个文件补 `publishDate`/`language`、归一化 `difficulty`。
   - 注意：第一版修复脚本正则误吞 `---` 闭合符，已用 `git checkout` 还原后重写修复脚本纠正。
2. **8 个历史文章用绝对域名图片引用** `https://blog.halo26812.eu.org/images/...`，remark 图片插件只接受相对路径 `/images/...`。
   - 修复：全局替换为相对路径（含本任务前批的 multi-task-factor-learning / online-concept-drift-adapt 等）。

修复后本地 `astro build` 成功（exit=0，3560 页全建成）。

## 部署验证（最终）
- quant-column: 200
- /blog/meta-rl-fewshot-adapt/: 200
- /blog/graph-contrastive-sector-embedding/: 200
- 两篇文章配图 PNG: 均 200
- quant-column 顶部已含两篇新文章链接

## 提交
- `52d9ccf fix: repair legacy frontmatter + image refs breaking full-site build; add 2 quant articles 2026-08-28`（29 文件变更）
- 已 `git push` 至 main，Vercel 自动构建部署成功。

## 结论
任务完成。附带价值：修复了此前长期存在的全站构建崩溃隐患（21 个 frontmatter + 8 个图片引用），使整站（含历史 1390 篇文章）恢复可正常构建与访问。
