# 量化博客自动发布任务 (2026-07-19 10:05)

## 目标
按 cron `quant-blog-2-posts-simplified` 流程自动生成 2 篇不重复量化博客并发布到 Vercel。

## 关键故障与修复（重要）
- **主题选择脚本崩溃**：`select_blog_topics.py` 的「写死兜底 slug 清单」已失效——所列 9 个 slug（volatility-target-timing / adf-kpss-stationarity / threshold-garch / macro-pca / optimal-transport / entropy-risk-parity / sparse-group-lasso / particle-filter / kalman-filter）今日已全部发布，去重过滤后只剩 1 个 → `assert len(selected)==2` 直接失败。
- **修复**：把写死清单替换为**运行时磁盘扫描**——从一份新鲜候选池逐条检查 `slug` 是否真在 `src/content/blog/` 未发布，取前 2 个空闲 slug；标题按 slug 即时生成，避免再次失效。脚本本身已验证可跑通（输出 Resampled Efficiency / Asian-Lookback），并提交到 workspace-ybgrxae3pgfdysn7 仓库。
- **本次手动选定的 2 个主题（经文件系统校验 slug 全库唯一）**：
  1. Hull-White 短期利率模型 → `hull-white-rate-model`
  2. Dupire 局部波动率 → `local-volatility-dupire`

## 生成内容
- **配图**：各 4 张真实 matplotlib 图（种子 20260719），全部自包含 Python 真实计算，非占位符。
  - Hull-White (`gen_hull_white.py`)：1500 条利率路径向 6% 中枢回复 / 4 快照收益率曲线随 r_t 切换上凸↔倒挂 / 池化 OU MLE 校准 a=0.145(RMSE 0.0147) b=0.060(RMSE 0.0038) / 期限溢价机制。稳态波动 4.56%。**诚实陷阱**：朴素单路径 OLS 校准 a 的 RMSE 0.8827 = 池化法 60 倍（慢回复下不可用）。
  - Dupire (`gen_dupire_local_vol.py`)：已知局部波动率曲面(微笑+期限衰减) / 正向 CN-PDE 解出 C(K,T) 曲面 / Dupire 反演恢复 vs 真值 / 截面切片恢复。良定义区域 RMSE≈0.058。**诚实陷阱**：ATM 拐点处 C_KK→0 分母趋零除零炸裂（已用邻域外推正则化，并在切片图显式剔除奇点带并标注）。
- **文章**：各 ~2000–3000 中文字 + 完整可运行 Python（已实跑通过）+ 真实陷阱披露 + 诚实声明，frontmatter 严格对齐现有规范（title/description/publishDate/tags/language:Chinese/difficulty:intermediate）。两文互为「表兄弟」对照（多一个状态变量）。
- **quant-column.md**：在「最新文章」顶部插入新日期分组 `### 2026-07-19 新发布（Hull-White 利率模型 / Dupire 局部波动率）`，2 条链接 + 简介。

## 构建与部署
- `npx astro build` 通过（1916 页, exit 0, 无错误）。
- commit: `feat: add 2 quant articles 2026-07-19` → push 成功（80a6c99）。
- Vercel 自动构建约 3 分钟后传播完成。

## 部署验证（最终, 全部 200）
- quant-column: 200
- /blog/hull-white-rate-model/: 200
- /blog/local-volatility-dupire/: 200
- 8 张配图全部 200（远程确认）
- 两文正文标志性短语命中, 渲染正确。

## 结论
2 篇文章成功生成、配图真实、更新专栏页、推送并验证部署通过。全程未编造虚假收益指标；对「Hull-White 常数 σ 不能解释波动率微笑 / 单路径 OLS 校准极噪」与「Dupire ATM 奇点除零 / 微笑套利 / 无模型风险」均做了诚实披露,符合量化策略专家的 Iron Rules。同时修复了导致 cron 后续运行必然失败的脚本根因。
