# 创意路线图（Creative Roadmap）

> 本文件是「每日功能创新」定时任务的**持久化记忆**：点子 backlog + 实施日志。
> 每天 14:00 的任务会：读这里 → 扫描全站找新机会 → 追加 3–5 个新点子并评分 → 挑 1 个做出来 → 构建验证 → 提交推送 → 回写日志。
> 评分维度：Impact（影响 1–5）/ Cost（成本 1–5，越低越省）/ Risk（风险 1–5，越低越稳）。Priority = Impact×2 − Cost − Risk。
>
> **工作约定：新功能 / 新页面必须先出设计稿**——用「设计工作台」（ui-design 技能）在 `docs/designs/YYYY-MM-DD-{slug}/` 做高保真原型：一屏一个 HTML + 黑金 `tokens.css`（复用 `docs/designs/_shared/tokens.css`），preview_page 开画布、check_design + look_design 体检通过后再 commit 设计稿，然后动手实现，实现以设计稿为准。

## 📋 Backlog（待做点子）

| # | 点子 | Impact | Cost | Risk | 分 | 状态 |
|---|------|--------|------|------|----|------|
| 1 | 术语悬浮卡：正文量化术语（如 VRP、IRF）hover 显示定义卡，数据源独立 JSON，中英双语 | 4 | 2 | 1 | 7 | ✅ 已做 11-19 |
| 2 | QuantLab 实验扩展：给期权类文章嵌入「蒙特卡洛定价」交互 demo（复用凯利 demo 的 Web Component 模式） | 5 | 3 | 2 | 6 | ✅ 已做 11-19 |
| 3 | 阅读体验：长文阅读进度条 + TOC 当前节高亮跟随 | 3 | 1 | 1 | 6 | ✅ 已做 11-19（进度条；TOC 高亮 astro-pure 原生已带） |
| 4 | 文章系列导航：同 tag 系列文章的「系列进度 + 上一篇/下一篇」卡片（学习路径已有反向链，可泛化） | 3 | 2 | 1 | 6 | ✅ 已做 11-19 |
| 5 | /dash 站点自检面板：死链检测、断图检测、构建耗时趋势（数据来自 CI/build 产物 JSON） | 4 | 4 | 2 | 4 | 待做 |
| 6 | 动态 OG 卡片：文章页 og:image 按标题/标签生成 SVG→PNG，替代静态图 | 4 | 4 | 3 | 3 | 待做 |
| 7 | 学习路径「今日复习」：按艾宾浩斯间隔从已打勾课程中挑 3 篇提醒复习 | 3 | 2 | 1 | 6 | ✅ 已做 11-19 |
| 8 | 量化专栏筛选增强：按 difficulty × 子领域（统计/期权/风控/执行）二维筛选，URL 可分享 | 3 | 2 | 1 | 6 | ✅ 已做 11-19（二维筛选原有，补 URL 可分享 + popstate） |

## 📝 实施日志

| 日期 | 实施的点子 | 结果 / 备注 |
|------|-----------|------------|
| 2026-11-19 | 术语悬浮卡 + 阅读进度条 + 专栏筛选 URL 分享 + 今日复习（用户点名的批量首跑） | 术语表 `src/data/glossary.ts` 38 词双语；进度条只动 transform；URL 参数 level/topic/q/page 稳定 tid；复习卡按 1/2/4/7/15/30/60 天阶梯，进度存储升级 v2（旧 true 自动迁移）。顺带修复：cto 文章正文唯一 unsplash 外链图本地化（远程图在 inferSize 拉取失败时会炸整个 build） |
| 2026-11-19 | QuantLab 蒙特卡洛定价 demo（#2）+ 文章系列导航（#4）（第二次批量跑） | #2：纯数学内核 `src/lib/mcbs.ts`（BSM/GBM 模拟/crude+对偶两估计量/收敛轨迹）与 node 核对脚本共用，`bun scripts/test-mcbs.ts` 19 项过（Haug 基准 10.4506/5.5735、幂等平价、双估计量无偏、对偶方差缩减 ×0.71）；`<quant-lab-mc>` Shadow DOM 组件对数轴收敛图 + BSM 金色锚线 + 重新抽样，先出数后载图；hasLab/hasLabMc 分离打包。#4：设计稿 `docs/designs/2026-11-19-series-nav/`（黑金 tokens，`_shared/` 上版后拷入屏目录——画布服务根是设计目录，上级路径载不到；tokens 缺 box-sizing reset 会在窄屏溢出）；`SeriesNav.astro` 取本篇最窄的 3–120 成员 tag（泛标签黑名单）当系列，点条 ≤21 全画、超长 ±10 开窗两端淡点，设计稿体检 1280/390 双尺寸 0 问题，实现页面 check 无 sp- 归因问题。补交上次会话 6 篇 numpy 实跑文章 |

## 🚫 已做/已存在（避免重复造轮子）

- 凯利下注交互实验（QuantLab，Web Component + Shadow DOM + echarts 懒加载）
- /learn 学习路线图（7 主线 63 篇，localStorage 进度）
- 黑金主题全站重构；归档/关于/学习/条款页重设计
- SEO：JSON-LD 结构化数据、og:locale、sitemap 排除 404、robots
- RSS、英文 i18n、图片懒加载、站内搜索页
