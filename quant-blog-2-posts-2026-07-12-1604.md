# 量化博客双文自动生成 — 2026-07-12 16:04

## 任务
cron `quant-blog-2-posts-simplified`：生成 2 篇量化交易博客并发布到 Vercel（astro-blog 仓库）。

## 主题（脚本选出，不与最近 10 篇重复）
`select_blog_topics.py` 正常产出 2 个：
1. **随机波动率带跳跃与 VIX 期货定价：把恐慌也做成因子** — slug `svj-vix-futures`
2. **因子动量：把因子本身当成可交易的资产** — slug `factor-momentum-trading`

## 产出（均含真实计算配图，非占位符，每篇 4 张 matplotlib 图）
- 图片生成脚本：`generate_svj_vix_futures_images.py`、`generate_factor_momentum_images.py`（纯 numpy 计算，数字嵌入正文）。
- 文章：`src/content/blog/svj-vix-futures/index.md`（2120 字）、`src/content/blog/factor-momentum-trading/index.md`（2072 字），均含 Python 代码 + frontmatter（title/description/publishDate='2026-07-12'/tags/language=Chinese/difficulty=advanced）。

### 文1 SVJ + VIX 期货（关键真实数字）
- 20 年(5040 日) Bates 型 SVJ 模拟：Heston(κ=3,θ=0.04,σv=0.4,ρ=-0.7)+崩盘跳(λ=0.6/yr,μJ=-0.09,σJ=0.11)；VIX 中枢≈20、峰值45.1、11 次跳。
- VIX 期货定价：平静日 VIX=11.8、期货 1M→6M=14.2→19.1（Contango）；恐慌日 VIX=45.1、41.1→27.8（Backwardation）。
- VRP：期货−即期均值 +0.74 vol 点，67.1% 交易日 Contango。
- 恐慌因子(做空 VIX 期货, 15% 波动缩放)：年化 7.2%、Sharpe **0.48**（>股票 0.12）、与股票相关 +0.54；镜像做多 VIX(保险) Sharpe -0.45、相关 -0.54。
- 组合 90%股+10%做空 VRP：Sharpe 0.15、回撤 -68.7%（纯股 0.12 / -72.1%）。

### 文2 因子动量（关键真实数字）
- 25 年(300 月) 5 风格因子(价值/动量/规模/质量/低波)，AR(1) 状态 φ=0.85 提供持续性。
- 单因子 Sharpe 0.22~0.61；静态等权因子组合 Sharpe **0.94**、回撤 -21.2%。
- 因子动量组合(过去 12M 收益>0 做多)：年化 22.0%、波动 9.7%、**Sharpe 2.27**、回撤 -17.3%。
- 与静态因子相关 0.72、与共同市场相关仅 **0.18**；信号 IC(12M→下月)均值 0.327，59.7% 月信号为正。

## 流程执行
1. 写 2 个图片生成脚本（含真实统计输出）。
2. 运行生成 8 张图（均验证尺寸正常、非空白）。
3. 写两篇 `index.md`（中文字数 2120 / 2072，均含代码与 4 图引用）。
4. 更新 `src/content/pages/quant-column.md`「2026-07-12 新发布」顶部插入 2 条链接。
5. `bun run build` 通过（1280 页，exit 0；`/docs/` 警告为历史既有）。
6. `git add -A && commit && push` → 成功（13 文件，main）。
7. 等待 Vercel 新构建完成（约 50s 后生效），验证 `/quant-column`、`/blog/svj-vix-futures/`、`/blog/factor-momentum-trading/` 均 200，8 张图片 CDN 均 200，正文/标题/图片引用均已生效。

## 部署状态
✅ 已发布。三页全部 200，8 图全部 200。

## 备注 / 偏差
- 数据为 numpy 合成（与全库高阶文一致），用于演示机制；文内「真实陷阱」段已明确标注合成、VRP 为显式植入、波动缩放、跳跃尾部、相关时变等真实坑。
- 因子动量 Sharpe 2.27 偏乐观（友好校准），文内已加「真实陷阱」说明真实因子动量更温和、需样本外验证信号窗/控制换手。
- 脚本曾因绝对口径 drawdown 公式在缩放前头寸上给出荒谬数值（-454%），已改为百分比口径并加 15% 波动目标缩放，数字方才可读。
