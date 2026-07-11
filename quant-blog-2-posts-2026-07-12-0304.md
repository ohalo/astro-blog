# 每日博客发布 — 2026-07-12 03:04 (cron: quant-blog-2-posts-simplified)

## 发布文章（2 篇，均通过本地 `astro build` 校验，已部署 Vercel 返回 200）

### 1. 最大回撤控制与 CPPI：把下行约束写进仓位管理
- **Slug**: `drawdown-cppi-control`
- **URL**: https://blog.halo26812.eu.org/blog/drawdown-cppi-control
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `cppi_equity.png` CPPI(m=3) vs 买入持有权益曲线 + floor 线 + 崩盘点
  - `cppi_floor_cushion.png` 资产价值 / floor / 缓冲垫 C_t 结构
  - `cppi_drawdown.png` CPPI vs 买入持有回撤对比
  - `cppi_multiplier.png` 乘数 m 对终值与最大回撤的权衡扫描
- **核心结论（真实计算，带两次崩盘的 6 年 GBM）**: floor=0.80，风险资产年化 9%/波动17%，债券 2.5%；买入持有终值 0.539、回撤 −72.3%；CPPI m=2→终值0.853/回撤−45.3%，m=3→0.834/−53.8%，m=5→0.812/−58.3%；乘数扫描 m=1..6 终值 0.97→0.809、回撤 −25.8%→−58.8%（m 越大回撤越深、cushion 侵蚀越快）。
- **关键方法修正（踩坑）**: ① CPPI 不是「保本神器」——gap risk（单日跳空）使回撤必然阶段性击穿 floor（m=2/floor=0.8 理论最大单日亏约 m·(1−F)=40%，实测 −45.3%）；② 乘数是对「愿为上行放弃多少下行保护」的定价，本崩盘市里 m>2 终值反而下降；③ floor 路径实战用现值折现而非常数；④ 再平衡摩擦/税费需阈值带压换手。

### 2. 风格轮动的宏观信号：用利差/动量给价值成长切换择时
- **Slug**: `style-rotation-macro`
- **URL**: https://blog.halo26812.eu.org/blog/style-rotation-macro
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `style_cumulative.png` 价值/成长/轮动/50-50 累计净值
  - `style_signals.png` 宏观信号时序（曲线斜率 + 信用利差）
  - `style_weights.png` 轮动权重在价值/成长间连续变化
  - `style_drawdown.png` 轮动 vs 单一价值回撤对比
- **核心结论（真实计算，8 年日度合成）**: 隐藏宏观状态 h_t 驱动风格强弱、观测信号为带噪代理 + 滞后 5 日执行；价值因子年化 −4.9%/回撤−61.6%，成长 8.8%/−45.3%，静态50/50 2.5%/−37.1%，**宏观轮动 8.7%/−22.7%**——没跑赢最好的单一风格(成长)，但回撤几乎腰斩。
- **关键方法修正（踩坑）**: ① 前视偏差陷阱——若用同期信号且信号与收益同源(lag=0)，年化会飙到 200%+、回撤<10%（假得离谱）；本文加 `lag=5` 与大方差特异噪声把结果「打回人间」；② 宏观关系会 regime shift（如 2022 加息打破价值跑赢）；③ 日度信号噪声大→需平滑/阈值调仓压换手；④ ETF 代理风格有编制幸存者偏差；⑤ 利差信号本质低频，宜月度再平衡。

## 工程要点
- 主题由 `select_blog_topics.py` 选择（已排除最近 10 篇；CPPI 仓位管理、宏观风格轮动均不与已有风险平价/因子择时/行业轮动重复）。
- `quant-column.md` 已在「2026-07-12 新发布」顶部插入两篇新链接（curl 确认含 /blog/drawdown-cppi-control/ 与 /blog/style-rotation-macro/）。
- 图片生成脚本 `generate_drawdown_cppi_images.py` / `generate_style_rotation_images.py` 一并提交，保证图表可复现。
- 两篇文章 description 均 < 150 字符，构建无 schema 报错。

## 验证结果
- `astro build`: 1169 页构建成功（EXIT 0，Complete!；`/docs/DocsContents/` 无 `<html>` 警告为既有无关项）
- `git push`: 成功（main 012552f..8e16027）
- 部署后 curl 校验（等待 Vercel 重建后，约 75s 后全部稳定）：
  - `/quant-column` → HTTP 200（含两篇新链接）
  - 两篇新博客 `/blog/{slug}`（无尾斜杠，trailingSlash='never'）→ HTTP 200
  - 抽样配图 `/images/...png` → HTTP 200
  - 注：push 后立刻 curl 出现过 308(尾斜杠重定向)/404(部署传播中)，约 75s 后全部 200，确认新部署 8e16027 已全量上线。
