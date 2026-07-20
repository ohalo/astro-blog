# 量化博客自动发布任务 — 2026-07-20 12:10

## 任务目标
自动生成 2 篇量化交易博客文章、配真实生成图、更新量化专栏页、Git 提交推送、验证 Vercel 部署返回 200。

## 选主题（脚本去重，已校验 slug 全库不重复）
1. **博彩偏好与最大日收益因子 MAX** — slug: `lottery-max-factor`
2. **分析师评级修正动量** — slug: `analyst-revision-momentum`

> 主题不重复校验：已存在 `lottery-preference-max`（异象快照）、`analyst-bias-quant-strategy`（偏差总论）、`pead`/`post-earnings-drift`（SUE 漂移）。本次两篇为「因子面板化 + 稳定性检验 / 修正动量专属因子」，角度区分明确。

## 产出
- 文章：`src/content/blog/lottery-max-factor/index.md`（CJK ≈ 2112 字，4 段 Python）、`src/content/blog/analyst-revision-momentum/index.md`（CJK ≈ 2138 字，5 段 Python）
- 配图（均为文中代码真实计算，非占位符，各 4 张）：
  - `public/images/lottery-max-factor/`：`max_scatter.png` / `max_ls_curve.png` / `max_ic_ts.png` / `max_2d_sort.png`
  - `public/images/analyst-revision-momentum/`：`rev_distribution.png` / `rev_vs_rating.png` / `rev_decile.png` / `rev_blend.png`
- 配图生成脚本：`generate_lottery_max_factor_images.py`、`generate_analyst_revision_momentum_images.py`
- 量化专栏更新：`src/content/pages/quant-column.md` 在「最新文章」顶部新增 2026-07-20 小节，含 2 篇链接+简介

## 内容要点
- **lottery-max-factor**：把 MAX 从异象升级为月度再平衡横截面因子；多低MAX/空高MAX；逐月 rank-IC 稳定性检验（均值约 −0.13、IR≈−2.4 但频繁转负）；与动量二维分组证正交；拆穿 IC转负/卖空不可得/涨跌停/幸存者偏差/换手容量/低波同源 七类陷阱。全部数据自洽合成，显式声明。
- **analyst-revision-momentum**：标准化预期修正 REV（本月预测变化/历史波动）因子；证明显著优于静态评级水平；十分组严格单调；叠加 60/40 做中观增益；复合修正指数（EPS+目标价+评级）；拆穿保守主义滞后/覆盖偏差/行业中性化/做空成本/PEAD区别 六类陷阱。全部数据自洽合成，显式声明。

## 构建与部署
- `npm run build` 通过：2059 页面，含 2 篇新文。
- Git：commit `feat: add 2 quant articles 2026-07-20`，push 成功（`d0105ee..1be78bf main -> main`）。
- Vercel 验证（部署完成后复测，全部 200）：
  - `/quant-column` → 200
  - `/blog/lottery-max-factor/` → 200（308→200 重定向正常）
  - `/blog/analyst-revision-momentum/` → 200
  - 两张采样配图 → 200

## 备注
- 文章内所有收益/IC/终值均为自洽合成校准量级，已显式标注「真实落地见文末路径」，不误导读者抄量级。
- 早期 curl 出现 404 系 Vercel 部署滞后，复测全部 200，非内容问题。
