# 量化博客自动生成任务（2026-07-11 15:04）

## 目标
自动生成 2 篇不重复主题的量化交易博客，配真实生成的图表，更新量化专栏页，Git 推送并验证 Vercel 部署。

## 执行结果 ✅
- 主题选择（`select_blog_topics.py`）：
  1. 因果推断与双重差分(DID)：从相关性到策略可解释性 → slug `causal-inference-quant`
  2. 深度强化学习的离线批处理(Offline RL)在交易中的应用 → slug `offline-rl-trading`
  - 两主题均不在最近 10 篇（及全量）已发文章中。
- 文章：各约 2500–3000 字，含完整可运行 Python 代码 + 3 张真实 matplotlib 配图（共 6 张）。
- `quant-column.md`：两篇新文已插入「2026-07-11 新发布」区块顶部。
- Git：两次提交并推送至 `main`（第二次修复了 schema 超限）。
- 部署验证（curl）：
  - `/quant-column` → 200
  - `/blog/causal-inference-quant/` → 200（内容含「双重差分」）
  - `/blog/offline-rl-trading/` → 200
  - 6 张配图 URL 均 → 200

## 关键数据与结论（诚实可复现）
**文章1 — DID（因果推断）**
- 模拟：90 只股票 / 120 日 / 第 60 日事件，30 处理组 + 60 控制组；真实处理效应 τ=0.40%/日。
- DID 估计 = 0.415%/日；集群 bootstrap 95% CI = [0.328%, 0.498%]（覆盖真值）。
- 平行趋势检验：事件前处理组/控制组斜率差 ≈ 2.2e-5（≈0，平行趋势成立）。
- 事件研究 CAR（事件后 20 日窗口）≈ 9.78%（理论 τ×20≈8%，加噪声略高）。
- 诚实陷阱：平行趋势不可直接检验、预期效应前移、仅识别平均处理效应、时变混杂、控制组须可比。

**文章2 — Offline RL（含 CQL）**
- 1-D 状态 + 3 动作可解析环境，真实 Q* 已知；偏置行为策略制造分布偏移（「多」只在 x>0.35 出现，制造「多=高收益」伪相关；[-0.2,0.2] 状态盲区）。
- 真实平均收益（rollout）：Oracle 49.7 / 行为策略 33.4 / 行为克隆 BC 36.3 / **朴素离线 Q ≈ 0.0（崩塌）** / CQL ≈ 0.0（浅层线性演示中局部惩罚受共享权重限制，收益与朴素相近——文中诚实说明）。
- 分布外动作被虚高估值：下尾「多」朴素 16.4 vs 真实 10.0（+6.4pp）；上尾「空」朴素 16.4 vs 真实 10.0（+6.4pp）。
- 解法与边界：行为克隆锁在支撑集内稳健恢复；CQL（log-sum-exp 惩罚压低 OOD Q）数学给出，诚实指出其威力在高容量 Q 网 + 独立策略提取时才显；落地清单（数据质量、覆盖度检查、勿直接贪心部署、优先执行层、行为策略护栏）。

## 踩坑与修复
- **首次部署 404（两篇文章）**：`src/content.config.ts` 的 blog schema 要求 `description ≤ 160` 字符。Offline RL 文章 description 原 173 字符 → 构建失败，Vercel 回退到上一版部署（故文章 404）。本地 `npm run build` 复现并定位，将 description 缩至 139 字符后重新构建通过（BUILD_EXIT=0，1062 页），再次推送后全站 200。
- 注意：blog schema 会**静默剥离**未知字段（如 `difficulty`），故该字段无害；`title ≤ 60` 两篇均满足。

## 产物文件
- `src/content/blog/causal-inference-quant/index.md`
- `src/content/blog/offline-rl-trading/index.md`
- `public/images/causal-inference-quant/{fig_parallel_trends,fig_did_estimator,fig_event_study}.png`
- `public/images/offline-rl-trading/{fig_policy_return,fig_q_overestimation,fig_q_surface}.png`
- `generate_causal_inference_images.py`、`generate_offline_rl_images.py`（配图生成脚本）
- `src/content/pages/quant-column.md`（已更新）

## 复现命令
```bash
cd /Users/halo/workspace/astro-blog
python3 generate_causal_inference_images.py
python3 generate_offline_rl_images.py
npm run build   # 全量构建约 38s，1062 页
```
