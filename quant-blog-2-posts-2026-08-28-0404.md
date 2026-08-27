# 量化博客双文生成任务 - 执行记录 (2026-08-28 04:04)

## 任务目标
自动生成 2 篇量化交易博客并发布到 Vercel（blog.halo26812.eu.org）。
约束：2 篇、主题不与最近 10 篇重复、含真实配图（非占位）、push 前更新 quant-column.md。

## 主题（来自 select_blog_topics.py）
1. 多智能体强化学习交易：用对手建模学会在博弈中下单 → `multi-agent-rl-trading`
2. 状态空间模型 S4/S5：用结构化状态空间替代注意力做长序列 → `state-space-model-s4`

## 交付物
- 文章1：`src/content/blog/multi-agent-rl-trading/index.md`（约 14.8k 字符 / 中文约 3.7k 字，4 张真实 numpy 计算图）
- 文章2：`src/content/blog/state-space-model-s4/index.md`（约 12.7k 字符 / 中文约 3.2k 字，4 张真实 numpy+scipy 计算图）
- 配图：`public/images/{slug}/*.png`（每组 4 张，均为真实数值计算生成，非占位）
- 生成脚本：`scripts_gen/gen_marl_images.py`、`scripts_gen/gen_s4_images.py`（全部可复现 seed=20260828）
- quant-column.md 已更新（「最新文章」顶部新增 2026-08-28 多智能体RL / S4 链接块）

## 关键数值（全部来自真实运行，已嵌入文章）
### 文章1：多智能体强化学习交易
- 设定：Almgren-Chriss 纯瞬时冲击执行博弈，P0=100, V=1(归一化), T=20, λ=0.6, M=5
- 2 人混合博弈（对手前载 ρ=0.70，聪明交易员二选一）：
  - 朴素 TWAP 实现短缺 IS = **6.00 bps**
  - 对手建模最优反应 IS = **4.53 bps**
  - 省 **1.47 bps（≈24.5%）**
- 多智能体虚拟博弈（M=5，初始全前载 ρ=0.55）：
  - 第 0 轮撞车 IS = **87.1 bps**
  - 第 25 轮协调后 IS = **27.3 bps**
  - **降幅 68.7%**（离均匀理论下限 ≈15 bps 尚有余量，体现收敛需足够轮数）
- 可预测性消融（对手计划噪声 ε 0→4.0）：建模优势 1.47→1.32 bps（慢衰减；诚实边界：对手趋近均匀时优势归零）
- 复现 pitfall：best_response 初版 water-filling 写成 (μ−L) 且二分方向反，导致 NaN；改为 (L−μ) 且 s>V 时 lo=mid 后修正；永久冲击交叉项使 BR 退化稠密 QP，已如实标注为局限（用纯瞬时冲击换取可分离可解）

### 文章2：状态空间模型 S4/S5
- 设定：HiPPO(S4D-Lin) 对角 A，双线性离散化，seed=20260828
- SSM-as-convolution 正确性：卷积视角 vs 状态递推 逐点最大差 **1.7e-15**（机器精度内）
- 序列复杂度（N=512..8192 测时，log-log 拟合）：
  - SSM FFT 卷积斜率 **0.55（近线性）**
  - 朴素注意力 QKᵀ 斜率 **1.95（近 O(N²)）**
  - N=8192：注意力 23ms vs SSM <0.3ms（差约 100×）
- 长程依赖（早期事件 p0→末拍读出）：
  - a=0.99 长记忆 SSM **MSE≈0**（p0=10/30/60/100 全部接近机器零）
  - a=0.5 短记忆 **MSE≈1.0（失败）**、固定窗口基线(末30拍) **MSE≈1.0（失败）**
- 内容选择性短板（诚实红线）：LTI-SSM 线性读出 MSE=**0.324** ≈ 目标方差 1/3（随机水平）；显式内容选择器(找闸门取后一token) MSE=**0**；结论指向需 Mamba 选择性扫描
- 复现 pitfall：最小二乘读头 solve 初版矩阵维度错 (2000 vs 1)，改为标量 1x1 后修正

## 验证
- `git add -A && git commit && git push` → 成功（main: 7db06be → origin）
- `npm run build` → 成功（3485 pages，dist 含两篇文章 + 8 张图）
- 线上复核（Vercel 从 GitHub push 自动重建，本次约等待 120s 完成）：
  - `curl https://blog.halo26812.eu.org/quant-column` → **200**，页面含两个新 slug 链接 ✅
  - 两篇文章 URL（带尾斜杠，跟随重定向）→ **200**，正文含真实标题 ✅
  - 8 张配图 URL → 全部 **200** ✅

## 已知偏差 / 说明
- 与历史任务一致现象：push 后 Vercel 不会立即更新，需等待其从 GitHub 自动重建；本轮等待约 120s 后全部 200。
- 本环境无 Vercel 凭证（`vercel whoami` → No existing credentials），无法手动 redeploy，依赖 GitHub 自动构建（历史记录显示曾用空 commit `c459532 trigger: vercel rebuild` 触发，本次未需要）。
- 两篇文章正文均嵌入真实运行数字（非占位/非编造），与配图来源脚本一致。
- 实验为受控合成场景（执行博弈 / 长程读出任务），用于演示机制，非真实盘回测；永久冲击、LOB 非线性、异步部分可观测等真实约束已在文章「落地坑」段诚实标注。
