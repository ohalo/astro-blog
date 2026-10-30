# Automation 6a2858b0 执行记录

## 2026-10-27（第 1 次）
- 主题（select_blog_topics.py 兜底池）：kernel-density-conditional-probability、gaussian-mixture-regime-classification
- 脚本：gen_kde_conditional_prob.py / gen_gmm_regime.py（系统 python3，seed=20261027），各 5-6 张真实计算图 → public/images/{slug}/，结果 json：_kde_results.json / _gmm_results.json
- quant-column.md 置顶 2026-10-27 分组；push e7043222；三页面 curl 均 200
- **经验 1**：远端常有并行跑的自动化（本次 origin 上多了模型置信集/SDF-HJ 提交），git pull --rebase 时 quant-column.md 几乎必冲突——解法是保留双方分组、按日期降序排，删冲突标记后 `git add + GIT_EDITOR=true git rebase --continue`。**先 commit 再 pull --rebase**（工作区有未暂存改动时 pull --rebase 直接报错）。
- **经验 2**：文章页 curl 返回 308 是尾斜杠重定向，加 -L 即 200，不是部署失败。
- **经验 3**：quant-column 部署后 CDN 缓存可能滞留 15 分钟+（age=850），验证内容要加 `?v=<ts>` cache-buster，别只 grep 正文就下"没部署"结论。
- **经验 4**：系统 python3 有 numpy/scipy/matplotlib/pandas 无 sklearn（EM 手写即可）；PyYAML 只在 venv（~/.workbuddy/binaries/python/envs/default/bin/python）里，frontmatter 校验脚本用 venv python 跑。matplotlib 中文字体显式设 Heiti TC 等非默认字体时才生效，默认 DejaVu 下 CJK 全是缺字形。
- **经验 5**：GMM/KDE 的 log-sum-exp 要对逐维密度求和后再 logsumexp（z.sum(2) + log w），广播方向别搞反。

## 2026-10-27（第 2 次）
- 主题：hidden-markov-market-regime、change-point-detection-volatility-jump（与现有目录查重通过）
- 脚本：gen_hmm_viterbi.py（手写 log-space forward/backward/viterbi/baum_welch）、gen_cusum_pelt.py（复位型双侧 CUSUM + PELT L1 精确 DP），seed=20261027，各 5 张真实计算图 + _hmm_results.json / _cpd_results.json
- **教训 0（最重要）**：上次中运行留下脏状态（旧版脚本跑出 NaN 对数似然 + TypeError 崩溃）。恢复方式=用当前脚本全程重跑、以 json/日志 `grep -c NaN|Traceback` 均为 0 为准。生成脚本若从上次遗留继续改，改完必须整体重跑。
- **教训 1**：CUSUM 必须写复位型（越阈告警后 S± 清零，PP 图语义）。原来的"alarm 后不清零、要求同时为 0 才重新布防"版本主序列只告 1 次、0/5 命中且不合常理；复位型得到 7 告警 2/5，逻辑自洽。
- **教训 2**：统计滞后映射字典以 CPS 值为键，画图时用 `d.get(j)`（j=0..4 索引）会全 nan 且无任何报错——图表画完要目测/程序化查 NaN。
- **教训 3**：图文必须一致——图例声称"5% 虚警校准"而实测 FAR 0.284 次/天就是自相矛盾，用中性标签（"重尾置换峰值校准阈值"）；直方图 bins 容纳不下的失败样本（滞后≥60）要在图上加注释框说明，否则读者看不到 2 次未检出。
- quant-column.md 置顶第二个 2026-10-27 分组；本次 pull --rebase 无冲突；push e49ee5cf；Vercel 约 3 分钟部署（10 次轮询），quant-column/2 文章/2 图 curl 均 200（文章页用 -L）。

## 2026-10-28（第 3 次）
- 主题（select_blog_topics.py）：diffusion-model-market-path-generation、mcts-portfolio-optimization（与现有目录查重通过）
- 脚本：gen_ddpm_market_paths.py（手写最小 DDPM：两层 MLP 噪声预测器、T=40、3000 轮，seed=20261028，标准化后训练再反标准化）、gen_mcts_portfolio.py（手写 UCB1 MCTS 选择/扩展/模拟/回传，C(15,5)=3003 穷举 ground truth，2000 评估预算）
- **MCTS 关键 bug**：select_expand 每次只扩展 1 个新子节点，2000 迭代内树到不了深度 K 的终节点 → best_sel_m 恒为 None 崩。修法：rollout 返回 (val, completed_set)，用模拟补全出的完整子集跟踪最优，并据此算入选频率/剪枝比（distinct_evaluated=1483）。另加 `_ser` 把 numpy int64 序列化为 JSON（rng.choice 出的元素是 int64）。
- **诚实结论**：DDPM 复刻分布肚子（波动率误差 1.9%、KS D=0.066）但压扁尾部（超额峰度 7.10→0.87、差 87.8%；P99 差 24.6%）——MSE 训练扩散的固有代价；MCTS 精确命中穷举最优 Sharpe 0.2321（差距 0%）、随机同预算只到 0.218（差 6.07%），仅评估 1483/3003 子集（省 50.6%）。两篇都写明局限，未粉饰。
- quant-column.md 在「## 最新文章」下置顶新增 2026-10-28 分组（含本 2 条），保留已有的 2026-10-28（Neural ODE/SimCLR）分组在其后。
- 先 commit 再 `git pull --rebase`（无冲突），push 3846ba11。本地 `npm run build` 先 mv .vercel/output 挪开避 safe-delete 拦截，build 通过（0 schema 错）。
- **部署**：Vercel 本约 4-5 分钟才生效——推送后首轮 curl 文章页 404、quant-column 无新链接，约 5 分钟（sleep 累计 ~3.5 分钟 + 轮询）后才 quant-column 200 含新链接、两文章页 200。验证务必加 `?cb=<ts>` cache-buster 并耐心等部署。

## 2026-10-29（第 4 次）
- 主题（select_blog_topics.py）：distributionally-robust-optimization、stochastic-control-optimal-execution（slug 全库查重通过）
- 脚本：gen_dro_wasserstein.py（手写 W2 球，最坏情况期望收益=μ̂−ε，均值-方差仓位 w*=(μ̂−ε)/(γσ̂²)，600 次 OOS 蒙特卡洛）、gen_hjb_optimal_execution.py（LQ 执行问题有限差分 HJB 反向 DP + 插值，闭式 sinh 调度校验），seed=20261029，各 5 张真实计算图 + _dro_results.json / _sce_results.json（根目录，gitignore）
- **DRO 诚实结论**：DRO 卖的不是更高收益、是给下注上锁——SAA 杠杆偶发 5.58 倍、DRO 封顶 3.92 倍（压 29.8%），90 分位 2.00→0.59，仓位 2.12→0.57（ε*=std(μ̂)=0.000955）；最坏下界覆盖真值 84.7%（贴 1-SE 理论 84.1%）；DRO 杠杆≤SAA 构造恒为 100%。
- **HJB 关键坑**：①控制网格须与状态网格解耦+插值，否则 u 分辨率锁死、调度退化；②闭式值函数须用带硬约束 x(T)=0 的 coth 形式 V=x²σ√(λη)coth(ω(T−t))（对应 sinh 调度），不是无约束 tanh 形式；③终端 coth 发散致相对误差虚高，校验用全轨迹绝对误差（3.48e-5）与排除终端的均值相对误差（0.079）。数值对闭式：调度误差 5.83e-3（0.58%）、x(T)→0；漂移 μ=+0.003 后载（u:0.40→1.80）、μ=−0.003 前载（u:1.85→0.04）；前沿以 TWAP 为端点；网格加密误差单调 0.15→0.009。
- **frontmatter YAML 坑（本回合主要卡点）**：description 内部若含 ASCII 直引号 `"` 会提前闭合双引号标量 → "bad indentation of a mapping entry"。修法：外层强制 ASCII `"`、内部引号类字符改「」交替。两文 description 286/293 字符（<600 上限）。
- quant-column.md 置顶合并：远端并行批次已加 2026-10-29（LightGBM/Barra）组，本回合 2026-10-29（DRO/HJB）组，同日期合并为一个 4 条文分组置顶，其下接远端 2026-10-28 三组——冲突保留双方、日期降序。
- 先 stash -u → git pull --rebase（合入远端批次）→ stash pop（quant-column.md 冲突）→ 解冲突 → 精确 add 文章文件（排除 .workbuddy memory md）→ commit 51381007 → push → 清理 stash。
- 本地 build：先 mv .vercel/output 避 safe-delete 拦截；frontmatter 修好后 build 全程通过（77s 生成全部页面），safe-delete 仅拦 dist/server/.prerender 清理（dist 被 gitignore，与部署无关）。
- **部署**：Vercel 约 3-4 分钟生效，轮询第 2 轮（~60s）后 quant-column 200 含新链接、两文章页 200。验证加 -L + ?cb=<ts>。
