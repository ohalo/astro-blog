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
