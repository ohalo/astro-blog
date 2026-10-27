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
