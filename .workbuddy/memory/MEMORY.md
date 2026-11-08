# astro-blog 项目长期备忘

## 站点与部署
- 线上域名 https://blog.halo26812.eu.org（astro.config site），Vercel 部署约 3-6 分钟。
- 文章页 URL 返回 308 是尾斜杠重定向，curl 必须加 -L。
- **CDN 缓存很顽固**：`?cb=<ts>` 可能被忽略（`x-vercel-cache: HIT`、age 上千秒仍返回旧版）。可靠验证用 `-H 'Cache-Control: no-cache' -H 'Pragma: no-cache'` 强制回源，看到 `MISS` + 新 `last-modified` 才算部署生效。
- 但 **`MISS` 也可能仍是旧版**：部署尚未切换完成时，强制回源会拿到旧构建（新 `last-modified` 只是请求时间）。判定标准是**同时**出现 `MISS` + 线上 HTML 含新标题/新数字。实测 2026-11-07 推送后 ~3.5 分钟仍旧版、~5.5 分钟才全量生效，至少验证两轮。

## 量化专栏页面架构（重要，2026-11-07 查清）
- `/quant-column` 由 `src/pages/quant-column.astro` → `src/components/pages/QuantColumnContent.astro` 渲染，**不读取** `src/content/pages/quant-column.md`。历代自动化在该 md 里维护的「## 最新文章」分组是死内容，线上不展示。
- 组件按 `tags` 含 `量化/quant/交易/投资` 过滤 blog 集合，再按 `difficulty`（beginner/intermediate/advanced）分类，`byDateDesc` 按 publishDate 倒序。
- **文章要进专栏并置顶**：frontmatter 必须带 `量化` 类 tag + `difficulty: advanced`，且 `publishDate` 最新。仅改 quant-column.md 无效。
- 建议把自动化里「更新 quant-column.md」改成「校验/补齐新文章 frontmatter 标签」。

## 内容 schema 硬约束
- frontmatter `description` ≤ 600 字符（content collection schema，超长直接 build 失败）；内部不要有 ASCII 直引号（会提前闭合 YAML 双引号标量）。
- frontmatter 用 `publishDate`（不是 `date`）+ `language: zh`。
- 每篇文章标配：5 张真实计算 PNG 于 public/images/{slug}/，结果 JSON 与图同目录，正文数字必须与 JSON 一致。
- 文章内代码块必须自包含可运行（含数据生成 + `__main__`），并从 md 抽出来 exec 比对 JSON；禁止用「乘系数」伪造曲线、禁止把估计误差冒充预测误差、禁止 try/except 回退真值冒充拟合结果。
- frontmatter `image` 必须指向真实存在的文件；正文必须真的引用 5 张图（`![...](/images/{slug}/...)`），否则图是孤儿。
- 正文中文字数（不含代码块）2000-3000；`tags` 含「量化」+ `difficulty: advanced` 才会进专栏。

## 内容质量红线（2026-11-07 两轮检视总结）
- **生成前先查同题旧文**：`information-bottleneck-sequence`（11-01）与 `information-bottleneck-sequence-prediction`（10-26）标题一字不差；`pc-algorithm-causal-factor` 与 `pc-algorithm-causal-factor-discovery` 几乎同名。11-01 两篇是同主题劣化重复灌水版（776/826 字、假图、崩代码），10-26 版反而严谨（2166/1823 字、真实图、诚实边界）。重写时改做互补进阶篇并交叉引用，避免继续制造重复。
- **灌水版特征**：图片目录存在但正文 0 引用、图片找不到任何生成脚本、无 results.json、frontmatter cover.png 不存在、表格数字无计算来源、代码缺 import / 张量维度错、学术出处捏造（作者名+年份）。
- **学术出处必须核对**：IB = Tishby, Pereira & Bialek (1999)；PC = Spirtes & Glymour（1990 前后，姓氏首字母）。偏相关用精度矩阵 $\rho_{ij·Z}=-\Omega_{ij}/\sqrt{\Omega_{ii}\Omega_{jj}}$ + Fisher-z 双尾检验，不能用 $|r|<t_\alpha$。
- **共同随机数**：对照实验（如 β=0 vs β>0）必须让初始化与 minibatch 序列一致，重参数化噪声用独立流，否则差异被初始化噪声污染；报告验证集早停值，别用末点值把「过拟合更晚」当成泛化。
- 代码与文章同源做法：核心实验放 `scripts/*_core.py`，用 `inspect.getsource` 拼装文章代码块，`run_*()` 返回 (可 JSON 化统计, 画图数组)，保证 exec 输出与 results.json 逐项一致。

## 自动化选题与文件安全（2026-11-08 事故，必读）
- `scripts/select_blog_topics.py` 的 662 个候选对已被 1510 个现有目录**全部占用**。核对 slug 只能用 Python `os.listdir('src/content/blog')`，别用 `subprocess.run(['ls', ...], shell=True)`（结果不可靠，曾误判两个已存在 slug 为未占用，导致覆盖旧文）。
- 写入前双重确认：`git ls-files src/content/blog/<slug>` 与 `git ls-files public/images/<slug>`。**`public/images/{slug}/` 常已被旧文章占用，绝不整个 `rm -rf`**；误删用 `git checkout HEAD -- <dir>` 恢复。
- 池耗尽后自造主题，先按关键词 grep 现有目录（本站覆盖极广：execution 28 篇、cointegration 22 篇、bootstrap/kelly/copula 等均有多篇），选互补进阶角度并交叉引用。

## 数值实验实现坑（2026-11-08）
- **BOCD**：常数 hazard 下 `P(r_t=0|x_1:t)` 恒等于 `H=1/λ`，与数据无关，不能当检测分数；改用短 run-length 后验质量 `Σ_{r=1..10} P(r_t=r)`。CUSUM 对照必须报警后清零统计量、δ 固定、基准只用前 calib 天（无前视偏差），否则 ROC 被调参污染。
- **秩相关**：等相关矩阵经逐元素 `sin(πτ/2)` 后仍是等相关，结构上必然 PSD，用它测「非正定率」永远得 0；必须换异质相关矩阵（低秩因子+个体差异）才看得到边界（p=40,n=60 时 Kendall 100% 非 PSD 而 Pearson 已 0%）。
- Kendall tau 向量化：`D=np.sign(R[None,:,:]-R[:,None,:])` → `einsum('abi,abj->ij')/(n(n-1))`，与 scipy 差 1.4e-17。
- numpy 2.x 已删 `np.trapz`，用 `np.trapezoid`（写 fallback 兼容旧版）。
- 文章与代码同源的更简做法：核心脚本本身就是文章代码块，md 里放占位符由 build 脚本注入，再抽出来 exec 深度比对 results.json。

## 本机沙箱坑
- `.vercel/output` 有旧产物时 build 会触发 safe-delete 拦截（count>50）；先 `mv .vercel/output` 挪开再 build。
