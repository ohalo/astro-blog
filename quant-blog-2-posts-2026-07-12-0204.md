# 每日博客发布 — 2026-07-12 02:04 (cron: quant-blog-2-posts-simplified)

## 发布文章（2 篇，均通过本地 `astro build` 校验，已部署 Vercel 返回 200）

### 1. 粒子滤波估计随机波动率：当 MCMC 太慢时的序贯解法
- **Slug**: `particle-filter-sv`
- **URL**: https://blog.halo26812.eu.org/blog/particle-filter-sv/
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `pf_particles.png` 真实隐状态 h_t + Bootstrap 粒子云 + 滤波均值
  - `pf_filtered_credible.png` 滤波均值 vs 真值 + 90% 可信带（RMSE）
  - `pf_ess.png` Bootstrap PF vs APF 的有效样本数(ESS)时序
  - `pf_convergence.png` 滤波 RMSE 与 SMC 对数似然随粒子数 N 收敛
- **核心结论（真实计算）**: T=800, μ=-9.0, φ=0.97, σ_η=0.60；Bootstrap PF 滤波 RMSE(h_t)=1.378（解释约 69.6% 隐状态波动），平均 ESS≈1483/2000；APF 平均 ESS≈1760/2000（退化更慢）；SMC 对数似然≈-1961.89。
- **关键方法修正（踩坑）**: ① 对原始收益 r_t~N(0,exp(h)) 直接滤波几乎不可追踪（RMSE≈信号自身 std，等于没滤波）——必须改用对数平方收益 z_t=log(r_t²)、用 log(χ²₁) 似然；② 精确 log(χ²₁) 众数在 0、均值在 -1.27，叠有信息量先验会产生水平偏差，改用其 Gaussian 近似(μ=-1.2704, σ²=π²/2) 才无偏；③ 收敛扫描显示 RMSE 触顶（观测噪声地板）、SMC 对数似然随 N 收敛——N 决定边际似然精度而非隐状态信息上限。

### 2. VIX 期限结构与波动率风险溢价：跨期结构
- **Slug**: `vix-term-structure`
- **URL**: https://blog.halo26812.eu.org/blog/vix-term-structure/
- **配图**: 4 张（均由文中 Python 真实计算生成）
  - `vix_term_structure.png` contango 与 backwardation 两种状态的期限结构快照
  - `vix_term_slope.png` 期限结构斜率时序（contango/backwardation 切换）
  - `vix_vrp_term.png` VRP 期限结构（短端最肥、随期限递减）
  - `vix_strategy.png` 做多曲线 carry（卖近买远）权益曲线
- **核心结论（真实计算，合成但自洽模型）**: 模型 IV(M)=LV+(V0-LV)e^{-M/τ}+VRP0·e^{-M/τ_vrp}，LV=20、V0 均值 5、带 Poisson 危机跳；contango 占比 81.5%、backwardation 18.5%；VRP 期限结构均值 1M→6M = 1.49%→0.74%→0.37%→0.18%→0.09%→0.04%（短端最肥）；做多曲线 carry 累计 +123.6 vol 点、最大回撤 -60.2（危机翻转日近端跳涨、空头爆亏）。
- **关键方法修正（踩坑）**: ① 初版 V0 缩放错误（V0≈0.2 而非 ~15-18）+ carry 方向写反 → 重写为 V0=LV+shock 且 shock 去均值使 contango 为常态；② carry 循环误用 t 而非 t+1 估值导致零回撤 → 修正为「次日期限下滑一个档」重新估值，才复现真实 gap risk（回撤 -60.2）；③ 平滑 AR(1) 无法建模跳空，注入 Poisson 向上跳才有真实危机片段与回撤。

## 工程要点
- 主题由 `select_blog_topics.py` 选择（已排除最近 10 篇；粒子滤波/ MCMC-SV 与 VIX-期限结构均为不同角度，不与已有 stochastic-volatility-mcmc / volatility-risk-premium 重复）。
- `quant-column.md` 已在「2026-07-12 新发布」顶部插入两篇新链接（已 curl 确认含 /blog/particle-filter-sv/ 与 /blog/vix-term-structure/）。
- 图片生成脚本 `generate_particle_filter_images.py` / `generate_vix_term_structure_images.py` 一并提交，保证图表可复现。
- 两篇文章 description 均 < 150 字符，构建无 schema 报错。

## 验证结果
- `astro build`: 1159 页构建成功（EXIT 0，Complete!）
- `git push`: 成功（main 222b986..012552f）
- 部署后 curl 校验（等待 Vercel 重建后）：
  - `/quant-column` → HTTP 200
  - 两篇新博客 `/blog/{slug}/` → HTTP 200（标题正确渲染）
  - 抽样配图 `/images/...png` → HTTP 200
  - quant-column 页面含两篇新链接 ✅
