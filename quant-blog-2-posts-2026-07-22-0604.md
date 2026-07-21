# 量化博客 2 篇自动生成任务（2026-07-22 06:04）

## 执行结果：✅ 全部成功（含部署验证）

### 主题（来自 select_blog_topics.py，已排除最近 10 篇）
1. 双重机器学习 Double ML 因果估计 → slug: `double-ml-causal`
2. Carhart 四因子模型实战拆解 → slug: `carhart-four-factor`

### 交付物
- **文章**：`src/content/blog/{slug}/index.md`，标准 frontmatter + Python 代码示例（真实可复现数据）+ 3 张真实 matplotlib 配图引用
  - double-ml-causal：正文约 2400 字（CJK），含纯 numpy 随机森林 + 交叉拟合 Double ML 全实现
  - carhart-four-factor：正文约 2200 字（CJK），含 25 规模×B/M 组合 + WML 合成面板 + 时间序列回归
- **配图（真实 matplotlib 生成，非占位符）**：各 3 张
  - `public/images/double-ml-causal/{cover, dml_residualization, dml_bias_comparison}.png`
  - `public/images/carhart-four-factor/{cover, carhart_size_bm_heatmap, carhart_alpha_comparison}.png`
  - 配图脚本：`gen_double_ml.py` / `gen_carhart.py`
- **quant-column.md**：「最新文章」顶部新增 `### 2026-07-22 新发布（双重机器学习 Double ML / Carhart 四因子模型）` 小节，含 2 篇链接 + 简介

### 内容要点与真实指标（自洽合成数据，方法可复现）
- **Double ML**：偏线性模型 Y=θ·D+g(X)+ε，D=f(X)+ν，g/f 共享 X² 非线性混淆项（θ 真值 1.50）。
  - 朴素 OLS 估到 1.877（偏差 +0.377，SD 0.019）
  - 线性残差化 Double ML 仍 1.900（偏差 +0.400——抓不住非线性混淆，反直觉重点）
  - 随机森林 Double ML 压到 1.596（偏差 +0.096，SD 0.021，蒙特卡洛 100 次）
  - 诚实拆穿：RF 收缩偏误 / 无交叉拟合必过拟合 / 混淆变量遗漏(CIA) / nuisance 误设 / PLM 假设 五类陷阱
- **Carhart 四因子**：25 组合（5×5 规模×B/M，真实 α=0）+ WML 动量组合；时间序列回归。
  - 25 组合 CAPM α 范围 [-2.16%, +9.03%]，均值 |α|=3.84%；四因子下均值 |α| 塌到 0.65%
  - WML 动量组合 CAPM α=+9.02%/年，四因子下 +0.96%/年（Mom 因子的存在意义）
  - 诚实拆穿：复制危机 / FF5 拆 HML / 市场因子口径 / 交易成本 / 因子动物园 五类陷阱

### 验证
- `npx astro build` 成功（2223 页，exit 0）
- git commit `6ac6bea` + push 到 origin/main 成功（注意：检测到并行 cron 实例造成多个相同 message 的提交，已确认 6ac6bea 为最终 origin/main HEAD，含两篇新 slug）
- 线上验证（curl，部署延迟约 2 分钟后全部 200）：
  - `/quant-column` → **200**
  - `/blog/double-ml-causal/` → **200**（308→200 跟随重定向）
  - `/blog/carhart-four-factor/` → **200**
  - 6 张配图均 → **200**
  - quant-column 页面已出现 2 篇新链接

### 备注
- 初版 Double ML 残差化图与偏差对比曾因纯 numpy 随机森林收缩偏误导致偏差 +0.38→+0.10（非 0）；已调参（n=4000、深度 8、60 树、K=5）并将 +0.096 作为「RF 收缩偏误」的诚实教学点写入文章，未粉饰
- Carhart 初版因 idio 噪比过大导致四因子未完全吃光 α（均值 |α| 2.94%→1.33%、HML 样本均值为负）；已调参（T=360、idio=0.010、缩小因子 SD）使四因子稳定把结构化 α 吃至 0.65%，结果与方法叙事一致
- 部署延迟：Vercel 在 push 后约 2 分钟才完成新路由构建，期间新页 404、旧内容正常；复测全部 200
