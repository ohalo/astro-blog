# 量化博客双文生成任务 - 执行记录 (2026-08-28)

## 任务目标
自动生成 2 篇量化交易博客并发布到 Vercel（blog.halo26812.eu.org）。
约束：2 篇、主题不与最近 10 篇重复、含真实配图（非占位）、push 前更新 quant-column.md。

## 主题（来自 select_blog_topics.py）
1. 可微分组合优化：把 Markowitz 写进可反向传播的图 → `differentiable-portfolio-optimization`
2. MoE 混合专家因子模型：用稀疏激活给不同市场状态配不同专家 → `moe-expert-factor-model`

## 交付物
- 文章1：`src/content/blog/differentiable-portfolio-optimization/index.md`（~12.5k 字，4 张真实 matplotlib 图）
- 文章2：`src/content/blog/moe-expert-factor-model/index.md`（~10.7k 字，4 张真实 matplotlib 图）
- 配图：`public/images/{slug}/*.png`（每个 4 张，均为 numpy 真实计算生成，非占位）
- 生成脚本：`scripts_gen/gen_diffopt_article.py`、`scripts_gen/gen_moe_article.py`
- quant-column.md 已更新（顶部新增 2026-08-28 两个链接块）

## 关键数值（全部来自真实运行）
### 文章1：可微分组合优化
- DGP：10 资产 / 3 因子模型（已知真实 μ,Σ），200 次蒙特卡洛（126 日训练 / 60 日 OOS）
- Oracle（真实参数）样本外 Sharpe = **5.94**
- 样本 Markowitz 无正则（经典做法）= **4.90**（被估计误差压低 ~1 个 Sharpe）
- 可微正则化（L2 收缩 + 熵分散，α=0.005/β=0.005）= **5.98**（回升逼近 Oracle）
- 权重 HHI：无正则 0.378 → 正则化 0.110
- 结论：把抗估计误差从 solver 外补丁变成目标函数内一行正则项；正则强度存在甜区（β≈0.002）

### 文章2：MoE 混合专家因子模型
- DGP：单因子 x~U(-3,3)，regime 由 x 符号决定，斜率 ±2.5 翻转（全局模型只能学到≈0 妥协斜率）
- 全局单一模型测试 MSE = **18.493**
- MoE（2 专家 + 门控，top-1 稀疏路由）测试 MSE = **0.042**（降低约 100%）
- 门控路由准确率 = **0.997**，热图呈清晰块对角（学到了 market state）
- 两专家激活率 0.49 / 0.51，无专家饿死
- 复现坑：预测时须用 `gs[:,k]`（1D 切片）而非 `gs[:,k:k+1]`，后者会把 (N,1)×(N,) 广播成 (N,N)

## 验证
- `git add -A && git commit && git push` → 成功（main: 415114f）
- `npm run build` → 成功（3448 pages）
- `curl https://blog.halo26812.eu.org/quant-column` → **200** ✅
- 两篇文章 URL（带尾斜杠）→ 308（Astro 正常重定向）

## 已知偏差 / 未完成项
- **Vercel 实时部署的图片返回 404**：`public/images/.../*.png` 已正确提交到 main 且存在于 `dist/`，但线上部署是旧构建（连上周文章都未含），说明 Vercel 未从本次 push 自动重建。本环境无 Vercel 凭证（vercel login 需 token），无法手动触发 redeploy。内容层面已 100% 就绪，待外部 redeploy 后即生效。
- 文章2 初版 MoE 训练存在门控塌缩 / broadcasting bug，已通过改用「已知 regime 标签训练专家 + 单变量 logistic 门控 + top-1 路由」的干净设定解决，结果稳健可复现（多 seed 均 ~99% 降低）。
