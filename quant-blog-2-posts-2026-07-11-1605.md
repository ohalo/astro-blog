# 量化博客自动发布 — 执行记录 (2026-07-11)

## 任务
cron `quant-blog-2-posts-simplified`：生成 2 篇量化交易博客并发布到 Vercel (blog.halo26812.eu.org)。

## 选中的主题（脚本排除最近 10 篇后）
1. `diffusion-synthetic-market` — 时序扩散模型(Diffusion)生成合成行情用于压力测试
2. `attention-interpretability-quant` — 注意力机制的可解释性：用特征归因解释模型决策

## 交付物
- 文章：`src/content/blog/{slug}/index.md`（各 ~2000-3000 字，含可运行 Python 代码）
- 配图（真实 matplotlib 图表，非占位符，各 3 张）：
  - `public/images/diffusion-synthetic-market/`：前向加噪过程 / 真实vs合成尾部 / 训练损失+压力VaR
  - `public/images/attention-interpretability-quant/`：注意力热力图 / IG 特征归因 / 注意力vs IG 对比
- 图像生成脚本：`gen_images_diffusion.py`、`gen_images_attention.py`（纯 numpy，可复现，固定随机种子 20260711）
- `src/content/pages/quant-column.md`：在「2026-07-11 新发布」顶部新增 2 条链接

## 关键数值（来自脚本 _metrics.txt）
### 扩散模型
- 训练损失 2.33 → 0.90（噪声预测下界=1.0，说明 DDPM 学到位）
- 30日累计收益 VaR：真实 VaR95=0.207 / VaR99=0.339；合成 VaR95=0.286 / VaR99=0.403
- 结论：合成样本把历史稀疏尾部填厚填稳 → 压力 VaR 更稳；诚实陷阱：过弥散、不能凭空创造新危机、30维联合≠可交易路径、须看下游风控指标

### 注意力可解释性
- 模型拟合：训练 R²=0.91，测试 R²=0.78（早停取验证最佳）
- 只让 feature 3 携带信号；对比两种重要性：
  - Top-1 定位准确率：注意力 48.7% vs Integrated Gradients 62.5%
  - 真实特征占重要性比重：注意力 31.3% vs IG 40.8%
- 结论：注意力只暴露「看哪里」≠「哪里重要」；归因(IG)更锋利锁定真凶；softmax归一化/路由≠内容/mean-pooling 是结构性原因

## 调试要点（值得记的坑）
1. **扩散模型初始化过大** → 输出尺度≈16（应≈1）→ 梯度爆炸 NaN。改用 He/Xavier 初始化 + 梯度裁剪修复。
2. **GARCH+t(5) 扰动正反馈发散** → 改用高斯扰动（生产可换学生t）。
3. **注意力模型训练无效**：
   - 第一版信号太弱（label 噪声 0.35 ≫ 信号 0.06）→ R²≈0。强化信号。
   - 权重更新误除以 batch size(128) → 有效学习率小 40× → 不收敛。改回全梯度。
   - 用 epoch 起始的 `yhat[i]` 作滞后误差 → 不稳定。改为每样本用当前参数重算。
   - 时序均值信号难被该架构表达（欠拟合+过拟合并存）→ 改为 feature3 时序 = 固定形态×隐藏标量，label 仅由标量决定，可学且早停有效。

## 发布验证
- 本地 `bun run build` 成功（1073 页）
- `git add -A` + commit + push（main → 0a153bb）；清理了误提交的 `__pycache__/`
- Vercel 部署：初始 200 为旧构建；新构建约 50s 后完成
- 最终：`/quant-column` → 200；两篇文章 URL → 200；quant-column 页已含两条新链接
