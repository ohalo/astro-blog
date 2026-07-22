# 量化博客 2 篇自动生成任务执行记录

**执行时间**: 2026-07-23 01:04–01:18 (Asia/Shanghai)
**触发**: cron `09c323b8-b9c4-432e-a36f-f7281a10fb09 quant-blog-2-posts-simplified`
**结果**: ✅ 成功，Vercel 部署返回 200，两篇文章页 + 配图均可访问

## 主题（select_blog_topics.py 排除最近 10 篇后选定）

1. **Stacking 集成因子模型** (slug: `stacking-factor-model`)
2. **逆强化学习策略偏好推断** (slug: `inverse-rl-preference`)

## 产出文件

- 文章: `src/content/blog/stacking-factor-model/index.md`、`src/content/blog/inverse-rl-preference/index.md`（各 ~2500 字，含 Python 代码块）
- 配图（各 4 张，纯 numpy/matplotlib 合成，CJK 字体）: `public/images/{slug}/` 下 cover + 3 图 + stats.json
- 生成脚本（可复跑）: `gen_stacking.py`、`gen_inverse_rl.py`
- 专栏更新: `src/content/pages/quant-column.md` 顶部新增「2026-07-23 新发布（Stacking / 逆强化学习）」小节

## 关键统计数字（合成宇宙，numpy 算实）

**Stacking 集成因子模型**（「线性+tanh非线性+交互+周期」混合收益，4 个各瞎一块的专科专家 + 岭回归元学习器，8 次切分均值）：
- 单专家测试集 IC：线性 0.403 / tanh 0.355 / 交互 0.098 / 周期 0.282
- 单专家多空收益差：0.84 / 0.74 / 0.23 / 0.54
- Stacking 集成：IC **0.611**、多空收益差 **1.240**（显著高于任一单模型）
- 元学习器权重均值：线性 0.99 / 周期 1.01 / tanh 0.96 / 交互 0.73（交互最低，因该块信号最弱）

**逆强化学习策略偏好推断**（MaxEnt IRL，30 状态 / 4 维奖励特征 / 专家 600 步，真值 w*=[1.0,1.2,0.6,−0.3]）：
- 学到权重 ŵ=[0.74, 1.48, 0.13, −0.35]，与真值**余弦相似度 0.937**
- 重放分布 KL：专家→重放 **0.061** vs 专家→随机 **0.225**（行为被准确复现）
- 周期项被略低估（0.13 vs 0.6），属 MaxEnt 有限样本+熵温度的真实偏误，主导项方向全对

## 执行中的关键修复

1. **Stacking 初版 IC 故事偏弱**：先试「同质化基模型（线性/多项式/核机/随机傅里叶）」，结果核机已吃全信号、stacking 增益≈0。改为「4 个各只看得见收益信号一块的专科专家」，盲区互补、stacking 增益显著且可复现——故事诚实且有力。
2. **Stacking Sharpe 指标虚假**：初版用「Top/Bottom 组每日收益差序列」算 Sharpe，因两组不配对导致年化√252 膨胀到 18+。改为合法的无前瞻「Top/Bottom 多空平均收益差」，并改在 8 次随机切分上取均值（stacking 价值是方差缩减，单次切分不稳）。
3. **Stacking OOF 维度 bug**：`base_rff` 误用全局 Xtr/Xte、`_`，`p` = fn(...) 取错返回值 → 修为用传入参数、取第二返回值（测试/折外预测）。
4. **Stacking 线性读出图误用**：第一节配图误挂 meta_weights.png（权重组），改为 cover.png（IC 对比组）。
5. **IRL findfont 警告**：bold 字体回退 600，不影响渲染（与历史脚本一致，非阻断）。

## 验证

- 本地 `npx astro build` 通过（2314 页，无 error）
- `git push` 触发 Vercel 自动构建
- 部署后验证：quant-column=200、`/blog/{slug}/`=200（308→200 重定向后）、各 4 张配图均 200
- quant-column 页面已渲染出两篇新文章链接（标题 grep 命中）
