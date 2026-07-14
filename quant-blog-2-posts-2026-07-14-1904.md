# 量化博客自动发布（2026-07-14 第 2 批 2 篇）

## 任务
按 cron `quant-blog-2-posts-simplified` 执行：选 2 个不重复主题 → 写 2000-3000 字文章（含 Python 代码 + 真实配图）→ 更新 quant-column.md → git 提交推送 → Vercel 验证。

## 主题选取
`select_blog_topics.py` 主池用尽，兜底选了 2 个与最近 10 篇不重复的主题：
1. **FTRL 在线学习组合：用正则化跟随引领历史** (slug: `ftrl-online-portfolio`)
2. **注意力机制因子：让模型自己决定看哪些特征** (slug: `attention-factor`)

## 关键数字（均为自洽合成数据，算法机制演示用）
### FTRL-Proximal 在线组合（α=0.10, β=1, λ1=0.02, λ2=1.0, 6 资产/756 日/带赢家切换+崩盘）
- 总收益 **80.1%**，年化 21.67%，Sharpe **0.94**，波动 23.8%，MDD −24.8%，日均换手 **3.9%**，L1 稀疏度 **41%**
- 对比：UCRP 39.7%/Sharpe 0.61，OLMAR 36.5%/0.46，等权买入持有 27.8%，事后最优单资产 106.2%（作弊上界）
- 20 种子：FTRL Sharpe 0.73±0.27 vs UCRP 0.62±0.03，**12/20 跑赢**

### 注意力机制因子（F=10 因子/20 日窗/1600 样本，仅动量携带信号）
- 训练 MSE 0.073 / 测试 0.105（轻微过拟合）
- 注意力入度 vs 真值 one-hot IC=**0.999**；IG 归因 IC=**0.994**（两者均锁定动量，但 IG 更干净）
- 门控注意力因子 vs 真信号：IC=0.965, R²=**0.590**；等权平均因子：IC=0.534, R²=0.151（门控 3.9 倍）

## 调试踩坑（关键经验）
1. **FTRL 梯度必须用净收益 `g = 1 - X[t]`**，不能拿价格相对 `X[t]≈1` 当梯度——否则 `z` 量级错乱、权重被数值问题打回均匀，FTRL 静默退化成 UCRP（收益从 80% 塌到 39.7%）。
2. **注意力训练不能用单样本 Adam**（发散爆炸）；改用参考脚本的 minibatch SGD（lr=0.01, 128 batch, 验证集早停 5000 轮）才收敛。
3. **Astro content schema 对 `description` 有 160 字符上限**（首轮 push 后 build 报错 InvalidContentEntryDataError）。两篇 description 都超限，缩短到 ≤160 后 build 通过。
4. Vercel 部署有 ~2 分钟 build 延迟：首轮验证新文章 404，是旧构建未刷新，不是代码缺陷；等待后复测全部 200。

## 交付物
- `src/content/blog/ftrl-online-portfolio/index.md`（180 行 / ~14.9K 字符）
- `src/content/blog/attention-factor/index.md`（158 行 / ~12.2K 字符）
- 各 4 张真实配图：`public/images/ftrl-online-portfolio/{ftrl_equity,ftrl_rates,ftrl_weights,ftrl_params}.png`、`public/images/attention-factor/{attn_matrix,attn_importance,attn_gate_vs_equal,attn_train}.png`
- `src/content/pages/quant-column.md` 顶部新增 2 条（2026-07-14 新发布）
- 脚本：`generate_ftrl_online_portfolio_images.py`、`generate_attention_factor_images.py`
- Git：`a521794`（文章+图）→ `1221ac7`（description 缩短修复）

## 验证结果（最终）
| 资源 | HTTP |
|---|---|
| /quant-column | 200 |
| /blog/ftrl-online-portfolio/ | 200 |
| /blog/attention-factor/ | 200 |
| /images/ftrl-online-portfolio/ftrl_equity.png | 200 |
| /images/attention-factor/attn_matrix.png | 200 |

✅ 全部 200，部署成功。
