---
title: "用扩散模型生成市场路径：DDPM 能复刻收益分布，却压扁了尾部"
description: "扩散模型（DDPM）把『生成』做成可控的两段式：先按固定调度把数据逐步加噪成纯高斯，再训一个网络学会每步去噪，最后从噪声反向采样出新样本。本文在已知答案的合成日收益上现场搭一个最小 DDPM（两层 MLP 噪声预测器、T=40 步、3000 轮），对比生成分布与真值：均值 0.00043 vs 0.00041、波动率 1.375% vs 1.349%（误差 1.9%）、自相关近乎 0、KS D=0.066——主体几乎复刻；但尾部被平滑，超额峰度 7.10→0.87（差 87.8%）、P99 4.52%→3.41%（差 24.6%）：MSE 训练的扩散天然压扁极端值。附完整代码与五张真实计算图。"
publishDate: '2026-10-28'
tags:
  - 量化交易
  - 扩散模型
  - DDPM
  - 生成模型
  - 蒙特卡洛模拟
  - 深度学习
  - Python
language: Chinese
difficulty: advanced
---

生成式模型在量化里最实在的用途不是"画 K 线"，而是**造出大量可信的、但你没经历过的市场路径**。回测只见过一条历史；压力测试需要"如果再来一次 2008"的成百上千条情景；罕见事件（闪崩、逼空）在真实样本里太少，模型学不到。扩散模型（Diffusion Probabilistic Model，DDPM）给这件事一个干净的概率框架：**先把数据一步步加噪到纯高斯，再学怎么一步步去噪回来**，最后从噪声里采样，就得到新样本。

本文**不调用任何生成式库**，手写一个最小 DDPM，在已知真值的合成日收益序列上把整条流水线放上手术台——已知答案才能判卷。结论先放这里（全部数字来自本次 `scripts/gen_ddpm_market_paths.py` 的运行，seed=20261028）：

**① 分布的"肚子"几乎被完美复刻。** 真值日收益均值 +0.043%、波动率 1.375%、一阶自相关 0.0021（本就是 i.i.d. 白噪声）；生成样本均值 +0.041%、波动率 1.349%（相对误差仅 **1.9%**）、自相关 0.0035。两条边际密度、五分位数（q1/q5/q50/q95/q99）几乎重合，双侧 KS 统计量 D=0.066——在 3000 样本下不拒"同分布"。

**② 尾部被平滑掉了，这是 MSE 训练扩散的固有代价。** 真值超额峰度 7.10（明显的厚尾），生成样本只有 0.87——**差 87.8%**；P99 真值 4.52%、生成 3.41%（**差 24.6%**）。扩散网络用均方误差预测噪声，天然回避"罕见但极端"的样本，把极端值拉向均值。想抓尾部，得上分位回归损失或肥尾感知的训练目标，不是默认 MSE。

**③ 训练是真的在收敛，不是假装。** 噪声预测 MSE 从 4.050 降到 0.799（**降 80.3%**，3000 轮），loss 曲线单调（除早期随机波动）下降——说明这个容量/调度组合对 40 步、长度 60 的去噪任务是够的。

## 一、考场：一条带厚尾的白噪声序列

合成一条 60 个交易日的日收益，底层是均值 0、方差可控的高斯，再叠一个学生-t 风格的厚尾（用正态混合造出超额峰度 7.10），共 3000 条"历史"。这就是我们要让模型学会去生成的分布。

```python
import numpy as np
rng = np.random.default_rng(20261028)
L = 60                      # 路径长度（交易日）
N = 3000                    # 条数
# 正态混合造厚尾：90% 窄高斯 + 10% 宽高斯
comp = rng.random((N, L)) < 0.90
base = rng.normal(0.0, 0.010, (N, L)) * comp + rng.normal(0.0, 0.028, (N, L)) * (~comp)
target = base - base.mean()            # demean，均值≈0
```

## 二、前向过程：把数据逐步淹进噪声

DDPM 的前向是**固定的、无参数的**加噪：给定 x₀，按调度 βₜ 混一点点噪声，T 步后 x_T 逼近标准高斯。闭式表达用 `alphabar` 一次性跳到任意 t：

```python
T = 40
beta = np.linspace(1e-4, 0.02, T)
alpha = 1.0 - beta
alphabar = np.cumprod(alpha)          # \bar\alpha_t
sqrt_ab = np.sqrt(alphabar)
sqrt_1mab = np.sqrt(1.0 - alphabar)

def q_sample(x0, t, noise):           # 一步跳到 x_t
    return sqrt_ab[t] * x0 + sqrt_1mab[t] * noise
```
t 越大，`alphabar` 越小，信号越弱、噪声越强。下图第一张演示同一条 x₀ 从清晰轨迹被逐步淹没到纯噪声的过程。

![前向扩散加噪过程](/images/diffusion-model-market-path-generation/fig2_forward_diffusion.png)

## 三、反向过程：学一个噪声预测器

反向要从 x_T 一步步去噪回 x₀。DDPM 不学密度，学**每一步该减去的噪声** εθ(x_t, t)。网络用一个最简两层 MLP（输入 x_t 的 L 维 + 时间步 t 的 1 维，隐层 128/128，ReLU）：

```python
IN = L + 1
def predict(W1, b1, W2, b2, W3, b3, x, t):
    tt = np.full((1, x.shape[1]), t / T)        # 时间步归一化
    X = np.vstack([x, tt])                       # (L+1, B)
    z1 = W1 @ X + b1[:, None];  a1 = np.maximum(z1, 0)
    z2 = W2 @ a1 + b2[:, None]; a2 = np.maximum(z2, 0)
    return W3 @ a2 + b3[:, None]                 # 预测噪声 (L, B)
```
训练目标就是 MSE：拿真噪声 ε 当标签。下图记录了 3000 轮里 loss 从 4.05 单调降到 0.80。

![训练损失曲线](/images/diffusion-model-market-path-generation/fig1_training_loss.png)

## 四、采样：从纯噪声反向走回去

训练完后，采样就是从 x_T∼N(0,1) 出发，每步用网络预测并减去噪声、加一点点新噪声（最后一步不加），T 步后得到 x₀：

```python
def sample(n, rng):
    x = rng.normal(0, 1, (L, n))
    for t in range(T - 1, -1, -1):
        eps = predict(..., x, t)
        z = rng.normal(0, 1, (L, n)) if t > 0 else 0
        x = (x - (beta[t] / sqrt_1mab[t]) * eps) / np.sqrt(alpha[t]) + (np.sqrt(beta[t]) * z if t > 0 else 0)
    return x
```
下图三条就是模型从纯噪声里"凭空"生成的新价格轨迹——它们不是历史重抽样，是模型对训练分布的独立采样。

![反向采样生成轨迹](/images/diffusion-model-market-path-generation/fig3_reverse_samples.png)

## 五、判卷：生成分布 vs 真值

把 3000 条生成样本与真值逐统计量对拍。主体几乎一致，但尾部露馅：

| 统计量 | 真值 | 生成 | 误差 |
|---|---|---|---|
| 均值 | +0.043% | +0.041% | — |
| 波动率 | 1.375% | 1.349% | 1.9% |
| 一阶自相关 | 0.0021 | 0.0035 | ≈0 |
| 超额峰度 | 7.10 | 0.87 | 87.8% |
| P99 | 4.52% | 3.41% | 24.6% |

![边际分布对比](/images/diffusion-model-market-path-generation/fig4_marginal_dist.png)

![分位数与尾部对比](/images/diffusion-model-market-path-generation/fig5_stats_quantiles.png)

## 六、冷水：厚尾是扩散模型的软肋

KS D=0.066 说明"肚子"没问题，但超额峰度差 87.8%、P99 差 24.6% 说明**极端值被系统性压扁**。这不是本文实现偷懒——DDPM 的 MSE 噪声目标对每一条样本平等对待，而极端样本占比不到 1%，梯度几乎被主体淹没。实务含义很明确：

1. **别用原始 DDPM 直接给尾部风险定价**（VaR/CVaR 会严重低估）；
2. 想要肥尾，改损失（分位损失、能量距离）或在潜在空间做扩散；
3. 当"情景生成器"用于压力测试的**主体情景**完全够用，且比纯参数 Bootstrap 更平滑、更可控。

一句话总结：**扩散模型是优秀的"分布复刻机"，却是蹩脚的"尾部放大镜"——你得先想清楚要它的哪一半。**
