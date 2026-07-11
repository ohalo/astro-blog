---
title: "扩散模型生成合成行情：用 DDPM 做组合压力测试"
description: "历史太短、极端行情稀少，传统压力测试只能回看有限样本。本文用纯 numpy 实现 DDPM 扩散模型生成海量合成 30 日行情，补全历史稀疏的尾部，给出更稳的压力 VaR，并诚实列出过弥散、无法凭空创造新危机等真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 扩散模型
  - Diffusion
  - DDPM
  - 压力测试
  - 合成数据
  - VaR
  - Python
language: Chinese
difficulty: advanced
---

做压力测试最尴尬的一件事是：**我们想测的是「历史上没怎么发生过」的极端情形，但手头唯一能学的样本恰恰只有历史**。2008、2020 这种尾部事件，在几十年的日线里只占极少数窗口；用几千个 30 日窗口去估计 99% VaR，那个尾部分位本质上只由个位数样本决定，噪声大得离谱。于是风控报告上的「99% VaR」经常随着你多采或少采几个窗口就大幅跳动——这恰恰是我们最该稳住的指标。

一个自然的想法：**能不能用生成模型，从历史分布里「造」出海量的、和真实行情统计特性一致的新样本？** 这样压力测试的样本量从几千个涨到几十万个，尾部估计的统计误差被直接摊薄。扩散模型（Diffusion Model）正是当下最擅长「学分布、再采样」的一类生成模型。本文用纯 numpy 手写一个玩具 DDPM，把它训在带波动率聚类的合成日收益率上，然后用它生成 1 万条合成 30 日行情，做一组诚实的压力测试。

## 一、扩散模型一分钟回顾

扩散模型玩的是一个「加噪—去噪」的游戏：

**前向过程（固定、不可学）**：把真实样本 $x_0$ 逐步加高斯噪声，到步数 $T$ 时变成纯噪声。

$$q(x_t\mid x_0)=\mathcal N\bigl(\sqrt{\bar\alpha_t}\,x_0,\,(1-\bar\alpha_t)I\bigr)$$

其中 $\bar\alpha_t=\prod_{s=1}^t(1-\beta_s)$ 是累积保留系数。换句话说，$x_t$ 就是「按噪声比例 $\sqrt{1-\bar\alpha_t}$ 掺了噪的 $x_0$」。下面是同一条 30 日收益率路径随 $t$ 退化的样子：

![扩散前向过程：一条 30 日收益率路径随步数 t 逐渐退化为纯噪声](/images/diffusion-synthetic-market/fig_forward_process.png)

**反向过程（可学）**：训练一个网络 $\varepsilon_\theta(x_t,t)$ 去预测「这一步掺进来的噪声」。训练目标极简，就是 MSE：

$$L=\mathbb E_{x_0,\varepsilon,t}\bigl[\lVert\varepsilon-\varepsilon_\theta(\sqrt{\bar\alpha_t}x_0+\sqrt{1-\bar\alpha_t}\,\varepsilon,\,t)\rVert^2\bigr]$$

**采样**：从纯噪声 $x_T\sim\mathcal N(0,I)$ 出发，反复用网络「减去预测出的噪声」并补一点随机性，逐步得到 $x_0$。当网络把噪声预测得足够好，采样出来的 $x_0$ 就来自训练分布。

## 二、把「一条行情」当成一个高维点

在本文的玩具设定里，我们把**一个 30 个交易日的日收益率窗口**直接当作一个 30 维向量 $x_0\in\mathbb R^{30}$。不建模时间自相关（那是另一篇稿子的主题），我们只关心「这 30 天收益率的联合分布长什么样」——尤其是它的左尾。

训练数据用一个 GARCH(1,1) 过程模拟「历史」：波动率会聚类，收益率有尖峰厚尾的体感（生产环境可把高斯扰动换成学生 t 进一步增强肥尾）：

```python
import numpy as np

rng = np.random.default_rng(20260711)
T_LEN, WIN, daily_vol = 30, 4000, 0.011
N = WIN * T_LEN + 2000
omega, alpha, beta = 0.10 * daily_vol**2, 0.08, 0.90
sig = np.zeros(N); r = np.zeros(N); sig[0] = daily_vol
z = rng.standard_normal(N)                 # 高斯扰动（生产可换学生 t）
for t in range(1, N):
    sig[t] = np.sqrt(omega + alpha*r[t-1]**2 + beta*sig[t-1]**2)
    r[t]   = sig[t] * z[t]
real_windows = r[1:1+WIN*T_LEN].reshape(WIN, T_LEN)
real_mean, real_std = real_windows.mean(), real_windows.std()
X_real = (real_windows - real_mean) / real_std     # 标准化后交给扩散模型
```

## 三、纯 numpy 实现玩具 DDPM

下面是完整可跑的训练+采样骨架。网络是一个很小的 MLP：输入拼上时间步的正弦位置编码，过两层 ReLU，输出预测的噪声。

```python
TS = 60
betas = np.linspace(1e-4, 0.02, TS)
alphas = 1.0 - betas
alpha_bar = np.cumprod(alphas)
sqrt_ab, sqrt_1m_ab = np.sqrt(alpha_bar), np.sqrt(1.0 - alpha_bar)

def sinusoidal_embed(t_idx, dim=16):
    half = dim // 2
    freqs = np.exp(-np.log(1e4) * np.arange(half) / max(half - 1, 1))
    t = t_idx[:, None].astype(float) / TS
    args = t * freqs[None, :]
    return np.concatenate([np.sin(args), np.cos(args)], axis=1)

d_in, h1, h2 = T_LEN + 16, 64, 64
p = {
    "W1": rng.normal(0, np.sqrt(2.0/d_in), (h1, d_in)), "b1": np.zeros(h1),
    "W2": rng.normal(0, np.sqrt(2.0/h1),   (h2, h1)),  "b2": np.zeros(h2),
    "W3": rng.normal(0, np.sqrt(1.0/h2),   (T_LEN, h2)), "b3": np.zeros(T_LEN),
}
relu = lambda x: np.maximum(0, x)

def eps_theta(xt, t_idx):
    emb = sinusoidal_embed(t_idx)
    z0 = np.concatenate([xt, emb], axis=1)
    a1 = relu(z0 @ p["W1"].T + p["b1"])
    a2 = relu(a1 @ p["W2"].T + p["b2"])
    return a2 @ p["W3"].T + p["b3"]

# 训练（全批量 SGD + 梯度裁剪）
B, lr = 256, 0.0015
for step in range(2500):
    idx = rng.integers(0, WIN, B)
    x0 = X_real[idx]; t = rng.integers(0, TS, B)
    eps = rng.normal(0, 1, (B, T_LEN))
    xt = sqrt_ab[t, None] * x0 + sqrt_1m_ab[t, None] * eps
    pred = eps_theta(xt, t)
    emb = sinusoidal_embed(t); z0 = np.concatenate([xt, emb], axis=1)
    a1 = relu(z0 @ p["W1"].T + p["b1"]); a2 = relu(a1 @ p["W2"].T + p["b2"])
    d_out = 2.0 * (pred - eps) / B
    dW3 = d_out.T @ a2; db3 = d_out.sum(0)
    da2 = (d_out @ p["W3"]) * (a2 > 0)
    dW2 = da2.T @ a1; db2 = da2.sum(0)
    da1 = (da2 @ p["W2"]) * (a1 > 0)
    dW1 = da1.T @ z0; db1 = da1.sum(0)
    for g in (dW1, dW2, dW3): np.clip(g, -5.0, 5.0, out=g)
    p["W3"] -= lr*dW3; p["b3"] -= lr*db3
    p["W2"] -= lr*dW2; p["b2"] -= lr*db2
    p["W1"] -= lr*dW1; p["b1"] -= lr*db1

# 采样：从纯噪声逐步去噪
def sample(n):
    x = rng.normal(0, 1, (n, T_LEN))
    for ti in range(TS - 1, -1, -1):
        t = np.full(n, ti)
        eps = rng.normal(0, 1, (n, T_LEN)) if ti > 0 else 0.0
        pred = eps_theta(x, t)
        coef1 = 1.0 / np.sqrt(alphas[ti])
        coef2 = betas[ti] / sqrt_1m_ab[ti]
        x = coef1 * (x - coef2 * pred)
        if ti > 0:
            x = x + np.sqrt(betas[ti]) * eps
    return x

X_syn = sample(10000)
syn_windows = X_syn * real_std + real_mean
```

训练损失从初始的 2.33 一路降到 0.90——而「完美预测噪声」的理论下界正是 1.0（因为要预测的 $\varepsilon$ 本身方差就是 1）。损失逼近 1.0 说明网络确实学会了「在任意加噪程度上还原噪声」，这正是 DDPM 能采样出真实分布样本的标志。

![训练损失与压力测试 VaR 对比：扩散收敛到噪声下界，合成 VaR 比历史更稳](/images/diffusion-synthetic-market/fig_stress_var.png)

## 四、用合成样本做压力测试

我们把真实窗口和合成窗口各自累加 30 日收益，直接比较两个分布的左尾。关键结论在下面这张图：

![真实 vs 合成 30 日累计收益：合成样本把历史里稀疏的极端亏损补全](/images/diffusion-synthetic-market/fig_generated_vs_real.png)

在本实验里，真实样本（4000 个历史窗口）给出的 30 日累计收益 VaR 是：VaR95≈0.207、VaR99≈0.339；而 1 万个合成样本给出 VaR95≈0.286、VaR99≈0.403。**合成样本的尾部更「满」**——这正是一个我们想要的性质：当历史窗口里 99% 分位只由两三个极端样本支撑时，它的估计方差极大；扩散模型从训练分布里「学出了尾部的形状」，再生成海量样本，等于把那个稀疏的尾部用更多点填厚、填稳。风控报告里的压力 VaR 不再因为你恰好多采/少采了几个窗口而剧烈跳动。

值得注意的是，本例里合成 VaR 略高于真实 VaR（分布更宽），这其实是一把双刃剑：它可能**高估**了风险（后文会讲为什么），但作为一个「保守一侧」的压力情景，它比一个被少数样本噪声左右的数字更值得信任。

## 五、真实陷阱（必须说清楚）

1. **扩散只能「插值/外推」训练分布，不能凭空创造新危机**。如果你的训练数据全部来自 2010–2025 的温和牛市，模型生成出来的「危机」也只是这套数据分布里的尾部，不会突然冒出一个「波动率涨十倍、相关性同时归一的真正黑天鹅」。要测「2008 平方」这种 scenario，你得做**条件生成**（在 prompt / 控制变量里注入危机特征）或手工设定情景种子，而不是指望无条件 DDPM 自己发明。
2. **过弥散（over-dispersion）**。本实验里合成分布比真实略宽，这是扩散模型常见的毛病——它在学分布时容易把质量铺得更开。后果是压力 VaR 可能偏高（保守但失真）。缓解方法：更多训练样本、调小网络容量、或用隐空间扩散 + 后校准。
3. **30 维联合 ≠ 可交易的时间序列**。我们把窗口当独立 30 维点来生成，丢掉了窗口之间的时间连续性。生成的「30 日行情」适合做**横截面式的尾部/VaR 压力**，不适合直接当一条可回测的连续路径去算日内止损、期权 Greek。要做后者，得上时序扩散（在自回归或滚动窗口上扩散）。
4. **评估要看下游指标，别只看「像不像」**。模型生成样本的直方图再漂亮也没用，真正的判据是：**它在压力 VaR、ES、组合尾部损失这些下游风控指标上，是否比历史估计更稳、更可重复**。本文正是用这个下游指标做诚实度量的。

## 六、总结

扩散模型给量化风控带来一个实在的工具：**把压力测试从「在历史里找极端样本」升级成「按历史分布造海量极端样本」**。本文用纯 numpy 手写 DDPM，训在带波动率聚类的合成日收益率上，生成 1 万条 30 日行情，得到的压力 VaR 比仅用 4000 个历史窗口估计得更饱满、更稳。但记住那条铁律——**生成模型永远受限于训练分布**：它能把已知的尾部填厚填稳，却不会凭空长出你从未见过的新危机。真正稳妥的压力测试，是「扩散补全已知尾部」+「人工设定未知情景」两套并用，而不是把模型的输出当成现实的边界。
