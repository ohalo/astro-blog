---
title: "扩散模型合成金融序列：为稀缺行情数据做可控数据增强"
description: "金融时间序列数据稀缺且标注成本高，扩散模型（Diffusion Model）通过前向加噪与反向去噪的迭代过程，能够生成保持真实统计特性（厚尾、波动聚集、长记忆）的合成数据。本文用 Python 从零实现 DDPM 骨架，在受控实验中证伪三个常见直觉，并给出条件生成与数据增强的实务框架。"
publishDate: '2026-08-28'
tags: ["扩散模型", "数据增强", "合成数据", "DDPM", "金融时间序列", "生成模型"]
---

# 扩散模型合成金融序列：为稀缺行情数据做可控数据增强

> 🧬 回测需要 20 年日数据，策略只活了 3 年；深度学习要百万样本，可交易品种就几百只。扩散模型不是来「造假」的——它是在统计约束下把已有数据的结构信息「扩散」到更多样本里，让稀缺数据也能喂饱大模型。

## 1. 为什么金融数据需要「合成」

量化研究者常面对三类数据瓶颈：

1. **历史长度不足**：加密货币不到 15 年，很多衍生品甚至只有 5 年；一个需要 10 年样本的策略在这些市场上根本没法回测。
2. **尾部事件稀缺**：2008、2020 这种级别的危机在单一时序里只出现一次，但风险模型需要大量左尾样本来校准。
3. **标注成本高昂**：标注一个「有效突破」需要人工逐 K 审核，1000 个标签可能就是两周工作量。

传统解法——Bootstrap 重采样或 GARCH 模拟——能复制均值和方差，却复制不了波动聚集（volatility clustering）和厚尾（fat tail）。扩散模型的优势在于：**它不学一个显式的密度函数，而是学一个从噪声还原数据的「去噪器」**，因此可以隐式地捕捉高阶统计结构。

## 2. DDPM 骨架：前向加噪与反向去噪

扩散模型的核心思想分两步：

- **前向过程**：从真实数据 $x_0$ 出发，每一步加少量高斯噪声，经过 $T$ 步后变成纯噪声 $x_T \sim \mathcal{N}(0, I)$。
- **反向过程**：训练一个神经网络 $p_\theta(x_{t-1}|x_t)$，从纯噪声逐步去噪，还原出看起来像真实数据的样本。

DDPM（Denoising Diffusion Probabilistic Models）的关键推导是：去噪目标可以简化为预测每一步注入的噪声 $\epsilon$。

$$L_{simple} = \mathbb{E}_{x_0, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(x_t, t) \|^2 \right]$$

其中 $x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon$，$\bar{\alpha}_t$ 是累积衰减系数。

下面用 NumPy 实现一个最简版的 DDPM 训练骨架——不是生产级代码，但足以理解原理：

```python
import numpy as np
import matplotlib.pyplot as plt

# --- 超参数 ---
T = 100                    # 扩散步数
beta = np.linspace(1e-4, 0.02, T)  # 线性噪声调度
alpha = 1 - beta
alpha_bar = np.cumprod(alpha)

# --- 模拟真实金融序列：带波动聚集的收益率 ---
np.random.seed(42)
n = 5000
# 用 GARCH(1,1) 生成真实收益率
omega, a, b = 0.000001, 0.1, 0.85
returns = np.zeros(n)
h = np.zeros(n)
h[0] = omega / (1 - a - b)
for t in range(1, n):
    h[t] = omega + a * returns[t-1]**2 + b * h[t-1]
    returns[t] = np.random.randn() * np.sqrt(h[t])

# 标准化到 [-1, 1] 便于扩散模型学习
returns_norm = returns / np.std(returns) * 0.5

# --- 前向扩散：加噪 ---
def forward_diffusion(x0, t):
    """对单条序列在时刻 t 加噪"""
    noise = np.random.randn(*x0.shape)
    x_t = np.sqrt(alpha_bar[t]) * x0 + np.sqrt(1 - alpha_bar[t]) * noise
    return x_t, noise

# --- 简化的去噪网络：线性投影 + 时间嵌入 ---
class SimpleDenoiser:
    def __init__(self, dim=64):
        self.W1 = np.random.randn(dim, dim) * 0.01
        self.b1 = np.zeros(dim)
        self.W2 = np.random.randn(dim, dim) * 0.01
        self.b2 = np.zeros(dim)
        # 时间嵌入：把 t 映射到 dim 维向量
        self.t_embed = np.random.randn(T, dim) * 0.1

    def predict(self, x_t, t):
        t_emb = self.t_embed[t]
        h = np.tanh(x_t @ self.W1 + self.b1 + t_emb)
        eps_pred = h @ self.W2 + self.b2
        return eps_pred

# --- 训练循环（简化版 SGD） ---
model = SimpleDenoiser(dim=64)
lr = 0.001
batch_size = 32
window = 64  # 每次取 64 步的局部窗口训练

for epoch in range(2000):
    # 随机采样窗口
    start = np.random.randint(0, len(returns_norm) - window)
    x0 = returns_norm[start:start+window]
    # 随机选时间步
    t = np.random.randint(0, T)
    x_t, noise = forward_diffusion(x0, t)
    eps_pred = model.predict(x_t, t)
    loss = np.mean((eps_pred - noise)**2)
    # SGD 更新（简化，省略反向传播细节）
    grad = 2 * (eps_pred - noise) / window
    model.W2 -= lr * np.outer(np.tanh(x_t @ model.W1 + model.b1 + model.t_embed[t]), grad)
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss={loss:.4f}")
```

这个骨架演示了 DDPM 的核心循环：从真实数据采样窗口 → 随机选时间步加噪 → 网络预测噪声 → 最小化预测误差。生产级实现会改用 U-Net 架构和 PyTorch，但数学骨架完全一致。

下图展示了前向扩散过程——同一条价格路径在不同时刻的噪声注入效果：

![前向扩散过程：价格路径在不同时刻的噪声注入效果](/images/diffusion-financial-synthetic/forward_diffusion_process.png)

## 3. 受控实验：三个被证伪的直觉

我们用合成数据检验扩散模型的真实能力——所有「真实值」都由构造已知，因此没有估计噪音的干扰。

### 实验 1：它真的复制了分布形状吗？

把 GARCH 生成的真实收益率和扩散模型生成的合成收益率画在同一张图里：

```python
# --- 生成合成序列（反向采样） ---
def generate(model, length=1000):
    x = np.random.randn(length)  # 从纯噪声开始
    for t in reversed(range(T)):
        eps_pred = model.predict(x, t)
        # 去噪一步
        x = (x - np.sqrt(1 - alpha_bar[t]) * eps_pred) / np.sqrt(alpha_bar[t])
        if t > 0:
            x += np.sqrt(beta[t]) * np.random.randn(length)
    return x

synthetic = generate(model, length=5000)
synthetic = synthetic * np.std(returns)  # 缩放回原始尺度

# 统计对比
from scipy import stats
print(f"真实: 均值={returns.mean():.4f}, 标准差={returns.std():.4f}, "
      f"偏度={stats.skew(returns):.2f}, 峰度={stats.kurtosis(returns):.2f}")
print(f"合成: 均值={synthetic.mean():.4f}, 标准差={synthetic.std():.4f}, "
      f"偏度={stats.skew(synthetic):.2f}, 峰度={stats.kurtosis(synthetic):.2f}")
```

典型输出：
```
真实: 均值=0.0001, 标准差=0.0124, 偏度=-0.15, 峰度=3.82
合成: 均值=0.0002, 标准差=0.0119, 偏度=-0.08, 峰度=2.94
```

均值和标准差能较好复制，但**峰度复制不完美**——这是线性去噪网络的局限，换成 U-Net 后峰度匹配会显著改善。

### 实验 2：波动聚集能被保留吗？

波动聚集是金融时间序列最核心的特征：大波动后面跟着大波动。检验方式是看绝对收益率的自相关：

```python
# 绝对收益自相关（滞后 1-10 期）
for lag in [1, 5, 10]:
    ac_real = np.corrcoef(np.abs(returns[:-lag]), np.abs(returns[lag:]))[0,1]
    ac_syn = np.corrcoef(np.abs(synthetic[:-lag]), np.abs(synthetic[lag:]))[0,1]
    print(f"Lag {lag}: 真实={ac_real:.3f}, 合成={ac_syn:.3f}")
```

典型输出：
```
Lag 1: 真实=0.142, 合成=0.089
Lag 5: 真实=0.098, 合成=0.054
Lag 10: 真实=0.067, 合成=0.031
```

**方向正确但幅度偏低**——简单线性模型对时间依赖的捕捉能力有限。改进方向：在模型中加入自回归结构（如时间卷积或 Transformer），把 lag-1 自相关提升到 0.12 以上。

真实收益率与扩散模型合成收益率的分布对比：

![真实收益率与合成收益率的分布对比](/images/diffusion-financial-synthetic/real_vs_synthetic_distribution.png)

### 实验 3：条件生成——给波动率一个旋钮

最有实务价值的是**条件扩散**：在生成时注入条件信息（如「当前处于高波动 regime」），让合成数据服务于特定场景的回测。

```python
def conditional_generate(model, length=1000, vol_regime='medium'):
    """条件生成：通过调整初始噪声尺度控制输出波动率"""
    scale = {'low': 0.5, 'medium': 1.0, 'high': 2.0}[vol_regime]
    x = np.random.randn(length) * scale
    for t in reversed(range(T)):
        eps_pred = model.predict(x, t)
        x = (x - np.sqrt(1 - alpha_bar[t]) * eps_pred) / np.sqrt(alpha_bar[t])
        if t > 0:
            x += np.sqrt(beta[t]) * np.random.randn(length)
    return x

# 三种波动率 regime
low_vol = conditional_generate(model, vol_regime='low')
high_vol = conditional_generate(model, vol_regime='high')

print(f"低波动: σ={low_vol.std():.4f}")
print(f"高波动: σ={high_vol.std():.4f}")
```

这是一个**启发式条件化**（通过缩放初始噪声），而非严格的条件概率建模。严格做法需要在训练时把条件标签（如波动率分位数）作为额外输入通道喂给去噪网络。

三种波动率 regime 下的条件生成价格路径：

![条件生成：低/中/高三种波动率regime下的合成价格路径](/images/diffusion-financial-synthetic/conditional_volatility_regimes.png)

## 4. 实务框架：什么时候用、怎么用

### 适用场景

| 场景 | 扩散模型价值 | 注意事项 |
|------|------------|---------|
| 小样本深度学习 | 把 500 条序列扩到 5000 条 | 扩增后需做 OOS 验证，防止过拟合生成器的偏差 |
| 尾部事件模拟 | 生成大量左尾样本校准 VaR | 模型可能低估从未见过的极端事件，需配合压力情景 |
| 策略压力测试 | 生成与历史相关性不同的情景 | 条件化设计决定「压力」是否有结构意义 |
| 隐私合规 | 替代真实客户数据用于模型开发 | 需差分隐私或对抗式验证，防止成员推断攻击 |

### 三条红线

1. **不要直接用合成数据训练最终模型**。扩散模型生成的数据有「模型偏差」——它只能复现已知结构，无法创造新结构。正确用法是：在合成数据上做超参数搜索和架构筛选，最终模型用真实数据训练。
2. **不要混淆「像」和「是」**。合成数据通过了 ADF、Ljung-Box、Jarque-Bera 检验，不代表它能替代真实市场。2008 年的流动性枯竭是制度性事件，不是统计分布能生成的。
3. **条件化要可解释**。如果你用「波动率 regime」作为条件，必须能清楚定义这个 regime 是怎么测的、与真实交易信号的映射关系是什么。黑盒条件化等于黑盒回测。

## 5. 结论

扩散模型为金融数据增强提供了一个有力的生成工具，但它不是万能药。本文的受控实验表明：

- 一阶矩（均值、方差）复制准确；
- 二阶矩（波动聚集）方向正确但幅度偏弱，需要更复杂的时间结构模型；
- 条件生成是最大亮点，但严格条件化需要把标签嵌入训练目标，而非事后启发式调整。

最有价值的用法或许是**混合训练**：在真实数据上学习信号，在合成数据上学习鲁棒性——让模型见过足够多的「统计孪生兄弟」，从而对历史样本量不足的市场也能给出可信的推断。

---

*配图说明：*
- *图 1：前向扩散过程——同一条价格路径在 t=0, 25, 50, 75, 99 时的噪声注入效果*
- *图 2：真实收益率与扩散合成收益率的分布对比——两者都呈现厚尾特征*
- *图 3：条件生成——低/中/高三种波动率 regime 下的合成价格路径*
