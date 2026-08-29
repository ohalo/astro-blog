---
title: "已实现核与微观结构噪声：用子采样估计无偏积分波动率"
description: "秒级 tick 价格里混着买卖价差、错单、报价粒度带来的微观结构噪声——直接平方求和会把积分波动率(IV)估高 161%。本文用 numpy 从零实现 Parzen 已实现核(Realized Kernel)，靠 lag1 自协方差把 2Nη² 噪声项精确抵消，在 h=1 全采样下就还原真实 IV=0.0016；并给出 γ_1=-η² 的噪声方差估计。附完整 numpy 实现与四张真实计算图。"
publishDate: '2026-08-29'
tags:
  - 量化交易
  - 已实现核
  - 微观结构噪声
  - 高频数据
  - 积分波动率
  - 子采样
  - Python
language: Chinese
difficulty: intermediate
---

积分波动率（Integrated Variance, IV）是波动率建模的「真值」：\\(IV = \int_0^T \sigma_t^2 dt\\)。如果你有秒级 tick 数据，最朴素的想法是「把 tick 收益平方加起来」——这就是已实现方差（Realized Variance）。问题是：**tick 价格里混着微观结构噪声**。买卖价差、报价粒度、错单、收盘竞价，都会让观测到的价格 \\(y_i\\) 偏离真实有效价格 \\(p_i\\)：

$$y_i = p_i + \varepsilon_i,\qquad \varepsilon_i\sim \text{iid}(0,\eta^2)$$

于是直接平方求和会把噪声方差算进去，估出来的 IV 系统性偏高。本文用 numpy 从零实现 **已实现核（Realized Kernel, RK）**，它不靠降采样「回避」噪声，而是用自协方差结构把噪声项精确抵消，在全采样下就给出无偏估计。

## 1. 噪声怎么污染「已实现方差」

设有效 tick 收益 \\(r_i^{eff}\sim\mathcal N(0,\sigma_\Delta^2)\\)（iid，连续路径），观测收益是

$$r_i = r_i^{eff} + \varepsilon_i - \varepsilon_{i-1}$$

它的方差被抬高了两项噪声：\\(\mathbb E[r_i^2] = \sigma_\Delta^2 + 2\eta^2\\)。对全部 N 个 tick 平方求和：

$$RV = \sum_{i=1}^N r_i^2 \;\approx\; \underbrace{N\sigma_\Delta^2}_{IV_{true}} + \underbrace{2N\eta^2}_{\text{噪声偏差}}$$

偏差 \\(2N\eta^2\\) 与 tick 数成正比——**采样越密，偏差越大**。下面的图把一条含噪路径和纯净有效路径叠在一起，整段看噪声被波动盖住了，但放大到 100 个 tick 就能看到典型的「锯齿」。

![有效价格 vs 含噪观测价格](/images/realized-kernel-microstructure-noise/price_path.png)

子采样（每隔 h 个 tick 聚合一次再平方）能减小偏差，但代价是丢掉连续路径的高频信息。我们实测了不同采样间隔 h 下的 RV：

| 采样间隔 h | RV(h) | 相对真实 IV |
|---|---|---|
| 1 (全采样) | 0.0042 | +161% |
| 20 | 0.0024 | +47% |
| 80 | 0.0013 | +-17% |

采样越稀，RV 越接近真实 IV=0.0016（红虚线），但这是「用信息换无偏」的妥协。

![RV(h) 随采样频率变化](/images/realized-kernel-microstructure-noise/rv_vs_frequency.png)

## 2. 已实现核：用 lag1 自协方差抵消噪声

关键观察在**自协方差**上。因为噪声只在相邻 tick 重叠一次：

$$\gamma_1 = \mathrm{Cov}(r_i, r_{i+1}) = \mathbb E[(\varepsilon_i-\varepsilon_{i-1})(\varepsilon_{i+1}-\varepsilon_i)] = -\eta^2$$

而 \\(\gamma_h\approx 0\;(h\ge 2)\\)（有效收益 iid、噪声只重叠一阶）。所以 **lag1 自协方差的负尖就是噪声方差的指纹**：

$$\hat{\eta}^2 = -\gamma_1, \qquad \hat{\sigma}_\Delta^2 = \gamma_0 + 2\gamma_1$$

Barndorff-Nielsen-Hansen-Lunde-Shephard（2008）的已实现核把这件事做成加权自协方差和：

$$RV_K = \sum_{h=-H}^{H} k\!\left(\frac{h}{H}\right)\gamma_h = \gamma_0\cdot N + 2\sum_{h=1}^{H} k\!\left(\frac{h}{H}\right)\gamma_h\cdot N$$

核权重 \\(k(\cdot)\\)（本文用 Parzen）在 \\(h=0\\) 处为 1、随滞后衰减、且保留 lag1 的权重≈1。于是

$$RV_K \approx N(\sigma_\Delta^2+2\eta^2) + 2\cdot 1\cdot N(-\eta^2) = N\sigma_\Delta^2 = IV_{true}$$

噪声项被自己 lag1 的贡献抵消掉。下面是本例算出的自协方差（lag1 负尖）与 Parzen 权重。

![自协方差与 Parzen 核权重](/images/realized-kernel-microstructure-noise/autocov_kernel.png)

## 3. 完整 numpy 实现

```python
import numpy as np

def parzen_kernel(x):
    x = np.abs(np.asarray(x, float))
    return np.where(x <= 0.5, 1 - 6*x**2 + 6*x**3,
             np.where(x <= 1.0, 2*(1 - x)**3, 0.0))

def realized_kernel(ret, H=20):
    N = len(ret)
    ac = np.array([np.mean(ret[:N-h] * ret[h:]) for h in range(H + 1)])
    w  = parzen_kernel(np.arange(H + 1) / H)
    return N * (ac[0] + 2 * np.sum(w[1:] * ac[1:]))   # 抵消 2Nη² 噪声偏差

def subsampled_rv(ret, h):
    M = len(ret) // h
    sub = ret[: M*h].reshape(M, h).sum(axis=1)        # h 期聚合收益
    return np.sum(sub**2)

# 噪声方差估计：γ_1 = -η²
def noise_variance(ret):
    N = len(ret)
    g0 = np.mean(ret**2)
    g1 = np.mean(ret[:N-1] * ret[1:])
    return -g1, g0 + 2*g1   # η²估计, 有效单tick方差估计

# 用法
r  = np.diff(np.log(price_tick))        # tick 收益
rk = realized_kernel(r)                  # 无偏 IV 估计
eta2, sig2 = noise_variance(r)          # 微观结构噪声方差
```

本例实测：直接用全采样 RV 得到 0.0042，偏高达 +161%；Parzen 已实现核得到 0.0015，几乎贴住真实 IV=0.0016（紫点线）；由 γ_1 估出的噪声方差 \\(\hat\eta^2=1.68e-07\\)，与设定 \\(\eta^2=(0.0004)^2=1.60e-07\\) 吻合，据此还原的有效单 tick 方差 \\(\hat\sigma_\Delta^2=1.85e-07\\)。

## 4. Monte Carlo：谁最稳？

单次实现有运气成分。我们对 300 条独立合成路径各估一次 IV，比较三种方法相对真值 0.0016 的误差：

- **RV(全采样)**：均值偏差 0.0026，RMSE 0.0026（噪声偏差主导）
- **RV(80 聚合)**：均值偏差 0.0001，RMSE 0.0002（降采样换来无偏，但方差略增）
- **已实现核**：均值偏差 0.0000，RMSE 0.0001（全采样 + 无偏，综合最优）

![三种 IV 估计的 Monte Carlo 误差](/images/realized-kernel-microstructure-noise/mc_comparison.png)

已实现核在「保留全部高频信息」的同时把偏差压到最低，这就是它在高频波动率估计里成为标配的原因。

## 5. 实战提醒（诚实版）

- **H 的选择**：核窗口 H 一般取 \\(O(N^{1/3\sim 1/2})\\)。H 太小消不掉噪声、太大引入高 lag 的估计方差。本例 H=20。
- **噪声不只有「加性 iid」**：真实市场的噪声是异方差的（开盘/收盘更吵、流动性差时更吵），这时 η² 会随时间变化，`-γ_1` 只能给一个平均值；更稳妥用 **Two-Scale RV / 多尺度** 同时估 IV 和 η²。
- **跳跃**：若价格有跳（overnight gap、重大新闻），RV 会同时吸收跳方差，需用 **bipower variation** 把连续部分和跳分开。
- **本文是合成数据**：用了恒定 σ、iid 有效收益、同方差噪声，所以 RK 几乎完美还原。真实 tick 数据（限价单簿、买卖盘不对称）会让数字没这么漂亮，但方法框架不变。

## 小结

微观结构噪声让「直接平方求和」的已实现方差系统性偏高，偏差 ∝ tick 数。已实现核不靠降采样回避，而是用 **lag1 自协方差 γ₁=−η² 把 2Nη² 噪声项精确抵消**，在全采样下就给出无偏积分波动率；顺手还能由 γ₁ 估出噪声方差。完整 numpy 实现不到 20 行，四张图都是真实算出来的。
