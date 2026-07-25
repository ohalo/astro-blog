---
title: "Newey-West 稳健标准误：给自相关的收益率算一个诚实的 t 值"
publishDate: '2026-07-26'
description: "Newey-West HAC 标准误 - 3000次蒙特卡洛复现：策略收益真实均值为0但注入AR(1)自相关时，iid标准误把假阳性率从名义5%抬到17.3%，Newey-West修正回7.1%；Bartlett核带宽选择、多策略t值对比、夏普率显著性去膨胀 - halo的技术博客"
tags:
  - 量化交易
  - 统计推断
  - Newey-West
  - 自相关
  - 假设检验
  - HAC
  - Python
language: Chinese
difficulty: intermediate
---

你回测出一个策略，20 年月度超额收益均值为正，算出 t 值 2.4，看起来稳稳越过了 1.96 的显著线，于是准备上线。但如果这条收益序列有**正自相关**——动量策略、carry 策略、任何持仓周期跨越多个 bar 的策略几乎都有——那么这个 2.4 是**虚高的**。真实的 t 值可能只有 1.5，根本不显著。你以为发现了 alpha，其实只是被一个错误的标准误骗了。

Newey-West 标准误（更准确地说是 HAC，Heteroskedasticity and Autocorrelation Consistent 标准误）就是用来堵这个漏洞的。它不改你的收益均值，只修正标准误——把序列自相关和异方差带来的额外不确定性诚实地计入分母。这篇文章用 3000 次蒙特卡洛把"iid 标准误在自相关下会撒谎"这件事量化出来，讲清带宽 L 怎么选，以及它在夏普率检验里的日常用法。

结论先放这：**当策略月度收益真实均值为 0、但注入 AR(1) 系数 0.35 的正自相关时，用普通 iid 标准误做 t 检验，名义 5% 的显著性水平实际假阳性率高达 17.3%——每 6 个"显著"的策略里有 1 个多是纯噪声冒充的。换成 Newey-West 标准误（带宽 L=5），假阳性率回落到 7.1%，接近名义水平。** 正自相关越强，iid 越乐观，你被骗得越惨。

## 问题的根源：iid 标准误假设了不存在的独立性

估计一个均值 $\bar{r}$ 的标准误，教科书公式是 $\text{SE} = s / \sqrt{n}$，其中 $s$ 是样本标准差。这个公式背后藏着一个致命假设：**观测之间相互独立**。当观测独立时，$n$ 个样本提供了 $n$ 份独立信息，标准误按 $\sqrt{n}$ 收缩。

但收益率序列往往不独立。如果 $r_t$ 和 $r_{t-1}$ 正相关，那么相邻两个观测携带的信息是**重叠的**——它们没有提供两份独立信息，更像是 1.3 份。有效样本量比 $n$ 小，真实标准误比 $s/\sqrt{n}$ 大。iid 公式用了虚高的有效样本量，算出的标准误偏小，t 值偏大，于是把噪声误判成信号。

先看一条真实均值为 0、但注入了 AR(1) 自相关的策略收益序列，以及它的自相关函数：

![左：真实均值为0但有自相关的月度收益序列；右：残差ACF在低阶显著为正](/images/newey-west-robust-se/autocorr_series_acf.png)

右图的 ACF 很清楚：lag 1、lag 2 的自相关系数明显超出 95% 置信带（红色虚线）。这条序列的均值确实是 0，但相邻观测正相关，正是 iid 标准误会翻车的典型场景。

## Newey-West 怎么修

HAC 标准误的核心思想：估计均值的**长期方差**（long-run variance）时，不只用方差 $\gamma_0$，还要把各阶自协方差 $\gamma_k$ 加进来：

$$
\text{LRV} = \gamma_0 + 2\sum_{k=1}^{L} w_k \, \gamma_k
$$

其中 $w_k$ 是 **Bartlett 核**权重 $w_k = 1 - k/(L+1)$，随滞后阶数线性衰减；$L$ 是带宽（bandwidth），决定纳入多少阶自协方差。这个加权衰减的设计不是随意的——它保证了估计出的长期方差恒为非负（半正定），否则你可能算出一个负的方差。

用纯 numpy 实现估计均值的 Newey-West 标准误：

```python
import numpy as np

def nw_se(x, L):
    """Newey-West HAC 标准误：估计样本均值的标准误。"""
    x = np.asarray(x, float)
    n = len(x)
    u = x - x.mean()
    gamma0 = np.sum(u * u) / n           # 0阶：方差
    s = gamma0
    for k in range(1, L + 1):
        w = 1.0 - k / (L + 1.0)          # Bartlett 核权重
        gk = np.sum(u[k:] * u[:n - k]) / n   # k阶自协方差
        s += 2 * w * gk
    return np.sqrt(s / n)

def iid_se(x):
    return np.std(x, ddof=1) / np.sqrt(len(x))
```

当 $L=0$ 时，Newey-West 退化成 iid 标准误（只有 $\gamma_0$ 项）。每增加一阶，就把那一阶的自相关贡献按 Bartlett 权重计入。正自相关会让 $\gamma_k > 0$，长期方差变大，标准误变大——这正是我们要的诚实。

## 3000 次模拟：假阳性率的直接证据

把上面那条"真实均值 0 + AR(1)"的数据生成过程重复 3000 次，每次分别用 iid 和 Newey-West 算 t 值，统计 |t| > 1.96 的比例（即错误拒绝"均值为 0"原假设的比例）：

```python
n_sim = 3000
T = 240
phi = 0.35
L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))   # 经验带宽 ≈ 5

fp_iid, fp_nw = 0, 0
for _ in range(n_sim):
    e = rng.normal(0, 0.045, T)
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = phi * x[t - 1] + e[t]     # 真实均值=0
    m = x.mean()
    fp_iid += abs(m / iid_se(x)) > 1.96
    fp_nw  += abs(m / nw_se(x, L)) > 1.96
```

![3000次模拟t值分布：iid假阳性率17.3%，Newey-West回落到7.1%](/images/newey-west-robust-se/tstat_distribution.png)

红色（iid）的 t 值分布明显比标准正态更胖——尾巴超出 ±1.96 的部分远大于名义的 5%，实测 **17.3%**。蓝色（Newey-West）的分布收窄回接近正态，超出比例 **7.1%**，仍略高于 5%（HAC 在有限样本下有已知的轻微 size 扭曲，但已经是数量级的改善）。

这张图是整篇文章的核心证据：**在有自相关的世界里用 iid 标准误，你有约 1/6 的概率把一个纯噪声策略判成"统计显著"。** 量化研究里同时测几十上百个策略/因子，这个膨胀的假阳性率意味着你的"显著发现"里混进了大量假货。

## 带宽 L 怎么选：不是越大越好

带宽 $L$ 是唯一的调参。$L$ 太小，漏掉高阶自相关，标准误仍偏低；$L$ 太大，纳入太多噪声自协方差估计，标准误方差变大、不稳定。看 $L$ 对标准误的影响：

![Newey-West标准误随带宽L上升而增大，L=0退化为iid，经验带宽L*=5](/images/newey-west-robust-se/bandwidth_effect.png)

从 $L=0$（iid）开始，随着 $L$ 增大，标准误快速上升——因为正自相关被逐步计入——然后在经验带宽附近趋于平稳。实践中最常用的自动选择是 Newey-West (1994) 的经验规则：

```python
L = int(np.floor(4 * (T / 100.0) ** (2.0 / 9.0)))
```

对 $T=240$（20 年月度），这给出 $L=5$。另一个常见做法：如果你的策略持仓周期是 $h$ 个 bar（比如季度调仓的月度数据 $h=3$），至少取 $L \geq h-1$，因为重叠持仓天然制造 $h-1$ 阶的移动平均自相关。statsmodels 里 `cov_type='HAC'` 配 `maxlags` 参数就是干这个的，但理解手算逻辑能让你判断它给的数字合不合理。

## 日常用法：给夏普率一个诚实的 t 值

量化里最常见的场景是检验策略的夏普率是否显著异于 0。夏普率的 t 值近似等于均值收益的 t 值（$t_{\text{Sharpe}} \approx \text{Sharpe} \times \sqrt{n}$），所以均值检验的自相关问题会原样传导到夏普率上。看五个不同自相关强度的策略：

![五个策略的均值显著性：正自相关越强，iid t值越高估](/images/newey-west-robust-se/sharpe_significance.png)

动量策略（AR 系数 0.40，强正自相关）的 iid t 值远高于 Newey-West t 值——差距最大，因为它自相关最强；反转策略（AR 系数 -0.15，负自相关）反而是 Newey-West t 值略高于 iid（负自相关会让 iid 高估标准误、低估 t）。这揭示了一个常被忽略的事实：**Newey-West 不总是让 t 值变小**——它让 t 值变对。正自相关下修正向下，负自相关下修正向上。

配合 Lo (2002) 的夏普率标准误修正、以及 Bailey-López de Prado 的去膨胀夏普率（Deflated Sharpe Ratio），HAC 标准误是"策略显著性去水分"工具箱里最基础的一件。它不解决多重检验问题（那需要 Bonferroni/FDR），但解决了单个策略内部的自相关污染。

## 四个实盘会踩的坑

**一、重叠样本是自相关的头号来源，且常被忽视。** 如果你用月度数据但计算的是"过去 12 个月滚动收益"，相邻观测共享 11 个月的数据，人为制造了 11 阶的强自相关。这种情况下 iid t 值可以虚高好几倍，$L$ 至少要取到重叠窗口长度。很多"年化收益显著为正"的结论一换 HAC 就崩了。

**二、带宽选择本身有主观性，别用它来 p-hacking。** 既然 $L$ 越大标准误通常越大，反过来你可以通过调小 $L$ 让结果"更显著"。诚实的做法是预先用经验规则确定 $L$（比如 Newey-West 1994 或 Andrews 1991 的自动带宽），并在报告里写清楚，而不是试到 t 值刚好过 1.96 为止。

**三、有限样本下 HAC 仍有 size 扭曲。** 模拟里 Newey-West 的假阳性率是 7.1%，不是完美的 5%。短样本（比如只有 60 个月）加强自相关时，HAC 的过度拒绝会更明显。此时可以考虑 HAC 的小样本修正版本，或直接用 block bootstrap 做推断——后者对自相关结构的假设更少。

**四、HAC 只修正推断，不修正点估计。** Newey-West 让你的标准误诚实，但如果你的均值收益估计本身有偏（比如幸存者偏差、前视偏差污染了收益序列），HAC 一点忙都帮不上。它是推断层的工具，不是数据质量的补丁。别指望它救回一个数据有问题的回测。

## 总结

Newey-West / HAC 标准误解决的是一个具体而普遍的问题：收益率序列的自相关会让 iid 标准误撒谎。3000 次模拟给出的数字很直白——真实均值为 0 的策略，在 AR(1)=0.35 的自相关下，iid 标准误把假阳性率从名义 5% 抬到 17.3%，Newey-West 用 Bartlett 核加权计入各阶自协方差，把它拉回 7.1%。

它的用法可以一句话概括：**任何时候你要给一个时间序列的均值（或夏普率）算 t 值，先问一句"这条序列有自相关吗"——只要答案不是斩钉截铁的"没有"，就用 HAC 标准误，别用 iid。** 在动量、carry、重叠样本、多步预测这些量化的家常场景里，答案几乎永远是"有"。

## 参考文献

1. Newey, W. K., & West, K. D. (1987). A Simple, Positive Semi-Definite, Heteroskedasticity and Autocorrelation Consistent Covariance Matrix. *Econometrica*, 55(3).
2. Newey, W. K., & West, K. D. (1994). Automatic Lag Selection in Covariance Matrix Estimation. *Review of Economic Studies*, 61(4).
3. Andrews, D. W. K. (1991). Heteroskedasticity and Autocorrelation Consistent Covariance Matrix Estimation. *Econometrica*, 59(3).
4. Lo, A. W. (2002). The Statistics of Sharpe Ratios. *Financial Analysts Journal*, 58(4).
