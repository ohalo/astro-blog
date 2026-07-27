---
title: "NIG 正态逆高斯分布：用半重尾分布刻画收益的尖峰厚尾"
publishDate: '2026-07-27'
description: "NIG 正态逆高斯分布：用半重尾分布刻画收益的尖峰厚尾 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

金融收益率几乎从不服从正态分布。任何做过实证的人都知道，日收益率的直方图总是比正态曲线更尖、尾巴更肥，偶尔还带点偏度。用正态分布去算 VaR、定权益期权、或做蒙特卡洛模拟，本质上都是在系统性地低估极端事件的概率。问题在于：怎样找一个既能抓住尖峰厚尾、又不像稳定分布那样连方差都不存在的分布？正态逆高斯分布（Normal Inverse Gaussian，NIG）就是一个非常实用的答案。

## 为什么正态分布不够用

先看一组直觉。标准正态分布的峰度是 3，但股票日收益的超额峰度经常在 3 到 10 甚至更高。这意味着实际分布中间更集中、两端更极端。2008 年 10 月标普 500 单日跌幅超过 9%，在正态假设下这是十几个标准差的事件，理论概率小到"宇宙年龄里都不该发生一次"，可它真的发生了。

处理厚尾有几条路：t 分布、稳定分布（Stable/Lévy）、广义双曲分布（GH）家族。t 分布简单但只有对称版本，且尾部是幂律衰减，有时太肥；稳定分布理论优雅但方差可能无穷，参数估计困难，闭式密度也不存在。NIG 恰好卡在一个甜蜜点上：它是广义双曲分布的一个特例，尾部是**半重尾**（semi-heavy），介于正态的指数衰减和幂律衰减之间，方差永远有限，还能独立刻画偏度和峰度。

![收益分布的尖峰厚尾](/images/nig-distribution-returns/return-distribution.jpg)

## NIG 分布的四个参数

NIG 分布由四个参数决定，每个都有清晰的金融含义：

- **α（alpha）**：尾部重量/陡峭度。α 越大，尾巴越轻、分布越接近正态；α 越小，尾巴越肥。
- **β（beta）**：偏度参数，满足 |β| < α。β > 0 右偏，β < 0 左偏，β = 0 对称。股票收益通常 β < 0（崩盘比暴涨更极端）。
- **δ（delta）**：尺度参数，类似标准差的作用，δ > 0。
- **μ（mu）**：位置参数，控制分布中心。

它的构造很漂亮：NIG 是一个正态分布，但其方差本身是一个服从**逆高斯分布**的随机变量。这就是"正态逆高斯"名字的由来——它是一个方差混合（variance-mean mixture）模型。这个构造使得 NIG 天然适合刻画"平静期波动小、危机期波动骤增"的市场状态切换。

均值和方差有闭式表达：

- 均值：`μ + δβ / √(α² − β²)`
- 方差：`δα² / (α² − β²)^{3/2}`

一个关键性质是 NIG 对卷积封闭：若把时间尺度从日收益聚合到周收益，只要 α、β 不变，δ 和 μ 按天数线性相加，结果仍是 NIG 分布。这对多周期风险聚合极为友好。

## 用 Python 拟合真实收益

`scipy.stats` 里直接提供了 `norminvgauss`，但它的参数化和上面的经典参数化略有不同——scipy 用形状参数 `a`（对应 δα）和 `b`（对应 δβ），再加 `loc`、`scale`。下面用最大似然拟合一段真实收益：

```python
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

# 构造一段带尖峰厚尾的收益序列（实际用中替换为真实数据）
np.random.seed(42)
# 混合两个正态模拟"平静+危机"两种波动状态
calm = np.random.normal(0.0005, 0.008, 4500)
crisis = np.random.normal(-0.002, 0.035, 500)
returns = np.concatenate([calm, crisis])
np.random.shuffle(returns)

# 拟合 NIG（scipy 参数化：a=δα, b=δβ, loc=μ, scale=δ）
a, b, loc, scale = stats.norminvgauss.fit(returns)
print(f"a (δα)   = {a:.4f}")
print(f"b (δβ)   = {b:.4f}")
print(f"loc (μ)  = {loc:.6f}")
print(f"scale(δ) = {scale:.6f}")

# 还原经典参数
alpha = a / scale
beta = b / scale
print(f"\nα = {alpha:.2f}, β = {beta:.2f}  (β<0 => 左偏)")

# 对比正态拟合的对数似然
nig_ll = np.sum(stats.norminvgauss.logpdf(returns, a, b, loc, scale))
mu_n, sd_n = returns.mean(), returns.std()
norm_ll = np.sum(stats.norm.logpdf(returns, mu_n, sd_n))
print(f"\nNIG  log-likelihood: {nig_ll:.1f}")
print(f"Norm log-likelihood: {norm_ll:.1f}")
print(f"NIG 胜出 {nig_ll - norm_ll:.1f} 个对数似然单位")
```

对数似然的差距通常非常显著。我们可以进一步用 AIC/BIC 做正式模型选择：NIG 多了两个参数（4 个 vs 正态的 2 个），但只要似然提升足够，AIC 依然会青睐 NIG。

## 可视化：密度对比与 QQ 图

```python
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左：直方图 + NIG/正态密度叠加
x = np.linspace(returns.min(), returns.max(), 500)
axes[0].hist(returns, bins=120, density=True, alpha=0.4,
             color='gray', label='实际收益')
axes[0].plot(x, stats.norminvgauss.pdf(x, a, b, loc, scale),
             'r-', lw=2, label='NIG 拟合')
axes[0].plot(x, stats.norm.pdf(x, mu_n, sd_n),
             'b--', lw=2, label='正态拟合')
axes[0].set_yscale('log')          # 对数纵轴放大尾部差异
axes[0].set_title('密度对比（对数纵轴）')
axes[0].legend()

# 右：NIG QQ 图
stats.probplot(returns, dist=stats.norminvgauss,
               sparams=(a, b, loc, scale), plot=axes[1])
axes[1].set_title('NIG QQ 图')
plt.tight_layout()
plt.savefig('nig_fit.png', dpi=120)
```

在对数纵轴上看密度对比最直观：正态曲线在尾部会"塌下去"远低于实际直方图，而 NIG 曲线能一路贴合到极端区域。QQ 图里，正态假设下的点会在两端明显偏离参考线（尾部翘起），NIG 则基本落在直线上。

![NIG 密度与尾部拟合](/images/nig-distribution-returns/nig-density-tails.jpg)

## 实战应用：更靠谱的 VaR 与 ES

拟合出 NIG 之后，最直接的用途是重算风险指标。正态 VaR 会低估尾部风险，NIG VaR 则能给出更保守也更真实的估计：

```python
alpha_level = 0.01  # 99% 置信

# 正态 VaR / ES
var_norm = stats.norm.ppf(alpha_level, mu_n, sd_n)
es_norm = mu_n - sd_n * stats.norm.pdf(stats.norm.ppf(alpha_level)) / alpha_level

# NIG VaR（分位数直接用 ppf）
var_nig = stats.norminvgauss.ppf(alpha_level, a, b, loc, scale)

# NIG ES：分位数以下的条件期望，用数值积分
from scipy import integrate
def nig_es(level):
    q = stats.norminvgauss.ppf(level, a, b, loc, scale)
    integrand = lambda t: t * stats.norminvgauss.pdf(t, a, b, loc, scale)
    tail, _ = integrate.quad(integrand, -np.inf, q)
    return tail / level

es_nig = nig_es(alpha_level)

print(f"99% VaR  —  正态: {var_norm:.4f}   NIG: {var_nig:.4f}")
print(f"99% ES   —  正态: {es_norm:.4f}   NIG: {es_nig:.4f}")
print(f"NIG VaR 比正态严格 {(var_nig/var_norm - 1)*100:.1f}%")
```

对典型股票收益，NIG 的 99% VaR 往往比正态深 15%–40%，这个差距在杠杆头寸或期权组合上会被进一步放大。如果你的风控用正态，本质上是每天都在给自己的尾部风险打折。

## 使用 NIG 时要注意的坑

**参数估计不稳定。** NIG 的对数似然面在 α 很大时会变得很平（此时接近正态），MLE 优化可能收敛到边界。建议给多个初值、检查收敛，样本量最好在 500 以上。

**尾部并非幂律。** NIG 是半重尾，尾部指数衰减（带一个多项式修正）。如果你的资产真的是幂律尾（比如某些加密货币或流动性极差的品种），NIG 可能仍然低估极端尾部，这时要考虑广义帕累托或稳定分布。

**独立同分布假设。** 上面的拟合把收益当成 i.i.d.，但真实收益有波动率聚集。更严谨的做法是先用 GARCH 过滤掉条件异方差，再对标准化残差拟合 NIG——这套组合（GARCH-NIG）在期权定价和风险管理里非常常见。

**别忘了做样本外检验。** 分布拟合极易过拟合，务必用样本外的 Kupiec 检验或 Christoffersen 检验来验证 VaR 突破率是否符合理论水平。

## 小结

NIG 分布是量化工具箱里被低估的一件利器。它用四个有金融解释力的参数，同时刻画了收益的偏度和厚尾，方差有限、卷积封闭、密度可算，还能无缝接入 VaR/ES 计算和期权定价。当你发现正态假设一次次在极端行情面前失灵时，NIG 往往是那个既不过于激进、又足够诚实的替代方案。下次做风险模型时，不妨先花十分钟拟合一个 NIG，对比一下尾部——你可能会重新审视自己对"小概率事件"的定义。
