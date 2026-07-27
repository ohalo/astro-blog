---
title: "Expectile 风险度量：兼具一致性与可引出性的另一条尾部路线"
description: "VaR 可回测但不次可加，ES 次可加但单独不可引出——尾部风险度量似乎总要二选一。Expectile 是已被证明的唯一例外：既是一致风险度量又可引出（Ziegel 2016）。本文从非对称平方损失出发从零实现样本 expectile，用经典违约债券反例实测 VaR 违反次可加（0+0=0 < 100）而 expectile 不违反（88.4 > 63.0），给出 τ↔α 映射表（正态 5% VaR 对应 τ≈1.24%，t(4) 要 2.39%）并诚实指出映射依赖整个分布形状这一最大软肋，附完整 Python 与四类真实陷阱（中阶）。"
publishDate: '2026-07-27'
tags:
  - 量化交易
  - Expectile
  - 风险度量
  - VaR
  - 一致性
  - 可引出性
  - 风险管理
  - Python
language: Chinese
difficulty: intermediate
---

尾部风险度量的世界里有一个著名的"不可能三角"氛围：**VaR 可引出（elicitable）所以好回测，但不次可加，合并组合可能凭空放大风险；ES 次可加是一致风险度量，但单独不可引出，回测要绕弯**（绕弯的方式见 [Acerbi-Szekely ES 检验](/blog/acerbi-szekely-es-test/)）。于是从业者默认了一个心智模型：监管口径用 ES，模型比较用 VaR，两头受气。

结论先放这：**expectile 是目前已被证明的唯一同时满足"一致性 + 可引出性"的风险度量**（Ziegel 2016, Mathematical Finance；Bellini et al. 2014）。它用非对称平方损失定义，天生对尾部损失的**幅度**敏感（VaR 只数频率不看幅度），且有自己的一阶条件可直接做回测。实测验证：经典违约债券反例中 VaR(X)+VaR(Y)=0 却 VaR(X+Y)=100，次可加性被撕碎；同一水平的 expectile 给出 44.1+44.3=88.4 > 63.0，合并组合风险不增反降，分散化逻辑恢复正常。代价也要说清楚：**expectile 没有"被击穿概率 α%"这种直白解释，τ↔α 的换算依赖整个分布形状**——正态下 5% VaR 等效于 τ≈1.24%，t(4) 下要 τ≈2.39%，差了近一倍。

## 从损失函数出发：expectile 是"非对称的均值"

分位数是 pinball 损失的最小化解，均值是平方损失的最小化解。expectile 把两者杂交：**非对称加权的平方损失**（Newey & Powell 1987，最初叫 asymmetric least squares）：

$$e_\tau = \arg\min_e \; E\Big[\,|\tau - \mathbf{1}\{X < e\}| \cdot (X - e)^2\,\Big]$$

τ=0.5 时就是均值。τ 偏离 0.5 时，一侧的平方误差被加重，解被拉向那一侧。一阶条件等价于一个漂亮的平衡式：

$$\tau \, E[(X - e_\tau)^+] = (1-\tau) \, E[(e_\tau - X)^+]$$

左边是"超出 $e_\tau$ 的期望超额"，右边是"低于 $e_\tau$ 的期望缺口"。**expectile 是让两侧期望超额按 τ:(1-τ) 配平的位置**——它同时看频率和幅度，而分位数只看频率（pinball 对误差幅度是线性的，平方损失是二次的）。

样本 expectile 用不动点迭代十行写完：

```python
import numpy as np

def expectile(x, tau, tol=1e-10, max_iter=200):
    """样本 expectile：加权均值不动点迭代"""
    e = np.mean(x)
    for _ in range(max_iter):
        w = np.where(x <= e, 1 - tau, tau)   # 低于 e 的点权重 1-tau
        e_new = np.sum(w * x) / np.sum(w)
        if abs(e_new - e) < tol:
            break
        e = e_new
    return e
```

每一步都是"按当前位置重新分配权重后的加权均值"，目标函数严格凸，收敛又快又稳——比分位数回归的线性规划省心得多。

![expectile 曲线 vs 分位数曲线](/images/expectile-risk-evar/expectile-vs-quantile.jpg)

上图对比标准正态与方差标准化 t(4) 下的两条曲线。expectile 曲线比分位数曲线**平缓**（因为平方损失把估计拉向均值），但在尾部水平上**反应更早**：τ 稍微偏离 0.5，expectile 已经开始感知尾部幅度，而分位数只有 α 真正走到尾部才动。

## 一致性实测：违约债券反例

Artzner et al.（1999）拆 VaR 的经典反例：两只**独立**债券，各有 4% 概率违约损失 100，96% 概率损失 0。看 95% 水平：

```python
rng = np.random.default_rng(42)
n = 2_000_000
X = np.where(rng.random(n) < 0.04, 100.0, 0.0)
Y = np.where(rng.random(n) < 0.04, 100.0, 0.0)

# 95% VaR（损失方向取高分位）
print(np.quantile(X, 0.95), np.quantile(Y, 0.95))      # 0.0, 0.0
print(np.quantile(X + Y, 0.95))                        # 100.0 —— 违反次可加！
# 95% expectile
print(expectile(X, 0.95) + expectile(Y, 0.95))         # 88.4
print(expectile(X + Y, 0.95))                          # 63.0 —— 分散化生效
```

单只债券违约概率 4% < 5%，95% VaR 直接瞎掉报 0；合并后"至少一只违约"的概率升到 7.8% > 5%，VaR 跳到 100。**分开看零风险、合起来风险 100，按 VaR 配资本会鼓励拆分组合来隐藏风险**。expectile 因为对 4% 概率的损失幅度敏感，单只就报出 44.1 的风险；合并后 63.0 < 88.4，次可加性完好——这不是巧合，Bellini et al.（2014）证明了 **τ≥0.5（损失方向）的 expectile 是一致风险度量**。

![损失函数与次可加性](/images/expectile-risk-evar/expectile-loss-coherence.jpg)

## τ↔α 映射：expectile 的最大软肋

用 expectile 替代 VaR 的第一个现实问题：**τ 不是突破概率**。τ=5% 的 expectile 被击穿的概率不是 5%，是多少取决于分布。想让 expectile 匹配"5% 突破率"的直觉，要找 τ 使 $e_\tau = q_{5\%}$——这个映射对分布形状敏感：

| 目标 α | 正态 | t(10) | t(6) | t(4) | t(3) |
|---|---|---|---|---|---|
| 5% | τ=1.34%* | 1.59% | 1.90% | 2.39% | 3.03% |
| 2.5% | 0.53% | 0.65% | 0.81% | 1.07% | 1.42% |
| 1% | 0.17% | 0.22% | 0.28% | 0.39% | 0.54% |

*正态行取 df=30 近似；精确正态值为 τ=1.24%（α=5%）、0.145%（α=1%）。

![tau-alpha 映射](/images/expectile-risk-evar/expectile-tau-alpha-map.jpg)

规律很清楚：**尾部越厚，同一 α 对应的 τ 越大**。这意味着如果你按正态校准 τ=1.24% 去匹配"5% 突破率"，而真实分布是 t(4)，你的 expectile-VaR 会系统性偏深（过度保守）。映射本身成了一个需要估计的对象——这是 expectile 阵营必须承认的税。

## GARCH 世界滚动回测

在 GARCH(1,1)-t(6) 模拟序列上（波动持续性 0.97），用 500 天滚动窗口分别计算 5% 分位数 VaR 和校准后的 expectile-VaR（τ 在首个窗口上校准到与 5% 分位数重合，得 τ=2.54%——顺带验证了上表：数据肥尾+波动混合，τ 远高于正态的 1.24%）：

- 750 个评估日，期望突破 38 次：分位数 VaR 突破 **53 次**，expectile-VaR 突破 **49 次**——都超标，因为**滚动无条件方法对波动聚集都反应迟钝**（这是方法族的通病，不是 expectile 的锅，条件化的解法见 [GARCH 滤波历史模拟](/blog/garch-filtered-historical-simulation/)）
- 按真实波动率分半：高波动期突破率分位数 10.9% vs expectile 9.6%，低波动期 3.2% vs 3.5%。expectile 的高低波动失衡略轻，因为**窗口里进来一笔大损失时，expectile 立即按幅度加深，而分位数要等足够多的点挤过 5% 门槛才动**

![GARCH 世界滚动回测](/images/expectile-risk-evar/expectile-var-backtest.jpg)

这个实验的诚实结论：expectile 在无条件滚动框架下只带来边际改善，**它的真正价值在结构层面**——一致性让它能做组合聚合与资本分配，可引出性让它能像 VaR 一样直接进 [Kupiec](/blog/kupiec-pof-test/)/[DQ](/blog/dynamic-quantile-test-dq/) 式的回测流水线（对 expectile 用推广的一阶条件检验），而 ES 两者只占一头。

## 陷阱清单

1. **τ 直接当突破概率用**。τ=5% 的 expectile 突破率通常在 12%~15%（视分布而定）。要匹配目标突破率必须做 τ↔α 校准，且校准结果依赖分布假设——换个市场状态映射就漂移。
2. **和 EVaR（entropic VaR）搞混**。文献里 "EVaR" 有两个完全不同的所有者：expectile-VaR（Kuan et al. 2009）和 entropic VaR（Ahmadi-Javid 2012，见[这篇](/blog/entropic-var-evar/)）。读论文先看定义再看结论。
3. **对离群点比分位数敏感**。平方损失是双刃剑：对幅度敏感意味着一笔极端脏数据（错价、拆股未复权）会把 expectile 拖走，而分位数几乎不动。数据清洗的标准要按均值类估计量的要求来。
4. **expectile 不是逐段可加的**。它不满足 comonotonic additivity（共单调可加性），两个完全同向的头寸的 expectile 之和不等于组合 expectile。做风险分解时不能像 ES 那样用 Euler 分配的直觉硬套，要用 expectile 专用的边际贡献公式（Emmer et al. 2015）。

expectile 至今没有成为监管标准，最大障碍就是解释性：向风控委员会解释"τ=2.5% 的非对称平方损失平衡点"远比"1% 概率下的最大损失"费劲。但在内部模型里——尤其是需要同时做组合聚合和严格回测的场景——它是那个理论上没有明显短板的选项。

## 参考文献

- Newey, W. K., & Powell, J. L. (1987). Asymmetric Least Squares Estimation and Testing. *Econometrica*, 55(4), 819-847.
- Ziegel, J. F. (2016). Coherence and Elicitability. *Mathematical Finance*, 26(4), 901-918.
- Bellini, F., Klar, B., Müller, A., & Rosazza Gianin, E. (2014). Generalized Quantiles as Risk Measures. *Insurance: Mathematics and Economics*, 54, 41-48.
- Kuan, C.-M., Yeh, J.-H., & Hsu, Y.-C. (2009). Assessing Value at Risk with CARE, the Conditional Autoregressive Expectile Models. *Journal of Econometrics*, 150(2), 261-270.
- Emmer, S., Kratz, M., & Tasche, D. (2015). What Is the Best Risk Measure in Practice? *Journal of Risk*, 18(2), 31-60.
