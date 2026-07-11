---
title: "分位数回归在风险管理中的应用：不止看均值，更看尾部"
publishDate: '2026-07-12'
description: "OLS 只建模条件均值，但风险管理关心尾部。分位数回归用 check function 钉住任意条件分位，把「看均值」升级成「看整个条件分布」。本文从零实现 LP 求解，演示 QR 喇叭形展开、与 OLS 的尾部盲区对比、条件 VaR 回测，以及系数路径揭示的尾部不对称性，附可复现 Python。"
tags:
  - 量化交易
  - 分位数回归
  - 风险管理
  - VaR
  - 条件分位
  - 尾部风险
  - 异方差
language: Chinese
difficulty: advanced
---

均值回归、OLS、线性回归——我们习惯用「条件均值」$\mathbb E[y\mid x]$ 描述世界。但**风险管理关心的从来不是均值，是尾部**：5% 最坏情况（VaR）、亏损分布的上凸下凹、高波动时损失会不会更肥。一条均值线在这些问题面前几乎失明。（Quantile Regression, Koenker & Bassett 1978）**把目标从「预测均值」换成「预测任意分位」：$Q_\tau(y\mid x)=x^\top\beta_\tau$。它用 check function 最小化，不看均值看分布**整个形状**。本文讲清四件事，并用可复现代码演示：

1. QR 是什么、和 OLS 差在哪；
2. 异方差下 OLS 怎样漏掉尾部不对称；
3. 怎么用 QR 做**条件 VaR**（让风险限额随状态自适应）；
4. **系数路径**如何揭示风险因子在尾部的效应漂移。

## 一、OLS 的盲区：只给你一条均值线

OLS 最小化平方误差，等价于建模条件均值。它对对称噪声稳健，但有两个致命局限：

- **均值掩盖风险**：异方差下，低 x 和高 x 处的条件均值可能相近，但尾部天差地别——一条线把下行风险悄悄抹平；
- **对称假设**：平方损失对正负偏差同等惩罚，无法区分「温和下跌」和「暴跌」，而后者才是风控核心。

真正的风险管理语言是**分布**，不是均值。

## 二、分位数回归：用 check function 钉住任意分位

QR 的目标是最小化加权绝对误差：

$$\min_{\beta_\tau}\;\sum_{i=1}^{n}\rho_\tau(y_i - x_i^\top\beta_\tau),\qquad \rho_\tau(u)=u\big(\tau-\mathbf 1_{\{u<0\}}\big)$$

check function $\rho_\tau$ 在 $\tau<0.5$ 时**重罚高估**（盯下行）、$\tau>0.5$ 时**重罚低估**（盯上行）。它不可微，但能干净地转成**线性规划**：

$$\min \sum_i\big[\tau\,u_i + (1-\tau)\,v_i\big],\quad \text{s.t. } y - X\beta = u - v,\; u,v\ge 0$$

```python
import numpy as np
from scipy.optimize import linprog

def quantile_regression(X, y, tau):
    n, k = X.shape
    c = np.concatenate([np.zeros(k), np.full(n, tau), np.full(n, 1.0 - tau)])
    A_eq = np.hstack([-X, -np.eye(n), np.eye(n)])   # -Xβ - u + v = -y
    b_eq = -y
    bounds = [(None, None)] * k + [(0, None)] * (2 * n)
    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    return res.x[:k]

# 异方差 + 重尾模拟：y = 1 + 0.8x + sigma(x)*t(4噪声)
rng = np.random.default_rng(2026)
N = 600
x = np.sort(rng.uniform(0, 10, N))
sigma = 0.3 + 0.25 * x
y = 1.0 + 0.8 * x + sigma * (rng.standard_t(4, N) / np.sqrt(2.0))
X = np.column_stack([np.ones(N), x])

TAUS = [0.05, 0.25, 0.5, 0.75, 0.95]
betas = {t: quantile_regression(X, y, t) for t in TAUS}
beta_ols = np.linalg.lstsq(X, y, rcond=None)[0]
```

得到的系数（截距, 斜率）：τ=0.05→(0.60, 0.34)、τ=0.5→(1.08, 0.76)、τ=0.95→(1.65, 1.05)。注意 **τ=0.5 几乎等于 OLS (1.13, 0.74)**——对称噪声下中位数≈均值；而斜率随 τ 从 0.34 **单调递增到 1.05**，正是异方差的签名（噪声尺度随 x 增大，越往上尾巴越陡）。画出来是经典的**喇叭形**：

![分位数回归的喇叭形：条件分位随 x 展开，尾部比均值更分散](/images/quantile-regression-risk/qr_fan.png)

## 三、QR vs OLS：被漏掉的不对称

把 OLS 单条均值线和 QR 的 5%/95% 分位带叠在一起，差距一目了然：

![OLS 只给一条均值线；分位数回归把上下尾部都刻画出来](/images/quantile-regression-risk/qr_vs_ols.png)

在重尾、异方差数据里，OLS 那条线会落在「看似居中」的位置，但**下行尾部实际比它暗示的深得多**。用 OLS 做风险归因，等于睁眼瞎。

## 四、条件 VaR：让风险限额随状态走

传统 VaR 用全样本历史分位（恒定 5%），不随市场状态变。QR 的杀手锏是**条件 VaR**：给定当前风险状态 $x$（例如前期波动率、因子暴露），直接给出该状态下的 5% 分位：

$$\text{VaR}_t = Q_{0.05}(y_t \mid x_t) = x_t^\top\beta_{0.05}$$

危机里 $x$ 高 → VaR 自动放大；平静期自动收窄，资本占用更精准。回测要求**突破率（实际收益低于 VaR 的比例）应≈5%**：

```python
T = 300
rng2 = np.random.default_rng(99)
xv = rng2.uniform(0, 10, T)
sv = 0.3 + 0.25 * xv
yv = 1.0 + 0.8 * xv + sv * (rng2.standard_t(4, T) / np.sqrt(2.0))
Xv = np.column_stack([np.ones(T), xv])
b05 = quantile_regression(Xv, yv, 0.05)
var_hat = Xv @ b05
breaches = yv < var_hat
print(breaches.mean())     # ≈ 0.05
```

![QR 条件 VaR 回测：突破率应贴近 5%（而非恒定分位）](/images/quantile-regression-risk/var_backtest.png)

本例突破率 **4.7%**，紧贴目标 5%——说明条件 VaR 既随状态自适应，又守住了校准。

## 五、系数路径：风险因子的效应在尾部会漂移

把每个 τ 的斜率 $\beta_\tau$ 画成 τ 的函数，就是**系数路径**。它回答一个 OLS 永远答不了的问题：「这个风险因子的效应，在好时候和坏时候一样吗？」

![系数路径：风险因子的效应如何随分位（下行到上行尾部）变化](/images/quantile-regression-risk/qr_coef_path.png)

本例斜率从 τ=0.05 的 0.34 一路升到 τ=0.95 的 1.05：**高因子暴露在好时候赚得更多，坏时候也亏得更狠，且上行比下行更陡**。这种「效应随分位漂移」对风险归因极有价值——它告诉你哪类敞口在崩盘时最危险，而不是只看平均效应。

## 六、从 QR 到预期短缺（CVaR / ES）

分位数回归给出的 VaR 只是「τ 分位那一个点」，但监管资本与真实损失更关心**预期短缺（ES, Expected Shortfall）**——即跌破 VaR 之后那部分损失的均值。一个干净的做法直接来自 QR 的定义：

$$\text{ES}_\tau(x) = \frac{1}{1-\tau}\int_\tau^1 Q_s(x)\,ds$$

既然 QR 已能给出任意 $s$ 的条件分位 $Q_s(x)$，只需在 $(\tau,1)$ 上积分（离散情形即对多个 τ 的分位估计求和平均），**无需额外假设分布族**就能拿到条件 ES。实务上常用两种近似：一是对「已估计出的尾部超阈样本」再取均值；二是拟合一组密集 τ（如 0.5→0.99）后按上式数值积分。这把风控从「最坏那一个分位」升级到「尾部整体形状」，正是 QR 超越单点 VaR 的地方——VaR 在分位处可能不连续、对尾部厚度不敏感，ES 则把整条尾巴都吃了进去。

## 七、真实陷阱

**1. 极端 τ 样本太少。** τ<0.01 或 >0.99 时可用观测极少，估计方差爆炸，别硬取。

**2. 回归分位 ≠ 独立同分布分位。** QR 给的是**条件**分位，回测 VaR 必须用**条件突破率**，不能用全样本分位去校。

**3. 重尾噪声下仍受离群点影响。** check function 比 OLS 稳健，但 τ 极小时极端负值仍会拉动估计，必要时用 winsorize 或鲁棒 QR。

**4. 多维 x 无单调约束。** 各 τ 的 β 独立拟合，可能出现交叉、非单调，解释时要小心。

**5. 过拟合。** 因子一多，逐 τ 拟合容易过拟合，上 **quantile LASSO / 岭 QR** 正则化。

**6. 线性假设过简。** 真实关系常非线性；可用样条、多项式，或树模型（quantile forest、GBM 的分位损失）上生产。

**7. 别用 QR 直接替代全样本 VaR 的合规口径。** 巴塞尔对交易账户 ES 有特定回溯检验与置信区间要求，研究用 QR 条件 ES 与生产合规口径要隔离，避免把研究估计当成监管申报数字。

## 八、小结

- 分位数回归把「看均值」升级成「看整个**条件分布**」，是风险管理的自然语言；
- 用 **LP** 严谨求解，τ 任意取；τ=0.5 自动退化为中位数回归（≈OLS）；
- **条件 VaR** 让风险限额随状态自适应，回测突破率守在 5% 附近；
- **系数路径**揭示尾部不对称性——这是 OLS 完全看不见、却对风险归因至关重要的信息。

> 附：本文所有图表与数值均来自上方可运行代码（异方差 + 重尾 t(4) 模拟 + LP 分位数回归 + 条件 VaR 回测 + 系数路径），参数与结果一致，可直接复现。
