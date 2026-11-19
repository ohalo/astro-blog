---
title: "协整配对交易：用误差修正模型把两只股票的价差做成均值回归信号"
description: "相关性高的两只股票未必能做配对交易——相关只是同涨同跌，协整才保证价差会回归。本文从伪回归讲起，用已知真值的误差修正模型造出一对协整股票（对冲比率 1.6、价差 AR 系数 0.97、半衰期 22.76 天），走一遍 Engle-Granger 两步法：OLS 估出对冲比率 1.5433，对残差做 ADF 得 -3.442（1% 临界 -3.43，拒绝单位根），而两条价格水平序列 ADF 仅 -0.749/-0.746 无法拒绝。半衰期从回归系数估出 27.11 天接近真值。样本外 499 天 z-score 策略 Sharpe 0.976，跑赢买入持有的 -0.212/-0.244；但当对冲比率从第 1250 天起漂移、协整失效，同一策略 Sharpe 崩到 -9.018、累计亏损 -2.35。附完整代码、交易稀疏的诚实披露与四张真实计算图。"
publishDate: '2026-11-19'
tags:
  - 量化交易
  - 配对交易
  - 协整
  - 均值回归
  - 误差修正模型
  - ADF检验
  - Python
language: Chinese
difficulty: advanced
---

配对交易的第一课通常是"找两只相关的股票"。但这几乎是个陷阱。两只股票可以高度相关——趋势相同、同涨同跌——却永远不回头，价差一路走远。你以为在做均值回归，其实是在赌一个永不兑现的回归。真正让配对交易站得住脚的不是相关性，而是**协整**：价差本身必须是平稳的，偏离之后迟早会回来。

这篇文章把协整配对交易从头走一遍，而且用的是**已知真值**的数据——我知道对冲比率是多少、价差多久回归一次——这样每一步估得准不准都能当场对答案。最后我会做一件很多人不愿意做的事：故意破坏协整关系，看策略怎么死。

## 为什么相关不够：伪回归的幽灵

设两只股票的对数价是 $y_t$ 和 $x_t$。如果它们各自都是随机游走（一阶单整 I(1)），哪怕彼此毫无关系，直接把 $y_t$ 回归到 $x_t$ 也很容易得到漂亮的 $R^2$ 和显著系数——这就是**伪回归**。两个独立的随机游走会在样本里碰巧走出相似的漂移，OLS 会把这种巧合当成关系。

Granger-Newbold 早在七十年代就把这件事说透了：对 I(1) 序列做回归，t 统计量不再服从标准分布，你以为的"显著"有极大概率是假的。解法不是放弃回归，而是问一个更硬的问题：**回归残差平稳吗？** 如果 $y_t - \beta x_t$ 是平稳的 I(0)，那么这对序列就是协整的，$\beta$ 是真实的长期均衡关系，回归不是巧合。这个平稳的残差，就是配对交易里那个"终将回归"的价差。

协整还有一个更直观的名字：**误差修正机制**。价差短期会偏离均衡，但存在一个把它拉回去的力量。这个力量有多强、回归要多久，决定了策略能不能赚钱。

## 实验设计：一块真值已知的试验田

我造了 1500 个交易日的合成数据，机制完全透明：

- 股票 B 的对数价 $x_t$ 是带漂移的随机游走，日波动 1.1%，代表共同的市场趋势；
- 价差 $e_t$ 是一个 AR(1) 过程，$e_t = 0.97 e_{t-1} + \varepsilon_t$，波动 0.75%，代表围绕均衡的短期偏离；
- 股票 A 的对数价 $y_t = 1.6 x_t + e_t$。

对冲比率真值是 1.6，价差 AR 系数 0.97。由 AR(1) 系数可以算出**半衰期**——偏离均衡后衰减一半所需的天数：

$$h = \frac{\ln 2}{-\ln 0.97} \approx 22.76 \text{ 天}$$

这个 22.76 天是我埋在数据里的标准答案。接下来所有估计都要拿它来对。

!![协整股票对与价差](/images/cointegration-pairs-trading/fig1_prices_spread.png)

图1上半部分：两条价格序列共享一条随机趋势，肉眼看几乎重合。真正的机会藏在下半部分的价差里——它围绕零线上下摆动，两条橙色虚线是 ±2 倍标准差的开仓阈值。注意 x=1000 处的红色竖线：左边是估计期，右边是样本外交易期。

## Engle-Granger 两步法

### 第一步：OLS 估对冲比率

只用前 1000 天，把 $y$ 回归到 $x$：

$$
y_t = \hat\alpha + \hat\beta x_t + \hat e_t
$$

估出 $\hat\beta = 1.5433$，接近真值 1.6（样本误差约 3.5%）。这个 $\hat\beta$ 就是配对交易里"买一份 A 要配多少份 B"的对冲比率。残差 $\hat e_t = y_t - \hat\alpha - \hat\beta x_t$ 是被估出来的价差序列。

### 第二步：对残差做 ADF 检验

关键一步。对残差做增广 Dickey-Fuller 检验，回归：

$$
\Delta \hat e_t = a + b\,\hat e_{t-1} + \gamma \Delta \hat e_{t-1} + u_t
$$

检验 $b=0$（存在单位根）。结果见下图。

!![ADF检验结果对比](/images/cointegration-pairs-trading/fig2_adf_test.png)

三条柱子的含义截然不同。价格水平 A 和 B 的 ADF 统计量分别是 **-0.749** 和 **-0.746**，远离 5% 临界值 -2.86——无法拒绝单位根，它们各自是随机游走。而回归残差的 ADF 是 **-3.442**，比 1% 临界值 -3.43 还要负——在 1% 水平拒绝单位根，残差平稳。

这正是协整的定义：两个 I(1) 序列的线性组合是 I(0)。价格本身没有均值回归，但它们的**价差有**。配对交易赚的就是这个回归的钱。

## 半衰期：这个回归有多快

对残差再拟合一次 $\Delta \hat e_t = \gamma \hat e_{t-1} + c$，得到 $\hat\gamma = -0.02557$，对应半衰期：

$$h = \frac{\ln 2}{0.02557} = 27.11 \text{ 天}$$

真值是 22.76 天，估计值 27.11 天——量级对了，但偏慢了约 19%。这个偏差来自估计误差和有限样本，提醒我们：**实盘里你永远不知道真实的回归速度，只能估，而且会估错**。

!![价差自相关衰减与半衰期](/images/cointegration-pairs-trading/fig3_halflife.png)

图中蓝色圆点是价差的自相关函数，红色虚线是真实的 $0.97^t$ 衰减，绿色点线是拟合衰减。三者吻合得不错。紫色竖线标出半衰期位置——大约三周后，偏离会衰减一半。这个数字直接决定持仓周期：太快则交易成本吃掉利润，太慢则资金占用久、不确定性大。

## z-score 策略：把价差变成仓位

有了平稳价差，交易规则就很自然。把价差标准化成 z-score：

$$z_t = \frac{\hat e_t - \mu}{\sigma}$$

- $z > 2$：价差过高，做空价差（卖 A、买 $\hat\beta$ 份 B），赌它回落；
- $z < -2$：价差过低，做多价差；
- $|z| < 0.5$：价差回到均衡，平仓。

在样本外 499 天里，策略 Sharpe 达到 **0.976**，而同期买入持有 A 是 **-0.212**、买入持有 B 是 **-0.244**——两只票本身都在阴跌，配对策略却靠来回做价差赚了正收益。累计对数收益 0.0589。

**诚实披露**：499 天里只触发了 **2 次完整交易**。2σ 是很严格的阈值，价差很少走到那么远，导致信号稀疏。0.976 的 Sharpe 建立在极少数几笔交易上，统计意义有限——这远不足以当作实盘业绩，只是演示方法有效性的玩具例子。真要用，得放松阈值、扩大股票池、并认真计入交易成本和冲击成本。

## 当协整失效：策略是怎么死的

现在做那件重要的事。假设从第 1250 天起，股票 A 和 B 的基本面关系变了，对冲比率从 1.6 缓慢漂移到 2.1。价差不再是平稳的 $e_t$，而混入了一个越来越大的趋势项。协整关系破裂了。

把同一套策略原封不动跑上去，结果触目惊心：

!![策略净值与协整失效风险](/images/cointegration-pairs-trading/fig4_equity.png)

绿色是正常情形的净值曲线，缓缓向上；红色是协整失效后的曲线，在第 1250 天（图中黑色竖线）之后一路俯冲。Sharpe 从 0.976 崩到 **-9.018**，累计对数收益 **-2.35**——亏掉净值的一大半。黑色竖线右边，策略反复在错误的方向上加仓：它以为价差会回归，但价差已经不再有回归机制，越做越亏。

这就是配对交易最真实的风险。协整不是永久的物理定律，而是历史统计规律。公司合并、业务重组、监管冲击都可能让一对曾经协整的股票分道扬镳。所以实盘中必须：滚动重估协整关系、监控残差的平稳性、设置止损、以及永远不要把全部仓位压在一对股票上。**每笔亏损的配对交易背后，几乎都有一句"它们以前一直很准"。**

## 完整代码

下面是可独立运行的完整实现（需 numpy、matplotlib）：

```python
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(20261119)
N = 1500
TRAIN = 1000

# 1) 造已知真值的协整数据
sig_x = 0.011
x = np.empty(N); x[0] = 4.5
x[1:] = x[0] + np.cumsum(rng.normal(0, sig_x, N - 1))
rho_true, sig_e = 0.97, 0.0075
e = np.empty(N)
e[0] = rng.normal(0, sig_e / np.sqrt(1 - rho_true**2))
for t in range(1, N):
    e[t] = rho_true * e[t-1] + rng.normal(0, sig_e)
beta_true = 1.6
y = beta_true * x + e
half_life_true = np.log(2) / (-np.log(rho_true))   # 22.76

# 2) 手写 ADF 检验：dy[j]=a+b*y[j]+sum_i g_i*dy[j-i]
def adf_test(series, maxlag=1):
    ys = np.asarray(series, float)
    dy = np.diff(ys); L = len(dy)
    idx = np.arange(maxlag, L)
    cols = [np.ones(len(idx)), ys[idx]]
    for i in range(1, maxlag + 1):
        cols.append(dy[idx - i])
    X = np.column_stack(cols); Y = dy[idx]
    coef, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ coef
    s2 = resid @ resid / (len(Y) - X.shape[1])
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    return coef[1] / se

# 3) Engle-Granger 两步法（仅用训练段）
Xtr = np.column_stack([np.ones(TRAIN), x[:TRAIN]])
a_hat, beta_hat = np.linalg.lstsq(Xtr, y[:TRAIN], rcond=None)[0]
resid_train = y[:TRAIN] - (a_hat + beta_hat * x[:TRAIN])
print("beta_hat", beta_hat, "true", beta_true)
print("ADF y/x/resid:", adf_test(y[:TRAIN]), adf_test(x[:TRAIN]),
      adf_test(resid_train))   # -0.75 / -0.75 / -3.44

# 4) 半衰期
de = np.diff(resid_train); elag = resid_train[:-1]
gamma = np.polyfit(elag, de, 1)[0]
print("half-life", np.log(2) / -gamma, "true", half_life_true)  # 27.11 / 22.76

# 5) z-score 策略（样本外）
resid_full = y - (a_hat + beta_hat * x)
mu, sd = resid_train.mean(), resid_train.std(ddof=1)
z = (resid_full - mu) / sd
TEST = np.arange(TRAIN, N - 1)
pos, cur = np.zeros(len(TEST), int), 0
for i, zz in enumerate(z[TEST]):
    if cur == 0:
        if zz > 2: cur = -1
        elif zz < -2: cur = 1
    elif abs(zz) < 0.5:
        cur = 0
    pos[i] = cur
pnl = pos * np.diff(resid_full)[TEST]
sharpe = pnl.mean() / pnl.std(ddof=1) * np.sqrt(252)
print("pairs Sharpe", sharpe)   # 0.976
```

把结构设计成可复用的：`adf_test` 是通用函数，`beta_hat` 和半衰期的估计流程可直接换成真实行情数据。换数据时唯一要注意的是——**先在你的样本上重新验证协整性**，别拿这里的结果想当然。

## 小结

协整配对交易的逻辑链条其实很干净：相关性不够，协整才是前提；Engle-Granger 两步法用 OLS 估对冲比率、用 ADF 验证残差平稳；半衰期告诉你回归有多快；z-score 把价差翻译成仓位。合成数据上，这套流程把 1.6 的对冲比率估到 1.5433、把 22.76 天的半衰期估到 27.11 天、让策略 Sharpe 从买入持有的负值翻正到 0.976。

但真正的教训在最后一段：协整是会失效的。关系一变，同一个策略 Sharpe 从 0.976 塌到 -9.018。方法本身不难，难的是对"历史规律可能不再成立"这件事保持敬畏。能把赚的钱留住的，从来不是更精巧的模型，而是对模型失效的持续监控和严格的止损纪律。
