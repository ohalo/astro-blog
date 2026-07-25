---
title: "Almgren-Chriss 最优执行：在冲击成本与时间风险之间找平衡点"
publishDate: '2026-07-26'
description: "Almgren-Chriss 最优执行：在冲击成本与时间风险之间找平衡点 - 100万股清仓实验：TWAP期望缺口26万但标准差58万，AC(λ=2e-6)期望38万标准差42万，95%VaR从121万降到108万；风险厌恶买的是确定性 - halo的技术博客"
tags:
 - 量化交易
language: Chinese
---

![Almgren-Chriss 最优清仓轨迹](/images/almgren-chriss-optimal-execution/trajectories.png)

假设你管理的组合要在今天之内卖掉某只股票 100 万股，市值约 5000 万元。你面前只有两个坏选项：

- **卖得快**：一小时内砸完。市场没有那么多即时买盘，你的卖单会把价格打下去——这叫**冲击成本**。砸得越急，成本越高，而且是超线性的。
- **卖得慢**：拖满全天甚至几天。每一片没卖掉的仓位都继续暴露在市场波动里——万一下午出个利空，剩余仓位跌 2%，损失比冲击成本大一个数量级。这叫**时间风险**。

快了付冲击，慢了担风险。这是所有大单执行绕不开的根本权衡。2000 年 Almgren 和 Chriss 发表的《Optimal Execution of Portfolio Transactions》把这个权衡写成了一个可以解析求解的优化问题，给出的答案至今仍是几乎所有券商算法交易（TWAP/VWAP/IS 算法）的理论底座。

这篇文章从模型假设讲到闭式解，再用 2 万次蒙特卡洛验证：**风险厌恶买到的不是更低的成本，而是更窄的成本分布**。

## 模型：把执行写成均值-方差问题

Almgren-Chriss 的框架有三块积木。

**第一块：交易轨迹。** 把交易日切成 $N$ 个时间片（本文取 60 片、每片约 4 分钟），$x_k$ 表示第 $k$ 片结束时的剩余持仓，$x_0 = X$（初始 100 万股），$x_N = 0$（收盘前必须清空）。每片卖出量 $n_k = x_{k-1} - x_k$。

**第二块：价格冲击，分两种。**

- **永久冲击**：你的卖出改变了市场对均衡价格的认知，价格被压低后不回来。线性形式：每卖出速度 $v$ 使价格永久下移 $\gamma v$。
- **临时冲击**：为了立刻成交你吃掉了盘口深度，价格暂时被压低，你走了它就弹回来。线性形式：你的实际成交价比中间价低 $\eta v$，其中 $v = n_k / \tau$ 是交易速率。

**第三块：价格随机游走。** 除了你自己的冲击，价格还有外生波动 $\sigma$。剩余持仓 $x_k$ 越大、拖的时间越长，暴露在这个波动里的敞口就越大。

把三块拼起来，实施缺口（初始市值减实际收入）的期望和方差是：

$$E[\text{cost}] = \frac{1}{2}\gamma X^2 + \eta \sum_k \frac{n_k^2}{\tau}, \qquad V[\text{cost}] = \sigma^2 \tau \sum_k x_k^2$$

注意两个关键结构：期望成本里的 $n_k^2$ 惩罚"单片卖太多"，方差里的 $x_k^2$ 惩罚"仓位留太久"。两个目标天然打架。

Almgren-Chriss 的做法是经典的均值-方差：最小化 $E + \lambda V$，其中 $\lambda$ 是风险厌恶系数。这个二次规划有闭式解：

$$x_k = X \cdot \frac{\sinh\left(\kappa (T - t_k)\right)}{\sinh(\kappa T)}$$

其中 $\kappa \approx \sqrt{\lambda \sigma^2 / \eta}$ 叫**urgency（紧迫度）**。整条轨迹是一条双曲正弦衰减曲线，形状完全由 $\kappa$ 一个数决定。

两个极限很有启发：

- $\lambda \to 0$（风险中性）：$\kappa \to 0$，轨迹退化成直线——**TWAP 就是 AC 在风险中性下的特例**；
- $\lambda \to \infty$：轨迹变成开盘瞬间全砸——极度怕风险的人愿意付任何冲击成本换取立刻落袋。

## 实现：60 行核心代码

合成市场参数：股价 50 元，日波动 2%，临时冲击系数 $\eta = 2.5 \times 10^{-7}$（卖出全仓 1/60 大约压价 8 个 bp），永久冲击为临时的 1/10。

```python
import numpy as np

X, T, N = 1_000_000, 1.0, 60          # 100万股, 1个交易日, 60片
tau = T / N
S0, sigma = 50.0, 0.02
sigma_abs = sigma * S0                 # 绝对价格波动（元）
eta, gamma_perm = 2.5e-7, 2.5e-8       # 临时/永久冲击系数

def ac_trajectory(lam):
    """Almgren-Chriss 闭式解：给定风险厌恶 lambda 返回持仓轨迹"""
    if lam <= 0:
        return np.linspace(X, 0, N + 1)          # 风险中性 = TWAP
    k2 = lam * sigma_abs**2 / (eta * (1 - gamma_perm * tau / (2 * eta)))
    kappa = np.arccosh(k2 * tau**2 / 2 + 1) / tau  # 离散修正的 urgency
    k = np.arange(N + 1)
    return X * np.sinh(kappa * (T - k * tau)) / np.sinh(kappa * T)

def cost_stats(x):
    """期望冲击成本与方差"""
    n = -np.diff(x)
    v = n / tau
    temp = np.sum(eta * v * n)                    # 临时冲击
    perm = 0.5 * gamma_perm * X**2                # 永久冲击（轨迹无关）
    var = sigma_abs**2 * tau * np.sum(x[1:]**2)   # 时间风险
    return temp + perm, var
```

注意 `arccosh` 那一行：连续时间解里 $\kappa = \sqrt{\lambda\sigma^2/\eta}$，但离散时间下直接套会有系统偏差，Almgren-Chriss 原文给出了 $\cosh$ 修正形式。片数少的时候（比如只切 10 片）这个修正不可忽略。

第一张图展示了四个 $\lambda$ 下的轨迹：$\lambda=0$ 是直线（TWAP），$\lambda$ 越大曲线越凸、前半段卖得越多。$\lambda = 2 \times 10^{-6}$ 时，前四分之一的时间里已经卖掉了 51% 的仓位。

## 有效前沿：省下的方差要用期望成本买

对一串 $\lambda$ 扫描，把每条轨迹的期望成本和标准差画在同一张图上：

![执行的有效前沿](/images/almgren-chriss-optimal-execution/efficient-frontier.png)

这条曲线和 Markowitz 组合前沿是同一个数学对象，只是坐标换了：横轴是执行收入的不确定性，纵轴是期望冲击成本。TWAP 坐在曲线最右端——期望成本最低（26 万元），但标准差最大（57 万元）。沿曲线往左走，每减少一万元标准差，就要多付一点期望成本，而且越往左边际价格越贵。

这张图对实务的意义在于：**不存在"最好的执行算法"，只存在"匹配你风险偏好的执行算法"**。一个每天调仓、成本会在几百次执行中平均掉的量化基金，应该选右端（接近 TWAP）；一个在重仓股上做一次性清仓、这笔执行的结果直接进当月损益的主观基金，应该往左走。

## 蒙特卡洛：分布的形状才是重点

期望和方差是理论值，实际执行一次只抽一个样本。用 2 万条模拟价格路径分别跑 TWAP 和 AC（$\lambda = 2\times10^{-6}$）：

```python
def simulate(lam, n_sim=20000, rng=np.random.default_rng(42)):
    x = ac_trajectory(lam)
    n = -np.diff(x); v = n / tau
    dW = rng.normal(0, sigma_abs * np.sqrt(tau), size=(n_sim, N))
    perm = gamma_perm * np.cumsum(np.tile(v * tau, (n_sim, 1)), axis=1)
    S = S0 + np.cumsum(dW, axis=1) - perm         # 中间价路径
    revenue = (S - eta * v) @ n                   # 临时冲击后的实际成交
    return (X * S0 - revenue) / 1e4               # 实施缺口（万元）
```

![蒙特卡洛成本分布](/images/almgren-chriss-optimal-execution/monte-carlo.png)

结果一目了然：

| 策略 | 期望缺口 | 标准差 | 95% VaR |
|---|---|---|---|
| TWAP | 26.1 万 | 57.7 万 | 120.8 万 |
| AC (λ=2e-6) | 38.0 万 | 42.2 万 | 107.6 万 |

AC 的期望成本比 TWAP 高 12 万元——这是前置卖出多付的冲击费。但它的标准差窄了 27%，95% VaR 低了 13 万。换句话说：**多付 12 万的确定成本，买掉了尾部最坏情形里十几万的损失**。对一个把执行成本纳入风险预算的机构，这笔交易是划算的；对一个只看平均成本的账户，它是亏的。这正是 $\lambda$ 存在的意义——它不是拟合出来的参数，是你自己的效用函数。

还有一个不那么直观的观察：两个分布都近似正态，但 AC 的分布整体右移又收窄。执行算法改变不了市场波动，它只能决定你在波动面前暴露多久。

## 半衰期：一个比 λ 更好沟通的参数

直接跟交易员说"风险厌恶取 2e-6"没有任何直觉。更好的沟通语言是**交易半衰期**——清掉一半仓位需要的时间，它与 urgency 的关系近似 $t_{1/2} \approx \ln 2 / \kappa$：

![交易半衰期随风险厌恶下降](/images/almgren-chriss-optimal-execution/half-life.png)

$\lambda$ 从 $10^{-7.5}$ 到 $10^{-4.5}$ 三个数量级的变化，对应半衰期从接近 180 分钟压缩到 20 分钟以内。实务中的标定流程往往是反过来的：交易台先说"这单希望两小时内完成大半"，再反解出 $\kappa$ 和 $\lambda$，最后由模型生成整条平滑轨迹。

## 模型没告诉你的事

Almgren-Chriss 是理论底座，不是可以直接上线的执行系统。它的几个假设在真实市场里都要打折：

**线性冲击是近似。** 实证研究（Almgren et al. 2005 对花旗订单库的标定）发现临时冲击更接近交易速率的 0.6 次方——凹函数意味着"拆得越细越省钱"的边际收益递减，最优轨迹形状会变，但"前凸后缓"的定性结论不变。

**波动率和流动性不是常数。** 开盘和收盘的成交量占全天的大头，午盘流动性最薄。实务系统会把日历时间换成"成交量时间"，在成交量密集时段多卖——这就是 VWAP 类算法对 AC 的修正。

**没有信号项。** 模型假设价格是外生随机游走。如果你有短期 alpha（比如预测未来一小时价格要跌），最优轨迹会进一步前置；Gârleanu-Pedersen (2013) 把动态信号加进了这个框架。

**没有博弈。** 如果市场上其他人能识别你的执行模式（比如每天固定用同一条 AC 曲线），他们可以在你前面卖。这是执行算法要加随机化的原因。

即便有这些局限，AC 框架的核心洞见——**执行是一个风险-成本权衡问题，而不是单纯的成本最小化问题**——依然是对的。TWAP 不是"没有策略"，它是风险中性这个特定效用函数下的最优策略；意识到这一点，你才知道自己什么时候不该用它。

## 参考文献

- Almgren, R., & Chriss, N. (2000). Optimal Execution of Portfolio Transactions. *Journal of Risk*, 3(2), 5-39.
- Almgren, R., Thum, C., Hauptmann, E., & Li, H. (2005). Direct Estimation of Equity Market Impact. *Risk*, 18(7), 58-62.
- Gârleanu, N., & Pedersen, L. H. (2013). Dynamic Trading with Predictable Returns and Transaction Costs. *Journal of Finance*, 68(6), 2309-2340.
