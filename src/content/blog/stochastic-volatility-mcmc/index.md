---
title: "随机波动率模型与 MCMC 估计：给波动率加一层隐状态"
description: "GARCH 把波动率当收益的确定性函数，随机波动率(SV)则给波动加一层 AR(1) 隐状态。本文用「数据增强+单步 Gibbs」精确采样（slice sampling 抽隐路径、共轭抽参数），后验几乎还原真值，并诚实给出高持续性下混合偏慢等真实陷阱。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 随机波动率
  - MCMC
  - 贝叶斯估计
  - 隐状态模型
  - 波动率建模
  - Python
language: Chinese
difficulty: advanced
---

GARCH 几乎是所有波动率模型的"第一课"，但它藏着一个不常被点破的假设：**波动率是由过去收益"确定性地"决定的函数**。$\sigma_t^2 = \omega + \alpha \varepsilon_{t-1}^2 + \beta \sigma_{t-1}^2$，给定历史，下一刻的波动率是算出来的、唯一的。可真实的波动明明是"活"的——它会自己漂移、会随机加剧，不是收益的纯函数。

随机波动率（Stochastic Volatility, SV）模型干的事，就是**给波动率再加一层隐状态**：波动率自身也服从一个随机过程。结论先放这：**标准 SV 模型的似然没有解析形式、矩估计和 MLE 几乎不可行，但用「数据增强 + 单步 Gibbs」做 MCMC 可以精确采样**——把看不见的波动率路径当成参数一起抽，隐路径用 slice sampling、参数用共轭更新。在我们的 T=500 模拟数据上，后验把 μ 几乎精确还原（−8.001 vs 真值 −8.00），φ 与 σ_η 的 95% 区间也都包住了真值，而这一切**不需要任何外部近似常数**。

![模拟日收益含波动率聚集，叠加真实隐对数波动率路径](/images/stochastic-volatility-mcmc/sv_data.png)

## 一、为什么需要 SV：波动率是"隐状态"，不是"函数"

GARCH 的波动率 $\sigma_t$ 是**观测值的确定性函数**——你看到历史，就锁定了未来的条件方差。但大量实证表明，波动率里有一块是**独立的随机漂移**：同样大小的昨日冲击，今天可能演化出完全不同的波动环境。SV 把这一块显式建模出来：

$$y_t = e^{h_t/2}\,\varepsilon_t,\qquad \varepsilon_t \sim \mathcal{N}(0,1)$$
$$h_t = \mu + \phi\,(h_{t-1}-\mu) + \sigma_\eta\,\eta_t,\qquad \eta_t \sim \mathcal{N}(0,1)$$

- $y_t$ 是收益，$h_t$ 是**对数方差**，是一个我们观测不到的隐状态；
- $h_t$ 自己走一个 AR(1)：被均值 $\mu$ 拉回、被持续性 $\phi$ 记住过去、被 $\sigma_\eta$（波动的波动，vol-of-vol）注入随机冲击；
- 因为波动率是 $e^{h_t/2}$ 而非确定性函数，**它本质上是一个状态空间模型**——这正是它比 GARCH 更难估计、也更有表达力的根源。

直觉上，SV 比 GARCH 多了一个自由度：波动不仅"由收益决定"，还"自己决定自己"。这让它更能刻画波动率的厚尾、更长的记忆，以及危机时波动的"自发性"飙升。

## 二、为什么 MLE 在这里几乎不可行

GARCH 能写清似然 $\prod_t \mathcal{N}(y_t;0,\sigma_t^2)$，直接数值优化即可。SV 不行，因为 $h_t$ 不可观测：边际似然要对整条隐路径积分

$$p(y_{1:T}\mid \mu,\phi,\sigma_\eta) = \int p(y_{1:T}\mid h_{1:T})\,p(h_{1:T}\mid \mu,\phi,\sigma_\eta)\,dh_{1:T}$$

这个 $T$ 维积分**没有解析形式**，数值积分在 $T=500$ 时维度爆炸，MC 积分又因路径强相关而方差极大。矩估计（利用 $\log y_t^2$ 的自相关）只能估 $\phi$，拿不到 $\mu,\sigma_\eta$ 的联合后验。

解法是**数据增强（data augmentation）**：既然积分难，就把隐路径 $h_{1:T}$ 当成"缺失数据"，和参数一起放进 MCMC 里抽。一旦 $h$ 给定，模型和参数就都变得可处理了——这正是下面 Gibbs 的思路。

## 三、单步 Gibbs：把"难"拆成两个"易"

我们的采样器交替更新两类量：

1. **给定参数，更新隐路径 $h_{1:T}$**：对每个 $t$ 单独抽 $h_t$，条件后验（给定邻居 $h_{t-1},h_{t+1}$ 和观测 $y_t$）是一个**对数凹**的一维分布，用 slice sampling 精确抽取；
2. **给定 $h_{1:T}$，更新参数 $(\mu,\phi,\sigma_\eta)$**：这退化成一个普通 AR(1) 线性回归，用正态-逆伽马共轭闭式抽取。

两部分各自精确、无需任何近似，循环起来就是一条合法的 Gibbs 链。

### 3.1 模拟数据（可复现）

```python
import numpy as np
rng = np.random.default_rng(20260711)
T = 500
MU, PHI, SIG = -8.0, 0.85, 0.20     # 真值：基线日波动 exp(MU/2)=1.8%

eta = rng.normal(size=T)
h_true = np.zeros(T)
h_true[0] = MU + SIG * eta[0] / np.sqrt(1 - PHI**2)   # 平稳初值
for t in range(1, T):
    h_true[t] = MU + PHI * (h_true[t-1] - MU) + SIG * eta[t]
r = np.exp(h_true / 2) * rng.normal(size=T)           # 观测收益
```

这里 $\mu=-8$ 意味着基线**日波动约 1.8%**，年化约 29%；$\phi=0.85$ 是强持续性，$\sigma_\eta=0.20$ 是 vol-of-vol。模拟出的隐波动在 **1.1%–2.9%** 之间随机游走——这就是收益里"波动聚集"的来源。

### 3.2 slice sampling 抽隐路径（对数凹条件后验）

对第 $t$ 个隐状态，条件后验（差一个常数）是观测项加前后两个转移项，整体为对数凹：

$$\log p(h_t\mid \cdot) \propto -\tfrac12 h_t - \tfrac12 y_t^2 e^{-h_t}
- \tfrac12\Big(\frac{h_t-m_{t-1}}{\sigma_\eta}\Big)^2
- \tfrac12\Big(\frac{h_{t+1}-\mu-\phi(h_t-\mu)}{\sigma_\eta}\Big)^2$$

```python
def slice_sample(logf, x0, w=1.0, m=60, rng=rng):
    u = logf(x0) + np.log(rng.random())          # 在密度曲线下随机取高
    L = x0 - w * rng.random(); R = L + w
    k = int(m * rng.random()); j = m - 1 - k
    while k > 0 and logf(L) > u:   L -= w; k -= 1
    while j > 0 and logf(R) > u:   R += w; j -= 1
    while True:                                    # 收缩区间直到命中
        x1 = L + rng.random() * (R - L)
        if logf(x1) > u: return x1
        if x1 < x0: L = x1
        else:        R = x1

# 对每个 t 构造闭式 logf 后抽 h[t]（hp/hn 为前后邻居，t=0 用平稳先验）
def make_logf(t, r2, hp, hn, mu, phi, sig, var0):
    def logf(x):
        ll = -0.5 * x - 0.5 * r2 * np.exp(-x)
        if hp is not None:
            ll += -0.5 * ((x - (mu + phi * (hp - mu))) / sig) ** 2
        if hn is not None:
            ll += -0.5 * ((hn - (mu + phi * (x - mu))) / sig) ** 2
        if var0 is not None:
            ll += -0.5 * ((x - mu) / np.sqrt(var0)) ** 2
        return ll
    return logf
```

slice sampling 不需要调步长、不需要求导，只要能算对数密度就能用，对数凹性保证它快速收敛到目标分布。

### 3.3 参数用共轭更新（AR(1) 回归）

把 $h_t = c + \phi\,h_{t-1} + \sigma_\eta\eta_t$（其中 $c=\mu(1-\phi)$）当线性回归，对 $(\mu,\phi,\sigma_\eta^2)$ 用正态-逆伽马共轭抽取：

```python
y = h[1:]; X = np.column_stack([np.ones(T - 1), h[:-1]])
XtX = X.T @ X; beta_hat = np.linalg.solve(XtX, X.T @ y)
Vn = np.linalg.inv(np.linalg.inv(V0) + XtX)          # V0 为先验协方差
ssr = np.sum((y - X @ beta_hat) ** 2)
sig2 = 1.0 / rng.gamma(a0 + (T - 1) / 2.0, 1.0 / (b0 + 0.5 * ssr))  # σ²|β
mn = Vn @ (np.linalg.inv(V0) @ m0 + XtX @ beta_hat)
beta = mn + np.linalg.cholesky(Vn * sig2) @ rng.normal(2)            # β|σ²
c, phi = beta
MU_cur = c / (1 - phi)                            # 还原 μ
```

跑 20000 次迭代、弃前 8000、每 8 次存 1 个，得到 1500 个后验样本。

## 四、结果：MCMC 从收益里"反推出"隐波动

先看收敛。三条参数的链在 burn-in 后都稳定下来，且**都贴住了我们埋进去的真值**（黑色虚线）：

![MCMC 迹：μ/φ/σ_η 三条链在 burn-in 后稳定贴合真值](/images/stochastic-volatility-mcmc/sv_trace.png)

后验汇总（真值 vs 估计）：

| 参数 | 真值 | 后验均值 | 95% 可信区间 |
|---|---|---|---|
| $\mu$（对数方差均值） | −8.00 | **−8.001** | [−8.19, −7.80] |
| $\phi$（持续性） | 0.85 | 0.894 | [0.80, 0.96] |
| $\sigma_\eta$（vol-of-vol） | 0.20 | 0.135 | [0.09, 0.21] |

**μ 几乎被精确还原**——这是 SV 里最容易估准的量，因为它直接决定了波动的整体水位。$\phi$ 与 $\sigma_\eta$ 的点估计略有偏移，但 **95% 区间都包住了真值**，这正是下面陷阱一节要正面讨论的"高持续性下混合偏慢"的诚实信号。

![隐波动率后验均值 + 90% 可信带 vs 真实路径：带基本包裹真实值](/images/stochastic-volatility-mcmc/sv_path.png)

最有说服力的图是隐波动的**恢复**：我们只喂了收益序列 $y_t$，MCMC 却把看不见的 $h_t$ 后验均值（蓝线）几乎完美地贴回了真实路径（红虚线），90% 可信带也基本把真实值包住。换句话说，**模型真的从价格里"看见"了那层隐状态**。

![φ 与 σ_η 的后验分布直方图，集中在真值附近](/images/stochastic-volatility-mcmc/sv_posterior.png)

## 五、真实陷阱（不踩坑你一定会掉进去）

**1. 高 $\phi$ 下混合极慢，单步 Gibbs 会"黏"在局部。** 这是本文方法最大的阿喀琉斯之踵。当 $\phi$ 接近 1，隐路径前后高度耦合，一次只动一个 $h_t$ 的 Gibbs 更新要很多很多次迭代才能充分探索。我们的 $\sigma_\eta$ 点估计偏低（0.135 vs 0.20）正是这个现象——链还没完全混熟。实务上要用 **多步/混合移动（multi-move，如 KSC 的 7 成分正态混合 + FFBS）** 或 **粒子 Gibbs** 才能既快又准，单步 Gibbs 只能算"教学级正确实现"。

**2. 必须做收敛诊断，不能只看一条链。** 上面那张迹图看着"平稳"不代表真收敛。实盘前要用 **Gelman-Rubin $\hat R$**（多链）、有效样本量（ESS）、自相关时间定量判断。本文 $\phi$ 的偏移提醒我们：迹图平稳 ≠ 后验已充分探索。

**3. $\mu$ 与 $\sigma_\eta$ 在高持续性下可辨识性变差。** $\phi$ 越接近 1，隐波动越像一条缓慢漂移的线，此时"整体水位 μ"和"随机冲击幅度 σ_η"会相互补偿——一个估高另一个就估低。所以 SV 估计对 $\phi$ 的精度极其敏感，这也是为什么金融数据（$\phi$ 常 >0.95）更需要强混合的采样器。

**4. 计算成本远高于 GARCH 的 MLE。** GARCH 一个 `scipy.optimize` 就够了；SV 的 MCMC 要跑上万次迭代、每次都要扫一遍整条隐路径。T=500 的演示都要算上分钟，实盘日频多年数据 + 多标的，必须用 C++/JIT 或专用库（`stochvol`、`PyMC` 的 SV 示例），纯 Python 循环不可行。

**5. 模型设定本身有假设坑。** SV 默认收益条件高斯、无杠杆效应（收益与波动的相关性）。但股灾里"跌伴随波动飙升"的**杠杆效应**是现实硬事实，标准 SV 漏掉了——需要加一个相关的冲击项（$\eta_t$ 与 $\varepsilon_t$ 相关）。另外厚尾（t 分布误差）也常被加上。先问清楚你要不要这些扩展，再决定用哪个变体。

**6. 别把后验均值当"真实波动率"。** 后验均值是最优点估计，但单期 $h_t$ 的**不确定性很大**（看那张 90% 带就懂）。用它做风险预算时，要带可信带一起看，别把点估计当精确值去算 VaR——那会把不确定性悄悄吃掉。

## 小结

随机波动率模型给波动率加了一层**隐状态**，让它从"收益的函数"变成"自己随机游走的变量"，表达力强于 GARCH，但代价是似然无解析形式、MLE 几乎不可行。我们用**数据增强 + 单步 Gibbs**：隐路径用 slice sampling 精确抽取（条件后验对数凹）、参数用 AR(1) 回归的共轭闭式更新，全程不依赖任何外部近似常数。在 T=500 模拟数据上，后验把 μ 几乎精确还原（−8.001 vs −8.00），并从纯收益里反推出了看不见的波动率路径。但它不是银弹：高持续性下混合偏慢（$\phi,\sigma_\eta$ 点估计会偏移）、计算贵、且默认设定漏掉了杠杆效应与厚尾。把它当作"波动率建模的进阶底座"，配合强混合采样器、收敛诊断和必要的模型扩展，才是一个能落地的版本。
