---
title: "贝叶斯状态空间择时：把『现在是牛是熊』变成后验概率"
description: "市场牛熊不是开关而是隐状态。HMM+Baum-Welch EM 估参数、前向-后向算平滑后验 P(牛|全部观测)；后验>0.5 满仓否则空仓：合成回撤 −47%→−9.3%、Sharpe 0.22→0.99，附 Python 与六类陷阱（高阶）。"
publishDate: '2026-07-14'
tags:
  - 量化交易
  - 市场择时
  - 隐马尔可夫模型
  - HMM
  - 贝叶斯滤波
  - 状态空间
  - 前向-后向
  - Python
language: Chinese
difficulty: advanced
---

「现在到底是牛市还是熊市？」大多数择时策略把这个问题当成非黑即白的开关——金叉是牛、死叉是熊。但市场的牛熊更像**藏在价格背后的隐状态**：你看不到它，只能从每天 noisy 的涨跌里去推断它。

结论先放这：**把行情建模成隐马尔可夫模型（HMM），用 Baum-Welch EM 估出两个 regime（牛/熊）各自的漂移、波动与黏性转移，再用前向-后向（forward-backward）算法算平滑后验 γ_t = P(牛市 | 全部观测)，就能把「现在是牛是熊」变成一个 0~1 的概率，而不是硬切的标签。在我们的自洽合成里，这个后验驱动的择时把最大回撤从 −47% 砍到 −9.3%、Sharpe 从 0.22 拉到 0.99，且全年平均只有 44.4% 时间满仓——它靠的不是预测方向，而是「熊市阶段少亏」。** 完整可运行脚本已生成：`generate_bayesian_regime_images.py`。

![贝叶斯滤波把牛熊变成后验概率，紫线为平滑后的 P(牛)](/images/bayesian-regime-timing/posterior_probability.png)

## 一、HMM 框架：把牛熊当隐变量

HMM 的设定很自然：

- **隐藏状态** $z_t \in \{\text{牛}, \text{熊}\}$，你不能直接观测；
- **观测** $y_t$ 是月度市场收益，由当前状态生成：牛市 $y_t \sim \mathcal N(\mu_B, \sigma_B^2)$，熊市 $y_t \sim \mathcal N(\mu_S, \sigma_S^2)$；
- **状态转移** 用黏性矩阵 $A$：牛→牛、熊→熊的概率很高（0.95），意味着 regime 一旦确立就倾向于延续；
- **初值** $\pi$。

于是整个模型参数就是 $\theta = \{\mu, \sigma, A, \pi\}$。给定观测序列 $y_{1:T}$，我们想做两件事：(1) 估参数 $\theta$（EM）；(2) 算后验 $\gamma_t = P(z_t = \text{牛} \mid y_{1:T})$（前向-后向）。

## 二、生成合成数据（造数用，EM 不知道真值）

```python
import numpy as np
rng = np.random.default_rng(20260714)
T = 240
TRUE = dict(
    mu=np.array([0.007, -0.005]),   # 牛 +0.7%/月, 熊 -0.5%/月
    sig=np.array([0.035, 0.050]),   # 熊波动更大
    A=np.array([[0.95, 0.05], [0.05, 0.95]]),
    pi=np.array([0.5, 0.5]),
)
z = np.zeros(T, dtype=int); z[0] = rng.choice(2, p=TRUE["pi"])
for t in range(1, T):
    z[t] = rng.choice(2, p=TRUE["A"][z[t-1]])
y = np.array([rng.normal(TRUE["mu"][z[t]], TRUE["sig"][z[t]]) for t in range(T)])
```

注意真值里「熊市波动更大、漂移为负」是我们熟悉的现实特征。EM 只会看到 `y`，看不到 `z`。

## 三、Baum-Welch EM：从收益序列估计参数

EM 在 HMM 里有标准闭式解。E-step 算前向 $\alpha$、后向 $\beta$（用 log-sum-exp 防溢出），M-step 更新参数。下面是**可直接运行、与文末脚本同构**的紧凑实现：

```python
def baum_welch(y, K=2, n_iter=200, seed=7):
    r = np.random.default_rng(seed)
    T = len(y)
    mu = r.normal(y.mean(), y.std(), K)
    sig = np.full(K, y.std())
    A = np.full((K, K), 0.05) + np.eye(K) * 0.9
    A /= A.sum(1, keepdims=True)
    pi = np.full(K, 1.0 / K)
    from numpy import logaddexp
    for _ in range(n_iter):
        # E-step: 发射概率(对数) + 前向/后向
        logP = np.array([ -0.5*((y-mu[k])/sig[k])**2 - np.log(sig[k]*np.sqrt(2*np.pi))
                          for k in range(K)]).T
        la = np.full((T, K), -np.inf)
        la[0] = np.log(pi) + logP[0]
        for t in range(1, T):
            for j in range(K):
                la[t, j] = logP[t, j] + logaddexp.reduce(la[t-1] + np.log(A[:, j]))
        lb = np.zeros((T, K))
        for t in range(T-2, -1, -1):
            for i in range(K):
                lb[t, i] = logaddexp.reduce(lb[t+1] + np.log(A[i]) + logP[t+1])
        loglik = logaddexp.reduce(la[T-1])
        gamma = np.exp((la + lb) - loglik)
        xi = np.zeros((T-1, K, K))
        for t in range(T-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = np.exp(la[t, i] + np.log(A[i, j]) + logP[t+1, j] + lb[t+1, j] - loglik)
        # M-step
        pi = gamma[0] / gamma[0].sum()
        for j in range(K):
            mu[j] = (gamma[:, j] * y).sum() / gamma[:, j].sum()
            sig[j] = np.sqrt((gamma[:, j] * (y-mu[j])**2).sum() / gamma[:, j].sum())
        A = xi.sum(0) / gamma[:-1].sum(0)[:, None]
        A /= A.sum(1, keepdims=True)
    return mu, sig, A, pi

mu, sig, A, pi = baum_welch(y, K=2, n_iter=200)
# 对齐状态：mu 大的当牛市
order = np.argsort(mu); mu, sig = mu[order], sig[order]
A = A[order][:, order]; pi = pi[order]
# 估计结果：mu≈[−0.0024, +0.0096], sig≈[0.051, 0.040], 转移对角≈0.93/0.92
```

我们跑出来的估计与真值对得上方向：**牛市漂移为正、熊市为负；熊市波动（≈5.1%）大于牛市（≈4.0%）；转移矩阵对角线 ≈0.93**（真值 0.95，略低是因为样本里发生了真实的切换）。μ 的绝对数值有偏差（估计牛 +0.96%/月 vs 真 +0.7%），这是有限样本 + 状态重叠下的固有估计误差，文末陷阱会展开。

## 四、前向-后向：平滑后验 γ_t = P(牛 | 全部观测)

上面 EM 里已经算过 `gamma`，它就是我们要的后验。这里单独抽出，强调它的**「平滑」**属性——它用了**全部** 240 个月的观测，而不只是到第 t 月为止的数据：

```python
def forward_backward(y, mu, sig, A, pi):
    T, K = len(y), len(mu)
    logP = np.array([ -0.5*((y-mu[k])/sig[k])**2 - np.log(sig[k]*np.sqrt(2*np.pi))
                      for k in range(K)]).T
    from numpy import logaddexp
    la = np.full((T, K), -np.inf); la[0] = np.log(pi) + logP[0]
    for t in range(1, T):
        for j in range(K):
            la[t, j] = logP[t, j] + logaddexp.reduce(la[t-1] + np.log(A[:, j]))
    lb = np.zeros((T, K))
    for t in range(T-2, -1, -1):
        for i in range(K):
            lb[t, i] = logaddexp.reduce(lb[t+1] + np.log(A[i]) + logP[t+1])
    loglik = logaddexp.reduce(la[T-1])
    gamma = np.exp((la + lb) - loglik)
    return gamma, loglik

gamma, _ = forward_backward(y, mu, sig, A, pi)
p_bull = gamma[:, 1]          # P(牛市 | 全部观测)
```

图 1 里紫线就是 `p_bull`：它在真值牛市区间（绿柱）里爬升过 0.5，在真值熊市区间（红柱）里回落到 0.5 以下。关键点——**这是「回头看」的平滑概率，用到了未来的信息**。实盘里你只能用到截至当前的信息（filtered 概率，即前向 α 归一化），否则就是 look-ahead。文末陷阱 1 重点讲这个。

## 五、把概率变成仓位：γ > 0.5 满仓，否则空仓

最直接的择时规则：

```python
pos = (p_bull > 0.5).astype(float)     # 0/1 持仓
strat_ret = pos * y
bh_cum = np.cumprod(1 + y)
strat_cum = np.cumprod(1 + strat_ret)

def maxdd(cum):
    peak = np.maximum.accumulate(cum)
    return (cum / peak - 1).min()

# 买入持有: 年化 +3.49%, Sharpe 0.215, 最大回撤 -47.0%
# 贝叶斯择时: 年化 +8.07%, Sharpe 0.990, 最大回撤 -9.3%
# 全年平均 P(牛)=0.444, 持仓切换 12 次
```

结果读起来很诱人，但要分清它**到底赚了什么**：

| 指标 | 买入持有 | 贝叶斯择时 |
|---|---|---|
| 年化收益 | +3.49% | **+8.07%** |
| Sharpe | 0.215 | **0.99** |
| 最大回撤 | −47.0% | **−9.3%** |
| 平均仓位 | 100% | 44.4% |

择时本质不是「牛市多赚」（满仓时也只拿到市场收益），而是**「熊市少亏」**——一旦 `p_bull` 跌破 0.5 就空仓，避开了那段 −47% 的深跌。Sharpe 的飞跃主要来自回撤被砍掉、波动被压低，而非方向 alpha。

图 2 把两条累积净值画在一起：择时（蓝）在熊市段几乎走平（空仓），而买入持有（灰虚）一路下探。

![贝叶斯择时累积净值 vs 买入持有，回撤被大幅削减](/images/bayesian-regime-timing/timing_vs_buyhold.png)

## 六、局部放大：滤波如何跟进一次 regime 切换

图 3 放大了某次「熊→牛」切换点附近。可以看到真值状态在第 $t$ 月翻多（绿柱出现），后验概率 `p_bull` 在随后几期内被推过 0.5 阈值、仓位随之翻多。这正是 HMM 的实用价值：**它不要求你精准判断切换日，而是在概率越过阈值时「承认」新 regime 已经到来**，把择时噪声从硬切标签降到了概率软判。

![一次熊→牛切换点附近，后验概率的数期跟进](/images/bayesian-regime-timing/regime_switch_zoom.png)

## 七、六类真实陷阱（高阶必读）

**1. Look-ahead：平滑概率用了未来信息。** 图 1 的紫线是 smoothed γ_t（用全部 240 月），实盘只能用 filtered 概率（前向 α 归一化，只用到第 t 月）。用 smoothed 回测会虚高 Sharpe——这是本文最该警惕的点。落地必须改成「截至 t 月的前向概率」。

**2. 标签交换（label switching）。** EM 对状态没有固定语义，"状态 0/1" 每次随机初始化可能互换。我们靠 `mu` 排序对齐，但极端样本下可能判错牛熊，导致仓位反向。落地要加语义约束（如「漂移为正的是牛」）并多次初始化取稳。

**3. 参数非平稳。** 真实市场的 regime 参数会随时间漂移（牛市的波动结构在 2008 与 2020 完全不同）。固定全样本 EM 估计出的参数是「平均态」，对近期失效。应改用滚动窗口或在线 EM。

**4. 状态数假设的敏感性。** 我们硬设 K=2。真实市场可能是 3~4 个 regime（震荡/慢牛/快牛/崩盘）。K 选错会让后验失去经济含义。要用 BIC / 预测似然做模型选择，而不是拍脑袋。

**5. 转移矩阵过黏导致滞后。** 对角线 0.95 意味着状态极难切换，滤波会「反应慢半拍」——切换发生后好几期才翻仓，错过 early recovery 或晚逃顶。这是 HMM 择时的系统性滞后，必须在 Sharpe 里如实体现。

**6. 估计误差放大到仓位。** 第四节里 μ 的绝对估计有偏差（+0.96% vs +0.7%）。当牛市/熊市漂移接近、波动重叠时，后验会长期在 0.5 附近游走，仓位频繁切换、交易成本高企。必须加阈值缓冲带（如 0.4~0.6 之间保持原仓位）并计入成本。

## 八、落地的正确姿势

1. 用真实指数/ETF 日收益聚合到月，或直接从日收益建模（HMM 同样适用，只是状态更细）；
2. **实盘用 filtered 概率**（前向递归），绝不用 smoothed；
3. 多次 EM 初始化 + 语义对齐，确保「牛」漂移为正；
4. 滚动窗口重估参数，或上在线 EM 跟踪非平稳；
5. 仓位规则加缓冲带，并扣除交易成本后再评估 Sharpe；
6. 把它当「尾部保护 / 仓位旋钮」，而非独立 alpha 来源——它的价值是**把回撤砍下来**，不是把收益做上去。

HMM 给我们的最重要的东西，不是一条择时曲线，而是一个**可解释的「市场状态信念」**：当你不再问「现在是不是牛」，而是问「我现在有 73% 把握是牛」，你的风控和仓位才有了贝叶斯意义上的依据。

---

*本文为量化专栏第 N 篇。所有数值来自自洽合成，仅用于演示 HMM 估计与滤波机制；真实市场的 regime 边界模糊、参数非平稳，落地请以真实数据 + filtered 概率 + 成本建模为准。*
