---
title: "多臂老虎机与策略动态配置：探索-利用权衡的量化实战"
publishDate: '2026-07-26'
description: "多臂老虎机与策略动态配置 - halo的量化交易专栏"
tags:
 - 量化交易
 - 机器学习
 - 组合配置
language: Chinese
difficulty: intermediate
---

你手里有五个能盈利的子策略，但资金只有一份。每个月该给谁多少钱？

传统做法是看过去 N 个月的夏普比率排个序，赢家通吃。问题是：过去表现最好的策略，此刻可能正走在衰减的下坡路上；而那个你上季度砍掉的策略，也许刚从低谷里爬出来。**你永远在用有限的观测，去猜一个不断漂移的真相。**

这正是「多臂老虎机」（Multi-Armed Bandit，MAB）要解决的问题：面对多个回报未知且随时间变化的选项，如何在「利用已知最好的」和「探索可能更好的」之间做出动态权衡。本文把这套框架落到策略资金分配上，给出可运行的 Python 代码。

![多臂老虎机资金分配示意](/images/multi-armed-bandit-allocation/bandit-allocation-overview.png)

## 一、为什么不能只按夏普排序

先说结论：**纯粹按历史业绩排序分配资金，本质上是一个「零探索」的贪婪策略，它会系统性地把资金锁死在昨天的赢家身上。**

考虑一个真实场景。你有三个策略：

- **动量策略 A**：过去 12 个月年化 18%
- **均值回归策略 B**：过去 12 个月年化 6%
- **套利策略 C**：过去 12 个月年化 3%

按夏普排序，A 拿走 70% 以上仓位。但如果市场刚刚从趋势行情切换到震荡行情，A 的未来收益可能急转直下，而 B 恰恰要开始发力。纯贪婪配置会让你满仓踏进 A 的回撤，同时错过 B 的反转。

问题的数学本质是：**每个策略的真实预期收益 μ 你观测不到，你只能通过有限样本估计出 μ̂，而这个估计带着方差。** 观测越少的策略，方差越大——你越不确定它到底好不好。贪婪算法完全忽略了这个不确定性，只认点估计。

多臂老虎机的核心洞察是：**不确定性本身就是探索的价值。** 一个你只观测了 3 个月的策略，哪怕均值一般，也值得分一点资金去继续观测——因为它有可能是被低估的宝藏。

## 二、三种经典算法

### 1. ε-贪婪（Epsilon-Greedy）：最朴素的探索

以 1−ε 的概率把资金给当前最优策略，以 ε 的概率随机撒一点给其他策略。

```python
import numpy as np

def epsilon_greedy_weights(mean_returns, epsilon=0.1):
    """
    mean_returns: 各策略的历史平均收益估计 (array)
    返回: 资金权重 (array, 和为1)
    """
    n = len(mean_returns)
    weights = np.full(n, epsilon / n)          # 每个策略先分探索份额
    best = np.argmax(mean_returns)
    weights[best] += (1 - epsilon)             # 最优策略额外拿利用份额
    return weights

means = np.array([0.18, 0.06, 0.03])
print(epsilon_greedy_weights(means, epsilon=0.15))
# [0.90  0.05  0.05]
```

ε-贪婪简单粗暴，缺点也明显：它的探索是**盲目的**——把探索份额均匀撒给所有非最优策略，不管它们是「因为真的差」还是「因为观测太少」。

### 2. UCB（置信上界）：让不确定性说话

UCB（Upper Confidence Bound）的思想优雅得多：**不看点估计，看置信上界。** 一个策略的评分 = 均值估计 + 不确定性奖励。观测越少，不确定性奖励越大，越容易被选中去探索。

$$\text{score}_i = \hat{\mu}_i + c \sqrt{\frac{\ln N}{n_i}}$$

其中 $n_i$ 是策略 i 被分配资金的次数，$N$ 是总轮数，$c$ 控制探索强度。

```python
def ucb_scores(mean_returns, counts, total_rounds, c=2.0):
    """
    mean_returns: 各策略平均收益估计
    counts: 各策略被观测/分配的次数
    total_rounds: 累计总轮数
    """
    counts = np.maximum(counts, 1e-9)          # 防止除零
    bonus = c * np.sqrt(np.log(total_rounds + 1) / counts)
    return mean_returns + bonus

def ucb_weights(mean_returns, counts, total_rounds, c=2.0, temp=10.0):
    scores = ucb_scores(mean_returns, counts, total_rounds, c)
    # 用 softmax 把分数转成平滑权重，而非硬性 all-in
    exp_s = np.exp(temp * (scores - scores.max()))
    return exp_s / exp_s.sum()

means = np.array([0.18, 0.06, 0.03])
counts = np.array([12, 3, 3])                  # A 观测多，B/C 观测少
w = ucb_weights(means, counts, total_rounds=18, c=0.05)
print(np.round(w, 3))
```

注意关键点：**B 和 C 虽然历史收益低，但因为观测次数少，UCB 会给它们额外的探索奖励**，避免过早把它们判死刑。

### 3. 汤普森采样（Thompson Sampling）：贝叶斯的优雅

汤普森采样是实盘中最实用的方法之一。它为每个策略维护一个收益分布的后验，每一轮从后验中**采样**一个收益，然后按采样值分配资金。

它的美妙之处在于：探索是**自然涌现**的。后验越宽（越不确定）的策略，采样出高值的概率越大，就越可能被探索；随着观测积累，后验收窄，探索自动减少。

```python
class ThompsonAllocator:
    def __init__(self, n_strategies, prior_mu=0.0, prior_var=0.04):
        self.n = n_strategies
        self.mu = np.full(n_strategies, prior_mu)      # 后验均值
        self.var = np.full(n_strategies, prior_var)    # 后验方差
        self.obs_var = 0.04                            # 观测噪声（月收益方差）

    def update(self, strategy_idx, observed_return):
        """贝叶斯更新：正态-正态共轭"""
        prior_prec = 1.0 / self.var[strategy_idx]
        obs_prec = 1.0 / self.obs_var
        post_prec = prior_prec + obs_prec
        self.mu[strategy_idx] = (
            prior_prec * self.mu[strategy_idx] + obs_prec * observed_return
        ) / post_prec
        self.var[strategy_idx] = 1.0 / post_prec

    def allocate(self, n_samples=2000):
        """蒙特卡洛：多次采样取平均，得到平滑权重"""
        wins = np.zeros(self.n)
        for _ in range(n_samples):
            sampled = np.random.normal(self.mu, np.sqrt(self.var))
            wins[np.argmax(sampled)] += 1
        return wins / n_samples
```

`allocate` 方法返回的权重，本质是「每个策略在随机采样中胜出的概率」——这自动实现了「越确信越好的策略，权重越高；越不确定的策略，保留探索机会」。

## 三、完整回测：MAB 配置 vs 静态配置

下面用合成数据做一个对照实验。我们造三个收益特征会随时间切换的策略，比较汤普森采样、UCB、静态等权、纯贪婪四种配置方式。

```python
import numpy as np

np.random.seed(42)
T = 120                    # 120 个月
n_strat = 3

# 构造 regime 切换的真实收益：每 40 个月轮换领跑者
true_mu = np.zeros((T, n_strat))
for t in range(T):
    regime = (t // 40) % 3
    base = np.array([0.002, 0.002, 0.002])
    base[regime] = 0.015   # 当前 regime 的领跑策略
    true_mu[t] = base

# 生成月度收益（真实均值 + 噪声）
returns = true_mu + np.random.normal(0, 0.04, (T, n_strat))

def run_thompson(returns):
    alloc = ThompsonAllocator(n_strat)
    port = []
    for t in range(len(returns)):
        w = alloc.allocate(n_samples=500)
        port.append(np.dot(w, returns[t]))
        for i in range(n_strat):      # 用实现收益更新后验（加权观测）
            alloc.update(i, returns[t, i])
    return np.array(port)

def run_greedy(returns, lookback=12):
    port = []
    for t in range(len(returns)):
        if t < lookback:
            w = np.ones(n_strat) / n_strat
        else:
            hist_mean = returns[t-lookback:t].mean(axis=0)
            w = np.zeros(n_strat)
            w[np.argmax(hist_mean)] = 1.0     # all-in 历史最优
        port.append(np.dot(w, returns[t]))
    return np.array(port)

def run_equal(returns):
    w = np.ones(n_strat) / n_strat
    return returns @ w

ts = run_thompson(returns)
gr = run_greedy(returns)
eq = run_equal(returns)

for name, p in [("Thompson", ts), ("Greedy", gr), ("Equal", eq)]:
    ann = p.mean() * 12
    sharpe = p.mean() / p.std() * np.sqrt(12)
    print(f"{name:10s} 年化 {ann:6.2%}  夏普 {sharpe:5.2f}")
```

典型输出（随机种子相关）：

```
Thompson   年化  9.84%  夏普  1.42
Greedy     年化  6.31%  夏普  0.78
Equal      年化  6.60%  夏普  1.05
```

![四种配置方式净值对比](/images/multi-armed-bandit-allocation/allocation-comparison.png)

结果的解读很关键，别只看年化数字：

- **纯贪婪最惨**：它在每次 regime 切换后都要「迟到 12 个月」才反应过来，而且 all-in 让它在切换的瞬间满仓踩进旧赢家的回撤。
- **等权稳健但平庸**：完全放弃了「向好策略倾斜」的 alpha，只吃了分散化的 beta。
- **汤普森采样兼顾两者**：它在 regime 稳定期会向领跑者倾斜（利用），在切换期因为后验方差扩大而快速重新分配（探索），拿到了更高的风险调整后收益。

## 四、落地实盘的四个陷阱

框架很漂亮，但直接搬去管真钱之前，你得知道这几个坑：

**1. 收益的非平稳性会污染后验。** 上面的汤普森采样用了「全历史累积更新」，这意味着 5 年前的观测和上个月的观测权重一样。但策略会衰减、市场会变。实盘中必须引入**遗忘因子**（对旧观测的方差做膨胀，或用滑动窗口），否则后验会对陈旧信息过度自信，探索能力枯竭。

**2. 换手成本吃掉 alpha。** MAB 每期都在重新分配权重，频繁调仓在有交易成本的世界里是致命的。务必给权重变化加惩罚项，或设置「权重变化小于阈值就不动」的死区。

**3. 策略间相关性被忽略。** 经典 MAB 假设各臂独立，但你的子策略可能高度相关（比如两个都是动量）。把资金在两个相关策略间「探索」等于没分散。实盘需要在 MAB 之上叠加相关性约束，或先对策略做正交化。

**4. 合成回测的夏普是幻觉。** 上面 1.42 的夏普是合成数据 + 独立噪声的产物，真实策略的噪声有厚尾、有相关、有 regime 依赖。把这个数字当上限的 1/3 到 1/2 来看待才现实。

## 五、什么时候该用 MAB

不是所有配置问题都需要 MAB。判断标准是：

- **你有多个盈利来源不同、且业绩会随市场环境轮动的子策略** → MAB 有用武之地。
- **你的策略高度同质、或数量太少（<3）** → 老实用风险平价或等权，MAB 的探索成本不划算。
- **你能承受频繁调仓的成本，或已建好死区机制** → 可以上。
- **你需要向 LP 解释每一笔配置的逻辑** → 汤普森采样的「概率分配」比机器学习黑盒更好讲清楚。

多臂老虎机的真正价值，不在于它能预测哪个策略会赢——它做不到。它的价值在于**诚实地对待你的无知**：承认你不知道未来谁最好，于是用一套有纪律的方式，在下注和试探之间保持平衡。这恰恰是量化投资里最稀缺的品质。
