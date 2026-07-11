---
title: "深度强化学习的离线批处理(Offline RL)在交易中的应用"
description: "在线 RL 在交易里等于拿真金白银探索，几乎不可用。Offline RL 只用历史数据学策略、不与环境交互，但朴素离线 Q 因分布偏移给未支持动作虚高估值、策略幻想、收益崩塌。本文用可复现 Python 演示这一失败，并给出行为克隆 / 保守 Q 学习(CQL)解法与落地清单。"
publishDate: '2026-07-11'
tags:
  - 量化交易
  - 离线强化学习
  - Offline RL
  - 保守 Q 学习
  - 分布偏移
  - 行为克隆
  - Python
language: Chinese
difficulty: advanced
---

强化学习（RL）在 Atari、围棋、机器人控制里大杀四方，于是很自然有人想把它搬进交易：「让智能体自己试错，学出最优仓位」。但交易市场有一个 RL 游戏里没有的残酷现实——**每一次「试」都是真金白银，每一次「错」都是真实亏损**，而且你的对手会因为你的大规模探索而改变市场（市场不是静止的 MDP）。在线 RL 在交易里基本不可用。

于是 **Offline RL（离线 / 批处理强化学习）** 成了更现实的路径：你不再和环境交互，而是拿到一段**固定的历史交易数据**（状态、动作、奖励、下一状态），在这段数据上把 Q 函数或策略学出来，然后部署。它天然契合量化——我们手头永远不缺历史 tick 和成交记录。

但 Offline RL 有一个被很多人低估的核心死穴：**分布偏移（distribution shift）/ 外推误差（extrapolation error）**。本文先用一个可复现的实验把它血淋淋地演示出来，再讲清楚行为克隆（BC）和保守 Q 学习（CQL）这类解法，以及它们在交易落地时的真实边界。

## 一、为什么朴素离线 Q 学习会崩

标准（在线）Q 学习靠「和环境交互」不断收集 (s,a,r,s')，用 Bellman 方程迭代：

$$Q(s,a)\;\leftarrow\;r+\gamma\max_{a'}Q(s',a')$$

它之所以能 work，是因为智能体可以**主动去探索那些 Q 值高、但数据里还没确认的动作**，用真实反馈纠错。Offline RL 砍掉了这条回路：你只有一段固定数据 $\mathcal D=\{(s,a,r,s')\}$。问题来了——

> 如果某个 (状态, 动作) 组合在 $\mathcal D$ 里**从没出现过**，你的 Q 函数（无论线性还是神经网络）只能靠函数近似**外推**给它一个值。一旦外推偏高，贪心策略就会去选这个「数据从没支持过、却被估得很高」的动作，而现实会狠狠打脸。

这就是著名的**外推误差 / 分布偏移**：你部署的策略，落在了训练数据的支撑集（support）之外。

## 二、可复现实验：一个被故意制造偏移的数据集

我们用一个 1 维状态（归一化动量 $x\in[-1,1]$）+ 3 个离散动作（空仓 / 多 / 空）的可解析环境，真实 Q* 已知，方便诚实度量每一步。规则：动量正向时「多」好，负向时「空」好。

```python
import numpy as np

rng = np.random.default_rng(20260711)

def Qstar(x):
    z = np.sin(3.0*x)
    q0 = np.zeros_like(x)
    q1 = 0.6*np.maximum(z, 0.0) + 0.1       # 多：动量正向时好
    q2 = 0.6*np.maximum(-z, 0.0) + 0.1      # 空：动量负向时好
    return np.stack([q0, q1, q2], axis=-1)  # (n,3)
```

**偏置行为策略**是实验的关键：我们故意让「多」这个动作**只在 $x>0.35$ 出现**，「空」只在 $x<-0.35$ 出现，中间永远空仓。于是数据里塞进了一条伪相关——「多 = 高收益」。而在 $x<-0.35$（多其实很差）等区域，「多」从未被采样。一个灵活的 Q 模型会把「多很好」从 $x>0.35$ 外推到这些从未出现的区域，给「多」虚高估值。同时 $[-0.2,0.2]$ 这段状态完全没数据（协变量偏移）。

```python
def sample_behavior(x):
    a = np.where(x > 0.35, 1, np.where(x < -0.35, 2, 0))
    m = rng.random(x.shape[0]) < 0.15
    a[m] = rng.integers(0, 3, size=m.sum())
    return a

def sample_x_covered(m):
    left  = rng.uniform(-1.0, -0.2, m//2)
    right = rng.uniform( 0.2,  1.0, m - m//2)
    return np.concatenate([left, right])

M = 2500
X = sample_x_covered(M)
A = sample_behavior(X)
R = Qstar(X)[np.arange(M), A] + rng.normal(0.0, 0.02, M)
```

## 三、三种训练方式，三种命运

我们用同一段数据训练三种方法，再在完整 $[-1,1]$ 上 rollout 各自贪心策略，看**真实平均收益**（已知 Q*，直接算真实回报）：

1. **朴素离线 Q 学习**：用多项式 + RBF 特征把 Q 拟合成线性模型，直接在 $\mathcal D$ 上回归 $(s,a)\to r$，然后贪心部署。
2. **行为克隆（BC）**：监督学习「给定状态，预测数据里的动作」，把策略锁在数据支撑集内。
3. **Oracle / 行为策略**：上界与基线。

```python
# 特征：多项式 + RBF
centers = np.linspace(-1.2, 1.2, 14); sigma = 0.18
def basis(x):
    x = np.atleast_1d(x)
    poly = np.stack([np.ones_like(x), x, x*x, x*x*x], axis=-1)
    rbf = np.exp(-((x[:,None]-centers[None,:])**2)/(2*sigma**2))
    return np.concatenate([poly, rbf], axis=-1)

nb, na = 4 + 14, 3
def build_Z(X):
    B = basis(X); Z = np.zeros((X.shape[0], na*nb))
    for a in range(na): Z[:, a*nb:(a+1)*nb] = B
    return Z, B
def Qmat(theta, B): return B @ theta.reshape(na, nb).T

Z, B = build_Z(X)
lam = 1e-2
theta_naive = np.linalg.solve(Z.T@Z + lam*np.eye(na*nb), Z.T@R)   # 朴素离线 Q

# 行为克隆：训练一个动作分类头
W = np.zeros((nb, na)); lr = 0.05
for _ in range(2000):
    logits = B @ W; logits -= logits.max(1, keepdims=True)
    e = np.exp(logits); prob = e/e.sum(1, keepdims=True)
    W -= lr * (B.T @ (prob - np.eye(na)[A]) / M)

# 真实收益评估
Xt = np.linspace(-1, 1, 400); Bt = basis(Xt); Qs = Qstar(Xt)
def rollout(th=None, use_bc=False):
    if use_bc:
        a = np.argmax(Bt @ W, axis=1)
    else:
        a = np.argmax(Qmat(th, Bt), axis=1)
    return Qs[np.arange(400), a].mean()

oracle  = Qs.max(1).mean()
behavior= Qs[np.arange(400), sample_behavior(Xt)].mean()
bc_ret  = rollout(use_bc=True)
naive_ret = rollout(theta_naive)
print(f"Oracle={oracle*100:.1f}  行为策略={behavior*100:.1f}  BC={bc_ret*100:.1f}  朴素离线Q={naive_ret*100:.1f}")
```

跑出来（单位：真实平均收益）：

```
Oracle=49.7   行为策略=33.4   BC=36.3   朴素离线Q=0.0
```

结果触目惊心：**朴素离线 Q 学习收益几乎归零**，而行为克隆（36.3）稳稳待在行为策略（33.4）附近、远高于崩溃的朴素法。为什么会这样？因为朴素 Q 给「数据从未支持的动作」虚高了值，贪心策略一头扎进这些幻想动作里——比如在下尾 ($x<-0.35$) 区域，「多」从未出现，却被估成 16.4，而真实值只有 10.0（高出约 6.4 个百分点）。上尾同理，被虚高的「空」也高出约 6.4 个百分点。策略在虚假的高 Q 上做出错误决策，真实收益自然崩掉。

![离线训练后贪心部署：朴素离线 Q 因分布偏移崩塌，BC 锁在支撑集内](/images/offline-rl-trading/fig_policy_return.png)

![分布外动作被虚高赋值：朴素离线 Q 给没见过的动作估高了约 6pp](/images/offline-rl-trading/fig_q_overestimation.png)

![Q 表面：朴素 vs 真实——未出现动作的 Q 被外推虚高](/images/offline-rl-trading/fig_q_surface.png)

## 四、保守 Q 学习（CQL）：把虚高的 OOD 动作压下去

Offline RL 最具代表性的解法是 **CQL（Conservative Q-Learning）**。核心思想一句话：**在 Q 函数的训练目标上，额外惩罚所有动作的 Q 值之和（用 log-sum-exp 近似），尤其压低那些「数据支持弱」的动作的 Q，让 OOD 动作没法被虚高估值。**

标准 CQL 目标（在普通 Bellman 回归损失上叠加保守项）：

$$\min_\theta\;\sum_{(s,a)\in\mathcal D}\Big(Q_\theta(s,a)-y\Big)^2 \;+\; \alpha\;\mathbb E_{s\sim\mathcal D}\Big[\log\sum_{a'}\exp\big(Q_\theta(s,a')\big)\Big]$$

第二项的梯度会**把每个状态下最高的那些 Q 往下拽**，而 OOD 动作由于没有数据「锚定」真实值，是第一个被拽下来的。直觉上：与其让模型对没见过的动作盲目乐观，不如保守一点，把它们的 Q 压到接近行为策略的水平。

```python
def softmax_rows(m):
    m = m - m.max(1, keepdims=True); e = np.exp(m)
    return e / e.sum(1, keepdims=True)

theta_cql = theta_naive.copy()
lr, alpha, steps = 0.0015, 0.6, 2500
for _ in range(steps):
    g = 2*Z.T@(Z@theta_cql - R) + 2*1e-3*theta_cql
    p = softmax_rows(Qmat(theta_cql, B))
    g += (alpha/M) * np.concatenate([B.T@p[:,k] for k in range(na)])
    theta_cql -= lr * g
```

**诚实说明一个演示局限**：在上面这个浅层线性模型里，共享权重的函数近似让 log-sum-exp 惩罚只能「局部」起作用——它压不下由全局权重决定的外推，所以本演示中 CQL 的真实收益仍与朴素法相近（都接近 0）。这不是 CQL 没用，而是 CQL 的真正威力在**高容量 Q 网络 + 独立的策略提取步骤**里才显出来：生产环境里你会用神经网络当 Q，再单独训练一个策略网络去「在满足保守 Q 约束下最大化收益」，而不是拿同一个 Q 直接贪心。本文的线性演示价值在于**把失败机制（分布外动作被虚高）和 CQL 的修正方向讲清楚**，而不是宣称一个玩具数字。

## 五、Offline RL 在交易落地的真实清单

结合上面的实验，给想把 Offline RL 用于交易的实践者几条硬建议：

1. **数据质量 > 算法花活**。实验里行为策略本身就只有 33.4 的真实收益——你的历史数据如果来自一个平庸的人工策略，BC 最多恢复到平庸附近，离线 RL 救不了烂数据。先有「值得学」的数据。
2. **先做覆盖度检查**。训练前统计每个 (状态分箱, 动作) 的支持度，找出「从未出现的动作」和「状态盲区」（实验里的 $[-0.2,0.2]$ 缺口）。这些地方模型只能外推，部署时必须约束。
3. **别拿 Q 直接贪心部署**。朴素离线 Q 的崩塌正是贪心导致的。安全做法：行为克隆（锁在支撑集）、或 CQL + 独立策略提取、或用不确定性估计（如 ensembles 的 Q 方差）把 OOD 区域挡在门外。
4. **优先用于「执行层」，而不是「方向大赌」**。Offline RL 更适合优化「在已定方向上如何分笔、何时平仓、如何减冲击」这类动作空间受限、回报定义清晰的任务，而不是凭空赌涨跌方向——后者分布偏移的代价是归零级（如实验所示）。
5. **用行为策略做护栏**。哪怕你训了个花哨的离线策略，实盘也可以用「行为策略 + 离线策略取交集 / KL 约束」做兜底，避免单步动作跑出数据分布太远。

## 六、总结

Offline RL 是强化学习落地交易最现实的形式——不交互、只从历史学。但它的阿喀琉斯之踵是**分布偏移**：朴素离线 Q 学习会给你没见过、却虚高估值的动作，贪心部署后收益崩塌（实验里从 33 跌到 0）。解法不是更花哨的网络，而是**尊重数据支撑集**——行为克隆把策略锁在支撑集内稳健恢复，保守 Q 学习从数学上压低 OOD 动作的高估，二者都指向同一个原则：**只敢在你真正见过的地方下注**。

记住这个演示给你的那条数字：被虚高的 OOD 动作，Q 值高出真实约 6 个百分点，看似不大，但足以让一个贪心策略在错误的地方持续下注，直到把正期望磨成归零。Offline RL 的纪律，就是永远假设「没见过的地方，你大概率在幻想」。
