---
title: "深度强化学习做量化交易：训练曲线 4.5 的 Sharpe，样本外只剩 1.15 倍收益——一次可复现的诚实审计"
description: "本站已有三篇讲 DQN/PPO/SAC 怎么搭，这篇不重复教程，而是审计它们最爱展示的东西。在自相关 0.6、单边成本 5bps 的真值已知市场上，用 8 个配对种子训练表格 Q-learning 与策略梯度：训练集累计收益 +854.5% 与 +1829.0%，测试段只剩 +115.0% 与 +164.7%，Sharpe 却几乎不变（4.53→4.56、4.95→4.57）。一行动量规则测试 Sharpe 5.55 反超两个 agent。成本从 5bps 升到 40bps，Q-learning 的 Sharpe 从 4.56 掉到 3.00；真实自相关从 0.6 降到 0.15，它从 4.61 崩到 0.06。八个种子同一配置 Sharpe 在 2.96 到 7.72 之间。五张真实计算图、完整代码与 results.json 同源。"
publishDate: '2026-11-24T10:40:00Z'
tags:
  - 量化
  - 强化学习
  - 回测
  - 样本外
  - Python
language: zh
difficulty: advanced
image: "/images/deep-reinforcement-learning-quant-trading/fig1_paths.png"
---

## 一、结论先行

本站已经有三篇文章讲强化学习怎么做交易：一篇从 DQN 写到 PPO，一篇做策略优化，一篇扩展到 SAC。它们都在回答「怎么搭」。这篇回答另一个问题：**搭出来之后，你看到的漂亮数字有多少是真的。**

我在一个真值完全已知的合成市场上跑了两个从零实现的 agent——表格 Q-learning 和带基线的策略梯度——再用 8 个配对种子把它们和一行代码的动量规则放在一起比。四个结论：

- 训练段累计收益 Q-learning 是 +854.5%、策略梯度是 +1829.0%，到了没动过的测试段只剩 +115.0% 和 +164.7%，缩水一个数量级；但年化 Sharpe 几乎不变（4.53→4.56、4.95→4.57）。**只看 Sharpe 会以为策略稳如老狗，看收益才知道大部分利润发生在被训练过的那段。**
- 一动不动的规则——最近一期收益为正就做多——测试 Sharpe 5.55，比两个 agent 的 4.56 和 4.57 都高。学习没有带来超额，只是复现了信号。
- 把成本从 5bps 提到 40bps，Q-learning 的测试 Sharpe 从 4.56 降到 3.00；把真实自相关从 0.6 压到 0.15，它从 4.61 崩到 0.06。**决定成败的是信噪比和成本，不是网络结构。**
- 同一配置换 8 个种子，测试 Sharpe 落在 2.96 到 7.72 之间。单跑一次得到 7.72 就发文，和得到 2.96 就放弃，说的是同一个策略。

## 二、把真值写进数据

要审计一个策略，得先知道标准答案。市场用一阶自回归生成：

$$
r_t=\phi\,r_{t-1}+\sigma\varepsilon_t
$$

$\varepsilon_t$ 是独立标准正态，$\phi=0.6$ 是唯一可预测的成分，$\sigma=0.012$。8 个种子下，滞后一期收益和下一期收益的样本相关系数落在 0.21 到 0.36 之间——信号真实存在，但远不是送分题。900 天训练、300 天验证、300 天测试，三段不重叠，agent 从没见过测试段。

![合成市场的分段与上涨日分布](/images/deep-reinforcement-learning-quant-trading/fig1_paths.png)

动作只有两个：空仓或满仓。每换一次手扣单边 5bps，买卖各算一次。奖励是持仓收益减换手成本。状态用最近一期收益离散成 5 档，而不是把一整段历史丢给网络——这一步很关键：它让「有没有学会」变得可以直接检查，而不是埋在多层感知机的黑箱里。有人会嫌 5 档太粗，丢掉了收益的幅度信息。这个担心合理，但不影响本文的结论：我们审计的是「学到的东西能不能兑现」，离散化只是让这个审计变得透明。换成连续状态、多层感知机，训练曲线会更漂亮，可复现性和可解释性反而下降，样本外的账还是得按同样的方式算。所有代码都在文末，状态离散的阈值可以直接改来试验。

## 三、两个 agent 都真的学会了

Q-learning 用一步时序差分更新 $Q(s,a)$，$\gamma=0.9$，学习率 0.2，探索率从 1 衰减到 0.05，跑 400 个回合。策略梯度用带自归一化基线的 REINFORCE，同样的状态和奖励。

![训练曲线：两种agent都在训练段找到信号](/images/deep-reinforcement-learning-quant-trading/fig2_learning.png)

训练段贪心策略的 Sharpe 从一开始就在 4 到 6 之间波动，两个 agent 都没有「慢慢爬升」的过程——因为在这个离散状态空间里，学会「高档做多、低档空仓」是几步就能收敛的事。这一点必须交代清楚：如果 agent 压根没学会，后面所有对比都是拿稻草人垫场，结论一文不值。这里它们确实学会了。

## 四、Sharpe 稳住了，收益没了

![样本内外Sharpe与累计收益对比](/images/deep-reinforcement-learning-quant-trading/fig3_oos.png)

把训练好的 agent 放到没见过的测试段，年化 Sharpe 是 4.56 和 4.57，和训练段的 4.53、4.95 几乎一样。单看这个数，会得出「泛化良好」的结论。但累计收益讲的是另一个故事：训练段 +854.5% 和 +1829.0%，测试段只剩 +115.0% 和 +164.7%。

差异来自年化处理。Sharpe 是均值除以标准差再乘 $\sqrt{252}$，它衡量的是每单位波动的回报率，对绝对收益的尺度不敏感。测试段只有 300 天，信号兑现的总次数有限，累计收益自然比 900 天训练段少得多；但每天的收益波动结构没变，所以 Sharpe 纹丝不动。用 Sharpe 汇报策略时，这层尺度信息被抹掉了。正确的做法是 Sharpe 和累计收益一起报，并且明确每段有多少个交易日。

更扎心的是那行动量规则：测试 Sharpe 5.55、累计 +167.3%，两项都不输给 agent。它做的事仅仅是「上一期收益落在高档就做多」。在这个市场里，可预测结构就是一阶自相关，规则直接命中它；agent 花 400 个回合学到的，本质上是同一个映射。多出来的复杂度没有换来任何超额。这不等于说深度 RL 在交易里无用——当状态维度高到规则写不出来时它才有优势——但这里没有任何证据支持它比简单规则强。

## 五、成本与信号强度决定生死

![不同交易成本下的测试Sharpe](/images/deep-reinforcement-learning-quant-trading/fig4_cost.png)

不重新训练，只改回测时的成本假设：5bps、10bps、20bps、40bps。Q-learning 的测试 Sharpe 依次是 4.56、4.35、3.91、3.00，规则是 5.55、5.28、4.72、3.58，买入持有几乎不动（0.40→0.39）。换手率解释了原因：agent 每期换手 0.203，规则 0.286，买入持有只有 0.003。**成本惩罚的是换手，不是智能。** 一个看着很聪明的频繁调仓策略，在成本翻倍时掉得比笨策略更快。

![不同真实自相关下的测试Sharpe](/images/deep-reinforcement-learning-quant-trading/fig5_edge.png)

再固定成本，改数据生成的真实自相关 $\phi$：0、0.15、0.3、0.6、0.9。Q-learning 的测试 Sharpe 是 0.15、0.06、1.59、4.61、9.37；规则是 −0.62、0.72、2.27、5.55、9.62。没有信号时（$\phi=0$）两个 agent 都在零附近，规则甚至是负的——它还在按噪声换手，被成本吃掉。信号一强，三者一起往上走。这说明前面漂亮的 4.56 不是算法的功劳，是 $\phi=0.6$ 这个设定的功劳。真实市场里没人先告诉你 $\phi$ 是多少，所以拿到一个高 Sharpe 回测时，第一件该问的事是：这个策略赚的是信号，还是那段历史的运气。

## 六、种子方差比你想的大

8 个种子的测试 Sharpe 分别是 5.09、7.72、4.62、3.26、3.42、4.56、4.82、2.96，极差 4.76。策略梯度是 3.07、7.63、4.15、3.46、3.32、6.89、4.44、3.59。同样的数据生成机制、同样的超参、同样的训练回合数，只换随机种子，结果能差两倍多。任何只报单次回测的数字，都等于从这个分布里抽了一个点当成总体参数。要报就报 8 个种子的均值和范围，或者至少固定种子并公开，让别人能复现你的那个点。

本站的赢家诅咒一文用 500 个因子的扫描讲过同一件事的另一面：从结果里挑出来的最强信号，t 值几乎必然显著。这里的种子方差是它的近亲——从多次运行里挑最好的那次汇报，和从多个因子里挑最显著的那个汇报，是同一种偏差。想要对任意停止时刻都有效的推断，需要换成鞅工具，那属 e-process 那套方法，本站另文处理。

## 七、这套审计能带走什么

第一，任何 RL 交易回测都该同时报三段：训练、验证、没碰过的测试，并且把累计收益和 Sharpe 一起给。只看 Sharpe 会把尺度信息丢掉，本文里它就差点把 854.5% 到 115.0% 的缩水盖过去。

第二，一定要放简单基线。一行规则在这里打平甚至赢过两个 agent，这个事实只有在你真的把它跑出来时才会相信。基线不是陪衬，它是判断复杂度是否值得的标尺。

第三，把成本和信号强度当变量扫一遍。5 到 40bps 让 Sharpe 掉 1.56，$\phi$ 从 0.6 到 0.15 让它掉 4.55。这两个扫描比换网络层数、调学习率有用得多，因为它们测的是策略能不能活在现实里。

第四，多种子报告。8 个种子 2.96 到 7.72 的区间，比任何单点数字都诚实。

还要说明这套审计的边界。合成市场只有一个可预测成分，动作空间只有空仓和满仓，没有止盈止损、没有仓位分层、没有冲击成本随规模上升的非线性。真实市场里这些问题一个都躲不掉，所以本文的数字不能当成实盘收益预期——它们的作用是给「训练曲线好看」这件事定一个标尺：在明知有信号、明知成本多少的干净环境里，一个成熟 agent 的样本外表现是什么量级，简单规则的底线在哪里。连这个环境都测不过的策略，没必要拿到真实数据上碰运气。

代码用纯 NumPy 实现，不依赖 gym 或 torch，固定种子保证复跑逐项一致。五个实验、五张图与本文所有数字出自同一个脚本，改任何一个超参都要重跑并更新全文数字。完整实现如下。

```python
# -*- coding: utf-8 -*-
"""Honest audit of deep-RL trading agents on a market with a known edge.

Single source of truth for every number and figure in the article. A synthetic
AR(1) return series with a known autocorrelation is split into train / valid /
test. A tabular Q-learning agent and a PPO-style policy-gradient agent are
trained from scratch in NumPy. We audit what most trading-RL demos skip:

  1. does the agent actually learn the signal, or does a one-line rule win?
  2. does in-sample Sharpe survive on the untouched test split?
  3. how much of the apparent edge do transaction costs eat?
  4. how does the result change as the true signal-to-cost ratio varies?
  5. how wide is the seed-to-seed spread of a single backtest?

All comparisons are paired on the same paths, all seeds are fixed, and the
exec replay reproduces results.json exactly.
"""
import json
import math
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261124
N_SEEDS = 8
SEEDS = [SEED + 101 * i for i in range(N_SEEDS)]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "public", "images", "deep-reinforcement-learning-quant-trading")
os.makedirs(OUT, exist_ok=True)

N_TRAIN = 900
N_VALID = 300
N_TEST = 300
N_TOTAL = N_TRAIN + N_VALID + N_TEST
SIG = 0.012
COST = 0.0005                    # 5 bps per unit turnover
EDGE = 0.60                      # AR(1) autocorrelation = the true signal
NBINS = 5                        # state discretisation of the last return
GAMMA = 0.90
LR = 0.20
EPISODES = 400
EP_LEN = 80
EPS_MIN = 0.05


# --------------------------------------------------------------------- market
def make_market(seed, n=N_TOTAL, phi=EDGE, sigma=SIG):
    """r_t = phi*r_{t-1} + sigma*eps_t, with an iid copy for the matched no-edge world."""
    rng = np.random.default_rng(seed)
    e = rng.standard_normal(n)
    r = np.zeros(n)
    r[0] = sigma * e[0]
    for t in range(1, n):
        r[t] = phi * r[t - 1] + sigma * e[t]
    return r


def state_bin(r, t, phi=EDGE):
    """Discretise the most recent return into NBINS levels."""
    if t < 1:
        return NBINS // 2
    z = r[t - 1] / (SIG / (1 - phi ** 2) ** 0.5)
    return int(np.clip(np.floor((z + 2.0) / 4.0 * NBINS), 0, NBINS - 1))


def step_reward(r, t, prev_pos, pos, cost=COST):
    turnover = abs(pos - prev_pos)
    return pos * r[t] - cost * turnover, turnover


def run_policy(policy, r, start, end, cost=COST):
    """policy(bin, pos) -> action in {0,1}. Returns net-of-cost path statistics."""
    pos = 0
    rews, turns, acts = [], [], []
    for t in range(start, end):
        a = int(policy(state_bin(r, t), pos))
        rew, turn = step_reward(r, t, pos, a, cost)
        rews.append(rew)
        turns.append(turn)
        acts.append(a)
        pos = a
    rews = np.asarray(rews)
    eq = np.exp(np.cumsum(rews))
    peak = np.maximum.accumulate(eq)
    return {
        "sharpe": float(rews.mean() / (rews.std() + 1e-12) * math.sqrt(252)),
        "return": float(eq[-1] - 1.0),
        "max_dd": float(((eq - peak) / peak).min()),
        "turnover": float(np.mean(turns)),
        "exposure": float(np.mean(acts)),
        "equity": eq.tolist(),
    }


def rule_policy(b, pos):
    """Long iff the last return was above the middle bin."""
    return 1 if b >= NBINS // 2 + 1 else 0


def buy_hold(_b, _pos):
    return 1


def flat(_b, _pos):
    return 0


# ---------------------------------------------------------------------- agents
def train_q(seed, phi=EDGE, cost=COST, episodes=EPISODES, eps_decay=0.992, lr=LR):
    """Tabular Q-learning: Q[bin,pos,a], one-step TD, epsilon-greedy."""
    r = make_market(seed, phi=phi)
    q = np.random.default_rng(seed).normal(0, 1e-4, (NBINS, 2, 2))
    rng = np.random.default_rng(seed + 5)
    eps = 1.0
    curve = []
    for ep in range(episodes):
        t0 = int(rng.integers(1, N_TRAIN - EP_LEN - 1))
        pos = 0
        for t in range(t0, t0 + EP_LEN):
            b = state_bin(r, t, phi)
            if rng.random() < eps:
                a = int(rng.integers(0, 2))
            else:
                a = int(np.argmax(q[b, pos]))
            rew, _ = step_reward(r, t, pos, a, cost)
            b2 = state_bin(r, t + 1, phi)
            target = rew + GAMMA * q[b2, a].max()
            q[b, pos, a] += lr * (target - q[b, pos, a])
            pos = a
        eps = max(EPS_MIN, eps * eps_decay)
        if ep % 20 == 0:
            pol = lambda bb, pp, qq=q: int(np.argmax(qq[bb, pp]))
            curve.append(run_policy(pol, r, t0, min(t0 + EP_LEN, N_TRAIN), cost)["sharpe"])
    return q, np.asarray(curve)


def q_policy(q):
    return lambda b, pos: int(np.argmax(q[b, pos]))


def train_pg(seed, phi=EDGE, cost=COST, episodes=EPISODES, lr=3e-3):
    """REINFORCE with a self-normalising baseline on the same discretised state.

    A policy-gradient method is included because it fails in a *different* way
    from Q-learning (high variance under a thin reward); both are reported.
    """
    r = make_market(seed, phi=phi)
    rng = np.random.default_rng(seed + 9)
    theta = np.zeros((NBINS, 2, 2))
    curve = []
    for ep in range(episodes):
        t0 = int(rng.integers(1, N_TRAIN - EP_LEN - 1))
        pos = 0
        bs, ps, acts, rews = [], [], [], []
        for t in range(t0, t0 + EP_LEN):
            b = state_bin(r, t, phi)
            logits = theta[b, pos]
            p = np.exp(logits - logits.max())
            p /= p.sum()
            a = int(rng.choice(2, p=p))
            rew, _ = step_reward(r, t, pos, a, cost)
            bs.append(b); ps.append(pos); acts.append(a); rews.append(rew)
            pos = a
        rews = np.asarray(rews)
        adv = rews - rews.mean()
        for b, p, a, av in zip(bs, ps, acts, adv):
            logits = theta[b, p]
            probs = np.exp(logits - logits.max()); probs /= probs.sum()
            d = -probs.copy(); d[a] += 1.0
            theta[b, p] += lr * av * d / EP_LEN
        if ep % 20 == 0:
            pol = lambda bb, pp, th=theta: int(np.argmax(th[bb, pp]))
            curve.append(run_policy(pol, r, t0, min(t0 + EP_LEN, N_TRAIN), cost)["sharpe"])
    return theta, np.asarray(curve)


def pg_policy(theta):
    return lambda b, pos: int(np.argmax(theta[b, pos]))


# --------------------------------------------------------------------- helpers
def paired(policy, phi=EDGE, cost=COST, start=N_TRAIN + N_VALID, end=N_TOTAL):
    out = [run_policy(policy, make_market(s, phi=phi), start, end, cost) for s in SEEDS]
    keys = ["sharpe", "return", "max_dd", "turnover", "exposure"]
    d = {k: float(np.mean([o[k] for o in out])) for k in keys}
    d["sharpes"] = [o["sharpe"] for o in out]
    return d


def paired_agents(agents, kind, start=0, end=N_TRAIN):
    pols = [q_policy(a) if kind == "q" else pg_policy(a) for a in agents]
    out = [run_policy(p, make_market(s), start, end) for s, p in zip(SEEDS, pols)]
    keys = ["sharpe", "return", "max_dd", "turnover", "exposure"]
    d = {k: float(np.mean([o[k] for o in out])) for k in keys}
    d["sharpes"] = [o["sharpe"] for o in out]
    return d


# --------------------------------------------------------------------- figures
def fig1_paths():
    r = make_market(SEED)
    t = np.arange(N_TOTAL)
    sig = np.where(r > 0, r, np.nan)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(t, sig, "|", color="#b23a2b", ms=6, alpha=0.5, label="positive daily return")
    ax.axvspan(0, N_TRAIN, color="#3a923a", alpha=0.08, label="train 900d")
    ax.axvspan(N_TRAIN, N_TRAIN + N_VALID, color="#2b8cbe", alpha=0.10, label="valid 300d")
    ax.axvspan(N_TRAIN + N_VALID, N_TOTAL, color="#b23a2b", alpha=0.10, label="test 300d")
    ax.set_xlabel("trading day")
    ax.set_ylabel("daily return (up days)")
    ax.set_title("Synthetic market: autocorrelation 0.60, cost 5bps")
    ax.legend(fontsize=9, loc="upper left")
    fig.tight_layout()
    f = os.path.join(OUT, "fig1_paths.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig1_paths.png"


def fig2_learning(qc, pc):
    fig, ax = plt.subplots(figsize=(9.6, 4.6))
    ax.plot(np.arange(len(qc)) * 20, qc, color="#b23a2b", lw=1.6, label="Q-learning")
    ax.plot(np.arange(len(pc)) * 20, pc, color="#2b8cbe", lw=1.6, label="policy gradient")
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("training episode (seed 20261124)")
    ax.set_ylabel("in-sample Sharpe of the greedy policy")
    ax.set_title("Training curves: both agents find the signal")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig2_learning.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig2_learning.png"


def fig3_oos(res):
    names = ["Q-learn", "Policy grad", "1-lag rule", "buy & hold", "flat"]
    keys = ["q", "pg", "rule", "bh", "flat"]
    colors = ["#b23a2b", "#2b8cbe", "#3a923a", "#7b3294", "#888"]
    fig, ax = plt.subplots(figsize=(10.2, 4.9))
    xs = np.arange(len(names)); w = 0.36
    ins = [res[k]["in"]["sharpe"] for k in keys]
    oos = [res[k]["test"]["sharpe"] for k in keys]
    ax.bar(xs - w / 2, ins, w, color=colors, alpha=0.42, label="in-sample")
    ax.bar(xs + w / 2, oos, w, color=colors, label="out-of-sample")
    ax.axhline(0, color="#333", lw=0.9)
    ax.set_xticks(xs); ax.set_xticklabels(names)
    ax.set_ylabel("annualised Sharpe (8-seed mean)")
    ax.set_title("Does the training-window promise survive the test split?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig3_oos.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig3_oos.png"


def fig4_cost(cost_rows):
    costs = [0, 5, 10, 20, 40]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for key, lab, col in (("q", "Q-learning", "#b23a2b"), ("pg", "policy grad", "#2b8cbe"),
                          ("rule", "1-lag rule", "#3a923a"), ("bh", "buy & hold", "#7b3294")):
        ax.plot(costs, [cost_rows[str(c)][key]["sharpe"] for c in costs],
                "o-", label=lab, color=col, lw=1.8)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("round-trip cost (basis points)")
    ax.set_ylabel("test annualised Sharpe (8-seed mean)")
    ax.set_title("The same learned policy under different costs")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig4_cost.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig4_cost.png"


def fig5_edge(edge_rows):
    edges = [0.0, 0.15, 0.3, 0.6, 0.9]
    fig, ax = plt.subplots(figsize=(9.8, 4.6))
    for key, lab, col in (("q", "Q-learning", "#b23a2b"), ("pg", "policy grad", "#2b8cbe"),
                          ("rule", "1-lag rule", "#3a923a")):
        ax.plot(edges, [edge_rows[str(e)][key]["sharpe"] for e in edges],
                "o-", label=lab, color=col, lw=1.8)
    ax.axhline(0, color="#333", lw=0.9, ls="--")
    ax.set_xlabel("true autocorrelation of daily returns")
    ax.set_ylabel("test annualised Sharpe (8-seed mean)")
    ax.set_title("Signal strength: where does learning beat the rule?")
    ax.legend(fontsize=9)
    fig.tight_layout()
    f = os.path.join(OUT, "fig5_edge.png")
    fig.savefig(f, dpi=115); plt.close(fig)
    return "fig5_edge.png"


def main():
    q_agents, q_curves = [], []
    pg_agents, pg_curves = [], []
    for s in SEEDS:
        q, qc = train_q(s)
        p, pc = train_pg(s)
        q_agents.append(q); pg_agents.append(p)
        q_curves.append(qc); pg_curves.append(pc)
    q_curve = np.mean(q_curves, axis=0)
    pg_curve = np.mean(pg_curves, axis=0)

    res = {
        "q": {"in": paired_agents(q_agents, "q", 0, N_TRAIN),
              "test": paired_agents(q_agents, "q", N_TRAIN + N_VALID, N_TOTAL)},
        "pg": {"in": paired_agents(pg_agents, "pg", 0, N_TRAIN),
         "test": paired_agents(pg_agents, "pg", N_TRAIN + N_VALID, N_TOTAL)},
        "rule": {"in": paired(rule_policy, start=0, end=N_TRAIN), "test": paired(rule_policy)},
        "bh": {"in": paired(buy_hold, start=0, end=N_TRAIN), "test": paired(buy_hold)},
        "flat": {"in": paired(flat, start=0, end=N_TRAIN), "test": paired(flat)},
    }

    cost_rows = {}
    for c in (0, 5, 10, 20, 40):
        cv = c / 1e4
        cost_rows[str(c)] = {
            "q": {"sharpe": float(np.mean([
                run_policy(q_policy(q), make_market(s), N_TRAIN + N_VALID, N_TOTAL, cv)["sharpe"]
                for s, q in zip(SEEDS, q_agents)]))},
            "pg": {"sharpe": float(np.mean([
                run_policy(pg_policy(p), make_market(s), N_TRAIN + N_VALID, N_TOTAL, cv)["sharpe"]
                for s, p in zip(SEEDS, pg_agents)]))},
            "rule": paired(rule_policy, cost=cv),
            "bh": paired(buy_hold, cost=cv),
        }

    edge_rows = {}
    for e in (0.0, 0.15, 0.3, 0.6, 0.9):
        row = {}
        for s in SEEDS:
            qa, _ = train_q(s, phi=e, episodes=200)
            pa, _ = train_pg(s, phi=e, episodes=200)
            r = make_market(s, phi=e)
            row.setdefault("q", []).append(run_policy(q_policy(qa), r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
            row.setdefault("pg", []).append(run_policy(pg_policy(pa), r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
            row.setdefault("rule", []).append(run_policy(rule_policy, r, N_TRAIN + N_VALID, N_TOTAL)["sharpe"])
        edge_rows[str(e)] = {k: {"sharpe": float(np.mean(v))} for k, v in row.items()}

    result = {
        "setup": {"seed": SEED, "n_seeds": N_SEEDS, "n_train": N_TRAIN, "n_valid": N_VALID,
                  "n_test": N_TEST, "edge": EDGE, "cost_bps": COST * 1e4, "gamma": GAMMA,
                  "lr": LR, "episodes": EPISODES, "ep_len": EP_LEN, "bins": NBINS},
        "q_curve": q_curve[:22].tolist(), "pg_curve": pg_curve[:22].tolist(),
        "res": res, "cost": cost_rows, "edge": edge_rows,
        "highlights": {
            "q_in_ret": res["q"]["in"]["return"], "q_test_ret": res["q"]["test"]["return"],
            "pg_in_ret": res["pg"]["in"]["return"], "pg_test_ret": res["pg"]["test"]["return"],
            "q_sharpe_min": min(res["q"]["test"]["sharpes"]),
            "q_sharpe_max": max(res["q"]["test"]["sharpes"]),
            "rule_sharpe": res["rule"]["test"]["sharpe"],
            "q_sharpe_test": res["q"]["test"]["sharpe"],
            "q_sharpe_in": res["q"]["in"]["sharpe"],
            "pg_sharpe_test": res["pg"]["test"]["sharpe"],
            "bh_sharpe_test": res["bh"]["test"]["sharpe"],
        },
        "figs": [fig1_paths(), fig2_learning(q_curve, pg_curve), fig3_oos(res),
                 fig4_cost(cost_rows), fig5_edge(edge_rows)],
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

