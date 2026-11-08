---
title: "非平稳 Thompson Sampling：共轭后验越学越窄，策略配置越换不动"
description: "Beta-Bernoulli 共轭后验的伪计数每观测一次就加一，采样噪声按 1/sqrt(n) 衰减，Thompson Sampling 在平稳世界里叫收敛，在区制切换的世界里叫冻结。本文用同一条 1200 天收益路径对拍四种算法：标准 TS 的后验浓度从 10 涨到 1210、主导臂后验标准差从 0.0300 掉到 0.0167，三次变点后 100 天内选对新最优臂的比例只有 0.00 / 0.03 / 0.00，总遗憾 383.6；折扣 TS（γ=0.99）把浓度锁在 110，遗憾降到 169.7；滑窗 TS 161.6；静态 UCB1 反而最低 101.3。折扣因子扫描呈 U 型，γ=0.90 与 γ=1.0 几乎一样差。附五张真实计算图、区制长度扫描与完整可运行代码。"
publishDate: '2026-11-08T09:20:00Z'
tags:
  - 量化
  - 交易
  - 强化学习
  - Python
language: zh
difficulty: advanced
---

## 一、收敛是优点，也是陷阱

假设手上有 5 个候选策略，每天开盘只能挑一个上场，收盘记一个二值结果：跑赢基准记 1，否则记 0。这是最朴素的多臂老虎机，Thompson Sampling（TS）也是这类问题里最省心的解法——给每个臂挂一个 Beta 后验，每天从各臂后验里抽一个数，选最大的那个，用观测结果更新后验。它不需要调探索率，理论保证也干净。

问题出在「收敛」二字上。Beta 后验的浓度 $\alpha+\beta$ 每观测一次就加一，采样噪声按 $1/\sqrt{\alpha+\beta}$ 衰减。跑得越久，抽出来的 $\theta$ 越贴近后验均值，TS 就越像贪心。在概率恒定不变的世界里这叫收敛，正是我们想要的；可市场区制会切换，昨天胜率 0.60 的策略今天可能只剩 0.10。**一个积累了上千次伪计数的臂，要把它已经过时的高均值拉下来，需要的连续失败次数与它积累的样本量同量级**——而新区制根本不会给你那么久。

本站上一篇 [贝叶斯在线变点检测](/blog/bayesian-online-changepoint-run-length/) 讨论的重点是「怎么尽早发现分布变了」。这篇换个角度：假设你不装变点检测器，只靠后验自己适应，代价具体有多大？以及给后验人为装一个「记忆上限」要付什么价。

## 二、同一条世界路径上的四个算法

实验设定 1200 个交易日、5 个臂，在第 300、600、900 天切换区制，四段里跑赢基准的概率向量依次把最优臂放在 2、0、3、1 号，最优概率 0.60 / 0.55 / 0.62 / 0.60，其余臂落在 0.10～0.15。也就是说每次切换都会把上一轮的冠军打回平庸，且新冠军此前长期坐冷板凳。

![](/images/thompson-sampling-nonstationary-posterior-collapse/fig1_regime_dgp.png)

图 1 是这四段概率的阶梯图。为了公平，先生成一张 2400×5 的均匀随机数表，所有算法、所有扫描共用同一张表，因此第 $t$ 天第 $k$ 个臂「本来会不会成功」是固定的，只有被选中时才揭晓；四个算法的采样随机流也统一用同一个种子。伪遗憾用真实概率矩阵逐日重算 $\sum_t(p_{\max}(t)-p_{\text{chosen}}(t))$，而不是拿观测到的 0/1 收益去算，这样运气不会混进方法差异。

四套做法：标准 TS；折扣 TS（每步先把偏离先验的部分乘 $\gamma$，即 $\alpha\leftarrow1+\gamma(\alpha-1)$，$\beta$ 同理，$\gamma=0.99$）；滑窗 TS（只保留最近 100 次观测重建后验）；以及静态 UCB1 作为对照。三者都不知道变点位置，切点只用于事后评估。

## 三、后验塌缩看得见

先看两个直接指标。标准 TS 的总伪计数 $\sum_k(\alpha_k+\beta_k)$ 从初始的 10 线性涨到 1210，正好等于 $1200+2\times5$；折扣 TS 与滑窗 TS 都稳定在 110 附近，前者由 $1/(1-\gamma)\approx100$ 加上先验地板给出，后者由窗口长度 100 给出——两个完全不同的机制，收敛到几乎同一个有效样本量，这个巧合后面还会用到。

更关键的是主导臂（累计观测最多的那个臂）的后验标准差 $\sqrt{\alpha\beta/((\alpha+\beta)^2(\alpha+\beta+1))}$，它衡量「采样噪声还剩多少」。第 299 天时标准 TS 是 0.0300，到第 1199 天只剩 0.0167，单调趋近于零；折扣 TS 从 0.0557 到 0.0528、滑窗从 0.0535 到 0.0509，都维持在一个不消失的水平。

![](/images/thompson-sampling-nonstationary-posterior-collapse/fig3_concentration_and_posterior_sd.png)

图 3 左半边是浓度的线性爬升，右半边是采样噪声的塌缩，三条竖虚线是变点。**噪声消失意味着算法不再有能力「偶然试一下别的臂」，而偶然尝试恰恰是它发现区制切换的唯一通道**。

用数字说更直白：第一个区制里 2 号臂大约被选了 290 次，后验均值约 0.60。切换后它的真实概率掉到 0.10，若算法继续选它，每次失败只让 $\beta$ 加一；要把均值从 0.60 拉到 0.12 以下，大致需要与原始样本量同量级的连续失败，而一个区制只有 300 天。于是它整个区制都赖在旧冠军身上。

## 四、赖着不走的代价

四个算法的累计伪遗憾分别是：标准 TS 383.6、折扣 TS 169.7、滑窗 TS 161.6、UCB1 101.3。标准 TS 的遗憾曲线在每次变点后都出现一段近乎直线的陡升，正是「用旧概率打新区制」的代价。

![](/images/thompson-sampling-nonstationary-posterior-collapse/fig2_cumulative_regret.png)

把镜头拉近到每次变点后的 100 天，看「选对新冠军的比例」：标准 TS 是 0.00 / 0.03 / 0.00，等于三次里有两次完全没适应；折扣 TS 是 0.24 / 0.29 / 0.65；滑窗 TS 是 0.24 / 0.31 / 0.46；UCB1 是 0.72 / 0.63 / 0.67。首次选对新臂的滞后天数（100 表示整个窗口都没命中）依次是标准 TS 100 / 30 / 100、折扣 TS 2 / 9 / 14、滑窗 TS 43 / 10 / 22、UCB1 14 / 10 / 34。

再看「抱住上一区制最优臂」的天数占比：标准 TS 高达 0.88 / 0.22 / 0.72，折扣 TS 0.20 / 0.19 / 0.12，滑窗 TS 0.18 / 0.17 / 0.12，UCB1 0.08 / 0.06 / 0.11。第一个区制切换后，标准 TS 有 88% 的日子还在用一个已经从 0.60 掉到 0.10 的策略。

![](/images/thompson-sampling-nonstationary-posterior-collapse/fig4_recovery_speed.png)

图 4 把这三组恢复速度画成并列柱状。**结论不是「TS 不好」，而是「无界记忆的共轭后验与非平稳环境不匹配」**：修正方式不是换更复杂的先验，而是给记忆设上限。

## 五、两个反直觉的观察

**第一，静态 UCB1 反而最好。** 它同样把浓度涨到 1210，却在各次变点后迅速切到新臂。这不说明 UCB1 懂非平稳，而说明它的置信 bonus $\sqrt{2\ln t/n_k}$ 对冷门臂（$n_k$ 很小）始终很大，会强制去抽查这些臂。标准 TS 的冷门臂后验虽宽，但窄后验的旧冠军几乎每次都被抽到最高值，宽后验要「碰巧超过它」的概率反而被压得很低。两种探索机制对「被长期冷落」的处理方式不同，结果就差了近四倍遗憾。

**第二，探索率高不等于适应快。** 若把「选中的不是后验均值最大臂」记成探索率，标准 TS 最后 100 天是 0.51，比折扣 TS 的 0.13、UCB1 的 0.08 都高，可它的恢复率几乎是零。原因是冷门臂浓度只有个位数，采样 $\theta$ 方差极大，偶尔冒尖被选中，但一次观测根本不足以积累证据，下一轮又沉回去——这是无效抽查，不是有效探索。**评估非平稳 bandit 时，别拿探索率或平均奖励当健康指标，要看变点后选对新臂的比例、首次命中滞后，以及抱住旧臂的时间**。这一点在实盘尤其容易踩：监控面板上「仍在探索」看起来很健康，仓位却早已钉死在过气策略上。

## 六、记忆上限不是免费的

折扣因子把「记多久」变成一个显式旋钮，但两端都有代价。扫描结果：

| γ | 1.0 | 0.999 | 0.995 | 0.99 | 0.98 | 0.95 | 0.90 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 总遗憾 | 383.6 | 247.4 | 194.2 | **169.7** | 190.0 | 292.8 | 354.6 |
| 末期浓度 | 1210 | 709 | 209.5 | 110 | 60 | 30 | 20 |

明显的 U 型：γ 太大适应不了，γ 太小（有效记忆只剩 10～30 次观测）后验被单次成败剧烈摇晃，稳态遗憾重新抬头。γ=0.90 与完全不折扣几乎一样差，只是差的原因相反。

区制长度扫描进一步说明没有万能值。把每个区制长度设成 50、100、200、300、600 天重跑，最优 γ 依次为 0.98、0.98、0.995、0.99、0.995；短区制偏向更快遗忘，但最优值始终挤在 0.98～0.995 的窄带内，且受有限样本噪声扰动、不是严格单调。更实用的经验是让有效记忆 $1/(1-\gamma)$ 与预期区制长度同量级但略短：本例区制 300 天，γ=0.99 给出约 100 天记忆，恰好落在最优点上；而各长度下不折扣 TS 的平均逐步遗憾始终在 0.27～0.33，没有一次接近最优。

![](/images/thompson-sampling-nonstationary-posterior-collapse/fig5_gamma_and_regime_length.png)

图 5 左是 U 型曲线，右是最优 γ 与区制长度的关系（双轴：左轴最优 γ，右轴标准 TS 的平均逐步遗憾）。顺带一提，滑窗 W=100 与 γ=0.99 的浓度同为 110，但滑窗三次恢复率更均衡（0.24 / 0.31 / 0.46），折扣 TS 的第三次冲到 0.65、第一次只有 0.24，波动更大；代价是滑窗要保留 W 条记录、每步重建一遍后验。工程上选哪个，取决于你更在意稳定的最坏表现还是实现成本。

## 七、边界与落地建议

必须说清局限：奖励是伯努利的，真实策略收益是连续、自相关、带偏度的；概率被设成分段常数且变点已知（仅供事后评估，算法本身看不到）；臂之间独立，没有策略容量、交易成本、换仓冲击，也没考虑同时上场多个策略。这些都会改变最优遗忘速度，不能把 γ=0.99 直接抄进实盘。

真正可迁移的是三条：**把后验浓度当成受监控的风险参数**（不是超参），它一旦无界增长，就意味着算法正在丧失自我修正能力；**用变点后的恢复指标而非整体平均收益做验收**；**把显式变点检测与折扣/滑窗结合**，检测到了就清空或加速遗忘，没检测到时折扣提供保底——这正是 [贝叶斯在线变点检测](/blog/bayesian-online-changepoint-run-length/) 那篇的下游用法。

## 八、完整代码

下面这段代码生成全部数字与五张图：世界路径、四种算法、恢复指标、γ 扫描、区制长度扫描，最后写 `results.json`。在项目根目录直接运行即可复现，随机种子固定为 20261108，所有算法共用同一张奖励随机数表和同一个采样流。

```python
"""非平稳多臂老虎机里的 Thompson Sampling 后验塌缩：完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_ts_20261108pm.py

产出：
    public/images/thompson-sampling-nonstationary-posterior-collapse/fig1..fig5.png
    public/images/thompson-sampling-nonstationary-posterior-collapse/results.json

核心问题：Beta-Bernoulli 共轭后验的浓度 alpha+beta 随观测线性增长，
采样噪声 ~ 1/sqrt(n) 趋于零，Thompson Sampling 于是悄悄退化成贪心。
在平稳世界里这叫「收敛」，在区制切换的世界里这叫「冻结」——
旧最优臂积累了上千次伪计数，哪怕概率已经掉下来也要很久才肯让位。
下面用共同随机数把标准 TS、折扣 TS、滑窗 TS 与 UCB1 放在同一条
收益路径上对拍，并扫描折扣因子与区制长度，量化这个「记忆上限」的代价。
"""
import json
import os
from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/thompson-sampling-nonstationary-posterior-collapse"
K = 5
T = 1200
T_MAX = 2400
CPS = (300, 600, 900)
REGIME_P = np.array([
    [0.10, 0.15, 0.60, 0.12, 0.10],
    [0.55, 0.12, 0.10, 0.15, 0.10],
    [0.10, 0.10, 0.10, 0.62, 0.12],
    [0.12, 0.60, 0.10, 0.10, 0.14],
])
GAMMAS = (1.0, 0.999, 0.995, 0.99, 0.98, 0.95, 0.90)
LENGTHS = (50, 100, 200, 300, 600)
WINDOW = 100
RECOVERY_H = 100


def make_world(n, cps, u_all, probs=REGIME_P):
    """用同一张潜在均匀随机数表构造奖励，保证所有算法看到同一条世界路径。"""
    U = u_all[:n]
    P = np.zeros((n, K))
    bounds = (0,) + tuple(cps) + (n,)
    for j, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        P[a:b] = probs[j % len(probs)]
    R = (U < P).astype(float)
    best = P.argmax(axis=1)
    pmax = P.max(axis=1)
    return R, P, best, pmax


def run_ts(R, mode="standard", gamma=0.99, window=WINDOW, seed=1000):
    """返回 (浓度, 非贪心率, 选中臂, 主导臂后验标准差)。

    主导臂 = 当前后验浓度（累计伪计数）最高的臂；它的后验标准差
    sqrt(ab / ((a+b)^2 (a+b+1))) 就是「采样噪声还剩多少」的直接度量。
    """
    rng = np.random.default_rng(seed)
    n = len(R)
    alpha = np.ones(K)
    beta = np.ones(K)
    hist = deque(maxlen=window) if mode == "window" else None
    conc = np.zeros(n)
    explore = np.zeros(n)
    arms = np.zeros(n, dtype=int)
    sd_dom = np.zeros(n)
    for t in range(n):
        if mode == "discount":
            # 先验地板 + 对偏离部分折扣：有效样本量上限约 1/(1-gamma)
            alpha = 1.0 + gamma * (alpha - 1.0)
            beta = 1.0 + gamma * (beta - 1.0)
        elif mode == "window":
            alpha = np.ones(K)
            beta = np.ones(K)
            for a, r in hist:
                if r > 0:
                    alpha[a] += 1.0
                else:
                    beta[a] += 1.0
        mean = alpha / (alpha + beta)
        greedy = int(np.argmax(mean))
        if mode == "ucb":
            counts = alpha + beta - 2.0
            bonus = np.sqrt(2.0 * np.log(t + 1) / np.maximum(counts, 1.0))
            arm = int(np.argmax((alpha - 1.0) / np.maximum(counts, 1.0)
                                + bonus))
        else:
            arm = int(np.argmax(rng.beta(alpha, beta)))
        r = R[t, arm]
        if mode != "window":
            alpha[arm] += r
            beta[arm] += 1.0 - r
        else:
            hist.append((arm, r))
        conc[t] = float(np.sum(alpha + beta))
        explore[t] = float(arm != greedy)
        arms[t] = arm
        dom = int(np.argmax(alpha + beta))
        s = alpha[dom] + beta[dom]
        sd_dom[t] = np.sqrt(alpha[dom] * beta[dom] / (s ** 2 * (s + 1.0)))
    return conc, explore, arms, sd_dom


def run_algorithm(R, P, pmax, mode, gamma=0.99, window=WINDOW, seed=1000):
    """用真实概率矩阵逐日重算被选臂的收益，得到正确的伪遗憾。"""
    conc, explore, arms, sd_dom = run_ts(
        R, mode=mode, gamma=gamma, window=window, seed=seed)
    chosen_p = P[np.arange(len(R)), arms]
    correct = np.cumsum(pmax - chosen_p)
    return {"cum_regret": correct, "conc": conc, "explore": explore,
            "arms": arms, "sd_dom": sd_dom}


def recovery_share(arms, best, cps, h=RECOVERY_H):
    """每个变点后 h 步内选对当前最优臂的比例。"""
    out = []
    for c in cps:
        lo, hi = c, min(c + h, len(arms))
        out.append(float(np.mean(arms[lo:hi] == best[lo:hi])))
    return out


def recovery_lag(arms, best, cps, h=RECOVERY_H):
    """变点后第一次选中新最优臂用了多少天；窗口内始终没选到则记 h。"""
    out = []
    for c in cps:
        lo, hi = c, min(c + h, len(arms))
        hits = np.flatnonzero(arms[lo:hi] == best[lo:hi])
        out.append(int(hits[0]) + 1 if len(hits) else h)
    return out


def stale_arm_share(arms, P, cps):
    """每个新区制里，仍然抱住上一个区制最优臂的天数占比。"""
    bounds = (0,) + tuple(cps) + (len(arms),)
    out = []
    for j in range(1, len(bounds) - 1):
        a, b = bounds[j], bounds[j + 1]
        prev_best = int(np.argmax(P[a - 1]))
        out.append(float(np.mean(arms[a:b] == prev_best)))
    return out


def run():
    os.makedirs(OUT, exist_ok=True)
    rng_u = np.random.default_rng(SEED)
    U_all = rng_u.random((T_MAX, K))
    R, P, best, pmax = make_world(T, CPS, U_all)

    # 共同随机数：所有算法、所有扫描共用同一条采样随机流（seed 2001），
    # 差异只来自算法本身，不被采样运气污染。
    algos = {
        "standard": run_algorithm(R, P, pmax, "standard", seed=2001),
        "discount": run_algorithm(R, P, pmax, "discount", gamma=0.99,
                                  seed=2001),
        "window": run_algorithm(R, P, pmax, "window", window=WINDOW,
                                seed=2001),
        "ucb1": run_algorithm(R, P, pmax, "ucb", seed=2001),
    }

    main = {
        "total_regret": {k: float(v["cum_regret"][-1]) for k, v in algos.items()},
        "posterior_concentration_at_1199": {
            k: float(v["conc"][-1]) for k, v in algos.items()},
        "explore_rate_first100": {
            k: float(v["explore"][:100].mean()) for k, v in algos.items()},
        "explore_rate_last100": {
            k: float(v["explore"][-100:].mean()) for k, v in algos.items()},
        "recovery_share": {
            k: recovery_share(v["arms"], best, CPS) for k, v in algos.items()},
        "recovery_lag_days": {
            k: recovery_lag(v["arms"], best, CPS) for k, v in algos.items()},
        "stale_arm_share": {
            k: stale_arm_share(v["arms"], P, CPS) for k, v in algos.items()},
        "final_posterior_sd_dominant": {
            k: float(v["sd_dom"][-1]) for k, v in algos.items()},
        "posterior_sd_at_299": {
            k: float(v["sd_dom"][299]) for k, v in algos.items()},
    }

    # 折扣因子扫描（同一条世界路径，采样种子也固定）
    gamma_scan = []
    for g in GAMMAS:
        if g == 1.0:
            res = run_algorithm(R, P, pmax, "standard", seed=2001)
        else:
            res = run_algorithm(R, P, pmax, "discount", gamma=g, seed=2001)
        gamma_scan.append({"gamma": g,
                           "total_regret": float(res["cum_regret"][-1]),
                           "final_concentration": float(res["conc"][-1])})

    # 区制长度扫描：每个 L 下的平均逐步遗憾与最优 gamma
    length_scan = []
    for L in LENGTHS:
        cps = (L, 2 * L, 3 * L)
        n = 4 * L
        Ri, Pi, besti, pmaxi = make_world(n, cps, U_all)
        row = []
        for g in GAMMAS:
            if g == 1.0:
                res = run_algorithm(Ri, Pi, pmaxi, "standard", seed=2001)
            else:
                res = run_algorithm(Ri, Pi, pmaxi, "discount", gamma=g,
                                    seed=2001)
            row.append({"gamma": g,
                        "avg_regret": float(res["cum_regret"][-1] / n)})
        best_row = min(row, key=lambda d: d["avg_regret"])
        length_scan.append({"regime_length": L,
                            "best_gamma": best_row["gamma"],
                            "best_avg_regret": best_row["avg_regret"],
                            "standard_avg_regret": row[0]["avg_regret"],
                            "curve": row})

    results = {
        "config": {"T": T, "K": K, "changepoints": list(CPS),
                   "seed": SEED, "window": WINDOW,
                   "recovery_horizon": RECOVERY_H},
        "main": main, "gamma_scan": gamma_scan, "length_scan": length_scan,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # ---------------- 五张真实计算图 ----------------
    plt.rcParams.update({"font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "figure.dpi": 110})
    tt = np.arange(T)

    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    for k in range(K):
        ax.plot(tt, P[:, k], lw=1.6, label=f"arm {k} : win prob")
    for c in CPS:
        ax.axvline(c, color="#888", ls="--", lw=1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("P(strategy beats benchmark)")
    ax.set_title("(a) piecewise-stationary reward, 3 regime switches")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_regime_dgp.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    for name, v in algos.items():
        ax.plot(tt, v["cum_regret"], lw=1.8, label=name)
    for c in CPS:
        ax.axvline(c, color="#888", ls="--", lw=1)
    ax.set_xlabel("trading day")
    ax.set_ylabel("cumulative pseudo-regret")
    ax.set_title("Cumulative regret under the same reward path")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_cumulative_regret.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    for name in ("standard", "discount", "window"):
        ax[0].plot(tt, algos[name]["conc"], lw=1.6, label=name)
    ax[0].set_xlabel("trading day")
    ax[0].set_ylabel("sum(alpha+beta)")
    ax[0].set_title("(a) posterior concentration")
    ax[0].legend(fontsize=8)
    for name in ("standard", "discount", "window"):
        ax[1].plot(tt, algos[name]["sd_dom"], lw=1.4, label=name)
    for c in CPS:
        ax[1].axvline(c, color="#888", ls="--", lw=1)
    ax[1].set_xlabel("trading day")
    ax[1].set_ylabel("posterior SD of dominant arm")
    ax[1].set_title("(b) sampling noise collapses toward zero")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_concentration_and_posterior_sd.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    names = ("standard", "discount", "window", "ucb1")
    x = np.arange(len(CPS))
    w = 0.2
    for i, name in enumerate(names):
        ax.bar(x + (i - 1.5) * w, main["recovery_share"][name], w,
               label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([f"after switch {i + 1}" for i in range(len(CPS))])
    ax.set_ylabel(f"share of days choosing the new best arm ({RECOVERY_H}d)")
    ax.set_title("Adaptation speed in the 100 days after each switch")
    ax.set_ylim(0, 1.0)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_recovery_speed.png")
    plt.close(fig)

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot([d["gamma"] for d in gamma_scan],
               [d["total_regret"] for d in gamma_scan], "o-", lw=2)
    ax[0].set_xlabel("discount gamma (1.0 = standard TS)")
    ax[0].set_ylabel("total regret")
    ax[0].set_title("(a) discount factor: a real trade-off")
    ax2 = ax[1].twinx()
    l1, = ax[1].plot([d["regime_length"] for d in length_scan],
                     [d["best_gamma"] for d in length_scan], "o-", lw=2,
                     color="#3b6ea5", label="best gamma (left)")
    l2, = ax2.plot([d["regime_length"] for d in length_scan],
                   [d["standard_avg_regret"] for d in length_scan],
                   "s--", lw=1.6, color="#c0654a",
                   label="standard TS avg regret (right)")
    ax[1].set_xlabel("regime length (days)")
    ax[1].set_ylabel("best gamma", color="#3b6ea5")
    ax2.set_ylabel("average regret per day", color="#c0654a")
    ax2.grid(False)
    ax[1].set_title("(b) best gamma vs regime length")
    ax[1].legend(handles=[l1, l2], fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_gamma_and_regime_length.png")
    plt.close(fig)

    print("Nonstationary TS experiment (T=%d, K=%d, switches=%s)" %
          (T, K, CPS))
    for name in algos:
        print("  %-9s regret %7.1f  conc@1199 %7.1f  sd@299/1199 "
              "%.4f/%.4f  recovery %s  lag %s  stale %s" %
              (name, main["total_regret"][name],
               main["posterior_concentration_at_1199"][name],
               main["posterior_sd_at_299"][name],
               main["final_posterior_sd_dominant"][name],
               ", ".join("%.2f" % v for v in
                         main["recovery_share"][name]),
               main["recovery_lag_days"][name],
               ", ".join("%.2f" % v for v in
                         main["stale_arm_share"][name])))
    for d in gamma_scan:
        print("  gamma=%-6g regret %8.1f  final conc %8.1f" %
              (d["gamma"], d["total_regret"], d["final_concentration"]))
    for d in length_scan:
        print("  regime_len=%3d  best gamma %-6g  best avg regret %.4f  "
              "standard %.4f" %
              (d["regime_length"], d["best_gamma"], d["best_avg_regret"],
               d["standard_avg_regret"]))
    return results


if __name__ == "__main__":
    run()
```

想自己动手验证塌缩机制，最简单的办法是把 `gamma` 改成 1.0 再跑一遍，然后盯住图 3 右半边那条蓝线：它不会在某次变点后抬头，只会一路贴着零往下走。所有「适应失败」的故事，都藏在这一条曲线里。
