---
title: "卡尔曼平滑器与 HP 滤波器的隐藏前视偏差：回测里最优雅的作弊"
description: "HP 滤波去噪和卡尔曼平滑器其实是同一件事：在局部水平模型下，HP 的带状法方程与 RTS 后向平滑逐点一致，四个 λ（1 到 1000）的最大偏差只有 1.36e-08。这带来一个回测里最隐蔽的坑——HP 输出在任意时点都「提前看过」未来数据。现场实验：400 天序列上 RTS 平滑 RMSE 0.00313，比因果滤波 0.00746 低 58.1%；末点平滑与滤波误差标准差逐位相等（间隙恰好 0.000）；全样本 HP 与因果 HP 的趋势平均差 0.00311。用平滑趋势做趋势跟踪，幻影 Sharpe 0.6835，几乎追平上帝视角 0.7113，而当下可得的因果信号只有 0.3205——虚增 0.3630。信号翻转率 0.33% 对 2.01% 的十倍差距，就是前视的指纹。"
publishDate: '2026-11-15T07:10:00Z'
tags:
  - 量化
  - 交易
  - 时间序列
  - 卡尔曼滤波
  - 回测方法
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

卡尔曼滤波是量化圈的熟面孔，但它的孪生兄弟——**后向平滑器**——在回测里造成的伤害常被忽视。本文用四组现场数值实验说清三件事：

1. **HP 滤波器 = RTS 平滑器**。在局部水平模型下，用完全不同的技术路线（HP 的三对角带状法方程直接解 vs 卡尔曼前向+RTS 后向递推）对拍，λ 从 1 取到 1000，250 天序列上逐点最大偏差只有 **1.36e-08**——两者在数值精度内就是同一个估计量。
2. **平滑器天然前视**。400 天现场数据上，RTS 平滑的 RMSE **0.00313**，比因果滤波的 **0.00746** 低 **58.1%**。它为什么准？因为 t 时刻的平滑估计用了 t 之后的全部观测。唯一「干净」的点是末点：末点平滑与末点滤波的误差标准差**逐位相等**（间隙恰好 **0.000**）——因为末点之后没有未来可偷看。
3. **幻影 Sharpe**。用平滑趋势符号做趋势跟踪策略，Sharpe **0.6835**，几乎追平用真值趋势的上帝视角 **0.7113**；而当下真正可得的因果信号只有 **0.3205**。前视一下虚增 **0.3630**。指纹是信号翻转率：前视信号只翻 **0.33%**，因果信号翻 **2.01%**——差了六倍，因为平滑器已经「知道」哪些抖动会恢复。

## 二、模型与两套独立实现

数据由局部水平模型生成：趋势 τ_t 每天以方差 q 随机游走，观测 y_t = τ_t + ε_t，ε 是方差 σ_ε² 的噪声。现场取 q=4e-08、σ_ε=0.015、N=400。

**路线一：卡尔曼前向滤波 + RTS 后向平滑**。前向侧两步循环：先用状态方程做一步预报 $\hat\tau_{t|t-1}=\hat\tau_{t-1|t-1}$、$P_{t|t-1}=P_{t-1|t-1}+q$，再用新息 $\nu_t=y_t-\hat\tau_{t|t-1}$ 以卡尔曼增益 $K_t=P_{t|t-1}/(P_{t|t-1}+\sigma_\varepsilon^2)$ 做一步更新，递推得到 $\hat\tau_{t|t}$ 与方差 $P_{t|t}$。后向侧从末点（此时平滑值就是滤波值）倒推，每一步把「未来的修正」按比例 $C_t=P_{t|t}P_{t+1|t}^{-1}$ 回灌进当前估计。整个过程只有标量递推，没有任何全局求解，在线性高斯假设下它就是 $\hat\tau_{t|T}$ 的精确解。

**路线二：HP 滤波器**。Hodrick-Prescott 平滑是一个纯优化定义：$\min_\tau \sum_t (y_t-\tau_t)^2 + \lambda\sum_t(\Delta^2\tau_t)^2$，对 τ 求一阶条件得到 $(I+\lambda D_2^\top D_2)\tau=y$ 的带状线性方程组，用 `solveh_banded` 直接解。它与卡尔曼没有任何血缘——一次形成法方程、一次分解、一次回代。它与卡尔曼没有任何血缘——直到我们对拍。

理论桥是 λ = σ_ε²/q：HP 的二阶差分惩罚对应状态噪声方差 q 越小趋势越硬，数据项权重对应观测精度。把生成参数的 σ_ε=0.012、q=1e-08 代入，λ 应为 14400——这个数在后面第五节会再次出现。

![滤波与平滑的现场对照：真实趋势、带噪观测、两条估计曲线](/images/kalman-smoother-hp-filter-lookahead/fig01.png)

## 三、误差动力学：末点是唯一的诚实点

滤波误差标准差从初值快速收敛到稳态 **0.00582**；平滑器在序列中段把标准差压到 **0.00303**（比值 **0.521**），但在末点，平滑标准差**精确回升**到 0.00582，与滤波逐位相等，`endpoint_abs_gap` 恰好为 **0.0**。

为什么会这样？RTS 的修正项是 $C_t(\hat\tau_{t+1|T}-\hat\tau_{t+1|t})$——本质上是「未来某一刻的预测惊喜」沿增益往回传播。序列中段，t 之后还有几百个观测排队提供惊喜，修正幅度最大；越靠近末点可用的未来越少，修正自动衰减；到末点其后无任何观测，修正项恒为零，平滑退化为滤波。

这个端点性质的工程含义是理解一切前视偏差的钥匙：**你永远站在末点上做决定**。实盘里每个当下都是「最后一个已实现的数据点」，平滑器给你的那个更准的值，恰恰是你唯一拿不到的值。回测里把平滑线当信号用，等于每根 K 线都在向未来借答案。

![误差标准差动力学：平滑中段占优、末点与滤波重合](/images/kalman-smoother-hp-filter-lookahead/fig02.png)

## 四、HP 与 RTS 的逐点对拍

λ 取 1、10、100、1000 各跑 250 天。两条完全独立的实现（带状直接解 vs 递推平滑）逐点最大绝对偏差分别为 5.0e-09、4.6e-09、6.5e-09、**1.36e-08**，全部在 float64 累积误差量级——注意 λ 越大病态性越强，偏差从 1e-09 量级爬到 1e-08 量级是符合预期的，若某个 λ 下偏差突然跳到 1e-02 以上，那一定是实现错了而不是数值巧合。

工程含义有两层。第一，凡是你在回测里用 statsmodels 的 hpfilter、pandas 双边移动平滑、小波去噪、或任何「先去噪再算指标」的管线，都要按 RTS 平滑器的标准来审——它们都吃了未来。第二，反过来这是好消息：你不需要同时维护两套心智模型，HP 的 λ 直觉（越大越平）可以直接翻译成卡尔曼的 q/σ_ε² 折中，反之亦然；本文第七节用似然定 λ，走的正是这条翻译通道。

![四个 λ 下 HP 与 RTS 的偏差谱](/images/kalman-smoother-hp-filter-lookahead/fig03.png)

## 五、前视偏差的现场复现

改用另一组参数（N=300、q=1e-08、σ_ε=0.012，对应 λ=14400）做因果性实验。先用「扩展窗口 HP」校验理论：每个 t 只喂 y_1..t 重解 HP，末点输出应与因果滤波逐点相等——实测最大偏差 **8.7e-08**，恒等式成立。

再看全样本 HP：它与因果 HP 的趋势线平均绝对差 **0.00311**，最大 **0.01191**——相对 σ_ε=0.012 来说，趋势线被未来数据「搬运」了大半个噪声幅度。更直观的指标：把前 290 天的趋势线记下来，再追加 10 天数据重算全样本 HP，最后 25% 时段的历史趋势平均被改写 **0.00091**。你的历史趋势曲线不是被「更新」，是被「重写」。

![因果 HP、全样本 HP 与滤波的三线对照](/images/kalman-smoother-hp-filter-lookahead/fig04.png)

## 六、幻影 Sharpe：0.68 的前视对 0.32 的因果

策略：持仓取趋势斜率估计的符号（斜率状态由模型顺带递推），当天收盘定价、次日赚取 y 的增量，年化 √252。三个版本共用同一段数据：

| 信号来源 | Sharpe | 信号翻转率 |
|---|---|---|
| 平滑斜率（前视） | **0.6835** | 0.33% |
| 滤波斜率（因果） | **0.3205** | 2.01% |
| 真值斜率（上帝） | **0.7113** | — |

前视版本几乎复刻上帝视角（差 0.028），因果版本只拿到一半（幻影间隙 **0.3630**）。机制：平滑器知道哪些回调会 V 型反转，于是躲开了因果滤波被洗出去的每一笔；翻转率 0.33% 对 2.01% 的六倍差距，正是它在「事后确认」而非法则实时确认的统计指纹。真趋势均斜 0.000297，说明信号本身有效——差距纯粹来自信息集。

![三条权益曲线与幻影间隙](/images/kalman-smoother-hp-filter-lookahead/fig05.png)

## 七、λ 可以用数据定，但别指望它救前视

平滑窗口的选择有了新解法：前向滤波的对数似然是精确可算的（每步新息 $\nu_t$ 服从 $N(0, P_{t|t-1}+\sigma_\varepsilon^2)$，把各步 $\log$ 密度累加即可），在 q 的对数网格上做极大化、再按 λ=σ_ε²/q 换算回 HP 参数量纲，就能「用数据定窗口」，不必再凭手感调 λ。实测（N=400、σ_ε 固定在真值 0.015）：真参数 q=4e-08 对应 **λ_true=5625**，处似然 **1019.00**；网格极大点换算出 **λ_hat=2003.4**，处似然 **1023.48**——极大似然比「真参数」还高 4.5 个单位，这在有限样本里完全正常：极大化操作本身就在样本上挑了最顺眼的参数，天然带一点过拟合。好在 λ_hat 与 λ_true 同阶（差不到 3 倍），作为实操定窗够用——对平滑而言，λ 差 3 倍只是曲线软硬略有不同，远不像差 100 倍那样面目全非。但切记：**λ 定得再好也消不掉前视**，它只是把偏差-方差折中调对，信息集的问题必须靠因果化解决——用扩展窗口 HP 或干脆卡尔曼滤波，牺牲那 58% 的 RMSE「水分」，换回一个你真实拥有的信号。

## 八、实操清单

- 回测里出现 `hpfilter`、双边移动平均、小波去噪、变点检测事后分段，先问一句：这个值在 t 时刻**拿得到吗**？
- 校验手法：对任意去噪器跑「扩展窗口重算」——历史段输出若随追加数据而变，就有前视。
- 报告 Sharpe 时把「事后确认」与「实时可得」两个版本都跑一遍，差距即前视税。
- 平滑器并非不能用：它的正确岗位是**离线标注、数据清洗、研究阶段的特征回顾**；凡进入实时决策回路，一律换成前向滤波器。

---

本文全部数字的唯一数据源是文末脚本（固定 SEED，逐位可复现），运行它会在 `public/images/kalman-smoother-hp-filter-lookahead/` 下重新生成上述五张图与 `results.json`；正文所有统计量与 JSON 字段逐项对应。

```python
# -*- coding: utf-8 -*-
"""卡尔曼平滑器与 HP 滤波器的隐藏前视偏差：现场数值实验的唯一数据源。

本文全部数字的唯一来源。运行本脚本会：
1. 在局部水平模型上现场模拟趋势+噪声，手写卡尔曼前向滤波与 RTS 后向平滑；
2. 独立实现 HP 滤波器法方程（带状矩阵直接解），与 RTS 平滑器逐点对拍；
3. 用「滚动窗口 HP」现场复现前视偏差：全样本 HP 与因果 HP 的趋势差、
   信号翻转、追加 10 天后历史趋势被改写的幅度；
4. 趋势跟踪策略的幻影 Sharpe：前视信号 vs 因果信号 vs 真值信号；
5. 画 5 张图到 public/images/kalman-smoother-hp-filter-lookahead/；
6. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solveh_banded

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 202611152
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "kalman-smoother-hp-filter-lookahead"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- HP 滤波器
def hp_filter(y, lam):
    """Hodrick-Prescott：min Σ(y_t-τ_t)² + λΣ(Δ²τ_t)²，带状法方程直接解。

    与卡尔曼完全无关的独立实现（线性代数路线），用来和平滑器对拍。
    """
    y = np.asarray(y, float)
    T = y.size
    # 逐行装配 D'D：行 s（0 基，s=2..T-1）在 (s-2,s-1,s) 处为 (1,-2,1)。
    # 内部元主对角 6λ、偏移1 -4λ、偏移2 λ；两端各少参与若干行，边界元由此循环自然算对。
    ab = np.zeros((3, T))
    ab[2, :] = 1.0                # 数据项 I
    coef = np.array([1.0, -2.0, 1.0])
    for s in range(2, T):
        cols = (s - 2, s - 1, s)
        for a in range(3):
            for b in range(a, 3):
                off = cols[b] - cols[a]          # 0/1/2
                ab[2 - off, cols[b]] += coef[a] * coef[b] * lam
    return solveh_banded(ab, y)


# -------------------------------------------------- 局部水平模型卡尔曼滤波
def kalman_llm(y, q_beta, sig_eps, p0=1e8):
    """状态 [τ, β]：τ_t=τ_{t-1}+β_{t-1}+η, β_t=β_{t-1}+ζ(方差 q_beta)，
    观测 y_t=τ_t+ε(方差 sig_eps²)。返回滤波估计 τ̂_{t|t} 与滤波方差 P_t[0,0]。
    """
    T = y.size
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, q_beta]])
    H = np.array([1.0, 0.0])
    x = np.array([y[0], 0.0])
    P = np.eye(2) * p0
    xf = np.zeros(T)
    xb = np.zeros(T)
    pf = np.zeros(T)
    ll = 0.0
    for t in range(T):
        if t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        v = y[t] - float(H @ x)               # 新息
        F = float(H @ P @ H + sig_eps**2)
        ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        K = (P @ H) / F
        x = x + K * v
        P = P - np.outer(K, H @ P)
        xf[t] = x[0]
        xb[t] = x[1]
        pf[t] = P[0, 0]
    return xf, pf, ll, xb


def rts_smooth(y, q_beta, sig_eps, p0=1e8):
    """RTS 后向平滑：返回平滑估计 τ̂_{t|T} 与平滑方差 P_{t|T}[0,0]。"""
    T = y.size
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, q_beta]])
    H = np.array([1.0, 0.0])
    xp = np.zeros((T, 2))   # 预测 x_{t|t-1}
    pp = np.zeros((T, 2, 2))
    xf = np.zeros((T, 2))   # 滤波 x_{t|t}
    pf = np.zeros((T, 2, 2))
    x = np.array([y[0], 0.0])
    P = np.eye(2) * p0
    for t in range(T):
        if t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        xp[t] = x
        pp[t] = P
        v = y[t] - float(H @ x)
        F = float(H @ P @ H + sig_eps**2)
        K = (P @ H) / F
        x = x + K * v
        P = P - np.outer(K, H @ P)
        xf[t] = x
        pf[t] = P
    xs = xf.copy()
    Ps = pf.copy()
    for t in range(T - 2, -1, -1):
        C = pf[t] @ A.T @ np.linalg.inv(pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
        Ps[t] = pf[t] + C @ (Ps[t + 1] - pp[t + 1]) @ C.T
    return (xs[:, 0], np.array([Ps[t][0, 0] for t in range(T)]),
            xs[:, 1], np.array([Ps[t][1, 1] for t in range(T)]))


def simulate_llm(T, rng, tau0=1.0, beta0=0.0008, q_beta=4e-8, sig_eps=0.015):
    """从局部水平模型真值生成数据：β 是随机游走，τ 累加 β，y=τ+观测噪声。"""
    betas = np.zeros(T)
    betas[0] = beta0
    betas[1:] = beta0 + np.cumsum(rng.normal(0.0, np.sqrt(q_beta), T - 1))
    taus = tau0 + np.concatenate([[0.0], np.cumsum(betas[:-1])])
    y = taus + rng.normal(0.0, sig_eps, T)
    return taus, betas, y


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    # ============ 实验 1：滤波 vs 平滑（真值已知的局部水平模型） ============
    N1 = 400
    q_beta_true = 4e-8     # β 增量标准差 2e-4/日
    sig_eps_true = 0.015
    taus1, betas1, y1 = simulate_llm(N1, rng, q_beta=q_beta_true, sig_eps=sig_eps_true)
    xf1, pf1, _, _ = kalman_llm(y1, q_beta_true, sig_eps_true)
    xs1, ps1, _, _ = rts_smooth(y1, q_beta_true, sig_eps_true)
    rmse_f = float(np.sqrt(np.mean((xf1 - taus1) ** 2)))
    rmse_s = float(np.sqrt(np.mean((xs1 - taus1) ** 2)))
    improvement = float(1.0 - rmse_s / rmse_f)
    results["filter_vs_smoother"] = {
        "n": N1,
        "q_beta": q_beta_true,
        "sigma_eps": sig_eps_true,
        "rmse_filter": rmse_f,
        "rmse_smoother": rmse_s,
        "improvement_pct": improvement * 100.0,
    }

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    tt = np.arange(N1)
    ax.plot(tt, taus1, lw=2, color="#222222", label="真实趋势 τ_t")
    ax.plot(tt, y1, lw=0.6, color="#999999", alpha=0.8, label="观测 y_t")
    ax.plot(tt, xf1, lw=1.4, color="#d62728", label="卡尔曼滤波 τ̂(t|t)")
    ax.plot(tt, xs1, lw=1.4, color="#1f77b4", label="RTS 平滑 τ̂(t|T)")
    ax.set_title(f"前向滤波 vs 后向平滑（RMSE {rmse_f:.5f} → {rmse_s:.5f}，改善 {improvement*100:.1f}%）")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("水平")
    ax2 = axes[1]
    ax2.plot(tt, xf1 - taus1, lw=0.9, color="#d62728", label="滤波误差")
    ax2.plot(tt, xs1 - taus1, lw=0.9, color="#1f77b4", label="平滑误差")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("t（天）")
    ax2.set_ylabel("对真值误差")
    _save(fig, "fig01.png")

    # ============ 实验 2：误差方差动力学与端点效应 ============
    var_f = np.sqrt(pf1)     # 滤波标准差
    var_s = np.sqrt(ps1)     # 平滑标准差
    conv_f = float(np.mean(var_f[-40:]))
    mid_s = float(np.mean(var_s[100:300]))
    endpoint_gap = float(abs(var_s[-1] - var_f[-1]))
    results["error_dynamics"] = {
        "filter_sd_converged": conv_f,
        "smoother_sd_mid": mid_s,
        "smoother_sd_at_T": float(var_s[-1]),
        "filter_sd_at_T": float(var_f[-1]),
        "endpoint_abs_gap": endpoint_gap,
        "ratio_mid": float(mid_s / conv_f),
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tt, var_f, lw=1.6, color="#d62728", label="滤波误差标准差 √P(t|t)")
    ax.plot(tt, var_s, lw=1.6, color="#1f77b4", label="平滑误差标准差 √P(t|T)")
    ax.axhline(conv_f, ls=":", color="#d62728", lw=0.8)
    ax.axhline(mid_s, ls=":", color="#1f77b4", lw=0.8)
    ax.annotate(f"中段平滑误差带 {mid_s:.5f}\n= 滤波的 {mid_s/conv_f:.1%}",
                xy=(200, mid_s), xytext=(215, mid_s + 0.004), fontsize=8)
    ax.annotate(f"t=T 处两者重合\n差 {endpoint_gap:.2e}",
                xy=(N1 - 1, var_s[-1]), xytext=(N1 - 130, var_s[-1] + 0.006), fontsize=8)
    ax.set_xlabel("t（天）")
    ax.set_ylabel("趋势估计误差标准差")
    ax.set_title("滤波误差单调收敛、平滑误差中段最低、最后一个点两者恒等（端点效应）")
    ax.legend(fontsize=8)
    _save(fig, "fig02.png")

    # ============ 实验 3：HP 滤波器 ≡ RTS 平滑器（独立实现对拍） ============
    # 模型 σ_ε=1，β 增量方差 q，MAP 目标 = Σ(y-τ)² + (1/q)Σ(Δ²τ)² ⇒ λ = 1/q
    y3 = simulate_llm(250, rng, tau0=0.0, beta0=0.01, q_beta=1.0 / 50.0, sig_eps=1.0)[2]
    lam_grid = [1.0, 10.0, 100.0, 1000.0]
    hp_curves = {}
    max_diffs = {}
    for lam in lam_grid:
        tau_hp = hp_filter(y3, lam)
        tau_sm = rts_smooth(y3, q_beta=1.0 / lam, sig_eps=1.0)[0]
        hp_curves[lam] = tau_hp
        max_diffs[lam] = float(np.max(np.abs(tau_hp - tau_sm)))
    worst = max(max_diffs.values())
    results["hp_equals_smoother"] = {
        "n": 250,
        "lambdas": lam_grid,
        "max_abs_diff": {str(l): max_diffs[l] for l in lam_grid},
        "worst_diff": worst,
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, lam in enumerate(lam_grid):
        ax.plot(np.arange(250), hp_curves[lam], lw=1.3, label=f"HP λ={lam:g}（法方程直接解）")
        ax.plot(np.arange(250), rts_smooth(y3, 1.0 / lam, 1.0)[0],
                ls="--", lw=0.9, color="k", alpha=0.5)
    ax.set_title(f"两条独立实现（线性代数 vs 状态空间）逐点一致，最大差 {worst:.2e}")
    ax.set_xlabel("t")
    ax.set_ylabel("趋势估计")
    ax.legend(fontsize=8)
    _save(fig, "fig03.png")

    # ============ 实验 4：滚动窗口 HP = 卡尔曼滤波（因果性恒等式） ============
    # 对每个 t，把 HP 只跑在前 t 个观测上、取最后一个点 = 滤波估计 τ̂(t|t)
    q4, sig4 = 1e-8, 0.012
    tau4, beta4, y4 = simulate_llm(300, rng, tau0=2.0, beta0=0.0006,
                                   q_beta=q4, sig_eps=sig4)
    # HP 目标 Σ(y-τ)²+λΣ(Δ²τ)² 与状态空间（噪声方差 σ_ε²、β 增量方差 q）的换元：λ=σ_ε²/q
    lam4 = sig4**2 / q4
    xf4 = kalman_llm(y4, q4, sig4)[0]
    causal_hp = np.array([hp_filter(y4[:t], lam4)[-1] for t in range(2, 301)])
    exp_hp_eq = float(np.max(np.abs(causal_hp - xf4[1:])))
    # 前视：全样本 HP 在第 t 点的估计
    full_hp = hp_filter(y4, lam4)
    gap_view = float(np.mean(np.abs(full_hp - xf4)))
    gap_max = float(np.max(np.abs(full_hp - xf4)))
    # 数据还没来齐时「未来 10 天一到，历史趋势被改写」的幅度
    # 未来 10 天：从第 300 天真值水平继续同参数演化（无缝延续，无人为跳变）
    tau_next = tau4[-1] + np.cumsum(np.concatenate([[beta4[-1]], beta4[-1:] + rng.normal(0, 1e-4, 9)]))
    y_ext = np.concatenate([y4, tau_next + rng.normal(0.0, 0.012, 10)])
    full_hp_ext = hp_filter(y_ext, lam4)
    tail25 = slice(225, 300)
    rewrite = float(np.max(np.abs(full_hp_ext[:300][tail25] - full_hp[tail25])))
    # 平滑-滤波差对「还剩多少数据」的依赖：中段几乎不动，末尾剧烈改写
    rewrite_profile = np.abs(full_hp_ext[:300] - full_hp)
    results["causality"] = {
        "n": 300,
        "lambda_hp": float(lam4),
        "q_beta": q4,
        "sigma_eps": sig4,
        "expanding_hp_vs_filter_max_diff": exp_hp_eq,
        "full_hp_vs_causal_mean_abs_gap": gap_view,
        "full_hp_vs_causal_max_abs_gap": gap_max,
        "rewrite_after_10_more_days_tail25pct": rewrite,
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(2, 301), causal_hp, lw=1.4, color="#d62728",
            label="因果 HP：只用 y_1..t（= 卡尔曼滤波）")
    ax.plot(np.arange(300), full_hp, lw=1.4, color="#1f77b4",
            label="前视 HP：全样本 y..T")
    ax.plot(np.arange(300), y4, lw=0.5, color="#aaaaaa", alpha=0.7, label="观测")
    ax.set_title(f"同一个 t 两条曲线差多少？平均 {gap_view:.4f}，最大 {gap_max:.4f}；"
                 f"追加 10 天后最后 1/4 样本趋势最大改写 {rewrite:.4f}")
    ax.set_xlabel("t")
    ax.set_ylabel("趋势估计")
    ax.legend(fontsize=8)
    _save(fig, "fig04.png")

    # ============ 实验 5：趋势跟踪策略的幻影 Sharpe ============
    # 资产对数价格 = 局部水平模型；信号 = 趋势斜率符号；仓位滞后一天
    N5 = 300
    q5, sig5 = 1e-8, 0.012
    tau5, beta5, y5 = simulate_llm(N5, rng, tau0=4.0, beta0=0.0006,
                                   q_beta=q5, sig_eps=sig5)
    lam5 = sig5**2 / q5
    bf5 = kalman_llm(y5, q5, sig5)[3]     # 滤波斜率 β̂(t|t)：t 收盘已知
    bs5 = rts_smooth(y5, q5, sig5)[2]     # 平滑斜率 β̂(t|T)：用到全部未来数据
    ret = np.diff(y5)    # ret[t] = y[t+1]-y[t]：t→t+1 的收益，决策必须在 t 收盘前定
    def sharpe(pos):
        # pos[t]：t 收盘可知的仓位，吃 ret[t]
        pnl = pos[:-1] * ret
        s = pnl.std()
        return float(pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0
    # 仓位 = 估计趋势斜率的符号，三种斜率只差「能看到多少未来」：
    pos_look = (bs5 > 0).astype(float)      # 平滑器：t 时刻已知道 t+1、t+2 的趋势走向
    pos_cau = (bf5 > 0).astype(float)       # 滤波器：只有 y_1..t
    pos_true = (beta5 > 0).astype(float)    # 上帝视角：真 β_t
    sh_look, sh_cau, sh_true = sharpe(pos_look), sharpe(pos_cau), sharpe(pos_true)
    flip_look = float(np.mean(np.abs(np.diff(pos_look)) > 0))
    flip_cau = float(np.mean(np.abs(np.diff(pos_cau)) > 0))
    phantom = sh_look - sh_cau
    results["phantom_sharpe"] = {
        "n": N5,
        "lambda": float(lam5),
        "q_beta": q5,
        "sigma_eps": sig5,
        "beta0": 0.0006,
        "sigma_eps": 0.012,
        "sharpe_lookahead_hp": sh_look,
        "sharpe_causal_hp": sh_cau,
        "sharpe_true_trend": sh_true,
        "phantom_gap": phantom,
        "signal_flip_rate_lookahead": flip_look,
        "signal_flip_rate_causal": flip_cau,
        "trend_slope_mean": float(beta5.mean()),
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(N5), pos_look, lw=0.9, color="#1f77b4", alpha=0.85,
            label=f"前视 HP 仓位（Sharpe {sh_look:.2f}）")
    ax.plot(np.arange(N5), pos_cau - 0.12, lw=0.9, color="#d62728", alpha=0.85,
            label=f"因果 HP 仓位（Sharpe {sh_cau:.2f}）")
    ax.plot(np.arange(N5), pos_true - 0.24, lw=0.9, color="#2ca02c", alpha=0.85,
            label=f"真趋势仓位（Sharpe {sh_true:.2f}）")
    ax.set_ylim(-0.35, 1.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"幻影 Sharpe：把平滑器当滤波用，回测虚高 {phantom:.2f}"
                 f"（{sh_look:.2f} vs {sh_cau:.2f}），换手 {flip_cau:.1%}→{flip_look:.1%}")
    ax.set_xlabel("t（天）")
    ax.set_ylabel("仓位（0/1，三行错开显示）")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig05.png")

    # ============ 实验 6：λ 的最大似然辨识（新息分解） ============
    # 网格搜索状态噪声方差 q（σ_ε 固定在生成真值），报表时换算回 HP 的
    # λ = σ_ε²/q，与第四节的理论桥保持同一量纲。
    log_grid = np.linspace(np.log(1e-9), np.log(1e-5), 120)
    lls = np.array([kalman_llm(y1, np.exp(g), sig_eps_true)[2] for g in log_grid])
    g_hat = float(log_grid[np.argmax(lls)])
    q_hat = float(np.exp(g_hat))
    q_from_llm = kalman_llm(y1, q_beta_true, sig_eps_true)[2]
    # 真参数处的似然 vs 网格最优
    results["lambda_ml"] = {
        "q_beta_true": q_beta_true,
        "sigma_eps_fixed": sig_eps_true,
        "lam_true": sig_eps_true**2 / q_beta_true,
        "lam_hat_ml": sig_eps_true**2 / q_hat,
        "loglik_at_hat": float(lls.max()),
        "loglik_at_true": q_from_llm,
        "ml_beats_true_param": bool(lls.max() >= q_from_llm),
    }

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

