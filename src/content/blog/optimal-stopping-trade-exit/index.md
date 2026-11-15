---
title: "最优停止出场边界：止盈和止损都是它的特例，动态规划一次算清"
description: "「涨了卖还是拿着等」是每一笔交易的开放问题。把价格建成 OU 均值回复过程（持有还有 0.02%/日的机会成本），卖出决策就是一个最优停止问题：倒推动态规划 V_t(p)=max(p, ρ·E[V_{t+1})] 直接给出每时每刻的卖出边界 b_t。60 天期、8000 条共同随机数路径对拍：DP 边界期望折现卖出价 102.84，固定止盈 102.71，回本就卖 100.95，移动止损 99.69，持有到期只有 98.84，上帝视角上限 105.42。纯随机游走校验里 DP 值 100.00000036 对理论值 100 的误差仅 3.6e-09，而先知才能拿到的 E[max] 是 108.28——鞅的可选停时定理说明那是不可达的幻觉。参数错配扫描：σ 从 1.5 低估到 0.75，期望损失 0.69 元，是最大死扛代价。"
publishDate: '2026-11-15T07:12:00Z'
tags:
  - 量化
  - 交易
  - 风险管理
  - 动态规划
  - 随机过程
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

「浮盈 4 个点是落袋还是再拿拿？」——几乎所有交易者都靠拍脑袋回答这个问题。它的数学本体是**最优停止**：在给定价格动态下，找一个停时 τ 最大化 $\mathbb{E}[\rho^\tau p_\tau]$。一旦动态可以采样、转移密度可算，这个问题就有**精确解**：倒推动态规划，每一步输出一个卖出边界 $b_t$，报价上穿即走，否则持有。

60 天期限、8000 条共同随机数测试路径上的对拍结果（期望折现卖出价，建仓价 100）：

| 策略 | 期望折现卖出价 | 平均出场日 | 提前出场率 |
|---|---|---|---|
| DP 最优边界 | **102.84** | 29.3 天 | 90.4% |
| 固定止盈（最优 4.0%） | **102.71** | 30.1 天 | 77.6% |
| 回本就卖 | **100.95** | 5.1 天 | 99.6% |
| 移动止损（最优 0.5） | **99.69** | 2.5 天 | 100% |
| 持有到期 | **98.84** | 60.0 天 | 0% |
| 上帝视角（先知） | **105.42** | — | — |

三个要点：一，DP 的初值 $V_0(100)=102.84$ 通过了纯随机游走校验——鞅的可选停时定理预言最优值恰为 100，DP 算出 **100.00000036**，误差 **3.6e-09**；而「全程最高点均价」$E[\max p_t]$ = **108.28** 是只有先知才能兑现的幻觉。二，精心调参的移动止损在这组动态下是灾难：最优参数 0.5 意味着 2.5 天就被洗出、100% 提前离场。三，边界对**波动率低估**最敏感：把 σ=1.5 当成 0.75 标定，真实行情里每条路径亏 **0.69 元**。

## 二、问题的建立

价格动态取 Ornstein-Uhlenbeck：$p_{t+1} = p_t + \theta(m - p_t) + \sigma z_t$，现场参数 m=100、θ=0.08（半衰期 $\ln 2/\theta$ 约 8.7 天）、σ=1.5、T=60、p0=100。选 OU 不是偷懒：它是一阶自回归在连续时间的极限，是刻画「围绕均衡涨落」的最简可解模型，大量均值回复类标的（价差、估值回归、超额跌后的反弹）都可以先拿它当零假设。θ>0 意味着**等待有价值**——价格跌了大概率回抬，早卖等于把回复红利让给别人；但持有每天收 0.02% 的**机会成本**（折现因子 ρ=1/1.0002，对应资金另投无风险资产的收益），等待也有代价。回复收益与时间租金的拉锯，才让「何时卖」成为非平凡问题：若 θ=0 且 ρ=1，答案退化成了卖不卖都无所谓；若 ρ 更小（利率更高），最优策略整体前移。

贝尔曼方程一行写完：

$$V_t(p) = \max\big(p,\ \rho\,\mathbb{E}_p[V_{t+1}(p')]\big)$$

卖出就落袋 p；持有则拿明日价值的折现期望。这里 $\mathbb{E}_p$ 是对转移核 $p'=p+\theta(m-p)+\sigma z$ 的积分，用 **60 点 Gauss-Hermite 求积**在价格网格（40 到 160，步长 0.1，共 1201 个节点）上数值完成：高斯节点的权重把正态期望变成加权和，60 阶对 OU 这类光滑转移的积分误差远小于价格网格的离散误差。从 t=59（此时持有只剩到期落袋，$V_T=p$）一路倒推到 t=0，同时记录每个 t 的卖出边界 b_t：持有价值首次不低于 p 的价格临界点。边界以下的区域就是「继续扛」的领地。

## 三、一条路径与一族边界

![一条样本路径：报价第 23 天上穿 b_t，以 105.3 卖出](/images/optimal-stopping-trade-exit/fig01.png)

上图是一条样本路径与边界轨迹：价格早盘下探到 94 一线——DP 不慌，因为 b_t 还悬在 104 附近，此时卖出远不如等回复；第 **23** 天报价上穿边界，以 **105.3** 成交。注意折现后报价（虚线）与裸价格的分叉：机会成本把尾部的高价「折」了回去，这正是第 55 天后不再恋战的定价逻辑。

边界族图揭示了三个单调性：剩余时间越长，要价越高（T-t 大时 b_t 逼近 105）；均值回复越快（θ 大），边界整体下移、更早落袋；**没有机会成本**（ρ=1，虚线）时边界全程更高——不赶时间，就该等更好的价。临到期边界单调下行至均衡 m 附近，「越接近到期越肯卖」。

![边界族：θ 与 ρ 如何塑造卖出边界](/images/optimal-stopping-trade-exit/fig02.png)

## 四、校验：DP 值必须过鞅这一关

数值 DP 最容易犯的错是「看起来像答案」。这里用解析可证的极限做检验：令 θ=0、ρ=1，价格成纯随机游走。鞅的可选停时定理：对**任何**适应于价格流的有界停时 τ，$\mathbb{E}[p_\tau]=p_0=100$。所以最优停止值必须恰为 100。DP 算出 **100.00000036**，相对误差 **3.6e-09**，即离散化噪声量级——实现正确。

对照组是蒙特卡洛的先知期望 $\mathbb{E}[\max_t p_t]$ = **108.28**。它比 DP 值高 8.3 元，差在哪？「先看完全部路径再挑最高点」不是停时——决策不能依赖未来。这 8.3 元就是**信息租金**：任何回测里若你的出场规则隐含引用了未来（事后高低点、全样本分位数），你测到的就是先知线而不是可得线。

## 五、五策略对拍与调参纪律

规则策略的参数（固定止盈档位、止损距离）**只在 4000 条训练路径上标定**，再到另外 8000 条测试路径上出成绩；DP 边界无自由参数。所有策略在同一条随机数流上评估（共同随机数），路径噪声不偏袒任何一方——如果不做共同随机数，两个策略的均值差里混杂着「谁运气好抽到好路径」，几百分之一元的差距就完全失去意义。

![五策略柱状图与上帝视角上限](/images/optimal-stopping-trade-exit/fig03.png)

几个值得咀嚼的数字。**固定止盈**标定到 4.0%（训练 102.768、测试 102.707，无过拟合迹象），已接近 DP——因为 OU 慢回复下「等够就卖」本就近似最优。**回本就卖**（100.95）输 1.9 元：99.6% 的路径在 5 天内小赚出局，系统性地砍掉了右侧厚尾。**移动止损**最反直觉：最优距离竟标定到网格下界 0.5，均值 **2.45** 天全部被震出——σ=1.5 的日内波幅下，任何宽止损都会先被 0.5 的窄带击穿，它防的回撤远小于洗出再入场的期望损失。DP 出场日分布（下图）则是「中间厚、两头轻」：前期边界高悬，绝大多数路径耐心持有；中后期边界回落到可达区域，仓位逐日释放在真正走出高度的行情上。这正是它比两条规则都好的原因——固定止盈对所有路径说同一句话（到 104 就卖），移动止损对所有路径说另一句话（回撤 0.5 就跑），而 DP 对每条路径在每一天问一次「此刻的报价，配不配得上剩余时间的期望」。

![出场时点结构：固定止盈挤在触线日，DP 分布宽而平滑](/images/optimal-stopping-trade-exit/fig05.png)

## 六、标错了参数，亏多少？

真实世界的 θ、σ 未知，只能用历史标定。把「用 θ̂ 的边界打真实 θ 的行情」的期望损失画成柱状图：θ 从 0.02 到 0.32 扫五档，偏离真值越远亏越多；同样倍数的错估里**高估更疼**——θ 低估一半（0.04）亏 0.018 元，高估一倍（0.16）亏 0.050 元，高估到四倍（0.32）亏 **0.117 元**。直觉：误以为马上回复，就会提前在低价落袋。

![参数错配代价：θ 与 σ 两维扫描](/images/optimal-stopping-trade-exit/fig04.png)

σ 维度呈 V 型且**不对称**：σ=1.5 时损失 **0.006 元**（恰好正确），低估到 0.75 亏 **0.69 元**——边界定得太高，死扛到深度回撤；高估到 3.0 亏 **1.33 元**——边界压得太低，过早贱卖。两个方向的代价都比 θ 维度大一个量级：**标边界最怕的不是看不懂回复，是量不准波动**。另外，无视机会成本（用 ρ=1 的边界）每天要多扛 0.02% 的折现，合计每条路径亏 **0.040 元**、平均出场拖到 32.6 天——不付房租的人，房子也不会自己变好。

## 七、怎么用

- 把「止盈止损」从**规则**升级为**边界**：先给价格动态一个可检验的模型（OU、或任何你能采样的过程），再倒推 DP，得到随剩余期限移动的 b_t。
- 校验永远从鞅极限做起：θ=0、ρ=1 时 DP 必须回到建仓价。对不上就停，先修实现。
- 标定用训练集、汇报用测试集、对拍用共同随机数——少了哪一条，结论都可能来自噪声。
- 模型失真时优先保 σ：波动率低估的代价（0.69 元）远超回复速度错估的最坏情形（0.117 元）。

---

本文全部数字唯一来源是文末脚本（SEED 固定为 202611153，逐位可复现）。它会重解 DP、重跑五个策略并重新生成 `public/images/optimal-stopping-trade-exit/` 下的五张图与 `results.json`；正文每个统计量都能与 JSON 字段对上。

```python
# -*- coding: utf-8 -*-
"""最优停止退出规则：动态规划求解「何时卖出」的最优边界，现场量化止盈的代价。

本文全部数字的唯一来源。运行本脚本会：
1. 在 OU 均值回复价格过程 + 资金机会成本（折现）环境里，后向归纳解最优停止 DP，
   得到逐日卖出边界 b_t；
2. 对拍常数期望检验：纯随机游走（θ=0、无折现）下 DP 值应等于蒙特卡洛 E[max p_t]；
3. 用共同随机数对拍五类实战规则：DP 最优 / 持有到期 / 固定止盈 / 移动止损 / 回本就卖；
4. 参数错配代价：用错 θ、错 σ 的边界在真实环境下的期望损失；
5. 画 5 张图到 public/images/optimal-stopping-trade-exit/；
6. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 202611153
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "optimal-stopping-trade-exit"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# 标准正交求积点（概率家 Hermite：权重对应 exp(-x²/2)）
_QN, _QW = np.polynomial.hermite_e.hermegauss(60)
_QW = _QW / _QW.sum()

PG = np.linspace(40.0, 160.0, 1201)   # 价格网格，步长 0.1


def solve_dp(T, m, theta, sig, rho):
    """后向归纳：V_t(p) = max(p, rho * E_p[V_{t+1}])，返回 (V_0..V_T, 边界 b_0..b_{T-1})。

    期望用 Gauss-Hermite 求积 + 分段线性插值；边界取 g(p)=rho*E[V]-p 的首个下穿零点。
    """
    V = [None] * (T + 1)
    V[T] = PG.copy()
    b = np.full(T, np.inf)
    for t in range(T - 1, -1, -1):
        Vn = np.array([np.interp(PG + theta * (m - PG) + sig * q, PG, V[t + 1])
                       for q in _QN])
        cont = rho * (Vn * _QW[:, None]).sum(axis=0)
        g = cont - PG                      # g>0 继续持有，g<=0 立刻卖出
        V[t] = np.maximum(PG, cont)
        idx = np.where(g[:-1] > 0)[0]
        if idx.size and idx[-1] + 1 < PG.size and g[idx[-1] + 1] <= 0:
            j = idx[-1]                    # 最后一个正值的格点
            b[t] = PG[j] + g[j] * (PG[j + 1] - PG[j]) / (g[j] - g[j + 1])
        elif g[0] <= 0:
            b[t] = PG[0]
    return np.array(V), b


def sim_paths(T, m, theta, sig, Z, p0):
    """给定共同随机数矩阵 Z (n,T)，模拟 OU 价格路径 (n,T+1)。"""
    n, _ = Z.shape
    P = np.empty((n, T + 1))
    P[:, 0] = p0
    for t in range(T):
        p = P[:, t]
        P[:, t + 1] = p + theta * (m - p) + sig * Z[:, t]
    return P


def main():
    rng = np.random.default_rng(SEED)
    T, m, theta, sig, rho = 60, 100.0, 0.08, 1.5, 1.0 / (1.0 + 0.0002)
    p0 = 100.0
    disc = rho ** np.arange(T + 1)
    results = {}

    # ---------- DP 求解与纯随机游走校验（θ=0、ρ=1） ----------
    # 鞅的可选停时定理：对任意有界停时 τ 有 E[p_τ] = p0，因此随机游走下
    # 任何"可测"策略的最优值恰为 p0（不是 E[max p_t]——那是先知的上界，
    # 只有看得见整条路径才能兑现）。DP 若把 V 算成 p0 即通过实现性校验。
    V, b = solve_dp(T, m, theta, sig, rho)
    i0 = int(np.argmin(np.abs(PG - p0)))
    dp_val = float(V[0][i0])

    Zrw = rng.standard_normal((8000, T))
    Prw = sim_paths(T, m, 0.0, sig, Zrw, p0)
    prophet_emax = float(np.mean(Prw.max(axis=1)))  # 先知上界，不可达
    Vrw, _ = solve_dp(T, m, 0.0, sig, 1.0)
    dp_rw = float(Vrw[0][i0])
    results["dp_validation"] = {
        "dp_value_p0": dp_val,
        "random_walk_dp": dp_rw,
        "random_walk_p0": p0,
        "prophet_emax": prophet_emax,
        "rel_error": abs(dp_rw - p0) / p0,
    }

    # ---------- 五类规则对拍（共同随机数） ----------
    Ztr = rng.standard_normal((4000, T))
    Zte = rng.standard_normal((8000, T))
    Ptr = sim_paths(T, m, theta, sig, Ztr, p0)
    Pte = sim_paths(T, m, theta, sig, Zte, p0)

    def payoff(P, stop_idx):
        tau = stop_idx
        return (disc[tau] * P[np.arange(P.shape[0]), tau])

    def exit_dp(P, bb):
        n = P.shape[0]
        tau = np.full(n, T)
        done = np.zeros(n, bool)
        for t in range(T):
            hit = (~done) & (P[:, t] >= bb[t])
            tau[hit] = t
            done |= hit
        return tau

    def exit_fixed_tp(P, lvl):
        n = P.shape[0]
        tau = np.full(n, T)
        hit = P >= lvl
        first = hit.argmax(axis=1)
        any_hit = hit.any(axis=1)
        tau[any_hit] = first[any_hit]
        return tau

    def exit_trailing(P, d):
        n = P.shape[0]
        peak = P[:, 0].copy()
        tau = np.full(n, T)
        done = np.zeros(n, bool)
        for t in range(1, T + 1):
            peak = np.maximum(peak, P[:, t - 1])
            hit = (~done) & (P[:, t] <= peak - d)
            tau[hit] = t
            done |= hit
        return tau

    def exit_on_profit(P):
        n = P.shape[0]
        tau = np.full(n, T)
        hit = P >= p0
        hit[:, 0] = False          # 建仓价本身不算「回本」
        first = hit.argmax(axis=1)
        any_hit = hit.any(axis=1)
        tau[any_hit] = first[any_hit]
        return tau

    # 固定止盈 / 移动止损的最优参数：只在训练集上标定（避免用同批路径调参再报成绩）
    tp_grid = np.arange(0.005, 0.1201, 0.005)
    tp_scores = [float(payoff(Ptr, exit_fixed_tp(Ptr, m * (1 + x))).mean()) for x in tp_grid]
    k_tp = int(np.argmax(tp_scores))
    tl_grid = np.arange(0.5, 12.01, 0.5)
    tl_scores = [float(payoff(Ptr, exit_trailing(Ptr, d)).mean()) for d in tl_grid]
    k_tl = int(np.argmax(tl_scores))

    strat = {}
    taus = {}
    taus["dp"] = exit_dp(Pte, b)
    taus["hold"] = np.full(Pte.shape[0], T)
    taus["tp"] = exit_fixed_tp(Pte, m * (1 + tp_grid[k_tp]))
    taus["trailing"] = exit_trailing(Pte, tl_grid[k_tl])
    taus["profit"] = exit_on_profit(Pte)
    names = {"dp": "DP 最优边界", "hold": "持有到期", "tp": "固定止盈",
             "trailing": "移动止损", "profit": "回本就卖"}
    for k in taus:
        e = payoff(Pte, taus[k])
        strat[k] = {
            "name": names[k],
            "mean_disc_payoff": float(e.mean()),
            "loss_vs_dp_pct": float((dp_val - e.mean()) / max(dp_val - p0, 1e-12) * 100.0),
            "mean_exit_day": float(taus[k].mean()),
            "pct_early_exit": float(np.mean(taus[k] < T) * 100.0),
        }
    best_fixed = {
        "tp_level_pct": float(tp_grid[k_tp] * 100), "tp_score_train": tp_scores[k_tp],
        "tp_score_test": strat["tp"]["mean_disc_payoff"],
        "trail_dist": float(tl_grid[k_tl]), "trail_score_train": tl_scores[k_tl],
        "trail_score_test": strat["trailing"]["mean_disc_payoff"],
    }
    Zpf = rng.standard_normal((8000, T))
    Ppf = sim_paths(T, m, theta, sig, Zpf, p0)
    omniscient = float(np.mean((disc[None, :] * Ppf).max(axis=1)))
    results["strategies"] = strat
    results["best_fixed_params"] = best_fixed
    results["omniscient_ceiling"] = omniscient

    # ---------- 参数错配代价：用错 θ / 错 σ 的边界，在真实环境的同一批路径上跑 ----------
    theta_grid = [0.02, 0.04, 0.08, 0.16, 0.32]
    sigma_grid = [0.75, 1.1, 1.5, 2.2, 3.0]
    loss_theta = []
    for th in theta_grid:
        _, bb = solve_dp(T, m, th, sig, rho)
        e = payoff(Pte, exit_dp(Pte, bb)).mean()
        loss_theta.append(float(dp_val - e))
    loss_sigma = []
    for sg in sigma_grid:
        _, bb = solve_dp(T, m, theta, sg, rho)
        e = payoff(Pte, exit_dp(Pte, bb)).mean()
        loss_sigma.append(float(dp_val - e))
    # 无视机会成本：用 ρ=1（不折现）的边界在 r=0.0002/日 的世界里
    _, b_und = solve_dp(T, m, theta, sig, 1.0)
    e_und = payoff(Pte, exit_dp(Pte, b_und)).mean()
    results["misspecification"] = {
        "theta_true": theta, "sigma_true": sig,
        "theta_grid": theta_grid, "loss_theta": loss_theta,
        "sigma_grid": sigma_grid, "loss_sigma": loss_sigma,
        "boundary_undiscounted_loss": float(dp_val - e_und),
        "boundary_undiscounted_mean_day": float(np.mean(exit_dp(Pte, b_und))),
    }

    # ---------- fig01：一条样本路径 + 卖出边界轨迹 + 出场点 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    Zshow = Zte[0]
    Pshow = sim_paths(T, m, theta, sig, Zshow[None, :], p0)[0]
    tshow = int(exit_dp(Pshow[None, :], b)[0])
    ax.plot(np.arange(T + 1), Pshow, lw=1.2, color="#333333", label="样本价格路径")
    bb_plot = np.where(np.isinf(b), np.nan, b)
    ax.plot(np.arange(T), np.maximum(bb_plot, Pshow.min()), lw=1.6, color="#d62728",
            label="卖出边界 b_t（低于下界则恒持有）")
    ax.plot(np.arange(T + 1), disc * Pshow, lw=1.0, ls="--", color="#1f77b4",
            alpha=0.7, label="折现后报价 ρ^t p_t")
    ax.scatter([tshow], [Pshow[tshow]], s=90, zorder=5, color="#2ca02c",
               label=f"最优出场 t={tshow}，卖 {Pshow[tshow]:.1f}")
    ax.axhline(m, color="#999999", ls=":", lw=0.8)
    ax.set_title(f"OU(θ={theta}, m={m:g}, σ={sig:g}) + 机会成本 0.02%/日：最优出场 = 报价首次上穿 b_t")
    ax.set_xlabel("t（天）")
    ax.set_ylabel("价格")
    ax.legend(fontsize=8)
    _save(fig, "fig01.png")

    # ---------- fig02：边界族 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    for th in [0.02, 0.08, 0.32]:
        _, bb = solve_dp(T, m, th, sig, rho)
        ax.plot(np.arange(T), np.where(np.isinf(bb), np.nan, bb), lw=1.5,
                label=f"θ={th:g}（半衰期 {np.log(2)/(-np.log(max(1-th,1e-9))):.1f} 天）" if th != theta else f"θ={th:g}（基准）")
    _, bb0 = solve_dp(T, m, theta, sig, 1.0)
    ax.plot(np.arange(T), np.where(np.isinf(bb0), np.nan, bb0), ls="--", lw=1.4,
            color="#1f77b4", label="无机会成本（ρ=1）")
    ax.axhline(m, color="#999999", ls=":", lw=0.8)
    ax.annotate("越接近到期越肯卖：\nb_t 单调下行至均衡附近", xy=(52, 99.6),
                xytext=(30, 103.5), fontsize=8,
                arrowprops=dict(arrowstyle="->", lw=0.7))
    ax.set_xlabel("t（天）")
    ax.set_ylabel("卖出边界 b_t")
    ax.set_title("最优卖出的形状：剩余时间越长要价越高；均值回复越快、机会成本越高，越早就卖")
    ax.legend(fontsize=8)
    _save(fig, "fig02.png")

    # ---------- fig03：五策略 + 上帝视角的期望折现报价 ----------
    order = ["dp", "tp", "trailing", "profit", "hold"]
    vals = [strat[k]["mean_disc_payoff"] for k in order] + [omniscient]
    labels = [strat[k]["name"] for k in order] + ["上帝视角上限"]
    colors = ["#2ca02c", "#ff7f0e", "#1f77b4", "#d62728", "#999999", "#8172b3"]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(np.arange(6), vals, color=colors, width=0.62)
    for i, (v, k) in enumerate(zip(vals, order + ["omni"])):
        txt = f"{v:.2f}"
        if k != "omni" and k != "dp":
            pct = strat[k]["loss_vs_dp_pct"]
            txt += f"\n距最优 {pct:.0f}%"
        ax.text(i, v + 0.08, txt, ha="center", fontsize=8.5)
    ax.axhline(p0, color="k", ls=":", lw=0.8)
    ax.text(-0.4, p0 - 0.5, "建仓价 100", fontsize=8)
    ax.set_xticks(np.arange(6), labels, fontsize=8.5)
    ax.set_ylabel("期望折现卖出价 E[ρ^t p_τ]")
    ax.set_ylim(98.5, max(vals) + 1.6)
    ax.set_title(f"同一批 8000 条测试路径：DP 最优 {dp_val:.2f}，固定止盈 {strat['tp']['mean_disc_payoff']:.2f}，"
                 f"回本就卖 {strat['profit']['mean_disc_payoff']:.2f}")
    _save(fig, "fig03.png")

    # ---------- fig04：参数错配代价 ----------
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.bar(np.arange(len(theta_grid)), loss_theta, color="#d62728", width=0.55)
    ax.set_xticks(np.arange(len(theta_grid)), [f"{t:g}" for t in theta_grid], fontsize=8.5)
    ax.axvline(theta_grid.index(theta), color="k", ls=":", lw=0.8)
    ax.set_xlabel("标定用的 θ（真实 θ=0.08）")
    ax.set_ylabel("期望损失（元）")
    ax.set_title("以为均值回复更快 → 更早更低地卖")
    ax2 = axes[1]
    ax2.bar(np.arange(len(sigma_grid)), loss_sigma, color="#1f77b4", width=0.55)
    ax2.set_xticks(np.arange(len(sigma_grid)), [f"{s:g}" for s in sigma_grid], fontsize=8.5)
    ax2.axvline(sigma_grid.index(sig), color="k", ls=":", lw=0.8)
    ax2.set_xlabel("标定用的 σ（真实 σ=1.5）")
    ax2.set_title("低估波动 → 边界过高死扛；高估 → 过早卖")
    fig.suptitle("边界的鲁棒性：拿错参数标定，在真实行情里亏多少", fontsize=11)
    _save(fig, "fig04.png")

    # ---------- fig05：出场时点分布 ----------
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bins = np.arange(0, T + 2, 2)
    ax.hist(taus["dp"], bins=bins, alpha=0.75, color="#2ca02c",
            label=f"DP 最优（均值 {strat['dp']['mean_exit_day']:.1f} 天，提前率 {strat['dp']['pct_early_exit']:.0f}%）")
    ax.hist(taus["tp"], bins=bins, alpha=0.75, color="#ff7f0e",
            label=f"固定止盈（均值 {strat['tp']['mean_exit_day']:.1f} 天，提前率 {strat['tp']['pct_early_exit']:.0f}%）")
    ax.hist(taus["trailing"], bins=bins, alpha=0.75, color="#1f77b4",
            label=f"移动止损（均值 {strat['trailing']['mean_exit_day']:.1f} 天，提前率 {strat['trailing']['pct_early_exit']:.0f}%）")
    ax.set_xlabel("出场日 t")
    ax.set_ylabel("路径数（共 8000 条）")
    ax.set_title("出场时点结构：固定止盈全挤在触线日，移动止损偏早，DP 最优把仓位留给高位")
    ax.legend(fontsize=8)
    _save(fig, "fig05.png")

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
```

