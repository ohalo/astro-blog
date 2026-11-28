---
title: "Sinkhorn 熵正则最优传输调仓：把换手成本写进运输计划，为什么温度控制稀疏性、又在目标权重极端时失真"
description: "8 资产调仓实测：从旧权重搬到新权重的理论最小单边换手是 0.4300，Sinkhorn 在 ε=0.0001 时恰好达到 0.4300、只产生 15 笔交易；ε=0.01 时换手涨到 0.8614、交易数 56 笔、成本从 0.003492 升到 0.01013173。高温度还凭空造出隐性回路：由欠配仓位向外提供的资金从 0 升到 0.2453。目标权重突变到单资产 60% 时，ε=0.03 下流入该资产 0.4164，超过实际需要的 0.3004，其中 0.1709 是绕道而来的虚假流量。附完整可运行代码与五张真实图。"
publishDate: '2026-11-28T14:21:00Z'
tags:
  - 量化
  - 组合优化
  - 最优传输
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

大多数调仓代码只算一个结果：新权重减去旧权重，再乘上费率。可是同样的权重变化，可以对应完全不同的交易指令：A 仓多出来的钱可以直接买 C，也可以先经过 B、D 再绕到 C。只要存在固定票费和多档冲击成本，这笔「资金从哪来、到哪去」的匹配就不是无关紧要的细节。

把旧权重当作供给、新权重当作需求、把每对资产的买卖成本当作运输代价，调仓就变成一个离散最优传输问题。用 Sinkhorn 算法解熵正则版本，8 资产的实测结果是：

- 理论最小单边换手是 **0.4300**，ε=0.0001 时 Sinkhorn 恰好做到 **0.4300**，只有 **15** 笔交易，总成本 **0.003492**；
- ε=0.001 时换手升到 **0.4984**、交易数 **39**，成本跳到 **0.00658871**；ε=0.01 时换手 **0.8614**、交易数 **56**、成本 **0.01013173**；
- 更隐蔽的是隐性回路：由本来就应该加仓的资产向外提供的资金，从 ε=0.0001 的 **0** 升到 ε=0.01 的 **0.2453**；
- 目标权重突变到单一资产占 60% 时，ε=0.0003 下流入该资产 **0.3004**，恰好等于需要补足的额度；ε=0.03 下流入 **0.4164**，超出 **38.6%**，其中 **0.1709** 是绕道产生的虚假流量。

关键结论有点反直觉：熵正则让求解变得平滑、可微、稳定，但它同时**改变了交易清单**。最终权重仍然严格等于目标权重，收益端毫发无损；成本端却可能翻倍。把 ε 当成纯粹的数值参数随手调小调大，等价于默默改写了执行方案。

![Sinkhorn 运输计划热力图](/images/sinkhorn-entropic-portfolio-rebalancing/fig1_plan_heatmap.png)

## 二、把调仓写成运输问题

设旧权重向量 a、新权重 b 都和为 1，运输矩阵 P 的第 i 行第 j 列表示「从资产 i 卖出、用于买入资产 j」的金额。行边际等于旧权重、列边际等于新权重，对角线 Pᵢᵢ 表示原地保留的仓位。代价矩阵取 Cᵢⱼ = 卖出冲击费ᵢ + 买入冲击费ⱼ，对角元为零。问题写成：

$$
\min_{P\mathbf{1}=a,\ P^\top\mathbf{1}=b}\ \langle P,C\rangle-\varepsilon H(P),\qquad H(P)=-\sum_{i,j}P_{ij}\log P_{ij}.
$$

ε=0 时是线性规划，最优解天然稀疏且可能有多个；ε>0 后目标严格凸，Sinkhorn 迭代在 log 域里只需交替缩放行与列，就能把边际误差压到 1e-12 以下。具体做法是维护两个势向量 u 与 v，交替更新 u = log a − logsumexp(v − C/ε)、v = log b − logsumexp(u − C/ε)，最后用 exp(uᵢ+vⱼ−Cᵢⱼ/ε) 还原计划。必须在 log 域里算，是因为 ε 很小时 C/ε 的差异会放大到几百，直接对 exp(−C/ε) 做乘除会下溢成一堆零，行边际永远对不上。脚本每次都记录最大边际误差，实测全部小于 1e-12，所以下文所有换手与成本差异都不是求解器没收敛造成的假象。

这里有一个常被搞混的点：**无论 ε 取多少，P 的行列边际都被约束死，所以新权重恒等于 b**。温度改变的不是配置结果，而是「哪一笔卖单配哪一笔买单」。换手定义为非对角元之和，它的理论下界是 0.5‖a−b‖₁，也就是只做必须做的那部分交易；对角线保留量达到各资产新旧权重的较小值之和。

## 三、温度就是执行成本

实验用 8 个资产，旧权重从 0.30 递减到 0.02，新信号把资金往中后段资产挪，冲击费从 0.0008 线性升到 0.0036，另加每笔 0.00012 的固定票费。扫描 ε 从 1e-4 到 0.1：

![温度与换手、交易笔数](/images/sinkhorn-entropic-portfolio-rebalancing/fig2_turnover_epsilon.png)

ε 很小时，计划集中在少数几条最便宜的供给—需求配对上，换手贴着 0.4300 的理论下界走，交易笔数 15；ε 增大后，同一笔买入被摊薄到几乎所有卖出来源，换手一路升到 0.8950，交易笔数停在 56，也就是 8×8 去掉对角线之后几乎所有格子都在交易。这种「每个人都给每个人转一点」的结构，正是熵正则的数学后果，不是数据里的真实需求。

![实际成本随温度上升](/images/sinkhorn-entropic-portfolio-rebalancing/fig3_cost_epsilon.png)

成本曲线单调上升：纯比例费用从 ε=0.0001 的 0.001692 升到 ε=0.1 的 0.00355844，固定票费则把 15 笔变成 56 笔后叠加了 0.00672。对照实验里「按超配比例直接分摊资金」的朴素做法，换手同样是 0.4300、成本同样是 0.003492——小 ε 下 Sinkhorn 自动收敛到这个直觉解；大 ε 下它走向了朴素方法永远不会产生的绕道结构。

## 四、隐性回路与极端目标

把「本来欠配、应当净买入的资产却在向外输送资金」的量定义为隐性流量。它的经济含义很直接：这类仓位必须先卖后买，白白承担两次冲击成本。实测中它从 ε=0.0001 的 0 升到 ε=0.03 的 **0.2626**，占同期总换手 0.8871 的近三成；运输矩阵的香农熵同步从 2.8525 升到 3.8414。

![欠配资产向外输血的隐性回路](/images/sinkhorn-entropic-portfolio-rebalancing/fig4_ghost_flow.png)

目标权重越极端，这个问题越尖锐。把 A1 的目标从 0.30 直接拉到 0.60，其余资产等比例压缩，需要净流入 0.3004。低温 ε=0.0003 下计划精准地只搬 0.3004，隐性流量 0.0004；高温 ε=0.03 下流入 A1 的名义金额达到 **0.4164**，超出的部分正是各资产互相倒手的结果，隐性流量 **0.1709**，占该资产总流入的四成。目标权重没有变，配置结果也完全正确，但执行报告上会多出几十笔毫无经济意义的交易。

![极端目标权重下的运输计划对比](/images/sinkhorn-entropic-portfolio-rebalancing/fig5_extreme_target.png)

## 五、稳健性、边界和用法

对目标权重施加 2% 的随机扰动，再比较运输矩阵的 L1 变化：ε=0.0001 时相对变化是 1.0918，ε=0.001 时 1.1681，ε≥0.01 后降到约 1.0。也就是说，小 ε 下计划确实对目标扰动更敏感一些，但幅度远没有换手和成本那么夸张；主要风险不是「解法不稳」，而是「解法稳定地走向昂贵」。

更重要的边界有三条。第一，Sinkhorn 只回答如何搬，不回答搬去哪，目标权重 b 的质量完全由上游模型负责。第二，线性代价假设冲击成本与交易量成正比，真实市场的大单冲击是非线性的，需要把 C 换成凸函数并改用半离散或二次正则版本。第三，固定票费本身不在线性目标里，必须像本文这样在得到计划后按交易笔数事后计价，否则优化器会系统性地低估小 ε 的优势。这一点与常见的二次换手惩罚很不一样：二次惩罚只关心每个资产净变化了多少，无法区分「一笔直接买」和「三笔绕道买」，而运输矩阵把资金路径完整保留下来，才让固定票费与隐性回路变得可观测。代价是未知量从 n 个净值变成 n² 个配对，资产数上百之后要借助稀疏化或低秩近似。

实践建议是：把 ε 当成执行参数而不是算法参数，按「成本—笔数—隐性流量」三条曲线联合校准；目标权重变动剧烈时优先检查低温计划是否可行，再决定是否为了平滑而升温。

## 六、完整可运行代码

下面脚本固定种子，一次生成全部数值、五张图和 results.json，可独立运行。

```python
"""Entropic optimal-transport rebalancing: Sinkhorn plan, cost and robustness."""

def main():
    import json
    import os

    import numpy as np
    from scipy.special import logsumexp

    SEED = 20261129
    OUT = "/Users/halo/workspace/astro-blog/public/images/sinkhorn-entropic-portfolio-rebalancing"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    n = 8
    labels = [f"A{i+1}" for i in range(n)]
    w0 = np.array([0.30, 0.22, 0.16, 0.12, 0.08, 0.06, 0.04, 0.02])
    w1 = np.array([0.05, 0.08, 0.12, 0.15, 0.18, 0.16, 0.14, 0.12])
    fee = 0.0008 + 0.0004 * np.arange(n)
    C = fee[:, None] + fee[None, :]
    np.fill_diagonal(C, 0.0)
    FIXED = 0.00012
    TRADE_EPS = 1e-4


    def sinkhorn(a, b, eps, iters=6000, tol=1e-12):
        """Log-domain Sinkhorn; returns coupling P and marginal error."""
        a = np.maximum(a, 1e-300)
        b = np.maximum(b, 1e-300)
        la, lb = np.log(a), np.log(b)
        M = -C / eps
        u = np.zeros(n)
        v = np.zeros(n)
        err = np.inf
        for _ in range(iters):
            u_new = la - logsumexp(v[None, :] + M, axis=1)
            v_new = lb - logsumexp(u_new[:, None] + M, axis=0)
            P_new = np.exp(u_new[:, None] + v_new[None, :] + M)
            err = max(np.abs(P_new.sum(axis=1) - a).max(), np.abs(P_new.sum(axis=0) - b).max())
            u, v = u_new, v_new
            if err < tol:
                break
        P = np.exp(u[:, None] + v[None, :] + M)
        return P, float(err)


    def plan_stats(P, b, eps):
        off = P - np.diag(np.diag(P))
        turn = float(off.sum())
        n_trade = int(np.count_nonzero(off > TRADE_EPS))
        prop_cost = float((off * C).sum())
        total_cost = prop_cost + FIXED * n_trade
        H = float(-np.sum(P * np.log(np.maximum(P, 1e-300))))
        underweight = w0 < b
        ghost = float(off[underweight, :].sum())
        return {
            "eps": eps,
            "turnover": round(turn, 6),
            "n_trades": n_trade,
            "proportional_cost": round(prop_cost, 8),
            "cost_with_ticket": round(total_cost, 8),
            "entropy": round(H, 6),
            "ghost_flow": round(ghost, 6),
            "diag_retention": round(float(np.trace(P)), 6),
            "max_offdiag": round(float(off.max()), 6),
        }


    lower_bound = round(float(0.5 * np.abs(w0 - w1).sum()), 6)
    eps_grid = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1]
    results = {
        "seed": SEED,
        "w0": w0.tolist(),
        "w1": w1.tolist(),
        "fee": fee.tolist(),
        "fixed_ticket": FIXED,
        "turnover_lower_bound": lower_bound,
        "grid": {},
        "sensitivity": {},
        "extreme": {},
        "naive": {},
    }

    # proportional rebalancing: sell from every overweight line in proportion to its surplus
    def naive_plan(a, b):
        surplus = np.maximum(a - b, 0)
        need = np.maximum(b - a, 0)
        sell = surplus[:, None] * need[None, :] / surplus.sum()
        P = np.diag(np.minimum(a, b)) + sell
        return P

    P_naive = naive_plan(w0, w1)
    np.fill_diagonal(C, 0.0)
    results["naive"] = plan_stats(P_naive, w1, 0.0)
    results["naive"]["label"] = "proportional funding"

    plans = {}
    for eps in eps_grid:
        P, err = sinkhorn(w0, w1, eps)
        plans[eps] = P
        st = plan_stats(P, w1, eps)
        st["marginal_error"] = err
        results["grid"][str(eps)] = st

        delta = rng.normal(size=n)
        delta = delta / np.abs(delta).sum() * 0.02
        Pb, _ = sinkhorn(w0, w1 + delta, eps)
        sens = float(np.abs(P - Pb).sum() / 0.02)
        results["sensitivity"][str(eps)] = round(sens, 4)

    # extreme target: one asset takes 60% of the book
    b_ext = np.array([0.60, 0.05, 0.05, 0.05, 0.05, 0.05, 0.075, 0.075])
    for eps in [3e-4, 3e-2]:
        P, err = sinkhorn(w0, b_ext, eps)
        st = plan_stats(P, b_ext, eps)
        st["inflow_to_A1"] = round(float(P[:, 0].sum() - P[0, 0]), 6)
        st["marginal_error"] = err
        results["extreme"][str(eps)] = st

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # fig1 coupling heatmap
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    P0 = plans[1e-3]
    im = ax.imshow(P0, cmap="Blues", vmin=0, vmax=P0.max())
    ax.set_xticks(range(n), labels); ax.set_yticks(range(n), labels)
    ax.set_xlabel("buy asset"); ax.set_ylabel("sell asset")
    for i in range(n):
        for j in range(n):
            if P0[i, j] > 0.006:
                ax.text(j, i, f"{P0[i,j]:.3f}", ha="center", va="center", fontsize=7,
                        color="white" if P0[i, j] > P0.max() * 0.55 else "black")
    ax.set_title("Sinkhorn coupling, eps=1e-3 (diagonal = kept position)")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_plan_heatmap.png"), dpi=150); plt.close(fig)

    # fig2 turnover and trade count
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    tv = [results["grid"][str(e)]["turnover"] for e in eps_grid]
    nt = [results["grid"][str(e)]["n_trades"] for e in eps_grid]
    ax.plot(eps_grid, tv, "o-", color="#4c78a8", label="one-way turnover")
    ax.axhline(lower_bound, ls="--", color="gray", label=f"L1 lower bound {lower_bound:.4f}")
    ax.set_xscale("log"); ax.set_xlabel("entropy temperature eps"); ax.set_ylabel("turnover")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx(); ax2.plot(eps_grid, nt, "s--", color="#e45756", label="trades > 1e-4")
    ax2.set_ylabel("number of trades", color="#e45756"); ax2.grid(False)
    ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("Entropy creates extra round trips")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_turnover_epsilon.png"), dpi=150); plt.close(fig)

    # fig3 realized cost
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    pc = [results["grid"][str(e)]["proportional_cost"] for e in eps_grid]
    tc = [results["grid"][str(e)]["cost_with_ticket"] for e in eps_grid]
    ax.plot(eps_grid, pc, "o-", color="#4c78a8", label="proportional cost")
    ax.plot(eps_grid, tc, "s-", color="#e45756", label="with fixed ticket cost")
    ax.axhline(results["naive"]["cost_with_ticket"], ls="--", color="gray", label="proportional funding")
    ax.set_xscale("log"); ax.set_xlabel("entropy temperature eps"); ax.set_ylabel("realized cost")
    ax.legend(fontsize=8); ax.set_title("Entropy tax: diffuse plans cost more, tickets amplify it")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_cost_epsilon.png"), dpi=150); plt.close(fig)

    # fig4 ghost flows: underweight lines funding other buys
    fig, ax = plt.subplots(figsize=(7.0, 4.5))
    gf = [results["grid"][str(e)]["ghost_flow"] for e in eps_grid]
    ent = [results["grid"][str(e)]["entropy"] for e in eps_grid]
    ax.plot(eps_grid, gf, "o-", color="#e45756", label="flow out of underweight lines")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xscale("log"); ax.set_xlabel("entropy temperature eps"); ax.set_ylabel("ghost flow (notional)")
    ax.legend(loc="upper left", fontsize=8)
    ax2 = ax.twinx(); ax2.plot(eps_grid, ent, "s--", color="#4c78a8", label="plan entropy")
    ax2.set_ylabel("Shannon entropy of P", color="#4c78a8"); ax2.grid(False)
    ax2.legend(loc="lower right", fontsize=8)
    ax.set_title("High temperature makes underweight positions fund other buys")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_ghost_flow.png"), dpi=150); plt.close(fig)

    # fig5 extreme target
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.6))
    for ax, eps in zip(axes, [3e-4, 3e-2]):
        P, _ = sinkhorn(w0, b_ext, eps)
        off = P - np.diag(np.diag(P))
        ax.imshow(off, cmap="Reds", vmin=0, vmax=max(off.max(), 1e-9))
        ax.set_xticks(range(n), labels); ax.set_yticks(range(n), labels)
        ax.set_xlabel("buy asset"); ax.set_ylabel("sell asset")
        st = results["extreme"][str(eps)]
        ax.set_title(f"eps={eps:g}: ghost flow {st['ghost_flow']:.4f}, trades {st['n_trades']}")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_extreme_target.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

## 七、结语

熵正则最优传输最容易被误读成「又快又稳的调仓器」。它确实快，也确实稳，但稳定的是目标函数，不是你的交易成本。行列边际把最终权重钉死，让人误以为温度无关紧要；实际上温度决定的是资金流动的拓扑。把换手下界、交易笔数和隐性流量一起摆到台面上，才能分清哪些交易是信号要求的，哪些只是算法为了平滑而多绕的路。

还有一个容易忽略的用法：运输矩阵本身就是一份可审计的执行指令。合规审查可以直接追问某笔买入由哪几笔卖出融资、经过了几层中转，而传统的净权重差分根本无法回答这类问题。若把风控规则写成对 P 的线性约束，例如限制单一卖出来源的融资比例、禁止某类资产互为中转，Sinkhorn 框架里只要把对应格子的代价抬高即可，不需要重写整个优化器。当然，任何对代价矩阵的修改都要重新跑温度扫描，因为约束生效的临界点同样随 ε 移动。
