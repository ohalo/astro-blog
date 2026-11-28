---
title: "嵌套模拟估计希腊字母的偏差-方差闸门：外层路径套内层定价，为什么内层样本数决定 Delta 的方向性偏差、又在随机波动率下代价爆炸"
description: "复合期权解析 Delta 0.486575。嵌套模拟内层 2 条路径时估到 0.462687（偏差 -0.023888）、误判行权 32.41%；内层 500 条时偏差收窄到 -0.002083、误判 2.82%。价格端 J=8 正偏差 1.4236，M=100 后转为微小负偏差。固定 40 万内层抽样预算下，M=100 的 Delta RMSE 最低 0.004286，M=5 与 M=1000 分别恶化到 0.016759 与 0.018165。Heston 下 M=5 的 Delta 偏低 0.032242，M=1000 参考值 0.541295 需要 200 万次内层抽样。附完整代码与五张真实计算图。"
publishDate: '2026-11-28T19:35:00Z'
tags:
  - 量化
  - 蒙特卡洛
  - 衍生品定价
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

嵌套模拟的常识是「内层路径越多越准」。这句话对价格只说对一半，对希腊字母则更危险：内层噪声进入的不是最终的数值，而是行权判断本身。

本文用一个有解析解的复合期权做对照。外层在 0.25 年到期，标的必须先付一笔内层看涨；内层看涨 1 年到期、执行价 100，外层执行价 6。Geske 公式给出价格 4.265136、Delta 0.486575，临界股价 96.3301。用外层 2 万条路径、内层从 2 条扫到 500 条，Delta 从 0.462687 升到 0.484492，偏差从 -0.023888 收到 -0.002083；误判行权的路径比例从 32.41% 降到 2.82%。价格端更夸张：内层 2 条时期望价格 5.6887，比解析值高 1.4236，内层 100 条后偏差才降到 -0.0053。

但「多加内层」不是免费午餐。固定 40 万次内层抽样，M=100 时 Delta 的 RMSE 最低 0.004286；M=5 因为偏差大，RMSE 是 0.016759；M=1000 因为外层只剩 400 条路径，RMSE 反弹到 0.018165。换成 Heston 随机波动率，M=5 的 Delta 比 M=1000 参考值低 0.032242，而把内层提到 1000 条意味着 200 万次内层定价。

## 二、路径导数为什么不够

复合期权的收益是外层时刻的内层价值减去执行价再取正部。记内层价值为 C(S_t)，外层收益为 max(C(S_t)-K2, 0)。

路径wise 导数可以直接写出来：对每个外层路径，内层收益的导数只在那条内层路径确实实值不为零，乘上 S_T 除以初始股价。折现后就是

$$
\Delta=e^{-rT}\,E\big[\,\mathbf 1\{C(S_{t_1})>K_2\}\;\mathbf 1\{S_T>K_1\}\;\frac{S_T}{S_0}\,\big].
$$

问题出在第一个指示函数。真实 C 是条件期望，是 S_t 的平滑函数；嵌套模拟里只能用 M 条内层路径的均值代替它。这个均值在临界点附近会上下抖动，于是本该行权的路径被判成不行权，反之亦然。导数的漏项和增项都来自这批被误判的路径，而它们恰好集中在临界点附近——也就是 Delta 最大的地方。

价格端的偏差机制不同。max 是凸函数，内层均值是真实条件期望的无偏估计，但 E[max(噪声版本-K2, 0)] 大于 max(E[ C ]-K2, 0)，这是 Jensen 不等式。M 越小噪声越大，价格被抬得越高。两种偏差来源不同，方向也不必一致，这是理解结果的关键。

还有一个常见误判：把 Delta 的偏差与价格的偏差当成同一件事。价格偏差是 Jensen 项，正值为主且随 M 单调下降；Delta 偏差来自行权指示的误判，符号取决于被错纳与被错排路径上路径导数的均值对比。本例里被错纳的路径更集中在低 Delta 区域，所以净偏差为负。换一个外层执行价，错纳与错排的角色对调，偏差完全可能变号。因此不能靠「价格偏高就调低 Delta」这类经验规则补救。

## 三、解析基准与嵌套估计量

Geske 公式把复合期权写成两个二元正态分布函数之差。先解 S*，使外层时刻的内层看涨等于外层执行价 6，得到 S*=96.3301；再用相关系数 sqrt(t1/T)=0.5 的二元正态算两段概率。数值微分给出 Delta 0.486575，多种步长下的结果稳定。

嵌套估计量按外层模拟 S_t，内层从该状态重模拟 S_T。价格估计取内层收益均值的正部再折现；Delta 估计把内层均值替换成「实值指示乘 S_T/S0」的内层均值，并乘上 outer 的行权指示。所有随机种子固定，外层路径共用，因此 M 之间的差异是配对比较，不是独立抽样噪声。

![内层样本数与解析值收敛](/images/nested-simulation-greeks-bias/fig1_bs_convergence.png)

## 四、误判行权是偏差的真正来源

M=2 时内层均值只有两个收益平均，方差极大，32.41% 的外层路径行权判断与解析条件相反；此时 Delta 只有 0.462687，比真值低 4.91%。M=5 误判 23.50%、Delta 0.468452；M=20 误判 13.63%、Delta 0.477288；M=50 误判 8.86%、Delta 0.482288；到 M=500 误判降到 2.82%，Delta 0.484492，偏差只剩 -0.002083。样本里处于临界价 0.5 元内的外层路径占 8.06%，正是这批路径决定了误判率的下限。

价格端同样单调但方向相反。M=2 的价格 5.6887，偏差 +1.4236；M=5 是 4.9198，偏差 +0.6547；M=20 是 4.4205，偏差 +0.1554；M=50 是 4.3044，偏差 +0.0393；M=100 之后偏差在正负之间抖动，说明 Jensen 项已经被抽样误差盖过。换句话说，价格要收敛到小数点后两位需要几十条内层路径，Delta 要收敛到同样精度需要几百条。

![行权误判与相对偏差](/images/nested-simulation-greeks-bias/fig2_misclassification.png)

## 五、固定预算下的最优内层样本数

把预算固定成 40 万次内层抽样，外层路径数就是 40 万除以 M。每个配置重复 10 次，用解析 Delta 作参照算 RMSE。

M=5 时外层有 8 万条路径，但偏差 -0.016702 主导，RMSE 0.016759。M=20、50 依次降到 0.007632 和 0.005074。M=100 时外层 4000 条，偏差 -0.001057、RMSE 最低 0.004286。再往上，M=200 的外层只剩 2000 条，RMSE 回到 0.008102；M=1000 只有 400 条外层路径，RMSE 恶化到 0.018165，几乎和 M=5 一样差。

这个预算分配对工程实现也有影响。内层路径可以批量向量化，外层路径通常按块推进；当 M 从 100 提到 1000，外层从 4000 条掉到 400 条，估计量的标准误放大约三倍，而偏差只从 -0.001057 变成 -0.004133，方向还不一定有利。若系统只能承受固定总时长，宁可把外层路径数维持在几千条，再选一个让偏差小于统计噪声的 M，也不要单层追求精度。

这条 U 形曲线说明内层样本数不是越大越好。目标是最小化偏差平方加方差除以预算，最优 M 由收益在临界点附近的局部曲率决定。实务里应先在简化模型上扫出这个 M，再拿去跑昂贵的完整模型，而不是凭经验取一个固定数字。

![固定预算下的偏差与 RMSE](/images/nested-simulation-greeks-bias/fig3_fixed_budget.png)

## 六、随机波动率把代价放大

把内层换成 Heston 模型：初始方差 0.04、均值回复 1.5、长期方差 0.04、波动率之波动 0.6、相关系数 -0.5。外层 6 步到 0.25 年，内层 4 步到 1 年，2000 条外层路径，内层最多 1000 条。

以 M=1000 的 Delta 0.541295 为参照，M=5 给出 0.509053，偏低 0.032242，相对误差 5.96%；与参考估计的行权判断不一致的路径占 20.10%。M=20 把差距压到 0.004874、不一致 10.85%；M=50 是 0.541046，几乎对齐；M=200 是 0.541142，不一致降到 3.25%。代价从 1 万次内层抽样涨到 200 万次，正好 200 倍。

随机波动率放大代价有两层原因。一是方差路径本身随机，内层价值的条件分布更宽，临界点附近的误判更多；二是内层每步都要维护方差状态，单条路径的成本远高于常数波动率模型。若用同一个绝对预算，Heston 能承受的外层路径数会更少，方差项更早开始主导 RMSE。

![Heston 下的 Delta 偏差](/images/nested-simulation-greeks-bias/fig4_heston_bias.png)

![内层抽样成本](/images/nested-simulation-greeks-bias/fig5_cost_explosion.png)

## 七、实务边界

本文的内层估计量用的是路径导数，适用于连续扩散。若收益含跳跃或数字型不连续，路径导数为零，需要改用似然比，误差结构会再变一层。外层路径共用同一批内层随机数做配对比较，这是研究设置的便利；实盘里不同 M 的报价来自不同模拟，比较时还要计入独立抽样噪声。

还有一个容易忽略的口径问题：本文的 Delta 是对初始股价的敏感性，内层随机数固定。若改求对初始方差的 Vega，方差路径本身决定行权判断，误差结构与本文完全不同，不能把 M 的选择直接搬过去。

另一个边界是临界点厚度。本文外层执行价选得让约 8% 的路径落在临界价 0.5 元内，属于中等难度。若外层执行价更低，几乎所有路径都行权，误判率与偏差都会小很多；若执行价高到只有极少数深度实值路径行权，M 不足时会直接把整个期权判成零价值，偏差不再随 M 平滑衰减。上线前必须报告临界带占比，否则「偏差 0.002」这种数字没有可迁移性。

## 八、完整可运行代码

下面代码生成全部数据、五张图与 results.json，可独立运行。

```python
# -*- coding: utf-8 -*-
"""Nested simulation Greeks: inner-sample bias, variance trade-off and Heston cost."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq
from scipy.stats import multivariate_normal, norm

SEED = 20261128
S0 = 100.0
R = 0.03
SIG = 0.20
T = 1.0
T1 = 0.25
TAU = T - T1
K1 = 100.0
K2 = 6.0
N_OUT = 20000
M_GRID = [2, 5, 10, 20, 50, 100, 200, 500]
BUDGET = 400000
BUDGET_M = [5, 20, 50, 100, 200, 1000]
BUDGET_REPS = 10
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "nested-simulation-greeks-bias")

V0 = 0.04
KAPPA = 1.5
THETA = 0.04
XI = 0.60
RHO = -0.50
HN_OUT = 2000
H_M_MAX = 1000
H_INNER_STEPS = 4
H_OUTER_STEPS = 6


def bs_call(s, k, tt):
    d1 = (np.log(s / k) + (R + 0.5 * SIG ** 2) * tt) / (SIG * np.sqrt(tt))
    d2 = d1 - SIG * np.sqrt(tt)
    return s * norm.cdf(d1) - k * np.exp(-R * tt) * norm.cdf(d2)


def geske_price(s):
    rho = np.sqrt(T1 / T)
    cov = [[1.0, rho], [rho, 1.0]]
    a2 = (np.log(s / S_STAR) + (R - 0.5 * SIG ** 2) * T1) / (SIG * np.sqrt(T1))
    a1 = a2 + SIG * np.sqrt(T1)
    b1 = (np.log(s / K1) + (R + 0.5 * SIG ** 2) * T) / (SIG * np.sqrt(T))
    b2 = b1 - SIG * np.sqrt(T)
    p = multivariate_normal.cdf([a1, b1], cov=cov)
    q = multivariate_normal.cdf([a2, b2], cov=cov)
    return s * p - K1 * np.exp(-R * T) * q - K2 * np.exp(-R * T1) * norm.cdf(a2)


def bs_nested(rng, n_out, m, want_exact=False):
    z1 = rng.standard_normal(n_out)
    s1 = S0 * np.exp((R - 0.5 * SIG ** 2) * T1 + SIG * np.sqrt(T1) * z1)
    zi = rng.standard_normal((n_out, m))
    st = s1[:, None] * np.exp((R - 0.5 * SIG ** 2) * TAU + SIG * np.sqrt(TAU) * zi)
    inner_pay = np.maximum(st - K1, 0.0)
    inner_val = np.exp(-R * TAU) * inner_pay.mean(axis=1)
    inner_dlt = np.exp(-R * TAU) * ((st > K1).astype(float) * st / S0).mean(axis=1)
    take = (inner_val > K2).astype(float)
    price = np.exp(-R * T1) * np.maximum(inner_val - K2, 0.0).mean()
    delta = np.exp(-R * T1) * (take * inner_dlt).mean()
    result = {
        "price": float(price),
        "delta": float(delta),
        "delta_se": float(np.exp(-R * T1) * np.std(take * inner_dlt, ddof=1) / np.sqrt(n_out)),
        "price_se": float(np.exp(-R * T1) * np.std(np.maximum(inner_val - K2, 0.0), ddof=1) / np.sqrt(n_out)),
    }
    if want_exact:
        exact_val = bs_call(s1, K1, TAU)
        exact_take = (exact_val > K2)
        mis = float(np.mean(take.astype(bool) != exact_take))
        result["misclassification"] = mis
        result["near_boundary_share"] = float(np.mean(np.abs(exact_val - K2) < 0.5))
    return result


def heston_nested(rng, n_out, m_grid, dt_out, dt_in, want_inner_paths=False):
    z2o = rng.standard_normal((n_out, H_OUTER_STEPS))
    zio = rng.standard_normal((n_out, H_OUTER_STEPS))
    z1o = RHO * z2o + np.sqrt(1.0 - RHO ** 2) * zio
    s = np.full(n_out, S0)
    v = np.full(n_out, V0)
    for k in range(H_OUTER_STEPS):
        vp = np.maximum(v, 0.0)
        s = s * np.exp((R - 0.5 * vp) * dt_out + np.sqrt(vp * dt_out) * z1o[:, k])
        v = v + KAPPA * (THETA - vp) * dt_out + XI * np.sqrt(vp) * np.sqrt(dt_out) * z2o[:, k]
    s1 = s.copy()
    v1 = np.maximum(v, 0.0)

    mmax = max(m_grid)
    z2i = rng.standard_normal((n_out, mmax, H_INNER_STEPS))
    zii = rng.standard_normal((n_out, mmax, H_INNER_STEPS))
    z1i = RHO * z2i + np.sqrt(1.0 - RHO ** 2) * zii
    si = np.broadcast_to(s1[:, None], (n_out, mmax)).copy()
    vi = np.broadcast_to(v1[:, None], (n_out, mmax)).copy()
    growth = np.ones((n_out, mmax))
    for k in range(H_INNER_STEPS):
        vp = np.maximum(vi, 0.0)
        factor = np.exp((R - 0.5 * vp) * dt_in + np.sqrt(vp * dt_in) * z1i[:, :, k])
        si = si * factor
        growth = growth * factor
        vi = vi + KAPPA * (THETA - vp) * dt_in + XI * np.sqrt(vp) * np.sqrt(dt_in) * z2i[:, :, k]
    pay = np.maximum(si - K1, 0.0)
    disc_in = np.exp(-R * TAU)
    counts = np.arange(1, mmax + 1, dtype=float)[None, :]
    inner_val = disc_in * np.cumsum(pay, axis=1) / counts
    inner_dlt = disc_in * np.cumsum((si > K1).astype(float) * growth * s1[:, None] / S0, axis=1) / counts
    ref_take = (inner_val[:, mmax - 1] > K2)
    out = {}
    for m in m_grid:
        iv = inner_val[:, m - 1]
        idl = inner_dlt[:, m - 1]
        tk = (iv > K2)
        out[str(m)] = {
            "price": float(np.exp(-R * T1) * np.maximum(iv - K2, 0.0).mean()),
            "delta": float(np.exp(-R * T1) * (tk.astype(float) * idl).mean()),
            "misclassification_vs_max": float(np.mean(tk != ref_take)),
        }
    return out


def main():
    global S_STAR
    S_STAR = brentq(lambda s: bs_call(s, K1, TAU) - K2, 1.0, 300.0)
    analytic_price = float(geske_price(S0))
    h = 1e-4
    analytic_delta = float((geske_price(S0 + h) - geske_price(S0 - h)) / (2.0 * h))

    rng = np.random.default_rng(SEED)
    z1 = rng.standard_normal(N_OUT)
    s1 = S0 * np.exp((R - 0.5 * SIG ** 2) * T1 + SIG * np.sqrt(T1) * z1)
    zi = rng.standard_normal((N_OUT, max(M_GRID)))
    st = s1[:, None] * np.exp((R - 0.5 * SIG ** 2) * TAU + SIG * np.sqrt(TAU) * zi)
    pay = np.maximum(st - K1, 0.0)
    counts = np.arange(1, max(M_GRID) + 1, dtype=float)[None, :]
    disc_in = np.exp(-R * TAU)
    inner_vals = disc_in * np.cumsum(pay, axis=1) / counts
    inner_dlts = disc_in * np.cumsum((st > K1).astype(float) * st / S0, axis=1) / counts
    exact_val = bs_call(s1, K1, TAU)
    exact_take = exact_val > K2
    study = {}
    for m in M_GRID:
        iv = inner_vals[:, m - 1]
        idl = inner_dlts[:, m - 1]
        tk = iv > K2
        delta_path = np.exp(-R * T1) * tk.astype(float) * idl
        study[str(m)] = {
            "price": float(np.exp(-R * T1) * np.maximum(iv - K2, 0.0).mean()),
            "delta": float(delta_path.mean()),
            "delta_se": float(np.std(delta_path, ddof=1) / np.sqrt(N_OUT)),
            "misclassification": float(np.mean(tk != exact_take)),
            "near_boundary_share": float(np.mean(np.abs(exact_val - K2) < 0.5)),
        }

    budget = {}
    for m in BUDGET_M:
        n_out = BUDGET // m
        ds = []
        ps = []
        for rep in range(BUDGET_REPS):
            res = bs_nested(rng, n_out, m)
            ds.append(res["delta"] - analytic_delta)
            ps.append(res["price"] - analytic_price)
        budget[str(m)] = {
            "n_outer": int(n_out),
            "delta_bias": round(float(np.mean(ds)), 6),
            "delta_rmse": round(float(np.sqrt(np.mean(np.square(ds)))), 6),
            "price_bias": round(float(np.mean(ps)), 6),
        }

    h_rng = np.random.default_rng(SEED + 31)
    heston = heston_nested(h_rng, HN_OUT, [5, 20, 50, 200, H_M_MAX],
                           T1 / H_OUTER_STEPS, TAU / H_INNER_STEPS)
    ref_delta = heston[str(H_M_MAX)]["delta"]

    # Figure 1: convergence of nested price and delta with inner sample size
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(M_GRID, [study[str(m)]["price"] - analytic_price for m in M_GRID], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Nested price bias (BS compound call)")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("price minus analytic")
    axes[1].plot(M_GRID, [study[str(m)]["delta"] - analytic_delta for m in M_GRID], "o-", color="#3b5b8c")
    axes[1].axhline(0.0, color="#555555", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_title("Nested delta bias (BS compound call)")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("delta minus analytic")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_bs_convergence.png"), dpi=150)
    plt.close(fig)

    # Figure 2: exercise-boundary misclassification drives the delta bias
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(M_GRID, [study[str(m)]["misclassification"] * 100.0 for m in M_GRID],
                 "o-", color="#8e6cbb")
    axes[0].set_xscale("log")
    axes[0].set_title("Wrong exercise decisions")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("misclassified outer paths (%)")
    axes[1].plot(M_GRID,
                 [study[str(m)]["delta"] / analytic_delta - 1.0 for m in M_GRID],
                 "o-", color="#e07b39", label="delta")
    axes[1].plot(M_GRID,
                 [study[str(m)]["price"] / analytic_price - 1.0 for m in M_GRID],
                 "s-", color="#3b5b8c", label="price")
    axes[1].axhline(0.0, color="#555555", ls="--")
    axes[1].set_xscale("log")
    axes[1].set_title("Relative bias")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("relative error")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_misclassification.png"), dpi=150)
    plt.close(fig)

    # Figure 3: fixed budget variance penalty
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    axes[0].plot(BUDGET_M, [budget[str(m)]["delta_bias"] for m in BUDGET_M], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Bias at fixed inner-draw budget")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("delta bias")
    axes[1].plot(BUDGET_M, [budget[str(m)]["delta_rmse"] for m in BUDGET_M], "o-", color="#3b5b8c")
    axes[1].set_xscale("log")
    axes[1].set_title("RMSE at fixed inner-draw budget")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("delta RMSE")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_fixed_budget.png"), dpi=150)
    plt.close(fig)

    # Figure 4: Heston nested delta versus inner sample size
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    hm = [5, 20, 50, 200]
    axes[0].plot(hm, [heston[str(m)]["delta"] - ref_delta for m in hm], "o-", color="#c0392b")
    axes[0].axhline(0.0, color="#555555", ls="--")
    axes[0].set_xscale("log")
    axes[0].set_title("Heston delta bias versus M=1000 reference")
    axes[0].set_xlabel("inner paths M")
    axes[0].set_ylabel("delta difference")
    axes[1].plot(hm, [heston[str(m)]["misclassification_vs_max"] * 100.0 for m in hm],
                 "o-", color="#8e6cbb")
    axes[1].set_xscale("log")
    axes[1].set_title("Exercise decisions versus reference")
    axes[1].set_xlabel("inner paths M")
    axes[1].set_ylabel("disagreeing paths (%)")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_heston_bias.png"), dpi=150)
    plt.close(fig)

    # Figure 5: inner-draw cost explosion
    fig, ax = plt.subplots(figsize=(9, 4))
    ms = [5, 20, 50, 200, H_M_MAX]
    ax.plot(ms, [HN_OUT * m for m in ms], "o-", color="#e07b39")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("Inner path evaluations in the Heston experiment")
    ax.set_xlabel("inner paths M")
    ax.set_ylabel("outer x inner draws")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_cost_explosion.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "s0": S0,
        "strike_inner": K1,
        "strike_outer": K2,
        "t1": T1,
        "maturity": T,
        "sigma": SIG,
        "rate": R,
        "critical_s_star": round(S_STAR, 6),
        "analytic_price": round(analytic_price, 6),
        "analytic_delta": round(analytic_delta, 6),
        "n_outer": N_OUT,
        "m_grid": M_GRID,
        "nested_price": {k: round(v["price"], 6) for k, v in study.items()},
        "nested_delta": {k: round(v["delta"], 6) for k, v in study.items()},
        "nested_delta_se": {k: round(v["delta_se"], 6) for k, v in study.items()},
        "price_bias": {k: round(v["price"] - analytic_price, 6) for k, v in study.items()},
        "delta_bias": {k: round(v["delta"] - analytic_delta, 6) for k, v in study.items()},
        "misclassification": {k: round(v["misclassification"], 6) for k, v in study.items()},
        "near_boundary_share": round(study["500"]["near_boundary_share"], 6),
        "budget": budget,
        "budget_total": BUDGET,
        "budget_reps": BUDGET_REPS,
        "heston_ref_m": H_M_MAX,
        "heston_ref_delta": round(ref_delta, 6),
        "heston_delta": {k: round(v["delta"], 6) for k, v in heston.items()},
        "heston_misclassification": {k: round(v["misclassification_vs_max"], 6) for k, v in heston.items()},
        "heston_inner_draws": {k: HN_OUT * int(k) for k in heston},
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
```

