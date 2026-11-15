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
