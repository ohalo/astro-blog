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
