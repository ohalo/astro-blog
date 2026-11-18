# -*- coding: utf-8 -*-
"""Wald / 似然比 / Score 三大渐近等价检验：有限样本下为何给出不同结论。

本文全部数字的唯一来源。运行本脚本会：
1. 在 AR(1) 条件高斯似然下把 sigma^2 profile 掉，解析得到三个统计量都是
   同一个量 u 的函数：LM = n*u/(1+u) <= LR = n*log(1+u) <= Wald = n*u；
2. 用 20000 次 H0 蒙特卡洛核对三者的经验分布、95% 分位与实际 size；
3. 扫描 n=30..1000 与正态/t(5) 两种新息，看 size 随样本量收敛的速度；
4. 用一族光滑正则重参数化 phi = psi/(1-a*psi) 重跑三个检验，
   核对 LR 与 Fisher 版 LM 不变、Wald 随 a 变化（解析比 (1+a*phi_hat)^2），
   并统计结论翻转率；
5. 共同随机数下画功效曲线，观察 Score 检验在远离原假设时的非单调性；
6. 画 5 张图到 public/images/wald-lr-score-test-divergence/，
   全部统计量写入同目录 results.json。
"""
import json
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from scipy.optimize import minimize_scalar

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 20261118
_OUT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "public", "images", "wald-lr-score-test-divergence",
)
os.makedirs(_OUT, exist_ok=True)

ALPHA = 0.05
CRIT = float(stats.chi2.ppf(1 - ALPHA, 1))  # 3.841458820694124
BURN = 60


def _save(fig, name):
    path = os.path.join(_OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def innovations(nrep, total, rng, innov="normal", df=5):
    if innov == "normal":
        return rng.standard_normal((nrep, total))
    if innov == "t5":
        return rng.standard_t(df, size=(nrep, total)) / math.sqrt(df / (df - 2.0))
    if innov == "garch":
        # GARCH(1,1) 条件异方差：omega=0.05, a=0.10, b=0.85，
        # 持久性 0.95、无条件方差 omega/(1-a-b)=1.0，与前两种新息同尺度。
        z = rng.standard_normal((nrep, total))
        s2 = np.ones(nrep)
        ep2 = np.ones(nrep)
        E = np.empty((nrep, total))
        om, ga, gb = 0.05, 0.10, 0.85
        for t in range(total):
            s2 = om + ga * ep2 + gb * s2
            e = np.sqrt(s2) * z[:, t]
            E[:, t] = e
            ep2 = e * e
        return E
    raise ValueError(innov)


def simulate(phi, n, E, burn=BURN):
    """给定同一批新息 E（共同随机数），模拟 AR(1) 并返回三个充分统计量。

    S0 = sum x_{t-1}^2, S1 = sum x_{t-1} x_t, S2 = sum x_t^2。
    丢弃 burn 步预热，避免初值 0 带来的非平稳偏差。
    """
    total = E.shape[1]
    x = np.zeros(E.shape[0])
    for t in range(total):
        x = phi * x + E[:, t]
        if t == burn - 1:
            x0 = x.copy()
        if t >= burn:
            if t == burn:
                s0 = x0 * x0
                s1 = x0 * x
                s2 = x * x
            else:
                s0 = s0 + prev * prev
                s1 = s1 + prev * x
                s2 = s2 + x * x
        prev = x
    return s0, s1, s2


def three_stats(s0, s1, s2, n, phi0=0.0):
    """profile 掉 sigma^2 后三个检验统计量的解析式。"""
    ph = s1 / s0
    rss_u = s2 - s1 * s1 / s0
    rss_r = s2 - 2.0 * phi0 * s1 + phi0 * phi0 * s0
    u = (ph - phi0) ** 2 * s0 / rss_u
    W = n * u
    LR = n * np.log(rss_r / rss_u)
    LM = n * (ph - phi0) ** 2 * s0 / rss_r
    return ph, W, LR, LM, rss_u, rss_r


def g_a(psi, a):
    return psi / (1.0 - a * psi)


def psi_of_phi(phi, a):
    return phi / (1.0 + a * phi)


def reparam_wald(phi_hat, s0, sig2_u, a):
    """Wald 在 psi 坐标系下的值：I_psi(psi_hat) = g'(psi_hat)^2 * s0 / sig2_u。

    g'(psi) = 1/(1-a*psi)^2，代入 psi_hat 得 (1+a*phi_hat)^2。
    """
    ps_hat = psi_of_phi(phi_hat, a)
    gp = 1.0 / (1.0 - a * ps_hat) ** 2
    return (ps_hat ** 2) * (gp ** 2) * s0 / sig2_u


def obs_info_lm(x_series, a, phi0=0.0):
    """用观测信息（profile 似然的数值 Hessian）算的 LM，检验不变性边界。"""
    n = len(x_series) - 1
    xl = x_series[:-1]
    xr = x_series[1:]
    s0 = float((xl * xl).sum())
    s1 = float((xl * xr).sum())
    s2 = float((xr * xr).sum())
    rss_r = s2 - 2.0 * phi0 * s1 + phi0 * phi0 * s0
    sig2_r = rss_r / n

    def ell(psi):
        ph = g_a(psi, a)
        rss = s2 - 2.0 * ph * s1 + ph * ph * s0
        return -0.5 * n * math.log(rss / n)

    h = 1e-5
    d1 = (ell(h) - ell(-h)) / (2.0 * h)
    d2 = (ell(h) - 2.0 * ell(0.0) + ell(-h)) / (h * h)
    info_obs = -d2
    info_fisher = s0 / sig2_r  # g'(0)=1，链式后不变
    return float(d1 * d1 / info_obs), float(d1 * d1 / info_fisher), float(info_obs)


def main():
    rng = np.random.default_rng(SEED)
    results = {"seed": SEED, "alpha": ALPHA, "chi2_crit": CRIT, "figs": []}

    # ---- 实验 1：H0 下三统计量的经验分布 ----
    n0, rep0 = 252, 20000
    E0 = innovations(rep0, n0 + BURN, rng)
    s0, s1, s2 = simulate(0.0, n0, E0)
    ph, W, LR, LM, rss_u, rss_r = three_stats(s0, s1, s2, n0)
    dist = {}
    for key, v in [("wald", W), ("lr", LR), ("lm", LM)]:
        dist[key] = {
            "mean": float(v.mean()),
            "q95": float(np.quantile(v, 0.95)),
            "q99": float(np.quantile(v, 0.99)),
            "size": float((v > CRIT).mean()),
        }
    results["h0_distribution"] = {
        "n": n0, "reps": rep0, **dist,
        "ordering_violations": int(((LM > LR + 1e-9) | (LR > W + 1e-9)).sum()),
        "max_lm_minus_lr": float((LM - LR).max()),
        "max_lr_minus_wald": float((LR - W).max()),
    }

    # 图1：三统计量尾部直方图 + chi2(1) 密度 + 经验 95% 分位
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.9), sharex=True)
    grid = np.linspace(0.001, 9.0, 400)
    dens = stats.chi2.pdf(grid, 1)
    for ax, (key, v, label, color) in zip(
        axes,
        [("wald", W, "Wald = n·u", "#c0392b"),
         ("lr", LR, "LR = n·log(1+u)", "#e67e22"),
         ("lm", LM, "LM = n·u/(1+u)", "#2980b9")],
    ):
        ax.hist(v, bins=110, range=(0, 9), density=True, color=color, alpha=0.55)
        ax.plot(grid, dens, "k-", lw=1.5, label="χ²(1) 理论密度")
        ax.axvline(CRIT, color="k", ls="--", lw=1.3, label=f"名义临界 {CRIT:.3f}")
        q = dist[key]["q95"]
        ax.axvline(q, color=color, ls="-", lw=2.0, label=f"经验95%分位 {q:.3f}")
        ax.set_title(f"{label}\n实际 size = {dist[key]['size']:.4f}", fontsize=10)
        ax.set_xlabel("统计量取值")
        ax.set_xlim(0, 9)
        ax.set_ylim(0, 0.85)
        ax.legend(fontsize=7, loc="upper right")
    axes[0].set_ylabel("密度")
    fig.suptitle("图1：同一个 u 的三种函数——Wald 最激进、LM 最保守，位移方向确定", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig1_h0_distributions.png"))

    # ---- 实验 2：三个统计量作为 u 的解析函数 ----
    us = np.linspace(0, 0.06, 300)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    ax.plot(us, n0 * us / (1 + us), "-", color="#2980b9", lw=2.2, label="LM = n·u/(1+u)")
    ax.plot(us, n0 * np.log(1 + us), "-", color="#e67e22", lw=2.2, label="LR = n·log(1+u)")
    ax.plot(us, n0 * us, "-", color="#c0392b", lw=2.2, label="Wald = n·u")
    ax.fill_between(us, n0 * us / (1 + us), n0 * us, color="#95a5a6", alpha=0.18,
                    label="LM 与 Wald 的分歧带")
    ax.axhline(CRIT, color="k", ls="--", lw=1.4, label=f"χ²(1) 5% 临界 {CRIT:.3f}")
    crossing = {}
    for name, f, c in [("Wald", lambda u: n0 * u, "#c0392b"),
                       ("LR", lambda u: n0 * math.log(1 + u), "#e67e22"),
                       ("LM", lambda u: n0 * u / (1 + u), "#2980b9")]:
        lo, hi = 0.0, 0.06
        for _ in range(80):
            mid = (lo + hi) / 2
            if f(mid) < CRIT:
                lo = mid
            else:
                hi = mid
        crossing[name] = float((lo + hi) / 2)
        ax.plot([crossing[name]], [CRIT], "o", color=c, ms=8)
    ax.text(0.02, 0.96,
            "三者跨过 5% 临界线所需的 u：\n"
            f"Wald  {crossing['Wald']:.6f}\n"
            f"LR      {crossing['LR']:.6f}\n"
            f"LM      {crossing['LM']:.6f}",
            transform=ax.transAxes, fontsize=9, va="top",
            bbox=dict(boxstyle="round", fc="white", ec="#bdc3c7", alpha=0.92))
    results["crossing_u"] = crossing
    ax.set_xlabel("u = (φ̂−φ0)²·Σx²/RSS(φ̂)（与参数化无关的充分统计量之比）")
    ax.set_ylabel("统计量取值（n=252）")
    ax.set_title("图2：LM ≤ LR ≤ Wald 是恒等式，不是抽样误差——三者跨临界线的位置必然不同")
    ax.legend(fontsize=8)
    fig.tight_layout()
    results["figs"].append(_save(fig, "fig2_analytic_ordering.png"))

    # ---- 实验 3：size 随 n 收敛；正态 / t(5) / GARCH(1,1) ------
    ns = [30, 60, 120, 252, 504, 1000]
    rep_s = 6000
    size_scan = {}
    for innov in ["normal", "t5", "garch"]:
        per = {}
        for n in ns:
            E = innovations(rep_s, n + BURN, rng, innov=innov)
            ss0, ss1, ss2 = simulate(0.0, n, E)
            _, w, l, m, _, _ = three_stats(ss0, ss1, ss2, n)
            per[f"n{n}"] = {
                "wald": float((w > CRIT).mean()),
                "lr": float((l > CRIT).mean()),
                "lm": float((m > CRIT).mean()),
            }
        size_scan[innov] = per
    results["size_vs_n"] = size_scan
    results["size_gaps"] = {
        "gap_wald_minus_lm_n30": size_scan["normal"]["n30"]["wald"] - size_scan["normal"]["n30"]["lm"],
        "gap_wald_minus_lm_n252": size_scan["normal"]["n252"]["wald"] - size_scan["normal"]["n252"]["lm"],
        "gap_wald_minus_lm_n1000": size_scan["normal"]["n1000"]["wald"] - size_scan["normal"]["n1000"]["lm"],
    }

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.2), sharey=False)
    for ax, innov, title in zip(
        axes, ["normal", "t5", "garch"],
        ["正态新息（似然设定正确）", "t(5) 厚尾（尺度仍正确）", "GARCH(1,1) 条件异方差"]):
        for key, label, color, mk in [("wald", "Wald", "#c0392b", "o"),
                                     ("lr", "LR", "#e67e22", "s"),
                                     ("lm", "LM / Score", "#2980b9", "^")]:
            ys = [size_scan[innov][f"n{n}"][key] for n in ns]
            ax.plot(range(len(ns)), ys, mk + "-", color=color, lw=1.9, label=label)
        ax.axhline(ALPHA, color="k", ls="--", lw=1.3, label="名义 0.05")
        ax.set_xticks(range(len(ns)))
        ax.set_xticklabels(ns)
        ax.set_xlabel("样本量 n")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("H0 实际拒绝率")
    fig.suptitle("图3：分歧只在小样本显著；厚尾让三家一起保守，异方差才把 size 顶上去", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig3_size_vs_n.png"))

    # ---- 实验 4：重参数化不变性 ----
    # 固定一条 phi_true=0.15 的样本（保留原始序列以便数值重跑 profile 似然）
    n1, rep1 = 252, 4000
    E1 = innovations(rep1, n1 + BURN, rng)
    ss0, ss1, ss2 = simulate(0.15, n1, E1)
    ph1, w1, l1, m1, ru1, rr1 = three_stats(ss0, ss1, ss2, n1)
    sig2u1 = ru1 / n1
    # 取一条 phi_hat 明显为正、Wald 恰好未过临界线的样本
    cand = np.where((ph1 > 0.08) & (w1 < CRIT))[0]
    idx = int(cand[np.argmin(np.abs(w1[cand] - 3.0))])
    total = E1.shape[1]
    xs = np.zeros(total + 1)
    for t in range(total):
        xs[t + 1] = 0.15 * xs[t] + E1[idx, t]
    x_demo = xs[BURN:]
    d0, d1, d2 = simulate(0.15, n1, E1[idx:idx + 1])
    p_demo, W_demo, LR_demo, LM_demo, ru_demo, rr_demo = three_stats(d0, d1, d2, n1)
    su_demo = ru_demo / n1

    a_grid = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0]
    rep_rows = []
    flip_rates = []
    for a in a_grid:
        Wp = float(reparam_wald(p_demo, d0, su_demo, a)[0])
        # 数值重跑：以 psi 为参数极大化同一条 profile 似然
        def neg_ell(psi):
            ph = g_a(psi, a)
            rss = d2 - 2.0 * ph * d1 + ph * ph * d0
            return 0.5 * n1 * math.log(float((rss / n1)[0]))
        opt = minimize_scalar(neg_ell, bounds=(-0.4, 0.4), method="bounded",
                              options={"xatol": 1e-12})
        LRp = 2.0 * (neg_ell(0.0) - opt.fun)
        LM_obs, LM_fisher, info_obs = obs_info_lm(x_demo, a)
        # MC 翻转率：phi 坐标系不拒绝、psi 坐标系拒绝
        Wp_all = reparam_wald(ph1, ss0, sig2u1, a)
        flip = float(((w1 < CRIT) & (Wp_all > CRIT)).mean())
        flip_rates.append(flip)
        rep_rows.append({
            "a": a, "wald_psi": Wp, "lr_psi": float(LRp),
            "lm_fisher_psi": LM_fisher, "lm_obs_psi": LM_obs,
            "obs_info": info_obs,
            "wald_ratio_to_closed_form": float(Wp / (W_demo[0] * (1.0 + a * p_demo[0]) ** 2)),
            "psi_hat_numeric": float(opt.x), "psi_hat_closed": float(psi_of_phi(p_demo[0], a)),
            "flip_rate": flip,
        })
    results["reparam"] = {
        "n": n1, "reps_mc": rep1, "phi_true": 0.15,
        "demo": {
            "phi_hat": float(p_demo[0]), "wald_phi": float(W_demo[0]),
            "lr_phi": float(LR_demo[0]), "lm_phi": float(LM_demo[0]),
            "sample_index": idx,
        },
        "a_grid": a_grid, "rows": rep_rows,
        "lr_max_dev": max(abs(r["lr_psi"] - float(LR_demo[0])) for r in rep_rows),
        "lm_fisher_max_dev": max(abs(r["lm_fisher_psi"] - float(LM_demo[0])) for r in rep_rows),
        "lm_obs_max_dev": max(abs(r["lm_obs_psi"] - float(LM_demo[0])) for r in rep_rows),
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.3))
    ax = axes[0]
    ax.plot(a_grid, [r["wald_psi"] for r in rep_rows], "o-", color="#c0392b", lw=2.1,
            label="Wald(ψ) 随坐标系变化")
    ax.plot(a_grid, [r["lr_psi"] for r in rep_rows], "s-", color="#e67e22", lw=1.8,
            label="LR(ψ) 数值重跑")
    ax.plot(a_grid, [r["lm_fisher_psi"] for r in rep_rows], "^-", color="#2980b9", lw=1.8,
            label="LM(ψ) Fisher 信息版")
    ax.plot(a_grid, [r["lm_obs_psi"] for r in rep_rows], "d--", color="#8e44ad", lw=1.5,
            label="LM(ψ) 观测信息版")
    ax.axhline(CRIT, color="k", ls="--", lw=1.3, label="χ²(1) 5% 临界")
    ax.set_xlabel("重参数化旋钮 a：φ = ψ/(1−aψ)")
    ax.set_ylabel("统计量取值")
    ax.set_title(f"同一份样本 φ̂={p_demo[0]:.4f}，Wald(φ)={W_demo[0]:.3f}", fontsize=10)
    ax.legend(fontsize=7.5)
    ax = axes[1]
    ax.plot(a_grid, flip_rates, "o-", color="#16a085", lw=2.2)
    ax.set_xlabel("重参数化旋钮 a")
    ax.set_ylabel("结论翻转的比例")
    ax.set_title("φ_true=0.15 的 4000 条样本里，\n「φ 系不拒绝、ψ 系拒绝」的占比", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_ylim(0, max(flip_rates) * 1.25 + 0.01)
    fig.suptitle("图4：换坐标系只改 Wald——同一数据、同一 H0、结论可以翻转", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig4_reparameterization.png"))

    # ---- 实验 5：共同随机数下的功效曲线 ----
    phis = np.round(np.linspace(0.0, 0.30, 13), 4)
    rep_p = 6000
    Ep = innovations(rep_p, n0 + BURN, rng)
    power = {}
    for phi in phis:
        ps0, ps1, ps2 = simulate(float(phi), n0, Ep)
        _, wp, lp, mp, _, _ = three_stats(ps0, ps1, ps2, n0)
        power[f"phi{phi:.4f}"] = {
            "wald": float((wp > CRIT).mean()),
            "lr": float((lp > CRIT).mean()),
            "lm": float((mp > CRIT).mean()),
            "mean_lm": float(mp.mean()),
        }
    gaps = [power[f"phi{p:.4f}"]["wald"] - power[f"phi{p:.4f}"]["lm"] for p in phis]
    lm_last = power[f"phi{phis[-1]:.4f}"]["mean_lm"]
    results["power_curve"] = {
        "phis": [float(p) for p in phis], **power,
        "max_gap_wald_minus_lm": float(max(gaps)),
        "lm_mean_at_max_phi": lm_last,
        "lm_theoretical_upper_bound": n0,
    }

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.2))
    ax = axes[0]
    for key, label, color, mk in [("wald", "Wald", "#c0392b", "o"),
                                  ("lr", "LR", "#e67e22", "s"),
                                  ("lm", "LM / Score", "#2980b9", "^")]:
        ys = [power[f"phi{p:.4f}"][key] for p in phis]
        ax.plot(phis, ys, mk + "-", color=color, lw=2.0, label=label)
    ax.axhline(ALPHA, color="k", ls="--", lw=1.2)
    ax.set_xlabel("真实 φ（H0: φ=0）")
    ax.set_ylabel("拒绝率 / 功效")
    ax.set_title(f"n={n0}，6000 条共同随机数路径", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(phis, gaps, "o-", color="#8e44ad", lw=2.1, label="Wald 功效 − LM 功效")
    ax.axhline(0.0, color="k", ls="--", lw=1.2)
    ax.set_xlabel("真实 φ（H0: φ=0）")
    ax.set_ylabel("功效差")
    ax.set_title(f"三家功效最大差仅 {max(gaps):.4f}；\n"
                 f"LM 均值 {lm_last:.2f} 远未触及理论上界 n={n0}", fontsize=10)
    ax.legend(fontsize=8)
    fig.suptitle("图5：正确设定的线性模型里，三家的功效几乎重合", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    results["figs"].append(_save(fig, "fig5_power_curve.png"))

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print(json.dumps(results, ensure_ascii=False))
    return results


if __name__ == "__main__":
    main()
