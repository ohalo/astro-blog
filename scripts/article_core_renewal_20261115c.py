# -*- coding: utf-8 -*-
"""更新过程检查悖论：随机时点观察到的"平均间隔"为什么系统性偏大。

本文全部数字的唯一来源。运行本脚本会：
1. 用对数正态间隔生成更新过程，蒙特卡洛测「检查悖论」——随机观察时刻所
   落入区间长度的期望 E[L*]，与理论 E[L*]=E[L^2]/E[L]=E[L](1+CV^2) 对拍；
2. 扫 CV：不同分散度下偏差倍数 1+CV^2 的仿真-理论曲线；
3. 残差寿命分布 vs 平衡分布理论 CDF 对拍（K-S 偏离）；
4. 固定日历采样估计平均挂单间隔：朴素均值偏差随样本量不收敛（≈CV²），
   而长度偏差的无偏校正——观测样本的调和均值——偏差收敛到 0；
5. 持仓快照的截面平均被长度偏差放大的倍数；
6. 画 5 张图到 public/images/renewal-process-inspection-paradox/；
7. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 202611154
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "renewal-process-inspection-paradox"
)
os.makedirs(_OUT, exist_ok=True)

SIG0 = 0.55  # 主实验用的对数正态形状参数


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


def lognormal_params(sigma):
    """L=exp(sigma*Z-mu0) 归一到均值 1：mu0=sigma^2/2。返回 (mu1, mu2, cv2)。"""
    mu0 = sigma ** 2 / 2.0
    mu1 = 1.0
    mu2 = np.exp(sigma ** 2)  # E[L^2]，均值已归一
    return mu1, mu2, mu2 - 1.0


def draw_L(rng, sigma, n):
    return np.exp(sigma * rng.standard_normal(n) - sigma ** 2 / 2.0)


def inspected_lengths(rng, sigma, n_obs, horizon=None):
    """跨实现观察：每 400 个观察时刻用一条独立的长窗口（2 万天）更新过程，

    长窗口保证时间平均遍历收敛，跨实现平均消除单条实现的随机涨落。
    返回观察时刻落入区间的 (长度, 起点, 已老化时长 age, 残差 resid)。
    """
    w = 20000.0
    o_L, o_S, o_A, o_R = [], [], [], []
    rem = n_obs
    while rem > 0:
        g = min(rem, 400)
        rem -= g
        L = draw_L(rng, sigma, int(2.5 * w) + 10)
        S = np.concatenate(([0.0], np.cumsum(L)))
        while S[-1] < 2.5 * w + 10.0:  # 覆盖保证：观测窗 [1.5w, 2.5w)
            L = np.concatenate([L, draw_L(rng, sigma, int(2.5 * w))])
            S = np.concatenate(([0.0], np.cumsum(L)))
        t = rng.uniform(1.5 * w, 2.5 * w, g)
        i = np.searchsorted(S, t, side="right") - 1
        o_L.append(L[i]); o_S.append(S[i] - 1.5 * w)
        o_A.append(t - S[i]); o_R.append(S[i + 1] - t)
    return (np.concatenate(o_L), np.concatenate(o_S),
            np.concatenate(o_A), np.concatenate(o_R))


def exp1_inspection(rng):
    """检查悖论主实验：观察区间长度 vs 原始间隔分布。"""
    mu1, mu2, cv2 = lognormal_params(SIG0)
    L_true = draw_L(rng, SIG0, 400_000)
    L_ins, _, _, _ = inspected_lengths(rng, SIG0, 400_000, 300.0)
    stat = {
        "sigma": SIG0,
        "E_L": float(L_true.mean()),
        "E_L_inspected": float(L_ins.mean()),
        "ratio_sim": float(L_ins.mean() / L_true.mean()),
        "ratio_theory": float(mu2 / mu1),  # = 1+CV^2
        "cv2_theory": float(cv2),
    }
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.linspace(0, 5.0, 80)
    ax.hist(L_true, bins=bins, density=True, alpha=0.55, color="#2f6db3",
            label="原始间隔 L 的密度")
    ax.hist(L_ins, bins=bins, density=True, alpha=0.55, color="#c94f4f",
            label="随机时点观察到的区间 L*")
    xg = np.linspace(1e-3, 5, 400)
    dens = np.exp(-np.log(xg) ** 2 / (2 * SIG0 ** 2)) / (xg * SIG0 * np.sqrt(2 * np.pi))
    ax.plot(xg, xg * dens / mu1, "k--", lw=1.6,
            label=r"理论密度 $x f(x)/E[L]$")
    ax.axvline(L_true.mean(), color="#2f6db3", lw=1, ls=":")
    ax.axvline(L_ins.mean(), color="#c94f4f", lw=1, ls=":")
    ax.annotate(f"E[L]={L_true.mean():.3f}", (L_true.mean(), 1.6),
                xytext=(0.35, 1.6), color="#2f6db3")
    ax.annotate(f"E[L*]={L_ins.mean():.3f}", (L_ins.mean(), 1.2),
                xytext=(2.6, 1.2), color="#c94f4f")
    ax.set_xlabel("挂单间隔（天）")
    ax.set_ylabel("密度")
    ax.set_title(f"检查悖论：观察到的间隔被拉长 {stat['ratio_sim']:.3f} 倍"
                 f"（理论 1+CV²={stat['ratio_theory']:.3f}）")
    ax.legend()
    _save(fig, "fig1_inspection_paradox.png")
    return stat


def exp2_cv_sweep(rng):
    """扫 CV：偏差倍数 1+CV^2 的仿真点 vs 理论曲线。"""
    sigmas = np.array([0.0, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95, 1.1])
    rows = []
    for s in sigmas:
        _, mu2, cv2 = lognormal_params(s)
        if s == 0.0:
            r = 1.0
        else:
            L_ins, _, _, _ = inspected_lengths(rng, s, 150_000, 400.0)
            r = float(L_ins.mean())  # E[L]=1 归一
        rows.append({"sigma": float(s), "cv2": float(cv2),
                     "ratio_sim": r, "ratio_theory": float(mu2)})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    xs = [q["cv2"] for q in rows]
    ax.plot(xs, [q["ratio_theory"] for q in rows], "-", color="#2f6db3",
            lw=1.8, label="理论 1+CV²")
    ax.scatter(xs, [q["ratio_sim"] for q in rows], color="#c94f4f", zorder=5,
               label="蒙特卡洛仿真")
    for q in rows[1::2]:
        ax.annotate(f"CV²={q['cv2']:.2f}→{q['ratio_sim']:.2f}",
                    (q["cv2"], q["ratio_sim"]), textcoords="offset points",
                    xytext=(6, 7), fontsize=8)
    ax.set_xlabel("间隔分布的变异系数平方 CV²")
    ax.set_ylabel("观察均值 / 真值")
    ax.set_title("分散度越大，快照估计的偏差越大")
    ax.legend()
    _save(fig, "fig2_cv_sweep.png")
    return {"sweep": rows, "max_abs_dev": max(abs(q["ratio_sim"] - q["ratio_theory"])
                                              for q in rows)}


def eq_cdf(x, sigma):
    """平衡分布 CDF：F_e(x)=E[min(L,x)]/E[L]，L 为归一均值 1 的对数正态。"""
    x = np.asarray(x, dtype=float)
    from scipy.stats import norm

    # E[min(L,x)] = ∫0^x P(L>t)dt = x - E[(L-x)^+]；用数值积分更稳
    tt = np.linspace(0.0, x.max(), 20001)
    tail = np.empty_like(tt)
    pos = tt > 0
    tail[pos] = norm.sf(np.log(tt[pos]) / sigma + sigma / 2.0)
    tail[~pos] = 1.0
    integ = np.concatenate(([0.0], np.cumsum((tail[1:] + tail[:-1]) / 2 * np.diff(tt))))
    Fe = np.interp(x, tt, integ)  # E[L]=1
    return Fe


def exp3_residual(rng):
    """残差寿命 vs 平衡分布理论 CDF。"""
    _, _, L_start, resid = inspected_lengths(rng, SIG0, 300_000, 300.0)
    sr = np.sort(resid)
    xs = sr[np.linspace(0, sr.size - 1, 300).astype(int)]
    emp = (np.arange(1, xs.size + 1)) / xs.size  # xs 取自 sr 的等距格点
    theo = eq_cdf(xs, SIG0)
    ks = float(np.max(np.abs(emp - theo)))
    stat = {"sigma": SIG0, "n": int(resid.size), "ks_stat": ks,
            "mean_residual": float(resid.mean()),
            "mean_age": float(L_start.mean()),
            "half_quantile_theory": float(xs[np.argmax(theo >= 0.5)]),
            "half_quantile_sim": float(np.quantile(resid, 0.5))}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot(xs, emp, "o", ms=4, color="#c94f4f", label="仿真残差寿命经验 CDF")
    ax.plot(xs, theo, "-", lw=1.8, color="#2f6db3", label="平衡分布理论 CDF")
    ax.set_xlabel("残差寿命（天）")
    ax.set_ylabel("累计概率")
    ax.set_title(f"残差寿命服从平衡分布（K-S 偏离 {ks:.4f}）")
    ax.legend()
    _save(fig, "fig3_residual_life.png")
    return stat


def exp4_estimator(rng):
    """固定日历采样估计平均挂单间隔：朴素均值 vs 1/(1+CV²) 校正。"""
    mu1, mu2, cv2 = lognormal_params(SIG0)
    ns = [25, 60, 150, 400, 1000, 3000, 10000]
    reps = 300
    rows = []
    for n in ns:
        naive, corr = [], []
        for _ in range(reps):
            L_ins, _, _, _ = inspected_lengths(rng, SIG0, n, 300.0)
            naive.append(L_ins.mean())
            # size-biased 样本的调和均值是 E[L] 的无偏估计：E[1/L*]=1/E[L]
            corr.append(n / np.sum(1.0 / L_ins))
        naive = np.array(naive)
        corr = np.array(corr)
        rows.append({"n": n, "naive_bias_pct": float((naive.mean() / mu1 - 1) * 100),
                     "corr_bias_pct": float((corr.mean() / mu1 - 1) * 100),
                     "naive_rmse_pct": float(np.sqrt(np.mean((naive / mu1 - 1) ** 2)) * 100),
                     "corr_rmse_pct": float(np.sqrt(np.mean((corr / mu1 - 1) ** 2)) * 100)})
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    ax.plot([q["n"] for q in rows], [q["naive_bias_pct"] for q in rows],
            "o-", color="#c94f4f", label="朴素均值偏差（不收敛）")
    ax.plot([q["n"] for q in rows], [q["corr_bias_pct"] for q in rows],
            "s-", color="#2f6db3", label="调和均值校正偏差（→0）")
    ax.axhline(0, color="gray", lw=1, ls=":")
    ax.axhline(cv2 * 100, color="#c94f4f", lw=1, ls="--",
               label=f"理论偏差 CV²={cv2:.3f}")
    ax.set_xscale("log")
    ax.set_xlabel("快照样本量 n")
    ax.set_ylabel("对真值 E[L] 的偏差（%）")
    ax.set_title("样本再多也修不掉偏差：必须做长度偏差校正")
    ax.legend()
    _save(fig, "fig4_estimator_bias.png")
    return {"cv2_theory_pct": float(cv2 * 100), "sweep": rows}


def exp5_snapshot(rng):
    """持仓快照：截面上「平均持仓时长」被长度偏差放大的倍数。"""
    n_traders = 120_000
    H = draw_L(rng, SIG0, n_traders)  # 单位=天，均值≈1
    t_obs = rng.uniform(0, 40.0, 1)
    # 开单时刻在 [0,t_obs] 内均匀分布；快照只看到 open_t<=t_obs<open_t+H 的在场单
    open_t = rng.uniform(0, 40.0, n_traders)
    active = (open_t <= t_obs) & (open_t + H > t_obs)
    snap = H[active]
    stat = {"n_active": int(active.sum()), "true_mean_H": float(H.mean()),
            "snapshot_mean_H": float(snap.mean()),
            "amplification": float(snap.mean() / H.mean()),
            "theory_1pCV2": float(lognormal_params(SIG0)[1]),
            "pct_long_holders_true": float((H > 3).mean() * 100),
            "pct_long_holders_snap": float((snap > 3).mean() * 100)}
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bins = np.linspace(0, 6, 60)
    ax.hist(H, bins=bins, density=True, alpha=0.55, color="#2f6db3",
            label="全部订单的真实持仓时长")
    ax.hist(snap, bins=bins, density=True, alpha=0.55, color="#c94f4f",
            label="快照中在场订单的持仓时长")
    ax.axvline(H.mean(), color="#2f6db3", ls=":", lw=1.2)
    ax.axvline(snap.mean(), color="#c94f4f", ls=":", lw=1.2)
    ax.set_xlabel("持仓时长（天）")
    ax.set_ylabel("密度")
    ax.set_title(f"持仓快照截面均值 {snap.mean():.3f} 天 vs 真值 {H.mean():.3f} 天"
                 f"（放大 {stat['amplification']:.3f}×）")
    ax.legend()
    _save(fig, "fig5_position_snapshot.png")
    return stat


def main():
    rng = np.random.default_rng(SEED)
    stats = {
        "seed": SEED,
        "exp1_inspection": exp1_inspection(rng),
        "exp2_cv_sweep": exp2_cv_sweep(rng),
        "exp3_residual": exp3_residual(rng),
        "exp4_estimator": exp4_estimator(rng),
        "exp5_snapshot": exp5_snapshot(rng),
    }
    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

