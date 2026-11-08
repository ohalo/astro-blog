"""回测里的「偷看」问题：固定样本检验、Bonferroni 与 anytime-valid e-process。

运行方式（在项目根目录）：
    python3 scripts/article_core_eval_20261108night.py

产出：
    public/images/anytime-valid-backtest-e-process/fig1..fig5.png
    public/images/anytime-valid-backtest-e-process/results.json

核心问题：回测跑了 250 天，研究员每周看一次 p 值，只要某次 p<0.05 就宣布策略有效。
这种「可选停止」会把 I 类错误放大数倍。e-process 用混合似然比鞅 + Ville 不等式
给出对任意停止时刻都成立的 1/alpha 阈值，代价是功效与检出延迟。本文把所有数字
现场算出来，并且诚实地检验模型错配（方差设错、厚尾）时 anytime-valid 会不会失效。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logsumexp
from scipy.stats import norm, t as student_t

SEED = 20261108
OUT = "public/images/anytime-valid-backtest-e-process"
T = 250                 # 回测交易日
ALPHA = 0.05
SIGMA = 1.0             # 研究员假定日收益标准差
N_SIM_H0 = 3000         # 原假设下的独立回测次数
N_SIM_ALT = 1500        # 备择下的独立回测次数
TAU = 0.20              # 混合先验 N(0, tau^2) 的标准差
N_MU = 41               # 混合网格点数
PEEK_EVERYS = (250, 125, 50, 25, 10, 5, 1)
EFFECTS = (0.05, 0.10, 0.15, 0.20, 0.30)
TAUS = (0.03, 0.08, 0.20, 0.40, 0.80)
MIS_SIGMAS = (0.8, 1.0, 1.3, 1.6)


def mixing_grid(tau, n_mu=N_MU):
    """把 N(0,tau^2) 混合先验离散成有限混合（有限混合仍是精确的鞅）。"""
    mus = np.linspace(-4.0 * tau, 4.0 * tau, n_mu)
    w = np.exp(-0.5 * (mus / tau) ** 2)
    w /= w.sum()
    return mus, np.log(w + 1e-300)


def paths(seed, n_sim, mu=0.0, sigma_true=SIGMA, heavy=False):
    """生成 (n_sim, T) 条日超额收益路径；共同随机数通过 seed 保证。"""
    rng = np.random.default_rng(seed)
    if heavy:
        z = student_t.rvs(df=3, size=(n_sim, T), random_state=rng) / np.sqrt(3.0)
    else:
        z = rng.standard_normal((n_sim, T))
    return mu + sigma_true * z


def log_e_process(S, mus, logw, sigma=SIGMA):
    """log E_n = logsumexp_k [ log w_k + mu_k S_n/sigma^2 - n mu_k^2/(2 sigma^2) ].

    这是备择「r_i ~ N(mu_k, sigma^2)」对原假设「r_i ~ N(0, sigma^2)」的
    混合似然比；H0 下它是非负鞅且 E[E_n]=1，Ville 不等式给出
    P(存在 n 使 E_n >= 1/alpha) <= alpha。
    """
    n = np.arange(1, S.shape[1] + 1, dtype=float)
    coef = mus / sigma ** 2
    quad = mus ** 2 / (2.0 * sigma ** 2)
    out = np.empty_like(S, dtype=float)
    for i in range(0, S.shape[0], 500):          # 分块控制内存
        term = (logw[None, None, :]
                + coef[None, None, :] * S[i:i + 500, :, None]
                - quad[None, None, :] * n[None, :, None])
        out[i:i + 500] = logsumexp(term, axis=2)
    return out


def decisions(r, tau=TAU, alpha=ALPHA, sigma=SIGMA):
    """对一批路径给出四种检验的拒绝判定与首次越过时刻。"""
    S = np.cumsum(r, axis=1)
    n = np.arange(1, r.shape[1] + 1, dtype=float)
    Z = S / (sigma * np.sqrt(n))
    zcrit = norm.ppf(1.0 - alpha / 2.0)
    mus, logw = mixing_grid(tau)
    logE = log_e_process(S, mus, logw, sigma=sigma)
    thr = np.log(1.0 / alpha)

    fixed = np.abs(Z[:, -1]) > zcrit
    # 偷看 / Bonferroni：按给定网格检查
    res = {"fixed": fixed}
    for every in PEEK_EVERYS:
        grid = np.arange(every, T + 1, every)
        zz = np.abs(Z[:, grid - 1])
        res[f"peek_{every}"] = (zz > zcrit).any(axis=1)
        zb = norm.ppf(1.0 - alpha / (2.0 * len(grid)))
        res[f"bonf_{every}"] = (zz > zb).any(axis=1)
    # e-process：每个时刻都检查（Ville 不等式覆盖任意停止时刻）
    e_hit = logE >= thr
    res["eproc"] = e_hit.any(axis=1)
    first = np.argmax(e_hit, axis=1) + 1
    res["eproc_first"] = np.where(res["eproc"], first, T + 1)
    res["logE"] = logE
    res["Z"] = Z
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
        try:
            plt.rcParams["font.family"] = [f]
            break
        except Exception:
            continue
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 130
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.3

    R = {}
    zcrit = float(norm.ppf(1.0 - ALPHA / 2.0))
    R["settings"] = {"T": T, "alpha": ALPHA, "sigma_assumed": SIGMA,
                     "n_sim_h0": N_SIM_H0, "n_sim_alt": N_SIM_ALT,
                     "tau": TAU, "n_mu": N_MU, "zcrit": zcrit,
                     "seed": SEED}

    # ---------- 1) 原假设下的 I 类错误，以及偷看频次扫描 ----------
    r0 = paths(SEED, N_SIM_H0)
    d0 = decisions(r0)
    R["h0_rejection"] = {
        "fixed": float(d0["fixed"].mean()),
        "peek_weekly": float(d0["peek_5"].mean()),
        "peek_daily": float(d0["peek_1"].mean()),
        "bonferroni_weekly": float(d0["bonf_5"].mean()),
        "eproc": float(d0["eproc"].mean()),
    }
    R["peek_scan"] = []
    for every in PEEK_EVERYS:
        R["peek_scan"].append({
            "every": every,
            "n_looks": int(T // every),
            "peek": float(d0[f"peek_{every}"].mean()),
            "bonferroni": float(d0[f"bonf_{every}"].mean()),
            "eproc": float(d0["eproc"].mean()),
        })

    # ---------- 2) 示意路径：从上面这批 H0 模拟里挑一条「偷看误报、e-process 不误报」----------
    p0 = 2.0 * norm.sf(np.abs(d0["Z"]))
    cand = np.flatnonzero((p0 < ALPHA).any(axis=1) & ~d0["eproc"])
    idx = int(cand[0])
    demo_r = r0[idx]
    demo_Z = d0["Z"][idx]
    demo_p = p0[idx]
    demo_logE = d0["logE"][idx]
    R["demo"] = {"path_index": idx,
                 "n_candidates": int(len(cand)),
                 "max_abs_z": float(np.max(np.abs(demo_Z))),
                 "min_p": float(np.min(demo_p)),
                 "first_peek_day": int(np.flatnonzero(demo_p < ALPHA)[0] + 1),
                 "max_logE": float(np.max(demo_logE)),
                 "logE_threshold": float(np.log(1 / ALPHA)),
                 "final_p": float(demo_p[-1]),
                 "final_logE": float(demo_logE[-1])}

    # ---------- 3) 功效与检出时间 ----------
    R["power"] = []
    first_times = {}
    for mu in EFFECTS:
        r1 = paths(SEED + 1000, N_SIM_ALT, mu=mu)
        d1 = decisions(r1)
        grid = np.arange(5, T + 1, 5)
        Zg = np.abs(d1["Z"][:, grid - 1])
        zb = norm.ppf(1.0 - ALPHA / (2.0 * len(grid)))
        peek_first = np.where((Zg > zcrit).any(axis=1),
                              grid[np.argmax(Zg > zcrit, axis=1)], T + 1)
        bonf_first = np.where((Zg > zb).any(axis=1),
                              grid[np.argmax(Zg > zb, axis=1)], T + 1)
        row = {
            "mu": mu,
            "fixed": float(d1["fixed"].mean()),
            "peek": float(d1["peek_5"].mean()),
            "bonferroni": float(d1["bonf_5"].mean()),
            "eproc": float(d1["eproc"].mean()),
            "eproc_mean_hit": float(d1["eproc_first"][d1["eproc"]].mean())
            if d1["eproc"].any() else None,
            "bonf_mean_hit": float(bonf_first[bonf_first <= T].mean())
            if (bonf_first <= T).any() else None,
            "peek_mean_hit": float(peek_first[peek_first <= T].mean())
            if (peek_first <= T).any() else None,
        }
        R["power"].append(row)
        if abs(mu - 0.15) < 1e-9:
            first_times = {
                "eproc": d1["eproc_first"][d1["eproc"]].tolist(),
                "bonferroni": bonf_first[bonf_first <= T].tolist(),
                "peek": peek_first[peek_first <= T].tolist(),
            }
    R["first_hit_mu015"] = {
        "eproc_median": float(np.median(first_times["eproc"])),
        "bonferroni_median": float(np.median(first_times["bonferroni"])),
        "peek_median": float(np.median(first_times["peek"])),
        "eproc_n": len(first_times["eproc"]),
        "bonferroni_n": len(first_times["bonferroni"]),
        "peek_n": len(first_times["peek"]),
    }

    # ---------- 4) 模型错配：真实方差与厚尾 ----------
    R["misspec"] = []
    for s in MIS_SIGMAS:
        rm = paths(SEED + 2000, N_SIM_H0, sigma_true=s)
        dm = decisions(rm)
        R["misspec"].append({
            "label": f"sigma={s:.1f}", "true_sigma": s,
            "peek": float(dm["peek_5"].mean()),
            "eproc": float(dm["eproc"].mean()),
            "bonferroni": float(dm["bonf_5"].mean()),
        })
    rh = paths(SEED + 3000, N_SIM_H0, heavy=True)
    dh = decisions(rh)
    R["misspec"].append({
        "label": "t(3)/sqrt(3)", "true_sigma": 1.0,
        "peek": float(dh["peek_5"].mean()),
        "eproc": float(dh["eproc"].mean()),
        "bonferroni": float(dh["bonf_5"].mean()),
    })

    # ---------- 5) 混合先验宽度 tau 的权衡 ----------
    r_tau = paths(SEED + 4000, N_SIM_ALT, mu=0.15)
    R["tau_scan"] = []
    for tau in TAUS:
        dt = decisions(r_tau, tau=tau)
        hit = dt["eproc_first"][dt["eproc"]]
        R["tau_scan"].append({
            "tau": tau,
            "power": float(dt["eproc"].mean()),
            "mean_hit": float(hit.mean()) if len(hit) else None,
        })

    with open(f"{OUT}/results.json", "w", encoding="utf-8") as f:
        json.dump(R, f, ensure_ascii=False, indent=2)

    # ---------- 图 1：单条 H0 路径的 p 值与 e-process ----------
    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    ax[0].plot(np.arange(1, T + 1), demo_p, lw=1.6, color="#2b6cb0")
    ax[0].axhline(ALPHA, ls="--", color="#c53030", label="alpha=0.05")
    ax[0].set_yscale("log")
    ax[0].set_ylabel("双侧 p 值（对数轴）")
    ax[0].set_title("原假设下的一条真实路径：p 值穿越 0.05，e-process 未越过阈值")
    ax[0].legend()
    ax[1].plot(np.arange(1, T + 1), demo_logE, lw=1.6, color="#2f855a")
    ax[1].axhline(np.log(1 / ALPHA), ls="--", color="#c53030",
                  label="log(1/alpha)")
    ax[1].set_ylabel("log E_n")
    ax[1].set_xlabel("回测交易日 n")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_peeking_paths.png")
    plt.close(fig)

    # ---------- 图 2：I 类错误随偷看次数 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    k = [x["n_looks"] for x in R["peek_scan"]]
    ax.plot(k, [x["peek"] for x in R["peek_scan"]], "o-", color="#c53030",
            label="反复偷看（固定 0.05 阈值）")
    ax.plot(k, [x["bonferroni"] for x in R["peek_scan"]], "s-", color="#dd6b20",
            label="Bonferroni 校正")
    ax.plot(k, [x["eproc"] for x in R["peek_scan"]], "^-", color="#2f855a",
            label="e-process（任意时刻检验）")
    ax.axhline(ALPHA, ls="--", color="0.4", label="名义水平 0.05")
    ax.set_xscale("log")
    ax.set_xlabel("250 天内的检查次数 K（对数轴）")
    ax.set_ylabel("H0 下至少拒绝一次的比例")
    ax.set_title("偷看越多，普通 p 值越不可信；Ville 阈值与 K 无关")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_type1_vs_looks.png")
    plt.close(fig)

    # ---------- 图 3：功效曲线与检出延迟 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    mus = [x["mu"] for x in R["power"]]
    for key, label, c in [("fixed", "固定终点检验", "#718096"),
                          ("peek", "每周偷看", "#c53030"),
                          ("bonferroni", "Bonferroni", "#dd6b20"),
                          ("eproc", "e-process", "#2f855a")]:
        ax[0].plot(mus, [x[key] for x in R["power"]], "o-", color=c, label=label)
    ax[0].set_xlabel("真实日均超额收益 mu")
    ax[0].set_ylabel("检出比例（功效）")
    ax[0].set_title("功效：偷看组虚高，有效方法要付代价")
    ax[0].legend(fontsize=8)
    for key, label, c in [("eproc_mean_hit", "e-process", "#2f855a"),
                          ("bonf_mean_hit", "Bonferroni", "#dd6b20"),
                          ("peek_mean_hit", "每周偷看", "#c53030")]:
        vals = [x[key] if x[key] is not None else np.nan for x in R["power"]]
        ax[1].plot(mus, vals, "s-", color=c, label=label)
    ax[1].set_xlabel("真实日均超额收益 mu")
    ax[1].set_ylabel("平均首次越过的交易日")
    ax[1].set_title("检出延迟：有效性不是免费的")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_power_and_delay.png")
    plt.close(fig)

    # ---------- 图 4：mu=0.15 时首次检出时间分布 ----------
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.arange(0, T + 26, 25)
    for key, label, c in [("peek", "每周偷看", "#c53030"),
                          ("bonferroni", "Bonferroni", "#dd6b20"),
                          ("eproc", "e-process", "#2f855a")]:
        ax.hist(first_times[key], bins=bins, alpha=0.55, label=label, color=c)
    ax.axvline(R["first_hit_mu015"]["eproc_median"], ls="--", color="#2f855a")
    ax.set_xlabel("首次越过的交易日")
    ax.set_ylabel("路径条数")
    ax.set_title("mu=0.15：有效检验把检出时间整体推后")
    ax.legend()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig4_first_hit_hist.png")
    plt.close(fig)

    # ---------- 图 5：模型错配与 tau 权衡 ----------
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
    labels = [x["label"] for x in R["misspec"]]
    xp = np.arange(len(labels))
    ax[0].bar(xp - 0.2, [x["peek"] for x in R["misspec"]], 0.4,
              color="#c53030", label="每周偷看")
    ax[0].bar(xp + 0.2, [x["eproc"] for x in R["misspec"]], 0.4,
              color="#2f855a", label="e-process")
    ax[0].axhline(ALPHA, ls="--", color="0.4")
    ax[0].set_xticks(xp)
    ax[0].set_xticklabels(labels, rotation=20, fontsize=8)
    ax[0].set_ylabel("H0 下拒绝比例")
    ax[0].set_title("真实波动被设错 / 厚尾时，Ville 保证也失效")
    ax[0].legend(fontsize=8)
    ts = [x["tau"] for x in R["tau_scan"]]
    ax[1].plot(ts, [x["power"] for x in R["tau_scan"]], "o-", color="#2b6cb0",
               label="功效（mu=0.15）")
    ax[1].set_xscale("log")
    ax[1].set_xlabel("混合先验宽度 tau（对数轴）")
    ax[1].set_ylabel("检出比例", color="#2b6cb0")
    ax2 = ax[1].twinx()
    ax2.plot(ts, [x["mean_hit"] if x["mean_hit"] else np.nan
                  for x in R["tau_scan"]], "s--", color="#805ad5",
             label="平均检出日")
    ax2.set_ylabel("平均首次越过日", color="#805ad5")
    ax2.grid(False)
    ax[1].set_title("先验宽度要匹配效应量：过宽过窄都亏")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig5_misspec_and_tau.png")
    plt.close(fig)

    print("=== H0 拒绝率 (alpha=0.05) ===")
    for k, v in R["h0_rejection"].items():
        print(f"  {k:20s} {v:.4f}")
    print("=== 偷看次数扫描 ===")
    for row in R["peek_scan"]:
        print(f"  K={row['n_looks']:3d}  peek={row['peek']:.4f}  "
              f"bonf={row['bonferroni']:.4f}  eproc={row['eproc']:.4f}")
    print("=== 功效 (mu=0.15) ===")
    p = [x for x in R["power"] if abs(x["mu"] - 0.15) < 1e-9][0]
    for k in ("fixed", "peek", "bonferroni", "eproc",
              "eproc_mean_hit", "bonf_mean_hit", "peek_mean_hit"):
        print(f"  {k:20s} {p[k]:.4f}")
    print("=== 模型错配下 e-process 拒绝率 ===")
    for row in R["misspec"]:
        print(f"  {row['label']:16s} eproc={row['eproc']:.4f}  peek={row['peek']:.4f}")
    print("=== tau 扫描 ===")
    for row in R["tau_scan"]:
        print(f"  tau={row['tau']:.2f}  power={row['power']:.4f}  "
              f"mean_hit={row['mean_hit']}")
    print(f"demo: path#{R['demo']['path_index']}  "
          f"first_peek_day={R['demo']['first_peek_day']}  "
          f"min_p={R['demo']['min_p']:.5f}  "
          f"max_logE={R['demo']['max_logE']:.4f} "
          f"(threshold {R['demo']['logE_threshold']:.4f})")


if __name__ == "__main__":
    main()
