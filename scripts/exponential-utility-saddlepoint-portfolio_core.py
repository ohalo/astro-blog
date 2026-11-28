# -*- coding: utf-8 -*-
"""CARA portfolio certainty equivalents: normal plug-in versus mixture saddlepoint."""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261129
A = 10.0
P = np.array([0.92, 0.05, 0.03])
MU = np.array([0.0015, -0.0030, -0.0150])
SIG = np.array([0.0040, 0.0120, 0.0450])
N_MC = 300000
N_GRID = 301
TRIALS = 240
SIZES = (60, 250, 1000)
IMG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "public", "images", "exponential-utility-saddlepoint-portfolio")


def mixture_moments():
    m = float(P @ MU)
    e2 = float(P @ (SIG ** 2 + MU ** 2))
    e3 = float(P @ (3 * MU * SIG ** 2 + MU ** 3))
    e4 = float(P @ (3 * SIG ** 4 + 6 * MU ** 2 * SIG ** 2 + MU ** 4))
    var = e2 - m * m
    m3 = e3 - 3 * m * e2 + 2 * m ** 3
    m4 = e4 - 4 * m * e3 + 6 * m * m * e2 - 3 * m ** 4
    return m, var, m3 / var ** 1.5, m4 / var ** 2 - 3.0


def cgf(w, t):
    z = t * w * MU + 0.5 * (t * w * SIG) ** 2
    zmax = z.max()
    z -= zmax
    return float(np.log(P @ np.exp(z)) + zmax)


def ce_saddle(w, a=A):
    return -cgf(w, -a) / a


def ce_normal(w, m, var, a=A):
    return w * m - 0.5 * a * w * w * var


def ce_empirical(w, sample, a=A):
    z = -a * w * sample
    zmax = z.max()
    return -float(np.log(np.mean(np.exp(z - zmax))) + zmax) / a


def sample_draws(rng, n):
    k = rng.choice(len(P), size=n, p=P)
    return MU[k] + SIG[k] * rng.standard_normal(n)


def best_weight(ce_fn, grid):
    vals = np.array([ce_fn(float(w)) for w in grid])
    return float(grid[int(np.argmax(vals))]), float(vals.max())


def main():
    os.makedirs(IMG, exist_ok=True)
    m, var, skew, kurt = mixture_moments()
    rng = np.random.default_rng(SEED)
    draws = sample_draws(rng, N_MC)
    fine = np.linspace(0.0, 1.5, N_GRID)

    w_norm_true = float(np.clip(m / (A * var), 0.0, 1.5))
    w_sad, ce_sad = best_weight(lambda w: ce_saddle(w), fine)
    ce_norm_best = ce_normal(w_norm_true, m, var)
    ce_norm_at_sad = ce_normal(w_sad, m, var)
    ce_sad_at_norm = ce_saddle(w_norm_true)
    ce_mc_sad = -float(np.log(np.mean(np.exp(-A * w_sad * draws)))) / A
    ce_mc_norm = -float(np.log(np.mean(np.exp(-A * w_norm_true * draws)))) / A
    mc_se = float(np.std(np.exp(-A * w_sad * draws)) /
                  (np.sqrt(N_MC) * np.mean(np.exp(-A * w_sad * draws))))

    a_grid = np.array([2.0, 5.0, 10.0, 20.0, 40.0])
    a_sweep = {}
    for a in a_grid:
        wn = float(np.clip(m / (a * var), 0.0, 1.5))
        ws, _ = best_weight(lambda w, aa=a: ce_saddle(w, aa), fine)
        a_sweep[str(a)] = {"w_normal": round(wn, 6), "w_saddle": round(ws, 6),
                           "ce_normal": round(ce_normal(wn, m, var, a), 8),
                           "ce_saddle": round(ce_saddle(ws, a), 8)}

    coarse = np.linspace(0.0, 1.5, 61)
    estimation = {}
    for n in SIZES:
        trial_rng = np.random.default_rng(SEED + n)
        wn_list, ws_list, reg_n, reg_s = [], [], [], []
        wins = 0
        for _ in range(TRIALS):
            s = sample_draws(trial_rng, n)
            mh = float(s.mean())
            vh = float(s.var(ddof=1))
            whn = float(np.clip(mh / (A * vh), 0.0, 1.5))
            whs, _ = best_weight(lambda w, ss=s: ce_empirical(w, ss), coarse)
            wn_list.append(whn)
            ws_list.append(whs)
            rn = ce_sad - ce_saddle(whn)
            rs = ce_sad - ce_saddle(whs)
            reg_n.append(rn)
            reg_s.append(rs)
            wins += int(rs < rn)
        estimation[str(n)] = {
            "mean_w_normal": round(float(np.mean(wn_list)), 6),
            "mean_w_saddle": round(float(np.mean(ws_list)), 6),
            "mean_regret_normal": round(float(np.mean(reg_n)), 8),
            "mean_regret_saddle": round(float(np.mean(reg_s)), 8),
            "median_regret_normal": round(float(np.median(reg_n)), 8),
            "median_regret_saddle": round(float(np.median(reg_s)), 8),
            "saddle_win_rate": round(wins / TRIALS, 6),
        }

    t_grid = np.linspace(-A * 1.5, 0.0, 13)
    k_exact = np.array([cgf(w_sad, float(t)) for t in t_grid])
    k_mc = np.array([np.log(np.mean(np.exp(t * w_sad * draws))) for t in t_grid])

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.hist(draws, bins=180, density=True, color="#b9c8dc", edgecolor="none")
    xs = np.linspace(-0.08, 0.04, 600)
    for i in range(len(P)):
        ax.plot(xs, P[i] / (SIG[i] * np.sqrt(2 * np.pi)) *
                np.exp(-0.5 * ((xs - MU[i]) / SIG[i]) ** 2),
                lw=1.5, label="component " + str(i + 1))
    ax.axvline(m, color="#c0392b", ls="--", lw=1.2, label="mixture mean")
    ax.set_title("Three-normal mixture of daily excess returns")
    ax.set_xlabel("daily excess return")
    ax.set_ylabel("density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig1_mixture_density.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4.3))
    ax.plot(fine, [ce_normal(float(w), m, var) for w in fine],
            lw=1.6, color="#3b5b8c", label="normal plug-in CE")
    ax.plot(fine, [ce_saddle(float(w)) for w in fine],
            lw=1.6, color="#c0392b", label="mixture saddlepoint CE")
    ax.plot(fine, [-np.log(np.mean(np.exp(-A * w * draws))) / A for w in fine],
            lw=1.0, color="#7f8c8d", label="Monte Carlo CE")
    ax.axvline(w_norm_true, color="#3b5b8c", ls="--", lw=1.0)
    ax.axvline(w_sad, color="#c0392b", ls="--", lw=1.0)
    ax.set_title("Certainty equivalent versus portfolio weight")
    ax.set_xlabel("weight on risky asset")
    ax.set_ylabel("daily CE")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig2_ce_curves.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(a_grid, [a_sweep[str(a)]["w_normal"] for a in a_grid], "o-",
            color="#3b5b8c", label="normal weight")
    ax.plot(a_grid, [a_sweep[str(a)]["w_saddle"] for a in a_grid], "o-",
            color="#c0392b", label="saddlepoint weight")
    ax.fill_between(a_grid, [a_sweep[str(a)]["w_normal"] for a in a_grid],
                    [a_sweep[str(a)]["w_saddle"] for a in a_grid], alpha=0.15, color="#c0392b")
    ax.set_title("Optimal weight versus risk aversion")
    ax.set_xlabel("CARA coefficient a")
    ax.set_ylabel("weight")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig3_weights_vs_aversion.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ns = [str(n) for n in SIZES]
    axes[0].plot(ns, [estimation[n]["mean_w_normal"] for n in ns], "o-",
                 color="#3b5b8c", label="normal plug-in")
    axes[0].plot(ns, [estimation[n]["mean_w_saddle"] for n in ns], "o-",
                 color="#c0392b", label="empirical CE")
    axes[0].axhline(w_sad, ls="--", lw=1.0, color="0.35", label="true optimum")
    axes[0].set_title("Mean estimated weight")
    axes[0].set_xlabel("sample size")
    axes[0].set_ylabel("weight")
    axes[0].legend()
    axes[1].bar(np.arange(3) - 0.18, [estimation[n]["mean_regret_normal"] for n in ns],
                width=0.36, color="#3b5b8c", label="normal")
    axes[1].bar(np.arange(3) + 0.18, [estimation[n]["mean_regret_saddle"] for n in ns],
                width=0.36, color="#c0392b", label="saddlepoint")
    axes[1].set_xticks(np.arange(3))
    axes[1].set_xticklabels(ns)
    axes[1].set_title("Out-of-sample CE regret")
    axes[1].set_xlabel("sample size")
    axes[1].set_ylabel("daily CE")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig4_finite_sample.png"), dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(t_grid, k_exact, "o-", color="#c0392b", label="analytic CGF")
    axes[0].plot(t_grid, k_mc, "s--", color="#7f8c8d", label="MC CGF")
    axes[0].set_title("Cumulant generating function check")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("K(t)")
    axes[0].legend()
    axes[1].plot(t_grid, k_mc - k_exact, "o-", color="#2c6e7f")
    axes[1].axhline(0, color="0.3", lw=0.8)
    axes[1].set_title("MC minus analytic CGF")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("error")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG, "fig5_saddlepoint_check.png"), dpi=150)
    plt.close(fig)

    out = {
        "seed": SEED,
        "a": A,
        "mix_p": P.tolist(),
        "mix_mu": MU.tolist(),
        "mix_sigma": SIG.tolist(),
        "n_mc": N_MC,
        "trials": TRIALS,
        "true_mean": round(m, 8),
        "true_var": round(var, 10),
        "true_sd": round(float(np.sqrt(var)), 8),
        "skewness": round(float(skew), 6),
        "excess_kurtosis": round(float(kurt), 6),
        "w_normal": round(w_norm_true, 6),
        "w_saddle": round(w_sad, 6),
        "ce_normal_best": round(ce_norm_best, 8),
        "ce_normal_at_saddle": round(ce_norm_at_sad, 8),
        "ce_saddle_at_normal": round(ce_sad_at_norm, 8),
        "ce_saddle_best": round(ce_sad, 8),
        "ce_mc_at_saddle": round(ce_mc_sad, 8),
        "ce_mc_at_normal": round(ce_mc_norm, 8),
        "mc_relative_se": round(mc_se, 8),
        "weight_gap": round(w_norm_true - w_sad, 6),
        "ce_regret_of_normal": round(ce_sad - ce_sad_at_norm, 8),
        "ce_regret_annual_bp": round((ce_sad - ce_sad_at_norm) * 252 * 10000, 3),
        "a_sweep": a_sweep,
        "estimation": estimation,
        "cgf_max_abs_error": round(float(np.max(np.abs(k_mc - k_exact))), 6),
    }
    with open(os.path.join(IMG, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2, sort_keys=True)
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
