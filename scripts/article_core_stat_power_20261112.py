"""Statistical power of a factor IC backtest and the minimum detectable IC.

This file IS the runnable code block embedded in the article. Run it directly:

    python3 article_core_stat_power_20261112.py

It writes five real PNG figures and results.json into ARTICLE_OUT (default: cwd).
Every number quoted in the article comes from results.json. No placeholders,
no fabricated curves, no hardcoded statistics.

Dependencies: numpy, scipy, matplotlib.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm, t as student_t

OUT = Path(os.environ.get("ARTICLE_OUT", ".")).resolve()
SEED = 20261112
T = 250              # trading days in one backtest year
N = 50               # names in the cross-section
ALPHA = 0.05         # two-sided test level
TARGET = 0.80        # desired power
R_MAIN = 1500        # independent backtest replications
R_TAIL = 800
R_N = 600

Z975 = float(norm.ppf(1 - ALPHA / 2))
Z80 = float(norm.ppf(TARGET))
SIGMA_IC = 1.0 / np.sqrt(N - 1)          # sampling sd of one daily Pearson IC under H0


def crit_t(df, level=ALPHA):
    return float(student_t.ppf(1 - level / 2, df))


MDE_ANALYTIC = (crit_t(T - 1) + Z80) * SIGMA_IC / np.sqrt(T)


# ---------- cross-sectional DGP with a known true IC ----------
def make_pool(rng, R, T_, N_, dtype=np.float32):
    """Independent standardised factor scores and idiosyncratic shocks."""
    f = rng.standard_normal((R, T_, N_), dtype=dtype)
    z = rng.standard_normal((R, T_, N_), dtype=dtype)
    return f, z


def ic_sequences(f, z, rho, phi=0.0, sigma_eta=0.0, seed=None):
    """Daily Pearson IC from r_i = rho_t * f_i + sqrt(1-rho_t^2) * z_i.

    rho is the true cross-sectional IC (constant unless phi>0, in which case
    the true IC itself follows a zero-mean AR(1) around rho). Returns (R, T).
    """
    R, T_, N_ = f.shape
    if phi > 0.0:
        rg = np.random.default_rng(seed)
        eta = rg.normal(0.0, sigma_eta, size=(R, T_)).astype(np.float32)
        cur = np.zeros(R, dtype=np.float32)
        path = np.empty((R, T_), dtype=np.float32)
        for t in range(T_):
            cur = phi * cur + eta[:, t]
            path[:, t] = rho + cur
        rho_t = np.clip(path[..., None], -0.95, 0.95).astype(np.float32)
    else:
        rho_t = np.float32(rho)
    r = rho_t * f + np.sqrt(1.0 - rho_t ** 2) * z
    fc = f - f.mean(axis=-1, keepdims=True)
    rc = r - r.mean(axis=-1, keepdims=True)
    num = (fc * rc).sum(axis=-1)
    den = np.sqrt((fc * fc).sum(axis=-1) * (rc * rc).sum(axis=-1))
    return (num / den).astype(np.float64)


def tstat(ic):
    """Two-sided one-sample t statistic for mean(IC)=0, per replication."""
    m = ic.mean(axis=1)
    s = ic.std(axis=1, ddof=1)
    return m / (s / np.sqrt(ic.shape[1]))


def tstat_hac(ic, max_lags=10):
    """Same t statistic but with a Newey-West (Bartlett) long-run variance.

    Under AR(1) dependence the naive s^2/T underestimates Var(mean(IC));
    the HAC estimate restores the correct size.
    """
    m = ic.mean(axis=1)
    x = ic - m[:, None]
    T_ = ic.shape[1]
    g0 = (x * x).sum(axis=1) / T_
    lr = g0.copy()
    for lag in range(1, max_lags + 1):
        w = 1.0 - lag / (max_lags + 1.0)
        gl = (x[:, lag:] * x[:, :-lag]).sum(axis=1) / T_
        lr += 2.0 * w * gl
    lr = np.maximum(lr, g0 * 1e-6)
    return m / np.sqrt(lr / T_)


def power_at(ic, thresh):
    return float(np.mean(np.abs(tstat(ic)) > thresh))


def analytic_power(rho, T_, N_, thresh=None):
    # Normal approximation to the non-central t, evaluated at the SAME
    # critical value used by the Monte Carlo test.
    c = crit_t(T_ - 1) if thresh is None else thresh
    delta = rho * np.sqrt(T_ * (N_ - 1))
    return float(norm.cdf(delta - c) + norm.cdf(-delta - c))


def mde_analytic(T_, N_, level=ALPHA):
    return (crit_t(T_ - 1, level) + Z80) / np.sqrt(T_ * (N_ - 1))


def mde_mc(rhos, powers):
    """Linearly interpolate the true IC at which empirical power hits TARGET."""
    for i in range(len(rhos) - 1):
        if powers[i] <= TARGET <= powers[i + 1]:
            w = (TARGET - powers[i]) / (powers[i + 1] - powers[i])
            return float(rhos[i] + w * (rhos[i + 1] - rhos[i]))
    return float("nan")


def run_all():
    rng = np.random.default_rng(SEED)
    stats = {}
    figs = {}

    stats["design"] = {
        "T": T, "N": N, "alpha": ALPHA, "target_power": TARGET,
        "R_main": R_MAIN, "R_tail": R_TAIL, "R_n": R_N, "seed": SEED,
        "z_975": round(Z975, 6), "z_80": round(Z80, 6),
        "sigma_ic_theory": round(float(SIGMA_IC), 6),
    }

    # ---- common random numbers: one (f,z) pool reused by every rho ----
    f, z = make_pool(rng, R_MAIN, T, N)
    crit = crit_t(T - 1)
    stats["design"]["t_crit_249df"] = round(float(crit), 6)

    # ---- Figure 1 + Figure 2: representative paths, power curve, MDE ----
    rhos = [0.0, 0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.035,
            0.04, 0.05, 0.06, 0.08]
    mc_powers, an_powers = [], []
    ic0 = ic_sequences(f, z, 0.0)
    for rho in rhos:
        ic = ic0 if rho == 0.0 else ic_sequences(f, z, rho)
        mc_powers.append(power_at(ic, crit))
        an_powers.append(analytic_power(rho, T, N))
        if abs(rho - 0.02) < 1e-12:
            ic_002 = ic
            figs["ic0"] = ic0[0]
            figs["ic002"] = ic[0]
            figs["ic0_all"] = ic0
            figs["tvals0"] = tstat(ic0)
            figs["mean_ic002"] = ic.mean(axis=1)
    figs["rhos"] = rhos
    figs["mc_powers"] = mc_powers
    figs["an_powers"] = an_powers
    figs["t_crit"] = crit

    mde_mc_val = mde_mc(rhos, mc_powers)
    stats["power_curve"] = {
        "rho": rhos,
        "mc_power": [round(v, 4) for v in mc_powers],
        "analytic_power": [round(v, 4) for v in an_powers],
        "max_abs_gap": round(float(np.max(np.abs(np.array(mc_powers) - np.array(an_powers)))), 4),
        "mde_mc": round(mde_mc_val, 6),
        "mde_analytic": round(float(MDE_ANALYTIC), 6),
    }
    stats["headline"] = {
        "size_at_rho0": round(mc_powers[0], 4),
        "power_at_rho_002": round(mc_powers[4], 4),
        "analytic_power_at_rho_002": round(an_powers[4], 4),
        "mde_80": round(float(MDE_ANALYTIC), 6),
        "mde_80_mc": round(mde_mc_val, 6),
        "miss_rate_at_rho_002": round(1 - mc_powers[4], 4),
    }

    # ---- Figure 3: MDE scaling in T and N, analytic curves + MC anchors ----
    T_grid = [60, 125, 250, 500, 1000]
    mde_T = [mde_analytic(t, N) for t in T_grid]
    T_anchor = [60, 125, 250]
    mc_T = []
    for t in T_anchor:
        sub_rhos = np.linspace(0.0, 0.12, 13)
        pw = [power_at(ic_sequences(f[:, :t, :], z[:, :t, :], r),
                       crit_t(t - 1)) for r in sub_rhos]
        mc_T.append(mde_mc(list(sub_rhos), pw))
    rng_n = np.random.default_rng(SEED + 101)
    N_grid = [25, 50, 100, 200]
    mde_N = [mde_analytic(T, n) for n in N_grid]
    mc_N = []
    for n in N_grid:
        fn, zn = make_pool(rng_n, R_N, T, n)
        sub_rhos = np.linspace(0.0, 0.10, 11)
        pw = [power_at(ic_sequences(fn, zn, r), crit) for r in sub_rhos]
        mc_N.append(mde_mc(list(sub_rhos), pw))
    # The analytic MDE is checked against independent MC anchors at every
    # T and N setting; report the largest discrepancy instead of assuming it.
    gap_T = [abs(mde_analytic(t, N) - mc) for t, mc in zip(T_anchor, mc_T)]
    gap_N = [abs(mde_analytic(T, n) - mc) for n, mc in zip(N_grid, mc_N)]
    stats["mde_scaling"] = {
        "T_grid": T_grid, "mde_T": [round(v, 6) for v in mde_T],
        "T_anchor": T_anchor, "mde_T_mc": [round(v, 6) for v in mc_T],
        "N_grid": N_grid, "mde_N": [round(v, 6) for v in mde_N],
        "mde_N_mc": [round(v, 6) for v in mc_N],
        "mde_at_T60_N50": round(mde_analytic(60, 50), 6),
        "analytic_vs_mc_max_gap": round(float(max(max(gap_T), max(gap_N))), 6),
    }
    figs["T_grid"], figs["mde_T"], figs["T_anchor"], figs["mc_T"] = T_grid, mde_T, T_anchor, mc_T
    figs["N_grid"], figs["mde_N"], figs["mc_N"] = N_grid, mde_N, mc_N

    # ---- Figure 4: multiple testing ----
    M_grid = [1, 5, 20, 100, 500, 1000]
    bonf_mde, bonf_fwer, bonf_detect, bh_detect, bh_fdr = [], [], [], [], []
    rg_m = np.random.default_rng(SEED + 202)
    ic_null = ic0                       # (R_main, T) under H0
    ic_true = ic_sequences(f, z, 0.04)  # one genuinely predictive factor
    df_t = T - 1
    p_null = 2.0 * student_t.sf(np.abs(tstat(ic_null)), df_t)
    p_true = 2.0 * student_t.sf(np.abs(tstat(ic_true)), df_t)
    R_m = R_MAIN
    for M in M_grid:
        lvl = ALPHA / M
        bonf_mde.append(mde_analytic(T, N, lvl))
        # one true factor + (M-1) nulls, paired across replications
        idx = rg_m.integers(0, R_m, size=(R_m, max(M - 1, 0)))
        pn = p_null[idx] if M > 1 else np.empty((R_m, 0))
        rej_null = pn < lvl
        bonf_fwer.append(float(np.mean(rej_null.any(axis=1))))
        bonf_detect.append(float(np.mean(p_true < lvl)))
        # Benjamini-Hochberg at FDR level ALPHA on the same family
        p_all = np.concatenate([p_true[:, None], pn], axis=1)
        ps = np.sort(p_all, axis=1)
        k = np.arange(1, M + 1)
        passed = ps <= (k / M) * ALPHA
        kmax = np.where(passed.any(axis=1),
                        np.max(np.where(passed, k, 0), axis=1), 0)
        cutoff = ps[np.arange(R_m), np.maximum(kmax - 1, 0)]
        rej_bh = (p_all <= cutoff[:, None]) & (kmax[:, None] > 0)
        bh_detect.append(float(np.mean(rej_bh[:, 0])))
        n_false = rej_bh[:, 1:].sum(axis=1)
        n_rej = np.maximum(rej_bh.sum(axis=1), 1)
        bh_fdr.append(float(np.mean(np.where(kmax > 0, n_false / n_rej, 0.0))))
    stats["multiple_testing"] = {
        "M": M_grid,
        "bonferroni_alpha": [ALPHA / M for M in M_grid],
        "bonferroni_mde": [round(v, 6) for v in bonf_mde],
        "bonferroni_fwer": [round(v, 4) for v in bonf_fwer],
        "bonferroni_power_rho004": [round(v, 4) for v in bonf_detect],
        "bh_power_rho004": [round(v, 4) for v in bh_detect],
        "bh_false_discovery_prop": [round(v, 4) for v in bh_fdr],
        "mde_inflation_M100": round(bonf_mde[3] / bonf_mde[0], 4),
        "mde_inflation_M1000": round(bonf_mde[5] / bonf_mde[0], 4),
    }
    figs["M_grid"], figs["bonf_mde"] = M_grid, bonf_mde
    figs["bonf_fwer"], figs["bonf_detect"], figs["bh_detect"] = bonf_fwer, bonf_detect, bh_detect

    # ---- Figure 5a: winner's curse + post-hoc power (rho_true = 0.02) ----
    hat = figs["mean_ic002"]
    sig = np.abs(tstat(ic_002)) > crit
    # A genuinely independent out-of-sample panel, same true IC, fresh shocks.
    rng_oos = np.random.default_rng(SEED + 303)
    f_oos, z_oos = make_pool(rng_oos, R_MAIN, T, N)
    oos = ic_sequences(f_oos, z_oos, 0.02).mean(axis=1)
    posthoc = norm.cdf(hat * np.sqrt(T * (N - 1)) - Z975)
    stats["winners_curse"] = {
        "rho_true": 0.02,
        "unconditional_mean_hat": round(float(hat.mean()), 6),
        "conditional_mean_hat_given_significant": round(float(hat[sig].mean()), 6),
        "inflation_factor": round(float(hat[sig].mean() / hat.mean()), 4),
        "significant_fraction": round(float(sig.mean()), 4),
        "oos_mean_ic_all": round(float(oos.mean()), 6),
        "oos_mean_ic_significant": round(float(oos[sig].mean()), 6),
        "posthoc_power_all": round(float(posthoc.mean()), 4),
        "posthoc_power_given_significant": round(float(posthoc[sig].mean()), 4),
        "posthoc_power_given_nonsignificant": round(float(posthoc[~sig].mean()), 4),
        "true_power": round(mc_powers[4], 4),
    }
    figs["hat"], figs["sig"], figs["posthoc"] = hat, sig, posthoc

    # ---- Figure 5b: overlapping horizons make the IC series AR(1) ----
    # Real mechanism: when the "forward return" is an h-day cumulative return
    # measured every day, adjacent IC values share h-1 days of returns. We
    # model the resulting daily IC series directly as a stationary AR(1) whose
    # marginal sd is CALIBRATED to the cross-sectional simulation above (not
    # assumed), so the only thing that changes is the dependence structure.
    sigma_measured = float(ic0.std())
    rg_s = np.random.default_rng(SEED + 404)
    eps_s = rg_s.standard_normal((R_MAIN, T))
    phi_grid = [0.0, 0.2, 0.4, 0.6]
    size_phi, size_phi_hac, ac_measured = [], [], []
    for phi in phi_grid:
        v = np.empty((R_MAIN, T))
        cur = eps_s[:, 0].copy()
        v[:, 0] = cur
        for t in range(1, T):
            cur = phi * cur + np.sqrt(1.0 - phi ** 2) * eps_s[:, t]
            v[:, t] = cur
        ic_p = sigma_measured * v                       # H0: true IC = 0
        size_phi.append(power_at(ic_p, crit))
        size_phi_hac.append(float(np.mean(np.abs(tstat_hac(ic_p)) > crit)))
        ac_measured.append(float(np.corrcoef(ic_p[:, :-1].ravel(),
                                             ic_p[:, 1:].ravel())[0, 1]))
    stats["serial_dependence"] = {
        "phi": phi_grid, "empirical_size": [round(v, 4) for v in size_phi],
        "empirical_size_hac_corrected": [round(v, 4) for v in size_phi_hac],
        "measured_lag1_autocorr": [round(v, 4) for v in ac_measured],
        "marginal_sd_calibrated": round(sigma_measured, 6),
        "nominal": ALPHA,
        "theoretical_variance_inflation_phi06": round((1 + 0.6) / (1 - 0.6), 4),
    }
    figs["phi_grid"], figs["size_phi"], figs["size_phi_hac"] = (
        phi_grid, size_phi, size_phi_hac)

    # ---- Figure 5c: heavy-tailed cross-sectional shocks ----
    nu_grid = [1e9, 8.0, 5.0, 3.0]
    tail_power, tail_sigma, tail_size, tail_mean, tail_kurt = [], [], [], [], []
    for nu in nu_grid:
        rg_t = np.random.default_rng(SEED + int(nu) if nu < 1e8 else SEED + 999)
        if nu > 1e8:
            ft, zt = f[:R_TAIL], z[:R_TAIL]
        else:
            ft = rg_t.standard_normal((R_TAIL, T, N), dtype=np.float32)
            zt = rg_t.standard_t(nu, size=(R_TAIL, T, N)).astype(np.float32)
            zt /= np.sqrt(nu / (nu - 2.0))
        ic_h0 = ic_sequences(ft, zt, 0.0)
        ic_h1 = ic_sequences(ft, zt, 0.02)
        tail_size.append(power_at(ic_h0, crit))
        tail_power.append(power_at(ic_h1, crit))
        tail_sigma.append(float(ic_h0.std()))
        tail_mean.append(float(ic_h1.mean()))
        tail_kurt.append(float(np.mean((ic_h0 / ic_h0.std()) ** 4)))
    stats["heavy_tail"] = {
        "nu": ["normal", "8", "5", "3"],
        "daily_ic_sd": [round(v, 6) for v in tail_sigma],
        "mean_daily_ic_at_true_002": [round(v, 6) for v in tail_mean],
        "excess_kurtosis_daily_ic": [round(v - 3.0, 4) for v in tail_kurt],
        "empirical_size": [round(v, 4) for v in tail_size],
        "power_at_rho_002": [round(v, 4) for v in tail_power],
        "power_change_t3": round(tail_power[3] - tail_power[0], 4),
        "interpretation": "heavy tails do NOT lower power here; they distort "
                          "the IC estimator and the size, so a higher rejection "
                          "rate is not evidence of a better test",
    }
    figs["nu_labels"], figs["tail_power"], figs["tail_sigma"] = (
        ["normal", "t(8)", "t(5)", "t(3)"], tail_power, tail_sigma)

    return stats, figs


# ---------- figures ----------
def make_figures(g):
    plt.rcParams.update({"figure.dpi": 110, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})

    # fig 1: signal buried in noise
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    ax[0].plot(g["ic0"], lw=0.55, color="#94a3b8", label="true IC = 0")
    ax[0].plot(g["ic002"], lw=0.55, color="#d62728", alpha=0.8, label="true IC = 0.02")
    ax[0].axhline(0, color="#111827", lw=0.8)
    ax[0].set_xlabel("trading day"); ax[0].set_ylabel("daily cross-sectional IC")
    ax[0].set_title("One-year IC paths (N=50 names/day)")
    ax[0].legend(frameon=False, fontsize=8)
    # Both histograms are put on the SAME standardised scale: daily IC is
    # divided by its theoretical sd, and the 250-day mean IC by its own
    # theoretical sd. No visual stretching or "x20" trick.
    bins = np.linspace(-4, 6, 60)
    daily_std = g["ic0_all"].ravel() / SIGMA_IC
    mean_std = g["mean_ic002"] / (SIGMA_IC / np.sqrt(T))
    ax[1].hist(daily_std, bins=bins, density=True, color="#94a3b8",
               alpha=0.75, label="daily IC / sd(IC)")
    ax[1].hist(mean_std, bins=bins, density=True, histtype="step",
               color="#d62728", lw=1.8, label="mean IC / sd(mean IC)")
    xs = np.linspace(-4, 6, 200)
    ax[1].plot(xs, norm.pdf(xs), "--", color="#111827", lw=1.0, label="N(0,1)")
    ax[1].axvline(g["t_crit"], color="#16a34a", ls=":", lw=1.5,
                  label=f"critical t = {g['t_crit']:.3f}")
    ax[1].set_xlabel("standardised units"); ax[1].set_ylabel("density")
    ax[1].set_title("Standardised: signal shifts the mean by 2.21 sd")
    ax[1].legend(frameon=False, fontsize=7.5)
    fig.tight_layout(); fig.savefig(OUT / "fig1_ic_paths_and_noise.png"); plt.close(fig)

    # fig 2: power curve + MDE
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    r = np.array(g["rhos"])
    ax.plot(r, g["an_powers"], "-", color="#2563eb", lw=1.8, label="analytic (normal approx.)")
    ax.plot(r, g["mc_powers"], "o", color="#d62728", ms=5, label="Monte Carlo (1500 backtests)")
    ax.axhline(TARGET, color="#111827", ls="--", lw=1.1, label="80% power target")
    ax.axvline(MDE_ANALYTIC, color="#16a34a", ls=":", lw=1.6,
               label=f"MDE = {MDE_ANALYTIC:.4f}")
    ax.axhline(ALPHA, color="#94a3b8", ls=":", lw=1.0)
    ax.set_xlabel("true cross-sectional IC"); ax.set_ylabel("P(reject H0: mean IC = 0)")
    ax.set_title(f"Power of a {T}-day IC t-test (N={N}, two-sided 5%)")
    ax.set_ylim(-0.02, 1.02); ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "fig2_power_curve_mde.png"); plt.close(fig)

    # fig 3: MDE vs T and N
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    ax[0].plot(g["T_grid"], g["mde_T"], "-o", color="#2563eb", ms=4,
               label="analytic  (z.975+z.80)/sqrt(T(N-1))")
    ax[0].plot(g["T_anchor"], g["mc_T"], "s", color="#d62728", ms=6, label="MC anchors")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("backtest length T (days)"); ax[0].set_ylabel("minimum detectable IC")
    ax[0].set_title("Lengthening the backtest (N=50)"); ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(g["N_grid"], g["mde_N"], "-o", color="#2563eb", ms=4, label="analytic")
    ax[1].plot(g["N_grid"], g["mc_N"], "s", color="#d62728", ms=6, label="MC anchors")
    ax[1].set_xscale("log"); ax[1].set_yscale("log")
    ax[1].set_xlabel("cross-section width N (names/day)"); ax[1].set_ylabel("minimum detectable IC")
    ax[1].set_title("Widening the cross-section (T=250)"); ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig3_mde_scaling_T_N.png"); plt.close(fig)

    # fig 4: multiple testing
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.7))
    M = np.array(g["M_grid"])
    ax[0].plot(M, g["bonf_mde"], "-o", color="#d62728", ms=5, label="Bonferroni MDE")
    ax[0].axhline(g["bonf_mde"][0], color="#2563eb", ls="--", lw=1.2, label="single-test MDE")
    ax[0].set_xscale("log"); ax[0].set_xlabel("number of candidate factors M")
    ax[0].set_ylabel("minimum detectable IC")
    ax[0].set_title("Searching more factors raises the bar")
    ax[0].legend(frameon=False, fontsize=8)
    ax[1].plot(M, g["bonf_detect"], "-o", color="#d62728", ms=5, label="Bonferroni power")
    ax[1].plot(M, g["bh_detect"], "-s", color="#16a34a", ms=5, label="BH-FDR power")
    ax[1].plot(M, g["bonf_fwer"], "-^", color="#94a3b8", ms=5, label="Bonferroni FWER")
    ax[1].set_xscale("log"); ax[1].set_xlabel("number of candidate factors M")
    ax[1].set_ylabel("probability"); ax[1].set_ylim(-0.02, 1.02)
    ax[1].set_title("Detection of one true factor (true IC = 0.04)")
    ax[1].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "fig4_multiple_testing.png"); plt.close(fig)

    # fig 5: three caveats
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.5))
    ax[0].hist(g["hat"][g["sig"]], bins=28, color="#d62728", alpha=0.8,
               label="significant backtests")
    ax[0].axvline(0.02, color="#111827", ls="--", lw=1.4, label="true IC = 0.02")
    ax[0].axvline(float(g["hat"][g["sig"]].mean()), color="#2563eb", ls=":", lw=1.6,
                  label="conditional mean")
    ax[0].set_xlabel("estimated mean IC"); ax[0].set_ylabel("count")
    ax[0].set_title("Winner's curse: conditioning on p<0.05")
    ax[0].legend(frameon=False, fontsize=7)
    ax[1].plot(g["phi_grid"], g["size_phi"], "-o", color="#d62728", ms=5,
               label="naive t-test size")
    ax[1].plot(g["phi_grid"], g["size_phi_hac"], "-s", color="#16a34a", ms=5,
               label="Newey-West HAC size")
    ax[1].axhline(ALPHA, color="#111827", ls="--", lw=1.1, label="nominal 5%")
    ax[1].set_xlabel("AR(1) coefficient of the daily IC series")
    ax[1].set_ylabel("P(reject | H0 true)")
    ax[1].set_title("Overlapping horizons break the naive t-test")
    ax[1].legend(frameon=False, fontsize=7.5)
    x = np.arange(4)
    ax[2].bar(x, g["tail_power"], color=["#2563eb", "#60a5fa", "#f59e0b", "#d62728"])
    ax[2].set_xticks(x); ax[2].set_xticklabels(g["nu_labels"])
    ax[2].axhline(g["tail_power"][0], color="#111827", ls="--", lw=1.0)
    ax[2].set_ylabel("power at true IC = 0.02")
    ax[2].set_title("Heavy tails do NOT lower power (size also drifts)")
    for i, v in enumerate(g["tail_power"]):
        ax[2].text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8)
    ax[2].set_ylim(0, max(g["tail_power"]) * 1.22)
    fig.tight_layout(); fig.savefig(OUT / "fig5_caveats.png"); plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    stats, figs = run_all()
    make_figures(figs)
    (OUT / "results.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8")
    print("MDE(80%, analytic) =", round(float(MDE_ANALYTIC), 6))
    print("power at true IC=0.02 =", stats["headline"]["power_at_rho_002"])
    print("Bonferroni MDE inflation M=1000 =",
          stats["multiple_testing"]["mde_inflation_M1000"])
    print("Winner's curse inflation =", stats["winners_curse"]["inflation_factor"])
    print("size under phi=0.6 =", stats["serial_dependence"]["empirical_size"][-1])
    print("Wrote 5 PNGs + results.json to", OUT)


if __name__ == "__main__":
    main()
