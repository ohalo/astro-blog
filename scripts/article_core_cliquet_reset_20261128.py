"""Cliquet option: reset frequency, capped/floored returns and jump risk."""

def main():
    import json
    import math
    import os

    import numpy as np
    from scipy.stats import norm, poisson

    SEED = 20261128
    OUT = "/Users/halo/workspace/astro-blog/public/images/cliquet-reset-frequency-path-dependence"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    S0 = 100.0
    R = 0.04
    T = 1.0
    SIG = 0.20
    LAM = 0.8
    MUJ = -0.06
    SDJ = 0.12
    CAP = 0.08
    FLOOR = 0.08
    STEPS = 252
    PATHS = 120000
    CHUNK = 20000
    RESETS = [1, 2, 4, 12, 63, 252]

    kappa = math.exp(MUJ + 0.5 * SDJ ** 2) - 1.0
    mu_log = R - LAM * kappa - 0.5 * SIG ** 2
    dt = T / STEPS
    a_log = math.log(1.0 - FLOOR)
    b_log = math.log(1.0 + CAP)

    def analytic(lam, muj, sdj, sig, horizon):
        """Closed-form per-period expectation by Poisson mixing over jump counts."""
        kap = math.exp(muj + 0.5 * sdj ** 2) - 1.0
        drift = R - lam * kap - 0.5 * sig ** 2
        per = 0.0
        pc = 0.0
        pf = 0.0
        for k in range(16):
            w = poisson.pmf(k, lam * horizon)
            mk = drift * horizon + k * muj
            sk = math.sqrt(max(sig ** 2 * horizon + k * sdj ** 2, 1e-18))
            za = (a_log - mk) / sk
            zb = (b_log - mk) / sk
            Fa, Fb = norm.cdf(za), norm.cdf(zb)
            Da = norm.cdf(za - sk)
            Db = norm.cdf(zb - sk)
            growth = math.exp(mk + 0.5 * sk * sk)
            per += w * (-FLOOR * Fa + growth * (Db - Da) - (Fb - Fa) + CAP * (1.0 - Fb))
            pc += w * (1.0 - Fb)
            pf += w * Fa
        return per, pc, pf

    # Flat-vol benchmark: match total variance rate of the jump-diffusion log return.
    sig_flat = math.sqrt(SIG ** 2 + LAM * (MUJ ** 2 + SDJ ** 2))
    true_daily, true_cap_p, true_floor_p = analytic(LAM, MUJ, SDJ, SIG, dt)
    flat_daily, _, _ = analytic(0.0, 0.0, 1.0, sig_flat, dt)

    # One shared panel of daily log returns, reused for every reset frequency.
    log_paths = np.empty((PATHS, STEPS), dtype=np.float64)
    for start in range(0, PATHS, CHUNK):
        end = min(start + CHUNK, PATHS)
        n = end - start
        z = rng.standard_normal((n, STEPS))
        counts = rng.poisson(LAM * dt, size=(n, STEPS))
        jumps = MUJ * counts + SDJ * np.sqrt(counts) * rng.standard_normal((n, STEPS))
        log_paths[start:end] = mu_log * dt + SIG * math.sqrt(dt) * z + jumps

    def mc_payoff(n):
        step = STEPS // n
        per_log = log_paths[:, : n * step].reshape(PATHS, n, step).sum(axis=2)
        gross = np.exp(per_log) - 1.0
        clipped = np.clip(gross, -FLOOR, CAP)
        payoff = clipped.sum(axis=1) * math.exp(-R * T)
        se = payoff.std(ddof=1) / math.sqrt(PATHS)
        return payoff.mean(), se, float((gross > CAP).mean()), float((gross < -FLOOR).mean()), payoff

    mc = {}
    for n in RESETS:
        mean, se, cp, fp, p = mc_payoff(n)
        mc[n] = {"price": float(mean), "se": float(se), "cap_hit": cp, "floor_hit": fp}
        if n == 12:
            payoff_12 = p
        if n == 252:
            payoff_252 = p

    prices = {
        "jd_analytic": {},
        "flat_analytic": {},
        "mc": {str(n): mc[n]["price"] for n in RESETS},
    }
    hit_exact = {}
    for n in RESETS:
        h = T / n
        per, cp, fp = analytic(LAM, MUJ, SDJ, SIG, h)
        flat_p, _, _ = analytic(0.0, 0.0, 1.0, sig_flat, h)
        prices["jd_analytic"][str(n)] = float(math.exp(-R * T) * n * per)
        prices["flat_analytic"][str(n)] = float(math.exp(-R * T) * n * flat_p)
        hit_exact[str(n)] = {"cap": cp, "floor": fp}
    gap = {str(n): prices["jd_analytic"][str(n)] - prices["flat_analytic"][str(n)] for n in RESETS}

    # Jump sensitivity: vary intensity with either negative or positive jump mean.
    sens = {}
    for sign, muj in (("negative", MUJ), ("positive", -MUJ)):
        rows = {}
        for lam in [0.0, 0.4, 0.8, 1.6, 3.0]:
            sig_flat_l = math.sqrt(SIG ** 2 + lam * (muj ** 2 + SDJ ** 2))
            rows[str(lam)] = {}
            for n in (12, 252):
                pp, _, _ = analytic(lam, muj, SDJ, SIG, T / n)
                ff, _, _ = analytic(0.0, 0.0, 1.0, sig_flat_l, T / n)
                rows[str(lam)][f"n{n}"] = float(math.exp(-R * T) * n * (pp - ff))
        sens[sign] = rows

    results = {
        "params": {"S0": S0, "r": R, "T": T, "sigma": SIG, "lambda": LAM, "muJ": MUJ,
                   "sdJ": SDJ, "cap": CAP, "floor": FLOOR, "paths": PATHS, "seed": SEED,
                   "sigma_flat": float(sig_flat), "kappa": float(kappa)},
        "per_period": {"jd_daily": float(true_daily), "flat_daily": float(flat_daily),
                       "cap_prob_daily": float(true_cap_p), "floor_prob_daily": float(true_floor_p),
                       "hit_by_n": hit_exact},
        "prices": prices,
        "gap": gap,
        "mc": {str(n): mc[n] for n in RESETS},
        "sensitivity": sens,
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # fig1: representative paths and the clipped increments they generate.
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    idxs = [7, 41, 88]
    step12 = STEPS // 12
    for j, idx in enumerate(idxs):
        s = S0 * np.exp(np.concatenate([[0.0], np.cumsum(log_paths[idx])]))
        axes[0].plot(np.arange(STEPS + 1), s, lw=1.2, label=f"path {j + 1}")
        per_log = log_paths[idx, :12 * step12].reshape(12, step12).sum(axis=1)
        per_ret = np.exp(per_log) - 1.0
        axes[1].vlines(np.arange(1, 13) * step12, 0, np.clip(per_ret, -FLOOR, CAP),
                       alpha=0.75, label=f"path {j + 1}")
    axes[0].set_ylabel("underlying price")
    axes[0].set_title("Jump-diffusion paths with 12 reset dates")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].axhline(CAP, color="red", ls="--", lw=0.8)
    axes[1].axhline(-FLOOR, color="green", ls="--", lw=0.8)
    axes[1].set_ylabel("capped increment")
    axes[1].set_xlabel("trading day")
    axes[1].legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_paths_clipped.png"), dpi=150)
    plt.close(fig)

    # fig2: price vs reset frequency.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    ns = np.array(RESETS, dtype=float)
    jd = np.array([prices["jd_analytic"][str(n)] for n in RESETS])
    fl = np.array([prices["flat_analytic"][str(n)] for n in RESETS])
    mp = np.array([mc[n]["price"] for n in RESETS])
    ms = np.array([mc[n]["se"] for n in RESETS])
    ax.plot(ns, jd, "o-", color="#4c78a8", label="jump-diffusion exact")
    ax.plot(ns, fl, "s--", color="#e45756", label="flat-vol benchmark")
    ax.errorbar(ns, mp, yerr=2 * ms, fmt="^", color="black", capsize=3, label="Monte Carlo +/-2SE")
    ax.set_xscale("log")
    ax.set_xlabel("number of resets per year")
    ax.set_ylabel("discounted cliquet price")
    ax.set_title("Reset frequency changes the price of path dependence")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_vs_frequency.png"), dpi=150)
    plt.close(fig)

    # fig3: hit probabilities and expected annual counts.
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.4))
    ax = axes[0]
    ax.plot(ns, [mc[n]["cap_hit"] for n in RESETS], "o", color="#e45756", label="MC cap hit")
    ax.plot(ns, [mc[n]["floor_hit"] for n in RESETS], "s", color="#2ca02c", label="MC floor hit")
    ax.plot(ns, [hit_exact[str(n)]["cap"] for n in RESETS], "--", color="#e45756", lw=1.2)
    ax.plot(ns, [hit_exact[str(n)]["floor"] for n in RESETS], "--", color="#2ca02c", lw=1.2)
    ax.set_xscale("log")
    ax.set_xlabel("resets per year")
    ax.set_ylabel("per-period hit probability")
    ax.set_title("Monte Carlo (markers) vs exact (dashed)")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(ns, [12 * mc[n]["cap_hit"] for n in RESETS], "o-", color="#e45756", label="cap events/year")
    ax.plot(ns, [12 * mc[n]["floor_hit"] for n in RESETS], "s-", color="#2ca02c", label="floor events/year")
    ax.set_xscale("log")
    ax.set_xlabel("resets per year")
    ax.set_ylabel("expected events per year (12-period scale)")
    ax.set_title("Frequent resets multiply barrier touches")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_hit_rates.png"), dpi=150)
    plt.close(fig)

    # fig4: mispricing against jump intensity, with both skew signs.
    fig, ax = plt.subplots(figsize=(8.4, 5.0))
    lams = [0.0, 0.4, 0.8, 1.6, 3.0]
    for sign, color in (("negative", "#4c78a8"), ("positive", "#e45756")):
        for key, ls in (("n12", "-"), ("n252", "--")):
            ax.plot(lams, [sens[sign][str(l)][key] for l in lams], color=color, ls=ls,
                    marker="o", label=f"{sign} skew, {key}")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("jump intensity lambda")
    ax.set_ylabel("exact price minus flat-vol price")
    ax.set_title("A flat-vol quote can err in either direction")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_jump_sensitivity.png"), dpi=150)
    plt.close(fig)

    # fig5: realised payoff distributions.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.hist(payoff_12, bins=70, density=True, alpha=0.55, color="#4c78a8", label="12 resets")
    ax.hist(payoff_252, bins=70, density=True, alpha=0.45, color="#e45756", label="252 resets")
    ax.axvline(prices["jd_analytic"]["12"], color="#4c78a8", ls="--", lw=1.4)
    ax.axvline(prices["jd_analytic"]["252"], color="#e45756", ls="--", lw=1.4)
    ax.set_xlabel("discounted total payoff")
    ax.set_ylabel("density")
    ax.set_title("Reset schedule reshapes the whole payoff distribution")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_payoff_distribution.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
