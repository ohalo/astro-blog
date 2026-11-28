"""Bachelier normal-vol pricing: parity, smile inversion, negative strikes and shift sensitivity."""

def main():
    import json
    import math
    import os

    import numpy as np
    from scipy.optimize import brentq
    from scipy.stats import norm

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "/Users/halo/workspace/astro-blog/public/images/bachelier-normal-volatility-pricing"
    os.makedirs(OUT, exist_ok=True)

    S0 = 100.0
    R = 0.04
    T = 1.0
    DF = math.exp(-R * T)
    F = S0 * math.exp(R * T)
    SIG_L = 0.20
    SIG_N = F * SIG_L
    STRIKES = [60.0, 80.0, 100.0, F, 120.0, 140.0]

    def bach_call(k, sn):
        d = (F - k) / (sn * math.sqrt(T))
        return DF * ((F - k) * norm.cdf(d) + sn * math.sqrt(T) * norm.pdf(d))

    def bach_put(k, sn):
        d = (F - k) / (sn * math.sqrt(T))
        return DF * ((k - F) * norm.cdf(-d) + sn * math.sqrt(T) * norm.pdf(d))

    def bs_call(k, sl):
        if k <= 0:
            return float("nan")
        d1 = (math.log(S0 / k) + (R + 0.5 * sl * sl) * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return S0 * norm.cdf(d1) - DF * k * norm.cdf(d2)

    def bs_put(k, sl):
        if k <= 0:
            return float("nan")
        d1 = (math.log(S0 / k) + (R + 0.5 * sl * sl) * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return DF * k * norm.cdf(-d2) - S0 * norm.cdf(-d1)

    def shift_call(k, sl, a):
        return bs_call(k + a, sl)

    # 1) exact parity for both models.
    parity = {
        "bach_max_abs": max(abs(bach_call(k, SIG_N) - bach_put(k, SIG_N) - DF * (F - k)) for k in STRIKES),
        "bs_max_abs": max(abs(bs_call(k, SIG_L) - bs_put(k, SIG_L) - DF * (F - k)) for k in STRIKES),
    }

    # 2) ATM-matched price curves.
    prices = {
        "strike": STRIKES,
        "bachelier": [bach_call(k, SIG_N) for k in STRIKES],
        "lognormal": [bs_call(k, SIG_L) for k in STRIKES],
    }
    abs_diff = [abs(a - b) for a, b in zip(prices["bachelier"], prices["lognormal"])]
    kgrid = np.linspace(60.0, 140.0, 161)

    # 3) cross-model smile inversion: each model prices, the other model inverts.
    k_smile = [70.0, 80.0, 90.0, 100.0, F, 110.0, 120.0, 130.0]
    normal_iv = [brentq(lambda s: bach_call(k, s) - bs_call(k, SIG_L), 0.01, 400.0) for k in k_smile]
    lognormal_iv = [brentq(lambda s: bs_call(k, s) - bach_call(k, SIG_N), 1e-4, 4.0) for k in k_smile]
    smile = {
        "strike": k_smile,
        "normal_iv_from_lognormal": normal_iv,
        "lognormal_iv_from_normal": lognormal_iv,
        "normal_iv_spread": max(normal_iv) - min(normal_iv),
        "lognormal_iv_spread": max(lognormal_iv) - min(lognormal_iv),
    }

    # 4) near-zero forward: negative strikes are legal in Bachelier, not in lognormal.
    F2 = 1.0
    SN2 = 1.6
    DF2 = DF
    neg_strikes = [-0.4, -0.2, 0.0, 0.5, F2, 2.0]
    shifts = [0.6, 1.0, 2.0, 5.0]

    def bach_call_f(k, fwd, sn):
        d = (fwd - k) / (sn * math.sqrt(T))
        return DF2 * ((fwd - k) * norm.cdf(d) + sn * math.sqrt(T) * norm.pdf(d))

    def shifted_price(k, a, sl):
        d1 = (math.log((F2 + a) / (k + a)) + 0.5 * sl * sl * T) / (sl * math.sqrt(T))
        d2 = d1 - sl * math.sqrt(T)
        return DF2 * ((F2 + a) * norm.cdf(d1) - (k + a) * norm.cdf(d2))

    # Calibrate every shifted model to the same at-the-forward price.
    atm_normal = bach_call_f(F2, F2, SN2)
    shift_sl = {a: brentq(lambda s, aa=a: shifted_price(F2, aa, s) - atm_normal, 1e-4, 10.0)
                for a in shifts}
    shift_grid = {str(a): [shifted_price(k, a, shift_sl[a]) for k in neg_strikes] for a in shifts}
    shift_atm_err = {str(a): abs(shifted_price(F2, a, shift_sl[a]) - atm_normal) for a in shifts}
    neg = {
        "strikes": neg_strikes,
        "bachelier": [bach_call_f(k, F2, SN2) for k in neg_strikes],
        "shifted": shift_grid,
        "shift_atm_error": shift_atm_err,
        "spread_k0": max(v[1] for v in shift_grid.values()) - min(v[1] for v in shift_grid.values()),
        "spread_kneg": max(v[0] for v in shift_grid.values()) - min(v[0] for v in shift_grid.values()),
        "atm_normal": atm_normal,
    }

    # 5) ATM normal/lognormal conversion error grows with volatility.
    vols = [0.05, 0.10, 0.20, 0.40, 0.80]
    conv = []
    for sl in vols:
        p_bs = bs_call(F, sl)
        p_ba = bach_call(F, F * sl)
        conv.append({"sigma_log": sl, "bs": p_bs, "bachelier": p_ba, "rel_error": (p_ba - p_bs) / p_bs})

    # 6) Monte Carlo cross-check of both closed forms (fixed seed).
    rng = np.random.default_rng(20261128)
    paths = 300000
    zn = rng.standard_normal(paths)
    st_normal = F + SIG_N * math.sqrt(T) * zn
    st_log = F * np.exp(-0.5 * SIG_L * SIG_L * T + SIG_L * math.sqrt(T) * zn)
    mc = {}
    for k in (80.0, F, 120.0):
        pn = np.maximum(st_normal - k, 0.0)
        pl = np.maximum(st_log - k, 0.0)
        mc[str(k)] = {
            "normal": DF * float(pn.mean()),
            "normal_se": DF * float(pn.std(ddof=1)) / math.sqrt(paths),
            "log": DF * float(pl.mean()),
            "log_se": DF * float(pl.std(ddof=1)) / math.sqrt(paths),
        }

    results = {
        "params": {"S0": S0, "r": R, "T": T, "DF": DF, "F": F, "sigma_log": SIG_L,
                   "sigma_normal": SIG_N, "paths": paths},
        "parity": parity,
        "prices": prices,
        "max_abs_price_diff": max(abs_diff),
        "smile": smile,
        "negative_forward": neg,
        "atm_conversion": conv,
        "monte_carlo": mc,
    }

    # fig1: terminal density, normal vs lognormal at matched ATM volatility.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    x = np.linspace(-20, 180, 700)
    dens_n = norm.pdf(x, loc=F, scale=SIG_N * math.sqrt(T))
    dens_l = np.zeros_like(x)
    pos = x > 0
    dens_l[pos] = np.exp(norm.logpdf(np.log(x[pos]), math.log(F) - 0.5 * SIG_L ** 2 * T,
                                     SIG_L * math.sqrt(T))) / x[pos]
    ax.plot(x, dens_n, color="#4c78a8", lw=2, label="Bachelier normal")
    ax.plot(x, dens_l, color="#e45756", lw=2, label="GBM lognormal")
    ax.axvline(120, color="black", ls="--", lw=1, label="K = 120")
    ax.set_xlabel("terminal forward price")
    ax.set_ylabel("density")
    ax.set_title("Same forward and ATM volatility, different tails")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_terminal_density.png"), dpi=150)
    plt.close(fig)

    # fig2: price curves and parity residuals.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(kgrid, [bach_call(float(k), SIG_N) for k in kgrid], color="#4c78a8", label="Bachelier")
    axes[0].plot(kgrid, [bs_call(float(k), SIG_L) for k in kgrid], color="#e45756", label="Black-Scholes")
    axes[0].axvline(F, color="gray", ls=":", lw=1)
    axes[0].set_xlabel("strike K")
    axes[0].set_ylabel("discounted call price")
    axes[0].set_title("ATM-matched, not strike-matched")
    axes[0].legend(fontsize=9)
    axes[1].bar(range(len(STRIKES)), abs_diff, color="#72b7b2")
    axes[1].set_xticks(range(len(STRIKES)), [str(int(k)) for k in STRIKES], rotation=30)
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("absolute price gap")
    axes[1].set_title("Largest gap away from the forward")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_curves.png"), dpi=150)
    plt.close(fig)

    # fig3: implied vol smiles seen by the other model.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(k_smile, normal_iv, "o-", color="#4c78a8")
    axes[0].axhline(SIG_N, color="gray", ls="--", lw=1, label="true normal vol")
    axes[0].set_xlabel("strike K")
    axes[0].set_ylabel("implied normal vol")
    axes[0].set_title("Lognormal prices -> normal smile")
    axes[0].legend(fontsize=9)
    axes[1].plot(k_smile, lognormal_iv, "o-", color="#e45756")
    axes[1].axhline(SIG_L, color="gray", ls="--", lw=1, label="true log vol")
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("implied lognormal vol")
    axes[1].set_title("Normal prices -> lognormal smile")
    axes[1].legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_cross_smiles.png"), dpi=150)
    plt.close(fig)

    # fig4: near-zero forward and shifted-lognormal instability.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    ax.plot(neg_strikes, neg["bachelier"], "ko-", lw=2, label="Bachelier (K may be <= 0)")
    for a in shifts:
        ax.plot(neg_strikes, shift_grid[str(a)], "o-", lw=1.4, label=f"shifted lognormal, a={a}")
    ax.axvline(0, color="gray", ls=":", lw=1)
    ax.set_xlabel("strike K")
    ax.set_ylabel("discounted call price")
    ax.set_title("All shifts match ATM, then disagree between strikes")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_negative_forward.png"), dpi=150)
    plt.close(fig)

    # fig5: ATM conversion error and Monte Carlo validation.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot(vols, [c["rel_error"] for c in conv], "o-", color="#e45756")
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].set_xlabel("lognormal volatility")
    axes[0].set_ylabel("relative ATM pricing error")
    axes[0].set_title("sigma_N = F sigma_L is only a first-order rule")
    ks = sorted(mc, key=float)
    exact_n = [bach_call(float(k), SIG_N) for k in ks]
    exact_l = [bs_call(float(k), SIG_L) for k in ks]
    axes[1].errorbar(ks, [mc[k]["normal"] for k in ks], yerr=[2 * mc[k]["normal_se"] for k in ks],
                     fmt="o", capsize=4, color="#4c78a8", label="normal MC")
    axes[1].plot(ks, exact_n, "s--", color="#4c78a8", label="normal exact")
    axes[1].errorbar(ks, [mc[k]["log"] for k in ks], yerr=[2 * mc[k]["log_se"] for k in ks],
                     fmt="o", capsize=4, color="#e45756", label="log MC")
    axes[1].plot(ks, exact_l, "s--", color="#e45756", label="log exact")
    axes[1].set_xlabel("strike K")
    axes[1].set_ylabel("discounted call price")
    axes[1].set_title("300k-path MC vs closed forms (2 SE bars)")
    axes[1].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_conversion_and_mc.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
