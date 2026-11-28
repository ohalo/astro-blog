"""Model-free basket call bounds from marginals: comonotonic, countermonotonic and copula prices."""

def main():
    import json
    import math
    import os

    import numpy as np
    from numpy.polynomial.legendre import leggauss
    from scipy.stats import qmc

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    OUT = "/Users/halo/workspace/astro-blog/public/images/comonotonic-basket-option-bounds"
    os.makedirs(OUT, exist_ok=True)

    R = 0.04
    T = 1.0
    DF = math.exp(-R * T)
    K = 100.0

    # Gauss-Legendre on u in (0,1); endpoints are clipped before ndtri.
    nodes, weights = leggauss(4000)
    u = np.clip((nodes + 1.0) / 2.0, 1e-9, 1 - 1e-9)
    wu = weights / 2.0

    def quantile(uu, forward, vol):
        return forward * np.exp(-0.5 * vol * vol * T + vol * math.sqrt(T) * ndtri_safe(uu))

    def ndtri_safe(uu):
        from scipy.special import ndtri
        return ndtri(uu)

    def bound_price(k, forwards, vols, wt, coupling):
        qs = [quantile(u, f, s) for f, s in zip(forwards, vols)]
        if coupling == "comonotonic":
            basket = sum(wi * qi for wi, qi in zip(wt, qs))
        elif coupling == "countermonotonic":
            basket = wt[0] * qs[0] + wt[1] * qs[1][::-1]
        else:
            raise ValueError(coupling)
        payoff = np.maximum(basket - k, 0.0)
        return float(DF * np.dot(wu, payoff))

    F2 = [100.0, 100.0]
    V2 = [0.20, 0.30]
    W2 = [0.5, 0.5]
    upper2 = bound_price(K, F2, V2, W2, "comonotonic")
    lower2 = bound_price(K, F2, V2, W2, "countermonotonic")

    def gaussian_price(rhos, forwards, vols, wt, k, log2m=20, seed=1234):
        dim = len(forwards)
        sobol = qmc.Sobol(d=dim, scramble=True, seed=seed)
        uv = sobol.random_base2(log2m)
        z = ndtri_safe(uv)
        if dim == 2:
            rho = rhos[0]
            z2 = rho * z[:, 0] + math.sqrt(1 - rho * rho) * z[:, 1]
            coords = np.column_stack([z[:, 0], z2])
        else:
            rho = rhos[0]
            coords = np.empty_like(z)
            coords[:, 0] = z[:, 0]
            for j in range(1, dim):
                coords[:, j] = math.sqrt(rho) * z[:, 0] + math.sqrt(1 - rho) * z[:, j]
        basket = np.zeros(len(uv))
        for wi, f, s, zz in zip(wt, forwards, vols, coords.T):
            basket += wi * f * np.exp(-0.5 * s * s * T + s * math.sqrt(T) * zz)
        payoff = np.maximum(basket - k, 0.0)
        return float(DF * payoff.mean())

    def spearman(rho):
        return (6.0 / math.pi) * math.asin(rho / 2.0)

    rho_grid = [-0.9, -0.5, 0.0, 0.5, 0.9, 0.99]
    rho_sweep = []
    for rho in rho_grid:
        p = gaussian_price([rho], F2, V2, W2, K)
        rho_sweep.append({"rho": rho, "spearman": spearman(rho), "price": p,
                          "retention": (p - lower2) / (upper2 - lower2)})

    indep2 = rho_sweep[2]["price"]

    k_grid = [80.0, 90.0, 100.0, 110.0, 120.0]
    strike_sweep = []
    for kk in k_grid:
        up = bound_price(kk, F2, V2, W2, "comonotonic")
        lo = bound_price(kk, F2, V2, W2, "countermonotonic")
        strike_sweep.append({"K": kk, "upper": up, "lower": lo, "width": up - lo})

    F5 = [95.0, 98.0, 100.0, 102.0, 105.0]
    V5 = [0.15, 0.20, 0.25, 0.30, 0.35]
    W5 = [0.2] * 5
    upper5 = bound_price(K, F5, V5, W5, "comonotonic")
    rho5_grid = [0.0, 0.25, 0.5, 0.9, 0.99, 0.9999]
    rho5 = [{"rho": rr, "price": gaussian_price([rr], F5, V5, W5, K, log2m=19, seed=77)}
            for rr in rho5_grid]

    # Heterogeneous vol spread: scale all vols, keep marginals fixed within each scenario.
    spread = []
    for scale in [0.5, 1.0, 1.5, 2.0]:
        vs = [v * scale for v in V5]
        up = bound_price(K, F5, vs, W5, "comonotonic")
        px = gaussian_price([0.0], F5, vs, W5, K, log2m=19, seed=77)
        spread.append({"scale": scale, "upper": up, "independent": px, "gap": up - px})

    results = {
        "params": {"r": R, "T": T, "DF": DF, "K": K, "quad_nodes": 4000, "sobol_log2m": 20},
        "n2": {"forwards": F2, "vols": V2, "weights": W2,
               "upper": upper2, "lower": lower2, "width": upper2 - lower2,
               "independent": indep2,
               "indep_retention": (indep2 - lower2) / (upper2 - lower2)},
        "rho_sweep": rho_sweep,
        "strike_sweep": strike_sweep,
        "n5": {"forwards": F5, "vols": V5, "upper": upper5, "rho": rho5,
               "retention_099": (rho5[-1]["price"] - rho5[1]["price"]) / (upper5 - rho5[1]["price"])},
        "vol_spread": spread,
    }

    # fig1: quantile paths under the two extremal couplings.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    q1 = quantile(u, F2[0], V2[0])
    q2 = quantile(u, F2[1], V2[1])
    ax.plot(u, q1, color="#4c78a8", lw=1.6, label="asset 1 quantile")
    ax.plot(u, q2, color="#e45756", lw=1.6, label="asset 2 quantile")
    ax.plot(u, 0.5 * q1 + 0.5 * q2, color="#2ca02c", lw=2.2, label="comonotonic basket")
    ax.plot(u, 0.5 * q1 + 0.5 * q2[::-1], color="#9467bd", lw=2.2, label="countermonotonic basket")
    ax.axhline(K, color="black", ls="--", lw=1, label="strike 100")
    ax.set_xlabel("uniform quantile level u")
    ax.set_ylabel("terminal basket value")
    ax.set_title("Same marginals, opposite couplings")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig1_quantile_paths.png"), dpi=150)
    plt.close(fig)

    # fig2: basket price across Gaussian copula correlation, with exact bounds.
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    xs = [e["rho"] for e in rho_sweep]
    ax.plot(xs, [e["price"] for e in rho_sweep], "o-", color="#4c78a8", label="Gaussian copula (Sobol)")
    ax.axhline(upper2, color="#2ca02c", ls="--", lw=1.8, label="comonotonic upper bound")
    ax.axhline(lower2, color="#9467bd", ls="--", lw=1.8, label="countermonotonic lower bound")
    ax.set_xlabel("Gaussian correlation rho")
    ax.set_ylabel("basket call price")
    ax.set_title("Dependence moves price inside a fixed marginal-implied range")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig2_price_vs_rho.png"), dpi=150)
    plt.close(fig)

    # fig3: upper-bound retention against rank correlation.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot([e["spearman"] for e in rho_sweep], [e["retention"] for e in rho_sweep],
            "o-", color="#e45756")
    ax.axhline(1.0, color="#2ca02c", ls="--", lw=1)
    ax.set_xlabel("Spearman rank correlation")
    ax.set_ylabel("fraction of bound interval")
    ax.set_title("Almost-comonotonic is not comonotonic")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig3_rank_retention.png"), dpi=150)
    plt.close(fig)

    # fig4: strike sweep of the two bounds.
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(k_grid, [e["upper"] for e in strike_sweep], "o-", color="#2ca02c", label="upper")
    ax.plot(k_grid, [e["lower"] for e in strike_sweep], "o-", color="#9467bd", label="lower")
    ax.fill_between(k_grid, [e["lower"] for e in strike_sweep], [e["upper"] for e in strike_sweep],
                    color="#bbbbbb", alpha=0.45, label="unidentified range")
    ax.set_xlabel("strike K")
    ax.set_ylabel("basket call price")
    ax.set_title("Marginals alone leave a strike-dependent price band")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig4_strike_band.png"), dpi=150)
    plt.close(fig)

    # fig5: five-asset basket and vol-spread sensitivity.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    axes[0].plot([e["rho"] for e in rho5], [e["price"] for e in rho5], "o-", color="#4c78a8",
                 label="equicorrelation")
    axes[0].axhline(upper5, color="#2ca02c", ls="--", lw=1.6, label="comonotonic upper")
    axes[0].set_xlabel("equicorrelation rho")
    axes[0].set_ylabel("basket call price")
    axes[0].set_title("Five assets, heterogeneous vols")
    axes[0].legend(fontsize=8)
    axes[1].plot([e["scale"] for e in spread], [e["upper"] for e in spread], "o-",
                 color="#2ca02c", label="upper")
    axes[1].plot([e["scale"] for e in spread], [e["independent"] for e in spread], "o-",
                 color="#4c78a8", label="independent")
    axes[1].set_xlabel("volatility scale")
    axes[1].set_ylabel("basket call price")
    axes[1].set_title("Wider vol dispersion widens the gap")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "fig5_five_assets.png"), dpi=150)
    plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
