"""秩相关在厚尾与污染数据下的稳健性：完整可运行实验。

运行方式（在项目根目录）：
    python3 scripts/article_core_kendall_20261108.py

产出：
    public/images/kendall-tau-robust-covariance/fig1..fig5.png
    public/images/kendall-tau-robust-covariance/results.json

对照四种相关矩阵估计量：Pearson、Winsorize+Pearson、Spearman 变换、
Kendall 变换。所有数字均由本脚本现场计算，无外部数据。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SEED = 20261108
OUT = "public/images/kendall-tau-robust-covariance"
SIGMA = 0.015          # 日波动率
RHO = 0.35             # 真实等相关系数
CONTAM_LEVELS = (0.0, 0.01, 0.02, 0.05, 0.10)
METHODS = ("Pearson", "Winsorized", "Spearman", "Kendall")


def true_cov(p, rho=RHO, sigma=SIGMA):
    c = np.full((p, p), rho)
    np.fill_diagonal(c, 1.0)
    return (sigma ** 2) * c


def hetero_corr(rng, p, rank=3):
    """异质相关矩阵：低秩因子 + 个体差异，逐元素非线性变换后才可能破坏正定。"""
    B = rng.normal(size=(p, rank)) * 0.6
    C = B @ B.T + np.diag(rng.uniform(0.5, 1.5, size=p) ** 2) + np.eye(p) * 0.3
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def gen_data(rng, n, p, dgp, eps=0.0, rho=RHO, sigma=SIGMA, corr=None):
    """dgp: 'gaussian' | 't3' | 'contaminated'；返回 (n,p) 观测矩阵。

    给定 corr 时使用任意相关结构，否则使用等相关 rho。
    """
    if corr is None:
        c = np.full((p, p), rho)
        np.fill_diagonal(c, 1.0)
    else:
        c = corr
    L = np.linalg.cholesky(c)
    G = rng.standard_normal((n, p)) @ L.T
    if dgp == "t3":
        w = rng.chisquare(3.0, size=n) / 3.0
        X = G / np.sqrt(w)[:, None]
    else:
        X = G.copy()
    if dgp == "contaminated" and eps > 0:
        n_bad = max(1, int(round(n * eps)))
        rows = rng.choice(n, size=n_bad, replace=False)
        for r in rows:
            cols = rng.choice(p, size=2, replace=False)
            X[r, cols] += rng.choice([-1.0, 1.0], size=2) * 8.0
    return X * sigma


def kendall_from_ranks(R):
    """一次性算出所有配对的 Kendall tau（O(n^2 p^2) 的 einsum 实现）。

    D[a,b,i] = sign(rank[b,i] - rank[a,i])，
    sum_{a,b} D[a,b,i]*D[a,b,j] = 2*(concordant - discordant)，
    除以 n(n-1) 即得 tau。
    """
    n, p = R.shape
    D = np.sign(R[None, :, :] - R[:, None, :])
    return np.einsum("abi,abj->ij", D, D, optimize=True) / (n * (n - 1))


def estimates(X):
    """返回四种估计量的相关矩阵字典。"""
    n, p = X.shape
    R = np.argsort(np.argsort(X, axis=0), axis=0).astype(float)
    pear = np.corrcoef(X, rowvar=False)
    Z = np.column_stack([np.clip(X[:, j],
                                 np.quantile(X[:, j], 0.01),
                                 np.quantile(X[:, j], 0.99))
                         for j in range(p)])
    wins = np.corrcoef(Z, rowvar=False)
    spear = np.corrcoef(R, rowvar=False)
    kend = kendall_from_ranks(R)
    return {
        "Pearson": pear,
        "Winsorized": wins,
        "Spearman": 2.0 * np.sin(np.pi * spear / 6.0),
        "Kendall": np.sin(np.pi * kend / 2.0),
    }


def rel_frobenius(chat, ctrue):
    mask = ~np.eye(chat.shape[0], dtype=bool)
    return float(np.linalg.norm((chat - ctrue)[mask]) /
                 np.linalg.norm(ctrue[mask]))


def bias(chat, rho=RHO):
    p = chat.shape[0]
    off = chat[~np.eye(p, dtype=bool)]
    return float(off.mean() - rho)


def make_psd(A, floor=1e-8):
    ev, V = np.linalg.eigh(A)
    return (V * np.clip(ev, floor, None)) @ V.T, int((ev < 0).any())


def gmv_oos_vol(chat, ctrue, annual=252):
    """最小方差组合在真实协方差下的年化波动；非正定时先做 PSD 投影。"""
    A, bad = make_psd(chat)
    ones = np.ones(A.shape[0])
    try:
        w = np.linalg.solve(A, ones)
    except np.linalg.LinAlgError:
        w = np.linalg.lstsq(A, ones, rcond=None)[0]
    w = w / w.sum()
    var = float(w @ ctrue @ w)
    return float(np.sqrt(annual * max(var, 0.0))), bad


def oracle_vol(ctrue, annual=252):
    p = ctrue.shape[0]
    w = np.ones(p) / p  # 等相关矩阵的最优 GMV 权重就是等权
    return float(np.sqrt(annual * float(w @ ctrue @ w)))


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=110, bbox_inches="tight")
    plt.close(fig)


def run():
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n, p = 200, 10
    C = true_cov(p) / (SIGMA ** 2)  # 相关系数矩阵，作为估计目标
    COV = true_cov(p)
    R = 120

    # 图 1：三种 DGP 的散点示例
    samples = {
        "Gaussian": gen_data(rng, 400, 2, "gaussian", rho=RHO),
        "Student-t(3)": gen_data(rng, 400, 2, "t3", rho=RHO),
        "Contaminated (5%)": gen_data(rng, 400, 2, "contaminated", 0.05, RHO),
    }
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.7))
    for ax, (name, X) in zip(axes, samples.items()):
        ax.scatter(X[:, 0], X[:, 1], s=8, alpha=0.55, color="#2f5597")
        ax.set_title(name)
        ax.set_xlabel("asset 1")
        lim = 0.12 if "Contaminated" in name else (0.09 if "t(3)" in name else 0.06)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
    axes[0].set_ylabel("asset 2")
    fig.suptitle("Same true correlation rho=0.35, three noise models")
    fig.tight_layout()
    save(fig, "fig1_scatter_dgps.png")

    # 主实验：污染比例扫描
    scan = []
    for eps in CONTAM_LEVELS:
        acc = {m: {"fro": [], "bias": [], "vol": [], "psd_bad": 0}
               for m in METHODS}
        for _ in range(R):
            X = gen_data(rng, n, p, "contaminated" if eps > 0 else "gaussian",
                         eps=eps)
            for m, chat in estimates(X).items():
                acc[m]["fro"].append(rel_frobenius(chat, C))
                acc[m]["bias"].append(bias(chat))
                v, bad = gmv_oos_vol(chat, COV)
                acc[m]["vol"].append(v)
                acc[m]["psd_bad"] += bad
        row = {"eps": eps}
        for m in METHODS:
            row[m] = {
                "rel_frobenius": float(np.mean(acc[m]["fro"])),
                "bias": float(np.mean(acc[m]["bias"])),
                "gmv_oos_vol": float(np.mean(acc[m]["vol"])),
                "psd_fail_rate": acc[m]["psd_bad"] / R,
            }
        scan.append(row)

    oracle = oracle_vol(COV)

    # 图 2：相对 Frobenius 误差 vs 污染比例
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    colors = {"Pearson": "#c0392b", "Winsorized": "#e67e22",
              "Spearman": "#2f5597", "Kendall": "#1e8449"}
    for m in METHODS:
        ax.plot(CONTAM_LEVELS, [r[m]["rel_frobenius"] for r in scan],
                "o-", color=colors[m], label=m)
    ax.set_xlabel("contamination fraction epsilon")
    ax.set_ylabel("relative Frobenius error of correlation matrix")
    ax.set_title(f"n={n}, p={p}, {R} replications")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig2_frobenius_vs_contamination.png")

    # 图 4：GMV 样本外年化波动 vs 污染比例
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    for m in METHODS:
        ax.plot(CONTAM_LEVELS, [r[m]["gmv_oos_vol"] for r in scan],
                "o-", color=colors[m], label=m)
    ax.axhline(oracle, ls="--", color="#555555", label="oracle (equal weight)")
    ax.set_xlabel("contamination fraction epsilon")
    ax.set_ylabel("out-of-sample annualized volatility")
    ax.set_title("Global minimum variance portfolio under corrupted inputs")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig4_gmv_oos_volatility.png")

    # 无偏性验证：sin(pi*tau/2) 在椭圆分布下是否还原 Pearson rho
    rhos = (0.0, 0.2, 0.4, 0.6, 0.8)
    unbiased = []
    for dgp in ("gaussian", "t3"):
        for rho in rhos:
            vals = {m: [] for m in METHODS}
            for _ in range(60):
                X = gen_data(rng, 250, p, dgp, rho=rho)
                for m, chat in estimates(X).items():
                    vals[m].append(bias(chat, rho) + rho)
            unbiased.append({"dgp": dgp, "rho": rho,
                             **{m: float(np.mean(vals[m])) for m in METHODS}})

    # 图 3：估计均值 vs 真实 rho
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.3), sharey=True)
    for ax, dgp in zip(axes, ("gaussian", "t3")):
        sub = [u for u in unbiased if u["dgp"] == dgp]
        for m in ("Pearson", "Spearman", "Kendall"):
            ax.plot(rhos, [u[m] for u in sub], "o-", color=colors[m], label=m)
        ax.plot(rhos, rhos, ls="--", color="#666666", label="identity")
        ax.set_title("Gaussian" if dgp == "gaussian" else "Student-t(3)")
        ax.set_xlabel("true rho")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("estimated rho")
    axes[0].legend(fontsize=8)
    fig.suptitle("Does sin(pi*tau/2) recover the Pearson correlation?")
    fig.tight_layout()
    save(fig, "fig3_sin_transform_unbiasedness.png")

    # p/n 扫描：误差与正定性失败率
    pn = []
    for pp in (6, 10, 20):
        Cp = true_cov(pp) / (SIGMA ** 2)
        COVp = true_cov(pp)
        for nn in (60, 120, 250, 500):
            fp, fk, bad_k = [], [], 0
            for _ in range(40):
                X = gen_data(rng, nn, pp, "contaminated", 0.05)
                e = estimates(X)
                fp.append(rel_frobenius(e["Pearson"], Cp))
                fk.append(rel_frobenius(e["Kendall"], Cp))
                _, b = gmv_oos_vol(e["Kendall"], COVp)
                bad_k += b
            pn.append({"p": pp, "n": nn,
                       "pearson_fro": float(np.mean(fp)),
                       "kendall_fro": float(np.mean(fk)),
                       "kendall_psd_fail_rate": bad_k / 40})

    # 正定性压力测试：等相关矩阵被逐元素变换后仍是等相关，结构上必然 PSD；
    # 只有异质相关结构才能检验「逐元素 sin 变换破坏正定性」这条边界。
    psd_stress = []
    p_stress = 40
    for nn in (30, 60, 120, 250):
        bad = {m: 0 for m in METHODS}
        mineigs = []
        for _ in range(40):
            Ch = hetero_corr(rng, p_stress)
            X = gen_data(rng, nn, p_stress, "contaminated", 0.05, corr=Ch)
            e = estimates(X)
            for m, chat in e.items():
                if np.linalg.eigvalsh(chat)[0] < 0:
                    bad[m] += 1
            mineigs.append(np.linalg.eigvalsh(e["Kendall"])[0])
        psd_stress.append({
            "p": p_stress, "n": nn,
            **{m: bad[m] / 40 for m in METHODS},
            "kendall_mean_min_eig": float(np.mean(mineigs)),
        })

    # 图 5：p/n 扫描 + 正定性压力测试
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3))
    for pp in (6, 10, 20):
        sub = [r for r in pn if r["p"] == pp]
        xs = [r["n"] for r in sub]
        axes[0].plot(xs, [r["pearson_fro"] for r in sub], "s--",
                     color=colors["Pearson"], alpha=0.8, label=f"Pearson p={pp}")
        axes[0].plot(xs, [r["kendall_fro"] for r in sub], "o-",
                     color=colors["Kendall"], alpha=0.8, label=f"Kendall p={pp}")
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sample size n")
    axes[0].set_ylabel("relative Frobenius error")
    axes[0].set_title("Error vs sample size (5% contamination)")
    axes[0].legend(fontsize=7, ncol=2)
    for m in ("Pearson", "Spearman", "Kendall"):
        axes[1].plot([r["n"] for r in psd_stress],
                     [r[m] for r in psd_stress], "o-",
                     color=colors[m], label=m)
    axes[1].set_xscale("log")
    axes[1].set_ylim(-0.03, 1.03)
    axes[1].set_xlabel("sample size n")
    axes[1].set_ylabel("non-PSD rate")
    axes[1].set_title(f"Heterogeneous corr (p={p_stress}): PSD failure")
    axes[1].legend(fontsize=8)
    for a in axes:
        a.grid(alpha=0.25)
    fig.tight_layout()
    save(fig, "fig5_pn_scan_and_psd.png")

    # DGP 对照（无污染 vs 厚尾 vs 污染）
    dgp_cmp = []
    for dgp, eps in (("gaussian", 0.0), ("t3", 0.0), ("contaminated", 0.05)):
        vals = {m: [] for m in METHODS}
        for _ in range(R):
            X = gen_data(rng, n, p, dgp, eps)
            for m, chat in estimates(X).items():
                vals[m].append(rel_frobenius(chat, C))
        dgp_cmp.append({"dgp": dgp, "eps": eps,
                        **{m: float(np.mean(vals[m])) for m in METHODS}})

    results = {
        "seed": SEED,
        "n": n, "p": p, "rho": RHO, "sigma": SIGMA,
        "replications_main": R,
        "contamination_scan": scan,
        "dgp_comparison": dgp_cmp,
        "unbiasedness": unbiased,
        "pn_scan": pn,
        "psd_stress": psd_stress,
        "oracle_gmv_annual_vol": oracle,
    }
    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"oracle GMV annual vol = {oracle:.6f}")
    print("=== contamination scan (relative Frobenius error) ===")
    for r in scan:
        print(f"eps={r['eps']:.2f}  " + "  ".join(
            f"{m}={r[m]['rel_frobenius']:.4f}" for m in METHODS))
    print("=== GMV out-of-sample annual vol @ eps=0.05 ===")
    row = [r for r in scan if r["eps"] == 0.05][0]
    for m in METHODS:
        print(f"  {m:11s} vol={row[m]['gmv_oos_vol']:.6f}  "
              f"psd_fail={row[m]['psd_fail_rate']:.3f}")
    print("=== DGP comparison (relative Frobenius error) ===")
    for r in dgp_cmp:
        print(f"{r['dgp']:13s} " + "  ".join(
            f"{m}={r[m]:.4f}" for m in METHODS))
    print("=== unbiasedness (Kendall sin transform, Gaussian) ===")
    for u in [u for u in unbiased if u["dgp"] == "gaussian"]:
        print(f"  rho={u['rho']:.1f}  Pearson={u['Pearson']:.4f}  "
              f"Spearman={u['Spearman']:.4f}  Kendall={u['Kendall']:.4f}")
    print("=== PSD stress (heterogeneous corr, p=40) ===")
    for r in psd_stress:
        print(f"  n={r['n']:3d}  " + "  ".join(
            f"{m}={r[m]:.3f}" for m in METHODS) +
            f"  min_eig={r['kendall_mean_min_eig']:.4f}")
    return results


if __name__ == "__main__":
    run()
