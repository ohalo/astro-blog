---
title: "Stein 变分梯度下降 SVGD 贝叶斯组合后验：用一组粒子代替 MCMC 链，为什么带宽决定成败、又在粒子坍缩时失灵"
description: "5 资产、20 期收益的实测：样本均值最高的 A3 只比 A1 高 0.00003，后验里它真正最好的概率却只有 51.67%——plug-in 权重把一次掷硬币当成了确定性结论。120 个 SVGD 粒子把 MMD² 压到 0.000657、协方差比 1.1304；带宽缩到 0.2 倍时粒子过度散开（MMD² 0.03942、协方差比 2.2108），放到 6 倍时同样恶化到 0.05608 与 2.5602。双峰后验里，从单一区域热启动的粒子右峰占比为 0，带宽 2.4 时粒子挤到中间、平均对数密度从 −0.9784 掉到 −1.5784。附完整可运行代码与五张真实图。"
publishDate: '2026-11-28T14:21:00Z'
tags:
  - 量化
  - 贝叶斯
  - 组合优化
  - Python
language: zh
difficulty: advanced
---

## 一、结论先行

均值方差组合的最大问题不是公式，而是把估计出来的期望收益当成真相。20 期样本估出的五个年化收益里，A1 是 0.058037，A3 是 0.058067，后者只比前者高 **0.00003**。 plug-in 优化器据此给出 A3 权重 0.4905、A1 权重 0.3329，看起来言之凿凿。

把 20 期收益、已知协方差和一个宽先验写进贝叶斯模型，后验里 A3 真正是最优资产的概率只有 **51.67%**，A1 是 **48.33%**。这不是「A3 更好」，而是一次几乎公平的掷硬币。SVGD 用 120 个粒子把这个后验直接摆出来：

- 带宽取匹配尺度时，粒子与精确后验的 MMD² 为 **0.000657**，粒子协方差迹与理论值的比是 **1.1304**；
- 带宽缩到 0.2 倍，MMD² 恶化到 **0.03942**、协方差比 **2.2108**；放到 6 倍，MMD² **0.05608**、协方差比 **2.5602**；两端都在丢掉后验形状；
- A3 权重的 95% 区间是 **[0.45873, 0.53322]**，宽 0.07449；plug-in 的 0.49052 落在区间里，但它没有区间这个概念；
- 期望效用：plug-in **0.033614**，SVGD 后验平均 **0.033663**，等权 **0.042836**，真实均值 **0.034413**。贝叶斯平均只改善 0.000049，而等权反而赢了 0.0092。

最后这一条比前面所有数字都重要：在这个样本里，SVGD 卖的不是更高的收益，而是**对权重不确定性的诚实报价**。想靠贝叶斯方法榨出超额收益，通常会输给最简单的一比一平均；想回答「我这个 49% 的仓位有多可信」，plug-in 根本无法回答。

![SVGD 粒子与精确后验的二维投影对比](/images/svgd-bayesian-portfolio-posterior/fig1_posterior_scatter.png)

## 二、为什么用粒子，而不是再跑一条链

给定已知协方差 Σ 和 n 期收益样本 x，均值 μ 的共轭后验是高斯分布：精度等于样本精度加先验精度，均值是两者的精度加权平均。既然有解析后验，为什么还要 SVGD？因为组合权重是 μ 的**非线性函数**：w(μ) = Σ⁻¹μ / (1ᵀΣ⁻¹μ)。知道 μ 的均值和方差，不等于知道 w 的分布；而 w 的分布才是风险管理真正要用的对象。MCMC 能给样本，但高维、非共轭、带约束的扩展模型里要重新设计采样器。SVGD 的思路是把后验压成一团确定性的粒子，用「梯度上升 + 核斥力」把粒子推到高概率区域。

对粒子集合 {zᵢ}，Stein 变分梯度下降的更新方向是：

$$
\phi(z_i)=\frac{1}{m}\sum_{j=1}^{m}\Big[k(z_j,z_i)\,\nabla\log p(z_j)+\nabla_{z_j}k(z_j,z_i)\Big].
$$

第一项把每个粒子往高概率处拉，第二项来自核函数对源粒子的导数，把靠得太近的粒子推开。若用 RBF 核 k = exp(−‖z−z′‖²/2h²)，斥力方向就是 (zᵢ−zⱼ)/h² 乘以核权重。这里 h 是唯一的自由参数，也是这套方法最容易被低估的地方。

为什么不能先算一个「最优均值」再算权重？因为 w(μ) 对 μ 是有理函数，括号里的分母本身随 μ 变化，均值处的函数值不等于函数的均值。偏差的方向由曲率决定，系数越是被样本均值推高的资产，plug-in 越容易放大它的相对优势。贝叶斯的做法是把后验粒子逐个映射成权重，再对权重求平均，这个积分才是风险调整后的配置。

为了不被量纲干扰，实验把 μ 白化成 z = L⁻¹(μ−m)，其中 L 是后验协方差的 Cholesky 因子。目标后验因此是标准正态，得分函数简化为 −z，带宽可以用 h/√(2d) 这个无量纲比例来扫描。粒子初始化在标准差 2 的过散高斯里，优化 320 步、步长 0.12。白化还有一个工程上的好处：不同资产的收益量纲不同，直接在原始 μ 空间用单一核宽，会让波动大的资产主导核距离，带宽扫描的结果失去可比性。

## 三、带宽是一条 U 形曲线

用固定核宽的高斯样本算 MMD²，把 0.2 到 6 倍参考尺度扫一遍，误差与粒子协方差都呈 U 形：

![不同带宽下的 SVGD 收敛轨迹](/images/svgd-bayesian-portfolio-posterior/fig2_convergence.png)

匹配带宽 1.0 倍时，MMD² 在 320 步后降到 0.000657，粒子两两中位距离接近理论值 √(2d)。带宽 0.2 倍时斥力项按 1/h² 放大，粒子互相推得过远，协方差比升到 2.2108；带宽 6 倍时核函数在粒子间几乎恒定， 吸引力项压过斥力，粒子又聚成一团偏离目标尺度，协方差比升到 2.5602。注意这里的「聚成一团」并非严格塌缩到一个点：RBF 核下的 SVGD 对高斯目标本身就有有限粒子偏差，粒子会在一个由带宽决定的尺度上停下，而不是精确复现 N(0,I)。这正是必须实测 MMD 而不能只看收敛曲线是否平稳的原因。

![带宽扫描：粒子离散度与 MMD](/images/svgd-bayesian-portfolio-posterior/fig3_bandwidth.png)

粒子数从 20 增到 240 时，MMD² 在 0.00071 到 0.00090 之间波动，并没有单调改善；协方差比从 0.997 升到 1.205。也就是说，盲目加粒子并不能修复带宽选错带来的偏差，先调 h 再谈粒子数。常见的中位数启发式取初始粒子两两距离的中位数作带宽，在过散初始化下会给出偏大的 h，正好落在图 3 右侧的劣化区；这也是脚本改用相对后验尺度扫描的原因。判断是否收敛不能只看目标密度不再上升，还要看粒子二阶矩与解析后验是否接近，以及 MMD 是否真的降下来。

## 四、组合权重的分布，而不是一个点

把每个粒子映射回 μ，再逐个算均值方差权重，得到的是权重后验。A3 的后验均值权重 0.49105，95% 区间 [0.45873, 0.53322]；A2 的区间是 [−0.27130, −0.10315]，意味着「做空 A2」这件事在模型里相当确定；A5 的区间则跨过零，从 −0.04749 到 0.10441，方向本身就不明确。

![后验平均权重与 95% 区间](/images/svgd-bayesian-portfolio-posterior/fig4_weights.png)

plug-in 权重全部落在这个区间里，因此说它「错」并不公平。它真正丢掉的是区间所携带的决策信息：A3 与 A1 的样本均值差只有 0.00003，后验概率 51.67% 对 48.33%，任何基于「谁排第一」的离散调仓规则在这里都近似抛硬币。更诚实的做法是把仓位按后验概率平滑，或者直接承认等权基准难以击败。期望效用 0.033663 对 0.042836 的差距提醒我们：贝叶斯后验是风险语言，不是收益神话。

## 五、双峰后验里的模式坍缩

单峰高斯后验只暴露带宽问题，真正的失效发生在多峰。构造一个两成分后验，两个峰位于 z₁ = −1.6 和 +1.6、标准差 0.32，代表「牛市均值」与「熊市均值」两种无法用样本区分的解释。120 个粒子从对称位置出发、带宽 0.45 时，右峰占比 **0.5333**，平均对数密度 **−0.9784**，两个峰都被覆盖。

![双峰后验中的模式坍缩](/images/svgd-bayesian-portfolio-posterior/fig5_mode_collapse.png)

把初始化改成从左侧峰附近开始，同样 120 个粒子、同样带宽，右峰占比变成 **0**，平均对数密度 −0.9873；粒子数降到 18 时右峰占比也是 0。粒子没有足够的斥力跨过两峰之间的低密度区，于是整体留在初始峰里，后验的另一半解释凭空消失。把带宽放大到 2.4 更糟：右峰占比 0.175，粒子被拉到两峰之间的低密度地带，平均对数密度跌到 **−1.5784**，比坍缩还差。实践中若怀疑市场存在多个机制，应当多起点初始化并检查每个峰的粒子占比，而不是盯着一条看起来已经平稳的收敛曲线。

## 六、完整可运行代码

下面脚本固定种子一次生成全部数值与五张图，可独立运行，结果写入 results.json。

```python
"""SVGD posterior particles for Bayesian mean-variance weights; deterministic simulation."""

def main():
    import json
    import os

    import numpy as np

    SEED = 20261129
    OUT = "/Users/halo/workspace/astro-blog/public/images/svgd-bayesian-portfolio-posterior"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ---- data generating process (annualized) ----
    d = 5
    B = np.array([
        [0.12, 0.00],
        [0.10, 0.05],
        [-0.06, 0.09],
        [0.02, -0.11],
        [0.04, 0.03],
    ])
    Sigma = B @ B.T + np.diag([0.012 ** 2, 0.011 ** 2, 0.013 ** 2, 0.010 ** 2, 0.014 ** 2])
    invS = np.linalg.inv(Sigma)
    cholS = np.linalg.cholesky(Sigma)
    mu_true = np.array([0.10, 0.07, 0.03, -0.01, 0.04])
    n_obs = 20
    X = mu_true + rng.normal(size=(n_obs, d)) @ cholS.T
    xbar = X.mean(axis=0)
    tau = 0.15
    prec = n_obs * invS + np.eye(d) / tau ** 2
    V = np.linalg.inv(prec)
    m = V @ (n_obs * invS @ xbar)
    L = np.linalg.cholesky(V)

    # whiten: z = L^{-1}(mu - m) ~ N(0, I) under exact posterior
    def to_mu(z):
        return m + z @ L.T


    def svgd(z0, h, iters=320, step=0.12, trace_every=20):
        z = z0.copy()
        n = len(z)
        trace = []
        for it in range(iters + 1):
            if it % trace_every == 0:
                trace.append((it, mmd_to_gaussian(z)))
            diff = z[:, None, :] - z[None, :, :]
            sq = np.einsum("ijn,ijn->ij", diff, diff)
            k = np.exp(-sq / (2.0 * h * h))
            rep = np.einsum("ijn,ij->in", diff, k) / (n * h * h)
            score = -z
            phi = (k @ score) / n + rep
            z += step * phi
        return z, trace


    def mmd_to_gaussian(z, nsamp=800):
        # fixed kernel bandwidth = median distance of N(0,I) samples in d dimensions
        y = rng.normal(size=(nsamp, d))
        x = z
        xy = np.vstack([x, y])
        sq = np.sum((xy[:, None, :] - xy[None, :, :]) ** 2, axis=2)
        h = np.sqrt(2.0 * d)
        k = np.exp(-sq / (2.0 * h * h))
        nx = len(x)
        return float(k[:nx, :nx].mean() + k[nx:, nx:].mean() - 2.0 * k[:nx, nx:].mean())


    def cov_ratio(z):
        return float(np.trace(np.cov(z, rowvar=False)) / d)


    def utility(w):
        return float(w @ mu_true - 1.5 * w @ Sigma @ w)


    def weights(mu):
        raw = np.asarray(mu) @ invS
        return raw / raw.sum(axis=-1, keepdims=True)


    def plan(n_particles, h_rel, iters=320):
        z0 = rng.normal(size=(n_particles, d)) * 2.0
        h = h_rel * np.sqrt(2.0 * d)
        z, trace = svgd(z0, h, iters=iters)
        w = weights(to_mu(z)).mean(axis=0)
        return z, trace, w


    h_ref = np.sqrt(2.0 * d)
    z0 = rng.normal(size=(120, d)) * 2.0
    h_grid = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    results = {
        "seed": SEED,
        "n_obs": n_obs,
        "prior_tau": tau,
        "sample_mean": xbar.tolist(),
        "posterior_mean": m.tolist(),
        "posterior_sd": np.sqrt(np.diag(V)).tolist(),
        "true_mu": mu_true.tolist(),
        "h_grid": h_grid,
        "bandwidth": {},
        "convergence": {},
        "weights": {},
        "utility": {},
        "particles": {},
    }

    # convergence under three bandwidths
    for label, hr in [("small", 0.25), ("matched", 1.0), ("large", 4.0)]:
        z, trace, w = plan(120, hr)
        results["convergence"][label] = {"h_rel": hr, "mmd_trace": [[int(i), round(v, 8)] for i, v in trace]}

    # bandwidth sweep
    for hr in h_grid:
        z, trace, w = plan(120, hr)
        vals = {
            "h_rel": hr,
            "cov_ratio": round(cov_ratio(z), 5),
            "median_pair_dist": round(float(np.median(np.sqrt(np.sum((z[:, None] - z[None, :]) ** 2, axis=2)))), 5),
            "mmd2": round(mmd_to_gaussian(z), 8),
            "utility": round(utility(w), 6),
            "max_abs_weight": round(float(np.max(np.abs(w))), 4),
        }
        results["bandwidth"][str(hr)] = vals
        z0 = rng.normal(size=(120, d)) * 2.0  # independent init per bandwidth

    # reference portfolios
    w_plugin = weights(xbar)
    w_true = weights(mu_true)
    w_equal = np.ones(d) / d
    z_best, _, w_svgd = plan(120, 1.0)
    w_all = weights(to_mu(z_best))
    mu_all = to_mu(z_best)
    w_lo = np.quantile(w_all, 0.025, axis=0)
    w_hi = np.quantile(w_all, 0.975, axis=0)
    prob_top = np.mean(mu_all == mu_all.max(axis=1, keepdims=True), axis=0)
    results["uncertainty"] = {
        "w_2.5pct": w_lo.round(5).tolist(),
        "w_97.5pct": w_hi.round(5).tolist(),
        "prob_best_asset": prob_top.round(5).tolist(),
        "plugin_top_asset": int(np.argmax(xbar)),
        "plugin_top_posterior_prob": round(float(prob_top[np.argmax(xbar)]), 5),
        "w3_interval_width": round(float(w_hi[2] - w_lo[2]), 5),
    }
    z_coll, _, w_coll = plan(120, 6.0)
    results["weights"] = {
        "equal": w_equal.round(5).tolist(),
        "plugin": w_plugin.round(5).tolist(),
        "svgd": w_svgd.round(5).tolist(),
        "true": w_true.round(5).tolist(),
        "collapsed": w_coll.round(5).tolist(),
    }
    results["utility"] = {
        "equal": round(utility(w_equal), 6),
        "plugin": round(utility(w_plugin), 6),
        "svgd": round(utility(w_svgd), 6),
        "collapsed": round(utility(w_coll), 6),
        "true": round(utility(w_true), 6),
        "plugin_max_abs_weight": round(float(np.max(np.abs(w_plugin))), 4),
        "svgd_max_abs_weight": round(float(np.max(np.abs(w_svgd))), 4),
    }

    # particle count sensitivity at matched bandwidth
    for n_p in [20, 60, 120, 240]:
        z, _, w = plan(n_p, 1.0)
        results["particles"][str(n_p)] = {
            "mmd2": round(mmd_to_gaussian(z), 8),
            "cov_ratio": round(cov_ratio(z), 5),
            "utility": round(utility(w), 6),
        }

    # ---- bimodal target: regime ambiguity -> mode collapse ----
    cent = np.array([[-1.6, 0.0], [1.6, 0.0]])
    s2 = 0.32 ** 2


    def logp_bimodal(z):
        dev = z[:, None, :] - cent[None, :, :]
        ll = -0.5 * np.sum(dev ** 2, axis=2) / s2
        mx = ll.max(axis=1, keepdims=True)
        w = np.exp(ll - mx)
        return np.log(w.sum(axis=1)) + mx[:, 0]


    def score_bimodal(z):
        dev = z[:, None, :] - cent[None, :, :]
        ll = -0.5 * np.sum(dev ** 2, axis=2) / s2
        mx = ll.max(axis=1, keepdims=True)
        w = np.exp(ll - mx)
        w /= w.sum(axis=1, keepdims=True)
        return -np.einsum("ij,ijk->ik", w, dev) / s2


    def svgd_bimodal(z0, h, iters=600, step=0.06):
        z = z0.copy()
        n = len(z)
        for _ in range(iters):
            diff = z[:, None, :] - z[None, :, :]
            sq = np.einsum("ijn,ijn->ij", diff, diff)
            k = np.exp(-sq / (2.0 * h * h))
            rep = np.einsum("ijn,ij->in", diff, k) / (n * h * h)
            z += step * ((k @ score_bimodal(z)) / n + rep)
        return z


    def mode_stats(z):
        right = float(np.mean(z[:, 0] > 0))
        dp = z[:, None, :] - cent[None, :, :]
        ll = -0.5 * np.sum(dp ** 2, axis=2) / s2
        mx = ll.max(axis=1, keepdims=True)
        lse = np.log(np.exp(ll - mx).sum(axis=1)) + mx[:, 0]
        return round(right, 4), round(float(np.mean(lse)), 4)


    rng2 = np.random.default_rng(SEED + 7)
    bimodal = {}
    cases = {
        "balanced_small_h": (120, 0.45, np.array([0.0, 0.0]), 1.6),
        "few_particles": (18, 0.45, np.array([-1.6, 0.0]), 0.4),
        "warm_start_one_mode": (120, 0.45, np.array([-1.6, 0.0]), 0.4),
        "large_h_merge": (120, 2.4, np.array([0.0, 0.0]), 1.6),
    }
    for name, (n_p, h, center, scale) in cases.items():
        zz = center + rng2.normal(size=(n_p, 2)) * scale
        zz = svgd_bimodal(zz, h)
        right, mean_ll = mode_stats(zz)
        bimodal[name] = {
            "n_particles": n_p, "h": h, "right_mode_share": right,
            "mean_log_density": mean_ll,
            "z_samples": np.round(zz[::max(1, len(zz) // 60)][:60], 4).tolist(),
        }
    results["bimodal"] = {k: {kk: vv for kk, vv in v.items() if kk != "z_samples"} for k, v in bimodal.items()}

    # ---- figures ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.3})

    # fig1 posterior scatter
    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    y = rng.normal(size=(1200, 2))
    ax.scatter(y[:, 0], y[:, 1], s=8, alpha=0.35, label="Exact posterior N(0,I)", color="#4c78a8")
    zb, _, _ = plan(120, 1.0)
    ax.scatter(zb[:, 0], zb[:, 1], s=18, alpha=0.85, label="SVGD particles (h matched)", color="#e45756", marker="o")
    ax.scatter([0], [0], marker="x", s=90, color="black", label="Posterior mean")
    th = np.linspace(0, 2 * np.pi, 200)
    ax.plot(np.cos(th), np.sin(th), "k--", lw=1.2, label="1 std ellipse")
    ax.set_xlabel("whitized mu1"); ax.set_ylabel("whitized mu2")
    ax.set_xlim(-4, 4); ax.set_ylim(-4, 4); ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("SVGD particles vs exact posterior (2D projection)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_posterior_scatter.png"), dpi=150); plt.close(fig)

    # fig2 convergence
    fig, ax = plt.subplots(figsize=(6.6, 4.5))
    colors = {"small": "#e45756", "matched": "#2ca02c", "large": "#4c78a8"}
    for label, rec in results["convergence"].items():
        its = [p[0] for p in rec["mmd_trace"]]
        vv = [max(p[1], 1e-6) for p in rec["mmd_trace"]]
        ax.semilogy(its, vv, lw=1.8, color=colors[label], label=f"{label} h_rel={rec['h_rel']}")
    ax.set_xlabel("SVGD iteration"); ax.set_ylabel("MMD^2 to exact posterior")
    ax.set_title("Convergence: bandwidth controls the fixed point")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_convergence.png"), dpi=150); plt.close(fig)

    # fig3 bandwidth sweep
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2))
    cr = [results["bandwidth"][str(h)]["cov_ratio"] for h in h_grid]
    md = [results["bandwidth"][str(h)]["median_pair_dist"] for h in h_grid]
    axes[0].plot(h_grid, cr, "o-", color="#4c78a8"); axes[0].axhline(1.0, ls="--", color="gray")
    axes[0].set_xscale("log"); axes[0].set_xlabel("h / sqrt(2d)"); axes[0].set_ylabel("trace cov / d")
    axes[0].set_title("Posterior dispersion of particles")
    axes[1].plot(h_grid, md, "o-", color="#e45756"); axes[1].axhline(np.sqrt(2*d), ls="--", color="gray")
    axes[1].set_xscale("log"); axes[1].set_xlabel("h / sqrt(2d)"); axes[1].set_ylabel("median pair distance")
    axes[1].set_title("Particle spread")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig3_bandwidth.png"), dpi=150); plt.close(fig)

    # fig4 weights
    fig, ax = plt.subplots(figsize=(8.0, 4.6))
    labels = [f"A{i+1}" for i in range(d)]
    xp = np.arange(d); width = 0.18
    for j, (name, color) in enumerate([("equal", "#999999"), ("plugin", "#e45756"), ("svgd", "#2ca02c"), ("true", "#4c78a8")]):
        ax.bar(xp + (j - 1.5) * width, results["weights"][name], width, label=name, color=color)
    lo = np.array(results["uncertainty"]["w_2.5pct"])
    hi = np.array(results["uncertainty"]["w_97.5pct"])
    ax.errorbar(xp + 0.5 * width, results["weights"]["svgd"],
                yerr=[np.maximum(results["weights"]["svgd"] - lo, 0), np.maximum(hi - results["weights"]["svgd"], 0)],
                fmt="none", ecolor="#2ca02c", capsize=3, lw=1.2, label="SVGD 95% interval")
    ax.set_xticks(xp); ax.set_xticklabels(labels); ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("portfolio weight"); ax.set_title("Posterior weights: mean and 95% interval")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig4_weights.png"), dpi=150); plt.close(fig)

    # fig5 mode collapse in bimodal posterior
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5))
    for ax, name in zip(axes, ["balanced_small_h", "warm_start_one_mode"]):
        zz = np.array(bimodal[name]["z_samples"])
        ax.scatter(zz[:, 0], zz[:, 1], s=16, alpha=0.8, color="#4c78a8")
        ax.scatter(cent[:, 0], cent[:, 1], marker="x", s=120, color="black")
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-2.0, 2.0)
        ax.set_xlabel("z1"); ax.set_ylabel("z2")
        ax.set_title(f"{name}: right share {bimodal[name]['right_mode_share']:.3f}")
    axes[0].set_title("Both regimes covered (120 particles)")
    axes[1].set_title("Mode collapse (warm-started in one regime)")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig5_mode_collapse.png"), dpi=150); plt.close(fig)

    with open(os.path.join(OUT, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    main()
```

## 七、边界与用法

SVGD 的粒子不是独立同分布样本，不能直接拿来做标准误估计；核宽固定时，它收敛到的是带核偏差的分布而非精确后验。高维下核矩阵是 m×m，粒子数上千之后成本迅速上升，需要用随机特征或分块近似。对量化实践最稳妥的用法，是把它当成一个比 MCMC 更容易嵌入优化循环的**不确定性可视化工具**：先看粒子是否覆盖了已知的解释，再谈要不要用它来加权仓位。当后验分布只有一个峰、且期望效用打不过等权时，诚实的结论就是老老实实用等权。

与常见的拉普拉斯近似相比，SVGD 不要求把后验压成一个高斯，原则上能保留非对称和多个峰；代价是结果依赖初始化、核宽和迭代步数，三个旋钮都要检查。落地时建议按固定顺序做三件事：先用解析后验或少量 MCMC 样本对拍一阶二阶矩，再扫描核宽看 MMD 是否出现 U 形最小值，最后用多起点初始化检查有没有丢失的峰。三步都过了，粒子给出的权重区间才值得拿去做仓位上限约束。
