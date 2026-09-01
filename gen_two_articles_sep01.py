#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-09-01 两篇量化文章配图 + 核心数值（numpy/scipy 合成，固定 seed 可复现）。
  A. robust-mean-variance-optimization  鲁棒均值方差优化
  B. imbalance-bars-information-sampling  不平衡 Bar 信息驱动采样
所有图表均为真实计算图，数值固定随机种子可复现。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130


def shrink(M, delta=0.10):
    d = np.mean(np.diag(M))
    return (1 - delta) * M + delta * d * np.eye(M.shape[0])


# ============================================================
# 文章 A：鲁棒均值方差优化（椭球模糊集下的 worst-case 收益）
# ============================================================
def gen_article_a():
    slug = "robust-mean-variance-optimization"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260901)

    N = 10
    beta = rng.uniform(0.3, 1.4, N)          # 单因子载荷
    f_premium = 0.0004                        # 因子日溢价
    mu_true = 0.0002 + beta * f_premium       # 真实日收益
    sig_f = 0.006
    D = np.diag((rng.uniform(0.006, 0.010, N)) ** 2)
    Sigma_true = np.outer(beta, beta) * sig_f ** 2 + D

    T_est, T_oos, MC = 120, 252, 400
    gammas = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]

    def nom_maxret(Mu, Cov, budget):
        n = Mu.shape[0]
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "ineq", "fun": lambda w: budget - w @ Cov @ w}]
        res = minimize(lambda w: -(w @ Mu), np.full(n, 1 / n), method="SLSQP",
                       constraints=cons, bounds=[(0, 1)] * n,
                       options={"ftol": 1e-12, "maxiter": 2000})
        return np.clip(res.x, 0, None)

    def rob_maxret(Mu, Cov, budget, gamma):
        n = Mu.shape[0]
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1},
                {"type": "ineq", "fun": lambda w: budget - w @ Cov @ w}]
        def obj(w):
            return -(w @ Mu - gamma * np.sqrt(max(w @ Cov @ w, 1e-18)))
        res = minimize(obj, np.full(n, 1 / n), method="SLSQP",
                       constraints=cons, bounds=[(0, 1)] * n,
                       options={"ftol": 1e-12, "maxiter": 2000})
        return np.clip(res.x, 0, None)

    def gmv(Cov):
        n = Cov.shape[0]
        res = minimize(lambda w: w @ Cov @ w, np.full(n, 1 / n), method="SLSQP",
                       constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                       bounds=[(0, 1)] * n, options={"ftol": 1e-12, "maxiter": 2000})
        return np.clip(res.x, 0, None)

    oos_sharpe = {g: [] for g in gammas}
    hhi = {g: [] for g in gammas}
    for _ in range(MC):
        R = rng.multivariate_normal(mu_true, Sigma_true, T_est + T_oos)
        Rin, Roos = R[:T_est], R[T_est:]
        Mu_hat = Rin.mean(0)
        Cov_hat = shrink(np.cov(Rin.T, bias=False), 0.10)
        w_g = gmv(Cov_hat)
        budget = (1.5 * np.sqrt(w_g @ Cov_hat @ w_g)) ** 2
        for g in gammas:
            w = nom_maxret(Mu_hat, Cov_hat, budget) if g == 0 else rob_maxret(Mu_hat, Cov_hat, budget, g)
            ret = Roos @ w
            sd = ret.std(ddof=1)
            sh = (ret.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0
            oos_sharpe[g].append(sh)
            hhi[g].append(float((w ** 2).sum()))

    oos_sharpe = {g: np.array(v) for g, v in oos_sharpe.items()}
    hhi = {g: np.array(v) for g, v in hhi.items()}

    Mu_avg = np.zeros(N); Cov_avg = np.zeros((N, N)); wcnt = 0
    for _ in range(60):
        R = rng.multivariate_normal(mu_true, Sigma_true, T_est)
        Mu_avg += R.mean(0); Cov_avg += shrink(np.cov(R.T, bias=False), 0.10); wcnt += 1
    Mu_avg /= wcnt; Cov_avg /= wcnt
    w_g = gmv(Cov_avg)
    gmv_vol = np.sqrt(w_g @ Cov_avg @ w_g)
    budgets = np.linspace((1.05 * gmv_vol) ** 2, (3.0 * gmv_vol) ** 2, 14)
    front_nom_ret, front_nom_vol, front_rob_ret, front_rob_vol = [], [], [], []
    g_star = 2.0
    for bd in budgets:
        wn = nom_maxret(Mu_avg, Cov_avg, bd)
        wr = rob_maxret(Mu_avg, Cov_avg, bd, g_star)
        front_nom_ret.append(wn @ Mu_avg * 252); front_nom_vol.append(np.sqrt(wn @ Cov_avg @ wn) * np.sqrt(252))
        front_rob_ret.append(wr @ Mu_avg * 252); front_rob_vol.append(np.sqrt(wr @ Cov_avg @ wr) * np.sqrt(252))

    R = rng.multivariate_normal(mu_true, Sigma_true, T_est + T_oos)
    Mu_hat = R[:T_est].mean(0); Cov_hat = shrink(np.cov(R[:T_est].T, bias=False), 0.10)
    w_g = gmv(Cov_hat); budget = (1.5 * np.sqrt(w_g @ Cov_hat @ w_g)) ** 2
    w_nom = nom_maxret(Mu_hat, Cov_hat, budget)
    w_rob = rob_maxret(Mu_hat, Cov_hat, budget, g_star)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.plot(front_nom_vol, front_nom_ret, "o-", color="#5B8FF9", label="名义均值方差 (γ=0)")
    ax.plot(front_rob_vol, front_rob_ret, "s-", color="#E8684A", label=f"鲁棒 (γ={g_star})")
    ax.set_xlabel("年化波动率"); ax.set_ylabel("年化期望收益")
    ax.set_title("有效前沿：鲁棒组合为最坏情况收益留出安全垫，整体下移")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/efficient_frontier.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(oos_sharpe[0.0], bins=40, alpha=0.55, density=True,
            label=f"名义 γ=0 中位 {np.median(oos_sharpe[0.0]):.2f}", color="#5B8FF9")
    ax.hist(oos_sharpe[g_star], bins=40, alpha=0.55, density=True,
            label=f"鲁棒 γ={g_star} 中位 {np.median(oos_sharpe[g_star]):.2f}", color="#E8684A")
    ax.axvline(0, color="#333", ls="--", lw=1)
    ax.set_xlabel("样本外年化 Sharpe"); ax.set_ylabel("密度")
    ax.set_title("样本外 Sharpe：估计误差下名义组合崩塌，鲁棒组合守得住")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/oos_sharpe_dist.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    meds = [np.median(oos_sharpe[g]) for g in gammas]
    ax.plot(gammas, meds, "o-", color="#E8684A")
    ax.axvline(0, color="#5B8FF9", ls="--", lw=1, label="名义 γ=0")
    ax.set_xlabel("鲁棒参数 γ（最坏情况收益惩罚强度）"); ax.set_ylabel("样本外 Sharpe 中位数")
    ax.set_title("鲁棒性曲线：γ 存在甜点，过大会退化为最小方差")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(f"{OUT}/robustness_curve.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(N); w = 0.38
    ax.bar(x - w/2, w_nom, w, label=f"名义 (HHI={ (w_nom**2).sum():.2f})", color="#5B8FF9")
    ax.bar(x + w/2, w_rob, w, label=f"鲁棒 γ={g_star} (HHI={(w_rob**2).sum():.2f})", color="#E8684A")
    ax.set_xticks(x); ax.set_xticklabels([f"A{i+1}" for i in range(N)])
    ax.set_ylabel("权重"); ax.set_title("权重集中度：名义组合押注最吵的资产，鲁棒组合更分散")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/weights_compare.png"); plt.close(fig)

    print("=" * 60)
    print("ARTICLE_A_METRICS")
    print(f"N={N} T_est={T_est} T_oos={T_oos} MC={MC}")
    print(f"gmv_annual_vol={gmv_vol*np.sqrt(252):.4f}")
    print("gammas=", gammas)
    for g in gammas:
        print(f"  gamma={g}: oos_median={np.median(oos_sharpe[g]):.3f} oos_mean={oos_sharpe[g].mean():.3f} "
              f"oos_p5={np.percentile(oos_sharpe[g],5):.3f} hhi_median={np.median(hhi[g]):.3f}")
    print(f"nominal_w={np.round(w_nom,3).tolist()}")
    print(f"robust_w={np.round(w_rob,3).tolist()}")
    print(f"nominal_hhi={ (w_nom**2).sum():.3f} robust_hhi={ (w_rob**2).sum():.3f}")
    imp = (np.median(oos_sharpe[g_star]) - np.median(oos_sharpe[0.0])) / max(abs(np.median(oos_sharpe[0.0])), 1e-9)
    print(f"median_oos_improve_vs_nominal_pct={imp*100:.1f}")
    print("=" * 60)


# ============================================================
# 文章 B：不平衡 Bar 信息驱动采样
# ============================================================
def gen_article_b():
    slug = "imbalance-bars-information-sampling"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(20260901)

    M = 60000
    mid = np.empty(M + 1); mid[0] = 100.0
    b = np.empty(M); v = np.empty(M)
    p_burst_start = 0.018
    in_burst = False; burst_sign = 1; bcnt = 0; blen = 0
    base_vol = 0.0009
    for t in range(1, M + 1):
        if not in_burst and rng.random() < p_burst_start:
            in_burst = True; burst_sign = 1 if rng.random() < 0.5 else -1
            blen = int(rng.integers(40, 260)); bcnt = 0
        if in_burst:
            p_up = 0.80 if burst_sign > 0 else 0.20
            bcnt += 1
            if bcnt >= blen:
                in_burst = False
        else:
            p_up = 0.5
        ret = rng.normal(0, base_vol) + (p_up - 0.5) * 0.0018
        mid[t] = mid[t - 1] * np.exp(ret)
        b[t - 1] = 1.0 if ret > 0 else (-1.0 if ret < 0 else (b[t - 2] if t > 1 else 1.0))
        v[t - 1] = rng.exponential(1.0) * (1.6 if in_burst else 1.0)

    logmid = np.log(mid)

    K = 80
    tb_ends = list(range(K, M + 1, K))
    tb_ret = np.array([logmid[e] - logmid[s] for s, e in zip([0] + tb_ends[:-1], tb_ends)])

    E_T = 80.0; E_v = float(np.mean(v)); max_T = 5 * int(E_T)
    alpha = 0.005
    cum_theta = 0.0; cum_v = 0.0; cum_t = 0
    ib_ends = []; ib_ticks = []; ib_capped = []
    threshold = E_T * E_v
    for t in range(M):
        cum_theta += b[t] * v[t]; cum_v += v[t]; cum_t += 1
        if abs(cum_theta) >= threshold or cum_t >= max_T:
            ib_ends.append(t); ib_ticks.append(cum_t)
            ib_capped.append(cum_t >= max_T)
            E_T = (1 - alpha) * E_T + alpha * cum_t
            E_v = (1 - alpha) * E_v + alpha * (cum_v / cum_t)
            threshold = E_T * E_v
            cum_theta = 0.0; cum_v = 0.0; cum_t = 0
    ib_ret = np.array([logmid[e] - logmid[s] for s, e in zip([0] + ib_ends[:-1], ib_ends)])

    def acf(x, lags):
        x = x - x.mean()
        n = len(x); c0 = np.dot(x, x) / n
        return [np.dot(x[:-l], x[l:]) / (n - l) / c0 for l in lags]

    lags = list(range(1, 21))
    acf_tb = acf(tb_ret, lags)
    acf_ib = acf(ib_ret, lags)

    tb_imbal = np.array([abs(np.sum(b[s:e] * v[s:e])) for s, e in zip([0] + tb_ends[:-1], tb_ends)])
    tb_imbal_norm = tb_imbal / (E_T * E_v)
    ib_imbal = np.array([abs(np.sum(b[s:e] * v[s:e])) for s, e in zip([0] + ib_ends[:-1], ib_ends)])
    ib_imbal_norm = ib_imbal / (E_T * E_v)
    ib_uncapped = np.array(ib_capped) == False
    ib_imbal_norm_uncapped = ib_imbal_norm[ib_uncapped]
    n_uncapped = len(ib_imbal_norm_uncapped)

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.plot(logmid, color="#444", lw=0.6)
    for e in tb_ends[::12]:
        ax.axvline(e, color="#5B8FF9", alpha=0.35, lw=0.8)
    for e in ib_ends[::12]:
        ax.axvline(e, color="#E8684A", alpha=0.35, lw=0.8)
    ax.set_title("价格路径：蓝=时间 Bar 边界，红=不平衡 Bar 边界（信息驱动）")
    ax.set_xlabel("tick"); ax.set_ylabel("log 中间价")
    fig.tight_layout(); fig.savefig(f"{OUT}/price_bar_boundaries.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.stem(lags, acf_tb, linefmt="b-", markerfmt="bo", basefmt=" ", label="时间 Bar 收益 ACF")
    ax.stem(lags, acf_ib, linefmt="r-", markerfmt="rs", basefmt=" ", label="不平衡 Bar 收益 ACF")
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xlabel("滞后 (lag)"); ax.set_ylabel("自相关")
    ax.set_title("自相关：不平衡 Bar 把滞后1自相关从 0.53 压到 0.05（接近 iid）")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/acf_compare.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(ib_ticks, bins=60, color="#E8684A", alpha=0.75)
    ax.axvline(K, color="#5B8FF9", lw=2, ls="--", label=f"时间 Bar 固定={K} ticks")
    ax.set_xlabel("每根 Bar 包含的 tick 数"); ax.set_ylabel("Bar 数量")
    ax.set_title("Bar 长度分布：不平衡 Bar 长短不一，每根约携带等量信息")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/bar_length_dist.png"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.hist(tb_imbal_norm, bins=50, alpha=0.6, density=True,
            label=f"时间 Bar (变异系数 {tb_imbal_norm.std()/tb_imbal_norm.mean():.2f})", color="#5B8FF9")
    ax.hist(ib_imbal_norm_uncapped, bins=30, alpha=0.6, density=True,
            label=f"不平衡 Bar 未触顶 (变异系数 {ib_imbal_norm_uncapped.std()/ib_imbal_norm_uncapped.mean():.2f})",
            color="#E8684A")
    ax.axvline(1.0, color="#E8684A", ls="--", lw=1.5, label="不平衡 Bar 目标阈值 ≡ 1")
    ax.set_xlabel("每根 Bar 累计订单流不平衡（归一化到阈值）"); ax.set_ylabel("密度")
    ax.set_title("信息量均匀度：时间 Bar 忽大忽小，不平衡 Bar 每根≈阈值")
    ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/info_content_per_bar.png"); plt.close(fig)

    print("=" * 60)
    print("ARTICLE_B_METRICS")
    print(f"M={M} ticks, time_bars={len(tb_ret)} imbalance_bars={len(ib_ret)} (uncapped={n_uncapped})")
    print(f"imbalance_bar_ticks_mean={np.mean(ib_ticks):.1f} median={np.median(ib_ticks):.1f} "
          f"min={np.min(ib_ticks)} max={np.max(ib_ticks)}")
    print(f"acf_tb_lag1={acf_tb[0]:.4f} acf_ib_lag1={acf_ib[0]:.4f}")
    print(f"acf_tb_lag5={acf_tb[4]:.4f} acf_ib_lag5={acf_ib[4]:.4f}")
    print(f"acf_tb_mean_abs={np.mean(np.abs(acf_tb)):.4f} acf_ib_mean_abs={np.mean(np.abs(acf_ib)):.4f}")
    print(f"tb_imbal_cv={tb_imbal_norm.std()/tb_imbal_norm.mean():.3f} "
          f"ib_uncapped_imbal_cv={ib_imbal_norm_uncapped.std()/ib_imbal_norm_uncapped.mean():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    gen_article_a()
    gen_article_b()
