#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-08-31 两篇量化文章配图 + 核心数值（numpy 合成，固定 seed 可复现）。
  A. semivariance-downside-portfolio  半方差与下行风险组合
  B. limit-order-book-hmm             限价单簿隐马尔可夫建模
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

rng = np.random.default_rng(20260831)


from scipy.optimize import minimize


def shrink(M, delta=0.1):
    d = np.mean(np.diag(M))
    return (1 - delta) * M + delta * d * np.eye(M.shape[0])


def long_only_port(M):
    """最小风险组合（Σ 或 S），多空受限 w>=0, sum w = 1。"""
    N = M.shape[0]
    scale = 1e6  # 缩放以稳定 SLSQP 数值
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    res = minimize(lambda w: w @ (scale * M) @ w, np.full(N, 1/N), method="SLSQP",
                   constraints=cons, bounds=[(0, 1)] * N, options={"ftol": 1e-12, "maxiter": 1000})
    return res.x


# ============================================================
# 文章 A：半方差与下行风险组合
# ============================================================
def gen_article_a():
    slug = "semivariance-downside-portfolio"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)

    N, T = 8, 600
    # 设计「强不对称」：
    #   低波动组(0-3)：基准波动低，但 3.5% 概率遇到 -4.5% 崩盘跳 -> 左尾极厚、半方差大
    #   高波动组(4-7)：基准波动高，但只有 +3% 上行跳、绝不崩盘 -> 左尾薄、半方差小
    # 于是 min-σ(Σ) 会重仓低波动的崩盘组，min-半方差(S) 会改仓几乎不崩的高波动组。
    drift_A = 0.0005; vol_A = 0.0035; crash_p = 0.035; crash_amp = 0.045
    drift_B = 0.0008; vol_B = 0.0080; up_p = 0.035; up_amp = 0.030
    R = np.empty((T, N))
    for i in range(N):
        if i < 4:
            base = drift_A + rng.normal(0, vol_A, T)
            jump = rng.random(T) < crash_p
            R[:, i] = base - jump * crash_amp
        else:
            base = drift_B + rng.normal(0, vol_B, T)
            jump = rng.random(T) < up_p
            R[:, i] = base + jump * up_amp
    mu = R.mean(0)
    # 校验：前4（低波动+崩盘） vs 后4（高波动+上行跳）

    # 全协方差（样本）
    Sigma = np.cov(R.T, bias=False)
    # 下行半协方差（target = 0）：只取亏损那一边的乘积
    below = np.minimum(R, 0.0)
    S = (below.T @ below) / T
    Sigmas = shrink(Sigma, 0.1)
    Ss = shrink(S, 0.1)

    # 最小方差组合（多空受限）
    w_mv = long_only_port(Sigmas)
    # 最小半方差组合（多空受限，独立最优，不强制同收益）
    w_dn = long_only_port(Ss)

    def stats(w):
        rp = w @ mu
        var = w @ Sigmas @ w
        semi = w @ Ss @ w
        vol = np.sqrt(max(var, 1e-12))
        semidev = np.sqrt(max(semi, 1e-12))
        return rp, vol, semidev, rp / vol, rp / semidev

    rp_mv, vol_mv, semi_mv, sh_mv, so_mv = stats(w_mv)
    rp_dn, vol_dn, semi_dn, sh_dn, so_dn = stats(w_dn)

    # 最大回撤：对两个组合的历史日收益序列做 iid 自助（3000 路径 × 252 日）
    ret_mv = R @ w_mv
    ret_dn = R @ w_dn
    B = 3000
    mdd_mv, mdd_dn = [], []
    for _ in range(B):
        s1 = rng.choice(ret_mv, 252)
        s2 = rng.choice(ret_dn, 252)
        def mdd(x):
            eq = np.cumprod(1 + x)
            return (eq / np.maximum.accumulate(eq) - 1).min()
        mdd_mv.append(mdd(s1)); mdd_dn.append(mdd(s2))
    mdd_mv = np.array(mdd_mv); mdd_dn = np.array(mdd_dn)

    # ---------- 图 1：单资产收益分布 + 0 目标线，阴影标出下行半方差区 ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(R[:, 0], bins=60, color="#5B8FF9", alpha=0.75, density=True)
    ax.axvline(0, color="#E8684A", lw=2, label="目标线 τ=0（下行分界）")
    ax.axvspan(ax.get_xlim()[0], 0, color="#E8684A", alpha=0.10)
    ax.set_title("资产 1 日收益分布：只有跌破 τ=0 的半边计入半方差")
    ax.set_xlabel("日收益率"); ax.set_ylabel("概率密度"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/dist_target.png"); plt.close(fig)

    # ---------- 图 2：两种组合权重对比 ----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = np.arange(N); w = 0.38
    ax.bar(x - w/2, w_mv, w, label="最小方差 (Σ)", color="#5B8FF9")
    ax.bar(x + w/2, w_dn, w, label="最小半方差 (S)", color="#E8684A")
    ax.set_xticks(x); ax.set_xticklabels([f"A{i+1}" for i in range(N)])
    ax.set_title("权重对比：下行风险组合更少押注「同跌」资产")
    ax.set_ylabel("权重"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/weights_compare.png"); plt.close(fig)

    # ---------- 图 3：风险分解（总波动 vs 下行半偏差）----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    labels = ["最小方差", "最小半方差"]
    vols = [vol_mv, vol_dn]; semis = [semi_mv, semi_dn]
    xx = np.arange(2); w = 0.38
    ax.bar(xx - w/2, vols, w, label="总波动 σ (全协方差)", color="#5B8FF9")
    ax.bar(xx + w/2, semis, w, label="下行半偏差 (半协方差)", color="#E8684A")
    ax.set_xticks(xx); ax.set_xticklabels(labels)
    ax.set_title("风险口径：相同收益下，半方差组合把『下行半偏差』压低")
    ax.set_ylabel("风险度量"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/risk_decomp.png"); plt.close(fig)

    # ---------- 图 4：最大回撤分布对比（3000 次自助）----------
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.hist(mdd_mv*100, bins=50, alpha=0.6, label=f"最小方差 中位 {np.median(mdd_mv)*100:.1f}%", color="#5B8FF9", density=True)
    ax.hist(mdd_dn*100, bins=50, alpha=0.6, label=f"最小半方差 中位 {np.median(mdd_dn)*100:.1f}%", color="#E8684A", density=True)
    ax.set_title("最大回撤分布：最小半方差组合尾部更轻")
    ax.set_xlabel("最大回撤 (%)"); ax.set_ylabel("密度"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/mdd_dist.png"); plt.close(fig)

    print("="*60)
    print("ARTICLE_A_METRICS")
    print(f"N={N}, T={T}, 低波动崩盘组(0-3) vs 高波动上行跳组(4-7)")
    print(f"每组平均下行半偏差: low={np.sqrt(np.mean([ (np.minimum(R[:,i],0)**2).mean() for i in range(4)])):.5f} high={np.sqrt(np.mean([ (np.minimum(R[:,i],0)**2).mean() for i in range(4,8)])):.5f}")
    print(f"每组平均总波动: low={np.std(R[:,:4]):.5f} high={np.std(R[:,4:]):.5f}")
    print(f"mu0_mv={w_mv@mu:.5f} mu0_dn={w_dn@mu:.5f} ret_diff={(w_dn@mu - w_mv@mu):.5f}")
    print(f"w_mv={np.round(w_mv,3).tolist()}")
    print(f"w_dn={np.round(w_dn,3).tolist()}")
    print(f"rp_mv={rp_mv:.5f} vol_mv={vol_mv:.5f} semi_mv={semi_mv:.5f} sharpe_mv={sh_mv:.3f} sortino_mv={so_mv:.3f}")
    print(f"rp_dn={rp_dn:.5f} vol_dn={vol_dn:.5f} semi_dn={semi_dn:.5f} sharpe_dn={sh_dn:.3f} sortino_dn={so_dn:.3f}")
    print(f"mdd_mv_median={np.median(mdd_mv):.4f} mdd_dn_median={np.median(mdd_dn):.4f}")
    print(f"mdd_mv_p95={np.percentile(mdd_mv,95):.4f} mdd_dn_p95={np.percentile(mdd_dn,95):.4f}")
    print(f"semi_reduction={(1-semi_dn/semi_mv):.3f} mdd_median_reduction={(1-np.median(mdd_dn)/np.median(mdd_mv)):.3f}")
    print(f"var_change={(vol_dn/vol_mv-1):.3f} sortino_improve={(so_dn/so_mv-1):.3f} ret_dn_vs_mv={(rp_dn/rp_mv-1):.3f}")
    print("="*60)


# ============================================================
# 文章 B：限价单簿隐马尔可夫建模（数值稳定的缩放 Baum-Welch）
# ============================================================
def gen_article_b():
    slug = "limit-order-book-hmm"
    OUT = f"public/images/{slug}"
    os.makedirs(OUT, exist_ok=True)

    Tn = 1500
    A_true = np.array([[0.99, 0.01], [0.06, 0.94]])
    pi_true = np.array([0.85, 0.15])
    mean0 = np.array([0.00, 0.0002]); cov0 = np.array([[0.0004, 0.0], [0.0, 0.0008]])
    mean1 = np.array([0.09, 0.0000]); cov1 = np.array([[0.0025, 0.0], [0.0, 0.0040]])

    states = np.empty(Tn, dtype=int)
    states[0] = 0 if rng.random() < pi_true[0] else 1
    for t in range(1, Tn):
        states[t] = 0 if rng.random() < A_true[states[t-1], 0] else 1
    obs = np.empty((Tn, 2))
    for t in range(Tn):
        mu_, cov_ = (mean0, cov0) if states[t] == 0 else (mean1, cov1)
        obs[t] = rng.multivariate_normal(mu_, cov_)
    mid = 100 * np.cumprod(1 + obs[:, 1])

    def log_gauss(x, mu, cov):
        d = x - mu
        sign, logdet = np.linalg.slogdet(cov)
        return -0.5 * (d @ np.linalg.solve(cov, d) + logdet + 2*np.log(2*np.pi))

    K = 2
    pi = np.array([0.7, 0.3])
    A = np.array([[0.95, 0.05], [0.05, 0.95]])
    mu = np.array([[0.01, 0.0003], [0.06, 0.0]])
    cov = np.array([[[0.001, 0], [0, 0.001]], [[0.002, 0], [0, 0.004]]])

    # 发射对数概率矩阵
    def emit_log():
        return np.stack([np.array([log_gauss(obs[t], mu[k], cov[k]) for k in range(K)]) for t in range(Tn)])

    for it in range(80):
        logB = emit_log()
        # ---- 缩放前向 ----
        c = np.zeros(Tn)
        alpha = np.zeros((Tn, K))
        log_alpha0 = np.log(pi) + logB[0]
        c[0] = np.logaddexp.reduce(log_alpha0)
        alpha[0] = np.exp(log_alpha0 - c[0])
        for t in range(1, Tn):
            log_num = np.logaddexp.reduce(np.log(alpha[t-1])[:, None] + np.log(A), axis=0) + logB[t]
            c[t] = np.logaddexp.reduce(log_num)
            alpha[t] = np.exp(log_num - c[t])
        # ---- 缩放后向 ----
        beta = np.zeros((Tn, K))
        beta[Tn-1] = 1.0
        for t in range(Tn-2, -1, -1):
            beta[t] = (A @ (beta[t+1] * np.exp(logB[t+1]))) / np.exp(c[t+1])
            beta[t] /= beta[t].sum()
        # gamma
        gamma = alpha * beta
        gamma /= gamma.sum(1, keepdims=True)
        # xi
        xi = np.zeros((Tn-1, K, K))
        for t in range(Tn-1):
            for i in range(K):
                for j in range(K):
                    xi[t, i, j] = alpha[t, i] * A[i, j] * np.exp(logB[t+1, j]) * beta[t+1, j]
            xi[t] /= xi[t].sum()
        # M-step
        pi = gamma[0] / gamma[0].sum()
        A = xi.sum(0); A /= A.sum(1, keepdims=True)
        for k in range(K):
            w = gamma[:, k]
            mu[k] = (w[:, None] * obs).sum(0) / w.sum()
            d = obs - mu[k]
            cov[k] = (w[:, None, None] * (d[:, :, None] @ d[:, None, :])).sum(0) / w.sum()
            cov[k] += 1e-8 * np.eye(2)

    # Viterbi 解码（log 域）
    logB = emit_log()
    delta = np.zeros((Tn, K)); psi = np.zeros((Tn, K), dtype=int)
    delta[0] = np.log(pi) + logB[0]
    for t in range(1, Tn):
        for k in range(K):
            seq = delta[t-1] + np.log(A[:, k]) + logB[t, k]
            psi[t, k] = np.argmax(seq); delta[t, k] = np.max(seq)
    path = np.zeros(Tn, dtype=int); path[Tn-1] = np.argmax(delta[Tn-1])
    for t in range(Tn-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    # 状态匹配
    acc = {}
    for perm in [(0, 1), (1, 0)]:
        pred = np.array([perm[s] for s in path])
        acc[perm] = (pred == states).mean()
    best_perm = max(acc, key=acc.get)
    pred_states = np.array([best_perm[s] for s in path])
    accuracy = (pred_states == states).mean()
    stressed_prob = gamma[:, best_perm[1]] if best_perm[1] == 1 else gamma[:, best_perm[0]]

    # 做市商仿真：固定价差 vs 状态感知价差（受压时减仓 + 加宽价差）
    s0 = 0.0008
    inv_f = 0.0; inv_s = 0.0; eq_fixed = 0.0; eq_state = 0.0
    inv_fixed = []; inv_state = []
    pnl_fixed = []; pnl_state = []
    for t in range(Tn):
        base_trade = np.sign(obs[t, 0]) if abs(obs[t, 0]) > 0.005 else 0.0
        mid_ret = obs[t, 1]
        stressed = pred_states[t] == 1
        # 固定策略：始终吃满 1 单位
        fill_f = 1.0
        eq_fixed += s0 * fill_f - inv_f * mid_ret
        inv_f += fill_f * base_trade
        # 状态感知：受压时只吃 0.4 单位 + 加宽价差到 1.6 倍
        fill_s = 0.4 if stressed else 1.0
        spread_s = s0 * (1.6 if stressed else 1.0)
        eq_state += spread_s * fill_s - inv_s * mid_ret
        inv_s += fill_s * base_trade
        inv_fixed.append(inv_f); inv_state.append(inv_s)
        pnl_fixed.append(eq_fixed); pnl_state.append(eq_state)
    pnl_fixed = np.array(pnl_fixed); pnl_state = np.array(pnl_state)
    inv_std_fixed = np.std(inv_fixed); inv_std_state = np.std(inv_state)
    def mdd_eq(eq):
        return (eq / np.maximum.accumulate(eq) - 1).min()
    mdd_fixed = mdd_eq(pnl_fixed); mdd_state = mdd_eq(pnl_state)

    # ---------- 图 1 ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(mid, color="#444", lw=0.8)
    for t in range(1, Tn):
        if states[t] == 1:
            ax.axvspan(t, t+1, color="#E8684A", alpha=0.18)
    ax.set_title("中间价路径（红色背景 = 真实受压/挤兑状态）")
    ax.set_xlabel("步"); ax.set_ylabel("中间价")
    fig.tight_layout(); fig.savefig(f"{OUT}/mid_price_states.png"); plt.close(fig)

    # ---------- 图 2 ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.hist(obs[states == 0, 0], bins=50, alpha=0.55, density=True, label="状态0 平静", color="#5B8FF9")
    ax.hist(obs[states == 1, 0], bins=50, alpha=0.55, density=True, label="状态1 受压", color="#E8684A")
    ax.axvline(mu[best_perm[0], 0], color="#5B8FF9", ls="--", lw=1.5)
    ax.axvline(mu[best_perm[1], 0], color="#E8684A", ls="--", lw=1.5)
    ax.set_title("订单流失衡 OFI：HMM 把两簇分布分开")
    ax.set_xlabel("OFI"); ax.set_ylabel("密度"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/ofi_states.png"); plt.close(fig)

    # ---------- 图 3 ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(stressed_prob, color="#E8684A", lw=0.8)
    ax.axhline(0.5, color="#888", ls=":", lw=1)
    ax.set_title("HMM 滤波：平滑后的『受压』状态后验概率")
    ax.set_xlabel("步"); ax.set_ylabel("P(受压)")
    fig.tight_layout(); fig.savefig(f"{OUT}/stressed_prob.png"); plt.close(fig)

    # ---------- 图 4 ----------
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(pnl_fixed, label=f"固定价差 (库存σ={inv_std_fixed:.2f}, MDD={mdd_fixed:.2f})", color="#5B8FF9", lw=1)
    ax.plot(pnl_state, label=f"状态感知 (库存σ={inv_std_state:.2f}, MDD={mdd_state:.2f})", color="#E8684A", lw=1)
    ax.set_title("做市商净值：状态感知策略库存风险更低")
    ax.set_xlabel("步"); ax.set_ylabel("累计净值"); ax.legend()
    fig.tight_layout(); fig.savefig(f"{OUT}/mm_pnl.png"); plt.close(fig)

    print("="*60)
    print("ARTICLE_B_METRICS")
    print(f"Tn={Tn}, 真实受压占比={ (states==1).mean():.3f}")
    print(f"transmat_est={np.round(A,3).tolist()}")
    print(f"mu_est={np.round(mu,4).tolist()}")
    print(f"cov_est={np.round(cov,5).tolist()}")
    print(f"viterbi_accuracy={accuracy:.3f} best_perm={best_perm}")
    print(f"inv_std_fixed={inv_std_fixed:.3f} inv_std_state={inv_std_state:.3f} inv_reduction={(1-inv_std_state/inv_std_fixed):.3f}")
    print(f"mdd_fixed={mdd_fixed:.4f} mdd_state={mdd_state:.4f} mdd_reduction={(1-mdd_state/mdd_fixed):.3f}")
    print(f"pnl_final_fixed={pnl_fixed[-1]:.3f} pnl_final_state={pnl_state[-1]:.3f}")
    print("="*60)


if __name__ == "__main__":
    gen_article_a()
    gen_article_b()
