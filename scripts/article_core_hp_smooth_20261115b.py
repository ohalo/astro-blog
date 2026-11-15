# -*- coding: utf-8 -*-
"""卡尔曼平滑器与 HP 滤波器的隐藏前视偏差：现场数值实验的唯一数据源。

本文全部数字的唯一来源。运行本脚本会：
1. 在局部水平模型上现场模拟趋势+噪声，手写卡尔曼前向滤波与 RTS 后向平滑；
2. 独立实现 HP 滤波器法方程（带状矩阵直接解），与 RTS 平滑器逐点对拍；
3. 用「滚动窗口 HP」现场复现前视偏差：全样本 HP 与因果 HP 的趋势差、
   信号翻转、追加 10 天后历史趋势被改写的幅度；
4. 趋势跟踪策略的幻影 Sharpe：前视信号 vs 因果信号 vs 真值信号；
5. 画 5 张图到 public/images/kalman-smoother-hp-filter-lookahead/；
6. 把全部统计量写入同目录 results.json。
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import solveh_banded

plt.rcParams["font.sans-serif"] = ["Heiti TC", "PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

SEED = 202611152
_HERE = os.path.dirname(os.path.abspath(__file__))
_OUT = os.path.join(
    os.path.dirname(_HERE), "public", "images", "kalman-smoother-hp-filter-lookahead"
)
os.makedirs(_OUT, exist_ok=True)


def _save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(_OUT, name), dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------- HP 滤波器
def hp_filter(y, lam):
    """Hodrick-Prescott：min Σ(y_t-τ_t)² + λΣ(Δ²τ_t)²，带状法方程直接解。

    与卡尔曼完全无关的独立实现（线性代数路线），用来和平滑器对拍。
    """
    y = np.asarray(y, float)
    T = y.size
    # 逐行装配 D'D：行 s（0 基，s=2..T-1）在 (s-2,s-1,s) 处为 (1,-2,1)。
    # 内部元主对角 6λ、偏移1 -4λ、偏移2 λ；两端各少参与若干行，边界元由此循环自然算对。
    ab = np.zeros((3, T))
    ab[2, :] = 1.0                # 数据项 I
    coef = np.array([1.0, -2.0, 1.0])
    for s in range(2, T):
        cols = (s - 2, s - 1, s)
        for a in range(3):
            for b in range(a, 3):
                off = cols[b] - cols[a]          # 0/1/2
                ab[2 - off, cols[b]] += coef[a] * coef[b] * lam
    return solveh_banded(ab, y)


# -------------------------------------------------- 局部水平模型卡尔曼滤波
def kalman_llm(y, q_beta, sig_eps, p0=1e8):
    """状态 [τ, β]：τ_t=τ_{t-1}+β_{t-1}+η, β_t=β_{t-1}+ζ(方差 q_beta)，
    观测 y_t=τ_t+ε(方差 sig_eps²)。返回滤波估计 τ̂_{t|t} 与滤波方差 P_t[0,0]。
    """
    T = y.size
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, q_beta]])
    H = np.array([1.0, 0.0])
    x = np.array([y[0], 0.0])
    P = np.eye(2) * p0
    xf = np.zeros(T)
    xb = np.zeros(T)
    pf = np.zeros(T)
    ll = 0.0
    for t in range(T):
        if t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        v = y[t] - float(H @ x)               # 新息
        F = float(H @ P @ H + sig_eps**2)
        ll += -0.5 * (np.log(2 * np.pi * F) + v * v / F)
        K = (P @ H) / F
        x = x + K * v
        P = P - np.outer(K, H @ P)
        xf[t] = x[0]
        xb[t] = x[1]
        pf[t] = P[0, 0]
    return xf, pf, ll, xb


def rts_smooth(y, q_beta, sig_eps, p0=1e8):
    """RTS 后向平滑：返回平滑估计 τ̂_{t|T} 与平滑方差 P_{t|T}[0,0]。"""
    T = y.size
    A = np.array([[1.0, 1.0], [0.0, 1.0]])
    Q = np.array([[0.0, 0.0], [0.0, q_beta]])
    H = np.array([1.0, 0.0])
    xp = np.zeros((T, 2))   # 预测 x_{t|t-1}
    pp = np.zeros((T, 2, 2))
    xf = np.zeros((T, 2))   # 滤波 x_{t|t}
    pf = np.zeros((T, 2, 2))
    x = np.array([y[0], 0.0])
    P = np.eye(2) * p0
    for t in range(T):
        if t > 0:
            x = A @ x
            P = A @ P @ A.T + Q
        xp[t] = x
        pp[t] = P
        v = y[t] - float(H @ x)
        F = float(H @ P @ H + sig_eps**2)
        K = (P @ H) / F
        x = x + K * v
        P = P - np.outer(K, H @ P)
        xf[t] = x
        pf[t] = P
    xs = xf.copy()
    Ps = pf.copy()
    for t in range(T - 2, -1, -1):
        C = pf[t] @ A.T @ np.linalg.inv(pp[t + 1])
        xs[t] = xf[t] + C @ (xs[t + 1] - xp[t + 1])
        Ps[t] = pf[t] + C @ (Ps[t + 1] - pp[t + 1]) @ C.T
    return (xs[:, 0], np.array([Ps[t][0, 0] for t in range(T)]),
            xs[:, 1], np.array([Ps[t][1, 1] for t in range(T)]))


def simulate_llm(T, rng, tau0=1.0, beta0=0.0008, q_beta=4e-8, sig_eps=0.015):
    """从局部水平模型真值生成数据：β 是随机游走，τ 累加 β，y=τ+观测噪声。"""
    betas = np.zeros(T)
    betas[0] = beta0
    betas[1:] = beta0 + np.cumsum(rng.normal(0.0, np.sqrt(q_beta), T - 1))
    taus = tau0 + np.concatenate([[0.0], np.cumsum(betas[:-1])])
    y = taus + rng.normal(0.0, sig_eps, T)
    return taus, betas, y


def main():
    rng = np.random.default_rng(SEED)
    results = {}

    # ============ 实验 1：滤波 vs 平滑（真值已知的局部水平模型） ============
    N1 = 400
    q_beta_true = 4e-8     # β 增量标准差 2e-4/日
    sig_eps_true = 0.015
    taus1, betas1, y1 = simulate_llm(N1, rng, q_beta=q_beta_true, sig_eps=sig_eps_true)
    xf1, pf1, _, _ = kalman_llm(y1, q_beta_true, sig_eps_true)
    xs1, ps1, _, _ = rts_smooth(y1, q_beta_true, sig_eps_true)
    rmse_f = float(np.sqrt(np.mean((xf1 - taus1) ** 2)))
    rmse_s = float(np.sqrt(np.mean((xs1 - taus1) ** 2)))
    improvement = float(1.0 - rmse_s / rmse_f)
    results["filter_vs_smoother"] = {
        "n": N1,
        "q_beta": q_beta_true,
        "sigma_eps": sig_eps_true,
        "rmse_filter": rmse_f,
        "rmse_smoother": rmse_s,
        "improvement_pct": improvement * 100.0,
    }

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1]})
    ax = axes[0]
    tt = np.arange(N1)
    ax.plot(tt, taus1, lw=2, color="#222222", label="真实趋势 τ_t")
    ax.plot(tt, y1, lw=0.6, color="#999999", alpha=0.8, label="观测 y_t")
    ax.plot(tt, xf1, lw=1.4, color="#d62728", label="卡尔曼滤波 τ̂(t|t)")
    ax.plot(tt, xs1, lw=1.4, color="#1f77b4", label="RTS 平滑 τ̂(t|T)")
    ax.set_title(f"前向滤波 vs 后向平滑（RMSE {rmse_f:.5f} → {rmse_s:.5f}，改善 {improvement*100:.1f}%）")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_ylabel("水平")
    ax2 = axes[1]
    ax2.plot(tt, xf1 - taus1, lw=0.9, color="#d62728", label="滤波误差")
    ax2.plot(tt, xs1 - taus1, lw=0.9, color="#1f77b4", label="平滑误差")
    ax2.axhline(0, color="k", lw=0.5)
    ax2.legend(fontsize=8)
    ax2.set_xlabel("t（天）")
    ax2.set_ylabel("对真值误差")
    _save(fig, "fig01.png")

    # ============ 实验 2：误差方差动力学与端点效应 ============
    var_f = np.sqrt(pf1)     # 滤波标准差
    var_s = np.sqrt(ps1)     # 平滑标准差
    conv_f = float(np.mean(var_f[-40:]))
    mid_s = float(np.mean(var_s[100:300]))
    endpoint_gap = float(abs(var_s[-1] - var_f[-1]))
    results["error_dynamics"] = {
        "filter_sd_converged": conv_f,
        "smoother_sd_mid": mid_s,
        "smoother_sd_at_T": float(var_s[-1]),
        "filter_sd_at_T": float(var_f[-1]),
        "endpoint_abs_gap": endpoint_gap,
        "ratio_mid": float(mid_s / conv_f),
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(tt, var_f, lw=1.6, color="#d62728", label="滤波误差标准差 √P(t|t)")
    ax.plot(tt, var_s, lw=1.6, color="#1f77b4", label="平滑误差标准差 √P(t|T)")
    ax.axhline(conv_f, ls=":", color="#d62728", lw=0.8)
    ax.axhline(mid_s, ls=":", color="#1f77b4", lw=0.8)
    ax.annotate(f"中段平滑误差带 {mid_s:.5f}\n= 滤波的 {mid_s/conv_f:.1%}",
                xy=(200, mid_s), xytext=(215, mid_s + 0.004), fontsize=8)
    ax.annotate(f"t=T 处两者重合\n差 {endpoint_gap:.2e}",
                xy=(N1 - 1, var_s[-1]), xytext=(N1 - 130, var_s[-1] + 0.006), fontsize=8)
    ax.set_xlabel("t（天）")
    ax.set_ylabel("趋势估计误差标准差")
    ax.set_title("滤波误差单调收敛、平滑误差中段最低、最后一个点两者恒等（端点效应）")
    ax.legend(fontsize=8)
    _save(fig, "fig02.png")

    # ============ 实验 3：HP 滤波器 ≡ RTS 平滑器（独立实现对拍） ============
    # 模型 σ_ε=1，β 增量方差 q，MAP 目标 = Σ(y-τ)² + (1/q)Σ(Δ²τ)² ⇒ λ = 1/q
    y3 = simulate_llm(250, rng, tau0=0.0, beta0=0.01, q_beta=1.0 / 50.0, sig_eps=1.0)[2]
    lam_grid = [1.0, 10.0, 100.0, 1000.0]
    hp_curves = {}
    max_diffs = {}
    for lam in lam_grid:
        tau_hp = hp_filter(y3, lam)
        tau_sm = rts_smooth(y3, q_beta=1.0 / lam, sig_eps=1.0)[0]
        hp_curves[lam] = tau_hp
        max_diffs[lam] = float(np.max(np.abs(tau_hp - tau_sm)))
    worst = max(max_diffs.values())
    results["hp_equals_smoother"] = {
        "n": 250,
        "lambdas": lam_grid,
        "max_abs_diff": {str(l): max_diffs[l] for l in lam_grid},
        "worst_diff": worst,
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    for i, lam in enumerate(lam_grid):
        ax.plot(np.arange(250), hp_curves[lam], lw=1.3, label=f"HP λ={lam:g}（法方程直接解）")
        ax.plot(np.arange(250), rts_smooth(y3, 1.0 / lam, 1.0)[0],
                ls="--", lw=0.9, color="k", alpha=0.5)
    ax.set_title(f"两条独立实现（线性代数 vs 状态空间）逐点一致，最大差 {worst:.2e}")
    ax.set_xlabel("t")
    ax.set_ylabel("趋势估计")
    ax.legend(fontsize=8)
    _save(fig, "fig03.png")

    # ============ 实验 4：滚动窗口 HP = 卡尔曼滤波（因果性恒等式） ============
    # 对每个 t，把 HP 只跑在前 t 个观测上、取最后一个点 = 滤波估计 τ̂(t|t)
    q4, sig4 = 1e-8, 0.012
    tau4, beta4, y4 = simulate_llm(300, rng, tau0=2.0, beta0=0.0006,
                                   q_beta=q4, sig_eps=sig4)
    # HP 目标 Σ(y-τ)²+λΣ(Δ²τ)² 与状态空间（噪声方差 σ_ε²、β 增量方差 q）的换元：λ=σ_ε²/q
    lam4 = sig4**2 / q4
    xf4 = kalman_llm(y4, q4, sig4)[0]
    causal_hp = np.array([hp_filter(y4[:t], lam4)[-1] for t in range(2, 301)])
    exp_hp_eq = float(np.max(np.abs(causal_hp - xf4[1:])))
    # 前视：全样本 HP 在第 t 点的估计
    full_hp = hp_filter(y4, lam4)
    gap_view = float(np.mean(np.abs(full_hp - xf4)))
    gap_max = float(np.max(np.abs(full_hp - xf4)))
    # 数据还没来齐时「未来 10 天一到，历史趋势被改写」的幅度
    # 未来 10 天：从第 300 天真值水平继续同参数演化（无缝延续，无人为跳变）
    tau_next = tau4[-1] + np.cumsum(np.concatenate([[beta4[-1]], beta4[-1:] + rng.normal(0, 1e-4, 9)]))
    y_ext = np.concatenate([y4, tau_next + rng.normal(0.0, 0.012, 10)])
    full_hp_ext = hp_filter(y_ext, lam4)
    tail25 = slice(225, 300)
    rewrite = float(np.max(np.abs(full_hp_ext[:300][tail25] - full_hp[tail25])))
    # 平滑-滤波差对「还剩多少数据」的依赖：中段几乎不动，末尾剧烈改写
    rewrite_profile = np.abs(full_hp_ext[:300] - full_hp)
    results["causality"] = {
        "n": 300,
        "lambda_hp": float(lam4),
        "q_beta": q4,
        "sigma_eps": sig4,
        "expanding_hp_vs_filter_max_diff": exp_hp_eq,
        "full_hp_vs_causal_mean_abs_gap": gap_view,
        "full_hp_vs_causal_max_abs_gap": gap_max,
        "rewrite_after_10_more_days_tail25pct": rewrite,
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(2, 301), causal_hp, lw=1.4, color="#d62728",
            label="因果 HP：只用 y_1..t（= 卡尔曼滤波）")
    ax.plot(np.arange(300), full_hp, lw=1.4, color="#1f77b4",
            label="前视 HP：全样本 y..T")
    ax.plot(np.arange(300), y4, lw=0.5, color="#aaaaaa", alpha=0.7, label="观测")
    ax.set_title(f"同一个 t 两条曲线差多少？平均 {gap_view:.4f}，最大 {gap_max:.4f}；"
                 f"追加 10 天后最后 1/4 样本趋势最大改写 {rewrite:.4f}")
    ax.set_xlabel("t")
    ax.set_ylabel("趋势估计")
    ax.legend(fontsize=8)
    _save(fig, "fig04.png")

    # ============ 实验 5：趋势跟踪策略的幻影 Sharpe ============
    # 资产对数价格 = 局部水平模型；信号 = 趋势斜率符号；仓位滞后一天
    N5 = 300
    q5, sig5 = 1e-8, 0.012
    tau5, beta5, y5 = simulate_llm(N5, rng, tau0=4.0, beta0=0.0006,
                                   q_beta=q5, sig_eps=sig5)
    lam5 = sig5**2 / q5
    bf5 = kalman_llm(y5, q5, sig5)[3]     # 滤波斜率 β̂(t|t)：t 收盘已知
    bs5 = rts_smooth(y5, q5, sig5)[2]     # 平滑斜率 β̂(t|T)：用到全部未来数据
    ret = np.diff(y5)    # ret[t] = y[t+1]-y[t]：t→t+1 的收益，决策必须在 t 收盘前定
    def sharpe(pos):
        # pos[t]：t 收盘可知的仓位，吃 ret[t]
        pnl = pos[:-1] * ret
        s = pnl.std()
        return float(pnl.mean() / s * np.sqrt(252)) if s > 0 else 0.0
    # 仓位 = 估计趋势斜率的符号，三种斜率只差「能看到多少未来」：
    pos_look = (bs5 > 0).astype(float)      # 平滑器：t 时刻已知道 t+1、t+2 的趋势走向
    pos_cau = (bf5 > 0).astype(float)       # 滤波器：只有 y_1..t
    pos_true = (beta5 > 0).astype(float)    # 上帝视角：真 β_t
    sh_look, sh_cau, sh_true = sharpe(pos_look), sharpe(pos_cau), sharpe(pos_true)
    flip_look = float(np.mean(np.abs(np.diff(pos_look)) > 0))
    flip_cau = float(np.mean(np.abs(np.diff(pos_cau)) > 0))
    phantom = sh_look - sh_cau
    results["phantom_sharpe"] = {
        "n": N5,
        "lambda": float(lam5),
        "q_beta": q5,
        "sigma_eps": sig5,
        "beta0": 0.0006,
        "sigma_eps": 0.012,
        "sharpe_lookahead_hp": sh_look,
        "sharpe_causal_hp": sh_cau,
        "sharpe_true_trend": sh_true,
        "phantom_gap": phantom,
        "signal_flip_rate_lookahead": flip_look,
        "signal_flip_rate_causal": flip_cau,
        "trend_slope_mean": float(beta5.mean()),
    }
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(np.arange(N5), pos_look, lw=0.9, color="#1f77b4", alpha=0.85,
            label=f"前视 HP 仓位（Sharpe {sh_look:.2f}）")
    ax.plot(np.arange(N5), pos_cau - 0.12, lw=0.9, color="#d62728", alpha=0.85,
            label=f"因果 HP 仓位（Sharpe {sh_cau:.2f}）")
    ax.plot(np.arange(N5), pos_true - 0.24, lw=0.9, color="#2ca02c", alpha=0.85,
            label=f"真趋势仓位（Sharpe {sh_true:.2f}）")
    ax.set_ylim(-0.35, 1.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title(f"幻影 Sharpe：把平滑器当滤波用，回测虚高 {phantom:.2f}"
                 f"（{sh_look:.2f} vs {sh_cau:.2f}），换手 {flip_cau:.1%}→{flip_look:.1%}")
    ax.set_xlabel("t（天）")
    ax.set_ylabel("仓位（0/1，三行错开显示）")
    ax.legend(fontsize=8, loc="lower right")
    _save(fig, "fig05.png")

    # ============ 实验 6：λ 的最大似然辨识（新息分解） ============
    # 网格搜索状态噪声方差 q（σ_ε 固定在生成真值），报表时换算回 HP 的
    # λ = σ_ε²/q，与第四节的理论桥保持同一量纲。
    log_grid = np.linspace(np.log(1e-9), np.log(1e-5), 120)
    lls = np.array([kalman_llm(y1, np.exp(g), sig_eps_true)[2] for g in log_grid])
    g_hat = float(log_grid[np.argmax(lls)])
    q_hat = float(np.exp(g_hat))
    q_from_llm = kalman_llm(y1, q_beta_true, sig_eps_true)[2]
    # 真参数处的似然 vs 网格最优
    results["lambda_ml"] = {
        "q_beta_true": q_beta_true,
        "sigma_eps_fixed": sig_eps_true,
        "lam_true": sig_eps_true**2 / q_beta_true,
        "lam_hat_ml": sig_eps_true**2 / q_hat,
        "loglik_at_hat": float(lls.max()),
        "loglik_at_true": q_from_llm,
        "ml_beats_true_param": bool(lls.max() >= q_from_llm),
    }

    with open(os.path.join(_OUT, "results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
