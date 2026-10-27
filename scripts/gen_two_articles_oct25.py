#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-25 两篇量化文章配图 + 核心数值（numpy 合成，固定 seed 可复现）。
  A. kalman-pairs-cointegration-fusion  协整配对 Kalman 融合
  B. gnn-sector-correlation-network     图神经网络板块关联
所有图表均为真实计算图，数值固定随机种子可复现。
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

BASE = "/Users/halo123456/workspace/astro-blog/public/images"


# ============================================================
# 文章 A：协整配对 Kalman 融合
# ============================================================
def gen_kalman():
    slug = "kalman-pairs-cointegration-fusion"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    T, split = 1500, 900
    PHI_SP, NU = 0.80, 0.05      # spread AR(1)：半衰期约 3.1 天
    BETA_DRIFT, BETA_NOISE = 0.00025, 0.0008
    XVOL, XDRIFT = 0.015, 0.0002
    QB = 0.001                   # Kalman β 过程噪声（与真实量级一致）
    SEED = 20261025

    def make_pair(seed):
        rg = np.random.default_rng(seed)
        x = np.empty(T); beta = np.empty(T); y = np.empty(T); sp = np.empty(T)
        x[0], beta[0], y[0], sp[0] = 4.6, 0.85, 4.6, 0.0
        for t in range(1, T):
            x[t] = x[t - 1] + XDRIFT + XVOL * rg.normal()
            beta[t] = beta[t - 1] + BETA_DRIFT + BETA_NOISE * rg.normal()
            sp[t] = PHI_SP * sp[t - 1] + NU * rg.normal()
            y[t] = beta[t] * x[t] + sp[t]
        return x, beta, y, sp

    def kalman_run(x, y, q_beta):
        th = np.array([1.0, 0.0])
        P = np.diag([0.05, 0.05])
        F = np.array([[1.0, 0.0], [0.0, PHI_SP]])
        Q = np.diag([q_beta ** 2, NU ** 2])
        b_est = np.empty(T); s_est = np.empty(T)
        for t in range(T):
            th = F @ th
            P = F @ P @ F.T + Q
            H = np.array([x[t], 1.0])
            v = y[t] - H @ th
            S = H @ P @ H + 1e-10
            K = (P @ H) / S
            th = th + K * v
            P = (np.eye(2) - np.outer(K, H)) @ P
            b_est[t], s_est[t] = th
        return b_est, s_est

    x, beta, y, sp = make_pair(SEED)
    r_x, r_y = np.diff(x), np.diff(y)

    kal_beta, kal_spread = kalman_run(x, y, QB)

    # 静态 OLS（仅前 60% 训练段）
    xm, ym = x[:split].mean(), y[:split].mean()
    static_beta = ((x[:split] - xm) * (y[:split] - ym)).sum() / ((x[:split] - xm) ** 2).sum()
    static_spread = y - static_beta * x

    # 滚动 OLS（120 日窗口）
    win = 120
    roll_beta = np.full(T, static_beta)
    for t in range(win - 1, T):
        xs = x[t - win + 1:t + 1]; ys = y[t - win + 1:t + 1]
        xw = xs - xs.mean()
        roll_beta[t] = (xw * (ys - ys.mean())).sum() / (xw ** 2).sum()
    roll_spread = y - roll_beta * x
    m_roll = np.arange(T) >= win - 1

    rmse_static = float(np.sqrt(np.mean((static_beta - beta) ** 2)))
    rmse_roll = float(np.sqrt(np.mean((roll_beta[m_roll] - beta[m_roll]) ** 2)))
    rmse_kal = float(np.sqrt(np.mean((kal_beta - beta) ** 2)))
    corr_kal_s = float(np.corrcoef(kal_spread, sp)[0, 1])
    corr_roll_s = float(np.corrcoef(roll_spread[m_roll], sp[m_roll])[0, 1])
    corr_static_s = float(np.corrcoef(static_spread, sp)[0, 1])

    # q_beta 敏感性：β RMSE 与 spread 还原度的权衡
    qb_grid = [0.00025, 0.0005, 0.001, 0.002, 0.004, 0.008]
    sens_rmse, sens_corr = [], []
    for qb in qb_grid:
        bq, sq = kalman_run(x, y, qb)
        sens_rmse.append(float(np.sqrt(np.mean((bq - beta) ** 2))))
        sens_corr.append(float(np.corrcoef(sq, sp)[0, 1]))

    # ---- 交易回测：z-score 均值回复 + 波动率目标（日波动 0.5%）----
    TGT_VOL = 0.005

    def backtest(spread_series, beta_series, rx, ry, z_enter=1.5, z_exit=0.5, cost=0.0005):
        lam = 0.97
        mu, var = spread_series[0], spread_series.var()
        # 用训练段单位仓位波动初始化 EWMA 方差
        u0 = np.zeros(T - 1)
        pos = 0
        ntr, turnover = 0, 0.0
        # 第一遍：单位仓位 pnl（用于波动率目标）
        for t in range(T - 1):
            z = (spread_series[t] - mu) / np.sqrt(var)
            new_pos = pos
            if pos == 0:
                if z > z_enter:
                    new_pos = -1
                elif z < -z_enter:
                    new_pos = 1
            elif abs(z) < z_exit:
                new_pos = 0
            d = abs(new_pos - pos)
            ntr += int(d > 0)
            turnover += d
            pos = new_pos
            u0[t] = pos * (ry[t] - beta_series[t] * rx[t]) - cost * d
            mu = lam * mu + (1 - lam) * spread_series[t]
            var = lam * var + (1 - lam) * (spread_series[t] - mu) ** 2
        # 第二遍：波动率目标缩放
        v_ew = np.var(u0[:split])
        mu, var = spread_series[0], spread_series.var()
        pos = 0
        pnl = np.zeros(T - 1)
        pos_hist = np.zeros(T - 1)
        for t in range(T - 1):
            z = (spread_series[t] - mu) / np.sqrt(var)
            new_pos = pos
            if pos == 0:
                if z > z_enter:
                    new_pos = -1
                elif z < -z_enter:
                    new_pos = 1
            elif abs(z) < z_exit:
                new_pos = 0
            d = abs(new_pos - pos)
            pos = new_pos
            k = TGT_VOL / np.sqrt(max(v_ew, 1e-10))
            k = min(k, 20.0)
            pnl[t] = k * (pos * (ry[t] - beta_series[t] * rx[t]) - cost * d)
            pos_hist[t] = pos
            v_ew = lam * v_ew + (1 - lam) * pnl[t] ** 2 / max(k, 1e-12) ** 2
            mu = lam * mu + (1 - lam) * spread_series[t]
            var = lam * var + (1 - lam) * (spread_series[t] - mu) ** 2
        eq = np.cumprod(1 + pnl)
        return eq, pos_hist, pnl, ntr, turnover

    def perf(eq):
        rets = np.diff(eq) / eq[:-1]
        sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else float("nan")
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min()
        return sharpe, dd, eq[-1] - 1

    eq_kal, pos_kal, _, ntr_kal, to_kal = backtest(kal_spread, kal_beta, r_x, r_y)
    eq_roll, pos_roll, _, ntr_roll, to_roll = backtest(roll_spread, roll_beta, r_x, r_y)
    eq_static, pos_static, _, ntr_static, to_static = backtest(static_spread, np.full(T, static_beta), r_x, r_y)
    eq_oracle, pos_oracle, _, ntr_or, to_or = backtest(sp, beta, r_x, r_y)
    # 买入持有 X（同样波动率目标）
    kx = TGT_VOL / np.std(r_x[:split])
    eq_x = np.cumprod(1 + kx * r_x)

    res = {}
    for name, eq in [("Kalman", eq_kal), ("Oracle", eq_oracle), ("滚动OLS", eq_roll),
                     ("静态OLS", eq_static), ("买入持有X", eq_x)]:
        res[name] = perf(eq) + (eq,)

    # ---- 蒙特卡洛稳健性：30 条路径下四策略夏普分布 ----
    MC = 30
    mc = {"Kalman": [], "Oracle": [], "滚动OLS": [], "静态OLS": []}
    for m in range(MC):
        xs, bs, ys, sps = make_pair(90000 + m)
        kb, ks = kalman_run(xs, ys, QB)
        xm2, ym2 = xs[:split].mean(), ys[:split].mean()
        sb2 = ((xs[:split] - xm2) * (ys[:split] - ym2)).sum() / ((xs[:split] - xm2) ** 2).sum()
        rb2 = np.full(T, sb2)
        for t in range(win - 1, T):
            xw2 = xs[t - win + 1:t + 1]; yw2 = ys[t - win + 1:t + 1]
            xw2 = xw2 - xw2.mean()
            rb2[t] = (xw2 * (yw2 - yw2.mean())).sum() / (xw2 ** 2).sum()
        rx2, ry2 = np.diff(xs), np.diff(ys)
        e_kal, _, _, _, _ = backtest(ks, kb, rx2, ry2)
        e_or, _, _, _, _ = backtest(sps, bs, rx2, ry2)
        e_ro, _, _, _, _ = backtest(ys - rb2 * xs, rb2, rx2, ry2)
        e_st, _, _, _, _ = backtest(ys - sb2 * xs, np.full(T, sb2), rx2, ry2)
        mc["Kalman"].append(perf(e_kal)[0])
        mc["Oracle"].append(perf(e_or)[0])
        mc["滚动OLS"].append(perf(e_ro)[0])
        mc["静态OLS"].append(perf(e_st)[0])

    # ---------- 图1：配对价格 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(np.exp(x), color="#5B8FF9", lw=1.2, label="资产 X 价格")
    ax.plot(np.exp(y), color="#E8684A", lw=1.2, label="资产 Y 价格")
    ax.set_xlabel("交易日"); ax.set_ylabel("价格")
    ax.set_title("合成协整对：Y 与 X 的价差围绕均值回复，但对冲比例在持续漂移")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pair_prices.png")); plt.close(fig)

    # ---------- 图2：beta 追踪 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(beta, color="#333", lw=1.6, label="真实 βₜ（带漂移的随机游走）")
    ax.plot(kal_beta, color="#E8684A", lw=1.2, label=f"Kalman 估计（RMSE={rmse_kal:.4f}）")
    ax.plot(roll_beta, color="#5B8FF9", lw=1.0, alpha=0.85, label=f"120日滚动 OLS（RMSE={rmse_roll:.4f}）")
    ax.axhline(static_beta, color="#52A973", ls="--", lw=1.2,
               label=f"静态 OLS β={static_beta:.3f}（RMSE={rmse_static:.4f}）")
    ax.set_xlabel("交易日"); ax.set_ylabel("对冲比例 β")
    ax.set_title("对冲比例追踪：Kalman 紧跟漂移，OLS 类被伪回归污染")
    ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "beta_tracking.png")); plt.close(fig)

    # ---------- 图3：spread 与交易信号 ----------
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(7.6, 5.6), sharex=True,
                                  gridspec_kw={"height_ratios": [3, 1]})
    sd = kal_spread.std()
    ax.plot(sp, color="#999", lw=0.8, alpha=0.8, label="真实 spread")
    ax.plot(kal_spread, color="#5B8FF9", lw=1.0, label="Kalman 滤波 spread")
    ax.axhline(0, color="#888", ls="--", lw=0.8)
    ax.axhline(1.5 * sd, color="#E8684A", ls=":", lw=0.9)
    ax.axhline(-1.5 * sd, color="#52A973", ls=":", lw=0.9)
    ax.set_ylabel("spread"); ax.legend(loc="upper right", fontsize=9); ax.grid(alpha=0.3)
    ax.set_title(f"Kalman spread 与真实 spread 相关性 {corr_kal_s:.2f}（虚线为 ±1.5σ 入场带）")
    ax2.step(np.arange(T - 1), pos_kal, color="#E8684A", lw=1.0, where="post")
    ax2.set_yticks([-1, 0, 1]); ax2.set_yticklabels(["空 spread", "空仓", "多 spread"])
    ax2.set_xlabel("交易日")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "spread_signals.png")); plt.close(fig)

    # ---------- 图4：策略净值（波动率目标后）----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(eq_kal, color="#E8684A", lw=1.3, label=f"Kalman 动态对冲（Sharpe={res['Kalman'][0]:.2f}）")
    ax.plot(eq_oracle, color="#333", lw=1.1, ls="--", label=f"Oracle 真实 β（Sharpe={res['Oracle'][0]:.2f}）")
    ax.plot(eq_static, color="#52A973", lw=1.1, label=f"静态 OLS（Sharpe={res['静态OLS'][0]:.2f}）")
    ax.plot(eq_roll, color="#5B8FF9", lw=1.1, label=f"滚动 OLS（Sharpe={res['滚动OLS'][0]:.2f}）")
    ax.plot(eq_x, color="#999", lw=1.0, alpha=0.7, label=f"买入持有 X（Sharpe={res['买入持有X'][0]:.2f}）")
    ax.set_xlabel("交易日"); ax.set_ylabel("净值（日波动目标 0.5%）")
    ax.set_title("配对交易净值：Kalman 几乎摸到 Oracle 上界")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_curves.png")); plt.close(fig)

    # ---------- 图5：RMSE + q_beta 敏感性 ----------
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(7.8, 3.8))
    names = ["静态 OLS", "滚动 OLS", "Kalman"]
    vals = [rmse_static, rmse_roll, rmse_kal]
    colors = ["#52A973", "#5B8FF9", "#E8684A"]
    bars = ax.bar(names, vals, color=colors, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("β 估计 RMSE"); ax.set_title("对冲比例估计误差")
    ax.grid(alpha=0.3, axis="y")
    ax2.plot(qb_grid, sens_rmse, color="#E8684A", marker="o", lw=1.4, label="β RMSE（左轴）")
    ax2.set_xlabel("Kalman 过程噪声 q_β"); ax2.set_ylabel("β RMSE", color="#E8684A")
    ax2.set_xscale("log")
    ax2.set_xticks(qb_grid)
    ax2.set_xticklabels([f"{q:g}" for q in qb_grid])
    ax2.grid(alpha=0.3)
    ax2b = ax2.twinx()
    ax2b.plot(qb_grid, sens_corr, color="#5B8FF9", marker="s", lw=1.4, label="spread 还原相关性（右轴）")
    ax2b.set_ylabel("corr(spread, 真实)", color="#5B8FF9")
    ax2.set_title("q_β 敏感性：估计精度与价差还原的权衡")
    l1, lab1 = ax2.get_legend_handles_labels()
    l2, lab2 = ax2b.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, fontsize=8, loc="center right")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "rmse_sensitivity.png")); plt.close(fig)

    # ---------- 图6：蒙特卡洛稳健性 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    order = ["Kalman", "Oracle", "滚动OLS", "静态OLS"]
    data = [mc[k] for k in order]
    bp = ax.boxplot(data, positions=[1, 2, 3, 4], patch_artist=True, widths=0.5, showmeans=True)
    ax.set_xticks([1, 2, 3, 4]); ax.set_xticklabels(order)
    for patch, c in zip(bp["boxes"], ["#E8684A", "#333", "#5B8FF9", "#52A973"]):
        patch.set_facecolor(c); patch.set_alpha(0.75)
    for i, k in enumerate(order):
        ax.scatter([i + 1] * len(mc[k]), mc[k], color="#333", s=12, alpha=0.5, zorder=3)
        ax.text(i + 1, np.mean(mc[k]) + 0.12, f"均值 {np.mean(mc[k]):.2f}",
                ha="center", fontsize=9)
    ax.set_ylabel("夏普比率（30 条独立路径）")
    ax.set_title("稳健性：Kalman 的夏普稳定贴近 Oracle，OLS 类离散且常为负")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "monte_carlo_sharpe.png")); plt.close(fig)

    print("=" * 64)
    print("ARTICLE_A_KALMAN_METRICS")
    print(f"beta: start={beta[0]:.3f} end={beta[-1]:.3f} drift={BETA_DRIFT}/day noise={BETA_NOISE}")
    print(f"spread: phi={PHI_SP} nu={NU} halflife_true={-np.log(2)/np.log(PHI_SP):.1f}d std={sp.std():.4f}")
    print(f"static_ols beta={static_beta:.4f}")
    print(f"rmse: static={rmse_static:.4f} rolling={rmse_roll:.4f} kalman={rmse_kal:.4f}")
    print(f"corr(spread_est, true): kalman={corr_kal_s:.3f} rolling={corr_roll_s:.3f} static={corr_static_s:.3f}")
    print("q_beta sensitivity:")
    for qb, r_, c_ in zip(qb_grid, sens_rmse, sens_corr):
        print(f"  qb={qb:.5f}: rmse={r_:.4f} corr={c_:.3f}")
    for name in ["Kalman", "Oracle", "滚动OLS", "静态OLS", "买入持有X"]:
        sh, dd, tot, eq = res[name]
        print(f"strategy {name:8s}: total={tot:+.2%} sharpe={sh:.2f} maxDD={dd:.2%}")
    print(f"trades: kalman={ntr_kal} oracle={ntr_or} rolling={ntr_roll} static={ntr_static}")
    print(f"MC(30 paths) sharpe: " + " ".join(f"{k}={np.mean(v):.2f}(std {np.std(v):.2f})" for k, v in mc.items()))
    print("=" * 64)


# ============================================================
# 文章 B：图神经网络板块关联
# ============================================================
def gen_gnn():
    slug = "gnn-sector-correlation-network"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    # ---- 20 个行业，5 个板块 × 4 ----
    sectors = ["银行", "证券", "保险", "多元金融",
               "白酒", "食品饮料", "农业", "商贸零售",
               "电力", "燃气", "环保", "水务",
               "半导体", "软件", "消费电子", "通信",
               "新能源车", "光伏", "储能", "电网设备"]
    block_names = ["金融", "消费", "公用事业", "科技", "新能源"]
    blocks = np.repeat(np.arange(5), 4)
    N = len(sectors)
    T, split = 3000, 2100
    rg = np.random.default_rng(20261025)

    # ---- 数据生成：5 个板块因子（单日特征 + 邻居聚合才有真实增益）----
    nF = 5
    PHI_F, SIG_F = 0.30, 0.012      # 因子日度持续性
    LOAD_IN, LOAD_OUT = 1.5, 0.15   # 板块内/跨板块载荷
    SIG_IDIO = 0.025
    B = np.zeros((N, nF))
    for i in range(N):
        B[i, blocks[i]] = LOAD_IN
    B += LOAD_OUT * rg.random((N, nF))
    F = np.zeros((T, nF))
    for t in range(1, T):
        F[t] = PHI_F * F[t - 1] + SIG_F * rg.normal(size=nF)
    E = SIG_IDIO * rg.normal(size=(T, N))
    R = F @ B.T + E

    R_train, R_test = R[:split], R[split:]
    T_test = T - split

    # ---- 特征：昨日收益（单日窗口，制造「自身信息弱、邻居聚合强」的对比）----
    L = 1
    X_all = R[0:-1][:, :, None]                                        # (T-1, N, 1)
    Y_all = R[1:]                                                      # (T-1, N)
    n_train = split - 1
    X_train, Y_train = X_all[:n_train], Y_all[:n_train]
    X_test, Y_test = X_all[n_train:], Y_all[n_train:]
    mu_f, sd_f = X_train.mean(), X_train.std()
    X_train = (X_train - mu_f) / sd_f
    X_test = (X_test - mu_f) / sd_f
    X_all_std = (X_all - mu_f) / sd_f

    # ---- 相关性图（训练集，|corr|>0.25 连边）----
    C = np.corrcoef(R_train.T)
    A = np.where(np.abs(C) > 0.25, C, 0.0)
    np.fill_diagonal(A, 0.0)
    A = A + np.eye(N)
    Dm = np.diag(1.0 / np.sqrt(A.sum(1)))
    A_norm = Dm @ A @ Dm
    n_edges = int((np.abs(C) > 0.25).sum() // 2)

    # ---- 模型 ----
    def relu(z):
        return np.maximum(z, 0)

    rng = np.random.default_rng(20261026)

    def build_gnn(A_mat, seed):
        r2 = np.random.default_rng(seed)
        W1g = r2.normal(0, np.sqrt(2 / L), (L, 16)); b1g = np.zeros(16)
        W2g = r2.normal(0, np.sqrt(2 / 16), (16, 8)); b2g = np.zeros(8)
        w3g = r2.normal(0, 0.1, (8, 1))

        def fwd(X):
            Z1 = np.einsum("ij,sjl->sil", A_mat, X) @ W1g + b1g
            H1 = relu(Z1)
            Z2 = np.einsum("ij,sjf->sif", A_mat, H1) @ W2g + b2g
            H2 = relu(Z2)
            return (H2 @ w3g).squeeze(-1)

        return fwd, [W1g, b1g, W2g, b2g, w3g]

    gnn_forward, gnn_params = build_gnn(A_norm, 20261026)
    # 消融：同一架构但把图换成单位阵（只有自环，无邻居传播）
    nograph_forward, nograph_params = build_gnn(np.eye(N), 20261027)

    M_W1 = rng.normal(0, np.sqrt(2 / L), (L, 16)); M_b1 = np.zeros(16)
    M_w2 = rng.normal(0, 0.1, (16, 1))

    def mlp_forward(X):
        return (relu(X @ M_W1 + M_b1) @ M_w2).squeeze(-1)

    Xtr_flat = X_train.reshape(-1, L)
    ytr_flat = Y_train.reshape(-1)
    ridge_w = np.linalg.solve(Xtr_flat.T @ Xtr_flat + 1.0 * np.eye(L), Xtr_flat.T @ ytr_flat)

    def ridge_forward(X):
        return X @ ridge_w

    def fama_forward(X):   # 因子动量：昨日收益
        return X.mean(-1)

    models = {"GNN": gnn_forward, "无图GNN": nograph_forward, "MLP": mlp_forward, "Ridge": ridge_forward}

    lr, b1a, b2a, eps, epochs, bs = 0.01, 0.9, 0.999, 1e-8, 400, 256
    n_batches = int(np.ceil(len(X_train) / bs))

    def train_gnn(A_mat, params):
        W1, b1, W2, b2, w3 = params
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for ep in range(epochs):
            perm = rng.permutation(len(X_train))
            for bi in range(n_batches):
                idx = perm[bi * bs:(bi + 1) * bs]
                xb, yb = X_train[idx], Y_train[idx]
                S = xb.shape[0]
                Z1 = np.einsum("ij,sjl->sil", A_mat, xb) @ W1 + b1
                H1 = relu(Z1)
                Z2 = np.einsum("ij,sjf->sif", A_mat, H1) @ W2 + b2
                H2 = relu(Z2)
                yhat = (H2 @ w3).squeeze(-1)
                g = 2 * (yhat - yb) / (S * N)
                gw3 = np.einsum("sn,snh->h", g, H2).reshape(-1, 1)
                gZ2 = g[:, :, None] * w3[:, 0][None, None, :] * (Z2 > 0)
                gW2 = np.einsum("ij,sih,sjf->fh", A_mat, gZ2, H1)
                gb2 = gZ2.sum((0, 1))
                gH1 = np.einsum("sjh,fh->sjf", np.einsum("ij,sih->sjh", A_mat, gZ2), W2) * (Z1 > 0)
                gW1 = np.einsum("ij,sif,sjl->lf", A_mat, gH1, xb)
                gb1 = gH1.sum((0, 1))
                grads = [gW1, gb1, gW2, gb2, gw3]
                for i, (p, gp) in enumerate(zip(params, grads)):
                    m[i] = b1a * m[i] + (1 - b1a) * gp
                    v[i] = b2a * v[i] + (1 - b2a) * gp ** 2
                    mh = m[i] / (1 - b1a ** (ep + 1))
                    vh = v[i] / (1 - b2a ** (ep + 1))
                    p -= lr * mh / (np.sqrt(vh) + eps)

    train_gnn(A_norm, gnn_params)
    train_gnn(np.eye(N), nograph_params)

    def train_mlp():
        params = [M_W1, M_b1, M_w2]
        m = [np.zeros_like(p) for p in params]
        v = [np.zeros_like(p) for p in params]
        for ep in range(epochs):
            perm = rng.permutation(len(X_train))
            for bi in range(n_batches):
                idx = perm[bi * bs:(bi + 1) * bs]
                xb, yb = X_train[idx], Y_train[idx]
                S = xb.shape[0]
                Z1 = xb @ M_W1 + M_b1
                H1 = relu(Z1)
                yhat = (H1 @ M_w2).squeeze(-1)
                g = 2 * (yhat - yb) / (S * N)
                gw2 = np.einsum("sn,snh->h", g, H1).reshape(-1, 1)
                gZ1 = g[:, :, None] * M_w2[:, 0][None, None, :] * (Z1 > 0)
                gW1 = np.einsum("snf,snl->lf", gZ1, xb)
                gb1 = gZ1.sum((0, 1))
                grads = [gW1, gb1, gw2]
                for i, (p, gp) in enumerate(zip(params, grads)):
                    m[i] = b1a * m[i] + (1 - b1a) * gp
                    v[i] = b2a * v[i] + (1 - b2a) * gp ** 2
                    mh = m[i] / (1 - b1a ** (ep + 1))
                    vh = v[i] / (1 - b2a ** (ep + 1))
                    p -= lr * mh / (np.sqrt(vh) + eps)

    train_mlp()

    def rank_ic_series(pred, actual):
        ics = []
        for t in range(pred.shape[0]):
            rp = np.argsort(np.argsort(pred[t])).astype(float)
            ra = np.argsort(np.argsort(actual[t])).astype(float)
            rp -= rp.mean(); ra -= ra.mean()
            ics.append((rp * ra).sum() / np.sqrt((rp ** 2).sum() * (ra ** 2).sum()))
        return np.array(ics)

    preds, metrics = {}, {}
    for name, fn in models.items():
        p = fn(X_test)
        preds[name] = p
        r2 = 1 - np.sum((p - Y_test) ** 2) / np.sum((Y_test - Y_test.mean()) ** 2)
        metrics[name] = (r2, float(rank_ic_series(p, Y_test).mean()))

    # ---- 多期限 IC 衰减 ----
    horizons = [1, 5, 10, 20]
    ic_decay = {name: [] for name in models}
    for h in horizons:
        fwd = np.stack([R_test[t:t + h].sum(0) for t in range(T_test - h + 1)])
        for name, fn in models.items():
            p = fn(X_test)[:len(fwd)]
            ic_decay[name].append(float(rank_ic_series(p, fwd).mean()))

    # ---- 十分之一多空组合（波动率目标：日波动 0.5%）----
    TGT_VOL = 0.005
    k = N // 10
    ls = {}
    ls_train_vol = {}
    for name, p in preds.items():
        def ls_daily(pred, actual):
            d = np.zeros(len(pred))
            for t in range(len(pred)):
                order = np.argsort(pred[t])
                d[t] = actual[t, order[-k:]].mean() - actual[t, order[:k]].mean()
            return d
        d_train = ls_daily(models[name](X_train), Y_train)
        ls_train_vol[name] = d_train.std()
        daily = ls_daily(p, Y_test)
        scale = TGT_VOL / ls_train_vol[name] if ls_train_vol[name] > 0 else 1.0
        eq = np.cumprod(1 + scale * daily)
        sharpe = daily.mean() / daily.std() * np.sqrt(252)
        peak = np.maximum.accumulate(eq)
        dd = ((eq - peak) / peak).min()
        ls[name] = (eq, sharpe, dd, eq[-1] - 1, scale)

    C_full = np.corrcoef(R.T)
    block_colors = ["#E8684A", "#5B8FF9", "#52A973", "#9B6DD1", "#E8A33D"]

    # ---------- 图1：相关性网络 ----------
    ang0 = {0: 90, 1: 18, 2: -54, 3: -126, 4: 162}
    pos = {}
    for i in range(N):
        a = np.deg2rad(ang0[blocks[i]] + (i % 4) * 18 - 27)
        pos[i] = (np.cos(a), np.sin(a))
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for i in range(N):
        for j in range(i + 1, N):
            w = C[i, j]
            if abs(w) > 0.25:
                ax.plot([pos[i][0], pos[j][0]], [pos[i][1], pos[j][1]],
                        color="#888", lw=1.6 * abs(w), alpha=0.35, zorder=1)
    for i in range(N):
        x0, y0 = pos[i]
        ax.scatter(x0, y0, s=420, color=block_colors[blocks[i]], zorder=3,
                   edgecolors="white", linewidths=1.5)
        ax.text(x0, y0, sectors[i], ha="center", va="center", fontsize=9.5,
                color="white", zorder=4, weight="bold")
    for b in range(5):
        a = np.deg2rad(ang0[b])
        ax.text(1.30 * np.cos(a), 1.30 * np.sin(a), block_names[b], ha="center",
                va="center", fontsize=11, color=block_colors[b], weight="bold")
    ax.set_xlim(-1.6, 1.6); ax.set_ylim(-1.6, 1.6); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"行业相关性网络：{n_edges} 条边，板块内稠密、板块间稀疏（|ρ|>0.25 连边）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "correlation_network.png")); plt.close(fig)

    # ---------- 图2：图传播去噪 ----------
    k_f = 0   # 金融板块因子
    own = X_test[:, 0, 0]                       # 单行业（银行）昨日收益
    Wc = np.where(np.abs(C_full) > 0.25, C_full, 0.0)
    Wc /= Wc.sum(1, keepdims=True)
    f_true = F[split - 1: split - 1 + T_test, k_f]   # 与 X_test 特征日对齐的当期因子
    graph_agg = np.einsum("ij,sj->si", Wc, X_test[:, :, 0])
    f_graph = graph_agg[:, blocks == k_f].mean(-1)

    def corr(a, b):
        a = a - a.mean(); b = b - b.mean()
        return float((a * b).sum() / np.sqrt((a ** 2).sum() * (b ** 2).sum()))

    c_own, c_graph = corr(own, f_true), corr(f_graph, f_true)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    tt = np.arange(len(f_true))
    ax.plot(tt, f_true / f_true.std(), color="#333", lw=1.4, label="真实因子 1（金融）")
    ax.plot(tt, own / own.std(), color="#999", lw=0.9, alpha=0.85,
            label=f"单行业昨日收益（corr={c_own:.2f}）")
    ax.plot(tt, f_graph / f_graph.std(), color="#E8684A", lw=1.1,
            label=f"图传播聚合估计（corr={c_graph:.2f}）")
    ax.set_xlabel("测试集交易日"); ax.set_ylabel("因子（标准化）")
    ax.set_title("图传播的本质：邻居信号加权聚合，把特异噪声平均掉")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "factor_denoising.png")); plt.close(fig)

    # ---------- 图3：IC 衰减 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    style = {"GNN": ("#E8684A", "o", 2.0), "无图GNN": ("#9B6DD1", "v", 1.4),
             "MLP": ("#5B8FF9", "s", 1.4), "Ridge": ("#52A973", "^", 1.4)}
    for name, ics in ic_decay.items():
        c, mk, lw = style[name]
        ax.plot(horizons, ics, color=c, marker=mk, lw=lw, ms=5, label=name)
    ax.set_xlabel("预测期限（天）"); ax.set_ylabel("平均截面 rank-IC")
    ax.set_title("IC 衰减曲线：GNN 在短周期占优，优势随期限收敛")
    ax.set_xticks(horizons); ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "ic_decay.png")); plt.close(fig)

    # ---------- 图4：多空组合净值 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    for name, (eq, sharpe, dd, tot, scale) in ls.items():
        ax.plot(eq, color=style[name][0], lw=1.3, label=f"{name}（Sharpe={sharpe:.2f}）")
    ax.set_xlabel("测试集交易日"); ax.set_ylabel("净值（十分之一多空，日波动目标 0.5%）")
    ax.set_title("截面十分之一多空组合：信息优势能否变成真金白银")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "longshort_equity.png")); plt.close(fig)

    # ---------- 图5：图传播消融实验 ----------
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    abl_order = ["GNN", "无图GNN", "Ridge", "MLP"]
    abl_vals = [metrics[n][1] for n in abl_order]
    abl_colors = ["#E8684A", "#9B6DD1", "#52A973", "#5B8FF9"]
    bars = ax.bar(abl_order, abl_vals, color=abl_colors, alpha=0.85, width=0.55)
    for b, v in zip(bars, abl_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.4f}",
                ha="center", va="bottom", fontsize=10)
    uplift = (abl_vals[0] / abl_vals[2] - 1) * 100
    ax.annotate(f"图传播带来 +{uplift:.0f}% IC", xy=(0, abl_vals[0]), xytext=(0.55, abl_vals[0] * 1.02),
                fontsize=10, color="#E8684A",
                arrowprops=dict(arrowstyle="->", color="#E8684A"))
    ax.set_ylabel("测试集 1 日截面 rank-IC")
    ax.set_title("消融实验：拿掉图传播，GNN 退化到 Ridge 水平——增量全部来自邻居聚合")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "graph_ablation.png")); plt.close(fig)

    print("=" * 64)
    print("ARTICLE_B_GNN_METRICS")
    print(f"N={N} nF={nF} T={T} train={split} test={T_test} edges={n_edges}")
    print(f"factor: phi={PHI_F} sigma={SIG_F} load_in={LOAD_IN} load_out={LOAD_OUT} idio={SIG_IDIO}")
    within = [C_full[i, j] for i in range(N) for j in range(i + 1, N) if blocks[i] == blocks[j]]
    cross = [C_full[i, j] for i in range(N) for j in range(i + 1, N) if blocks[i] != blocks[j]]
    print(f"corr: within-block={np.mean(within):.3f} cross-block={np.mean(cross):.3f}")
    for name in models:
        r2, ic = metrics[name]
        print(f"1d test: {name:6s} R2={r2:.4f} rank-IC={ic:.4f}")
    for name in models:
        print(f"IC decay {name:6s}: " + " ".join(f"h{h}={v:.4f}" for h, v in zip(horizons, ic_decay[name])))
    for name in models:
        eq, sharpe, dd, tot, scale = ls[name]
        print(f"L/S 1/10: {name:6s} sharpe={sharpe:.2f} maxDD={dd:.2%} total={tot:.2%} scale={scale:.2f}")
    print(f"factor denoise corr: own={c_own:.3f} graph={c_graph:.3f}")
    print("=" * 64)


if __name__ == "__main__":
    gen_kalman()
    gen_gnn()
    print("ALL DONE")
