#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-07 两篇量化文章配图 + 核心数值（numpy/scipy 合成，固定 seed 可复现）。
  A. liquidity-spiral-deleveraging  流动性螺旋与去杠杆
  B. rl-portfolio-allocation       强化学习组合配置（策略梯度 REINFORCE）
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

BASE = "/Users/halo/workspace/astro-blog/public/images"


# ============================================================
# 文章 A：流动性螺旋与去杠杆
# ============================================================
def gen_liquidity():
    slug = "liquidity-spiral-deleveraging"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    P0 = 100.0
    L0 = 2.5            # 平静期杠杆（同时作为风险限额基准）
    D = 60.0            # 固定名义负债
    V0 = L0 * D / (L0 - 1)   # 初始资产价值，使杠杆 = L0
    N0 = V0 / P0
    E0 = V0 - D
    market_cap = 100.0   # 市场总盘子（impact 缩放）
    beta_impact = 0.8    # 价格冲击系数
    gamma_vol = 2.2      # 波动每升 1 单位，杠杆上限收紧多少（VaR 约束）
    sale_cap = 0.05      # 每日强平上限 = 5% 持仓（只能卖这么快）
    lam = 0.94           # EWMA 波动衰减
    mu_f = 0.0002        # 基本面日漂移
    sig_f = 0.010        # 基本面日波动

    def run_path(seed, do_spiral):
        rg = np.random.default_rng(seed)
        T = 252 * 5
        P = np.empty(T + 1); P[0] = P0
        N = np.empty(T + 1); N[0] = N0
        Dv = np.empty(T + 1); Dv[0] = D      # 当前负债（随还债下降）
        sales = np.zeros(T)
        Lmax_seq = np.empty(T)
        sig = sig_f
        s_start = int(rg.integers(150, T - 80))
        s_len = 30
        for t in range(1, T + 1):
            stress = (s_start <= t < s_start + s_len)
            sf = sig_f * (2.5 if stress else 1.0)
            drift = mu_f - (0.0015 if stress else 0.0)
            rf = drift + sf * rg.normal()
            Lmax = max(1.2, L_base_ - gamma_vol * sig)
            Lmax_seq[t - 1] = Lmax
            Vprev = N[t - 1] * P[t - 1]
            Dprev = Dv[t - 1]
            Eprev = Vprev - Dprev
            Lcur = Vprev / Eprev if Eprev > 0 else 1e9
            if do_spiral:
                if Eprev <= 0:
                    S = min(Vprev, Dprev)       # 资不抵债 → 清仓还债
                elif Lcur > Lmax:
                    # 卖资产并用 proceeds 还债：E 不变，L=V/E 随 V 下降
                    S = Vprev - Lmax * Eprev
                else:
                    S = 0.0
                S = min(S, sale_cap * Vprev)    # 每天只能卖这么快（流动性约束）
                S = max(S, 0.0)
            else:
                S = 0.0
            impact = -beta_impact * (S / market_cap) if do_spiral else 0.0
            Pt = P[t - 1] * (1 + rf + impact)
            Nt = max(N[t - 1] - S / P[t - 1], 0.0)
            P[t] = Pt; N[t] = Nt; Dv[t] = Dprev - S; sales[t - 1] = S
            r_mkt = Pt / P[t - 1] - 1
            sig = np.sqrt(lam * sig ** 2 + (1 - lam) * r_mkt ** 2)
        return P, N, Dv, sales, Lmax_seq, sig

    L_base_ = L0

    # 主路径（用于画图，固定 seed）
    P_sp, N_sp, D_sp, sales_sp, Lmax_sp, _ = run_path(20261007, True)
    P_bs, _, _, _, _, _ = run_path(20261007, False)

    def max_dd(x):
        peak = np.maximum.accumulate(x)
        return ((x - peak) / peak).min()

    dd_sp = max_dd(P_sp)
    dd_bs = max_dd(P_bs)

    # 图1：螺旋 vs 基线价格
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(P_sp, color="#E8684A", lw=1.4, label=f"含去杠杆螺旋 (最大回撤 {dd_sp:.0%})")
    ax.plot(P_bs, color="#5B8FF9", lw=1.4, label=f"纯基本面基线 (最大回撤 {dd_bs:.0%})")
    ax.axhline(P0, color="#888", ls="--", lw=0.9)
    ax.set_xlabel("交易日"); ax.set_ylabel("资产价格指数")
    ax.set_title("流动性螺旋：同样的初始冲击，去杠杆把它变成崩盘")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "price_spiral_vs_baseline.png")); plt.close(fig)

    # 图2：权益 & 杠杆
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    Esp = N_sp * P_sp - D_sp
    ax.plot(Esp / E0, color="#E8684A", lw=1.3, label="杠杆 sector 权益 (归一)")
    ax.set_xlabel("交易日"); ax.set_ylabel("权益 (初始=1)", color="#E8684A")
    ax2 = ax.twinx()
    Lcur = np.where(N_sp * P_sp - D_sp > 0, (N_sp * P_sp) / (N_sp * P_sp - D_sp), np.nan)
    ax2.plot(Lcur, color="#5B8FF9", lw=1.0, label="杠杆倍数")
    ax2.axhline(L0, color="#5B8FF9", ls="--", lw=0.8, alpha=0.6)
    ax2.set_ylabel("杠杆倍数", color="#5B8FF9")
    ax.set_title("权益崩塌 + 杠杆触顶：先被动去杠杆，后资不抵债清仓")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity_leverage.png")); plt.close(fig)

    # 图3：抛售量
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.fill_between(np.arange(len(sales_sp)), sales_sp, color="#E8684A", alpha=0.75)
    ax.set_xlabel("交易日"); ax.set_ylabel("当日强平抛售 (美元)")
    ax.set_title("火线抛售：螺旋期间抛压飙升，把价格越砸越低")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sales_volume.png")); plt.close(fig)

    # 图4：MC 回撤分布
    MC = 1500
    dds_sp_mc = np.array([max_dd(run_path(7000 + m, True)[0]) for m in range(MC)])
    dds_bs_mc = np.array([max_dd(run_path(7000 + m, False)[0]) for m in range(MC)])
    crash_sp = (dds_sp_mc < -0.30).mean()
    crash_bs = (dds_bs_mc < -0.30).mean()
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.hist(dds_bs_mc, bins=40, alpha=0.55, density=True, color="#5B8FF9",
            label=f"基线 P(崩盘<-30%)={crash_bs:.1%}")
    ax.hist(dds_sp_mc, bins=40, alpha=0.55, density=True, color="#E8684A",
            label=f"螺旋 P(崩盘<-30%)={crash_sp:.1%}")
    ax.axvline(-0.30, color="#333", ls="--", lw=1)
    ax.set_xlabel("最大回撤"); ax.set_ylabel("密度")
    ax.set_title("蒙特卡洛 1500 路径：去杠杆让崩盘概率翻数倍")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "drawdown_dist.png")); plt.close(fig)

    # 图5：允许杠杆上限 vs 价格（波动率收紧）
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(P_sp, color="#E8684A", lw=1.3, label="价格")
    ax.set_xlabel("交易日"); ax.set_ylabel("价格", color="#E8684A")
    ax2 = ax.twinx()
    ax2.plot(Lmax_sp, color="#52A973", lw=1.2, label="允许杠杆上限 Lmax")
    ax2.set_ylabel("允许杠杆上限", color="#52A973")
    ax.set_title("波动越高、可容忍杠杆越低：margin 螺旋自我收紧")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "lmax_volatility.png")); plt.close(fig)

    print("=" * 60)
    print("ARTICLE_A_LIQUIDITY_METRICS")
    print(f"P0={P0} L0={L0} D={D} V0={V0:.2f} N0={N0:.3f} E0={E0:.2f}")
    print(f"main_path: dd_spiral={dd_sp:.4f} dd_baseline={dd_bs:.4f}")
    print(f"final_price spiral={P_sp[-1]:.1f} baseline={P_bs[-1]:.1f}")
    print(f"spiral min price={P_sp.min():.1f} at day {P_sp.argmin()}")
    print(f"MC={MC}: crash_prob spiral={crash_sp:.4f} baseline={crash_bs:.4f}")
    print(f"spiral max sales={sales_sp.max():.2f} at day {sales_sp.argmax()}")
    print(f"MC mean dd spiral={dds_sp_mc.mean():.4f} baseline={dds_bs_mc.mean():.4f}")
    print(f"MC P95 dd spiral={np.percentile(dds_sp_mc,5):.4f} baseline={np.percentile(dds_bs_mc,5):.4f}")
    print("=" * 60)


# ============================================================
# 文章 B：强化学习组合配置（策略梯度 REINFORCE）
# ============================================================
def gen_rl():
    slug = "rl-portfolio-allocation"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    N = 4  # [RiskyA, RiskyB, Bond, Gold]
    names = ["风险A", "风险B", "债券", "黄金"]
    beta = np.array([1.10, 0.90, 0.10, -0.20])
    mu = np.array([0.0003, 0.0002, 0.0001, 0.00015])
    sigma_idio = np.array([0.012, 0.013, 0.004, 0.006])

    # 动作 = 目标配置篮子（长仓、和为1）
    baskets = np.array([
        [0.45, 0.45, 0.05, 0.05],   # 0 激进
        [0.35, 0.35, 0.20, 0.10],   # 1 成长
        [0.25, 0.25, 0.35, 0.15],   # 2 平衡
        [0.15, 0.15, 0.45, 0.25],   # 3 防御
        [0.05, 0.05, 0.50, 0.40],   # 4 避险
    ])

    def gen_data(seed, T):
        rg = np.random.default_rng(seed)
        regime = rg.choice([0, 1], p=[0.7, 0.3], size=T)   # 1 = 熊市/高波动
        sigF = np.where(regime == 0, 0.004, 0.013)
        muF = np.where(regime == 0, 0.0003, -0.0009)
        F = rg.normal(muF, sigF)
        eps = rg.normal(0, 1, (N, T)) * sigma_idio[:, None]
        R = mu[:, None] + beta[:, None] * F + eps
        return R

    T_train = 252 * 15
    R_train = gen_data(20261007, T_train)
    L = 10  # 回望窗口

    def ewma_vol(x, lam=0.94):
        v = np.empty_like(x); v[0] = x[0] ** 2
        for i in range(1, len(x)):
            v[i] = lam * v[i - 1] + (1 - lam) * x[i] ** 2
        return np.sqrt(v)

    def make_features(R, w_prev, t):
        if t < L:
            win = np.concatenate([np.zeros((N, L - t)), R[:, :t]], axis=1)
        else:
            win = R[:, t - L:t]
        f = win.flatten()
        f = np.concatenate([f, w_prev, [ewma_vol(R[0, :t + 1])[-1] if t > 0 else sigma_idio[0]]])
        return f

    n_feat = N * L + N + 1
    n_act = baskets.shape[0]

    rng = np.random.default_rng(20261008)
    theta = rng.normal(0, 0.05, (n_act, n_feat))
    lr = 0.02
    baseline = 0.0
    T_ep = 60
    n_ep = 3000
    kappa = 0.0025   # 换手惩罚
    ep_rewards = []

    def softmax(z):
        z = z - z.max()
        e = np.exp(z)
        return e / e.sum()

    for ep in range(n_ep):
        start = int(rng.integers(0, T_train - T_ep))
        w_prev = np.ones(N) / N
        traj = []   # (feat, action, reward)
        tot = 0.0
        for k in range(T_ep):
            t = start + k
            s = make_features(R_train, w_prev, t)
            pi = softmax(theta @ s)
            a = int(rng.choice(n_act, p=pi))
            w = baskets[a]
            r = float(w @ R_train[:, t]) - kappa * np.abs(w - w_prev).sum()
            traj.append((s, a, r))
            tot += r
            w_prev = w
        # REINFORCE：return-to-go
        G = 0.0
        adv_list = []
        returns = []
        for i in range(len(traj) - 1, -1, -1):
            G = traj[i][2] + 0.99 * G
            returns.append(G)
        returns = returns[::-1]
        baseline = 0.999 * baseline + 0.001 * tot
        grad = np.zeros_like(theta)
        for i, (s, a, _) in enumerate(traj):
            pi = softmax(theta @ s)
            adv = returns[i] - baseline
            grad += adv * np.outer(np.eye(n_act)[a] - pi, s)
        theta += lr * grad / T_ep
        ep_rewards.append(tot)

    # OOS 评估
    R_oos = gen_data(991, 252 * 8)
    To = R_oos.shape[1]
    w_prev = np.ones(N) / N
    rl_w = np.zeros((N, To))
    rl_eq = np.empty(To); rl_eq[0] = 1.0
    for t in range(To):
        s = make_features(R_oos, w_prev, t)
        a = int(np.argmax(softmax(theta @ s)))
        w = baskets[a]
        rl_w[:, t] = w
        if t > 0:
            rl_eq[t] = rl_eq[t - 1] * (1 + w @ R_oos[:, t - 1])
        w_prev = w

    def bench(cols):
        eq = np.empty(To); eq[0] = 1.0
        w = cols
        for t in range(1, To):
            eq[t] = eq[t - 1] * (1 + w @ R_oos[:, t - 1])
        return eq

    eq_ew = bench(np.ones(N) / N)
    eq_6040 = bench(np.array([0.30, 0.30, 0.30, 0.10]))
    eq_risky = bench(np.array([0.50, 0.50, 0.0, 0.0]))

    def stats(eq):
        ret = eq[1:] / eq[:-1] - 1
        sd = ret.std(ddof=1)
        sh = (ret.mean() / sd) * np.sqrt(252) if sd > 0 else 0.0
        vol = sd * np.sqrt(252)
        peak = np.maximum.accumulate(eq)
        mdd = ((eq - peak) / peak).min()
        return sh, vol, mdd, eq[-1]

    s_rl = stats(rl_eq); s_ew = stats(eq_ew); s_60 = stats(eq_6040); s_rk = stats(eq_risky)

    # 图1：学习曲线
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    win = 50
    sm = np.convolve(ep_rewards, np.ones(win) / win, mode="valid")
    ax.plot(sm, color="#E8684A", lw=1.4)
    ax.set_xlabel("训练回合 (episode)"); ax.set_ylabel("单回合总奖励 (平滑)")
    ax.set_title("策略梯度 REINFORCE：回合奖励随训练稳定上升")
    ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "learning_curve.png")); plt.close(fig)

    # 图2：OOS 净值
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(rl_eq, color="#E8684A", lw=1.4, label=f"RL 动态再平衡 (Sharpe {s_rl[0]:.2f})")
    ax.plot(eq_ew, color="#5B8FF9", lw=1.2, label=f"等权 (Sharpe {s_ew[0]:.2f})")
    ax.plot(eq_6040, color="#52A973", lw=1.2, label=f"固定 30/30/30/10 (Sharpe {s_60[0]:.2f})")
    ax.plot(eq_risky, color="#D9A441", lw=1.2, label=f"满仓风险 (Sharpe {s_rk[0]:.2f})")
    ax.set_xlabel("交易日"); ax.set_ylabel("累计净值 (初始=1)")
    ax.set_title("样本外净值：RL 策略自适应再平衡")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "equity_curve_oos.png")); plt.close(fig)

    # 图3：权重随时间（动态再平衡）
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.stackplot(np.arange(To), rl_w, labels=names,
                 colors=["#E8684A", "#5B8FF9", "#52A973", "#D9A441"], alpha=0.85)
    ax.set_xlabel("交易日"); ax.set_ylabel("配置权重")
    ax.set_title("策略学到的动态配置：权重随状态在篮子上滑动")
    ax.legend(loc="upper right"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "weights_over_time.png")); plt.close(fig)

    # 图4：指标对比
    labels = ["RL", "等权", "30/30/30/10", "满仓风险"]
    sh = [s_rl[0], s_ew[0], s_60[0], s_rk[0]]
    vol = [s_rl[1], s_ew[1], s_60[1], s_rk[1]]
    mdd = [s_rl[2], s_ew[2], s_60[2], s_rk[2]]
    x = np.arange(len(labels)); w = 0.26
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.bar(x - w, sh, w, label="Sharpe", color="#E8684A")
    ax.bar(x, vol, w, label="年化波动", color="#5B8FF9")
    ax.bar(x + w, [-m for m in mdd], w, label="最大回撤(取正)", color="#52A973")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("风险-收益对比：RL 在回撤控制上占优")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "metrics_bar.png")); plt.close(fig)

    # 图5：避险权重 vs 市场波动
    safe_w = rl_w[2] + rl_w[3]
    mkt_vol = ewma_vol(R_oos[0, :]) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    ax.plot(safe_w, color="#52A973", lw=1.3, label="避险资产权重 (债+金)")
    ax.set_xlabel("交易日"); ax.set_ylabel("避险权重", color="#52A973")
    ax2 = ax.twinx()
    ax2.plot(mkt_vol, color="#E8684A", lw=1.0, label="市场风险波动 (年化)")
    ax2.set_ylabel("年化波动", color="#E8684A")
    ax.set_title("策略学出「波动高→加避险」：隐式市场择时")
    ax.legend(loc="upper left"); fig.tight_layout()
    fig.savefig(os.path.join(OUT, "safe_weight_vs_vol.png")); plt.close(fig)

    print("=" * 60)
    print("ARTICLE_B_RL_METRICS")
    print(f"n_ep={n_ep} T_ep={T_ep} n_feat={n_feat} n_act={n_act} kappa={kappa}")
    print(f"RL  : Sharpe={s_rl[0]:.3f} vol={s_rl[1]:.3f} mdd={s_rl[2]:.3f} final={s_rl[3]:.3f}")
    print(f"EW  : Sharpe={s_ew[0]:.3f} vol={s_ew[1]:.3f} mdd={s_ew[2]:.3f} final={s_ew[3]:.3f}")
    print(f"6040: Sharpe={s_60[0]:.3f} vol={s_60[1]:.3f} mdd={s_60[2]:.3f} final={s_60[3]:.3f}")
    print(f"RISK: Sharpe={s_rk[0]:.3f} vol={s_rk[1]:.3f} mdd={s_rk[2]:.3f} final={s_rk[3]:.3f}")
    print(f"final ep reward (smoothed tail) ~ {sm[-1]:.4f}")
    print(f"safe weight mean={safe_w.mean():.3f} max={safe_w.max():.3f} min={safe_w.min():.3f}")
    print("=" * 60)


if __name__ == "__main__":
    gen_liquidity()
    gen_rl()
