#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成 2026-10-25（第四弹）两篇量化文章的核心数值 + 配图（固定 seed 可复现）。
  A. survival-analysis-highfreq-default  生存分析高频违约预测：用 Cox 比例风险建模跳变
     - 3000 家合成公司 × 60 个月面板；违约 = 时变协变量驱动的跳变（非齐次 Poisson）
     - 从零实现 Cox 部分似然 Newton-Raphson + Breslow 基线风险 + Schoenfeld PH 检验
     - 对比静态 logit；C-index；12 个月 PD 期限结构
  B. bayesian-structural-breakpoint      贝叶斯结构断点检测：用变点把行情的变脸时刻找出来
     - 合成 500 日三段制度收益（calm / crash / recovery）
     - 从零实现 Normal-Gamma 共轭的贝叶斯在线变点检测（BOCD）
       约定：r_t = 当前段观测数（含 x_t）；变点分支用「先验预测」解释新段首点
     - Viterbi 离线 MAP 分割；在线 P(r_t=1) 检测；与 CUSUM 对比延迟/误报/漏报
所有图表均为真实计算图，数值固定随机种子可复现。
"""
import os
import math
import json
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
rcParams["figure.facecolor"] = "white"

BASE = "/Users/halo/workspace/astro-blog/public/images"


# ============================================================
# 文章 A：生存分析高频违约预测（Cox 比例风险，numpy 从零实现）
# ============================================================
def gen_survival_cox():
    slug = "survival-analysis-highfreq-default"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    rng = np.random.default_rng(20261025)
    N, T = 6000, 60
    LAM0 = lambda t: 0.0008 * np.exp(0.01 * t)
    BETA_TRUE = np.array([1.5, 1.2, -2.0])

    # ---- 时变协变量（AR(1)）----
    lev = np.zeros((N, T)); vol = np.zeros((N, T)); roa = np.zeros((N, T))
    lev[:, 0] = rng.normal(0.40, 0.15, N).clip(0.05, 0.95)
    vol[:, 0] = rng.normal(0.25, 0.08, N).clip(0.05, 1.00)
    roa[:, 0] = rng.normal(0.02, 0.020, N).clip(-0.15, 0.20)
    for t in range(1, T):
        lev[:, t] = (0.95 * lev[:, t-1] + 0.05 * 0.40 + rng.normal(0, 0.04, N)).clip(0.05, 0.95)
        vol[:, t] = (0.90 * vol[:, t-1] + 0.10 * 0.25 + rng.normal(0, 0.06, N)).clip(0.05, 1.00)
        roa[:, t] = (0.90 * roa[:, t-1] + 0.10 * 0.02 + rng.normal(0, 0.025, N)).clip(-0.15, 0.20)

    # ---- 生成违约（跳变）：逐月非齐次 Poisson ----
    default_t = np.full(N, T + 1, dtype=int)
    alive = np.ones(N, dtype=bool)
    for t in range(T):
        lam = LAM0(t) * np.exp(lev[alive, t] * BETA_TRUE[0]
                               + vol[alive, t] * BETA_TRUE[1] + roa[alive, t] * BETA_TRUE[2])
        p = 1.0 - np.exp(-lam)
        hit = rng.random(alive.sum()) < p
        idx = np.where(alive)[0][hit]
        default_t[idx] = t
        alive[idx] = False
    exit_t = np.minimum(default_t, T)
    event = (default_t <= T).astype(int)

    X = np.stack([lev[:, 0], vol[:, 0], roa[:, 0]], axis=1)
    Xlast = np.stack([lev[:, T-1], vol[:, T-1], roa[:, T-1]], axis=1)
    Xt = np.stack([lev, vol, roa], axis=2)               # (N, T, 3)

    # ---- Cox 拟合函数（可限定公司子集）----
    def fit_cox(mask):
        idx = np.where(mask)[0]
        ev = np.where((event == 1) & mask)[0]
        ev_t = exit_t[ev]
        o = np.argsort(ev_t, kind="stable")
        ev, ev_t = ev[o], ev_t[o]
        Xev_ = Xt[ev, ev_t, :]
        rc = {te: np.where(mask & (exit_t >= te))[0] for te in np.unique(ev_t)}

        def nll_grad_hess(beta):
            denom = np.empty(len(ev))
            for k, te in enumerate(ev_t):
                denom[k] = np.exp(Xt[rc[te], te, :] @ beta).sum()
            ll = float((Xev_ @ beta - np.log(denom)).sum())
            grad = np.zeros(3); hess = np.zeros((3, 3))
            for k, te in enumerate(ev_t):
                xm = Xt[rc[te], te, :]
                w = np.exp(xm @ beta); sw = w.sum()
                xbar = (w[:, None] * xm).sum(0) / sw
                grad += Xev_[k] - xbar
                xc = xm - xbar
                hess += (w[:, None, None] * xc[:, :, None] * xc[:, None, :]).sum(0) / sw
            return -ll, -grad, hess

        b = np.zeros(3)
        for it in range(100):
            f, g, h = nll_grad_hess(b)
            step = np.linalg.solve(h, g)
            s = 1.0
            for _ in range(30):
                f_new, _, _ = nll_grad_hess(b - s * step)
                if f_new <= f + 1e-12:
                    break
                s *= 0.5
            b = b - s * step
            if np.max(np.abs(s * step)) < 1e-10:
                break
        f, g, h = nll_grad_hess(b)
        cov = np.linalg.inv(h)
        se_ = np.sqrt(np.diag(cov))
        # Breslow 基线（该子集）
        lh = np.zeros(T)
        for t in range(T):
            m = np.where(mask & (exit_t >= t))[0]
            s_ = np.exp(Xt[m, t, :] @ b).sum()
            d = int((ev_t == t).sum())
            lh[t] = d / s_ if d > 0 else np.nan
        ok_ = ~np.isnan(lh)
        lh_i = np.interp(np.arange(T), np.arange(T)[ok_], lh[ok_])
        # Schoenfeld 残差（该子集）
        sl = np.empty(len(ev)); sv = np.empty(len(ev))
        for k, te in enumerate(ev_t):
            xm = Xt[rc[te], te, :]
            w = np.exp(xm @ b); sw = w.sum()
            xbar = (w[:, None] * xm).sum(0) / sw
            sl[k] = Xev_[k, 0] - xbar[0]
            sv[k] = Xev_[k, 1] - xbar[1]
        tcc = ev_t - ev_t.mean()
        slope_l = float((sl * tcc).sum() / (tcc * tcc).sum())
        slope_v = float((sv * tcc).sum() / (tcc * tcc).sum())
        return b, se_, lh_i, slope_l, slope_v

    # 训练/测试划分（按公司）
    perm = rng.permutation(N)
    tr_mask = np.zeros(N, dtype=bool); tr_mask[perm[:2100]] = True
    te_mask = ~tr_mask
    beta, se, lam0_interp, slope_lev, slope_vol = fit_cox(tr_mask)
    lam0_true_arr = LAM0(np.arange(T))
    Lam0 = np.cumsum(lam0_interp)
    hr = np.exp(beta)
    zvals = beta / se

    # ---- 生存曲线（按风险三分位）----
    lp0 = Xt[:, 0, :] @ beta
    terc = np.quantile(lp0, [1/3, 2/3])
    groups = np.digitize(lp0, terc)
    grid = np.arange(T + 1)
    surv_model = {}
    for g, lab in [(0, "低风险组"), (1, "中风险组"), (2, "高风险组")]:
        lp_g = lp0[groups == g].mean()
        surv_model[lab] = np.exp(-np.concatenate([[0.0], Lam0]) * np.exp(lp_g))

    def km_curve(mask):
        times = exit_t[mask]; evs = event[mask]
        o = np.argsort(times, kind="stable")
        times, evs = times[o], evs[o]
        S = [1.0]
        for t in range(T):
            at_risk = (times >= t).sum()
            d = ((times == t) & (evs == 1)).sum()
            S.append(S[-1] * (1 - d / at_risk) if at_risk > 0 and d > 0 else S[-1])
        return np.array(S)
    km = {lab: km_curve(groups == g) for g, lab in [(0, "低风险组"), (1, "中风险组"), (2, "高风险组")]}

    # ---- 12 个月 PD 期限结构 ----
    pd12 = {}
    for lev_lab, lev_v in [("低杠杆 20%", 0.20), ("中杠杆 45%", 0.45), ("高杠杆 80%", 0.80)]:
        x = np.array([lev_v, 0.25, 0.02])
        lp = float(x @ beta)
        S_t = np.exp(-np.concatenate([[0.0], Lam0]) * np.exp(lp))
        pd12[lev_lab] = np.array([1 - S_t[t + 12] / S_t[t] for t in range(0, T - 12)])

    # ---- 12 个月前瞻违约排名：逐月重评（pooled C-index）----
    def pooled_cindex_12m(score_mat, firms, months=range(0, 48)):
        """score_mat[i, m] = 公司 i 在月 m 的风险分（只允许用 m 月及以前的信息）。
        单元 (i,m)：仍在市（exit_t>m）；标签 = 12 个月内是否违约。
        同月内截面配对：违约方分数应高于存活方。"""
        conc = tot = 0.0
        for m in months:
            alive = firms[exit_t[firms] > m]
            if len(alive) == 0:
                continue
            d = exit_t[alive] - m
            y1 = alive[(event[alive] == 1) & (d <= 12)]      # 12 个月内违约
            y0 = alive[d > 12]                                # 观察期内活过整个窗口
            if len(y1) == 0 or len(y0) == 0:
                continue
            s1 = score_mat[y1, m][:, None]
            s0 = score_mat[y0, m][None, :]
            tot += len(y1) * len(y0)
            conc += (s1 > s0).sum() + 0.5 * (s1 == s0).sum()
        return conc / tot if tot > 0 else np.nan

    # Cox 逐月重评：分数 = x_{i,m}·β̂（训练集估计）
    score_cox = np.einsum("nmk,k->nm", Xt, beta)
    te_idx = np.where(te_mask)[0]

    # 静态 logit：m=0 特征训练，预测 12 个月内违约；系数冻结，每月套最新特征
    def logit_fit(Xf, y, iters=200):
        Xb = np.column_stack([np.ones(len(Xf)), Xf])
        w = np.zeros(Xb.shape[1])
        for _ in range(iters):
            p = 1 / (1 + np.exp(-(Xb @ w)))
            g = Xb.T @ (y - p) / len(y) - 0.01 * w
            H = (Xb * (p * (1 - p))[:, None]).T @ Xb / len(y) + 0.01 * np.eye(Xb.shape[1])
            w = w + np.linalg.solve(H, g)
        return w

    tr_idx = np.where(tr_mask)[0]
    y12 = ((event == 1) & (exit_t <= 12)).astype(float)
    w_static = logit_fit(X[tr_idx], y12[tr_idx])
    score_logit = np.einsum("nmk,k->nm", Xt, w_static[1:]) + w_static[0]

    c_cox_pooled = pooled_cindex_12m(score_cox, te_idx)
    c_logit_pooled = pooled_cindex_12m(score_logit, te_idx)
    c_logit_annual = pooled_cindex_12m(score_logit, te_idx, months=range(0, 1))  # 仅 m=0 单元（年度体检）
    c_cox_m0 = pooled_cindex_12m(score_cox, te_idx, months=range(0, 1))

    # 高风险组：m=0 静态模型 PD12 vs 实际 12 个月违约率
    terc = np.quantile(score_cox[:, 0], [1/3, 2/3])
    groups = np.digitize(score_cox[:, 0], terc)
    hi = np.where(te_mask & (groups == 2))[0]
    lp_hi = score_cox[hi, 0].mean()
    S0 = np.exp(-np.concatenate([[0.0], Lam0]) * np.exp(lp_hi))
    pd12_static_hi = 1 - S0[12] / S0[0]
    realized_hi = np.mean((event[hi] == 1) & (exit_t[hi] <= 12))

    print("=" * 70)
    print("文章 A：生存分析高频违约预测（Cox）")
    print(f"样本：{N} 公司 × {T} 月；违约 {event.sum()} 家（{event.mean()*100:.2f}%），存活 {N - event.sum()}")
    print(f"真实 β: 杠杆 {BETA_TRUE[0]}, 波动 {BETA_TRUE[1]}, ROA {BETA_TRUE[2]}")
    print(f"估计 β: 杠杆 {beta[0]:.3f}(se {se[0]:.3f}, z {zvals[0]:.1f}), 波动 {beta[1]:.3f}(se {se[1]:.3f}, z {zvals[1]:.1f}), ROA {beta[2]:.3f}(se {se[2]:.3f}, z {zvals[2]:.1f})")
    print(f"风险比 HR: 杠杆 {hr[0]:.2f}, 波动 {hr[1]:.2f}, ROA {hr[2]:.3f}")
    print(f"12个月前瞻 C-index（测试公司）: Cox逐月重评 {c_cox_pooled:.4f} | 静态logit冻结系数 {c_logit_pooled:.4f} | 静态logit仅m=0(年度体检) {c_logit_annual:.4f} | Cox仅m=0 {c_cox_m0:.4f}")
    print(f"Schoenfeld PH: 杠杆斜率 {slope_lev:+.5f}, 波动斜率 {slope_vol:+.5f}")
    print(f"基线强度 λ0(0): 真值 {LAM0(0):.5f} vs 估计 {lam0_interp[0]:.5f}; λ0(59): {LAM0(59):.5f} vs {lam0_interp[59]:.5f}")
    x_ref = np.array([0.40, 0.25, 0.02])
    cum5_true = sum(LAM0(t) * float(np.exp(x_ref @ BETA_TRUE)) for t in range(T))
    print(f"5年累计违约率（参考公司）: 真值 {1-np.exp(-cum5_true):.4f}")
    for lab in pd12:
        print(f"PD12({lab}): 首月 {pd12[lab][0]*100:.2f}% → 末年 {pd12[lab][-1]*100:.2f}%")
    for lab in surv_model:
        print(f"S(60) {lab}: 模型 {surv_model[lab][-1]*100:.1f}% vs KM {km[lab][-1]*100:.1f}%")
    print(f"高风险组 m=0: 静态PD12 {pd12_static_hi*100:.2f}% | 逐月重评PD12 {pd12_rescored_hi*100:.2f}% | 实际12月违约率 {realized_hi*100:.2f}%")
    ev_all = np.where(event == 1)[0]
    d_idx = ev_all[:3]
    for i in d_idx:
        print(f"违约例: 公司{i} 退出月 {exit_t[i]}, 违约月协量 lev={lev[i, exit_t[i]]:.2f} vol={vol[i, exit_t[i]]:.2f} roa={roa[i, exit_t[i]]:.3f}")
    print(f"违约公司违约月均: lev={lev[ev_all, exit_t[ev_all]].mean():.3f} vol={vol[ev_all, exit_t[ev_all]].mean():.3f} roa={roa[ev_all, exit_t[ev_all]].mean():.3f}")
    surv_idx = np.where(event == 0)[0]
    print(f"存活公司期末均:   lev={lev[surv_idx, T-1].mean():.3f} vol={vol[surv_idx, T-1].mean():.3f} roa={roa[surv_idx, T-1].mean():.3f}")

    # ============ 图 1：协变量路径 ============
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    surv0 = surv_idx[0]
    for i in d_idx[:3]:
        axes[0].plot(lev[i], lw=1.2, label=f"公司{i}（第{exit_t[i]}月违约）")
    axes[0].plot(lev[surv0], lw=1.0, color="gray", alpha=0.7, label="存活公司")
    axes[0].set_title("杠杆路径（违约 vs 存活）"); axes[0].set_xlabel("月"); axes[0].legend(fontsize=8)
    for i in d_idx[:3]:
        axes[1].plot(vol[i], lw=1.2, label=f"公司{i}")
    axes[1].plot(vol[surv0], lw=1.0, color="gray", alpha=0.7, label="存活公司")
    axes[1].set_title("已实现波动路径"); axes[1].set_xlabel("月"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "covariate_paths.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 2：基线风险 ============
    fig, ax = plt.subplots(figsize=(7.2, 4))
    ax.plot(lam0_true_arr, drawstyle="steps-mid", lw=1.6, label="真实 λ0(t)（数据生成）")
    ax.plot(lam0_interp, lw=1.3, alpha=0.85, label="Breslow 估计 λ̂0(t)")
    ax.set_title("基线违约强度：真实 vs Breslow 估计"); ax.set_xlabel("月"); ax.set_ylabel("月违约强度"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "baseline_hazard.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 3：生存曲线 ============
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for lab, c in [("低风险组", "#2ca02c"), ("中风险组", "#1f77b4"), ("高风险组", "#d62728")]:
        ax.plot(grid, surv_model[lab], lw=1.8, color=c, label=f"{lab}（模型）")
        ax.plot(grid, km[lab], lw=1.2, ls="--", color=c, alpha=0.7, label=f"{lab}（KM 经验）")
    ax.set_title("生存曲线：模型 vs 经验（按风险三分位）"); ax.set_xlabel("月"); ax.set_ylabel("生存概率 S(t)"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "survival_curves.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 4：PD 期限结构 ============
    fig, ax = plt.subplots(figsize=(7.2, 4))
    for lab, c in [("低杠杆 20%", "#2ca02c"), ("中杠杆 45%", "#1f77b4"), ("高杠杆 80%", "#d62728")]:
        ax.plot(np.arange(len(pd12[lab])), pd12[lab] * 100, lw=1.8, color=c, label=lab)
    ax.set_title("12 个月边际违约概率 PD 期限结构（协量固定）"); ax.set_xlabel("评分月份"); ax.set_ylabel("PD12 (%)"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "pd_term_structure.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 5：模型对比 + 系数恢复 ============
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4))
    models = ["Cox\n逐月重评", "静态logit\n冻结系数", "静态logit\n年度体检(m=0)"]
    vals = [c_cox_pooled, c_logit_pooled, c_logit_annual]
    bars = axes[0].bar(models, vals, color=["#d62728", "#7f7f7f", "#7f7f7f"], width=0.55)
    for b, v in zip(bars, vals):
        axes[0].text(b.get_x() + b.get_width() / 2, v + 0.002, f"{v:.3f}", ha="center", fontsize=9)
    axes[0].set_ylim(0.4, max(vals) + 0.04)
    axes[0].set_title("12个月前瞻违约排名 C-index（测试公司）")
    xpos = np.arange(3)
    axes[1].bar(xpos - 0.18, BETA_TRUE, width=0.36, label="真实 β", color="#1f77b4")
    axes[1].bar(xpos + 0.18, beta, width=0.36, yerr=se, label="Cox 估计 β̂（训练集）", color="#ff7f0e", capsize=3)
    axes[1].set_xticks(xpos); axes[1].set_xticklabels(["杠杆", "波动", "ROA"]); axes[1].set_title("系数恢复（含 ±1 SE）"); axes[1].legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "model_comparison.png"), bbox_inches="tight"); plt.close(fig)

    return {
        "n_default": int(event.sum()), "rate": float(event.mean()),
        "beta": beta.tolist(), "se": se.tolist(), "z": zvals.tolist(), "hr": hr.tolist(),
        "c_cox_pooled": float(c_cox_pooled), "c_logit_pooled": float(c_logit_pooled),
        "c_logit_annual": float(c_logit_annual), "c_cox_m0": float(c_cox_m0),
        "slope_lev": slope_lev, "slope_vol": slope_vol,
        "pd12_hi": pd12["高杠杆 80%"].tolist(), "pd12_lo": pd12["低杠杆 20%"].tolist(), "pd12_mid": pd12["中杠杆 45%"].tolist(),
        "surv_model": {k: v.tolist() for k, v in surv_model.items()},
        "km": {k: v.tolist() for k, v in km.items()},
        "lam0_interp": lam0_interp.tolist(), "lam0_true": lam0_true_arr.tolist(),
        "pd12_static_hi": float(pd12_static_hi), "pd12_rescored_hi": float(pd12_rescored_hi), "realized_hi": float(realized_hi),
        "d_mean": {"lev": float(lev[ev_all, exit_t[ev_all]].mean()), "vol": float(vol[ev_all, exit_t[ev_all]].mean()), "roa": float(roa[ev_all, exit_t[ev_all]].mean())},
        "s_mean": {"lev": float(lev[surv_idx, T-1].mean()), "vol": float(vol[surv_idx, T-1].mean()), "roa": float(roa[surv_idx, T-1].mean())},
        "cum5_true": float(1 - np.exp(-cum5_true)),
    }


# ============================================================
# 文章 B：贝叶斯结构断点检测（BOCD，numpy 从零实现）
# ============================================================
def gen_bocd():
    slug = "bayesian-structural-breakpoint"
    OUT = os.path.join(BASE, slug)
    os.makedirs(OUT, exist_ok=True)

    rng = np.random.default_rng(4262026)
    T = 500
    true_mu = np.concatenate([np.full(150, 0.03), np.full(150, -0.10), np.full(200, 0.06)])
    true_sd = np.concatenate([np.full(150, 0.80), np.full(150, 2.00), np.full(200, 1.00)])
    true_cps = [150, 300]
    x = rng.normal(true_mu, true_sd)

    H = 1.0 / 100.0                       # 恒定危险率：先验期望段长 100 日
    K0, M0, A0, B0 = 0.01, 0.0, 0.01, 0.01
    Rmax = T + 2

    def ng_stats_from_prior():
        return 0.0, M0, 0.0, K0, A0, B0   # n, mean, M2, kappa, a, b

    def update_stats(n, m, M2, kap, a, b, xv):
        n1 = n + 1
        m1 = m + (xv - m) / n1
        M21 = M2 + (xv - m) * (xv - m1)
        kap1 = kap + 1
        a1 = a + 0.5
        b1 = b + kap * (xv - m) ** 2 / (2 * (kap + 1))
        return n1, m1, M21, kap1, a1, b1

    def pred_logp(xv, m, kap, a, b):
        """Normal-Gamma 后验预测（Student-t）对数密度"""
        nu = 2 * a
        scale = math.sqrt(b * (kap + 1) / (a * kap))
        z = (xv - m) / scale
        return (math.lgamma((nu + 1) / 2) - math.lgamma(nu / 2)
                - 0.5 * math.log(nu * math.pi) - math.log(scale)
                - (nu + 1) / 2 * math.log(1 + z * z / nu))

    # 状态：run r = 1..t+1（当前段含 x_t 共 r 个观测）
    # stats[r] = run r 的充分统计量（在 t-1 时刻，即 r 个观测 x_{t-r}..x_{t-1}）
    log_R = np.full(Rmax, -np.inf)
    n_run = np.zeros(Rmax); mean_run = np.zeros(Rmax); M2_run = np.zeros(Rmax)
    kap_run = np.full(Rmax, K0); a_run = np.full(Rmax, A0); b_run = np.full(Rmax, B0)
    log_R[1] = 0.0                        # t=0 之前不存在；占位

    P_cp = np.zeros(T)                    # P(r_t = 1)：当天开启新段
    map_run = np.zeros(T, dtype=int)
    vit = np.full((T, Rmax), -np.inf); back = np.zeros((T, Rmax), dtype=int)
    post_mean_map = np.zeros(T); post_scale_map = np.zeros(T)
    R_hist = np.zeros((T, Rmax))
    logp_prior_const = None

    for t in range(T):
        xv = x[t]
        # 各 run 的预测对数密度（run r 有 r 个观测：x_{t-r}..x_{t-1}）
        logp = np.full(Rmax, -np.inf)
        for r in range(1, t + 2):         # r = 1..t+1（上一时刻存在的 run）
            if log_R[r] > -np.inf:
                logp[r] = pred_logp(xv, mean_run[r], kap_run[r], a_run[r], b_run[r])
        # 先验预测（新段首点）：用 NG 先验
        lp_prior = pred_logp(xv, M0, K0, A0, B0)
        # 递推（新约定）：
        #   R_t(1)   = H · p(x_t|先验) · Σ_r R_{t-1}(r)
        #   R_t(r+1) = R_{t-1}(r) · (1-H) · p(x_t|run r)
        sum_prev = np.logaddexp.reduce(log_R[1:t + 2]) if t + 1 >= 1 else -np.inf
        new_log_R = np.full(Rmax, -np.inf)
        new_log_R[1] = math.log(H) + lp_prior + sum_prev
        new_log_R[2:t + 3] = (log_R[1:t + 2] + math.log(1 - H) + logp[1:t + 2])
        # Viterbi
        best_prev = int(np.argmax(vit[t - 1, 1:t + 2])) + 1 if t > 0 else 1
        vit[t, 1] = (vit[t - 1, best_prev] if t > 0 else 0.0) + math.log(H) + lp_prior
        back[t, 1] = best_prev
        for r in range(2, t + 3):
            vit[t, r] = vit[t - 1, r - 1] + math.log(1 - H) + logp[r - 1]
            back[t, r] = r - 1
        # 统计量更新：新 run r = 旧 run r-1 + xv；run 1 = 先验 + xv
        new_n = np.zeros(Rmax); new_mean = np.zeros(Rmax); new_M2 = np.zeros(Rmax)
        new_kap = np.full(Rmax, K0); new_a = np.full(Rmax, A0); new_b = np.full(Rmax, B0)
        n1, m1, M21, k1, a1, b1 = update_stats(0.0, M0, 0.0, K0, A0, B0, xv)
        new_n[1], new_mean[1], new_M2[1], new_kap[1], new_a[1], new_b[1] = n1, m1, M21, k1, a1, b1
        for r in range(2, t + 3):
            ro = r - 1
            if n_run[ro] == 0 and ro != 1:
                continue
            n1, m1, M21, k1, a1, b1 = update_stats(n_run[ro], mean_run[ro], M2_run[ro], kap_run[ro], a_run[ro], b_run[ro], xv)
            new_n[r], new_mean[r], new_M2[r], new_kap[r], new_a[r], new_b[r] = n1, m1, M21, k1, a1, b1
        log_R = new_log_R
        n_run, mean_run, M2_run = new_n, new_mean, new_M2
        kap_run, a_run, b_run = new_kap, new_a, new_b
        mx = log_R.max()
        Rr = np.exp(log_R - mx); Rr /= Rr.sum()
        R_hist[t] = Rr
        P_cp[t] = Rr[1]                    # 当天开启新段的概率
        r_star = int(np.argmax(Rr)); map_run[t] = r_star
        post_mean_map[t] = mean_run[r_star]
        post_scale_map[t] = math.sqrt(b_run[r_star] * (kap_run[r_star] + 1) / (a_run[r_star] * kap_run[r_star]))

    # ---- 在线检测：P(r=1) 局部峰 > 阈值，5 日合并 ----
    thresh = 0.30
    cand_t = []
    for t in range(2, T - 2):
        if P_cp[t] > thresh and P_cp[t] == max(P_cp[max(0, t - 2):t + 3]):
            cand_t.append(t)
    merged = []
    for c in cand_t:
        if merged and c - merged[-1][-1] <= 5:
            merged[-1].append(c)
        else:
            merged.append([c])
    det_online = [int(np.mean(g)) for g in merged]

    # ---- Viterbi 离线 MAP 分割 ----
    path = np.zeros(T, dtype=int)
    r = int(np.argmax(vit[T - 1]))
    for t in range(T - 1, -1, -1):
        path[t] = r
        r = back[t, r]
    det_viterbi = [t for t in range(1, T) if path[t] < path[t - 1]]

    # ---- CUSUM（稳健标准化，双侧，首次越阈）----
    med = np.median(x); mad = np.median(np.abs(x - med)) * 1.4826
    z = (x - med) / mad
    k_c, h_c = 0.5, 5.0
    Sp = np.zeros(T); Sm = np.zeros(T)
    for t in range(T):
        pp = Sp[t - 1] if t > 0 else 0.0
        pm = Sm[t - 1] if t > 0 else 0.0
        Sp[t] = max(0.0, pp + z[t] - k_c)
        Sm[t] = max(0.0, pm - z[t] - k_c)
    cusum = np.maximum(Sp, Sm)
    det_cusum = []
    prev = -100
    for t in range(50, T):
        if cusum[t] > h_c:
            if t - prev > 5:
                det_cusum.append(t)
            prev = t

    # ---- 指标 ----
    def metrics(det, truths, tol=10):
        lat, matched = [], list(truths)
        for d in det:
            near = [tr for tr in matched if abs(d - tr) <= tol]
            if near:
                tr = min(near, key=lambda zz: abs(zz - d))
                lat.append(d - tr); matched.remove(tr)
        fp = [d for d in det if not any(abs(d - tr) <= tol for tr in truths)]
        return lat, matched, fp
    lat_on, miss_on, fp_on = metrics(det_online, true_cps)
    lat_vi, miss_vi, fp_vi = metrics(det_viterbi, true_cps)
    lat_cu, miss_cu, fp_cu = metrics(det_cusum, true_cps)
    deep = map_run > 20
    err_scale = float(np.abs(post_scale_map[deep] - true_sd[deep]).mean())
    err_mean = float(np.abs(post_mean_map[deep] - true_mu[deep]).mean())

    print("=" * 70)
    print("文章 B：贝叶斯结构断点检测（BOCD）")
    print(f"真实变点: {true_cps}；BOCD在线检测: {det_online}；Viterbi: {det_viterbi}；CUSUM: {det_cusum}")
    print(f"BOCD在线: 延迟 {lat_on}, 漏报 {miss_on}, 误报 {len(fp_on)}")
    print(f"Viterbi : 延迟 {lat_vi}, 漏报 {miss_vi}, 误报 {len(fp_vi)}")
    print(f"CUSUM   : 延迟 {lat_cu}, 漏报 {miss_cu}, 误报 {len(fp_cu)}")
    print(f"制度跟踪（run>20）: 尺度 MAE {err_scale:.3f}, 均值 MAE {err_mean:.4f}")
    print(f"P(r=1) 峰值 t={int(np.argmax(P_cp))} ({P_cp.max():.3f}); t=150 附近 {np.round(P_cp[147:156], 3).tolist()}; t=300 附近 {np.round(P_cp[297:306], 3).tolist()}")
    print(f"平静期 P(r=1) 基线: 前150日均值 {P_cp[:150].mean():.4f}, 最大 {P_cp[:150].max():.3f}")
    print(f"危险率 H={H}（先验期望段长 {1/H:.0f} 日）")

    # ============ 图 1：合成收益与制度 ============
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(x, lw=0.8, color="#1f77b4", alpha=0.85)
    ax.axvspan(0, 150, alpha=0.08, color="green"); ax.axvspan(150, 300, alpha=0.08, color="red"); ax.axvspan(300, 500, alpha=0.08, color="orange")
    for cp in true_cps:
        ax.axvline(cp, color="red", ls="--", lw=1.2)
    ax.text(75, x.max() * 0.90, "制度1 calm\nμ=0.03% σ=0.8%", ha="center", fontsize=9)
    ax.text(225, x.max() * 0.90, "制度2 crash\nμ=-0.10% σ=2.0%", ha="center", fontsize=9)
    ax.text(400, x.max() * 0.90, "制度3 recovery\nμ=0.06% σ=1.0%", ha="center", fontsize=9)
    ax.set_title("合成日收益序列（500 日，3 段制度，红虚线=真实变点）"); ax.set_xlabel("交易日"); ax.set_ylabel("日收益 %")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "synthetic_returns.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 2：run-length 后验热力图 ============
    fig, ax = plt.subplots(figsize=(11, 4.2))
    RR = R_hist[:, :260].T
    im = ax.imshow(np.log10(RR[:, 1:] + 1e-12), aspect="auto", origin="lower", cmap="viridis",
                   extent=[0, T, 1, 261], vmin=-8, vmax=0)
    for cp in true_cps:
        ax.axvline(cp, color="red", ls="--", lw=1.2)
    ax.set_title("Run-length 后验 log10 P(r_t | x_1:t)（红虚线=真实变点；亮带=段结构）")
    ax.set_xlabel("交易日 t"); ax.set_ylabel("run length r（当前段观测数）")
    fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "runlength_posterior.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 3：在线变点概率 ============
    fig, ax = plt.subplots(figsize=(11, 3.8))
    ax.plot(P_cp, lw=1.0, color="#1f77b4")
    ax.axhline(thresh, color="gray", ls=":", lw=1)
    for cp in true_cps:
        ax.axvline(cp, color="red", ls="--", lw=1.2, label="真实变点" if cp == true_cps[0] else None)
    for d in det_online:
        ax.axvline(d, color="green", ls="-", lw=1.4, alpha=0.85, label="BOCD 检测" if d == det_online[0] else None)
    ax.set_title(f"在线变点概率 P(r_t=1)（检测阈值 {thresh}）"); ax.set_xlabel("交易日"); ax.set_ylabel("概率"); ax.legend()
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "cp_probability.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 4：检测质量对比 ============
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, (name, lat, miss, fp, color) in zip(axes, [
        ("BOCD 在线", lat_on, miss_on, fp_on, "#1f77b4"),
        ("CUSUM", lat_cu, miss_cu, fp_cu, "#ff7f0e")]):
        vals_ = [len(lat), len(miss), len(fp)]
        ax.bar(["检测到的\n真实变点", "漏报", "误报"], vals_, color=[color, "#d62728", "#7f7f7f"], width=0.5)
        for i, v in enumerate(vals_):
            ax.text(i, v + 0.03, str(v), ha="center")
        med_txt = f"延迟中位 {np.median(lat):.0f} 日" if len(lat) else "无检测"
        ax.set_title(f"{name}：{med_txt}")
    axes[0].set_ylabel("变点个数")
    fig.suptitle("检测质量对比（±10 日容差匹配）")
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "detection_quality.png"), bbox_inches="tight"); plt.close(fig)

    # ============ 图 5：制度跟踪 ============
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True)
    axes[0].plot(true_mu, lw=1.4, color="gray", label="真实制度均值 μ")
    axes[0].plot(post_mean_map, lw=1.0, color="#1f77b4", label="MAP run 后验均值")
    for cp in true_cps:
        axes[0].axvline(cp, color="red", ls="--", lw=1)
    axes[0].set_title("制度均值跟踪（MAP run 后验）"); axes[0].set_ylabel("日收益 %"); axes[0].legend(fontsize=8)
    axes[1].plot(true_sd, lw=1.4, color="gray", label="真实制度波动 σ")
    axes[1].plot(post_scale_map, lw=1.0, color="#d62728", label="MAP run 后验预测尺度")
    for cp in true_cps:
        axes[1].axvline(cp, color="red", ls="--", lw=1)
    axes[1].set_title("制度波动跟踪（MAP run 后验预测尺度）"); axes[1].set_xlabel("交易日"); axes[1].set_ylabel("日波动 %"); axes[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "regime_tracking.png"), bbox_inches="tight"); plt.close(fig)

    return {
        "det_online": det_online, "det_viterbi": det_viterbi, "det_cusum": det_cusum,
        "lat_on": lat_on, "lat_vi": lat_vi, "lat_cu": lat_cu,
        "miss_on": miss_on, "miss_vi": miss_vi, "miss_cu": miss_cu,
        "fp_on": len(fp_on), "fp_vi": len(fp_vi), "fp_cu": len(fp_cu),
        "err_scale": err_scale, "err_mean": err_mean,
        "p_cp_max": float(P_cp.max()), "p_cp_argmax": int(np.argmax(P_cp)),
        "p_cp_baseline_mean": float(P_cp[:150].mean()), "p_cp_baseline_max": float(P_cp[:150].max()),
        "p_cp_150": [float(v) for v in P_cp[147:156]], "p_cp_300": [float(v) for v in P_cp[297:306]],
    }


if __name__ == "__main__":
    res_a = gen_survival_cox()
    res_b = gen_bocd()
    with open("/Users/halo/workspace/astro-blog/_oct25d_results.json", "w") as f:
        json.dump({"A": res_a, "B": res_b}, f)
    print("\n全部完成：10 张配图 + 数值 JSON 已保存")
