#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分数阶差分与平稳性：在保留记忆和通过 ADF 之间求平衡
全部图表由真实计算生成，固定随机种子可复现。
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import adfuller

plt.rcParams["font.sans-serif"] = ["Heiti SC", "Arial Unicode MS", "PingFang HK"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25

OUT = "/Users/halo/workspace/astro-blog/public/images/fractional-differencing-stationarity"
os.makedirs(OUT, exist_ok=True)
SEED = 20260801
C_A, C_B, C_C, C_D = "#2563eb", "#dc2626", "#16a34a", "#f59e0b"

N = 5000
TAU = 1e-4
SPLIT = int(N * 0.6)
THETA, SIG_X, SIG_T, DRIFT = 0.02, 0.012, 0.012, 0.0003


# ---------------- 分数阶差分核心 ----------------
def frac_weights(d, size):
    w = [1.0]
    for k in range(1, size):
        w.append(-w[-1] * (d - k + 1) / k)
    return np.array(w)


def frac_weights_ffd(d, tau=TAU, max_size=20000):
    w = [1.0]
    k = 1
    while k < max_size:
        x = -w[-1] * (d - k + 1) / k
        if abs(x) < tau:
            break
        w.append(x)
        k += 1
    return np.array(w)


def frac_diff_ffd(series, d, tau=TAU):
    """FFD：等长输出，前 width-1 个为 NaN"""
    w = frac_weights_ffd(d, tau)
    width = len(w)
    out = np.full(len(series), np.nan)
    if width <= len(series):
        out[width - 1:] = np.convolve(series, w, mode="valid")
    return out, width


def frac_diff_expanding(series, d, tau=TAU):
    """扩张窗口：权重不截断，可用历史全部用上"""
    n = len(series)
    w = frac_weights(d, n)
    skip = len(frac_weights_ffd(d, tau))
    out = np.full(n, np.nan)
    for t in range(skip, n):
        out[t] = np.dot(w[:t + 1][::-1], series[:t + 1])
    return out


def adf_stat_p(arr, min_len=200):
    a = arr[~np.isnan(arr)]
    if len(a) < min_len or np.std(a) < 1e-14:
        return np.nan, np.nan
    r = adfuller(a, regression="c", autolag="AIC")
    return r[0], r[1]


def rolling_z(f, win=250):
    f = np.asarray(f, float)
    z = np.full(len(f), np.nan)
    for t in range(win, len(f)):
        w_ = f[t - win:t]
        if np.isnan(w_).any():
            continue
        s = w_.std()
        if s > 1e-14:
            z[t] = (f[t] - w_.mean()) / s
    return z


def make_series(seed=SEED, n=N):
    """log_price = 随机游走趋势(I(1)) + OU 平稳偏离(唯一可预测源)"""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = (1 - THETA) * x[t - 1] + SIG_X * rng.standard_normal()
    trend = np.cumsum(DRIFT + SIG_T * rng.standard_normal(n))
    return np.log(100.0) + trend + x, x


def make_rw(seed, n=N):
    rng = np.random.default_rng(seed)
    return np.log(100.0) + np.cumsum(DRIFT + 0.017 * rng.standard_normal(n))


def main():
    logp, x_true = make_series()
    ret_next = np.append(np.diff(logp), np.nan)
    t_axis = np.arange(N)
    half_life = np.log(2) / THETA

    # ============ 1. d 网格扫描 ============
    d_grid = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    rows, d_star = [], None
    for d in d_grid:
        ffd, width = frac_diff_ffd(logp, d)
        stat, p = adf_stat_p(ffd)
        m = ~np.isnan(ffd)
        corr = float(np.corrcoef(ffd[m], logp[m])[0, 1]) if m.sum() > 200 else np.nan
        rows.append(dict(d=float(d), adf=float(stat), p=float(p), corr=corr,
                         width=int(width), usable=bool(m.sum() > 200)))
        if d_star is None and p == p and p < 0.05:
            d_star = float(d)
    if d_star and d_star > 0:
        for dd in np.round(np.arange(max(0.0, d_star - 0.05), d_star + 1e-9, 0.01), 2):
            f_, _ = frac_diff_ffd(logp, dd)
            _, p_ = adf_stat_p(f_)
            if p_ == p_ and p_ < 0.05:
                d_star = float(dd)
                break

    adf_5pct = adfuller(logp, regression="c", autolag="AIC")[4]["5%"]
    ffd_star, width_star = frac_diff_ffd(logp, d_star)
    adf_star, p_star = adf_stat_p(ffd_star)
    m_s = ~np.isnan(ffd_star)
    corr_star = float(np.corrcoef(ffd_star[m_s], logp[m_s])[0, 1])
    adf_d0, p_d0 = adf_stat_p(logp)
    ffd_d1, w_d1 = frac_diff_ffd(logp, 1.0)
    adf_d1, p_d1 = adf_stat_p(ffd_d1)
    m_1 = ~np.isnan(ffd_d1)
    corr_d1 = float(np.corrcoef(ffd_d1[m_1], logp[m_1])[0, 1])

    # ============ 2. 预测力 ============
    def eval_ols(feat, name, ret=ret_next):
        f = np.asarray(feat, float)
        ok = (~np.isnan(f)) & (~np.isnan(ret))
        ins = ok.copy(); ins[SPLIT:] = False
        oos = ok.copy(); oos[:SPLIT] = False
        if ins.sum() < 200 or oos.sum() < 200:
            return dict(name=name, ok=False, r2_in=np.nan, r2_oos=np.nan, sharpe_oos=np.nan)
        X = np.column_stack([np.ones(ins.sum()), f[ins]])
        beta = np.linalg.lstsq(X, ret[ins], rcond=None)[0]
        pin = X @ beta
        r2_in = 1 - ((ret[ins] - pin) ** 2).sum() / ((ret[ins] - ret[ins].mean()) ** 2).sum()
        poos = beta[0] + beta[1] * f[oos]
        y = ret[oos]
        r2_oos = 1 - ((y - poos) ** 2).sum() / ((y - ret[ins].mean()) ** 2).sum()
        pnl = np.sign(poos) * y
        sh = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 1e-14 else 0.0
        return dict(name=name, ok=True, r2_in=float(r2_in), r2_oos=float(r2_oos), sharpe_oos=sh)

    def eval_z(feat, name, ret=ret_next):
        z = rolling_z(feat)
        ok = (~np.isnan(z)) & (~np.isnan(ret))
        ins = ok.copy(); ins[SPLIT:] = False
        oos = ok.copy(); oos[:SPLIT] = False
        if ins.sum() < 200 or oos.sum() < 200:
            return dict(name=name, ok=False, ic_in=np.nan, ic_oos=np.nan, sharpe_oos=np.nan)
        ic_in = float(spearmanr(z[ins], ret[ins])[0])
        sign = -1.0 if ic_in < 0 else 1.0
        pnl = (np.clip(sign * z, -2, 2) * ret)[oos]
        sh = float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 1e-14 else 0.0
        return dict(name=name, ok=True, ic_in=ic_in,
                    ic_oos=float(spearmanr(z[oos], ret[oos])[0]), sharpe_oos=sh)

    ols = dict(d0=eval_ols(logp, "d=0"), dstar=eval_ols(ffd_star, f"d={d_star}"),
               d1=eval_ols(ffd_d1, "d=1"), oracle=eval_ols(x_true, "oracle"))
    zev = dict(d0=eval_z(logp, "d=0"), dstar=eval_z(ffd_star, f"d={d_star}"),
               d1=eval_z(ffd_d1, "d=1"), oracle=eval_z(x_true, "oracle"))

    # ============ 3. 对抗式检验 ============
    # 3a 截断宽度如何伪造平稳性（纯随机游走，30 种子）
    rw_sweep = []
    for tau in [1e-2, 3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5]:
        ps, W = [], 0
        for sd in range(30):
            f_, W = frac_diff_ffd(make_rw(5000 + sd), d_star, tau=tau)
            _, p_ = adf_stat_p(f_)
            if p_ == p_:
                ps.append(p_)
        ps = np.array(ps)
        rw_sweep.append(dict(tau=tau, width=int(W), reject=float((ps < 0.05).mean()),
                             med_p=float(np.median(ps))))

    # 3b 主设定下随机游走的伪 alpha
    rw = make_rw(SEED + 7)
    ret_rw = np.append(np.diff(rw), np.nan)
    ffd_rw, _ = frac_diff_ffd(rw, d_star)
    _, p_rw = adf_stat_p(ffd_rw)
    rw_z = eval_z(ffd_rw, "RW", ret=ret_rw)
    rw_ols = eval_ols(ffd_rw, "RW", ret=ret_rw)

    # 3c look-ahead 对照
    def eval_lookahead(feat, ret):
        f = np.asarray(feat, float)
        z = (f - np.nanmean(f)) / np.nanstd(f)
        ok = (~np.isnan(z)) & (~np.isnan(ret))
        ins = ok.copy(); ins[SPLIT:] = False
        oos = ok.copy(); oos[:SPLIT] = False
        ic_in = spearmanr(z[ins], ret[ins])[0]
        sign = -1.0 if ic_in < 0 else 1.0
        pnl = (np.clip(sign * z, -2, 2) * ret)[oos]
        return float(pnl.mean() / pnl.std() * np.sqrt(252)) if pnl.std() > 1e-14 else 0.0

    sh_la_real = eval_lookahead(ffd_star, ret_next)
    sh_la_rw = eval_lookahead(ffd_rw, ret_rw)

    # 3d 置换检验
    z_star = rolling_z(ffd_star)
    base_ok = (~np.isnan(z_star)) & (~np.isnan(ret_next))
    idx = np.where(base_ok)[0]
    rng_p = np.random.default_rng(SEED + 21)
    perm = []
    for _ in range(300):
        rp = ret_next.copy()
        sh_ = rp[idx].copy(); rng_p.shuffle(sh_); rp[idx] = sh_
        perm.append(eval_z(ffd_star, "p", ret=rp)["sharpe_oos"])
    perm = np.array([v for v in perm if v == v])
    real_sh = zev["dstar"]["sharpe_oos"]
    perm_p = float((perm >= real_sh).mean())

    # 3e d* 跨样本稳定性
    d_stars = []
    for sd in range(20):
        lp_, _ = make_series(seed=SEED + 100 + sd)
        for d in d_grid:
            f_, _ = frac_diff_ffd(lp_, d)
            _, p_ = adf_stat_p(f_)
            if p_ == p_ and p_ < 0.05:
                d_stars.append(float(d))
                break
        else:
            d_stars.append(np.nan)
    ds_arr = np.array([v for v in d_stars if v == v])

    # 3f τ 敏感性（真实序列）
    tau_rows = []
    for tau in [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]:
        f_, wd = frac_diff_ffd(logp, d_star, tau=tau)
        st, pp = adf_stat_p(f_)
        mk = ~np.isnan(f_)
        cr = float(np.corrcoef(f_[mk], logp[mk])[0, 1]) if mk.sum() > 200 else np.nan
        tau_rows.append(dict(tau=tau, width=int(wd), adf=float(st), p=float(pp), corr=cr,
                             sharpe_oos=eval_z(f_, "t")["sharpe_oos"],
                             lost_pct=float(100.0 * min(wd - 1, N) / N)))

    # 3g FFD vs 扩张窗口
    exp_star = frac_diff_expanding(logp, d_star)
    adf_exp, p_exp = adf_stat_p(exp_star)
    both = (~np.isnan(exp_star)) & (~np.isnan(ffd_star))
    corr_ffd_exp = float(np.corrcoef(exp_star[both], ffd_star[both])[0, 1])
    ev, fv = exp_star[~np.isnan(exp_star)], ffd_star[m_s]
    std_ratio_exp = float(ev[len(ev) // 2:].std() / ev[:len(ev) // 2].std())
    std_ratio_ffd = float(fv[len(fv) // 2:].std() / fv[:len(fv) // 2].std())
    z_exp = eval_z(exp_star, "expanding")

    m_x = m_s & (~np.isnan(x_true))
    corr_ffd_x = float(np.corrcoef(ffd_star[m_x], x_true[m_x])[0, 1])
    corr_d1_x = float(np.corrcoef(ffd_d1[m_1], x_true[m_1])[0, 1])
    corr_d0_x = float(np.corrcoef(logp, x_true)[0, 1])

    # ================= 绘图 =================
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.4), sharex=True,
                           gridspec_kw={"height_ratios": [1.2, 1, 1]})
    ax[0].plot(t_axis, logp, color=C_A, lw=0.9,
               label=f"原始对数价格 d=0（ADF p={p_d0:.2f}，不平稳）")
    ax[0].set_ylabel("对数价格"); ax[0].legend(loc="upper left", fontsize=9)
    ax[0].set_title(f"分数阶差分：d={d_star} 处刚通过 ADF，同时保住 {corr_star:.1%} 记忆",
                    fontsize=12.5, fontweight="bold")
    ax[1].plot(t_axis, ffd_star, color=C_C, lw=0.75,
               label=f"FFD d={d_star}（ADF p={p_star:.2e}，与原序列相关 {corr_star:.3f}）")
    ax[1].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[1].set_ylabel("FFD 取值"); ax[1].legend(loc="upper left", fontsize=8.8)
    ax[2].plot(t_axis, ffd_d1, color=C_B, lw=0.5, alpha=0.75,
               label=f"一阶差分 d=1（平稳但记忆归零，相关 {corr_d1:.3f}）")
    ax[2].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[2].set_ylabel("日收益"); ax[2].set_xlabel("交易日")
    ax[2].legend(loc="upper left", fontsize=8.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/cover.png"); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for d, c in zip([0.1, 0.3, d_star, 0.7, 1.0], ["#0ea5e9", C_C, C_D, "#9333ea", C_B]):
        ax[0].plot(range(50), frac_weights(d, 50), marker="o", ms=2.6, lw=1.1, color=c,
                   label=f"d={d}")
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("滞后阶数 k"); ax[0].set_ylabel("权重 $w_k$")
    ax[0].set_title("权重衰减：d 越小尾巴越长、记忆越多", fontsize=11)
    ax[0].legend(fontsize=8.5)
    ds_g = np.round(np.arange(0.05, 1.001, 0.05), 2)
    ax[1].plot(ds_g, [len(frac_weights_ffd(d)) for d in ds_g], marker="o", ms=4.2, color=C_A)
    ax[1].axvline(d_star, color=C_B, ls="--", lw=1.3, label=f"d*={d_star} → 窗宽 {width_star}")
    ax[1].axhline(N, color="#64748b", ls=":", lw=1.2, label=f"样本长度 N={N}")
    ax[1].set_yscale("log"); ax[1].set_xlabel("差分阶数 d")
    ax[1].set_ylabel("FFD 窗宽（对数轴）")
    ax[1].set_title(f"τ={TAU:g} 下窗宽随 d 爆炸：小 d 直接吃光样本", fontsize=11)
    ax[1].legend(fontsize=8.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/weights_decay.png"); plt.close()

    ok_rows = [r for r in rows if r["usable"] and r["adf"] == r["adf"]]
    fig, ax1 = plt.subplots(figsize=(10.5, 5))
    ax1.plot([r["d"] for r in ok_rows], [r["adf"] for r in ok_rows], marker="o", ms=4.5,
             color=C_A, label="ADF 统计量")
    ax1.axhline(adf_5pct, color=C_B, ls="--", lw=1.3, label=f"5% 临界值 {adf_5pct:.2f}")
    ax1.axvline(d_star, color=C_D, ls=":", lw=2.0, label=f"最小可用 d*={d_star}")
    ax1.set_xlabel("差分阶数 d"); ax1.set_ylabel("ADF 统计量（越负越平稳）", color=C_A)
    ax1.tick_params(axis="y", labelcolor=C_A)
    ax2 = ax1.twinx(); ax2.grid(False)
    ax2.plot([r["d"] for r in ok_rows], [r["corr"] for r in ok_rows], marker="s", ms=4.2,
             color=C_C, label="与原序列相关（记忆保留）")
    ax2.set_ylabel("与原对数价格的相关系数", color=C_C)
    ax2.tick_params(axis="y", labelcolor=C_C); ax2.set_ylim(-0.05, 1.05)
    h1, l1 = ax1.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="center right", fontsize=9)
    ax1.set_title(f"平稳性 vs 记忆：d*={d_star} 保住 {corr_star:.1%}，d=1 只剩 {corr_d1:.1%}",
                  fontsize=12, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/adf_memory_tradeoff.png"); plt.close()

    labels = ["d=0", f"d={d_star}", "d=1", "oracle"]
    colors = [C_B, C_C, C_D, "#64748b"]
    ol = [ols["d0"], ols["dstar"], ols["d1"], ols["oracle"]]
    zl = [zev["d0"], zev["dstar"], zev["d1"], zev["oracle"]]
    xp = np.arange(4)
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 4.3))
    ax[0].bar(xp - 0.2, [e["r2_in"] * 100 for e in ol], 0.4, color=colors, alpha=0.55,
              label="样本内 R²")
    ax[0].bar(xp + 0.2, [e["r2_oos"] * 100 for e in ol], 0.4, color=colors, label="样本外 R²")
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(xp); ax[0].set_xticklabels(labels, fontsize=9.5)
    ax[0].set_ylabel("R² (%)"); ax[0].legend(fontsize=8.5)
    ax[0].set_title("固定系数迁移：只有 oracle 样本外为正", fontsize=10.8)
    ax[1].bar(xp, [e["sharpe_oos"] for e in ol], 0.55, color=colors)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xp); ax[1].set_xticklabels(labels, fontsize=9.5)
    ax[1].set_ylabel("样本外 Sharpe"); ax[1].set_title("OLS 固定系数 · 样本外 Sharpe", fontsize=10.8)
    for i, e in enumerate(ol):
        ax[1].text(i, e["sharpe_oos"], f"{e['sharpe_oos']:.2f}", ha="center",
                   va="bottom" if e["sharpe_oos"] >= 0 else "top", fontsize=9)
    ax[2].bar(xp, [e["sharpe_oos"] for e in zl], 0.55, color=colors)
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_xticks(xp); ax[2].set_xticklabels(labels, fontsize=9.5)
    ax[2].set_ylabel("样本外 Sharpe"); ax[2].set_title("因果滚动 z 之后 · 样本外 Sharpe", fontsize=10.8)
    for i, e in enumerate(zl):
        ax[2].text(i, e["sharpe_oos"], f"{e['sharpe_oos']:.2f}", ha="center",
                   va="bottom" if e["sharpe_oos"] >= 0 else "top", fontsize=9)
    plt.suptitle("d=1 丢记忆、d=0 系数不可迁移，d* 居中但远不到 oracle",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(); plt.savefig(f"{OUT}/predictive_power.png"); plt.close()

    # 伪平稳主图
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ws = [r["width"] for r in rw_sweep]
    rj = [r["reject"] * 100 for r in rw_sweep]
    ax[0].plot(ws, rj, marker="o", ms=6, color=C_B, lw=1.8)
    ax[0].axhline(5, color=C_C, ls="--", lw=1.4, label="名义显著性水平 5%")
    ax[0].set_xscale("log")
    ax[0].set_xlabel("FFD 截断窗宽 W（对数轴）")
    ax[0].set_ylabel("ADF 错误拒绝率 (%)")
    ax[0].set_title(f"纯随机游走上做 d={d_star} 分数阶差分\n窗口越长，ADF 越确信「已平稳」",
                    fontsize=11)
    for w_, r_ in zip(ws, rj):
        ax[0].annotate(f"{r_:.0f}%", (w_, r_), textcoords="offset points",
                       xytext=(0, 7), ha="center", fontsize=8.5)
    ax[0].legend(fontsize=8.8)
    ax[1].hist(ds_arr, bins=np.arange(-0.025, 1.05, 0.05), color="#94a3b8", edgecolor="white")
    ax[1].axvline(d_star, color=C_B, lw=2.0, label=f"本文样本 d*={d_star}")
    ax[1].axvline(float(np.median(ds_arr)), color=C_D, ls="--", lw=1.6,
                  label=f"20 样本中位 {np.median(ds_arr):.2f}")
    ax[1].set_xlabel("最小可用 d*"); ax[1].set_ylabel("频数")
    ax[1].set_title(f"同一数据生成过程换 20 个种子\nd* 从 {ds_arr.min():.2f} 摆到 {ds_arr.max():.2f}",
                    fontsize=11)
    ax[1].legend(fontsize=8.8)
    plt.tight_layout(); plt.savefig(f"{OUT}/spurious_stationarity.png"); plt.close()

    # 实现差异 + 对抗
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.3))
    ax[0].plot(t_axis, exp_star, color=C_B, lw=0.7,
               label=f"扩张窗口（后半/前半 std={std_ratio_exp:.2f}）")
    ax[0].plot(t_axis, ffd_star, color=C_C, lw=0.7, alpha=0.85,
               label=f"固定宽度 FFD（后半/前半 std={std_ratio_ffd:.2f}）")
    ax[0].axhline(0, color="k", lw=0.6, alpha=0.4)
    ax[0].set_xlabel("交易日"); ax[0].set_ylabel("差分后取值")
    ax[0].set_title(f"同一个 d={d_star}，两种实现相关仅 {corr_ffd_exp:.2f}", fontsize=11)
    ax[0].legend(fontsize=8.2)
    grp = ["真实序列\n（含 OU 记忆）", "纯随机游走\n（无可预测成分）"]
    v_c = [real_sh, rw_z["sharpe_oos"]]
    v_l = [sh_la_real, sh_la_rw]
    xq = np.arange(2)
    ax[1].bar(xq - 0.2, v_c, 0.4, color=C_C, label="因果滚动 z")
    ax[1].bar(xq + 0.2, v_l, 0.4, color=C_B, label="全样本 z（look-ahead）")
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].set_xticks(xq); ax[1].set_xticklabels(grp, fontsize=9)
    ax[1].set_ylabel("样本外 Sharpe")
    ax[1].set_title("随机游走上不该有 alpha", fontsize=11)
    ax[1].legend(fontsize=8.2)
    for i, (a_, b_) in enumerate(zip(v_c, v_l)):
        ax[1].text(i - 0.2, a_, f"{a_:.2f}", ha="center", va="bottom" if a_ >= 0 else "top", fontsize=8.8)
        ax[1].text(i + 0.2, b_, f"{b_:.2f}", ha="center", va="bottom" if b_ >= 0 else "top", fontsize=8.8)
    ax[2].hist(perm, bins=40, color="#94a3b8", edgecolor="white")
    ax[2].axvline(float(np.percentile(perm, 95)), color=C_D, ls="--", lw=1.6,
                  label=f"置换 95 分位 {np.percentile(perm,95):.2f}")
    ax[2].axvline(real_sh, color=C_B, lw=2.2, label=f"真实 {real_sh:.2f}（p={perm_p:.3f}）")
    ax[2].set_xlabel("样本外 Sharpe"); ax[2].set_ylabel("频数")
    ax[2].set_title(f"打乱收益顺序 300 次（最大 {perm.max():.2f}）", fontsize=11)
    ax[2].legend(fontsize=8.2)
    plt.tight_layout(); plt.savefig(f"{OUT}/adversarial_checks.png"); plt.close()

    stats = dict(
        seed=SEED, N=N, tau=TAU, split=SPLIT, half_life=float(half_life),
        adf_5pct=float(adf_5pct),
        d0=dict(adf=float(adf_d0), p=float(p_d0), corr=1.0),
        d_star=dict(d=d_star, adf=float(adf_star), p=float(p_star), corr=corr_star,
                    width=int(width_star)),
        d1=dict(adf=float(adf_d1), p=float(p_d1), corr=corr_d1, width=int(w_d1)),
        grid=rows, eval_ols=ols, eval_z=zev,
        rw_width_sweep=rw_sweep,
        adversarial=dict(rw_ffd_adf_p=float(p_rw), rw_z=rw_z, rw_ols=rw_ols,
                         sharpe_lookahead_real=sh_la_real, sharpe_lookahead_rw=sh_la_rw,
                         perm_p95=float(np.percentile(perm, 95)), perm_max=float(perm.max()),
                         perm_mean=float(perm.mean()), perm_pvalue=perm_p,
                         corr_ffd_xtrue=corr_ffd_x, corr_d1_xtrue=corr_d1_x,
                         corr_d0_xtrue=corr_d0_x),
        d_star_stability=dict(values=[float(v) for v in ds_arr], median=float(np.median(ds_arr)),
                              min=float(ds_arr.min()), max=float(ds_arr.max()),
                              std=float(ds_arr.std())),
        tau_sensitivity=tau_rows,
        expanding=dict(adf=float(adf_exp), p=float(p_exp), corr_with_ffd=corr_ffd_exp,
                       std_ratio_exp=std_ratio_exp, std_ratio_ffd=std_ratio_ffd,
                       sharpe_oos=z_exp["sharpe_oos"]),
    )
    with open(f"{OUT}/stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(json.dumps(stats, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
