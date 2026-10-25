#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct25f: 10 figures + key stats for
A) rolling-cointegration-bre  滚动窗口协整：用 DCC 抓住会『断』的长期关系
B) low-volatility-anomaly-cn  低波动异象：低风险股票为何长期跑赢高风险
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy import optimize

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
OUT_A = os.path.join(BASE, "rolling-cointegration-bre")
OUT_B = os.path.join(BASE, "low-volatility-anomaly-cn")
os.makedirs(OUT_A, exist_ok=True)
os.makedirs(OUT_B, exist_ok=True)

RED, GREEN, BLUE, GRAY, ORANGE, BLACK = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436"
CV = -2.86  # ADF 5% 临界值（含常数项，大样本 MacKinnon 近似）


def roll_mean(x, w):
    c = np.cumsum(np.insert(x, 0, 0.0))
    return (c[w:] - c[:-w]) / w


def roll_std(x, w):
    m = roll_mean(x, w)
    m2 = roll_mean(x ** 2, w)
    return np.sqrt(np.maximum(m2 - m ** 2, 1e-12))


def zscore(x, w):
    """返回 (idx, z)：idx 为窗口结束日（信号当日收盘可用）。"""
    mu, sd = roll_mean(x, w), roll_std(x, w)
    idx = np.arange(w - 1, len(x))
    return idx, (x[idx] - mu) / sd


def roll_corr(x, y, w):
    """滚动 Pearson 相关（必须减窗口均值乘积，否则窗口均值大时会算出 >1 的假相关）。"""
    mx, my = roll_mean(x, w), roll_mean(y, w)
    sx, sy = roll_std(x, w), roll_std(y, w)
    cxy = roll_mean(x * y, w)
    return (cxy - mx * my) / (sx * sy)


def ols(y, x):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def adf(resid, maxlag=8):
    """ADF t 统计量（BIC 选滞后阶）。"""
    d = np.diff(resid)
    n0 = len(d)
    best = (np.inf, 0.0, 0)
    for lag in range(0, maxlag + 1):
        n = n0 - lag
        if n < 40:
            break
        y = d[lag:]
        cols = [np.ones(n), resid[lag:n0]]
        for i in range(1, lag + 1):
            cols.append(d[lag - i:n0 - i])
        X = np.column_stack(cols)
        XtXi = np.linalg.pinv(X.T @ X)
        b = XtXi @ (X.T @ y)
        e = y - X @ b
        dof = max(n - X.shape[1], 1)
        s2 = e @ e / dof
        tg = b[1] / np.sqrt(s2 * XtXi[1, 1])
        bic = n * np.log(s2) + X.shape[1] * np.log(n)
        if bic < best[0]:
            best = (bic, tg, lag)
    return best[1], best[2]


def garch_fit(x):
    """方差目标化 GARCH(1,1) QMLE，返回 (标准化残差, (a,b), 条件波动)。"""
    v = np.var(x)

    def nll(theta):
        a, b = theta
        if a <= 1e-6 or b <= 1e-6 or a + b >= 0.999:
            return 1e10
        s2 = np.empty_like(x)
        s2[0] = v
        for t in range(1, len(x)):
            s2[t] = (1 - a - b) * v + a * x[t - 1] ** 2 + b * s2[t - 1]
        return 0.5 * np.sum(np.log(2 * np.pi * s2) + x ** 2 / s2)

    res = optimize.minimize(nll, [0.08, 0.90], method="Nelder-Mead",
                            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-6})
    a, b = res.x
    s2 = np.empty_like(x)
    s2[0] = v
    for t in range(1, len(x)):
        s2[t] = (1 - a - b) * v + a * x[t - 1] ** 2 + b * s2[t - 1]
    return x / np.sqrt(s2), (a, b), np.sqrt(s2)


def dcc_corr(za, zb):
    """DCC(1,1) 条件相关序列。"""
    r0 = np.corrcoef(za, zb)[0, 1]
    Qbar = np.array([[1.0, r0], [r0, 1.0]])
    Z = np.column_stack([za, zb])
    n = len(Z)

    def nll(theta):
        a, b = theta
        if a <= 1e-6 or b <= 1e-6 or a + b >= 0.999:
            return 1e10
        Q = Qbar.copy()
        ll = 0.0
        for t in range(1, n):
            Qt = (1 - a - b) * Qbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
            det = Qt[0, 0] * Qt[1, 1] - Qt[0, 1] ** 2
            inv_zz = (Z[t, 0] ** 2 * Qt[1, 1] - 2 * Z[t, 0] * Z[t, 1] * Qt[0, 1]
                      + Z[t, 1] ** 2 * Qt[0, 0]) / det
            ll += 0.5 * (np.log(det) + inv_zz - Z[t] @ Z[t])
            Q = Qt
        return ll

    res = optimize.minimize(nll, [0.02, 0.97], method="Nelder-Mead",
                            options={"maxiter": 200, "xatol": 1e-4, "fatol": 1e-5})
    a, b = res.x
    Q = Qbar.copy()
    rho = np.full(n, r0)
    for t in range(1, n):
        Qt = (1 - a - b) * Qbar + a * np.outer(Z[t - 1], Z[t - 1]) + b * Q
        rho[t] = Qt[0, 1] / np.sqrt(Qt[0, 0] * Qt[1, 1])
        Q = Qt
    return rho, (a, b)


def max_dd(cum):
    peak = np.maximum.accumulate(cum)
    return float(np.min(cum / peak - 1.0))


def roll_sr(pnl, w=250):
    out = np.full(len(pnl) - w, np.nan)
    for i in range(len(out)):
        seg = pnl[i:i + w]
        out[i] = seg.mean() / seg.std() * np.sqrt(252)
    return out


def synth_pair(seed, T=3000, break_at=1500, ramp=20, beta_lo=0.80, beta_hi=1.30,
               kappa=0.22, s_eq=0.004):
    """合成协整对。break_at=None 表示无断点（对照组）。"""
    rng = np.random.default_rng(seed)
    t_idx = np.arange(T)
    rA = rng.normal(0.0004, 0.011, T)
    logA = np.cumsum(rA)
    spread = np.zeros(T)
    for t in range(1, T):
        spread[t] = spread[t - 1] + kappa * (0.0 - spread[t - 1]) + rng.normal(0, s_eq)
    if break_at is None:
        beta_true = np.full(T, beta_lo)
    else:
        beta_true = np.full(T, beta_lo)
        seg = np.arange(ramp)
        beta_true[break_at:break_at + ramp] = beta_lo + (beta_hi - beta_lo) * (seg + 1) / ramp
        beta_true[break_at + ramp:] = beta_hi
    logB = beta_true * logA + spread
    rB = np.diff(logB)
    return t_idx, logA, logB, rA, rB, beta_true


def run_pair(t_idx, logA, logB, rA, rB, beta_true, T, BREAK, W_ADF=250, tag=""):
    rA1 = rA[1:]
    # E1 全样本 EG
    coef_full, resid_full = ols(logB, logA)
    adf_full, _ = adf(resid_full)
    # 冻结 beta（前 1000 日估计，样本内）
    beta_frz = float(ols(logB[:1000], logA[:1000])[0][1])
    resid_frz = logB - beta_frz * logA
    # 滚动 ADF（冻结 beta 残差，250 日窗口）：断点后【无法】拒绝单位根 = 协整断裂
    ends = np.array(list(range(W_ADF, T, 5)))
    roll_adf = np.array([adf(resid_frz[e - W_ADF:e])[0] for e in ends])
    det = ends[(ends > BREAK) & (roll_adf > CV)]
    lag_det = int(det[0] - BREAK) if len(det) else -1
    fp_pre = int((roll_adf[ends < BREAK] > CV).sum())
    n_pre = int((ends < BREAK).sum())
    # 滚动 OLS beta（两种窗口）
    def roll_beta(w):
        b = np.full(T, beta_frz)
        for t in range(w, T):
            b[t] = ols(logB[t - w:t], logA[t - w:t])[0][1]
        return b
    beta_roll120 = roll_beta(120)
    beta_roll250 = roll_beta(250)
    def rmse(b, sl):
        return float(np.sqrt(np.mean((b[sl] - beta_true[sl]) ** 2)))
    post = slice(T // 2, T)
    out = dict(adf_full=adf_full, coef_full=float(coef_full[1]), beta_frz=beta_frz,
               lag_det=lag_det, roll_adf=roll_adf, ends=ends, resid_frz=resid_frz,
               beta_roll120=beta_roll120, beta_roll250=beta_roll250,
               rmse_static=rmse(beta_frz + np.zeros(T), slice(0, T)),
               rmse_static_post=rmse(beta_frz + np.zeros(T), post),
               rmse_120=rmse(beta_roll120, slice(120, T)), rmse_120_post=rmse(beta_roll120, post),
               rmse_250=rmse(beta_roll250, slice(250, T)), rmse_250_post=rmse(beta_roll250, post))
    print(f"[A{tag}] 全样本 EG: beta={coef_full[1]:.3f}, ADF t={adf_full:.2f}; "
          f"冻结 beta(前1000日)={beta_frz:.3f}; 检测滞后={lag_det}d; 断点前误报 {fp_pre}/{n_pre}")
    print(f"[A{tag}] RMSE β: 静态全样本 {out['rmse_static']:.3f} 断点后 {out['rmse_static_post']:.3f}; "
          f"滚动120 {out['rmse_120']:.3f}/{out['rmse_120_post']:.3f}; "
          f"滚动250 {out['rmse_250']:.3f}/{out['rmse_250_post']:.3f}")
    print(f"[A{tag}] 滚动ADF 断点前中位 {np.median(roll_adf[ends < BREAK]):.2f} -> "
          f"断点后中位 {np.median(roll_adf[ends > BREAK + 100]):.2f}")
    return out


COST = 0.0005


def backtest(logB, logA, rA, rB, beta_hat, T):
    spread_h = logB - beta_hat * logA
    idx_z, z_arr = zscore(spread_h, 60)
    z = np.full(T, np.nan)
    z[idx_z] = z_arr
    pos = np.zeros(T)
    pnl = np.zeros(T)
    cur = 0
    for t in range(1, T):
        zt = z[t - 1]
        if not np.isnan(zt):
            if cur == 0:
                if zt < -2:
                    cur = 1
                elif zt > 2:
                    cur = -1
            elif abs(zt) < 0.5:
                cur = 0
        pos[t] = cur
        d = abs(cur - pos[t - 1])
        pnl[t] = cur * (rB[t - 1] - beta_hat[t] * rA[t]) - COST * d * (1 + beta_hat[t])
    cum = np.cumprod(1 + pnl)
    return pos, cum, pnl


def bt_stats(pos, cum, pnl, T, BREAK, label):
    post = slice(BREAK, T)
    return {"label": label, "total": cum[-1] - 1,
            "sharpe": pnl[1:].mean() / pnl[1:].std() * np.sqrt(252),
            "mdd": max_dd(cum), "trades": int((np.abs(np.diff(pos)) > 0).sum()),
            "post_total": cum[-1] / cum[BREAK] - 1,
            "post_sharpe": pnl[post].mean() / pnl[post].std() * np.sqrt(252),
            "post_mdd": max_dd(cum[BREAK:]),
            "winrate": float((pnl[pos != 0] > 0).mean())}


# ============================================================
# Article A: rolling-cointegration-bre
# ============================================================
print("=" * 70)
print("ARTICLE A: rolling-cointegration-bre")
print("=" * 70)

T, BREAK, W_ROLL = 3000, 1500, 120
t_idx, logA, logB, rA, rB, beta_true = synth_pair(42, break_at=BREAK)
info = run_pair(t_idx, logA, logB, rA, rB, beta_true, T, BREAK, tag="(断点世界)")
beta_frz, beta_roll = info["beta_frz"], info["beta_roll120"]

# GARCH + DCC
za, (ga, gb), _ = garch_fit(rA[1:])
zb, (gc, gd), _ = garch_fit(rB)
rho_dcc, (da_, db_) = dcc_corr(za, zb)
rho_roll60 = roll_corr(rA[1:], rB, 60)
rho_pre_dcc = float(rho_dcc[1000:1500].mean())
rho_post_dcc = float(rho_dcc[1600:2500].mean())
corr_direct_pre = float(np.corrcoef(rA[1:][1000:1500], rB[1000:1500])[0, 1])
corr_direct_post = float(np.corrcoef(rA[1:][1600:2500], rB[1600:2500])[0, 1])
# 错误公式（不减均值乘积）的对照：窗口均值大时系统性高估
rho_naive = roll_mean(rA[1:] * rB, 60) / (roll_std(rA[1:], 60) * roll_std(rB, 60))
print(f"[A] GARCH: A(a={ga:.3f},b={gb:.3f}) B(a={gc:.3f},b={gd:.3f}); DCC(a={da_:.3f},b={db_:.3f})")
print(f"[A] DCC rho 前 {rho_pre_dcc:.3f} -> 后 {rho_post_dcc:.3f}; 滚动60d corr(正确) 前 "
      f"{float(rho_roll60[1000:1500].mean()):.3f} -> 后 {float(rho_roll60[1600:2500].mean()):.3f}; "
      f"直接 corr 前 {corr_direct_pre:.3f} -> 后 {corr_direct_post:.3f}")
print(f"[A] 偷懒公式 mean(ab)/(sa*sb): 前 {float(rho_naive[1000:1500].mean()):.3f} -> "
      f"后 {float(rho_naive[1600:2500].mean()):.3f}, max={float(rho_naive.max()):.3f} (>1 即穿帮)")
print(f"[A] 窗口内波动: rA {rA[1:][1600:2500].std():.4f} rB {rB[1600:2500].std():.4f} (日)")

# 回测：静态 / 滚动 / Oracle
pos_f, cum_f, pnl_f = backtest(logB, logA, rA, rB, np.full(T, beta_frz), T)
pos_r, cum_r, pnl_r = backtest(logB, logA, rA, rB, beta_roll, T)
pos_o, cum_o, pnl_o = backtest(logB, logA, rA, rB, beta_true, T)
s_f = bt_stats(pos_f, cum_f, pnl_f, T, BREAK, "静态")
s_r = bt_stats(pos_r, cum_r, pnl_r, T, BREAK, "滚动120d")
s_o = bt_stats(pos_o, cum_o, pnl_o, T, BREAK, "Oracle")
cum_bh = np.cumprod(1 + 0.5 * (rA[1:] + rB))
for s in (s_f, s_r, s_o):
    print(f"[A] {s['label']}: total={s['total']:+.2%} SR={s['sharpe']:.2f} DD={s['mdd']:.2%} "
          f"trades={s['trades']} post={s['post_total']:+.2%} postSR={s['post_sharpe']:.2f} "
          f"postDD={s['post_mdd']:.2%} win={s['winrate']:.1%}")
bh_post = cum_bh[-1] / cum_bh[BREAK - 1] - 1
print(f"[A] buy&hold 等权: total={cum_bh[-1]-1:+.2%} post={bh_post:+.2%}")

# 对照组：无断点世界
t_idx2, logA2, logB2, rA2, rB2, beta_true2 = synth_pair(7, break_at=None)
info2 = run_pair(t_idx2, logA2, logB2, rA2, rB2, beta_true2, T, BREAK, tag="(无断点对照)")
beta_frz2, beta_roll2 = info2["beta_frz"], info2["beta_roll120"]
pos_f2, cum_f2, pnl_f2 = backtest(logB2, logA2, rA2, rB2, np.full(T, beta_frz2), T)
pos_r2, cum_r2, pnl_r2 = backtest(logB2, logA2, rA2, rB2, beta_roll2, T)
pos_o2, cum_o2, pnl_o2 = backtest(logB2, logA2, rA2, rB2, beta_true2, T)
s_f2 = bt_stats(pos_f2, cum_f2, pnl_f2, T, BREAK, "静态")
s_r2 = bt_stats(pos_r2, cum_r2, pnl_r2, T, BREAK, "滚动120d")
s_o2 = bt_stats(pos_o2, cum_o2, pnl_o2, T, BREAK, "Oracle")
for s in (s_f2, s_r2, s_o2):
    print(f"[A-对照] {s['label']}: total={s['total']:+.2%} SR={s['sharpe']:.2f} DD={s['mdd']:.2%} "
          f"trades={s['trades']} win={s['winrate']:.1%}")

# ---- 图 A1: 价格路径 + 对冲比 + 价差 ----
fig, axes = plt.subplots(3, 1, figsize=(11, 9.5), sharex=True)
ax = axes[0]
ax.plot(t_idx, np.exp(logA), color=BLUE, lw=1.1, label="资产 A")
ax.plot(t_idx, np.exp(logB), color=RED, lw=1.1, label="资产 B")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.text(BREAK + 20, np.exp(logA).max() * 0.6, "协整关系断裂\nβ: 0.80 → 1.30（20 日渐进）", fontsize=9, color=GRAY)
ax.set_title("合成协整对：3000 个交易日，断点 t=1500 处对冲比渐进跳升", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("价格")
ax = axes[1]
ax.plot(t_idx, beta_true, color=BLACK, lw=2, label="真实 β_t")
ax.plot(t_idx, beta_roll, color=ORANGE, lw=1.1, label="滚动 120 日 OLS")
ax.axhline(beta_frz, color=RED, lw=1.1, ls="--", label=f"静态估计 β={beta_frz:.2f}（前 1000 日，冻结）")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_ylim(0.6, 1.5); ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3)
ax.set_ylabel("对冲比 β"); ax.set_title("对冲比跟踪：静态估计断点后永久偏低约 0.5", fontsize=11)
ax = axes[2]
idx_zr, zr = zscore(logB - beta_roll * logA, 60)
idx_zs, zs = zscore(logB - beta_frz * logA, 60)
ax.plot(idx_zs, zs, color=RED, lw=0.8, alpha=0.8, label="冻结 β 价差 z-score（漂成趋势）")
ax.plot(idx_zr, zr, color=BLUE, lw=0.8, label="滚动 β 价差 z-score（回归均值）")
ax.axhline(2, color=GRAY, lw=0.8, ls=":"); ax.axhline(-2, color=GRAY, lw=0.8, ls=":")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_title("价差 z-score：旧关系的价差断点后不再回复", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("z"); ax.set_xlabel("交易日")
fig.tight_layout()
fig.savefig(os.path.join(OUT_A, "price_paths_break.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 A2: 冻结 beta 残差的滚动 ADF ----
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax = axes[0]
ax.plot(info["ends"], info["roll_adf"], color=BLUE, lw=1.2, label="滚动 120 日 ADF t 值（冻结 β 残差）")
ax.axhline(CV, color=RED, ls="--", lw=1.2, label=f"5% 临界值 {CV}")
ax.axhline(info["adf_full"], color=ORANGE, ls=":", lw=1.2,
           label=f"全样本 ADF t={info['adf_full']:.1f}（仍拒绝单位根）")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
if info["lag_det"] > 0:
    ax.axvline(BREAK + info["lag_det"], color=RED, ls="-.", lw=1)
    ax.annotate(f"协整消失被确认：滞后 {info['lag_det']} 天", xy=(BREAK + info["lag_det"], CV),
                xytext=(BREAK + info["lag_det"] + 150, CV + 1.0), fontsize=9, color=RED,
                arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("滚动 ADF：断点前深度负值（协整成立）→ 断点后升破临界值（无法拒绝单位根）", fontsize=11)
ax.legend(fontsize=9, ncol=3); ax.grid(alpha=0.3); ax.set_ylabel("ADF t")
ax = axes[1]
p_approx = np.where(info["roll_adf"] < -3.43, 0.01, np.where(info["roll_adf"] < CV, 0.05, 0.10))
ax.plot(info["ends"], p_approx, color=BLUE, lw=1.0, drawstyle="steps-post")
ax.axhline(0.05, color=RED, ls="--", lw=1)
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_ylabel("近似 p 值（分档）"); ax.set_xlabel("窗口结束日 t")
ax.set_title("p 值分档示意（0.10 / 0.05 / 0.01）：断点后稳定在 0.10", fontsize=11); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_A, "rolling_adf.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 A3: 跟踪误差 + DCC ----
fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True)
ax = axes[0]
ax.plot(t_idx, np.abs(beta_roll - beta_true), color=ORANGE, lw=1.0, label="滚动 120 日：|β̂−β|")
ax.plot(t_idx, np.abs(beta_frz - beta_true), color=RED, lw=1.0, alpha=0.8, label="静态冻结：|β̂−β|")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_title(f"对冲比绝对误差：静态断点后永久约 0.5；滚动120 RMSE {info['rmse_120']:.3f}"
             f"（滚动250 {info['rmse_250']:.3f}，静态 {info['rmse_static']:.3f}）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("|β̂ − β|")
ax = axes[1]
ax.plot(np.arange(1, T), rho_dcc, color=BLUE, lw=1.2, label=f"DCC(1,1) ρ_t（a={da_:.2f}, b={db_:.2f}）")
ax.plot(np.arange(60, 60 + len(rho_roll60)), rho_roll60, color=ORANGE, lw=0.9, alpha=0.85,
        label="滚动 60 日相关（正确去均值）")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_title(f"条件相关：断点前 {rho_pre_dcc:.3f} → 断点后 {rho_post_dcc:.3f}"
             "（相关矩阵只给弱信号，强信号在残差漂移）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("相关系数"); ax.set_xlabel("交易日")
fig.tight_layout()
fig.savefig(os.path.join(OUT_A, "hedge_ratio_tracking.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 A4: 回测权益曲线 ----
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(t_idx, cum_o, color=BLUE, lw=1.6, label=f"Oracle（真值 β）SR={s_o['sharpe']:.2f}")
ax.plot(t_idx, cum_r, color=ORANGE, lw=1.4, label=f"滚动 120d SR={s_r['sharpe']:.2f}")
ax.plot(t_idx, cum_f, color=RED, lw=1.4, label=f"静态冻结 SR={s_f['sharpe']:.2f}")
ax.plot(t_idx[1:], cum_bh, color=GRAY, lw=1.1, ls="--", label="buy&hold 等权")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.axvspan(BREAK, T, color=RED, alpha=0.05)
ax.text(BREAK + 20, cum_o[-1] * 0.9, "断点后", fontsize=9, color=GRAY)
ax.set_yscale("log")
ax.set_title("配对交易净值（对数轴，单边 5bps 成本）：静态对冲断点后持续流血", fontsize=11)
ax.legend(fontsize=9, loc="upper left"); ax.grid(alpha=0.3); ax.set_ylabel("净值"); ax.set_xlabel("交易日")
fig.tight_layout()
fig.savefig(os.path.join(OUT_A, "backtest_equity.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 A5: z-score 交易 + 指标 + 滚动 Sharpe ----
fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.2))
ax = axes[0]
ax.plot(idx_zr, zr, color=BLUE, lw=0.7)
buy = (pos_r == 1); sell = (pos_r == -1)
ax.scatter(t_idx[buy], zr[np.searchsorted(idx_zr, t_idx[buy])], s=8, color=RED, label="做多价差")
ax.scatter(t_idx[sell], zr[np.searchsorted(idx_zr, t_idx[sell])], s=8, color=GREEN, label="做空价差")
ax.axhline(2, color=GRAY, ls=":", lw=0.8); ax.axhline(-2, color=GRAY, ls=":", lw=0.8)
ax.axvline(BREAK, color=GRAY, ls="--", lw=1)
ax.set_title("滚动对冲：z-score 与开平仓点", fontsize=10); ax.legend(fontsize=8); ax.grid(alpha=0.3)
ax = axes[1]
labels = ["静态", "滚动120d", "Oracle"]
sh = [s_f["sharpe"], s_r["sharpe"], s_o["sharpe"]]
dd = [-s_f["mdd"], -s_r["mdd"], -s_o["mdd"]]
xpos = np.arange(3)
b1 = ax.bar(xpos - 0.2, sh, 0.4, color=BLUE, label="Sharpe")
b2 = ax.bar(xpos + 0.2, dd, 0.4, color=ORANGE, label="最大回撤(取正)")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.03, f"{b.get_height():.2f}",
            ha="center", fontsize=8.5)
ax.set_xticks(xpos); ax.set_xticklabels(labels, fontsize=9)
ax.set_title("全样本：Sharpe 与最大回撤", fontsize=10); ax.legend(fontsize=8.5); ax.grid(alpha=0.3, axis="y")
ax = axes[2]
ax.plot(np.arange(250, T), roll_sr(pnl_f), color=RED, lw=1.0, label="静态")
ax.plot(np.arange(250, T), roll_sr(pnl_r), color=ORANGE, lw=1.0, label="滚动120d")
ax.plot(np.arange(250, T), roll_sr(pnl_o), color=BLUE, lw=1.0, label="Oracle")
ax.axvline(BREAK, color=GRAY, ls="--", lw=1); ax.axhline(0, color=GRAY, lw=0.6)
ax.set_title("滚动 250 日 Sharpe", fontsize=10); ax.legend(fontsize=8.5); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_A, "signal_attribution.png"), bbox_inches="tight")
plt.close(fig)
print("[A] figures saved:", sorted(os.listdir(OUT_A)))

# ============================================================
# Article B: low-volatility-anomaly-cn
# ============================================================
print("=" * 70)
print("ARTICLE B: low-volatility-anomaly-cn")
print("=" * 70)

rngB = np.random.default_rng(7)
N, TB = 500, 2500
blocks = [(0, 500, 0.0006, 0.009), (500, 900, -0.0004, 0.016),
          (900, 1400, 0.0007, 0.008), (1400, 1900, -0.0003, 0.018), (1900, 2500, 0.0005, 0.010)]
mkt = np.zeros(TB)
for a, b, mu, sd in blocks:
    mkt[a:b] = rngB.normal(mu, sd, b - a)

beta_t = rngB.uniform(0.6, 1.7, N)
idio = 0.003 + np.abs(rngB.normal(0.006, 0.0025, N))
alpha_beta = 0.00018 - 0.00022 * (beta_t - 1.15)     # 杠杆约束通道（日频）
alpha_idio = -0.012 * (idio - 0.009)                 # 彩票偏好通道（日频）
alpha_noise = rngB.normal(0, 0.00006, N)
alpha_t = alpha_beta + alpha_idio + alpha_noise
R = alpha_t[None, :] + beta_t[None, :] * mkt[:, None] + rngB.normal(0, 1, (TB, N)) * idio[None, :]

rf_annual = 0.02
vol_r = R.std(axis=0) * np.sqrt(252)
beta_r = np.array([np.cov(R[:, i], mkt)[0, 1] / np.var(mkt) for i in range(N)])
alpha_r = R.mean(axis=0) - beta_r * mkt.mean()
ret_ann = R.mean(axis=0) * 252

# ---- 全样本波动五分位（静态分组） ----
order = np.argsort(vol_r)
qs = np.array_split(order, 5)
q_labels = ["Q1 低波", "Q2", "Q3", "Q4", "Q5 高波"]
qcolors = [BLUE, GREEN, "#fdcb6e", ORANGE, RED]
print("[B] 全样本波动五分位（静态分组等权）：")
qstats = {}
for k, idx in enumerate(qs):
    rq = R[:, idx].mean(axis=1)
    qstats[k] = {"ret": rq.mean() * 252, "vol": rq.std() * np.sqrt(252),
                 "sr": rq.mean() / rq.std() * np.sqrt(252),
                 "mdd": max_dd(np.cumprod(1 + rq)), "beta": float(beta_r[idx].mean()),
                 "idio": float(idio[idx].mean()), "n": len(idx)}
    print(f"  {q_labels[k]}: n={len(idx)} 年化收益={qstats[k]['ret']:+.2%} 波动={qstats[k]['vol']:.2%} "
          f"Sharpe={qstats[k]['sr']:.2f} DD={qstats[k]['mdd']:.2%} β={qstats[k]['beta']:.2f} idio={qstats[k]['idio']:.3%}")

# ---- 月度调仓版本（排序用过去 250 日已实现波动，无前视） ----
mdays = list(range(250, TB, 21))
q_monthly = {k: [] for k in range(5)}
turnover_q1 = []
prev_names = None
for m in mdays:
    win = R[m - 250:m]
    v_win = win.std(axis=0) * np.sqrt(252)
    o = np.argsort(v_win)
    idxs = np.array_split(o, 5)
    if prev_names is not None:
        cur = set(idxs[0].tolist())
        turnover_q1.append(1 - len(cur & prev_names) / len(cur))
    prev_names = set(idxs[0].tolist())
    if m + 21 > TB:
        break
    nxt = R[m:m + 21]
    for k, idx in enumerate(idxs):
        q_monthly[k].append(nxt[:, idx].mean(axis=1).sum())
qm = {k: np.array(v) for k, v in q_monthly.items()}
qstats_m = {}
print("[B] 月度调仓波动五分位（年化=月均×12）：")
for k in range(5):
    r = qm[k]
    qstats_m[k] = {"ret": r.mean() * 12, "vol": r.std() * np.sqrt(12),
                   "sr": r.mean() / r.std() * np.sqrt(12), "mdd": max_dd(np.cumprod(1 + r))}
    print(f"  {q_labels[k]}: 年化={qstats_m[k]['ret']:+.2%} 波动={qstats_m[k]['vol']:.2%} "
          f"Sharpe={qstats_m[k]['sr']:.2f} DD={qstats_m[k]['mdd']:.2%}")
turn = float(np.mean(turnover_q1))
print(f"[B] Q1 组合月均名单更替比例={turn:.1%}")

# ---- BAB 因子（按 β 排序，月度调仓） ----
bab_m, bh_m = [], []
b_L, b_H = [], []
for m in mdays:
    win = R[m - 250:m]
    mk = mkt[m - 250:m]
    b_win = np.array([np.cov(win[:, i], mk)[0, 1] / np.var(mk) for i in range(N)])
    o = np.argsort(b_win)
    lo, hi = o[:100], o[-100:]
    if m + 21 > TB:
        break
    nxt = R[m:m + 21]
    rL = nxt[:, lo].mean(axis=1).sum()
    rH = nxt[:, hi].mean(axis=1).sum()
    rf_m = rf_annual / 12
    bab_m.append((rL - rf_m) / b_win[lo].mean() - (rH - rf_m) / b_win[hi].mean())
    bh_m.append(mkt[m:m + 21].sum())
    b_L.append(b_win[lo].mean()); b_H.append(b_win[hi].mean())
bab_m = np.array(bab_m); bh_m = np.array(bh_m)
bab_stats = {"ret": bab_m.mean() * 12, "vol": bab_m.std() * np.sqrt(12),
             "sr": bab_m.mean() / bab_m.std() * np.sqrt(12), "mdd": max_dd(np.cumprod(1 + bab_m))}
mkt_stats = {"ret": bh_m.mean() * 12, "vol": bh_m.std() * np.sqrt(12),
             "sr": bh_m.mean() / bh_m.std() * np.sqrt(12), "mdd": max_dd(np.cumprod(1 + bh_m))}
print(f"[B] BAB: 年化={bab_stats['ret']:+.2%} 波动={bab_stats['vol']:.2%} SR={bab_stats['sr']:.2f} "
      f"DD={bab_stats['mdd']:.2%}; 低波组均β={np.mean(b_L):.2f} 高波组均β={np.mean(b_H):.2f}")
print(f"[B] 市场: 年化={mkt_stats['ret']:+.2%} 波动={mkt_stats['vol']:.2%} SR={mkt_stats['sr']:.2f} DD={mkt_stats['mdd']:.2%}")

# ---- A股长仓倾斜版（不做空不加杠杆：超配低波、低配高波） ----
tilt_m, eqw_m = [], []
for m in mdays:
    win = R[m - 250:m]
    mk = mkt[m - 250:m]
    b_win = np.array([np.cov(win[:, i], mk)[0, 1] / np.var(mk) for i in range(N)])
    raw = 1.0 / b_win
    w = np.maximum(raw - np.median(raw), 0)
    w = w / w.sum()
    if m + 21 > TB:
        break
    nxt = R[m:m + 21]
    tilt_m.append((nxt * w).sum(axis=1).sum())
    eqw_m.append(nxt.mean(axis=1).sum())
tilt_m, eqw_m = np.array(tilt_m), np.array(eqw_m)
tilt_stats = {"ret": tilt_m.mean() * 12, "vol": tilt_m.std() * np.sqrt(12),
              "sr": tilt_m.mean() / tilt_m.std() * np.sqrt(12), "mdd": max_dd(np.cumprod(1 + tilt_m))}
eqw_stats = {"ret": eqw_m.mean() * 12, "vol": eqw_m.std() * np.sqrt(12),
             "sr": eqw_m.mean() / eqw_m.std() * np.sqrt(12), "mdd": max_dd(np.cumprod(1 + eqw_m))}
print(f"[B] 长仓低波倾斜: 年化={tilt_stats['ret']:+.2%} 波动={tilt_stats['vol']:.2%} "
      f"SR={tilt_stats['sr']:.2f} DD={tilt_stats['mdd']:.2%}")
print(f"[B] 等权基准:     年化={eqw_stats['ret']:+.2%} 波动={eqw_stats['vol']:.2%} "
      f"SR={eqw_stats['sr']:.2f} DD={eqw_stats['mdd']:.2%}")

# ---- 机制分解：alpha ~ beta + idio ----
X = np.column_stack([np.ones(N), beta_t, idio])
XtXi = np.linalg.pinv(X.T @ X)
b_mech = XtXi @ (X.T @ alpha_r)
resid_mech = alpha_r - X @ b_mech
r2_mech = 1 - resid_mech.var() / alpha_r.var()
print(f"[B] 机制分解: alpha = {b_mech[0]:.5f} + ({b_mech[1]:.5f})*beta + ({b_mech[2]:.4f})*idio, R2={r2_mech:.3f}")
d_beta_ch = b_mech[1] * (qstats[4]["beta"] - qstats[0]["beta"]) * 252
d_idio_ch = b_mech[2] * (qstats[4]["idio"] - qstats[0]["idio"]) * 252
d_total = qstats[4]["ret"] - qstats[0]["ret"]
print(f"[B] Q5-Q1 年化收益差 {d_total:+.2%}: beta 通道 {d_beta_ch:+.2%}, idio 通道 {d_idio_ch:+.2%}, "
      f"残差/交互 {d_total - d_beta_ch - d_idio_ch:+.2%}")

# ---- 年度表现 ----
years = TB // 250
y_q1, y_q5 = [], []
for y in range(years):
    seg = slice(y * 250, (y + 1) * 250)
    y_q1.append(R[seg][:, qs[0]].mean(axis=1).mean() * 252)
    y_q5.append(R[seg][:, qs[4]].mean(axis=1).mean() * 252)
print("[B] 分年 Q1 vs Q5:", [f"{a:+.1%}/{b:+.1%}" for a, b in zip(y_q1, y_q5)])

# ---- 成本敏感性 ----
print(f"[B] 成本敏感性（Q1 月换手 {turn:.1%}，Q1 月均毛收益 {qstats_m[0]['ret']/12:.4%}）：")
for c in [0.0, 0.0005, 0.001, 0.002, 0.005]:
    net_ret = qstats_m[0]["ret"] - turn * c * 12
    print(f"  c={c*1e4:.0f}bps: Q1 净年化 {net_ret:+.2%}")

# ---- 图 B1: 波动-收益散点 ----
fig, ax = plt.subplots(figsize=(8.6, 5.6))
for k in range(5):
    idx = qs[k]
    ax.scatter(vol_r[idx], ret_ann[idx], s=10, alpha=0.55, color=qcolors[k], label=q_labels[k])
for k in range(5):
    ax.scatter(qstats[k]["vol"], qstats[k]["ret"], s=130, marker="D", color=qcolors[k],
               edgecolor=BLACK, zorder=5)
    ax.annotate(f"{qstats[k]['ret']:+.1%}", (qstats[k]["vol"], qstats[k]["ret"]),
                textcoords="offset points", xytext=(8, 6), fontsize=9)
z_fit = np.polyfit(vol_r, ret_ann, 1)
xx = np.linspace(vol_r.min(), vol_r.max(), 50)
ax.plot(xx, np.polyval(z_fit, xx), color=BLACK, ls="--", lw=1.2,
        label=f"拟合斜率 {z_fit[0]:.3f}/单位波动")
ax.set_xlabel("年化已实现波动"); ax.set_ylabel("年化收益")
ax.set_title("500 只合成股票：波动-收益截面（菱形=五分位均值）——低波组收益反而更高", fontsize=11)
ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_B, "vol_return_scatter.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 B2: 五分位累计净值 ----
fig, ax = plt.subplots(figsize=(11, 5.2))
mx = np.arange(1, len(qm[0]) + 1)
for k in range(5):
    ax.plot(mx, np.cumprod(1 + qm[k]), color=qcolors[k], lw=1.4,
            label=f"{q_labels[k]} SR={qstats_m[k]['sr']:.2f}")
ax.axhline(1, color=GRAY, lw=0.8)
ax.set_title("月度调仓波动五分位累计净值（排序用过去 250 日已实现波动，无前视）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax.set_ylabel("净值"); ax.set_xlabel("月份")
fig.tight_layout()
fig.savefig(os.path.join(OUT_B, "quintile_cumulative.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 B3: BAB 因子 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
ax.plot(mx, np.cumprod(1 + bab_m), color=RED, lw=1.5, label=f"BAB 因子 SR={bab_stats['sr']:.2f}")
ax.plot(mx, np.cumprod(1 + tilt_m), color=ORANGE, lw=1.3, label=f"长仓低波倾斜 SR={tilt_stats['sr']:.2f}")
ax.plot(mx, np.cumprod(1 + bh_m), color=BLUE, lw=1.2, label=f"市场 SR={mkt_stats['sr']:.2f}")
ax.axhline(1, color=GRAY, lw=0.8)
ax.set_title("BAB（β 中性化）与 A 股可执行版：月度调仓", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("净值")
ax = axes[1]
def roll_sr_b(r, w=36):
    return np.array([r[i:i + w].mean() / r[i:i + w].std() * np.sqrt(12) for i in range(len(r) - w)])
ax.plot(np.arange(37, len(bab_m) + 1), roll_sr_b(bab_m), color=RED, lw=1.1, label="BAB")
ax.plot(np.arange(37, len(tilt_m) + 1), roll_sr_b(tilt_m), color=ORANGE, lw=1.1, label="长仓低波倾斜")
ax.plot(np.arange(37, len(bh_m) + 1), roll_sr_b(bh_m), color=BLUE, lw=1.1, label="市场")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_title("滚动 36 个月 Sharpe", fontsize=11); ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT_B, "bab_factor.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 B4: 机制分解 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
ax = axes[0]
ax.scatter(beta_t, alpha_r * 252, s=12, alpha=0.5, color=BLUE)
xx = np.linspace(0.6, 1.7, 50)
ax.plot(xx, (b_mech[0] + b_mech[1] * xx) * 252, color=RED, lw=1.6,
        label=f"斜率 {b_mech[1]*252:.4f}/β")
ax.set_xlabel("真实 β"); ax.set_ylabel("年化 alpha（时间序列回归）")
ax.set_title("杠杆约束通道：β 越高 alpha 越低", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
ax.scatter(idio * 100, alpha_r * 252, s=12, alpha=0.5, color=ORANGE)
xx = np.linspace(idio.min() * 100, idio.max() * 100, 50)
ax.plot(xx, (b_mech[0] + b_mech[2] * xx / 100) * 252, color=RED, lw=1.6,
        label=f"斜率 {b_mech[2]*252/100:.4f}/1%idio")
ax.set_xlabel("特质波动（%/日）"); ax.set_ylabel("年化 alpha")
ax.set_title("彩票偏好通道：特质波动越高 alpha 越低", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.suptitle(f"机制分解回归：alpha ~ β + idio，R²={r2_mech:.2f}（合成世界中机制是人为植入的）", fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(OUT_B, "mechanism_decomp.png"), bbox_inches="tight")
plt.close(fig)

# ---- 图 B5: 成本 + 分年 ----
fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
ax = axes[0]
cs = [0.0, 0.0005, 0.001, 0.002, 0.005]
net_q1 = [(qstats_m[0]["ret"] - turn * c * 12) * 100 for c in cs]
net_eqw = [(eqw_stats["ret"] - 0.02 * c * 12) * 100 for c in cs]  # 等权月换手近似 2%
ax.plot([c * 1e4 for c in cs], net_q1, "o-", color=BLUE, label=f"Q1 低波（月换手 {turn:.0%}）")
ax.plot([c * 1e4 for c in cs], net_eqw, "s--", color=GRAY, label="等权（月换手 2%）")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xlabel("单边成本（bps）"); ax.set_ylabel("净年化收益（%）")
ax.set_title("成本敏感性：低波策略的敌人是换手不是信号", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
ax = axes[1]
xpos = np.arange(years)
ax.bar(xpos - 0.2, y_q1, 0.4, color=BLUE, label="Q1 低波")
ax.bar(xpos + 0.2, y_q5, 0.4, color=RED, label="Q5 高波")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_xticks(xpos)
ax.set_xticklabels([f"Y{y+1}" for y in range(years)], fontsize=9)
ax.set_title("分年收益：低波在熊市年全面占优（全样本分组）", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig.savefig(os.path.join(OUT_B, "cost_subperiod.png"), bbox_inches="tight")
plt.close(fig)

print("[B] figures saved:", sorted(os.listdir(OUT_B)))
print("DONE")
