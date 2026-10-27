#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oct26b-B: ACD 成交持续时间模型 文章配图 + 核心统计
DGP: 6.5 小时交易日, 日内强度 U 型 m(τ)=1+A(2τ-1)^2 (开盘/收盘 2.5 倍),
     x_i = ψ_i ε_i / m(τ_i), ε~Exp(1); ψ_i = ω + α x̃_{i-1} + β ψ_{i-1},
     x̃_i = x_i·m(τ_i) 为去季节化持续期。真值 ω=1.0, α=0.10, β=0.85。
估计: 指数 ACD QMLE / Weibull ACD QMLE / 不去季节化的对照。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.optimize import minimize
from scipy import stats

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
rcParams["figure.facecolor"] = "white"
rcParams["axes.grid"] = True
rcParams["grid.alpha"] = 0.3

OUT = "/Users/halo123456/workspace/astro-blog/public/images/acd-duration"
os.makedirs(OUT, exist_ok=True)
RED, GREEN, BLUE, GRAY, ORANGE, BLACK, PURPLE = "#d63031", "#00b894", "#0984e3", "#636e72", "#e17055", "#2d3436", "#6c5ce7"

DAY = 23400.0
A_SEAS = 1.5
OMEGA, ALPHA, BETA = 1.0, 0.10, 0.85
N_DAYS = 20


def m_int(tau):
    return 1.0 + A_SEAS * (2 * tau - 1) ** 2


def simulate(n_days, seed, omega=OMEGA, alpha=ALPHA, beta=BETA, nu=None):
    """返回成交时刻 t、持续期 x、条件持续期 ψ、日内比例 τ"""
    rng = np.random.default_rng(seed)
    cap = int(n_days * DAY / 12)
    ts = np.empty(cap); xs = np.empty(cap); psis = np.empty(cap); taus = np.empty(cap)
    t = 0.0
    psi = omega / (1 - alpha - beta)
    i = 0
    while i < cap and t < n_days * DAY:
        tau = (t % DAY) / DAY
        m = m_int(tau)
        eps = rng.standard_exponential() if nu is None else rng.weibull(nu)
        x = psi * eps / m
        t += x
        if t >= n_days * DAY:
            break
        ts[i] = t; xs[i] = x; psis[i] = psi; taus[i] = tau
        i += 1
        psi = omega + alpha * (x * m) + beta * psi
    return ts[:i], xs[:i], psis[:i], taus[:i]


# ---------------- 季节因子非参数估计 ----------------
def estimate_seasonal(ts, xs, n_bins=39):
    """按 15 分钟分箱估平均持续期 → 平滑 → 归一化到全天均值 1"""
    edges = np.linspace(0, DAY, n_bins + 1)
    idx = np.digitize(ts % DAY, edges) - 1
    idx = np.clip(idx, 0, n_bins - 1)
    means = np.full(n_bins, np.nan)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 20:
            means[b] = xs[sel].mean()
    ok = ~np.isnan(means)
    means[~ok] = np.interp(np.flatnonzero(~ok), np.flatnonzero(ok), means[ok])
    # 3 箱中心移动平均平滑
    k = np.ones(3) / 3
    sm = np.convolve(means, k, mode="same")
    sm[0] = means[:2].mean(); sm[-1] = means[-2:].mean()
    grand = xs.mean()
    s_hat = sm / grand          # 期望持续期乘数 (开盘/收盘 >1)
    def s_of_tau(tau):
        b = np.clip((tau * DAY / (DAY / n_bins)).astype(int), 0, n_bins - 1)
        return s_hat[b]
    return s_hat, s_of_tau


# ---------------- ACD 估计器 ----------------
def _acd_filter(x, params, kind="exp"):
    w, a, b = params[:3]
    n = len(x)
    psi = np.empty(n)
    psi[0] = w / (1 - a - b)
    for i in range(1, n):
        psi[i] = w + a * x[i - 1] + b * psi[i - 1]
    return psi


def acd_qmle(x, kind="exp", x0s=((-2.0, -2.3, 2.0, 0.1), (-1.5, -2.0, 2.2, 0.0))):
    """kind='exp': ε~Exp(1); kind='weibull': ε~Weibull(ν)"""
    def negll(p):
        w = np.exp(p[0]); a = 1 / (1 + np.exp(-p[1])); b = 1 / (1 + np.exp(-p[2]))
        if a + b >= 0.999:
            return 1e12
        nu = np.exp(p[3]) if kind == "weibull" else 1.0
        psi = _acd_filter(x, (w, a, b))
        u = x / psi
        if kind == "exp":
            ll = np.sum(-np.log(psi) - u)
        else:
            ll = np.sum(np.log(nu) - np.log(psi) + (nu - 1) * np.log(u) - u ** nu)
        return -ll
    best = None
    for x0 in x0s:
        res = minimize(negll, x0, method="Nelder-Mead",
                       options={"maxiter": 2500, "xatol": 1e-7, "fatol": 1e-8})
        if best is None or res.fun < best.fun:
            best = res
    if kind == "weibull":
        return np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), 1 / (1 + np.exp(-best.x[2])), np.exp(best.x[3])
    return np.exp(best.x[0]), 1 / (1 + np.exp(-best.x[1])), 1 / (1 + np.exp(-best.x[2]))


def acf(x, nlags):
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    denom = np.dot(x, x)
    return np.array([np.dot(x[:-k], x[k:]) / denom for k in range(1, nlags + 1)])


def lb_test(x, nlags=20):
    r = acf(x, nlags)
    n = len(x)
    Q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, nlags + 1)))
    return Q, 1 - stats.chi2.cdf(Q, nlags)


def overdispersion(x):
    n = len(x)
    T = (n - 1) * x.var(ddof=1) / x.mean() ** 2
    return T, 1 - stats.chi2.cdf(T, n - 1), x.var(ddof=1) / x.mean() ** 2


# ================= 主实验 =================
ts, xs, psis, taus = simulate(N_DAYS, 20261026)
n = len(xs)
print(f"[数据] {N_DAYS} 个交易日, {n} 笔成交, 日均 {n/N_DAYS:.0f} 笔, "
      f"平均持续期 {xs.mean():.1f}s, 中位 {np.median(xs):.1f}s")

# 1) 过度分散检验（指数原假设）
T, p_od, od_ratio = overdispersion(xs)
print(f"[检验] 过度分散: var/mean^2 = {od_ratio:.2f}, T = {T:.0f}, p = {p_od:.3g}")

# 2) 原始持续期的自相关
Q_raw, p_raw = lb_test(xs)
print(f"[检验] 原始持续期 Ljung-Box(20): Q = {Q_raw:.0f}, p = {p_raw:.3g}")

# 3) 季节因子估计 + 去季节化（s(τ) 是期望持续期乘数: 开盘/收盘短、中庸长,
#    去季节化 = 除以 s, 即 x̃ = x·m_int(τ)）
s_hat, s_of_tau = estimate_seasonal(ts, xs)
x_tilde = xs / s_of_tau(taus)
print(f"[季节] 估计的持续期乘数 s(τ): 开盘 {s_hat[0]:.2f}x / 中庸 {s_hat[19]:.2f}x / 收盘 {s_hat[-1]:.2f}x (真值 0.40/1.00/0.40)")

# 3b) 原始 vs 去季节化的过度分散
T_t, p_t, od_tilde = overdispersion(x_tilde)
print(f"[检验] 去季节化后过度分散: var/mean^2 = {od_tilde:.2f}, T = {T_t:.0f}, p = {p_t:.3g}")

# 4) 指数 ACD / Weibull ACD
par_e = acd_qmle(x_tilde, "exp")
par_w = acd_qmle(x_tilde, "weibull")
print(f"[估计] 指数 ACD: w={par_e[0]:.3f} a={par_e[1]:.4f} b={par_e[2]:.4f} (a+b={par_e[1]+par_e[2]:.4f})")
print(f"[估计] Weibull ACD: w={par_w[0]:.3f} a={par_w[1]:.4f} b={par_w[2]:.4f} nu={par_w[3]:.3f}")

# 5) 不去季节化的对照
par_raw = acd_qmle(xs, "exp")
print(f"[对照] 不去季节化: w={par_raw[0]:.3f} a={par_raw[1]:.4f} b={par_raw[2]:.4f} (a+b={par_raw[1]+par_raw[2]:.4f})")

# 6) 标准化残差诊断
psi_e = _acd_filter(x_tilde, par_e)
resid = x_tilde / psi_e
Q_res, p_res = lb_test(resid)
print(f"[诊断] 标准化残差 Ljung-Box(20): Q = {Q_res:.0f}, p = {p_res:.3g}")
print(f"[诊断] 残差均值 {resid.mean():.3f} (真值 1), 标准差 {resid.std():.3f} (指数原假设 1)")

# 7) 半衰期
hl = np.log(0.5) / np.log(par_e[1] + par_e[2])
print(f"[动态] 冲击半衰期 = {hl:.1f} 笔 (估计 a+b = {par_e[1]+par_e[2]:.3f}, 真值 0.95 → {np.log(0.5)/np.log(0.95):.1f} 笔)")

# 8) 创新分布误设实验: DGP 用 Weibull(1.3) 创新, 仍用指数 ACD 拟合
ts_w, xs_w, _, taus_w = simulate(N_DAYS, 777, nu=1.3)
_, s_of_tau_w = estimate_seasonal(ts_w, xs_w)
x_tw = xs_w / s_of_tau_w(taus_w)
par_ew = acd_qmle(x_tw, "exp")
psi_ew = _acd_filter(x_tw, par_ew)
resid_w = x_tw / psi_ew
Q_rw, p_rw = lb_test(resid_w)
print(f"[误设] Weibull(1.3) DGP + 指数 ACD 拟合: a={par_ew[1]:.4f} b={par_ew[2]:.4f} (真值 0.10/0.85)")
print(f"[误设] 残差峰度: 指数DGP {stats.kurtosis(resid)+3:.2f} vs Weibull-DGP {stats.kurtosis(resid_w)+3:.2f} (指数原假设 9)")

# ================= 图 1: 日内结构与成交 =================
fig, axes = plt.subplots(2, 1, figsize=(9.4, 5.6))
# 上: 30 分钟分箱成交笔数 + 理论强度
bin_edges = np.arange(0, DAY + 1, 1800)
counts, _ = np.histogram(ts % DAY, bins=bin_edges)
centers = (bin_edges[:-1] + bin_edges[1:]) / 2 / DAY
axes[0].bar(centers, counts, width=1.8 / 52, color=BLUE, alpha=0.75, label="每 30 分钟成交笔数")
tau_grid = np.linspace(0, 1, 200)
axes[0].plot(tau_grid, m_int(tau_grid) / m_int(tau_grid).max() * counts.max() * 0.92,
             color=RED, lw=2, label="理论强度 $m(\\tau)=1+1.5(2\\tau-1)^2$")
axes[0].set_xlim(0, 1)
axes[0].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
axes[0].set_xticklabels(["09:30", "11:00", "13:00", "14:30", "16:00"])
axes[0].set_ylabel("成交笔数")
axes[0].legend(fontsize=9, loc="upper center")
axes[0].set_title("日内 U 型强度：开盘与收盘的成交密度是中庸的 2.5 倍", fontsize=11)
# 下: 第 3 日前 400 笔持续期
d3 = (ts >= 2 * DAY) & (ts < 3 * DAY)
seg = np.flatnonzero(d3)[:400]
axes[1].plot(np.arange(len(seg)), xs[seg], color=GRAY, lw=0.8, marker="o", ms=2.5, label="实际持续期 $x_i$")
axes[1].plot(np.arange(len(seg)), psis[seg] / m_int(taus[seg]), color=RED, lw=1.4, label="期望持续期 $\\psi_i/m(\\tau_i)$")
axes[1].set_yscale("log")
axes[1].set_ylabel("持续期（秒，对数轴）")
axes[1].set_xlabel("第 3 日前 400 笔成交")
axes[1].legend(fontsize=9, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "intraday_pattern.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 2: 持续期分布 + 残差诊断 =================
fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6))
bins = np.logspace(np.log10(0.5), np.log10(xs.max() * 1.1), 60)
axes[0].hist(x_tilde, bins=bins, density=True, color=BLUE, alpha=0.65,
             label="去季节化持续期 $\\tilde{x}_i$")
grid = np.logspace(np.log10(0.5), np.log10(x_tilde.max() * 1.1), 400)
axes[0].plot(grid, stats.expon.pdf(grid, scale=x_tilde.mean()), color=RED, lw=2,
             label=f"指数分布（均值 {x_tilde.mean():.1f}s）")
axes[0].set_xscale("log")
axes[0].set_xlabel("持续期（秒，对数轴）")
axes[0].set_ylabel("密度")
axes[0].set_title(f"右尾远比指数厚：var/mean² = {od_tilde:.1f}（指数原假设下 ACD 聚类也应 >2）", fontsize=10.5)
axes[0].legend(fontsize=9.5)
# 右: 标准化残差 vs 指数(1) / Weibull-DGP 残差
axes[1].hist(resid, bins=np.logspace(np.log10(0.02), np.log10(12), 70), density=True,
             color=BLUE, alpha=0.6, label=f"指数创新 DGP 的残差（峰度 {stats.kurtosis(resid)+3:.1f}）")
axes[1].hist(resid_w, bins=np.logspace(np.log10(0.02), np.log10(12), 70), density=True,
             color=ORANGE, alpha=0.55, label=f"Weibull(1.3) 创新 DGP 的残差（峰度 {stats.kurtosis(resid_w)+3:.1f}）")
g2 = np.logspace(np.log10(0.02), np.log10(12), 400)
axes[1].plot(g2, stats.expon.pdf(g2), color=RED, lw=2, label="Exp(1) 参考（峰度 9）")
axes[1].set_xscale("log")
axes[1].set_xlabel("标准化残差 $\\tilde{x}_i/\\psi_i$（对数轴）")
axes[1].set_title("创新分布误设会留在残差峰度里：指数 ACD 的 (α,β) 仍然估得住", fontsize=10.5)
axes[1].legend(fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "duration_dist.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 3: ACD 过滤 =================
fig, ax = plt.subplots(figsize=(9.4, 4.6))
seg = slice(20000, 21200)
ax.scatter(np.arange(seg.stop - seg.start), x_tilde[seg], s=7, color=BLUE, alpha=0.5, label="去季节化持续期")
ax.plot(np.arange(seg.stop - seg.start), psi_e[seg], color=RED, lw=1.8, label="ACD 条件持续期 $\\psi_i$")
imax = np.argmax(x_tilde[seg])
ax.annotate(f"最长持续期 {x_tilde[seg][imax]:.0f}s\n≈ 期望的 {x_tilde[seg][imax]/psi_e[seg][imax]:.1f} 倍",
            xy=(imax, x_tilde[seg][imax]), xytext=(imax + 60, x_tilde[seg][imax] * 1.15),
            fontsize=9.5, color=BLACK,
            arrowprops=dict(arrowstyle="->", color=BLACK, lw=1))
ax.set_yscale("log")
ax.set_ylabel("秒（对数轴）")
ax.set_xlabel("成交序号")
ax.set_title(f"ACD 滤波：一笔长持续期把后续期望整体抬高，再按 (α+β)={par_e[1]+par_e[2]:.2f} 缓慢衰减", fontsize=11)
ax.legend(fontsize=9.5, loc="upper right")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "acd_filter.png"), bbox_inches="tight")
plt.close(fig)

# ================= 图 4: ACF 对比 =================
nlags = 20
r_raw = acf(xs, nlags)
r_res = acf(resid, nlags)
fig, ax = plt.subplots(figsize=(9.0, 4.6))
xx = np.arange(1, nlags + 1)
ax.bar(xx - 0.2, r_raw, width=0.4, color=GRAY, label=f"原始持续期（LB Q={Q_raw:.0f}, p≈0）")
ax.bar(xx + 0.2, r_res, width=0.4, color=RED, label=f"ACD 标准化残差（LB Q={Q_res:.0f}, p={p_res:.2f}）")
ci = 1.96 / np.sqrt(n)
ax.axhline(ci, color=BLUE, lw=1, ls=":"); ax.axhline(-ci, color=BLUE, lw=1, ls=":")
ax.axhline(0, color=BLACK, lw=0.8)
ax.text(nlags + 0.3, ci, "95% 白噪声带", fontsize=8.5, color=BLUE, va="center")
ax.set_xlabel("滞后阶数")
ax.set_ylabel("自相关")
ax.set_title("持续期的可预测性被 ACD 收进条件项：原始 ACF 一阶 0.35+，残差掉进白噪声带", fontsize=11)
ax.legend(fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "acf_compare.png"), bbox_inches="tight")
plt.close(fig)
print(f"[图4] ACF(1): 原始 {r_raw[0]:.3f} vs 残差 {r_res[0]:.3f}")

# ================= 图 5: 预测 + 季节化对照 =================
fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))
# 左: 冲击后的 ψ 多步预测
i0 = 20000 + int(np.argmax(x_tilde[seg]))
pers = par_e[1] + par_e[2]
psi_now = psi_e[i0]
k = np.arange(0, 60)
psi_fc = par_e[0] + (psi_now - par_e[0]) * pers ** k
axes[0].plot(k, psi_fc, color=RED, lw=2, label="$E[\\psi_{i+k}|F_i]$（指数 ACD）")
axes[0].axhline(par_e[0] / (1 - pers), color=GRAY, lw=1.2, ls="--",
                label=f"无条件水平 {par_e[0]/(1-pers):.1f}s")
axes[0].axvline(hl, color=ORANGE, lw=1.2, ls=":")
axes[0].text(hl + 1.5, psi_fc.max() * 0.8, f"半衰期 {hl:.0f} 笔", fontsize=9.5, color=ORANGE)
axes[0].set_xlabel("冲击后第 k 笔")
axes[0].set_ylabel("期望持续期（秒）")
axes[0].set_title("长持续期冲击后的均值回复：把「下一笔可能很慢」写进排单", fontsize=11)
axes[0].legend(fontsize=9)
# 右: 季节化与否的参数对照
labels = ["$\\alpha$", "$\\beta$", "$\\alpha+\\beta$"]
vals_deseas = [par_e[1], par_e[2], par_e[1] + par_e[2]]
vals_raw = [par_raw[1], par_raw[2], par_raw[1] + par_raw[2]]
xx2 = np.arange(3)
axes[1].bar(xx2 - 0.19, vals_deseas, width=0.38, color=RED, label="去季节化后估计")
axes[1].bar(xx2 + 0.19, vals_raw, width=0.38, color=GRAY, label="原始持续期直接估计")
for i, v in enumerate(vals_deseas):
    axes[1].text(i - 0.19, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=RED)
for i, v in enumerate(vals_raw):
    axes[1].text(i + 0.19, v + 0.012, f"{v:.3f}", ha="center", fontsize=9, color=GRAY)
axes[1].axhline(ALPHA, color=RED, lw=0.8, ls=":")
axes[1].axhline(BETA, color=RED, lw=0.8, ls=":")
axes[1].set_xticks(xx2); axes[1].set_xticklabels(labels)
axes[1].set_ylim(0, 1.05)
axes[1].set_ylabel("参数值")
axes[1].set_title(f"虚线=真值 α={ALPHA}, β={BETA}：不去季节化把 β 从 {par_e[2]:.3f} 虚抬到 {par_raw[2]:.3f}，持续性从 {par_e[1]+par_e[2]:.3f} 虚抬到 {par_raw[1]+par_raw[2]:.3f}", fontsize=10.5)
axes[1].legend(fontsize=9.5)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "forecast_seasonal.png"), bbox_inches="tight")
plt.close(fig)

print("\n[图] 5 张已保存至", OUT)
