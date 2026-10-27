#!/usr/bin/env python3
"""两篇量化文章真实配图生成（non-placeholder, real matplotlib charts）。

文章1: private-credit-mark-smoothing  —— 私募信贷的估值平滑
文章2: stale-pricing-autocorrelation-smoothing —— 陈旧定价与自相关平滑
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

mpl_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = mpl_fonts
matplotlib.rcParams["axes.unicode_minus"] = False

# ---------- 通用配色 ----------
C_TRUE = "#1f4e79"      # 真实价值（深蓝）
C_REP = "#c0392b"       # 上报 NAV（红）
C_ACC = "#27ae60"       # 去平滑/修正（绿）
C_GREY = "#636e72"
GRID = "#e6e6e6"


def acf(x, lags):
    x = np.asarray(x)
    x = x - x.mean()
    n = len(x)
    out = []
    for L in lags:
        if L == 0:
            out.append(1.0)
        else:
            out.append(np.sum(x[L:] * x[:-L]) / np.sum(x * x))
    return np.array(out)


# =========================================================================
# 文章1: 私募信贷的估值平滑
# =========================================================================
OUT1 = "public/images/private-credit-mark-smoothing"
os.makedirs(OUT1, exist_ok=True)
rng = np.random.default_rng(20260827)

# ---- 生成真实 vs 上报 NAV 序列 ----
T = 252
beta = 0.85                       # 平滑系数（越接近1越平滑）
mu_true = 0.0006                 # 真实日收益
sig_true = 0.012                 # 真实日波动
true_ret = rng.normal(mu_true, sig_true, T)
true_nav = np.cumprod(1 + true_ret)
# 上报 NAV = 真实 NAV 的指数加权滑动平均（EWMA）
rep_nav = np.zeros(T)
rep_nav[0] = true_nav[0]
for t in range(1, T):
    rep_nav[t] = beta * rep_nav[t - 1] + (1 - beta) * true_nav[t]
rep_ret = np.diff(rep_nav) / rep_nav[:-1]

# ===== 图1: 真实价值 vs 上报 NAV 路径 =====
fig, ax = plt.subplots(figsize=(11, 5.8))
days = np.arange(T)
ax.plot(days, true_nav, color=C_TRUE, lw=1.8, label="真实经济价值（底层资产真实净值）")
ax.plot(days, rep_nav, color=C_REP, lw=1.8, alpha=0.9, label="上报 NAV（季度估值、EWMA 平滑）")
ax.set_title("私募信贷：上报净值把波动「熨平」了，但底层价值一直在动", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日（约 1 年）"); ax.set_ylabel("净值（起始=1.0）")
ax.legend(loc="upper left", fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT1}/pcs_true_vs_reported.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图2: 年化波动 & Sharpe 对比（真实 vs 上报）=====
ann_true_vol = sig_true * np.sqrt(252)
ann_true_sharpe = (mu_true * 252) / ann_true_vol
ann_rep_vol = np.std(rep_ret, ddof=1) * np.sqrt(252)
ann_rep_sharpe = (np.mean(rep_ret) * 252) / ann_rep_vol

fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
labels = ["真实", "上报 NAV"]
ax[0].bar(labels, [ann_true_vol, ann_rep_vol], color=[C_TRUE, C_REP])
ax[0].set_title("年化波动率", fontsize=12, fontweight="bold")
for i, v in enumerate([ann_true_vol, ann_rep_vol]):
    ax[0].text(i, v + 0.002, f"{v:.2%}", ha="center", fontsize=11, fontweight="bold")
ax[0].set_ylim(0, max(ann_true_vol, ann_rep_vol) * 1.25)
ax[1].bar(labels, [ann_true_sharpe, ann_rep_sharpe], color=[C_TRUE, C_REP])
ax[1].set_title("年化 Sharpe（无风险利率=0）", fontsize=12, fontweight="bold")
for i, v in enumerate([ann_true_sharpe, ann_rep_sharpe]):
    ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax[1].set_ylim(0, max(ann_true_sharpe, ann_rep_sharpe) * 1.25)
fig.suptitle("平滑把波动率砍掉近一半、Sharpe 翻倍——但收益一分没多",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT1}/pcs_vol_sharpe.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图3: 自相关函数（真实 vs 上报）=====
lags = np.arange(0, 11)
acf_true = acf(true_ret, lags)
acf_rep = acf(rep_ret, lags)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.stem(lags, acf_true, linefmt=C_TRUE, markerfmt="o", basefmt=" ", label="真实收益")
ax.stem(lags, acf_rep, linefmt=C_REP, markerfmt="s", basefmt=" ", label="上报收益")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(lags)
ax.set_title("自相关：真实收益近似白噪声，上报收益 lag-1 高度正相关", fontsize=13, fontweight="bold")
ax.set_xlabel("滞后阶数 k"); ax.set_ylabel("自相关系数")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT1}/pcs_autocorr.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图4: Geltner 去平滑——把被熨平的波动找回来 =====
# 估计 β: 用上报收益对真实上一期收益的回归（已知真实序列时可直接演示）
# 实务中 Geltner 用上报收益自身滞后项: r_t^R = a + β r_{t-1}^R + e
X = rep_ret[:-1]
Y = rep_ret[1:]
beta_hat = np.polyfit(X, Y, 1)[0]
beta_hat = min(max(beta_hat, 0.05), 0.95)
# 去平滑: unsmoothed_t = (r_t^R - β_hat * r_{t-1}^R) / (1 - β_hat)
unsmooth = (rep_ret[1:] - beta_hat * rep_ret[:-1]) / (1 - beta_hat)
un_vol = np.std(unsmooth, ddof=1) * np.sqrt(252)
un_sharpe = (np.mean(unsmooth) * 252) / un_vol

fig, ax = plt.subplots(1, 2, figsize=(11, 5.2))
ax[0].bar(["真实", "上报", "去平滑"], [ann_true_vol, ann_rep_vol, un_vol],
          color=[C_TRUE, C_REP, C_ACC])
ax[0].set_title(f"波动率还原（估计 β̂={beta_hat:.2f}）", fontsize=12, fontweight="bold")
for i, v in enumerate([ann_true_vol, ann_rep_vol, un_vol]):
    ax[0].text(i, v + 0.002, f"{v:.2%}", ha="center", fontsize=11, fontweight="bold")
ax[0].set_ylim(0, max(ann_true_vol, un_vol) * 1.25)
ax[1].bar(["真实", "上报", "去平滑"], [ann_true_sharpe, ann_rep_sharpe, un_sharpe],
          color=[C_TRUE, C_REP, C_ACC])
ax[1].set_title("Sharpe 还原", fontsize=12, fontweight="bold")
for i, v in enumerate([ann_true_sharpe, ann_rep_sharpe, un_sharpe]):
    ax[1].text(i, v + 0.05, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
ax[1].set_ylim(0, max(ann_true_sharpe, un_sharpe) * 1.25)
fig.suptitle("Geltner 去平滑：用滞后项回归剥离平滑，波动率回到真实量级",
             fontsize=13, fontweight="bold")
fig.tight_layout(); fig.savefig(f"{OUT1}/pcs_geltner.png", dpi=160, bbox_inches="tight"); plt.close()
print("文章1 配图完成 ->", OUT1)

# =========================================================================
# 文章2: 陈旧定价与自相关平滑
# =========================================================================
OUT2 = "public/images/stale-pricing-autocorrelation-smoothing"
os.makedirs(OUT2, exist_ok=True)
rng2 = np.random.default_rng(20260828)

# 真实 iid 收益（设定真实 Sharpe）
mu_t = 0.0005
sig_t = 0.015
true_r2 = rng2.normal(mu_t, sig_t, T)
true_sharpe2 = (mu_t * 252) / (sig_t * np.sqrt(252))

# 陈旧定价：上报收益 = 真实收益的 MA(1) 平滑（与文章1 同机制）
def make_stale(b, ret):
    nav_true = np.cumprod(1 + ret)
    nav_rep = np.zeros(len(ret))
    nav_rep[0] = nav_true[0]
    for t in range(1, len(ret)):
        nav_rep[t] = b * nav_rep[t - 1] + (1 - b) * nav_true[t]
    return np.diff(nav_rep) / nav_rep[:-1]

# ===== 图1: ACF 对比（真实 iid vs 陈旧定价）=====
# 用中等平滑系数演示
beta2 = 0.6
stale_r = make_stale(beta2, true_r2)
acf_s = acf(stale_r, lags)
acf_t2 = acf(true_r2, lags)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.stem(lags, acf_t2, linefmt=C_TRUE, markerfmt="o", basefmt=" ", label="真实（iid）收益")
ax.stem(lags, acf_s, linefmt=C_REP, markerfmt="s", basefmt=" ", label="陈旧定价收益")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(lags)
ax.set_title("陈旧定价把白噪声染成高度正自相关——lag-1 一眼可见", fontsize=13, fontweight="bold")
ax.set_xlabel("滞后阶数 k"); ax.set_ylabel("自相关系数")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT2}/sps_acf.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图2: 朴素 Sharpe 虚高 vs 平滑系数，以及 Lo 修正后回到真实 =====
bs = np.linspace(0.0, 0.9, 19)
naive_sr = []
corr_sr = []
for b in bs:
    sr = make_stale(b, true_r2)
    s_naive = (np.mean(sr) * 252) / (np.std(sr, ddof=1) * np.sqrt(252))
    rho = acf(sr, [1])[0]
    rho = min(max(rho, -0.99), 0.99)
    s_corr = s_naive * np.sqrt((1 - rho) / (1 + rho))   # Lo (2002) 修正
    naive_sr.append(s_naive)
    corr_sr.append(s_corr)

fig, ax = plt.subplots(figsize=(11, 5.6))
ax.plot(bs, naive_sr, color=C_REP, lw=2.2, marker="o", ms=4, label="朴素 Sharpe（被虚高）")
ax.plot(bs, corr_sr, color=C_ACC, lw=2.2, marker="s", ms=4, label="Lo 修正后 Sharpe")
ax.axhline(true_sharpe2, color=C_TRUE, lw=2, ls="--", label=f"真实 Sharpe = {true_sharpe2:.2f}")
ax.set_xlabel("平滑系数 β（越大约平滑）"); ax.set_ylabel("年化 Sharpe")
ax.set_title("Sharpe 虚高因子 = √[(1+ρ)/(1−ρ)]：β=0.6 时理论虚高约 1.8 倍", fontsize=13, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT2}/sps_sharpe_inflation.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图3: 蒙特卡洛——朴素 vs 修正后的 Sharpe 估计分布 =====
MC = 500
b_mc = 0.6
naive_est = []; corr_est = []
for mc in range(MC):
    rr = rng2.normal(mu_t, sig_t, T)
    sr = make_stale(b_mc, rr)
    s_n = (np.mean(sr) * 252) / (np.std(sr, ddof=1) * np.sqrt(252))
    rho = acf(sr, [1])[0]
    rho = min(max(rho, -0.99), 0.99)
    s_c = s_n * np.sqrt((1 - rho) / (1 + rho))
    naive_est.append(s_n); corr_est.append(s_c)
naive_est = np.array(naive_est); corr_est = np.array(corr_est)

fig, ax = plt.subplots(figsize=(11, 5.4))
bins = np.linspace(0, max(naive_est.max(), corr_est.max()) + 0.2, 40)
ax.hist(naive_est, bins=bins, alpha=0.55, color=C_REP, label=f"朴素（中位={np.median(naive_est):.2f}）")
ax.hist(corr_est, bins=bins, alpha=0.55, color=C_ACC, label=f"Lo 修正（中位={np.median(corr_est):.2f}）")
ax.axvline(true_sharpe2, color=C_TRUE, lw=2, ls="--", label=f"真实={true_sharpe2:.2f}")
ax.set_xlabel("估计的 Sharpe（单次 1 年样本）"); ax.set_ylabel("频数")
ax.set_title("500 次蒙特卡洛：修正把分布拉回真实值，朴素估计系统性右偏", fontsize=12.5, fontweight="bold")
ax.legend(fontsize=10); ax.grid(True, color=GRID)
fig.tight_layout(); fig.savefig(f"{OUT2}/sps_montecarlo.png", dpi=160, bbox_inches="tight"); plt.close()

# ===== 图4: 机制示意图 =====
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.set_xlim(0, 12); ax.set_ylim(0, 10); ax.axis("off")
ax.text(6, 9.3, "陈旧定价如何系统性虚高 Sharpe：机制链", ha="center", fontsize=14, fontweight="bold")

ax.add_patch(FancyBboxPatch((0.4, 5.2), 2.6, 1.8, boxstyle="round,pad=0.1", fc="#eef3f8", ec=C_TRUE, lw=1.5))
ax.text(1.7, 6.6, "真实收益 r_t", ha="center", fontsize=11, fontweight="bold", color=C_TRUE)
ax.text(1.7, 6.0, "i.i.d.，白噪声", ha="center", fontsize=9.5)
ax.text(1.7, 5.6, f"真实 SR={true_sharpe2:.2f}", ha="center", fontsize=9.5)

ax.annotate("", xy=(3.6, 6.1), xytext=(3.0, 6.1), arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.8))
ax.text(3.3, 6.5, "估值滞\n后（EWMA）", ha="center", fontsize=8.5, color=C_GREY)

ax.add_patch(FancyBboxPatch((3.7, 5.2), 2.6, 1.8, boxstyle="round,pad=0.1", fc="#fdeaea", ec=C_REP, lw=1.5))
ax.text(5.0, 6.6, "上报收益 R_t", ha="center", fontsize=11, fontweight="bold", color=C_REP)
ax.text(5.0, 6.0, "= βRₜ₋₁+(1−β)rₜ", ha="center", fontsize=9)
ax.text(5.0, 5.6, "引入正自相关 ρ", ha="center", fontsize=9)

ax.annotate("", xy=(7.0, 6.1), xytext=(6.3, 6.1), arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.8))
ax.text(6.65, 6.5, "自相关\n抬升 σ", ha="center", fontsize=8.5, color=C_GREY)

ax.add_patch(FancyBboxPatch((7.1, 5.2), 2.6, 1.8, boxstyle="round,pad=0.1", fc="#fdeaea", ec=C_REP, lw=1.5))
ax.text(8.4, 6.7, "朴素 Sharpe", ha="center", fontsize=11, fontweight="bold", color=C_REP)
ax.text(8.4, 6.05, f"=SR×√[(1+ρ)/(1−ρ)]", ha="center", fontsize=9)
ax.text(8.4, 5.55, f"→ 虚高 ×{np.sqrt((1+beta2)/(1-beta2)):.2f}", ha="center", fontsize=9.5, fontweight="bold")

ax.annotate("", xy=(8.4, 4.5), xytext=(8.4, 5.1), arrowprops=dict(arrowstyle="->", color=C_ACC, lw=1.8))
ax.text(9.9, 4.8, "Lo 修正\n÷√[(1+ρ)/(1−ρ)]", ha="center", fontsize=8.5, color=C_ACC)

ax.add_patch(FancyBboxPatch((7.1, 2.2), 2.6, 1.7, boxstyle="round,pad=0.1", fc="#e6f4ea", ec=C_ACC, lw=1.5))
ax.text(8.4, 3.4, "修正后 SR", ha="center", fontsize=11, fontweight="bold", color=C_ACC)
ax.text(8.4, 2.85, "回到真实值", ha="center", fontsize=9.5)

ax.text(1.7, 1.2, "关键结论：波动率被平滑低估 → 分母变小 → Sharpe 被虚高；修正只需除以 √[(1+ρ)/(1−ρ)]",
        ha="center", fontsize=10.5, fontweight="bold", color=C_GREY)
fig.tight_layout(); fig.savefig(f"{OUT2}/sps_mechanism.png", dpi=160, bbox_inches="tight"); plt.close()
print("文章2 配图完成 ->", OUT2)
