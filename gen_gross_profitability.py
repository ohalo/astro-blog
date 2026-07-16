#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「毛利率因子：Novy-Marx 之后，利润表的现金含量才是硬 alpha」生成真实配图与统计数字。

核心逻辑(Novy-Marx 2013, "Profitability, growth, and the growth of book equity value"):
  - 毛利率(Gross Profitability) = (营业收入 - 营业成本)/ 总资产(或/账面权益)
      -> 衡量「收入里扣掉直接成本后还剩多少真金白银」, 是利润表里最「现金化」的一块
  - 与 Fama-French RMW 用的「营业盈利能力」(operating profitability, 再扣 SGA/利息)对比:
      GP 更持久、更"干净"(SGA/利息是噪声), 因此截面预测力更强、更稳定
  - 合成面板: N 只股票 x T 月,  latent 质量 q 用 AR(1) 持久演化;
      GP_i,t = q_i,t + 小噪声        (持久, 现金含量)
      OP_i,t = 0.6 q_i,t + 大噪声     (不持久, 含 SGA/利息式噪声)
      下月收益 r_{i,t+1} = beta*市场 + c*GP_i,t + 小价值暴露 + 个股噪声
  - 验证: 十分位分层单调、多空(L-S)年化/夏普/CAPM alpha、rank-IC(GP vs OP)、自相关持久性

全部数字由文中 Python 真实计算(自包含, 仅依赖 numpy/scipy)。
图片:
  gp_decile_returns.png  —— 按毛利率十分位分层下月平均收益(单调, 标 L-S)
  gp_cum_ls.png          —— 多空组合累计净值
  gp_ic_compare.png      —— GP vs OP 的逐月 rank-IC(及整体均值)
  gp_persistence.png     —— GP vs OP 的面板自相关(毛利率更持久)
"""
import os
import numpy as np
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "gross-profitability-factor")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260717)
N, T = 240, 120          # 240 只股票, 120 个月
n_dec = 10

# ---------- 1) 合成面板 ----------
q = np.zeros((N, T))
q[:, 0] = rng.normal(0, 1, N)
for t in range(1, T):
    q[:, t] = 0.95 * q[:, t - 1] + 0.10 * rng.normal(0, 1, N)     # 持久质量
GP = q + 0.05 * rng.normal(0, 1, (N, T))                          # 毛利率: 持久 + 极小噪声
OP = 0.60 * q + 0.80 * rng.normal(0, 1, (N, T))                   # 营业盈利能力: 同 q 但噪声大
mkt = rng.normal(0, 0.04, T)                                      # 市场月度收益
bm = rng.normal(0, 1, (N, T))                                     # 账面市值比代理(价值)

ret = np.zeros((N, T + 1))
for t in range(T):
    ret[:, t + 1] = (0.80 * mkt[t]
                     + 0.008 * GP[:, t]          # 毛利率信号(系数小, 避免"作弊"式暴利)
                     + 0.004 * bm[:, t]          # 轻微价值暴露
                     + rng.normal(0, 0.06, N))   # 个股噪声

# ---------- 2) 十分位分层(用 t 月 GP 预测 t+1 月收益) ----------
dec_ret = np.zeros((n_dec, T))
for t in range(T):
    order = np.argsort(GP[:, t])
    ranks = np.empty(N); ranks[order] = np.arange(N)
    d = np.clip(ranks // (N // n_dec), 0, n_dec - 1)
    for k in range(n_dec):
        sel = (d == k)
        dec_ret[k, t] = ret[sel, t + 1].mean()
ls_monthly = dec_ret[-1, :] - dec_ret[0, :]
ann_ret = 12.0 * ls_monthly.mean()
ann_vol = ls_monthly.std() * np.sqrt(12)
sharpe = ann_ret / ann_vol

# CAPM alpha / beta
Xm = np.vstack([np.ones(T), mkt]).T
coef, *_ = np.linalg.lstsq(Xm, ls_monthly, rcond=None)
beta, alpha_m = coef[1], coef[0]
capm_alpha_ann = 12.0 * alpha_m

# ---------- 3) rank-IC (GP vs OP) ----------
def rank_ic_signal(sig, t):
    y = ret[:, t + 1]
    return np.corrcoef(rankdata(sig[:, t]), rankdata(y))[0, 1]
ic_gp = np.array([rank_ic_signal(GP, t) for t in range(T)])
ic_op = np.array([rank_ic_signal(OP, t) for t in range(T)])
ic_gp_mean, ic_op_mean = ic_gp.mean(), ic_op.mean()
t_gp = ic_gp_mean / (ic_gp.std() / np.sqrt(T))

# ---------- 4) 面板自相关(持久性) ----------
def acf_panel(X, maxlag):
    out = np.zeros(maxlag)
    for L in range(1, maxlag + 1):
        a, b = X[:, :-L], X[:, L:]
        ma, mb = a.mean(1, keepdims=True), b.mean(1, keepdims=True)
        num = ((a - ma) * (b - mb)).sum(1)
        den = np.sqrt(((a - ma) ** 2).sum(1) * ((b - mb) ** 2).sum(1))
        out[L - 1] = np.nanmean(num / den)
    return out
acf_gp = acf_panel(GP, 12)
acf_op = acf_panel(OP, 12)

metrics = {
    "N": N, "T_months": T,
    "ls_ann_ret": round(float(ann_ret), 4),
    "ls_ann_vol": round(float(ann_vol), 4),
    "ls_sharpe": round(float(sharpe), 3),
    "capm_alpha_ann": round(float(capm_alpha_ann), 4),
    "capm_beta": round(float(beta), 3),
    "ic_gp_mean": round(float(ic_gp_mean), 4),
    "ic_op_mean": round(float(ic_op_mean), 4),
    "ic_gp_t": round(float(t_gp), 2),
    "acf_gp_lag1": round(float(acf_gp[0]), 3),
    "acf_op_lag1": round(float(acf_op[0]), 3),
    "acf_gp_lag12": round(float(acf_gp[11]), 3),
    "acf_op_lag12": round(float(acf_op[11]), 3),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k, v in metrics.items():
        f.write(f"{k}={v}\n")
print("METRICS", metrics)

# ============ 图1: 十分位分层收益 ============
means = dec_ret.mean(1)
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = ["#C44E52" if k in (0, 9) else "#4C72B0" for k in range(n_dec)]
bars = ax.bar([f"D{k+1}" for k in range(n_dec)], means, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("按毛利率十分位分层（t 月毛利率 → t+1 月收益）", fontsize=13, fontweight="bold")
ax.set_xlabel("毛利率十分位（D1=最低，D10=最高）")
ax.set_ylabel("下月平均收益")
ax.grid(True, alpha=0.3, axis="y")
for k, v in enumerate(means):
    ax.text(k, v + (0.0008 if v >= 0 else -0.0014), f"{v:.3%}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
ax.annotate(f"多空 L-S 月均 = {ls_monthly.mean():.3%}\n年化 ≈ {ann_ret:.1%} / Sharpe {sharpe:.2f}",
            xy=(9, means[9]), xytext=(5.2, max(means) * 1.15),
            fontsize=10, color="#C44E52", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#C44E52"))
plt.tight_layout()
plt.savefig(os.path.join(D, "gp_decile_returns.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2: 多空累计净值 ============
cum = np.cumprod(1 + ls_monthly)
fig, ax = plt.subplots(figsize=(9, 5.0))
ax.plot(range(T), cum, color="#2F4B7C", lw=1.8)
ax.set_title(f"毛利率多空组合累计净值（年化 {ann_ret:.1%}，Sharpe {sharpe:.2f}）",
             fontsize=13, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("净值（起始=1）")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "gp_cum_ls.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3: GP vs OP 逐月 rank-IC ============
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.0))
ax1.plot(range(T), ic_gp, color="#2F4B7C", lw=1.2, label=f"毛利率 GP (均值 {ic_gp_mean:.3f})")
ax1.plot(range(T), ic_op, color="#DD8452", lw=1.2, label=f"营业盈利 OP (均值 {ic_op_mean:.3f})")
ax1.axhline(0, color="black", lw=0.7)
ax1.set_title("逐月 rank-IC：毛利率 vs 营业盈利能力", fontsize=12, fontweight="bold")
ax1.set_xlabel("月份"); ax1.set_ylabel("rank-IC")
ax1.legend(fontsize=9); ax1.grid(True, alpha=0.3)
b = ax2.bar(["毛利率 GP", "营业盈利 OP"], [ic_gp_mean, ic_op_mean],
            color=["#2F4B7C", "#DD8452"], width=0.5)
ax2.set_title("整体平均 rank-IC", fontsize=12, fontweight="bold")
ax2.set_ylabel("平均 rank-IC"); ax2.grid(True, alpha=0.3, axis="y")
for bb, v in zip(b, [ic_gp_mean, ic_op_mean]):
    ax2.text(bb.get_x() + bb.get_width() / 2, v + 0.001, f"{v:.3f}", ha="center", fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(D, "gp_ic_compare.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图4: 面板自相关（持久性） ============
lags = np.arange(1, 13)
fig, ax = plt.subplots(figsize=(9, 5.0))
ax.plot(lags, acf_gp, "o-", color="#2F4B7C", lw=1.8, label=f"毛利率 GP (lag1={acf_gp[0]:.2f})")
ax.plot(lags, acf_op, "s-", color="#DD8452", lw=1.8, label=f"营业盈利 OP (lag1={acf_op[0]:.2f})")
ax.axhline(0, color="black", lw=0.7)
ax.set_title("面板自相关：毛利率更持久（'现金含量'更稳定）", fontsize=13, fontweight="bold")
ax.set_xlabel("滞后月数"); ax.set_ylabel("自相关系数")
ax.legend(fontsize=10); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "gp_persistence.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE gross-profitability images")
