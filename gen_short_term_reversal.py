#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「短期反转因子：用隔夜过度反应捡日内均值回归的钱」生成真实配图与统计数字。

核心逻辑(基于 Birru "Day and Night" / Jegadeesh 短期反转 文献):
  - 日收益可拆成「隔夜(close→open)」与「日内(open→close)」两部分
  - 隔夜收益承载"过度反应"成分 R: 是均值回复的(AR(1), phi<0), 这个月涨多了下月回吐
  - 日内收益承载"真实信息"成分 P: 是慢变量(AR(1), phi≈0.9 持久), 带来轻微动量
  - 因此: ① 1 个月跨截面反转(过去赢家→下月输家)成立, 且主要由隔夜/噪声成分驱动
          ② 该效应在短期限(1M)为负(反转), 在中长期限(6~12M)翻正(动量) —— 经典 horizon flip
  - 验证(自包含合成面板, 固定种子):
      ① 按过去 1 月隔夜收益排序的十分位: D1(最大隔夜输家)→ D10(最大隔夜赢家) 单调下滑
      ② 隔夜反转 L-S(D1-D10) / 日内动量 L-S(D10-D1) / 总反转 L-S 累计净值对比
      ③ rank-IC 随持有期: 1M 负(反转) → 12M 正(动量)
      ④ 隔夜 vs 日内 成分分解: 隔夜含均值回复冲击、日内含持久信息

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  str_decile_overnight.png       —— 隔夜反转十分位(猛跌→猛涨)
  str_curves.png                 —— 隔夜反转 / 日内动量 / 总反转 累计净值
  str_ic_horizon.png             —— rank-IC 随持有期(反转→动量 flip)
  str_overnight_intraday_decomp.png —— 隔夜 vs 日内 均值回复/持久 分解
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
D = os.path.join(BASE, "short-term-reversal-factor")
os.makedirs(D, exist_ok=True)

rng = np.random.default_rng(20260717)
N, YEARS, DAYS_Y = 400, 10, 252
DAYS = YEARS * DAYS_Y
M = YEARS * 12
n_dec = 10
TD = 21  # 每月交易日

# ---------- 月度持久成分 P (动量, phi=0.92) 与 隔夜过度反应 R (反转, phi=-0.6) ----------
# 设定稳态波动后反推创新波动
sigP_steady, phiP = 0.00765, 0.92
sigR_steady, phiR = 0.0110, -0.60
innov_P = sigP_steady * np.sqrt(1 - phiP**2)   # ≈0.003
innov_R = sigR_steady * np.sqrt(1 - phiR**2)   # ≈0.006

P = np.zeros((N, M)); R = np.zeros((N, M))
P[:, 0] = rng.normal(0, sigP_steady, N)
R[:, 0] = rng.normal(0, sigR_steady, N)
for m in range(1, M):
    P[:, m] = phiP * P[:, m-1] + rng.normal(0, innov_P, N)
    R[:, m] = phiR * R[:, m-1] + rng.normal(0, innov_R, N)

# ---------- 日收益: 月度成分均摊到 21 天 + 日度特异噪声 ----------
sd_intra_day = 0.009 / np.sqrt(TD)   # 月内噪声 sd 0.009
sd_on_day = 0.009 / np.sqrt(TD)
ret_intra = np.zeros((N, DAYS))
ret_on = np.zeros((N, DAYS))
for d in range(DAYS):
    m = d // TD
    ret_intra[:, d] = P[:, m] / TD + rng.normal(0, sd_intra_day, N)
    ret_on[:, d] = R[:, m] / TD + rng.normal(0, sd_on_day, N)

# 月度聚合 (N x M)
ret_total_m = np.array([ret_intra[:, m*TD:(m+1)*TD].sum(1) + ret_on[:, m*TD:(m+1)*TD].sum(1) for m in range(M)]).T
ret_intra_m = np.array([ret_intra[:, m*TD:(m+1)*TD].sum(1) for m in range(M)]).T
ret_on_m = np.array([ret_on[:, m*TD:(m+1)*TD].sum(1) for m in range(M)]).T

# 月度个股特异噪声(让截面 L-S 不至于因成批平均而 Sharpe 爆表, 也制造真实回撤)
eps_m = rng.normal(0, 0.055, (N, M))
ret_intra_m = ret_intra_m + eps_m

def perf(ls):
    a = 12.0 * ls.mean(); vol = ls.std() * np.sqrt(12)
    eq = np.cumprod(1 + ls); dd = (eq / np.maximum.accumulate(eq) - 1).min()
    return a, vol, a / vol, dd

# ---------- 十分位 L-S ----------
def decile_ls(sig, fut):
    dec_ret = np.zeros((n_dec, M - 1))
    for m in range(M - 1):
        s = sig[:, m]; order = np.argsort(s); ranks = np.empty(N); ranks[order] = np.arange(N)
        d = np.clip(ranks // (N // n_dec), 0, n_dec - 1)
        for k in range(n_dec):
            sel = (d == k)
            dec_ret[k, m] = fut[sel, m + 1].mean()
    return dec_ret, dec_ret[-1, :] - dec_ret[0, :]   # D10 - D1

# 隔夜反转: 买过去隔夜输家(D1)、卖过去隔夜赢家(D10) → 取 D1-D10 为正向
dec_on, ls_on_raw = decile_ls(ret_on_m, ret_total_m)
ls_on = -ls_on_raw
# 日内动量: 买过去日内赢家(D10)、卖输家(D1) → D10-D1
dec_int, ls_int = decile_ls(ret_intra_m, ret_total_m)
# 总反转: 用过去 1 月总收益排序
dec_tot, ls_tot_raw = decile_ls(ret_total_m, ret_total_m)
ls_tot = -ls_tot_raw

a_on, v_on, s_on, dd_on = perf(ls_on)
a_tot, v_tot, s_tot, dd_tot = perf(ls_tot)
a_int, v_int, s_int, dd_int = perf(ls_int)

# ---------- rank-IC 随持有期 ----------
def rank_ic(sig_hist, fut, h):
    ics = []
    for m in range(h, M - 1):
        s = sig_hist[:, m - h + 1:m + 1].sum(1)   # 过去 h 月累计
        f = fut[:, m + 1]
        ic = np.corrcoef(rankdata(s), rankdata(f))[0, 1]
        ics.append(ic)
    return np.mean(ics)

ic_1m = rank_ic(ret_total_m, ret_total_m, 1)     # 短期反转(应负)
ic_3m = rank_ic(ret_total_m, ret_total_m, 3)
ic_6m = rank_ic(ret_total_m, ret_total_m, 6)
ic_12m = rank_ic(ret_total_m, ret_total_m, 12)   # 中期动量(应正)

# ---------- 隔夜 vs 日内 统计 ----------
on_mean = ret_on_m.mean(); on_std = ret_on_m.std()
int_mean = ret_intra_m.mean(); int_std = ret_intra_m.std()
# 成分自相关(验证 R 反转 / P 持久)
R_autocorr = np.mean([np.corrcoef(R[:, m], R[:, m+1])[0,1] for m in range(M-1)])
P_autocorr = np.mean([np.corrcoef(P[:, m], P[:, m+1])[0,1] for m in range(M-1)])

metrics = {
    "N": N, "years": YEARS, "months": M,
    "overnight_rev_ann_ret": round(float(a_on),4), "overnight_rev_sharpe": round(float(s_on),3),
    "overnight_rev_dd": round(float(dd_on),4),
    "total_rev_ann_ret": round(float(a_tot),4), "total_rev_sharpe": round(float(s_tot),3),
    "total_rev_dd": round(float(dd_tot),4),
    "intraday_mom_ann_ret": round(float(a_int),4), "intraday_mom_sharpe": round(float(s_int),3),
    "intraday_mom_dd": round(float(dd_int),4),
    "ic_1m": round(float(ic_1m),4), "ic_3m": round(float(ic_3m),4),
    "ic_6m": round(float(ic_6m),4), "ic_12m": round(float(ic_12m),4),
    "R_autocorr": round(float(R_autocorr),3), "P_autocorr": round(float(P_autocorr),3),
    "overnight_mean": round(float(on_mean),5), "overnight_std": round(float(on_std),4),
    "intraday_mean": round(float(int_mean),5), "intraday_std": round(float(int_std),4),
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    for k,vv in metrics.items(): f.write(f"{k}={vv}\n")
print("METRICS", metrics)

# ============ 图1: 隔夜反转十分位 ============
fig, ax = plt.subplots(figsize=(10.5, 5.2))
x = np.arange(n_dec); w = 0.8
vals = dec_on.mean(1)   # D1..D10 of future return when sorted by past overnight
colors = ["#C44E52" if v >= 0 else "#2F4B7C" for v in vals]
ax.bar(x, vals, w, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels([f"D{k+1}" for k in range(n_dec)])
ax.set_title("短期反转：按过去 1 月『隔夜收益』排序的十分位下月平均收益", fontsize=12, fontweight="bold")
ax.set_xlabel("隔夜收益十分位（D1=最大隔夜输家 → D10=最大隔夜赢家）")
ax.set_ylabel("下月平均收益"); ax.grid(True, alpha=0.3, axis="y")
ax.annotate(f"买输家卖赢家: 年化 {a_on:.1%} / Sharpe {s_on:.2f} / 回撤 {dd_on:.1%}",
            xy=(0, vals[0]), xytext=(2.6, vals[0]*0.7), fontsize=10, color="#2F4B7C", fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#2F4B7C"))
plt.tight_layout()
plt.savefig(os.path.join(D, "str_decile_overnight.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图2: 三类策略累计净值 ============
cum_on = np.cumprod(1+ls_on); cum_tot = np.cumprod(1+ls_tot); cum_int = np.cumprod(1+ls_int)
fig, ax = plt.subplots(figsize=(9.5, 5.2))
ax.plot(range(M-1), cum_on, color="#2F4B7C", lw=1.9, label=f"隔夜反转(买输家卖赢家) Sharpe {s_on:.2f}")
ax.plot(range(M-1), cum_tot, color="#C44E52", lw=1.8, label=f"总收益反转 Sharpe {s_tot:.2f}")
ax.plot(range(M-1), cum_int, color="#55A868", lw=1.8, label=f"日内动量 Sharpe {s_int:.2f}")
ax.set_title("隔夜反转 / 总反转 / 日内动量：累计净值对比", fontsize=12, fontweight="bold")
ax.set_xlabel("月份"); ax.set_ylabel("净值（起始=1）"); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(D, "str_curves.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图3: rank-IC 随持有期 (反转→动量 flip) ============
fig, ax = plt.subplots(figsize=(9.5, 5.0))
hs = ["1M","3M","6M","12M"]; ics = [ic_1m, ic_3m, ic_6m, ic_12m]
bars = ax.bar(hs, ics, color=["#2F4B7C","#4C78A8","#9ECAE1","#C44E52"], width=0.55)
ax.axhline(0, color="black", lw=0.8)
ax.set_title("rank-IC 随持有期：短期限反转(负) → 中期限动量(正)", fontsize=12, fontweight="bold")
ax.set_ylabel("rank-IC（信号=过去 h 月收益，预测下月收益）"); ax.grid(True, alpha=0.3, axis="y")
for b, vv in zip(bars, ics):
    ax.text(b.get_x()+b.get_width()/2, vv + (0.002 if vv>=0 else -0.004), f"{vv:+.3f}",
            ha="center", fontsize=11, fontweight="bold")
ax.annotate("反转区", xy=(0.5, ic_1m), xytext=(0.4, ic_1m-0.006), fontsize=9, color="#2F4B7C")
ax.annotate("动量区", xy=(3.5, ic_12m), xytext=(3.4, ic_12m+0.004), fontsize=9, color="#C44E52")
plt.tight_layout()
plt.savefig(os.path.join(D, "str_ic_horizon.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============ 图4: 隔夜 vs 日内 分解(自相关印证) ============
fig, ax = plt.subplots(figsize=(9.5, 5.0))
labels = ["隔夜 R 成分\n(过度反应)", "日内 P 成分\n(真实信息)"]
acs = [R_autocorr, P_autocorr]
bars = ax.bar(labels, acs, color=["#C44E52","#55A868"], width=0.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_ylabel("成分一阶自相关 (月)")
ax.set_title("日收益分解：隔夜含均值回复冲击(负自相关)，日内含持久信息(正自相关)", fontsize=11.5, fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
for b, vv in zip(bars, acs):
    ax.text(b.get_x()+b.get_width()/2, vv + (0.02 if vv>=0 else -0.05), f"{vv:+.2f}",
            ha="center", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(D, "str_overnight_intraday_decomp.png"), dpi=150, bbox_inches="tight")
plt.close()
print("DONE short-term-reversal images")
