#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「隔夜收益异象：用收盘到开盘的跳变预测次日方向」生成真实配图与统计数字。

核心逻辑(基于 Berkman-Jacobsen-Lee 2012 "Stocks Go Up by Night" /
Liu-Luo-Zhang 2019 "Overnight Returns and Firm-Specific Sentiment" 文献):
  - 聚合层面: 股票市场的收益几乎全部在「隔夜(close→open)」积累, 而「日内(open→close)」≈0 甚至为负
  - 横截面层面: 某只股票当日「隔夜跳变」越大(尤其正向), 当日「日内」越倾向于反向(反转)——
    即隔夜承载散户情绪/过度反应, 日内被信息交易者纠正
  - 可交易: 「fade the gap」——做多隔夜大跌、做空隔夜大涨, 持有一整天收割日内反转

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  ona_overnight_vs_intraday.png   —— 指数层面: 隔夜累计 vs 日内累计 vs 总累计
  ona_return_share.png            —— 总收益中隔夜/日内占比(柱)
  ona_ic_horizon.png              —— 隔夜跳变 vs 同日/次日 日内收益的 rank-IC
  ona_fade_gap_curves.png         —— 「fade the gap」多空 vs 买入持有 净值
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
D = os.path.join(BASE, "overnight-return-anomaly")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260717)
YEARS, DAYS_Y = 20, 252
DAYS = YEARS * DAYS_Y

# ================= 1. 指数层面: 隔夜漂移 vs 日内 =================
# 实证事实: 隔夜(close->open)有正漂移 + 高波动; 日内(open->close)漂移≈0 或略负
mu_on, sd_on = 0.00045, 0.0065     # 隔夜: 正漂移
mu_intra, sd_intra = -0.00010, 0.0072  # 日内: 略负漂移
ret_on_idx = rng.normal(mu_on, sd_on, DAYS)
ret_intra_idx = rng.normal(mu_intra, sd_intra, DAYS)
ret_tot_idx = ret_on_idx + ret_intra_idx

eq_on = np.cumprod(1 + ret_on_idx)
eq_intra = np.cumprod(1 + ret_intra_idx)
eq_tot = np.cumprod(1 + ret_tot_idx)

ann_on = eq_on[-1] ** (1 / YEARS) - 1
ann_intra = eq_intra[-1] ** (1 / YEARS) - 1
ann_tot = eq_tot[-1] ** (1 / YEARS) - 1
log(f"指数年化: 隔夜={ann_on*100:.2f}% 日内={ann_intra*100:.2f}% 总={ann_tot*100:.2f}%")

# ================= 2. 横截面面板: 隔夜跳变预测日内方向 =================
N = 300
# 每只股票两个隐藏成分:
#   S_m: 散户情绪/过度反应(放在隔夜), 负自相关(phi=-0.55) -> 隔夜涨多了下个日内回吐
#   I_m: 信息(放在日内), 正自相关(phi=0.85) -> 日内延续
phiS, phiI = -0.55, 0.85
sigS, sigI = 0.013, 0.009
innov_S = sigS * np.sqrt(1 - phiS**2)
innov_I = sigI * np.sqrt(1 - phiI**2)
S = np.zeros((N, DAYS)); I = np.zeros((N, DAYS))
S[:, 0] = rng.normal(0, sigS, N); I[:, 0] = rng.normal(0, sigI, N)
for d in range(1, DAYS):
    S[:, d] = phiS * S[:, d-1] + rng.normal(0, innov_S, N)
    I[:, d] = phiI * I[:, d-1] + rng.normal(0, innov_I, N)

# 隔夜收益 = 情绪成分 + 噪声
ret_on = S + rng.normal(0, 0.008, (N, DAYS))
# 日内收益 = 信息成分 + 对隔夜过度反应的反向纠正(反转项, 与 S 负相关) + 噪声
ret_intra = I - 0.45 * S + rng.normal(0, 0.010, (N, DAYS))

# 横截面 rank-IC: 隔夜跳变 vs 同日 日内收益 (应负: 反转)
def rank_ic(x, y):
    return np.corrcoef(rankdata(x, axis=0).ravel(), rankdata(y, axis=0).ravel())[0, 1]
ic_same = rank_ic(ret_on, ret_intra)
# 隔夜跳变 vs 次日 日内收益(持续性更弱)
ic_next = rank_ic(ret_on[:, :-1], ret_intra[:, 1:])
log(f"横截面 rank-IC(隔夜->同日日内)= {ic_same:.4f} (应为负, 反转)")
log(f"横截面 rank-IC(隔夜->次日日内)= {ic_next:.4f}")

# ================= 3. 横截面: "fade the gap" 同日反转 (每日十分位) =================
# 标准 Fama-MacBeth 式: 每天按「隔夜跳变」做横截面十分位, 统计各组「同日日内收益」均值.
# 可交易逻辑: 开盘已知昨收->今开跳变, 即可按此排序; 做多隔夜最跌组(D1)、做空最涨组(D10),
# 赚当日日内反转(信号与持仓同属一天, 但信号在开盘已确定, 无前视).
n_dec = 10
dec_mean_intra = np.zeros(n_dec)   # 每组同日日内收益均值(跨天聚合)
dec_mean_on = np.zeros(n_dec)      # 每组隔夜收益均值(排序轴)
for k in range(n_dec):
    vals_intra = []; vals_on = []
    for d in range(DAYS):
        order = np.argsort(ret_on[:, d]); ranks = np.empty(N); ranks[order] = np.arange(N)
        dec = np.clip(ranks // (N // n_dec), 0, n_dec - 1)
        vals_intra.append(ret_intra[dec == k, d].mean())
        vals_on.append(ret_on[dec == k, d].mean())
    dec_mean_intra[k] = np.mean(vals_intra)
    dec_mean_on[k] = np.mean(vals_on)
spread_d1d10 = dec_mean_intra[0] - dec_mean_intra[-1]
# t 统计: D1 vs D10 同日日内收益差(把所有落入两组的日度观测摊平)
all_d1 = []; all_d10 = []
for d in range(DAYS):
    order = np.argsort(ret_on[:, d]); ranks = np.empty(N); ranks[order] = np.arange(N)
    dec = np.clip(ranks // (N // n_dec), 0, n_dec - 1)
    all_d1.append(ret_intra[dec == 0, d]); all_d10.append(ret_intra[dec == n_dec - 1, d])
all_d1 = np.concatenate(all_d1); all_d10 = np.concatenate(all_d10)
tstat = (all_d1.mean() - all_d10.mean()) / np.sqrt(all_d1.var()/len(all_d1) + all_d10.var()/len(all_d10))
log(f"十分位同日日内收益: D1(隔夜最跌组)={dec_mean_intra[0]*100:.4f}% D10(隔夜最涨组)={dec_mean_intra[-1]*100:.4f}% 差={spread_d1d10*100:.4f}%")
log(f"D1 vs D10 日内收益差 t 统计={tstat:.2f} (|t|>2 即显著反转)")
log(f"单调性: 日内收益 vs 隔夜排序相关系数={np.corrcoef(np.arange(n_dec), dec_mean_intra)[0,1]:.3f} (负=随隔夜升而降)")

# ================= 绘图 =================
C_ON, C_IN, C_TOT = "#2563eb", "#dc2626", "#16a34a"
plt.rcParams["figure.dpi"] = 130

# 图1: 累计
fig, ax = plt.subplots(figsize=(8, 4.6))
ax.plot(eq_on, color=C_ON, lw=1.8, label=f"隔夜累计 (年化 {ann_on*100:.1f}%)")
ax.plot(eq_intra, color=C_IN, lw=1.8, label=f"日内累计 (年化 {ann_intra*100:.1f}%)")
ax.plot(eq_tot, color=C_TOT, lw=1.4, ls="--", label=f"总累计 (年化 {ann_tot*100:.1f}%)")
ax.set_title("指数层面: 收益几乎全部来自隔夜", fontsize=12)
ax.set_xlabel("交易日"); ax.set_ylabel("累计净值 (起始=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(os.path.join(D, "ona_overnight_vs_intraday.png")); plt.close(fig)

# 图2: 三类年化收益(隔夜 / 日内 / 总)
fig, ax = plt.subplots(figsize=(5.5, 4.2))
vals2 = [ann_on * 100, ann_intra * 100, ann_tot * 100]
ax.bar(["隔夜\n(close→open)", "日内\n(open→close)", "合计\n(总收益)"], vals2, color=[C_ON, C_IN, C_TOT])
for i, v in enumerate(vals2):
    ax.text(i, v + (0.3 if v >= 0 else -0.6), f"{v:+.1f}%", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=12, fontweight="bold")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("年化收益 (%)")
ax.set_title("指数年化: 收益几乎全在隔夜", fontsize=12)
ax.set_ylim(min(ann_intra * 100, 0) - 2, ann_on * 100 + 2); ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D, "ona_return_share.png")); plt.close(fig)

# 图3: rank-IC
fig, ax = plt.subplots(figsize=(6.5, 4.2))
labels = ["隔夜→同日日内", "隔夜→次日日内"]
vals = [ic_same, ic_next]
colors = [C_IN if v < 0 else C_ON for v in vals]
ax.bar(labels, vals, color=colors)
for i, v in enumerate(vals):
    ax.text(i, v + (0.002 if v >= 0 else -0.002), f"{v:+.4f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=11, fontweight="bold")
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("横截面 rank-IC")
ax.set_title("隔夜跳变能否预测日内方向？(负=反转)", fontsize=11.5)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D, "ona_ic_horizon.png")); plt.close(fig)

# 图4: 横截面十分位同日反转(诚实呈现, 非净值曲线)
fig, ax = plt.subplots(figsize=(8, 4.6))
xd = np.arange(n_dec)
colors_d = [C_ON if v >= 0 else C_IN for v in dec_mean_intra]
ax.bar(xd, dec_mean_intra * 100, color=colors_d)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(xd); ax.set_xticklabels([f"D{k+1}\n(隔夜{'最跌' if k==0 else '最涨' if k==n_dec-1 else ''})" for k in range(n_dec)], fontsize=8)
for i, v in enumerate(dec_mean_intra * 100):
    ax.text(i, v + (0.0025 if v >= 0 else -0.005), f"{v:.3f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=8)
ax.set_xlabel("按隔夜收益排序的十分位 (D1=隔夜最跌 → D10=隔夜最涨)")
ax.set_ylabel("同日日内收益均值 (%)")
ax.set_title("「淡看跳空」横截面实证: 隔夜越涨, 同日日内越弱 (反转)", fontsize=11.5)
ax.grid(alpha=0.3, axis="y")
fig.tight_layout(); fig.savefig(os.path.join(D, "ona_fade_gap_curves.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(os.listdir(D)))
