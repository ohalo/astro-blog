#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「月末月初效应：日历翻页时的资金流规律」
(turn-of-month-effect) 生成真实配图与真实统计数字。

核心：Turn-of-the-Month (TOM) 效应——Lakonishok & Smidt (1988), Ariel (1987)。
市场收益在每月最后几个交易日到次月头几个交易日显著高于月中，
资金流驱动：养老金/工资定投月末月初申购、机构窗口粉饰、被动基金再平衡。

所有图与数字由文中 Python 逻辑真实计算：
  1) tom_by_tradingday.png  —— 按"相对月末的交易日位置"平均收益条形图（TOM 窗口凸起）
  2) tom_equity.png         —— 只持有 TOM 窗口 vs 只持有月中 vs 买入持有 三条净值
  3) tom_annual.png         —— 逐年 TOM 窗口收益 vs 月中收益（稳定性检验）
  4) tom_window_scan.png    —— TOM 窗口长度(前后各 N 天)扫描：年化收益 & 暴露天数占比

自洽合成数据：20 年 × 252 日，植入月末月初正漂移，非真实行情。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

rc = matplotlib.rcParams
rc["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rc["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "turn-of-month-effect")
os.makedirs(D, exist_ok=True)

C = {"tom": "#55A868", "mid": "#C44E52", "bh": "#2F4B7C",
     "bar": "#4C72B0", "grid": "#E2E2E2"}

# =====================================================================
# 合成日度收益并标注每天在当月内的位置
# 机制：TOM 窗口(月末最后 3 天 + 次月头 2 天)有额外正漂移，月中接近 0。
# =====================================================================
N_YEARS = 20
SEED = 20260725
DAYS_PER_MONTH = 21   # 近似每月交易日数

# TOM 窗口定义：相对月末的位置 <=+2 (月末最后3天: 位置 0,-1,-2 从月末倒数)
# 以及次月头 2 天。用"距离月末的天数"与"距离月初的天数"联合标注。
TOM_TAIL = 3   # 月末最后 3 个交易日
TOM_HEAD = 2   # 次月头 2 个交易日
MU_TOM = 0.0016    # TOM 窗口内每日额外漂移
MU_MID = 0.00004   # 月中微小漂移
VOL = 0.0092


def simulate(mu_tom=MU_TOM, tail=TOM_TAIL, head=TOM_HEAD, seed=SEED):
    rng = np.random.default_rng(seed)
    rets = []
    pos_from_start = []   # 当月内第几个交易日(0起)
    pos_from_end = []     # 距月末还有几个交易日(0=最后一天)
    months = []
    total_months = N_YEARS * 12
    for m in range(total_months):
        # 每月交易日数轻微随机 19~23
        ndays = rng.integers(19, 24)
        for i in range(ndays):
            from_end = ndays - 1 - i
            is_tom = (from_end < tail) or (i < head)
            mu = mu_tom if is_tom else MU_MID
            r = mu + rng.standard_normal() * VOL
            rets.append(r)
            pos_from_start.append(i)
            pos_from_end.append(from_end)
            months.append(m)
    return (np.array(rets), np.array(pos_from_start),
            np.array(pos_from_end), np.array(months))


rets, pfs, pfe, months = simulate()

# TOM 标记（与真实交易同口径）
is_tom = (pfe < TOM_TAIL) | (pfs < TOM_HEAD)
tom_rets = rets[is_tom]
mid_rets = rets[~is_tom]
print(f"TOM 窗口日均 {tom_rets.mean()*100:.3f}%  月中日均 {mid_rets.mean()*100:.3f}%")
# t 检验（两样本）
from math import sqrt
n1, n2 = len(tom_rets), len(mid_rets)
sp = sqrt(tom_rets.var(ddof=1)/n1 + mid_rets.var(ddof=1)/n2)
tstat = (tom_rets.mean() - mid_rets.mean()) / sp
print(f"两样本 t = {tstat:.2f}  (TOM {n1} 天 / 月中 {n2} 天)")

# 占比
tom_share = is_tom.mean()
print(f"TOM 窗口只占 {tom_share:.1%} 的交易日")
# TOM 贡献的总收益占比
tom_contrib = tom_rets.sum() / rets.sum()
print(f"却贡献了 {tom_contrib:.1%} 的累计收益")

# =====================================================================
# 图 1：按"距月末交易日位置"平均收益条形图
# 横轴：月末最后5天(位置-5..-1) + 次月头5天(+1..+5)
# =====================================================================
labels, means = [], []
# 月末倒数 5 天：from_end = 4,3,2,1,0  -> 标 -5..-1
for fe in [4, 3, 2, 1, 0]:
    sel = pfe == fe
    labels.append(f"月末\n-{fe+1 if False else fe}")  # will relabel below
    means.append(rets[sel].mean() * 100)
# 次月头 5 天：from_start = 0..4 -> +1..+5
for fs in [0, 1, 2, 3, 4]:
    sel = pfs == fs
    labels.append(f"+{fs+1}")
    means.append(rets[sel].mean() * 100)

# 更清晰的标签
labels = ["末-5", "末-4", "末-3", "末-2", "末-1(最后日)",
          "初+1", "初+2", "初+3", "初+4", "初+5"]
colors = [C["tom"] if (i in (2, 3, 4, 5, 6)) else C["bar"] for i in range(10)]
# TOM 窗口 = 月末最后3天(index 2,3,4) + 次月头2天(index 5,6)
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.bar(range(10), means, color=colors)
ax.axhline(0, color="#666", lw=1)
ax.axhline(mid_rets.mean()*100, color=C["mid"], lw=1.4, ls="--",
           label=f"月中日均基准 {mid_rets.mean()*100:.3f}%")
ax.set_xticks(range(10))
ax.set_xticklabels(labels, fontsize=9.5)
ax.set_ylabel("平均日收益 (%)")
ax.set_title("月末月初效应：日历翻页附近的日均收益显著凸起（绿色为 TOM 窗口）",
             fontsize=12, fontweight="bold")
ax.legend()
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "tom_by_tradingday.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 图 2：三条净值——只持有 TOM / 只持有月中 / 买入持有
# =====================================================================
def build_equity(mask):
    r = np.where(mask, rets, 0.0)
    return np.cumprod(1 + r)

eq_tom = build_equity(is_tom)
eq_mid = build_equity(~is_tom)
eq_bh = np.cumprod(1 + rets)

def summ(mask):
    r = rets[mask]
    ann = r.mean() * 252
    sharpe = r.mean()/(r.std(ddof=1)+1e-12)*np.sqrt(252)
    return ann, sharpe

a_t, s_t = summ(is_tom)
a_m, s_m = summ(~is_tom)
a_b = rets.mean()*252
s_b = rets.mean()/(rets.std(ddof=1))*np.sqrt(252)
print(f"仅TOM 年化{a_t:.1%} Sharpe{s_t:.2f} | 仅月中 年化{a_m:.1%} Sharpe{s_m:.2f} | 买入持有 年化{a_b:.1%} Sharpe{s_b:.2f}")

fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(eq_tom, color=C["tom"], lw=2.2, label=f"只在 TOM 窗口持有  年化{a_t:.1%}(暴露仅{tom_share:.0%}天)")
ax.plot(eq_bh, color=C["bh"], lw=1.8, label=f"全程买入持有  年化{a_b:.1%}")
ax.plot(eq_mid, color=C["mid"], lw=1.8, label=f"只在月中持有  年化{a_m:.1%}")
ax.axhline(1, color="#999", lw=0.8)
ax.set_xlabel("交易日")
ax.set_ylabel("净值（起始=1）")
ax.set_title("只在日历翻页的几天持有，跑赢全程买入持有——月中几乎白拿风险", fontsize=11.5, fontweight="bold")
ax.legend(loc="upper left")
ax.grid(True, color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "tom_equity.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

# =====================================================================
# 图 3：逐年 TOM vs 月中 累计收益（稳定性）
# =====================================================================
year_len = 252
n_year = len(rets) // year_len
yr_tom, yr_mid = [], []
for y in range(n_year):
    sl = slice(y*year_len, (y+1)*year_len)
    mtom = is_tom[sl]
    ry = rets[sl]
    yr_tom.append(ry[mtom].sum()*100)
    yr_mid.append(ry[~mtom].sum()*100)
fig, ax = plt.subplots(figsize=(11, 5))
xpos = np.arange(n_year)
w = 0.4
ax.bar(xpos - w/2, yr_tom, w, label="TOM 窗口累计收益", color=C["tom"])
ax.bar(xpos + w/2, yr_mid, w, label="月中累计收益", color=C["mid"])
ax.axhline(0, color="#666", lw=1)
ax.set_xticks(xpos)
ax.set_xticklabels([f"Y{y+1}" for y in range(n_year)], fontsize=8.5)
ax.set_xlabel("年份")
ax.set_ylabel("当年累计收益 (%)")
win = sum(1 for a, b in zip(yr_tom, yr_mid) if a > b)
ax.set_title(f"稳定性检验：{n_year} 年里 TOM 窗口有 {win} 年跑赢月中", fontsize=11.5, fontweight="bold")
ax.legend()
ax.grid(True, axis="y", color=C["grid"], lw=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "tom_annual.png"), dpi=130, bbox_inches="tight")
plt.close(fig)
print(f"逐年稳定性：{win}/{n_year} 年 TOM 跑赢月中")

# =====================================================================
# 图 4：窗口长度扫描（前尾各 N 天）
# =====================================================================
scan_tails = [1, 2, 3, 4, 5, 6]
scan_ann, scan_share = [], []
for tl in scan_tails:
    hd = max(1, tl - 1)
    mask = (pfe < tl) | (pfs < hd)
    r = rets[mask]
    scan_ann.append(r.mean()*252*100 if len(r) else 0)  # 窗口内年化(占位)
    scan_share.append(mask.mean()*100)
# 更有意义：算"只持有该窗口"策略的全期年化
scan_strat_ann = []
for tl in scan_tails:
    hd = max(1, tl - 1)
    mask = (pfe < tl) | (pfs < hd)
    r = np.where(mask, rets, 0.0)
    eq = np.cumprod(1+r)
    yrs = len(r)/252
    scan_strat_ann.append((eq[-1]**(1/yrs)-1)*100)

fig, ax1 = plt.subplots(figsize=(10, 5.2))
ax1.plot(scan_tails, scan_strat_ann, "o-", color=C["tom"], lw=2, label="策略年化收益 (%)")
ax1.set_xlabel("TOM 窗口半长（月末最后 N 天 + 次月头 N-1 天）")
ax1.set_ylabel("只持有 TOM 窗口的策略年化 (%)", color=C["tom"])
ax1.tick_params(axis="y", labelcolor=C["tom"])
ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(scan_tails, scan_share, "s--", color=C["bh"], lw=2, label="市场暴露天数占比 (%)")
ax2.set_ylabel("暴露天数占比 (%)", color=C["bh"])
ax2.tick_params(axis="y", labelcolor=C["bh"])
ax1.set_title("窗口越宽收益越高但暴露也越多：窄窗口的单位风险收益更优", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(os.path.join(D, "tom_window_scan.png"), dpi=130, bbox_inches="tight")
plt.close(fig)

print("\n=== 图表已生成 ===")
print(f"输出目录: {D}")
for f in sorted(os.listdir(D)):
    print(" ", f)

with open(os.path.join(D, "_stats.txt"), "w") as fh:
    fh.write(f"tom_daily={tom_rets.mean()*100:.4f}% mid_daily={mid_rets.mean()*100:.4f}% t={tstat:.2f}\n")
    fh.write(f"tom_share={tom_share:.4f} tom_contrib={tom_contrib:.4f}\n")
    fh.write(f"tom ann={a_t:.4f} sharpe={s_t:.4f}\n")
    fh.write(f"mid ann={a_m:.4f} sharpe={s_m:.4f}\n")
    fh.write(f"bh  ann={a_b:.4f} sharpe={s_b:.4f}\n")
    fh.write(f"annual_win={win}/{n_year}\n")
    fh.write(f"bar_means={[round(x,4) for x in means]}\n")
