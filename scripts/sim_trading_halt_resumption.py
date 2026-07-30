#!/usr/bin/env python3
"""长期停牌复牌效应模拟：补跌缺口、流动性折价与可交易性
生成 4 张配图到 public/images/trading-halt-resumption-effect/
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)
OUT = Path(__file__).resolve().parent.parent / "public/images/trading-halt-resumption-effect"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------- 事件构造 ----------------
# 每个事件：停牌前价格路径 + 停牌期间市场路径 + 复牌后价格路径
N_EV = 900            # 停牌事件数（12年）
PRE, POST = 120, 60   # 复牌前后观察窗（交易日）

# 停牌时长分布：大多数 1-20 日，长尾到 250 日（重大资产重组）
halt_len = np.clip(rng.lognormal(2.6, 1.1, N_EV).astype(int) + 1, 1, 250)

# 停牌原因：0=一般事项(60%)，1=重组利好(25%)，2=风险警示/坏消息(15%)
reason = rng.choice([0, 1, 2], N_EV, p=[0.60, 0.25, 0.15])

beta = np.clip(rng.normal(1.0, 0.25, N_EV), 0.4, 1.8)

# 停牌期间市场累计收益（市场日波动 1.1%，可能大跌——2015式）
mkt_daily_all = rng.normal(0.0002, 0.011, (N_EV, 250))
mkt_cum_halt = np.array([mkt_daily_all[i, :halt_len[i]].sum() for i in range(N_EV)])

# 停牌期间个股"应有"的信息积累 = beta*市场 + 事件本身的公告效应 + 个股噪声
announce = np.where(reason == 1, rng.normal(0.18, 0.10, N_EV),
            np.where(reason == 2, rng.normal(-0.22, 0.12, N_EV),
                     rng.normal(0.0, 0.04, N_EV)))
idio_halt = rng.normal(0, 0.02, N_EV) * np.sqrt(halt_len)
fair_gap = beta * mkt_cum_halt + announce + idio_halt   # 复牌时的"公允"跳空

# 流动性折价：长期停牌后投资者要求补偿，与停牌时长的平方根成正比
liq_discount = -0.012 * np.sqrt(halt_len) * (1 + rng.normal(0, 0.3, N_EV))

# 涨跌停约束：A股复牌首日 ±10%（重组复牌历史上部分不设限，这里统一 ±10%）
LIMIT = 0.10
# 复牌后路径：两条通道分开建模——
# 通道1：缺口释放（fair_gap + 流动性折价一起砸出来），受涨跌停约束
# 通道2：流动性折价的缓慢修复（第3日起，每日修复剩余的8%），不会触及涨跌停
post_ret = np.zeros((N_EV, POST))
gap_un = fair_gap + liq_discount     # 待释放缺口
rec_un = -liq_discount.copy()        # 待修复的流动性折价（正数）
release_speed = 0.85
for d in range(POST):
    step_gap = np.clip(gap_un * release_speed + rng.normal(0, 0.015, N_EV), -LIMIT, LIMIT)
    gap_un = gap_un - step_gap
    step_rec = np.where(d >= 3, rec_un * 0.08, 0.0)
    rec_un = rec_un - step_rec
    post_ret[:, d] = np.clip(step_gap + step_rec, -LIMIT, LIMIT)

# 连续跌停/涨停天数（阈值 9.95% 抗噪声）
def limit_days(rets, side):
    out = np.zeros(N_EV, dtype=int)
    for i in range(N_EV):
        d = 0
        while d < POST and side * rets[i, d] >= LIMIT - 5e-4:
            d += 1
        out[i] = d
    return out

down_days = limit_days(post_ret, -1)
up_days = limit_days(post_ret, +1)

cum20 = post_ret[:, :20].sum(axis=1)
cum60 = post_ret.sum(axis=1)
day1 = post_ret[:, 0]

print(f"事件数 {N_EV}，停牌时长中位 {np.median(halt_len):.0f} 日，>60日占比 {(halt_len>60).mean()*100:.1f}%")
print(f"复牌首日平均 {day1.mean()*100:.2f}%，跌停开盘占比 {(day1<=-LIMIT+1e-9).mean()*100:.1f}%")
long_mask = halt_len > 60
print(f"长期停牌(>60日)复牌首日 {day1[long_mask].mean()*100:.2f}%，连续跌停天数中位 {np.median(down_days[long_mask]):.0f}")

# 市场缺口回归：复牌累计收益 vs beta*停牌期市场收益
from numpy.polynomial import polynomial as P
X = beta * mkt_cum_halt
slope, intercept = np.polyfit(X, cum20, 1)
resid = cum20 - (slope*X + intercept)
r2 = 1 - resid.var()/cum20.var()
tstat = slope / (np.sqrt(resid.var()/ (X.var()*(N_EV-2))))
print(f"\n补跌回归: 20日累计 = {intercept*100:.2f}% + {slope:.3f} × beta×市场缺口, R²={r2:.3f}, t={tstat:.1f}")

# 流动性反转策略：只做跌停开盘的事件，第一个打开跌停日买入，持有20日
# 注意：涨停开盘的事件一字板买不进，必须排除（这是第一版代码的 bug：
# 把涨停股当作 entry_day=0 可买入，白捡重组利好的未释放缺口，净收益虚高到 5.15%）
COST = 0.003
strat_ret = np.full(N_EV, np.nan)
for i in range(N_EV):
    if down_days[i] < 1:          # 只做跌停开盘的补跌事件
        continue
    e = down_days[i]
    if e >= POST - 20:
        continue
    strat_ret[i] = post_ret[i, e:e+20].sum()
valid = ~np.isnan(strat_ret)
net = strat_ret[valid] - COST*2
print(f"\n复牌抄底策略(仅跌停开盘,第一个打开跌停日买入,持有20日): 事件 {valid.sum()}, 平均毛收益 {strat_ret[valid].mean()*100:.2f}%, 净 {net.mean()*100:.2f}%, 胜率 {(net>0).mean()*100:.1f}%")
t_net = net.mean()/(net.std()/np.sqrt(len(net)))
print(f"t = {t_net:.2f}")

# 分停牌原因
for r, name in [(0, "一般事项"), (1, "重组利好"), (2, "风险警示")]:
    m = valid & (reason == r)
    if m.sum() > 0:
        print(f"  {name}: n={m.sum()}, 净收益 {(strat_ret[m]-COST*2).mean()*100:.2f}%")

# 被 bug 版本当作可交易的涨停事件（对比用）
buggy = np.full(N_EV, np.nan)
for i in range(N_EV):
    e = down_days[i]
    if e < POST - 20:
        buggy[i] = post_ret[i, e:e+20].sum()
bv = ~np.isnan(buggy)
print(f"[bug版对照] 不排除涨停开盘: 净收益 {(buggy[bv]-COST*2).mean()*100:.2f}% (虚高)")

# 对照：随机日期入场（同一个股票池，无复牌事件）
placebo = rng.normal(0.0002*20, 0.02*np.sqrt(20), max(valid.sum(), 1)) - COST*2
print(f"随机日期对照: 净收益 {placebo.mean()*100:.2f}%")

# ---------------- 画图 ----------------
C1, C2, C3, C4 = "#2563eb", "#dc2626", "#16a34a", "#f59e0b"

# 图1: 停牌时长 vs 复牌首日收益散点
fig, ax = plt.subplots(figsize=(9, 5))
colors = np.array([C1, C3, C2])[reason]
ax.scatter(halt_len, day1*100, s=10, c=colors, alpha=0.5)
for r, name, c_ in [(0, "一般事项", C1), (1, "重组利好", C3), (2, "风险警示", C2)]:
    ax.scatter([], [], c=c_, label=name)
ax.axhline(-10, color="gray", ls="--", lw=0.8)
ax.axhline(10, color="gray", ls="--", lw=0.8)
ax.set_xscale("log")
ax.set_xlabel("停牌时长（交易日，log）")
ax.set_ylabel("复牌首日收益 (%)")
ax.set_title("停牌越久，复牌首日越挤在涨跌停板上——价格发现被推迟而非取消")
ax.legend()
fig.tight_layout(); fig.savefig(OUT/"halt-length-day1.png", dpi=110); plt.close(fig)

# 图2: 补跌回归
fig, ax = plt.subplots(figsize=(8.5, 5))
ax.scatter(X*100, cum20*100, s=8, alpha=0.4, color=C1)
xs = np.linspace(X.min(), X.max(), 50)
ax.plot(xs*100, (slope*xs+intercept)*100, color=C2, lw=2,
        label=f"斜率 {slope:.2f} (t={tstat:.1f}), R²={r2:.2f}")
ax.set_xlabel("beta × 停牌期间市场累计收益 (%)")
ax.set_ylabel("复牌后20日累计收益 (%)")
ax.set_title("复牌后价格几乎完整补上停牌期间的市场缺口")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"catchup-regression.png", dpi=110); plt.close(fig)

# 图3: 事件时间平均路径（按停牌时长分组）
fig, ax = plt.subplots(figsize=(9, 5))
groups = [(halt_len <= 5, "≤5日", C1), ((halt_len > 5) & (halt_len <= 60), "6-60日", C4),
          (halt_len > 60, ">60日", C2)]
for m, label, c_ in groups:
    # 事件时间：去掉市场缺口后的超额路径
    excess = post_ret[m] - 0  # 简化展示原始路径
    path = np.concatenate([[0], np.cumsum(excess.mean(axis=0))])
    ax.plot(range(POST+1), path*100, label=f"停牌{label} (n={m.sum()})", color=c_, lw=1.5)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("复牌后交易日")
ax.set_ylabel("平均累计收益 (%)")
ax.set_title("复牌后平均路径：长期停牌组先深跌、后 20 日缓慢修复流动性折价")
ax.legend(); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(OUT/"event-time-paths.png", dpi=110); plt.close(fig)

# 图4: 抄底策略收益分布 vs 对照
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(net*100, bins=50, alpha=0.6, color=C3, label=f"复牌抄底 净均值 {net.mean()*100:.2f}% (t={t_net:.1f})")
ax.hist(placebo*100, bins=50, alpha=0.5, color="gray", label=f"随机日期对照 {placebo.mean()*100:.2f}%")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("20日净收益 (%)")
ax.set_ylabel("事件数")
ax.set_title("第一个打开跌停日买入：右移的均值 + 危险的左尾")
ax.legend()
fig.tight_layout(); fig.savefig(OUT/"dip-buying-distribution.png", dpi=110); plt.close(fig)

print("\n图片已生成:", sorted(p.name for p in OUT.glob("*.png")))
