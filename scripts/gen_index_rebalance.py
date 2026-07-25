#!/usr/bin/env python3
"""指数调仓前瞻交易文章配图生成：纳入/剔除事件研究 + 前瞻策略"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/index-rebalance-front-running"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(2026)

BLUE, RED, GREEN, ORANGE, GRAY = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#6b7280"

# ============================================================
# 事件研究：纳入 vs 剔除，围绕生效日 T0 的累计超额收益
# 关键时点：AD(公告日, T-5) → ED(生效日, T0)
# ============================================================
t = np.arange(-15, 21)   # T-15 到 T+20
AD = -5   # 公告日

def event_path(peak_pre, peak_post, sign, n_events=500, noise=0.006):
    """公告后到生效日单向漂移，生效日后反转"""
    paths = []
    for _ in range(n_events):
        p = np.zeros(len(t))
        for i, tt in enumerate(t):
            if tt < AD:
                p[i] = 0.0
            elif AD <= tt <= 0:
                # 公告到生效：指数基金前瞻买盘/卖盘推动
                frac = (tt - AD) / (0 - AD)
                p[i] = sign * peak_pre * frac
            else:
                # 生效后反转（价格压力回吐）
                p[i] = sign * peak_pre - sign * peak_post * (1 - np.exp(-tt / 6.0))
        p += rng.normal(0, noise, len(t)).cumsum() * 0.4
        paths.append(p)
    return np.array(paths)

incl = event_path(0.050, 0.028, +1)    # 纳入：涨后回吐
excl = event_path(0.045, 0.024, -1)    # 剔除：跌后修复

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
ax.plot(t, incl.mean(0)*100, color=GREEN, lw=2.6, label="纳入成份股")
ax.plot(t, excl.mean(0)*100, color=RED, lw=2.6, label="剔除成份股")
ax.axvline(AD, color=BLUE, ls="--", lw=1.6, alpha=0.8)
ax.axvline(0, color="black", ls=":", lw=1.6, alpha=0.8)
ax.text(AD+0.2, ax.get_ylim()[1]*0.85, "公告日 AD", fontsize=10, color=BLUE)
ax.text(0.3, ax.get_ylim()[1]*0.85, "生效日 ED", fontsize=10, color="black")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_title("指数纳入/剔除的累计超额收益：公告→生效漂移，生效后反转", fontsize=14.5, weight="bold")
ax.set_xlabel("相对生效日 (ED) 的交易日")
ax.set_ylabel("累计超额收益 (%)")
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/event-study.png", bbox_inches="tight")
plt.close()

# ============================================================
# 图2：成交量放大 —— 生效日的指数基金"收盘冲击"
# ============================================================
vol_mult = np.ones(len(t))
for i, tt in enumerate(t):
    if AD <= tt < 0:
        vol_mult[i] = 1.0 + 0.4 * (tt - AD) / (0 - AD)
    elif tt == 0:
        vol_mult[i] = 6.5   # 生效日成交量暴增（被动基金一次性建仓）
    elif tt > 0:
        vol_mult[i] = 1.0 + 0.8 * np.exp(-tt / 3.0)

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
colors = [RED if tt == 0 else (ORANGE if AD <= tt < 0 else GRAY) for tt in t]
ax.bar(t, vol_mult, color=colors, alpha=0.85, width=0.7)
ax.axhline(1.0, color=GRAY, ls="--", lw=1)
ax.axvline(AD, color=BLUE, ls="--", lw=1.4, alpha=0.7)
ax.set_title("生效日成交量放大 6.5 倍：被动基金的收盘刚需买盘", fontsize=14.5, weight="bold")
ax.set_xlabel("相对生效日 (ED) 的交易日")
ax.set_ylabel("成交量 / 正常日均（倍）")
ax.text(0.3, 6.0, "生效日\n(ED)", fontsize=10, color=RED, weight="bold")
ax.grid(alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(f"{OUT}/volume-spike.png", bbox_inches="tight")
plt.close()

# ============================================================
# 图3：前瞻策略净值 —— 公告日买入纳入股、卖出剔除股，生效日平仓
# ============================================================
n_events = 60
np.random.seed(7)
# 每个事件的收益：纳入多头吃 AD→ED 漂移
long_leg = rng.normal(0.042, 0.03, n_events)   # 纳入前瞻多头
short_leg = rng.normal(0.038, 0.03, n_events)  # 剔除前瞻空头
combo = (long_leg + short_leg) / 2
nav = np.cumprod(1 + combo)
nav = np.concatenate([[1.0], nav])

# 对照：生效日才跟随（被动基金一样吃亏在价格压力顶点）
naive = rng.normal(-0.012, 0.025, n_events)
nav_naive = np.cumprod(1 + naive)
nav_naive = np.concatenate([[1.0], nav_naive])

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
ax.plot(nav, color=GREEN, lw=2.6, label="前瞻策略（公告日进、生效日出）")
ax.plot(nav_naive, color=RED, lw=2.2, label="被动跟随（生效日才买）")
ax.axhline(1.0, color=GRAY, lw=0.8)
ax.set_title("前瞻交易 vs 被动跟随：抢在指数基金前面的价差", fontsize=14.5, weight="bold")
ax.set_xlabel("累计调仓事件数")
ax.set_ylabel("净值（起始 = 1）")
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/strategy-nav.png", bbox_inches="tight")
plt.close()

# ============================================================
# 图4：漂移幅度 vs 需求冲击（指数被动资金占流通比）
# ============================================================
demand = np.linspace(0.5, 12, 200)   # 指数买盘占流通市值 %
drift = 1.5 * np.sqrt(demand) + rng.normal(0, 0.4, 200)  # 平方根冲击
fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
ax.scatter(demand, drift, s=18, color=BLUE, alpha=0.5)
xs = np.linspace(0.5, 12, 100)
ax.plot(xs, 1.5*np.sqrt(xs), color=RED, lw=2.5, label="平方根冲击拟合")
ax.set_title("价格漂移随被动需求冲击呈平方根放大", fontsize=14.5, weight="bold")
ax.set_xlabel("指数被动买盘占流通市值 (%)")
ax.set_ylabel("公告→生效累计超额漂移 (%)")
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/demand-impact.png", bbox_inches="tight")
plt.close()

print("指数调仓前瞻 配图完成：", os.listdir(OUT))
