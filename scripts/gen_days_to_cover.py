#!/usr/bin/env python3
"""轧空天数 Days-to-Cover 文章配图生成：合成融券面板 + 逼空事件研究"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/days-to-cover-short-squeeze"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(42)

BLUE, RED, GREEN, ORANGE, GRAY = "#2563eb", "#dc2626", "#16a34a", "#ea580c", "#6b7280"

# ============================================================
# 合成横截面：1000 只股票，DTC = 融券余额 / 日均成交量
# ============================================================
n = 1000
# 融券余额比率（右偏），日均成交量占流通比（右偏）
short_shares = np.exp(rng.normal(-3.0, 0.85, n))       # 融券占流通股比例
turnover = np.exp(rng.normal(-3.4, 0.7, n))            # 日均成交量占流通股比例
dtc = short_shares / turnover                          # 轧空天数
dtc = np.clip(dtc, 0.1, 40)

# ---------- 图1：DTC 分布 ----------
fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
ax.hist(dtc, bins=60, color=BLUE, alpha=0.75, edgecolor="white", linewidth=0.4)
med = np.median(dtc)
ax.axvline(med, color=RED, ls="--", lw=2, label=f"中位数 {med:.1f} 天")
ax.axvline(10, color=ORANGE, ls="--", lw=2, label="逼空警戒线 10 天")
ax.set_title("轧空天数 Days-to-Cover 的横截面分布（右偏长尾）", fontsize=15, weight="bold")
ax.set_xlabel("Days-to-Cover（融券余额 / 日均成交量，单位：天）")
ax.set_ylabel("股票数量")
ax.legend(fontsize=11)
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/dtc-distribution.png", bbox_inches="tight")
plt.close()

# ============================================================
# 事件研究：高 DTC + 触发买盘 后的逼空累计收益
# ============================================================
# 三组：低DTC(<3)、中DTC(3-10)、高DTC(>10)，触发正向催化剂后 T0..T20 的路径
horizon = 20
t = np.arange(-5, horizon + 1)

def squeeze_path(peak, noise_sc, n_events=400):
    """逼空累计超额路径：T0 前平缓，T0 后陡升再回吐"""
    paths = []
    for _ in range(n_events):
        base = np.zeros(len(t))
        # T0 前小幅积累
        for i, tt in enumerate(t):
            if tt <= 0:
                base[i] = 0.002 * tt
            else:
                # 逼空：快速上冲后部分回吐
                rise = peak * (1 - np.exp(-tt / 4.0))
                giveback = peak * 0.35 * max(0, (tt - 8)) / (horizon - 8)
                base[i] = rise - giveback
        base += rng.normal(0, noise_sc, len(t)).cumsum() * 0.15
        paths.append(base)
    return np.array(paths)

high = squeeze_path(0.28, 0.02)
mid = squeeze_path(0.11, 0.018)
low = squeeze_path(0.04, 0.015)

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
for arr, c, lab in [(high, RED, "高 DTC (>10天)"), (mid, ORANGE, "中 DTC (3-10天)"), (low, GRAY, "低 DTC (<3天)")]:
    m = arr.mean(axis=0) * 100
    ax.plot(t, m, color=c, lw=2.5, label=lab)
ax.axvline(0, color="black", ls=":", lw=1.5, alpha=0.7)
ax.text(0.3, ax.get_ylim()[1]*0.9, "T0 催化剂触发", fontsize=10, color="black")
ax.axhline(0, color=GRAY, lw=0.8)
ax.set_title("正向催化剂后累计超额收益：DTC 越高逼空越猛", fontsize=15, weight="bold")
ax.set_xlabel("相对催化剂日的交易日")
ax.set_ylabel("累计超额收益 (%)")
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/squeeze-event-study.png", bbox_inches="tight")
plt.close()

# ============================================================
# 图3：DTC 五分组的未来收益 & 波动
# ============================================================
Q = 5
# 高 DTC 平时收益偏低（被做空是负信号），但波动更大、右尾更肥
dtc_rank = np.argsort(np.argsort(dtc)) / (n - 1)
base_ret = 0.008 - 0.010 * dtc_rank                    # 平时高DTC收益低
tail = np.where(rng.random(n) < 0.03 * dtc_rank, rng.exponential(0.25, n), 0)  # 逼空尾部
fwd = base_ret + rng.normal(0, 0.05, n) + tail

order = np.argsort(dtc)
bins = np.array_split(order, Q)
mean_ret = [fwd[b].mean() * 100 for b in bins]
vol = [fwd[b].std() * 100 for b in bins]

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
x = np.arange(Q)
colors = [GRAY, GRAY, ORANGE, ORANGE, RED]
bars = ax.bar(x, mean_ret, color=colors, alpha=0.85, width=0.6)
ax2 = ax.twinx()
ax2.plot(x, vol, color=BLUE, marker="o", lw=2.5, label="月收益波动率")
ax.set_xticks(x)
ax.set_xticklabels([f"Q{i+1}\n{'最低' if i==0 else '最高' if i==Q-1 else ''}DTC" for i in range(Q)])
ax.axhline(0, color="black", lw=0.8)
ax.set_title("DTC 五分组：平均收益偏低但尾部波动放大", fontsize=15, weight="bold")
ax.set_ylabel("月均收益 (%)", color=GRAY)
ax2.set_ylabel("月收益波动率 (%)", color=BLUE)
for b, v in zip(bars, mean_ret):
    ax.text(b.get_x()+b.get_width()/2, v + (0.05 if v>=0 else -0.12), f"{v:.2f}", ha="center", fontsize=10)
ax2.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig(f"{OUT}/dtc-quintile.png", bbox_inches="tight")
plt.close()

# ============================================================
# 图4：逼空信号策略净值 vs 基准
# ============================================================
T = 240
mkt = np.cumprod(1 + rng.normal(0.006, 0.045, T))
# 逼空捕捉策略：高DTC + 动量确认时进场，胜率低但赢时大
strat_ret = rng.normal(0.004, 0.03, T)
squeeze_hits = rng.random(T) < 0.06
strat_ret[squeeze_hits] += rng.exponential(0.08, squeeze_hits.sum())
strat = np.cumprod(1 + strat_ret)

fig, ax = plt.subplots(figsize=(11.7, 6.5), dpi=100)
ax.plot(mkt, color=GRAY, lw=2, label="市场基准 (buy & hold)")
ax.plot(strat, color=RED, lw=2.5, label="DTC 逼空捕捉策略")
ax.set_title("逼空捕捉策略净值：收益由少数大赢单驱动", fontsize=15, weight="bold")
ax.set_xlabel("交易月")
ax.set_ylabel("净值（起始 = 1）")
ax.legend(fontsize=11, loc="upper left")
ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig(f"{OUT}/strategy-nav.png", bbox_inches="tight")
plt.close()

print("Days-to-Cover 配图完成：", os.listdir(OUT))
