# -*- coding: utf-8 -*-
"""加权价格贡献 WPC 配图生成（Barclay-Hendershott 2003）"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/weighted-price-contribution"
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ---------------------------------------------------------------
# 模拟设定：一天分三个时段（盘前 / 连续竞价 / 盘后）
# 有效价格是随机游走，信息到达强度在三个时段不同
# 每个时段的观测收益 = 该时段信息增量 + 微观结构噪声
# ---------------------------------------------------------------
N_DAYS = 1000
TRUE_SHARE = np.array([0.15, 0.70, 0.15])   # 真实信息份额：盘前/盘中/盘后
NOISE_STD = np.array([0.0030, 0.0015, 0.0035])  # 时段噪声（盘前盘后更吵）
DAILY_VOL = 0.02

def simulate(n_days, true_share, noise_std, seed=42):
    r = np.random.default_rng(seed)
    # 每天的信息总量（有肥尾：有的日子几乎没信息）
    day_info = r.normal(0, DAILY_VOL, n_days) * np.where(r.random(n_days) < 0.25, 2.2, 0.6)
    sess_info = np.empty((n_days, 3))
    for k in range(3):
        sess_info[:, k] = day_info * true_share[k]
    noise = r.normal(0, noise_std, (n_days, 3))
    # 噪声在日内会部分回吐：让相邻时段噪声负相关（bid-ask bounce 式）
    sess_ret = sess_info + noise - np.roll(noise, 1, axis=1) * 0.5
    day_ret = sess_ret.sum(axis=1)
    return sess_ret, day_ret

sess_ret, day_ret = simulate(N_DAYS, TRUE_SHARE, NOISE_STD)

def wpc(sess_ret, day_ret):
    """Barclay-Hendershott 加权价格贡献"""
    w = np.abs(day_ret) / np.abs(day_ret).sum()
    ratio = sess_ret / day_ret[:, None]
    return (w[:, None] * ratio).sum(axis=0)

def naive_share(sess_ret, day_ret, eps=1e-6):
    """不加权：直接对每日 ratio 取平均"""
    mask = np.abs(day_ret) > eps
    ratio = sess_ret[mask] / day_ret[mask, None]
    return ratio.mean(axis=0), ratio

WPC = wpc(sess_ret, day_ret)
NAIVE, ratios = naive_share(sess_ret, day_ret)
print("true :", TRUE_SHARE)
print("WPC  :", np.round(WPC, 3))
print("naive:", np.round(NAIVE, 3))

SESS = ["盘前", "连续竞价", "盘后"]
COLORS = ["#e07b39", "#3a7ca5", "#7b5ea7"]

# ---------------- 图1：一天的价格路径分时段 ----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
r1 = np.random.default_rng(7)
seg_len = [30, 120, 30]
path = [100.0]
bounds = [0]
for k, L in enumerate(seg_len):
    steps = r1.normal(TRUE_SHARE[k] * 0.9 / L, [0.10, 0.035, 0.11][k], L)
    for s in steps:
        path.append(path[-1] * (1 + s / 100 * 4))
    bounds.append(len(path) - 1)
path = np.array(path)
x = np.arange(len(path))
for k in range(3):
    lo, hi = bounds[k], bounds[k + 1]
    ax.plot(x[lo:hi + 1], path[lo:hi + 1], color=COLORS[k], lw=1.8, label=SESS[k])
    ax.axvspan(lo, hi, color=COLORS[k], alpha=0.06)
for k in range(3):
    lo, hi = bounds[k], bounds[k + 1]
    ax.annotate(f"{SESS[k]}\nΔp={path[hi]-path[lo]:+.2f}",
                xy=((lo + hi) / 2, path.min() + 0.05),
                ha="center", fontsize=10, color=COLORS[k])
ax.set_title("一天的价格路径：日收益被三个时段瓜分，谁的份额算「价格发现」？")
ax.set_xlabel("时间（bar）")
ax.set_ylabel("价格")
ax.legend(loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/wpc-day-sessions.png", dpi=110)
plt.close(fig)

# ---------------- 图2：WPC vs 不加权平均 vs 真实份额 ----------------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4))
ax = axes[0]
xw = np.arange(3)
w_ = 0.26
ax.bar(xw - w_, TRUE_SHARE, w_, label="真实信息份额", color="#666")
ax.bar(xw, WPC, w_, label="WPC（加权）", color="#3a7ca5")
ax.bar(xw + w_, NAIVE, w_, label="不加权平均", color="#c0504d")
ax.set_xticks(xw, SESS)
ax.set_ylabel("份额")
ax.set_title("WPC 贴住真实份额，不加权平均被小收益日炸飞")
ax.axhline(0, color="k", lw=0.7)
ax.legend(fontsize=9)

ax = axes[1]
# 展示 ratio 的分布：小 |day_ret| 时 ratio 爆炸
mask = np.abs(day_ret) > 1e-6
ax.scatter(np.abs(day_ret[mask]) * 100, ratios[:, 2], s=6, alpha=0.35, color="#7b5ea7")
ax.set_ylim(-8, 8)
ax.axhline(TRUE_SHARE[2], color="k", ls="--", lw=1, label=f"真实盘后份额 {TRUE_SHARE[2]:.2f}")
ax.set_xlabel("|日收益|（%）")
ax.set_ylabel("盘后时段贡献比 (r_k / r_day)")
ax.set_title("日收益越小，单日贡献比越疯：这就是必须加权的原因")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/wpc-vs-naive.png", dpi=110)
plt.close(fig)

# ---------------- 图3：样本量收敛性（WPC 抽样分布） ----------------
fig, ax = plt.subplots(figsize=(9, 4.4))
sizes = [60, 125, 250, 500, 1000]
box_data = []
for n in sizes:
    vals = []
    for b in range(400):
        sr, dr = simulate(n, TRUE_SHARE, NOISE_STD, seed=1000 + b * 7 + n)
        vals.append(wpc(sr, dr)[2])
    box_data.append(vals)
bp = ax.boxplot(box_data, tick_labels=[f"{n}天" for n in sizes], showfliers=False,
                patch_artist=True)
for patch in bp["boxes"]:
    patch.set_facecolor("#b3cde0")
ax.axhline(TRUE_SHARE[2], color="#c0504d", ls="--", lw=1.4, label="真实盘后份额 0.15")
ax.set_ylabel("WPC（盘后）")
ax.set_title("WPC 的抽样分布：60 天窗口噪声很大，250 天以上才稳定")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/wpc-convergence.png", dpi=110)
plt.close(fig)

# ---------------- 图4：滚动 WPC 侦测价格发现迁移 ----------------
# 前 500 天盘后份额 0.15，后 500 天升到 0.40（模拟盘后交易活跃化）
sr1, dr1 = simulate(500, np.array([0.15, 0.70, 0.15]), NOISE_STD, seed=21)
sr2, dr2 = simulate(500, np.array([0.10, 0.50, 0.40]), NOISE_STD, seed=22)
sr_all = np.vstack([sr1, sr2])
dr_all = np.concatenate([dr1, dr2])
win = 250
roll = np.full((len(dr_all), 3), np.nan)
for t in range(win, len(dr_all)):
    roll[t] = wpc(sr_all[t - win:t], dr_all[t - win:t])
fig, ax = plt.subplots(figsize=(9.5, 4.6))
for k in range(3):
    ax.plot(roll[:, k], color=COLORS[k], lw=1.6, label=SESS[k])
ax.axvline(500, color="k", ls=":", lw=1.2)
ax.text(505, 0.82, "制度变化：盘后交易活跃化", fontsize=10)
ax.axhline(0.70, color=COLORS[1], ls="--", lw=0.8, alpha=0.6)
ax.axhline(0.40, color=COLORS[2], ls="--", lw=0.8, alpha=0.6)
ax.set_xlabel("交易日")
ax.set_ylabel("滚动 250 日 WPC")
ax.set_title("滚动 WPC：价格发现从盘中向盘后迁移，约一个窗口长度后完全反映")
ax.legend(loc="center left")
fig.tight_layout()
fig.savefig(f"{OUT}/wpc-rolling-shift.png", dpi=110)
plt.close(fig)

print("done:", os.listdir(OUT))
