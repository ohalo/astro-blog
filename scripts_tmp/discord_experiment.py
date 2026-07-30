#!/usr/bin/env python3
"""Discord 异常子序列实验 v2：
核心教学点——discord 的前提是『正常段有孪生兄弟』。
实验A：IID 随机游走本底 → discord 无对比度（失败案例，故意保留）
实验B：结构化日内模式本底 → 三种植入异常全部被 top-3 discord 命中
消融：排除区 / 窗口长度 / 无异常对照
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

rng = np.random.default_rng(2026)
OUT = "/Users/halo/workspace/astro-blog/public/images/discord-anomaly-detection"
os.makedirs(OUT, exist_ok=True)

# ---------- 工具 ----------
def znorm(a):
    s = a.std()
    if s < 1e-10:
        return np.zeros_like(a)
    return (a - a.mean()) / s

def matrix_profile_brute(x, m, excl):
    n = len(x) - m + 1
    subs = np.array([znorm(x[i:i+m]) for i in range(n)])
    mp = np.full(n, np.inf)
    nn = np.zeros(n, dtype=int)
    block = 500
    for i0 in range(0, n, block):
        i1 = min(i0 + block, n)
        D = np.sqrt(np.maximum(0, 2 * m - 2 * subs[i0:i1] @ subs.T))
        for k, i in enumerate(range(i0, i1)):
            d = D[k].copy()
            lo, hi = max(0, i - excl), min(n, i + excl + 1)
            d[lo:hi] = np.inf
            j = np.argmin(d)
            mp[i] = d[j]
            nn[i] = j
    return mp, nn

def top_discords(mp, m, k=3):
    mp2 = mp.copy()
    out = []
    for _ in range(k):
        i = int(np.argmax(mp2))
        out.append((i, mp2[i]))
        lo, hi = max(0, i - m), min(len(mp2), i + m)
        mp2[lo:hi] = -np.inf
    return out

# ========== 实验B：结构化日内本底（主实验） ==========
D, BARS = 60, 50           # 60 个交易日 × 50 根 bar
m = BARS                   # 窗口 = 1 天
# 日内模板：开盘下探 → 午盘修复 → 尾盘拉升（合成 U 型日内节奏）
tt = np.linspace(0, 1, BARS)
template = -0.0025 * np.sin(np.pi * tt) + 0.004 * tt**3  # bar 收益模板
template = template - template.mean() + 0.0001

ANOM_DAYS = {"闪崩回补": 15, "波动冻结": 34, "锯齿振荡": 48}
rets = []
for d in range(D):
    amp = rng.uniform(0.7, 1.3)
    day = amp * template + rng.normal(0, 0.0009, BARS)
    if d == ANOM_DAYS["闪崩回补"]:
        v = np.zeros(BARS)
        v[10:16] = -0.012          # 6 根 bar 崩 7%
        v[16:40] = 0.0022          # 缓慢回补
        day = v + rng.normal(0, 0.0009, BARS)
    elif d == ANOM_DAYS["波动冻结"]:
        day = rng.normal(0, 0.00006, BARS)   # 数据卡死/流动性冻结
    elif d == ANOM_DAYS["锯齿振荡"]:
        day = 0.006 * np.array([1 if i % 2 == 0 else -1 for i in range(BARS)]) \
              + rng.normal(0, 0.0009, BARS)
    rets.append(day)
x = np.concatenate(rets)
price = 100 * np.cumprod(1 + x)
T = len(x)
anom_pos = {k: v * BARS for k, v in ANOM_DAYS.items()}
print(f"序列长度 {T} bars, 异常位置: {anom_pos}")

mp, nn = matrix_profile_brute(x, m, excl=m // 2)
discords = top_discords(mp, m, k=3)
print("Top-3 discords:", [(i, round(d, 2)) for i, d in discords])
print(f"MP 中位数={np.median(mp):.2f}  最大值={mp.max():.2f}")

hits = 0
for name, pos in anom_pos.items():
    ok = any(abs(i - pos) <= m for i, _ in discords)
    hits += ok
    print(f"  {name}@{pos}: {'命中' if ok else '未命中'}")
print(f"命中 {hits}/3")

# 图1: 价格 + MP
colors = ["#d62728", "#ff7f0e", "#9467bd"]
fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})
axes[0].plot(price, lw=0.8, color="#333")
for (name, pos), c in zip(anom_pos.items(), colors):
    axes[0].axvspan(pos, pos + BARS, color=c, alpha=0.25, label=f"植入异常：{name}（第{pos//BARS}天）")
axes[0].set_ylabel("价格"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_title("60 个交易日 × 50 bar：共享日内模板的结构化本底 + 三个异常日")
axes[1].plot(mp, lw=0.8, color="#1f77b4")
for rank, (i, d) in enumerate(discords, 1):
    axes[1].plot([i], [d], "v", color="#d62728", ms=8)
    axes[1].annotate(f"#{rank} d={d:.1f}", (i, d), textcoords="offset points",
                     xytext=(0, 8), fontsize=9, color="#d62728", ha="center")
axes[1].set_ylabel("Matrix Profile"); axes[1].set_xlabel("bar 序号")
axes[1].set_title("MP 曲线：三个异常日全部成为最高峰（top-3 discord）")
axes[1].grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/discord-overview.png", dpi=130); plt.close(fig)

# 图2: 三个异常日 vs 正常日的日内形状
fig, axes = plt.subplots(1, 4, figsize=(13, 3.4))
normal_day = 20
cum_n = np.cumsum(rets[normal_day])
axes[0].plot(cum_n * 100, lw=1.4, color="#2ca02c")
axes[0].set_title(f"正常日（第{normal_day}天）", fontsize=11)
for ax, ((name, d_idx), c) in zip(axes[1:], zip(ANOM_DAYS.items(), colors)):
    cum = np.cumsum(rets[d_idx])
    ax.plot(cum * 100, lw=1.4, color=c)
    ax.set_title(f"{name}（第{d_idx}天）", fontsize=11)
for ax in axes:
    ax.grid(alpha=0.3); ax.set_xlabel("日内 bar")
axes[0].set_ylabel("日内累计收益 %")
fig.suptitle("日内累计收益路径：正常日彼此相似（有孪生兄弟），异常日独一无二", y=1.04)
fig.tight_layout(); fig.savefig(f"{OUT}/discord-shapes.png", dpi=130, bbox_inches="tight"); plt.close(fig)

# ========== 实验A：IID 本底失败案例 ==========
x_iid = rng.normal(0.0003, 0.011, T)
# 在同位置植入同样的三种异常（幅度等比放大到日线尺度）
x_iid[anom_pos["闪崩回补"]:anom_pos["闪崩回补"]+BARS] = np.concatenate(
    [np.full(6, -0.018), np.full(34, 0.0028), rng.normal(0, 0.002, 10)])
x_iid[anom_pos["波动冻结"]:anom_pos["波动冻结"]+BARS] = rng.normal(0, 0.0005, BARS)
x_iid[anom_pos["锯齿振荡"]:anom_pos["锯齿振荡"]+BARS] = 0.015 * np.array(
    [1 if i % 2 == 0 else -1 for i in range(BARS)])
mp_iid, _ = matrix_profile_brute(x_iid, m, excl=m // 2)
disc_iid = top_discords(mp_iid, m, k=3)
hits_iid = sum(any(abs(i - pos) <= m for i, _ in disc_iid) for pos in anom_pos.values())
print(f"[IID 本底] top-3 discords={[(i, round(d,2)) for i,d in disc_iid]}  命中 {hits_iid}/3")
print(f"[IID 本底] MP 中位数={np.median(mp_iid):.2f} 最大值={mp_iid.max():.2f}")

fig, axes = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
axes[0].plot(mp, lw=0.8, color="#1f77b4",
             label=f"结构化本底：中位数 {np.median(mp):.2f} / 峰值 {mp.max():.2f}（对比度 {mp.max()/np.median(mp):.1f}×）")
axes[1].plot(mp_iid, lw=0.8, color="#d62728",
             label=f"IID 噪声本底：中位数 {np.median(mp_iid):.2f} / 峰值 {mp_iid.max():.2f}（对比度 {mp_iid.max()/np.median(mp_iid):.1f}×）")
for ax in axes:
    for pos in anom_pos.values():
        ax.axvspan(pos, pos + BARS, color="gray", alpha=0.2)
    ax.legend(fontsize=9); ax.grid(alpha=0.3); ax.set_ylabel("MP")
axes[1].set_xlabel("bar 序号")
axes[0].set_title("同样的异常、不同的本底：discord 的对比度来自『正常段有多相似』")
fig.tight_layout(); fig.savefig(f"{OUT}/discord-background-contrast.png", dpi=130); plt.close(fig)

# ========== 消融1: 排除区 ==========
mp_bad, _ = matrix_profile_brute(x, m, excl=1)  # 只排除自身及紧邻
disc_bad = top_discords(mp_bad, m, k=3)
hits_bad = sum(any(abs(i - pos) <= m for i, _ in disc_bad) for pos in anom_pos.values())
print(f"[排除区=1] MP 中位数={np.median(mp_bad):.2f} 最大值={mp_bad.max():.2f}  命中 {hits_bad}/3")
print(f"[排除区=1] 模体端（MP 最小值）：正确 {mp.min():.3f} vs 无排除区 {mp_bad.min():.3f}")
fig, ax = plt.subplots(figsize=(10, 4.2))
ax.plot(mp, lw=0.9, color="#1f77b4", label=f"排除区 = m/2：峰值 {mp.max():.2f} / 谷值 {mp.min():.2f}")
ax.plot(mp_bad, lw=0.9, color="#d62728", alpha=0.75, label=f"排除区 ≈ 0：峰值 {mp_bad.max():.2f} / 谷值 {mp_bad.min():.2f}")
for pos in anom_pos.values():
    ax.axvspan(pos, pos + BARS, color="gray", alpha=0.15)
ax.set_xlabel("bar 序号"); ax.set_ylabel("MP")
ax.set_title("排除区消融：discord（峰）对自匹配相对鲁棒，但谷值被拉低——模体端会先失效")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/discord-exclusion-ablation.png", dpi=130); plt.close(fig)

# ========== 消融2: 窗口长度 ==========
ms = [10, 25, 50, 75, 100, 150]
hit_list = []
for mt in ms:
    mp_t, _ = matrix_profile_brute(x, mt, excl=mt // 2)
    ds = top_discords(mp_t, mt, k=3)
    h = sum(any(abs(i - pos) <= mt for i, _ in ds) for pos in anom_pos.values())
    hit_list.append(h)
    print(f"  m={mt:3d}  命中 {h}/3")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.plot(ms, hit_list, "o-", color="#d62728", lw=1.8)
ax.axvline(BARS, color="gray", ls="--", lw=1, label="异常真实长度 = 50 bar（1 天）")
ax.set_xlabel("MP 窗口长度 m"); ax.set_ylabel("Top-3 discord 命中数（/3）")
ax.set_yticks([0, 1, 2, 3])
ax.set_title("窗口长度扫描：m 与异常时间尺度匹配决定检出能力")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/discord-window-scan.png", dpi=130); plt.close(fig)

# ========== 消融3: 无异常对照 ==========
rets_null = []
for d in range(D):
    amp = rng.uniform(0.7, 1.3)
    rets_null.append(amp * template + rng.normal(0, 0.0009, BARS))
x_null = np.concatenate(rets_null)
mp_null, _ = matrix_profile_brute(x_null, m, excl=m // 2)
print(f"[无异常对照] MP max={mp_null.max():.2f}  中位数={np.median(mp_null):.2f}")
fig, ax = plt.subplots(figsize=(9, 4.2))
ax.hist(mp_null, bins=60, alpha=0.6, color="#2ca02c", density=True, label="无异常对照本底")
ax.hist(mp, bins=60, alpha=0.5, color="#1f77b4", density=True, label="含 3 处植入异常")
for i, d in discords:
    ax.axvline(d, color="#d62728", ls="--", lw=1)
ax.annotate("三处异常的 discord 分数", (discords[2][1], ax.get_ylim()[1]*0.7),
            fontsize=9, color="#d62728", ha="right")
ax.set_xlabel("MP 距离"); ax.set_ylabel("密度")
ax.set_title(f"Discord 分数 vs 本底分布：无异常世界的 MP 峰值也有 {mp_null.max():.1f}——阈值必须用对照校准")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/discord-null-dist.png", dpi=130); plt.close(fig)

print("done")
