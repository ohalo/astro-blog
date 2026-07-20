#!/usr/bin/env python3
"""
为文章「Piotroski F-score 财务健康打分」(piotroski-fscore) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月，每只股票有一个潜在「财务健康度」h_i。
  - 用 9 条基本面准则（Piotroski 2000 原文口径）各生成一个与 h_i 相关的代理信号，
    逐条判定 0/1，加总成 0-9 的 F-score。
  - 未来 1 月收益被设定为随 F-score 正相关（财务健康公司长期跑赢），叠加市场与噪声。
  - 验证：F-score 分布、高分组 vs 低分组 vs 市场累计净值、按 F 桶的未来收益单调性、
    高低 F 长短因子。
"""
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "piotroski-fscore")
os.makedirs(D, exist_ok=True)

C = {
    "grid": "#DDDDDD",
    "pos": "#2F4B7C",
    "neg": "#C44E52",
    "ls": "#55A868",
    "mk": "#8172B3",
    "gold": "#E1A100",
    "blue": "#4C72B0",
}


# ============================================================
# 1) 合成面板 + 9 条 Piotroski 准则
# ============================================================
def simulate_panel(N=600, M=144, seed=20260720):
    rng = np.random.default_rng(seed)
    # 潜在财务健康度
    h = rng.normal(0.0, 1.0, size=N)
    drift = rng.normal(0.0, 0.12, size=(N, M)).cumsum(axis=1) * 0.05

    def sig(beta, noise=0.6):
        return beta * h[:, None] + 0.3 * drift + rng.normal(0.0, noise, size=(N, M))

    # 盈利能力（4 条）
    roa = sig(1.0)                       # ROA > 0
    d_roa = sig(1.0)                     # ΔROA > 0
    cfo = sig(1.0)                       # 经营现金流 > 0
    cfo_minus_roa = sig(0.8) - sig(0.8)  # 经营现金流 > 净利润（应计低）
    # 杠杆/流动性/融资（3 条）
    d_lev = -sig(1.0)                    # 杠杆下降（Δ长期债率 < 0）
    d_cr = sig(1.0)                      # 流动比率上升
    d_shr = -sig(1.0)                    # 未增发（Δ股本 ≤ 0）
    # 经营效率（2 条）
    d_gm = sig(1.0)                      # 毛利率上升
    d_at = sig(1.0)                      # 资产周转率上升

    criteria = np.stack([
        (roa > 0).astype(int),
        (d_roa > 0).astype(int),
        (cfo > 0).astype(int),
        (cfo_minus_roa > 0).astype(int),
        (d_lev > 0).astype(int),
        (d_cr > 0).astype(int),
        (d_shr > 0).astype(int),
        (d_gm > 0).astype(int),
        (d_at > 0).astype(int),
    ], axis=0)  # (9, N, M)

    F = criteria.sum(axis=0)  # (N, M) 取值 0-9

    mkt = rng.normal(0.005, 0.04, size=M)
    # 未来 1 月收益：随 F-score 正相关，叠加市场与噪声
    future = (0.004 + 0.0045 * (F / 9.0)
              + 0.35 * mkt[None, :]
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    return F, future, mkt


F, future, mkt = simulate_panel()
N, M = F.shape


# ============================================================
# 图一：F-score 截面分布（0-9）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(F.flatten(), bins=range(0, 11), color=C["blue"], alpha=0.85,
        edgecolor="white", align="left", rwidth=0.85)
ax.set_xlabel("F-score（0 = 最不健康，9 = 最健康）")
ax.set_ylabel("股票-月数")
ax.set_title("F-score 截面分布：多数公司落在 3-6 的健康中段")
ax.set_xticks(range(0, 10))
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "fscore_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：高 F（8-9）vs 低 F（0-2）vs 市场 累计净值
# ============================================================
def port_nav(mask_fn):
    nav = np.ones(M)
    for t in range(1, M):
        idx = mask_fn(t)
        nav[t] = nav[t - 1] * (1 + future[idx, t - 1].mean())
    return nav


def high_mask(t, lo=8):
    return F[:, t] >= lo

def low_mask(t, hi=2):
    return F[:, t] <= hi

nav_high = port_nav(high_mask)
nav_low = port_nav(low_mask)
nav_mkt = np.cumprod(1 + mkt)

months = np.arange(M)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, nav_high, color=C["ls"], lw=2.2, label="高 F 组（8-9）")
ax.plot(months, nav_low, color=C["neg"], lw=2.2, label="低 F 组（0-2）")
ax.plot(months, nav_mkt, color="gray", lw=1.6, ls="--", label="市场")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("高 F-score 组合显著跑赢低 F 组与市场")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "fscore_high_low_nav.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：按 F-score 桶的未来 1 月平均收益（单调）
# ============================================================
buckets = {}
for fv in range(0, 10):
    buckets[fv] = future[F == fv].mean() * 100
xs = list(buckets.keys())
ys = [buckets[k] for k in xs]
colors = [C["neg"] if v < 0 else C["pos"] for v in ys]

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(xs, ys, color=colors)
ax.set_xlabel("F-score（0 → 9，越健康）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("按 F-score 的未来收益：从 0 到 9 单调抬升")
ax.set_xticks(range(0, 10))
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "fscore_bucket_returns.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：高低 F 长短因子累计净值
# ============================================================
ls_ret = np.zeros(M)
for t in range(1, M):
    hi = F[:, t] >= 8
    lo = F[:, t] <= 2
    ls_ret[t] = future[hi, t - 1].mean() - future[lo, t - 1].mean()
ls_ret[0] = 0
nav_ls = np.cumprod(1 + ls_ret)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, nav_ls, color=C["gold"], lw=2.2,
        label="高 F 多/低 F 空 长短因子")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("Piotroski 长短因子：多健康 / 空病弱，长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "fscore_ls_curve.png"), dpi=130)
plt.close(fig)

print("✅ piotroski-fscore 配图生成完成：", os.listdir(D))
