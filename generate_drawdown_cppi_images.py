#!/usr/bin/env python3
"""
为文章「最大回撤控制与 CPPI：把下行约束写进仓位管理」(drawdown-cppi-control)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

CPPI（Constant Proportion Portfolio Insurance，恒定比例投资组合保险）：
    缓冲垫 C_t = A_t - F_t
    风险敞口 E_t = m · C_t   （clamp 到 [0, A_t]）
    安全资产(债券)持有 = A_t - E_t
    A_{t+1} = E_t·(1+r_risky) + (A_t - E_t)·(1+r_safe)

本文用「带一次崩盘的 GBM 市场」演示：CPPI 如何把下行锁在 floor 附近，
以及乘数 m 在收益与 cushion 侵蚀之间的权衡。
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
D = os.path.join(BASE, "drawdown-cppi-control")
os.makedirs(D, exist_ok=True)

C = {"eq": "#2F4B7C", "bh": "#C44E52", "floor": "#55A868", "cush": "#F2C0C0",
     "grid": "#DDDDDD", "m3": "#4C72B0", "m5": "#DD8452", "m2": "#8172B3",
     "bond": "#999999", "crash": "#B22222"}

def max_drawdown(equity):
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return dd.min(), dd

# ============================================================
# 1) 模拟「带崩盘的市场」日度收益
# ============================================================
def simulate_market(T=252 * 6, seed=2026):
    rng = np.random.default_rng(seed)
    mu_d, sig_d = 0.09 / 252, 0.17 / np.sqrt(252)   # 年化 9% / 17%
    r = rng.normal(mu_d, sig_d, size=T)
    # 在第 ~ 60% 处注入一次 -38% 崩盘 + 后续 5 日余波
    crash = int(T * 0.60)
    r[crash] += -0.38
    for k in range(1, 6):
        r[crash + k] += -0.05 * np.exp(-k / 2.0)
    # 再注入一次温和熊市（~85% 处 -18%）
    bear = int(T * 0.85)
    r[bear] += -0.18
    return r, crash, bear

def simulate_bonds(T=252 * 6, seed=11):
    rng = np.random.default_rng(seed)
    # 短端安全资产，年化 2.5% / 波动 1%
    return rng.normal(0.025 / 252, 0.01 / np.sqrt(252), size=T)

# ============================================================
# 2) CPPI 回测
# ============================================================
def run_cppi(r_risky, r_safe, m, floor_level, rebalance=True):
    A = np.empty(len(r_risky) + 1)
    E = np.empty(len(r_risky) + 1)         # 风险敞口
    C = np.empty(len(r_risky) + 1)         # 缓冲垫
    A[0] = 1.0
    for t in range(len(r_risky)):
        C[t] = A[t] - floor_level
        E_t = m * C[t]
        E_t = min(max(E_t, 0.0), A[t])     # clamp 到 [0, A_t]
        bond = A[t] - E_t
        A[t + 1] = E_t * (1 + r_risky[t]) + bond * (1 + r_safe[t])
        E[t] = E_t; C[t] = C[t]
    E[-1] = E[-2]; C[-1] = C[-2]
    return A, E, C

# ============================================================
# 主计算
# ============================================================
r_risky, crash, bear = simulate_market()
r_safe = simulate_bonds()
T = len(r_risky)
t = np.arange(T + 1)

floor_level = 0.80          # 保护底线 = 初始财富的 80%（最多亏 20%）
cppi3 = run_cppi(r_risky, r_safe, m=3.0, floor_level=floor_level)
cppi5 = run_cppi(r_risky, r_safe, m=5.0, floor_level=floor_level)
cppi2 = run_cppi(r_risky, r_safe, m=2.0, floor_level=floor_level)

# 买入持有（全仓风险资产）
bh = np.empty(T + 1); bh[0] = 1.0
for i in range(T):
    bh[i + 1] = bh[i] * (1 + r_risky[i])

# 纯债券
bond_only = np.empty(T + 1); bond_only[0] = 1.0
for i in range(T):
    bond_only[i + 1] = bond_only[i] * (1 + r_safe[i])

dd3, _ = max_drawdown(cppi3[0])
dd5, _ = max_drawdown(cppi5[0])
dd2, _ = max_drawdown(cppi2[0])
ddbh, _ = max_drawdown(bh)

# ---------- 图 1：CPPI(m=3) 与 买入持有 的权益曲线 ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(t, bh, color=C["bh"], lw=1.3, label="买入持有 (全仓风险资产)")
ax.plot(t, cppi3[0], color=C["eq"], lw=1.6, label="CPPI (m=3, floor=0.80)")
ax.axhline(floor_level, color=C["floor"], ls="--", lw=1.4, label="保护底线 floor=0.80")
ax.scatter([crash, bear], [bh[crash], bh[bear]], color=C["crash"], zorder=5,
           s=40, marker="v", label="崩盘日")
ax.set_xlabel("交易日"); ax.set_ylabel("组合净值 (初始=1.0)")
ax.set_title("CPPI 把下行锁在底线附近：崩盘后净值贴近 floor，而非腰斩")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cppi_equity.png"), dpi=130); plt.close()

# ---------- 图 2：资产价值 / floor / cushion 结构 ----------
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, cppi3[0], color=C["eq"], lw=1.5, label="资产总值 A_t")
ax.plot(t, np.full_like(t, floor_level), color=C["floor"], ls="--", lw=1.3, label="floor (底线)")
ax.fill_between(t, floor_level, cppi3[0], where=(cppi3[0] >= floor_level),
                color=C["cush"], alpha=0.6, label="缓冲垫 C_t = A_t - floor")
ax.scatter([crash], [cppi3[0][crash]], color=C["crash"], zorder=5, s=40, marker="v")
ax.set_xlabel("交易日"); ax.set_ylabel("净值")
ax.set_title("缓冲垫 C_t 是风险敞口的上限：崩盘时 cushion 被快速侵蚀后自动降仓")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cppi_floor_cushion.png"), dpi=130); plt.close()

# ---------- 图 3：回撤对比（CPPI vs 买入持有）----------
_, dd_series_bh = max_drawdown(bh)
_, dd_series_3 = max_drawdown(cppi3[0])
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.fill_between(t, dd_series_bh * 100, color=C["bh"], alpha=0.35, label="买入持有 回撤")
ax.plot(t, dd_series_bh * 100, color=C["bh"], lw=0.8)
ax.plot(t, dd_series_3 * 100, color=C["eq"], lw=1.4, label="CPPI (m=3) 回撤")
ax.axhline(dd3 * 100, color=C["eq"], ls="--", lw=1, alpha=0.7)
ax.set_xlabel("交易日"); ax.set_ylabel("回撤 (%)")
ax.set_title("CPPI 最大回撤被约束在 floor 缺口附近 (约 %.1f%%)" % (dd3 * 100))
ax.legend(loc="lower left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cppi_drawdown.png"), dpi=130); plt.close()

# ---------- 图 4：乘数 m 的权衡（终值 vs 最大回撤）----------
ms = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
finals, maxdds = [], []
for m in ms:
    A, _, _ = run_cppi(r_risky, r_safe, m=m, floor_level=floor_level)
    finals.append(A[-1])
    d, _ = max_drawdown(A)
    maxdds.append(d * 100)
fig, ax1 = plt.subplots(figsize=(11, 4.4))
ax1.plot(ms, finals, color=C["eq"], lw=1.6, marker="o", label="终值 A_T")
ax1.set_xlabel("乘数 m"); ax1.set_ylabel("终值 (初始=1.0)", color=C["eq"])
ax1.tick_params(axis="y", labelcolor=C["eq"]); ax1.grid(True, color=C["grid"], lw=0.6)
ax2 = ax1.twinx()
ax2.plot(ms, maxdds, color=C["bh"], lw=1.6, marker="s", label="最大回撤 (%)")
ax2.set_ylabel("最大回撤 (%)", color=C["bh"]); ax2.tick_params(axis="y", labelcolor=C["bh"])
ax1.set_title("乘数 m 的权衡：m 越大上行越多，但 cushion 侵蚀越快、回撤越深")
l1, la1 = ax1.get_legend_handles_labels(); l2, la2 = ax2.get_legend_handles_labels()
ax1.legend(l1 + l2, la1 + la2, loc="center right", fontsize=8)
plt.tight_layout(); plt.savefig(os.path.join(D, "cppi_multiplier.png"), dpi=130); plt.close()

print("=== 最大回撤控制与 CPPI 关键数字 ===")
print("样本: 日度 %d 天 (约 %.1f 年)，含 1 次 -38%% 崩盘 + 1 次 -18%% 熊市" % (T, T / 252))
print("floor=%.2f, 风险资产年化 %.1f%% / 波动 %.0f%%, 债券年化 %.1f%%" % (floor_level, 9.0, 17.0, 2.5))
print("买入持有: 终值=%.3f, 最大回撤=%.1f%%" % (bh[-1], ddbh * 100))
print("CPPI m=2: 终值=%.3f, 最大回撤=%.1f%%" % (cppi2[0][-1], dd2 * 100))
print("CPPI m=3: 终值=%.3f, 最大回撤=%.1f%%" % (cppi3[0][-1], dd3 * 100))
print("CPPI m=5: 终值=%.3f, 最大回撤=%.1f%%" % (cppi5[0][-1], dd5 * 100))
print("乘数扫描 m=%s: 终值=%s" % (ms, [round(x, 3) for x in finals]))
print("乘数扫描 m=%s: 最大回撤=%s" % (ms, [round(x, 1) for x in maxdds]))
print("\n图片已保存到:", D)
