#!/usr/bin/env python3
"""
为文章「隐含波动率与已实现波动率套利：波动率风险的双面」(implied-vs-realized-vol)
生成真实配图。所有图表均由文中 Python 代码真实计算生成（随机波动率模型模拟）。

图表：
  1. iv_rv_ts.png          隐含波动 vs 已实现波动 10 年时间序列（IV 整体高于 RV）
  2. iv_rv_scatter.png     IV vs RV 散点 + 45° 线，展示正楔形的「波动率风险溢价」
  3. vrp_hist.png          波动率风险溢价(VRP = IV − RV，单位 vol%) 分布（均值显著为正）
  4. strat_equity.png      做空 VRP 的累计净值 vs 买入持有底层指数
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
D = os.path.join(BASE, "implied-vs-realized-vol")
os.makedirs(D, exist_ok=True)
np.random.seed(20260711)

# ============================================================
# 1) 模拟随机波动率价格序列（Heston 式 Euler 离散），收益用小数
# ============================================================
N = 252 * 10 + 60
dt = 1.0 / 252
mu = 0.07
v0 = 0.04
kappa, theta, xi, rho = 4.0, 0.04, 0.55, -0.65
S = np.zeros(N); v = np.zeros(N)
S[0], v[0] = 100.0, v0
for t in range(1, N):
    z1 = np.random.randn()
    z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.randn()
    v[t] = max(v[t-1] + kappa * (theta - v[t-1]) * dt + xi * np.sqrt(max(v[t-1], 1e-6) * dt) * z2, 1e-5)
    r = (mu - 0.5 * v[t-1]) * dt + np.sqrt(v[t-1] * dt) * z1
    if np.random.rand() < 0.004:
        r += -np.random.uniform(0.06, 0.14)
    S[t] = S[t-1] * np.exp(r)
ret = np.diff(np.log(S))                     # 日收益（小数）
M = len(ret)

# ============================================================
# 2) 已实现波动（RV）：30 日滚动年化（小数）
# ============================================================
WIN = 30
rv = np.full(M, np.nan)
for i in range(WIN, M):
    rv[i] = np.std(ret[i - WIN + 1:i + 1], ddof=1) * np.sqrt(252)

# ============================================================
# 3) 隐含波动（IV）：前瞻 30 日「实际」已实现波动 + 正的风险溢价 + 噪声
#    （透明模型设定，用于演示；因拥有完整路径，未来波动可得）
# ============================================================
fwd = 30
iv = np.full(M, np.nan)
for i in range(M - fwd):
    fut = np.std(ret[i + 1:i + 1 + fwd], ddof=1) * np.sqrt(252)
    premium = 0.020 + 0.010 * np.sin(i / 120.0)        # 正且均值回复的波动率风险溢价（vol 单位）
    iv[i] = max(fut + premium + np.random.randn() * 0.004, 0.01)

# ============================================================
# 图 1：IV vs RV 时间序列（以 vol% 显示）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.8))
idx = np.arange(M)
m1 = ~np.isnan(iv) & ~np.isnan(rv)
ax.plot(idx[m1], iv[m1] * 100, color="#d62728", lw=1.4, label="隐含波动率 IV（前瞻）")
ax.plot(idx[m1], rv[m1] * 100, color="#1f77b4", lw=1.2, alpha=0.9, label="已实现波动率 RV（后顾）")
ax.fill_between(idx[m1], rv[m1] * 100, iv[m1] * 100, where=(iv[m1] > rv[m1]),
                color="#d62728", alpha=0.08, label="IV − RV > 0（溢价区）")
ax.set_xlabel("交易日（近 10 年）", fontsize=11)
ax.set_ylabel("年化波动率 (%)", fontsize=11)
ax.set_title("隐含波动整体高于已实现波动：波动率风险溢价的「常态」", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "iv_rv_ts.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 2：IV vs RV 散点 + 45° 线
# ============================================================
fig, ax = plt.subplots(figsize=(7.6, 7.0))
ax.scatter(rv[m1] * 100, iv[m1] * 100, s=8, color="#1f77b4", alpha=0.35, edgecolors="none")
lim = [max(np.nanmin(rv) * 100, 5), min(np.nanmax(iv) * 100, 80)]
ax.plot(lim, lim, "k--", lw=1.6, label="45° 线（IV = RV）")
p = np.polyfit(rv[m1] * 100, iv[m1] * 100, 1)
xs = np.linspace(lim[0], lim[1], 50)
ax.plot(xs, np.polyval(p, xs), color="#d62728", lw=2.0, label=f"拟合斜率={p[0]:.2f}")
ax.set_xlabel("已实现波动率 RV (%)", fontsize=11)
ax.set_ylabel("隐含波动率 IV (%)", fontsize=11)
ax.set_title("IV–RV 散点：点云整体落在 45° 线上方", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "iv_rv_scatter.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 3：波动率风险溢价 VRP = IV − RV（vol 单位，%）
# ============================================================
vrp = (iv[m1] - rv[m1]) * 100.0
mean_vrp = vrp.mean()
fig, ax = plt.subplots(figsize=(11, 5.4))
ax.hist(vrp, bins=60, color="#1f77b4", alpha=0.75, edgecolor="white", linewidth=0.4)
ax.axvline(0, color="black", lw=1.4, ls="--", label="零线")
ax.axvline(mean_vrp, color="#d62728", lw=2.2, label=f"均值 = {mean_vrp:.2f}%（显著为正）")
ax.set_xlabel("波动率风险溢价 VRP = IV − RV (vol %)", fontsize=11)
ax.set_ylabel("频数", fontsize=11)
ax.set_title("波动率风险溢价长期为正：做空它（卖波动）在数学上占优", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.25, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "vrp_hist.png"), dpi=150, bbox_inches="tight")
plt.close()

# ============================================================
# 图 4：做空 VRP 的累计净值 vs 买入持有底层指数
#     P&L 逻辑：每月初卖出 Delta 对冲跨式（≈ 卖出方差互换），
#     到期损益 ≈ 名义 × (IV_entry² − RV_realized²)，折算成月度收益率复利。
# ============================================================
lev = 2.0
rebal = 21
r_series = np.zeros(M)
for i in range(0, M - fwd, rebal):
    if np.isnan(iv[i]) or (i + fwd >= M):
        continue
    fut = np.std(ret[i + 1:i + 1 + fwd], ddof=1) * np.sqrt(252)
    r_series[i] = lev * (iv[i]**2 - fut**2)          # 月度收益率（小数）
equity = 100.0 * np.cumprod(1 + r_series)
eq_idx = S[1:] / S[1] * 100.0
# 对齐长度
mn = min(len(equity), M)
fig, ax = plt.subplots(figsize=(11, 5.8))
ax.plot(np.arange(mn), equity[:mn], color="#2ca02c", lw=1.8, label="做空 VRP 累计净值（卖波动）")
ax.plot(np.arange(mn), eq_idx[:mn], color="#ff7f0e", lw=1.5, alpha=0.85, label="买入持有底层指数")
ax.set_xlabel("交易日", fontsize=11)
ax.set_ylabel("净值（起点=100）", fontsize=11)
ax.set_title("做空波动率溢价：平稳爬升 vs 权益指数的剧烈回撤", fontsize=12.5, fontweight="bold")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(os.path.join(D, "strat_equity.png"), dpi=150, bbox_inches="tight")
plt.close()

print("✅ 隐含/已实现波动配图生成完成：", sorted(os.listdir(D)))
print(f"   平均 VRP={mean_vrp:.2f}%  期末做空VRP净值={equity[-1]:.1f}  指数={eq_idx[-1]:.1f}")
