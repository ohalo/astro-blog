#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「因子跨越检验：用 GRS 检验判断新因子是否真有增量信息」生成真实配图与统计数字。

核心机制(基于 Huberman & Kandel 1987 的因子跨越检验 / GRS 型联合 F 检验):
  我们已有一个旧因子模型(例如只有市场因子 mkt)。现在来了一个「候选新因子」g,
  要判断它到底有没有增量信息 —— 即它能否解释旧模型解释不了的资产收益变动。

  对每个测试资产 i 做回归:
      R_i = a_i + b_i * mkt + c_i * g + e_i        (无约束, 含截距)
  跨越检验的零假设 H0: 新因子无增量信息  ⇔  所有资产对新因子的载荷 c_i = 0 (联合)。
  若 H0 成立, 加不加 g 都一样, c_i ≈ 0; 若 g 真有增量, 部分 c_i 显著 ≠ 0。

  统计量(单新因子的 GRS 型联合 F 检验):
      F = [ (SSR_r - SSR_u) / K ] / [ SSR_u / (T - p) ]
      SSR_r = Σ_i (R_i 对 [1, mkt] 回归的残差平方和)
      SSR_u = Σ_i (R_i 对 [1, mkt, g] 回归的残差平方和)
      K = 限制个数(=资产数), p = 无约束参数数(=3)
      在 H0 下 F ~ F_{K, T-p}。p 值小 => 拒绝 H0 => 新因子真有增量(不可被旧模型跨越)。

  全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib), 无占位符。

  两个对照场景:
    (A) 候选因子 = 与资产收益正交的纯噪声  => H0 成立, F 不显著(p 大)
    (B) 候选因子 = 对部分资产有真实载荷且独立于市场  => 拒绝 H0, F 显著(p 小)

图片:
  grs_loadings.png        —— 两场景下各资产对新因子的载荷 c_i(场景A≈0 vs 场景B显著)
  grs_scenario.png        —— 两场景 F 统计量与 F 临界值对比
  grs_pvalue.png          —— 两场景 p 值(对数轴) vs 显著性阈值 0.05
  grs_r2_increment.png    —— 加入新因子带来的 R² 增量(冗余 vs 真增量)
"""
import os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "factor-spanning-test")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

C_BLUE = "#2c7fb8"
C_RED = "#d7301f"
C_GREEN = "#1a9850"
C_GREY = "#7f7f7f"
C_GRID = "#dddddd"

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260718)

# ================= 1. 数据: K 个测试组合 + 旧因子(市场) =================
K = 8               # 测试资产(组合)数
T = 240             # 月度观测
mkt = rng.normal(0.0075, 0.038, T)                 # 旧因子: 市场
beta = rng.uniform(0.4, 1.4, K)
R = np.zeros((T, K))
for i in range(K):
    R[:, i] = rng.normal(0, 0.002, 1) + beta[i] * mkt + rng.normal(0, 0.02, T)
port_names = [f"组合{i+1}" for i in range(K)]

def ssr(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ b
    return float(resid @ resid)

def spanning_test(R, mkt, g):
    """H0: 新因子 g 的载荷 c_i = 0 (联合)。返回 F 统计量、p 值、各资产 c_i 载荷。"""
    Tt, Kk = R.shape
    Xr = np.column_stack([np.ones(Tt), mkt])       # 受限: [1, mkt]
    Xu = np.column_stack([np.ones(Tt), mkt, g])    # 无约束: [1, mkt, g]
    SSR_r = sum(ssr(Xr, R[:, i]) for i in range(Kk))
    SSR_u = sum(ssr(Xu, R[:, i]) for i in range(Kk))
    p = 3                                          # 无约束参数数
    F = ((SSR_r - SSR_u) / Kk) / (SSR_u / (Tt - p))
    pval = 1.0 - stats.f.cdf(F, Kk, Tt - p)
    # 各资产载荷 c_i (无约束回归第三列)
    c = np.array([np.linalg.lstsq(Xu, R[:, i], rcond=None)[0][2] for i in range(Kk)])
    return F, pval, c, SSR_r, SSR_u

def r2_inc(R, mkt, g):
    """加入 g 后平均 R² 增量(单新因子相对 [1,mkt])。"""
    def r2(X, y):
        b, *_ = np.linalg.lstsq(X, y, rcond=None)
        resid = y - X @ b
        sst = ((y - y.mean())**2).sum()
        return 1 - (resid @ resid) / sst
    incs = []
    for i in range(R.shape[1]):
        r2_0 = r2(np.column_stack([np.ones(T), mkt]), R[:, i])
        r2_1 = r2(np.column_stack([np.ones(T), mkt, g]), R[:, i])
        incs.append(r2_1 - r2_0)
    return np.mean(incs)

# ---- 场景 A: 候选因子 = 纯噪声(与资产收益正交, 无增量) ----
g_A = rng.normal(0.004, 0.020, T)     # 独立于 R、mkt
# ---- 场景 B: 候选因子 = 对部分资产有真实载荷(真增量) ----
g_B = rng.normal(0.005, 0.025, T)     # 独立新因子
R_B = R.copy()
for i in range(K):
    if i % 2 == 0:                     # 第 1/3/5/7 组合对新因子有 +0.5 真实载荷
        R_B[:, i] += 0.5 * g_B

F_A, p_A, c_A, SSR_rA, SSR_uA = spanning_test(R, mkt, g_A)
F_B, p_B, c_B, SSR_rB, SSR_uB = spanning_test(R_B, mkt, g_B)

Fcrit = stats.f.ppf(0.95, K, T - 3)

log("===== 因子跨越检验 (GRS / 联合 F 检验) =====")
log(f"测试资产 K = {K}, 观测 T = {T}; 受限模型 [1, mkt], 无约束 [1, mkt, g]")
log(f"F 临界值(5%, df=({K},{T-3})) = {Fcrit:.3f}")
log("--- 场景 A: 候选因子=纯噪声(应被跨越, 无增量) ---")
log(f"F = {F_A:.3f}, p = {p_A:.4f}  => {'拒绝H0(有增量)' if p_A<0.05 else '不拒绝H0(无增量)'}")
log(f"各资产对新因子载荷 c_i 绝对值均值 = {np.abs(c_A).mean():.5f}")
log("--- 场景 B: 候选因子含独立增量信息(应拒绝H0, 有增量) ---")
log(f"F = {F_B:.3f}, p = {p_B:.4f}  => {'拒绝H0(有增量)' if p_B<0.05 else '不拒绝H0(无增量)'}")
log(f"各资产对新因子载荷 c_i: 偶数组合(真实载荷0.5)≈{c_B[0::2].mean():.3f}, 奇数组合≈{c_B[1::2].mean():.3f}")
inc_A = r2_inc(R, mkt, g_A)
inc_B = r2_inc(R_B, mkt, g_B)
log(f"平均 R² 增量: 场景 A = {inc_A:.4f} (冗余), 场景 B = {inc_B:.4f} (真增量)")

# ================= 2. 画图 =================
# 图1: 各资产对新因子载荷 c_i
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
for ax, c, title in [(axes[0], c_A, "场景A: 冗余因子\n(c_i ≈ 0)"),
                     (axes[1], c_B, "场景B: 真增量因子\n(c_i 显著≠0)")]:
    ax.bar(range(K), c, color=[C_GREEN if v >= 0 else C_RED for v in c])
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xticks(range(K)); ax.set_xticklabels(port_names, fontsize=8)
    ax.set_ylabel("对新因子的载荷 c_i")
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.suptitle("跨越检验: 各资产对候选新因子的载荷 c_i", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(D, "grs_loadings.png")); plt.close(fig)

# 图2: F 统计量 vs F 临界值
fig, ax = plt.subplots(figsize=(6.6, 4.4))
bars = ax.bar(["场景A\n(冗余因子)", "场景B\n(增量因子)"], [F_A, F_B],
              color=[C_BLUE, C_GREEN])
ax.axhline(Fcrit, color=C_RED, ls="--", lw=1.4, label=f"F 临界值(5%)={Fcrit:.2f}")
for bar, val in zip(bars, [F_A, F_B]):
    ax.text(bar.get_x()+bar.get_width()/2, val+max(F_A, F_B)*0.02, f"{val:.2f}",
            ha="center", fontsize=10)
ax.set_ylabel("GRS 联合 F 统计量")
ax.set_title("跨越检验 F 统计量 vs F 临界值", fontsize=11)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "grs_scenario.png")); plt.close(fig)

# 图3: p 值(对数轴)
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.bar(["场景A\n(冗余因子)", "场景B\n(增量因子)"], [p_A, p_B], color=[C_BLUE, C_GREEN])
ax.axhline(0.05, color=C_RED, ls="--", lw=1.4, label="显著性阈值 0.05")
ax.set_yscale("log")
ax.set_ylabel("p 值 (log)")
ax.set_title("跨越检验 p 值: 越小越说明新因子有增量信息", fontsize=10.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "grs_pvalue.png")); plt.close(fig)

# 图4: R² 增量
fig, ax = plt.subplots(figsize=(6.6, 4.4))
ax.bar(["场景A\n(冗余因子)", "场景B\n(增量因子)"], [inc_A, inc_B],
       color=[C_BLUE, C_GREEN])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("平均 R² 增量")
ax.set_title("加入新因子的解释力增量: 冗余 vs 真增量", fontsize=11)
for i, v in enumerate([inc_A, inc_B]):
    ax.text(i, v + (0.002 if v >= 0 else -0.004), f"{v:.4f}", ha="center",
            va="bottom" if v >= 0 else "top", fontsize=10)
ax.grid(alpha=0.3, axis="y", color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "grs_r2_increment.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(sorted(os.listdir(D))))
