#!/usr/bin/env python3
"""
为文章「已实现波动率双幂变差:用 Bipower 检验把连续波动与跳跃剥开」
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由高频几何布朗运动 + 泊松跳跃自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 日内 5 分钟对数收益: dr_t = σ·dW_t + Σ_k J_k·1_{t=τ_k}  (泊松跳跃, 跳跃大小~N(0,σj²))
  - 已实现方差 RV = Σ dr_t²              (捕获 连续波动 + 跳跃, 对跳跃敏感)
  - 双幂变差 BV = μ₁⁻²·Σ|dr_t|·|dr_{t-1}|, μ₁=√(2/π)  (对跳跃稳健, 只估连续积分方差)
  - 跳跃变差 JV = max(RV − BV, 0)
  - 跳跃检验: 比值 RV/BV > 1 显著 ⇒ 存在跳跃 (adj. test statistic 另算)
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
D = os.path.join(BASE, "realized-bipower-variation")
os.makedirs(D, exist_ok=True)

C = {"grid": "#DDDDDD", "blue": "#2F4B7C", "red": "#C44E52", "green": "#55A868",
     "purple": "#8172B3", "orange": "#FF7F0E"}
MU1 = np.sqrt(2 / np.pi)     # = 0.7979

rng = np.random.default_rng(20260719)

# 日内 5 分钟收益: 78 根(6.5h × 12)
N_INTRA = 78
SIGMA_DAY = 0.013            # 日连续波动 ~1.3%
SIG_INT = SIGMA_DAY / np.sqrt(N_INTRA)
SIG_JUMP = 0.013             # 单跳大小 ~1.3%
LAMBDA = 0.45                # 每天跳跃强度(平均 0.45 跳/日)

def sim_day(jump=True, seed=None):
    r = rng if seed is None else np.random.default_rng(seed)
    cont = SIG_INT * r.normal(size=N_INTRA)
    if jump and r.uniform() < LAMBDA:
        k = r.integers(5, N_INTRA - 1)         # 盘中某处跳
        cont[k] += r.normal(0, SIG_JUMP)
        return cont, k
    return cont, None

def rv(x):
    return np.sum(x ** 2)

def bv(x):
    return (1.0 / MU1 ** 2) * np.sum(np.abs(x[1:]) * np.abs(x[:-1]))

# ---- 单次演示日(含跳) 用于图1 ----
day_j, jump_k = sim_day(jump=True, seed=7)
RV_d, BV_d = rv(day_j), bv(day_j)
logp = np.cumsum(day_j)
print(f"演示日(含跳): RV={RV_d:.5f}  BV={BV_d:.5f}  RV/BV={RV_d/BV_d:.3f}  跳跃在 bar#{jump_k}")
print(f"连续日波动理论 ≈ IV={SIGMA_DAY**2:.5f}, BV 估计={BV_d:.5f}")

# ---- 蒙特卡洛:200 天,跳跃/无跳跃混搭 ----
M = 200
RVs, BVs, ratios, has_jump = [], [], [], []
for i in range(M):
    x, k = sim_day(jump=True, seed=100 + i)
    r_ = rv(x); b_ = bv(x)
    RVs.append(r_); BVs.append(b_); ratios.append(r_ / b_)
    has_jump.append(k is not None)
RVs = np.array(RVs); BVs = np.array(BVs); ratios = np.array(ratios)
has_jump = np.array(has_jump)
print(f"\n蒙特卡洛 {M} 天:")
print(f"  含跳天 ({has_jump.sum()}): RV/BV 均值 = {ratios[has_jump].mean():.3f}")
print(f"  无跳天 ({ (~has_jump).sum()}): RV/BV 均值 = {ratios[~has_jump].mean():.3f}")
print(f"  BV 全部天数均值 = {BVs.mean():.5f} (应≈IV={SIGMA_DAY**2:.5f})")
print(f"  RV 全部天数均值 = {RVs.mean():.5f} (应>IV, 因含跳)")
# 跳跃检验命中率: 阈值 RV/BV > 1.10
TP = np.sum(ratios[has_jump] > 1.10)
FN = np.sum(ratios[has_jump] <= 1.10)
FP = np.sum(ratios[~has_jump] > 1.10)
TN = np.sum(ratios[~has_jump] <= 1.10)
print(f"  跳跃检验(阈值1.10): 命中 TP={TP}/{TP+FN}  误报 FP={FP}/{FP+TN}")

# ============================================================
# 图1:含跳日的日内路径 + 收益跳变高亮
# ============================================================
def fig_intraday():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.4), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    t = np.arange(N_INTRA)
    ax1.plot(t, logp, color=C["blue"], lw=1.4)
    if jump_k is not None:
        ax1.scatter([jump_k], [logp[jump_k]], color=C["red"], s=70, zorder=5,
                    label=f"跳跃 (bar #{jump_k})")
    ax1.axhline(0, color="gray", lw=0.6)
    ax1.set_ylabel("累计对数价格", fontsize=11)
    ax1.set_title("含跳跃的日内路径:跳跃是连续扩散『之外』的瞬时位移", fontsize=12)
    ax1.legend(loc="lower right", fontsize=9)
    ax2.bar(t, day_j, color=C["purple"], width=0.7)
    if jump_k is not None:
        ax2.bar([jump_k], [day_j[jump_k]], color=C["red"], width=0.7)
    ax2.set_ylabel("5分钟收益", fontsize=11)
    ax2.set_xlabel("日内 5 分钟 bar 序号", fontsize=11)
    ax2.grid(True, color=C["grid"], lw=0.5, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(D, "intraday_jump_path.png"), dpi=130)
    plt.close(fig)

# ============================================================
# 图2:RV vs BV 散点(含跳红/无跳蓝) + RV/BV 比值直方图
# ============================================================
def fig_scatter_hist():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.6))
    ax1.scatter(BVs[~has_jump], RVs[~has_jump], s=18, c=C["blue"], alpha=0.6,
                label="无跳跃日")
    ax1.scatter(BVs[has_jump], RVs[has_jump], s=18, c=C["red"], alpha=0.6,
                label="含跳跃日")
    lim = [BVs.min() * 0.9, RVs.max() * 1.05]
    ax1.plot(lim, lim, color="black", ls="--", lw=1, label="RV=BV (无跳基准线)")
    ax1.set_xlim(lim); ax1.set_ylim(lim)
    ax1.set_xlabel("双幂变差 BV (连续波动估计)", fontsize=11)
    ax1.set_ylabel("已实现方差 RV (含跳)", fontsize=11)
    ax1.set_title("RV-BV 散点:含跳日落在身份线上方", fontsize=12)
    ax1.legend(loc="upper left", fontsize=9)

    ax2.hist(ratios[~has_jump], bins=25, alpha=0.6, color=C["blue"],
             label="无跳跃日", density=True)
    ax2.hist(ratios[has_jump], bins=25, alpha=0.6, color=C["red"],
             label="含跳跃日", density=True)
    ax2.axvline(1.0, color="black", ls="--", lw=1, label="RV/BV = 1")
    ax2.axvline(1.10, color=C["orange"], ls=":", lw=1.5, label="检验阈值 1.10")
    ax2.set_xlabel("RV / BV 比值", fontsize=11)
    ax2.set_ylabel("密度", fontsize=11)
    ax2.set_title("比值分布:含跳日系统性右移", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "rv_bv_scatter_hist.png"), dpi=130)
    plt.close(fig)

# ============================================================
# 图3:RV/BV 比值时序(跳跃检测)+ 连续/跳跃分解
# ============================================================
def fig_ratio_ts():
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 5.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.4]})
    days = np.arange(M)
    ax1.plot(days, ratios, color=C["purple"], lw=1.0)
    ax1.axhline(1.0, color="black", ls="--", lw=1)
    ax1.axhline(1.10, color=C["orange"], ls=":", lw=1.5, label="检验阈值 1.10")
    # 标出被判定为含跳的天
    flagged = days[ratios > 1.10]
    ax1.scatter(flagged, ratios[ratios > 1.10], color=C["red"], s=18, zorder=3)
    ax1.set_ylabel("RV / BV", fontsize=11)
    ax1.set_title("RV/BV 比值时序:越过阈值即判为跳跃日", fontsize=12)
    ax1.legend(loc="upper right", fontsize=9)

    # 连续波动(BV) vs 跳跃变差(JV=max(RV-BV,0)) 堆叠
    JV = np.maximum(RVs - BVs, 0)
    ax2.bar(days, BVs, color=C["blue"], width=0.9, label="连续波动 (BV)")
    ax2.bar(days, JV, bottom=BVs, color=C["red"], width=0.9, label="跳跃变差 (JV)")
    ax2.set_ylabel("方差贡献", fontsize=11)
    ax2.set_xlabel("模拟交易日序号", fontsize=11)
    ax2.set_title("波动分解:RV = 连续(BV) + 跳跃(JV)", fontsize=12)
    ax2.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(D, "ratio_ts_decomposition.png"), dpi=130)
    plt.close(fig)

fig_intraday()
fig_scatter_hist()
fig_ratio_ts()
print("\n✅ 已生成 3 张配图:", os.listdir(D))
