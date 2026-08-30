#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成「Bootstrap IC 置信区间」文章配图 + 核心数值。
数据均为 numpy 合成（固定 seed 可复现）。

设定：跨 N=200 只股票、T=120 个月，构造一个真实 Rank-IC≈0.06 的因子
（因子值 z 对下期收益 y 的真实相关，叠加个股噪声）。用 block bootstrap
（按月整块重抽，保留自相关/截面相关结构）重抽 B=2000 次，每次重算 Rank-IC，
得到 IC 的经验分布，从而给出 95% percentile 置信区间，并算 t-stat / p-value，
判断「这个因子到底有没有显著预测力」。

配图保存到 public/images/bootstrap-ic-confidence-interval/
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["Heiti TC", "PingFang SC", "Songti SC", "STHeiti", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130

SEED = 20260830
rng = np.random.default_rng(SEED + 23)
OUT = "public/images/bootstrap-ic-confidence-interval"
os.makedirs(OUT, exist_ok=True)

N, T = 200, 120
rho_true = 0.06                                        # 真实 Rank-IC
z = rng.standard_normal((T, N))                         # 因子值（截面标准化）
noise = rng.standard_normal((T, N)) * 0.6
# 下期收益：因子方向 + 噪声。把因子值和收益都做截面 rank 后相关即为 Rank-IC
y = rho_true * z + noise

def rank_ic(zz, yy):
    return np.corrcoef(
        zz.argsort(axis=1).argsort(axis=1).ravel().astype(float),
        yy.argsort(axis=1).argsort(axis=1).ravel().astype(float),
    )[0, 1]

ic_obs = rank_ic(z, y)

# ===================== block bootstrap =====================
B = 2000
block = 6                                            # 整月块长，保留自相关
ic_boot = np.empty(B)
for b in range(B):
    starts = rng.integers(0, T - block + 1, size=int(np.ceil(T / block)))
    idx = np.r_[tuple(np.arange(s, s + block) for s in starts)] % T
    idx = idx[:T]
    ic_boot[b] = rank_ic(z[idx], y[idx])

ci_lo, ci_hi = np.percentile(ic_boot, [2.5, 97.5])
se_boot = ic_boot.std(ddof=1)
t_stat = ic_obs / se_boot
# 单样本 t 检验 H0: IC=0
from scipy import stats
p_val = 2 * (1 - stats.t.cdf(abs(t_stat), df=T - 1))

print("=== 核心统计（用于正文）===")
print(f"股票 N={N}, 月份 T={T}, 真实 Rank-IC={rho_true:.3f}")
print(f"观测 Rank-IC = {ic_obs:.4f}")
print(f"block bootstrap (B={B}, block={block}) 经验分布:")
print(f"  均值={ic_boot.mean():.4f}  标准差(SE)={se_boot:.4f}")
print(f"  95% 置信区间 = [{ci_lo:.4f}, {ci_hi:.4f}]")
print(f"  t-stat={t_stat:.2f}  p-value={p_val:.2e}  -> {'显著' if p_val<0.05 else '不显著'}")

# ===================== 图 1：IC 经验分布 + 置信区间 =====================
fig, ax = plt.subplots(figsize=(9.5, 5))
ax.hist(ic_boot, bins=60, color="#1b6ca8", alpha=0.8, density=True, edgecolor="white", linewidth=0.3)
ax.axvline(ic_obs, color="#d1495b", lw=2.2, label=f"观测 IC = {ic_obs:.3f}")
ax.axvline(ci_lo, color="#2a9d8f", ls="--", lw=1.8, label=f"95% CI 下限 = {ci_lo:.3f}")
ax.axvline(ci_hi, color="#2a9d8f", ls="--", lw=1.8, label=f"95% CI 上限 = {ci_hi:.3f}")
ax.axvline(0.0, color="gray", ls=":", lw=1.5, label="IC=0 (无预测力)")
ax.set_xlabel("Rank-IC"); ax.set_ylabel("概率密度")
ax.set_title("block bootstrap 给出的 Rank-IC 经验分布与 95% 置信区间", fontsize=12.5)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ic_bootstrap_dist.png")
plt.close(fig)

# ===================== 图 2：CI 宽度 vs bootstrap 次数 B =====================
Bs = [200, 500, 1000, 1500, 2000, 3000]
half = []
for bb in Bs:
    half.append((np.percentile(ic_boot[:bb], 97.5) - np.percentile(ic_boot[:bb], 2.5)) / 2)
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(Bs, half, marker="o", color="#6a4c93", lw=2)
ax.axhspan(min(half), max(half), color="#edae49", alpha=0.18)
ax.set_xlabel("bootstrap 次数 B"); ax.set_ylabel("置信区间半宽 (95%)")
ax.set_title("CI 半宽随 B 收敛；B≥2000 后基本稳定", fontsize=12.5)
ax.grid(alpha=0.3)
for bb, h in zip(Bs, half):
    ax.text(bb, h + 1.5e-4, f"{h:.4f}", ha="center", fontsize=8.5)
fig.tight_layout()
fig.savefig(f"{OUT}/ci_convergence.png")
plt.close(fig)

# ===================== 图 3：功率曲线示意（IC vs 可检显著性） =====================
ics_grid = np.linspace(-0.02, 0.18, 40)
# 用 bootstrap 的 SE 近似：t 检验在给定 IC 下的 p，反推最小可检 IC
min_ic = stats.t.ppf(0.975, df=T - 1) * se_boot      # 95% 下单侧显著所需 IC
fig, ax = plt.subplots(figsize=(9, 5))
ax.axvline(min_ic, color="#d1495b", ls="--", lw=1.8, label=f"最小可检 |IC|≈{min_ic:.4f}")
ax.axvline(0, color="gray", ls=":", lw=1.2)
ax.plot(ics_grid, 2 * (1 - stats.t.cdf(np.abs(ics_grid) / se_boot, df=T - 1)), color="#1b6ca8", lw=2,
        label="p-value(IC)")
ax.fill_between(ics_grid, 0, 2 * (1 - stats.t.cdf(np.abs(ics_grid) / se_boot, df=T - 1)),
                where=(np.abs(ics_grid) / se_boot) > stats.t.ppf(0.975, df=T - 1),
                color="#2a9d8f", alpha=0.25, label="显著区 (p<0.05)")
ax.set_xlabel("Rank-IC 真实值"); ax.set_ylabel("p-value")
ax.set_title("样本足够长时，一个真实 IC=0.06 的弱因子也能被稳健判显著", fontsize=12)
ax.set_ylim(-0.02, 1.05)
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(f"{OUT}/ic_power.png")
plt.close(fig)

print(f"图片已保存到 {OUT}")
