#!/usr/bin/env python3
"""
为文章「信用利差分解：违约补偿、流动性溢价与风险厌恶各占多少」
(credit-spread-decomposition-default-liquidity) 生成真实配图。

方法：用结构化+流动性+风险厌恶三因子模型合成一组（评级 × 时间）信用利差面板，
把观测利差 s_obs 可加地拆成三段：
    s_obs = s_default + s_liquidity + s_riskaversion
  - s_default    = (1 - R) * PD          （预期信用损失，PD 来自评级基线）
  - s_liquidity  = k_L * illiq_t          （Amihud 式非流动性因子，危机放大）
  - s_riskaversion = k_RA * risk_t        （VIX 式风险厌恶因子，危机放大）
所有数字均为真实计算，非占位图。
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
D = os.path.join(BASE, "credit-spread-decomposition-default-liquidity")
os.makedirs(D, exist_ok=True)

C = {"def": "#4C72B0", "liq": "#DD8452", "ra": "#C44E52",
     "grid": "#DDDDDD", "obs": "#333333"}

# ============================================================
# 1) 评级基线：年度违约概率 PD0 与回收率 R
# ============================================================
RATINGS = ["AAA", "AA", "A", "BBB", "BB", "B"]
PD0 = np.array([0.0002, 0.0006, 0.0020, 0.0040, 0.0180, 0.0600])  # 年违约概率
REC = np.array([0.55, 0.52, 0.48, 0.45, 0.40, 0.35])             # 回收率
# 非违约溢价随评级递减：越低评级越不流动、越被风险厌恶惩罚
LIQ_MULT = np.array([0.30, 0.40, 0.60, 1.00, 1.80, 3.00])
RA_MULT = np.array([0.40, 0.50, 0.70, 1.00, 1.60, 2.50])

# 时间轴：180 个月（15 年），两个危机窗（2008、2020）
N = 180
t = np.arange(N)
def crisis_factor(t, center, width, peak):
    return peak * np.exp(-((t - center) ** 2) / (2 * width ** 2))
illiq_t = 1.0 + crisis_factor(t, 18, 6, 4.0) + crisis_factor(t, 150, 5, 3.5) + 0.15 * np.sin(t / 18.0)
risk_t = 1.0 + crisis_factor(t, 18, 7, 4.5) + crisis_factor(t, 150, 5, 4.0) + 0.2 * (1 + 0.5 * np.sin(t / 22.0))

# 校准系数：让平静期 AAA≈35 / BBB≈120 / B≈640，危机期 BBB≈420
k_L = 40.0   # 每单位 illiq × 评级系数 贡献的 bps
k_RA = 35.0  # 每单位 risk × 评级系数 贡献的 bps

def spread_components(pd, rec, illiq, risk, liq_mult=1.0, ra_mult=1.0):
    illiq = np.asarray(illiq, dtype=float); risk = np.asarray(risk, dtype=float)
    s_def = (1 - rec) * pd * 1e4 * np.ones_like(illiq)
    s_liq = k_L * illiq * liq_mult
    s_ra = k_RA * risk * ra_mult
    return s_def, s_liq, s_ra

# 选 BBB（index 3）做时间序列
i_bbb = 3
s_def_b, s_liq_b, s_ra_b = spread_components(PD0[i_bbb], REC[i_bbb], illiq_t, risk_t,
                                              LIQ_MULT[i_bbb], RA_MULT[i_bbb])
s_obs_b = s_def_b + s_liq_b + s_ra_b

# ============================================================
# 图1：BBB 利差三因子堆叠面积（时间序列）
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.0))
ax.stackplot(t, s_def_b, s_liq_b, s_ra_b,
             labels=["违约补偿 (1-R)·PD", "流动性溢价 k_L·illiq", "风险厌恶溢价 k_RA·risk"],
             colors=[C["def"], C["liq"], C["ra"]], alpha=0.85)
ax.plot(t, s_obs_b, color=C["obs"], lw=1.6, label="观测利差 s_obs")
ax.set_title("BBB 信用利差的三因子分解（15 年月度，bps）", fontsize=15, fontweight="bold")
ax.set_xlabel("月份")
ax.set_ylabel("信用利差 (bps)")
ax.axvspan(8, 28, color="red", alpha=0.06)
ax.axvspan(140, 160, color="red", alpha=0.06)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "csd_timeseries.png"), dpi=130)
plt.close(fig)
print("图1 完成。BBB 平静期利差≈%.0fbps 危机峰值≈%.0fbps" % (s_obs_b[60], s_obs_b[18]))

# ============================================================
# 图2：各评级在平静期(中段)的分解横向对比（堆叠柱）
# ============================================================
mid = 90
s_def_r, s_liq_r, s_ra_r = [], [], []
for k in range(len(RATINGS)):
    d, l, ra = spread_components(PD0[k], REC[k], illiq_t[mid], risk_t[mid], LIQ_MULT[k], RA_MULT[k])
    s_def_r.append(d); s_liq_r.append(l); s_ra_r.append(ra)
s_def_r, s_liq_r, s_ra_r = map(np.array, (s_def_r, s_liq_r, s_ra_r))
s_obs_r = s_def_r + s_liq_r + s_ra_r

fig, ax = plt.subplots(figsize=(11, 6.0))
x = np.arange(len(RATINGS))
b1 = ax.bar(x, s_def_r, color=C["def"], label="违约补偿")
b2 = ax.bar(x, s_liq_r, bottom=s_def_r, color=C["liq"], label="流动性溢价")
b3 = ax.bar(x, s_ra_r, bottom=s_def_r + s_liq_r, color=C["ra"], label="风险厌恶溢价")
for xi, tot in zip(x, s_obs_r):
    ax.annotate(f"{tot:.0f}", (xi, tot), ha="center", va="bottom", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(RATINGS)
ax.set_title("平静期各评级利差分解（第 90 月，bps）", fontsize=15, fontweight="bold")
ax.set_ylabel("信用利差 (bps)")
ax.legend(fontsize=9)
ax.grid(axis="y", color=C["grid"], alpha=0.6)
fig.tight_layout()
fig.savefig(os.path.join(D, "csd_by_rating.png"), dpi=130)
plt.close(fig)
print("图2 完成。评级分解：", {r: round(float(s_obs_r[k]), 1) for k, r in enumerate(RATINGS)})

# ============================================================
# 图3：隐含 PD vs 真实 PD（忽略非违约成分会高估多少）
#  隐含 PD：把整段利差都当成违约补偿反推  ->  pd_impl = s_obs/((1-R)*1e4)
#  真实 PD：评级基线 PD0
# ============================================================
fig, ax = plt.subplots(figsize=(11, 6.0))
for k, r in enumerate(RATINGS):
    pd_impl = s_obs_r[k] / ((1 - REC[k]) * 1e4)
    ax.scatter(PD0[k] * 1e4, pd_impl * 1e4, s=120, color=C["ra"], zorder=3, label=(r if k == 0 else None))
    ax.plot([PD0[k] * 1e4, pd_impl * 1e4], [PD0[k] * 1e4, PD0[k] * 1e4],
            color=C["liq"], lw=1, alpha=0.6)
ax.plot([0, 700], [0, 700], "--", color="gray", label="隐含=真实(45°)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("真实违约概率 PD（bps，评级基线）")
ax.set_ylabel("隐含违约概率 PD_impl（bps，利差全当违约补偿）")
ax.set_title("把利差全当违约补偿会高估违约概率多少（平静期）", fontsize=14, fontweight="bold")
ax.annotate("虚线=真实值；红点=隐含值；\n水平线长度=非违约成分(流动性+风险厌恶)造成的偏差",
            xy=(0.05, 0.9), xycoords="axes fraction", fontsize=9, color="#555")
ax.legend()
ax.grid(True, which="both", color=C["grid"], alpha=0.5)
fig.tight_layout()
fig.savefig(os.path.join(D, "csd_implied_pd.png"), dpi=130)
plt.close(fig)
# 计算高估倍数（BBB）
pd_impl_bbb = s_obs_r[i_bbb] / ((1 - REC[i_bbb]) * 1e4)
print("图3 完成。BBB 隐含PD/真实PD = %.1fx" % (pd_impl_bbb / PD0[i_bbb]))

# ============================================================
# 图4：非违约成分占比的热力图（评级 × 时间 regime）
# ============================================================
regimes = ["平静期", "2008 危机", "2020 危机"]
cols_idx = [90, 18, 150]
share_nondefault = np.zeros((len(RATINGS), len(regimes)))
for j, ci in enumerate(cols_idx):
    for k in range(len(RATINGS)):
        d, l, ra = spread_components(PD0[k], REC[k], illiq_t[ci], risk_t[ci], LIQ_MULT[k], RA_MULT[k])
        tot = d + l + ra
        share_nondefault[k, j] = (l + ra) / tot * 100

fig, ax = plt.subplots(figsize=(9, 5.6))
im = ax.imshow(share_nondefault, aspect="auto", cmap="OrRd", vmin=0, vmax=100)
ax.set_xticks(range(len(regimes))); ax.set_xticklabels(regimes)
ax.set_yticks(range(len(RATINGS))); ax.set_yticklabels(RATINGS)
for ki in range(len(RATINGS)):
    for ji in range(len(regimes)):
        ax.text(ji, ki, f"{share_nondefault[ki, ji]:.0f}%", ha="center", va="center",
                color="black", fontsize=10, fontweight="bold")
ax.set_title("非违约成分（流动性+风险厌恶）占利差比重 (%)", fontsize=14, fontweight="bold")
ax.set_ylabel("评级")
fig.colorbar(im, ax=ax, label="占比 %")
fig.tight_layout()
fig.savefig(os.path.join(D, "csd_share_heatmap.png"), dpi=130)
plt.close(fig)
print("图4 完成。非违约占比矩阵(行=评级):")
print(np.round(share_nondefault, 1))

print("\ncredit-spread-decomposition 配图已生成：", sorted(os.listdir(D)))
