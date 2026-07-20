#!/usr/bin/env python3
"""
为文章「杜邦分析 ROE 拆解因子」(dupont-roe-decomposition) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月，每只股票有持续性质量 q_i（驱动净利率、
    资产周转率）与杠杆倾向 lev_i（驱动权益乘数 EM）。
  - DuPont 三因子分解：ROE = 净利率 × 资产周转率 × 权益乘数 = 经营 ROA × EM，
    其中经营 ROA = 净利率 × 资产周转率 是「剔除杠杆后的真实盈利能力」。
  - 未来 1 月收益：随经营 ROA 正相关、随权益乘数（杠杆脆弱性）负相关——
    高 ROE 若由高杠杆堆出来，是脆弱的，长期反而跑输。
  - 因子信号 = ROA_z − EM_z：奖励「经营驱动的高 ROE」、惩罚「杠杆驱动的高 ROE」。
  - 验证：ROE 的对数加性拆解、ROE 与权益乘数的散点（杠杆放大但脆弱）、
    「清洁 ROE 因子」净值、按信号分的未来收益单调性。
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
D = os.path.join(BASE, "dupont-roe-decomposition")
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
# 1) 合成面板 + DuPont 三因子
# ============================================================
def simulate_panel(N=600, M=144, seed=20260720):
    rng = np.random.default_rng(seed)
    q = rng.normal(0.0, 1.0, size=N)                 # 质量（驱动经营 ROA）
    lev = rng.normal(0.0, 1.0, size=N)               # 杠杆倾向（驱动 EM）
    drift = rng.normal(0, 0.12, size=(N, M)).cumsum(axis=1) * 0.04

    def sig(beta, noise=0.5):
        return beta * q[:, None] + 0.3 * drift + rng.normal(0, noise, size=(N, M))

    # 净利率（net margin）：受质量驱动，温和正值
    margin = np.clip(0.08 + 0.05 * sig(1.0) + rng.normal(0, 0.02, size=(N, M)), 0.005, 0.4)
    # 资产周转率（asset turnover）：受质量驱动
    turnover = np.clip(0.80 + 0.30 * sig(0.8) + rng.normal(0, 0.08, size=(N, M)), 0.2, 2.5)
    # 权益乘数（equity multiplier = 资产/权益）：受杠杆倾向驱动
    EM = np.clip(np.exp(0.45 * lev[:, None] + rng.normal(0, 0.10, size=(N, M))), 1.0, 6.0)

    ROA = margin * turnover                            # 经营 ROA（剔除杠杆）
    ROE = ROA * EM                                     # 杜邦 ROE

    mkt = rng.normal(0.005, 0.04, size=M)
    # 信号：奖励经营 ROA、惩罚杠杆
    ROA_z = (ROA - ROA.mean(axis=0, keepdims=True)) / (ROA.std(axis=0, keepdims=True) + 1e-6)
    EM_z = (EM - EM.mean(axis=0, keepdims=True)) / (EM.std(axis=0, keepdims=True) + 1e-6)
    signal = ROA_z - EM_z
    # 未来收益：随经营 ROA 正相关、随 EM 负相关
    future = (0.004 + 0.0040 * ROA_z - 0.0022 * EM_z + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    return dict(margin=margin, turnover=turnover, EM=EM, ROA=ROA, ROE=ROE,
                signal=signal, future=future, mkt=mkt)


PAN = simulate_panel()
margin = PAN["margin"]
turnover = PAN["turnover"]
EM = PAN["EM"]
ROA = PAN["ROA"]
ROE = PAN["ROE"]
signal = PAN["signal"]
future = PAN["future"]
mkt = PAN["mkt"]
M = ROE.shape[1]


# ============================================================
# 图一：杜邦对数加性拆解（ln ROE = ln margin + ln turnover + ln EM）
# ============================================================
lm = np.log(margin.mean(axis=(0, 1)) + 1e-9)
lt = np.log(turnover.mean(axis=(0, 1)) + 1e-9)
le = np.log(EM.mean(axis=(0, 1)) + 1e-9)
total = lm + lt + le

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh([0], [lm], color=C["pos"], label=f"净利率 ln(margin)={lm:.2f}")
ax.barh([0], [lt], left=[lm], color=C["mk"], label=f"资产周转率 ln(turnover)={lt:.2f}")
ax.barh([0], [le], left=[lm + lt], color=C["neg"], label=f"权益乘数 ln(EM)={le:.2f}")
ax.set_xlim(lm + lt + le + 0.3, lm - 0.3)
ax.set_yticks([])
ax.set_xlabel("ln(ROE) 的加性拆解（杜邦三因子）")
ax.set_title(f"平均 ROE = {np.exp(total):.3f}，由三部分对数相加构成")
ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
fig.tight_layout()
fig.savefig(os.path.join(D, "dupont_decomp_log.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：ROE vs 权益乘数 散点（杠杆放大 ROE，但越靠右越脆弱）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
# 用截面某月做散点，颜色按净利率深浅
t0 = M // 2
sc = ax.scatter(EM[:, t0], ROE[:, t0], c=margin[:, t0], cmap="viridis",
                s=22, alpha=0.6, edgecolors="none")
ax.set_xlabel("权益乘数 EM（资产/权益，越高=杠杆越高）")
ax.set_ylabel("ROE（净资产收益率）")
ax.set_title("ROE 与权益乘数：高杠杆放大 ROE，但密集在右下端（脆弱区）")
cb = fig.colorbar(sc, ax=ax)
cb.set_label("净利率（越深=盈利能力越强）")
ax.grid(True, color=C["grid"])
fig.tight_layout()
fig.savefig(os.path.join(D, "dupont_roe_vs_em.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：清洁 ROE 因子净值（多经营驱动 / 空杠杆驱动）
# ============================================================
def ls_curve(sig, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(sig[:, t])
        ret[t] = future[order[-n:], t].mean() - future[order[:n], t].mean()
    return np.cumprod(1 + ret)


nav_ls = ls_curve(signal)
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, nav_ls, color=C["ls"], lw=2.2, label="清洁 ROE 因子（多经营驱动/空杠杆驱动）")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("杜邦清洁 ROE 因子：长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "dupont_ls_curve.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：按信号十分位的未来收益（单调）
# ============================================================
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(signal[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M

fig, ax = plt.subplots(figsize=(8, 5))
colors = [C["neg"] if v < 0 else C["pos"] for v in dec_avg]
ax.bar(range(10), dec_avg * 100, color=colors)
ax.set_xlabel("清洁 ROE 信号十分位（D1 杠杆驱动 → D10 经营驱动）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("清洁 ROE 信号十分组：单调递增，经营驱动组显著跑赢")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "dupont_decile_returns.png"), dpi=130)
plt.close(fig)


print("dupont-roe-decomposition images saved:", sorted(os.listdir(D)))
print(f"avg ROE={ROE.mean():.4f} avg ROA={ROA.mean():.4f} avg EM={EM.mean():.3f}")
print(f"decile future ret % = {np.round(dec_avg*100,3)}")
print(f"ls final = {nav_ls[-1]:.3f}")
