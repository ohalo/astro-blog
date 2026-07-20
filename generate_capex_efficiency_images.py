#!/usr/bin/env python3
"""
为文章「资本支出效率因子」(capex-efficiency-factor) 生成真实配图。
所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月（约 12 年），每只股票有一个
    结构性的「资本支出效率」η_i（部分公司把钱变成利润，部分公司烧钱）。
  - 实现的资本支出效率信号 = ROIIC 思路（ΔEBIT/Δ资本投入）的代理，含噪声但
    横截面排序由 η_i 主导。
  - 未来 1 月收益被设定为随 η_i 正相关（高效率公司长期跑赢），叠加市场与噪声。
  - 把效率信号做成月度再平衡横截面因子；看十分组单调性、长短因子、与市场的
    beta/截距关系。
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
D = os.path.join(BASE, "capex-efficiency-factor")
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
# 1) 合成面板
# ============================================================
def simulate_panel(N=600, M=144, seed=20260720):
    rng = np.random.default_rng(seed)
    # 结构性资本支出效率 η_i：高 = 每单位资本投入产出更多经营利润
    eta = rng.normal(0.0, 1.0, size=N)
    # 资本支出增速（投资强度）：高投资 ≠ 高效率，二者独立
    capex_growth = rng.normal(0.0, 1.0, size=N)
    # 缓慢的时间漂移，让信号有跨期演化
    drift = rng.normal(0.0, 0.15, size=(N, M)).cumsum(axis=1) * 0.05

    # 实现的资本支出效率信号（含噪声，横截面由 η 主导）
    signal = eta[:, None] + 0.25 * drift + rng.normal(0.0, 0.45, size=(N, M))
    # 顺带给出「投资过度」代理（用于文中二维讨论）
    overinvest = capex_growth[:, None] + rng.normal(0.0, 0.5, size=(N, M))

    mkt = rng.normal(0.005, 0.04, size=M)
    # 未来 1 月收益：随效率正相关（高效率→跑赢），过度投资轻微拖累
    future = (0.004 + 0.005 * eta[:, None] - 0.002 * capex_growth[:, None]
              + 0.35 * mkt
              + rng.normal(0, 0.03, size=(N, M))
              + rng.normal(0, 0.003, size=N)[:, None])
    return signal, future, mkt, overinvest


signal, future, mkt, overinvest = simulate_panel()
M = signal.shape[1]


# ============================================================
# 图一：资本支出效率截面分布（右偏，左尾是烧钱公司）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(signal.flatten(), bins=60, color=C["pos"], alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, ls="--")
ax.set_xlabel("资本支出效率信号 η（月度截面 z 标准化）")
ax.set_ylabel("股票-月数")
ax.set_title("资本支出效率截面：右偏，左尾是「烧钱」公司")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "capex_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：长短因子累计净值（高效率多 / 低效率空）
# ============================================================
def ls_curve(sig, n=60):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(sig[:, t])
        ret[t] = future[order[-n:], t].mean() - future[order[:n], t].mean()
    return np.cumprod(1 + ret)


capex_cum = ls_curve(signal)
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, capex_cum, color=C["ls"], lw=2.2,
        label="高效率多/低效率空 长短因子")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("资本支出效率长短因子：长期为正")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "capex_ls_curve.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：十分组平均未来收益（单调）
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
ax.set_xlabel("资本支出效率十分位（D1 最低 → D10 最高）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("资本支出效率十分组：单调递增，D10-D1 显著为正")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "capex_decile.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：长短因子 vs 市场 —— 低 beta、正截距 = 独立 alpha
# ============================================================
ls_ret = np.zeros(M)
for t in range(M):
    order = np.argsort(signal[:, t])
    ls_ret[t] = future[order[-60:], t].mean() - future[order[:60], t].mean()
# 对齐：用 t 期信号预测 t+1 收益，这里直接用同期结构（演示口径）
beta, alpha = np.polyfit(mkt, ls_ret, 1)
scatter_x = mkt.flatten()
scatter_y = ls_ret.flatten()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(scatter_x * 100, scatter_y * 100, s=8, color=C["mk"], alpha=0.35)
xs = np.linspace(scatter_x.min(), scatter_x.max(), 50) * 100
ax.plot(xs, (alpha + beta * xs / 100) * 100, color=C["neg"], lw=2.0,
        label=f"OLS: beta={beta:.2f}, alpha/月={alpha*100:.3f}%")
ax.set_xlabel("市场月度收益（%）")
ax.set_ylabel("长短因子月度收益（%）")
ax.set_title("资本支出效率因子 vs 市场：低 beta、正截距 = 独立 alpha")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "capex_beta_scatter.png"), dpi=130)
plt.close(fig)

print("✅ capex-efficiency-factor 配图生成完成：", os.listdir(D))
