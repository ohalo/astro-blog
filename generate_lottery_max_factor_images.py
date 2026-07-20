#!/usr/bin/env python3
"""
为文章「博彩偏好与最大日收益因子 MAX：把彩票性做成可交易的横截面因子」
(lottery-max-factor) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造一个跨时间面板：N=500 只股票 × M=120 个月（约 10 年），每月每只股票
    有过去 21 日日收益，计算 MAX = 过去 21 日最大单日收益（彩票性强度）。
  - 未来 1 个月收益被设定为随 MAX 单调下降（博彩溢价 → 提前透支 → 未来走弱）
    叠加市场因子与特质噪声。
  - 把 MAX 做成一个「月度再平衡」的横截面因子：多低 MAX 十分位 / 空高 MAX 十分位，
    累计成因子净值；逐月算 rank-IC 看稳定性；再做 MAX × 动量 二维分组看冗余度。
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
D = os.path.join(BASE, "lottery-max-factor")
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
# 1) 合成面板：N 只股票 × M 个月，每月带过去 21 日日收益算 MAX
# ============================================================
def simulate_panel(N=500, M=120, T=21, seed=20260720):
    rng = np.random.default_rng(seed)
    # 每只股票的「彩票倾向」基线（截面异质）
    base_p = rng.uniform(0.02, 0.28, size=N)
    # 每月的市场状态：温和牛/熊切换
    mkt = rng.normal(0.004, 0.035, size=M)          # 月度市场收益
    # 月度特质漂移（轻微正）
    drift = rng.normal(0.003, 0.012, size=(N, M))

    MAX = np.zeros((N, M))
    future = np.zeros((N, M))
    mom = np.zeros((N, M))                            # 过去 12 月累计动量（用于二维分组）
    for t in range(M):
        # 彩票跳涨概率：熊市里散户更赌、跳涨略多
        p = np.clip(base_p * (1.0 + 0.4 * (mkt[t] < 0)), 0.01, 0.5)
        jumps = (rng.random((N, T)) < p[:, None]) * rng.exponential(0.05, size=(N, T))
        daily = rng.normal(0, 0.022, size=(N, T)) + jumps
        MAX[:, t] = daily.max(axis=1)
        # 未来 1 月收益：随 MAX 单调下降（核心异象），叠加市场与漂移
        future[:, t] = (0.010 - 0.16 * MAX[:, t]
                        + 0.35 * mkt[t]
                        + drift[:, t]
                        + rng.normal(0, 0.05, size=N))
        # 动量：用过去若干月未来收益近似（自洽：动量 = 之前累积强势）
        if t >= 12:
            mom[:, t] = future[:, t - 12:t].mean(axis=1)
        else:
            mom[:, t] = rng.normal(0, 0.01, size=N)
    return MAX, future, mom, mkt


def quintile_labels(x):
    """返回等长五分位标签 0..4（0=最低）。用 argsort 分箱，不依赖 pandas。"""
    order = np.argsort(np.argsort(x))
    return (order * 5 // len(x)).clip(0, 4)


MAX, future, mom, mkt = simulate_panel()

# ============================================================
# 图一：面板散点 MAX vs 未来 1 月收益（负斜率）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
flat_max = MAX.flatten()
flat_fut = future.flatten()
# 抽样避免过密
idx = np.random.default_rng(7).choice(len(flat_max), 6000, replace=False)
ax.scatter(flat_max[idx] * 100, flat_fut[idx] * 100, s=6,
           c=C["mk"], alpha=0.35, edgecolors="none")
# 拟合线
b1, b0 = np.polyfit(flat_max * 100, flat_fut * 100, 1)
xs = np.linspace(flat_max.min() * 100, flat_max.max() * 100, 50)
ax.plot(xs, b0 + b1 * xs, color=C["neg"], lw=2.5,
        label=f"拟合斜率 ≈ {b1:.3f}%/单位")
ax.set_xlabel("MAX（过去 21 日最大单日收益, %）")
ax.set_ylabel("未来 1 个月收益（%）")
ax.set_title("面板里 MAX 越高，未来 1 月收益越低（负向）")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "max_scatter.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图二：月度再平衡的长短因子累计净值
# ============================================================
M = MAX.shape[1]
ls_ret = np.zeros(M)
for t in range(M):
    order = np.argsort(MAX[:, t])
    long_ret = future[order[:50], t].mean()      # 低 MAX 十分位 → 多头
    short_ret = future[order[-50:], t].mean()    # 高 MAX 十分位 → 空头
    ls_ret[t] = long_ret - short_ret
ls_cum = np.cumprod(1 + ls_ret)
# 等权多空（仅多头腿，做参照）
long_cum = np.cumprod(1 + np.array([future[np.argsort(MAX[:, t])[:50], t].mean() for t in range(M)]))
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, ls_cum, color=C["ls"], lw=2.2, label="低MAX多/高MAX空 多空因子")
ax.plot(months, long_cum, color=C["blue"], lw=1.6, ls="--", label="仅低MAX多头")
ax.set_xlabel("月份（约 10 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("MAX 月度再平衡长短因子：长期为正但波动大")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "max_ls_curve.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图三：逐月 rank-IC 时间序列（稳定性检验）
# ============================================================
ic = np.array([np.corrcoef(MAX[:, t].argsort().argsort(),
                            future[:, t].argsort().argsort())[0, 1]
               for t in range(M)])
rolling = np.convolve(ic, np.ones(12) / 12, mode="same")

fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(months, ic, color=np.where(ic >= 0, C["pos"], C["neg"]), width=0.8, alpha=0.7)
ax.plot(months, rolling, color=C["gold"], lw=2.2, label="12 月滚动均值")
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("月份")
ax.set_ylabel("月度 rank-IC")
ax.set_title(f"MAX 的 rank-IC：均值 {ic.mean():.3f}、IR {ic.mean()/ic.std():.2f}，但频繁转负")
ax.grid(True, color=C["grid"], axis="y")
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "max_ic_ts.png"), dpi=130)
plt.close(fig)

# ============================================================
# 图四：MAX × 动量 二维分组（看冗余度）
# ============================================================
Q = 5
avg = np.zeros((Q, Q))
for t in range(M):
    qi = quintile_labels(MAX[:, t])
    qj = quintile_labels(mom[:, t])
    for i in range(Q):
        for j in range(Q):
            m = (qi == i) & (qj == j)
            if m.sum() > 0:
                avg[i, j] += future[:, t][m].mean()
avg /= M

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(avg * 100, cmap="RdYlGn", aspect="auto")
ax.set_xticks(range(Q)); ax.set_xticklabels([f"动量Q{q+1}" for q in range(Q)])
ax.set_yticks(range(Q)); ax.set_yticklabels([f"MAX Q{q+1}" for q in range(Q)])
for i in range(Q):
    for j in range(Q):
        ax.text(j, i, f"{avg[i, j]*100:.1f}", ha="center", va="center", fontsize=9,
                color="black")
ax.set_xlabel("横截面动量分组（Q1 弱 → Q5 强）")
ax.set_ylabel("MAX 分组（Q1 低彩票性 → Q5 高彩票性）")
ax.set_title("二维分组平均未来收益(%)：MAX 效应独立于动量")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="未来1月收益 %")
fig.tight_layout()
fig.savefig(os.path.join(D, "max_2d_sort.png"), dpi=130)
plt.close(fig)

print("lottery-max-factor images saved:", os.listdir(D))
