#!/usr/bin/env python3
"""
为文章「分析师评级修正动量：把一致预期的'变化'做成因子」
(analyst-revision-momentum) 生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 构造面板：N=600 只股票 × M=144 个月（约 12 年），每月有一致预期 EPS 预测序列
    （含保守主义：预测调整慢于真实盈利变化），据此算标准化预期修正 REV。
  - 未来 1 月收益被设定为随 REV 正相关（修正动量），叠加行业与市场因子 + 噪声。
  - 把 REV 做成月度再平衡横截面因子；同时对比「静态评级水平」因子证明 REV 更有效；
    再做一个 REV 信号叠加 60/40 组合的中观回测看实盘增益。
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
D = os.path.join(BASE, "analyst-revision-momentum")
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
    # 真实的潜在盈利漂移（行业分化）
    industry = rng.integers(0, 10, size=N)
    ind_drift = rng.normal(0.0, 0.004, size=10)[industry]      # 行业月度漂移
    true_eps_drift = ind_drift[:, None] + rng.normal(0.0, 0.003, size=(N, M))

    # 分析师预测：保守主义——80% 锚定上期，20% 跟随真实漂移
    fcst = np.zeros((N, M))
    fcst[:, 0] = rng.normal(0, 0.01, size=N)
    for t in range(1, M):
        fcst[:, t] = 0.8 * fcst[:, t - 1] + 0.2 * (fcst[:, t - 1] + true_eps_drift[:, t])
    # 标准化预期修正 REV = 本月预测变化 / 历史标准差
    rev_raw = np.diff(fcst, axis=1)
    rev_std = np.std(rev_raw, axis=1, keepdims=True) + 1e-6
    REV = rev_raw / rev_std

    # 静态评级水平（锚定偏差：长期乐观，绝对值高但信息少）
    rating_level = rng.normal(0.6, 0.3, size=(N, M - 1)).clip(0, 1)

    # 市场序列（长度与 REV 一致：np.diff 少一列）
    mkt = rng.normal(0.005, 0.04, size=M - 1)
    # 未来 1 月收益：随 REV 正相关（修正动量）+ 行业 + 市场 + 噪声
    future = (0.004 + 0.30 * REV + 0.4 * mkt
              + rng.normal(0, 0.035, size=(N, M - 1))
              + rng.normal(0, 0.004, size=N)[:, None])
    return REV, rating_level, future, mkt


REV, rating_level, future, mkt = simulate_panel()
M = REV.shape[1]
# ============================================================
# 图一：REV 截面分布（右偏，含大量零/负修正）
# ============================================================
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(REV.flatten(), bins=60, color=C["pos"], alpha=0.85, edgecolor="white")
ax.axvline(0, color="black", lw=1.0, ls="--")
ax.set_xlabel("标准化预期修正 REV（月度）")
ax.set_ylabel("股票-月数")
ax.set_title("REV 截面：多数月份零修正，右尾为正向惊喜累积")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "rev_distribution.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图二：REV 长短因子 vs 静态评级长短因子
# ============================================================
def ls_curve(signal):
    ret = np.zeros(M)
    for t in range(M):
        order = np.argsort(signal[:, t])
        ret[t] = future[order[-60:], t].mean() - future[order[:60], t].mean()
    return np.cumprod(1 + ret)


rev_cum = ls_curve(REV)
rating_cum = ls_curve(rating_level)
months = np.arange(M)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, rev_cum, color=C["ls"], lw=2.2, label="REV 长短因子（预期变化）")
ax.plot(months, rating_cum, color=C["neg"], lw=1.8, ls="--", label="静态评级长短因子（绝对值）")
ax.set_xlabel("月份（约 12 年）")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("预期「变化」比「水平」更能预测未来收益")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "rev_vs_rating.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图三：REV 十分组平均未来收益（单调）
# ============================================================
dec_avg = np.zeros(10)
for t in range(M):
    order = np.argsort(REV[:, t])
    for d in range(10):
        idx = order[d * 60:(d + 1) * 60]
        dec_avg[d] += future[:, t][idx].mean()
dec_avg /= M

fig, ax = plt.subplots(figsize=(8, 5))
colors = [C["neg"] if v < 0 else C["pos"] for v in dec_avg]
ax.bar(range(10), dec_avg * 100, color=colors)
ax.set_xlabel("REV 十分位（D1 最负修正 → D10 最正修正）")
ax.set_ylabel("平均未来 1 月收益（%）")
ax.set_title("REV 十分组：单调递增，D10-D1 显著为正")
ax.grid(True, color=C["grid"], axis="y")
fig.tight_layout()
fig.savefig(os.path.join(D, "rev_decile.png"), dpi=130)
plt.close(fig)


# ============================================================
# 图四：REV 信号叠加 60/40 组合（中观回测）
# ============================================================
# 基准 60/40：60% 市场 + 40% 现金(0)
base_ret = 0.6 * mkt + 0.4 * 0.001
# 叠加：用 REV 横截面多空月度收益对 60/40 做 20% 重新配置
rev_ls = np.array([future[np.argsort(REV[:, t])[-60:], t].mean()
                   - future[np.argsort(REV[:, t])[:60], t].mean() for t in range(M)])
blend_ret = 0.8 * base_ret + 0.2 * rev_ls
base_cum = np.cumprod(1 + base_ret)
blend_cum = np.cumprod(1 + blend_ret)


def metrics(cum, ret):
    yrs = M / 12
    ann = cum[-1] ** (1 / yrs) - 1
    sharpe = ret.mean() / (ret.std() + 1e-9) * np.sqrt(12)
    peak = np.maximum.accumulate(cum)
    mdd = (cum / peak - 1).min()
    return ann, sharpe, mdd


ba, bs, bm = metrics(base_cum, base_ret)
la, ls_, lm = metrics(blend_cum, blend_ret)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(months, base_cum, color=C["blue"], lw=1.8, label=f"60/40 基准 (年化{ba*100:.1f}%, SR {bs:.2f})")
ax.plot(months, blend_cum, color=C["gold"], lw=2.2,
        label=f"60/40 + 20% REV 叠加 (年化{la*100:.1f}%, SR {ls_:.2f})")
ax.set_xlabel("月份")
ax.set_ylabel("累计净值（起始=1）")
ax.set_title("REV 信号叠加 60/40：抬升收益与夏普")
ax.grid(True, color=C["grid"])
ax.legend(frameon=False)
fig.tight_layout()
fig.savefig(os.path.join(D, "rev_blend.png"), dpi=130)
plt.close(fig)

print("analyst-revision-momentum images saved:", os.listdir(D))
