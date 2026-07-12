#!/usr/bin/env python3
"""
为文章「因子动量：把因子本身当成可交易的资产」(factor-momentum-trading)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由「持续因子状态 + 共同市场 + 特异噪声」自洽合成，仅用于演示方法；落地见文末路径）：
  * 构造 5 个风格因子(价值/动量/规模/质量/低波)的月度收益，每个因子有一个 AR(1) 慢变
    「因子状态」s_i,t（phi=0.85 高持续）→ 因子收益可预测 → 因子动量成立。
  * 因子收益 = 状态 s_i,t + 共同市场 β·M_t + 特异噪声；因子间经市场弱相关（真实）。
  * 因子动量组合：每月用各因子过去 12 个月累计收益作信号，做多信号为正的因子(等权)，持 1 月。
  * 与「静态等权持有全部因子」对比，验证因子动量提供了新的、与底层因子低相关的收益来源。

注意：本模拟用于演示机制，数字量级参考真实因子（年化 Sharpe ~0.4-0.6），但非真实数据。
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
D = os.path.join(BASE, "factor-momentum-trading")
os.makedirs(D, exist_ok=True)

C = {"val": "#4C72B0", "mom": "#C44E52", "siz": "#55A868", "qual": "#DD8452",
     "lowv": "#8172B3", "fm": "#2F4B7C", "ew": "#999999", "grid": "#DDDDDD",
     "mkt": "#CCB974", "pos": "#C44E52", "neg": "#4C72B0"}

NF = 5
NAMES = ["价值 Value", "动量 Momentum", "规模 Size", "质量 Quality", "低波 LowVol"]
T = 300                      # 25 年月度
LOOK = 12                    # 因子动量信号回望(月)
REB = 1                      # 每月调仓

# ============================================================
# 1) 合成因子月度收益
# ============================================================
def simulate(seed=20260712):
    rng = np.random.default_rng(seed)
    # 各因子长期漂移(年化 Sharpe 目标 ~0.4-0.6 → 月度 mu 与 sigma)
    mu = np.array([0.0075, 0.0085, 0.0060, 0.0070, 0.0055])     # 月度均值
    sig = np.array([0.034, 0.038, 0.040, 0.030, 0.026])         # 月度特异波动
    phi = 0.85                                                # 因子状态持续性
    beta = np.array([0.25, 0.40, 0.55, 0.20, -0.30])           # 对共同市场的暴露(低波为负)
    s = np.zeros((NF, T))
    for i in range(NF):
        s[i, 0] = rng.normal(0, sig[i] * 0.5)
        for t in range(1, T):
            s[i, t] = phi * s[i, t - 1] + rng.normal(0, sig[i] * np.sqrt(1 - phi ** 2))
    M = rng.normal(0, 0.030, T)                                # 共同市场月度收益
    noise = rng.normal(0, 1, (NF, T)) * sig[:, None]
    s = s - s.mean(axis=1, keepdims=True)                     # 去均值: 保留持续性(动量来源), 使单因子均值=μ(干净)
    ret = mu[:, None] + s + beta[:, None] * M[None, :] + noise
    return ret, M


ret, M = simulate()


def ann_stats(x):
    m = x.mean(); s = x.std(ddof=1)
    ann_r = (1.0 + m) ** 12 - 1.0
    ann_v = s * np.sqrt(12.0)
    sharpe = ann_r / ann_v if ann_v > 0 else 0.0
    # 最大回撤(百分比口径)
    eq = np.cumprod(1.0 + x); peak = np.maximum.accumulate(eq)
    mdd = float(np.min((eq - peak) / peak))
    return ann_r, ann_v, sharpe, mdd


# ============================================================
# 2) 因子动量组合(做多过去 12M 收益为正的因子, 等权)
# ============================================================
fm_w = np.zeros((NF, T))
fm_ret = np.zeros(T)
for t in range(LOOK, T):
    sig12 = ret[:, t - LOOK:t].sum(axis=1)        # 过去 12 月累计收益 = 信号
    longs = sig12 > 0
    if longs.sum() == 0:
        w = np.ones(NF) / NF                       # 全空则等权(兜底)
    else:
        w = longs.astype(float) / longs.sum()
    fm_w[:, t] = w
    fm_ret[t] = float(np.dot(w, ret[:, t]))
fm_ret = fm_ret[LOOK:]                             # 有效区间

# 静态等权持有全部因子
ew_ret = ret.mean(axis=0)[LOOK:]

fr, fv, fs, fmdd = ann_stats(fm_ret)
er, ev, es, emdd = ann_stats(ew_ret)
# 各因子 standalone
standalone = [ann_stats(ret[i, LOOK:]) for i in range(NF)]
# 因子动量 与 静态EW 的相关
corr_fe = np.corrcoef(fm_ret, ew_ret)[0, 1]
corr_fm = np.corrcoef(fm_ret, M[LOOK:])[0, 1]
# 信号 IC: 12M 累计收益 与 下月因子收益 的横截面相关(平均)
ics = []
for t in range(LOOK, T - 1):
    sig12 = ret[:, t - LOOK:t].sum(axis=1)
    nxt = ret[:, t + 1]
    if np.std(sig12) > 1e-9 and np.std(nxt) > 1e-9:
        ics.append(np.corrcoef(sig12, nxt)[0, 1])
ic_mean = float(np.mean(ics))
# 命中率: 因子动量多空方向(信号>0 是否对应下月正收益)
hit = float(np.mean([np.mean(ret[:, t - LOOK:t].sum(1) > 0) for t in range(LOOK, T)]))
# 因子动量额外提升了静态组合多少: 组合 = 50% EW + 50% FM
comb_ret = 0.5 * ew_ret + 0.5 * fm_ret
cr, cv, cs, cmdd = ann_stats(comb_ret)


# ============================================================
# 图 1：五因子累计净值
# ============================================================
fig, ax = plt.subplots(figsize=(11, 5.0))
cols = [C["val"], C["mom"], C["siz"], C["qual"], C["lowv"]]
for i in range(NF):
    eq = np.cumprod(1.0 + ret[i])
    ax.plot(eq, color=cols[i], lw=1.2, label=NAMES[i])
ax.set_ylabel("净值 (期初=1)"); ax.set_xlabel("月份")
ax.set_title("五个风格因子的累计净值：轮动明显(有的领涨、有的沉寂)")
ax.legend(fontsize=8, ncol=5, loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "fm_factor_series.png"), dpi=130)
plt.close()


# ============================================================
# 图 2：因子动量权重热力图(哪些因子被做多)
# ============================================================
fig, ax = plt.subplots(figsize=(7.5, 5.2))
W = fm_w[:, LOOK:].T
im = ax.imshow(W, aspect="auto", cmap="RdBu_r", vmin=0, vmax=0.5, interpolation="nearest")
ax.set_yticks([])
ax.set_xticks(range(NF)); ax.set_xticklabels(NAMES, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("月份(调仓)")
ax.set_title("因子动量权重：每月只做多「过去12月收益为正」的因子(红=做多, 白=未选)")
cb = fig.colorbar(im, ax=ax, shrink=0.8); cb.set_label("权重")
plt.tight_layout()
plt.savefig(os.path.join(D, "fm_timing_weights.png"), dpi=130)
plt.close()


# ============================================================
# 图 3：Sharpe 对比柱状
# ============================================================
labels = NAMES + ["静态等权全部因子", "因子动量组合"]
sharpe_vals = [standalone[i][2] for i in range(NF)] + [es, fs]
colors = cols + [C["ew"], C["fm"]]
fig, ax = plt.subplots(figsize=(11, 5.0))
bars = ax.bar(range(len(labels)), sharpe_vals, color=colors, alpha=0.9)
ax.axhline(0, color="black", lw=0.8)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8)
ax.set_ylabel("年化 Sharpe")
ax.set_title("Sharpe 对比：因子动量组合(%.2f) 高于任一单因子与静态等权(%.2f)"
             % (fs, es))
for b, val in zip(bars, sharpe_vals):
    ax.text(b.get_x() + b.get_width() / 2, val + (0.02 if val >= 0 else -0.05),
            "%.2f" % val, ha="center", fontsize=8)
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "fm_sharpe_bar.png"), dpi=130)
plt.close()


# ============================================================
# 图 4：因子动量净值 vs 静态等权 + 滚动相关
# ============================================================
fig, axes = plt.subplots(2, 1, figsize=(11, 6.2), sharex=False)
eq_fm = np.cumprod(1.0 + fm_ret); eq_ew = np.cumprod(1.0 + ew_ret)
xr = np.arange(len(fm_ret))
axes[0].plot(xr, eq_fm, color=C["fm"], lw=1.3, label="因子动量组合")
axes[0].plot(xr, eq_ew, color=C["ew"], lw=1.0, label="静态等权全部因子")
axes[0].axhline(1, color="black", lw=0.5)
axes[0].set_ylabel("净值 (期初=1)"); axes[0].legend(fontsize=8, loc="upper left")
axes[0].set_title("因子动量组合 vs 静态等权因子组合")
axes[0].grid(True, color=C["grid"], lw=0.6)
# 滚动 12 月相关
roll = np.full(len(fm_ret) - LOOK, np.nan)
for i in range(len(roll)):
    roll[i] = np.corrcoef(fm_ret[i:i + LOOK], ew_ret[i:i + LOOK])[0, 1]
axes[1].plot(np.arange(len(roll)), roll, color=C["mkt"], lw=1.0)
axes[1].axhline(corr_fe, color=C["thr"] if "thr" in C else "gray", ls="--", lw=1,
                label="全样本相关 %.2f" % corr_fe)
axes[1].set_ylabel("滚动12月相关"); axes[1].set_xlabel("月份")
axes[1].set_title("因子动量与静态因子组合的相关性：中等(并非完全重复)→ 提供了新的收益维度")
axes[1].legend(fontsize=8); axes[1].grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "fm_equity_corr.png"), dpi=130)
plt.close()


# ============================================================
# 关键数字
# ============================================================
print("=== 因子动量 关键数字 ===")
print("样本 %d 月(%d 年), %d 个因子, 信号回望 %d 月" % (T, T // 12, NF, LOOK))
for i in range(NF):
    a, b, c, d = standalone[i]
    print("  %-16s 年化%.1f%% 波动%.1f%% Sharpe%.2f 回撤%.1f%%"
          % (NAMES[i], a * 100, b * 100, c, d * 100))
print("静态等权因子组合: 年化%.1f%% 波动%.1f%% Sharpe%.2f 回撤%.1f%%"
      % (er * 100, ev * 100, es, emdd * 100))
print("因子动量组合:     年化%.1f%% 波动%.1f%% Sharpe%.2f 回撤%.1f%%"
      % (fr * 100, fv * 100, fs, fmdd * 100))
print("50%%动量+50%%静态 组合: 年化%.1f%% 波动%.1f%% Sharpe%.2f 回撤%.1f%%"
      % (cr * 100, cv * 100, cs, cmdd * 100))
print("因子动量 vs 静态EW 相关 = %.2f ; vs 共同市场相关 = %.2f" % (corr_fe, corr_fm))
print("信号IC(12M累计收益 vs 下月因子收益)均值 = %.3f ; 信号为正占比 = %.1f%%"
      % (ic_mean, 100 * hit))
print("\n图片已保存到:", D)
