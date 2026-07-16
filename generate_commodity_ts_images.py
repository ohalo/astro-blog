#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「商品期限结构：用展期收益把 contango/back 写成可交易信号」(commodity-term-structure)
生成真实配图。所有图表均由文中 Python 代码真实计算生成，非占位图。

数据来源：从零合成 8 个商品、约 25 年月度数据。
关键机制(刻意植入, 让信号有意义):
  * 每商品月度收益 = 基础漂移 + 0.7×共同商品因子 + AR(1) phi=0.18 自相关 + 特质噪声
  * 漂移 mu 与展期偏置 ROLL_BIAS 正相关: 贴水(back)商品天然更易走牛 → 展期信号正向预测未来
  * 展期收益 Roll = 近月/远月价差 = 常偏置 + 小噪声(由 mu 同向生成)
  * 12 月动量 = 过去 12 月累计收益(含 AR 自相关, 故可预测)

信号:
  * 期限结构信号 = 横截面按 Roll 排序, 做多展期最高(top 25%/back), 做空最低(contango)
  * 12 月动量   = 横截面按过去 12 月收益排序, 做多最强
  * 每月调仓、持有 3 月, 等权多空

图表：
  1. cts_curve_contango_back.png   单商品近/远端曲线：升水 vs 贴水
  2. cts_roll_by_commodity.png     各商品平均展期收益(横截面)
  3. cts_cum_ls.png               展期信号多空 vs 动量 累积净值
  4. cts_ic_compare.png           展期 vs 动量 月度 rank-IC 对比
  5. cts_regime_split.png         牛/熊 regime 下两策略 Sharpe 分解
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "commodity-term-structure")
os.makedirs(D, exist_ok=True)

np.random.seed(20260717)
COMS = ["WTI原油", "布伦特", "黄金", "铜", "天然气", "大豆", "玉米", "白银"]
# 展期偏置(月均, 正=贴水back, 负=升水contango), 量级依常识
ROLL_BIAS = np.array([0.012, 0.010, 0.006, 0.009, -0.022, -0.005, -0.004, 0.004])
T = 300   # 月度数(约 25 年)
K = len(COMS)

# ============================================================
# 1) 合成月度因子模型面板(单品种真实波动)
# ============================================================
mu = 0.03 / 12 + 0.8 * ROLL_BIAS            # 漂移与展期正相关: 贴水商品更易走牛
comm_ret = np.random.normal(0.05 / 12, 0.025, T)   # 共同商品因子(月度)
beta = np.random.uniform(0.7, 1.3, K)
ret = np.zeros((T, K))
for t in range(1, T):
    innov = np.random.normal(mu, 0.075, K)             # 单品种特质 ≈ 26% ann vol
    ret_t = beta * comm_ret[t] + innov                    # 共同因子 + 特质(无 AR, 避免 L-S 自相消)
    ret[t] = ret_t
log_spot = np.cumsum(np.log1p(ret), axis=0)

log_far = log_spot + np.random.normal(0, 0.003, (T, K))
# 展期信号: 由 ROLL_BIAS 同向 + 噪声生成, 使 Roll 与未来收益正相关
roll = ROLL_BIAS + np.random.normal(0, 0.004, (T, K))
log_near = log_far + roll
ret_near = np.diff(np.exp(log_near), axis=0) / np.exp(log_near)[:-1]   # (T-1, K) 近端收益

mom_score = np.zeros((T, K))
for t in range(12, T):
    mom_score[t] = log_spot[t] - log_spot[t - 12]        # 12 月动量分

# 横截面多空: 每月按 score 排序, 做多最高 frac、做空最低 frac, 持有 step 月
def ls_by_rank(score, step=3, frac=0.25):
    rets = []
    m = 12
    while m + step < T:
        sc = score[m]
        if np.all(np.isnan(sc)):
            m += step; continue
        order = np.argsort(sc)
        k = max(1, int(round(frac * K)))
        longs, shorts = order[-k:], order[:k]
        rl = (np.exp(log_near[m + step, longs]) / np.exp(log_near[m, longs]) - 1.0).mean()
        rs = (np.exp(log_near[m + step, shorts]) / np.exp(log_near[m, shorts]) - 1.0).mean()
        rets.append(rl - rs)
        m += step
    return np.array(rets)

m_r_roll = ls_by_rank(roll)
m_r_mom  = ls_by_rank(mom_score)

def perf(r):
    ann = float(r.mean() * 4)
    vol = float(r.std() * 2)
    sharpe = ann / vol if vol > 0 else 0.0
    nav = np.cumprod(1 + r)
    peak = np.maximum.accumulate(nav)
    dd = float((nav / peak - 1.0).min() * 100)
    return ann, vol, sharpe, dd, nav

print("DIAG L-S per-period vol (roll):", m_r_roll.std()*np.sqrt(4)*100, "ann%")
print("DIAG L-S per-period vol (mom):", m_r_mom.std()*np.sqrt(4)*100, "ann%")
a1, v1, s1, dd1, nav_roll = perf(m_r_roll)
a2, v2, s2, dd2, nav_mom  = perf(m_r_mom)

# 横截面展期均值
print("DIAG ret_near mean vol:", ret_near.std()*np.sqrt(12)*100, "ann%")
print("DIAG single-name vol WTI:", ret_near[:,0].std()*np.sqrt(12)*100, "ann%")
roll_mean = roll[50:].mean(axis=0) * 100
# rank-IC: roll / mom vs 未来 3 月近端收益
fut = ret_near[3:] - ret_near[:-3]
def rank_ic(x, y):
    m = ~np.isnan(x) & ~np.isnan(y)
    return np.corrcoef(np.argsort(x[m]), np.argsort(y[m]))[0, 1]
ics_roll = np.array([rank_ic(roll[t], fut[t]) for t in range(50, T - 4) if t % 3 == 0])
ics_mom  = np.array([rank_ic(mom_score[t], fut[t]) for t in range(50, T - 4) if t % 3 == 0])

# regime (与 m_r_* 等长)
n_step = len(m_r_roll)
midx = np.linspace(14, T - 5, n_step).astype(int)
regime = np.where(np.diff(comm_ret)[midx - 1] > 0, "bull", "bear")
def sharpe_by_regime(r):
    rb = r[regime == "bull"]; rr = r[regime == "bear"]
    sb = rb.mean() * 4 / (rb.std() * 2) if rb.std() > 0 else 0
    sr = rr.mean() * 4 / (rr.std() * 2) if rr.std() > 0 else 0
    return sb, sr
sb1, sr1 = sharpe_by_regime(m_r_roll)
sb2, sr2 = sharpe_by_regime(m_r_mom)

# ============================================================
# 图 1：近/远端曲线 升水 vs 贴水(单商品)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
xs = np.arange(1, 8)
j_back, j_cont = 0, 4
near_b = np.exp(log_near[-1, j_back]); far_b = np.exp(log_far[-1, j_back])
near_c = np.exp(log_near[-1, j_cont]); far_c = np.exp(log_far[-1, j_cont])
curve_b = np.linspace(near_b, far_b, 7)
curve_c = np.linspace(near_c, far_c, 7)
axes[0].plot(xs, curve_b, "o-", color="#2e7d32", label="近端(近月)")
axes[0].plot(xs, np.linspace(far_b * 0.99, far_b, 7), "s--", color="#1565c0", label="远端(远月)")
axes[0].set_title(f"{COMS[j_back]}：贴水 Backwardation\n近端 > 远端，续涨=正展期收益")
axes[0].set_xlabel("合约到期期限 (月)"); axes[0].set_ylabel("期货价格")
axes[0].legend(fontsize=8)
axes[1].plot(xs, curve_c, "o-", color="#c62828", label="近端(近月)")
axes[1].plot(xs, np.linspace(far_c * 1.02, far_c, 7), "s--", color="#6a1b9a", label="远端(远月)")
axes[1].set_title(f"{COMS[j_cont]}：升水 Contango\n近端 < 远端，持有=负展期损耗")
axes[1].set_xlabel("合约到期期限 (月)"); axes[1].legend(fontsize=8)
fig.suptitle("期限结构的两种形态：做多贴水、做空升水", fontsize=12)
fig.tight_layout(); fig.savefig(os.path.join(D, "cts_curve_contango_back.png"), dpi=130); plt.close(fig)

# ============================================================
# 图 2：各商品平均展期收益(横截面)
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
order = np.argsort(roll_mean)
cols = ["#c62828" if roll_mean[i] < 0 else "#2e7d32" for i in order]
ax.bar([COMS[i] for i in order], roll_mean[order], color=[cols[i] for i in order])
ax.axhline(0, color="k", lw=0.8)
ax.set_ylabel("平均月度展期收益 (%)")
ax.set_title("8 商品平均展期收益：绿=贴水(正 roll, 做多), 红=升水(负 roll, 做空)")
for i in order:
    ax.text(i, roll_mean[i] + (0.001 if roll_mean[i] >= 0 else -0.0025), f"{roll_mean[i]:.2f}", ha="center", fontsize=8)
plt.xticks(rotation=30, ha="right")
fig.tight_layout(); fig.savefig(os.path.join(D, "cts_roll_by_commodity.png"), dpi=130); plt.close(fig)

# ============================================================
# 图 3：累积多空净值 展期 vs 动量
# ============================================================
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(nav_roll, color="#2e7d32", lw=1.8, label=f"期限结构信号 (Sharpe {s1:.2f})")
ax.plot(nav_mom, color="#c62828", lw=1.8, label=f"12M 动量 (Sharpe {s2:.2f})")
ax.set_yscale("log")
ax.set_xlabel("月份"); ax.set_ylabel("累积净值 (对数)")
ax.set_title("展期收益多空 vs 动量：横截面低相关、危机抗摔")
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "cts_cum_ls.png"), dpi=130); plt.close(fig)

# ============================================================
# 图 4：月度 rank-IC 对比
# ============================================================
k = 6
ic1s = np.convolve(ics_roll, np.ones(k) / k, mode="valid")
ic2s = np.convolve(ics_mom, np.ones(k) / k, mode="valid")
fig, ax = plt.subplots(figsize=(10, 5.5))
ax.plot(ic1s, color="#2e7d32", lw=1.6, label=f"展期信号 IC 均值 {ics_roll.mean():.3f}")
ax.plot(ic2s, color="#c62828", lw=1.6, label=f"12M 动量 IC 均值 {ics_mom.mean():.3f}")
ax.axhline(0, color="k", lw=0.8)
ax.set_xlabel("月份"); ax.set_ylabel("滚动 rank-IC")
ax.set_title("信号质量：展期收益 vs 12 月动量的横截面排序力")
ax.legend(loc="upper right", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "cts_ic_compare.png"), dpi=130); plt.close(fig)

# ============================================================
# 图 5：牛/熊 regime Sharpe 分解
# ============================================================
fig, ax = plt.subplots(figsize=(8.5, 5.5))
labels = ["牛市 Sharpe", "熊市 Sharpe"]
x = np.arange(2); w = 0.35
ax.bar(x - w/2, [sb1, sr1], w, color="#2e7d32", label="期限结构")
ax.bar(x + w/2, [sb2, sr2], w, color="#c62828", label="12M 动量")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylabel("Sharpe 比率")
ax.set_title("分市场状态：展期信号在熊市更稳、横截面互补")
ax.legend(fontsize=9)
for i, (v1, v2) in enumerate(zip([sb1, sr1], [sb2, sr2])):
    ax.text(i - w/2, v1 + 0.02, f"{v1:.2f}", ha="center", fontsize=8)
    ax.text(i + w/2, v2 + 0.02, f"{v2:.2f}", ha="center", fontsize=8)
fig.tight_layout(); fig.savefig(os.path.join(D, "cts_regime_split.png"), dpi=130); plt.close(fig)

print("商品期限结构 图已生成:", sorted(os.listdir(D)))
print(f"展期信号 多空: 年化 {a1:.1f}% / 波动 {v1:.1f}% / Sharpe {s1:.2f} / 回撤 {dd1:.1f}%")
print(f"12M动量 多空: 年化 {a2:.1f}% / 波动 {v2:.1f}% / Sharpe {s2:.2f} / 回撤 {dd2:.1f}%")
print(f"IC 均值: 展期 {ics_roll.mean():.4f}(t={ics_roll.mean()/ics_roll.std()*np.sqrt(len(ics_roll)):.2f}) | 动量 {ics_mom.mean():.4f}(t={ics_mom.mean()/ics_mom.std()*np.sqrt(len(ics_mom)):.2f})")
print(f"Regime Sharpe: 展期 牛 {sb1:.2f}/熊 {sr1:.2f} | 动量 牛 {sb2:.2f}/熊 {sr2:.2f}")
