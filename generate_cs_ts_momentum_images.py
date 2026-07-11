#!/usr/bin/env python3
"""
为文章「横截面动量 vs 时间序列动量：两种动量的分工与组合」(cross-sectional-ts-momentum)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

机制（数据由「共同市场趋势 + 持续横截面强弱 + 特异噪声」自洽合成，仅用于演示方法；真实落地见文末路径）：
  - 一个自相关的市场因子 F_t（带正自相关 → 有趋势）→ 时间序列动量(TSMOM)能捕捉它；
  - 每个资产有各自的「相对强弱」p_i,t（慢变 AR(1)）→ 横截面排名持续 → 横截面动量(CS)能捕捉它；
  - 资产收益 = 0.4*F_t + 0.3*(p_i,t - 截面均值) + 大方差特异噪声。
  因此 CS 赚「相对强弱」的钱（市场中性），TSMOM 赚「共同趋势」的钱（带方向），两者低相关、互补。
  组合（等权混合）的 Sharpe 高于任一单策略。
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
D = os.path.join(BASE, "cross-sectional-ts-momentum")
os.makedirs(D, exist_ok=True)

C = {"cs": "#2F4B7C", "ts": "#C44E52", "combo": "#55A868", "bench": "#999999",
     "grid": "#DDDDDD", "mkt": "#8172B3", "fill": "#DD8452"}

N = 12                      # 资产数
T = 252 * 10                # 约 10 年日度
L = 60                      # 动量信号回望窗（约 3 个月交易日）
REB = 21                    # 月度调仓（21 个交易日）

# 调参（经独立调参脚本标定，使两策略 Sharpe 落在可信区间、组合 Sharpe 高于任一单策略）
Fphi, Fvol = 0.72, 0.008    # 共同市场趋势因子：温和自相关（趋势弱而真实）
qphi, qvol = 0.95, 0.010    # 横截面相对强弱：慢变 AR(1)
GAMMA_Q, NOISE, FW, DRIFT = 0.13, 0.013, 0.30, 0.03 / 252

# ============================================================
# 1) 数据合成：共同市场趋势 + 持续横截面强弱 + 特异噪声
# ============================================================
def simulate(seed=42):
    rng = np.random.default_rng(seed)
    # 共同市场因子 F_t：温和正自相关 → 时间序列动量(TSMOM)能捕捉共同趋势
    F = np.zeros(T)
    for t in range(1, T):
        F[t] = Fphi * F[t - 1] + np.sqrt(1 - Fphi ** 2) * rng.normal(0, Fvol)
    # 每个资产的「相对强弱」p_i,t：慢变 AR(1) → 横截面排名持续 → 横截面动量(CS)可捕捉
    p = np.zeros((N, T))
    for i in range(N):
        for t in range(1, T):
            p[i, t] = qphi * p[i, t - 1] + np.sqrt(1 - qphi ** 2) * rng.normal(0, qvol)
    # 资产日收益 = 微小漂移 + 共同趋势暴露 + 横截面相对强弱 + 大方差特异噪声
    R = np.zeros((N, T))
    for t in range(1, T):
        cross = p[:, t] - p[:, t].mean()          # 横截面相对强弱（去均值，使 CS 市场中性）
        for i in range(N):
            R[i, t] = DRIFT + FW * F[t] + GAMMA_Q * cross[i] + rng.normal(0, NOISE)
    return F, p, R

# ============================================================
# 2) 横截面动量（CS）：每月按回望收益排名，多前 k 空后 k，美元中性
# ============================================================
def cs_momentum(R, k=3, L=L, REB=REB):
    pos = np.zeros((N, T))
    ret = np.zeros(T)
    for s in range(REB, T, REB):
        # 用 [s-L, s) 的累计收益排名（避免当日前视：只用 s 之前的数据）
        mom = R[:, s - L:s].sum(axis=1)
        order = np.argsort(mom)
        longs = order[-k:]; shorts = order[:k]
        w = np.zeros(N)
        w[longs] = 1.0 / k
        w[shorts] = -1.0 / k          # 多空相抵 → 美元中性
        pos[:, s:] = w.reshape(-1, 1)
    for t in range(1, T):
        ret[t] = pos[:, t] @ R[:, t]
    return np.cumprod(1 + ret), ret

# ============================================================
# 3) 时间序列动量（TSMOM）：每个资产各自 sign(回望收益)，等权聚合（带方向）
# ============================================================
def ts_momentum(R, L=L):
    pos = np.zeros((N, T))
    for t in range(L, T):
        mom = R[:, t - L:t].sum(axis=1)
        pos[:, t] = np.sign(mom) / N     # 每个资产独立方向，等权
    ret = np.zeros(T)
    for t in range(1, T):
        ret[t] = pos[:, t] @ R[:, t]
    return np.cumprod(1 + ret), ret

# ============================================================
# 4) 等权市场基准（买入持有等权）
# ============================================================
def buyhold(R):
    ret = R.mean(axis=0)
    return np.cumprod(1 + ret), ret

# ============================================================
# 主计算
# ============================================================
F, p, R = simulate()
cum_cs, rc = cs_momentum(R)
cum_ts, rt = ts_momentum(R)
cum_bh, rb = buyhold(R)
cum_combo = np.cumprod(1 + 0.5 * (rc + rt))
t = np.arange(T)

def ann_sharpe(ret):
    r = ret[1:]
    if r.std() == 0:
        return 0.0, 0.0
    return (r.mean() * 252) / (r.std() * np.sqrt(252)), (np.prod(1 + r)) ** (252.0 / len(r)) - 1

def max_dd(eq):
    peak = np.maximum.accumulate(eq)
    return (eq / peak - 1.0).min()

sh_cs, an_cs = ann_sharpe(rc); sh_ts, an_ts = ann_sharpe(rt)
sh_cb, an_cb = ann_sharpe(0.5 * (rc + rt)); sh_bh, an_bh = ann_sharpe(rb)
dd_cs = max_dd(cum_cs); dd_ts = max_dd(cum_ts); dd_cb = max_dd(cum_combo); dd_bh = max_dd(cum_bh)

# ---------- 图 1：CS / TSMOM / 组合 / 等权基准 累计净值 ----------
fig, ax = plt.subplots(figsize=(11, 4.8))
ax.plot(t, cum_cs, color=C["cs"], lw=1.6, label="横截面动量 (CS, 多空中性)")
ax.plot(t, cum_ts, color=C["ts"], lw=1.6, label="时间序列动量 (TSMOM, 带方向)")
ax.plot(t, cum_combo, color=C["combo"], lw=2.0, label="组合 (等权混合)")
ax.plot(t, cum_bh, color=C["bench"], lw=1.0, ls=":", label="等权买入持有")
ax.set_xlabel("交易日"); ax.set_ylabel("累计净值 (初始=1.0)")
ax.set_title("两种动量的分工：CS 做相对强弱，TSMOM 做共同趋势")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cs_ts_equity.png"), dpi=130); plt.close()

# ---------- 图 2：CS 与 TSMOM 收益滚动相关性（低相关→互补）----------
win = 126
a = rc[1:]; b = rt[1:]
corr = np.full(len(a) - win + 1, np.nan)
for i in range(len(corr)):
    x = a[i:i + win]; y = b[i:i + win]
    if x.std() > 0 and y.std() > 0:
        corr[i] = np.corrcoef(x, y)[0, 1]
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(np.arange(win, len(a) + 1), corr, color=C["mkt"], lw=1.2)
ax.axhline(corr.mean(), color=C["ts"], ls="--", lw=1.0, label="均值 %.2f" % corr.mean())
ax.set_xlabel("交易日"); ax.set_ylabel("滚动 %d 日相关系数" % win)
ax.set_title("CS 与 TSMOM 日收益相关性：长期偏低，组合有分散价值")
ax.legend(loc="upper right", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "cs_ts_correlation.png"), dpi=130); plt.close()

# ---------- 图 3：CS 调仓时的多空权重分布（某次调仓快照）----------
s = REB * 40
mom = R[:, s - L:s].sum(axis=1)
order = np.argsort(mom)
labels = ["A%d" % (i + 1) for i in range(N)]
w = np.zeros(N)
w[order[-3:]] = 1/3; w[order[:3]] = -1/3
colors = [C["cs"] if v > 0 else C["ts"] for v in w]
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.bar(labels, w, color=colors)
ax.axhline(0, color="#333", lw=0.8)
ax.set_ylabel("权重 (多正/空负)")
ax.set_title("横截面动量调仓快照：多前 3 / 空后 3，美元中性")
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout(); plt.savefig(os.path.join(D, "cs_weights.png"), dpi=130); plt.close()

# ---------- 图 4：组合 Sharpe 随混合比例 w 扫描 ----------
ws = np.linspace(0, 1, 41)
sharpes = []
for w_ in ws:
    rc_ = w_ * rc + (1 - w_) * rt
    sh, _ = ann_sharpe(rc_)
    sharpes.append(sh)
best = int(np.argmax(sharpes))
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(ws, sharpes, color=C["combo"], lw=2.0)
ax.axvline(0.5, color=C["bench"], ls="--", lw=1.0, label="等权 w=0.5")
ax.scatter([ws[best]], [sharpes[best]], color=C["ts"], zorder=5,
           label="最优 w=%.2f (Sharpe=%.2f)" % (ws[best], sharpes[best]))
ax.set_xlabel("组合中 CS 的权重 w（TSMOM 权重 = 1-w）")
ax.set_ylabel("组合年化 Sharpe")
ax.set_title("混合比例扫描：组合 Sharpe 高于任一单策略")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout(); plt.savefig(os.path.join(D, "combined_sharpe.png"), dpi=130); plt.close()

print("=== 横截面 vs 时间序列动量 关键数字 ===")
print("样本: 日度 %d 天 (约 %.1f 年)，%d 个资产" % (T, T / 252, N))
print("CS动量   : 年化=%.1f%%, Sharpe=%.2f, 最大回撤=%.1f%%" % (an_cs * 100, sh_cs, dd_cs * 100))
print("TSMOM    : 年化=%.1f%%, Sharpe=%.2f, 最大回撤=%.1f%%" % (an_ts * 100, sh_ts, dd_ts * 100))
print("组合(0.5): 年化=%.1f%%, Sharpe=%.2f, 最大回撤=%.1f%%" % (an_cb * 100, sh_cb, dd_cb * 100))
print("等权基准 : 年化=%.1f%%, Sharpe=%.2f, 最大回撤=%.1f%%" % (an_bh * 100, sh_bh, dd_bh * 100))
print("CS-TSMOM 滚动相关均值=%.2f" % corr.mean())
print("\n图片已保存到:", D)
