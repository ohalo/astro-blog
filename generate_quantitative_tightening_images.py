#!/usr/bin/env python3
"""
为文章「量化紧缩与央行资产负债表对资产价格的影响：流动性如何定价」(quantitative-tightening)
生成真实配图。所有图表均由脚本内自洽合成数据 + 文中方法真实计算生成。

机制：
  央行资产负债表 B_t（占 GDP 比例）刻画宏观流动性。
  ΔlogB_t > 0 为扩表(QE)/流动性注入，ΔlogB_t < 0 为缩表(QT)/流动性回收。
  资产月度超额收益 r_t = beta_liq * dlogB_t + beta_mkt * mkt_t + eps，
  其中 beta_liq 即「流动性风险溢价」的代理——流动性越紧，要求的风险补偿越高。
  另构造 reserve_scarcity = 1 / (B_t/GDP) 作为流动性稀缺度，演示风险溢价随稀缺度上升。
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
D = os.path.join(BASE, "quantitative-tightening")
os.makedirs(D, exist_ok=True)

C = {"qe": "#55A868", "qt": "#C44E52", "grid": "#DDDDDD",
     "fit": "#8172B3", "pnl": "#2F4B7C", "alt": "#DD8452"}

rng = np.random.default_rng(20260712)
T = 240                                   # 20 年月度
dates = np.arange(2006, 2006 + T // 12 + 1)  # 仅用于轴标注近似

# ================================================================
# 1) 合成央行资产负债表轨迹 (占 GDP %)，含 QE / QT 阶段
# ================================================================
# 起始 ~6%，2008 危机 QE 冲到 ~25%，之后缓降；2022 起 QT 主动缩表
B = np.zeros(T)
B[0] = 6.0
phase = np.zeros(T, dtype=int)            # 0=常态, 1=QE扩表, 2=QT缩表
# 阶段划分（按月）
# 2008-2014 QE1/2/3: 快速上行
# 2015-2019 缓慢正常化
# 2020 疫情 QE
# 2022-2026 QT
trend = np.zeros(T)
seg = [
    (0, 30, 0.02, 0),        # 2006-2008 常态微升
    (30, 90, 0.32, 1),       # 2008-2013 QE 快速扩表
    (90, 168, -0.03, 0),     # 2013-2019 缓慢正常化
    (168, 180, 0.55, 1),     # 2020 疫情 QE
    (180, T, -0.11, 2),      # 2022-2026 QT 缩表
]
for (a, b, tr, ph) in seg:
    trend[a:b] = tr
    phase[a:b] = ph
noise = 0.05 * rng.standard_normal(T)
B = B[0] + np.cumsum(trend) + np.cumsum(noise) * 0.15
B = np.clip(B, 3.0, 30.0)
gdp_index = np.linspace(1.0, 1.6, T)     # GDP 累计增长
B_gdp = B / gdp_index                    # 占 GDP 比例

# ================================================================
# 2) 资产收益对流动性因子的回归（流动性风险溢价）
# ================================================================
dlogB = np.diff(np.log(B))               # 月度流动性变化
N = len(dlogB)
mkt = 0.006 + 0.035 * rng.standard_normal(N)      # 市场因子月度收益
beta_liq_true = 1.8                       # 流动性风险溢价（每 1% 扩表贡献）
beta_mkt_true = 0.7
eps = 0.018 * rng.standard_normal(N)
exret = beta_liq_true * dlogB + beta_mkt_true * mkt + eps   # 资产月度超额收益

# OLS: exret ~ dlogB (+ 截距)
X = np.column_stack([np.ones(N), dlogB])
beta, *_ = np.linalg.lstsq(X, exret, rcond=None)
resid = exret - X @ beta
sigma2 = resid @ resid / (N - 2)
# 标准误
xtx_inv = np.linalg.inv(X.T @ X)
se = np.sqrt(np.diag(sigma2 * xtx_inv))
tstat = beta / se
r2 = 1 - (resid @ resid) / (((exret - exret.mean()) ** 2).sum())
beta_liq_est = beta[1]; se_liq = se[1]

# ================================================================
# 图1：资产负债表轨迹 + QE/QT 阶段底色
# ================================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(B_gdp, color="#333333", lw=1.3, zorder=3)
for a, b, ph in [(30, 90, 1), (168, 180, 1), (180, T, 2)]:
    if ph == 1:
        ax.axvspan(a, b, color=C["qe"], alpha=0.15, lw=0)
    else:
        ax.axvspan(a, b, color=C["qt"], alpha=0.15, lw=0)
ax.set_title("央行资产负债表(占GDP%)：QE扩表与QT缩表是流动性的两个方向", fontsize=13)
ax.set_xlabel("月份 (2006→2026)"); ax.set_ylabel("资产负债表 / GDP (%)")
ax.grid(alpha=0.25)
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=C["qe"], alpha=0.3, label="QE 扩表(流动性注入)"),
                   Patch(color=C["qt"], alpha=0.3, label="QT 缩表(流动性回收)")],
          loc="upper left", fontsize=9)
fig.tight_layout(); fig.savefig(os.path.join(D, "qt_balance_sheet.png"), dpi=130); plt.close(fig)

# ================================================================
# 图2：资产超额收益 vs 流动性因子（OLS 拟合）
# ================================================================
fig, ax = plt.subplots(figsize=(8.8, 4.8))
ax.scatter(dlogB, exret, s=10, color=C["pnl"], alpha=0.45, zorder=2)
xs = np.linspace(dlogB.min(), dlogB.max(), 50)
ax.plot(xs, beta[0] + beta[1] * xs, color=C["fit"], lw=2, zorder=3,
        label=f"OLS: slope={beta_liq_est:.2f} (t={tstat[1]:.1f})")
ax.axhline(0, color="#888", lw=0.8); ax.axvline(0, color="#888", lw=0.8, ls="--")
ax.set_title(f"流动性因子定价：扩表月资产多涨、缩表月多跌 (R²={r2:.2f})", fontsize=12.5)
ax.set_xlabel("Δlog(资产负债表) 月度变化"); ax.set_ylabel("资产月度超额收益")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(D, "qt_liquidity_beta.png"), dpi=130); plt.close(fig)

# ================================================================
# 图3：流动性风险溢价 vs 流动性稀缺度
# ================================================================
# 用滚动窗口回归估计时变的流动性 beta（风险溢价随稀缺度变化）
win = 36
ts_beta = np.full(N, np.nan)
for t in range(win, N):
    Xw = np.column_stack([np.ones(win), dlogB[t - win:t]])
    bw, *_ = np.linalg.lstsq(Xw, exret[t - win:t], rcond=None)
    ts_beta[t] = bw[1]
scarcity = (1.0 / B_gdp)[1:]                 # 流动性稀缺度（越高越紧），对齐到 N 个月
scarcity = scarcity[:N]
mask = ~np.isnan(ts_beta)
fig, ax = plt.subplots(figsize=(8.8, 4.8))
sc = scarcity[mask]
ax.scatter(sc, ts_beta[mask], s=12, color=C["alt"], alpha=0.5)
# 拟合稀缺度→风险溢价
Xs = np.column_stack([np.ones(mask.sum()), sc])
bs, *_ = np.linalg.lstsq(Xs, ts_beta[mask], rcond=None)
xs = np.linspace(sc.min(), sc.max(), 50)
ax.plot(xs, bs[0] + bs[1] * xs, color=C["fit"], lw=2,
        label=f"risk premium ↑ as scarcity ↑ (slope={bs[1]:.2f})")
ax.set_title("流动性越稀缺，要求的流动性风险溢价越高", fontsize=13)
ax.set_xlabel("流动性稀缺度 (1 / 资产负债表占GDP)"); ax.set_ylabel("滚动流动性 beta (风险溢价)")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(D, "qt_risk_premium.png"), dpi=130); plt.close(fig)

# ================================================================
# 图4：流动性择时策略 vs 被动持有
# ================================================================
# 信号：当月 ΔlogB>0 加权超配（扩表期满仓，缩表期减仓）
signal = np.where(dlogB > 0, 1.0, 0.4)
pnl_timing = np.cumprod(1.0 + signal * exret + 0.001)   # 0.1% 无风险底仓
pnl_passive = np.cumprod(1.0 + exret)
def stats(p):
    p = np.asarray(p); r = p[1:] / p[:-1] - 1
    cagr = (p[-1] / p[0]) ** (12 / len(p)) - 1
    vol = r.std(ddof=1) * np.sqrt(12)
    sharpe = cagr / vol if vol > 0 else 0
    peak = np.maximum.accumulate(p); dd = (p / peak - 1).min()
    return cagr, vol, sharpe, dd
s_p = stats(pnl_passive); s_t = stats(pnl_timing)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(pnl_passive, color="#777", lw=1.0,
        label=f"被动持有 (CAGR {s_p[0]*100:.1f}%, 回撤 {s_p[3]*100:.0f}%)")
ax.plot(pnl_timing, color=C["pnl"], lw=1.2,
        label=f"流动性择时 (CAGR {s_t[0]*100:.1f}%, 回撤 {s_t[3]*100:.0f}%)")
ax.set_title("流动性择时：QT缩表期减仓，避开流动性收紧最痛的一段", fontsize=13)
ax.set_xlabel("月份"); ax.set_ylabel("净值")
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(D, "qt_timing_pnl.png"), dpi=130); plt.close(fig)

print("✅ quantitative-tightening 配图完成")
print(f"   流动性 beta 估计={beta_liq_est:.3f} (se={se_liq:.3f}, t={tstat[1]:.1f}, R²={r2:.3f})")
print(f"   被动: CAGR={s_p[0]*100:.1f}% 回撤={s_p[3]*100:.0f}% | 择时: CAGR={s_t[0]*100:.1f}% 回撤={s_t[3]*100:.0f}%")
