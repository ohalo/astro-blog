"""时间序列动量(TSMOM) 配图生成 —— 基于合成多资产数据真实计算。
诚实演示: 动量信号来自温和的截面内自相关(非完美趋势), 杠杆封顶 1.0。
目标 Sharpe 落在可信区间 (~1.0–1.3), 不追求夸张收益。
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.colors import LinearSegmentedColormap

# ---- 中文字体 ----
_FP = "/System/Library/Fonts/STHeiti Medium.ttc"
fm.fontManager.addfont(_FP)
_CJK = fm.FontProperties(fname=_FP).get_name()
plt.rcParams["font.family"] = _CJK
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(20260717)
OUT = "public/images/time-series-momentum"
os.makedirs(OUT, exist_ok=True)

ACCENT = "#2563eb"; ACCENT2 = "#dc2626"; GREY = "#6b7280"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 13, "axes.titleweight": "normal", "font.weight": "normal",
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 110, "savefig.dpi": 110,
})

# ---- 1. 合成多资产月频数据(2007-2025, 8 个资产类别) ----
n_assets = 8
n_months = 228
dates = pd.date_range("2007-01-31", periods=n_months, freq="ME")
names = ["股票指数", "科技股", "国债", "信用债", "黄金", "原油", "铜", "美元指数"]
# 温和自相关(真实市场动量强度, 制造持续数月的方向但不完美)
rho = np.array([0.03, 0.035, 0.02, 0.015, 0.025, 0.04, 0.035, -0.02])
mu  = np.array([0.007, 0.009, 0.003, 0.004, 0.005, 0.008, 0.006, 0.001])
sig = np.array([0.045, 0.060, 0.012, 0.015, 0.035, 0.075, 0.055, 0.025])

rets = np.zeros((n_months, n_assets))
eps = np.zeros((n_months, n_assets))
for a in range(n_assets):
    trend = 0.0
    for t in range(n_months):
        shock = np.random.normal(0, sig[a])
        trend += np.random.normal(0, sig[a]*0.10)
        trend *= 0.92
        eps[t, a] = rho[a]*eps[t-1, a] if t > 0 else 0.0
        eps[t, a] += shock
        rets[t, a] = mu[a] + trend + eps[t, a]
rets = pd.DataFrame(rets, index=dates, columns=names)

# ---- 2. TSMOM 信号: 过去 12 月收益(跳过最近 1 月)定方向, 波动率目标化, 杠杆封顶 1.0 ----
lookback, skip, vol_tgt = 12, 1, 0.10
lev_cap = 1.0
pos = pd.DataFrame(index=dates, columns=names, dtype=float)
for t in range(lookback+skip, n_months):
    sig_ret = (1+rets.iloc[t-lookback-skip:t-skip]).prod() - 1
    direction = np.sign(sig_ret.values)
    hist = rets.iloc[t-lookback:t].std().values * np.sqrt(12)
    scale = np.where(hist > 0, np.minimum(vol_tgt/np.maximum(hist, 1e-6), lev_cap), 0.0)
    pos.iloc[t] = direction * scale
pos = pos.fillna(0.0)

strat_ret = (pos.shift(1).values * rets.values)
strat_ret = pd.Series(strat_ret.sum(axis=1), index=dates)
ew_ret = rets.mean(axis=1)

def perf(s):
    c = (1+s).cumprod(); n = len(s)
    ann = c.iloc[-1]**(12/n) - 1
    sharpe = s.mean()/s.std()*np.sqrt(12) if s.std() > 0 else 0
    peak = c.cummax(); dd = (c-peak)/peak
    return c, ann, sharpe, dd.min()

c_s, ann_s, sh_s, mdd_s = perf(strat_ret)
c_e, ann_e, sh_e, mdd_e = perf(ew_ret)

# ===== 图 1: 净值曲线 =====
fig, ax = plt.subplots(figsize=(11, 5.2))
ax.plot(c_s.index, c_s.values, color=ACCENT, lw=2.2, label=f"TSMOM(波动率目标) Sharpe={sh_s:.2f}")
ax.plot(c_e.index, c_e.values, color=GREY, lw=1.6, ls="--", label=f"等权买入持有 Sharpe={sh_e:.2f}")
ax.set_title("时间序列动量 vs 等权买入持有（合成多资产, 2007–2025）")
ax.set_ylabel("净值 (起始=1)"); ax.legend(frameon=False, loc="upper left")
ax.axhline(1, color="k", lw=0.6, alpha=0.3)
fig.tight_layout(); fig.savefig(f"{OUT}/tsmom-equity-curve.png"); plt.close(fig)

# ===== 图 2: 持仓方向热力图 =====
pos_show = pos.shift(1).iloc[lookback+skip:]
sub = pos_show.iloc[-80:]
cmap = LinearSegmentedColormap.from_list("pos", [ACCENT2, "#f3f4f6", ACCENT])
fig, ax = plt.subplots(figsize=(11, 5.0))
im = ax.imshow(sub.T.values, aspect="auto", cmap=cmap, vmin=-1, vmax=1)
ax.set_yticks(range(n_assets)); ax.set_yticklabels(names, fontsize=9)
step = 12
ax.set_xticks(range(0, len(sub), step))
ax.set_xticklabels([d.strftime("%Y-%m") for d in sub.index[::step]], rotation=45, fontsize=8)
ax.set_title("TSMOM 持仓方向热力图（蓝=多, 红=空, 灰=近零; 最近 80 个月）")
fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="方向×波动缩放")
fig.tight_layout(); fig.savefig(f"{OUT}/tsmom-position-heatmap.png"); plt.close(fig)

# ===== 图 3: 回撤对比 =====
def drawdown(s):
    c = (1+s).cumprod(); peak = c.cummax(); return (c-peak)/peak
dd_s = drawdown(strat_ret); dd_e = drawdown(ew_ret)
fig, ax = plt.subplots(figsize=(11, 5.0))
ax.fill_between(dd_s.index, dd_s.values*100, 0, color=ACCENT, alpha=0.55, label=f"TSMOM 最大回撤 {mdd_s*100:.1f}%")
ax.fill_between(dd_e.index, dd_e.values*100, 0, color=GREY, alpha=0.35, label=f"等权买入持有 最大回撤 {mdd_e*100:.1f}%")
ax.set_title("回撤对比（%）")
ax.set_ylabel("回撤 %"); ax.legend(frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(f"{OUT}/tsmom-drawdown.png"); plt.close(fig)

print("TSMOM sharpe=%.3f ann=%.3f maxdd=%.3f" % (sh_s, ann_s, mdd_s))
print("EW    sharpe=%.3f ann=%.3f maxdd=%.3f" % (sh_e, ann_e, mdd_e))
print("charts written to", OUT)
