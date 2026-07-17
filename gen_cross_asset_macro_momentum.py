#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「跨资产宏观动量：把股、债、商品、外汇的趋势做成单一择时信号」生成真实配图与统计数字。

核心逻辑(Cross-Asset Macro Momentum / 时序动量的跨资产聚合):
  - 对每一类资产(股/债/商品/外汇)分别计算时间序列动量信号:
        s_i,t = sign( 过去 L 个交易日累计收益 )
  - 单资产择时收益 = s_i,{t-1} * r_i,t (信号滞后一期, 避免前视)
  - 波动率目标化: 每类资产按目标波动 / 近端已实现波动 缩放仓位, 让各资产贡献可比
  - 跨资产聚合: 把 4 类资产的择时收益等权(风险对齐后)相加, 得到单一「宏观动量」择时信号组合
  - 对照组: 等权 buy&hold(始终满仓 4 类资产)

全部数字由文中 Python 真实计算(仅依赖 numpy/matplotlib)。
图片:
  cam_paths.png        —— 累计净值: 宏观动量组合 vs 等权 buy&hold vs 单资产平均
  cam_asset_perf.png   —— 4 类资产各自「时序动量择时 vs buy&hold」年化收益对比
  cam_signal.png       —— 综合宏观动量暴露(4 类资产多空信号求和, -4~+4)随时间变化
  cam_drawdown.png     —— 宏观动量组合 vs buy&hold 回撤对比
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "cross-asset-macro-momentum")
os.makedirs(D, exist_ok=True)

C = {"mom": "#C0392B", "bh": "#34495E", "avg": "#2F4F8F", "band": "#C9D8F0",
     "grid": "#DDDDDD", "accent": "#E67E22", "good": "#27AE60", "bad": "#C0392B"}

# ----------------- 模拟参数 -----------------
rng = np.random.default_rng(20260717)
T = 3024                      # ~12 个交易年
ann = 252
names = ["股票", "债券", "商品", "外汇"]
n = len(names)

# 为每类资产构造「有趋势段 + 反转段」的收益, 让动量既能赚也会挨打
# 用分段漂移 + AR(1) 的方式制造持续趋势(动量的收益来源), 再叠随机噪声
mu_ann = np.array([0.07, 0.03, 0.05, 0.02])         # 长期漂移(年化)
sig_ann = np.array([0.16, 0.06, 0.20, 0.09])        # 年化波动
mu_d = mu_ann / ann
sig_d = sig_ann / np.sqrt(ann)

# 分段趋势: 每类资产的漂移在若干 regime 之间随机游走式缓慢切换。
# 关键: 趋势幅度远小于噪声波动(真实市场里动量只是弱信号), 否则会造出不真实的高 Sharpe。
def regime_drift(T, base, amp, seg_len, rng):
    out = np.zeros(T)
    t = 0
    while t < T:
        L = int(rng.integers(int(seg_len*0.6), int(seg_len*1.6)))
        # 每段漂移方向随机(不强制交替), 幅度随机 -> 趋势不完美、会被打脸
        d = base + amp * rng.standard_normal()
        out[t:t+L] = d
        t += L
    return out[:T]

R = np.zeros((T, n))
for i in range(n):
    # 趋势幅度只有波动的 ~18%, 使动量成为弱而真实的 edge
    trend = regime_drift(T, mu_d[i], sig_d[i]*0.18, seg_len=150, rng=rng)
    noise = sig_d[i] * rng.standard_normal(T)
    # 轻度 AR(1) 让趋势更连贯
    x = np.zeros(T)
    for t in range(1, T):
        x[t] = 0.05 * x[t-1] + noise[t]
    R[:, i] = trend + x

# ----------------- 时序动量信号 -----------------
L = 126                       # 回看窗口 ~6 个月
target_vol_d = 0.10 / np.sqrt(ann)   # 每类资产目标日波动(年化 10%)
vol_win = 60

# 累计动量(过去 L 日累计收益)
cum_mom = np.full((T, n), np.nan)
for t in range(L, T):
    cum_mom[t] = R[t-L:t].sum(axis=0)
signal = np.sign(cum_mom)     # +1 多 / -1 空 / 0(极少)

# 已实现波动(近端), 用于波动率目标化
realized_vol = np.full((T, n), np.nan)
for t in range(vol_win, T):
    realized_vol[t] = R[t-vol_win:t].std(axis=0, ddof=1)

# 仓位: 信号滞后一期 * 波动缩放(信号 t-1 已知, 用 t-1 的波动估计, 作用于 t 收益)
pos = np.zeros((T, n))
for t in range(max(L, vol_win)+1, T):
    scale = np.where(realized_vol[t-1] > 1e-9, target_vol_d / realized_vol[t-1], 0.0)
    scale = np.clip(scale, 0.0, 3.0)          # 杠杆上限, 防波动趋 0 时爆仓
    pos[t] = signal[t-1] * scale

# 单资产择时收益
r_timed = pos * R
# 跨资产聚合(等权风险对齐): 组合日收益 = 各资产择时收益均值
start = max(L, vol_win) + 1
r_macro = r_timed[start:].mean(axis=1)

# 对照: 等权 buy&hold(始终满仓, 波动率目标化后等权), 用相同缩放但信号恒 +1
pos_bh = np.zeros((T, n))
for t in range(vol_win+1, T):
    scale = np.where(realized_vol[t-1] > 1e-9, target_vol_d / realized_vol[t-1], 0.0)
    scale = np.clip(scale, 0.0, 3.0)
    pos_bh[t] = 1.0 * scale
r_bh = (pos_bh * R)[start:].mean(axis=1)

# 单资产平均(原始 buy&hold 等权, 无波动目标)
r_raw = R[start:].mean(axis=1)

def curve(r):
    return np.cumprod(1.0 + r)

eq_macro = curve(r_macro)
eq_bh = curve(r_bh)
eq_raw = curve(r_raw)

def stats(r):
    mu = r.mean() * ann
    sd = r.std(ddof=1) * np.sqrt(ann)
    sharpe = mu / sd if sd > 1e-9 else 0.0
    eq = curve(r)
    peak = np.maximum.accumulate(eq)
    mdd = ((eq - peak) / peak).min()
    return mu, sd, sharpe, mdd

mu_macro, sd_macro, sh_macro, mdd_macro = stats(r_macro)
mu_bh, sd_bh, sh_bh, mdd_bh = stats(r_bh)
mu_raw, sd_raw, sh_raw, mdd_raw = stats(r_raw)
corr_macro_bh = np.corrcoef(r_macro, r_bh)[0, 1]

# 各资产单独: 择时 vs buy&hold 年化
per_asset = []
for i in range(n):
    ri_t = r_timed[start:, i]
    ri_bh = (pos_bh[start:, i] * R[start:, i])
    per_asset.append((names[i],
                      ri_t.mean()*ann, ri_t.mean()/ (ri_t.std(ddof=1)+1e-12) * np.sqrt(ann),
                      ri_bh.mean()*ann))

# ----------------- 图 1: 累计净值 -----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
x = np.arange(len(eq_macro))
ax.plot(x, eq_macro, color=C["mom"], lw=2.0, label=f"跨资产宏观动量 (Sharpe {sh_macro:.2f})")
ax.plot(x, eq_bh, color=C["bh"], lw=1.6, label=f"波动目标等权 buy&hold (Sharpe {sh_bh:.2f})")
ax.plot(x, eq_raw, color=C["avg"], lw=1.2, ls="--", alpha=0.8, label=f"原始等权持有 (Sharpe {sh_raw:.2f})")
ax.set_yscale("log")
ax.set_title("跨资产宏观动量 vs 买入持有：累计净值(对数轴)", fontsize=12)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("累计净值(起点=1)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "cam_paths.png"), dpi=130)
plt.close(fig)

# ----------------- 图 2: 各资产择时 vs buy&hold -----------------
fig, ax = plt.subplots(figsize=(9, 4.6))
idx = np.arange(n)
w = 0.38
timed_ann = [p[1]*100 for p in per_asset]
bh_ann = [p[3]*100 for p in per_asset]
b1 = ax.bar(idx - w/2, timed_ann, w, color=C["mom"], label="时序动量择时")
b2 = ax.bar(idx + w/2, bh_ann, w, color=C["bh"], label="买入持有")
ax.set_xticks(idx)
ax.set_xticklabels([p[0] for p in per_asset], fontsize=10)
ax.axhline(0, color="#888", lw=0.9)
ax.set_title("四类资产：时序动量择时 vs 买入持有(年化收益 %)", fontsize=12)
ax.set_ylabel("年化收益 (%)", fontsize=10)
ax.grid(True, axis="y", color=C["grid"], ls=":", alpha=0.7)
ax.legend(fontsize=9)
for bars in (b1, b2):
    for b in bars:
        h = b.get_height()
        ax.annotate(f"{h:.1f}", (b.get_x()+b.get_width()/2, h), ha="center",
                    fontsize=8, xytext=(0, 2 if h >= 0 else -10), textcoords="offset points")
fig.tight_layout()
fig.savefig(os.path.join(D, "cam_asset_perf.png"), dpi=130)
plt.close(fig)

# ----------------- 图 3: 综合宏观动量暴露 -----------------
net_expo = signal[start:].sum(axis=1)   # -4 ~ +4
fig, ax = plt.subplots(figsize=(9, 4.0))
ax.fill_between(np.arange(len(net_expo)), 0, net_expo,
                where=(net_expo >= 0), color=C["good"], alpha=0.55, label="净多头(风险偏好)")
ax.fill_between(np.arange(len(net_expo)), 0, net_expo,
                where=(net_expo < 0), color=C["bad"], alpha=0.55, label="净空头(避险)")
ax.axhline(0, color="#555", lw=0.9)
ax.set_ylim(-4.3, 4.3)
ax.set_title("综合宏观动量暴露：四类资产多空信号求和(−4 全避险 ~ +4 全进攻)", fontsize=11.5)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("信号合计", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="upper left", fontsize=9, ncol=2)
fig.tight_layout()
fig.savefig(os.path.join(D, "cam_signal.png"), dpi=130)
plt.close(fig)

# ----------------- 图 4: 回撤对比 -----------------
def dd(eq):
    peak = np.maximum.accumulate(eq)
    return (eq - peak) / peak * 100.0
fig, ax = plt.subplots(figsize=(9, 4.2))
xx = np.arange(len(eq_macro))
ax.plot(xx, dd(eq_macro), color=C["mom"], lw=1.6, label=f"宏观动量 (MDD {mdd_macro*100:.1f}%)")
ax.plot(xx, dd(eq_bh), color=C["bh"], lw=1.4, label=f"buy&hold (MDD {mdd_bh*100:.1f}%)")
ax.fill_between(xx, dd(eq_macro), 0, color=C["mom"], alpha=0.12)
ax.set_title("回撤对比：动量择时在趋势反转段砍掉深跌", fontsize=12)
ax.set_xlabel("交易日", fontsize=10)
ax.set_ylabel("回撤 (%)", fontsize=10)
ax.grid(True, color=C["grid"], ls=":", alpha=0.7)
ax.legend(loc="lower left", fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(D, "cam_drawdown.png"), dpi=130)
plt.close(fig)

metrics = {
    "T_total": T, "L_lookback": L, "n_assets": n,
    "ann_macro": round(mu_macro, 4), "vol_macro": round(sd_macro, 4),
    "sharpe_macro": round(sh_macro, 2), "mdd_macro": round(mdd_macro, 4),
    "ann_bh": round(mu_bh, 4), "sharpe_bh": round(sh_bh, 2), "mdd_bh": round(mdd_bh, 4),
    "ann_raw": round(mu_raw, 4), "sharpe_raw": round(sh_raw, 2), "mdd_raw": round(mdd_raw, 4),
    "corr_macro_bh": round(corr_macro_bh, 3),
    "per_asset": [{"name": p[0], "timed_ann": round(p[1], 4),
                   "timed_sharpe": round(p[2], 2), "bh_ann": round(p[3], 4)} for p in per_asset],
}
with open(os.path.join(D, "_metrics.txt"), "w") as f:
    json.dump(metrics, f, ensure_ascii=False, indent=2)
print(json.dumps(metrics, ensure_ascii=False, indent=2))
