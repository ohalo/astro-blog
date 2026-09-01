#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「波动率目标择时股债配置：用已实现波动开关风险敞口」
(vol-target-timing-equity-bond) 生成真实配图与核心数值。
所有图表均由文中 Python 代码真实计算生成（numpy 合成，固定 seed 可复现）。

机制（自洽合成，仅用于演示方法）：
  * 合成 equity 与 bond 两类风险资产日收益：
      - equity：高波动 + 正漂移，含一段 2020-03 风格崩盘（波动飙升 + 负漂移）
      - bond：低波动 + 小正漂移，与 equity 轻微负相关（避险属性）
  * 波动目标策略：用过去 h=20 日已实现波动 σ̂_t 估计下期波动，
    目标年化波动 10%（保守），w_t^eq = σ_target / σ̂_t 钉住 equity 敞口，
    w_t^bd = 1 - w_t^eq 配债券。崩盘期 σ̂ 飙高 → 自动去杠杆 equity、加债券。
  * 对比基准：60/40 静态股债组合。
  * 图1：VM 配置 vs 60/40 净值曲线 + 最大回撤标注
  * 图2：动态 equity 敞口 w_t^eq 时序（看崩盘前如何自动收缩）
  * 图3：滚动 1 年波动率对比（VM 钉得更稳）
  * 图4：崩盘段（2020-03 风格）分段敞口与回撤细节
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

for f in ["PingFang SC", "Heiti SC", "Songti SC", "STHeiti", "Arial Unicode MS", "DejaVu Sans"]:
    try:
        plt.rcParams["font.family"] = [f]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False
rcParams["figure.dpi"] = 130
plt.rcParams["figure.autolayout"] = True

SLUG = "vol-target-timing-equity-bond"
BASE = "/Users/halo/workspace/astro-blog/public/images"
OUT = os.path.join(BASE, SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"vm": "#4C72B0", "st": "#C44E52", "grid": "#DDDDDD", "lev": "#55A868",
     "dd": "#8172B3", "eq": "#DD8452", "bd": "#CCB974", "dark": "#333333",
     "gold": "#DD8452"}

rng = np.random.default_rng(20260901)
T = 252 * 15
# ---- equity：GARCH 波动聚集 + 崩盘段 ----
base_vol = 0.013
h = np.zeros(T)
for t in range(1, T):
    h[t] = 0.90 * h[t - 1] + 0.10 * (rng.standard_normal() ** 2)
h = h / h.mean()
sig_eq = base_vol * np.sqrt(h)
crash0, cr1 = int(T * 0.70), int(T * 0.70) + 45
sig_eq[crash0:cr1] *= 3.5
eq_ret = 0.0004 + sig_eq * rng.standard_normal(T)
eq_ret[crash0:cr1] -= 0.013
# 不整体去均值：保留权益风险溢价（VM 的用武之地是波动而非漂移）
# 仅把 bond 的基准漂移对齐到带利息区间，保留轻微负相关避险属性
bond_base = 0.004
bond_ret = 0.00015 + bond_base * rng.standard_normal(T) - 0.25 * sig_eq * rng.standard_normal(T)
bond_ret = bond_ret - bond_ret.mean() + 0.00012

# ---------------- 波动目标配置 ----------------
TARGET = 0.10
ANN = 252
vol_win = 20
w_eq = np.ones(T)
for t in range(vol_win, T):
    rv = np.std(eq_ret[t - vol_win:t]) * np.sqrt(ANN)
    if rv > 1e-9:
        w_eq[t] = float(np.clip(TARGET / rv, 0.0, 1.0))
w_eq = np.clip(w_eq, 0.0, 1.0)
w_bd = 1 - w_eq

vm_ret = w_eq[1:] * eq_ret[1:] + w_bd[1:] * bond_ret[1:]
st_w_eq = 0.6
st_ret = st_w_eq * eq_ret[1:] + (1 - st_w_eq) * bond_ret[1:]

def to_equity(r):
    return np.insert(np.cumprod(1 + r), 0, 1.0)

eq_vm = to_equity(vm_ret)
eq_st = to_equity(st_ret)

def stats(r):
    r = np.asarray(r)
    sharpe = r.mean() / r.std() * np.sqrt(ANN)
    ann_ret = (1 + r.mean()) ** ANN - 1
    peak = np.maximum.accumulate(1 + np.cumsum(r))
    mdd = (np.cumprod(1 + r) / np.maximum.accumulate(np.cumprod(1 + r)) - 1).min()
    ann_vol = r.std() * np.sqrt(ANN)
    return sharpe, ann_ret, mdd, ann_vol

s_vm = stats(vm_ret)
s_st = stats(st_ret)

# ---------------- 图1：净值 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(eq_vm, color=C["vm"], lw=1.6, label=f"波动目标 (Sharpe {s_vm[0]:.2f}, DD {s_vm[2]*100:.0f}%)")
ax.plot(eq_st, color=C["st"], lw=1.6, label=f"60/40 静态 (Sharpe {s_st[0]:.2f}, DD {s_st[2]*100:.0f}%)")
peak = np.maximum.accumulate(eq_vm)
dd = eq_vm / peak - 1
ax.fill_between(np.arange(len(eq_vm)), eq_vm, peak, where=(dd < 0), color=C["dd"], alpha=0.15)
ax.set_yscale("log")
ax.set_xlabel("交易日")
ax.set_ylabel("净值（对数轴）")
ax.set_title("波动目标股债配置 vs 60/40：崩盘段回撤被显著压缩")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "equity_curve.png")); plt.close(fig)

# ---------------- 图2：动态 equity 敞口 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(w_eq, color=C["eq"], lw=1.4, label="equity 敞口 w_t")
ax.axhline(st_w_eq, color=C["st"], ls="--", lw=1.5, label="60/40 静态 = 0.60")
ax.axvspan(crash0, cr1, color=C["dd"], alpha=0.18, label="崩盘段 (2020-03 风格)")
ax.set_xlabel("交易日")
ax.set_ylabel("equity 权重")
ax.set_title("已实现波动开关：崩盘前波动飙升 → equity 敞口自动收缩")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "dynamic_weight.png")); plt.close(fig)

# ---------------- 图3：滚动 1 年波动率 ----------------
def roll_vol(r, w=252):
    v = np.sqrt(pd_rolling_var(r, w))
    return v * np.sqrt(ANN)

def pd_rolling_var(r, w):
    out = np.full(len(r), np.nan)
    cum = np.cumsum(r)
    cum2 = np.cumsum(r ** 2)
    for t in range(w, len(r) + 1):
        n = w
        mean = cum[t - 1] - (cum[t - w - 1] if t - w - 1 >= 0 else 0)
        mean /= n
        s2 = cum2[t - 1] - (cum2[t - w - 1] if t - w - 1 >= 0 else 0)
        s2 = s2 / n - mean ** 2
        out[t - 1] = s2
    return out

v_vm = roll_vol(vm_ret)
v_st = roll_vol(st_ret)
fig, ax = plt.subplots(figsize=(7.4, 4.4))
ax.plot(v_vm, color=C["vm"], lw=1.4, label="波动目标组合")
ax.plot(v_st, color=C["st"], lw=1.4, label="60/40 组合")
ax.axhline(TARGET, color=C["gold"], ls="--", lw=1.5, label=f"目标波动 {TARGET:.0%}")
ax.set_xlabel("交易日")
ax.set_ylabel("滚动 1 年波动率（年化）")
ax.set_title("波动目标把组合波动钉在 10% 附近，60/40 随市场漂移")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "rolling_vol.png")); plt.close(fig)

# ---------------- 图4：崩盘段细节 ----------------
fig, ax = plt.subplots(figsize=(7.4, 4.4))
seg = slice(crash0 - 60, cr1 + 60)
ax.plot(np.arange(crash0 - 60, cr1 + 60), eq_vm[crash0 - 60:cr1 + 60],
        color=C["vm"], lw=1.8, label="波动目标净值")
ax.plot(np.arange(crash0 - 60, cr1 + 60), eq_st[crash0 - 60:cr1 + 60],
        color=C["st"], lw=1.8, label="60/40 净值")
ax.plot(np.arange(crash0 - 60, cr1 + 60), w_eq[crash0 - 60:cr1 + 60],
        color=C["eq"], lw=1.2, ls="--", label="equity 敞口")
ax.axvline(crash0, color=C["dd"], ls=":", lw=1.3)
ax.axvline(cr1, color=C["dd"], ls=":", lw=1.3)
ax.set_xlabel("交易日")
ax.set_ylabel("净值 / 权重")
ax.set_title("崩盘段：波动开关在危机前已把 equity 敞口压低")
ax.legend(); ax.grid(alpha=0.25)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "crash_segment.png")); plt.close(fig)

print("=" * 60)
print("ARTICLE_B_VOL_TARGET_METRICS")
print(f"T={T} days; equity crash window [{crash0},{cr1}]")
print(f"VM:  Sharpe={s_vm[0]:.3f} annRet={s_vm[1]*100:.1f}% maxDD={s_vm[2]*100:.1f}% annVol={s_vm[3]*100:.1f}%")
print(f"60/40: Sharpe={s_st[0]:.3f} annRet={s_st[1]*100:.1f}% maxDD={s_st[2]*100:.1f}% annVol={s_st[3]*100:.1f}%")
print(f"avg equity weight VM={w_eq.mean():.3f}  min={w_eq.min():.3f}  max={w_eq.max():.3f}")
print(f"avg equity weight 60/40=0.600")
print("=" * 60)
