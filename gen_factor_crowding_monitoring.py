#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""因子拥挤度监测与规避 配图生成 (4 张真实图表, 自洽合成)

机制(自洽合成, 仅用于演示方法):
  * 合成 60 股票面板, 各自基础因子暴露 f_i ~ N(0,1)
  * 构造一个"拥挤强度"隐藏过程 c_t: 前期 0 -> 升温(120-160) -> 过载(160-200) -> 退潮(200+)
  * 四种拥挤指标从 c_t 派生(带噪声, 体现真实可观测性):
      1) 因子估值 z-score: 抱团把便宜因子名买贵, 估值相对历史抬升
      2) 因子收益 AC(1): 抱团推高短期动量自相关
      3) 横截面暴露离散度(由"权重压缩"模型真实计算): 抱团=大家挤同方向, 仓位离散度塌缩
      4) 资金流/规模增长 z: 申购资金直接放大暴露
  * 因子收益: 过载段(160-200)发生崩盘(收益显著为负), 与拥挤峰值重合
  * 综合打分 > 70 触发降仓: 验证"提前嗅到抱团崩盘"
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

SLUG = "factor-crowding-monitoring"
OUT = os.path.join("/Users/halo/workspace/astro-blog/public/images", SLUG)
os.makedirs(OUT, exist_ok=True)

C = {"net": "#4C72B0", "red": "#C44E52", "green": "#55A868", "orange": "#DD8452",
     "purple": "#8172B3", "grid": "#DDDDDD", "line": "#999999"}

rng = np.random.default_rng(42)
N, T = 60, 240
f = rng.normal(0, 1, N)                       # 基础因子暴露(横截面)
val_proxy = rng.normal(0, 1, N)               # 各股"估值"代理(抱团时高暴露名被买贵)

# ---------- 拥挤强度隐藏过程 c_t ----------
# 升温期拉长, 让拥挤信号在崩盘前就持续抬升(提前预警); 崩盘在过载尾部发生
c = np.zeros(T)
c[110:170] = np.linspace(0.15, 1.0, 60)       # 升温(抱团累积)
c[170:210] = 1.0                              # 过载(崩盘段)
c[210:] = np.linspace(1.0, 0.05, T - 210)     # 退潮

# ---------- 因子收益时序: 过载段崩盘 ----------
base = rng.normal(0.005, 0.015, T)            # 平日温和噪声
crash = np.where(c > 0.8, rng.normal(-0.040, 0.040, T), 0.0)  # 崩盘主导回撤
factor_ret = base + crash

# ---------- 资金流(与 c_t 同向, 带噪声) ----------
flow = 0.5 * c + rng.normal(0, 0.25, T)

# ---------- 指标1: 因子估值 z-score (真实计算: 暴露加权估值) ----------
# 抱团升温 -> 高暴露股被买贵 -> 加权估值上升
w_impl = np.sign(f) * np.power(np.abs(f), 1.0 / (1.0 + c[:, None]))   # 权重压缩
w_impl = w_impl / w_impl.sum(1, keepdims=True)
val_raw = (w_impl * val_proxy[None, :]).sum(1)
val_z = (val_raw - val_raw[:120].mean()) / (val_raw[:120].std() + 1e-9)

# ---------- 指标2: 因子收益滚动 AC(1) ----------
def ac1(x, win=30):
    out = np.full(len(x), np.nan)
    for t in range(win, len(x)):
        seg = x[t - win:t]
        if np.std(seg) > 1e-9:
            out[t] = np.corrcoef(seg[:-1], seg[1:])[0, 1]
    return out
fac_ac1 = ac1(factor_ret, 30)

# ---------- 指标3: 横截面暴露离散度(由权重压缩模型真实计算) ----------
disp = w_impl.std(1)
disp_z = (disp - disp[:120].mean()) / (disp[:120].std() + 1e-9)

# ---------- 指标4: 资金流/规模增长 z ----------
size_z = (flow - flow[:120].mean()) / (flow[:120].std() + 1e-9)

# ---------- 综合拥挤度打分 (0-100) ----------
def to_score(z, lo, hi):
    return np.clip((z - lo) / (hi - lo) * 100, 0, 100)
s1 = to_score(val_z, 0.6, 1.8)
s2 = to_score(fac_ac1, 0.25, 0.5)
s3 = to_score(-disp_z, 0.6, 1.8)             # 离散度越低越拥挤
s4 = to_score(size_z, 1.2, 2.5)
crowd_score = 0.35 * s1 + 0.25 * s2 + 0.20 * s3 + 0.20 * s4

# ---------- 因子净值 + 拥挤规避 ----------
factor_nav = np.cumprod(1 + factor_ret)
pos = np.where(crowd_score > 70, 0.0, 1.0)
nav_avoid = np.cumprod(1 + factor_ret * np.concatenate([[1.0], pos[1:]]))
dd_factor = factor_nav / np.maximum.accumulate(factor_nav) - 1
dd_avoid = nav_avoid / np.maximum.accumulate(nav_avoid) - 1

first_warn = int(np.argmax(crowd_score > 70)) if np.any(crowd_score > 70) else -1
crash_days = np.where(c > 0.8)[0]
print(f"最大回撤(满仓因子)= {dd_factor.min():.2%}")
print(f"最大回撤(拥挤规避)= {dd_avoid.min():.2%}")
print(f"崩盘段 [{crash_days[0]},{crash_days[-1]}] 拥挤度峰值= {crowd_score[crash_days].max():.1f}")
print(f"净值(满仓)= {factor_nav[-1]:.3f}  规避= {nav_avoid[-1]:.3f}")
print(f"触发降仓首日= {first_warn} (崩盘首日= {crash_days[0]}), 提前 {(crash_days[0]-first_warn)} 天")
print(f"触发降仓交易日数= {int((crowd_score>70).sum())}")

# ---------- 图1: 四指标 + 因子净值 ----------
fig, axs = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
t = np.arange(T)
axs[0].plot(t, val_z, color=C["purple"], lw=1.6, label="因子估值 z")
axs[0].plot(t, fac_ac1, color=C["orange"], lw=1.6, label="因子收益 AC(1)")
axs[0].plot(t, -disp_z, color=C["green"], lw=1.6, label="暴露离散度反向 z")
axs[0].plot(t, size_z, color=C["red"], lw=1.6, label="资金流/规模增长 z")
axs[0].axhline(2.0, color="k", ls=":", lw=1.0)
axs[0].axvspan(160, 200, color=C["red"], alpha=0.08, label="拥挤崩盘段")
axs[0].set_title("因子拥挤四指标合成时间序列（预热→抱团崩盘情景）", fontsize=13)
axs[0].legend(fontsize=8, ncol=3); axs[0].grid(color=C["grid"], lw=0.5)
axs[0].set_ylabel("z / 指标值")
axs[1].plot(t, factor_nav, color=C["net"], lw=2.0, label="满仓因子净值")
axs[1].plot(t, nav_avoid, color=C["green"], lw=2.0, label="拥挤规避净值")
axs[1].axvspan(160, 200, color=C["red"], alpha=0.08)
axs[1].set_title("因子净值 vs 拥挤规避净值", fontsize=12)
axs[1].legend(fontsize=9); axs[1].grid(color=C["grid"], lw=0.5)
axs[1].set_ylabel("NAV"); axs[1].set_xlabel("交易日")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "crowding_indicators.png")); plt.close(fig)

# ---------- 图2: 估值 z vs 未来收益 ----------
fut = factor_ret[1:]
z_lag = val_z[:-1]
fig, ax = plt.subplots(figsize=(10, 5.2))
sc = ax.scatter(z_lag, fut * 100, c=z_lag, cmap="coolwarm", s=18, alpha=0.7)
bins = np.linspace(-1.5, 4, 8)
bc = 0.5 * (bins[:-1] + bins[1:])
bm = [np.nanmean(fut[(z_lag >= bins[i]) & (z_lag < bins[i + 1])].mean() * 100)
      if np.any((z_lag >= bins[i]) & (z_lag < bins[i + 1])) else np.nan for i in range(len(bins) - 1)]
ax.plot(bc, bm, "k-o", lw=2, label="分组平均未来1月收益")
ax.axvline(2.0, color=C["red"], ls="--", lw=1.3, label="拥挤阈值 z=2")
ax.set_title("因子估值 z-score 越高，未来 1 期收益越反向（抱团崩盘）", fontsize=13)
ax.set_xlabel("因子估值 z-score (t 期)"); ax.set_ylabel("未来1期因子收益 (%)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.colorbar(sc, ax=ax, label="z-score")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "crowding_val_vs_return.png")); plt.close(fig)

# ---------- 图3: 离散度塌缩 + 崩盘 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(t, disp, color=C["green"], lw=2.2, label="横截面因子暴露离散度")
ax.axvspan(160, 200, color=C["red"], alpha=0.08)
ax.axvline(160, color=C["red"], ls="--", lw=1.2, label="崩盘起点")
ax.set_title("抱团 = 大家都挤同一方向：暴露离散度塌缩", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("离散度 (std of exposure)")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "crowding_dispersion.png")); plt.close(fig)

# ---------- 图4: 综合拥挤度打分 + 触发 ----------
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.plot(t, crowd_score, color=C["purple"], lw=2.0, label="综合拥挤度打分 (0-100)")
ax.axhline(70, color=C["red"], ls="--", lw=1.6, label="规避阈值 = 70")
ax.fill_between(t, 0, crowd_score, where=crowd_score > 70, color=C["red"], alpha=0.18)
ax.axvspan(160, 200, color=C["red"], alpha=0.05)
ax.set_title("综合拥挤度打分：突破阈值即降仓规避", fontsize=13)
ax.set_xlabel("交易日"); ax.set_ylabel("拥挤度打分")
ax.legend(fontsize=9); ax.grid(color=C["grid"], lw=0.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "crowding_score.png")); plt.close(fig)

print("generated:", sorted(os.listdir(OUT)))
