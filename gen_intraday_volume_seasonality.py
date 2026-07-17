#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
为文章「日内成交量季节性：用 U 型曲线预测盘中流动性窗口」生成真实配图与统计数字。

核心机制(实证 Stylized Fact: 几乎所有市场的日内成交量呈 U 型/U-shape):
  - 开盘前 30~45 分钟: 隔夜信息集中释放 + 集合竞价延续, 成交量最高
  - 午盘 11:30~14:00: 信息真空, 成交量坍缩到日内最低(流动性最差窗口)
  - 收盘前 30~45 分钟: 机构调仓、指数再平衡、收盘竞价, 成交量二次冲高

全部数字由文中 Python 真实计算(仅依赖 numpy/scipy/matplotlib)。
图片:
  ivs_profile.png      —— 日内平均成交量分钟级曲线(U 型)
  ivs_cumvol.png       —— 累计成交量 vs 时钟(S 型, 标注流动性窗口)
  ivs_cost.png         —— 时段执行成本代理(与成交量反比): 午盘成本最高
  ivs_vwap.png         —— 1 万手大单: VWAP 调度 vs 均匀调度的实现差额(implementation shortfall)
"""
import os
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams["font.sans-serif"] = ["Arial Unicode MS", "PingFang SC", "SimHei", "DejaVu Sans"]
rcParams["axes.unicode_minus"] = False

BASE = "/Users/halo/workspace/astro-blog/public/images"
D = os.path.join(BASE, "intraday-volume-seasonality")
os.makedirs(D, exist_ok=True)
METRICS = os.path.join(D, "_metrics.txt")
lines = []

def log(s):
    print(s); lines.append(str(s))

rng = np.random.default_rng(20260717)
N_MIN = 240  # 一个交易日的分钟数(以 4 小时连续竞价段示意)

# ================= 1. 合成日内成交量 U 型曲线 =================
# U 型基准: 两端高、中间低。用二次函数 baseline + 平滑噪声刻画。
t = np.arange(N_MIN)
center = (N_MIN - 1) / 2.0
# 归一化时间 [-1, 1]
u = (t - center) / center
# U 型: 1 + k*u^2, 两端=1+k 中间=1
base = 1.0 + 2.2 * u ** 2
# 开盘跳变额外 boost(集合竞价延续) 与收盘竞价 boost
open_boost = np.exp(-((t - 5) ** 2) / (2 * 8 ** 2)) * 1.3      # 前 5 分钟附近
close_boost = np.exp(-((t - (N_MIN - 6)) ** 2) / (2 * 10 ** 2)) * 1.1
profile = base + open_boost + close_boost
# 加入逐分钟乘性噪声(流动性冲击), 并设午间更脆(波动更大)
noise = rng.lognormal(mean=0.0, sigma=0.18 + 0.12 * np.exp(-((t - center) ** 2) / (2 * 30 ** 2)), size=N_MIN)
vol_min = profile * noise
vol_min = np.clip(vol_min, 0.05, None)
vol_min = vol_min / vol_min.mean()  # 归一化到均值 1

peak_open = vol_min[:15].mean()
trough_mid = vol_min[108:132].mean()
peak_close = vol_min[-15:].mean()
log(f"开盘15分钟均量(归一)={peak_open:.3f}")
log(f"午盘均量(归一, 108-132分)={trough_mid:.3f}")
log(f"收盘15分钟均量(归一)={peak_close:.3f}")
log(f"U型比(两端/中段)={(peak_open + peak_close) / (2 * trough_mid):.2f}x")

# ================= 2. 累计成交量 vs 时钟 =================
cum = np.cumsum(vol_min)
cum = cum / cum[-1]
# 流动性窗口定义: 成交量位于前/后 20% 分钟为高流动性
high_liq_mask = vol_min >= np.quantile(vol_min, 0.7)
log(f"高流动性分钟占比={high_liq_mask.mean()*100:.1f}%")

# ================= 3. 时段执行成本代理(与成交量反比) =================
# 简化执行成本模型: cost = c0 + c1 / vol_min  (成交量越低, 冲击/价差成本越高)
cost = 0.8 + 2.5 / vol_min
cost_mean = cost.mean()
cost_mid = cost[108:132].mean()
cost_open = cost[:15].mean()
cost_close = cost[-15:].mean()
log(f"平均执行成本代理={cost_mean:.3f}bp")
log(f"午盘执行成本={cost_mid:.3f}bp | 开盘={cost_open:.3f}bp | 收盘={cost_close:.3f}bp")
log(f"午盘/开盘成本比={cost_mid/cost_open:.2f}x")

# ================= 4. 大单调度: VWAP vs 均匀 =================
# 假设要在盘中执行 10000 手大单, 拆成 N_MIN 份。
#   - 均匀调度: 每分 10000/N_MIN 手, 但低流动性分钟成交冲击大 -> 实际成交价偏离
#   - VWAP 调度: 按成交量分布分配手数, 天然集中在高流动性窗口
Q = 10000.0
uniform_qty = np.full(N_MIN, Q / N_MIN)
vwap_qty = Q * vol_min / vol_min.sum()
# 执行价格偏移: 单分钟成交量压力 = 分配手数 / 该分钟真实流动性容量(vol_min 比例)
# 冲击成本 ~ 分配占比 * 成本代理
exec_price_drift_uniform = (uniform_qty / (Q / N_MIN)) * 0.0 + (uniform_qty / (Q)) * cost * 100
exec_price_drift_vwap = (vwap_qty / Q) * cost * 100
# 实现差额(IS): 对执行量加权的总价格漂移(基点)
is_uniform = np.sum(uniform_qty * exec_price_drift_uniform) / Q
is_vwap = np.sum(vwap_qty * exec_price_drift_vwap) / Q
log(f"均匀调度实现差额(IS)={is_uniform:.3f}bp")
log(f"VWAP调度实现差额(IS)={is_vwap:.3f}bp")
log(f"VWAP 相对均匀节省={(is_uniform - is_vwap)/is_uniform*100:.1f}%")

# ================= 绘图 =================
C_BLUE, C_RED, C_GRID = "#2563eb", "#dc2626", "#E2E2E2"
plt.rcParams["figure.dpi"] = 130

# 图1: U 型曲线
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, vol_min, color=C_BLUE, lw=1.4, alpha=0.5, label="逐分钟成交量(含噪声)")
ax.plot(t, profile / profile.mean(), color="#1e3a8a", lw=2.2, label="平滑 U 型基准")
ax.axhspan(0, np.quantile(vol_min, 0.7), color=C_RED, alpha=0.06)
ax.fill_between(t, 0, vol_min, color=C_BLUE, alpha=0.12)
ax.set_title("日内成交量 U 型曲线：开盘与收盘最肥, 午盘最瘦", fontsize=12)
ax.set_xlabel("交易分钟 (0=开盘, 240=收盘)")
ax.set_ylabel("相对成交量 (均值=1)")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "ivs_profile.png")); plt.close(fig)

# 图2: 累计成交量 (S 型) + 流动性窗口标注
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, cum, color=C_BLUE, lw=2.2)
ax.set_title("累计成交量 vs 时钟：S 型, 高流动性窗口在头尾", fontsize=12)
ax.set_xlabel("交易分钟"); ax.set_ylabel("累计成交量占比")
# 标窗口
ax.axvspan(0, 30, color="#16a34a", alpha=0.12)
ax.axvspan(210, 240, color="#16a34a", alpha=0.12)
ax.axvspan(108, 132, color=C_RED, alpha=0.12)
ax.text(15, 0.05, "开盘窗口\n(高流动性)", ha="center", fontsize=8, color="#15803d")
ax.text(225, 0.05, "收盘窗口\n(高流动性)", ha="center", fontsize=8, color="#15803d")
ax.text(120, 0.05, "午盘\n(低流动性)", ha="center", fontsize=8, color=C_RED)
ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "ivs_cumvol.png")); plt.close(fig)

# 图3: 执行成本代理
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, cost, color=C_RED, lw=1.6)
ax.axvspan(108, 132, color=C_RED, alpha=0.15)
ax.set_title("时段执行成本代理：午盘流动性最差、成本最高", fontsize=12)
ax.set_xlabel("交易分钟"); ax.set_ylabel("执行成本代理 (bp)")
ax.text(120, cost_mid + 0.3, f"午盘 {cost_mid:.1f}bp", ha="center", fontsize=9, color=C_RED)
ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "ivs_cost.png")); plt.close(fig)

# 图4: VWAP vs 均匀 调度手数分布
fig, ax = plt.subplots(figsize=(8, 4.4))
ax.plot(t, uniform_qty, color="#6b7280", lw=1.6, label=f"均匀调度 ({is_uniform:.1f}bp)")
ax.plot(t, vwap_qty, color=C_BLUE, lw=1.8, label=f"VWAP 调度 ({is_vwap:.1f}bp)")
ax.set_title("1 万手大单调度：VWAP 把单子压进高流动性窗口", fontsize=11.5)
ax.set_xlabel("交易分钟"); ax.set_ylabel("每分钟执行手数")
ax.legend(fontsize=9); ax.grid(alpha=0.3, color=C_GRID)
fig.tight_layout(); fig.savefig(os.path.join(D, "ivs_vwap.png")); plt.close(fig)

with open(METRICS, "w") as f:
    f.write("\n".join(lines) + "\n")
print("\n=== IMAGES WRITTEN ===")
print("\n".join(os.listdir(D)))
