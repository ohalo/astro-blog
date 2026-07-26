#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日内波动率季节性 FFF（柔性傅里叶形式）配图生成"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

for name in ["PingFang SC", "Heiti TC", "Arial Unicode MS", "STSong"]:
    try:
        font_manager.findfont(name, fallback_to_default=False)
        plt.rcParams["font.sans-serif"] = [name]
        break
    except Exception:
        continue
plt.rcParams["axes.unicode_minus"] = False

OUT = "/Users/halo/workspace/astro-blog/public/images/intraday-seasonality-fff"
os.makedirs(OUT, exist_ok=True)

BLUE, RED, GRAY, GREEN, ORANGE, PURPLE = "#2c6fbb", "#c0392b", "#7f8c8d", "#27ae60", "#e67e22", "#8e44ad"
rng = np.random.default_rng(20260726)

# ---- 构造"真实"的日内波动季节性（U 型 + 午间凹陷 + 开盘尖峰）----
BINS = 48  # 每日 48 个 5 分钟 bin（近似 A 股 4 小时交易，仅演示）
tau = np.linspace(0, 1, BINS)  # 归一化交易时刻 [0,1]


def true_seasonality(tau):
    u = 1.0 + 1.4 * np.exp(-((tau - 0.0) ** 2) / (2 * 0.06 ** 2))   # 开盘尖峰
    u += 0.9 * np.exp(-((tau - 1.0) ** 2) / (2 * 0.10 ** 2))         # 收盘抬升
    u += 0.35 * np.cos(2 * np.pi * tau)                              # 整体 U 型
    u -= 0.25 * np.exp(-((tau - 0.5) ** 2) / (2 * 0.08 ** 2))        # 午间凹陷
    return u


s_true = true_seasonality(tau)
s_true = s_true / s_true.mean()  # 归一化，均值=1


# ---- FFF 设计矩阵：低阶多项式 + 若干正弦余弦谐波 ----
def fff_design(tau, P=4):
    cols = [np.ones_like(tau), tau, tau ** 2]  # 多项式趋势项
    for p in range(1, P + 1):
        cols.append(np.sin(2 * np.pi * p * tau))
        cols.append(np.cos(2 * np.pi * p * tau))
    return np.column_stack(cols)


# ---- 模拟多天的 |收益| 作为波动代理 ----
DAYS = 60
abs_ret = np.zeros((DAYS, BINS))
for d in range(DAYS):
    daily_level = np.exp(rng.normal(0, 0.25))  # 每天整体波动水平不同
    eps = np.abs(rng.normal(0, 1, BINS))
    abs_ret[d] = daily_level * s_true * eps * 0.01

mean_abs = abs_ret.mean(axis=0)
mean_abs_norm = mean_abs / mean_abs.mean()

# ============ 图1：日内波动的 U 型季节性（多天叠加 + 均值）============
fig, ax = plt.subplots(figsize=(9, 4.8))
for d in range(0, DAYS, 3):
    ax.plot(tau, abs_ret[d] / abs_ret[d].mean(), color=GRAY, lw=0.5, alpha=0.25)
ax.plot(tau, mean_abs_norm, color=BLUE, lw=2.4, label="跨日平均 |收益|（波动代理）")
ax.plot(tau, s_true, color=RED, lw=2.0, ls="--", label="真实季节性形状")
ax.set_title("日内波动率的 U 型季节性：开盘收盘高、午间低", fontsize=13.5, fontweight="bold")
ax.set_xlabel("归一化交易时刻 τ（0=开盘，1=收盘）")
ax.set_ylabel("相对波动（均值归一化）")
ax.legend(loc="upper center", framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fff-intraday-ushape.png", dpi=130)
plt.close(fig)

# ============ 图2：FFF 阶数 P 对拟合的影响 ============
y = np.log(mean_abs_norm)  # 对数波动做线性回归更稳
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.scatter(tau, mean_abs_norm, color=GRAY, s=18, alpha=0.7, label="观测（平均 |收益|）", zorder=3)
colors = {1: GREEN, 2: ORANGE, 4: BLUE, 8: PURPLE}
for P in [1, 2, 4, 8]:
    Xd = fff_design(tau, P)
    beta = np.linalg.lstsq(Xd, y, rcond=None)[0]
    fit = np.exp(Xd @ beta)
    ax.plot(tau, fit, color=colors[P], lw=1.8, label=f"FFF 阶数 P={P}（{Xd.shape[1]} 参数）")
ax.set_title("柔性傅里叶阶数 P 越高，越贴合开盘尖峰", fontsize=13.5, fontweight="bold")
ax.set_xlabel("归一化交易时刻 τ")
ax.set_ylabel("相对波动")
ax.legend(loc="upper center", fontsize=9, framealpha=0.9, ncol=2)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fff-order-fit.png", dpi=130)
plt.close(fig)

# ============ 图3：去季节化前后的 |收益| 序列（拉直 U 型）============
Xd = fff_design(tau, 4)
beta = np.linalg.lstsq(Xd, np.log(mean_abs_norm), rcond=None)[0]
season_hat = np.exp(Xd @ beta)          # 估计的季节因子
season_hat = season_hat / season_hat.mean()

# 取某一天，展示去季节化
day = 7
raw = abs_ret[day] / abs_ret[day].mean()
deseason = raw / season_hat

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 5.8), sharex=True)
ax1.bar(tau, raw, width=1.0 / BINS * 0.9, color=BLUE, alpha=0.8)
ax1.plot(tau, season_hat, color=RED, lw=2.0, label="FFF 估计季节因子")
ax1.set_ylabel("原始 |收益|（归一）")
ax1.set_title("去季节化：把 U 型从波动里剥掉，还原真实异常", fontsize=13.5, fontweight="bold")
ax1.legend(loc="upper center", framealpha=0.9)
ax1.grid(alpha=0.2)

ax2.bar(tau, deseason, width=1.0 / BINS * 0.9, color=GREEN, alpha=0.8)
ax2.axhline(1.0, color=GRAY, ls="--", lw=1)
ax2.set_ylabel("去季节化 |收益|")
ax2.set_xlabel("归一化交易时刻 τ")
ax2.text(0.02, 0.85, "开盘/收盘不再天然偏高\n此时的尖峰才是真正的异常波动",
         transform=ax2.transAxes, fontsize=9, color=GRAY,
         bbox=dict(boxstyle="round", fc="#f0fff0", ec=GREEN, alpha=0.85))
ax2.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(f"{OUT}/fff-deseasonalize.png", dpi=130)
plt.close(fig)

# ============ 图4：忽略季节性 vs FFF 校正对跳跃检测的影响 ============
# 在午间（低波动区）人为放一个真实跳跃；用未校正 vs 校正的阈值检测
test_day = raw.copy()
jump_bin = 24  # 午间
test_day[jump_bin] += 1.8  # 注入跳跃

# 未校正：用全天固定阈值（均值+k*std）
flat_thr = test_day.mean() + 2.5 * test_day.std()
# FFF 校正：用去季节化后统一阈值，再乘回季节因子
de = test_day / season_hat
de_thr_level = de.mean() + 2.5 * de.std()
fff_thr = de_thr_level * season_hat

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(tau, test_day, width=1.0 / BINS * 0.9, color=BLUE, alpha=0.65, label="|收益|（含午间跳跃）")
ax.plot(tau, np.full(BINS, flat_thr), color=RED, lw=1.8, ls="--", label="固定阈值（忽略季节性）")
ax.plot(tau, fff_thr, color=GREEN, lw=2.0, label="FFF 校正阈值（随季节起伏）")
ax.scatter([tau[jump_bin]], [test_day[jump_bin]], color=ORANGE, s=90, zorder=5,
           marker="*", label="真实午间跳跃")
ax.annotate("固定阈值漏报\n（午间跳跃低于全天线）", xy=(tau[jump_bin], test_day[jump_bin]),
            xytext=(0.55, 0.78), textcoords="axes fraction", fontsize=9, color=RED,
            arrowprops=dict(arrowstyle="->", color=RED))
ax.set_title("季节性校正改变跳跃检测：固定阈值会漏报午间跳跃", fontsize=13, fontweight="bold")
ax.set_xlabel("归一化交易时刻 τ")
ax.set_ylabel("相对波动")
ax.legend(loc="upper left", fontsize=8.8, framealpha=0.9)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(f"{OUT}/fff-jump-detection.png", dpi=130)
plt.close(fig)

print("FFF images done ->", OUT)
for f in sorted(os.listdir(OUT)):
    print("  ", f)
