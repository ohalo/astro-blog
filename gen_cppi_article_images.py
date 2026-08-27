#!/usr/bin/env python3
"""CPPI 组合保险文章配图生成（真实 matplotlib 图表，非占位符）"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Arrow

mpl_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = mpl_fonts
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "public/images/portfolio-insurance-cppi"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260827)

# ============ 图1: CPPI 机制原理图 ============
print("生成图1: CPPI 机制原理图 ...")
fig, ax = plt.subplots(figsize=(11, 6.5))
ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

ax.text(5, 9.3, "CPPI 组合保险：固定比例投资组合保险机制", ha="center",
        fontsize=16, fontweight="bold")

# 总资产
ax.add_patch(FancyBboxPatch((0.5, 1.0), 9.0, 7.0, boxstyle="round,pad=0.1",
             fc="#eef3f8", ec="#888", lw=1.5))
ax.text(5, 7.6, "组合总资产 (Total Portfolio)  =  风险资产 + 无风险资产", ha="center",
        fontsize=12, fontweight="bold")

# 风险资产
ax.add_patch(FancyBboxPatch((0.9, 1.4), 4.0, 5.2, boxstyle="round,pad=0.1",
             fc="#fde2e1", ec="#c0392b", lw=1.5))
ax.text(2.9, 5.8, "风险资产 (Risk)\n股票 / 权益头寸", ha="center", fontsize=11, color="#c0392b")
ax.text(2.9, 3.6, "暴露 = 乘数 m ×\n(总资产 − 底线)", ha="center", fontsize=10)
ax.text(2.9, 2.3, "m = 4 ~ 5", ha="center", fontsize=11, fontweight="bold")

# 无风险资产
ax.add_patch(FancyBboxPatch((5.1, 1.4), 4.0, 5.2, boxstyle="round,pad=0.1",
             fc="#e2f0e6", ec="#27ae60", lw=1.5))
ax.text(7.1, 5.8, "无风险资产 (Safe)\n债券 / 现金", ha="center", fontsize=11, color="#1e7e34")
ax.text(7.1, 3.6, "持有至底线\n到期保本", ha="center", fontsize=10)
ax.text(7.1, 2.3, "= 底线现值", ha="center", fontsize=11, fontweight="bold")

# 底线箭头
ax.annotate("", xy=(2.9, 1.4), xytext=(2.9, 0.5),
            arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.5))
ax.text(2.9, 0.25, "动态再平衡：市场涨→加仓, 市场跌→减仓", ha="center",
        fontsize=9.5, color="#c0392b")

ax.annotate("", xy=(7.1, 1.4), xytext=(7.1, 0.5),
            arrowprops=dict(arrowstyle="->", color="#1e7e34", lw=1.5))
ax.text(7.1, 0.25, "Floor = 本金 × e^(-rT)", ha="center",
        fontsize=9.5, color="#1e7e34")

plt.tight_layout()
plt.savefig(f"{OUT}/cppi_mechanism.png", dpi=160, bbox_inches="tight")
plt.close()

# ============ 图2: CPPI 路径模拟（含 1987 式闪崩） ============
print("生成图2: CPPI 路径模拟 ...")
T = 252 * 3
m = 4.0
floor_ratio = 0.8           # 保本底线 = 80% 本金
r_daily = 0.03 / 252
dt = 1.0

# 风险资产：先慢牛，第 600 个交易日闪崩 -30%，再恢复
mu = 0.08 / 252
sigma = 0.18 / np.sqrt(252)
risk_ret = rng.normal(mu, sigma, T)
crash_day = 600
risk_ret[crash_day] += -0.30      # 单日 -30% 闪崩
risk_ret[crash_day+1:crash_day+20] += 0.01  # 短暂反弹
risk_price = np.cumprod(1 + risk_ret)
risk_only = 1_000_000 * risk_price

# CPPI 路径
portfolio = np.zeros(T); floor = np.zeros(T); exposure = np.zeros(T)
cash = np.zeros(T)
V = 1_000_000; bond_rate = r_daily
bond_val = V * floor_ratio * np.exp(-bond_rate * T)  # 底线现值(期初投入无风险)
# 简化模型：无风险部分持有到期现值，风险部分动态
bond_account = V * floor_ratio * np.exp(-bond_rate * T)
for t in range(T):
    F = bond_account * np.exp(bond_rate * (t))  # 底线随时间升到 floor_ratio*V
    cushion = max(V - F, 0)
    e = m * cushion
    e = min(e, V)  # 不能超仓
    exposure[t] = e
    # 风险账户按当日收益变动，无风险账户按债券收益
    rb = bond_rate
    if t == 0:
        V = V
    else:
        V = e * (1 + risk_ret[t]) + (V - e) * (1 + rb)
    portfolio[t] = V
    floor[t] = F

fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(risk_price, color="#c0392b", lw=1.6, label="纯风险资产（未保险）")
ax.plot(portfolio / 1_000_000, color="#2c3e50", lw=2.0, label="CPPI 组合（m=4）")
ax.plot(floor / 1_000_000, color="#27ae60", lw=1.6, ls="--", label=f"保本底线（{int(floor_ratio*100)}%）")
ax.axvline(crash_day, color="gray", ls=":", lw=1.2)
ax.text(crash_day + 8, 0.95, "第 600 日闪崩 −30%", color="gray", fontsize=9.5)
ax.set_title("CPPI 组合保险路径模拟：闪崩后强制减仓锁住损失", fontsize=13, fontweight="bold")
ax.set_xlabel("交易日"); ax.set_ylabel("净值（起始 = 1.0）")
ax.legend(loc="upper left", fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.6, 1.8)
plt.tight_layout()
plt.savefig(f"{OUT}/cppi_paths.png", dpi=160, bbox_inches="tight")
plt.close()

# ============ 图3: CPPI vs 买入持有（Buy & Hold）在崩盘中的对比 ============
print("生成图3: CPPI vs 买入持有 崩盘对比 ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# 左图：连续下跌情景
n_days = 60
down = np.r_[-0.01 * np.ones(30), -0.02 * np.ones(20), -0.005 * np.ones(10)]
bh = np.cumprod(1 + down)
# CPPI: 乘数 4，每日减仓
Vc = 1.0; cppi = [1.0]; F = 0.8
for d in down:
    cushion = max(Vc - F, 0)
    e = min(4 * cushion, Vc)
    Vc = e * (1 + d) + (Vc - e) * (1 + 0.0001)
    cppi.append(Vc)
axes[0].plot(bh, color="#c0392b", lw=2, label="买入持有")
axes[0].plot(cppi, color="#2c3e50", lw=2, label="CPPI (m=4)")
axes[0].axhline(0.8, color="#27ae60", ls="--", label="底线 0.8")
axes[0].set_title("连续下跌：CPPI 自动减仓\n（损失被锁定在底线之上）", fontsize=11)
axes[0].set_xlabel("交易日"); axes[0].legend(fontsize=9); axes[0].grid(alpha=0.3)
axes[0].set_ylim(0.55, 1.02)

# 右图：复苏能力对比（先崩后 V 型反弹）
cycle = np.r_[-0.03*np.ones(10), 0.04*np.ones(10), -0.02*np.ones(8), 0.03*np.ones(15)]
bh2 = np.cumprod(1 + cycle)
Vc2 = 1.0; cppi2 = [1.0]; F2 = 0.8
for d in cycle:
    cushion = max(Vc2 - F2, 0)
    e = min(4 * cushion, Vc2)
    Vc2 = e * (1 + d) + (Vc2 - e) * (1 + 0.0001)
    cppi2.append(Vc2)
axes[1].plot(bh2, color="#c0392b", lw=2, label="买入持有")
axes[1].plot(cppi2, color="#2c3e50", lw=2, label="CPPI (m=4)")
axes[1].axhline(0.8, color="#27ae60", ls="--", label="底线 0.8")
axes[1].set_title("V 型反弹：CPPI 因低位减仓\n（反弹时头寸更小，回血更慢）", fontsize=11)
axes[1].set_xlabel("交易日"); axes[1].legend(fontsize=9); axes[1].grid(alpha=0.3)
axes[1].set_ylim(0.55, 1.08)

plt.tight_layout()
plt.savefig(f"{OUT}/cppi_crash_recovery.png", dpi=160, bbox_inches="tight")
plt.close()

print("CPPI 配图完成:", os.listdir(OUT))
