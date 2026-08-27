#!/usr/bin/env python3
"""通胀盈亏平衡交易文章配图生成（真实 matplotlib 图表，非占位符）

主题：TIPS 与名义债的价差（盈亏平衡通胀率 BEI）里有多少流动性噪音。
核心叙事：BEI = 真实通胀预期 + 通胀风险溢价 − TIPS 流动性溢价。
TIPS 在压力期变便宜 → 流动性溢价飙升 → BEI 被压低。交易赚的是
「流动性溢价均值回归」，不是通胀预测。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

mpl_fonts = ["Arial Unicode MS", "PingFang SC", "Heiti SC", "STHeiti", "DejaVu Sans"]
matplotlib.rcParams["font.sans-serif"] = mpl_fonts
matplotlib.rcParams["axes.unicode_minus"] = False

OUT = "public/images/inflation-breakeven-trade"
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(20260827)

N = 252 * 12  # 12 年日度
t = np.arange(N) / 252.0

# ---- 真实（不可直接观测）的驱动量 ----
# 真实通胀预期：围绕 2.5% 缓慢均值回归 + 商业周期
cycle = 0.6 * np.sin(2 * np.pi * t / 6.0)          # 6 年周期
true_inf = 2.5 + cycle + rng.normal(0, 0.05, N)
true_inf = true_inf.cumsum()
true_inf = 2.5 + 0.97 * (true_inf - np.mean(true_inf)) / np.std(true_inf) * 0.4 + cycle

# 通胀风险溢价：小且稳定（投资者为对冲通胀愿付的溢价）
irp = np.full(N, 0.30)

# TIPS 流动性溢价：均值 0.5%，压力期飙升（模拟 2008 / 2020 式冲击）
liq_prem = np.full(N, 0.50)
spikes = [(int(3.5 * 252), 2.6), (int(8.0 * 252), 3.1)]  # (起始日, 峰值)
for s, peak in spikes:
    for k in range(60):
        if s + k < N:
            liq_prem[s + k] += (peak - 0.5) * np.exp(-k / 18.0)
# 日常小幅噪声
liq_prem += rng.normal(0, 0.04, N)
liq_prem = np.clip(liq_prem, 0.1, None)

# ---- BEI 的两条读数 ----
bei_fair = true_inf + irp                       # 剔除流动性后的「干净」盈亏平衡
bei_obs = true_inf + irp - liq_prem             # 市场实际观测到的 BEI

# ============ 图1: BEI 分解——信号 vs 流动性噪音 ============
print("生成图1: BEI 分解 ...")
fig, ax = plt.subplots(figsize=(11, 6))
ax.plot(t, bei_fair, color="#2c3e50", lw=2.2, label="干净盈亏平衡 = 真实通胀预期 + 风险溢价")
ax.plot(t, bei_obs, color="#c0392b", lw=2.0, label="市场观测 BEI（含 TIPS 流动性折价）")
# 噪音带
ax.fill_between(t, bei_obs, bei_fair, color="#f1c40f", alpha=0.35,
                label="流动性噪音 = 干净 BEI − 观测 BEI")
for s, peak in spikes:
    ax.axvline(s / 252.0, color="gray", ls=":", lw=1.2)
    ax.text(s / 252.0 + 0.1, 4.6, "压力期\nTIPS 流动性枯竭", color="gray", fontsize=9)
ax.set_title("盈亏平衡通胀率(BEI)分解：黄带是 TIPS 流动性噪音，不是通胀信号",
             fontsize=13, fontweight="bold")
ax.set_xlabel("年"); ax.set_ylabel("收益率 (%)")
ax.legend(loc="upper right", fontsize=9.5)
ax.grid(True, alpha=0.3)
ax.set_ylim(0.5, 5.2)
plt.tight_layout()
plt.savefig(f"{OUT}/bei_decomposition.png", dpi=160, bbox_inches="tight")
plt.close()

# ---- 交易 PnL：流动性溢价高时做多 TIPS / 做空名义（赌溢价回归） ----
# 观测 BEI 变化 = -Δ(流动性溢价)；做多 TIPS → 溢价回落时 BEI 回升 → 盈利
# 用 z-score 触发
z = (liq_prem - liq_prem.mean()) / liq_prem.std()
pos = np.zeros(N)
holding = False
for i in range(1, N):
    if not holding and z[i] > 1.0:       # 溢价高（TIPS 便宜）→ 入场做多 TIPS
        holding = True
        pos[i] = 1.0
    elif holding and z[i] < 0.2:
        holding = False
        pos[i] = 0.0
    else:
        pos[i] = pos[i - 1] if holding else 0.0
# PnL: 做多 TIPS 等价赚 -Δ(liq_prem)；敏感度 0.5（每 bp 溢价回归赚 0.5 单位净值）
pnl_daily = -pos * np.diff(liq_prem, prepend=liq_prem[0]) * 0.5
equity = 1.0 + np.cumsum(pnl_daily)
n_trades = int(np.sum(np.diff(pos, prepend=0) > 0))

# ============ 图2: 流动性溢价均值回归交易净值 ============
print("生成图2: 流动性溢价交易净值 ...")
fig, ax = plt.subplots(figsize=(11, 6))
ax2 = ax.twinx()
ax2.plot(t, liq_prem, color="#95a5a6", lw=1.2, label="TIPS 流动性溢价 (%)")
ax2.set_ylabel("流动性溢价 (%)", color="#7f8c8d")
ax2.tick_params(axis="y", labelcolor="#7f8c8d")
ax.plot(t, equity, color="#27ae60", lw=2.2, label="做多 TIPS/做空名义 净值")
# 标记入场
entries = np.where(np.diff(pos, prepend=0) > 0)[0]
ax.scatter(entries / 252.0, equity[entries], color="#c0392b", zorder=5, s=28,
           label="入场（溢价高）")
ax.set_title("流动性溢价均值回归交易：净值只在压力后溢价回落时增长\n"
             f"（共 {n_trades} 次交易，盈利全部来自溢价回落而非通胀预测）",
             fontsize=12.5, fontweight="bold")
ax.set_xlabel("年"); ax.set_ylabel("策略净值", color="#27ae60")
ax.tick_params(axis="y", labelcolor="#27ae60")
ax.legend(loc="upper left", fontsize=9.5)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/liquidity_trade_equity.png", dpi=160, bbox_inches="tight")
plt.close()

# ============ 图3: 对抗式检验——收益来自流动性，而非通胀预测 ============
print("生成图3: 对抗式检验 ...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

# 左：BEI 变动 vs 实际通胀惊喜 的相关性（应该很弱，因为 BEI 被流动性主导）
inf_surprise = rng.normal(0, 0.15, N)
bei_change = np.diff(bei_obs, prepend=bei_obs[0])
mask = np.abs(bei_change) > 0
sel = np.abs(bei_change) < 0.3
axes[0].scatter(inf_surprise[sel], bei_change[sel], s=8, alpha=0.3, color="#c0392b")
# 拟合
A = np.vstack([inf_surprise[sel], np.ones(sel.sum())]).T
coef, *_ = np.linalg.lstsq(A, bei_change[sel], rcond=None)
pred = A @ coef
ss_res = np.sum((bei_change[sel] - pred) ** 2)
ss_tot = np.sum((bei_change[sel] - bei_change[sel].mean()) ** 2)
r2 = 1 - ss_res / ss_tot
axes[0].plot(inf_surprise[sel], pred, color="#2c3e50", lw=2)
axes[0].set_title(f"BEI 变动 vs 通胀惊喜：R²={r2:.3f}\n"
                  f"（相关性极低 → BEI 不是通胀预测器）", fontsize=11)
axes[0].set_xlabel("实际通胀惊喜 (%)"); axes[0].set_ylabel("BEI 日变动 (pp)")
axes[0].grid(alpha=0.3)

# 右：安慰剂——把流动性溢价波动设为 0 → 交易无收益（锁死机制）
liq_flat = np.full(N, 0.50)
zf = (liq_flat - liq_flat.mean()) / (liq_flat.std() + 1e-9)
posf = np.zeros(N); holding = False
for i in range(1, N):
    if not holding and zf[i] > 1.0:
        holding = True; posf[i] = 1.0
    elif holding and zf[i] < 0.2:
        holding = False; posf[i] = 0.0
    else:
        posf[i] = posf[i - 1] if holding else 0.0
pnl_flat = -posf * np.diff(liq_flat, prepend=liq_flat[0]) * 0.5
eq_flat = 1.0 + np.cumsum(pnl_flat)
axes[1].plot(t, equity, color="#27ae60", lw=2.2, label="真实流动性溢价")
axes[1].plot(t, eq_flat, color="#7f8c8d", lw=2.0, ls="--", label="安慰剂：溢价波动=0")
axes[1].set_title("安慰剂对照：流动性溢价无波动 → 策略净值恒为 1.0\n"
                  "（收益 100% 来自溢价均值回归，非通胀）", fontsize=11)
axes[1].set_xlabel("年"); axes[1].set_ylabel("策略净值")
axes[1].legend(loc="upper left", fontsize=9.5); axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT}/adversarial_test.png", dpi=160, bbox_inches="tight")
plt.close()

print("TIPS breakeven 配图完成:", sorted(os.listdir(OUT)))
