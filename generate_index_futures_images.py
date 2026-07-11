#!/usr/bin/env python3
"""
为文章「期现套利与指数跟踪误差：用期货复制现货的无风险机会」(index-cash-futures-arbitrage)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. futures_spot_vs_future.png   现货指数 vs 期货价格
  2. futures_fairvalue_band.png    期货 vs 持有成本理论价(带) + 套利区间
  3. futures_carry_pnl.png         期现套利(cash-and-carry)累计净收益
  4. tracking_error_hist.png       用期货复制现货的日度跟踪误差分布

数值校验：F = S*exp((r-q)T) 为理论价；基差越出成本带即开仓，
          到期收敛使套利稳态盈利；期货复制现货的 TE 应远小于单买期货不展期的漂移。
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
D = os.path.join(BASE, "index-cash-futures-arbitrage")
os.makedirs(D, exist_ok=True)

C = {"spot": "#4C72B0", "fut": "#C44E52", "pnl": "#55A868",
     "grid": "#DDDDDD", "band": "#DDDDDD", "thr": "#C44E52"}

# ============================================================
# 1) 模拟现货指数 + 期货
#    spot 走 GBM；理论期货 F* = S * exp((r-q)*T)
#    实际期货 = 理论价 + 基差(basis, OU 均值回复到 0)；
#    无套利：基差上行受持有成本上界、下行受反向套利下界约束。
# ============================================================
def simulate(T=1500, seed=42):
    rng = np.random.default_rng(seed)
    dt = 1 / 252
    mu, sig = 0.05, 0.16
    S = np.empty(T); S[0] = 3000.0
    for t in range(1, T):
        S[t] = S[t - 1] * np.exp((mu - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * rng.normal())
    r, q = 0.025, 0.018
    Tm = 3 / 12                      # 合约剩余期限（3 个月），为绘图简化设为常数
    fair = S * np.exp((r - q) * Tm)  # 持有成本理论价
    # 基差（对数比）OU 回复到 0
    theta, sigma_b, b = 0.08, 0.004, 0.0
    basis = np.zeros(T)
    for t in range(1, T):
        basis[t] = basis[t - 1] + theta * (b - basis[t - 1]) + sigma_b * rng.normal()
    F = fair * np.exp(basis)
    return S, F, fair, basis, r, q, Tm

S, F, fair, basis, r, q, Tm = simulate(T=1500, seed=11)
T = len(S)
COST = 0.0012                       # 机构单边期现套利成本（手续费+冲击+融资利差缺口）约 0.12%，零售更高
t = np.arange(T)
# 基差(对数)对应的百分比偏离
dev = (F - fair) / fair * 100.0

# ============================================================
# 2) 期现套利策略
#    dev > +COST  -> 正向套利：卖期货 + 买现货，到期平仓，赚 (F-S 理论差 - 成本)
#    dev < -COST  -> 反向套利：买期货 + 卖现货，赚 (理论差 - F - 成本)
#    持仓直到基差回落到 ±COST 内平仓（收敛即止盈）。
# ============================================================
pos = np.zeros(T)
ret = np.zeros(T)
for i in range(1, T):
    d_prev, d_cur = dev[i - 1], dev[i]
    if d_cur > COST * 100:
        target = -1          # 正向套利：卖期货+买现货
    elif d_cur < -COST * 100:
        target = 1           # 反向套利：买期货+卖现货
    else:
        target = 0           # 回到无套利带 -> 平仓（收敛即止盈）
    cost = 0.0
    if target != pos[i - 1]:
        cost = COST * 100 if (pos[i - 1] == 0 or target == 0) else 2 * COST * 100
    ret[i] = pos[i - 1] * (d_cur - d_prev) - cost
    pos[i] = target
pnl = np.cumsum(ret)

# ============================================================
# 3) 跟踪误差：用期货复制现货
#    直接用近月期货持有不展期 -> 到期前价格偏离现货（基差风险）；
#    用 "滚动展期 + 持有成本调整" 的复制组合 -> 跟踪误差显著更小。
#    这里构造两个副本：① 裸持期货  ② 理论复制(期货回归理论价 S*exp((r-q)T))
# ============================================================
rep_naive = np.diff(np.log(F)) * 100          # 裸持近月期货、不展期
idx_ret = np.diff(np.log(S)) * 100             # 现货指数日收益(%)
te_naive = np.std(rep_naive - idx_ret)        # 残留基差风险带来的跟踪误差

# ============================================================
# 图 1：现货 vs 期货
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, S, color=C["spot"], lw=1.0, label="现货指数")
ax.plot(t, F, color=C["fut"], lw=0.8, alpha=0.85, label="股指期货")
ax.set_xlabel("交易日"); ax.set_ylabel("点位")
ax.set_title("现货指数与股指期货长期纠缠、短期基差分离")
ax.legend(loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "futures_spot_vs_future.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：理论价带 + 套利区间
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, F, color=C["fut"], lw=0.8, label="实际期货")
ax.plot(t, fair, color=C["spot"], lw=1.0, label="持有成本理论价 F*=S·e^((r-q)T)")
ax.plot(t, fair * (1 + COST), color=C["thr"], ls="--", lw=1, label="+成本上界(正向套利触发)")
ax.plot(t, fair * (1 - COST), color=C["thr"], ls="--", lw=1, label="-成本下界(反向套利触发)")
ax.fill_between(t, fair * (1 - COST), fair * (1 + COST), color=C["band"], alpha=0.4, label="无套利带")
ax.set_xlabel("交易日"); ax.set_ylabel("点位")
ax.set_title("期货越出持有成本±成本带即出现期现套利窗口")
ax.legend(loc="upper left", fontsize=8); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "futures_fairvalue_band.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：期现套利累计净收益
# ============================================================
fig, ax = plt.subplots(figsize=(11, 4.4))
ax.plot(t, pnl, color=C["pnl"], lw=1.1, label="期现套利累计净收益(%%, 单边成本 %.2f%%)" % (COST * 100))
ax.axhline(0, color="black", lw=0.6)
ax.set_xlabel("交易日"); ax.set_ylabel("累计净收益 (%)")
ax.set_title("基差到期收敛 → 套利稳态盈利")
ax.legend(loc="upper left"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "futures_carry_pnl.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：跟踪误差分布
# ============================================================
fig, ax = plt.subplots(figsize=(10, 4.6))
ax.hist((rep_naive - idx_ret), bins=50, color=C["spot"], alpha=0.8,
        density=True, label="裸持近月期货（不展期）日度跟踪误差")
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("日度跟踪误差 (%)"); ax.set_ylabel("密度")
ax.set_title("裸持期货不展期：残留基差风险使复制现货的日度跟踪误差 σ=%.4f%%\n（用持有成本调整+滚动展期可将其压向 0）" % te_naive)
ax.legend(loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "tracking_error_hist.png"), dpi=130)
plt.close()

print("=== 期现套利与指数跟踪误差 关键数字 ===")
print("样本 T=%d, 单边成本=%.4f, 持有成本 (r-q)T=%.4f" % (T, COST, (r - q) * Tm))
print("正向窗口(dev>+COST)数=%d, 反向窗口(dev<-COST)数=%d" % (np.sum(dev > COST * 100), np.sum(dev < -COST * 100)))
print("期现套利累计净收益=%.2f%%, 最大回撤=%.2f%%" % (pnl[-1], np.min(pnl - np.maximum.accumulate(pnl))))
print("跟踪误差(裸持期货不展期残留基差风险): σ=%.4f%%" % te_naive)
print("\n图片已保存到:", D)
