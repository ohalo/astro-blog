#!/usr/bin/env python3
"""VWAP/TWAP 执行策略配图生成"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

plt.rcParams["font.family"] = ["PingFang SC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = os.path.join(os.path.dirname(__file__), "..", "public", "images", "vwap-twap-execution")
os.makedirs(OUT, exist_ok=True)

rng = np.random.default_rng(42)

# ---------- 日内 U 型成交量 + 价格路径模拟 ----------
n_bins = 48  # 48 个 5 分钟桶（4 小时交易时段）
t = np.linspace(0, 1, n_bins)

def u_shape_volume(t, seed_rng, noise=0.15):
    base = 1.8 * (t - 0.5) ** 2 + 0.55   # U 型
    base[:2] *= 1.6                        # 开盘冲高
    base[-3:] *= np.array([1.3, 1.6, 2.2]) # 尾盘拉升
    v = base * np.exp(seed_rng.normal(0, noise, len(t)))
    return v / v.sum()

def sim_day(seed):
    r = np.random.default_rng(seed)
    vol_prof = u_shape_volume(t, r)
    # 价格：GBM + 日内轻微漂移
    sigma_bin = 0.012 / np.sqrt(n_bins)
    rets = r.normal(0, sigma_bin, n_bins)
    price = 100 * np.exp(np.cumsum(rets))
    return vol_prof, price

vol_prof, price = sim_day(7)

# ============ 图1：U 型成交量 + VWAP/TWAP 切片对比 ============
fig, axes = plt.subplots(2, 1, figsize=(9, 6.4), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1]})
Q = 1_000_000  # 总股数
twap_slices = np.full(n_bins, Q / n_bins)
vwap_slices = Q * vol_prof

ax = axes[0]
ax.bar(np.arange(n_bins), vol_prof * 100, color="#94a3b8", alpha=0.8, label="市场成交量占比 (%)")
ax.set_ylabel("成交量占比 (%)")
ax.set_title("日内 U 型成交量剖面（48 个 5 分钟桶）", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(np.arange(n_bins), twap_slices / 1000, "o-", color="#f59e0b", ms=3.5, label="TWAP：等量切片")
ax.plot(np.arange(n_bins), vwap_slices / 1000, "s-", color="#2563eb", ms=3.5, label="VWAP：按成交量剖面切片")
ax.set_xlabel("时间桶")
ax.set_ylabel("子单量（千股）")
ax.set_title("同一张 100 万股母单的两种切法", fontsize=12)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "vwap-twap-slicing.png"), dpi=150)
plt.close()

# ============ 图2：执行价格 vs 基准（单日示例） ============
kappa = 8e-9  # 临时冲击系数
def exec_day(slices, vol_prof, price, Q_total, kappa=kappa):
    mkt_vol = vol_prof * 20_000_000  # 市场总量 2000 万股
    part = slices / (mkt_vol + slices)
    impact = kappa * (slices / (mkt_vol + 1)) * 1e6 * price  # 简化线性临时冲击
    exec_px = price * (1 + 0.02 * np.sqrt(part))   # 参与率驱动的平方根冲击
    avg_px = np.sum(exec_px * slices) / Q_total
    return exec_px, avg_px

vwap_mkt = np.sum(price * vol_prof)  # 市场 VWAP 基准
exec_t, avg_t = exec_day(twap_slices, vol_prof, price, Q)
exec_v, avg_v = exec_day(vwap_slices, vol_prof, price, Q)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(price, color="#475569", lw=1.6, label="市场中间价")
ax.axhline(vwap_mkt, color="#16a34a", ls="--", lw=1.4, label=f"市场 VWAP = {vwap_mkt:.2f}")
ax.axhline(avg_t, color="#f59e0b", ls=":", lw=1.8, label=f"TWAP 成交均价 = {avg_t:.2f}")
ax.axhline(avg_v, color="#2563eb", ls=":", lw=1.8, label=f"VWAP 策略成交均价 = {avg_v:.2f}")
ax.set_xlabel("时间桶")
ax.set_ylabel("价格")
ax.set_title("单日执行：两种策略的成交均价 vs 市场 VWAP 基准", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "single-day-execution.png"), dpi=150)
plt.close()

# ============ 图3：蒙特卡洛 500 天，VWAP 滑点分布 ============
n_days = 500
slip_twap, slip_vwap, slip_vwap_est = [], [], []
hist_prof = np.zeros(n_bins)
# 用前 60 天估计历史成交量剖面
for d in range(60):
    vp, _ = sim_day(1000 + d)
    hist_prof += vp
hist_prof /= hist_prof.sum()

for d in range(n_days):
    vp, px = sim_day(2000 + d)
    mkt_vwap = np.sum(px * vp)
    _, a_t = exec_day(np.full(n_bins, Q / n_bins), vp, px, Q)
    _, a_v = exec_day(Q * vp, vp, px, Q)             # 上帝视角：当日真实剖面
    _, a_e = exec_day(Q * hist_prof, vp, px, Q)      # 现实：历史剖面预测
    slip_twap.append((a_t / mkt_vwap - 1) * 1e4)
    slip_vwap.append((a_v / mkt_vwap - 1) * 1e4)
    slip_vwap_est.append((a_e / mkt_vwap - 1) * 1e4)

fig, ax = plt.subplots(figsize=(9, 5))
bins = np.linspace(min(min(slip_twap), min(slip_vwap_est)) - 1,
                   max(max(slip_twap), max(slip_vwap_est)) + 1, 55)
ax.hist(slip_twap, bins=bins, alpha=0.55, color="#f59e0b",
        label=f"TWAP（均值 {np.mean(slip_twap):.2f} bp, σ {np.std(slip_twap):.2f}）")
ax.hist(slip_vwap_est, bins=bins, alpha=0.55, color="#2563eb",
        label=f"VWAP-历史剖面（均值 {np.mean(slip_vwap_est):.2f} bp, σ {np.std(slip_vwap_est):.2f}）")
ax.hist(slip_vwap, bins=bins, alpha=0.45, color="#16a34a",
        label=f"VWAP-真实剖面（均值 {np.mean(slip_vwap):.2f} bp, σ {np.std(slip_vwap):.2f}）")
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("对市场 VWAP 的滑点（bp，买单为正=吃亏）")
ax.set_ylabel("天数")
ax.set_title(f"蒙特卡洛 {n_days} 天：三种执行的 VWAP 滑点分布", fontsize=12)
ax.legend(fontsize=9)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "slippage-distribution.png"), dpi=150)
plt.close()

# ============ 图4：订单规模 vs 滑点（参与率效应） ============
sizes = np.array([0.2, 0.5, 1, 2, 5, 10, 20]) * 1e6
mean_t, mean_v = [], []
for Qs in sizes:
    st, sv = [], []
    for d in range(120):
        vp, px = sim_day(5000 + d)
        mkt_vwap = np.sum(px * vp)
        _, a_t = exec_day(np.full(n_bins, Qs / n_bins), vp, px, Qs)
        _, a_e = exec_day(Qs * hist_prof, vp, px, Qs)
        st.append((a_t / mkt_vwap - 1) * 1e4)
        sv.append((a_e / mkt_vwap - 1) * 1e4)
    mean_t.append(np.mean(st))
    mean_v.append(np.mean(sv))

pr = sizes / 20_000_000 * 100
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(pr, mean_t, "o-", color="#f59e0b", lw=2, label="TWAP")
ax.plot(pr, mean_v, "s-", color="#2563eb", lw=2, label="VWAP（历史剖面）")
ax.set_xscale("log")
ax.set_xlabel("母单规模 / 日成交量 (%)")
ax.set_ylabel("平均滑点 (bp)")
ax.set_title("订单越大，剖面匹配的价值越大", fontsize=12)
ax.legend()
ax.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "size-vs-slippage.png"), dpi=150)
plt.close()

print("VWAP/TWAP images done:", os.listdir(OUT))
print(f"TWAP slip: {np.mean(slip_twap):.3f} bp (sd {np.std(slip_twap):.3f})")
print(f"VWAP-est slip: {np.mean(slip_vwap_est):.3f} bp (sd {np.std(slip_vwap_est):.3f})")
print(f"VWAP-true slip: {np.mean(slip_vwap):.3f} bp (sd {np.std(slip_vwap):.3f})")
print("size sweep TWAP:", [f"{x:.2f}" for x in mean_t])
print("size sweep VWAP:", [f"{x:.2f}" for x in mean_v])
