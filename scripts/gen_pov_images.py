#!/usr/bin/env python3
"""生成 POV 参与率执行文章配图"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = "public/images/pov-participation-execution"
rng = np.random.default_rng(42)

# ---- 图1: 日内成交量 U 型曲线与 POV 跟随执行 ----
n = 48  # 48 个 5 分钟 bar
t = np.arange(n)
u_shape = 1.8 * np.exp(-((t - 0) / 12) ** 2) + 2.2 * np.exp(-((t - 47) / 8) ** 2) + 0.6
mkt_vol = u_shape * (1 + 0.25 * rng.standard_normal(n))
mkt_vol = np.clip(mkt_vol, 0.2, None) * 10000

rate = 0.15
my_vol = mkt_vol * rate / (1 - rate)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(t, mkt_vol, color="#9ecae1", label="市场成交量（其他参与者）", width=0.8)
ax.bar(t, my_vol, bottom=mkt_vol, color="#e6550d", label="POV 算法成交量（目标参与率 15%）", width=0.8)
ax.set_xlabel("交易日内 5 分钟 bar 序号")
ax.set_ylabel("成交量（股）")
ax.set_title("POV 执行：订单量跟随日内 U 型成交量节奏")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pov-intraday-follow.png", dpi=130)
plt.close(fig)

# ---- 图2: 不同参与率下的完成时间 vs 冲击成本权衡 ----
rates = np.linspace(0.02, 0.40, 100)
Q = 1.0
duration = Q / rates  # 相对完成时间
impact = 25 * np.sqrt(rates)  # 平方根冲击, bp
risk = 8.0 / np.sqrt(rates) * 0.35  # 时间风险近似

fig, ax1 = plt.subplots(figsize=(9, 4.8))
ax1.plot(rates * 100, impact, color="#e6550d", lw=2, label="市场冲击成本 (bp)")
ax1.plot(rates * 100, risk, color="#3182bd", lw=2, label="价格漂移风险 (bp)")
total = impact + risk
ax1.plot(rates * 100, total, color="#31a354", lw=2.5, ls="--", label="总成本")
opt = rates[np.argmin(total)] * 100
ax1.axvline(opt, color="gray", ls=":", lw=1.5)
ax1.annotate(f"最优参与率 ≈ {opt:.0f}%", xy=(opt, total.min()),
             xytext=(opt + 5, total.min() + 6),
             arrowprops=dict(arrowstyle="->", color="gray"))
ax1.set_xlabel("参与率 (%)")
ax1.set_ylabel("成本 (bp)")
ax1.set_title("参与率的权衡：冲击成本 vs 时间风险")
ax1.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pov-tradeoff.png", dpi=130)
plt.close(fig)

# ---- 图3: POV vs TWAP vs VWAP 执行轨迹对比 ----
cum_mkt = np.cumsum(mkt_vol)
cum_mkt_pct = cum_mkt / cum_mkt[-1]

pov_done = np.minimum(cum_mkt_pct * rate / (rate * 1.0), 1.0)  # POV 跟随市场
pov_done = cum_mkt_pct  # 归一化后 POV 完成度 = 市场累计成交占比
twap_done = (t + 1) / n
# VWAP 用预测的 U 型曲线
pred = u_shape / u_shape.sum()
vwap_done = np.cumsum(pred)

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(t, pov_done * 100, lw=2.2, color="#e6550d", label="POV（跟随实际成交量）")
ax.plot(t, twap_done * 100, lw=2, color="#3182bd", ls="--", label="TWAP（时间均匀）")
ax.plot(t, vwap_done * 100, lw=2, color="#31a354", ls="-.", label="VWAP（预测成交量曲线）")
ax.set_xlabel("交易日内 5 分钟 bar 序号")
ax.set_ylabel("累计完成比例 (%)")
ax.set_title("三种执行算法的累计完成轨迹")
ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/pov-vs-twap-vwap.png", dpi=130)
plt.close(fig)

# ---- 图4: 模拟回测——不同参与率的滑点分布 ----
sims = 500
res = {}
for r in [0.05, 0.15, 0.30]:
    dur = 1.0 / r
    drift = rng.standard_normal(sims) * 6 * np.sqrt(dur / 6.67)
    imp = 25 * np.sqrt(r) + rng.standard_normal(sims) * 3
    res[r] = drift + imp

fig, ax = plt.subplots(figsize=(9, 4.8))
colors = {0.05: "#3182bd", 0.15: "#31a354", 0.30: "#e6550d"}
for r, v in res.items():
    ax.hist(v, bins=40, alpha=0.55, color=colors[r], label=f"参与率 {int(r*100)}%（均值 {v.mean():.1f} bp, σ {v.std():.1f}）")
ax.set_xlabel("执行滑点 (bp, 相对到达价)")
ax.set_ylabel("频数")
ax.set_title("蒙特卡洛模拟：参与率越低，均值越好但离散越大")
ax.legend(fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/pov-slippage-dist.png", dpi=130)
plt.close(fig)

print("POV images done")
