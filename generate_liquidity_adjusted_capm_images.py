#!/usr/bin/env python3
"""
为文章「流动性调整 CAPM：Pastor-Stambaugh 模型把流动性风险定价」(liquidity-adjusted-capm)
生成真实配图。所有图表均由文中 Python 代码真实计算生成。

图表：
  1. lacapm_liq_factor.png   聚合流动性因子 L_t 时间序列（含「流动性枯竭」阴影）
  2. lacapm_premium.png      按流动性 Beta 排序的组合：平均超额收益随流动性 Beta 单调上升
  3. lacapm_scatter.png      个股散点：流动性 Beta vs 市场 Beta，颜色=平均收益（风险定价结构）
  4. lacapm_pricing_err.png  CAPM 定价误差 vs Pastor-Stambaugh 定价误差（残差显著收窄）

数值校验：流动性 Beta 与平均收益正相关（流动性风险溢价为正）；
          加入流动性因子后截面定价误差 |alpha| 显著下降。
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
D = os.path.join(BASE, "liquidity-adjusted-capm")
os.makedirs(D, exist_ok=True)

C = {"p1": "#4C72B0", "p2": "#C44E52", "p3": "#55A868", "grid": "#DDDDDD",
     "band": "#DDDDDD", "accent": "#8172B3"}
np.set_printoptions(suppress=True, precision=4)

N = 40            # 股票数
T = 1500          # 交易日
rng = np.random.default_rng(20260714)

# ============================================================
# 模拟数据生成
#   市场超额收益 rM_t ~ N(muM, sM)，muM=0.0004
#   流动性因子 L_t = 流动性创新（零均值序列 + 一段系统性枯竭）
#   个股： r_i,t = beta_mkt_i * rM_t + beta_liq_i * L_t + eps_i,t
#     beta_liq_i 决定「流动性风险暴露」，其期望收益补偿 = beta_liq_i * 价格(流动性风险)
#   为体现流动性风险溢价，令 L_t 含一个正的流动性风险价格分量（长期均值 > 0）
# ============================================================
muM = 0.0
rM = rng.normal(muM, 0.010, T)
# 流动性因子：常态轻微正均值（流动性风险有正价格），中段一段枯竭(均值转负)
liq = np.zeros(T)
base = 0.0006
for tt in range(T):
    if 700 <= tt < 850:           # 流动性枯竭期
        liq[tt] = -0.0015 + rng.normal(0, 0.004)
    else:
        liq[tt] = base + rng.normal(0, 0.004)

beta_mkt = rng.uniform(0.5, 1.5, N)
beta_liq = rng.uniform(-1.0, 2.0, N)      # 流动性风险暴露
idio_sd = rng.uniform(0.005, 0.012, N)
R = np.zeros((T, N))
for i in range(N):
    R[:, i] = beta_mkt[i] * rM + beta_liq[i] * liq + rng.normal(0, idio_sd[i], T)

# 各股平均超额收益（截面）
avg_ret = R.mean(0)
X = np.column_stack([np.ones(T), rM])          # CAPM 设计矩阵

# ============================================================
# 截面估计：
#   (a) CAPM：     r_i = a_i + b_i·rM + ε   ->  alpha_capm = a_i
#   (b) Pastor-Stambaugh: r_i = a'_i + b_i·rM + c_i·L + ε  -> alpha_ps = a'_i
# 被流动性因子解释的「伪 alpha」应被吸收 -> |alpha_ps| << |alpha_capm|
# ============================================================
alpha_capm = np.zeros(N)
alpha_ps = np.zeros(N)
beta_mkt_est = np.zeros(N); beta_liq_est = np.zeros(N)
Xps = np.column_stack([np.ones(T), rM, liq])
for i in range(N):
    b1 = np.linalg.lstsq(X, R[:, i], rcond=None)[0]
    b2 = np.linalg.lstsq(Xps, R[:, i], rcond=None)[0]
    alpha_capm[i] = b1[0]
    alpha_ps[i] = b2[0]
    beta_mkt_est[i] = b2[1]
    beta_liq_est[i] = b2[2]

mae_capm = np.mean(np.abs(alpha_capm)) * 252
mae_ps = np.mean(np.abs(alpha_ps)) * 252
print("=== Pastor-Stambaugh 模拟 (N=%d, T=%d) ===" % (N, T))
print("CAPM 截面平均 |alpha| (年化) = %.4f" % mae_capm)
print("PS   截面平均 |alpha| (年化) = %.4f" % mae_ps)
print("定价误差下降 = %.1f%%" % (100 * (mae_capm - mae_ps) / mae_capm))

# ============================================================
# 图 1：聚合流动性因子 L_t，含枯竭阴影
# ============================================================
t = np.arange(T)
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.plot(t, liq * 100, color=C["p1"], lw=1.0)
ax.axhline(0, color="black", lw=0.7)
ax.axvspan(700, 850, color=C["p2"], alpha=0.15, label="流动性枯竭期")
ax.set_xlabel("交易日"); ax.set_ylabel("流动性因子 L_t (%)")
ax.set_title("聚合流动性因子：常态温和为正，危机窗口系统性枯竭")
ax.legend(loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6); ax.set_xlim(0, T)
plt.tight_layout()
plt.savefig(os.path.join(D, "lacapm_liq_factor.png"), dpi=130)
plt.close()

# ============================================================
# 图 2：按流动性 Beta 排序的组合，平均超额收益
# ============================================================
order = np.argsort(beta_liq_est)
port = order.reshape(5, -1)             # 5 分位
port_beta = np.array([beta_liq_est[p].mean() for p in port])
port_ret = np.array([avg_ret[p].mean() * 252 for p in port])
print("\n流动性 Beta 五分位组合：")
for k in range(5):
    print("  组%d: 流动性Beta=%.3f  平均年化超额收益=%.4f" % (k + 1, port_beta[k], port_ret[k]))

fig, ax = plt.subplots(figsize=(9, 5))
colors = [C["p3"] if v > 0 else C["p2"] for v in port_ret]
bars = ax.bar(["Q1\n低", "Q2", "Q3", "Q4", "Q5\n高"], port_ret, color=colors)
ax.axhline(0, color="black", lw=0.8)
ax.set_xlabel("按流动性 Beta 排序的组合"); ax.set_ylabel("平均年化超额收益")
ax.set_title("流动性风险溢价：流动性 Beta 越高，平均收益越高")
for b, v, bv in zip(bars, port_ret, port_beta):
    ax.text(b.get_x() + b.get_width() / 2, v + (0.002 if v >= 0 else -0.006),
            "β_L=%.2f" % bv, ha="center", fontsize=9)
ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "lacapm_premium.png"), dpi=130)
plt.close()

# ============================================================
# 图 3：个股散点 流动性 Beta vs 市场 Beta，颜色=平均收益
# ============================================================
fig, ax = plt.subplots(figsize=(9, 6.2))
sc = ax.scatter(beta_mkt_est, beta_liq_est, c=avg_ret * 252, cmap="RdYlGn",
                s=70, edgecolor="black", linewidth=0.5)
ax.set_xlabel("市场 Beta (β_M)"); ax.set_ylabel("流动性 Beta (β_L)")
ax.set_title("截面风险结构：横轴市场暴露，纵轴流动性暴露，颜色=年化超额收益")
cb = plt.colorbar(sc); cb.set_label("年化超额收益")
ax.grid(True, color=C["grid"], lw=0.6)
plt.tight_layout()
plt.savefig(os.path.join(D, "lacapm_scatter.png"), dpi=130)
plt.close()

# ============================================================
# 图 4：定价误差 CAPM vs Pastor-Stambaugh
# ============================================================
fig, ax = plt.subplots(figsize=(9, 5))
xpos = np.arange(N)
ax.bar(xpos - 0.2, alpha_capm * 252, width=0.4, color=C["p1"], label="CAPM alpha")
ax.bar(xpos + 0.2, alpha_ps * 252, width=0.4, color=C["p3"], label="P-S alpha")
ax.axhline(0, color="black", lw=0.7)
ax.set_xlabel("个股"); ax.set_ylabel("年化 alpha")
ax.set_title("加入流动性因子后：被流动性风险伪装成的『伪 alpha』被吸收")
ax.legend(loc="upper right"); ax.grid(True, color=C["grid"], lw=0.6, axis="y")
plt.tight_layout()
plt.savefig(os.path.join(D, "lacapm_pricing_err.png"), dpi=130)
plt.close()

print("\n图片已保存到:", D)
